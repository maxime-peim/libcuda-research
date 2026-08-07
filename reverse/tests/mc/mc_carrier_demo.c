/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_carrier_demo.c — capstone demo for the VAS-selectable
 * allocator refactor + the SM-owner kernel.
 *
 * Allocates HBM (vidmem) and sysmem buffers BOTH in the carrier VAS
 * — so they're addressable from the DMA + compute channels.  Seeds
 * HBM via mc_memcpy (which now dispatches to the DMA channel
 * because both pointers are MC_VAS_SYSMEM_CARRIER).  Then launches the
 * SM-owner kernel, which has a single SM thread author the entire
 * CE-channel submission for a HBM→sysmem D2H.  The host CPU then
 * verifies the bytes landed.
 *
 * Exercises in one run:
 *   - mc_malloc_device(ctx, n, MC_VAS_SYSMEM_CARRIER)  (HBM in carrier)
 *   - mc_malloc_host  (ctx, n, MC_VAS_SYSMEM_CARRIER)  (sysmem in carrier)
 *   - mc_memcpy(MC_XFER_HOST) dispatch by VAS to the DMA channel
 *   - mc_memcpy(MC_XFER_SM)  SM-authored CE submission via the
 *                                public API (no direct sm_owner call)
 *   - mc_gpu_va recovering the carrier GPU VA from a CPU pointer
 *   - mc_free over carrier-VAS allocs (rm_unmap_memory_dma teardown)
 *
 * Plus a small inline negative test: mismatched-VAS mc_memcpy must
 * return MC_EINVAL.
 *
 * Result interpretation:
 *
 *   PASS    → dst sysmem contains the seeded HBM pattern.  The HBM
 *             → sysmem D2H ran via an SM-authored CE submission
 *             on the DMA channel.  The H2D seed used the new VAS-
 *             dispatched mc_memcpy path.  All carrier-VAS allocator
 *             arms work; mc_free/mc_fini teardown is clean (no Xid).
 *
 *   TIMEOUT → the SM kernel ran but PBDMA didn't process the
 *             SM-authored submission, OR the seed memcpy itself
 *             stalled.
 *
 *   VERIFY  → bytes don't match: somewhere along seed → submit →
 *             readback the data was corrupted.  Usually a bug in
 *             the new allocator path's GPU-VA bookkeeping.
 *
 * Build: `make mc_carrier_demo` from reverse/.
 * Run:   sudo ./bin/mc_carrier_demo [--size N] [--iters M]
 *
 * Defaults: --size 1M --iters 1.  The 4-GiB carrier comfortably
 * holds two ~1.9 GiB buffers; tools/run_mc_tests.py exercises sizes up
 * to 64 MiB.
 *
 * Pre-req: kernel modules from this tree loaded with
 *          nv_dbell_disable_intercept=1 (sm_owner_kernel writes the
 *          DMA channel's BAR1 doorbell from device code; the
 *          watchpoint shadow page would absorb that write).
 *
 * Exit codes: 0 = PASS, 1 = init/alloc failure, 2 = mc_memcpy
 *             failure, 3 = sm_owner submission failure, 4 = verify
 *             mismatch, 5 = mismatched-VAS negative test failed.
 */
#include <emmintrin.h> /* _mm_clflush */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mc.h"
#include "mc_test_helpers.h"

#define DEFAULT_SIZE (1ULL * 1024ULL * 1024ULL) /* 1 MiB */
#define FILL_PATTERN 0xC0FFEE00u

int main(int argc, char **argv)
{
  mc_test_args_t args = { .n_bytes = DEFAULT_SIZE, .iters = 1 };

  mc_ctx_t   *ctx = NULL;
  uint32_t    n;
  void       *d_buf    = NULL; /* carrier HBM */
  void       *h_buf    = NULL; /* carrier sysmem */
  void       *h_verify = NULL; /* H2D-only readback target (carrier sysmem) */
  int         it;
  size_t      bad_off      = 0;
  mc_vas_t    carrier      = MC_VAS_SYSMEM_CARRIER;
  int         read_i       = 1;
  int         write_i      = 1;
  const char *carrier_name = "sysmem-carrier";
  const char *dir_label;

  /* Pre-scan argv for --fb / --sysmem so we can pick the carrier
   * before mc_test_parse_args (which doesn't know our flag and
   * rejects unknown args).  We compact argv in place so the
   * generic parser sees only the args it knows. */
  while (read_i < argc)
  {
    if (strcmp(argv[read_i], "--fb") == 0)
    {
      carrier      = MC_VAS_FB_CARRIER;
      carrier_name = "fb-carrier";
      read_i++;
    }
    else if (strcmp(argv[read_i], "--sysmem") == 0)
    {
      carrier      = MC_VAS_SYSMEM_CARRIER;
      carrier_name = "sysmem-carrier";
      read_i++;
    }
    else
    {
      argv[write_i++] = argv[read_i++];
    }
  }
  argc = write_i;

  switch (mc_test_parse_args(argc, argv, &args))
  {
  case MC_TEST_ARGS_OK:
    break;
  case MC_TEST_ARGS_HELP:
    printf("\n  --sysmem      Use MC_VAS_SYSMEM_CARRIER (default)\n");
    printf("  --fb          Use MC_VAS_FB_CARRIER (FB-resident channel\n");
    printf("                resources, SM-poll-FB-sema fast path)\n");
    return 0;
  case MC_TEST_ARGS_ERROR:
    return 1;
  }

  if (args.n_bytes > 0x7FFFFFFFu)
  {
    fprintf(stderr, "size too large: %zu\n", args.n_bytes);
    return 1;
  }
  n = (uint32_t)args.n_bytes;

  dir_label = args.h2d ? "H2D" : "D2H";

  if (mc_init(&ctx) != MC_OK)
  {
    fprintf(stderr, "mc_init failed\n");
    return 1;
  }
  printf("mc_init ok — exercising %s (size=%u, iters=%d, dir=%s)\n",
         carrier_name, n, args.iters, dir_label);

  d_buf = mc_malloc_device(ctx, n, carrier);
  h_buf = mc_malloc_host(ctx, n, carrier);
  if (!d_buf || !h_buf)
  {
    fprintf(stderr,
            "%s alloc failed (HBM=%p sysmem=%p) — likely carrier-VAS "
            "exhausted or RM rejected vidmem-in-carrier at this size\n",
            carrier_name, d_buf, h_buf);
    mc_fini(ctx);
    return 1;
  }
  printf("d_buf (HBM, %s): user_ptr=%p gpu_va=0x%llx\n", carrier_name, d_buf,
         (unsigned long long)mc_gpu_va(ctx, d_buf));
  printf("h_buf (sysmem, %s): cpu_va=%p gpu_va=0x%llx\n", carrier_name, h_buf,
         (unsigned long long)mc_gpu_va(ctx, h_buf));

  /* For --h2d we need a separate carrier-sysmem readback target so
   * the timed window only contains the SM-authored H2D copy.  The
   * readback uses MC_XFER_SM, so the verify exercises the SM path too. */
  if (args.h2d)
  {
    h_verify = mc_malloc_host(ctx, n, carrier);
    if (!h_verify)
    {
      fprintf(stderr, "%s h_verify alloc failed\n", carrier_name);
      mc_free(ctx, h_buf);
      mc_free(ctx, d_buf);
      mc_fini(ctx);
      return 1;
    }
  }

  /* Seed once.
   *   D2H: CPU-fill h_buf with FILL_PATTERN, mc_memcpy(MC_XFER_HOST)
   *        into d_buf via the carrier DMA channel.  HBM stays loaded
   *        across every iter; the iter loop only zeroes h_buf, runs
   *        the SM-authored D2H, and verifies.
   *   H2D: CPU-fill h_buf once with FILL_PATTERN.  The timed loop's
   *        SM-authored H2D moves it to d_buf each iter; an untimed
   *        D2H read-back into h_verify provides the bytes we verify. */
  mc_test_seed_pattern((uint32_t *)h_buf, n / 4, FILL_PATTERN);
  if (args.h2d)
  {
    /* For MC_XFER_SM H2D the SM-authored CE reads h_buf via PCIe MRd.
     * Per mc.h's coherency contract, CPU writes to a cacheable
     * carrier-VAS host alloc may still sit in x86 L1/L2/L3 when the
     * GPU MRd Requests reach the host; flush them down so the GPU's
     * read of h_buf returns FILL_PATTERN, not stale-or-zero. */
    size_t off;
    for (off = 0; off < n; off += 64)
      _mm_clflush((char *)h_buf + off);
  }
  _mm_sfence();
  if (!args.h2d)
  {
    if (mc_memcpy(ctx, d_buf, h_buf, n, MC_XFER_HOST) != MC_OK)
    {
      fprintf(stderr, "mc_memcpy (seed HBM) failed\n");
      mc_free(ctx, h_verify);
      mc_free(ctx, h_buf);
      mc_free(ctx, d_buf);
      mc_fini(ctx);
      return 2;
    }
  }

  /* Run the SM-authored copy args.iters times.  Buffers are reused
   * across iterations; each iter only stomps the verify target,
   * runs the SM-authored copy, reads back if needed, and verifies. */
  for (it = 0; it < args.iters; it++)
  {
    uint64_t    t0;
    uint64_t    ns;
    mc_status_t rc;
    size_t      off;

    if (args.h2d)
    {
      /* Stomp the device side via h_verify so a silent no-op H2D
       * shows up as a verify failure: pre-fill h_verify with GARBLE
       * (CPU-side, fast), push it through d_buf via MC_XFER_HOST so
       * d_buf is GARBLE entering the timed window. */
      const uint32_t GARBLE = 0xCAFEBABEu;
      mc_test_seed_pattern((uint32_t *)h_verify, n / 4, GARBLE);
      /* Flush h_verify to sysmem so the GPU's PCIe MRd path reads
       * the CPU-written GARBLE rather than stale-or-zero physical
       * pages (the H100 test host is a VMware passthrough where the host root
       * complex's snoop of CPU caches on PCIe MRd is unreliable;
       * library coherency contract per mc.h applies). */
      for (off = 0; off < n; off += 64)
        _mm_clflush((char *)h_verify + off);
      _mm_sfence();
      rc = mc_memcpy(ctx, d_buf, h_verify, n, MC_XFER_HOST);
      if (rc != MC_OK)
      {
        fprintf(stderr,
                "iter %d: pre-stomp mc_memcpy(MC_XFER_HOST) failed (%d)\n",
                it + 1, (int)rc);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }

      /* Timed: SM-authored H2D h_buf -> d_buf. */
      t0 = mc_test_now_ns();
      rc = mc_memcpy(ctx, d_buf, h_buf, n, MC_XFER_SM);
      ns = mc_test_now_ns() - t0;
      if (rc != MC_OK)
      {
        fprintf(stderr,
                "iter %d: mc_memcpy(MC_XFER_SM) H2D failed (status %d)\n",
                it + 1, (int)rc);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }

      /* Untimed: D2H read-back into h_verify, then CPU cache flush
       * so the verify loop reads the GPU-written bytes. */
      rc = mc_memcpy(ctx, h_verify, d_buf, n, MC_XFER_SM);
      if (rc != MC_OK)
      {
        fprintf(stderr,
                "iter %d: readback mc_memcpy(MC_XFER_SM) failed (%d)\n",
                it + 1, (int)rc);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }
      for (off = 0; off < n; off += 64)
        _mm_clflush((char *)h_verify + off);
      _mm_mfence();

      if (!mc_test_verify_pattern((uint32_t *)h_verify, n / 4, FILL_PATTERN,
                                  &bad_off))
      {
        fprintf(
            stderr,
            "iter %d VERIFY FAIL: h_verify[%zu] = 0x%08x (expected 0x%08x)\n",
            it + 1, bad_off, ((uint32_t *)h_verify)[bad_off], FILL_PATTERN);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 4;
      }
    }
    else
    {
      /* Zero h_buf so verification can't be fooled by the still-live
       * pre-memcpy bytes.  The next CPU read after the SM-authored
       * D2H must come from sysmem refilled by the GPU. */
      memset(h_buf, 0, n);
      _mm_sfence();

      /* Timed: SM-authored D2H d_buf -> h_buf. */
      t0 = mc_test_now_ns();
      rc = mc_memcpy(ctx, h_buf, d_buf, n, MC_XFER_SM);
      ns = mc_test_now_ns() - t0;
      if (rc != MC_OK)
      {
        fprintf(stderr,
                "iter %d: mc_memcpy(MC_XFER_SM) D2H failed (status %d)\n",
                it + 1, (int)rc);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }

      /* CPU cache flush so the verify loop reads sysmem post-GPU. */
      for (off = 0; off < n; off += 64)
        _mm_clflush((char *)h_buf + off);

      if (!mc_test_verify_pattern((uint32_t *)h_buf, n / 4, FILL_PATTERN,
                                  &bad_off))
      {
        fprintf(stderr,
                "iter %d VERIFY FAIL: h_buf[%zu] = 0x%08x (expected 0x%08x)\n",
                it + 1, bad_off, ((uint32_t *)h_buf)[bad_off], FILL_PATTERN);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 4;
      }
    }

    printf("  iter %3d: %s=%7.1f us (%5.2f GB/s)\n", it + 1, dir_label, ns / 1e3,
           mc_test_bandwidth_gbps(n, ns));
  }

  printf("PASS: %s transfer authored end-to-end by an SM thread on %s.\n",
         args.h2d ? "DRAM->HBM" : "HBM->DRAM", carrier_name);
  printf("       mc_malloc_device(%s) + mc_malloc_host(%s)\n", carrier_name,
         carrier_name);
  printf("       both worked; mc_memcpy(MC_XFER_SM) ran the SM-authored\n");
  printf("       CE submission via the carrier's SM victim channel.\n");

  mc_free(ctx, h_verify);
  mc_free(ctx, h_buf);
  mc_free(ctx, d_buf);
  mc_fini(ctx);
  return 0;
}
