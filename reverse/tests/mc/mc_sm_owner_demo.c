/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_sm_owner_demo.c — HBM→DRAM demo where a HOPPER_COMPUTE_A SM
 * thread authors an entire CE-channel submission (pushbuffer, GPFIFO
 * entry, USERD GP_PUT, BAR1 doorbell) for the DMA channel.  Same
 * shape as mc_carrier_demo and mc_compute_dbell_chain_demo:
 * allocate a device buffer + sysmem buffer, seed HBM via the
 * vanilla CE path, run the SM-authored D2H, verify in sysmem.
 *
 * The difference vs. mc_carrier_demo is just defaults — this
 * test starts at 4 KiB to keep the latency-bound regime in view; the
 * carrier_d2h test starts at 1 MiB.  Both exercise the same kernel
 * via the same public API:
 *
 *     mc_memcpy(ctx, d_buf, h_buf, n, MC_XFER_HOST)   // seed HBM
 *     mc_memcpy(ctx, h_buf, d_buf, n, MC_XFER_SM)     // SM-authored
 *
 * The MC_XFER_SM dispatch internally calls the sm_owner kernel which
 * runs on the compute channel and authors the DMA channel's
 * submission from device code.
 *
 * Result interpretation:
 *
 *   PASS    → h_buf contains the seed pattern.  Host did NOT ring
 *             the DMA channel's doorbell during the SM-authored
 *             call; an SM thread wrote pushbuffer + GPFIFO entry +
 *             USERD GP_PUT + BAR1 doorbell, PBDMA observed all four
 *             and ran the CE copy from HBM to DRAM.
 *
 *   TIMEOUT → either the compute sema or the DMA channel's sema
 *             didn't fire within MC_TIMEOUT_MS.
 *
 *   VERIFY  → both semas fired but h_buf doesn't match the seed.
 *             The SM-authored method stream was structurally
 *             accepted by PBDMA but produced wrong bytes (likely a
 *             kernel-side OFFSET_IN/OFFSET_OUT or LINE_LENGTH bug,
 *             or HBM data was corrupted between H2D seed and SM
 *             D2H).
 *
 * Build: `make mc_sm_owner_demo` from reverse/.
 * Run:   sudo ./bin/mc_sm_owner_demo [--size N] [--iters N]
 *
 * Pre-req: kernel modules from this tree loaded with
 *          nv_dbell_disable_intercept=1.  Without this the SM's
 *          STG to BAR1+0x90 lands in the watchpoint shadow and
 *          PBDMA never wakes.
 *
 * Exit codes: 0 = PASS, 1 = init/alloc failure, 2 = mc_memcpy
 *             seed failure, 3 = MC_XFER_SM submission failure,
 *             4 = verify mismatch.
 */
#include <emmintrin.h>           /* _mm_clflush, _mm_sfence */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mc.h"
#include "mc_test_helpers.h"

/* Default workload: 4 KiB.  Small enough that a stale-PTE bug
 * unrelated to this experiment can't easily mask the signal; large
 * enough that LINE_LENGTH_IN > 4 exercises the multi-line path on the
 * CE.  --size overrides. */
#define DEFAULT_SIZE   4096u
#define FILL_PATTERN   0xCAFEBABEu

int main(int argc, char **argv)
{
  mc_test_args_t args = { .n_bytes = DEFAULT_SIZE, .iters = 1 };

  mc_ctx_t   *ctx       = NULL;
  uint32_t    n;
  void       *d_buf     = NULL;   /* HBM, GPU-only */
  void       *h_buf     = NULL;   /* sysmem, CPU-readable */
  void       *h_verify  = NULL;   /* H2D-only readback target */
  int         it;
  uint64_t    best_ns   = UINT64_MAX;
  uint64_t    worst_ns  = 0;
  uint64_t    sum_ns    = 0;
  const char *dir_label;

  switch (mc_test_parse_args(argc, argv, &args))
  {
  case MC_TEST_ARGS_OK:    break;
  case MC_TEST_ARGS_HELP:  return 0;
  case MC_TEST_ARGS_ERROR: return 1;
  }
  if (args.n_bytes > 0x7FFFFFFFu)
  {
    fprintf(stderr, "size too large: %zu\n", args.n_bytes);
    return 1;
  }
  if (args.n_bytes & 0x3)
  {
    fprintf(stderr, "size must be dword-aligned: %zu\n", args.n_bytes);
    return 1;
  }
  n = (uint32_t)args.n_bytes;

  dir_label = args.h2d ? "H2D" : "D2H";

  if (mc_init(&ctx) != MC_OK)
  {
    fprintf(stderr, "mc_init failed\n");
    return 1;
  }
  printf("mc_init ok — UVM + DMA + compute channels ready "
         "(size=%u, iters=%d, dir=%s)\n", n, args.iters, dir_label);

  /* Allocate HBM + sysmem ONCE in the carrier VAS, reuse for every
   * iter.  d_buf is HBM (GPU-only, no CPU alias); h_buf is sysmem
   * (CPU-readable, also addressable from the carrier-bound DMA
   * channel via mc_gpu_va).  Both must live in MC_VAS_SYSMEM_CARRIER for
   * the SM-authored MC_XFER_SM path to reach them. */
  d_buf = mc_malloc_device(ctx, n, MC_VAS_SYSMEM_CARRIER);
  h_buf = mc_malloc_host  (ctx, n, MC_VAS_SYSMEM_CARRIER);
  if (!d_buf || !h_buf)
  {
    fprintf(stderr,
            "carrier alloc failed (HBM=%p sysmem=%p)\n", d_buf, h_buf);
    mc_fini(ctx);
    return 1;
  }

  /* For --h2d: a separate carrier-sysmem readback target so the timed
   * window only contains the SM-authored H2D copy. */
  if (args.h2d)
  {
    h_verify = mc_malloc_host(ctx, n, MC_VAS_SYSMEM_CARRIER);
    if (!h_verify)
    {
      fprintf(stderr, "carrier h_verify alloc failed\n");
      mc_free(ctx, h_buf);
      mc_free(ctx, d_buf);
      mc_fini(ctx);
      return 1;
    }
  }

  /* Seed once.
   *   D2H: CPU-fill h_buf, mc_memcpy(MC_XFER_HOST) into d_buf via
   *        the DMA channel.  HBM stays loaded across every iter.
   *   H2D: CPU-fill h_buf once.  The timed loop's SM-authored H2D
   *        moves it to d_buf each iter; an untimed D2H readback into
   *        h_verify is what verify_pattern reads. */
  mc_test_seed_pattern((uint32_t *)h_buf, n / 4u, FILL_PATTERN);
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

  for (it = 0; it < args.iters; it++)
  {
    uint64_t    t0;
    uint64_t    ns;
    mc_status_t rc;
    size_t      bad_off = 0;
    size_t      off;

    if (args.h2d)
    {
      /* Pre-stomp d_buf via an untimed CE-author H2D so a silent
       * no-op SM-authored H2D shows up as GARBLE on readback. */
      const uint32_t GARBLE = 0xCAFEBABEu;
      mc_test_seed_pattern((uint32_t *)h_verify, n / 4u, GARBLE);
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

      /* Timed: SM-authored DRAM->HBM h_buf -> d_buf. */
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

      /* Untimed: D2H readback + flush + verify. */
      rc = mc_memcpy(ctx, h_verify, d_buf, n, MC_XFER_HOST);
      if (rc != MC_OK)
      {
        fprintf(stderr,
                "iter %d: readback mc_memcpy(MC_XFER_HOST) failed (%d)\n",
                it + 1, (int)rc);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }
      for (off = 0; off < n; off += 64)
        _mm_clflush((char *)h_verify + off);

      if (!mc_test_verify_pattern((uint32_t *)h_verify, n / 4u, FILL_PATTERN,
                                  &bad_off))
      {
        fprintf(stderr,
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
      /* Zero h_buf so any verify success can't be from leftover bytes
       * — the CPU read after the SM-authored D2H must come from sysmem
       * refilled by the GPU. */
      memset(h_buf, 0, n);
      _mm_sfence();

      /* Timed: SM-authored HBM->DRAM via the public API. */
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

      /* CPU-side cache hygiene: GPU wrote h_buf through L2 + sysmem;
       * CPU cache may still hold pre-submit zeros until invalidated. */
      for (off = 0; off < n; off += 64)
        _mm_clflush((char *)h_buf + off);

      if (!mc_test_verify_pattern((uint32_t *)h_buf, n / 4u, FILL_PATTERN,
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

    if (ns < best_ns)  best_ns  = ns;
    if (ns > worst_ns) worst_ns = ns;
    sum_ns += ns;

    printf("  iter %3d: %s=%7.1f us (%5.2f GB/s)\n",
           it + 1, dir_label, ns / 1e3, mc_test_bandwidth_gbps(n, ns));
  }

  printf("summary: best=%.1f us mean=%.1f us worst=%.1f us "
         "(%d iters, %u bytes, dir=%s)\n",
         best_ns / 1e3, (sum_ns / args.iters) / 1e3, worst_ns / 1e3,
         args.iters, n, dir_label);

  printf("PASS: %s transfer authored end-to-end by an SM thread.\n",
         args.h2d ? "DRAM->HBM" : "HBM->DRAM");
  printf("       mc_memcpy(MC_XFER_SM) ran the SM-authored CE submission\n");
  printf("       (pushbuffer + GPFIFO entry + USERD GP_PUT + BAR1 doorbell),\n");
  printf("       and PBDMA observed all four and ran the carrier copy.\n");

  mc_free(ctx, h_verify);
  mc_free(ctx, h_buf);
  mc_free(ctx, d_buf);
  mc_fini(ctx);
  return 0;
}
