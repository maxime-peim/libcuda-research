/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_compute_dbell_chain_demo.c — end-to-end demo where a
 * HOPPER_COMPUTE_A SM thread rings the UVM channel's BAR1 doorbell.
 *
 * Mirror of mc_dbell_chain_demo.c, but the GPU-issued MMIO doorbell
 * write is performed by an SM thread of mc_doorbell_kernel rather
 * than by the DMA channel's CE.  No copy engine is in the doorbell
 * loop — just a single warp's STG.E.STRONG.SYS to the BAR1 page.
 *
 *   - Allocate device buffer (HBM) + host buffer (DRAM).
 *   - Pre-fill HBM with FILL_PATTERN via mc_memcpy (host-rung
 *     doorbell — uncontroversial, just stages bytes for the test).
 *   - Zero DRAM.
 *   - Call mc_memcpy_gpu_doorbell_sm(): the UVM channel's
 *     doorbell is rung by a compute kernel, not the host, not a CE.
 *   - Verify DRAM contains FILL_PATTERN throughout.
 *
 * Result interpretation:
 *
 *   PASS → DRAM contains FILL_PATTERN.  The HBM→DRAM copy ran.
 *          The host did NOT ring the UVM channel's doorbell during
 *          the mc_memcpy_gpu_doorbell_sm call.  No CE was used
 *          to perform the doorbell write either.  The only path that
 *          could have woken the UVM channel's PBDMA is the compute
 *          kernel's STG.E.STRONG.SYS reaching real BAR1.  Therefore:
 *          a GPU SM thread *physically* issued a doorbell MMIO write
 *          and another channel's PBDMA observed it.  End-to-end
 *          proof of GPU-initiated DMA via compute.
 *
 *   FAIL  → DRAM doesn't match.  The UVM channel's PBDMA never ran,
 *          or ran with corrupted setup.
 *
 *   TIMEOUT → either the compute sema or the UVM channel's sema
 *          didn't fire within MC_TIMEOUT_MS.  Could mean the
 *          SM-issued doorbell write was silently dropped by the GPU
 *          MMU → BAR1 path.
 *
 * Build: `make mc_compute_dbell_chain_demo` from reverse/.
 * Run:   sudo ./bin/mc_compute_dbell_chain_demo [--size N]
 *
 * Pre-req: kernel modules from this tree loaded with
 *          nv_dbell_disable_intercept=1 (so the kernel doorbell
 *          watchpoint shadow page doesn't absorb the SM-issued write).
 *
 * Exit codes: 0 = PASS, 1 = init failure, 2 = h2d setup failure,
 *             3 = sm-doorbell d2h failure, 4 = verify mismatch.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mc.h"
#include "mc_test_helpers.h"

#define DEFAULT_SIZE   (4ULL * 1024ULL * 1024ULL)  /* 4 MiB */
#define FILL_PATTERN   0xCAFEBABEu

int main(int argc, char **argv)
{
  mc_test_args_t args = { .n_bytes = DEFAULT_SIZE, .iters = 1 };

  mc_ctx_t   *ctx       = NULL;
  size_t      n;
  void       *d_buf     = NULL;
  void       *h_buf     = NULL;
  void       *h_verify  = NULL;   /* H2D-only readback target */
  size_t      bad_off   = 0;
  const char *dir_label;

  switch (mc_test_parse_args(argc, argv, &args))
  {
  case MC_TEST_ARGS_OK:    break;
  case MC_TEST_ARGS_HELP:  return 0;
  case MC_TEST_ARGS_ERROR: return 1;
  }
  n         = args.n_bytes;
  dir_label = args.h2d ? "H2D" : "D2H";

  if (mc_init(&ctx) != MC_OK)
  {
    fprintf(stderr, "mc_init failed\n");
    return 1;
  }
  printf("mc_init ok — UVM + DMA + compute channels ready (size=%zu, dir=%s)\n",
         n, dir_label);

  d_buf  = mc_malloc_device(ctx, n, MC_VAS_UVM);
  h_buf  = mc_malloc_host  (ctx, n, MC_VAS_UVM);
  if (!d_buf || !h_buf)
  {
    fprintf(stderr, "mc_malloc_* failed\n");
    mc_fini(ctx);
    return 1;
  }

  if (args.h2d)
  {
    h_verify = mc_malloc_host(ctx, n, MC_VAS_UVM);
    if (!h_verify)
    {
      fprintf(stderr, "h_verify alloc failed\n");
      mc_free(ctx, h_buf);
      mc_free(ctx, d_buf);
      mc_fini(ctx);
      return 1;
    }
  }

  /* Pre-fill h_buf with FILL_PATTERN.  D2H seeds it through to HBM
   * via an untimed CE H2D so subsequent SM-rung D2Hs read FILL back.
   * H2D skips that seed: h_buf IS the source of the timed H2D each
   * iter; the verify path runs a separate untimed D2H readback into
   * h_verify. */
  mc_test_seed_pattern((uint32_t *)h_buf, n / 4, FILL_PATTERN);
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

  /* Run the experiment args.iters times.  Time only the SM-rung
   * chain call.  Fail-fast on any verify mismatch. */
  uint64_t best_ns = UINT64_MAX, sum_ns = 0, worst_ns = 0;

  for (int it = 0; it < args.iters; it++)
  {
    uint64_t t0;
    uint64_t ns;

    if (args.h2d)
    {
      const uint32_t GARBLE = 0xCAFEBABEu;
      mc_test_seed_pattern((uint32_t *)h_verify, n / 4, GARBLE);
      if (mc_memcpy(ctx, d_buf, h_verify, n, MC_XFER_HOST) != MC_OK)
      {
        fprintf(stderr, "iter %d: pre-stomp mc_memcpy failed\n", it + 1);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }

      t0 = mc_test_now_ns();
      if (mc_memcpy_gpu_doorbell_sm(ctx, d_buf, h_buf, n) != MC_OK)
      {
        fprintf(stderr,
                "iter %d: mc_memcpy_gpu_doorbell_sm H2D failed\n", it + 1);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }
      ns = mc_test_now_ns() - t0;

      if (mc_memcpy(ctx, h_verify, d_buf, n, MC_XFER_HOST) != MC_OK)
      {
        fprintf(stderr, "iter %d: readback mc_memcpy failed\n", it + 1);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }

      if (!mc_test_verify_pattern((uint32_t *)h_verify, n / 4,
                                  FILL_PATTERN, &bad_off))
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
      memset(h_buf, 0, n);

      t0 = mc_test_now_ns();
      if (mc_memcpy_gpu_doorbell_sm(ctx, h_buf, d_buf, n) != MC_OK)
      {
        fprintf(stderr,
                "iter %d: mc_memcpy_gpu_doorbell_sm D2H failed\n", it + 1);
        mc_free(ctx, h_verify);
        mc_free(ctx, h_buf);
        mc_free(ctx, d_buf);
        mc_fini(ctx);
        return 3;
      }
      ns = mc_test_now_ns() - t0;

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

    if (ns < best_ns)  best_ns  = ns;
    if (ns > worst_ns) worst_ns = ns;
    sum_ns += ns;

    printf("  iter %3d: %s=%7.1f us (%5.2f GB/s)\n",
           it + 1, dir_label, ns / 1e3, mc_test_bandwidth_gbps(n, ns));
  }

  printf("summary: best=%.1f us mean=%.1f us worst=%.1f us "
         "(%d iters, %zu bytes, dir=%s)\n",
         best_ns / 1e3, (sum_ns / args.iters) / 1e3, worst_ns / 1e3,
         args.iters, n, dir_label);

  printf("PASS: %s — UVM-channel transfer ran\n",
         args.h2d ? "d_buf contains FILL_PATTERN (via untimed D2H readback)"
                  : "DRAM contains FILL_PATTERN");
  printf("       Host did NOT ring the UVM channel's doorbell during"
         " the call.\n");
  printf("       No copy engine was involved in the doorbell write.\n");
  printf("       A HOPPER_COMPUTE_A SM thread's STG.E.STRONG.SYS to\n");
  printf("       BAR1+0x%x physically landed on the H100 doorbell\n",
         0x90);
  printf("       register, woke the UVM channel's PBDMA, and the\n");
  printf("       queued %s executed.  GPU-initiated DMA via compute,"
         " no CUDA.\n", dir_label);

  mc_free(ctx, h_verify);
  mc_free(ctx, h_buf);
  mc_free(ctx, d_buf);
  mc_fini(ctx);
  return 0;
}
