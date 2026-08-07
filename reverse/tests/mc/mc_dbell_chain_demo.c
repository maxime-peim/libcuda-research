/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_dbell_chain_demo.c — end-to-end-verifiable demo that the GPU
 * itself rings a CE channel's doorbell.
 *
 *   - Allocate device buffer (HBM) + host buffer (DRAM).
 *   - Pre-fill HBM with FILL_PATTERN via mc_memcpy_h2d (host-rung
 *     doorbell — uncontroversial, just stages bytes for the test).
 *   - Zero DRAM.
 *   - Call mc_memcpy_d2h_gpu_doorbell_ce(): the UVM channel's
 *     doorbell is rung by the GPU's CE on the DMA channel, not the
 *     host.
 *   - Verify DRAM contains FILL_PATTERN throughout.
 *
 * Result interpretation:
 *
 *   PASS → DRAM contains FILL_PATTERN.  The HBM→DRAM copy ran.
 *          The host did NOT ring the UVM channel's doorbell during
 *          the call.  The only path that could have woken the UVM
 *          channel's PBDMA is the DMA-channel CE's MMIO write
 *          reaching real BAR1.  Therefore: a GPU-side MMIO store
 *          *physically* landed on the H100 BAR1 register, observable
 *          as DRAM bytes — outside libcuda, outside the kernel
 *          watchpoint shadow.  This is the end-to-end proof of the
 *          GPU MMU → BAR1 path.
 *
 *   FAIL  → DRAM doesn't match.  The UVM channel's PBDMA never ran
 *          (or ran with corrupted setup).  The DMA-channel CE either
 *          didn't execute, or its write didn't land on BAR1.
 *
 *   TIMEOUT → one of the two semaphores never fired.  Could mean
 *          the GPU MMU → BAR1 path silently dropped the doorbell
 *          write (primary's PBDMA never woke).
 *
 * Build: `make mc_dbell_chain_demo` from reverse/.
 * Run:   sudo ./bin/mc_dbell_chain_demo [--size N]
 *
 * Exit codes: 0 = PASS, 1 = init failure, 2 = h2d setup failure,
 *             3 = d2h-via-doorbell failure, 4 = verify mismatch.
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

  mc_ctx_t *ctx     = NULL;
  size_t    n;
  void     *d_buf   = NULL;
  void     *h_buf   = NULL;
  size_t    bad_off = 0;

  switch (mc_test_parse_args(argc, argv, &args))
  {
  case MC_TEST_ARGS_OK:    break;
  case MC_TEST_ARGS_HELP:  return 0;
  case MC_TEST_ARGS_ERROR: return 1;
  }
  n = args.n_bytes;

  if (mc_init(&ctx) != MC_OK)
  {
    fprintf(stderr, "mc_init failed\n");
    return 1;
  }
  printf("mc_init ok — UVM + DMA channels ready (size=%zu)\n", n);

  d_buf  = mc_malloc_device(ctx, n);
  h_buf  = mc_malloc_host(ctx, n);
  if (!d_buf || !h_buf)
  {
    fprintf(stderr, "mc_malloc_* failed\n");
    mc_fini(ctx);
    return 1;
  }

  /* Pre-fill h_seed with FILL_PATTERN, H2D it into HBM via the
   * normal (host-rung) UVM channel.  Stages the bytes we'll
   * later D2H out via the GPU-rung doorbell. */
  mc_test_seed_pattern((uint32_t *)h_buf, n / 4, FILL_PATTERN);
  if (mc_memcpy_h2d(ctx, d_buf, h_buf, n) != MC_OK)
  {
    fprintf(stderr, "mc_memcpy_h2d (seed HBM) failed\n");
    mc_fini(ctx);
    return 2;
  }

  /* Run the experiment args.iters times.  HBM stays loaded with
   * FILL_PATTERN across iterations (nothing else writes d_buf), so
   * each iter only needs to: zero DRAM target, run the GPU-doorbell
   * D2H, verify.  We time only the D2H call — that's the operation
   * whose latency is interesting.  Fail-fast on any verify mismatch. */
  uint64_t best_ns = UINT64_MAX, sum_ns = 0, worst_ns = 0;

  for (int it = 0; it < args.iters; it++)
  {
    memset(h_buf, 0, n);

    uint64_t t0 = mc_test_now_ns();
    if (mc_memcpy_d2h_gpu_doorbell_ce(ctx, h_buf, d_buf, n) != MC_OK)
    {
      fprintf(stderr,
              "iter %d: mc_memcpy_d2h_gpu_doorbell_ce failed\n", it);
      mc_fini(ctx);
      return 3;
    }
    uint64_t ns = mc_test_now_ns() - t0;

    if (!mc_test_verify_pattern((uint32_t *)h_buf, n / 4, FILL_PATTERN,
                                &bad_off))
    {
      fprintf(stderr,
              "iter %d VERIFY FAIL: h_buf[%zu] = 0x%08x (expected 0x%08x)\n",
              it, bad_off, ((uint32_t *)h_buf)[bad_off], FILL_PATTERN);
      mc_fini(ctx);
      return 4;
    }

    if (ns < best_ns)  best_ns  = ns;
    if (ns > worst_ns) worst_ns = ns;
    sum_ns += ns;

    printf("  iter %3d: d2h=%7.1f us (%5.2f GB/s)\n",
           it, ns / 1e3, mc_test_bandwidth_gbps(n, ns));
  }

  printf("summary: best=%.1f us mean=%.1f us worst=%.1f us "
         "(%d iters, %zu bytes)\n",
         best_ns / 1e3, (sum_ns / args.iters) / 1e3, worst_ns / 1e3,
         args.iters, n);

  printf("PASS: DRAM contains FILL_PATTERN — UVM-channel D2H ran\n");
  printf("       Host did NOT ring the UVM channel's doorbell during"
         " the call.\n");
  printf("       The DMA channel's CE-issued MMIO write to\n");
  printf("       BAR1+0x%x physically landed on the H100 doorbell\n",
         0x90);
  printf("       register, woke the UVM channel's PBDMA, and the\n");
  printf("       queued D2H executed.\n");

  mc_free(ctx, h_buf);
  mc_free(ctx, d_buf);
  mc_fini(ctx);
  return 0;
}
