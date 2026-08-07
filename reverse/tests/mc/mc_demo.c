/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_demo.c — user-facing demo of the libmc API.
 *
 * Shares its CLI (--size, --iters, --help) with cuda_reference so the two
 * can be run side by side and their numbers compared directly.
 *
 * This file touches no RM ioctls, no UVM ioctls, and no /dev/nvidia*
 * node directly — it only uses the 7-function libmc API.  That is the
 * whole point: all of that machinery is now behind the library.
 *
 * Build: `make mc_demo` in reverse/.  Run (on a host with a Hopper
 * GPU + the instrumented driver from this tree) with:
 *   sudo ./bin/mc_demo [--size 128M] [--iters 10]
 *
 * Exit codes: 0 = PASS, 1 = verify failure, 2 = CLI error, 3 = init failure.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mc.h"
#include "mc_test_helpers.h"

#define DEFAULT_TRANSFER_SIZE (256ULL * 1024ULL * 1024ULL)
#define DEFAULT_ITERS         10
#define FILL_PATTERN          0xDEADBEEFu
#define GARBLE_PATTERN        0xCAFEBABEu

#define MC_CHECK(call)                                            \
  do                                                              \
  {                                                               \
    mc_status_t _e = (call);                                      \
    if (_e != MC_OK)                                              \
    {                                                             \
      fprintf(stderr, "MC error at %s:%d\n", __FILE__, __LINE__); \
      exit_code = _e;                                             \
      goto out;                                                   \
    }                                                             \
  } while (0)

int main(int argc, char **argv)
{
  mc_test_args_t args = {
      .n_bytes = DEFAULT_TRANSFER_SIZE,
      .iters   = DEFAULT_ITERS,
  };

  mc_ctx_t *ctx       = NULL;
  uint32_t *d_buf     = NULL;
  uint32_t *h_buf     = NULL;
  int       ok        = 1;
  int       exit_code = 0;

  switch (mc_test_parse_args(argc, argv, &args))
  {
  case MC_TEST_ARGS_OK:    break;
  case MC_TEST_ARGS_HELP:  return 0;
  case MC_TEST_ARGS_ERROR: return 2;
  }

  const size_t nw = args.n_bytes / sizeof(uint32_t);

  printf("mc_demo: size=%llu bytes (%llu MiB), iters=%d\n",
         (unsigned long long)args.n_bytes,
         (unsigned long long)(args.n_bytes >> 20), args.iters);

  /* ── Library init ──────────────────────────────────────────────────── */
  MC_CHECK(mc_init(&ctx));

  d_buf = mc_malloc_device(ctx, args.n_bytes);
  h_buf = mc_malloc_host(ctx, args.n_bytes);
  if (!d_buf || !h_buf)
  {
    fprintf(stderr, "allocation failed\n");
    exit_code = 3;
    goto out;
  }

  /* ── Seed device with FILL_PATTERN via one H2D ──────────────────────── */
  mc_test_seed_pattern(h_buf, nw, FILL_PATTERN);
  MC_CHECK(mc_memcpy_h2d(ctx, d_buf, h_buf, args.n_bytes));

  /* Timed D2H loop.  Best + mean latency across iters runs. */
  uint64_t best_ns = UINT64_MAX, sum_ns = 0, ns, t0;
  size_t   first_bad;
  int      it;

  for (it = 0; it < args.iters; it++)
  {
    /* Overwrite h_buf with a distinct pattern so a silently no-op D2H would
     * leave GARBLE behind instead of spuriously passing on stale FILL. */
    mc_test_seed_pattern(h_buf, nw, GARBLE_PATTERN);

    t0 = mc_test_now_ns();
    MC_CHECK(mc_memcpy_d2h(ctx, h_buf, d_buf, args.n_bytes));
    ns = mc_test_now_ns() - t0;

    if (ns < best_ns) best_ns = ns;
    sum_ns += ns;

    if (!mc_test_verify_pattern(h_buf, nw, FILL_PATTERN, &first_bad))
    {
      fprintf(stderr,
              "verify FAIL at iter=%d dword=%zu: got 0x%08x expected 0x%08x\n",
              it, first_bad, h_buf[first_bad], FILL_PATTERN);
      ok = 0;
      break;
    }
  }

  printf("D2H %llu MiB x %d iters:\n",
         (unsigned long long)(args.n_bytes >> 20), args.iters);
  printf("  Peak: %.2f GB/s (%.2f ms)\n",
         mc_test_bandwidth_gbps(args.n_bytes, best_ns), best_ns / 1e6);
  printf("  Mean: %.2f GB/s (%.2f ms)\n",
         mc_test_bandwidth_gbps(args.n_bytes, sum_ns / args.iters),
         (sum_ns / args.iters) / 1e6);
  if (ok)
    printf("PASS: verification\n");
  else
    exit_code = 1;

out:
  mc_free(ctx, d_buf);
  mc_free(ctx, h_buf);
  mc_fini(ctx);
  return exit_code;
}
