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
 * node directly — it only uses the public libmc API.  That is the whole
 * point: all of that machinery is now behind the library.
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

  mc_ctx_t   *ctx       = NULL;
  uint32_t   *d_buf     = NULL;
  uint32_t   *h_buf     = NULL;
  uint32_t   *h_verify  = NULL;   /* H2D-only readback target */
  int         ok        = 1;
  int         exit_code = 0;
  const char *dir_label;

  switch (mc_test_parse_args(argc, argv, &args))
  {
  case MC_TEST_ARGS_OK:    break;
  case MC_TEST_ARGS_HELP:  return 0;
  case MC_TEST_ARGS_ERROR: return 2;
  }

  dir_label       = args.h2d ? "H2D" : "D2H";
  const size_t nw = args.n_bytes / sizeof(uint32_t);

  printf("mc_demo: size=%llu bytes (%llu MiB), iters=%d, dir=%s\n",
         (unsigned long long)args.n_bytes,
         (unsigned long long)(args.n_bytes >> 20), args.iters, dir_label);

  /* ── Library init ──────────────────────────────────────────────────── */
  MC_CHECK(mc_init(&ctx));

  d_buf = mc_malloc_device(ctx, args.n_bytes, MC_VAS_UVM);
  h_buf = mc_malloc_host  (ctx, args.n_bytes, MC_VAS_UVM);
  if (!d_buf || !h_buf)
  {
    fprintf(stderr, "allocation failed\n");
    exit_code = 3;
    goto out;
  }

  /* For --h2d we need a separate host-side readback target so the
   * timed window only contains the H2D mc_memcpy itself.  Allocate
   * lazily so the D2H default path keeps its existing alloc footprint. */
  if (args.h2d)
  {
    h_verify = mc_malloc_host(ctx, args.n_bytes, MC_VAS_UVM);
    if (!h_verify)
    {
      fprintf(stderr, "h_verify allocation failed\n");
      exit_code = 3;
      goto out;
    }
  }

  /* ── Seed once.  D2H seeds the device side via one untimed H2D so
   *    the timed loop can read FILL back.  H2D seeds the host side
   *    in CPU code; the timed loop's H2D copies move that pattern to
   *    the device, and an untimed D2H read-back per iter verifies. */
  if (args.h2d)
    mc_test_seed_pattern(h_buf, nw, FILL_PATTERN);
  else
  {
    mc_test_seed_pattern(h_buf, nw, FILL_PATTERN);
    MC_CHECK(mc_memcpy(ctx, d_buf, h_buf, args.n_bytes, MC_XFER_HOST));
  }

  /* Timed loop.  Best + mean latency across iters runs.  Only the
   * unidirectional copy is timed; the verify read-back under --h2d
   * is explicitly outside the t0/ns measurement. */
  uint64_t best_ns = UINT64_MAX, sum_ns = 0, ns, t0;
  size_t   first_bad;
  int      it;

  for (it = 0; it < args.iters; it++)
  {
    if (args.h2d)
    {
      /* Stomp the device-side readback target so a silent no-op H2D
       * is detectable (the readback must come from h_buf via d_buf). */
      mc_test_seed_pattern(h_verify, nw, GARBLE_PATTERN);

      t0 = mc_test_now_ns();
      MC_CHECK(mc_memcpy(ctx, d_buf, h_buf, args.n_bytes, MC_XFER_HOST));
      ns = mc_test_now_ns() - t0;

      /* Untimed read-back into the verify buffer. */
      MC_CHECK(mc_memcpy(ctx, h_verify, d_buf, args.n_bytes, MC_XFER_HOST));

      if (!mc_test_verify_pattern(h_verify, nw, FILL_PATTERN, &first_bad))
      {
        fprintf(stderr,
                "verify FAIL at iter=%d dword=%zu: got 0x%08x expected 0x%08x\n",
                it + 1, first_bad, h_verify[first_bad], FILL_PATTERN);
        ok = 0;
        break;
      }
    }
    else
    {
      /* Overwrite h_buf with a distinct pattern so a silently no-op
       * D2H would leave GARBLE behind instead of spuriously passing
       * on stale FILL. */
      mc_test_seed_pattern(h_buf, nw, GARBLE_PATTERN);

      t0 = mc_test_now_ns();
      MC_CHECK(mc_memcpy(ctx, h_buf, d_buf, args.n_bytes, MC_XFER_HOST));
      ns = mc_test_now_ns() - t0;

      if (!mc_test_verify_pattern(h_buf, nw, FILL_PATTERN, &first_bad))
      {
        fprintf(stderr,
                "verify FAIL at iter=%d dword=%zu: got 0x%08x expected 0x%08x\n",
                it + 1, first_bad, h_buf[first_bad], FILL_PATTERN);
        ok = 0;
        break;
      }
    }

    if (ns < best_ns) best_ns = ns;
    sum_ns += ns;
  }

  printf("%s %llu MiB x %d iters:\n", dir_label,
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
  mc_free(ctx, h_verify);
  mc_fini(ctx);
  return exit_code;
}
