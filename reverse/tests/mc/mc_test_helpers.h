/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_test_helpers.h — boilerplate helpers shared across the
 * tests/mc/ test programs.
 *
 *   - --size + --iters CLI parsing through a single args struct
 *   - dword-aligned pattern seed + verify
 *
 * #include this in any test that wants the helpers; the Makefile
 * automatically links mc_test_helpers.c (any tests/mc/ source ending
 * in _helpers.c is treated as a shared TU and linked into every test,
 * not built as a stand-alone binary).
 */
#ifndef MC_TEST_HELPERS_H
#define MC_TEST_HELPERS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Result of mc_test_parse_args.  OK is 0 so `if (parse_args(...))` stays
 * a valid quick-fail check, matching the mc_status_t convention. */
typedef enum {
    MC_TEST_ARGS_OK    = 0,  /* args valid, proceed */
    MC_TEST_ARGS_ERROR = 1,  /* CLI error; helper printed message to stderr */
    MC_TEST_ARGS_HELP  = 2,  /* -h/--help; helper printed flag table to stdout */
} mc_test_args_status_t;

/* Canonical CLI shape for tests/mc/ programs.  Every test in this
 * directory uses the same flag surface (--size, --iters, --h2d); individual
 * tests are free to ignore fields they don't care about (chain demos
 * read n_bytes only). */
typedef struct {
    size_t n_bytes;   /* --size; bounds: > 0, <= 4 GiB, dword-aligned */
    int    iters;     /* --iters; bounds: 1 .. INT_MAX */
    int    h2d;       /* --h2d; should the xfer be host to device */
} mc_test_args_t;

/* Parse argv into *args.  Caller seeds *args with its desired defaults
 * before calling — the parser only overwrites fields the user actually
 * passed on the command line.  The flag table + exit-code legend
 * printed on -h/--help is owned by this helper, so each test gets
 * uniform usage output for free. */
mc_test_args_status_t mc_test_parse_args(int argc, char **argv,
                                         mc_test_args_t *args);

/* Fill `n_dwords` dwords starting at `buf` with `pattern`.  Caller-
 * checked: buf must be 4-byte-aligned and at least n_dwords * 4 bytes
 * large. */
void mc_test_seed_pattern(uint32_t *buf, size_t n_dwords, uint32_t pattern);

/* Verify every dword in [buf, buf + n_dwords) equals `pattern`.  On
 * mismatch, *first_bad is set to the dword index of the first bad value
 * (if first_bad is non-NULL) and the function returns 0.  On full
 * match, returns 1. */
int mc_test_verify_pattern(const uint32_t *buf, size_t n_dwords,
                           uint32_t pattern, size_t *first_bad);

/* Monotonic-clock reading in nanoseconds.  Used for measuring elapsed
 * time across an operation:
 *   t0 = mc_test_now_ns();  ...  ns = mc_test_now_ns() - t0;
 */
uint64_t mc_test_now_ns(void);

/* Convert (bytes, elapsed nanoseconds) to GB/s (10^9 bytes/s).
 * Returns 0.0 if ns_elapsed == 0 to avoid divide-by-zero. */
double mc_test_bandwidth_gbps(uint64_t bytes, uint64_t ns_elapsed);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_TEST_HELPERS_H */
