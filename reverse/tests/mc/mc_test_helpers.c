/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_test_helpers.c — implementation of mc_test_helpers.h.
 *
 * The --size suffix parser (K/M/G + b/B) was lifted from mc_demo's
 * bespoke parse_size and generalised; the rest is glue around the
 * canonical mc_test_args_t shape so every test in this directory has
 * the same CLI surface and the same help text.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mc_test_helpers.h"

/* Returns 1 on success, 0 on parse error.  Same suffix rules as the
 * long-standing mc_demo parser. */
static int parse_size_value(const char *s, uint64_t *out)
{
  char    *end;
  uint64_t v;

  if (s == NULL || *s == '\0') return 0;
  errno = 0;
  v = (uint64_t)strtoull(s, &end, 0);
  if (errno != 0 || end == s) return 0;

  switch (*end)
  {
  case '\0': break;
  case 'k': case 'K': v <<= 10; end++; break;
  case 'm': case 'M': v <<= 20; end++; break;
  case 'g': case 'G': v <<= 30; end++; break;
  default: return 0;
  }
  if (*end == 'b' || *end == 'B') end++;
  if (*end != '\0') return 0;
  *out = v;
  return 1;
}

static int parse_int_value(const char *s, long *out)
{
  char *end;
  long  v;

  if (s == NULL || *s == '\0') return 0;
  errno = 0;
  v = strtol(s, &end, 0);
  if (errno != 0 || end == s || *end != '\0') return 0;
  *out = v;
  return 1;
}

static void print_usage(const char *prog)
{
  printf(
      "usage: %s [--size SIZE] [--iters N] [--h2d] [--help]\n"
      "\n"
      "  --size SIZE   Transfer size; K/M/G suffixes accepted.  Must be\n"
      "                > 0, dword-aligned, <= 4 GiB.\n"
      "  --iters N     Number of iterations (1 <= N <= %d).  Tests that\n"
      "                don't iterate ignore this field.\n"
      "  --h2d         Run the timed transfer host->device instead of the\n"
      "                default device->host.  Tests that aren't directional\n"
      "                ignore this field.\n"
      "  --wc          Allocate the host buffer write-combined\n"
      "                (mc_malloc_host_wc).  Requires --h2d.\n"
      "  -h, --help    Show this message.\n"
      "\n"
      "Exit codes (by convention): 0 pass, 1 verify FAIL, 2 CLI error,\n"
      "3 init failure.  Individual tests may use additional codes.\n",
      prog, INT_MAX);
}

mc_test_args_status_t mc_test_parse_args(int argc, char **argv,
                                         mc_test_args_t *args)
{
  int      i;
  uint64_t sz;
  long     it;

  for (i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      print_usage(argv[0]);
      return MC_TEST_ARGS_HELP;
    }
    if (!strcmp(argv[i], "--size") && i + 1 < argc)
    {
      if (!parse_size_value(argv[i + 1], &sz))
      {
        fprintf(stderr, "%s: bad --size '%s' (suffix K/M/G accepted)\n",
                argv[0], argv[i + 1]);
        return MC_TEST_ARGS_ERROR;
      }
      if (sz == 0 || sz > 0xFFFFFFFFu || (sz & 3))
      {
        fprintf(stderr,
                "%s: --size %llu out of range (must be > 0, "
                "<= 4 GiB, dword-aligned)\n",
                argv[0], (unsigned long long)sz);
        return MC_TEST_ARGS_ERROR;
      }
      args->n_bytes = (size_t)sz;
      i++;
      continue;
    }
    if (!strcmp(argv[i], "--iters") && i + 1 < argc)
    {
      if (!parse_int_value(argv[i + 1], &it))
      {
        fprintf(stderr, "%s: bad --iters '%s'\n", argv[0], argv[i + 1]);
        return MC_TEST_ARGS_ERROR;
      }
      if (it < 1 || it > INT_MAX)
      {
        fprintf(stderr, "%s: --iters %ld out of range (1 .. %d)\n",
                argv[0], it, INT_MAX);
        return MC_TEST_ARGS_ERROR;
      }
      args->iters = (int)it;
      i++;
      continue;
    }
    if (!strcmp(argv[i], "--h2d"))
    {
      args->h2d = 1;
      continue;
    }
    if (!strcmp(argv[i], "--wc"))
    {
      args->wc = 1;
      continue;
    }
    fprintf(stderr, "%s: unknown argument '%s'\n", argv[0], argv[i]);
    return MC_TEST_ARGS_ERROR;
  }
  return MC_TEST_ARGS_OK;
}

void mc_test_seed_pattern(uint32_t *buf, size_t n_dwords, uint32_t pattern)
{
  size_t i;
  for (i = 0; i < n_dwords; ++i) buf[i] = pattern;
}

int mc_test_verify_pattern(const uint32_t *buf, size_t n_dwords,
                           uint32_t pattern, size_t *first_bad)
{
  size_t i;
  for (i = 0; i < n_dwords; ++i)
  {
    if (buf[i] != pattern)
    {
      if (first_bad) *first_bad = i;
      return 0;
    }
  }
  return 1;
}

uint64_t mc_test_now_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

double mc_test_bandwidth_gbps(uint64_t bytes, uint64_t ns_elapsed)
{
  if (ns_elapsed == 0) return 0.0;
  return (double)bytes / (double)ns_elapsed;
}
