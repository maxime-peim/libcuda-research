/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_host_register_bench.c - break down mc_host_register latency.
 *
 * This is the mc analogue of tests/cuda/cuda_host_register_bench.cu.
 * It intentionally uses malloc-backed host ranges and compares the
 * steady-state mc registration cost against two CPU-side operations
 * which also make anonymous pages resident:
 *
 *   malloc_touch
 *       malloc, then time one write per page.  This is first-touch page
 *       population without any locking or mc work.
 *
 *   malloc_mlock
 *       malloc, then time mlock().  This faults pages in and marks them
 *       unevictable from userspace.  It is not equivalent to the NVIDIA
 *       driver's DMA pin/FOLL_PIN path, but it is the closest standard
 *       userspace "pin it myself" operation.
 *
 *   malloc_mc_host_register
 *       malloc, then time mc_host_register() on an untouched range.  This
 *       includes Linux page population plus mc's RM OS-descriptor and
 *       UVM external-range mapping work.
 *
 *   prefault_mc_host_register
 *       malloc, touch pages outside the timed window, then time
 *       mc_host_register().  This removes first-touch page faults.
 *
 *   mlock_mc_host_register
 *       malloc, mlock outside the timed window, then time
 *       mc_host_register().  This estimates the residual RM/UVM registration
 *       work after ordinary Linux residency/locking has already happened.
 *
 * With --alignment SIZE, each sample mallocs a larger backing range, uses a
 * SIZE-aligned interior pointer for the requested range, and frees the
 * original malloc pointer after the sample.  Without --alignment, samples use
 * the raw malloc pointer.
 *
 * Build: cd reverse && make mc_host_register_bench
 * Run:   sudo ./bin/mc_host_register_bench --size 4M --iters 30
 */

#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "mc.h"
#include "mc_test_helpers.h"

#define DEFAULT_SIZE  (4ULL * 1024ULL * 1024ULL)
#define DEFAULT_ITERS 30

typedef struct {
  uint64_t size;
  uint64_t alignment;
  int      iters;
} options_t;

typedef enum {
  PHASE_TOUCH,
  PHASE_MLOCK,
  PHASE_MC_COLD,
  PHASE_MC_PREFAULT,
  PHASE_MC_AFTER_MLOCK,
} phase_t;

typedef struct {
  const char *name;
  phase_t     phase;
} phase_desc_t;

typedef struct {
  double *usec;
  size_t  count;
  char    first_error[256];
  int     failures;
} result_t;

static const phase_desc_t k_phases[] = {
    {"malloc_touch", PHASE_TOUCH},
    {"malloc_mlock", PHASE_MLOCK},
    {"malloc_mc_host_register", PHASE_MC_COLD},
    {"prefault_mc_host_register", PHASE_MC_PREFAULT},
    {"mlock_mc_host_register", PHASE_MC_AFTER_MLOCK},
};

static long page_size(void)
{
  static long cached = 0;
  if (cached == 0)
  {
    cached = sysconf(_SC_PAGESIZE);
    if (cached <= 0)
      cached = 4096;
  }
  return cached;
}

static uint64_t pages_in_range(uint64_t bytes)
{
  const uint64_t page = (uint64_t)page_size();
  return (bytes + page - 1) / page;
}

static int is_power_of_two_u64(uint64_t v)
{
  return v != 0 && (v & (v - 1)) == 0;
}

static int parse_size(const char *s, uint64_t *out)
{
  char    *end;
  uint64_t v;
  unsigned shift = 0;

  if (s == NULL || *s == '\0')
    return 0;

  errno = 0;
  v = (uint64_t)strtoull(s, &end, 0);
  if (errno != 0 || end == s)
    return 0;

  switch (*end)
  {
  case '\0':
    break;
  case 'k':
  case 'K':
    shift = 10;
    end++;
    break;
  case 'm':
  case 'M':
    shift = 20;
    end++;
    break;
  case 'g':
  case 'G':
    shift = 30;
    end++;
    break;
  default:
    return 0;
  }

  if (*end == 'b' || *end == 'B')
    end++;
  if (*end != '\0')
    return 0;
  if (shift != 0 && v > (~0ULL >> shift))
    return 0;

  *out = v << shift;
  return 1;
}

static void usage(const char *prog)
{
  fprintf(stderr,
          "usage: %s [OPTIONS]\n"
          "\n"
          "  --size SIZE     registration range size. K/M/G suffixes accepted.\n"
          "                  Default: 4M.\n"
          "  --alignment SZ  align each sample's malloc-backed pointer.\n"
          "                  K/M/G suffixes accepted. Default: 0.\n"
          "  --iters N       samples per phase. Default: 30.\n"
          "  -h, --help      show this message.\n"
          "\n"
          "mc is initialized and a 1-page registration is warmed up\n"
          "before timing. Each sample uses a fresh malloc/free lifecycle.\n"
          "With --alignment, samples use an aligned interior pointer from a\n"
          "larger malloc backing allocation.\n",
          prog);
}

static int parse_args(int argc, char **argv, options_t *opts)
{
  int i;

  opts->size      = DEFAULT_SIZE;
  opts->alignment = 0;
  opts->iters     = DEFAULT_ITERS;

  for (i = 1; i < argc; ++i)
  {
    const char *arg = argv[i];

    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
    {
      usage(argv[0]);
      exit(0);
    }
    if (strcmp(arg, "--size") == 0)
    {
      if (++i >= argc || !parse_size(argv[i], &opts->size))
      {
        fprintf(stderr, "bad --size\n");
        return 0;
      }
      continue;
    }
    if (strcmp(arg, "--alignment") == 0)
    {
      if (++i >= argc || !parse_size(argv[i], &opts->alignment))
      {
        fprintf(stderr, "bad --alignment\n");
        return 0;
      }
      continue;
    }
    if (strcmp(arg, "--iters") == 0)
    {
      char *end;
      long  v;

      if (++i >= argc)
      {
        fprintf(stderr, "--iters needs an argument\n");
        return 0;
      }
      errno = 0;
      v = strtol(argv[i], &end, 0);
      if (errno != 0 || end == argv[i] || *end != '\0'
          || v <= 0 || v > INT_MAX)
      {
        fprintf(stderr, "bad --iters\n");
        return 0;
      }
      opts->iters = (int)v;
      continue;
    }

    fprintf(stderr, "unknown arg '%s'\n", arg);
    return 0;
  }

  if (opts->size == 0 || opts->size > UINT32_MAX || opts->size > (uint64_t)SIZE_MAX)
  {
    fprintf(stderr, "--size is out of range for mc_host_register\n");
    return 0;
  }
  if (opts->alignment != 0
      && (!is_power_of_two_u64(opts->alignment)
          || opts->alignment > (uint64_t)SIZE_MAX))
  {
    fprintf(stderr, "--alignment must be 0 or a power-of-two process-size value\n");
    return 0;
  }

  return 1;
}

static void touch_pages(void *ptr, uint64_t bytes)
{
  volatile uint8_t *p = (volatile uint8_t *)ptr;
  const uint64_t    page = (uint64_t)page_size();
  uint64_t          off;

  for (off = 0; off < bytes; off += page)
    p[off] = (uint8_t)(off >> 12);

  if (bytes != 0)
    p[bytes - 1] = 0x5a;
}

static const char *mc_status_name(mc_status_t st)
{
  switch (st)
  {
  case MC_OK:        return "MC_OK";
  case MC_ETIMEOUT:  return "MC_ETIMEOUT";
  case MC_EALLOC:    return "MC_EALLOC";
  case MC_EIOCTL:    return "MC_EIOCTL";
  case MC_EINVAL:    return "MC_EINVAL";
  case MC_EINTERNAL: return "MC_EINTERNAL";
  case MC_EHANG:     return "MC_EHANG";
  default:           return "MC_EUNKNOWN";
  }
}

static void set_errno_error(char *buf, size_t buf_size, const char *what)
{
  snprintf(buf, buf_size, "%s failed: errno=%d (%s)", what, errno,
           strerror(errno));
}

static void set_mc_error(char *buf, size_t buf_size, const char *what,
                         mc_status_t st)
{
  snprintf(buf, buf_size, "%s failed: %s (%d)", what, mc_status_name(st),
           (int)st);
}

static int time_mc_register(mc_ctx_t *ctx, void *ptr, uint64_t bytes,
                            double *usec, char *error, size_t error_size)
{
  uint64_t    t0;
  uint64_t    t1;
  mc_status_t st;

  t0 = mc_test_now_ns();
  st = mc_host_register(ctx, ptr, (size_t)bytes, MC_VAS_UVM);
  t1 = mc_test_now_ns();

  if (st != MC_OK)
  {
    set_mc_error(error, error_size, "mc_host_register", st);
    return 0;
  }

  *usec = (double)(t1 - t0) / 1000.0;

  st = mc_host_unregister(ctx, ptr);
  if (st != MC_OK)
  {
    set_mc_error(error, error_size, "mc_host_unregister", st);
    return 0;
  }

  return 1;
}

typedef struct {
  void *base;
  void *ptr;
} aligned_malloc_t;

static int malloc_aligned_sample_range(uint64_t bytes, uint64_t alignment,
                                       aligned_malloc_t *alloc, char *error,
                                       size_t error_size)
{
  uintptr_t raw;
  uintptr_t aligned;

  memset(alloc, 0, sizeof(*alloc));

  if (bytes == 0 || bytes > (uint64_t)SIZE_MAX)
  {
    snprintf(error, error_size, "malloc size is out of range");
    return 0;
  }

  if (alignment != 0
      && (alignment > (uint64_t)SIZE_MAX
          || bytes > (uint64_t)SIZE_MAX - alignment))
  {
    snprintf(error, error_size, "aligned malloc size is out of range");
    return 0;
  }

  alloc->base = malloc((size_t)(alignment == 0 ? bytes : bytes + alignment));
  if (alloc->base == NULL)
  {
    set_errno_error(error, error_size,
                    alignment == 0 ? "malloc" : "malloc aligned backing range");
    return 0;
  }

  if (alignment == 0)
  {
    alloc->ptr = alloc->base;
    return 1;
  }

  raw = (uintptr_t)alloc->base;
  aligned = (raw + 1 + (uintptr_t)alignment - 1)
            & ~((uintptr_t)alignment - 1);
  alloc->ptr = (void *)aligned;
  return 1;
}

static int run_one(mc_ctx_t *ctx, phase_t phase, uint64_t bytes,
                   uint64_t alignment, double *usec, char *error,
                   size_t error_size)
{
  aligned_malloc_t alloc;
  void            *ptr;
  int              ok = 1;

  if (!malloc_aligned_sample_range(bytes, alignment, &alloc, error,
                                   error_size))
    return 0;

  ptr = alloc.ptr;

  switch (phase)
  {
  case PHASE_TOUCH:
  {
    const uint64_t t0 = mc_test_now_ns();
    touch_pages(ptr, bytes);
    const uint64_t t1 = mc_test_now_ns();
    *usec = (double)(t1 - t0) / 1000.0;
    break;
  }

  case PHASE_MLOCK:
  {
    const uint64_t t0 = mc_test_now_ns();
    int ret = mlock(ptr, (size_t)bytes);
    const uint64_t t1 = mc_test_now_ns();
    if (ret != 0)
    {
      set_errno_error(error, error_size, "mlock");
      ok = 0;
      break;
    }
    *usec = (double)(t1 - t0) / 1000.0;
    munlock(ptr, (size_t)bytes);
    break;
  }

  case PHASE_MC_COLD:
    ok = time_mc_register(ctx, ptr, bytes, usec, error, error_size);
    break;

  case PHASE_MC_PREFAULT:
    touch_pages(ptr, bytes);
    ok = time_mc_register(ctx, ptr, bytes, usec, error, error_size);
    break;

  case PHASE_MC_AFTER_MLOCK:
    if (mlock(ptr, (size_t)bytes) != 0)
    {
      set_errno_error(error, error_size, "mlock setup");
      ok = 0;
      break;
    }
    ok = time_mc_register(ctx, ptr, bytes, usec, error, error_size);
    munlock(ptr, (size_t)bytes);
    break;
  }

  free(alloc.base);

  return ok;
}

static result_t run_phase(mc_ctx_t *ctx, const phase_desc_t *phase,
                          uint64_t bytes, uint64_t alignment, int iters)
{
  result_t r;
  int      i;

  memset(&r, 0, sizeof(r));
  r.usec = calloc((size_t)iters, sizeof(*r.usec));
  if (r.usec == NULL)
  {
    snprintf(r.first_error, sizeof(r.first_error), "calloc samples failed");
    r.failures = 1;
    return r;
  }

  for (i = 0; i < iters; ++i)
  {
    double usec = 0.0;
    char   error[256] = "";

    if (!run_one(ctx, phase->phase, bytes, alignment, &usec, error,
                 sizeof(error)))
    {
      r.failures++;
      if (r.first_error[0] == '\0')
        snprintf(r.first_error, sizeof(r.first_error), "%s", error);
      break;
    }

    r.usec[r.count++] = usec;
  }

  return r;
}

static int cmp_double(const void *a, const void *b)
{
  const double da = *(const double *)a;
  const double db = *(const double *)b;
  return (da > db) - (da < db);
}

static double percentile(const double *sorted, size_t count, double p)
{
  double idx;
  size_t lo;
  size_t hi;
  double frac;

  if (count == 0)
    return 0.0;

  idx = p * (double)(count - 1);
  lo = (size_t)idx;
  hi = lo + 1;
  if (hi >= count)
    hi = count - 1;
  frac = idx - (double)lo;
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static void print_result(const char *name, result_t *r, uint64_t pages)
{
  double sum = 0.0;
  double min;
  double mean;
  double p50;
  double p90;
  double us_per_page;
  size_t i;

  if (r->count == 0)
  {
    printf("%-28s  FAIL   %s\n", name,
           r->first_error[0] == '\0' ? "no samples" : r->first_error);
    return;
  }

  qsort(r->usec, r->count, sizeof(*r->usec), cmp_double);

  for (i = 0; i < r->count; i++)
    sum += r->usec[i];

  min = r->usec[0];
  mean = sum / (double)r->count;
  p50 = percentile(r->usec, r->count, 0.50);
  p90 = percentile(r->usec, r->count, 0.90);
  us_per_page = pages == 0 ? 0.0 : mean / (double)pages;

  printf("%-28s  %5zu  %10.2f  %10.2f  %10.2f  %10.2f  %10.4f",
         name, r->count, min, mean, p50, p90, us_per_page);

  if (r->failures != 0)
    printf("  first failure: %s", r->first_error);
  printf("\n");
}

static int warm_mc(mc_ctx_t *ctx)
{
  const size_t bytes = (size_t)page_size();
  mc_status_t  st;
  void        *ptr;

  ptr = malloc(bytes);
  if (ptr == NULL)
  {
    fprintf(stderr, "malloc failed during mc warmup\n");
    return 0;
  }

  touch_pages(ptr, bytes);
  st = mc_host_register(ctx, ptr, bytes, MC_VAS_UVM);
  if (st != MC_OK)
  {
    fprintf(stderr, "mc_host_register warmup failed: %s (%d)\n",
            mc_status_name(st), (int)st);
    free(ptr);
    return 0;
  }

  st = mc_host_unregister(ctx, ptr);
  if (st != MC_OK)
  {
    fprintf(stderr, "mc_host_unregister warmup failed: %s (%d)\n",
            mc_status_name(st), (int)st);
    free(ptr);
    return 0;
  }

  free(ptr);
  return 1;
}

static void print_memlock_limit(void)
{
  struct rlimit lim;

  if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0)
    return;

  if (lim.rlim_cur == RLIM_INFINITY)
  {
    printf("RLIMIT_MEMLOCK: unlimited\n");
    return;
  }

  printf("RLIMIT_MEMLOCK: %llu bytes\n", (unsigned long long)lim.rlim_cur);
}

int main(int argc, char **argv)
{
  options_t opts;
  mc_ctx_t *ctx = NULL;
  uint64_t  pages;
  size_t    i;

  if (!parse_args(argc, argv, &opts))
  {
    usage(argv[0]);
    return 2;
  }

  if (mc_init(&ctx) != MC_OK)
  {
    fprintf(stderr, "mc_init failed\n");
    return 3;
  }

  if (!warm_mc(ctx))
  {
    mc_fini(ctx);
    return 3;
  }

  pages = pages_in_range(opts.size);

  printf("mc_host_register_bench: size=%llu bytes, pages=%llu, alignment=%llu bytes, iters=%d\n",
         (unsigned long long)opts.size, (unsigned long long)pages,
         (unsigned long long)opts.alignment, opts.iters);
  print_memlock_limit();
  if (opts.alignment != 0)
    printf("samples use requested-size ranges from aligned interior malloc pointers.\n");
  printf("\n");
  printf("%-28s  %5s  %10s  %10s  %10s  %10s  %10s\n",
         "phase", "n", "min_us", "mean_us", "p50_us", "p90_us", "us/page");

  for (i = 0; i < sizeof(k_phases) / sizeof(k_phases[0]); i++)
  {
    result_t r;

    r = run_phase(ctx, &k_phases[i], opts.size, opts.alignment, opts.iters);
    print_result(k_phases[i].name, &r, pages);
    free(r.usec);
  }

  mc_fini(ctx);
  return 0;
}
