/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * cuda_host_register_bench.cu - break down cudaHostRegister latency.
 *
 * This benchmark intentionally uses malloc-backed host ranges.  It
 * compares the steady-state CUDA registration cost against two CPU-side
 * operations which also make anonymous pages resident:
 *
 *   malloc_touch
 *       malloc, then time one write per page.  This is first-touch page
 *       population without any locking or CUDA work.
 *
 *   malloc_mlock
 *       malloc, then time mlock().  This faults pages in and marks them
 *       unevictable from userspace.  It is not equivalent to the NVIDIA
 *       driver's DMA pin/FOLL_PIN path, but it is the closest standard
 *       userspace "pin it myself" operation.
 *
 *   malloc_cudaHostRegister
 *       malloc, then time cudaHostRegister() on an untouched range.  This
 *       includes Linux page population plus NVIDIA registration/mapping work.
 *
 *   prefault_cudaHostRegister
 *       malloc, touch pages outside the timed window, then time
 *       cudaHostRegister().  This removes first-touch page faults.
 *
 *   mlock_cudaHostRegister
 *       malloc, mlock outside the timed window, then time
 *       cudaHostRegister().  This estimates the residual CUDA/RM/UVM work
 *       after ordinary Linux residency/locking has already happened.
 *
 * With --alignment SIZE, each sample mallocs a larger backing range, uses a
 * SIZE-aligned interior pointer for the requested range, and frees the
 * original malloc pointer after the sample.  Without --alignment, samples use
 * the raw malloc pointer.
 *
 * Interpretation:
 *   If malloc_mlock is close to malloc_cudaHostRegister, page population and
 *   CPU-side locking are likely a large part of the cold registration cost.
 *   If mlock_cudaHostRegister is still large, CUDA's own DMA pinning, IOMMU,
 *   RM allocation/register, or UVM external-range mapping work is still
 *   significant.
 *
 * Build: cd reverse && make cuda_host_register_bench
 * Run:   ./bin/cuda_host_register_bench --size 4M --iters 30
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#define DEFAULT_SIZE  (4ULL * 1024ULL * 1024ULL)
#define DEFAULT_ITERS 30

#define CUDA_CHECK(call)                                                \
  do                                                                    \
  {                                                                     \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
              cudaGetErrorString(_e));                                  \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

struct options_t
{
  uint64_t size;
  uint64_t alignment;
  int      iters;
  int      device;
};

struct result_t
{
  std::vector<double> usec;
  std::string         first_error;
  int                 failures;
};

enum class phase_t
{
  Touch,
  Mlock,
  CudaCold,
  CudaPrefault,
  CudaAfterMlock,
};

struct phase_desc_t
{
  const char *name;
  phase_t     phase;
};

static const phase_desc_t k_phases[] = {
    {"malloc_touch", phase_t::Touch},
    {"malloc_mlock", phase_t::Mlock},
    {"malloc_cudaHostRegister", phase_t::CudaCold},
    {"prefault_cudaHostRegister", phase_t::CudaPrefault},
    {"mlock_cudaHostRegister", phase_t::CudaAfterMlock},
};

static uint64_t now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

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

static bool is_power_of_two(uint64_t v)
{
  return v != 0 && (v & (v - 1)) == 0;
}

static bool parse_size(const char *s, uint64_t *out)
{
  if (s == nullptr || *s == '\0')
    return false;

  char *end = nullptr;
  errno = 0;
  uint64_t v = (uint64_t)strtoull(s, &end, 0);
  if (errno != 0 || end == s)
    return false;

  unsigned shift = 0;
  switch (*end)
  {
    case '\0':
      break;
    case 'k':
    case 'K':
      shift = 10;
      ++end;
      break;
    case 'm':
    case 'M':
      shift = 20;
      ++end;
      break;
    case 'g':
    case 'G':
      shift = 30;
      ++end;
      break;
    default:
      return false;
  }

  if (*end == 'b' || *end == 'B')
    ++end;
  if (*end != '\0')
    return false;
  if (shift != 0 && v > (~0ULL >> shift))
    return false;

  *out = v << shift;
  return true;
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
          "  --device N      CUDA device ordinal. Default: 0.\n"
          "  -h, --help      show this message.\n"
          "\n"
          "The CUDA context is initialized and a 1-page registration is warmed\n"
          "up before timing. Each sample uses a fresh malloc/free lifecycle.\n"
          "With --alignment, samples use an aligned interior pointer from a\n"
          "larger malloc backing allocation.\n",
          prog);
}

static bool parse_args(int argc, char **argv, options_t *opts)
{
  opts->size = DEFAULT_SIZE;
  opts->alignment = 0;
  opts->iters = DEFAULT_ITERS;
  opts->device = 0;

  for (int i = 1; i < argc; ++i)
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
        return false;
      }
      continue;
    }
    if (strcmp(arg, "--alignment") == 0)
    {
      if (++i >= argc || !parse_size(argv[i], &opts->alignment))
      {
        fprintf(stderr, "bad --alignment\n");
        return false;
      }
      continue;
    }
    if (strcmp(arg, "--iters") == 0)
    {
      if (++i >= argc)
      {
        fprintf(stderr, "--iters needs an argument\n");
        return false;
      }
      char *end = nullptr;
      errno = 0;
      long v = strtol(argv[i], &end, 0);
      if (errno != 0 || end == argv[i] || *end != '\0' || v <= 0 || v > INT_MAX)
      {
        fprintf(stderr, "bad --iters\n");
        return false;
      }
      opts->iters = (int)v;
      continue;
    }
    if (strcmp(arg, "--device") == 0)
    {
      if (++i >= argc)
      {
        fprintf(stderr, "--device needs an argument\n");
        return false;
      }
      char *end = nullptr;
      errno = 0;
      long v = strtol(argv[i], &end, 0);
      if (errno != 0 || end == argv[i] || *end != '\0' || v < 0 || v > INT_MAX)
      {
        fprintf(stderr, "bad --device\n");
        return false;
      }
      opts->device = (int)v;
      continue;
    }

    fprintf(stderr, "unknown arg '%s'\n", arg);
    return false;
  }

  if (opts->size == 0 || opts->size > (uint64_t)SIZE_MAX)
  {
    fprintf(stderr, "--size is out of range for this process\n");
    return false;
  }
  if (opts->alignment != 0
      && (!is_power_of_two(opts->alignment)
          || opts->alignment > (uint64_t)SIZE_MAX))
  {
    fprintf(stderr, "--alignment must be 0 or a power-of-two process-size value\n");
    return false;
  }

  return true;
}

static void touch_pages(void *ptr, uint64_t bytes)
{
  volatile uint8_t *p = (volatile uint8_t *)ptr;
  const uint64_t page = (uint64_t)page_size();

  for (uint64_t off = 0; off < bytes; off += page)
    p[off] = (uint8_t)(off >> 12);

  if (bytes != 0)
    p[bytes - 1] = 0x5a;
}

static std::string errno_string(const char *what)
{
  char buf[256];
  snprintf(buf, sizeof(buf), "%s failed: errno=%d (%s)", what, errno, strerror(errno));
  return std::string(buf);
}

static std::string cuda_string(const char *what, cudaError_t err)
{
  char buf[256];
  snprintf(buf, sizeof(buf), "%s failed: %s", what, cudaGetErrorString(err));
  return std::string(buf);
}

static bool time_cuda_register(void *ptr, uint64_t bytes, double *usec, std::string *error)
{
  const uint64_t t0 = now_ns();
  cudaError_t err = cudaHostRegister(ptr, (size_t)bytes, cudaHostRegisterDefault);
  const uint64_t t1 = now_ns();

  if (err != cudaSuccess)
  {
    *error = cuda_string("cudaHostRegister", err);
    return false;
  }

  *usec = (double)(t1 - t0) / 1000.0;

  err = cudaHostUnregister(ptr);
  if (err != cudaSuccess)
  {
    *error = cuda_string("cudaHostUnregister", err);
    return false;
  }

  return true;
}

struct aligned_malloc_t
{
  void *base;
  void *ptr;
};

static bool malloc_sample_range(uint64_t bytes, uint64_t alignment,
                                aligned_malloc_t *alloc, std::string *error)
{
  *alloc = {};

  if (bytes == 0 || bytes > (uint64_t)SIZE_MAX)
  {
    *error = "malloc size is out of range";
    return false;
  }

  if (alignment != 0
      && (alignment > (uint64_t)SIZE_MAX
          || bytes > (uint64_t)SIZE_MAX - alignment))
  {
    *error = "aligned malloc size is out of range";
    return false;
  }

  const uint64_t backing_bytes = alignment == 0 ? bytes : bytes + alignment;
  alloc->base = malloc((size_t)backing_bytes);
  if (alloc->base == nullptr)
  {
    *error = errno_string(alignment == 0 ? "malloc" : "malloc aligned backing range");
    return false;
  }

  if (alignment == 0)
  {
    alloc->ptr = alloc->base;
    return true;
  }

  const uintptr_t raw = (uintptr_t)alloc->base;
  const uintptr_t aligned =
      (raw + 1 + (uintptr_t)alignment - 1) & ~((uintptr_t)alignment - 1);

  alloc->ptr = (void *)aligned;
  return true;
}

static bool run_one(phase_t phase, uint64_t bytes, uint64_t alignment,
                    double *usec, std::string *error)
{
  aligned_malloc_t alloc;
  bool ok = true;

  if (!malloc_sample_range(bytes, alignment, &alloc, error))
    return false;

  void *ptr = alloc.ptr;

  switch (phase)
  {
    case phase_t::Touch:
    {
      const uint64_t t0 = now_ns();
      touch_pages(ptr, bytes);
      const uint64_t t1 = now_ns();
      *usec = (double)(t1 - t0) / 1000.0;
      break;
    }

    case phase_t::Mlock:
    {
      const uint64_t t0 = now_ns();
      int ret = mlock(ptr, (size_t)bytes);
      const uint64_t t1 = now_ns();
      if (ret != 0)
      {
        *error = errno_string("mlock");
        ok = false;
        break;
      }
      *usec = (double)(t1 - t0) / 1000.0;
      munlock(ptr, (size_t)bytes);
      break;
    }

    case phase_t::CudaCold:
    {
      ok = time_cuda_register(ptr, bytes, usec, error);
      break;
    }

    case phase_t::CudaPrefault:
    {
      touch_pages(ptr, bytes);
      ok = time_cuda_register(ptr, bytes, usec, error);
      break;
    }

    case phase_t::CudaAfterMlock:
    {
      if (mlock(ptr, (size_t)bytes) != 0)
      {
        *error = errno_string("mlock setup");
        ok = false;
        break;
      }
      ok = time_cuda_register(ptr, bytes, usec, error);
      munlock(ptr, (size_t)bytes);
      break;
    }
  }

  free(alloc.base);

  return ok;
}

static result_t run_phase(const phase_desc_t &phase, uint64_t bytes,
                          uint64_t alignment, int iters)
{
  result_t r;
  r.failures = 0;
  r.usec.reserve((size_t)iters);

  for (int i = 0; i < iters; ++i)
  {
    double usec = 0.0;
    std::string error;
    if (!run_one(phase.phase, bytes, alignment, &usec, &error))
    {
      ++r.failures;
      if (r.first_error.empty())
        r.first_error = error;
      break;
    }
    r.usec.push_back(usec);
  }

  return r;
}

static double percentile(const std::vector<double> &sorted, double p)
{
  if (sorted.empty())
    return 0.0;
  const double idx = p * (double)(sorted.size() - 1);
  const size_t lo = (size_t)idx;
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = idx - (double)lo;
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static void print_result(const char *name, result_t r, uint64_t pages)
{
  if (r.usec.empty())
  {
    printf("%-28s  FAIL   %s\n", name,
           r.first_error.empty() ? "no samples" : r.first_error.c_str());
    return;
  }

  std::sort(r.usec.begin(), r.usec.end());

  double sum = 0.0;
  for (double v : r.usec)
    sum += v;

  const double min = r.usec.front();
  const double mean = sum / (double)r.usec.size();
  const double p50 = percentile(r.usec, 0.50);
  const double p90 = percentile(r.usec, 0.90);
  const double us_per_page = pages == 0 ? 0.0 : mean / (double)pages;

  printf("%-28s  %5zu  %10.2f  %10.2f  %10.2f  %10.2f  %10.4f",
         name, r.usec.size(), min, mean, p50, p90, us_per_page);

  if (r.failures != 0)
    printf("  first failure: %s", r.first_error.c_str());
  printf("\n");
}

static void warm_cuda(void)
{
  const size_t bytes = (size_t)page_size();
  void *ptr = malloc(bytes);
  if (ptr == nullptr)
  {
    fprintf(stderr, "malloc failed during CUDA warmup\n");
    exit(EXIT_FAILURE);
  }

  touch_pages(ptr, bytes);
  CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault));
  CUDA_CHECK(cudaHostUnregister(ptr));
  free(ptr);
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
  if (!parse_args(argc, argv, &opts))
  {
    usage(argv[0]);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(opts.device));
  CUDA_CHECK(cudaInitDevice(opts.device, 0, 0));
  warm_cuda();

  const uint64_t covered = pages_in_range(opts.size);

  printf("cuda_host_register_bench: size=%llu bytes, pages=%llu, alignment=%llu bytes, iters=%d, device=%d\n",
         (unsigned long long)opts.size, (unsigned long long)covered,
         (unsigned long long)opts.alignment, opts.iters, opts.device);
  print_memlock_limit();
  if (opts.alignment != 0)
    printf("samples use requested-size ranges from aligned interior malloc pointers.\n");
  printf("\n");
  printf("%-28s  %5s  %10s  %10s  %10s  %10s  %10s\n",
         "phase", "n", "min_us", "mean_us", "p50_us", "p90_us", "us/page");

  for (const phase_desc_t &phase : k_phases)
    print_result(phase.name,
                 run_phase(phase, opts.size, opts.alignment, opts.iters),
                 covered);

  return 0;
}
