/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * cuda_reference.cu — CUDA-runtime reference for mc_demo.
 *
 * This is the apples-to-apples CUDA analogue of
 * libmc/tests/mc/mc_demo.c: same CLI, same sizes, same patterns, same
 * timing shape.  Every iteration does one CPU-side garble followed by
 * a D2H cudaMemcpy and a byte-exact verify — matching the loop in
 * mc_demo so GB/s numbers from both sides are directly
 * comparable.
 *
 * Structure:
 *
 *   cudaInitDevice(0, 0, 0)  — eagerly initialize CUDA on device 0.
 *                              Without this, the first CUDA call in a
 *                              lazy-init app pays the ~100 ms ctx-
 *                              creation cost inside the measured
 *                              region; doing it explicitly up front
 *                              removes that jitter from the timed
 *                              loop.
 *   cudaMalloc      d_buf.
 *   cudaHostAlloc   h_buf  (pinned → the CE can DMA directly).
 *
 *   CPU fills h_buf = FILL_PATTERN.
 *   H2D cudaMemcpy h_buf → d_buf.
 *   cudaDeviceSynchronize.
 *
 *   for (it = 0 ; it < iters ; it++):
 *       CPU overwrites h_buf = GARBLE_PATTERN
 *       clock_gettime(t0)
 *       cudaMemcpy d_buf → h_buf (D2H)
 *       cudaDeviceSynchronize
 *       clock_gettime(t1)
 *       verify h_buf == FILL_PATTERN  (every word)
 *
 *   Report peak + mean GB/s from monotonic-clock deltas.  PASS/FAIL.
 *
 * Deliberate scope choices:
 *   - No GPU kernel fill — the reference is CE-only, matching libmc's
 *     mc_demo. The H2D seed is CPU-side.
 *   - Monotonic-clock timing, not cudaEvent.  Gets the same wall-
 *     clock shape mc_demo reports.
 *   - No PCIe sysfs probing — nothing in mc_demo measures
 *     link speed, so including it here would introduce asymmetry.
 *   - Pinned host memory (cudaHostAllocDefault) — matches the memory
 *     kind the separate libmc project returns from mc_malloc_host (RM's WRITE_
 *     COMBINE sysmem).  Both sides are pinned, so the CE DMA path is
 *     the same shape.
 *
 * Build: `make cuda_reference` (produces bin/cuda_reference).
 * Run:   ./bin/cuda_reference [--size 128M] [--iters 10]
 */

#include <cuda_runtime.h>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

/* Must match mc_demo.c so side-by-side runs
 * exercise the same semantics.  The bytes themselves don't surface in
 * any ioctl trace, but keeping them identical makes verify failures
 * directly comparable. */

#define DEFAULT_TRANSFER_SIZE (256ULL * 1024ULL * 1024ULL)
#define DEFAULT_ITERS         10
#define MIN_TRANSFER_SIZE     4096ULL
#define MAX_TRANSFER_SIZE     0xFFFFFFFFULL
#define FILL_PATTERN          0xDEADBEEFu
#define GARBLE_PATTERN        0xCAFEBABEu

#define CUDA_CHECK(call)                                                \
  do                                                                    \
  {                                                                     \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                                  \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

/*
 * Mirror of mc_demo.c::parse_size — K/M/G suffix handling,
 * optional trailing 'B'/'b'.  Keeping the parsing rules identical
 * means --size 128M has the same meaning to both binaries.
 */
static bool parse_size(const char *s, uint64_t *out)
{
  if (s == nullptr || *s == '\0')
    return false;

  char *end  = nullptr;
  errno      = 0;
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
  if (shift && v > (~0ULL >> shift))
    return false; /* overflow guard */
  v <<= shift;

  *out = v;
  return true;
}

static void usage(const char *prog)
{
  fprintf(
      stderr,
      "usage: %s [OPTIONS]\n"
      "\n"
      "  --size SIZE   Per-iteration transfer size.  K/M/G suffixes\n"
      "                accepted (e.g. 4M, 256M, 1G).\n"
      "                Range: %llu .. %llu bytes.  Default: %llu bytes\n"
      "                (%llu MiB).\n"
      "  --iters N     Number of timed D2H iterations (>= 1).  Default: %d.\n"
      "  -h, --help    Show this message.\n"
      "\n"
      "Exit codes: 0 pass, 1 verify FAIL, 2 CLI error.\n"
      "cuda_reference is the CUDA-runtime analogue of mc_demo;\n"
      "GB/s numbers from both programs are directly comparable.\n",
      prog, (unsigned long long)MIN_TRANSFER_SIZE,
      (unsigned long long)MAX_TRANSFER_SIZE,
      (unsigned long long)DEFAULT_TRANSFER_SIZE,
      (unsigned long long)(DEFAULT_TRANSFER_SIZE >> 20), DEFAULT_ITERS);
}

static uint64_t now_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

static double gb_per_s(uint64_t bytes, double ms)
{
  if (ms <= 0.0)
    return 0.0;
  return (double)bytes / (ms * 1e-3) / 1e9;
}

int main(int argc, char *argv[])
{
  uint64_t n_bytes = DEFAULT_TRANSFER_SIZE;
  int      iters   = DEFAULT_ITERS;

  for (int i = 1; i < argc; ++i)
  {
    const char *arg = argv[i];
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
    {
      usage(argv[0]);
      return 0;
    }
    if (strcmp(arg, "--size") == 0)
    {
      if (++i >= argc)
      {
        fprintf(stderr, "--size needs an arg\n");
        return 2;
      }
      if (!parse_size(argv[i], &n_bytes))
      {
        fprintf(stderr, "bad --size '%s'\n", argv[i]);
        return 2;
      }
      continue;
    }
    if (strcmp(arg, "--iters") == 0)
    {
      if (++i >= argc)
      {
        fprintf(stderr, "--iters needs an arg\n");
        return 2;
      }
      char *end;
      errno  = 0;
      long v = strtol(argv[i], &end, 0);
      if (errno != 0 || end == argv[i] || *end != '\0' || v <= 0 || v > INT_MAX)
      {
        fprintf(stderr, "bad --iters '%s'\n", argv[i]);
        return 2;
      }
      iters = (int)v;
      continue;
    }
    fprintf(stderr, "unknown arg '%s'\n", arg);
    usage(argv[0]);
    return 2;
  }

  if (n_bytes < MIN_TRANSFER_SIZE || n_bytes > MAX_TRANSFER_SIZE)
  {
    fprintf(stderr, "--size out of range\n");
    return 2;
  }
  /* The verify loop works in 32-bit words; round down so a sub-word
   * tail can't produce a silent verification mismatch. */
  n_bytes &= ~((uint64_t)sizeof(uint32_t) - 1);
  const size_t nw = n_bytes / sizeof(uint32_t);

  printf("cuda_reference: size=%llu bytes (%llu MiB), iters=%d\n",
         (unsigned long long)n_bytes, (unsigned long long)(n_bytes >> 20),
         iters);

  /* ── Eager CUDA init ──────────────────────────────────────────────────
   * Forces the Runtime to dlopen libcuda, call cuInit, and create the
   * primary context on device 0 now — NOT on the first cudaMalloc.
   * This moves the ~100 ms ctx-creation cost out of the timed region,
   * so iter 0's latency isn't an outlier.  Introduced in CUDA 12.0;
   * the older cudaFree(0) idiom does the same thing pre-12. */
  CUDA_CHECK(cudaInitDevice(0, 0, 0));

  /* ── Allocate ── */
  uint32_t *d_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, n_bytes));

  uint32_t *h_buf = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_buf, n_bytes, cudaHostAllocDefault));

  /* ── Seed device with FILL_PATTERN via one H2D ── */
  for (size_t w = 0; w < nw; ++w)
    h_buf[w] = FILL_PATTERN;
  CUDA_CHECK(cudaMemcpy(d_buf, h_buf, n_bytes, cudaMemcpyHostToDevice));

  /* ── Timed D2H loop ── */
  double   best_ms = 1e9, sum_ms = 0.0, ms;
  uint64_t t0;
  int      ok = 1;

  for (int it = 0; it < iters; ++it)
  {
    /* Overwrite h_buf so a silently no-op D2H can't spuriously
     * pass verification by finding stale FILL bytes. */
    for (size_t w = 0; w < nw; ++w)
      h_buf[w] = GARBLE_PATTERN;

    t0 = now_ns();
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, n_bytes, cudaMemcpyDeviceToHost));
    ms = (now_ns() - t0) / 1e6;

    if (ms < best_ms)
      best_ms = ms;
    sum_ms += ms;

    for (size_t w = 0; w < nw; ++w)
    {
      if (h_buf[w] != FILL_PATTERN)
      {
        ok = 0;
        break;
      }
    }
  }

  printf("D2H %llu MiB x %d iters:\n", (unsigned long long)(n_bytes >> 20),
         iters);
  printf("  Peak: %.2f GB/s (%.2f ms)\n", gb_per_s(n_bytes, best_ms), best_ms);
  printf("  Mean: %.2f GB/s (%.2f ms)\n", gb_per_s(n_bytes, sum_ms / iters),
         sum_ms / iters);

  int exit_code = 0;
  if (ok)
  {
    printf("PASS: verification\n");
  }
  else
  {
    fprintf(stderr, "verify FAILED\n");
    exit_code = 1;
  }

  CUDA_CHECK(cudaFreeHost(h_buf));
  CUDA_CHECK(cudaFree(d_buf));

  return exit_code;
}
