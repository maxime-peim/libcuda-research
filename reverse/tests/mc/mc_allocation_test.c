/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_allocation_test.c — allocation-API edge cases:
 *   1. mismatched-VAS mc_memcpy must return MC_EINVAL;
 *   2. mc_malloc_host_wc round-trips — a WC buffer seeds an H2D, the
 *      data comes back through a cached buffer intact.
 */
#include <emmintrin.h> /* _mm_clflush */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mc.h"
#include "mc_test_helpers.h"

typedef struct
{
  mc_vas_t vt_1;
  mc_vas_t vt_2;
} mc_alloc_test_t;

/* Tiny inline test: allocate two buffers in two VAS, attempt mc_memcpy
 * between them, expect MC_EINVAL. Exercises the VAS-mismatch dispatch
 * validation that nothing else tests today. */
static int test_vas_memcpy(mc_ctx_t *ctx, mc_alloc_test_t *test)
{
  void       *buf_1;
  void       *buf_2;
  int         rc = 0;
  mc_status_t status;

  buf_1 = mc_malloc_host(ctx, 4096, test->vt_1);
  buf_2 = mc_malloc_host(ctx, 4096, test->vt_2);
  if (buf_1 == NULL || buf_2 == NULL)
  {
    fprintf(stderr, "test_vas_memcpy: alloc failed\n");
    rc = -1;
    goto cleanup;
  }

  status = mc_memcpy(ctx, buf_1, buf_2, 4096, MC_XFER_HOST);
  if ((status != MC_EINVAL) ^ (test->vt_1 == test->vt_2))
  {
    fprintf(stderr,
            "test_vas_memcpy: expected MC_EINVAL, got %d for VAS (%d, %d)\n"
            "(VAS dispatch validation regressed)\n",
            (int)status, test->vt_1, test->vt_2);
    rc = -1;
  }

cleanup:
  mc_free(ctx, buf_1);
  mc_free(ctx, buf_2);
  return rc;
}

/* WC opt-in round trip: seed a write-combined host buffer (CPU writes —
 * the direction WC is for), H2D into a device buffer, D2H back into a
 * CACHED host buffer, verify there.  Verifying in the cached buffer keeps
 * the test fast (reading a WC buffer runs at ~30 MB/s) and exercises
 * both allocator variants side by side. */
static int test_wc_roundtrip(mc_ctx_t *ctx)
{
  enum { N = 1u << 20 };  /* 1 MiB */
  uint32_t *wc  = mc_malloc_host_wc(ctx, N, MC_VAS_UVM);
  uint32_t *dev = mc_malloc_device(ctx, N, MC_VAS_UVM);
  uint32_t *rd  = mc_malloc_host(ctx, N, MC_VAS_UVM);
  size_t    nw  = N / 4, i, bad;
  int       rc  = 0;

  if (wc == NULL || dev == NULL || rd == NULL)
  {
    fprintf(stderr, "test_wc_roundtrip: alloc failed (wc=%p dev=%p rd=%p)\n",
            (void *)wc, (void *)dev, (void *)rd);
    rc = -1;
    goto cleanup;
  }

  for (i = 0; i < nw; i++) wc[i] = 0xA5000000u | (uint32_t)i;
  for (i = 0; i < nw; i++) rd[i] = 0xDEADBEEFu;

  if (mc_memcpy(ctx, dev, wc, N, MC_XFER_HOST) != MC_OK ||
      mc_memcpy(ctx, rd, dev, N, MC_XFER_HOST) != MC_OK)
  {
    fprintf(stderr, "test_wc_roundtrip: mc_memcpy failed\n");
    rc = -1;
    goto cleanup;
  }

  for (bad = 0; bad < nw; bad++)
  {
    if (rd[bad] != (0xA5000000u | (uint32_t)bad))
    {
      fprintf(stderr,
              "test_wc_roundtrip: verify FAIL at dword %zu: got 0x%08x\n",
              bad, rd[bad]);
      rc = -1;
      break;
    }
  }

cleanup:
  mc_free(ctx, wc);
  mc_free(ctx, dev);
  mc_free(ctx, rd);
  return rc;
}

int main(int argc, char **argv)
{
  mc_ctx_t       *ctx           = NULL;
  mc_alloc_test_t test_matrix[] = {
    { .vt_1 = MC_VAS_UVM, .vt_2 = MC_VAS_UVM },
    { .vt_1 = MC_VAS_UVM, .vt_2 = MC_VAS_SYSMEM_CARRIER },
    { .vt_1 = MC_VAS_UVM, .vt_2 = MC_VAS_FB_CARRIER },
    { .vt_1 = MC_VAS_SYSMEM_CARRIER, .vt_2 = MC_VAS_SYSMEM_CARRIER },
    { .vt_1 = MC_VAS_SYSMEM_CARRIER, .vt_2 = MC_VAS_FB_CARRIER },
    { .vt_1 = MC_VAS_FB_CARRIER, .vt_2 = MC_VAS_FB_CARRIER },
  };
  int t;

  if (mc_init(&ctx) != MC_OK)
  {
    fprintf(stderr, "mc_init failed\n");
    return 1;
  }
  printf("mc_init ok\n");

  for (t = 0; t < sizeof(test_matrix) / sizeof(test_matrix[0]); t++)
  {
    if (test_vas_memcpy(ctx, &test_matrix[t]) != 0)
    {
      mc_fini(ctx);
      return 2;
    }
  }

  if (test_wc_roundtrip(ctx) != 0)
  {
    mc_fini(ctx);
    return 3;
  }
  printf("wc roundtrip ok\n");

  mc_fini(ctx);
  return 0;
}
