/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_allocation_test.c — mismatched-VAS mc_memcpy
 * must return MC_EINVAL.
 *
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

  mc_fini(ctx);
  return 0;
}
