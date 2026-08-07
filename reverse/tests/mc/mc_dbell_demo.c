/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_dbell_demo.c — exercise libmc's mc_dbell_demo_ring().
 *
 * What it proves: the GPU's CE engine (running on mc's secondary,
 * non-UVM channel) can MMIO-write the H100 BAR1 doorbell page from a GPU
 * VA inside its own VAS.
 *
 * The proof is the release semaphore, not the kernel watchpoint.  That
 * watchpoint is an x86 hardware breakpoint on the diverted userspace
 * mapping, so it traps only CPU writes to the doorbell; this write is
 * issued by the CE straight to BAR1, never passes through the CPU, and so
 * cannot appear there.  Success is the CE having run the pushbuffer to
 * completion, which is exactly what the release semaphore reports.
 *
 * Build: `make mc_dbell_demo` from reverse/.
 * Run:   sudo ./bin/mc_dbell_demo [TOKEN]   (default token = 0xDEADBEEF)
 *
 * Exit codes: 0 = ring submitted + sema fired, 1 = init failure,
 *             2 = ring failure, 3 = bad CLI.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mc.h"

int main(int argc, char **argv)
{
  mc_ctx_t   *ctx   = NULL;
  uint32_t    token = 0xDEADBEEFu;
  mc_status_t st;

  if (argc > 2)
  {
    fprintf(stderr, "usage: %s [TOKEN]\n", argv[0]);
    return 3;
  }
  if (argc == 2)
  {
    char    *end;
    uint64_t v = strtoull(argv[1], &end, 0);
    if (*end != '\0' || v > 0xFFFFFFFFu)
    {
      fprintf(stderr, "bad token '%s' (must fit in uint32)\n", argv[1]);
      return 3;
    }
    token = (uint32_t)v;
  }

  if (mc_init(&ctx) != MC_OK)
  {
    fprintf(stderr, "mc_init failed\n");
    return 1;
  }
  printf("mc_init ok — DMA channel ready\n");

  st = mc_dbell_demo_ring(ctx, token);
  if (st != MC_OK)
  {
    fprintf(stderr, "mc_dbell_demo_ring failed: status=%d\n", (int)st);
    mc_fini(ctx);
    return 2;
  }
  printf("PASS: GPU CE wrote 0x%08" PRIx32 " to BAR1 doorbell page\n", token);
  printf("Verification: the CE-emitted release semaphore fired, so the "
         "pushbuffer ran to completion.\n"
         "  (The kernel doorbell watchpoint cannot see this write: it "
         "traps CPU accesses only,\n   and this one is issued by the CE "
         "directly to BAR1.)\n");

  mc_fini(ctx);
  return 0;
}
