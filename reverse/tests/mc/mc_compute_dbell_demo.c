/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_compute_dbell_demo.c — has the GPU SM thread run
 * mc_doorbell_kernel and write a token into a sysmem cell that
 * the host can read back.  No BAR1, no doorbell-write side
 * effect — just "did the SASS execute?".
 *
 * Sequence:
 *   1. mc_init.  Brings up the UVM, DMA, and compute channels.
 *   2. Get a (cpu_ptr, gpu_va) pair via mc_compute_get_scratch — a
 *      small helper that exposes a sysmem dword living in the
 *      compute channel's VAS.
 *   3. mc_compute_doorbell_kernel(ctx, gpu_va, 0xdeadbeef) — submits
 *      the launch pushbuffer (SET_OBJECT + SEND_PCAS_A + PCAS2_B +
 *      release sema), the SM thread runs `*dst = token; exit`.
 *   4. Read back through cpu_ptr.  Expected: 0xdeadbeef.
 *
 * Run on the H100 test host with the kernel doorbell watchpoint disabled (set
 * via /etc/modprobe.d/nvidia-dbell-bypass.conf).
 *
 *   sudo ./bin/mc_compute_dbell_demo
 */

#include <emmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mc.h"

int main(void)
{
  mc_ctx_t           *ctx     = NULL;
  volatile uint32_t  *cell_cpu = NULL;
  uint64_t            cell_gpu_va = 0;

  if (mc_init(&ctx) != MC_OK || ctx == NULL)
  {
    fprintf(stderr, "FAIL: mc_init returned non-zero\n");
    return EXIT_FAILURE;
  }

  if (mc_compute_get_scratch(ctx, &cell_cpu, &cell_gpu_va) != MC_OK)
  {
    fprintf(stderr, "FAIL: mc_compute_get_scratch failed\n");
    mc_fini(ctx);
    return EXIT_FAILURE;
  }

  /* Pre-fill 16 adjacent dwords with distinctive sentinels so we
   * can detect any GPU write near the target cell, not just at the
   * exact target.  If the kernel writes to a slightly-off VA (off-
   * by-some-bytes), we'd still see the change. */
  const uint32_t SENTINEL = 0xCAFEBABE;
  for (int i = -8; i < 8; i++) cell_cpu[i] = SENTINEL + (uint32_t)i;
  _mm_sfence();

  printf("compute_dbell_demo: cell_cpu=%p cell_gpu_va=0x%llx\n",
         (void *)cell_cpu, (unsigned long long)cell_gpu_va);
  printf("  before nearby cells:\n");
  for (int i = -4; i < 8; i++)
    printf("    cell[%+d] @ va+0x%02x = 0x%08x\n",
           i, (unsigned)(i*4) & 0xff, cell_cpu[i]);

  uint32_t token = 0xdeadbeef;
  mc_status_t rc = mc_compute_doorbell_kernel(ctx, cell_gpu_va, token);

  if (rc == MC_ETIMEOUT)
  {
    fprintf(stderr,
            "FAIL: mc_compute_doorbell_kernel timed out after %d ms\n"
            "      (kernel didn't finish, or release sema never fired)\n"
            "      *cell = 0x%08x\n",
            2000, *cell_cpu);
    mc_fini(ctx);
    return EXIT_FAILURE;
  }
  if (rc != MC_OK)
  {
    fprintf(stderr, "FAIL: mc_compute_doorbell_kernel returned %d\n", rc);
    mc_fini(ctx);
    return EXIT_FAILURE;
  }

  /* CLFLUSH each nearby cell + mfence to ensure CPU sees DRAM.
   *
   * NOT optional — empirically required.  At N=20 on the H100 test host:
   *   no clflush, mfence only:  1/20 PASS
   *   clflush + mfence:        20/20 PASS
   *
   * Mechanism: the test pre-fills each cell with a sentinel using
   * `cell_cpu[i] = ...`, which loads the line into L1 in Modified
   * state.  The GPU's STG.E.STRONG.SYS writes the same line in
   * DRAM, but the snoop traffic doesn't always invalidate the
   * line in this core's L1 fast enough (or the line gets
   * re-cached by speculative prefetch before the host load).
   * mfence orders memory ops but does not touch cache state, so
   * a `volatile` load still returns the cached sentinel value
   * ~95% of the time.  clflush forces the line back to DRAM and
   * the next load misses cache, returning the GPU-written
   * bytes.
   *
   * Future work: try mapping these sysmem pages as write-through
   * or uncached on the CPU side; that would obviate clflush at
   * the cost of slower CPU stores.  Out of scope for the demo. */
  for (int i = -8; i < 8; i++) __builtin_ia32_clflush((const void *)&cell_cpu[i]);
  _mm_mfence();
  printf("  after  nearby cells:\n");
  for (int i = -4; i < 8; i++) {
    uint32_t v = cell_cpu[i];
    const char *tag = "";
    if (i == 0) tag = "  ← target";
    else if (v != (SENTINEL + (uint32_t)i)) tag = "  ← CHANGED!";
    printf("    cell[%+d] @ va+0x%02x = 0x%08x%s\n",
           i, (unsigned)(i*4) & 0xff, v, tag);
  }
  uint32_t observed = *cell_cpu;

  int ok = (observed == token);
  if (ok)
    printf("PASS: GPU SM thread wrote 0x%08x to sysmem cell via\n"
           "      mc_doorbell_kernel(dst, token) executed under\n"
           "      HOPPER_COMPUTE_A on mc's compute channel.\n",
           token);
  else if (observed == SENTINEL)
    fprintf(stderr,
            "FAIL: kernel did not run — sentinel 0x%08x intact.\n"
            "      Either the QMD launched 0 threads (CTA dim wrong),\n"
            "      or the SM never reached STG (predicate trap?).\n",
            SENTINEL);
  else if (observed == 0)
    fprintf(stderr,
            "FAIL: kernel ran but wrote 0 — likely cb0[0x210] / cb0[0x218]\n"
            "      didn't reach the SM with our patched values, so\n"
            "      *dst = token wrote token=0 instead of 0x%08x.\n",
            token);
  else
    fprintf(stderr,
            "FAIL: expected 0x%08x, got 0x%08x (unexpected — re-examine)\n",
            token, observed);

  mc_fini(ctx);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
