/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_compute_demo.c — demo-specific compute-kernel launch path.
 *
 * Hand-rolled SM-thread launch of mc_doorbell_kernel
 * (`*dst = token`) plus the chain-of-evidence variant that uses it
 * to ring primary's BAR1 doorbell.  These are bound tightly to one
 * specific embedded SASS (1×1×1 grid, hard-coded register count,
 * fixed CB0 layout) — they live here rather than in mc_core.c so
 * the channel-management library stays general and the kernel-launch
 * code can grow independently.
 *
 * Library-internal: includes mc_internal.h.  Compiled into
 * libmc.so alongside the other library TUs (mc_core.c, mc_rm.c,
 * mc_uvm.c, mc_submit.c, mc_vaspace.c, mc_compute_qmd.c); not
 * exposed as a separate public surface.
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "nvtypes.h"
#include "nvmisc.h"
#include "class/clc36f.h"        /* NVC36F_DMA_INCR_* (header-encoding macros) */
#include "class/clc7c0.h"        /* HOPPER-inherited compute methods */
#include "class/clc86f.h"        /* NVC86F_SET_OBJECT */
#include "class/clcbc0.h"        /* HOPPER_COMPUTE_A class id */
#include "class/cla0c0.h"        /* INVALIDATE_*_CACHE_NO_WFI */
#include "class/clb1c0.h"        /* INVALIDATE_SKED_CACHES */
#include "class/cla06fsubch.h"   /* NVA06F_SUBCHANNEL_COMPUTE */

#include "mc_internal.h"
#include "mc_compute_qmd.h"

/*
 * Method-stream header for the compute subchannel.  Same encoding
 * as the CE-side INCR_HEADER_SUB in mc_submit.c, but pinned to
 * NVA06F_SUBCHANNEL_COMPUTE (= 1) instead of COPY_ENGINE (= 4).
 * Macros are file-scoped so duplication is harmless.
 */
#define INCR_HEADER_COMPUTE_SUB(method, count, subch)  \
  (DRF_DEF(C36F, _DMA_INCR, _OPCODE, _VALUE)           \
   | DRF_NUM(C36F, _DMA_INCR, _COUNT, (count))         \
   | DRF_NUM(C36F, _DMA_INCR, _SUBCHANNEL, (subch))    \
   | DRF_NUM(C36F, _DMA_INCR, _ADDRESS, (method) >> 2))
#define INCR_HEADER_COMPUTE(method, count) \
  INCR_HEADER_COMPUTE_SUB((method), (count), NVA06F_SUBCHANNEL_COMPUTE)

/* ── HOPPER_COMPUTE_A method-stream builders ──────────────────────────
 *
 * Per-channel one-time setup, modelled on Mesa NVK's
 * `nvk_push_dispatch_state_init` (vulkan/nvk_cmd_dispatch.c:31-89) +
 * `nvk_cmd_buffer_begin_compute` (same file:99-115).  NVK proves that
 * launching a Hopper compute kernel does NOT require the libcuda-only
 * MME bytecode burst, the SET_CWD_REF_COUNTER ramp, the SET_SPA_VERSION
 * / SET_QMD_VERSION / SET_RESERVED_SW_METHOD07 / SET_VALID_SPAN_OVERFLOW
 * methods, or the texture-pool methods.
 *
 * The minimal NVK-style setup is:
 *   - SET_OBJECT(HOPPER_COMPUTE_A)
 *   - INVALIDATE_SKED_CACHES                        (clear stale QMD cache)
 *   - INVALIDATE_SAMPLER_CACHE_NO_WFI(LINES_ALL=0)
 *   - INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI(LINES_ALL=0)
 *   - SET_SHADER_SHARED_MEMORY_WINDOW_A/B = 0x400000000 (16 GiB)
 *       Hopper requires this NON-ZERO and 4 GiB-aligned.  The window
 *       is 4 GiB WIDE on Hopper: every GPU VA in [base, base + 4 GiB)
 *       is intercepted as SMEM by the SM front-end and never reaches
 *       the GMMU as a global store.  NVK uses 1<<32 because its Vulkan
 *       heap avoids that range, but mc's dma_ch carrier
 *       (NV50_MEMORY_VIRTUAL) lands at GPU VA ~0x120000000 and the
 *       va_pool at 0x200000000 — both inside a 4 GiB window starting
 *       at 1<<32.  16 GiB is well above any mc VA.
 *   - SET_SHADER_LOCAL_MEMORY_WINDOW_A/B = 0xffULL << 24 (NVK constant)
 */
static uint32_t *mc_write_compute_setup_methods(uint32_t *pb)
{
  uint64_t smem_window = 0x400000000ULL;     /* 16 GiB; above all mc VAs */
  uint64_t lmem_window = 0xffULL << 24;      /* NVK constant */

  *pb++ = INCR_HEADER_COMPUTE_SUB(NVC86F_SET_OBJECT, 1, NVA06F_SUBCHANNEL_COMPUTE);
  *pb++ = HOPPER_COMPUTE_A;

  /* INVALIDATE_SKED_CACHES (no payload bits matter — V=0 means
   * "invalidate now"; NVK passes 0). */
  *pb++ = INCR_HEADER_COMPUTE(NVB1C0_INVALIDATE_SKED_CACHES, 1);
  *pb++ = 0;

  *pb++ = INCR_HEADER_COMPUTE(NVA0C0_INVALIDATE_SAMPLER_CACHE_NO_WFI, 1);
  *pb++ = NVA0C0_INVALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ALL;

  *pb++ = INCR_HEADER_COMPUTE(NVA0C0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, 1);
  *pb++ = NVA0C0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ALL;

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2);
  *pb++ = (NvU32)(smem_window >> 32);
  *pb++ = (NvU32)(smem_window & 0xffffffffu);

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2);
  *pb++ = (NvU32)(lmem_window >> 32);
  *pb++ = (NvU32)(lmem_window & 0xffffffffu);

  return pb;
}

/* mc_write_compute_launch_methods — submit one QMD via SEND_PCAS_A.
 * The setup methods above have already been emitted on the channel
 * (gated by `compute_ch.setup_done`), so this builder doesn't repeat
 * SET_OBJECT.  NVK's launch path also does not re-emit it per-dispatch
 * (vulkan/nvk_cmd_dispatch.c:328-340 / 387-401). */
static uint32_t *mc_write_compute_launch_methods(uint32_t *pb,
                                                  uint64_t qmd_gpu_va,
                                                  uint64_t sema_va,
                                                  uint32_t sema_payload)
{
  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SEND_PCAS_A, 1);
  *pb++ = (NvU32)(qmd_gpu_va >> 8);

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SEND_SIGNALING_PCAS2_B, 1);
  *pb++ = 0x0000000a;                /* SCHEDULE | <bit 3>; verified
                                      * working on H100.  NVK uses
                                      * PCAS_ACTION_INVALIDATE_COPY_SCHEDULE
                                      * (= 0x3) which also works on
                                      * this driver. */

  /* Release semaphore after the kernel completes.  4-method incr
   * write; A=offset_upper, B=offset_lower, C=payload, D=control word
   * (RELEASE | FOUR_WORDS | FLUSH_DISABLE=FALSE). */
  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SET_REPORT_SEMAPHORE_A, 4);
  *pb++ = DRF_NUM(C7C0, _SET_REPORT_SEMAPHORE_A, _OFFSET_UPPER,
                  (NvU32)(sema_va >> 32));
  *pb++ = (NvU32)sema_va;
  *pb++ = sema_payload;
  *pb++ = DRF_DEF(C7C0, _SET_REPORT_SEMAPHORE_D, _OPERATION, _RELEASE)
        | DRF_DEF(C7C0, _SET_REPORT_SEMAPHORE_D, _STRUCTURE_SIZE, _ONE_WORD)
        | DRF_DEF(C7C0, _SET_REPORT_SEMAPHORE_D, _FLUSH_DISABLE, _FALSE);

  return pb;
}

/* mc_compute_doorbell_kernel — submit the embedded mc_doorbell_kernel
 * SASS on the compute channel.  See mc.h for the full contract. */
mc_status_t mc_compute_doorbell_kernel(mc_ctx_t *ctx, uint64_t dst_gpu_va,
                                       uint32_t token)
{
  mc_channel_t             *ch;
  struct mc_compute_extras *ex;
  struct mc_compute_module *mod;
  uint32_t                 *pb;
  uint32_t                 *pb_end;
  uint32_t                  copy_bytes;

  if (ctx == NULL) return MC_EINVAL;
  if (!ctx->ch[MC_CH_COMPUTE].h_channel) return MC_EINTERNAL;

  ch  = &ctx->ch[MC_CH_COMPUTE];
  ex  = &ch->x.compute;
  mod = &ex->module;

  /* Build the QMD from a zero-initialized buffer NVK-style: only
   * the fields `nak::qmd::Qmd4_0::fill_qmd` sets.  See
   * mc_compute_qmd.h for the setter list.  Single-thread (1,1,1)
   * grid; 8 GPRs (SASS uses up to R5); no SLM/SMEM/barriers.  CB0
   * binds the kernel argument tuple. */
  mc_qmd_init(mod->qmd_cpu);
  mc_qmd_set_global_size   (mod->qmd_cpu, 1, 1, 1);
  mc_qmd_set_local_size    (mod->qmd_cpu, 1, 1, 1);
  mc_qmd_set_prog_addr     (mod->qmd_cpu, mod->sass_gpu_va);
  mc_qmd_set_register_count(mod->qmd_cpu, 8);
  mc_qmd_set_barrier_count (mod->qmd_cpu, 0);
  mc_qmd_set_slm_size      (mod->qmd_cpu, 0);
  mc_qmd_set_smem_size     (mod->qmd_cpu, 0);
  mc_qmd_set_cbuf(mod->qmd_cpu, /*idx=*/0, mod->cb0_gpu_va,
                  MC_CB0_TOTAL_BYTES_ALIGNED);

  /* CB0: zero-init, then write only the kernel-argument dwords the
   * SASS actually reads (c[0x0][0x210..0x218]). */
  mc_cb0_init(mod->cb0_cpu);
  mc_cb0_set_args(mod->cb0_cpu, dst_gpu_va, token);

  if (mc_debug())
  {
    /* Dump the patched QMD's pointer-bearing fields.  Field offsets
     * derived from clcbc0qmd.h V04 MW(hi:lo) ranges:
     *   PROGRAM_ADDRESS_LOWER  MW(1247:1216) → byte 0x98 (dword 38)
     *   PROGRAM_ADDRESS_UPPER  MW(1272:1248) → byte 0x9c (dword 39)
     *   CB0_VALID(0)           bit 416        → byte 0x34 bit 0
     *   CB0_ADDR_LOWER_S6(0)   MW(1567:1536) → byte 0xc0 (dword 48)
     *   CB0_ADDR_UPPER+SIZE(0) MW(1599:1568) → byte 0xc4 (dword 49) */
    uint32_t *q = (uint32_t *)mod->qmd_cpu;
    fprintf(stderr,
            "[mc] DEBUG: mc_compute_doorbell:"
            " qmd_va=0x%llx sass_va=0x%llx cb0_va=0x%llx"
            " dst_va=0x%llx token=0x%x\n"
            "[mc] DEBUG: qmd patched fields (NVK style):\n"
            "  off=0x034 (CB_VALID): 0x%08x\n"
            "  off=0x098 (PROG_LO):  0x%08x\n"
            "  off=0x09c (PROG_HI):  0x%08x\n"
            "  off=0x0c0 (CB0_LO):   0x%08x\n"
            "  off=0x0c4 (CB0_UP+SZ):0x%08x\n",
            (unsigned long long)mod->qmd_gpu_va,
            (unsigned long long)mod->sass_gpu_va,
            (unsigned long long)mod->cb0_gpu_va,
            (unsigned long long)dst_gpu_va, token,
            q[0x034/4], q[0x098/4], q[0x09c/4],
            q[0x0c0/4], q[0x0c4/4]);
  }

  /* Bump per-channel sema_payload (skip 0). */
  ch->sema_payload++;
  if (ch->sema_payload == 0) ch->sema_payload = 1;

  /* Build the pushbuffer.  First call: setup methods + launch.
   * Subsequent calls: launch only. */
  pb = ch->pb_cpu;
  if (!ex->setup_done)
  {
    pb = mc_write_compute_setup_methods(pb);
    ex->setup_done = true;
  }
  pb_end = mc_write_compute_launch_methods(pb, mod->qmd_gpu_va,
                                           ch->sema_gpu_va,
                                           ch->sema_payload);
  copy_bytes = (uint32_t)((pb_end - ch->pb_cpu) * sizeof(uint32_t));

  return mc_channel_submit(ch, ctx->vf_doorbell, copy_bytes);
}

mc_status_t mc_compute_get_scratch(mc_ctx_t *ctx, volatile uint32_t **cpu_ptr,
                                   uint64_t *gpu_va)
{
  /* Trivial getter over the dedicated scratch dword that
   * mc_compute_module_init allocates via mc_va_space_alloc_scratch.
   * The dword has its own RM hMemory and GPU MMU PTE; the carrier
   * VAS owns both for the lifetime of the compute channel. */
  const struct mc_compute_module *mod;

  if (ctx == NULL || cpu_ptr == NULL || gpu_va == NULL) return MC_EINVAL;
  if (!ctx->ch[MC_CH_COMPUTE].h_channel) return MC_EINTERNAL;

  mod = &ctx->ch[MC_CH_COMPUTE].x.compute.module;
  if (mod->scratch_cpu == NULL || mod->scratch_gpu_va == 0)
    return MC_EINTERNAL;

  *cpu_ptr = mod->scratch_cpu;
  *gpu_va  = mod->scratch_gpu_va;
  return MC_OK;
}

/* ── SM-thread-rung D2H ────────────────────────────────────────────
 * Same chain-of-evidence as mc_memcpy_d2h_gpu_doorbell_ce, but the
 * GPU-issued BAR1 doorbell write is performed by an SM thread of
 * mc_doorbell_kernel rather than by the DMA channel's CE LAUNCH_DMA.
 * No copy engine is involved in the doorbell ring — just a single
 * warp's STG.E.STRONG.SYS to a GPU-mapped BAR1 page.
 *
 * Both timeouts (compute sema then UVM-channel sema) are charged
 * against the same wall-clock budget so a wedged UVM channel can't
 * hide behind the compute kernel having succeeded.
 */
mc_status_t mc_memcpy_d2h_gpu_doorbell_sm(mc_ctx_t *ctx, void *dst_host,
                                          const void *src_dev, size_t n)
{
  mc_channel_t         *prim;
  mc_channel_t         *dma;
  mc_channel_t         *cmp;
  NvU64                 dbell_gpu_va;
  uint32_t             *prim_pb_end;
  uint32_t              prim_copy_bytes;
  struct timespec       t0;
  mc_status_t           rc;

  if (ctx == NULL || dst_host == NULL || src_dev == NULL) return MC_EINVAL;
  if (n == 0 || n > MC_MAX_TRANSFER_SIZE)                 return MC_EINVAL;
  dbell_gpu_va = ctx->vas[MC_VAS_PRIMARY_CARRIER].dbell_gpu_va;
  if (!ctx->ch[MC_CH_DMA].h_channel || !dbell_gpu_va)
    return MC_EINTERNAL;
  if (!ctx->ch[MC_CH_COMPUTE].h_channel)
    return MC_EINTERNAL;

  prim = &ctx->ch[MC_CH_PRIMARY];

  /* Arm primary's HBM->DRAM submission: build pushbuffer + write
   * GPFIFO entry + advance USERD GPPut, but DO NOT ring primary's
   * vf_doorbell.  Same shape as mc_memcpy_d2h_gpu_doorbell_ce. */
  prim->sema_payload++;
  if (prim->sema_payload == 0) prim->sema_payload = 1;

  prim_pb_end = mc_write_transfer_methods(
      prim->pb_cpu,
      (NvU64)(uintptr_t)src_dev, (NvU64)(uintptr_t)dst_host,
      (uint32_t)n, prim->sema_gpu_va, prim->sema_payload);
  prim_copy_bytes = (uint32_t)((prim_pb_end - prim->pb_cpu) * sizeof(uint32_t));

  rc = mc_channel_arm(prim, prim_copy_bytes, &t0);
  if (rc != MC_OK) return rc;

  /* Launch the compute kernel.  Its STG.E.STRONG.SYS writes the UVM
   * channel's work_submit_token to (carrier-VAS dbell_gpu_va + 0x90)
   * — the exact MMIO address the host would write to ring the UVM
   * channel.
   *
   * mc_compute_doorbell_kernel returns only after the kernel's
   * release semaphore fires, which (per SET_REPORT_SEMAPHORE_D
   * _FLUSH_DISABLE = _FALSE) waits for L2 flush — therefore the
   * doorbell write has physically reached BAR1 by the time we
   * return. */
  rc = mc_compute_doorbell_kernel(ctx,
                                  dbell_gpu_va + MC_VF_DOORBELL_OFFSET,
                                  prim->work_submit_token);
  if (rc != MC_OK) return rc;

  /* Poll primary sema with the t0 captured before arm so the
   * end-to-end budget can't hide a wedged primary behind a
   * successful compute launch. */
  return mc_channel_poll_sema(prim, t0);
}
