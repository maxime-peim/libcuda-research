/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_core.c — libmc lifecycle layer.  Owns the per-kind channel
 * init/fini routines, mc_init/mc_fini, the allocation table,
 * mc_malloc_device, mc_malloc_host, mc_free, mc_memcpy/d2h,
 * and the two CE-doorbell demos (mc_dbell_demo_ring,
 * mc_memcpy_gpu_doorbell_ce).
 *
 * The underlying machinery lives in sibling TUs:
 *   mc_rm.c       RM ioctl wrappers + typed object allocators
 *   mc_uvm.c      UVM ioctl wrappers
 *   mc_submit.c   method-stream builders + channel submit primitives
 *   mc_vaspace.c  VA pool + mc_va_space_* helpers
 *
 * Paper-F1 invariant: every allocation with a CPU alias is anchored
 * into the VA pool so CPU VA == GPU VA.  Enforced by mc_malloc_host
 * routing through rm_alloc_sysmem_at + uvm_map_buffer_at (both with
 * caller-supplied want_va inside the pool).  mc_malloc_device has no
 * CPU alias and therefore doesn't need (and doesn't use) the pool.
 */

#define _GNU_SOURCE
#include <emmintrin.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "nvtypes.h"                   /* NvU32 / NvU64 / NvHandle */
#include "class/cla06fsubch.h"          /* NVA06F_SUBCHANNEL_COPY_ENGINE / _COMPUTE */
#include "class/cl2080_notification.h"  /* NV2080_ENGINE_TYPE_GR0 (needs nvtypes.h above) */
#include "ctrl/ctrlc36f.h"              /* NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN */
#include "mc_compute_qmd.h"             /* mc_doorbell_kernel_sass — staged once
                                         * at compute-channel init */
#include "mc_sm_owner_sass.h"           /* mc_sm_owner_kernel_sass — second
                                         * embedded kernel; the SM thread
                                         * authors the entire CE-channel
                                         * submission (PB+GPFIFO+USERD+
                                         * doorbell) instead of just the
                                         * doorbell write. */

#include "mc_internal.h"

/* mc_memcpy casts size_t n to uint32_t for the method
 * stream and the SM-owner CB0.  The size gate above each cast is
 * `n > MC_MAX_TRANSFER_SIZE`, so n <= MC_MAX_TRANSFER_SIZE always.
 * If MC_MAX_TRANSFER_SIZE is ever raised above UINT32_MAX (e.g. to
 * support chunked >4 GiB transfers), the cast will silently
 * truncate and the GPU will copy a tiny size while reading/writing
 * far past the buffer — a memory-corruption foot-gun.  Trip the
 * build before that ships. */
_Static_assert(MC_MAX_TRANSFER_SIZE <= UINT32_MAX,
               "(uint32_t)n cast in mc_memcpy would truncate; "
               "raise MC_MAX_TRANSFER_SIZE only with a chunking loop");

/* ── Allocation table helpers ──────────────────────────────────────────────
 *
 * The allocation table is a fixed-size flat array (MC_ALLOC_TABLE_SLOTS
 * = 256) in mc_ctx_t that tracks every user buffer returned by
 * mc_malloc_{device,host}.  It exists so mc_free can take the opaque
 * pointer the user got back and recover the associated RM handle, GPU
 * VA, size, and is_device bit.
 *
 * A flat linear scan is fast enough at 256 entries, and sidesteps
 * allocator-inside-allocator concerns (a hash table would need its own
 * memory management that could interact badly with the library's VA
 * pool invariants).  A free slot is one with ptr == NULL; mc_free
 * clears the whole struct on release.
 */

static mc_alloc_t *alloc_table_find_free(mc_ctx_t *ctx)
{
  int i;
  for (i = 0; i < MC_ALLOC_TABLE_SLOTS; i++)
    if (ctx->allocs[i].ptr == NULL)
      return &ctx->allocs[i];
  return NULL;
}

mc_alloc_t *alloc_table_lookup(mc_ctx_t *ctx, const void *ptr)
{
  int i;
  if (ptr == NULL)
    return NULL;
  for (i = 0; i < MC_ALLOC_TABLE_SLOTS; i++)
    if (ctx->allocs[i].ptr == ptr)
      return &ctx->allocs[i];
  return NULL;
}

typedef struct {
  NvU64 base;
  NvU64 length;
  NvU64 offset;
} mc_uvm_external_segment_t;

static NvU64 mc_host_page_size(void)
{
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return 4096;
  return (NvU64)page_size;
}

static int mc_build_host_register_segments(const void *ptr, size_t n,
                                           mc_uvm_external_segment_t segs[MC_UVM_EXTERNAL_SEGMENTS_MAX],
                                           int *out_count,
                                           NvU64 *out_map_base,
                                           NvU64 *out_map_size)
{
  uintptr_t user_va;
  uintptr_t user_end;
  uintptr_t map_base;
  uintptr_t map_end;
  uintptr_t page_size;
  uintptr_t page_mask;
  NvU64     map_size;
  int       count = 0;

  if (ptr == NULL || n == 0 || segs == NULL || out_count == NULL
      || out_map_base == NULL || out_map_size == NULL)
    return -1;

  page_size = (uintptr_t)mc_host_page_size();
  if (page_size == 0 || (page_size & (page_size - 1)) != 0)
    return -1;
  page_mask = page_size - 1;

  user_va = (uintptr_t)ptr;
  if ((uintptr_t)n > UINTPTR_MAX - user_va)
    return -1;
  user_end = user_va + (uintptr_t)n;
  if (user_end > UINTPTR_MAX - page_mask)
    return -1;

  map_base = user_va & ~page_mask;
  map_end  = (user_end + page_mask) & ~page_mask;
  if (map_end <= map_base)
    return -1;

  map_size = (NvU64)(map_end - map_base);

  if (((user_va & page_mask) == 0 && (((uintptr_t)n) & page_mask) == 0)
      || map_size <= page_size)
  {
    segs[count++] = (mc_uvm_external_segment_t) {
      .base   = (NvU64)map_base,
      .length = map_size,
      .offset = 0,
    };
  }
  else
  {
    segs[count++] = (mc_uvm_external_segment_t) {
      .base   = (NvU64)map_base,
      .length = (NvU64)page_size,
      .offset = 0,
    };

    if (map_size > 2 * (NvU64)page_size)
    {
      segs[count++] = (mc_uvm_external_segment_t) {
        .base   = (NvU64)(map_base + page_size),
        .length = map_size - 2 * (NvU64)page_size,
        .offset = (NvU64)page_size,
      };
    }

    segs[count++] = (mc_uvm_external_segment_t) {
      .base   = (NvU64)(map_end - page_size),
      .length = (NvU64)page_size,
      .offset = map_size - (NvU64)page_size,
    };
  }

  *out_count    = count;
  *out_map_base = (NvU64)map_base;
  *out_map_size = map_size;
  return 0;
}

static void mc_free_registered_uvm_ranges(mc_ctx_t *ctx, const mc_alloc_t *slot)
{
  int i;

  if (ctx == NULL || slot == NULL || ctx->uvm_fd < 0)
    return;

  for (i = slot->uvm_segment_count; i > 0; i--)
    uvm_free_range(ctx->uvm_fd, slot->uvm_segment_base[i - 1],
                   "h_registered");
}

/* ── Channel alloc/free core helpers ─────────────────────────────────
 *
 * Both mc_dma_channel_init and mc_compute_channel_init share the same
 * skeleton: allocate three sysmem buffers (gpfifo+USERD, pushbuffer,
 * semaphore), DMA-map each into the carrier VAS, allocate TSG + channel
 * + engine-class object, fetch the work-submit-token, and schedule.
 * mc_channel_alloc_via_carrier encapsulates the shared sequence; each
 * core buffer gets its own NV50_MEMORY_VIRTUAL carrier via
 * mc_va_space_dma_map_resource — libcuda's recipe (one carrier per
 * source hMemory, RM picks the GPU VA).
 */

/*
 * mc_channel_alloc_via_carrier — allocate the gpfifo/pb/sema buffers,
 * DMA-map them into the carrier VAS, allocate TSG + channel +
 * engine-class object, fetch the work-submit-token, schedule.  The
 * engine-class branch (CE vs HOPPER_COMPUTE_A) is determined by
 * ch->engine_type: NV2080_ENGINE_TYPE_GR0 -> compute, anything else
 * -> CE.
 *
 * Caller must have already populated ch->type, ch->vas_id,
 * ch->subchannel, and ch->engine_type.
 *
 * Returns 0 on success, -1 on failure.  Partial state is freed by
 * mc_channel_free_core.
 */
/* Allocate one channel-resource buffer (PB / GPFIFO+USERD / sema)
 * with a host-visible CPU pointer.  Aperture is picked by the carrier
 * kind:
 *
 *   MC_VAS_KIND_CARRIER     → sysmem allocation; CPU pointer is the
 *                             real backing memory.  (today's path)
 *   MC_VAS_KIND_CARRIER_FB  → vidmem allocation; CPU pointer is a
 *                             BAR1-aliased view of the FB pages
 *                             (REFLECTED mapping).  PBDMA / SM read
 *                             FB directly through the GPU MMU; the
 *                             host alias is for one-time SET_OBJECT
 *                             pushbuffer writes and diagnostics.
 *
 * `*out_h_mem` and `*out_cpu_ptr` are populated on success.  Returns
 * 0 on success, -1 on failure.
 */
static int mc_channel_alloc_resource(mc_ctx_t *ctx, mc_va_space_kind_t kind,
                                     NvU64 size,
                                     NvHandle *out_h_mem, void **out_cpu_ptr)
{
  *out_cpu_ptr = NULL;
  if (kind == MC_VAS_KIND_CARRIER)
  {
    *out_h_mem = rm_alloc_sysmem_wc_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                                    ctx->h_device, size, NULL, out_cpu_ptr);
  }
  else if (kind == MC_VAS_KIND_CARRIER_FB)
  {
    /* No want_va: FB BAR1 aliases for channel resources don't need
     * the Paper-F1 VA pool — they have no GPU VA == CPU VA invariant
     * to maintain (the GPU sees them via the carrier's bump-allocated
     * GPU VA, which is unrelated to where the host BAR1 mmap lands). */
    *out_h_mem = rm_alloc_vidmem_bar1_at(ctx->ctl_fd, MC_DEVICE_DEV_PATH,
                                         ctx->h_client, ctx->h_device,
                                         size, NULL, out_cpu_ptr);
  }
  else
  {
    return -1;
  }
  return (*out_h_mem == 0 || *out_cpu_ptr == NULL) ? -1 : 0;
}

static int mc_channel_alloc_via_carrier(mc_ctx_t *ctx, mc_channel_t *ch)
{
  NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS token_params = {};
  mc_va_space_t *vas = &ctx->vas[ch->vas_id];
  void *gpfifo_cpu_ptr = NULL;
  void *pb_cpu_ptr     = NULL;
  void *sema_cpu_ptr   = NULL;

  if (!mc_va_space_kind_is_carrier(vas->kind)) return -1;

  /* Three core allocations.  Aperture chosen by the VAS kind:
   * sysmem-carrier gets sysmem allocs, FB-carrier gets vidmem +
   * BAR1-aliased CPU mappings.  Either way the CPU pointer is
   * read/writeable for bring-up + diagnostics. */
  if (mc_channel_alloc_resource(ctx, vas->kind, MC_GPFIFO_USERD_SIZE,
                                &ch->h_gpfifo_mem, &gpfifo_cpu_ptr) != 0)
    return -1;
  ch->gpfifo_cpu  = gpfifo_cpu_ptr;
  ch->gpfifo_ring = (volatile uint32_t *)gpfifo_cpu_ptr;
  ch->userd_cpu   = (char *)gpfifo_cpu_ptr + MC_USERD_OFFSET;
  ch->userd       = (volatile HopperAControlGPFifo *)ch->userd_cpu;

  if (mc_channel_alloc_resource(ctx, vas->kind, MC_GPFIFO_USERD_SIZE,
                                &ch->h_pb_mem, &pb_cpu_ptr) != 0)
    return -1;
  ch->pb_cpu = (uint32_t *)pb_cpu_ptr;

  if (mc_channel_alloc_resource(ctx, vas->kind, MC_GPFIFO_USERD_SIZE,
                                &ch->h_sema_mem, &sema_cpu_ptr) != 0)
    return -1;
  ch->sema_ptr = (volatile uint32_t *)sema_cpu_ptr;
  *ch->sema_ptr = 0;

  /* DMA-map each core buffer through libcuda's per-resource carrier
   * shape (one NV50_MEMORY_VIRTUAL per source, dmaOffset=0, RM picks
   * GPU VA).  See reverse/mc/README.md, "VA spaces". */
  ch->gpfifo_gpu_va = mc_va_space_dma_map_resource(
      ctx, vas, ch->h_gpfifo_mem, MC_GPFIFO_USERD_SIZE);
  if (!ch->gpfifo_gpu_va) return -1;

  ch->pb_gpu_va = mc_va_space_dma_map_resource(
      ctx, vas, ch->h_pb_mem, MC_GPFIFO_USERD_SIZE);
  if (!ch->pb_gpu_va) return -1;

  ch->sema_gpu_va = mc_va_space_dma_map_resource(
      ctx, vas, ch->h_sema_mem, MC_GPFIFO_USERD_SIZE);
  if (!ch->sema_gpu_va) return -1;

  /* FB-carrier victim DMA channel: allocate a sysmem release-sema
   * cell that the SM-author path uses as the active sema for both
   * H2D and D2H.  PCIe producer-side ordering of posted writes from
   * one Requester ID into one address space (sysmem) then guarantees
   * the sema MWr cannot land at host DRAM ahead of any earlier data
   * MWr from the same channel — so the host polling this cell is a
   * valid completion observable.  The HBM cell allocated above
   * remains for diagnostic dumps but isn't on the success path.
   * See mc_sm_owner.c::mc_sm_owner_submit for the unified routing. */
  if (vas->kind == MC_VAS_KIND_CARRIER_FB
      && ch->role == MC_ROLE_SM_VICTIM_DMA)
  {
    void *sema_sys_cpu = NULL;
    ch->sema_sysmem_gpu_va = mc_va_space_alloc_scratch_wc(
        ctx, vas, MC_SEMA_SIZE, /*align=*/0,
        &ch->h_sema_sysmem_mem, &sema_sys_cpu);
    if (!ch->sema_sysmem_gpu_va) return -1;
    ch->sema_sysmem_ptr = (volatile uint32_t *)sema_sys_cpu;
    *ch->sema_sysmem_ptr = 0;
  }

  /* TSG + channel + engine-class object on the carrier's VAS. */
  ch->h_tsg = rm_alloc_tsg(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                           ch->engine_type, vas->h_vaspace);
  if (!ch->h_tsg) return -1;

  ch->h_channel = rm_alloc_channel(
      ctx->ctl_fd, ctx->h_client, ch->h_tsg, ch->gpfifo_gpu_va,
      MC_GPFIFO_ENTRIES, ch->engine_type, ch->h_gpfifo_mem,
      MC_USERD_OFFSET);
  if (!ch->h_channel) return -1;

  if (ch->engine_type == NV2080_ENGINE_TYPE_GR0)
    ch->h_engine = rm_alloc_compute(ctx->ctl_fd, ctx->h_client, ch->h_channel);
  else
    ch->h_engine = rm_alloc_ce(ctx->ctl_fd, ctx->h_client, ch->h_channel,
                               ch->engine_type);
  if (!ch->h_engine) return -1;

  /* Work-submit-token + schedule. */
  if (rm_control(ctx->ctl_fd, ctx->h_client, ch->h_channel,
                 NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &token_params,
                 sizeof(token_params)) != 0)
    return -1;
  ch->work_submit_token = token_params.workSubmitToken;

  if (rm_gpfifo_schedule(ctx->ctl_fd, ctx->h_client, ch->h_channel) != 0)
    return -1;

  ch->gp_put       = 0;
  ch->sema_payload = 0;
  return 0;
}

/*
 * mc_channel_free_core — drain + disable the channel, then DMA-unmap
 * and free the core (gpfifo/pb/sema) plumbing.  Best-effort: every
 * step checks its own precondition so partial-init states are safe.
 *
 * Callers run this FIRST in their kind-specific *_channel_fini, then
 * unmap and free their kind-specific extras (qmd buffers, BAR1
 * doorbell mapping, VAS, etc.).  This ordering matters: the channel
 * may have outstanding GPU work referencing those PTEs; drain +
 * disable here ensures no SM/CE/PBDMA touches them after we return.
 */
static void mc_channel_free_core(mc_ctx_t *ctx, mc_channel_t *ch)
{
  /* Drain (best-effort). */
  if (ch->userd != NULL)
    drain_channel(ch->userd, MC_TIMEOUT_MS);

  /* Disable the channel before unmapping anything it might still
   * dereference. */
  if (ch->h_channel && ctx->ctl_fd >= 0)
    rm_channel_disable(ctx->ctl_fd, ctx->h_client, ch->h_channel);

  /* DMA-unmap each per-resource carrier and free the carrier handle.
   * The carrier was allocated by mc_va_space_dma_map_resource and
   * recorded in vas->carriers[]; we take ownership at channel free
   * time so the h_mem can be safely freed below.
   *
   * Order matches alloc order: gpfifo, pb, sema (oldest carriers
   * first).  We pull each entry out of the table by handle match
   * and shift survivors down. */
  mc_va_space_release_carrier(ctx, &ctx->vas[ch->vas_id], ch->h_gpfifo_mem);
  mc_va_space_release_carrier(ctx, &ctx->vas[ch->vas_id], ch->h_pb_mem);
  mc_va_space_release_carrier(ctx, &ctx->vas[ch->vas_id], ch->h_sema_mem);
  /* No-op when h_sema_sysmem_mem is 0 (non-FB-victim channels). */
  mc_va_space_release_carrier(ctx, &ctx->vas[ch->vas_id],
                              ch->h_sema_sysmem_mem);

  /* CPU-side mappings. */
  if (ch->gpfifo_cpu)       munmap(ch->gpfifo_cpu, MC_GPFIFO_USERD_SIZE);
  if (ch->pb_cpu)           munmap(ch->pb_cpu,     MC_GPFIFO_USERD_SIZE);
  if (ch->sema_ptr)         munmap((void *)ch->sema_ptr, MC_GPFIFO_USERD_SIZE);
  if (ch->sema_sysmem_ptr)  munmap((void *)ch->sema_sysmem_ptr, MC_SEMA_SIZE);

  /* RM handles in reverse alloc order. */
  if (ctx->ctl_fd >= 0 && ctx->h_client)
  {
    if (ch->h_engine)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ch->h_channel,
                     ch->h_engine, "h_engine");
    if (ch->h_channel)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ch->h_tsg,
                     ch->h_channel, "h_channel");
    if (ch->h_tsg)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_tsg, "h_tsg");
    if (ch->h_sema_sysmem_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_sema_sysmem_mem, "h_sema_sysmem_mem");
    if (ch->h_sema_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_sema_mem, "h_sema_mem");
    if (ch->h_pb_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_pb_mem, "h_pb_mem");
    if (ch->h_gpfifo_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_gpfifo_mem, "h_gpfifo_mem");
  }
}

/* ── DMA-channel bring-up ─────────────────────────────────────────────
 *
 * Allocates a CE channel on a carrier VAS (sysmem-carrier or
 * FB-carrier).  Channel core (PB/GPFIFO/USERD/sema) follows the
 * VAS's aperture rule via mc_channel_alloc_resource.  The kind-
 * specific extras are the token cell — a small sysmem scratch
 * DMA-mapped into the carrier VAS, used as the CE source for
 * doorbell-write demos.
 *
 * `slot` selects which mc_ctx.vas[] entry to attach to; `role`
 * tags the resulting channel (HOST_DMA or SM_VICTIM_DMA).  The
 * caller's mc_vas_add_channel CHECK enforces role uniqueness within
 * the VAS.  Both DMA roles use the same bring-up code; only the role
 * tag and intended dispatch differ.
 */
static int mc_dma_channel_init(mc_ctx_t *ctx, mc_vas_t slot,
                               mc_channel_role_t role,
                               NvU32 lce_engine_type)
{
  mc_va_space_t         *vas = &ctx->vas[slot];
  mc_channel_t          *ch  = mc_vas_add_channel(vas, role);
  struct mc_dma_extras  *ex;
  void                  *cpu = NULL;

  if (ch == NULL) return -1;
  ex              = &ch->x.dma;
  ch->type        = MC_CH_TYPE_DMA;
  ch->vas_id      = slot;
  ch->subchannel  = NVA06F_SUBCHANNEL_COPY_ENGINE;
  ch->engine_type = lce_engine_type;

  if (mc_channel_alloc_via_carrier(ctx, ch) != 0) return -1;

  /* Token cell: always sysmem, regardless of carrier kind.  The
   * doorbell-write demo's CE op reads 4 bytes from this cell and
   * writes them to the BAR1 doorbell page; sysmem keeps the host
   * able to read/write the cell via a normal CPU pointer. */
  ex->token_cell_gpu_va = mc_va_space_alloc_scratch_wc(
      ctx, vas, 4, 4,
      &ex->h_token_mem, &cpu);
  if (!ex->token_cell_gpu_va) return -1;
  ex->token_cell = (volatile uint32_t *)cpu;
  *ex->token_cell = 0;

  DEBUG_LOG("mc_dma_channel ok: vas=%d role=%d ch=0x%x token_gpu_va=0x%llx work_token=0x%x",
            (int)slot, (int)role, ch->h_channel,
            (unsigned long long)ex->token_cell_gpu_va,
            ch->work_submit_token);
  return 0;
}

static void mc_dma_channel_fini(mc_ctx_t *ctx, mc_vas_t slot,
                                mc_channel_role_t role)
{
  mc_channel_t         *ch = mc_vas_find_channel(&ctx->vas[slot], role);
  struct mc_dma_extras *ex;
  if (ch == NULL) return;
  ex = &ch->x.dma;

  /* Core first: drain + disable so no in-flight CE op can be
   * referencing the token cell PTE we're about to unmap.  The core
   * also releases the gpfifo/pb/sema per-resource carriers and frees
   * the TSG/channel/engine + hMemory handles. */
  mc_channel_free_core(ctx, ch);

  /* Token cell teardown: release its per-resource carrier, then free
   * the sysmem hMemory.  The carrier VAS + BAR1 doorbell PTE belong
   * to globals, not us. */
  mc_va_space_release_carrier(ctx, &ctx->vas[slot], ex->h_token_mem);
  if (ex->token_cell)
    munmap((void *)ex->token_cell, 4);
  if (ctx->ctl_fd >= 0 && ctx->h_client && ex->h_token_mem)
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                   ex->h_token_mem, "dma_ch.token_cell");

  memset(ch, 0, sizeof(*ch));
}

/* ── COMPUTE-channel bring-up (mc_compute_channel_init) ────────────────
 *
 * Allocates a HOPPER_COMPUTE_A channel on the carrier VAS — the SAME
 * VAS the DMA channel uses, so SM threads walk the same PDB and can
 * reach the BAR1 doorbell PTE at ctx->vas[CARRIER].dbell_gpu_va.
 *
 * Differences from mc_dma_channel_init:
 *   - engineType = NV2080_ENGINE_TYPE_GR0 (compute, runlist 0 on Hopper)
 *     rather than an LCE.
 *   - Engine class = HOPPER_COMPUTE_A (0xCBC0) rather than HOPPER_DMA_COPY_A.
 *   - Loads a single compute module (QMD/CB0/SASS) into the carrier VAS.
 */

/* Allocate the QMD/CB0/SASS triple for one compute kernel and stage
 * its SASS bytes.  Each region lives in the carrier VAS via the
 * scratch allocator — sysmem-backed, GPU MMU PTE per region.  All
 * sysmem (not vidmem): keeps bring-up simple; libcuda would put
 * these in a vidmem code/data pool for performance, but a sysmem
 * alloc with a GPU MMU PTE is functionally identical for one launch. */
static int mc_compute_module_init(mc_ctx_t *ctx, mc_vas_t slot,
                                  struct mc_compute_module *mod,
                                  const uint8_t *sass_bytes,
                                  size_t sass_len)
{
  mc_va_space_t *vas = &ctx->vas[slot];
  void          *qmd_cpu_ptr     = NULL;
  void          *cb0_cpu_ptr     = NULL;
  void          *sass_cpu_ptr    = NULL;
  void          *scratch_cpu_ptr = NULL;
  NvU64          scratch_base_gpu_va;

  mod->qmd_gpu_va = mc_va_space_alloc_scratch_wc(
      ctx, vas, MC_GPFIFO_USERD_SIZE, 0, &mod->h_qmd_mem, &qmd_cpu_ptr);
  if (!mod->qmd_gpu_va) return -1;
  mod->qmd_cpu = (uint8_t *)qmd_cpu_ptr;

  mod->cb0_gpu_va = mc_va_space_alloc_scratch_wc(
      ctx, vas, MC_GPFIFO_USERD_SIZE, 0, &mod->h_cb0_mem, &cb0_cpu_ptr);
  if (!mod->cb0_gpu_va) return -1;
  mod->cb0_cpu = (uint8_t *)cb0_cpu_ptr;

  mod->sass_gpu_va = mc_va_space_alloc_scratch_wc(
      ctx, vas, MC_GPFIFO_USERD_SIZE, 0, &mod->h_sass_mem, &sass_cpu_ptr);
  if (!mod->sass_gpu_va) return -1;
  mod->sass_cpu = (uint8_t *)sass_cpu_ptr;

  /* Dedicated scratch dword exposed via mc_compute_get_scratch.
   *
   * This target proves the SM's global store is visible to the host.
   * Use BAR1-aliased vidmem rather than tiny sysmem here: on H200 with
   * the 610 ABI the old sysmem scratch store is not visible even though
   * the kernel completion semaphore fires, while the same kernel's BAR1
   * doorbell store is the proven working path.
   */
  mod->h_scratch_mem = rm_alloc_vidmem_bar1_at(ctx->ctl_fd, MC_DEVICE_DEV_PATH,
                                               ctx->h_client, ctx->h_device,
                                               MC_GPFIFO_USERD_SIZE, NULL,
                                               &scratch_cpu_ptr);
  if (!mod->h_scratch_mem || scratch_cpu_ptr == NULL) return -1;

  scratch_base_gpu_va = mc_va_space_dma_map_resource(
      ctx, vas, mod->h_scratch_mem, MC_GPFIFO_USERD_SIZE);
  if (!scratch_base_gpu_va)
  {
    munmap(scratch_cpu_ptr, MC_GPFIFO_USERD_SIZE);
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                   mod->h_scratch_mem, "compute_module.h_scratch_mem_rollback");
    mod->h_scratch_mem = 0;
    return -1;
  }
  mod->scratch_cpu    = (volatile uint32_t *)((uint8_t *)scratch_cpu_ptr + 32);
  mod->scratch_gpu_va = scratch_base_gpu_va + 32;

  /* Stage SASS bytes once; per-call code only patches QMD + CB0. */
  memcpy(mod->sass_cpu, sass_bytes, sass_len);
  _mm_sfence();
  return 0;
}

static void mc_compute_module_fini(mc_ctx_t *ctx, mc_vas_t slot,
                                   struct mc_compute_module *mod)
{
  /* Release each per-resource carrier in reverse alloc order (LIFO).
   * mc_va_space_release_carrier no-ops if the handle isn't tracked,
   * so partial-init failures are safe. */
  mc_va_space_release_carrier(ctx, &ctx->vas[slot], mod->h_scratch_mem);
  mc_va_space_release_carrier(ctx, &ctx->vas[slot], mod->h_sass_mem);
  mc_va_space_release_carrier(ctx, &ctx->vas[slot], mod->h_cb0_mem);
  mc_va_space_release_carrier(ctx, &ctx->vas[slot], mod->h_qmd_mem);
  /* munmap each CPU mapping created during mc_compute_module_init.
   * These were leaked previously: every mc_ctx teardown left orphan
   * mappings around.  Sizes mirror the alloc sites above. */
  if (mod->scratch_cpu)
    munmap((uint8_t *)mod->scratch_cpu - 32, MC_GPFIFO_USERD_SIZE);
  if (mod->sass_cpu) munmap(mod->sass_cpu, MC_GPFIFO_USERD_SIZE);
  if (mod->cb0_cpu)  munmap(mod->cb0_cpu,  MC_GPFIFO_USERD_SIZE);
  if (mod->qmd_cpu)  munmap(mod->qmd_cpu,  MC_GPFIFO_USERD_SIZE);

  if (ctx->ctl_fd >= 0 && ctx->h_client)
  {
    if (mod->h_scratch_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     mod->h_scratch_mem, "compute_module.h_scratch_mem");
    if (mod->h_sass_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     mod->h_sass_mem, "compute_module.h_sass_mem");
    if (mod->h_cb0_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     mod->h_cb0_mem, "compute_module.h_cb0_mem");
    if (mod->h_qmd_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     mod->h_qmd_mem, "compute_module.h_qmd_mem");
  }
}

static int mc_compute_channel_init(mc_ctx_t *ctx, mc_vas_t slot)
{
  mc_va_space_t            *vas = &ctx->vas[slot];
  mc_channel_t             *ch  = mc_vas_add_channel(vas, MC_ROLE_COMPUTE);
  struct mc_compute_extras *ex;

  if (ch == NULL) return -1;
  ex              = &ch->x.compute;
  ch->type        = MC_CH_TYPE_COMPUTE;
  ch->vas_id      = slot;
  ch->subchannel  = NVA06F_SUBCHANNEL_COMPUTE;
  ch->engine_type = NV2080_ENGINE_TYPE_GR0;

  if (mc_channel_alloc_via_carrier(ctx, ch) != 0) return -1;

  /* Load the embedded compute kernels.  Two modules per channel:
   *   - mc_doorbell_kernel: tiny `*dst = token` kernel used by
   *     mc_compute_doorbell_kernel / mc_memcpy_gpu_doorbell_sm.
   *   - sm_owner_kernel: the SM-authors-channel kernel used by
   *     mc_sm_owner_*.  Authors PB+GPFIFO+USERD+doorbell from a
   *     single SM thread instead of the host doing it.
   * Each module owns its own QMD/CB0/SASS triple in this VAS
   * (sysmem-backed via the scratch allocator regardless of carrier
   * kind — only the channel-core PB/GPFIFO/USERD/sema follow the
   * carrier's aperture rule). */
  if (mc_compute_module_init(ctx, slot, &ex->module,
                             mc_doorbell_kernel_sass,
                             mc_doorbell_kernel_sass_len) != 0)
    return -1;

  if (mc_compute_module_init(ctx, slot, &ex->sm_owner_module,
                             mc_sm_owner_kernel_sass,
                             MC_SM_OWNER_SASS_LEN) != 0)
    return -1;

  DEBUG_LOG("mc_compute_channel ok: vas=%d tsg=0x%x ch=0x%x compute=0x%x "
            "gpfifo_gpu_va=0x%llx token=0x%x",
            (int)slot, ch->h_tsg, ch->h_channel, ch->h_engine,
            (unsigned long long)ch->gpfifo_gpu_va,
            ch->work_submit_token);
  return 0;
}

/* Symmetric reverse-order teardown for the compute channel.  Best-effort.
 * Does NOT free the carrier VAS (that belongs to globals). */
static void mc_compute_channel_fini(mc_ctx_t *ctx, mc_vas_t slot)
{
  mc_channel_t             *ch = mc_vas_find_channel(
                                     &ctx->vas[slot], MC_ROLE_COMPUTE);
  struct mc_compute_extras *ex;
  if (ch == NULL) return;
  ex = &ch->x.compute;

  /* Core first: drain + disable so no in-flight kernel can be
   * dereferencing the QMD/CB0/SASS PTEs we're about to unmap. */
  mc_channel_free_core(ctx, ch);

  /* Tear down the loaded compute modules in reverse-init order. */
  mc_compute_module_fini(ctx, slot, &ex->sm_owner_module);
  mc_compute_module_fini(ctx, slot, &ex->module);

  memset(ch, 0, sizeof(*ch));
}

/* ── UVM-channel bring-up (mc_uvm_channel_init) ────────────────────────
 *
 * The UVM channel is the foundation: it owns the UVM-managed VAS
 * that mc_malloc_host / mc_malloc_device anchor user allocations
 * into, and it's the channel mc_memcpy / mc_memcpy submits
 * to.
 *
 * Unlike the DMA / compute channels, this one does NOT use
 * mc_channel_alloc_via_carrier: its GPFIFO + USERD live in vidmem
 * (BAR1-mapped via rm_map_memory_at) and the channel is registered
 * with UVM via UVM_REGISTER_CHANNEL.  The GPFIFO+USERD region must
 * be UVM-mapped before channel construct (gpFifoOffset is read at
 * construct time); the pushbuffer + semaphore are UVM-mapped after.
 *
 * Returns 0 on success, -1 on failure (mc_fini cleans up partial
 * state).
 */
static int mc_uvm_channel_init(mc_ctx_t *ctx, NvU32 lce_engine_type)
{
  NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS token_params = {};
  mc_channel_t         *ch;
  struct mc_uvm_extras *ex;
  void                 *pb_direct   = NULL;
  void                 *sema_direct = NULL;

  ch = mc_vas_add_channel(&ctx->vas[MC_VAS_UVM], MC_ROLE_UVM_CE);
  if (ch == NULL) return -1;

  ch->type        = MC_CH_TYPE_UVM;
  ch->vas_id      = MC_VAS_UVM;
  ch->subchannel  = NVA06F_SUBCHANNEL_COPY_ENGINE;
  ch->engine_type = lce_engine_type;

  ex = &ch->x.uvm;

  /* GPFIFO+USERD vidmem, pushbuffer sysmem, semaphore sysmem — see the
   * comment on the sema allocation below for why it is not in FB. */
  ex->h_gpfifo_userd_mem = rm_alloc_vidmem(ctx->ctl_fd, ctx->h_client,
                                      ctx->h_device, MC_GPFIFO_USERD_SIZE, NULL);
  if (!ex->h_gpfifo_userd_mem) return -1;
  /* The uniform mc_channel_t.h_gpfifo_mem field aliases the same RM
   * handle as ch->x.uvm.h_gpfifo_userd_mem (which the BAR1 mmap also
   * targets); the generic submit primitives use it. */
  ch->h_gpfifo_mem = ex->h_gpfifo_userd_mem;

  pb_direct = va_pool_reserve(MC_PB_SIZE, "pushbuffer");
  if (pb_direct == NULL) return -1;
  ch->h_pb_mem = rm_alloc_sysmem_wc_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                                    ctx->h_device, MC_PB_SIZE, pb_direct,
                                    &pb_direct);
  if (!ch->h_pb_mem) return -1;

  /* The release semaphore lives in sysmem, because that is where libcuda
   * puts it — libcuda's SET_SEMAPHORE_VA resolves to NV01_MEMORY_SYSTEM.
   * The host polls this cell in a tight loop; from an HBM cell that poll
   * is a BAR1 read, so every spin becomes a PCIe transaction competing
   * with the transfer it is waiting on.  Ordering holds either way: the
   * release launch carries FLUSH_ENABLE with FLUSH_TYPE=SYS, and for a
   * sysmem destination PCIe posted-write ordering from a single Requester
   * ID also keeps the release behind the data. */
  sema_direct = va_pool_reserve(MC_SEMA_SIZE, "sema");
  if (sema_direct == NULL) return -1;
  ch->h_sema_mem = rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                                      ctx->h_device, MC_SEMA_SIZE, sema_direct,
                                      &sema_direct);
  if (!ch->h_sema_mem) return -1;

  /* UVM-managed VA space + UVM setup.  The GPFIFO+USERD region must
   * be UVM-mapped before channel construct: HOPPER_CHANNEL_GPFIFO_A
   * reads gpFifoOffset at construct time. */
  if (mc_va_space_init_uvm(ctx) != 0) return -1;

  if (uvm_setup(ctx->ctl_fd, ctx->dev_fd, ctx->h_client, ctx->h_subdevice,
                ctx->vas[MC_VAS_UVM].h_vaspace,
                ctx->gpu_inst_uuid, &ctx->uvm_fd) != 0)
    return -1;

  ex->gpfifo_userd_cpu = va_pool_reserve(MC_GPFIFO_USERD_SIZE, "gpfifo_userd");
  if (ex->gpfifo_userd_cpu == NULL) return -1;
  ex->gpfifo_userd_cpu = rm_map_memory_at(ctx->ctl_fd, MC_DEVICE_DEV_PATH,
                                     ctx->h_client, ctx->h_device,
                                     ex->h_gpfifo_userd_mem, 0,
                                     MC_GPFIFO_USERD_SIZE,
                                     0, ex->gpfifo_userd_cpu);
  if (ex->gpfifo_userd_cpu == NULL) return -1;
  ch->gpfifo_cpu  = ex->gpfifo_userd_cpu;
  ch->gpfifo_ring = (volatile uint32_t *)ex->gpfifo_userd_cpu;
  ch->userd_cpu   = (char *)ex->gpfifo_userd_cpu + MC_USERD_OFFSET;
  ch->userd       = (volatile HopperAControlGPFifo *)ch->userd_cpu;

  ch->gpfifo_gpu_va = uvm_map_buffer_at(ctx->uvm_fd, ctx->dev_fd,
                                        ctx->h_client, ctx->gpu_inst_uuid,
                                        ex->h_gpfifo_userd_mem,
                                        ex->gpfifo_userd_cpu,
                                        MC_GPFIFO_USERD_SIZE, "gpfifo_userd");
  if (!ch->gpfifo_gpu_va) return -1;

  /* TSG + channel + CE engine, all bound to the UVM VAS. */
  ch->h_tsg = rm_alloc_tsg(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                           lce_engine_type,
                           ctx->vas[MC_VAS_UVM].h_vaspace);
  if (!ch->h_tsg) return -1;

  ch->h_channel = rm_alloc_channel(ctx->ctl_fd, ctx->h_client, ch->h_tsg,
                                   ch->gpfifo_gpu_va, MC_GPFIFO_ENTRIES,
                                   lce_engine_type, ex->h_gpfifo_userd_mem,
                                   MC_USERD_OFFSET);
  if (!ch->h_channel) return -1;

  ch->h_engine = rm_alloc_ce(ctx->ctl_fd, ctx->h_client, ch->h_channel,
                             lce_engine_type);
  if (!ch->h_engine) return -1;

  if (rm_control(ctx->ctl_fd, ctx->h_client, ch->h_channel,
                 NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &token_params,
                 sizeof(token_params)) != 0)
    return -1;
  ch->work_submit_token = token_params.workSubmitToken;

  /* UVM-map pushbuffer + semaphore.  User buffers added by mc_malloc_*. */
  ch->pb_gpu_va = uvm_map_buffer_at(ctx->uvm_fd, ctx->dev_fd, ctx->h_client,
                                    ctx->gpu_inst_uuid, ch->h_pb_mem,
                                    pb_direct, MC_PB_SIZE, "pushbuffer");
  if (!ch->pb_gpu_va) return -1;
  ch->pb_cpu = (uint32_t *)pb_direct;

  /* sema_direct already points at the sysmem cell allocated above. */
  ch->sema_gpu_va = uvm_map_buffer_at(ctx->uvm_fd, ctx->dev_fd, ctx->h_client,
                                      ctx->gpu_inst_uuid, ch->h_sema_mem,
                                      sema_direct, MC_SEMA_SIZE, "sema");
  if (!ch->sema_gpu_va) return -1;
  ch->sema_ptr = (volatile uint32_t *)sema_direct;

  if (uvm_register_channel(ctx->uvm_fd, ctx->dev_fd, ctx->h_client,
                           ctx->gpu_inst_uuid, ch->h_channel) != 0)
    return -1;

  if (rm_gpfifo_schedule(ctx->ctl_fd, ctx->h_client, ch->h_channel) != 0)
    return -1;

  ch->gp_put       = 0;
  ch->sema_payload = 0;
  return 0;
}

/* Symmetric reverse-order teardown for the UVM channel.  Best-effort:
 * every step checks its own precondition so partial-init states from
 * mc_uvm_channel_init failures are safe to pass in.  Mirrors the
 * mc_dma_channel_fini / mc_compute_channel_fini structure (drain +
 * disable first, then unmap + free).  Also unregisters the GPU + VA
 * space from UVM here — symmetric reverse of uvm_setup() + UVM_REGISTER_CHANNEL.
 * The DMA / compute channels never talk to UVM, so this state only
 * exists when the UVM channel does. */
static void mc_uvm_channel_fini(mc_ctx_t *ctx)
{
  mc_channel_t         *ch = mc_vas_find_channel(&ctx->vas[MC_VAS_UVM],
                                                 MC_ROLE_UVM_CE);
  struct mc_uvm_extras *ex;
  if (ch == NULL) return;
  ex = &ch->x.uvm;

  /* Drain + disable first, while UVM mappings + RM handles are still
   * intact.  No other GPU work can wake this channel afterwards. */
  if (ch->userd != NULL)
    drain_channel(ch->userd, MC_TIMEOUT_MS);

  if (ch->h_channel && ctx->ctl_fd >= 0)
    rm_channel_disable(ctx->ctl_fd, ctx->h_client, ch->h_channel);
  if (ch->h_channel && ctx->uvm_fd >= 0)
    uvm_unregister_channel(ctx->uvm_fd, ctx->h_client, ch->h_channel);

  /* CPU-side mappings (BAR1 for gpfifo_userd; sysmem for pushbuffer and sema). */
  if (ch->gpfifo_ring) munmap((void *)ch->gpfifo_ring, MC_GPFIFO_USERD_SIZE);
  if (ch->sema_ptr)    munmap((void *)ch->sema_ptr,    MC_SEMA_SIZE);
  if (ch->pb_cpu)      munmap(ch->pb_cpu,              MC_PB_SIZE);

  /* UVM unmaps for gpfifo_userd / pb / sema, then unregister VA space + GPU. */
  if (ctx->uvm_fd >= 0)
  {
    if (ch->pb_gpu_va)
      uvm_unmap_buffer(ctx->uvm_fd, ctx->gpu_inst_uuid, ch->pb_gpu_va,
                       MC_PB_SIZE, "pb");
    if (ch->gpfifo_gpu_va)
      uvm_unmap_buffer(ctx->uvm_fd, ctx->gpu_inst_uuid, ch->gpfifo_gpu_va,
                       MC_GPFIFO_USERD_SIZE, "gpfifo_userd");
    if (ch->sema_gpu_va)
      uvm_unmap_buffer(ctx->uvm_fd, ctx->gpu_inst_uuid, ch->sema_gpu_va,
                       MC_SEMA_SIZE, "sema");
    uvm_unregister_gpu_vaspace(ctx->uvm_fd, ctx->gpu_inst_uuid);
    uvm_unregister_gpu(ctx->uvm_fd, ctx->gpu_inst_uuid);
  }

  /* RM handle frees (reverse of allocation). */
  if (ctx->ctl_fd >= 0 && ctx->h_client)
  {
    if (ch->h_engine)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ch->h_channel,
                     ch->h_engine, "h_ce");
    if (ch->h_channel)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ch->h_tsg,
                     ch->h_channel, "h_channel");
    if (ch->h_tsg)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_tsg, "h_tsg");
    if (ch->h_sema_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_sema_mem, "h_sema_mem");
    if (ex->h_gpfifo_userd_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ex->h_gpfifo_userd_mem, "h_gpfifo_userd_mem");
    if (ch->h_pb_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ch->h_pb_mem, "h_pb_mem");
  }

  memset(ch, 0, sizeof(*ch));
}

/* One line summarising every channel mc_init brought up.  Kept out of mc_init
 * so the six lookups it needs do not land in that function's scope. */
static void mc_log_init_summary(mc_ctx_t *ctx)
{
  mc_channel_t *uvm_ch  = mc_vas_find_channel(&ctx->vas[MC_VAS_UVM],
                                              MC_ROLE_UVM_CE);
  mc_channel_t *sys_dma = mc_vas_find_channel(&ctx->vas[MC_VAS_SYSMEM_CARRIER],
                                              MC_ROLE_HOST_DMA);
  mc_channel_t *sys_cmp = mc_vas_find_channel(&ctx->vas[MC_VAS_SYSMEM_CARRIER],
                                              MC_ROLE_COMPUTE);
  mc_channel_t *fb_dma  = mc_vas_find_channel(&ctx->vas[MC_VAS_FB_CARRIER],
                                              MC_ROLE_HOST_DMA);
  mc_channel_t *fb_smv  = mc_vas_find_channel(&ctx->vas[MC_VAS_FB_CARRIER],
                                              MC_ROLE_SM_VICTIM_DMA);
  mc_channel_t *fb_cmp  = mc_vas_find_channel(&ctx->vas[MC_VAS_FB_CARRIER],
                                              MC_ROLE_COMPUTE);

  DEBUG_LOG("mc_init ok: client=0x%x device=0x%x uvm=0x%x "
            "sys{dma=0x%x compute=0x%x} fb{dma=0x%x sm_victim=0x%x compute=0x%x}",
            ctx->h_client, ctx->h_device,
            uvm_ch  ? uvm_ch->h_channel  : 0,
            sys_dma ? sys_dma->h_channel : 0,
            sys_cmp ? sys_cmp->h_channel : 0,
            fb_dma  ? fb_dma->h_channel  : 0,
            fb_smv  ? fb_smv->h_channel  : 0,
            fb_cmp  ? fb_cmp->h_channel  : 0);
}

mc_status_t mc_init(mc_ctx_t **out_ctx)
{
  mc_ctx_t *ctx;
  NvU32     lce_engine_type;

  if (out_ctx == NULL) return MC_EINVAL;
  *out_ctx = NULL;

  ctx = calloc(1, sizeof(*ctx));
  if (ctx == NULL) { ERROR_LOG("calloc ctx"); return MC_EALLOC; }
  ctx->ctl_fd = ctx->dev_fd = ctx->uvm_fd = -1;

  /* Open device files and link them. */
  ctx->ctl_fd = open(MC_CONTROL_DEV_PATH, O_RDWR | O_CLOEXEC);
  ctx->dev_fd = open(MC_DEVICE_DEV_PATH, O_RDWR | O_CLOEXEC);
  if (ctx->ctl_fd < 0 || ctx->dev_fd < 0)
  {
    ERROR_LOG("open device files: %s", strerror(errno));
    goto fail;
  }
  if (rm_register_client_fd(ctx->ctl_fd, ctx->dev_fd) != 0) goto fail;

  /* VA pool up-front — Paper F1 is non-negotiable. */
  if (va_pool_init() != 0) goto fail;

  /* Root / device / subdevice. */
  ctx->h_client    = rm_alloc_root(ctx->ctl_fd);
  ctx->h_device    = rm_alloc_device(ctx->ctl_fd, ctx->h_client);
  ctx->h_subdevice = rm_alloc_subdevice(ctx->ctl_fd, ctx->h_client,
                                        ctx->h_device);
  if (!ctx->h_client || !ctx->h_device || !ctx->h_subdevice) goto fail;

  /* Hold boost clocks for the life of the client, as libcuda does —
   * otherwise a copy-only workload runs the CE at the idle SM clock.
   * See rm_perf_boost. */
  rm_perf_boost(ctx->ctl_fd, ctx->h_client, ctx->h_subdevice);

  /* Engine selection: a non-GRCE LCE drives the TSG / channel / CE. */
  lce_engine_type = pick_non_grce_lce(ctx->ctl_fd, ctx->h_client,
                                      ctx->h_subdevice);
  if (lce_engine_type == (NvU32)-1) goto fail;

  /* HOPPER_USERMODE_A BAR0 + BAR1.  Both must be alive before the
   * carrier VAS so we can install the BAR1 doorbell PTE. */
  ctx->h_usermode_bar0 = rm_alloc_usermode(ctx->ctl_fd, ctx->h_client,
                                           ctx->h_subdevice, false);
  if (!ctx->h_usermode_bar0) goto fail;
  ctx->usermode_bar0_cpu = rm_map_memory(ctx->ctl_fd, MC_DEVICE_DEV_PATH,
                                         ctx->h_client, ctx->h_subdevice,
                                         ctx->h_usermode_bar0, 0,
                                         MC_USERMODE_SIZE, 0);
  if (ctx->usermode_bar0_cpu == NULL) goto fail;

  ctx->h_usermode_bar1 = rm_alloc_usermode(ctx->ctl_fd, ctx->h_client,
                                           ctx->h_subdevice, true);
  if (!ctx->h_usermode_bar1) goto fail;
  ctx->usermode_bar1_cpu = rm_map_memory(ctx->ctl_fd, MC_DEVICE_DEV_PATH,
                                         ctx->h_client, ctx->h_subdevice,
                                         ctx->h_usermode_bar1, 0,
                                         MC_USERMODE_SIZE, 0);
  if (ctx->usermode_bar1_cpu == NULL) goto fail;
  ctx->vf_doorbell = (volatile uint32_t *)(ctx->usermode_bar1_cpu
                                           + MC_VF_DOORBELL_OFFSET);

  /* Sysmem carrier VAS (DMA + compute channels) + its BAR1 doorbell
   * PTE.  Brought up before the channels themselves so neither has
   * to embed VAS handles in its extras. */
  if (mc_va_space_init_carrier(ctx) != 0) goto fail;
  if (mc_va_space_install_doorbell_pte(ctx, &ctx->vas[MC_VAS_SYSMEM_CARRIER])
      != 0)
    goto fail;

  /* FB carrier VAS (HOST_DMA + SM_VICTIM_DMA + COMPUTE channels with
   * FB-resident, BAR1-aliased channel resources) + its own BAR1
   * doorbell PTE.  Channels here exist for the SM-author hot path
   * where every byte of the submission protocol is FB↔GPU-L2 and the
   * only PCIe MWr is the doorbell. */
  if (mc_va_space_init_carrier_fb(ctx) != 0) goto fail;
  if (mc_va_space_install_doorbell_pte(ctx, &ctx->vas[MC_VAS_FB_CARRIER])
      != 0)
    goto fail;

  /* UVM channel (CE; mc_memcpy/d2h). */
  if (mc_uvm_channel_init(ctx, lce_engine_type) != 0) goto fail;

  /* Sysmem-carrier channels: HOST_DMA (also reused as the SM victim
   * for the existing sysmem SM-author path) + COMPUTE. */
  if (mc_dma_channel_init(ctx, MC_VAS_SYSMEM_CARRIER, MC_ROLE_HOST_DMA,
                          lce_engine_type) != 0) goto fail;
  if (mc_compute_channel_init(ctx, MC_VAS_SYSMEM_CARRIER) != 0) goto fail;

  /* FB-carrier channels: HOST_DMA (host-author CE on FB-resident PB),
   * SM_VICTIM_DMA (CE channel reserved as the SM author's victim),
   * COMPUTE.  All three CE channels reuse the same non-GRCE LCE
   * engine type — Hopper allows multiple channels per engine. */
  if (mc_dma_channel_init(ctx, MC_VAS_FB_CARRIER, MC_ROLE_HOST_DMA,
                          lce_engine_type) != 0) goto fail;
  if (mc_dma_channel_init(ctx, MC_VAS_FB_CARRIER, MC_ROLE_SM_VICTIM_DMA,
                          lce_engine_type) != 0) goto fail;
  if (mc_compute_channel_init(ctx, MC_VAS_FB_CARRIER) != 0) goto fail;

  mc_log_init_summary(ctx);
  *out_ctx = ctx;
  return MC_OK;

fail:
  mc_fini(ctx);
  return MC_EIOCTL;
}

/*
 * mc_fini — orchestrator.  Tears down the three channels in reverse
 * dependency order (compute cohabits dma's VAS so it must go first;
 * dma is independent of UVM; the UVM channel goes last among the
 * channels), THEN reaps any user allocations the caller didn't free
 * (after every channel is drained + disabled, so no in-flight GPU
 * work can possibly reference user PTEs being unmapped — Paper F1
 * invariant), THEN releases process-global state (USERMODE_A
 * mappings + RM handles, fds, ctx).
 *
 * Best-effort: every sub-step checks its own precondition so partial-
 * init states from mc_init failures are safe to pass in (mc_init's
 * fail path calls mc_fini on a half-built context).  Safe against
 * a NULL argument.
 */
void mc_fini(mc_ctx_t *ctx)
{
  int i;

  if (ctx == NULL) return;

  /* Channels: reverse-dependency order, FB carrier first (its compute
   * channel may still be authoring submissions to its SM_VICTIM_DMA),
   * then sysmem carrier, then UVM channel.  Each *_fini drains +
   * disables its channel before unmapping anything, so no GPU work
   * survives. */
  mc_compute_channel_fini(ctx, MC_VAS_FB_CARRIER);
  mc_dma_channel_fini    (ctx, MC_VAS_FB_CARRIER, MC_ROLE_SM_VICTIM_DMA);
  mc_dma_channel_fini    (ctx, MC_VAS_FB_CARRIER, MC_ROLE_HOST_DMA);
  mc_compute_channel_fini(ctx, MC_VAS_SYSMEM_CARRIER);
  mc_dma_channel_fini    (ctx, MC_VAS_SYSMEM_CARRIER, MC_ROLE_HOST_DMA);
  mc_uvm_channel_fini(ctx);

  /* VA spaces: free both carriers (each with its own BAR1 doorbell
   * PTE) and the UVM-managed VAS.  All channel buffers were unmapped
   * above. */
  mc_va_space_fini(ctx, MC_VAS_FB_CARRIER);
  mc_va_space_fini(ctx, MC_VAS_SYSMEM_CARRIER);
  mc_va_space_fini(ctx, MC_VAS_UVM);

  /* Now safe to reap any leftover user allocations: every channel is
   * disabled, so the GPU can't be referencing these PTEs.  This used
   * to run BEFORE channel disable — that violated the F1 invariant
   * (uvm_unmap on an active channel intermittently wedges the GPU). */
  for (i = 0; i < MC_ALLOC_TABLE_SLOTS; i++)
  {
    if (ctx->allocs[i].ptr == NULL) continue;
    /* GPU-mapping teardown gated on VAS.  See mc_free for rationale.
     *
     * Note: by the time we reach this loop the VA spaces themselves
     * have already been freed by mc_va_space_fini above (which
     * NVOS47-unmaps every per-resource carrier).  rm_free_handle on
     * the alloc's h_mem is sufficient here since the parent VAS is
     * already released.  (mc_free runs while the VAS is still live
     * and does the full release-carrier path; this loop only catches
     * user-leaked allocs.) */
    if (ctx->allocs[i].is_registered)
      mc_free_registered_uvm_ranges(ctx, &ctx->allocs[i]);
    else if (ctx->allocs[i].vas == MC_VAS_UVM
        && ctx->uvm_fd >= 0 && ctx->allocs[i].gpu_va)
      uvm_unmap_buffer(ctx->uvm_fd, ctx->gpu_inst_uuid, ctx->allocs[i].gpu_va,
                       ctx->allocs[i].size, "user");
    if (!ctx->allocs[i].is_device && !ctx->allocs[i].is_registered)
      munmap(ctx->allocs[i].ptr, ctx->allocs[i].size);
    if (ctx->allocs[i].h_mem)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     ctx->allocs[i].h_mem, "user_mem");
    ctx->allocs[i].ptr = NULL;
  }

  /* Process-global mappings + RM handles. */
  if (ctx->usermode_bar1_cpu)
    munmap((void *)ctx->usermode_bar1_cpu, MC_USERMODE_SIZE);
  if (ctx->usermode_bar0_cpu)
    munmap((void *)ctx->usermode_bar0_cpu, MC_USERMODE_SIZE);

  if (ctx->ctl_fd >= 0 && ctx->h_client)
  {
    if (ctx->h_usermode_bar1)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_subdevice,
                     ctx->h_usermode_bar1, "h_usermode_bar1");
    if (ctx->h_usermode_bar0)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_subdevice,
                     ctx->h_usermode_bar0, "h_usermode_bar0");
    if (ctx->h_subdevice)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_client,
                     ctx->h_subdevice, "h_subdevice");
    if (ctx->h_device)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_client, ctx->h_device,
                     "h_device");
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_client, ctx->h_client,
                   "h_client");
  }

  /* Close fds (uvm_fd first — its release does a final TLB flush). */
  if (ctx->uvm_fd >= 0) close(ctx->uvm_fd);
  if (ctx->dev_fd >= 0) close(ctx->dev_fd);
  if (ctx->ctl_fd >= 0) close(ctx->ctl_fd);

  free(ctx);
}

/* ── mc_malloc_device / mc_malloc_host / mc_free ─────────────────────────
 *
 * Both allocators record their result in the ctx-local allocation table
 * so mc_free can find them later.  The difference is whether the
 * allocation has a CPU alias:
 *   - mc_malloc_device: NV01_MEMORY_LOCAL_USER + plain uvm_map_buffer
 *     (kernel-chosen PROT_NONE VA).  Returned pointer IS the GPU VA;
 *     CPU dereference is UB.  Use for the source of H2D and destination
 *     of D2H when the CPU won't touch it.
 *   - mc_malloc_host: NV01_MEMORY_SYSTEM + VA-pool-anchored MAP_FIXED +
 *     uvm_map_buffer_at.  CPU VA == GPU VA (Paper F1), both valid, same
 *     pointer.  Use for anything the CPU needs to read/write.
 */

/*
 * Allocate device-only HBM into the requested VAS.  No CPU alias; the
 * returned void* is a GPU VA cast from uintptr_t.  Two VAS arms:
 *
 *   MC_VAS_UVM     — rm_alloc_vidmem + uvm_map_buffer (kernel-chosen
 *                    GPU VA).  Reachable from the UVM channel.
 *   MC_VAS_SYSMEM_CARRIER — mc_va_space_alloc_vidmem (rm_alloc_vidmem +
 *                    rm_map_memory_dma into the carrier).  Reachable
 *                    from the DMA + compute channels.  The 64-KiB
 *                    GPU-VA alignment requirement for vidmem PTEs is
 *                    handled inside the helper.
 *
 * On failure rolls back partial state and returns NULL.
 */
void *mc_malloc_device(mc_ctx_t *ctx, size_t n, mc_vas_t vas)
{
  mc_alloc_t *slot;
  NvHandle    h_mem  = 0;
  NvU64       gpu_va = 0;

  if (ctx == NULL || n == 0 || n > MC_MAX_TRANSFER_SIZE) return NULL;
  if (vas != MC_VAS_UVM
      && vas != MC_VAS_SYSMEM_CARRIER
      && vas != MC_VAS_FB_CARRIER) return NULL;
  slot = alloc_table_find_free(ctx);
  if (slot == NULL) { ERROR_LOG("mc_malloc_device: table full"); return NULL; }

  if (vas == MC_VAS_UVM)
  {
    h_mem = rm_alloc_vidmem(ctx->ctl_fd, ctx->h_client, ctx->h_device, n, NULL);
    if (!h_mem) return NULL;
    gpu_va = uvm_map_buffer(ctx->uvm_fd, ctx->dev_fd, ctx->h_client,
                            ctx->gpu_inst_uuid, h_mem, n, "d_user");
    if (!gpu_va)
    {
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_mem,
                     "d_user_rollback");
      return NULL;
    }
  }
  else /* either carrier — sysmem-carrier or FB-carrier */
  {
    /* mc_va_space_alloc_vidmem owns both the rm_alloc_vidmem and the
     * carrier DMA-map.  Default 64-KiB align is enforced inside.  The
     * helper's only carrier-kind dependency is the per-resource carrier,
     * which is identical for both carriers; user buffers in
     * FB_CARRIER are allocated the same way as in SYSMEM_CARRIER. */
    gpu_va = mc_va_space_alloc_vidmem(ctx, &ctx->vas[vas], n, 0, &h_mem);
    if (!gpu_va) return NULL;
  }

  slot->ptr       = (void *)(uintptr_t)gpu_va;
  slot->h_mem     = h_mem;
  slot->gpu_va    = gpu_va;
  slot->size      = n;
  slot->is_device = true;
  slot->vas       = vas;
  return slot->ptr;
}

/*
 * Allocate host DRAM mapped into the requested VAS.  Two VAS arms:
 *
 *   MC_VAS_UVM     — Paper F1 anchored: caller's pointer is BOTH the CPU
 *                    VA and the GPU VA.  va_pool_reserve →
 *                    rm_alloc_sysmem_at(MAP_FIXED) → uvm_map_buffer_at.
 *   MC_VAS_SYSMEM_CARRIER — kernel-chosen CPU VA from rm_alloc_sysmem_at;
 *                    GPU VA is bump-allocated inside the carrier and
 *                    DIFFERENT from the CPU VA.  Caller passes the
 *                    returned CPU pointer to memcpy/CPU access; uses
 *                    mc_gpu_va() to recover the GPU VA when calling
 *                    primitives that need it directly.
 *
 * On failure rolls back partial state and returns NULL.
 */
static void *malloc_host_impl(mc_ctx_t *ctx, size_t n, mc_vas_t vas,
                              int want_wc)
{
  mc_alloc_t *slot;
  void       *cpu_va = NULL;
  NvHandle    h_mem  = 0;
  NvU64       gpu_va = 0;

  if (ctx == NULL || n == 0 || n > MC_MAX_TRANSFER_SIZE) return NULL;
  if (vas != MC_VAS_UVM
      && vas != MC_VAS_SYSMEM_CARRIER
      && vas != MC_VAS_FB_CARRIER) return NULL;
  slot = alloc_table_find_free(ctx);
  if (slot == NULL) { ERROR_LOG("mc_malloc_host: table full"); return NULL; }

  if (vas == MC_VAS_UVM)
  {
    cpu_va = va_pool_reserve((NvU64)n, "h_user");
    if (cpu_va == NULL) return NULL;
    h_mem = want_wc
              ? rm_alloc_sysmem_wc_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                                      ctx->h_device, (NvU64)n, cpu_va, &cpu_va)
              : rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                                   ctx->h_device, (NvU64)n, cpu_va, &cpu_va);
    if (!h_mem) return NULL;
    gpu_va = uvm_map_buffer_at(ctx->uvm_fd, ctx->dev_fd, ctx->h_client,
                               ctx->gpu_inst_uuid, h_mem, cpu_va, (NvU64)n,
                               "h_user");
    if (!gpu_va)
    {
      munmap(cpu_va, n);
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_mem,
                     "h_user_rollback");
      return NULL;
    }
  }
  else /* either carrier */
  {
    /* No VA-pool anchor: kernel chooses cpu_va, the carrier picks
     * gpu_va independently.  Both are bookkept in mc_alloc_t and
     * mc_gpu_va() recovers gpu_va from the user's CPU pointer.  The
     * sysmem-backed scratch allocator is identical for both carrier
     * kinds (host buffers in the FB carrier are still sysmem; only
     * the FB carrier's *channel resources* — PB/GPFIFO/USERD/sema —
     * live in FB). */
    gpu_va = want_wc
               ? mc_va_space_alloc_scratch_wc(ctx, &ctx->vas[vas],
                                              (NvU64)n, 0, &h_mem, &cpu_va)
               : mc_va_space_alloc_scratch(ctx, &ctx->vas[vas],
                                           (NvU64)n, 0, &h_mem, &cpu_va);
    if (!gpu_va) return NULL;
  }

  slot->ptr       = cpu_va;
  slot->h_mem     = h_mem;
  slot->gpu_va    = gpu_va;
  slot->size      = n;
  slot->is_device = false;
  slot->vas       = vas;
  return cpu_va;
}

void *mc_malloc_host(mc_ctx_t *ctx, size_t n, mc_vas_t vas)
{
  return malloc_host_impl(ctx, n, vas, /*want_wc=*/0);
}

void *mc_malloc_host_wc(mc_ctx_t *ctx, size_t n, mc_vas_t vas)
{
  return malloc_host_impl(ctx, n, vas, /*want_wc=*/1);
}

mc_status_t mc_host_register(mc_ctx_t *ctx, void *ptr, size_t n, mc_vas_t vas)
{
  mc_uvm_external_segment_t segs[MC_UVM_EXTERNAL_SEGMENTS_MAX];
  mc_alloc_t               *slot;
  NvHandle                  h_mem = 0;
  NvU64                     map_base = 0;
  NvU64                     map_size = 0;
  int                       seg_count = 0;
  int                       mapped_count = 0;
  int                       i;

  if (ctx == NULL || ptr == NULL || n == 0 || n > MC_MAX_TRANSFER_SIZE)
    return MC_EINVAL;
  if (vas != MC_VAS_UVM)
  {
    WARN_LOG("mc_host_register: only MC_VAS_UVM is supported today");
    return MC_EINVAL;
  }
  if (ctx->uvm_fd < 0)
    return MC_EINTERNAL;
  if (alloc_table_lookup(ctx, ptr) != NULL)
  {
    WARN_LOG("mc_host_register(%p): pointer already tracked", ptr);
    return MC_EINVAL;
  }

  slot = alloc_table_find_free(ctx);
  if (slot == NULL)
  {
    ERROR_LOG("mc_host_register: table full");
    return MC_EALLOC;
  }

  if (mc_build_host_register_segments(ptr, n, segs, &seg_count,
                                      &map_base, &map_size) != 0)
    return MC_EINVAL;

  h_mem = rm_register_user_memory(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                                  ctx->h_device,
                                  (void *)(uintptr_t)map_base, map_size);
  if (!h_mem)
    return MC_EIOCTL;

  for (i = 0; i < seg_count; i++)
  {
    if (uvm_map_buffer_range_at(ctx->uvm_fd, ctx->dev_fd, ctx->h_client,
                                ctx->gpu_inst_uuid, h_mem,
                                segs[i].base, segs[i].length,
                                segs[i].offset, "h_registered") != 0)
      goto rollback;
    mapped_count++;
  }

  slot->ptr               = ptr;
  slot->h_mem             = h_mem;
  slot->gpu_va            = (NvU64)(uintptr_t)ptr;
  slot->size              = (NvU64)n;
  slot->is_device         = false;
  slot->is_registered     = true;
  slot->vas               = vas;
  slot->map_base          = map_base;
  slot->map_size          = map_size;
  slot->uvm_segment_count = seg_count;
  for (i = 0; i < seg_count; i++)
    slot->uvm_segment_base[i] = segs[i].base;

  DEBUG_LOG("mc_host_register: ptr=%p size=0x%zx map_base=0x%llx map_size=0x%llx segments=%d h=0x%x",
            ptr, n, (unsigned long long)map_base,
            (unsigned long long)map_size, seg_count, h_mem);
  return MC_OK;

rollback:
  for (i = mapped_count; i > 0; i--)
    uvm_free_range(ctx->uvm_fd, segs[i - 1].base, "h_registered_rollback");
  rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_mem,
                 "h_registered_rollback");
  return MC_EIOCTL;
}

mc_status_t mc_host_unregister(mc_ctx_t *ctx, void *ptr)
{
  mc_alloc_t *slot;
  NvHandle    h_mem;

  if (ctx == NULL) return MC_EINVAL;
  if (ptr == NULL) return MC_OK;

  slot = alloc_table_lookup(ctx, ptr);
  if (slot == NULL)
  {
    WARN_LOG("mc_host_unregister(%p): not found in allocation table", ptr);
    return MC_EINVAL;
  }
  if (!slot->is_registered)
  {
    WARN_LOG("mc_host_unregister(%p): pointer was not registered by mc_host_register",
             ptr);
    return MC_EINVAL;
  }

  mc_free_registered_uvm_ranges(ctx, slot);
  h_mem = slot->h_mem;
  memset(slot, 0, sizeof(*slot));

  if (h_mem)
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_mem,
                   "h_registered_mem");

  return MC_OK;
}

/*
 * Free a pointer previously returned by either mc_malloc.  No-op on
 * NULL.  Unknown pointer logs WARN and returns: we'd rather a caller
 * see a noisy log message than have mc_free silently do something
 * surprising with a bogus pointer.
 *
 * Order mirrors mc_fini's user-alloc loop: UVM unmap first (so PTEs
 * are dropped before we free the backing handle), then munmap for
 * host allocs (device allocs have no CPU alias), then RM handle free.
 * The allocation-table slot is zeroed last so it becomes reusable.
 */
void mc_free(mc_ctx_t *ctx, void *p)
{
  mc_alloc_t *slot;

  if (ctx == NULL || p == NULL) return;
  slot = alloc_table_lookup(ctx, p);
  if (slot == NULL)
  {
    WARN_LOG("mc_free(%p): not found in allocation table", p);
    return;
  }
  if (slot->is_registered)
  {
    (void)mc_host_unregister(ctx, p);
    return;
  }
  /* GPU-mapping teardown: dispatch on the slot's VAS.
   *   MC_VAS_UVM     — uvm_unmap_buffer (UVM-managed PTEs).
   *   carrier kinds — release the per-resource NV50 carrier
   *                   tracked in vas->carriers[].
   * Calling uvm_unmap_buffer against a non-UVM gpu_va wedges UVM
   * bookkeeping, hence the gate. */
  if (slot->vas == MC_VAS_UVM)
  {
    uvm_unmap_buffer(ctx->uvm_fd, ctx->gpu_inst_uuid, slot->gpu_va, slot->size,
                     slot->is_device ? "d_user" : "h_user");
  }
  else /* sysmem-carrier or fb-carrier */
  {
    /* Per-resource carrier: release pulls the entry out of
     * vas->carriers[] and frees the carrier handle itself. */
    mc_va_space_release_carrier(ctx, &ctx->vas[slot->vas], slot->h_mem);
  }

  if (!slot->is_device)
    munmap(slot->ptr, slot->size);
  rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, slot->h_mem,
                 slot->is_device ? "d_user_mem" : "h_user_mem");
  memset(slot, 0, sizeof(*slot));
}

/* Look up the GPU VA backing a pointer returned by mc_malloc_*.  See
 * mc.h for the contract (Paper-F1 round-trip for UVM-host; carrier-
 * host requires this to recover the GPU VA; device just returns its
 * own GPU VA back). */
uint64_t mc_gpu_va(mc_ctx_t *ctx, const void *user_ptr)
{
  const mc_alloc_t *slot;

  if (ctx == NULL || user_ptr == NULL) return 0;
  slot = alloc_table_lookup(ctx, user_ptr);
  if (slot == NULL) return 0;
  return slot->gpu_va;
}

/* ── mc_memcpy ─────────────────────────────
 *
 * Both src and dst arguments are user pointers returned by mc_malloc_*.
 * mc_memcpy looks them up in the allocation table, validates
 * they share a VAS (else MC_EINVAL), then dispatches to the agent
 * named by `agent`.
 *
 *   MC_XFER_HOST: host writes the matching channel's pushbuffer (UVM-
 *                 VAS → UVM channel, CARRIER-VAS → DMA channel) and
 *                 rings its doorbell.  Translates CPU pointers to GPU
 *                 VAs via the alloc table.
 *
 *   MC_XFER_SM:   launches the sm_owner kernel on the compute channel;
 *                 a single SM thread authors the entire DMA-channel
 *                 submission (PB + GPFIFO entry + USERD GPPut + BAR1
 *                 doorbell) from device code.  Both buffers must live
 *                 in a carrier VA space — MC_VAS_SYSMEM_CARRIER or
 *                 MC_VAS_FB_CARRIER (the SM kernel only addresses a
 *                 carrier).
 *
 * Why translation matters: for UVM-VAS host allocs the user's CPU
 * pointer happens to equal the GPU VA (Paper F1) — so passing it
 * directly to mc_write_transfer_methods would have worked.  For
 * CARRIER-VAS host allocs the CPU and GPU VAs are different.  The
 * alloc-table lookup gives us the right GPU VA in both cases.
 *
 * The sema_payload counter increments monotonically per submission per
 * channel; if it wraps to 0 we bump it to 1 so the payload == 0
 * "cleared" sentinel stays unambiguous.
 */
mc_status_t mc_memcpy(mc_ctx_t *ctx, const void *dst_ptr, const void *src_ptr,
                      size_t n, mc_xfer_t agent)
{
  const mc_alloc_t *src_slot;
  const mc_alloc_t *dst_slot;
  mc_channel_t     *ch;
  uint32_t         *pb_end;
  uint32_t          copy_bytes;
  mc_status_t       rc;

  if (ctx == NULL) return MC_EINVAL;
  if (n == 0 || n > MC_MAX_TRANSFER_SIZE) return MC_EINVAL;
  if (agent != MC_XFER_HOST && agent != MC_XFER_SM) return MC_EINVAL;

  src_slot = alloc_table_lookup(ctx, src_ptr);
  dst_slot = alloc_table_lookup(ctx, dst_ptr);
  if (src_slot == NULL || dst_slot == NULL)
  {
    WARN_LOG("mc_memcpy: src=%p dst=%p — at least one pointer not in "
             "alloc table; both must come from mc_malloc_* on this ctx",
             src_ptr, dst_ptr);
    return MC_EINVAL;
  }
  if (src_slot->vas != dst_slot->vas)
  {
    WARN_LOG("mc_memcpy: src VAS=%d dst VAS=%d mismatch; both buffers "
             "must live in the same VAS",
             (int)src_slot->vas, (int)dst_slot->vas);
    return MC_EINVAL;
  }
  if (src_slot->size < n || dst_slot->size < n)
  {
    WARN_LOG("mc_memcpy: transfer size %zu exceeds buffer (src=%llu, dst=%llu)",
             n, (unsigned long long)src_slot->size,
             (unsigned long long)dst_slot->size);
    return MC_EINVAL;
  }

  if (agent == MC_XFER_SM)
  {
    /* The SM-owner kernel only addresses a carrier VAS — it walks
     * that VAS's PDB and submits onto the VAS's victim DMA channel.
     * UVM-VAS pointers would dereference invalid GPU VAs from the
     * SM thread's perspective. */
    if (src_slot->vas == MC_VAS_UVM)
    {
      WARN_LOG("mc_memcpy(MC_XFER_SM): UVM-VAS pointers not supported "
               "(SM kernel only addresses carrier VAS)");
      return MC_EINVAL;
    }
    return mc_sm_owner_submit(ctx, src_slot->vas, src_slot->gpu_va,
                              dst_slot->gpu_va, (uint32_t)n);
  }

  /* MC_XFER_HOST: pick the channel by (vas, role).  UVM-VAS routes
   * through the UVM CE channel; carrier-VAS routes through the
   * carrier's HOST_DMA channel.  Both channels' sema state is
   * independent; the library is single-threaded so alternating
   * mc_memcpy calls don't race. */
  ch = (src_slot->vas == MC_VAS_UVM)
       ? mc_vas_find_channel(&ctx->vas[MC_VAS_UVM], MC_ROLE_UVM_CE)
       : mc_vas_find_channel(&ctx->vas[src_slot->vas], MC_ROLE_HOST_DMA);
  if (ch == NULL || !ch->h_channel) return MC_EINTERNAL;

  ch->sema_payload++;
  if (ch->sema_payload == 0) ch->sema_payload = 1;

  pb_end = mc_write_transfer_methods(ch->pb_cpu,
                                     src_slot->gpu_va, dst_slot->gpu_va,
                                     (uint32_t)n,
                                     ch->sema_gpu_va, ch->sema_payload);
  copy_bytes = (uint32_t)((pb_end - ch->pb_cpu) * sizeof(uint32_t));

  rc = mc_channel_submit(ch, ctx->vf_doorbell, copy_bytes);
  if (rc != MC_OK)
  {
    WARN_LOG("mc_memcpy: submit failed rc=%d agent=%d vas=%d bytes=%zu "
             "src_ptr=%p src_gpu_va=0x%llx dst_ptr=%p dst_gpu_va=0x%llx "
             "h_channel=0x%x sema_payload=0x%x",
             rc, (int)agent, (int)src_slot->vas, n,
             src_ptr, (unsigned long long)src_slot->gpu_va,
             dst_ptr, (unsigned long long)dst_slot->gpu_va,
             ch->h_channel, ch->sema_payload);
  }
  return rc;
}

/*
 * mc_dbell_demo_ring — submit a 4-byte CE LAUNCH_DMA on the DMA
 * channel that copies `token` from a sysmem cell into the BAR1
 * doorbell page at GPU VA (carrier VAS dbell_gpu_va + 0x90).  The CE
 * engine — not the host — issues the MMIO write.
 *
 * Method stream (mirrors mc_write_transfer_methods):
 *   SET_OBJECT      → HOPPER_DMA_COPY_A on subch 4
 *   OFFSET_IN_*     ← DMA channel's token_cell_gpu_va     (sysmem source)
 *   OFFSET_OUT_*    ← carrier vas->dbell_gpu_va + 0x90    (BAR1 destination)
 *   LINE_LENGTH_IN  ← 4 bytes
 *   SET_SEMAPHORE_* + LAUNCH_DMA(NON_PIPELINED|FLUSH_ENABLE|REL_ONE_WORD)
 *
 * The destination GPU VA resolves through the GPU MMU PTE that
 * mc_va_space_install_doorbell_pte installed via NV04_MAP_MEMORY_DMA →
 * usrmodeGetMemInterMapParams_IMPL substituting pBar1VF.  The CE's
 * write egresses through the GPU's host interface as a 4-byte PCIe
 * MWr that lands on the H100's BAR1 doorbell register.
 */
mc_status_t mc_dbell_demo_ring(mc_ctx_t *ctx, uint32_t token)
{
  mc_channel_t         *ch;
  struct mc_dma_extras *ex;
  NvU64                 dbell_gpu_va;
  uint32_t             *pb_end;
  uint32_t              copy_bytes;

  if (ctx == NULL) return MC_EINVAL;
  ch           = mc_vas_find_channel(&ctx->vas[MC_VAS_SYSMEM_CARRIER],
                                     MC_ROLE_HOST_DMA);
  dbell_gpu_va = ctx->vas[MC_VAS_SYSMEM_CARRIER].dbell_gpu_va;
  if (ch == NULL || !ch->h_channel || !dbell_gpu_va) return MC_EINTERNAL;
  ex           = &ch->x.dma;

  /* Pre-fill the source cell with the token value the GPU will copy. */
  *ex->token_cell = token;

  ch->sema_payload++;
  if (ch->sema_payload == 0) ch->sema_payload = 1;

  pb_end = mc_write_transfer_methods(
      ch->pb_cpu,
      ex->token_cell_gpu_va,                          /* src: sysmem token cell */
      dbell_gpu_va + MC_VF_DOORBELL_OFFSET,           /* dst: BAR1 +0x90 */
      4,                                              /* 4 bytes */
      ch->sema_gpu_va,
      ch->sema_payload);
  copy_bytes = (uint32_t)((pb_end - ch->pb_cpu) * sizeof(uint32_t));

  /* Two writes per submission: host's doorbell ring (kicks dma_ch's
   * PBDMA inside mc_channel_submit -> ring_doorbell) and the GPU-side
   * MMIO write the CE op then performs to BAR1+0x90.  The latter is
   * the point of this demo. */
  return mc_channel_submit(ch, ctx->vf_doorbell, copy_bytes);
}

/*
 * mc_memcpy_gpu_doorbell_ce — D2H whose UVM-channel doorbell is
 * rung by the GPU's CE engine on the DMA channel, not the host.
 *
 * Chain (top to bottom is wall-clock time):
 *
 *   host:     arm UVM-channel CE pushbuffer (HBM→DRAM + sema rel)
 *             write UVM-channel GPFIFO entry
 *             advance UVM-channel USERD GPPut       ← but NO doorbell ring
 *             arm DMA-channel CE pushbuffer (token→UVM-ch doorbell GPU VA)
 *             write DMA-channel GPFIFO entry
 *             ring DMA-channel doorbell (host MMIO)
 *             poll DMA-channel sema  ╮
 *                                    │  DMA-channel CE LAUNCH_DMA writes
 *                                    │  the UVM channel's work_submit_token
 *                                    │  to the BAR1 doorbell — this is the
 *                                    │  GPU-issued ring.
 *                                    ╯
 *             DMA-channel sema fires → CE op done, doorbell write committed
 *             poll UVM-channel sema  ╮
 *                                    │  UVM channel's PBDMA, woken by the
 *                                    │  GPU-issued doorbell, runs the
 *                                    │  HBM→DRAM CE LAUNCH_DMA.
 *                                    ╯
 *             UVM-channel sema fires → bytes are in dst_host
 *
 * Verification: the caller compares dst_host with expected bytes.
 * The host did NOT ring the UVM channel's doorbell; the only path
 * that could have woken its PBDMA is the DMA-channel CE's MMIO
 * write to the BAR1 doorbell GPU VA.  No libcuda, no watchpoint
 * shadow involvement, no kernel hooks beyond what was already
 * needed for UVM + DMA mapping.
 */
mc_status_t mc_memcpy_gpu_doorbell_ce(mc_ctx_t *ctx, void *dst,
                                      const void *src, size_t n)
{
  const mc_alloc_t     *src_slot;
  const mc_alloc_t     *dst_slot;
  mc_channel_t         *prim;
  mc_channel_t         *dma;
  struct mc_dma_extras *dma_ex;
  NvU64                 dbell_gpu_va;
  uint32_t             *prim_pb_end;
  uint32_t              prim_copy_bytes;
  uint32_t             *sec_pb_end;
  uint32_t              sec_copy_bytes;
  struct timespec       t0;
  mc_status_t           rc;

  if (ctx == NULL || dst == NULL || src == NULL) return MC_EINVAL;
  if (n == 0 || n > MC_MAX_TRANSFER_SIZE) return MC_EINVAL;

  /* Direction-agnostic validation, mirroring mc_memcpy: both pointers
   * must come from mc_malloc_* on this ctx, share a VAS, and be large
   * enough for n.  The chain primary is the UVM channel, so both
   * pointers must live in MC_VAS_UVM. */
  src_slot = alloc_table_lookup(ctx, src);
  dst_slot = alloc_table_lookup(ctx, dst);
  if (src_slot == NULL || dst_slot == NULL)
  {
    WARN_LOG("mc_memcpy_gpu_doorbell_ce: src=%p dst=%p — at least one "
             "pointer not in alloc table; both must come from mc_malloc_* "
             "on this ctx", src, dst);
    return MC_EINVAL;
  }
  if (src_slot->vas != MC_VAS_UVM || dst_slot->vas != MC_VAS_UVM)
  {
    WARN_LOG("mc_memcpy_gpu_doorbell_ce: src VAS=%d dst VAS=%d; both "
             "buffers must live in MC_VAS_UVM (the chain primary is the "
             "UVM CE channel)", (int)src_slot->vas, (int)dst_slot->vas);
    return MC_EINVAL;
  }
  if (src_slot->size < n || dst_slot->size < n)
  {
    WARN_LOG("mc_memcpy_gpu_doorbell_ce: transfer size %zu exceeds buffer "
             "(src=%llu, dst=%llu)", n,
             (unsigned long long)src_slot->size,
             (unsigned long long)dst_slot->size);
    return MC_EINVAL;
  }

  prim         = mc_vas_find_channel(&ctx->vas[MC_VAS_UVM], MC_ROLE_UVM_CE);
  dma          = mc_vas_find_channel(&ctx->vas[MC_VAS_SYSMEM_CARRIER],
                                     MC_ROLE_HOST_DMA);
  dbell_gpu_va = ctx->vas[MC_VAS_SYSMEM_CARRIER].dbell_gpu_va;
  if (prim == NULL || dma == NULL || !dma->h_channel || !dbell_gpu_va)
    return MC_EINTERNAL;
  dma_ex       = &dma->x.dma;

  /* Arm primary's submission: build pushbuffer + write GPFIFO entry +
   * advance USERD GPPut, but DO NOT ring vf_doorbell.  src/dst are
   * the user-supplied GPU VAs in either order — the CE method stream
   * is direction-agnostic; H2D vs D2H comes from which slot was
   * mc_malloc_host vs mc_malloc_device.  mc_channel_arm captures t0
   * before publish so the eventual poll starts its budget here. */
  prim->sema_payload++;
  if (prim->sema_payload == 0) prim->sema_payload = 1;

  prim_pb_end = mc_write_transfer_methods(
      prim->pb_cpu,
      src_slot->gpu_va, dst_slot->gpu_va,
      (uint32_t)n, prim->sema_gpu_va, prim->sema_payload);
  prim_copy_bytes = (uint32_t)((prim_pb_end - prim->pb_cpu) * sizeof(uint32_t));

  rc = mc_channel_arm(prim, prim_copy_bytes, &t0);
  if (rc != MC_OK) return rc;

  /* Arm + ring the DMA channel's CE op: copy 4 bytes from
   * token_cell (pre-filled with the UVM channel's work_submit_token)
   * to dbell_gpu_va + 0x90 — the same physical BAR1 doorbell page
   * the host would write to ring the UVM channel.  When this CE op
   * executes, its MMIO write to BAR1+0x90 wakes the UVM channel's
   * PBDMA. */
  *dma_ex->token_cell = prim->work_submit_token;

  dma->sema_payload++;
  if (dma->sema_payload == 0) dma->sema_payload = 1;

  sec_pb_end = mc_write_transfer_methods(
      dma->pb_cpu,
      dma_ex->token_cell_gpu_va,                          /* src */
      dbell_gpu_va + MC_VF_DOORBELL_OFFSET,               /* dst: UVM ch doorbell */
      4,
      dma->sema_gpu_va, dma->sema_payload);
  sec_copy_bytes = (uint32_t)((sec_pb_end - dma->pb_cpu) * sizeof(uint32_t));

  /* mc_channel_submit on dma rings the DMA channel's doorbell, polls
   * dma's sema until the CE op completes (LAUNCH_DMA's FLUSH_ENABLE
   * guarantees the BAR1 write has committed by then). */
  rc = mc_channel_submit(dma, ctx->vf_doorbell, sec_copy_bytes);
  if (rc != MC_OK) return rc;

  /* Then poll primary's sema (HBM->DRAM bytes flushed), charging
   * elapsed time against the same budget that started before the
   * first submission. */
  return mc_channel_poll_sema(prim->sema_ptr, prim->sema_payload, t0);
}
