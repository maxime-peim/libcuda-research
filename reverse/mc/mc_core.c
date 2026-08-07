/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_core.c — libmc lifecycle layer.  Owns the per-kind channel
 * init/fini routines, mc_init/mc_fini, the allocation table,
 * mc_malloc_device, mc_malloc_host, mc_free, mc_memcpy_h2d/d2h,
 * and the two CE-doorbell demos (mc_dbell_demo_ring,
 * mc_memcpy_d2h_gpu_doorbell_ce).
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

#include "mc_internal.h"




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

static mc_alloc_t *alloc_table_lookup(mc_ctx_t *ctx, const void *ptr)
{
  int i;
  if (ptr == NULL)
    return NULL;
  for (i = 0; i < MC_ALLOC_TABLE_SLOTS; i++)
    if (ctx->allocs[i].ptr == ptr)
      return &ctx->allocs[i];
  return NULL;
}

/* ── Channel alloc/free core helpers ─────────────────────────────────
 *
 * Both mc_dma_channel_init and mc_compute_channel_init share the same
 * skeleton: allocate three sysmem buffers (gpfifo+USERD, pushbuffer,
 * semaphore), DMA-map each into the carrier VAS, allocate TSG + channel
 * + engine-class object, fetch the work-submit-token, and schedule.
 * mc_channel_alloc_via_carrier encapsulates the shared sequence; the
 * carrier VAS hands out GPU-VA windows via mc_va_space_carve, so the
 * old hard-coded sub-offset table is gone.
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
static int mc_channel_alloc_via_carrier(mc_ctx_t *ctx, mc_channel_t *ch)
{
  NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS token_params = {};
  mc_va_space_t *vas = &ctx->vas[ch->vas_id];
  void *gpfifo_cpu_ptr = NULL;
  void *pb_cpu_ptr     = NULL;
  void *sema_cpu_ptr   = NULL;

  if (vas->kind != MC_VAS_KIND_CARRIER) return -1;

  /* Three sysmem allocs.  Kernel-chosen CPU VAs (no Paper-F1 anchor:
   * no UVM, no CPU-VA == GPU-VA invariant). */
  ch->h_gpfifo_mem =
      rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client, ctx->h_device,
                         MC_GPFIFO_USERD_SIZE, NULL, &gpfifo_cpu_ptr);
  if (!ch->h_gpfifo_mem) return -1;
  ch->gpfifo_cpu  = gpfifo_cpu_ptr;
  ch->gpfifo_ring = (volatile uint32_t *)gpfifo_cpu_ptr;
  ch->userd_cpu   = (char *)gpfifo_cpu_ptr + MC_USERD_OFFSET;
  ch->userd       = (volatile HopperAControlGPFifo *)ch->userd_cpu;

  ch->h_pb_mem =
      rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client, ctx->h_device,
                         MC_GPFIFO_USERD_SIZE, NULL, &pb_cpu_ptr);
  if (!ch->h_pb_mem) return -1;
  ch->pb_cpu = (uint32_t *)pb_cpu_ptr;

  ch->h_sema_mem =
      rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client, ctx->h_device,
                         MC_GPFIFO_USERD_SIZE, NULL, &sema_cpu_ptr);
  if (!ch->h_sema_mem) return -1;
  ch->sema_ptr = (volatile uint32_t *)sema_cpu_ptr;
  *ch->sema_ptr = 0;

  /* Carve a window for each core buffer and DMA-map it. */
  ch->gpfifo_gpu_va = mc_va_space_carve(vas, MC_GPFIFO_USERD_SIZE, 0);
  if (!ch->gpfifo_gpu_va) return -1;
  if (rm_map_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                        vas->h_virt, ch->h_gpfifo_mem, 0, MC_GPFIFO_USERD_SIZE,
                        ch->gpfifo_gpu_va) != ch->gpfifo_gpu_va)
    return -1;

  ch->pb_gpu_va = mc_va_space_carve(vas, MC_GPFIFO_USERD_SIZE, 0);
  if (!ch->pb_gpu_va) return -1;
  if (rm_map_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                        vas->h_virt, ch->h_pb_mem, 0, MC_GPFIFO_USERD_SIZE,
                        ch->pb_gpu_va) != ch->pb_gpu_va)
    return -1;

  ch->sema_gpu_va = mc_va_space_carve(vas, MC_GPFIFO_USERD_SIZE, 0);
  if (!ch->sema_gpu_va) return -1;
  if (rm_map_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                        vas->h_virt, ch->h_sema_mem, 0, MC_GPFIFO_USERD_SIZE,
                        ch->sema_gpu_va) != ch->sema_gpu_va)
    return -1;

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
  NvHandle h_virt = ctx->vas[ch->vas_id].h_virt;

  /* Drain (best-effort). */
  if (ch->userd != NULL)
    drain_channel(ch->userd, MC_TIMEOUT_MS);

  /* Disable the channel before unmapping anything it might still
   * dereference. */
  if (ch->h_channel && ctx->ctl_fd >= 0)
    rm_channel_disable(ctx->ctl_fd, ctx->h_client, ch->h_channel);

  /* DMA-unmap the three core buffers from the carrier. */
  if (ctx->ctl_fd >= 0 && h_virt)
  {
    if (ch->sema_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          h_virt, ch->h_sema_mem, ch->sema_gpu_va);
    if (ch->pb_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          h_virt, ch->h_pb_mem, ch->pb_gpu_va);
    if (ch->gpfifo_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          h_virt, ch->h_gpfifo_mem, ch->gpfifo_gpu_va);
  }

  /* CPU-side mappings. */
  if (ch->gpfifo_cpu) munmap(ch->gpfifo_cpu, MC_GPFIFO_USERD_SIZE);
  if (ch->pb_cpu)     munmap(ch->pb_cpu,     MC_GPFIFO_USERD_SIZE);
  if (ch->sema_ptr)   munmap((void *)ch->sema_ptr, MC_GPFIFO_USERD_SIZE);

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

/* ── DMA-channel bring-up (mc_dma_channel_init) ────────────────────────
 *
 * Allocates a CE channel on the carrier VAS (set up earlier in mc_init,
 * along with its BAR1 doorbell PTE).  The channel's
 * one piece of kind-specific state is the token cell — a 4-byte sysmem
 * scratch buffer DMA-mapped into the carrier VAS, used as the CE source
 * for the doorbell-write demos.
 */
static int mc_dma_channel_init(mc_ctx_t *ctx, NvU32 lce_engine_type)
{
  mc_channel_t          *ch  = &ctx->ch[MC_CH_DMA];
  struct mc_dma_extras  *ex  = &ch->x.dma;
  void                  *cpu = NULL;

  ch->type        = MC_CH_TYPE_DMA;
  ch->vas_id      = MC_VAS_PRIMARY_CARRIER;
  ch->subchannel  = NVA06F_SUBCHANNEL_COPY_ENGINE;
  ch->engine_type = lce_engine_type;

  /* Core: 3 sysmem buffers (gpfifo/pb/sema), TSG/channel/engine/token,
   * schedule.  The carrier VAS + its BAR1 doorbell PTE already exist
   * (set up before any channel by mc_init). */
  if (mc_channel_alloc_via_carrier(ctx, ch) != 0) return -1;

  /* Token cell: 4-byte sysmem scratch in the carrier VAS, used as the
   * CE source for doorbell-write demos. */
  ex->token_cell_gpu_va = mc_va_space_alloc_scratch(
      ctx, &ctx->vas[MC_VAS_PRIMARY_CARRIER], 4, 4,
      &ex->h_token_mem, &cpu);
  if (!ex->token_cell_gpu_va) return -1;
  ex->token_cell = (volatile uint32_t *)cpu;
  *ex->token_cell = 0;

  DEBUG_LOG("mc_dma_channel ok: ch=0x%x token_gpu_va=0x%llx work_token=0x%x",
            ch->h_channel,
            (unsigned long long)ex->token_cell_gpu_va,
            ch->work_submit_token);
  return 0;
}

static void mc_dma_channel_fini(mc_ctx_t *ctx)
{
  mc_channel_t         *ch = &ctx->ch[MC_CH_DMA];
  struct mc_dma_extras *ex = &ch->x.dma;
  NvHandle h_virt = ctx->vas[MC_VAS_PRIMARY_CARRIER].h_virt;

  /* Core first: drain + disable so no in-flight CE op can be
   * referencing the token cell PTE we're about to unmap.  The core
   * also DMA-unmaps the gpfifo/pb/sema buffers and frees their RM
   * handles + TSG/channel/engine. */
  mc_channel_free_core(ctx, ch);

  /* Token cell teardown: DMA-unmap, then free the sysmem hMemory.
   * The carrier VAS + BAR1 doorbell PTE belong to globals, not us. */
  if (ctx->ctl_fd >= 0 && h_virt && ex->token_cell_gpu_va)
    rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                        h_virt, ex->h_token_mem, ex->token_cell_gpu_va);
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
static int mc_compute_module_init(mc_ctx_t *ctx,
                                  struct mc_compute_module *mod,
                                  const uint8_t *sass_bytes,
                                  size_t sass_len)
{
  mc_va_space_t *vas = &ctx->vas[MC_VAS_PRIMARY_CARRIER];
  void          *qmd_cpu_ptr     = NULL;
  void          *cb0_cpu_ptr     = NULL;
  void          *sass_cpu_ptr    = NULL;
  void          *scratch_cpu_ptr = NULL;
  NvU64          scratch_base_gpu_va;

  mod->qmd_gpu_va = mc_va_space_alloc_scratch(
      ctx, vas, MC_GPFIFO_USERD_SIZE, 0, &mod->h_qmd_mem, &qmd_cpu_ptr);
  if (!mod->qmd_gpu_va) return -1;
  mod->qmd_cpu = (uint8_t *)qmd_cpu_ptr;

  mod->cb0_gpu_va = mc_va_space_alloc_scratch(
      ctx, vas, MC_GPFIFO_USERD_SIZE, 0, &mod->h_cb0_mem, &cb0_cpu_ptr);
  if (!mod->cb0_gpu_va) return -1;
  mod->cb0_cpu = (uint8_t *)cb0_cpu_ptr;

  mod->sass_gpu_va = mc_va_space_alloc_scratch(
      ctx, vas, MC_GPFIFO_USERD_SIZE, 0, &mod->h_sass_mem, &sass_cpu_ptr);
  if (!mod->sass_gpu_va) return -1;
  mod->sass_cpu = (uint8_t *)sass_cpu_ptr;

  /* Dedicated scratch dword exposed via mc_compute_get_scratch.
   * Allocate 64 bytes so the caller can write a few sentinels on
   * either side of the target dword and stay inside one mapped
   * region; the GPU kernel only ever writes the target dword itself,
   * so the surrounding bytes are CPU-readable inspection space, not
   * writable by the GPU. */
  scratch_base_gpu_va = mc_va_space_alloc_scratch(
      ctx, vas, 64, 64, &mod->h_scratch_mem, &scratch_cpu_ptr);
  if (!scratch_base_gpu_va) return -1;
  mod->scratch_cpu    = (volatile uint32_t *)((uint8_t *)scratch_cpu_ptr + 32);
  mod->scratch_gpu_va = scratch_base_gpu_va + 32;

  /* Stage SASS bytes once; per-call code only patches QMD + CB0. */
  memcpy(mod->sass_cpu, sass_bytes, sass_len);
  _mm_sfence();
  return 0;
}

static void mc_compute_module_fini(mc_ctx_t *ctx,
                                   struct mc_compute_module *mod)
{
  NvHandle h_virt = ctx->vas[MC_VAS_PRIMARY_CARRIER].h_virt;

  if (ctx->ctl_fd >= 0 && h_virt)
  {
    /* LIFO: scratch was allocated after sass, so unmap it first. */
    if (mod->scratch_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          h_virt, mod->h_scratch_mem,
                          mod->scratch_gpu_va - 32);
    if (mod->sass_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          h_virt, mod->h_sass_mem, mod->sass_gpu_va);
    if (mod->cb0_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          h_virt, mod->h_cb0_mem, mod->cb0_gpu_va);
    if (mod->qmd_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          h_virt, mod->h_qmd_mem, mod->qmd_gpu_va);
  }
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

static int mc_compute_channel_init(mc_ctx_t *ctx)
{
  mc_channel_t             *ch = &ctx->ch[MC_CH_COMPUTE];
  struct mc_compute_extras *ex = &ch->x.compute;

  ch->type        = MC_CH_TYPE_COMPUTE;
  ch->vas_id      = MC_VAS_PRIMARY_CARRIER;
  ch->subchannel  = NVA06F_SUBCHANNEL_COMPUTE;
  ch->engine_type = NV2080_ENGINE_TYPE_GR0;

  /* Core: 3 sysmem buffers (gpfifo/pb/sema), TSG/channel/engine/token,
   * schedule.  Same carrier VAS as the DMA channel. */
  if (mc_channel_alloc_via_carrier(ctx, ch) != 0) return -1;

  /* Load the one embedded compute kernel. */
  if (mc_compute_module_init(ctx, &ex->module,
                             mc_doorbell_kernel_sass,
                             mc_doorbell_kernel_sass_len) != 0)
    return -1;

  DEBUG_LOG("mc_compute_channel ok: tsg=0x%x ch=0x%x compute=0x%x "
            "gpfifo_gpu_va=0x%llx token=0x%x",
            ch->h_tsg, ch->h_channel, ch->h_engine,
            (unsigned long long)ch->gpfifo_gpu_va,
            ch->work_submit_token);
  return 0;
}

/* Symmetric reverse-order teardown for the compute channel.  Best-effort.
 * Does NOT free the carrier VAS (that belongs to globals). */
static void mc_compute_channel_fini(mc_ctx_t *ctx)
{
  mc_channel_t             *ch = &ctx->ch[MC_CH_COMPUTE];
  struct mc_compute_extras *ex = &ch->x.compute;

  /* Core first: drain + disable so no in-flight kernel can be
   * dereferencing the QMD/CB0/SASS PTEs we're about to unmap. */
  mc_channel_free_core(ctx, ch);

  /* Tear down the loaded compute module. */
  mc_compute_module_fini(ctx, &ex->module);

  memset(ch, 0, sizeof(*ch));
}

/* ── UVM-channel bring-up (mc_uvm_channel_init) ────────────────────────
 *
 * The UVM channel is the foundation: it owns the UVM-managed VAS
 * that mc_malloc_host / mc_malloc_device anchor user allocations
 * into, and it's the channel mc_memcpy_h2d / mc_memcpy_d2h submits
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

  ctx->ch[MC_CH_PRIMARY].type        = MC_CH_TYPE_UVM;
  ctx->ch[MC_CH_PRIMARY].vas_id      = MC_VAS_PRIMARY_UVM;
  ctx->ch[MC_CH_PRIMARY].subchannel  = NVA06F_SUBCHANNEL_COPY_ENGINE;
  ctx->ch[MC_CH_PRIMARY].engine_type = lce_engine_type;

  ch = &ctx->ch[MC_CH_PRIMARY];
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
  ch->h_pb_mem = rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
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
                ctx->vas[MC_VAS_PRIMARY_UVM].h_vaspace,
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
                           ctx->vas[MC_VAS_PRIMARY_UVM].h_vaspace);
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
  mc_channel_t         *ch = &ctx->ch[MC_CH_PRIMARY];
  struct mc_uvm_extras *ex = &ch->x.uvm;

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

  /* Carrier VAS shared by the DMA + compute channels, plus its BAR1
   * doorbell PTE (resolved by usrmodeGetMemInterMapParams_IMPL to
   * pBar1VF — usermode_api.c:112).  Brought up before the channels
   * themselves so neither has to embed VAS handles in its extras. */
  if (mc_va_space_init_carrier(ctx) != 0) goto fail;
  if (mc_va_space_install_doorbell_pte(ctx, &ctx->vas[MC_VAS_PRIMARY_CARRIER])
      != 0)
    goto fail;

  /* UVM channel (CE; mc_memcpy_h2d/d2h). */
  if (mc_uvm_channel_init(ctx, lce_engine_type) != 0) goto fail;

  /* DMA channel for the doorbell-write demos.  Bound to the carrier
   * VAS (where the BAR1 doorbell PTE lives). */
  if (mc_dma_channel_init(ctx, lce_engine_type) != 0) goto fail;

  /* Compute channel for the SM-thread doorbell-write demo.  Also
   * bound to the carrier VAS, so SM threads walk the same PDB and can
   * reach the same BAR1 doorbell PTE.  Engine type GR0, class
   * HOPPER_COMPUTE_A. */
  if (mc_compute_channel_init(ctx) != 0) goto fail;

  DEBUG_LOG("mc_init ok: client=0x%x device=0x%x uvm_ch=0x%x dma_ch=0x%x "
            "compute_ch=0x%x",
            ctx->h_client, ctx->h_device, ctx->ch[MC_CH_PRIMARY].h_channel,
            ctx->ch[MC_CH_DMA].h_channel, ctx->ch[MC_CH_COMPUTE].h_channel);
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

  /* Channels: reverse-dependency order.  Each *_fini drains + disables
   * its channel before unmapping anything, so no GPU work survives. */
  mc_compute_channel_fini(ctx);
  mc_dma_channel_fini(ctx);
  mc_uvm_channel_fini(ctx);

  /* VA spaces: free the carrier (with its BAR1 doorbell PTE) and the
   * UVM-managed VAS.  All channel buffers were unmapped above. */
  mc_va_space_fini(ctx, MC_VAS_PRIMARY_CARRIER);
  mc_va_space_fini(ctx, MC_VAS_PRIMARY_UVM);

  /* Now safe to reap any leftover user allocations: every channel is
   * disabled, so the GPU can't be referencing these PTEs.  This used
   * to run BEFORE channel disable — that violated the F1 invariant
   * (uvm_unmap on an active channel intermittently wedges the GPU). */
  for (i = 0; i < MC_ALLOC_TABLE_SLOTS; i++)
  {
    if (ctx->allocs[i].ptr == NULL) continue;
    if (ctx->uvm_fd >= 0 && ctx->allocs[i].gpu_va)
      uvm_unmap_buffer(ctx->uvm_fd, ctx->gpu_inst_uuid, ctx->allocs[i].gpu_va,
                       ctx->allocs[i].size, "user");
    if (!ctx->allocs[i].is_device)
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
 * Allocate device-only HBM.  No CPU alias; the returned void* is a GPU
 * VA (cast from uintptr_t) that's meaningful to the CE but not to the
 * CPU.  On allocation failure, any partially-constructed state
 * (vidmem handle without a UVM map) is rolled back before returning
 * NULL so subsequent mc_malloc calls aren't starved of slots/handles.
 */
void *mc_malloc_device(mc_ctx_t *ctx, size_t n)
{
  mc_alloc_t *slot;
  NvHandle    h_mem;
  NvU64       gpu_va;

  if (ctx == NULL || n == 0 || n > MC_MAX_TRANSFER_SIZE) return NULL;
  slot = alloc_table_find_free(ctx);
  if (slot == NULL) { ERROR_LOG("mc_malloc_device: table full"); return NULL; }

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

  slot->ptr       = (void *)(uintptr_t)gpu_va;
  slot->h_mem     = h_mem;
  slot->gpu_va    = gpu_va;
  slot->size      = n;
  slot->is_device = true;
  return slot->ptr;
}

/*
 * Allocate host DRAM with Paper-F1 anchoring.  The caller gets a
 * pointer that's both the CPU VA and the GPU VA of the allocation.
 *
 * Steps:
 *   1. Reserve a 2-MiB-aligned chunk inside the VA pool.
 *   2. rm_alloc_sysmem_at MAP_FIXED's its mmap into that chunk.
 *   3. uvm_map_buffer_at anchors UVM's external range at the same VA
 *      and installs GPU MMU PTEs pointing at the sysmem pages.
 *
 * Any rollback between steps releases the partial state so the
 * allocation table isn't polluted.
 */
void *mc_malloc_host(mc_ctx_t *ctx, size_t n)
{
  mc_alloc_t *slot;
  void       *cpu_va;
  NvHandle    h_mem;
  NvU64       gpu_va;

  if (ctx == NULL || n == 0 || n > MC_MAX_TRANSFER_SIZE) return NULL;
  slot = alloc_table_find_free(ctx);
  if (slot == NULL) { ERROR_LOG("mc_malloc_host: table full"); return NULL; }

  cpu_va = va_pool_reserve((NvU64)n, "h_user");
  if (cpu_va == NULL) return NULL;
  h_mem = rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
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

  slot->ptr       = cpu_va;
  slot->h_mem     = h_mem;
  slot->gpu_va    = gpu_va;
  slot->size      = n;
  slot->is_device = false;
  return cpu_va;
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
  uvm_unmap_buffer(ctx->uvm_fd, ctx->gpu_inst_uuid, slot->gpu_va, slot->size,
                   slot->is_device ? "d_user" : "h_user");
  if (!slot->is_device)
    munmap(slot->ptr, slot->size);
  rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, slot->h_mem,
                 slot->is_device ? "d_user_mem" : "h_user_mem");
  memset(slot, 0, sizeof(*slot));
}

/* ── mc_submit_transfer + mc_memcpy_{h2d,d2h} ─────────────────────────────
 *
 * All transfer calls funnel through mc_submit_transfer, which is direction-
 * agnostic — src_va and dst_va are arbitrary GPU VAs.  mc_memcpy_h2d and
 * mc_memcpy_d2h are thin wrappers that cast their pointer arguments and
 * pick the right (src, dst) order.
 *
 * The sema_payload counter increments monotonically per submission; if
 * it wraps to 0 we bump it to 1 so the payload == 0 "cleared" sentinel
 * stays unambiguous.  Practically, wrapping won't happen in any realistic
 * session (2^32 submissions at 1µs each is ~71 minutes of non-stop CE
 * traffic), but the check is cheap insurance.
 */

/*
 * Build one CE transfer method stream, submit it to the primary
 * channel, poll the semaphore until CE releases our payload (or
 * MC_TIMEOUT_MS elapses).  Synchronous by design — returns only
 * when the transfer has committed all bytes through the PCIe
 * fabric (LAUNCH_DMA's FLUSH_ENABLE=TRUE guarantees that before
 * the sema release fires).
 */
static mc_status_t mc_submit_transfer(mc_ctx_t *ctx, NvU64 src_va, NvU64 dst_va,
                                      size_t n)
{
  mc_channel_t *ch;
  uint32_t     *pb_end;
  uint32_t      copy_bytes;

  if (ctx == NULL) return MC_EINVAL;
  if (n == 0 || n > MC_MAX_TRANSFER_SIZE) return MC_EINVAL;

  ch = &ctx->ch[MC_CH_PRIMARY];
  ch->sema_payload++;
  if (ch->sema_payload == 0) ch->sema_payload = 1; /* avoid 0 */

  pb_end = mc_write_transfer_methods(ch->pb_cpu, src_va, dst_va,
                                     (uint32_t)n, ch->sema_gpu_va,
                                     ch->sema_payload);
  copy_bytes = (uint32_t)((pb_end - ch->pb_cpu) * sizeof(uint32_t));

  return mc_channel_submit(ch, ctx->vf_doorbell, copy_bytes);
}

/*
 * D2H — CE reads vidmem at src_dev, writes sysmem at dst_host.  Both
 * addresses are valid GPU VAs: dst_host works because mc_malloc_host
 * returned a Paper-F1-anchored pointer that IS the GPU VA.
 */
mc_status_t mc_memcpy_d2h(mc_ctx_t *ctx, void *dst_host, const void *src_dev,
                          size_t n)
{
  return mc_submit_transfer(ctx, (NvU64)(uintptr_t)src_dev,
                            (NvU64)(uintptr_t)dst_host, n);
}

/*
 * H2D — CE reads sysmem at src_host (also its GPU VA thanks to Paper
 * F1), writes vidmem at dst_dev.  Same mc_submit_transfer call as D2H
 * with the src/dst pair swapped — the method stream builder is
 * direction-agnostic, and the CE doesn't care which side is vidmem.
 */
mc_status_t mc_memcpy_h2d(mc_ctx_t *ctx, void *dst_dev, const void *src_host,
                          size_t n)
{
  return mc_submit_transfer(ctx, (NvU64)(uintptr_t)src_host,
                            (NvU64)(uintptr_t)dst_dev, n);
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
  ch           = &ctx->ch[MC_CH_DMA];
  ex           = &ch->x.dma;
  dbell_gpu_va = ctx->vas[MC_VAS_PRIMARY_CARRIER].dbell_gpu_va;
  if (!ch->h_channel || !dbell_gpu_va) return MC_EINTERNAL;

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
 * mc_memcpy_d2h_gpu_doorbell_ce — D2H whose UVM-channel doorbell is
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
mc_status_t mc_memcpy_d2h_gpu_doorbell_ce(mc_ctx_t *ctx, void *dst_host,
                                          const void *src_dev, size_t n)
{
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

  if (ctx == NULL || dst_host == NULL || src_dev == NULL) return MC_EINVAL;
  if (n == 0 || n > MC_MAX_TRANSFER_SIZE) return MC_EINVAL;

  prim         = &ctx->ch[MC_CH_PRIMARY];
  dma          = &ctx->ch[MC_CH_DMA];
  dma_ex       = &dma->x.dma;
  dbell_gpu_va = ctx->vas[MC_VAS_PRIMARY_CARRIER].dbell_gpu_va;
  if (!dma->h_channel || !dbell_gpu_va) return MC_EINTERNAL;

  /* Arm primary's HBM->DRAM submission: build pushbuffer + write
   * GPFIFO entry + advance USERD GPPut, but DO NOT ring vf_doorbell.
   * mc_channel_arm captures t0 before publish so the eventual poll
   * starts its budget here. */
  prim->sema_payload++;
  if (prim->sema_payload == 0) prim->sema_payload = 1;

  prim_pb_end = mc_write_transfer_methods(
      prim->pb_cpu,
      (NvU64)(uintptr_t)src_dev, (NvU64)(uintptr_t)dst_host,
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
  return mc_channel_poll_sema(prim, t0);
}
