/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_vaspace.c — VA pool (process-global Paper-F1 anchor) + VA-space
 * helpers (mc_va_space_*).
 *
 * The VA pool is a 4-GiB PROT_NONE anonymous reservation at
 * MC_VA_POOL_BASE that every CPU-aliased allocation lands inside, so
 * GPU VA == user VA (Paper Finding 1).  va_pool_reserve hands out
 * sub-windows; rm_alloc_sysmem_at + rm_map_memory_at + uvm_map_buffer_at
 * MAP_FIXED into them.
 *
 * The VA-space helpers wrap rm_alloc_vaspace + per-resource
 * rm_alloc_virtual_memory into a typed mc_va_space_t.  Each NVOS46
 * mapping into the carrier VAS allocates its own NV50_MEMORY_VIRTUAL
 * sized to the source hMemory, mirroring libcuda's non-UVM shape: the
 * (GPFIFO 8 KiB + SYSMEM 48 B + USERD 512 B)-per-channel triple, each in its own
 * carrier with dmaOffset=0.  The previous bump-allocator over a single
 * 4-GiB carrier did not match libcuda and wedged the FB-resident
 * channel experiment.
 */

#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#include "nvtypes.h"

#include "mc_internal.h"
/* ── VA pool (process-global) ──────────────────────────────────────────────
 *
 * libcuda strace shows an up-front `mmap(0x200000000, 4 GiB, PROT_NONE,
 * MAP_ANONYMOUS)` reservation, followed by MAP_FIXED re-mappings of
 * file-backed RM memory into sub-windows — then UVM_CREATE_EXTERNAL_RANGE
 * at those same addresses.  This guarantees GPU VA == user VA for every
 * UVM-mapped allocation (Paper Finding 1), which is what the kernel-side
 * doorbell watchpoint relies on and, empirically, what the GPU itself
 * needs for reliable submissions on H100 PCIe
 * (docs/mc_architecture.md §3.7; --size 128M went 66/100 before
 * adopting this layout and 1000/1000 at N=1000 after).
 *
 * Our pool is a 4-GiB PROT_NONE anonymous reservation at MC_VA_POOL_BASE
 * = 0x200000000.  rm_alloc_sysmem_at, rm_map_memory_at, and
 * uvm_map_buffer_at accept a caller-supplied target VA so the initial
 * mmap lands in the pool — no mremap step, no stale tracker entries.
 * mc_malloc_device skips the pool entirely (no CPU alias).
 *
 * The pool is process-global (file-scope g_va_pool): one context per
 * process, initialized once by mc_init's first call.  Calling mc_init
 * twice is a no-op for the pool.
 */
static struct
{
  NvU64 base;
  NvU64 end;
  NvU64 cursor;
  int   initialized;
} g_va_pool;

/*
 * Reserve the 4 GiB PROT_NONE window.  If ASLR placed something else in
 * the way, mmap returns a different address — we CHECK for exact match
 * and fail loudly with a remediation hint (disabling ASLR).  In a
 * fresh process this reservation always succeeds on Linux x86_64.
 */
int va_pool_init(void)
{
  void *got;
  if (g_va_pool.initialized)
    return 0;

  got = mmap((void *)(uintptr_t)MC_VA_POOL_BASE, MC_VA_POOL_SIZE, PROT_NONE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (got != (void *)(uintptr_t)MC_VA_POOL_BASE)
  {
    ERROR_LOG("va_pool_init: mmap(%#llx, %#llx) got %p — ASLR collision? "
              "Disable with `sudo sysctl -w kernel.randomize_va_space=0`.",
              (unsigned long long)MC_VA_POOL_BASE,
              (unsigned long long)MC_VA_POOL_SIZE, got);
    return -1;
  }
  g_va_pool.base        = MC_VA_POOL_BASE;
  g_va_pool.end         = MC_VA_POOL_BASE + MC_VA_POOL_SIZE;
  g_va_pool.cursor      = MC_VA_POOL_BASE;
  g_va_pool.initialized = 1;
  return 0;
}

/*
 * Reserve a chunk inside the pool.  Sizes are rounded up to a 2 MiB
 * boundary — that's Hopper's natural UVM big-page size, so a single-
 * big-page mapping covers each reservation without sub-page edge
 * cases.  Returns the base CPU VA (will be the GPU VA too once UVM
 * maps it), or NULL if the pool is exhausted.
 */
void *va_pool_reserve(NvU64 size, const char *label)
{
  NvU64 align        = MC_VA_POOL_ALIGN_BYTES;
  NvU64 aligned_size = (size + align - 1) & ~(align - 1);
  NvU64 base         = (g_va_pool.cursor + align - 1) & ~(align - 1);
  if (base + aligned_size > g_va_pool.end)
  {
    ERROR_LOG("va_pool_reserve(%s, size=%#llx): pool exhausted", label,
              (unsigned long long)size);
    return NULL;
  }
  g_va_pool.cursor = base + aligned_size;
  DEBUG_LOG("va_pool_reserve(%s): %#llx..%#llx", label,
            (unsigned long long)base,
            (unsigned long long)(base + aligned_size));
  return (void *)(uintptr_t)base;
}

/* ── VA-space helpers ──────────────────────────────────────────────────────
 *
 * Each carrier VAS is a plain FERMI_VASPACE_A.  Per-resource NV50
 * carriers are allocated lazily by mc_va_space_dma_map_resource — one
 * per source hMemory, sized to the source.  RM picks the GPU VA at
 * NVOS46 time (dmaOffset=0).  Tracked in vas->carriers[] for teardown.
 *
 * Per libcuda's recipe, channel resources break down as:
 *   GPFIFO   8 KiB  NV01_MEMORY_LOCAL_USER (FB)
 *   USERD    512 B  NV01_MEMORY_LOCAL_USER (FB)
 *   PB       N MiB  NV01_MEMORY_SYSTEM (sysmem)
 *   sema     ~48 B  NV01_MEMORY_SYSTEM (sysmem)
 * mc's current sizes are ~2 MiB chunks per resource (see
 * MC_GPFIFO_USERD_SIZE) — fine for now; we match libcuda's *shape*
 * (one carrier per source) without yet matching its tight sizing.
 */

int mc_va_space_init_uvm(mc_ctx_t *ctx)
{
  mc_va_space_t *vas = &ctx->vas[MC_VAS_UVM];

  vas->kind = MC_VAS_KIND_UVM;
  vas->h_vaspace = rm_alloc_vaspace(ctx->ctl_fd, ctx->h_client, ctx->h_device);
  if (!vas->h_vaspace) return -1;
  return 0;
}

/* Shared init core for both carrier kinds.  Allocates the FERMI VAS
 * only; per-resource carriers are added on demand by
 * mc_va_space_dma_map_resource.
 *
 * The caller picks which mc_ctx.vas[] slot to populate by passing its
 * index; the kind comes along to differentiate FB from sysmem at every
 * later kind-aware site (channel-core bring-up, teardown, dispatch). */
static int mc_va_space_init_carrier_kind(mc_ctx_t *ctx, mc_vas_t slot,
                                         mc_va_space_kind_t kind,
                                         const char *label)
{
  mc_va_space_t *vas = &ctx->vas[slot];

  vas->kind = kind;
  vas->h_vaspace = rm_alloc_vaspace_dma(ctx->ctl_fd, ctx->h_client,
                                        ctx->h_device);
  if (!vas->h_vaspace) return -1;
  vas->carrier_count = 0;
  DEBUG_LOG("%s VAS: h_vaspace=0x%x (per-resource carriers allocated lazily)",
            label, vas->h_vaspace);
  return 0;
}

int mc_va_space_init_carrier(mc_ctx_t *ctx)
{
  return mc_va_space_init_carrier_kind(ctx, MC_VAS_SYSMEM_CARRIER,
                                       MC_VAS_KIND_CARRIER, "sysmem-carrier");
}

int mc_va_space_init_carrier_fb(mc_ctx_t *ctx)
{
  return mc_va_space_init_carrier_kind(ctx, MC_VAS_FB_CARRIER,
                                       MC_VAS_KIND_CARRIER_FB, "fb-carrier");
}

/* Per-resource NV50 carrier allocation + NVOS46.  libcuda's shape:
 * one NV50_MEMORY_VIRTUAL per source hMemory, sized to the source,
 * NV04_MAP_MEMORY_DMA with dmaOffset=0 (RM picks GPU VA).  Returns
 * the RM-chosen GPU VA on success, 0 on failure.
 *
 * The carrier is recorded in vas->carriers[] so mc_va_space_fini can
 * NVOS47-unmap and NV01_FREE both handles.  The caller still owns
 * h_mem's lifecycle — this helper does not free h_mem on failure. */
NvU64 mc_va_space_dma_map_resource(mc_ctx_t *ctx, mc_va_space_t *vas,
                                   NvHandle h_mem, NvU64 size)
{
  NvHandle h_carrier;
  NvU64    gpu_va;

  if (vas == NULL || !mc_va_space_kind_is_carrier(vas->kind)) return 0;
  if (size == 0 || h_mem == 0)                                return 0;

  if (vas->carrier_count >= MC_VAS_CARRIER_MAX)
  {
    ERROR_LOG("mc_va_space_dma_map_resource: carrier table full "
              "(%d entries, increase MC_VAS_CARRIER_MAX)",
              vas->carrier_count);
    return 0;
  }

  /* Per-resource carrier: NV50_MEMORY_VIRTUAL sized to `size`, in the
   * caller's hVASpace.  RM will reject sub-page sizes — round up to
   * one MMU page (4 KiB) at minimum. */
  NvU64 carrier_size = (size + 0xFFFULL) & ~0xFFFULL;
  h_carrier = rm_alloc_virtual_memory(ctx->ctl_fd, ctx->h_client,
                                      ctx->h_device, vas->h_vaspace,
                                      carrier_size, NULL);
  if (!h_carrier) return 0;

  /* dmaOffset=0 → RM picks the GPU VA from the carrier's range. */
  gpu_va = rm_map_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                             h_carrier, h_mem, 0, size, 0);
  if (!gpu_va)
  {
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_carrier,
                   "carrier_rollback");
    return 0;
  }

  vas->carriers[vas->carrier_count++] = (mc_resource_carrier_t){
      .h_carrier = h_carrier,
      .h_mem     = h_mem,
      .gpu_va    = gpu_va,
      .size      = size,
  };
  return gpu_va;
}

/* Release the per-resource carrier(s) mapping `h_mem` into `vas`.
 * NVOS47-unmap each match, NV01_FREE the carrier handle, then shift
 * survivors down in vas->carriers[].  Caller invokes this BEFORE
 * freeing h_mem (channel teardown), so the NVOS47 still has a valid
 * source object.
 *
 * Cost: O(carrier_count) per call.  carrier_count <= MC_VAS_CARRIER_MAX
 * (16) — fine. */
void mc_va_space_release_carrier(mc_ctx_t *ctx, mc_va_space_t *vas,
                                 NvHandle h_mem)
{
  int i, j;

  if (ctx == NULL || vas == NULL) return;
  if (ctx->ctl_fd < 0 || ctx->h_client == 0) return;
  if (h_mem == 0) return;
  if (!mc_va_space_kind_is_carrier(vas->kind)) return;

  for (i = 0; i < vas->carrier_count; )
  {
    mc_resource_carrier_t *rc = &vas->carriers[i];
    if (rc->h_mem != h_mem)
    {
      i++;
      continue;
    }
    if (rc->gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          rc->h_carrier, rc->h_mem, rc->gpu_va);
    if (rc->h_carrier)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     rc->h_carrier, "vas.carrier_release");
    /* Compact the table: shift survivors down. */
    for (j = i + 1; j < vas->carrier_count; j++)
      vas->carriers[j - 1] = vas->carriers[j];
    vas->carrier_count--;
    /* Don't increment i — the slot we just compacted into may also
     * match (shouldn't, but cheap). */
  }
}

/* Install the BAR1 USERMODE_A doorbell page as a PTE inside the carrier
 * VAS.  Stores the resulting GPU VA on the VAS itself so every channel
 * bound to this VAS sees the same doorbell address.  Idempotent: if the
 * PTE is already installed (dbell_gpu_va != 0), returns success.
 *
 * The doorbell page gets its own per-resource NV50 carrier just like
 * any other resource — no shared mega-carrier. */
int mc_va_space_install_doorbell_pte(mc_ctx_t *ctx, mc_va_space_t *vas)
{
  NvU64 gpu_va;

  if (vas == NULL || !mc_va_space_kind_is_carrier(vas->kind)) return -1;
  if (vas->dbell_gpu_va) return 0;

  gpu_va = mc_va_space_dma_map_resource(ctx, vas, ctx->h_usermode_bar1,
                                        MC_USERMODE_SIZE);
  if (!gpu_va) return -1;

  vas->dbell_gpu_va = gpu_va;
  DEBUG_LOG("BAR1 doorbell PTE installed in carrier VAS at gpu_va=0x%llx",
            (unsigned long long)gpu_va);
  return 0;
}

/* Allocate `size` bytes of sysmem, DMA-map it into the carrier VAS, and
 * return both the host CPU pointer and the GPU VA.  The caller stores
 * the (h_mem, cpu, gpu_va) triple wherever it makes sense; this helper
 * doesn't track the allocation.  On success returns the carved GPU VA;
 * on failure returns 0 and leaves *out_h_mem / *out_cpu untouched.
 *
 * Used wherever a channel needs a small dedicated sysmem region with
 * its own RM hMemory and GPU MMU PTE — token cells, compute QMD/CB0/
 * SASS images, the compute scratch dword. */
NvU64 mc_va_space_alloc_scratch(mc_ctx_t *ctx, mc_va_space_t *vas,
                                       NvU64 size, NvU64 align,
                                       NvHandle *out_h_mem,
                                       void **out_cpu)
{
  NvU64    gpu_va;
  NvHandle h_mem;
  void    *cpu = NULL;

  (void)align;  /* RM picks the GPU VA in the per-resource carrier path. */

  if (vas == NULL || !mc_va_space_kind_is_carrier(vas->kind)) return 0;
  if (size == 0) return 0;

  h_mem = rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                             ctx->h_device, size, NULL, &cpu);
  if (!h_mem || cpu == NULL) return 0;

  gpu_va = mc_va_space_dma_map_resource(ctx, vas, h_mem, size);
  if (!gpu_va)
  {
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_mem,
                   "scratch_rollback");
    return 0;
  }

  *out_h_mem = h_mem;
  *out_cpu   = cpu;
  return gpu_va;
}

/* Allocate `size` bytes of vidmem (HBM), DMA-map it into the carrier
 * VAS, and return the GPU VA.  Counterpart to mc_va_space_alloc_scratch
 * for HBM-backed buffers — used by the upcoming carrier-VAS arm of
 * mc_malloc_device.
 *
 * Vidmem allocs have no CPU alias; the caller gets only an hMemory
 * handle (for free) and a GPU VA (for use in CE submissions).
 *
 * Alignment defaults to 2 MiB when caller passes 0.  RM picks a vidmem
 * PTE page size based on the allocation size:
 *
 *   small allocs (~ <= 1 MiB)   → 64 KiB pages   (BAR1 small page)
 *   large allocs (>= ~64 MiB)   → 2 MiB pages    (BAR1 big page)
 *
 * The page size determines the GPU-VA alignment RM accepts; passing a
 * 64-KiB-aligned GPU VA for a 64-MiB allocation gets rejected with
 * `dmaAllocMapping_GM107: Virtual address ... is not compatible with
 * page size 0x200000`.  2-MiB align is universally accepted (it's
 * already a safe choice; with per-resource carriers (RM picks the
 * GPU VA), the page-size compatibility check is RM's problem.  The
 * `align` parameter is now ignored — kept in the signature for ABI
 * compat with existing callers.
 *
 * On success returns the carved GPU VA and stores the hMemory in
 * *out_h_mem.  On failure returns 0; *out_h_mem is untouched. */
NvU64 mc_va_space_alloc_vidmem(mc_ctx_t *ctx, mc_va_space_t *vas,
                                      NvU64 size, NvU64 align,
                                      NvHandle *out_h_mem)
{
  NvU64    gpu_va;
  NvHandle h_mem;

  (void)align;  /* RM picks the GPU VA in the per-resource carrier path. */

  if (vas == NULL || !mc_va_space_kind_is_carrier(vas->kind)) return 0;
  if (size == 0 || out_h_mem == NULL) return 0;

  h_mem = rm_alloc_vidmem(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          size, NULL);
  if (!h_mem) return 0;

  gpu_va = mc_va_space_dma_map_resource(ctx, vas, h_mem, size);
  if (!gpu_va)
  {
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_mem,
                   "vidmem_rollback");
    return 0;
  }

  *out_h_mem = h_mem;
  return gpu_va;
}

void mc_va_space_fini(mc_ctx_t *ctx, mc_vas_t id)
{
  mc_va_space_t *vas = &ctx->vas[id];
  int i;

  if (ctx->ctl_fd < 0 || ctx->h_client == 0) return;

  if (mc_va_space_kind_is_carrier(vas->kind))
  {
    /* Walk per-resource carriers in reverse alloc order: NVOS47-unmap
     * each, then NV01_FREE the carrier handle.  The source hMemory
     * (h_mem) is owned by the caller of mc_va_space_dma_map_resource —
     * channel_free_core already freed those before reaching here.  We
     * only own the carriers themselves. */
    for (i = vas->carrier_count - 1; i >= 0; i--)
    {
      mc_resource_carrier_t *rc = &vas->carriers[i];
      if (rc->gpu_va)
        rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                            rc->h_carrier, rc->h_mem, rc->gpu_va);
      if (rc->h_carrier)
        rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                       rc->h_carrier, "vas.carrier");
    }
    vas->carrier_count = 0;
    vas->dbell_gpu_va  = 0;
  }

  if (vas->h_vaspace)
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                   vas->h_vaspace, "vas.h_vaspace");

  memset(vas, 0, sizeof(*vas));
}

/* ── Channel registry on a VAS ────────────────────────────────────────────
 *
 * Each VAS owns its channels inline in vas->channels[].  Bring-up calls
 * mc_vas_add_channel(vas, role) to reserve the next free slot; the
 * caller fills the rest of the channel struct.  mc_vas_find_channel
 * resolves (vas, role) → channel pointer for dispatch.  The role-uniqueness
 * invariant (one channel per role per VAS) is enforced here at add time —
 * a duplicate role aborts the process via CHECK rather than letting a
 * silently shadowed second channel hide a bring-up bug.
 */
mc_channel_t *mc_vas_add_channel(mc_va_space_t *vas, mc_channel_role_t role)
{
  int i;
  if (vas == NULL) return NULL;

  for (i = 0; i < vas->channel_count; i++)
    CHECK(vas->channels[i].role != role,
          "mc_vas_add_channel: role %d already present in this VAS",
          (int)role);

  if (vas->channel_count >= MC_VAS_CH_MAX)
  {
    ERROR_LOG("mc_vas_add_channel: VAS already holds MC_VAS_CH_MAX (%d) channels",
              MC_VAS_CH_MAX);
    return NULL;
  }
  mc_channel_t *ch = &vas->channels[vas->channel_count++];
  ch->role = role;
  return ch;
}

mc_channel_t *mc_vas_find_channel(mc_va_space_t *vas, mc_channel_role_t role)
{
  int i;
  if (vas == NULL) return NULL;
  for (i = 0; i < vas->channel_count; i++)
    if (vas->channels[i].role == role)
      return &vas->channels[i];
  return NULL;
}
