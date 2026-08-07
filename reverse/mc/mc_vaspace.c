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
 * The VA-space helpers wrap rm_alloc_vaspace + rm_alloc_virtual_memory
 * into a typed mc_va_space_t.  The carrier VAS (MC_VAS_PRIMARY_CARRIER)
 * also hosts a bump allocator (mc_va_space_carve) and the BAR1 doorbell
 * PTE shared between the DMA + compute channels.
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
 * The carrier VAS hands out GPU-VA windows via a simple bump allocator
 * (mc_va_space_carve).  The DMA + compute channels each grab a few 2 MiB
 * tiles for their gpfifo / pushbuffer / sema / kind-specific buffers; the
 * BAR1 doorbell PTE lives at a fixed offset inside the same VAS.
 *
 * Capacity is sized for today's two carrier-bound channels (DMA +
 * COMPUTE) plus head-room for a future H2D channel sharing the
 * carrier; bump the constant if more is needed.
 */
#define MC_CARRIER_VIRT_SIZE      (256ULL * 1024ULL * 1024ULL)
#define MC_CARRIER_DEFAULT_ALIGN  (2ULL * 1024ULL * 1024ULL)

int mc_va_space_init_uvm(mc_ctx_t *ctx)
{
  mc_va_space_t *vas = &ctx->vas[MC_VAS_PRIMARY_UVM];

  vas->kind = MC_VAS_KIND_UVM;
  vas->h_vaspace = rm_alloc_vaspace(ctx->ctl_fd, ctx->h_client, ctx->h_device);
  if (!vas->h_vaspace) return -1;
  return 0;
}

int mc_va_space_init_carrier(mc_ctx_t *ctx)
{
  mc_va_space_t *vas = &ctx->vas[MC_VAS_PRIMARY_CARRIER];

  vas->kind = MC_VAS_KIND_CARRIER;
  vas->h_vaspace = rm_alloc_vaspace_dma(ctx->ctl_fd, ctx->h_client,
                                        ctx->h_device);
  if (!vas->h_vaspace) return -1;

  vas->h_virt = rm_alloc_virtual_memory(
      ctx->ctl_fd, ctx->h_client, ctx->h_device, vas->h_vaspace,
      MC_CARRIER_VIRT_SIZE, &vas->virt_base);
  if (!vas->h_virt) return -1;
  vas->virt_size   = MC_CARRIER_VIRT_SIZE;
  vas->virt_cursor = 0;
  DEBUG_LOG("carrier VAS: h_virt=0x%x base=0x%llx size=%llu MiB",
            vas->h_virt, (unsigned long long)vas->virt_base,
            (unsigned long long)(vas->virt_size >> 20));
  return 0;
}

/* Bump-allocate a GPU-VA window inside the carrier VAS.  Returns the GPU
 * VA on success, 0 if the request would overflow virt_size.  The caller
 * still has to NV04_MAP_MEMORY_DMA something into that range; this helper
 * just hands out the address. */
NvU64 mc_va_space_carve(mc_va_space_t *vas, NvU64 size, NvU64 align)
{
  NvU64 aligned_cursor;

  if (vas == NULL || vas->kind != MC_VAS_KIND_CARRIER) return 0;
  if (size == 0) return 0;
  if (align == 0) align = MC_CARRIER_DEFAULT_ALIGN;

  aligned_cursor = (vas->virt_cursor + align - 1) & ~(align - 1);
  if (aligned_cursor + size > vas->virt_size)
  {
    ERROR_LOG("mc_va_space_carve: out of carrier VAS (cursor=%llu need=%llu cap=%llu)",
              (unsigned long long)aligned_cursor, (unsigned long long)size,
              (unsigned long long)vas->virt_size);
    return 0;
  }
  vas->virt_cursor = aligned_cursor + size;
  return vas->virt_base + aligned_cursor;
}

/* Install the BAR1 USERMODE_A doorbell page as a PTE inside the carrier
 * VAS.  Stores the resulting GPU VA on the VAS itself so every channel
 * bound to this VAS sees the same doorbell address.  Idempotent: if the
 * PTE is already installed (dbell_gpu_va != 0), returns success. */
int mc_va_space_install_doorbell_pte(mc_ctx_t *ctx, mc_va_space_t *vas)
{
  NvU64 gpu_va;

  if (vas == NULL || vas->kind != MC_VAS_KIND_CARRIER) return -1;
  if (vas->dbell_gpu_va) return 0;

  gpu_va = mc_va_space_carve(vas, MC_USERMODE_SIZE, MC_USERMODE_SIZE);
  if (!gpu_va) return -1;
  if (rm_map_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                        vas->h_virt, ctx->h_usermode_bar1, 0,
                        MC_USERMODE_SIZE, gpu_va) != gpu_va)
    return -1;
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

  if (vas == NULL || vas->kind != MC_VAS_KIND_CARRIER) return 0;
  if (size == 0) return 0;

  gpu_va = mc_va_space_carve(vas, size, align ? align : 0x1000);
  if (!gpu_va) return 0;

  h_mem = rm_alloc_sysmem_at(ctx->ctl_fd, ctx->dev_fd, ctx->h_client,
                             ctx->h_device, size, NULL, &cpu);
  if (!h_mem || cpu == NULL) return 0;

  if (rm_map_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                        vas->h_virt, h_mem, 0, size, gpu_va) != gpu_va)
  {
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device, h_mem,
                   "scratch_rollback");
    return 0;
  }

  *out_h_mem = h_mem;
  *out_cpu   = cpu;
  return gpu_va;
}

void mc_va_space_fini(mc_ctx_t *ctx, mc_va_space_id_t id)
{
  mc_va_space_t *vas = &ctx->vas[id];

  if (ctx->ctl_fd < 0 || ctx->h_client == 0) return;

  if (vas->kind == MC_VAS_KIND_CARRIER)
  {
    /* Doorbell PTE: only DMA-unmap if the PTE actually got installed.
     * The carrier itself is freed below; channels are expected to have
     * unmapped their own buffers already (channel_free_core does this
     * inside *_channel_fini, which runs before mc_va_space_fini). */
    if (vas->dbell_gpu_va)
      rm_unmap_memory_dma(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                          vas->h_virt, ctx->h_usermode_bar1,
                          vas->dbell_gpu_va);
    if (vas->h_virt)
      rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                     vas->h_virt, "vas.h_virt");
  }
  if (vas->h_vaspace)
    rm_free_handle(ctx->ctl_fd, ctx->h_client, ctx->h_device,
                   vas->h_vaspace, "vas.h_vaspace");


  memset(vas, 0, sizeof(*vas));
}
