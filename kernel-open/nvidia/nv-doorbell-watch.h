/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * nv-doorbell-watch: kernel-side Hopper VF_DOORBELL watchpoint.
 * After Yan et al. §5.1.  See
 * "Rebuilding CUDA From Scratch" Part 4 for the full design.
 *
 * Public surface consumed by nv-mmap.c.
 */

#ifndef _NV_DOORBELL_WATCH_H_
#define _NV_DOORBELL_WATCH_H_

#include "nv-linux.h"

/* Tag used to disambiguate our VMA ctx from sysmem nv_alloc_t* */
#define NV_DBELL_VMA_TAG   0xDB11D00BUL

/* Number of simultaneous BAR0 VF doorbell mappings we can watch.
 * x86 provides 4 hardware debug registers (DR0..DR3). */
#define NV_DBELL_SLOTS     4

/* Offset of the VF_DOORBELL dword inside the 64-KiB HOPPER_USERMODE_A
 * window (NV_VIRTUAL_FUNCTION_DOORBELL - NV_VIRTUAL_FUNCTION_BASE). */
#define NV_DBELL_OFFSET    0x90

/* Size of the HOPPER_USERMODE_A region.  nv_mmap uses this as a first
 * filter so we only try to intercept matching VMAs. */
#define NV_DBELL_REGION_SIZE  0x10000

/*
 * Called from nvidia_mmap_helper()'s device-node path, after the
 * register/FB caching branches and before the remap loop — deliberately
 * outside both of them, because the hook has to see BAR0 (IS_REG_OFFSET)
 * mappings as well as BAR1 (IS_FB_OFFSET) ones: mc allocates both
 * USERMODE variants.  It keys solely off the
 * size == NV_DBELL_REGION_SIZE (64 KiB) + single contig range signature,
 * which is what uniquely identifies a HOPPER_USERMODE_A mapping, and not
 * off an offset class at all.  Returns:
 *   > 0: shadow page installed, VMA fully populated by this helper.
 *        Caller must skip its own nv_io_remap_page_range loop.
 *   = 0: not intercepted — the nv_dbell_disable_intercept kill switch is
 *        set, no slot is free, or the mapping does not match the shape.
 *        Caller falls through to the normal remap path.
 *   < 0: install failed; the caller propagates this errno verbatim.
 */
int nv_dbell_intercept_mmap(nv_state_t *nv,
                            struct vm_area_struct *vma,
                            nv_alloc_mapping_context_t *mmap_context);

/*
 * Called from nvidia_vma_release() when NV_VMA_PRIVATE(vma) has our
 * tag.  Tears down the breakpoint and frees the shadow page.
 * Safe to call from any sleeping context.
 */
void nv_dbell_vma_release(struct vm_area_struct *vma);

/*
 * Tag probe: returns true iff vm_private_data points at one of our
 * ctx structs (used to route nvidia_vma_release()).
 */
int nv_dbell_is_watched_vma(struct vm_area_struct *vma);

/*
 * BAR1 mapping tracker — the FBMEM USERD resolution path.
 *
 * On Hopper, libcuda allocates USERD in vidmem.  Its phys address is
 * in the memdesc but memdescGetKernelMapping() returns NULL (no kmap
 * for FBMEM).  To read it we reuse the kernel ioremap that was already
 * going to be set up for CUDA's own 2 MiB BAR1 FB mapping (the one
 * containing pushbuffer + GPFIFO + USERDs).
 *
 * nv_dbell_bar1_track_add: called from nvidia_mmap_helper after a
 *   device-node non-UD FB mapping succeeds — records the (phys range,
 *   kernel VA) pair so resolver work items can look up a USERD phys
 *   and get a readable kernel VA.  Returns 0 on success, negative on
 *   failure (table full, ioremap failed).  Caller is expected to call
 *   _remove with the same vma before VMA tear-down completes.
 * nv_dbell_bar1_track_remove: iounmap + drop the table entry.
 * nv_dbell_bar1_lookup_kva: phys -> kernel VA or NULL.  Safe to call
 *   from the kthread worker (not from #DB context — takes a spinlock).
 */
int   nv_dbell_bar1_track_add(struct vm_area_struct *vma,
                              NvU64 phys_start, NvU64 phys_size);
void  nv_dbell_bar1_track_remove(struct vm_area_struct *vma);
void *nv_dbell_bar1_lookup_kva(NvU64 phys, NvU64 size);

/*
 * For the case where RM couldn't give us a pool_kva (userdBar1CpuPtr is
 * NULL, e.g. SRIOV vGPU or per-process BAR1 USERD pools on Hopper):
 * pick the first large tracked BAR1 mapping and add `off` to its kva.
 * Returns NULL if no mapping covers [off, off+size).
 */
void *nv_dbell_bar1_offset_kva(NvU64 off, NvU64 size);

/*
 * Translate a GPU VA (under UVM == userspace VA) to a kernel VA.  Uses
 * the `vma->vm_start` captured at track-add time to find the owning
 * BAR1 mapping and compute the offset.  Used to reach the per-channel
 * GPFIFO ring from RAMFC's GP_BASE value.
 */
void *nv_dbell_bar1_gpu_va_to_kva(NvU64 gpu_va, NvU64 size);

/*
 * Sysmem mapping tracker — the pushbuffer-bytes readout path.
 *
 * libcuda's pushbuffer pool on Hopper is a large sysmem allocation
 * (typical size 56 MiB, class NV01_MEMORY_SYSTEM) mapped via
 * NVOS33 RM_MAP_MEMORY + MAP_FIXED mmap of /dev/nvidiactl.  Under UVM
 * libcuda chooses the user VA deliberately so GPU VA == user VA
 * (Paper Finding 1).
 *
 * The #DB handler decodes a pb_va from the GPFIFO entry and wants to
 * read the method-stream bytes.  This tracker records (user_va_start,
 * user_va_end, kernel_va) keyed by the VMA, analogous to the BAR1
 * tracker but backed by vmap()'d sysmem pages instead of ioremap'd
 * device BARs.
 *
 * nv_dbell_sysmem_track_add: called from nvidia_mmap_helper after
 *   nvidia_mmap_sysmem succeeds.  Only runs for mappings ≥ min_pages
 *   (filter set in nv-mmap.c) to avoid vmap'ing every tiny RM alloc.
 *   Builds a temporary struct page* array from at->page_table, calls
 *   nv_vm_map_pages() to get a persistent kernel VA, and records
 *   (vma, user_va_start, user_va_end, kernel_va, num_pages).
 *
 * nv_dbell_sysmem_track_remove: nv_vm_unmap_pages + drop table entry.
 *
 * nv_dbell_sysmem_gpu_va_to_kva: lookup by user VA (= GPU VA under
 *   UVM).  Safe to call from the #DB handler (takes a spinlock but
 *   does no sleepable work).  Returns NULL if no tracked mapping
 *   covers [gpu_va, gpu_va+size).
 */
int   nv_dbell_sysmem_track_add(struct vm_area_struct *vma,
                                nvidia_pte_t *page_table,
                                NvU64 page_index,
                                NvU64 num_pages);
void  nv_dbell_sysmem_track_remove(struct vm_area_struct *vma);
void *nv_dbell_sysmem_gpu_va_to_kva(NvU64 gpu_va, NvU64 size);

/*
 * Cache pre-resolve: enqueue the async resolver at channel register time
 * rather than waiting for the first doorbell.  Called from
 * nvGpuOpsDbellGpfifoRegister (nv_gpu_ops.c) during channel construct,
 * so the per-(chid, runlist) cache slot is RESOLVED before PBDMA ever
 * sees a doorbell on that channel — eliminates the "pending_init, no
 * PB decode" gap on the first submission per channel.
 *
 * No-op if an entry for (chid, runlist) already exists (PENDING or
 * RESOLVED) or if the cache is full (the #DB-handler fallback still
 * covers those cases).  Uses GFP_ATOMIC / non-blocking kthread enqueue;
 * safe to call under GPU locks.
 */
void NV_API_CALL nv_dbell_cache_pre_resolve(NvU32 chid, NvU32 runlist,
                                            void *nv_opaque);

#endif /* _NV_DOORBELL_WATCH_H_ */
