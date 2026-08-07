/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc.h — minimal CUDA-runtime-like library over raw NVIDIA driver
 * ioctls.  Does one thing: synchronous Host⇄Device byte copies through
 * a single CE channel, with no CUDA runtime and no libcuda involved.
 *
 * Invariants the library enforces for you:
 *   - Paper F1:  for any allocation with a CPU alias (i.e. anything returned
 *                by mc_malloc_host), the pointer is BOTH the CPU VA and the
 *                GPU VA.  Pass the same pointer to memcpy on the host side
 *                and as source/destination to mc_memcpy.
 *   - mc_memcpy is synchronous: returns only after the semaphore fires, or
 *     MC_ETIMEOUT after TIMEOUT_MS (2 s).
 *   - Single-threaded.  Calling from multiple threads is undefined.
 *
 * Pointers returned by mc_malloc_device are NOT CPU-dereferenceable.  Reading
 * or writing them from CPU code is undefined behaviour; treat them as opaque
 * device handles and move data in/out via mc_memcpy_h2d / mc_memcpy_d2h.
 */
#ifndef MC_H
#define MC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mc_ctx mc_ctx_t;

typedef enum {
    MC_OK         = 0,
    MC_ETIMEOUT   = 1,  /* semaphore never fired within TIMEOUT_MS */
    MC_EALLOC     = 2,  /* allocation table full / OOM / vidmem refused */
    MC_EIOCTL     = 3,  /* driver ioctl failed */
    MC_EINVAL     = 4,  /* bad argument (size == 0, unknown pointer, ...) */
    MC_EINTERNAL  = 5,  /* library invariant violated (should not happen) */
} mc_status_t;

/* Opens /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm; reserves the
 * VA pool; brings up the UVM, DMA, and compute channels.  Returns
 * MC_OK on success and stores an opaque context pointer in *out_ctx.
 * On failure returns a status code (details go to stderr) and *out_ctx
 * is left NULL. */
mc_status_t mc_init(mc_ctx_t **out_ctx);

/* Strict-LIFO teardown: drain channel, unregister from UVM, munmap, rm_free
 * all handles, close fds, free mc_ctx_t.  Safe against NULL. */
void mc_fini(mc_ctx_t *ctx);

/* Allocate HBM with NO CPU alias.  Returned pointer is a GPU VA; do NOT
 * dereference from the CPU.  Returns NULL on failure. */
void *mc_malloc_device(mc_ctx_t *ctx, size_t n);

/* Allocate host memory with CPU VA == GPU VA (Paper F1 anchored into the
 * VA pool).  Returned pointer is usable on both sides.  Returns NULL on
 * failure. */
void *mc_malloc_host(mc_ctx_t *ctx, size_t n);

/* Free a pointer returned by either mc_malloc_*.  No-op on NULL. */
void mc_free(mc_ctx_t *ctx, void *p);

/* Synchronous H2D copy.  src must be a host-accessible pointer; dst must
 * have been returned by mc_malloc_device.  Returns MC_OK on success. */
mc_status_t mc_memcpy_h2d(mc_ctx_t *ctx, void *dst_dev,
                          const void *src_host, size_t n);

/* Synchronous D2H copy.  src must have been returned by mc_malloc_device;
 * dst must be host-writable.  Returns MC_OK on success. */
mc_status_t mc_memcpy_d2h(mc_ctx_t *ctx, void *dst_host,
                          const void *src_dev, size_t n);

/* Doorbell-write demo: submit a CE LAUNCH_DMA on the DMA channel
 * (whose non-UVM VAS holds a BAR1 doorbell PTE) that copies 4 bytes
 * from a sysmem cell into the BAR1 doorbell page at +0x90.  The cell
 * is pre-filled with `token`, so the GPU's CE engine — not the host —
 * issues the MMIO write that lands on the doorbell.
 *
 * Verification is the release semaphore, not the kernel watchpoint.  The
 * watchpoint is an x86 hardware breakpoint on the diverted userspace
 * mapping, so it only sees *CPU* writes to the doorbell; a CE-issued MMIO
 * write reaches BAR1 without the CPU touching it and is invisible there by
 * construction.  What proves the write happened is that the CE ran the
 * pushbuffer to completion: this call returns MC_OK only once the
 * CE-emitted release semaphore has fired.
 *
 * Synchronous: returns only after that semaphore fires (or MC_ETIMEOUT
 * after TIMEOUT_MS).
 *
 * `token` is the value the GPU writes to the doorbell.  0xDEADBEEF is
 * a safe choice: it's not a valid Hopper work-submit-token, so PBDMA
 * looks up its (chid, runlist) decoding, finds nothing, and silently
 * drops.  Note the kernel watchpoint does NOT see this write: the store
 * is issued by the GPU's Copy Engine, and an x86 debug register traps
 * only CPU accesses.  What proves the write happened is the demo's own
 * release semaphore. */
mc_status_t mc_dbell_demo_ring(mc_ctx_t *ctx, uint32_t token);

/* End-to-end-verifiable D2H copy whose UVM-channel doorbell ring is
 * issued by the GPU itself, not the host.
 *
 * Sequence:
 *   1. Arm the UVM channel: build a HBM→DRAM CE pushbuffer at
 *      its pb_cpu, write the GPFIFO entry, advance USERD GPPut.
 *      DO NOT ring the UVM channel's vf_doorbell.
 *   2. Build a tiny pushbuffer on the DMA channel that copies the
 *      UVM channel's work_submit_token (pre-filled into a sysmem
 *      token cell mapped into the DMA channel's VAS) to
 *      dbell_gpu_va + 0x90 — i.e. the BAR1 USERMODE_A doorbell as
 *      seen by the DMA channel's GPU MMU.
 *   3. Ring the DMA channel's doorbell from the host.  PBDMA wakes
 *      the DMA channel, its CE writes 4 bytes to BAR1 doorbell,
 *      which (on real hardware) wakes the UVM channel's PBDMA,
 *      which runs the queued HBM→DRAM copy, which fires the UVM
 *      channel's release semaphore.
 *   4. Poll the DMA channel's sema (CE op finished) then the UVM
 *      channel's sema (D2H finished + bytes flushed through PCIe
 *      per LAUNCH_DMA_FLUSH_ENABLE=TRUE).
 *
 * Verification: dst_host contains the bytes from src_dev iff the UVM
 * channel's doorbell got rung.  The host did not ring it.  Only
 * possible source of the ring is the DMA channel's CE-emitted MMIO
 * write reaching real BAR1.  Comparing dst_host vs. expected bytes
 * is therefore proof — outside libcuda, outside the watchpoint
 * shadow path, just direct CPU loads from sysmem.
 *
 * Returns MC_OK on full chain success, MC_ETIMEOUT if either sema
 * doesn't fire within TIMEOUT_MS. */
mc_status_t mc_memcpy_d2h_gpu_doorbell_ce(mc_ctx_t *ctx, void *dst_host,
                                          const void *src_dev, size_t n);

/* Submit a HOPPER_COMPUTE_A kernel launch on the compute channel
 * (which cohabits the DMA channel's non-UVM VAS).  The launched
 * kernel is the embedded mc_doorbell_kernel:
 *
 *   __global__ void mc_doorbell_kernel(volatile uint32_t *dst,
 *                                       uint32_t token)
 *   { *dst = token; }
 *
 * `dst_gpu_va` must be valid in the carrier VAS (i.e. allocated and
 * DMA-mapped through it).  Two intended call patterns:
 *
 *   - sysmem cell: pass the GPU VA of any DMA-mapped sysmem region.
 *     The host can read the cell back directly to verify the SM
 *     thread ran.
 *
 *   - BAR1 doorbell: pass `dbell_gpu_va + 0x90` so the SM thread
 *     issues an MMIO write to the H100's BAR1 doorbell register.
 *     Verifying this reached real BAR1 requires the chain-of-evidence
 *     pattern; see mc_memcpy_d2h_gpu_doorbell_sm.
 *
 * Synchronous: returns only after the kernel's release semaphore
 * fires (or MC_ETIMEOUT after MC_TIMEOUT_MS = 2 s).
 *
 * The first call also submits a one-time per-channel setup
 * pushbuffer (SET_OBJECT(HOPPER_COMPUTE_A), cache invalidates,
 * SHADER_*_MEMORY_WINDOW), built NVK-style — see
 * mc_write_compute_setup_methods.  Subsequent calls skip the setup. */
mc_status_t mc_compute_doorbell_kernel(mc_ctx_t *ctx, uint64_t dst_gpu_va,
                                       uint32_t token);

/* Expose a small, GPU-MMU-mapped scratch dword that lives in the
 * compute channel's VAS — useful as the `dst` for a sysmem-target
 * mc_compute_doorbell_kernel run.  Caller gets a host-readable CPU
 * pointer + the matching GPU VA the SM thread will dereference.  The
 * scratch lives for the lifetime of the mc_ctx and is not
 * refcounted — used as a one-shot test cell.
 *
 * Backed by a dedicated 64-byte sysmem allocation (one RM hMemory,
 * one GPU MMU PTE) carved out of the carrier VAS during mc_init and
 * freed by mc_fini.  The returned pointer/VA target a dword in the
 * middle of the region, so callers can also write a few sentinel
 * dwords on either side of the target without touching unrelated
 * memory. */
mc_status_t mc_compute_get_scratch(mc_ctx_t *ctx, volatile uint32_t **cpu_ptr,
                                   uint64_t *gpu_va);

/* End-to-end-verifiable D2H copy whose UVM-channel doorbell is rung
 * by an SM thread of a HOPPER_COMPUTE_A kernel — the GPU compute
 * engine itself, with no copy engine in the loop.
 *
 * Identical contract and verification as mc_memcpy_d2h_gpu_doorbell_ce,
 * but the doorbell write is issued by the compute kernel's STG.E.SYS
 * to dbell_gpu_va + 0x90 instead of by a DMA-channel CE LAUNCH_DMA op.
 *
 * Sequence:
 *   1. Arm the UVM channel: build a HBM→DRAM CE pushbuffer at its
 *      pb_cpu, write the GPFIFO entry, advance USERD GPPut.  DO NOT
 *      ring the UVM channel's vf_doorbell.
 *   2. Launch mc_compute_doorbell_kernel(ctx, dbell_gpu_va + 0x90,
 *                                         uvm_work_submit_token).
 *      A single SM thread runs `*dst = token` — that store is a 4-byte
 *      MMIO write to BAR1's USERMODE_A VF_DOORBELL.
 *   3. Compute kernel's release semaphore fires (kernel exited; STG
 *      flushed through L2 to BAR1).
 *   4. Poll the UVM channel's sema (HBM→DRAM bytes flushed).
 *
 * Verification: dst_host contains the bytes from src_dev iff the UVM
 * channel's doorbell got rung.  The host did NOT ring it.  The only
 * remaining possible source of the ring is the SM thread's MMIO store.
 * Comparing dst_host vs. expected bytes is therefore proof — no copy
 * engine, no host doorbell, just a compute kernel writing to BAR1.
 *
 * Returns MC_OK on full chain success, MC_ETIMEOUT if either sema
 * doesn't fire within MC_TIMEOUT_MS. */
mc_status_t mc_memcpy_d2h_gpu_doorbell_sm(mc_ctx_t *ctx, void *dst_host,
                                          const void *src_dev, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_H */
