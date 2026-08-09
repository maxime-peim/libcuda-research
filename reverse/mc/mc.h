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
 * device handles and move data in/out via mc_memcpy.
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
    MC_EHANG      = 6,  /* compute kernel exited but the SM-authored CE
                         * submission on the victim channel did not commit
                         * (PBDMA wedged, malformed pushbuffer, etc.) */
} mc_status_t;

/* Which agent performs an mc_memcpy transfer.  Both arms produce
 * bit-identical results; they differ only in who authors the
 * submission to PBDMA.
 *
 *   MC_XFER_HOST — the host writes the channel's pushbuffer methods,
 *                  advances USERD GPPut, and rings the BAR1 doorbell.
 *                  This is the canonical CE-copy path; works for any
 *                  matching-VAS pointer pair.
 *   MC_XFER_SM   — a single SM thread of the sm_owner kernel writes
 *                  the entire submission (pushbuffer + GPFIFO entry +
 *                  USERD GPPut + BAR1 doorbell) from device code.
 *                  PBDMA on the victim DMA channel wakes and runs the
 *                  CE-authored copy.  Both pointers MUST live in a
 *                  non-UVM carrier VAS (either MC_VAS_SYSMEM_CARRIER
 *                  or MC_VAS_FB_CARRIER) — the sm_owner kernel only
 *                  addresses carrier VAS GPU VAs; else MC_EINVAL.
 *                  The FB-carrier variant keeps PB / GPFIFO / USERD in
 *                  HBM, so the SM's submission writes never cross PCIe;
 *                  the only MMIO is the BAR1 doorbell at the end.  The
 *                  release semaphore stays in sysmem either way — it is
 *                  the cell the host polls. */
typedef enum {
    MC_XFER_HOST = 0,
    MC_XFER_SM   = 1,
} mc_xfer_t;

/* Which VA space an allocation lives in.  Selected at mc_malloc_* time
 * and recoverable via the allocation table for any pointer the library
 * returned.
 *
 *   MC_VAS_UVM             — UVM-managed VAS, IS_EXTERNALLY_OWNED, the
 *                            only VAS reachable from the UVM CE channel
 *                            (mc_memcpy/_d2h's UVM path).  For host
 *                            allocs the CPU and GPU VAs are equal
 *                            (Paper F1 anchored into the VA pool).
 *   MC_VAS_SYSMEM_CARRIER  — Non-UVM FERMI_VASPACE_A wrapping a
 *                            per-resource NV50_MEMORY_VIRTUAL carriers,
 *                            channel resources (PB/GPFIFO/USERD/sema)
 *                            in sysmem.  Reachable from the carrier's
 *                            DMA + compute channels.  For host allocs
 *                            the CPU VA is kernel-chosen and NOT equal
 *                            to the GPU VA — use mc_gpu_va() to recover
 *                            the GPU VA when an external primitive
 *                            needs the raw GPU address (mc_memcpy
 *                            itself does this translation internally
 *                            via the alloc table).
 *   MC_VAS_FB_CARRIER      — Sibling carrier whose own channel
 *                            resources (PB/GPFIFO/USERD/sema) live in
 *                            FB vidmem with BAR1-aliased CPU mappings
 *                            for host bring-up + diagnostics.  Holds
 *                            its own BAR1 doorbell PTE and three
 *                            channels (HOST_DMA, SM_VICTIM_DMA,
 *                            COMPUTE).  Operational on H100 PCIe:
 *                            mc_memcpy(MC_XFER_SM) on this VAS
 *                            keeps every byte of the submission
 *                            protocol as FB↔GPU-L2 traffic with only
 *                            the doorbell ring as a PCIe MWr.  The
 *                            release semaphore stays sysmem-resident
 *                            even here — the victim DMA channel owns
 *                            a dedicated sysmem sema cell that the
 *                            host polls directly, because spinning on
 *                            an FB cell would make every poll a PCIe
 *                            read competing with the transfer it is
 *                            waiting on.  Same CPU/GPU-VA contract as
 *                            MC_VAS_SYSMEM_CARRIER for user buffers.
 *
 * This is the canonical VAS identifier; mc_internal.h uses it directly
 * (no internal alias type). */
typedef enum {
    MC_VAS_UVM             = 0,
    MC_VAS_SYSMEM_CARRIER  = 1,
    MC_VAS_FB_CARRIER      = 2,
} mc_vas_t;

/* Opens /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm; reserves the
 * VA pool; brings up the UVM, DMA, and compute channels.  Returns
 * MC_OK on success and stores an opaque context pointer in *out_ctx.
 * On failure returns a status code (details go to stderr) and *out_ctx
 * is left NULL. */
mc_status_t mc_init(mc_ctx_t **out_ctx);

/* Strict-LIFO teardown: drain channel, unregister from UVM, munmap, rm_free
 * all handles, close fds, free mc_ctx_t.  Safe against NULL. */
void mc_fini(mc_ctx_t *ctx);

/* Allocate HBM (vidmem) with NO CPU alias, mapped into `vas`.  Returned
 * pointer is a GPU VA cast to void*; do NOT dereference from the CPU.
 * Returns NULL on failure.
 *
 *   MC_VAS_UVM     — uvm_map_buffer with kernel-chosen GPU VA.
 *   MC_VAS_SYSMEM_CARRIER, MC_VAS_FB_CARRIER —
 *                    each per-resource NV50_MEMORY_VIRTUAL carrier
 *                    DMA-maps the source `hMemory` with
 *                    `dmaOffset = 0`; RM picks the GPU VA. */
void *mc_malloc_device(mc_ctx_t *ctx, size_t n, mc_vas_t vas);

/* Allocate host memory mapped into `vas`.  Returned pointer is the CPU
 * VA, usable for direct CPU dereference.  Returns NULL on failure.
 *
 *   MC_VAS_UVM     — Paper F1 anchored: returned CPU VA is also the GPU VA.
 *                    Pass it directly to mc_memcpy/_d2h.
 *   MC_VAS_SYSMEM_CARRIER, MC_VAS_FB_CARRIER —
 *                    kernel-chosen CPU VA (sysmem mapping; FB
 *                    carriers expose user host allocs through the
 *                    sysmem-style path too, with the FB-residency
 *                    only affecting the channel's own resources, not
 *                    user buffers).  GPU VA recoverable via
 *                    mc_gpu_va().  Pass the CPU VA to
 *                    mc_memcpy/_d2h (it dispatches by VAS via
 *                    alloc-table lookup); pass mc_gpu_va(p) when an
 *                    external primitive needs the raw GPU VA.
 *
 * Cacheability: host allocations are write-back cacheable, in every VA
 * space.  This is deliberate and it matters — the same buffer allocated
 * write-combined reads back at roughly 30 MB/s instead of ~11 GB/s,
 * because write-combined memory is uncached on the load path.  mc still
 * uses write-combining internally for its control plane (pushbuffers,
 * GPFIFO, USERD, semaphores), which the host writes and the GPU reads;
 * that is not what this function hands you.
 *
 * Coherency contract for carrier-VAS host allocs (both
 * MC_VAS_SYSMEM_CARRIER and MC_VAS_FB_CARRIER — user host allocs are
 * sysmem either way; only the FB carrier's *channel* resources live
 * in FB): CPU writes you
 * make before handing the buffer to mc_memcpy(..., MC_XFER_HOST)
 * will be ordered correctly by the library's own fence discipline.
 * The same holds for mc_memcpy(..., MC_XFER_SM): the caller's buffer is
 * only ever read by the Copy Engine, never by the SM kernel, so no
 * caller-side flushing is needed.  The library's own SFENCE in the
 * submit path makes the caller's stores visible before the doorbell,
 * and where the buffer is cached, CE reads of sysmem are cache-coherent
 * DMA on x86 (dirty lines are snooped).  The channel state the SM
 * kernel does read (pushbuffer, GPFIFO, USERD, semaphores, CB0) is
 * library-owned write-combined memory whose visibility the library's
 * own SFENCE discipline guarantees; callers never touch it. */
void *mc_malloc_host(mc_ctx_t *ctx, size_t n, mc_vas_t vas);

/* Register an existing malloc/mmap-backed host range into mc's
 * allocation table, analogous to cudaHostRegister(ptr, n, 0).
 *
 * The implementation currently supports MC_VAS_UVM only, matching the
 * cudaHostRegister trace shape:
 *   - RM creates an NV01_MEMORY_SYSTEM_OS_DESCRIPTOR over the page-covered
 *     CPU VA range.  mc does not allocate or mmap the memory.
 *   - UVM creates/maps external ranges at the CPU VAs, so GPU VA == CPU VA.
 *
 * After MC_OK, pass `ptr` to mc_memcpy just like a pointer returned by
 * mc_malloc_host.  The caller still owns the storage and must free(3) it only
 * after mc_host_unregister has returned. */
mc_status_t mc_host_register(mc_ctx_t *ctx, void *ptr, size_t n, mc_vas_t vas);

/* Unregister a pointer previously passed to mc_host_register.  No-op on NULL;
 * returns MC_EINVAL for unknown pointers or pointers owned by mc_malloc_*.
 * Does not free(3) or munmap(2) the user's storage. */
mc_status_t mc_host_unregister(mc_ctx_t *ctx, void *ptr);

/* Free a pointer returned by either mc_malloc_*.  No-op on NULL. */
void mc_free(mc_ctx_t *ctx, void *p);

/* Look up the GPU VA backing a pointer returned by mc_malloc_*.
 *
 * For MC_VAS_UVM host allocs the GPU VA equals the CPU VA (Paper F1)
 * so this is a no-op for them.  For carrier-VAS host allocs (both
 * MC_VAS_SYSMEM_CARRIER and MC_VAS_FB_CARRIER) the GPU VA differs
 * from the CPU VA returned by mc_malloc_host, and this function is
 * the supported way to recover it.  For device allocs (which return
 * their GPU VA cast to void*), the result equals the
 * argument cast to uint64_t.
 *
 * Returns 0 if `user_ptr` was not returned by mc_malloc_* on this
 * context (unknown pointer). */
uint64_t mc_gpu_va(mc_ctx_t *ctx, const void *user_ptr);

/* Synchronous copy.  Both pointers must have been returned by
 * mc_malloc_* on this context, with matching VAS — otherwise returns
 * MC_EINVAL.  Dispatch is automatic: UVM-VAS pointers route through
 * the UVM channel; CARRIER-VAS pointers route through the DMA
 * channel.  CPU pointers (host allocs) are translated to their GPU
 * VA via the allocation table before the method stream is built.
 *
 * `agent` selects who authors the submission to PBDMA.  See mc_xfer_t
 * for the trade-offs.  MC_XFER_SM additionally requires both pointers
 * in a carrier VAS (MC_VAS_SYSMEM_CARRIER or MC_VAS_FB_CARRIER). */
mc_status_t mc_memcpy(mc_ctx_t *ctx, const void *dst, const void *src,
                      size_t n, mc_xfer_t agent);

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

/* End-to-end-verifiable copy whose UVM-channel doorbell ring is
 * issued by the GPU itself, not the host.  Direction is inferred from
 * the alloc table: pass any matched (host, device) pair returned by
 * mc_malloc_* on this context.  Both pointers MUST live in
 * MC_VAS_UVM (the primary channel is the UVM CE channel) — otherwise
 * MC_EINVAL.
 *
 * Sequence (D2H phrasing; H2D swaps src/dst at the LAUNCH_DMA level
 * but the rest is identical):
 *   1. Arm the UVM channel: build a CE pushbuffer at its pb_cpu
 *      (src→dst via OFFSET_IN/OFFSET_OUT), write the GPFIFO entry,
 *      advance USERD GPPut.  DO NOT ring the UVM channel's
 *      vf_doorbell.
 *   2. Build a tiny pushbuffer on the DMA channel that copies the
 *      UVM channel's work_submit_token (pre-filled into a sysmem
 *      token cell mapped into the DMA channel's VAS) to
 *      dbell_gpu_va + 0x90 — i.e. the BAR1 USERMODE_A doorbell as
 *      seen by the DMA channel's GPU MMU.
 *   3. Ring the DMA channel's doorbell from the host.  PBDMA wakes
 *      the DMA channel, its CE writes 4 bytes to BAR1 doorbell,
 *      which (on real hardware) wakes the UVM channel's PBDMA,
 *      which runs the queued copy, which fires the UVM channel's
 *      release semaphore.
 *   4. Poll the DMA channel's sema (CE op finished) then the UVM
 *      channel's sema (bytes flushed through PCIe per
 *      LAUNCH_DMA_FLUSH_ENABLE=TRUE).
 *
 * Verification (D2H): the host buffer contains the bytes from the
 * device buffer iff the UVM channel's doorbell got rung.  The host
 * did not ring it.  Only possible source of the ring is the DMA
 * channel's CE-emitted MMIO write reaching real BAR1.  Comparing
 * the host buffer vs. expected bytes is therefore direct proof —
 * outside libcuda, outside the watchpoint shadow path, just CPU
 * loads from sysmem.
 *
 * Verification (H2D): the device buffer contains the bytes from the
 * host buffer iff the UVM channel's doorbell got rung.  The CPU
 * cannot read the device buffer directly, so the natural follow-up
 * is an *untimed* mc_memcpy(MC_XFER_HOST) read-back into a scratch
 * host buffer; the same chain-of-evidence argument then applies to
 * those readback bytes.
 *
 * Returns MC_OK on full chain success, MC_ETIMEOUT if either sema
 * doesn't fire within TIMEOUT_MS, MC_EINVAL on invalid inputs (NULL,
 * unknown pointers, mismatched VAS, non-UVM VAS, undersized buffer). */
mc_status_t mc_memcpy_gpu_doorbell_ce(mc_ctx_t *ctx, void *dst,
                                      const void *src, size_t n);

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
 *     pattern; see mc_memcpy_gpu_doorbell_sm.
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
 * compute channel's VAS — useful as the `dst` for a host-visible
 * mc_compute_doorbell_kernel run.  Caller gets a host-readable CPU
 * pointer + the matching GPU VA the SM thread will dereference.  The
 * scratch lives for the lifetime of the mc_ctx and is not
 * refcounted — used as a one-shot test cell.
 *
 * Backed by a dedicated BAR1-aliased vidmem allocation (one RM hMemory,
 * one GPU MMU PTE) mapped into the carrier VAS during mc_init and
 * freed by mc_fini.  The returned pointer/VA target a dword in the
 * middle of the region, so callers can also write a few sentinel
 * dwords on either side of the target without touching unrelated
 * memory. */
mc_status_t mc_compute_get_scratch(mc_ctx_t *ctx, volatile uint32_t **cpu_ptr,
                                   uint64_t *gpu_va);

/* End-to-end-verifiable copy whose UVM-channel doorbell is rung by
 * an SM thread of a HOPPER_COMPUTE_A kernel — the GPU compute engine
 * itself, with no copy engine in the loop.  Direction is inferred
 * from the alloc table: pass any matched (host, device) pair
 * returned by mc_malloc_* on this context.  Both pointers MUST live
 * in MC_VAS_UVM — otherwise MC_EINVAL.
 *
 * Identical contract and verification framing as
 * mc_memcpy_gpu_doorbell_ce, but the doorbell write is issued by the
 * compute kernel's STG.E.SYS to dbell_gpu_va + 0x90 instead of by a
 * DMA-channel CE LAUNCH_DMA op.
 *
 * Sequence:
 *   1. Arm the UVM channel: build a CE pushbuffer at its pb_cpu
 *      (src→dst), write the GPFIFO entry, advance USERD GPPut.  DO
 *      NOT ring the UVM channel's vf_doorbell.
 *   2. Launch mc_compute_doorbell_kernel(ctx, dbell_gpu_va + 0x90,
 *                                         uvm_work_submit_token).
 *      A single SM thread runs `*p = token` — that store is a 4-byte
 *      MMIO write to BAR1's USERMODE_A VF_DOORBELL.
 *   3. Compute kernel's release semaphore fires (kernel exited; STG
 *      flushed through L2 to BAR1).
 *   4. Poll the UVM channel's sema (bytes flushed).
 *
 * Verification (D2H): the host buffer contains the bytes from the
 * device buffer iff the UVM channel's doorbell got rung.  The host
 * did NOT ring it.  No CE was used to ring it either.  The only
 * remaining possible source of the ring is the SM thread's MMIO
 * store.  Comparing the host buffer vs. expected bytes is proof —
 * no copy engine, no host doorbell, just a compute kernel writing
 * to BAR1.
 *
 * Verification (H2D): same caveat as mc_memcpy_gpu_doorbell_ce — the
 * load lands in the device buffer; verify by an *untimed*
 * mc_memcpy(MC_XFER_HOST) read-back into a scratch host buffer.
 *
 * Returns MC_OK on full chain success, MC_ETIMEOUT if either sema
 * doesn't fire within MC_TIMEOUT_MS, MC_EINVAL on invalid inputs. */
mc_status_t mc_memcpy_gpu_doorbell_sm(mc_ctx_t *ctx, void *dst,
                                      const void *src, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_H */
