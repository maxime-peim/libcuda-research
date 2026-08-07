/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_sm_owner_args.h — shared layout for the sm_owner_kernel argument
 * struct.  Included by both the device-side kernel (sm_owner.cu) and
 * the host-side glue (mc_sm_owner.c).
 *
 * Why a struct instead of 11 individual kernel parameters: NVCC's
 * `extern "C" __global__` parameter packing is an implementation
 * detail (start-at-CB0[0x210], natural-alignment with no padding
 * between back-to-back u32s on observed CUDA 13.x).  It is NOT an
 * ABI guarantee across CUDA toolkit versions.  Passing a single
 * `struct sm_owner_args *` lets the kernel read fields by C struct
 * offset (which IS an ABI: the C standard fixes natural alignment),
 * so future CUDA toolkits can shift the parameter base or repack
 * primitive parameters without breaking us.
 *
 * Layout discipline:
 *   - All u64 fields first (8-byte aligned trivially).
 *   - u32 fields packed at the end with explicit padding to keep
 *     the total size a multiple of 8 (avoid surprises if we ever
 *     embed this struct inside a larger one).
 *   - Field order is irrelevant to correctness; chosen for
 *     reading-order convenience (resources first, then workload,
 *     then control bookkeeping).
 *
 * Caller (mc_sm_owner.c) allocates an instance of this struct in
 * carrier sysmem and passes its GPU VA to the kernel via a single
 * pointer parameter.  The kernel does LDG.E.64 / LDG.E for each
 * field through that pointer.
 */
#ifndef MC_SM_OWNER_ARGS_H
#define MC_SM_OWNER_ARGS_H

#include <stdint.h>

/* Number of method dwords the SM-owner kernel emits into the victim
 * channel's pushbuffer per submission.  The NVC8B5 method stream is
 * 16 dwords:
 *   SET_OBJECT (2) + OFFSET_IN_UPPER (3) + OFFSET_OUT_UPPER (3)
 *   + LINE_LENGTH_IN (2) + SET_SEMAPHORE_A triplet (4) + LAUNCH_DMA (2)
 * Single LAUNCH_DMA with RELEASE_ONE_WORD_SEMAPHORE.  Ordering
 * guarantee comes from PCIe producer-side ordering of MWrs from one
 * Requester ID into one address space (sysmem): the active sema cell
 * is always sysmem-resident for any FB-carrier MC_XFER_SM transfer
 * that touches sysmem on either side, and the host polls it.  See
 * mc_sm_owner.c::mc_sm_owner_submit for the routing rule.
 * The kernel uses this constant to size the GPFIFO entry's LENGTH
 * field; the host post-timeout diagnostic uses it to reconstruct the
 * GPFIFO entry it expected to see.  Both sides MUST agree. */
#define MC_SM_OWNER_PB_METHOD_DWORDS  16u

struct mc_sm_owner_args {
    /* Channel resource VAs (carrier-VAS GPU addresses). */
    uint64_t pb_gpu_va;          /* PB methods land here */
    uint64_t gpfifo_gpu_va;      /* GPFIFO ring base */
    uint64_t userd_gpu_va;       /* HopperAControlGPFifo struct (already +slot offset) */
    uint64_t dbell_gpu_va;       /* HOPPER_USERMODE_A page base; kernel adds +0x90 */

    /* Workload payload (the CE op the SM authors). */
    uint64_t src_gpu_va;
    uint64_t dst_gpu_va;
    uint64_t sema_gpu_va;        /* CE-op release semaphore — sysmem-resident
                                  * for both H2D and D2H; host polls the cell
                                  * after the compute report sema fires. */

    /* Optional on-GPU DMA-sema poll — kept as a hook for a future
     * HBM-resident sema path.  All callers today set
     * dma_sema_poll_va = 0, so the SM kernel skips the poll loop and
     * the host's poll on the sysmem sema cell is the sole completion
     * gate.  If a future caller sets dma_sema_poll_va non-zero, the
     * SM kernel busy-waits on *(volatile uint32_t *)dma_sema_poll_va
     * until it observes dma_sema_poll_expected (or the iteration cap
     * dma_sema_poll_budget expires). */
    uint64_t dma_sema_poll_va;

    /* u32 tail.  Pairs of u32s share 8-byte slots; layout chosen so
     * the total is a multiple of 8 with no implicit tail padding. */
    uint32_t size_bytes;
    uint32_t sema_payload;

    uint32_t work_submit_token;  /* victim DMA channel's RM-issued token */
    uint32_t gp_put_in;          /* current GPPut at launch time */

    uint32_t dma_sema_poll_expected;  /* zeroed when dma_sema_poll_va == 0 */
    uint32_t dma_sema_poll_budget;    /* iteration cap; 0 disables the loop */
};

#endif /* MC_SM_OWNER_ARGS_H */
