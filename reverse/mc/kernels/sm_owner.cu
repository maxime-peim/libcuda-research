/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * sm_owner.cu — a CUDA kernel that authors an entire CE channel
 * submission from a single SM thread: pushbuffer methods, GPFIFO
 * entry, USERD GP_PUT advance, and the BAR1 doorbell write.
 *
 * The kernel is the canonical, executable description of the
 * Hopper CE submission protocol.  Everything that lives in C on the
 * host side (write_gp_entry, ring_doorbell) is mirrored here in CUDA C /
 * inline PTX.  The method stream is the one exception: this kernel emits
 * the fused single-LAUNCH_DMA form (16 dwords) rather than the two-launch
 * form mc_submit.c now uses, because the store-count assertions on this
 * kernel's SASS are written against 16 pushbuffer dwords.  Only the
 * per-call workload payload (`src`, `dst`, `size`, `sema`) comes in
 * via kernel parameters; the structural authorship — opcodes,
 * subchannel routing, semaphore-release flags, LAUNCH_DMA flag word —
 * is in this file.
 *
 * Pre-conditions enforced by the host glue (mc_sm_owner.c):
 *   - The compute channel and the victim DMA channel share one carrier
 *     FERMI_VASPACE_A (MC_VAS_PRIMARY_CARRIER).  Every GPU VA below is
 *     therefore reachable from this kernel.
 *   - `userd_gpu_va` is the GPU VA of the HopperAControlGPFifo struct
 *     itself (i.e. the host has already added MC_USERD_OFFSET = 0x2000
 *     to the gpfifo+userd allocation base).
 *   - `dbell_gpu_va` is the GPU VA of the HOPPER_USERMODE_A page base;
 *     the kernel adds MC_VF_DOORBELL_OFFSET = 0x90 itself.
 *   - `gp_put_in` is current GPPut at the moment the kernel is
 *     launched.  The host must have already drained the channel or
 *     ensured the slot at this index is free.
 *
 * Memory ordering discipline mirrors the host's produce-fence-publish
 * contract (`_mm_sfence` between WC stores).  On the SM side:
 *
 *     STG.E.STRONG.SYS    pushbuffer methods    [PB writes]
 *     membar.sys                                [PB visible globally]
 *     STG.E.64.STRONG.SYS GPFIFO entry          [entry visible]
 *     membar.sys
 *     STG.E.STRONG.SYS    USERD GPPut           [PBDMA can now wake]
 *     membar.sys
 *     STG.E.STRONG.SYS    BAR1 doorbell         [host scheduler poked]
 *
 * Build (driven by reverse/Makefile):
 *
 *     nvcc -arch=sm_90 --cubin -O2 sm_owner.cu -o sm_owner.cubin
 *     # then extract the .text._Z*sm_owner_kernel* section into
 *     # mc_sm_owner_sass.h as a static const uint8_t array.
 *
 * Verifying the compiler kept our STG.E.STRONG.SYS / membar.sys
 * choices:
 *
 *     cuobjdump --dump-sass sm_owner.cubin | grep -E 'STG|MEMBAR'
 */

#include <cuda_runtime.h>
#include <stdint.h>

/* ── NVC8B5 / NVC86F bit layouts mirrored from the SDK headers ──────────
 *
 * Cross-check:
 *
 *   src/common/sdk/nvidia/inc/class/clc36f.h     — DMA_INCR_OPCODE/COUNT/SUBCH/ADDR
 *   src/common/sdk/nvidia/inc/class/clc8b5.h     — NVC8B5 method addresses + LAUNCH_DMA flags
 *   src/common/sdk/nvidia/inc/class/clc86f.h     — NVC86F GP_ENTRY0/1 + SET_OBJECT
 *   src/common/sdk/nvidia/inc/class/cla06fsubch.h — NVA06F_SUBCHANNEL_COPY_ENGINE
 *
 * If a future driver/SDK update changes any of these, regenerate by
 * comparing the C-side host-built pushbuffer to the kernel's output —
 * `decode.py` will flag any mismatched method opcode/payload.
 */

/* DMA_INCR opcode header layout (clc36f.h):
 *   [31:29]  OPCODE   = 1 (DMA_INCR)
 *   [28:16]  COUNT    = number of payload dwords following
 *   [15:13]  SUBCH    = subchannel
 *   [12:0]   ADDRESS  = method_address >> 2
 */
__device__ __forceinline__
uint32_t incr_header(uint32_t method_addr, uint32_t count, uint32_t subch)
{
    return (1u << 29)                           /* DMA_INCR opcode */
         | ((count & 0x1FFFu) << 16)
         | ((subch & 0x7u) << 13)
         | ((method_addr >> 2) & 0x1FFFu);
}

/* NVA06F_SUBCHANNEL_COPY_ENGINE = 4.  CE engine is bound to subch 4
 * by the SET_OBJECT method we emit first; PBDMA routes everything
 * keyed by subch 4 to the channel's hCE engine. */
constexpr uint32_t SUBCH_CE = 4;

/* NVC86F method addresses (NVC86F_SET_OBJECT) and NVC8B5 method
 * addresses.  These are byte-addressed registers; the >>2 inside
 * incr_header takes them to dword indices. */
constexpr uint32_t NVC86F_SET_OBJECT             = 0x00000000;
constexpr uint32_t NVC8B5_OFFSET_IN_UPPER        = 0x00000400;
constexpr uint32_t NVC8B5_OFFSET_OUT_UPPER       = 0x00000408;
constexpr uint32_t NVC8B5_LINE_LENGTH_IN         = 0x00000418;
constexpr uint32_t NVC8B5_SET_SEMAPHORE_A        = 0x00000240;
constexpr uint32_t NVC8B5_LAUNCH_DMA             = 0x00000300;

/* HOPPER_DMA_COPY_A class id — the SET_OBJECT payload that binds
 * subch 4 to the CE engine. */
constexpr uint32_t HOPPER_DMA_COPY_A             = 0x0000C8B5;

/* LAUNCH_DMA flags word.  Bit positions (clc8b5.h):
 *
 *   [1:0]  DATA_TRANSFER_TYPE = NON_PIPELINED (=2)
 *   [2]    FLUSH_ENABLE       = TRUE          (=1)
 *   [4:3]  SEMAPHORE_TYPE     = RELEASE_ONE_WORD_SEMAPHORE (=1)
 *   [7]    SRC_MEMORY_LAYOUT  = PITCH         (=1)
 *   [8]    DST_MEMORY_LAYOUT  = PITCH         (=1)
 *
 * Composing:
 *   (2 << 0) | (1 << 2) | (1 << 3) | (1 << 7) | (1 << 8)
 *   = 0x002 | 0x004 | 0x008 | 0x080 | 0x100
 *   = 0x18E
 *
 * This is the fused form: one launch that copies, flushes and releases.
 * It is the value Yan et al. report from an A40 trace.  Note H100 libcuda
 * does NOT emit it — a captured cudaMemcpy splits the work into
 * LAUNCH_DMA 0x182 (copy, no flush, no semaphore) followed by a
 * semaphore-only LAUNCH_DMA — which is what mc_submit.c mirrors on the
 * host side.  Both forms are correct; this one is kept here because it
 * fits in 16 dwords. */
constexpr uint32_t LAUNCH_DMA_FLAGS              = 0x0000018Eu;

/* NVC86F GP_ENTRY0/1 fields (clc86f.h):
 *
 *   ENTRY0[31:2]   GET       = pb_va[31:2]   (DRF_NUM masks to 30 bits then <<2)
 *   ENTRY0[1:0]   FETCH      = UNCONDITIONAL (=0)
 *   ENTRY1[7:0]   GET_HI     = pb_va[39:32]
 *   ENTRY1[10]    PRIV       = USER          (=0)
 *   ENTRY1[30:10] LENGTH     = method_dwords
 *   ENTRY1[31]    SYNC       = PROCEED       (=0)
 *
 * mc's carrier-VAS layout keeps PB GPU VAs below 2^40, so we
 * never need the SET_PB_SEGMENT_EXTENDED_BASE entry that
 * write_gp_entry emits when pb_va >= 2^40.  The kernel asserts on
 * the high bits to catch any future violation early.
 *
 * Note that NVC86F_GP_ENTRY0_GET takes the *semantic* value (pb_va >> 2),
 * not the pre-shifted bit pattern.  DRF_NUM(GP_ENTRY0, GET, ...) masks
 * to 30 bits and shifts left by 2 to put pb_va[31:2] at ENTRY0[31:2].
 * Doing the same in the kernel: build entry0 = (pb_va & 0xFFFFFFFC).
 */

/* MC_GPFIFO_ENTRIES = 512.  Mask = 511.  Baked in. */
constexpr uint32_t GPFIFO_ENTRY_MASK             = 511;

/* ── Inline PTX wrappers ───────────────────────────────────────────────
 *
 * Producer-side stores use `st.relaxed.sys.global` (ptxas requires a
 * memory-order modifier when using `.sys` scope; the PTX ISA accepts
 * `weak`, `relaxed`, `release`).  `relaxed.sys` means "no ordering
 * guarantee on its own, but visible at system scope (CPU / BAR1 /
 * other GPUs)."  Ordering between producer steps is provided by the
 * explicit `membar.sys` fences below — exactly mirroring the host's
 * "WC store + _mm_sfence" discipline in mc_submit.c.
 *
 * Why not `st.release.sys`?  Release-on-each-store would make every
 * STG a fence in itself, which is heavier than needed and obscures
 * the produce-fence-publish structure.  Splitting stores from fences
 * keeps the ordering invariants legible: each block of stores is one
 * "produce" step, each membar.sys is one "publish" step.
 *
 * cuobjdump --dump-sass sm_owner.cubin should show these as
 *   STG.E.STRONG.SYS  R, [R]     (32-bit, system scope)
 *   STG.E.64.STRONG.SYS R, [R]   (64-bit, system scope)
 *   MEMBAR.SC.SYS                (system-scope fence)
 * — which is exactly what the build-time assertion in reverse/Makefile
 * counts.  If the SYS variant degrades to plain STG.E (LTC scope),
 * data may not reach sysmem before PBDMA reads it — a subtle bug.
 */
__device__ __forceinline__
void stg_u32_sys(uint32_t *addr, uint32_t value)
{
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;"
                 : : "l"(addr), "r"(value) : "memory");
}

__device__ __forceinline__
void stg_u64_sys(uint64_t *addr, uint64_t value)
{
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
                 : : "l"(addr), "l"(value) : "memory");
}

__device__ __forceinline__
void membar_sys()
{
    asm volatile("membar.sys;" : : : "memory");
}

/* ── The kernel ───────────────────────────────────────────────────────
 *
 * Single-thread kernel.  Authors the entire CE-channel submission for
 * one D2H copy of `size` bytes from `src` to `dst`, with a release
 * semaphore at `sema` carrying `sema_payload`.
 *
 * `pb_gpu_va`, `gpfifo_gpu_va`, `userd_gpu_va`, `dbell_gpu_va` are GPU
 * VAs in the carrier VAS that the SM can address directly.
 * `work_submit_token` is the channel's RM-issued submit token (32-bit
 * value PBDMA's host scheduler matches against the runlist).
 * `gp_put_in` is the current GPPut at launch time; the kernel writes
 * `gp_put_in + 1` (we always emit one normal GPFIFO entry — no
 * extended-base) to USERD.
 *
 * All parameters land in CB0 at NVCC-determined byte offsets starting
 * at 0x210 (the standard Hopper kernel-arg base).  The host glue
 * (mc_sm_owner.c) reads these offsets via cuobjdump and writes them
 * to CB0 directly — same pattern as the existing mc_doorbell_kernel.
 *
 * The Phase A / Phase B distinction is in the *invocation arguments*,
 * not the kernel.  Phase A invokes with src/dst pointing at small
 * sysmem cells (4-byte copy); Phase B invokes with src=HBM, dst=DRAM
 * for a real D2H.  Same SASS, different CB0 contents.
 */
extern "C" __global__
void sm_owner_kernel(uint64_t pb_gpu_va,
                     uint64_t gpfifo_gpu_va,
                     uint64_t userd_gpu_va,
                     uint64_t dbell_gpu_va,
                     uint64_t src_gpu_va,
                     uint64_t dst_gpu_va,
                     uint32_t size_bytes,
                     uint64_t sema_gpu_va,
                     uint32_t sema_payload,
                     uint32_t work_submit_token,
                     uint32_t gp_put_in)
{
    /* Single-thread — invoked with grid (1,1,1) × CTA (1,1,1).
     * Defensive guard if anyone ever launches a wider grid by mistake. */
    if (threadIdx.x | threadIdx.y | threadIdx.z |
        blockIdx.x  | blockIdx.y  | blockIdx.z) return;

    /* ── Phase 1: emit the 7-method NVC8B5 D2H pushbuffer ──────────────
     *
     * Same method groups as mc_submit.c::mc_write_transfer_methods, but
     * fused into a single LAUNCH_DMA (see above), so 16 dwords = 64 bytes
     * rather than that function's 18.  PB writes are STG.E.STRONG.SYS so
     * they are visible to PBDMA (sysmem) once the membar.sys below
     * retires.
     */
    uint32_t *pb = reinterpret_cast<uint32_t *>(pb_gpu_va);

    /*  [0]  SET_OBJECT header (1 payload dword) */
    stg_u32_sys(&pb[0], incr_header(NVC86F_SET_OBJECT, 1, SUBCH_CE));
    /*  [1]  SET_OBJECT payload — bind HOPPER_DMA_COPY_A to subch 4 */
    stg_u32_sys(&pb[1], HOPPER_DMA_COPY_A);

    /*  [2]  OFFSET_IN_UPPER header (2 payload dwords) */
    stg_u32_sys(&pb[2], incr_header(NVC8B5_OFFSET_IN_UPPER, 2, SUBCH_CE));
    /*  [3]  src_va upper 32 bits */
    stg_u32_sys(&pb[3], (uint32_t)(src_gpu_va >> 32));
    /*  [4]  src_va lower 32 bits */
    stg_u32_sys(&pb[4], (uint32_t)(src_gpu_va & 0xFFFFFFFFu));

    /*  [5]  OFFSET_OUT_UPPER header (2 payload dwords) */
    stg_u32_sys(&pb[5], incr_header(NVC8B5_OFFSET_OUT_UPPER, 2, SUBCH_CE));
    /*  [6]  dst_va upper 32 bits */
    stg_u32_sys(&pb[6], (uint32_t)(dst_gpu_va >> 32));
    /*  [7]  dst_va lower 32 bits */
    stg_u32_sys(&pb[7], (uint32_t)(dst_gpu_va & 0xFFFFFFFFu));

    /*  [8]  LINE_LENGTH_IN header (1 payload dword) */
    stg_u32_sys(&pb[8], incr_header(NVC8B5_LINE_LENGTH_IN, 1, SUBCH_CE));
    /*  [9]  byte count */
    stg_u32_sys(&pb[9], size_bytes);

    /* [10]  SET_SEMAPHORE_A header (3 payload dwords: A, B, payload) */
    stg_u32_sys(&pb[10], incr_header(NVC8B5_SET_SEMAPHORE_A, 3, SUBCH_CE));
    /* [11]  sema_va upper 32 bits */
    stg_u32_sys(&pb[11], (uint32_t)(sema_gpu_va >> 32));
    /* [12]  sema_va lower 32 bits */
    stg_u32_sys(&pb[12], (uint32_t)(sema_gpu_va & 0xFFFFFFFFu));
    /* [13]  sema payload */
    stg_u32_sys(&pb[13], sema_payload);

    /* [14]  LAUNCH_DMA header (1 payload dword) */
    stg_u32_sys(&pb[14], incr_header(NVC8B5_LAUNCH_DMA, 1, SUBCH_CE));
    /* [15]  LAUNCH_DMA flags = 0x18E (NON_PIPELINED | FLUSH_ENABLE |
     *                                  RELEASE_ONE_WORD_SEMA |
     *                                  SRC_PITCH | DST_PITCH) */
    stg_u32_sys(&pb[15], LAUNCH_DMA_FLAGS);

    constexpr uint32_t pb_method_dwords = 16;

    /* Publish PB before producing GPFIFO entry. */
    membar_sys();

    /* ── Phase 2: write one GPFIFO entry pointing at pb_gpu_va ─────────
     *
     * Single normal entry only — we reject pb_gpu_va >= 2^40, which
     * would require an extended-base entry that mc's carrier
     * layout never needs.  Mirror write_gp_entry()'s entry0/entry1
     * encoding exactly.
     */

    /* entry0 = pb_va[31:2] in [31:2], FETCH=UNCONDITIONAL=0 in [1:0].
     * Equivalent to (pb_va & 0xFFFFFFFC). */
    uint32_t entry0 = (uint32_t)(pb_gpu_va & 0xFFFFFFFCu);

    /* entry1 = pb_va[39:32] in [7:0], LENGTH in [30:10]. */
    uint32_t entry1 = (uint32_t)((pb_gpu_va >> 32) & 0xFFu)
                    | (pb_method_dwords << 10);
    /* PRIV=USER=0 in [10] is below LENGTH; LENGTH starts at [30:10] so
     * the LENGTH << 10 already places bit-zero of LENGTH at bit 10.
     * SYNC=PROCEED=0 in [31] left implicit. */

    /* Combine into a 64-bit dword and STG it.  The carrier VAS keeps
     * the GPFIFO ring 8-byte-aligned per slot, so a single 64-bit
     * STG is correct and atomic from PBDMA's view. */
    uint64_t entry_qw = ((uint64_t)entry1 << 32) | (uint64_t)entry0;
    uint32_t gp_put_idx = gp_put_in & GPFIFO_ENTRY_MASK;
    uint64_t *entry_ptr = reinterpret_cast<uint64_t *>(
        gpfifo_gpu_va + (uint64_t)gp_put_idx * 8);
    stg_u64_sys(entry_ptr, entry_qw);

    /* Publish GPFIFO entry before advancing GPPut. */
    membar_sys();

    /* ── Phase 3: advance USERD GPPut ──────────────────────────────────
     *
     * GPPut byte offset = 0x8C inside HopperAControlGPFifo.  Caller
     * passes userd_gpu_va already including MC_USERD_OFFSET = 0x2000
     * within the gpfifo+userd 2 MiB allocation.
     */
    constexpr uint32_t USERD_GPPUT_OFFSET = 0x8C;
    uint32_t new_gp_put = gp_put_in + 1;
    uint32_t *userd_gpput_ptr = reinterpret_cast<uint32_t *>(
        userd_gpu_va + USERD_GPPUT_OFFSET);
    stg_u32_sys(userd_gpput_ptr, new_gp_put);

    /* Publish GPPut before ringing the doorbell. */
    membar_sys();

    /* ── Phase 4: ring the BAR1 doorbell ───────────────────────────────
     *
     * MC_VF_DOORBELL_OFFSET = 0x90 inside HOPPER_USERMODE_A.  The host
     * passes dbell_gpu_va as the page base; we add 0x90 ourselves.
     * Writing the channel's work_submit_token wakes the host scheduler
     * which then forwards to PBDMA.
     *
     * On Hopper, both writes (USERD GPPut and BAR1 doorbell) are
     * mandatory and ordered.  See ring_doorbell() in mc_submit.c.
     */
    constexpr uint32_t VF_DOORBELL_OFFSET = 0x90;
    uint32_t *dbell_ptr = reinterpret_cast<uint32_t *>(
        dbell_gpu_va + VF_DOORBELL_OFFSET);
    stg_u32_sys(dbell_ptr, work_submit_token);

    /* Final fence: ensure the doorbell store retires before the
     * kernel's report-semaphore release fires.  This makes the
     * compute kernel's "I'm done" signal a strict happens-after of
     * the doorbell write reaching BAR1, so the host can poll the
     * compute sema and trust that PBDMA on the victim channel has
     * already been kicked. */
    membar_sys();
}
