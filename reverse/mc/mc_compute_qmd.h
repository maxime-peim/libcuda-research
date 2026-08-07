/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_compute_qmd.h — Hopper QMD V04 builder for mc's
 * compute-doorbell-write kernel.
 *
 * Built from zero, NVK-style.  See Mesa NVK
 * `src/nouveau/compiler/nak/qmd.rs` (Qmd4_0 + fill_qmd) for the
 * ground-truth recipe.  The previous implementation captured a
 * libcuda QMD verbatim and patched it; that approach left
 * libcuda-process-specific pointer fragments in the QMD that SKED
 * dereferenced as VA 0 (Xid 31).  This rewrite starts from a
 * zero-initialised 384-byte buffer and only sets the fields NVK
 * proves are required.
 *
 * Field positions come from Mesa NVK's `clcbc0qmd.h`; this tree ships
 * only the Kepler `cla0c0qmd.h`, so the NVCBC0_QMDV04_00_* macro names
 * are kept purely so the two can be read side by side.
 *
 * Usage (single-thread (1,1,1) grid running our doorbell-write SASS):
 *
 *   uint8_t *qmd  = vidmem_alloc(MC_QMD_BYTES, 0x100);
 *   uint8_t *cb0  = vidmem_alloc(MC_CB0_TOTAL_BYTES_ALIGNED, 0x100);
 *   uint8_t *sass = vidmem_alloc(roundup(mc_doorbell_kernel_sass_len, 256), 0x100);
 *   memcpy(sass, mc_doorbell_kernel_sass, mc_doorbell_kernel_sass_len);
 *
 *   mc_qmd_init(qmd);
 *   mc_qmd_set_global_size(qmd, 1, 1, 1);
 *   mc_qmd_set_local_size (qmd, 1, 1, 1);
 *   mc_qmd_set_prog_addr  (qmd, sass_gpu_va);
 *   mc_qmd_set_register_count(qmd, 8);
 *   mc_qmd_set_barrier_count (qmd, 0);
 *   mc_qmd_set_slm_size      (qmd, 0);
 *   mc_qmd_set_smem_size     (qmd, 0);
 *   mc_qmd_set_cbuf(qmd, 0, cb0_gpu_va, MC_CB0_TOTAL_BYTES_ALIGNED);
 *
 *   mc_cb0_init(cb0);
 *   mc_cb0_set_args(cb0, dst_gpu_va, token);
 *
 *   // Then issue the launch pushbuffer: SET_OBJECT(HOPPER_COMPUTE_A) +
 *   // a few cache-invalidate / window methods (one-time per channel) +
 *   // SEND_PCAS_A(qmd_gpu_va >> 8) + SEND_SIGNALING_PCAS2_B(0xa).
 */
#ifndef MC_COMPUTE_QMD_H
#define MC_COMPUTE_QMD_H

#include <stdint.h>

/* Hopper QMD V04 fixed size (96 dwords). */
#define MC_QMD_BYTES                 384u

/* CB0 carries kernel arguments at SASS-defined offsets:
 *   c[0x0][0x208] = global memdesc (must read 0)
 *   c[0x0][0x210] = dst_gpu_va low 32 bits
 *   c[0x0][0x214] = dst_gpu_va high 32 bits
 *   c[0x0][0x218] = token
 * The SM front-end fetches the whole CB0 once into local memory,
 * so any bytes past 0x21c are unread but the buffer must be at
 * least 0x21c bytes.  We round up to 0x100 = 256 (NVK's
 * min_cbuf_alignment for Hopper is 0x100 on this class). */
#define MC_CB0_TOTAL_BYTES           0x21Cu
#define MC_CB0_TOTAL_BYTES_ALIGNED   ((MC_CB0_TOTAL_BYTES + 0xFFu) & ~0xFFu)

/* User-arg offsets inside CB0 (the SASS reads via LDC.64 / LDC). */
#define MC_CB0_OFF_USER_DST_LO       0x210u
#define MC_CB0_OFF_USER_DST_HI       0x214u
#define MC_CB0_OFF_USER_TOKEN        0x218u

/* Initialize a QMD — memset 0 then set NVK's qmd_init defaults
 * (QMD_MAJOR_VERSION=4, QMD_MINOR_VERSION=0,
 *  API_VISIBLE_CALL_LIMIT=NO_CHECK,
 *  SAMPLER_INDEX=INDEPENDENTLY,
 *  QMD_TYPE=GRID_CTA).
 * The buffer must be MC_QMD_BYTES bytes, 256-byte-aligned. */
void mc_qmd_init(uint8_t *qmd);

/* QMD field setters mirroring NVK's Qmd4_0 trait (one per fill_qmd
 * call).  All of these write into bit-ranges defined by the
 * NVCBC0_QMDV04_00_* macros and sit inside the 384-byte QMD. */
void mc_qmd_set_barrier_count(uint8_t *qmd, uint8_t n);
void mc_qmd_set_global_size(uint8_t *qmd, uint32_t w, uint32_t h, uint32_t d);
void mc_qmd_set_local_size (uint8_t *qmd, uint16_t w, uint16_t h, uint16_t d);
void mc_qmd_set_prog_addr   (uint8_t *qmd, uint64_t prog_gpu_va);
void mc_qmd_set_register_count(uint8_t *qmd, uint8_t n);
void mc_qmd_set_slm_size    (uint8_t *qmd, uint32_t bytes);
void mc_qmd_set_smem_size   (uint8_t *qmd, uint32_t bytes);

/* Patch QMD V04's CONSTANT_BUFFER(idx) — bind a CB GPU VA + size into
 * the QMD so the SM front-end can fetch `c[0x0][...]` referenced by
 * the kernel's SASS.  Same encoding as NVK's qmd_impl_set_cbuf!
 * (SHIFTED6 for addr, SHIFTED4 for size).
 *
 * Constraints: addr must be 64-byte-aligned (>>6 form), size must be
 * 16-byte-aligned (>>4 form).  idx must be < 8 (Hopper has CB0..CB7). */
void mc_qmd_set_cbuf(uint8_t *qmd, uint8_t idx, uint64_t addr, uint32_t size);

/* CB0 builders.  CB0 is zero-init'd and only the three argument
 * dwords (c[0x0][0x210/0x214/0x218]) are written — the SASS reads
 * the global memdesc base from c[0x0][0x208] (must be 0 for raw-VA
 * mode) and the dst pointer + token from the arg dwords. */
void mc_cb0_init(uint8_t *cb0);
void mc_cb0_set_args(uint8_t *cb0, uint64_t dst_gpu_va, uint32_t token);

/* Embedded SASS for mc_doorbell_kernel: 256 bytes, 6 effective
 * instructions (LDC, LDC, ULDC, LDC, STG, EXIT) + BRA self + NOPs.
 * REG=8, no LMEM/SMEM/STACK.  Source kernel:
 *   extern "C" __global__
 *   void mc_doorbell_kernel(volatile unsigned int *dst, unsigned int token)
 *   {  *dst = token;  }
 * Compiled with `nvcc -arch=sm_90 --cubin -O0`, .text section extracted. */
extern const uint8_t  mc_doorbell_kernel_sass[];
extern const uint32_t mc_doorbell_kernel_sass_len;

#endif /* MC_COMPUTE_QMD_H */
