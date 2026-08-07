/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_compute_qmd.c — NVK-style Hopper QMD V04 builder.
 *
 * Built from zero, mirroring Mesa NVK's `Qmd4_0` (qmd.rs:543-580) +
 * `fill_qmd` (qmd.rs:627-666).  See mc_compute_qmd.h for rationale.
 *
 * Bit-range references are named after the NVCBC0_QMDV04_00_* macros in
 * NVIDIA's clcbc0qmd.h.  That header does NOT ship in the open kernel
 * modules tree — only cla0c0qmd.h (Kepler) does — so the V04 ranges here
 * follow the layout Mesa NVK publishes, and the macro names are kept only
 * so the two can be read side by side.  All MW(hi:lo) ranges are flat bit
 * positions over the 384-byte QMD (LSB = bit 0 of byte 0).
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "mc_compute_qmd.h"

/* ── Bit-range write helper ─────────────────────────────────────────
 * Writes `value` into the QMD at bits [hi:lo] inclusive (MW(hi:lo)
 * semantics).  Handles ranges that span up to two 32-bit dwords —
 * NVK's bitview does the same via `BitMutView::set_field`.
 *
 * The QMD is treated as an array of little-endian 32-bit dwords —
 * the SDK macros and Mesa NVK both view it this way.
 */
static void qmd_set_bits(uint8_t *qmd, unsigned hi, unsigned lo, uint64_t value)
{
    /* clcbc0qmd.h ranges are all <= 32 bits; assert defensively. */
    unsigned width = hi - lo + 1;
    assert(width <= 32);

    uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1ULL);
    value &= mask;

    uint32_t *qw = (uint32_t *)qmd;

    unsigned dw_lo  = lo / 32;
    unsigned bit_lo = lo % 32;
    unsigned bit_hi = hi - dw_lo * 32;        /* relative to dw_lo */

    if (bit_hi < 32) {
        /* Range fits in one dword. */
        uint32_t dw_mask = (uint32_t)(mask << bit_lo);
        qw[dw_lo] = (qw[dw_lo] & ~dw_mask) | (uint32_t)((value & mask) << bit_lo);
    } else {
        /* Range straddles dw_lo and dw_lo+1. */
        unsigned lo_bits = 32 - bit_lo;        /* bits going into dw_lo */
        uint32_t lo_mask = (uint32_t)(((1ULL << lo_bits) - 1ULL) << bit_lo);
        uint32_t hi_mask = (uint32_t)((1ULL << (width - lo_bits)) - 1ULL);

        qw[dw_lo] = (qw[dw_lo] & ~lo_mask)
                    | (uint32_t)((value & ((1ULL << lo_bits) - 1ULL)) << bit_lo);
        qw[dw_lo + 1] = (qw[dw_lo + 1] & ~hi_mask)
                    | (uint32_t)((value >> lo_bits) & hi_mask);
    }
}

/* ── QMD V04 field bit positions (from clcbc0qmd.h, NVCBC0_QMDV04_00_*) ─ */
/* QMD_TYPE                              MW(25:23)                       */
#define QMD_TYPE_HI               25u
#define QMD_TYPE_LO               23u
#define QMD_TYPE_GRID_CTA         0x00000002u
/* CONSTANT_BUFFER_VALID(i)              MW((416+i*4):(416+i*4))         */
#define CB_VALID_BIT(i)           (416u + (i) * 4u)
/* SHADER_LOCAL_MEMORY_LOW_SIZE          MW(535:512)                     */
#define SLM_LOW_SIZE_HI           535u
#define SLM_LOW_SIZE_LO           512u
/* SHADER_LOCAL_MEMORY_HIGH_SIZE         MW(567:544)                     */
#define SLM_HIGH_SIZE_HI          567u
#define SLM_HIGH_SIZE_LO          544u
/* API_VISIBLE_CALL_LIMIT                MW(568:568)                     */
#define API_VIS_CALL_LIMIT_BIT    568u
#define API_VIS_CALL_LIMIT_NO_CHECK 1u
/* SAMPLER_INDEX                         MW(569:569)                     */
#define SAMPLER_INDEX_BIT         569u
#define SAMPLER_INDEX_INDEPENDENTLY 0u
/* QMD_MINOR_VERSION                     MW(579:576)                     */
#define QMD_MINOR_VERSION_HI      579u
#define QMD_MINOR_VERSION_LO      576u
/* QMD_MAJOR_VERSION                     MW(583:580)                     */
#define QMD_MAJOR_VERSION_HI      583u
#define QMD_MAJOR_VERSION_LO      580u
/* SHARED_MEMORY_SIZE                    MW(601:584)                     */
#define SMEM_SIZE_HI              601u
#define SMEM_SIZE_LO              584u
/* GRID_WIDTH                            MW(1055:1024)                   */
#define GRID_W_HI                 1055u
#define GRID_W_LO                 1024u
/* GRID_HEIGHT                           MW(1071:1056)                   */
#define GRID_H_HI                 1071u
#define GRID_H_LO                 1056u
/* GRID_DEPTH                            MW(1103:1088)                   */
#define GRID_D_HI                 1103u
#define GRID_D_LO                 1088u
/* CTA_THREAD_DIMENSION0/1/2             MW(1167:1152) / 1183:1168 / 1199:1184 */
#define CTA_DIM0_HI               1167u
#define CTA_DIM0_LO               1152u
#define CTA_DIM1_HI               1183u
#define CTA_DIM1_LO               1168u
#define CTA_DIM2_HI               1199u
#define CTA_DIM2_LO               1184u
/* REGISTER_COUNT                        MW(1208:1200)                   */
#define REGISTER_COUNT_HI         1208u
#define REGISTER_COUNT_LO         1200u
/* BARRIER_COUNT                         MW(1215:1211)                   */
#define BARRIER_COUNT_HI          1215u
#define BARRIER_COUNT_LO          1211u
/* PROGRAM_ADDRESS_LOWER                 MW(1247:1216)                   */
#define PROG_ADDR_LO_HI           1247u
#define PROG_ADDR_LO_LO           1216u
/* PROGRAM_ADDRESS_UPPER                 MW(1272:1248)                   */
#define PROG_ADDR_HI_HI           1272u
#define PROG_ADDR_HI_LO           1248u
/* CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(i) MW((1567+i*64):(1536+i*64))    */
#define CB_ADDR_LO_HI(i)          (1567u + (i) * 64u)
#define CB_ADDR_LO_LO(i)          (1536u + (i) * 64u)
/* CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6(i) MW((1586+i*64):(1568+i*64))    */
#define CB_ADDR_HI_HI(i)          (1586u + (i) * 64u)
#define CB_ADDR_HI_LO(i)          (1568u + (i) * 64u)
/* CONSTANT_BUFFER_SIZE_SHIFTED4(i)      MW((1599+i*64):(1587+i*64))     */
#define CB_SIZE_HI(i)             (1599u + (i) * 64u)
#define CB_SIZE_LO(i)             (1587u + (i) * 64u)

/* ── Field setters ─────────────────────────────────────────────────── */

void mc_qmd_init(uint8_t *qmd)
{
    memset(qmd, 0, MC_QMD_BYTES);

    /* NVK qmd_init! defaults (qmd.rs:60-66): */
    qmd_set_bits(qmd, QMD_MAJOR_VERSION_HI, QMD_MAJOR_VERSION_LO, 4);
    qmd_set_bits(qmd, QMD_MINOR_VERSION_HI, QMD_MINOR_VERSION_LO, 0);
    qmd_set_bits(qmd, API_VIS_CALL_LIMIT_BIT, API_VIS_CALL_LIMIT_BIT,
                 API_VIS_CALL_LIMIT_NO_CHECK);
    qmd_set_bits(qmd, SAMPLER_INDEX_BIT, SAMPLER_INDEX_BIT,
                 SAMPLER_INDEX_INDEPENDENTLY);
    /* QMD_TYPE = GRID_CTA (a normal compute grid).  NVK leaves QMD_TYPE
     * at 0 for V04; clcbc0qmd.h says GRID_CTA = 2 and QUEUE = 0.  Since
     * we ARE launching a CTA grid, set it explicitly to GRID_CTA — if
     * GSP requires it nonzero this matches CUDA's behaviour, and if it
     * doesn't matter (NVK works with 0) the SM ignores it. */
    qmd_set_bits(qmd, QMD_TYPE_HI, QMD_TYPE_LO, QMD_TYPE_GRID_CTA);
}

void mc_qmd_set_barrier_count(uint8_t *qmd, uint8_t n)
{
    qmd_set_bits(qmd, BARRIER_COUNT_HI, BARRIER_COUNT_LO, n);
}

void mc_qmd_set_global_size(uint8_t *qmd, uint32_t w, uint32_t h, uint32_t d)
{
    qmd_set_bits(qmd, GRID_W_HI, GRID_W_LO, w);
    qmd_set_bits(qmd, GRID_H_HI, GRID_H_LO, h);
    qmd_set_bits(qmd, GRID_D_HI, GRID_D_LO, d);
}

void mc_qmd_set_local_size(uint8_t *qmd, uint16_t w, uint16_t h, uint16_t d)
{
    qmd_set_bits(qmd, CTA_DIM0_HI, CTA_DIM0_LO, w);
    qmd_set_bits(qmd, CTA_DIM1_HI, CTA_DIM1_LO, h);
    qmd_set_bits(qmd, CTA_DIM2_HI, CTA_DIM2_LO, d);
}

void mc_qmd_set_prog_addr(uint8_t *qmd, uint64_t prog_gpu_va)
{
    /* NVK's qmd_impl_set_prog_addr_64!(clcbc0, QMDV04_00, NONE):
     * no shift — write the full 64-bit VA across LOWER+UPPER. */
    qmd_set_bits(qmd, PROG_ADDR_LO_HI, PROG_ADDR_LO_LO, (uint32_t)prog_gpu_va);
    qmd_set_bits(qmd, PROG_ADDR_HI_HI, PROG_ADDR_HI_LO, prog_gpu_va >> 32);
}

void mc_qmd_set_register_count(uint8_t *qmd, uint8_t n)
{
    qmd_set_bits(qmd, REGISTER_COUNT_HI, REGISTER_COUNT_LO, n);
}

void mc_qmd_set_slm_size(uint8_t *qmd, uint32_t bytes)
{
    /* NVK qmd_impl_set_slm_size! with size_suffix=NONE: round up to
     * 0x10 (16-byte multiple), write LOW_SIZE = bytes, HIGH_SIZE = 0. */
    uint32_t rounded = (bytes + 0xFu) & ~0xFu;
    qmd_set_bits(qmd, SLM_HIGH_SIZE_HI, SLM_HIGH_SIZE_LO, 0);
    qmd_set_bits(qmd, SLM_LOW_SIZE_HI, SLM_LOW_SIZE_LO, rounded);
}

void mc_qmd_set_smem_size(uint8_t *qmd, uint32_t bytes)
{
    /* NVK qmd_impl_set_smem_size_bounded! rounds up to 0x100 then
     * writes SHARED_MEMORY_SIZE plus MIN/MAX/TARGET_SM_CONFIG_*.
     *
     * For a 0-byte SMEM kernel (our doorbell-write case), only
     * SHARED_MEMORY_SIZE is written (it stays 0); the
     * MIN/MAX/TARGET_SM_CONFIG fields would only matter for
     * non-zero SMEM and are handled by gv100_get_hw_smem_sizes
     * if that path is ever ported.  For now, assert the
     * simple-case precondition. */
    uint32_t rounded = (bytes + 0xFFu) & ~0xFFu;
    assert(rounded == 0 && "non-zero SMEM not implemented; port gv100_get_hw_smem_sizes if needed");
    qmd_set_bits(qmd, SMEM_SIZE_HI, SMEM_SIZE_LO, rounded);
}

void mc_qmd_set_cbuf(uint8_t *qmd, uint8_t idx, uint64_t addr, uint32_t size)
{
    /* NVK qmd_impl_set_cbuf!(clcbc0, QMDV04_00, SHIFTED6, SHIFTED4):
     *   addr must be 64-byte-aligned (>>6 form),
     *   size must be 16-byte-aligned (>>4 form).
     * We mirror the layout via per-CB stride 64 bits = 8 bytes. */
    if (idx >= 8) return;
    assert((addr & 0x3F) == 0);
    assert((size & 0x0F) == 0);

    uint64_t addr_s6 = addr >> 6;
    uint32_t size_s4 = size >> 4;

    qmd_set_bits(qmd, CB_ADDR_LO_HI(idx), CB_ADDR_LO_LO(idx), (uint32_t)addr_s6);
    qmd_set_bits(qmd, CB_ADDR_HI_HI(idx), CB_ADDR_HI_LO(idx), addr_s6 >> 32);
    qmd_set_bits(qmd, CB_SIZE_HI(idx),    CB_SIZE_LO(idx),    size_s4);
    qmd_set_bits(qmd, CB_VALID_BIT(idx),  CB_VALID_BIT(idx),  1);
}

/* ── CB0 ──────────────────────────────────────────────────────────── */

void mc_cb0_init(uint8_t *cb0)
{
    /* No libcuda VA fragments, no signature dwords — just zero.  The
     * SASS reads c[0x0][0x208] (global memdesc, must be 0),
     * c[0x0][0x210/0x214] (dst), c[0x0][0x218] (token).  All other
     * bytes are unread. */
    memset(cb0, 0, MC_CB0_TOTAL_BYTES_ALIGNED);
}

void mc_cb0_set_args(uint8_t *cb0, uint64_t dst_gpu_va, uint32_t token)
{
    *(uint32_t *)(cb0 + MC_CB0_OFF_USER_DST_LO) = (uint32_t)dst_gpu_va;
    *(uint32_t *)(cb0 + MC_CB0_OFF_USER_DST_HI) = (uint32_t)(dst_gpu_va >> 32);
    *(uint32_t *)(cb0 + MC_CB0_OFF_USER_TOKEN ) = token;
}

/* ── SASS bytes for mc_doorbell_kernel ────────────────────────────────
 * Compiled offline via:
 *
 *   nvcc -arch=sm_90 --cubin -O0 dbell_write.cu -o dbell_write.cubin
 *   xxd -i -s 0x680 -l 256 dbell_write.cubin
 *
 * dbell_write.cu is not a file in this tree; the same kernel ships as
 * `reverse/tests/cuda/cuda_dbell_kernel_launch.cu`.  Its source is also
 * reproduced here so the blob can be regenerated standalone:
 *   extern "C" __global__
 *   void mc_doorbell_kernel(volatile unsigned int *dst,
 *                           unsigned int token)
 *   {  *dst = token;  }
 *
 * Disassembly (cuobjdump --dump-sass):
 *   LDC R1, c[0x0][0x28]
 *   LDC R5, c[0x0][0x218]              ; load `token`
 *   ULDC.64 UR4, c[0x0][0x208]         ; load global memdesc
 *   LDC.64 R2, c[0x0][0x210]           ; load `dst`
 *   STG.E.STRONG.SYS desc[UR4][R2.64], R5
 *   EXIT
 *   BRA self
 *   (NOP padding to 256 bytes)
 *
 * Resources: REG=8, STACK=0, SHARED=0, LOCAL=0, CONSTANT[0]=540 bytes.
 */
const uint8_t mc_doorbell_kernel_sass[] = {
    0x82, 0x7b, 0x01, 0xff, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00,
    0x00, 0xf0, 0x0f, 0x00, 0x82, 0x7b, 0x05, 0xff, 0x00, 0x86, 0x00, 0x00,
    0x00, 0x08, 0x00, 0x00, 0x00, 0x22, 0x0e, 0x00, 0xb9, 0x7a, 0x04, 0x00,
    0x00, 0x82, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0xce, 0x0f, 0x00,
    0x82, 0x7b, 0x02, 0xff, 0x00, 0x84, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00,
    0x00, 0x24, 0x0e, 0x00, 0x86, 0x79, 0x00, 0x02, 0x05, 0x00, 0x00, 0x00,
    0x04, 0x59, 0x11, 0x0c, 0x00, 0xe2, 0x1f, 0x00, 0x4d, 0x79, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x03, 0x00, 0xea, 0x0f, 0x00,
    0x47, 0x79, 0xfc, 0x00, 0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0x83, 0x03,
    0x00, 0xc0, 0x0f, 0x00, 0x18, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x0f, 0x00, 0x18, 0x79, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x0f, 0x00,
    0x18, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xc0, 0x0f, 0x00, 0x18, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x0f, 0x00, 0x18, 0x79, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x0f, 0x00,
    0x18, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xc0, 0x0f, 0x00, 0x18, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x0f, 0x00, 0x18, 0x79, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x0f, 0x00,
    0x18, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xc0, 0x0f, 0x00,
};
const uint32_t mc_doorbell_kernel_sass_len = sizeof(mc_doorbell_kernel_sass);
