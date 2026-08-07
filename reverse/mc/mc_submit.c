/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_submit.c — method stream + GPFIFO entry builders + channel
 * submission primitives.
 *
 * Pure-ish: no fds opened, no handles allocated.  Inputs are
 * already-allocated mc_channel_t fields + the BAR1 doorbell pointer;
 * outputs are written-to GPFIFO/USERD/PB memory.  drain_channel is
 * here too because it's also a "talk to GPU through USERD" primitive
 * (just polling rather than producing).
 */

#define _GNU_SOURCE
#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "nvtypes.h"
#include "nvmisc.h"
#include "class/clc8b5.h"        /* HOPPER_DMA_COPY_A + methods */
#include "class/clc86f.h"        /* HOPPER_CHANNEL_GPFIFO_A + GP entry */
#include "class/clc36f.h"        /* NVC36F_DMA_INCR_* method headers */
#include "class/cla06fsubch.h"   /* NVA06F_SUBCHANNEL_COPY_ENGINE */

#include "mc_internal.h"

/*
 * Kepler-style method header.  One 32-bit word that describes the payload
 * dwords that follow; PBDMA parses it to figure out which method addresses
 * to program and on which subchannel.
 *
 *   NVC36F_DMA_INCR_OPCODE      [31:29] = NVC36F_DMA_INCR_OPCODE_VALUE (1)
 *   NVC36F_DMA_INCR_COUNT       [28:16] = number of data dwords following
 *   NVC36F_DMA_INCR_SUBCHANNEL  [15:13] = subchannel (CE uses 4)
 *   NVC36F_DMA_INCR_ADDRESS     [11:0]  = method_address >> 2
 *
 * Subchannel NVA06F_SUBCHANNEL_COPY_ENGINE (= 4) is not arbitrary: PBDMA
 * routes methods to the engine bound to each subchannel, and our first
 * emitted method is a SET_OBJECT that binds HOPPER_DMA_COPY_A to
 * subchannel 4.  Subchannel 0 has no engine bound and would trigger
 * Xid 32 "invalid pushbuffer stream".
 */
#define INCR_HEADER_SUB(method, count, subch)       \
  (DRF_DEF(C36F, _DMA_INCR, _OPCODE, _VALUE)        \
   | DRF_NUM(C36F, _DMA_INCR, _COUNT, (count))      \
   | DRF_NUM(C36F, _DMA_INCR, _SUBCHANNEL, (subch)) \
   | DRF_NUM(C36F, _DMA_INCR, _ADDRESS, (method) >> 2))
#define INCR_HEADER(method, count) \
  INCR_HEADER_SUB((method), (count), NVA06F_SUBCHANNEL_COPY_ENGINE)
/*
 * Build the NVC8B5 method stream for a 1D CE copy.
 *
 * Direction-agnostic: src_va and dst_va are arbitrary GPU VAs — swap
 * them to get H2D vs D2H with identical machinery.  Both H2D and D2H go
 * through the same CE (COPY0/COPY1 etc. per pick_non_grce_lce) because
 * a CE doesn't care about the direction of the transfer, only about
 * reading pages from src_va and writing them to dst_va.
 *
 * The stream is 18 dwords = 72 bytes and mirrors what libcuda emits on
 * H100 PCIe: the copy and its completion signal are two separate
 * LAUNCH_DMAs rather than one.  Launch 1 (0x182) moves the data and
 * carries neither a flush nor a semaphore; SET_SEMAPHORE_A/B/PAYLOAD then
 * program the release cell; launch 2 moves no data
 * (DATA_TRANSFER_TYPE=NONE) and exists only to flush and release.
 *
 * A single fused launch also works — 0x18e = NON_PIPELINED | FLUSH_ENABLE
 * | RELEASE_ONE_WORD_SEMAPHORE | SRC_PITCH | DST_PITCH — and that is the
 * value Yan et al. report from an A40 trace.  H100 libcuda does not use
 * it, so neither do we; the SM-authored path in kernels/sm_owner.cu still
 * does, for the reason given there.
 *
 * sema_va + sema_payload program a 4-byte release semaphore — the CE
 * writes sema_payload to sema_va once the transfer has committed (the
 * release launch carries FLUSH_ENABLE=TRUE).  The CPU polls this word to
 * detect completion.  libcuda releases four words (payload plus a
 * timestamp) here; the one-word form keeps that poll a single 32-bit
 * compare.
 *
 * Publishing this method stream to PBDMA is the channel-submit
 * primitives' job (mc_channel_submit / mc_channel_arm), which emit
 * the produce-fence-publish ordering internally; callers do not
 * sfence between mc_write_transfer_methods() and the publish step.
 */
uint32_t *mc_write_transfer_methods(uint32_t *pb, uint64_t src_va,
                                           uint64_t dst_va, uint32_t nbytes,
                                           uint64_t sema_va,
                                           uint32_t sema_payload)
{
  *pb++ = INCR_HEADER(NVC86F_SET_OBJECT, 1);
  *pb++ = HOPPER_DMA_COPY_A;

  *pb++ = INCR_HEADER(NVC8B5_OFFSET_IN_UPPER, 2);
  *pb++ = DRF_NUM(C8B5, _OFFSET_IN_UPPER, _UPPER, (NvU32)(src_va >> 32));
  *pb++ = (NvU32)src_va;

  *pb++ = INCR_HEADER(NVC8B5_OFFSET_OUT_UPPER, 2);
  *pb++ = DRF_NUM(C8B5, _OFFSET_OUT_UPPER, _UPPER, (NvU32)(dst_va >> 32));
  *pb++ = (NvU32)dst_va;

  *pb++ = INCR_HEADER(NVC8B5_LINE_LENGTH_IN, 1);
  *pb++ = nbytes;

  /* EXPERIMENT: libcuda's split form.  Launch 1 is the copy alone — no
   * flush, no semaphore (SEMAPHORE_TYPE is field 4:3 and 0 means none,
   * so it is simply omitted).  This is bit-for-bit libcuda's 0x182. */
  *pb++ = INCR_HEADER(NVC8B5_LAUNCH_DMA, 1);
  *pb++ =
      DRF_DEF(C8B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NON_PIPELINED)
      | DRF_DEF(C8B5, _LAUNCH_DMA, _FLUSH_ENABLE, _FALSE)
      | DRF_DEF(C8B5, _LAUNCH_DMA, _SRC_MEMORY_LAYOUT, _PITCH)
      | DRF_DEF(C8B5, _LAUNCH_DMA, _DST_MEMORY_LAYOUT, _PITCH);

  /* Then the semaphore, then launch 2: release only, moves no data.
   * libcuda uses the four-word (timestamped) release here; we keep the
   * one-word form so the host's existing 4-byte poll still works, which
   * isolates the split itself as the only variable. */
  *pb++ = INCR_HEADER(NVC8B5_SET_SEMAPHORE_A, 3);
  *pb++ = DRF_NUM(C8B5, _SET_SEMAPHORE_A, _UPPER, (NvU32)(sema_va >> 32));
  *pb++ = (NvU32)sema_va;
  *pb++ = sema_payload;

  *pb++ = INCR_HEADER(NVC8B5_LAUNCH_DMA, 1);
  *pb++ =
      DRF_DEF(C8B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NONE)
      | DRF_DEF(C8B5, _LAUNCH_DMA, _FLUSH_ENABLE, _TRUE)
      | DRF_DEF(C8B5, _LAUNCH_DMA, _SEMAPHORE_TYPE, _RELEASE_ONE_WORD_SEMAPHORE);

  return pb;
}

/*
 * Write one or two 8-byte GPFIFO entries pointing PBDMA at a method stream
 * in the pushbuffer.
 *
 * Normal GP entry format (NVC86F):
 *   ENTRY0[31:2]  NVC86F_GP_ENTRY0_GET      pb_va[31:2] (stored in-place)
 *   ENTRY0[1:0]   NVC86F_GP_ENTRY0_FETCH    0 = UNCONDITIONAL
 *   ENTRY1[7:0]   NVC86F_GP_ENTRY1_GET_HI   pb_va[39:32]
 *   ENTRY1[30:10] NVC86F_GP_ENTRY1_LENGTH   length in dwords
 *   ENTRY1[31]    NVC86F_GP_ENTRY1_SYNC     0 = PROCEED
 *
 * A single normal entry encodes 40 bits of pb_va (pb_va[39:0] — 1 TiB).
 * UVM under Paper F1 routinely places user buffers above bit 40
 * (typical VAs look like 0x7xxx_xxxx_xxxx), so the extra high bits
 * come via a separate SET_PB_SEGMENT_EXTENDED_BASE entry written
 * BEFORE the normal entry.  That control entry is identified by
 * OPCODE = 0x4 in ENTRY1[7:0], with operand pb_va[56:40] in
 * ENTRY0[24:8].  Returns 1 if no ext-base needed, else 2.
 *
 * NVC86F_GP_ENTRY0_GET takes the *semantic* value (pb_va >> 2), not
 * the pre-shifted bit pattern — DRF_NUM masks to 30 bits then shifts
 * left by 2 to place pb_va[31:2] at bit positions [31:2].  Passing
 * pb_va directly would put pb_va[29:0] at ENTRY0[29:0] and trigger
 * Xid 31 when PBDMA reads 4× the wrong address.
 *
 * An sfence is emitted at the end so the GPFIFO ring BAR1 writes are
 * flushed before the caller advances GPPut.
 */
uint32_t write_gp_entry(volatile uint32_t *gpfifo_ring,
                               uint32_t gp_put_index, uint32_t ring_entries,
                               uintptr_t pb_cpu_va, uint32_t pb_offset,
                               uint32_t method_bytes)
{
  uintptr_t pb_va    = pb_cpu_va + pb_offset;
  /* NVC86F_GP_ENTRY1_LENGTH is a count of 32-bit dwords. */
  uint32_t  length   = method_bytes / (uint32_t)sizeof(uint32_t);
  uint32_t  consumed = 0;
  NvU32     ext_base = (NvU32)(pb_va >> 40);
  NvU32     idx, entry0, entry1;

  if (ext_base != 0)
  {
    idx = (gp_put_index + consumed) & (ring_entries - 1);
    gpfifo_ring[idx * 2 + 0] =
        DRF_NUM(C86F, _GP_ENTRY0, _PB_EXTENDED_BASE_OPERAND, ext_base);
    gpfifo_ring[idx * 2 + 1] =
        DRF_DEF(C86F, _GP_ENTRY1, _OPCODE, _SET_PB_SEGMENT_EXTENDED_BASE);
    consumed++;
  }

  idx    = (gp_put_index + consumed) & (ring_entries - 1);
  entry0 = DRF_NUM(C86F, _GP_ENTRY0, _GET, (NvU32)(pb_va >> 2))
           | DRF_DEF(C86F, _GP_ENTRY0, _FETCH, _UNCONDITIONAL);
  entry1 = DRF_NUM(C86F, _GP_ENTRY1, _GET_HI, (NvU32)(pb_va >> 32))
           | DRF_NUM(C86F, _GP_ENTRY1, _LENGTH, length)
           | DRF_DEF(C86F, _GP_ENTRY1, _LEVEL, _MAIN)
           | DRF_DEF(C86F, _GP_ENTRY1, _SYNC, _PROCEED);
  gpfifo_ring[idx * 2 + 0] = entry0;
  gpfifo_ring[idx * 2 + 1] = entry1;
  consumed++;

  _mm_sfence();
  return consumed;
}

/*
 * Submit work to the GPU.  TWO writes are required on Hopper, in order:
 *
 *   1. Advance GPPut in USERD.  PBDMA reads GPPut (not continuously —
 *      only when the channel is scheduled onto it) to learn how many
 *      GPFIFO entries are pending.  Writing GPPut makes the new entries
 *      officially "submitted" but does not wake up the GPU on its own.
 *
 *   2. Write the channel's work-submit-token to the VF doorbell
 *      (NV_VIRTUAL_FUNCTION_DOORBELL = 0x30090 *relative to the VF
 *      register block*, not to BAR0; reached here at +0x90 inside the
 *      USERMODE mapping — see MC_VF_DOORBELL_OFFSET).  This is the
 *      actual wake-up signal: the host scheduler looks up the channel
 *      by token, verifies it's on the runlist, and forwards a
 *      notification to PBDMA.
 *
 * Writing only (1) hangs — GPPut advances but PBDMA is asleep and never
 * wakes.  Writing only (2) dispatches PBDMA to read GPPut and find no
 * new work, wasting a cycle.  Both writes are mandatory, in this order.
 *
 * On x86 both are Write-Combine stores; the sfences flush WC buffers so
 * the GPU observes the writes in causal order.
 */
void ring_doorbell(volatile HopperAControlGPFifo *userd,
                          uint32_t new_gp_put, volatile uint32_t *vf_doorbell,
                          uint32_t work_submit_token)
{
  userd->GPPut = new_gp_put;
  _mm_sfence();
  *vf_doorbell = work_submit_token;
  _mm_sfence();
}

/* Poll a release-sema cell until *cell == expected, with the standard
 * MC_TIMEOUT_MS budget.  The cell may live in sysmem (host-rung paths,
 * sysmem-carrier paths, FB-carrier sysmem-dst path) or in HBM via a
 * BAR1-aliased CPU mapping (FB-carrier HBM-dst path).  The caller picks
 * which cell + expected payload to poll. */
mc_status_t mc_channel_poll_sema(volatile uint32_t *cell, uint32_t expected,
                                        struct timespec t0)
{
  struct timespec now;
  long elapsed_ms;
  while (*cell != expected)
  {
    volatile int s;
    clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed_ms = (now.tv_sec - t0.tv_sec) * 1000
                 + (now.tv_nsec - t0.tv_nsec) / 1000000;
    if (elapsed_ms > MC_TIMEOUT_MS) return MC_ETIMEOUT;
    for (s = 0; s < MC_POLL_SPIN_ITERATIONS
                && *cell != expected; s++)
      ;
  }
  return MC_OK;
}

mc_status_t mc_channel_arm(mc_channel_t *ch,
                                  uint32_t copy_bytes,
                                  struct timespec *t0_out)
{
  uint32_t entries_used;

  /* Producer-side fence: bytes the caller wrote into pb_cpu must
   * reach DRAM before PBDMA snoops the GPFIFO entry pointing at
   * pb_gpu_va.  See produce -> fence -> publish contract. */
  _mm_sfence();

  /* Clear the release semaphore so the upcoming poll observes a
   * fresh transition, not a stale prior payload. */
  *ch->sema_ptr = 0;
  _mm_sfence();

  if (t0_out) clock_gettime(CLOCK_MONOTONIC, t0_out);

  /* Publish: write the GPFIFO entry pointing at pb_gpu_va. */
  entries_used = write_gp_entry(ch->gpfifo_ring, ch->gp_put,
                                MC_GPFIFO_ENTRIES, ch->pb_gpu_va, 0,
                                copy_bytes);
  /* PBDMA reads USERD GPPut as a ring INDEX (Hopper NVC86F GPPut is
   * the offset into the GPFIFO ring, not a monotonic counter), so
   * the mirror must wrap modulo MC_GPFIFO_ENTRIES every advance.
   * UVM does the same in uvm_channel.c (cpu_put = (cpu_put + 1) %
   * num_gpfifo_entries). */
  ch->gp_put = (ch->gp_put + entries_used) % MC_GPFIFO_ENTRIES;
  _mm_sfence();

  /* Advance USERD GPPut so PBDMA sees the new entry on its next
   * wake event.  ring_doorbell() inside mc_channel_submit() does
   * the same store + sfence + the BAR1 doorbell write; mc_channel_arm
   * stops here and lets a different agent ring. */
  ch->userd->GPPut = ch->gp_put;
  _mm_sfence();

  return MC_OK;
}

mc_status_t mc_channel_submit(mc_channel_t *ch,
                                     volatile uint32_t *vf_doorbell,
                                     uint32_t copy_bytes)
{
  uint32_t entries_used;
  struct timespec t0;
  mc_status_t rc;

  /* Producer-side fence (same contract as mc_channel_arm). */
  _mm_sfence();

  *ch->sema_ptr = 0;
  _mm_sfence();

  clock_gettime(CLOCK_MONOTONIC, &t0);

  entries_used = write_gp_entry(ch->gpfifo_ring, ch->gp_put,
                                MC_GPFIFO_ENTRIES, ch->pb_gpu_va, 0,
                                copy_bytes);
  /* See mc_channel_arm: GPPut wraps modulo MC_GPFIFO_ENTRIES. */
  ch->gp_put = (ch->gp_put + entries_used) % MC_GPFIFO_ENTRIES;
  _mm_sfence();

  /* ring_doorbell does USERD GPPut + sfence + BAR1 doorbell + sfence
   * internally — the two writes that submit work on Hopper. */
  ring_doorbell(ch->userd, ch->gp_put, vf_doorbell, ch->work_submit_token);

  rc = mc_channel_poll_sema(ch->sema_ptr, ch->sema_payload, t0);
  if (rc != MC_OK)
  {
    uint32_t userd_gp_put = ch->userd ? ch->userd->GPPut : 0xffffffffu;
    uint32_t userd_gp_get = ch->userd ? ch->userd->GPGet : 0xffffffffu;
    uint32_t sema_actual  = ch->sema_ptr ? *ch->sema_ptr : 0xffffffffu;
    WARN_LOG("mc_channel_submit: poll failed rc=%d role=%d type=%d vas=%d "
             "h_channel=0x%x gp_put_mirror=0x%x USERD{GPPut=0x%x GPGet=0x%x} "
             "sema_ptr=%p sema_gpu_va=0x%llx sema_actual=0x%x "
             "sema_expected=0x%x pb_cpu=%p pb_gpu_va=0x%llx copy_bytes=%u "
             "entries_used=%u work_token=0x%x",
             rc, (int)ch->role, (int)ch->type, (int)ch->vas_id,
             ch->h_channel, ch->gp_put, userd_gp_put, userd_gp_get,
             (const void *)ch->sema_ptr,
             (unsigned long long)ch->sema_gpu_va, sema_actual,
             ch->sema_payload, (void *)ch->pb_cpu,
             (unsigned long long)ch->pb_gpu_va, copy_bytes, entries_used,
             ch->work_submit_token);
  }
  return rc;
}

/*
 * drain_channel — poll USERD until GPPut == GPGet (all submitted work
 * consumed) or timeout_ms elapses.  Called from mc_fini so the channel
 * is idle before resources are freed.  Teardown-tolerant: WARN and
 * return on timeout; the subsequent rm_channel_disable / channel free
 * will force-preempt anything still in flight.
 *
 * In practice the last mc_memcpy already polled the semaphore to
 * completion, so GPPut should already equal GPGet at entry — this is
 * insurance, not a hot path.
 */
void drain_channel(volatile HopperAControlGPFifo *userd, long timeout_ms)
{
  struct timespec t0, now;
  long            elapsed_ms = 0;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  while (userd->GPPut != userd->GPGet)
  {
    volatile int s;
    clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed_ms = (now.tv_sec - t0.tv_sec) * 1000
                 + (now.tv_nsec - t0.tv_nsec) / 1000000;
    if (elapsed_ms > timeout_ms)
    {
      WARN_LOG("drain_channel: timeout after %ld ms (GPPut=0x%x GPGet=0x%x)",
               elapsed_ms, userd->GPPut, userd->GPGet);
      return;
    }
    for (s = 0; s < MC_POLL_SPIN_ITERATIONS
                && userd->GPPut != userd->GPGet; s++)
      ;
  }
}
