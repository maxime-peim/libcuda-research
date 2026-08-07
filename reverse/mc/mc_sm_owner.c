/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_sm_owner.c — host-side glue for the sm_owner_kernel SASS.
 *
 * sm_owner_kernel is the canonical, executable description of the
 * Hopper CE submission protocol: a single SM thread writes the
 * pushbuffer methods, the GPFIFO entry, advances USERD GPPut, and
 * rings the BAR1 doorbell.  This file is the host's side of that
 * dance:
 *
 *   - patches a QMD with PROG_ADDR=sass_gpu_va + REGISTER_COUNT
 *     (NVK-style, mirroring mc_compute_doorbell_kernel)
 *   - writes CB0 with the DMA channel's resource VAs and the per-call
 *     workload payload
 *   - builds a compute pushbuffer that launches the kernel via
 *     SEND_PCAS_A + SET_REPORT_SEMAPHORE_RELEASE
 *   - rings the *compute channel's* doorbell from the host (this is
 *     the only host-issued doorbell in the experiment — everything
 *     after is the SM thread's job)
 *   - polls the compute channel's semaphore (proves the SM kernel
 *     ran to completion)
 *   - polls the DMA channel's semaphore (proves PBDMA woke and
 *     consumed the SM-authored bytes — the actual end-to-end success
 *     signal)
 *
 * Two semaphores, two failure modes.  If the compute sema fires but
 * the DMA sema doesn't, the SM-authored submission was malformed.
 * If neither fires, the kernel itself died (likely a bad address /
 * Xid 31 — check dmesg).
 *
 * CB0 parameter offsets are NVCC-determined, starting at 0x210.  If
 * sm_owner.cu's parameter list is reordered, regenerate the header
 * (`make mc/mc_sm_owner_sass.h`) and update the offsets below by
 * inspecting `cuobjdump --dump-sass mc/kernels/sm_owner.cubin`.
 */

#include <emmintrin.h>           /* _mm_clflush — no-op on today's WC
                                  * pages, kept as insurance; see
                                  * patch_cb0 for the reasoning */
#include <stddef.h>              /* offsetof for the struct-ABI assert */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "nvtypes.h"
#include "nvmisc.h"
#include "class/clc36f.h"        /* NVC36F_DMA_INCR_* (header-encoding macros) */
#include "class/clc7c0.h"        /* HOPPER-inherited compute methods */
#include "class/clc86f.h"        /* NVC86F_SET_OBJECT */
#include "class/clcbc0.h"        /* HOPPER_COMPUTE_A class id */
#include "class/cla0c0.h"        /* INVALIDATE_*_CACHE_NO_WFI */
#include "class/clb1c0.h"        /* INVALIDATE_SKED_CACHES */
#include "class/cla06fsubch.h"   /* NVA06F_SUBCHANNEL_COMPUTE */

#include "mc_internal.h"
#include "mc_compute_qmd.h"
#include "mc_sm_owner_args.h"
#include "mc_sm_owner_sass.h"

/* ── struct mc_sm_owner_args host/device ABI guards ────────────────────
 *
 * The args struct is shared between the host glue (this file) and
 * the SM-owner kernel (kernels/sm_owner.cu).  Both compilers use
 * natural alignment for u64 / u32, but the struct's layout is part
 * of the host/device ABI: if NVCC ever changed how it packs the
 * struct relative to gcc/clang, the host would write fields at
 * offsets the kernel doesn't read from.  Pin the layout via
 * compile-time asserts — these trip on the host build (which is
 * the only side that includes this header through C, not C++), so
 * a divergent NVCC build would surface as a host-side mismatch
 * the next time someone ran `make mc/mc_sm_owner_sass.h`.
 *
 * Total size: 8 u64 (64 bytes) + 6 u32 (24 bytes) = 88 bytes.  No
 * tail padding (struct alignment is 8, last field at offset 84
 * brings us to 88 = multiple of 8). */
_Static_assert(sizeof(struct mc_sm_owner_args) == 88,
               "struct mc_sm_owner_args size drifted from the host/device ABI");
_Static_assert(offsetof(struct mc_sm_owner_args, pb_gpu_va)              == 0,
               "mc_sm_owner_args.pb_gpu_va offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, sema_gpu_va)            == 48,
               "mc_sm_owner_args.sema_gpu_va offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, dma_sema_poll_va)       == 56,
               "mc_sm_owner_args.dma_sema_poll_va offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, size_bytes)             == 64,
               "mc_sm_owner_args.size_bytes offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, sema_payload)           == 68,
               "mc_sm_owner_args.sema_payload offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, work_submit_token)      == 72,
               "mc_sm_owner_args.work_submit_token offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, gp_put_in)              == 76,
               "mc_sm_owner_args.gp_put_in offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, dma_sema_poll_expected) == 80,
               "mc_sm_owner_args.dma_sema_poll_expected offset drift");
_Static_assert(offsetof(struct mc_sm_owner_args, dma_sema_poll_budget)   == 84,
               "mc_sm_owner_args.dma_sema_poll_budget offset drift");

/* ── CB0 layout for sm_owner_kernel ─────────────────────────────────────
 *
 * The kernel takes a single `struct mc_sm_owner_args *` parameter
 * (see mc_sm_owner_args.h).  NVCC packs that pointer at the start of
 * CB0's kernel-parameter band, byte 0x210 on observed CUDA 13.x
 * (the system-reserved band ends there).  We patch a single u64
 * (the args struct GPU VA) at that offset; the kernel does
 * LDG.E.64 / LDG.E through the pointer for each field.
 *
 * Why a struct + pointer instead of 11 individual params: NVCC's
 * per-parameter offset table is an implementation detail (start
 * base + per-type natural alignment + no padding between back-to-
 * back u32s), not an ABI guarantee.  C struct field offsets ARE an
 * ABI (fixed by the standard); using the struct insulates us from
 * any future toolkit shifting the parameter base or repacking
 * primitives.
 *
 * EARLIER MISTAKE (kept here as a war story for the next person who
 * touches this): the prior implementation passed 11 individual
 * parameters and assumed back-to-back u32s had 4-byte alignment
 * padding inserted between them.  They don't.  That put
 * work_submit_token at the wrong offset, the kernel read
 * uninitialised bytes as gp_put_in, USERD GPPut advanced to a
 * garbage value (0x1000c on one bring-up run), and PBDMA stalled
 * waiting for a slot that never existed.  The struct-pointer ABI
 * makes this kind of bug structurally impossible.
 *
 * Where the args struct lives: we co-locate it inside the same
 * 2 MiB CB0 sysmem allocation that the kernel parameter pointer
 * lives in, at offset 0x300 (well past the parameter band, which
 * ends around 0x260 with one pointer).  Saves a separate sysmem
 * allocation; same cacheable mapping the kernel reads through
 * normal GPU L2 / GMMU. */
#define MC_SMO_OFF_ARGS_PTR            0x210u
#define MC_SMO_ARGS_CB0_OFFSET         0x300u

/* ── Compute pushbuffer header macro (mirrors mc_compute_demo.c) ──────── */
#define INCR_HEADER_COMPUTE(method, count)                 \
  (DRF_DEF(C36F, _DMA_INCR, _OPCODE, _VALUE)               \
   | DRF_NUM(C36F, _DMA_INCR, _COUNT, (count))             \
   | DRF_NUM(C36F, _DMA_INCR, _SUBCHANNEL,                 \
             NVA06F_SUBCHANNEL_COMPUTE)                    \
   | DRF_NUM(C36F, _DMA_INCR, _ADDRESS, (method) >> 2))

/* ── Per-channel one-time compute setup methods ─────────────────────────
 *
 * Same recipe as mc_write_compute_setup_methods in mc_compute_demo.c.
 * Duplicated here (rather than exposed across TUs) because both files
 * are tightly coupled to their kernel-launch pattern and the setup is
 * <30 lines.  Both call sites share the `setup_done` flag in
 * mc_compute_extras so the methods only emit once per channel — the
 * DMA channel's PBDMA does not care which compute kernel executes
 * after SET_OBJECT, and HOPPER_COMPUTE_A binding is sticky.
 */
static uint32_t *write_compute_setup_methods(uint32_t *pb)
{
  uint64_t smem_window = 0x400000000ULL;     /* 16 GiB; above all mc VAs */
  uint64_t lmem_window = 0xffULL << 24;      /* NVK constant */

  *pb++ = INCR_HEADER_COMPUTE(NVC86F_SET_OBJECT, 1);
  *pb++ = HOPPER_COMPUTE_A;

  *pb++ = INCR_HEADER_COMPUTE(NVB1C0_INVALIDATE_SKED_CACHES, 1);
  *pb++ = 0;

  *pb++ = INCR_HEADER_COMPUTE(NVA0C0_INVALIDATE_SAMPLER_CACHE_NO_WFI, 1);
  *pb++ = NVA0C0_INVALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ALL;

  *pb++ = INCR_HEADER_COMPUTE(NVA0C0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, 1);
  *pb++ = NVA0C0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ALL;

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2);
  *pb++ = (NvU32)(smem_window >> 32);
  *pb++ = (NvU32)(smem_window & 0xffffffffu);

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2);
  *pb++ = (NvU32)(lmem_window >> 32);
  *pb++ = (NvU32)(lmem_window & 0xffffffffu);

  return pb;
}

static uint32_t *write_compute_launch_methods(uint32_t *pb,
                                              uint64_t qmd_gpu_va,
                                              uint64_t sema_va,
                                              uint32_t sema_payload)
{
  /* Per-launch INVALIDATE_SHADER_CACHES_NO_WFI(CONSTANT=TRUE).
   *
   * The SM front-end's constant cache holds prior CB0 reads keyed
   * by GPU VA.  Because we reuse the SAME CB0 GPU VA across every
   * launch and patch its contents per-call, the cache will return
   * stale bytes from a prior launch unless explicitly invalidated.
   *
   * Symptom without this method: iter 0 succeeds, iter 1+ wedges.
   * Iter 1's kernel reads iter 0's gp_put_in and sema_payload, so
   * the SM's USERD GPPut store is a no-op (writes the same value
   * already there) and PBDMA never wakes.  The dma_ch sema stays at
   * 0 and times out.
   *
   * The existing mc_compute_doorbell_kernel path didn't expose this
   * because all its callers pass identical CB0 args every iter
   * (dst_va = const doorbell, token = const work_submit_token).
   * mc_sm_owner_submit_d2h is the first multi-arg-per-iter consumer.
   *
   * NO_WFI variant: don't stall the engine waiting for inflight
   * compute to drain.  We're at the start of a new launch; nothing
   * is inflight on this channel since the prior launch already
   * fired its release sema before we polled it. */
  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1);
  *pb++ = DRF_DEF(C7C0, _INVALIDATE_SHADER_CACHES_NO_WFI, _CONSTANT,    _TRUE)
        | DRF_DEF(C7C0, _INVALIDATE_SHADER_CACHES_NO_WFI, _GLOBAL_DATA, _TRUE);

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SEND_PCAS_A, 1);
  *pb++ = (NvU32)(qmd_gpu_va >> 8);

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SEND_SIGNALING_PCAS2_B, 1);
  *pb++ = 0x0000000a;                /* SCHEDULE | <bit 3>; verified working
                                      * — same value mc_compute_demo.c uses. */

  *pb++ = INCR_HEADER_COMPUTE(NVC7C0_SET_REPORT_SEMAPHORE_A, 4);
  *pb++ = DRF_NUM(C7C0, _SET_REPORT_SEMAPHORE_A, _OFFSET_UPPER,
                  (NvU32)(sema_va >> 32));
  *pb++ = (NvU32)sema_va;
  *pb++ = sema_payload;
  *pb++ = DRF_DEF(C7C0, _SET_REPORT_SEMAPHORE_D, _OPERATION, _RELEASE)
        | DRF_DEF(C7C0, _SET_REPORT_SEMAPHORE_D, _STRUCTURE_SIZE, _ONE_WORD)
        | DRF_DEF(C7C0, _SET_REPORT_SEMAPHORE_D, _FLUSH_DISABLE, _FALSE);

  return pb;
}

/* ── Build the sm_owner_kernel argument tuple ─────────────────────────
 *
 * The kernel takes one parameter — a pointer to struct mc_sm_owner_args.
 * We place the args struct inside the same CB0 sysmem allocation at
 * offset MC_SMO_ARGS_CB0_OFFSET (0x300), and write its GPU VA at
 * MC_SMO_OFF_ARGS_PTR (0x210, NVCC's first kernel-parameter slot).
 *
 * CB0's parameter band (0x210..) holds only the single u64 args
 * pointer; the system band (0x000-0x210) is owned by NVCC.  No need
 * to zero either — we always overwrite both the pointer slot and
 * every field of the args struct on every call. */
static void patch_cb0(uint8_t                       *cb0,
                      uint64_t                       cb0_gpu_va,
                      const struct mc_sm_owner_args *args)
{
  /* Place the args struct at the agreed-upon offset inside CB0. */
  struct mc_sm_owner_args *cb0_args =
      (struct mc_sm_owner_args *)(cb0 + MC_SMO_ARGS_CB0_OFFSET);
  *cb0_args = *args;

  /* Write the args-struct GPU VA into the kernel-parameter slot. */
  *(uint64_t *)(cb0 + MC_SMO_OFF_ARGS_PTR) =
      cb0_gpu_va + MC_SMO_ARGS_CB0_OFFSET;

  /* Make the CB0 stores globally visible before the kernel launch.
   * CB0 is allocated write-combined,
   * so the stores above sit in the CPU's write-combining buffers,
   * not in cache lines — the SFENCE below is what drains those
   * buffers to DRAM, and it is the instruction doing the work here.
   * The SM kernel then reads CB0 — the parameter slot at 0x210 and
   * the args struct at 0x300 — through GPU L2 + GMMU PTE -> sysmem.
   *
   * The three CLFLUSHes are retained deliberately even though on WC
   * pages they are architecturally no-ops (WC never allocates cache
   * lines, so there is nothing to flush).  If this allocation is
   * ever moved to cached memory, the flushes become the thing that
   * pushes dirty L1/L2/L3 lines out before the SM's read, closing
   * the stale-prior-call-values hazard deterministically — so they
   * stay as insurance at ~ns per line.  They cover what we wrote:
   *   - the line containing CB0[0x210] (the args ptr)
   *   - the two lines spanning CB0[0x300..0x357] (the 88-byte
   *     args struct: 0x300-0x33F + 0x340-0x357) */
  _mm_clflush(cb0 + MC_SMO_OFF_ARGS_PTR);
  _mm_clflush(cb0 + MC_SMO_ARGS_CB0_OFFSET);
  _mm_clflush(cb0 + MC_SMO_ARGS_CB0_OFFSET + 64);
  __asm__ __volatile__("sfence" ::: "memory");
}

/* ── Failure-path diagnostic helper ─────────────────────────────────────
 *
 * Called when mc_sm_owner_submit_d2h's DMA-channel sema poll times
 * out — i.e. the compute kernel ran (its own sema fired) but PBDMA
 * on the DMA channel never released our payload.  Dumps the raw
 * post-submit state so the failure mode is debuggable from stderr:
 *
 *   - USERD GPPut/GPGet (did the SM actually advance GPPut?)
 *   - GPFIFO entry at the slot the SM was supposed to write
 *   - What the same entry would look like if the host had written it
 *     via write_gp_entry (so we can diff SM-authored vs. host-authored
 *     bytes and localise where the kernel encoding went wrong).
 *
 * Reads dma_ch->userd->GPPut/GPGet through volatile-qualified
 * pointers (declared in mc_internal.h::mc_channel_t).  No state is
 * modified; this is read-only diagnostics. */
static void dump_dma_timeout_diag(const mc_channel_t *dma_ch)
{
  uint32_t got_gpput;
  uint32_t got_gpget;
  uint32_t slot;
  uint32_t e0;
  uint32_t e1;
  uint64_t pb_va;
  uint32_t e0_expected;
  uint32_t e1_expected;

  got_gpput = dma_ch->userd ? dma_ch->userd->GPPut : 0xDEADBEEFu;
  got_gpget = dma_ch->userd ? dma_ch->userd->GPGet : 0xDEADBEEFu;
  slot      = dma_ch->gp_put & (MC_GPFIFO_ENTRIES - 1);
  e0        = dma_ch->gpfifo_ring ? dma_ch->gpfifo_ring[slot * 2 + 0] : 0;
  e1        = dma_ch->gpfifo_ring ? dma_ch->gpfifo_ring[slot * 2 + 1] : 0;

  /* Reconstruct what host-side write_gp_entry would have produced
   * for the same (pb_gpu_va, length=MC_SM_OWNER_PB_METHOD_DWORDS,
   * slot) so we can diff against what the SM wrote. */
  pb_va       = dma_ch->pb_gpu_va;
  e0_expected = (uint32_t)(pb_va & 0xFFFFFFFCu);
  e1_expected = (uint32_t)((pb_va >> 32) & 0xFFu)
              | (MC_SM_OWNER_PB_METHOD_DWORDS << 10);

  fprintf(stderr,
          "[mc] mc_sm_owner_submit: DMA sema did not fire "
          "(SM-authored submission was malformed or PBDMA wedged) --\n"
          "  dma_ch->gp_put_mirror=%u expected_USERD_GPPut=%u\n"
          "  USERD GPPut=0x%08x GPGet=0x%08x sema_actual=0x%08x sema_expected=0x%08x\n"
          "  pb_gpu_va=0x%llx slot=%u\n"
          "  GPFIFO[slot]: entry0=0x%08x (expected 0x%08x)  entry1=0x%08x (expected 0x%08x)\n",
          dma_ch->gp_put, (dma_ch->gp_put + 1) % MC_GPFIFO_ENTRIES,
          got_gpput, got_gpget, *dma_ch->sema_ptr, dma_ch->sema_payload,
          (unsigned long long)pb_va, slot,
          e0, e0_expected, e1, e1_expected);
}

/* ── Public entry: SM authors a CE copy on a chosen carrier ─────────────
 *
 * Submits the sm_owner_kernel on the chosen VAS's compute channel.
 * The kernel authors the entire CE-channel submission for the VAS's
 * SM-victim DMA channel — pushbuffer methods, GPFIFO entry, USERD
 * GP_PUT, BAR1 doorbell — then PBDMA wakes and runs the CE copy.
 *
 * `src_gpu_va` and `dst_gpu_va` must both be GPU VAs valid in the
 * chosen carrier VAS.
 *
 *   MC_VAS_SYSMEM_CARRIER → victim role = HOST_DMA (today's reuse).
 *   MC_VAS_FB_CARRIER     → victim role = SM_VICTIM_DMA, which owns
 *                           a dedicated sysmem sema cell of its own.
 *
 * Either way the active release semaphore is sysmem-resident and the
 * host polls it directly, with the existing 2 s budget.  The routing
 * comment in the body says why: an earlier design put the FB-carrier
 * sema in HBM and had the SM kernel poll it, which proved unreliable.
 *
 * Returns MC_OK on success, MC_ETIMEOUT if the compute kernel
 * itself doesn't exit, MC_EHANG if the kernel exited but the victim
 * channel never released the DMA sema.
 */
mc_status_t mc_sm_owner_submit(mc_ctx_t *ctx, mc_vas_t vas, uint64_t src_gpu_va,
                               uint64_t dst_gpu_va, uint32_t size_bytes)
{
  mc_channel_t             *compute_ch;
  mc_channel_t             *dma_ch;
  struct mc_compute_extras *cex;
  struct mc_compute_module *mod;
  uint64_t                  dbell_gpu_va;
  uint32_t                 *pb;
  uint32_t                 *pb_end;
  uint32_t                  copy_bytes;
  struct timespec           t0;
  mc_status_t               rc;
  mc_channel_role_t         victim_role;
  volatile uint32_t        *active_sema_ptr;
  NvU64                     active_sema_gpu_va;

  if (ctx == NULL)                            return MC_EINVAL;
  if (size_bytes == 0)                        return MC_EINVAL;
  if (size_bytes > MC_MAX_TRANSFER_SIZE)      return MC_EINVAL;

  /* (vas, agent=SM) → victim role.  Sysmem-carrier reuses HOST_DMA;
   * FB-carrier has a dedicated SM_VICTIM_DMA. */
  if (vas == MC_VAS_SYSMEM_CARRIER)
  {
    victim_role = MC_ROLE_HOST_DMA;
  }
  else if (vas == MC_VAS_FB_CARRIER)
  {
    victim_role = MC_ROLE_SM_VICTIM_DMA;
  }
  else
  {
    return MC_EINVAL;
  }

  compute_ch = mc_vas_find_channel(&ctx->vas[vas], MC_ROLE_COMPUTE);
  dma_ch     = mc_vas_find_channel(&ctx->vas[vas], victim_role);
  if (compute_ch == NULL || !compute_ch->h_channel) return MC_EINTERNAL;
  if (dma_ch     == NULL || !dma_ch->h_channel)     return MC_EINTERNAL;
  cex        = &compute_ch->x.compute;
  mod        = &cex->sm_owner_module;
  dbell_gpu_va = ctx->vas[vas].dbell_gpu_va;
  if (!dbell_gpu_va) return MC_EINTERNAL;

  /* Active sema cell + verification path.
   *
   * Both directions of FB-carrier MC_XFER_SM (and both directions of
   * sysmem-carrier MC_XFER_SM) cross PCIe on at least one side: one of
   * (src_gpu_va, dst_gpu_va) is sysmem-backed in every case mc
   * exposes today.  The single-LAUNCH_DMA NVC8B5 method stream emits
   * the data MWrs/MRds and the release-sema MWr from the same channel
   * (one Requester ID, one address space when the sema is sysmem-
   * resident), so PCIe producer-side ordering of posted writes
   * guarantees the sema MWr cannot land at host DRAM ahead of any
   * earlier data MWr.  For H2D specifically the LAUNCH_DMA's
   * FLUSH_ENABLE forces all CplDs to be absorbed into HBM via L2
   * before the sema-release method retires, so by the time the host
   * observes the sema the data is in HBM.
   *
   * Routing rule, simple and direction-symmetric: the active sema is
   * always sysmem-resident; the host always polls it.  FB-carrier
   * uses the dedicated sysmem sibling cell allocated alongside the
   * SM_VICTIM_DMA channel; sysmem-carrier uses its existing sema.
   *
   * History: a prior architecture put the FB-carrier H2D sema in HBM
   * and had the SM kernel poll it via `.cg` LDG.  That poll proved
   * unreliable (LTC-line aliasing from the SM's own prior FB stores
   * in the same kernel can mask PBDMA's later release-store), and a
   * separate two-LAUNCH_DMA pattern was introduced to enforce GPU-
   * side method ordering between the data and the release.  Both are
   * superseded — the sysmem-sema host-poll path subsumes them. */
  if (vas == MC_VAS_FB_CARRIER)
  {
    if (dma_ch->sema_sysmem_ptr == NULL || dma_ch->sema_sysmem_gpu_va == 0)
      return MC_EINTERNAL;  /* bring-up didn't allocate the sysmem sema */
    active_sema_ptr    = dma_ch->sema_sysmem_ptr;
    active_sema_gpu_va = dma_ch->sema_sysmem_gpu_va;
  }
  else
  {
    active_sema_ptr    = dma_ch->sema_ptr;
    active_sema_gpu_va = dma_ch->sema_gpu_va;
  }

  /* Bump victim-channel semaphore payload (skip 0). */
  dma_ch->sema_payload++;
  if (dma_ch->sema_payload == 0) dma_ch->sema_payload = 1;

  /* Clear the active semaphore so the host's poll observes a fresh
   * transition vs. a stale prior payload.  The cell is write-combined
   * sysmem, so the SFENCE is what
   * makes the store globally visible — it drains the CPU's
   * write-combining buffers before the kernel launch.  The CLFLUSH is
   * an architectural no-op on WC (no cache line is ever allocated)
   * and is retained only so this sequence stays correct if the cell
   * ever moves to cached memory. */
  *active_sema_ptr = 0;
  _mm_clflush((const void *)active_sema_ptr);
  __asm__ __volatile__("sfence" ::: "memory");

  /* Patch the QMD: NVK-style from zero.  Same field set as
   * mc_compute_doorbell_kernel — only PROGRAM_ADDRESS and
   * REGISTER_COUNT differ from the embedded-doorbell module.
   * CB0 binding uses MC_CB0_TOTAL_BYTES_ALIGNED for size. */
  mc_qmd_init(mod->qmd_cpu);
  mc_qmd_set_global_size   (mod->qmd_cpu, 1, 1, 1);
  mc_qmd_set_local_size    (mod->qmd_cpu, 1, 1, 1);
  mc_qmd_set_prog_addr     (mod->qmd_cpu, mod->sass_gpu_va);
  mc_qmd_set_register_count(mod->qmd_cpu, MC_SM_OWNER_REGS);
  mc_qmd_set_barrier_count (mod->qmd_cpu, 0);
  mc_qmd_set_slm_size      (mod->qmd_cpu, 0);
  mc_qmd_set_smem_size     (mod->qmd_cpu, 0);
  mc_qmd_set_cbuf(mod->qmd_cpu, /*idx=*/0, mod->cb0_gpu_va,
                  MC_CB0_TOTAL_BYTES_ALIGNED);

  /* Patch CB0 with the per-call argument tuple.  The kernel reads
   * one u64 (the args-struct GPU VA) at CB0[0x210], then dereferences
   * it for every field.  Sysmem path leaves dma_sema_poll_va = 0
   * which the kernel treats as "skip the poll loop". */
  struct mc_sm_owner_args args = {
      .pb_gpu_va         = dma_ch->pb_gpu_va,
      .gpfifo_gpu_va     = dma_ch->gpfifo_gpu_va,
      /* userd_gpu_va: gpfifo_gpu_va points at the start of the
       * gpfifo+userd alloc; the kernel needs the address of the
       * HopperAControlGPFifo struct itself, so add the slot offset. */
      .userd_gpu_va      = dma_ch->gpfifo_gpu_va + MC_USERD_OFFSET,
      .dbell_gpu_va      = dbell_gpu_va,
      .src_gpu_va        = src_gpu_va,
      .dst_gpu_va        = dst_gpu_va,
      .sema_gpu_va       = active_sema_gpu_va,
      /* dma_sema_poll_* fields are zeroed: the SM kernel's optional
       * `.cg` LDG poll loop is dead under the unified sysmem-sema
       * host-poll architecture.  The kernel still has the loop body
       * gated on `dma_sema_poll_va != 0` so it skips it cleanly. */
      .dma_sema_poll_va  = 0,
      .size_bytes        = size_bytes,
      .sema_payload      = dma_ch->sema_payload,
      .work_submit_token = dma_ch->work_submit_token,
      .gp_put_in         = dma_ch->gp_put,
      .dma_sema_poll_expected = 0,
      .dma_sema_poll_budget   = 0,
  };
  patch_cb0(mod->cb0_cpu, mod->cb0_gpu_va, &args);

  if (mc_debug())
  {
    fprintf(stderr,
        "[mc] DEBUG: mc_sm_owner_submit(vas=%d): src=0x%llx dst=0x%llx "
        "n=%u\n  pb=0x%llx gpfifo=0x%llx userd=0x%llx dbell=0x%llx\n"
        "  sema=0x%llx payload=0x%x work_token=0x%x gp_put=%u\n",
        (int)vas,
        (unsigned long long)src_gpu_va, (unsigned long long)dst_gpu_va,
        size_bytes,
        (unsigned long long)dma_ch->pb_gpu_va,
        (unsigned long long)dma_ch->gpfifo_gpu_va,
        (unsigned long long)(dma_ch->gpfifo_gpu_va + MC_USERD_OFFSET),
        (unsigned long long)dbell_gpu_va,
        (unsigned long long)active_sema_gpu_va,
        dma_ch->sema_payload, dma_ch->work_submit_token, dma_ch->gp_put);
  }

  /* Bump compute-channel semaphore payload — the kernel's release
   * sema fires after the kernel returns. */
  compute_ch->sema_payload++;
  if (compute_ch->sema_payload == 0) compute_ch->sema_payload = 1;

  /* Build the compute-channel pushbuffer.  First call: setup methods
   * + launch.  Subsequent calls: launch only. */
  pb = compute_ch->pb_cpu;
  if (!cex->setup_done)
  {
    pb = write_compute_setup_methods(pb);
    cex->setup_done = true;
  }
  pb_end = write_compute_launch_methods(pb, mod->qmd_gpu_va,
                                        compute_ch->sema_gpu_va,
                                        compute_ch->sema_payload);
  copy_bytes = (uint32_t)((pb_end - compute_ch->pb_cpu) * sizeof(uint32_t));

  /* Submit on the compute channel: rings the host doorbell, polls
   * the compute sema until the kernel exits.  Compute-sema fire only
   * proves the SM kernel ran to completion (PB / GPFIFO / USERD /
   * doorbell stores were emitted); it does NOT prove PBDMA has
   * processed the GPFIFO entry yet.  The DMA-sema poll below is the
   * actual end-to-end success signal. */
  rc = mc_channel_submit(compute_ch, ctx->vf_doorbell, copy_bytes);
  if (rc != MC_OK)
  {
    fprintf(stderr, "[mc] mc_sm_owner_submit(vas=%d): COMPUTE sema "
                    "did not fire (sm_owner_kernel itself failed) -- "
                    "compute_ch->gp_put=%u sema_payload_expected=%u "
                    "sema_actual=0x%08x\n",
                    (int)vas, compute_ch->gp_put, compute_ch->sema_payload,
                    *compute_ch->sema_ptr);
    return rc;
  }

  /* Host-poll the active (sysmem) sema as ground truth.  PBDMA may
   * not have processed its newly-published GPFIFO entry yet (CE work
   * is asynchronous), so the compute-sema fire above does NOT imply
   * the CE LAUNCH_DMA committed.  t0 starts here so the budget covers
   * PBDMA work, not compute-launch latency. */
  clock_gettime(CLOCK_MONOTONIC, &t0);
  rc = mc_channel_poll_sema(active_sema_ptr, dma_ch->sema_payload, t0);
  if (rc == MC_OK)
  {
    /* GPPut wraps modulo MC_GPFIFO_ENTRIES; see mc_submit.c. */
    dma_ch->gp_put = (dma_ch->gp_put + 1) % MC_GPFIFO_ENTRIES;
  }
  else
  {
    /* DMA sema timed out — the SM kernel ran (compute sema fired)
     * but PBDMA on the DMA channel never released our payload.
     * Dump the raw post-submit state so we can see what the SM
     * actually wrote vs. what was expected. */
    dump_dma_timeout_diag(dma_ch);
    /* Translate ETIMEOUT to EHANG: kernel observably ran, victim
     * channel didn't.  Same semantic distinction as the FB path. */
    if (rc == MC_ETIMEOUT) rc = MC_EHANG;
  }
  return rc;
}

