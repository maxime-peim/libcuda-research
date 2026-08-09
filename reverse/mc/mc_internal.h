/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_internal.h — private header for libmc.  Library-internal
 * .c files (mc_core.c, mc_rm.c, mc_uvm.c, mc_submit.c, mc_vaspace.c,
 * mc_sm_owner.c, mc_compute_demo.c) include this; public consumers
 * of the library use mc.h.
 *
 * Defines the full mc_ctx_t layout (each VAS owns its channels
 * inline via mc_va_space_t.channels[]), the MC_*
 * named constants, log macros (DEBUG_LOG /
 * WARN_LOG / ERROR_LOG), the CHECK fatal-assert macro, and the
 * cross-TU helper API surface (RM/UVM ioctl wrappers, submission
 * primitives, VA-space helpers).  Kept separate from mc.h so
 * mc_ctx_t stays opaque to users (they hold a mc_ctx_t * only —
 * no sizeof, no stack allocation, no field access).
 */
#ifndef MC_INTERNAL_H
#define MC_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>          /* fprintf, fputc — used in log macros */
#include <stdlib.h>         /* abort — used in CHECK macro */
#include <time.h>           /* struct timespec — used in submit primitives */

#include "mc.h"

/* NV SDK types — needed in this header for mc_ctx_t fields. */
#include "nvtypes.h"
#include "nvCpuUuid.h"      /* NV_UUID_LEN */
#include "class/clc86f.h"   /* HopperAControlGPFifo */

#define MC_UNUSED __attribute__((unused))

/* ── Named constants ───────────────────────────────────────────────────────
 * Each carries its rationale inline.  These are not arbitrary: most encode
 * a hardware or ABI constraint, and changing one without reading why will
 * usually produce an Xid rather than a compile error.
 */
/* GPFIFO ring geometry: entry count (power of two, PBDMA masks modulo-wise)
 * and per-entry byte size (two 32-bit dwords = GP_ENTRY0 + GP_ENTRY1). */
#define MC_GPFIFO_ENTRIES       512
#define MC_GPFIFO_ENTRY_BYTES   8

/* Buffer sizes.  The gpfifo+userd region holds the GPFIFO ring @ +0
 * and USERD @ +MC_USERD_OFFSET inside one 2 MiB alloc (UVM big-page
 * size: vidmem for the UVM channel, sysmem for the carrier-bound
 * channels).  Pushbuffer is 1 MiB of sysmem so the kernel's
 * sysmem-tracker (256-page threshold in nv-mmap.c) admits it into
 * the lookup table.  USERMODE_A is the 64 KiB VF-register aperture.
 * Semaphore is one page of sysmem — the host spins on it, so an FB
 * cell would turn every poll into a PCIe read competing with the
 * transfer it is waiting on. */
#define MC_GPFIFO_USERD_SIZE         (2ULL * 1024ULL * 1024ULL)
#define MC_USERMODE_SIZE        (64ULL * 1024ULL)
#define MC_SEMA_SIZE            4096
#define MC_PB_SIZE              (1u * 1024u * 1024u)

/* mc_memcpy semaphore poll timeout.  2 s is >> any legit CE copy
 * up to MC_MAX_TRANSFER_SIZE at the worst PCIe rates observed in practice. */
#define MC_TIMEOUT_MS           2000

/* USERD offset inside the shared gpfifo+userd allocation: the GPFIFO
 * ring sits at +0 and USERD at +0x2000.  That is slot 16 of the 512-byte
 * Hopper USERD slots — 4 KiB-aligned, matching libcuda's placement. */
#define MC_USERD_OFFSET         0x2000

/* VF doorbell offset inside the USERMODE mapping.
 *
 * dev_vm.h publishes NV_VIRTUAL_FUNCTION_DOORBELL = 0x30090 and
 * NV_VIRTUAL_FUNCTION = 0x0003FFFF:0x00030000.  Both are *VF-block
 * relative*, and the USERMODE window is anchored at the VF base
 * (kern_gpu_tu102.c -> GPU_GET_VREG_OFFSET(DRF_BASE(NV_VIRTUAL_FUNCTION))),
 * so the doorbell always lands at 0x30090 - 0x30000 = 0x90 inside it.
 *
 * Careful: 0x30090 is NOT the bare-metal BAR0 offset.  RM reaches the
 * register as GPU_GET_VREG_OFFSET(pGpu, NV_VIRTUAL_FUNCTION_DOORBELL)
 * = sriovState.virtualRegPhysOffset + 0x30090, and
 * gpuGetVirtRegPhysOffset_TU102() (used by every non-Tegra chip,
 * incl. GH100/GA10x) returns DRF_BASE(NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET)
 * = 0xB80000 on a host, returning 0 only inside an SR-IOV guest.  So on
 * bare metal the register is really at BAR0+0xBB0090; 0x30090 is the
 * guest's view.
 *
 * +0x90 is invariant regardless: the window base and the register pick up
 * the same virtualRegPhysOffset, and it holds whether the window is
 * carved from BAR0 (Ampere, or bBar1Mapping=FALSE) or BAR1 (Hopper as
 * libcuda and mc allocate it). */
#define MC_VF_DOORBELL_OFFSET   0x90

/* VA pool (Paper F1 anchor).  A 4 GiB PROT_NONE reservation at
 * 0x200000000 for every mc_malloc_host allocation; each reservation is
 * rounded up to a 2 MiB boundary (Hopper UVM big-page size) so a single
 * big-page PTE covers each region without sub-page edge cases. */
#define MC_VA_POOL_BASE         0x200000000ULL
#define MC_VA_POOL_SIZE         (4ULL * 1024ULL * 1024ULL * 1024ULL)
#define MC_VA_POOL_ALIGN_BYTES  (2ULL * 1024ULL * 1024ULL)

/* NV_MEMORY_ALLOCATION_PARAMS.alignment for vidmem allocs — 64 KiB is the
 * Hopper-friendly choice matching libcuda's bundle. */
#define MC_VIDMEM_ALIGN_BYTES   0x10000ULL

/* Per-channel UVM reservation: a 4 MiB window at 0x7f0000000000 passed to
 * UVM_REGISTER_CHANNEL as base/length.  Well above the VA pool at
 * MC_VA_POOL_BASE and above the UVM-internal region; user mappings must
 * not overlap. */
#define MC_UVM_CHANNEL_BASE     0x7f0000000000ULL
#define MC_UVM_CHANNEL_LENGTH   (4ULL * 1024ULL * 1024ULL)

/* Inner busy-wait iterations between wall-clock re-checks in the
 * semaphore-poll loops of drain_channel and mc_memcpy.  Sized
 * so the tight inner loop spins for ~microseconds between syscalls to
 * clock_gettime — small enough to keep the timeout bound meaningful,
 * large enough that the clock syscall isn't the hot path at GB/s
 * transfer rates. */
#define MC_POLL_SPIN_ITERATIONS 65536

/* Upper bound per single CE launch (LINE_LENGTH_IN is a 32-bit byte count). */
#define MC_MAX_TRANSFER_SIZE    (0xFFFFFFFFULL)
/* Lower bound: UVM_CREATE_EXTERNAL_RANGE needs GPU-MMU-page alignment. */
#define MC_MIN_TRANSFER_SIZE    4096ULL

#define MC_CONTROL_DEV_PATH     "/dev/nvidiactl"
#define MC_DEVICE_DEV_PATH      "/dev/nvidia0"

/* MC_VAS_COUNT is the size of mc_ctx.vas[] — one slot per public
 * mc_vas_t value (UVM, SYSMEM_CARRIER, FB_CARRIER). */
#define MC_VAS_COUNT 3

/* Allocation-table entry.  One slot per live buffer returned by
 * mc_malloc_device or mc_malloc_host; cleared on mc_free. */
#define MC_ALLOC_TABLE_SLOTS 256
#define MC_UVM_EXTERNAL_SEGMENTS_MAX 3

typedef struct {
    void            *ptr;        /* user-visible pointer (CPU VA for host, GPU VA for device).
                                  * NULL = slot free; this is also the canonical key
                                  * alloc_table_lookup matches on. */
    NvHandle         h_mem;      /* RM memory handle */
    NvU64            gpu_va;     /* GPU VA (== (NvU64)ptr for UVM-VAS host allocs by
                                  * Paper F1; distinct from ptr otherwise). */
    NvU64            size;
    bool             is_device;  /* true for mc_malloc_device (no CPU alias), false for host */
    bool             is_registered; /* true for mc_host_register: ptr is caller-owned
                                     * storage, so teardown must not munmap/free it. */
    mc_vas_t         vas;        /* which VAS the GPU mapping lives in.  Determines free
                                  * path: MC_VAS_UVM uses uvm_unmap_buffer;
                                  * MC_VAS_SYSMEM_CARRIER uses rm_unmap_memory_dma.
                                  * Default zero == MC_VAS_UVM, so any unpopulated
                                  * slot already reads as today's behavior. */
    NvU64            map_base;   /* page-aligned VA range described to RM/UVM for
                                  * mc_host_register; 0 for regular mc_malloc_* slots. */
    NvU64            map_size;
    int              uvm_segment_count;
    NvU64            uvm_segment_base[MC_UVM_EXTERNAL_SEGMENTS_MAX];
} mc_alloc_t;

/* Channel role — what the library uses a channel for.  Each VAS
 * exposes at most one channel per role; mc_memcpy dispatches by
 * (vas, mc_xfer_t) → role → mc_vas_find_channel.  Two channels with
 * the same role inside the same VAS is a bring-up bug (caught at
 * mc_init time).  Roles are orthogonal to mc_channel_type_t below:
 * type describes how the channel is brought up (RM class allocation
 * pattern); role describes how the library uses it. */
typedef enum {
    MC_ROLE_UVM_CE        = 0,    /* UVM-registered CE channel; mc_memcpy(MC_VAS_UVM) */
    MC_ROLE_HOST_DMA      = 1,    /* host-author CE channel inside a carrier VAS */
    MC_ROLE_SM_VICTIM_DMA = 2,    /* CE channel whose submissions are SM-authored */
    MC_ROLE_COMPUTE       = 3,    /* HOPPER_COMPUTE_A inside a carrier VAS */
    MC_ROLE_COUNT         = 4,
} mc_channel_role_t;

/* Channel kind — what kind of bring-up this channel uses.  Independent
 * of role: the sysmem-carrier reuses its HOST_DMA channel as the SM
 * victim too, while the FB-carrier has a dedicated SM_VICTIM_DMA
 * distinct from HOST_DMA. */
typedef enum {
    MC_CH_TYPE_UVM     = 0,
    MC_CH_TYPE_DMA     = 1,
    MC_CH_TYPE_COMPUTE = 2,
} mc_channel_type_t;

/* Maximum channels a single VAS may own.  UVM has 1 (UVM_CE);
 * sysmem-carrier has 2 (HOST_DMA + COMPUTE); the FB-carrier has 3
 * (HOST_DMA + SM_VICTIM_DMA + COMPUTE). */
#define MC_VAS_CH_MAX 3

/* ── VA-space objects ────────────────────────────────────────────────────
 * Two flavours: the UVM-managed VAS (allocated with IS_EXTERNALLY_OWNED
 * so UVM can install GPU MMU PTEs into it) and the carrier VAS (a plain
 * FERMI_VASPACE_A wrapping one NV50_MEMORY_VIRTUAL carrier per mapped
 * resource, which NV04_MAP_MEMORY_DMA installs PTEs into directly,
 * including the
 * BAR1 doorbell PTE).  Channels reference a VAS by id so an H2D channel
 * sharing the UVM VAS, or a future second compute channel sharing the
 * carrier, doesn't have to embed the same handles twice.
 *
 * mc_va_space_kind_t and mc_va_space_t live here where the VA-space
 * machinery itself is used; the per-VAS index type is the public
 * mc_vas_t (mc.h), used directly without an internal alias.
 */
typedef enum {
    MC_VAS_KIND_UVM,         /* IS_EXTERNALLY_OWNED + UVM-registered */
    MC_VAS_KIND_CARRIER,     /* per-resource NV50_MEMORY_VIRTUAL; sysmem channel resources */
    MC_VAS_KIND_CARRIER_FB,  /* per-resource NV50_MEMORY_VIRTUAL; FB-vidmem (BAR1-aliased)
                              * channel resources */
} mc_va_space_kind_t;

/* True for either carrier kind.  Used by helpers
 * (mc_va_space_dma_map_resource, mc_va_space_release_carrier,
 * doorbell-PTE install, channel-core bring-up) that share machinery
 * across the two carrier flavours. */
static inline bool mc_va_space_kind_is_carrier(mc_va_space_kind_t k) {
    return k == MC_VAS_KIND_CARRIER || k == MC_VAS_KIND_CARRIER_FB;
}

/* mc_va_space_t is defined further down: it embeds mc_channel_t
 * inline, so its full layout depends on mc_channel_t which itself
 * uses the per-kind extras structs declared below. */

/* ── Per-channel-kind extras ─────────────────────────────────────────────
 * Each kind owns a small struct with the state unique to it; the
 * mc_channel_t below holds them in a discriminated union keyed on
 * `type`, so call sites read `ch->x.uvm.field` / `ch->x.dma.field` /
 * `ch->x.compute.field` with compiler-checked access.  Fields are
 * flat, one-per-line.
 */
struct mc_uvm_extras {
    /* The UVM-managed VAS lives in ctx->vas[MC_VAS_UVM]; the UVM
     * channel just references it by id (mc_channel_t.vas_id). */
    NvHandle                       h_gpfifo_userd_mem;      /* 2 MiB vidmem: GPFIFO+USERD */
    void                          *gpfifo_userd_cpu;        /* BAR1 mmap of the above */
};

struct mc_dma_extras {
    /* The VAS / carrier / doorbell-PTE state lives in
     * ctx->vas[MC_VAS_SYSMEM_CARRIER].  The DMA channel's only
     * kind-specific data is the token cell — a small sysmem scratch
     * allocation used as the CE source for doorbell-write demos. */
    NvHandle                       h_token_mem;        /* sysmem hMemory backing the cell */
    volatile uint32_t             *token_cell;         /* CPU pointer */
    NvU64                          token_cell_gpu_va;  /* GPU VA in the carrier VAS */
};

/* A loaded compute kernel: SASS code + a CB0 image + a QMD scratch
 * region.  Today exactly one module is loaded per channel (the
 * embedded mc_doorbell_kernel); the separation from mc_compute_extras
 * means future multi-kernel scenarios add more modules without
 * touching the channel struct. */
struct mc_compute_module {
    NvHandle                       h_qmd_mem;          /* 2 MiB sysmem: QMD scratch */
    NvHandle                       h_cb0_mem;          /* 2 MiB sysmem: CB0 image */
    NvHandle                       h_sass_mem;         /* 2 MiB sysmem: SASS code */
    NvHandle                       h_scratch_mem;      /* BAR1-aliased vidmem scratch page */
    uint8_t                       *qmd_cpu;            /* 384 B QMD scratch */
    uint8_t                       *cb0_cpu;            /* CB0 image */
    uint8_t                       *sass_cpu;           /* SASS bytes */
    volatile uint32_t             *scratch_cpu;        /* host-readable scratch dword */
    NvU64                          qmd_gpu_va;
    NvU64                          cb0_gpu_va;
    NvU64                          sass_gpu_va;
    NvU64                          scratch_gpu_va;     /* GPU VA of *scratch_cpu */
};

struct mc_compute_extras {
    struct mc_compute_module       module;             /* mc_doorbell_kernel: 1-store SM kernel */
    struct mc_compute_module       sm_owner_module;    /* sm_owner_kernel: SM-authors-channel kernel */
    bool                           setup_done;         /* one-time-per-channel setup? */
};

/* ── Uniform per-channel struct ──────────────────────────────────────────
 * Holds the GPFIFO/pushbuffer/USERD/sema plumbing every channel needs,
 * plus a kind tag and a discriminated union of kind-specific extras.
 */
typedef struct {
    /* identity */
    mc_channel_role_t              role;              /* what mc_memcpy / mc_sm_owner_* dispatches to */
    mc_channel_type_t              type;              /* discriminates `x` below; selects bring-up shape */
    mc_vas_t                       vas_id;            /* index into mc_ctx.vas[] */
    NvU32                          subchannel;        /* NVA06F_SUBCHANNEL_* */
    NvU32                          engine_type;       /* NV2080_ENGINE_TYPE_* */

    /* RM handles (allocated only when the channel's init succeeds). */
    NvHandle                       h_tsg;
    NvHandle                       h_channel;
    NvHandle                       h_engine;          /* h_ce or h_compute */
    NvHandle                       h_gpfifo_mem;
    NvHandle                       h_pb_mem;
    NvHandle                       h_sema_mem;
    NvHandle                       h_sema_sysmem_mem; /* FB-victim only: sysmem-resident active sema (host-polled) */

    /* CPU pointers (BAR1 / sysmem mappings). */
    void                          *gpfifo_cpu;        /* base of the gpfifo+userd alloc */
    void                          *userd_cpu;         /* CPU base of the USERD region */
    volatile uint32_t             *gpfifo_ring;
    volatile HopperAControlGPFifo *userd;             /* typed alias of userd_cpu */
    uint32_t                      *pb_cpu;
    volatile uint32_t             *sema_ptr;
    volatile uint32_t             *sema_sysmem_ptr;   /* FB-victim only: CPU alias of sema_sysmem_gpu_va */

    /* GPU virtual addresses. */
    NvU64                          gpfifo_gpu_va;
    NvU64                          pb_gpu_va;
    NvU64                          sema_gpu_va;
    NvU64                          sema_sysmem_gpu_va;/* FB-victim only: sysmem-resident active sema (host polls for both H2D and D2H) */

    /* Hot-loop state. */
    NvU32                          work_submit_token;
    NvU32                          gp_put;            /* mutated by submit */
    NvU32                          sema_payload;      /* mutated by submit */

    /* Kind-specific extras.  Read the variant matching `type`. */
    union {
        struct mc_uvm_extras     uvm;
        struct mc_dma_extras     dma;
        struct mc_compute_extras compute;
    } x;
} mc_channel_t;

/* ── VA-space ─────────────────────────────────────────────────────────────
 * Each VAS owns its channels inline.  Bring-up populates
 * channels[0..channel_count-1]; mc_vas_find_channel scans the populated
 * prefix for a role match.  Two channels with the same role inside one
 * VAS is a bring-up bug (each (vas, role) tuple must resolve to one
 * channel; mc_vas_add_channel CHECKs this).
 */
/* One per-resource carrier record.  libcuda allocates a fresh
 * NV50_MEMORY_VIRTUAL per source hMemory, sized exactly to the
 * resource, with NV04_MAP_MEMORY_DMA(dmaOffset=0) — RM picks the GPU
 * VA from the FERMI VAS's heap.  We track each (h_carrier, h_mem,
 * gpu_va) so teardown can NVOS47-unmap and free both handles. */
typedef struct {
    NvHandle           h_carrier;       /* NV50_MEMORY_VIRTUAL */
    NvHandle           h_mem;           /* source hMemory (informational) */
    NvU64              gpu_va;          /* RM-chosen GPU VA from the NVOS46 */
    NvU64              size;            /* mapped length */
} mc_resource_carrier_t;

/* Per-VAS carrier-table capacity.  Sized for the worst case (FB-carrier
 * VAS): 2 modules × 4 buffers + 3 channels × 3 core buffers + token +
 * doorbell + N user allocs.  64 covers everything mc allocates
 * today plus a few dozen user buffers; if the user blows through this
 * the error is loud rather than a silent overwrite. */
#define MC_VAS_CARRIER_MAX 64

typedef struct {
    mc_va_space_kind_t kind;
    NvHandle           h_vaspace;       /* FERMI_VASPACE_A handle */

    /* Per-resource carriers — libcuda's shape (one NV50_MEMORY_VIRTUAL
     * per source hMemory, sized to the source).  Populated lazily by
     * mc_va_space_dma_map; freed in reverse order by mc_va_space_fini. */
    mc_resource_carrier_t carriers[MC_VAS_CARRIER_MAX];
    int                   carrier_count;

    NvU64              dbell_gpu_va;    /* BAR1 doorbell PTE address; 0 if none */

    /* Channels owned by this VAS.  Populated up to `channel_count` by
     * mc_init's per-VAS bring-up.  Zero-initialised channels (slots
     * past channel_count) are skipped by mc_vas_find_channel. */
    mc_channel_t       channels[MC_VAS_CH_MAX];
    int                channel_count;
} mc_va_space_t;

/* ── Library context ──────────────────────────────────────────────────────
 * All the state that survives across API calls.  Channels live inside
 * each mc_va_space_t (vas->channels[]); mc_vas_find_channel resolves
 * (vas, role) → mc_channel_t * for dispatch.
 */
struct mc_ctx {
    /* Device fds + UVM fd */
    int                            ctl_fd;
    int                            dev_fd;
    int                            uvm_fd;

    /* Global RM object handles */
    NvHandle                       h_client;
    NvHandle                       h_device;
    NvHandle                       h_subdevice;
    NvHandle                       h_usermode_bar0;
    NvHandle                       h_usermode_bar1;

    /* GPU UUID for UVM calls */
    NvU8                           gpu_inst_uuid[NV_UUID_LEN];

    /* CPU pointers shared across channels */
    volatile uint8_t              *usermode_bar0_cpu;
    volatile uint8_t              *usermode_bar1_cpu;
    volatile uint32_t             *vf_doorbell;

    /* Allocation table for mc_malloc / mc_free tracking */
    mc_alloc_t                     allocs[MC_ALLOC_TABLE_SLOTS];

    /* VA spaces, indexed by mc_vas_t.  Each VAS owns its
     * channels inline (mc_va_space_t.channels[]). */
    mc_va_space_t                  vas[MC_VAS_COUNT];
};

/* ── Cross-file helpers ──────────────────────────────────────────────
 * Shared between library .c files (mc_core.c, mc_rm.c, mc_uvm.c,
 * mc_submit.c, mc_vaspace.c, mc_compute_demo.c).  Not part of the
 * public API.
 */

/* Verbose logging gate (env: MC_VERBOSE).  Cached after first call.
 * Defined in mc_rm.c (always linked). */
int mc_debug(void);

/* Library-wide log macros.  All call sites expand to direct fprintf;
 * mc_debug() gates DEBUG output but the macro expands unconditionally
 * so log-string formatting cost vanishes when MC_VERBOSE is unset.
 *
 * Kept in this header (rather than per-file) so every TU sees the
 * same definition; otherwise a copy in each .c file would drift. */
#define DEBUG_LOG(...)                                       \
  do                                                         \
  {                                                          \
    if (mc_debug())                                          \
    {                                                        \
      fprintf(stderr, "[mc] DEBUG: " __VA_ARGS__);    \
      fputc('\n', stderr);                                   \
    }                                                        \
  } while (0)
#define WARN_LOG(...)                                        \
  do                                                         \
  {                                                          \
    fprintf(stderr, "[mc] WARN: " __VA_ARGS__);       \
    fputc('\n', stderr);                                     \
  } while (0)
#define ERROR_LOG(...)                                       \
  do                                                         \
  {                                                          \
    fprintf(stderr, "[mc] ERROR: " __VA_ARGS__);      \
    fputc('\n', stderr);                                     \
  } while (0)

/* Library-internal fatal check.  Library users should not have their
 * process aborted by library code, so this is used sparingly:
 * we log and abort only for invariant violations that indicate a bug
 * (asserts in a stricter language).  Setup/ioctl failures return
 * mc_status_t instead. */
#define CHECK(cond, ...)                                              \
  do                                                                  \
  {                                                                   \
    if (!(cond))                                                      \
    {                                                                 \
      fprintf(stderr, "[mc] FATAL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                                   \
      fputc('\n', stderr);                                            \
      abort();                                                        \
    }                                                                 \
  } while (0)

/* ── mc_rm.c — RM ioctl wrappers ────────────────────────────────── */
void     rm_free_handle(int ctl_fd, NvHandle h_root, NvHandle h_parent,
                        NvHandle h_target, const char *label);
NvHandle rm_alloc(int ctl_fd, NvHandle root, NvHandle parent,
                  NvU32 hclass, void *alloc_params);
NvHandle rm_alloc_vidmem(int ctl_fd, NvHandle root, NvHandle device,
                         NvU64 size, NvU64 *out_phys);
int      rm_register_client_fd(int ctl_fd, int dev_fd);
/* Cached (write-back) host memory — the default, and what every buffer a
 * user program reads should use. */
NvHandle rm_alloc_sysmem_at(int ctl_fd, int dev_fd, NvHandle root,
                            NvHandle device, NvU64 size, void *want_va,
                            void **out_cpu_va);
/* Write-combined host memory — control plane only (pushbuffer, GPFIFO,
 * USERD, semaphores, QMD/CB0/SASS).  Host-write, GPU-read; do not use for
 * anything the host reads back. */
NvHandle rm_alloc_sysmem_wc_at(int ctl_fd, int dev_fd, NvHandle root,
                               NvHandle device, NvU64 size, void *want_va,
                               void **out_cpu_va);
NvHandle rm_register_user_memory(int ctl_fd, int dev_fd, NvHandle root,
                                 NvHandle device, void *page_base,
                                 NvU64 page_covered_size);
/* Allocate a vidmem region and ALSO produce a host-visible BAR1
 * alias for it via NV04_MAP_MEMORY with NVOS33_FLAGS_MAPPING_REFLECTED.
 * Result: PBDMA / SM see the FB allocation through the GPU MMU; the
 * host gets `*out_cpu_va` pointing at the same bytes through BAR1.
 *
 * `want_va`, if non-NULL, anchors the BAR1 mmap into the Paper-F1 VA
 * pool (caller pre-reserves via va_pool_reserve).  Pass NULL to let
 * the kernel choose the CPU VA (rare; FB-carrier callers always
 * anchor).  Returns the new vidmem hMemory on success, 0 on failure
 * (no allocation leaks; the helper rolls back on partial failure). */
NvHandle rm_alloc_vidmem_bar1_at(int ctl_fd, const char *dev_path,
                                 NvHandle client, NvHandle device,
                                 NvU64 size, void *want_va,
                                 void **out_cpu_va);
NvHandle rm_alloc_vaspace(int ctl_fd, NvHandle root, NvHandle device);
NvHandle rm_alloc_vaspace_dma(int ctl_fd, NvHandle root, NvHandle device);
NvHandle rm_alloc_virtual_memory(int ctl_fd, NvHandle root, NvHandle device,
                                 NvHandle h_vaspace, NvU64 size,
                                 NvU64 *out_base);
NvU64    rm_map_memory_dma(int ctl_fd, NvHandle client, NvHandle device,
                           NvHandle h_virt, NvHandle h_memory,
                           NvU64 mem_offset, NvU64 size, NvU64 want_offset);
void     rm_unmap_memory_dma(int ctl_fd, NvHandle client, NvHandle device,
                             NvHandle h_virt, NvHandle h_memory,
                             NvU64 dma_offset);
void    *rm_map_memory_at(int ctl_fd, const char *dev_path, NvHandle client,
                          NvHandle parent, NvHandle h_memory,
                          NvU64 mem_offset, NvU64 size, NvU32 flags,
                          void *want_cpu_va);
void    *rm_map_memory(int ctl_fd, const char *dev_path,
                              NvHandle client, NvHandle parent,
                              NvHandle h_memory, NvU64 mem_offset,
                              NvU64 size, NvU32 flags);
int      rm_control(int ctl_fd, NvHandle client, NvHandle object, NvU32 cmd,
                    void *params, NvU32 params_size);
NvU32    pick_non_grce_lce(int ctl_fd, NvHandle h_client, NvHandle h_subdevice);
void     rm_perf_boost(int ctl_fd, NvHandle client, NvHandle subdevice);

NvHandle rm_alloc_root(int ctl_fd);
NvHandle rm_alloc_device(int ctl_fd, NvHandle root);
NvHandle rm_alloc_subdevice(int ctl_fd, NvHandle root, NvHandle device);
NvHandle rm_alloc_usermode(int ctl_fd, NvHandle root, NvHandle subdevice,
                           bool bar1_mapping);
NvHandle rm_alloc_tsg(int ctl_fd, NvHandle root, NvHandle device,
                      NvU32 engine_type, NvHandle vaspace);
NvHandle rm_alloc_channel(int ctl_fd, NvHandle root, NvHandle tsg,
                          NvU64 gp_fifo_offset, NvU32 gp_fifo_entries,
                          NvU32 engine_type, NvHandle h_userd,
                          NvU64 userd_offset);
NvHandle rm_alloc_ce(int ctl_fd, NvHandle root, NvHandle channel,
                     NvU32 engine_type);
NvHandle rm_alloc_compute(int ctl_fd, NvHandle root, NvHandle channel);
int      rm_gpfifo_schedule(int ctl_fd, NvHandle h_client, NvHandle h_channel);
void     rm_channel_disable(int ctl_fd, NvHandle h_client, NvHandle h_channel);

/* ── mc_uvm.c — UVM ioctl wrappers ──────────────────────────────── */
int      uvm_setup(int ctl_fd, int dev_fd, NvHandle h_client,
                   NvHandle h_subdevice, NvHandle h_vaspace,
                   NvU8 out_inst_uuid[NV_UUID_LEN], int *out_uvm_fd);
int      uvm_register_channel(int uvm_fd, int dev_fd, NvHandle h_client,
                              const NvU8 inst_uuid[NV_UUID_LEN],
                              NvHandle h_channel);
NvU64    uvm_map_buffer(int uvm_fd, int dev_fd, NvHandle h_client,
                        const NvU8 inst_uuid[NV_UUID_LEN], NvHandle h_memory,
                        NvU64 size, const char *label);
NvU64    uvm_map_buffer_at(int uvm_fd, int dev_fd, NvHandle h_client,
                           const NvU8 inst_uuid[NV_UUID_LEN], NvHandle h_memory,
                           void *want_va, NvU64 size, const char *label);
int      uvm_map_buffer_range_at(int uvm_fd, int dev_fd, NvHandle h_client,
                                 const NvU8 inst_uuid[NV_UUID_LEN],
                                 NvHandle h_memory, NvU64 base, NvU64 size,
                                 NvU64 offset, const char *label);
void     uvm_free_range(int uvm_fd, NvU64 base, const char *label);
void     uvm_unregister_channel(int uvm_fd, NvHandle h_client,
                                NvHandle h_channel);
void     uvm_unmap_buffer(int uvm_fd, const NvU8 inst_uuid[NV_UUID_LEN],
                          NvU64 gpu_va, NvU64 size, const char *label);
void     uvm_unregister_gpu_vaspace(int uvm_fd,
                                    const NvU8 inst_uuid[NV_UUID_LEN]);
void     uvm_unregister_gpu(int uvm_fd, const NvU8 inst_uuid[NV_UUID_LEN]);

/* ── mc_submit.c — method stream + submission primitives ───────── */
/* Build the NVC8B5 method stream for one CE copy.  Returns the
 * pushbuffer-end pointer (caller computes byte length as
 * (end - pb) * sizeof(uint32_t)). */
uint32_t *mc_write_transfer_methods(uint32_t *pb,
                                    uint64_t src_va, uint64_t dst_va,
                                    uint32_t nbytes,
                                    uint64_t sema_va, uint32_t sema_payload);

/* GPFIFO-entry builder + USERD GPPut bump.  Both used by submit
 * primitives but are useful standalone for tests/diagnostics. */
uint32_t write_gp_entry(volatile uint32_t *gpfifo_ring,
                        uint32_t gp_put_index, uint32_t ring_entries,
                        uintptr_t pb_cpu_va, uint32_t pb_offset,
                        uint32_t method_bytes);
void     ring_doorbell(volatile HopperAControlGPFifo *userd,
                       uint32_t new_gp_put,
                       volatile uint32_t *vf_doorbell,
                       uint32_t work_submit_token);

/* Wait for the channel to drain (USERD GPGet catches up to GPPut)
 * or timeout_ms elapses.  Used during channel teardown. */
void     drain_channel(volatile HopperAControlGPFifo *userd, long timeout_ms);

/* Channel submission primitives — see comments above the
 * implementations in mc_submit.c for the produce -> fence -> publish
 * contract. */
mc_status_t mc_channel_submit   (mc_channel_t *ch,
                                 volatile uint32_t *vf_doorbell,
                                 uint32_t copy_bytes);
mc_status_t mc_channel_arm      (mc_channel_t *ch, uint32_t copy_bytes,
                                 struct timespec *t0_out);
mc_status_t mc_channel_poll_sema(volatile uint32_t *cell, uint32_t expected,
                                 struct timespec t0);

/* ── mc_vaspace.c — VA pool + VA-space helpers ─────────────────── */
int    va_pool_init(void);
void  *va_pool_reserve(NvU64 size, const char *label);
int    mc_va_space_init_uvm(mc_ctx_t *ctx);
int    mc_va_space_init_carrier(mc_ctx_t *ctx);
int    mc_va_space_init_carrier_fb(mc_ctx_t *ctx);

/* Reserve the next free channels[] slot in `vas` for a channel with
 * role `role`.  Returns a pointer to the new slot (caller fills it
 * in) on success, or NULL if the VAS already owns a channel with
 * this role or has no free slot.  Increments vas->channel_count. */
mc_channel_t *mc_vas_add_channel(mc_va_space_t *vas, mc_channel_role_t role);

/* Look up the unique channel in `vas` whose role is `role`.  Returns
 * NULL if no channel with that role exists in this VAS.  O(MC_VAS_CH_MAX)
 * scan; tiny constant. */
mc_channel_t *mc_vas_find_channel(mc_va_space_t *vas,
                                  mc_channel_role_t role);
int    mc_va_space_install_doorbell_pte(mc_ctx_t *ctx, mc_va_space_t *vas);

/* DMA-map an existing hMemory into `vas` via libcuda's per-resource
 * carrier shape: allocate a fresh NV50_MEMORY_VIRTUAL sized to the
 * source, NV04_MAP_MEMORY_DMA with dmaOffset=0, return RM-chosen GPU
 * VA.  Records the carrier in vas->carriers[] for teardown.  Returns
 * 0 on failure. */
NvU64  mc_va_space_dma_map_resource(mc_ctx_t *ctx, mc_va_space_t *vas,
                                    NvHandle h_mem, NvU64 size);

/* Release the per-resource carrier mapping `h_mem` into `vas`:
 * NVOS47-unmap, NV01_FREE the carrier handle, and remove the entry
 * from vas->carriers[].  Used by channel teardown to detach a
 * channel's carriers BEFORE freeing the channel's h_mem (NVOS47 on a
 * freed h_mem returns INVALID_OBJECT_HANDLE).  No-op if h_mem isn't
 * tracked in this VAS.  Multiple matching entries (shouldn't happen
 * but cheap to handle) are all released. */
void   mc_va_space_release_carrier(mc_ctx_t *ctx, mc_va_space_t *vas,
                                   NvHandle h_mem);

NvU64  mc_va_space_alloc_scratch(mc_ctx_t *ctx, mc_va_space_t *vas,
                                 NvU64 size, NvU64 align,
                                 NvHandle *out_h_mem, void **out_cpu);
/* Write-combined scratch — control plane only; see rm_alloc_sysmem_wc_at. */
NvU64  mc_va_space_alloc_scratch_wc(mc_ctx_t *ctx, mc_va_space_t *vas,
                                    NvU64 size, NvU64 align,
                                    NvHandle *out_h_mem, void **out_cpu);
NvU64  mc_va_space_alloc_vidmem (mc_ctx_t *ctx, mc_va_space_t *vas,
                                 NvU64 size, NvU64 align,
                                 NvHandle *out_h_mem);
void   mc_va_space_fini(mc_ctx_t *ctx, mc_vas_t id);

/* ── mc_sm_owner.c — SM-authored submission helper ─────────────── */
/* Library-internal: invoked from mc_core.c::mc_memcpy when
 * the caller passes MC_XFER_SM.  Both gpu_va arguments must be valid
 * GPU VAs in the chosen carrier VAS; the caller is responsible for
 * alloc-table lookup and validation.
 *
 * `vas` selects which carrier to author into.  The submitter looks
 * up `(compute_ch, victim_ch)` from that VAS's role table:
 *   MC_VAS_SYSMEM_CARRIER → (COMPUTE, HOST_DMA)
 *   MC_VAS_FB_CARRIER     → (COMPUTE, SM_VICTIM_DMA)
 *
 * Functionally: launches the sm_owner kernel on the compute channel.
 * A single SM thread authors the entire CE-channel submission for
 * the victim channel — pushbuffer methods, GPFIFO entry, USERD
 * GP_PUT, BAR1 doorbell — then PBDMA wakes and runs the CE copy.
 *
 * Sema synchronization (unified across both VAS / both directions):
 * the active release-sema cell is always sysmem-resident, the host
 * polls it as ground truth after the compute report sema fires.
 * Compute-sema-fire only proves the SM kernel exited (the doorbell
 * was rung); the sysmem-sema poll proves PBDMA actually retired the
 * CE LAUNCH_DMA.  PCIe producer-side ordering of MWrs from one
 * Requester ID into sysmem gives the data-then-sema guarantee for
 * D2H; LAUNCH_DMA's FLUSH_ENABLE gives MRd-CplD-into-HBM-then-sema
 * for H2D.
 *
 * Returns MC_OK on success, MC_ETIMEOUT if the compute kernel itself
 * doesn't exit within MC_TIMEOUT_MS, MC_EHANG if the kernel exited
 * but the victim's CE LAUNCH_DMA never released the DMA sema. */
mc_status_t mc_sm_owner_submit(mc_ctx_t *ctx, mc_vas_t vas, uint64_t src_gpu_va,
                               uint64_t dst_gpu_va, uint32_t size_bytes);

/* Look up an mc_alloc_t slot by user pointer.  Returns NULL if `ptr`
 * was not returned by mc_malloc_* on this context (or `ptr` is NULL).
 * Cross-file callers (e.g. the chain-doorbell functions in
 * mc_compute_demo.c) use this to validate user-supplied pointer
 * pairs before dispatching a submission. */
mc_alloc_t *alloc_table_lookup(mc_ctx_t *ctx, const void *ptr);

#endif /* MC_INTERNAL_H */
