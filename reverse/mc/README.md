# libmc

A small CUDA-runtime-like library over the raw NVIDIA driver ioctl
surface — no `libcuda`, no `cudart`, no `nvcc` required at *runtime*
(but `nvcc` is needed at build time to compile the embedded sm_owner
kernel; the resulting cubin is committed as a hex header so a
production checkout builds without it).

The library wraps the raw ioctl-and-submit machinery behind a small
public API.  It supports:

- Synchronous **Host ⇄ Device byte copies** through a CE engine, using
  either a host-authored or SM-authored submission strategy.
- Allocation in either the **UVM-managed VA space** (Paper-F1
  anchored, single-pool ergonomics) or the **carrier VA space** (the
  non-UVM FERMI_VASPACE_A shared by the DMA + compute channels).
- A **GPU-resident submission** path: a HOPPER_COMPUTE_A SM thread
  authors the entire CE-channel submission (pushbuffer, GPFIFO entry,
  USERD GP_PUT, BAR1 doorbell) from device code.

## API (`mc.h`)

```c
/* Lifecycle. */
mc_status_t  mc_init(mc_ctx_t **out_ctx);
void         mc_fini(mc_ctx_t *ctx);

/* Allocator: both functions take a VA-space selector.
 *   MC_VAS_UVM            — UVM-managed VAS (Paper-F1 anchored for host allocs).
 *   MC_VAS_SYSMEM_CARRIER — carrier VAS with channel resources (PB/GPFIFO/
 *                           USERD/sema) in sysmem.  Reachable by the
 *                           SM-owner kernel.
 *   MC_VAS_FB_CARRIER     — sibling carrier with channel resources in FB
 *                           (HBM).  No PCIe round-trip on the SM's
 *                           critical path; the doorbell ring is the only
 *                           MMIO write per SM-authored submission.
 */
typedef enum {
    MC_VAS_UVM            = 0,
    MC_VAS_SYSMEM_CARRIER = 1,
    MC_VAS_FB_CARRIER     = 2,
} mc_vas_t;

void        *mc_malloc_device(mc_ctx_t *ctx, size_t n, mc_vas_t vas);
void        *mc_malloc_host  (mc_ctx_t *ctx, size_t n, mc_vas_t vas);
void         mc_free         (mc_ctx_t *ctx, void *p);
uint64_t     mc_gpu_va       (mc_ctx_t *ctx, const void *user_ptr);

/* Copy: both functions take an agent selector.
 *   MC_XFER_HOST — host writes the channel's pushbuffer + rings the
 *                  doorbell; CE engine performs the copy.
 *   MC_XFER_SM   — a single SM thread of the sm_owner kernel writes
 *                  the entire submission (PB + GPFIFO + USERD GPPut +
 *                  BAR1 doorbell) from device code.  Both pointers
 *                  must live in a carrier VAS (MC_VAS_SYSMEM_CARRIER
 *                  or MC_VAS_FB_CARRIER).
 */
typedef enum { MC_XFER_HOST = 0, MC_XFER_SM = 1 } mc_xfer_t;

mc_status_t  mc_memcpy(mc_ctx_t *ctx, const void *dst_dev, const void *src_host,
                       size_t n, mc_xfer_t agent);
```

One entry point covers both directions.  `mc_memcpy` looks each pointer up
in the allocation table, so it already knows which side is device and which
is host, and builds the method stream accordingly.  The parameter names
predate that unification and read more narrowly than the function behaves.

Plus several research-oriented helper entry points: `mc_dbell_demo_ring`,
`mc_memcpy_gpu_doorbell_ce`, `mc_memcpy_gpu_doorbell_sm`,
`mc_compute_doorbell_kernel`, `mc_compute_get_scratch`.  See `mc.h` for
contracts; each is a chain-of-evidence demo for a particular path the
library can drive (host-doorbell, CE-doorbell, SM-doorbell, etc.).

## VA spaces

Three VAS arms reachable from the public allocators:

- **`MC_VAS_UVM`** — UVM-managed `FERMI_VASPACE_A` allocated with
  `IS_EXTERNALLY_OWNED`.  For host allocs, the returned CPU pointer
  is **also the GPU VA** (Paper Finding 1: anchored into a 4-GiB
  pre-reserved VA pool at `0x200000000`).  Used by `mc_memcpy`
  through the *primary UVM channel* — same as today's vanilla CUDA
  H2D/D2H path.

- **`MC_VAS_SYSMEM_CARRIER`** — non-UVM `FERMI_VASPACE_A` whose channel
  resources (PB / GPFIFO / USERD / sema) live in sysmem.  Each
  NVOS46 mapping into the VAS allocates its own
  `NV50_MEMORY_VIRTUAL` carrier sized to the source `hMemory`, with
  `dmaOffset = 0` (RM picks the GPU VA).  This mirrors libcuda's
  per-resource shape; the older single-bump-allocator design is
  gone.  Holds the BAR1 doorbell PTE and is shared by the **DMA
  channel** (a second carrier-bound CE) and the **compute
  channel**.  For host allocs, CPU and GPU VAs are **distinct**
  (kernel-chosen CPU mapping; RM-chosen GPU VA); recover the GPU VA
  via `mc_gpu_va(ctx, ptr)`.  `mc_memcpy` on a carrier pair routes
  through the DMA channel; with `MC_XFER_SM`, the sm_owner kernel
  runs on the compute channel and authors the DMA channel's
  submission from device code.

- **`MC_VAS_FB_CARRIER`** — sibling carrier whose channel resources
  (PB / GPFIFO / USERD) live in **FB (HBM)**, not sysmem.  Same
  per-resource carrier shape as `MC_VAS_SYSMEM_CARRIER`, but three
  channels (HOST_DMA + SM_VICTIM_DMA + COMPUTE) where the sysmem
  carrier has two and reuses HOST_DMA as its SM victim.  The
  point is the SM-authored hot path: with everything in FB, an SM
  thread's pushbuffer / GPFIFO entry / USERD GPPut writes never
  cross PCIe — only the BAR1 doorbell ring at the end is an MMIO
  write.  The release semaphore for the SM-authored copy is
  sysmem-resident (allocated alongside the FB victim DMA channel)
  and the host polls it after the compute kernel returns:  PCIe
  producer-side ordering of MWrs from one Requester ID into
  sysmem (D2H) and `FLUSH_ENABLE` on the LAUNCH_DMA (H2D) make
  the sema observation a valid completion gate without an
  on-GPU spin-wait.

`mc_init` brings up all three carriers' channels (UVM, sysmem-DMA,
sysmem-compute, FB-HOST_DMA, FB-SM_VICTIM_DMA, FB-compute)
unconditionally.  Pointers from `mc_malloc_*(ctx, n, MC_VAS_UVM)`
are not addressable from carrier-bound channels, and vice versa;
`mc_memcpy` validates that both arguments share a VAS and returns
`MC_EINVAL` otherwise.

### Paper-F1 invariant

`mc_malloc_host(ctx, n, MC_VAS_UVM)` returns a pointer that is **both
the CPU VA and the GPU VA**.  This is a hard invariant baked into UVM's
external-range registration — violating it produces intermittent
GPU-MMU corruptions on H100 PCIe (25–34 % failure rate before this was
understood; see `docs/findings.md §13`).  The library anchors every
UVM-host allocation into the VA pool, so users don't think about it.

`mc_malloc_host(ctx, n, MC_VAS_SYSMEM_CARRIER)` and the
`MC_VAS_FB_CARRIER` variant are **not** F1-anchored: the returned
pointer is the CPU VA only.  The GPU VA is RM-chosen inside the
carrier and recoverable via `mc_gpu_va(ctx, ptr)`.

`mc_malloc_device` always returns a GPU-only VA cast to `void *`; not
CPU-dereferenceable in either VAS.

### Coherency note

The library handles ordering in both transfer arms.  Under
`MC_XFER_SM` the caller's buffer is only ever read by the Copy Engine,
and CE reads of sysmem are cache-coherent DMA on x86 — dirty CPU cache
lines are snooped, so no caller-side flushing is needed.  The channel
state the SM kernel itself reads (pushbuffer, GPFIFO, USERD,
semaphores, CB0) is library-owned write-combined memory made visible by
the library's own SFENCE discipline; callers never touch it.  See the
`mc_malloc_host` doc-comment in `mc.h`.

## Minimal user program

```c
#include "mc.h"
#include <string.h>

int main(void) {
    mc_ctx_t *ctx;
    mc_init(&ctx);

    size_t    n = 64 * 1024 * 1024;
    uint32_t *h = mc_malloc_host  (ctx, n, MC_VAS_UVM);
    uint32_t *d = mc_malloc_device(ctx, n, MC_VAS_UVM);

    memset(h, 0xAB, n);
    mc_memcpy(ctx, d, h, n, MC_XFER_HOST);
    memset(h, 0, n);
    mc_memcpy(ctx, h, d, n, MC_XFER_HOST);
    /* h now holds 0xAB bytes again. */

    mc_free(ctx, d);
    mc_free(ctx, h);
    mc_fini(ctx);
}
```

Build: `cc -Imc myapp.c -Llib -lmc -Wl,-rpath,'$ORIGIN/../lib'`.
From `reverse/` run `make libmc` or `make mc_demo` (see
`tests/mc/mc_demo.c` for the full-featured demo).

For the SM-authored path, allocate carrier-VAS buffers (sysmem or
FB) and pass `MC_XFER_SM`:

```c
mc_vas_t carrier = MC_VAS_SYSMEM_CARRIER;   /* or MC_VAS_FB_CARRIER */
void *d = mc_malloc_device(ctx, n, carrier);
void *h = mc_malloc_host  (ctx, n, carrier);
mc_memcpy(ctx, d, h, n, MC_XFER_HOST);   /* seed HBM */
mc_memcpy(ctx, h, d, n, MC_XFER_SM);     /* SM authors the D2H */
```

See `tests/mc/mc_carrier_demo.c` (use `--fb` to exercise the
FB-carrier path) and `tests/mc/mc_sm_owner_demo.c`.

## Scope / restrictions

- **Single-threaded.**  Calling any API from more than one thread at a
  time is undefined.  One `mc_ctx_t` per process; `mc_init` brings up
  the UVM channel plus three channels per carrier (HOST_DMA / one CE,
  SM_VICTIM_DMA / a second CE on the FB carrier, COMPUTE).
- **Synchronous.**  `mc_memcpy` returns only after the relevant
  semaphore fires or `MC_ETIMEOUT` (2 s) elapses.  No streams, no
  events.
- **One-shot init.**  `mc_init` reserves a process-global VA pool;
  calling it a second time in the same process is a no-op.
- **Size bounds.**  Per allocation and per transfer: 4 KiB ≤ n ≤ 4 GiB − 1.
  CE's `LINE_LENGTH_IN` is a 32-bit method; carrier-vidmem allocs
  additionally need 64-KiB or 2-MiB GPU-VA alignment depending on
  size (handled internally).
- **Environment.**  Set `MC_VERBOSE=1` for debug-level stderr output.
  Requires `sudo` for `/dev/nvidia0`, `/dev/nvidiactl`,
  `/dev/nvidia-uvm`.
- **MC_XFER_SM precondition.**  The kernel modules from this tree must
  be loaded with `nv_dbell_disable_intercept=1` so the kernel doorbell
  watchpoint shadow page doesn't absorb the SM-issued doorbell write.

## Reference — source layout

The library is split into eight translation units plus the embedded
SM-owner kernel.  Every `.c` includes `mc_internal.h`, which declares
the cross-TU helper API + log macros.

**Public surface:**

- `mc.h` — the whole public API.
- `mc_internal.h` — internal types (`mc_channel_t`, `mc_va_space_t`,
  `mc_alloc_t`, per-kind extras union) + cross-TU prototypes + log
  macros + `CHECK`.  Public consumers do not include this.

**Library translation units:**

- `mc_rm.c` — RM ioctl wrappers (root/device/subdevice/usermode/tsg/
  channel/ce/compute object allocators; `rm_alloc_vidmem`,
  `rm_alloc_sysmem_at`, `rm_alloc_vaspace*`, `rm_alloc_virtual_memory`,
  `rm_map_memory*`, `rm_control`, `pick_non_grce_lce`,
  `rm_gpfifo_schedule`, `rm_channel_disable`).  Also hosts
  `mc_debug()`.
- `mc_uvm.c` — UVM ioctl wrappers (`uvm_setup`,
  `uvm_register_channel`, `uvm_map_buffer*`, `uvm_unmap_buffer`,
  `uvm_unregister_*`).
- `mc_submit.c` — method-stream + GPFIFO entry builders
  (`mc_write_transfer_methods`, `write_gp_entry`, `ring_doorbell`) +
  channel submission primitives (`mc_channel_arm/submit/poll_sema`) +
  `drain_channel`.  No fds or RM handles opened here.
- `mc_vaspace.c` — VA pool (`va_pool_init`, `va_pool_reserve`) +
  carrier VA space helpers (`mc_va_space_init_uvm`,
  `mc_va_space_init_carrier`, `mc_va_space_init_carrier_fb`,
  `mc_va_space_dma_map_resource` (allocates a fresh
  `NV50_MEMORY_VIRTUAL` per source `hMemory`, NV04_MAP_MEMORY_DMA
  with `dmaOffset = 0`, RM picks the GPU VA),
  `mc_va_space_release_carrier` (NVOS47 + free, called by
  channel teardown before freeing the source `hMemory`),
  `mc_va_space_install_doorbell_pte`,
  `mc_va_space_alloc_scratch`, `mc_va_space_alloc_vidmem`,
  `mc_va_space_fini`).
- `mc_core.c` — lifecycle: per-kind channel init/fini, `mc_init`,
  `mc_fini`, the allocation table, `mc_malloc_*`, `mc_free`,
  `mc_gpu_va`, `mc_memcpy` (with VAS dispatch + agent dispatch),
  and the host-authored chain-of-evidence demos (`mc_dbell_demo_ring`,
  `mc_memcpy_gpu_doorbell_ce`).
- `mc_compute_qmd.{c,h}` — Hopper QMD V04 builder + embedded SASS for
  the simple `mc_doorbell_kernel`.
- `mc_compute_demo.c` — HOPPER_COMPUTE_A demos
  (`mc_compute_doorbell_kernel`, `mc_memcpy_gpu_doorbell_sm`,
  `mc_compute_get_scratch`).  Builds the per-channel one-time setup
  pushbuffer (cache invalidates + shader-memory windows).
- `mc_sm_owner.c` — SM-authored CE submission helper.  Library-internal
  (declared in `mc_internal.h`); reached through the public API via
  `mc_memcpy(..., MC_XFER_SM)`.  Wraps the embedded sm_owner kernel
  in a launch-and-poll path against the compute channel.

**Embedded sm_owner kernel:**

- `kernels/sm_owner.cu` — single-thread Hopper kernel that authors an
  entire CE-channel submission from one SM thread.  Uses `st.relaxed.sys.global`
  + `membar.sys` for fence discipline; each method-stream dword and
  the GPFIFO entry / USERD GPPut / BAR1 doorbell are emitted as
  system-scope strong-ordered global stores.  Build-time assertion in
  the Makefile validates that the cubin contains ≥ 19 such stores.
- `mc_sm_owner_args.h` — shared host/device argument-struct layout.
  The kernel takes a single `struct mc_sm_owner_args *` parameter
  (insulates the host glue from NVCC parameter-packing details).
- `mc_sm_owner_sass.h` — auto-generated by the Makefile from the cubin
  produced by `nvcc -arch=sm_90`.  Committed so a fresh checkout builds
  without `nvcc`; regenerated by `make mc/mc_sm_owner_sass.h`.

**Reference / cross-check:**

- `../../docs/mc_architecture.md` — the driver-level flow behind this
  API: every ioctl, every kernel path, and the bug log from bring-up.
