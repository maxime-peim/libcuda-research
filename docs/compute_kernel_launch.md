# Compute kernel launch — Architecture Deep-Dive

**Launching a real Hopper compute kernel from raw NVIDIA driver ioctls,
and using it to ring a BAR1 doorbell from a GPU SM thread**

This document is the "missing manual" for the compute path —
`reverse/mc/mc_compute_qmd.{c,h}` plus the compute-launch sections of
`reverse/mc/mc_core.c`. It covers the complete software
and hardware path from `mc_compute_doorbell_kernel()` to a single SM thread
executing `*dst = token` and exiting, plus the chain-of-evidence demo where
that store is itself a BAR1 MMIO doorbell write that wakes another channel's
PBDMA.

It is grounded in:

- The open-gpu-kernel-modules SDK headers (driver 610.43.02, Hopper / H100 PCIe).
- **Mesa NVK**, the Vulkan-on-Nouveau driver, used as ground truth for the
  Hopper QMD V04 layout and the minimum begin-compute pushbuffer.  Every
  `src/nouveau/…` path in this document is a path in the **Mesa** source
  tree, not in this one.
- A captured libcuda trace of `cudaLaunchKernel` running an identical
  source-level kernel, used as a sanity cross-check on field bit positions.
- Empirical observation: every fix in this document was confirmed by an
  Xid-signature change on the H100 test host between iterations.

**Companion docs:**
- `docs/mc_architecture.md` — the prerequisite "raw ioctl D2H" path
  whose primary channel and BAR1 mapping are reused here.
- `docs/gpfifo_pushbuffer_reference.md` — bit-exact formats of GPFIFO entries
  and CE methods. Compute methods (NVC7C0/NVCBC0) follow the same pushbuffer
  framing, just on subchannel 1 instead of 4.
- `docs/findings.md` — research log; see §12 for the kernel doorbell-watchpoint.

---

## Table of Contents

1. [Overview and end state](#1-overview-and-end-state)
2. [Why this is interesting](#2-why-this-is-interesting)
3. [Hopper compute software model](#3-hopper-compute-software-model)
4. [The compute channel — third channel in mc_ctx](#4-the-compute-channel--third-channel-in-mc_ctx)
5. [QMD V04 — the launch descriptor](#5-qmd-v04--the-launch-descriptor)
6. [Constant Buffer 0 — kernel arguments](#6-constant-buffer-0--kernel-arguments)
7. [Per-channel begin-compute pushbuffer](#7-per-channel-begin-compute-pushbuffer)
8. [Per-launch pushbuffer (SEND_PCAS_A)](#8-per-launch-pushbuffer-send_pcas_a)
9. [The SASS — `*dst = token` in 6 instructions](#9-the-sass--dst--token-in-6-instructions)
10. [SM thread rings BAR1 doorbell](#10-sm-thread-rings-bar1-doorbell)
11. [Bug log — three Xid signatures, three fixes](#11-bug-log--three-xid-signatures-three-fixes)
12. [Operational invariants](#12-operational-invariants)
13. [What we *did not* need (vs libcuda)](#13-what-we-did-not-need-vs-libcuda)
14. [Quick-reference: pushbuffer methods used](#14-quick-reference-pushbuffer-methods-used)

---

## 1. Overview and end state

`mc_compute_dbell_demo` and `mc_compute_dbell_chain_demo` are the
two reference programs. Both run on the H100 test host with zero Xid in dmesg.

**`mc_compute_dbell_demo`** — SM thread writes a sysmem cell:
```
sudo ./bin/mc_compute_dbell_demo
  before nearby cells: cell[+0] = 0xcafebabe ...
  after  nearby cells: cell[+0] = 0xdeadbeef  ← target
PASS: GPU SM thread wrote 0xdeadbeef to sysmem cell via
      mc_doorbell_kernel(dst, token) executed under HOPPER_COMPUTE_A.
```

**`mc_compute_dbell_chain_demo`** — SM thread rings BAR1 doorbell:
```
sudo ./bin/mc_compute_dbell_chain_demo
  mc_init ok — primary + dma + compute channels ready (size=4194304)
PASS: DRAM contains FILL_PATTERN — primary D2H ran
       Host did NOT ring primary's doorbell during the call.
       No copy engine was involved in the doorbell write.
       A HOPPER_COMPUTE_A SM thread's STG.E.STRONG.SYS to
       BAR1+0x90 physically landed on the H100 doorbell register.
```

The compute channel uses raw `/dev/nvidiactl` + `/dev/nvidia0` ioctls
only — no UVM, no libcuda, no CUDA runtime, no nvcc-launched kernel.
The kernel SASS itself is compiled offline by nvcc once and embedded
in `mc_compute_qmd.c` as a 256-byte byte array.

---

## 2. Why this is interesting

`mc`'s host-driven copy path (`mc_memcpy` with `MC_XFER_HOST`) proves
the host can drive the H100 to perform a copy-engine DMA without CUDA. That
uses MMIO, GPFIFO, USERD, and BAR0 — all "control plane" hardware.

The compute path is different:
- The **SM scheduler** (SKED) is involved, parsing a 384-byte Queue Meta
  Data (QMD) block in vidmem.
- The **constant-buffer cache** binds CB0 into per-CTA fast storage.
- An actual SASS program runs on the SMs.
- That program can issue normal global stores. **Including stores to
  BAR1-mapped addresses** — meaning a GPU thread can ring a doorbell
  the same way the host CPU does.

This last property is the architectural payoff. It collapses the
host-CPU-as-orchestrator assumption that runs through every "GPU
programming" tutorial. A compute kernel can submit work to other
channels without going through the host. Nouveau/NVK already implies
this is possible (its compute path uses a host-rung doorbell, but the
GPU MMU's ability to map BAR1 into a GPU VA is the same plumbing);
this branch's contribution is the end-to-end demonstration with no
copy-engine intermediary in the doorbell loop.

---

## 3. Hopper compute software model

A compute kernel on Hopper is dispatched via three layers:

```
       ┌─────────────────────────────────────────────────┐
       │  GPFIFO ring  (host writes 8-byte entries)      │
       └────────────────────────┬────────────────────────┘
                                ↓
       ┌─────────────────────────────────────────────────┐
       │  Pushbuffer  (method stream, ≤8 KiB per submit) │
       │    SET_OBJECT(HOPPER_COMPUTE_A=0xCBC0)           │
       │    [one-time per-channel setup methods]         │
       │    SEND_PCAS_A(qmd_va >> 8)                     │
       │    SEND_SIGNALING_PCAS2_B(0x0a)                 │
       │    SET_REPORT_SEMAPHORE_*  (release)            │
       └────────────────────────┬────────────────────────┘
                                ↓
       ┌─────────────────────────────────────────────────┐
       │  QMD (Queue Meta Data)  V04  - 384 bytes        │
       │    PROGRAM_ADDRESS  (kernel entry GPU VA)       │
       │    GRID_W/H/D, CTA_DIM_0/1/2                    │
       │    REGISTER_COUNT, SLM/SMEM, BARRIER_COUNT      │
       │    CONSTANT_BUFFER(0)  (CB0 GPU VA + size)      │
       └────────────────────────┬────────────────────────┘
                                ↓
       ┌─────────────────────────────────────────────────┐
       │  SM scheduler dispatches CTAs.  Each thread     │
       │  reads cb0 / global / shared / local memory.    │
       └─────────────────────────────────────────────────┘
```

Key class IDs on H100:

| Class | Hex | Header | Purpose |
|---|---|---|---|
| `HOPPER_COMPUTE_A` | `0xCBC0` | `clcbc0.h` | Compute engine class, bound via `SET_OBJECT` on subch 1 |
| `NVC7C0` | — | `clc7c0.h` | Hopper-inherited compute methods (SEND_PCAS_A, SET_QMD_VERSION, SET_REPORT_SEMAPHORE_*, etc.) |
| `NVB1C0` | — | `clb1c0.h` | Maxwell-B compute methods inherited (INVALIDATE_SKED_CACHES) |
| `NVA0C0` | — | `cla0c0.h` | Kepler compute methods inherited (INVALIDATE_SAMPLER_CACHE_NO_WFI, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI) |
| `QMDV04_00_*` | — | `clcbc0qmd.h` (not in this tree — see below) | QMD V04 bit-field positions (`MW(hi:lo)` macros) |

Subchannel layout (mc follows libcuda + NVK convention):
- subch 0 — graphics (unused here)
- **subch 1 — HOPPER_COMPUTE_A**
- subch 4 — HOPPER_DMA_COPY_A (CE)

The compute and CE paths are independent; the same channel can drive
both subchannels but mc uses three *separate* channels (primary
CE, secondary CE for the older chain demo, compute) for clean isolation.

---

## 4. The compute channel — third channel in mc_ctx

`mc_compute_channel_init` (in `mc_core.c`) builds a third channel
inside `mc_ctx_t`:

```
KEPLER_CHANNEL_GROUP_A   (TSG, engineType = NV2080_ENGINE_TYPE_GR0)
  └── HOPPER_CHANNEL_GPFIFO_A
       └── HOPPER_COMPUTE_A          ← class 0xCBC0
       └── (no CE — compute only)
```

Critical choices:

- **Cohabits the carrier VAS**, not the UVM-managed VAS. The
  compute channel's TSG and channel are allocated against
  `ctx->vas[MC_VAS_SYSMEM_CARRIER].h_vaspace` (a `FERMI_VASPACE_A`
  *without* `IS_EXTERNALLY_OWNED`).  This makes any buffer
  `rm_map_memory_dma`-mapped under `ctx->vas[MC_VAS_SYSMEM_CARRIER].h_virt`
  reachable from the SM. The same VAS already holds the BAR1
  doorbell GPU VA at `ctx->vas[MC_VAS_SYSMEM_CARRIER].dbell_gpu_va`,
  so the SM can reach BAR1 with no additional plumbing.

- **engineType = GR0**, not a CE engine. The QMD scheduler runs on
  the GR engine on Hopper, and PROMOTE_CTX is *not* required: a
  compute channel runs without a GR context buffer being promoted,
  which is what makes this path reachable without libcuda.

- **Per-kernel sysmem buffers** for QMD (384 B), CB0 (≥540 B), and
  SASS (256 B), grouped into a `mc_compute_module` (see `mc_internal.h`).
  These are allocated via `mc_va_space_alloc_scratch` and
  `rm_map_memory_dma`'d into the carrier VAS:

| Buffer | GPU VA | Use |
|---|---|---|
| QMD scratch | carved from carrier VAS via `mc_va_space_alloc_scratch` | rebuilt each launch |
| CB0 image   | carved from carrier VAS via `mc_va_space_alloc_scratch` | kernel args |
| SASS bytes  | carved from carrier VAS via `mc_va_space_alloc_scratch` | written once at init |

  Sysmem (not vidmem) is a deliberate choice — it lets the host
  inspect the QMD/CB0/SASS bytes without a D2H copy, which was
  invaluable during debugging. The cost is ~PCIe latency on first
  fetch; for our 1-thread kernel this is negligible.

- **Compute channel pushbuffer / GPFIFO / sema** — separate from
  primary and dma_ch, allocated the same way as for any GPFIFO
  channel.

The channel-init sequence and engine binding are the same as for
the secondary CE channel; only the engine class differs. See
`mc_compute_channel_init` in `mc_core.c`.

---

## 5. QMD V04 — the launch descriptor

The QMD is a 384-byte (96-dword) packed bit-field structure that
the SM scheduler reads to dispatch a kernel. Its layout for Hopper
is `QMDV04_00_*` in NVIDIA's `clcbc0qmd.h` — a header that does **not**
ship in the open kernel modules tree (the only QMD class header here is
`cla0c0qmd.h`, for Kepler), so the V04 ranges below follow the layout Mesa
NVK publishes, defined as `MW(hi:lo)` bit ranges over the 384-byte buffer
(LSB = bit 0 of byte 0).

### Ground truth: NVK's `Qmd4_0`

The simplest existence proof of "what fields actually need to be
set" is `Qmd4_0::fill_qmd` in Mesa's `src/nouveau/compiler/nak/qmd.rs`.
It calls **exactly nine setters**:

```rust
qmd.set_barrier_count(...)
qmd.set_global_size(w, h, d)
qmd.set_local_size(w, h, d)
qmd.set_prog_addr(addr)
qmd.set_register_count(n)
qmd.set_crs_size(0)             // call/return stack
qmd.set_slm_size(slm_size)      // shader local memory
qmd.set_smem_size(smem, ...)
for cb in cbufs: qmd.set_cbuf(idx, addr, size)
```

Plus the four `qmd_init!` defaults:
- `QMD_MAJOR_VERSION = 4`, `QMD_MINOR_VERSION = 0`
- `API_VISIBLE_CALL_LIMIT = NO_CHECK`
- `SAMPLER_INDEX = INDEPENDENTLY`

Nothing else. **No MME bytecode load, no SET_CWD_REF_COUNTER, no
"runtime helper QMD #2" pointer, no per-context VA fragments**, all
of which appear in libcuda's QMD bytes but are libcuda-internal
bookkeeping the SM does not require.

### mc's implementation

`mc_compute_qmd.c` follows NVK exactly. The core helper:

```c
static void qmd_set_bits(uint8_t *qmd, unsigned hi, unsigned lo,
                        uint64_t value);
```

writes a `MW(hi:lo)` range (handles cross-dword spans). Around it,
nine field setters mirror NVK's trait:

```c
void mc_qmd_init(uint8_t *qmd);                      // memset 0 + defaults
void mc_qmd_set_barrier_count(qmd, n);
void mc_qmd_set_global_size(qmd, w, h, d);
void mc_qmd_set_local_size(qmd, w, h, d);
void mc_qmd_set_prog_addr(qmd, va);
void mc_qmd_set_register_count(qmd, n);
void mc_qmd_set_slm_size(qmd, bytes);
void mc_qmd_set_smem_size(qmd, bytes);
void mc_qmd_set_cbuf(qmd, idx, addr, size);
```

For our `*dst = token` kernel:

```c
mc_qmd_init(qmd);
mc_qmd_set_global_size(qmd, 1, 1, 1);     // 1 CTA
mc_qmd_set_local_size (qmd, 1, 1, 1);     // 1 thread
mc_qmd_set_prog_addr  (qmd, sass_gpu_va);
mc_qmd_set_register_count(qmd, 8);        // SASS uses up to R5
mc_qmd_set_barrier_count (qmd, 0);
mc_qmd_set_slm_size      (qmd, 0);
mc_qmd_set_smem_size     (qmd, 0);
mc_qmd_set_cbuf(qmd, 0, cb0_gpu_va, MC_CB0_TOTAL_BYTES_ALIGNED);
```

The resulting QMD has **13 nonzero dwords out of 96**. Every nonzero
bit is one we explicitly wrote.

### Verifying the bit math

We cross-check `mc_qmd_set_prog_addr` and `mc_qmd_set_cbuf` against
captured libcuda QMD bytes from a real `cudaLaunchKernel` of the
same source-level kernel:

| Field | Macro | Byte offset | Captured value | Decoded |
|---|---|---|---|---|
| `PROGRAM_ADDRESS_LOWER` | `MW(1247:1216)` | `0x098` | `0xfb7b9d00` | low 32 of `0x7598fb7b9d00` ✓ |
| `PROGRAM_ADDRESS_UPPER` | `MW(1272:1248)` | `0x09c` | `0x00007598` | high 32 of same ✓ |
| `CB0_ADDR_LOWER_S6` | `MW(1567:1536)` | `0x0c0` | `0x63e8a000` | `(0x7598fa280000 >> 6)` low 32 ✓ |
| `CB0_ADDR_UPPER_S6` | `MW(1586:1568)` | `0x0c4` bits 0..18 | `0x1d6` | `((0x7598fa280000 >> 6) >> 32)` ✓ |

Every row above was checked by hand against the `mc_qmd_set_*` call
that produces it, which is what pins the bit math as byte-exact.

---

## 6. Constant Buffer 0 — kernel arguments

CUDA passes kernel parameters to the SM via `c[0x0]` ("constant bank
zero"). nvcc-compiled SASS reads:

- `c[0x0][0x208]` — global memdesc base (must be 0 for raw-VA mode)
- `c[0x0][0x210/0x214]` — first 64-bit argument (kernel `dst`)
- `c[0x0][0x218]` — second 32-bit argument (kernel `token`)

In mc we allocate a single CB0 buffer per channel,
zero-initialise it, and patch only the three argument dwords:

```c
void mc_cb0_init(uint8_t *cb0)
{
    memset(cb0, 0, MC_CB0_TOTAL_BYTES_ALIGNED);
}

void mc_cb0_set_args(uint8_t *cb0, uint64_t dst_gpu_va, uint32_t token)
{
    *(uint32_t *)(cb0 + 0x210) = (uint32_t)dst_gpu_va;
    *(uint32_t *)(cb0 + 0x214) = (uint32_t)(dst_gpu_va >> 32);
    *(uint32_t *)(cb0 + 0x218) = token;
}
```

CB0's GPU VA is bound into the QMD via `mc_qmd_set_cbuf(qmd, 0,
cb0_gpu_va, size)` (NVK's `set_cbuf` with `SHIFTED6` addr,
`SHIFTED4` size), and the SM front-end fetches it on warp dispatch.

**libcuda's CB0 ≠ ours.** A captured libcuda CB0 image is 132 dwords
of mostly per-process VA prefixes (e.g. self-pointers, signature
checksums). The SASS doesn't read most of them — only the three
argument dwords matter. We zero everything else; PASS confirms this
is fine.

---

## 7. Per-channel begin-compute pushbuffer

Before the first `SEND_PCAS_A` on a compute channel, several methods
must be issued once.  NVK's recipe — `nvk_cmd_bind_compute_shader` plus
`nvk_cmd_buffer_begin_compute` in Mesa's
`src/nouveau/vulkan/nvk_cmd_dispatch.c` — is the minimum:

| Method | Class | Purpose |
|---|---|---|
| `SET_OBJECT` | C86F | Bind HOPPER_COMPUTE_A to subch 1 |
| `INVALIDATE_SKED_CACHES` | NVB1C0 | Clear stale QMD scheduler cache |
| `INVALIDATE_SAMPLER_CACHE_NO_WFI(LINES_ALL)` | NVA0C0 | Per-cmdbuf compute begin-state |
| `INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI(LINES_ALL)` | NVA0C0 | Per-cmdbuf compute begin-state |
| `SET_SHADER_SHARED_MEMORY_WINDOW_A/B` | NVC7C0 | Hopper requires nonzero, 4 GiB-aligned |
| `SET_SHADER_LOCAL_MEMORY_WINDOW_A/B` | NVC7C0 | Reservation for thread-local stack |

`mc_write_compute_setup_methods` (in `mc_compute_demo.c`) emits this
sequence verbatim. It runs once per channel, gated by
`compute_ch.setup_done`.

### The `SHADER_SHARED_MEMORY_WINDOW` choice

Hopper SMs resolve every memory access via three "windows":

```
if  va ∈ [SMEM_window_base, SMEM_window_base + 4 GiB):
        route to per-CTA shared memory
elif va ∈ [LMEM_window_base, LMEM_window_base + 16 MiB):
        route to per-thread local memory
else:
        route to GMMU as a global virtual address
```

Both window widths are *fixed by the architecture* on Hopper; only
their bases are configurable. NVK uses `SMEM_base = 1ULL << 32 =
0x100000000` and notes "shared memory window needs 4GB alignment on
hopper+" (in `nvk_cmd_buffer_begin_compute`). NVK's allocator deliberately
keeps `[1<<32, 2<<32)` free.

mc's `dma_ch` carrier lands at `virt_base = 0x120000000`,
right inside that range. Picking NVK's default would route every
global STG to SMEM, where offsets > 228 KiB (the H100 max per-CTA
SMEM) abort the warp with "Invalid Address Space" (Xid 13 — see §11).

**Fix:** SMEM window base = `0x400000000` (16 GiB, 4 GiB-aligned).
This puts the entire 4 GiB SMEM aperture above any VA mc
currently allocates. The LMEM window stays at NVK's default
`0xff << 24 = 0xff000000`.

```
0x000000000  ┌─────────────────────────────┐
             │ unused                      │
0x0ff000000  ├─────────────────────────────┤
             │ LMEM window (16 MiB)        │  ← NVK default
0x100000000  ├─────────────────────────────┤
             │ unused                      │
0x120000000  ├─────────────────────────────┤
             │ mc dma_ch carrier           │  ← qmd, cb0, sass, dst cell
0x140000000  ├─────────────────────────────┤
             │ unused                      │
0x200000000  ├─────────────────────────────┤
             │ mc va_pool (4 GiB)          │  ← UVM-mapped buffers
0x300000000  ├─────────────────────────────┤
             │ unused                      │
0x400000000  ├─────────────────────────────┤
             │ SMEM window (4 GiB)         │  ← bumped here
0x500000000  ├─────────────────────────────┤
             │ free                        │
```

An equally valid alternative would be to refactor `rm_alloc_carrier`
to place dma_ch above `0x200000000`, freeing `1<<32` for SMEM. We
chose to bump the window because it's a one-line change with no
risk to the working d2h-only path.

### What we *don't* emit

Compared to a captured libcuda begin-compute pushbuffer, we drop:

- `SET_SPA_VERSION = 0x00000900`
- undocumented `method_0x07ac = 1`
- `SET_QMD_VERSION = 0x00010040`
- `SET_CWD_REF_COUNTER` × 64 (a credit-counter ramp)
- `SET_RESERVED_SW_METHOD07 = 1`
- `SET_VALID_SPAN_OVERFLOW_AREA_A/B/C`
- `SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C = 0x39`
- `LOAD_MME_INSTRUCTION_RAM` × 39 dwords (libcuda's MME bytecode)
- `LOAD_MME_START_ADDRESS_RAM`
- `SET_TEX_HEADER_POOL_A/B/C` (we use no textures)
- `SET_TEX_SAMPLER_POOL_A/B/C` (ditto)

NVK emits **none** of these and launches Hopper compute kernels
correctly. Each was tested by removal: PASS.

---

## 8. Per-launch pushbuffer (SEND_PCAS_A)

Per-launch the work is just three method writes plus a release
semaphore (`mc_write_compute_launch_methods` in `mc_compute_demo.c`):

```c
SEND_PCAS_A(qmd_va >> 8)              // method 0x02b4
SEND_SIGNALING_PCAS2_B(0x0a)          // method 0x02c0; PCAS_ACTION is one 4-bit
                                      // enum: 0x0a = PREFETCH_SCHEDULE
SET_REPORT_SEMAPHORE_A(va_upper)      // methods 0x1b00..0x1b0c
SET_REPORT_SEMAPHORE_B(va_lower)
SET_REPORT_SEMAPHORE_C(payload)
SET_REPORT_SEMAPHORE_D(RELEASE | ONE_WORD | FLUSH_DISABLE=FALSE)
```

`SEND_PCAS_A` takes the QMD's GPU VA shifted right 8 bits — the QMD
must therefore be 256-byte-aligned (the SDK header's
`_QMD_ADDRESS_SHIFTED8` field name reflects this). Our QMD is in
`MC_GPFIFO_USERD_SIZE` (2 MiB) sysmem, naturally aligned.

`SEND_SIGNALING_PCAS2_B` is the actual scheduler kick. NVK uses
`PCAS_ACTION_INVALIDATE_COPY_SCHEDULE = 0x3`; libcuda uses `0x0a`
(SCHEDULE plus an additional bit). We use `0x0a` for parity with
the captured libcuda trace; both have been observed to work on this
driver/GSP version. If you see "kernel didn't dispatch" with no Xid,
trying `0x3` is one obvious knob.

`SET_REPORT_SEMAPHORE_*` writes a 4-byte completion signal into
sysmem (the sema_payload value into sema_gpu_va) when the kernel
retires. With `FLUSH_DISABLE = FALSE`, the release waits on L2
flush — meaning by the time the host sees the new payload value,
any global stores the kernel issued (including STGs to BAR1) have
physically reached their destinations.

**SET_OBJECT is not repeated per launch.** NVK does it once at
channel init and trusts it persists. We do too.

---

## 9. The SASS — `*dst = token` in 6 instructions

Our kernel:

```c
extern "C" __global__
void mc_doorbell_kernel(volatile unsigned int *dst, unsigned int token)
{
    *dst = token;
}
```

Compiled offline (`nvcc -arch=sm_90 --cubin -O0`) and disassembled
with `cuobjdump --dump-sass`:

```
LDC    R1, c[0x0][0x28]                  ; stack pointer (boilerplate)
LDC    R5, c[0x0][0x218]                 ; R5 = token
ULDC.64 UR4, c[0x0][0x208]               ; UR4 = global memdesc base
LDC.64 R2, c[0x0][0x210]                 ; R2 = dst (64-bit)
STG.E.STRONG.SYS desc[UR4][R2.64], R5    ; *dst = R5
EXIT
BRA self  ; + NOP padding to 256 bytes
```

The interesting instruction is the STG. On Hopper:
- **`STG.E.STRONG.SYS`** — system-coherent strong global store. The
  store is fenced wrt the warp's exit and (with the report-semaphore
  release using `FLUSH_DISABLE=FALSE`) reaches DRAM/PCIe before the
  release sema fires.
- **`desc[UR4][R2.64]`** — descriptor + offset addressing. With
  `UR4 = 0` (CB0[0x208] is zero in our kernel and in libcuda's), the
  hardware degenerates to "use R2.64 as a raw 64-bit virtual
  address", which is exactly what we want.

The 256 SASS bytes are committed verbatim in
`mc_compute_qmd.c::mc_doorbell_kernel_sass[]`.

---

## 10. SM thread rings BAR1 doorbell

`mc_compute_dbell_demo` proved an SM thread can store to a sysmem
cell.  The chain demo makes the *target* of that store a BAR1-mapped
MMIO doorbell — specifically
`ctx->vas[MC_VAS_SYSMEM_CARRIER].dbell_gpu_va + 0x90`, the
HOPPER_USERMODE_A VF doorbell page.

The chain (`mc_memcpy_gpu_doorbell_sm` in `mc_compute_demo.c`):

```
HOST                           GPU compute                   GPU primary
────                           ───────────                    ───────────
arm primary CE pushbuffer
  (HBM→DRAM 4 MiB + sema rel)
write primary GPFIFO entry
advance primary USERD GPPut
                                                              (PBDMA idle —
                                                               no doorbell yet)
mc_compute_doorbell_kernel(
  dst = ctx->vas[MC_VAS_SYSMEM_CARRIER].dbell_gpu_va + 0x90,
  token = primary_work_submit_token)
  ↓
build QMD + push setup methods
push SEND_PCAS_A + sema rel
ring COMPUTE doorbell  ────→   PBDMA wakes
                                dispatch QMD to SKED
                                SKED reads PROGRAM_ADDRESS,
                                  CB0_ADDR, dispatches 1 CTA
                                SM warp executes:
                                  R5 = primary_work_submit_token
                                  R2 = 0x120c00090
                                  STG.E.STRONG.SYS [R2.64], R5
                                                       ↓
                              GPU MMU resolves 0x120c00090
                                → BAR1 USERMODE_A page
                                → PCIe MMIO write
                                                       ───→ primary BAR1 +0x90
                                                            sees write
                                                            PBDMA wakes
                                                            fetches GPFIFO
                                                            runs CE LAUNCH_DMA
                                                            HBM streams to DRAM
                                                            sema fires
                                kernel exits, sema fires
poll compute sema (returned)
poll primary sema (returned)
verify DRAM contents
```

Because the host *never rings primary's doorbell*, and *no copy
engine is in the doorbell loop*, the only path that could have
woken primary's PBDMA is the SM-issued MMIO write physically
landing on the H100's BAR1 register. PASS proves it did.

The implementation is **88 lines** in `mc_core.c`. It calls the
existing `mc_compute_doorbell_kernel` unchanged — that kernel's API
was deliberately parameterized so this transition is a one-line
delta in the test. The minimal-delta nature of the change is itself
a sanity check on the architecture.

---

## 11. Bug log — three Xid signatures, three fixes

The investigation went through three observable failure modes, each
peeled back by one fix. Listed in encounter order; cumulative state
when we hit each one is described.

### Bug #1 — Xid 31 SKED@VA0 (libcuda template's hidden NULL pointers)

**Signature:**
```
NVRM: Xid (PCI:0000:23:00): 31, ... MMU Fault: ENGINE GRAPHICS
HUBCLIENT_SKED faulted @ 0x0_00000000. FAULT_PDE
ACCESS_TYPE_VIRT_READ
```

**State at observation:** Initial template-based QMD (from a captured
libcuda trace), with `PROGRAM_ADDRESS`, `PREFETCH`, `CB0_ADDR`
patched to mc VAs.

**Root cause:** The captured 384-byte QMD contained ~30 dwords of
libcuda-process-specific values whose meaning we hadn't decoded —
including bytes `0x040..0x05c` with values like `0x90000000`,
`0x03000640`, `0x9e89bc040040`, `0x04437f7c`. At least one of these
is interpreted by SKED as a pointer; in libcuda's process those VAs
exist, in mc's process they don't (or are 0), so SKED
dereferences NULL.

**Fix:** Rewrite QMD construction NVK-style — start from `[0; 96]`
zero buffer and only set the 9 fields NVK proves are required. See
§5. Result: 13 nonzero dwords vs ~50, every one we explicitly
wrote.

### Bug #2 — Xid 13 SM Warp Exception, ESR=0x10 (SMEM window collision)

**Signature:**
```
NVRM: Xid (PCI:0000:23:00): 13, ... Graphics SM Warp Exception on
(GPC 1, TPC 0, SM 0): Invalid Address Space
NVRM: ... ESR 0x515730=0x10
```

**State at observation:** QMD now NVK-style; `SHARED_MEMORY_WINDOW`
set to NVK's default `1ULL << 32`. **Xid 31 gone**, replaced by Xid
13 — meaning the SM is now actually dispatching warps and
*executing the kernel*, but the STG faults.

**Root cause:** Hopper's SMEM window is 4 GiB wide. With base
`0x100000000`, the range `[0x100000000, 0x200000000)` is interpreted
as SMEM. Our `dst_gpu_va = 0x120c00040` (in dma_ch's carrier) falls
inside it, so the SM treats the STG as a SMEM offset of
`0x20c00040 ≈ 525 MiB` — far past the 228 KiB max per-CTA SMEM →
"Invalid Address Space".

**Fix:** Bump `SHARED_MEMORY_WINDOW` base to `0x400000000` (16 GiB,
4 GiB-aligned), parking the entire SMEM aperture above all
mc VAs. See §7's address-space diagram.

### Bug #3 — sentinel intact despite kernel running (CPU cache, not a GPU bug)

**Signature:** No Xid. `mc_compute_doorbell_kernel` returns MC_OK
(release sema fires). Test reads `*cell_cpu` and sees the
pre-launch sentinel, not `token`.

**State at observation:** SMEM window fixed; kernel actually
executing; STG actually completing.

**Root cause:** The sysmem page backing the cell is mapped
**writeback-cached** on the CPU side. The test pre-fills the cell
with a sentinel via `cell_cpu[i] = ...`, which loads the line into
this core's L1 in **Modified** state. The GPU's STG.E.STRONG.SYS
writes the same line in DRAM, but the snoop traffic doesn't
reliably invalidate the line in L1 fast enough (or speculative
prefetch re-caches it before the host load). `volatile uint32_t *`
forces the compiler to emit a load instruction every read, but
that load can still hit the cached line. `mfence` orders memory
operations w.r.t. each other but does not touch cache state, so
adding `mfence` alone does not help.

**Fix:** `__builtin_ia32_clflush(cell) + mfence` before reading
forces the line back to DRAM (or invalidates it), so the next
load misses cache and returns the GPU-written bytes. See
`mc_compute_dbell_demo.c`.

**Empirically measured at N=20 on the H100 test host:**

| Configuration | PASS rate |
|---|---|
| `volatile` only | not measured separately, expected ≈ mfence-only |
| `volatile` + `mfence` (no clflush) | **1 / 20** |
| `volatile` + `clflush` + `mfence` | **20 / 20** |

The ~5% PASS rate without clflush is what makes this bug
particularly nasty — single-shot manual testing will appear to
"work" with high probability while the underlying race is real
and reproducible at modest sample sizes. This is the same lesson
already learned during the D2H bring-up, where an N=100 retest
uncovered a bug that every single-shot run had missed:
**N=1 PASS is not evidence in this project.**

Why other paths in the codebase don't hit this:

- The release-semaphore poll in `mc_compute_doorbell_kernel`
  reads from `*ctx->compute_ch.sema_ptr` (offset 0). The sema
  cell sits in a *different* 64-byte cache line than the test
  scratch (which is at offset 0x40). The host never wrote the
  sema line beforehand, so its L1 state stays Invalid /
  Shared, and a fresh load misses cache and hits DRAM. The
  bug is specific to **cells the host wrote and then expects
  the GPU to overwrite within the same cache window**.
- mc's primary-channel D2H test reads `h_buf` (a fresh
  UVM-mapped sysmem buffer), not a host-pre-written one.
- The CE-engine variants of the chain demo write to
  host-zeroed `h_buf`; same situation.

Future work: try mapping the per-channel sysmem with
write-through or uncached PAT attributes on the CPU side; that
would obviate clflush at the cost of slower CPU stores.  Out
of scope for the demo.

### The Xid progression as diagnostic

The Xid signature changed cleanly across iterations:

| Run | Xid | Layer being tested | What it tells you |
|---|---|---|---|
| Pre-rewrite (template) | 31 SKED@VA0 | QMD parsing | Scheduler hit NULL pointer in QMD itself |
| Post-rewrite, NVK SMEM_WINDOW | 13 SM Warp ESR=0x10 | SM execution | Warp dispatched, STG executed, address mapped wrong |
| Post-bump SMEM_WINDOW | (none) | DRAM | STG reached DRAM but host cache hid it |
| Post-CLFLUSH | (none) | host | PASS |

Each fix peeled back one layer. That's the value of leaving
detailed Xid logs in dmesg during this kind of bring-up — the
specific number is a precise diagnostic of which subsystem failed.

---

## 12. Operational invariants

These are required for the demos to PASS:

- **Build the modules from this tree.** The kernel-side functional
  workaround in `nv_gpu_ops.c` (engine-type lookup on Hopper) is
  required for the underlying `mc` D2H path to work; if the loaded
  `nvidia.ko` doesn't match this tree, channel allocation will fail.

- **`/sys/module/nvidia/parameters/nv_dbell_disable_intercept = 1`**
  (set via `/etc/modprobe.d/nvidia-dbell-bypass.conf` on the H100 test host).
  The kernel doorbell-watchpoint shadow page diverts BAR1 writes —
  including GPU-issued ones from `STG.E.STRONG.SYS`. With
  `disable_intercept=1`, BAR1 is reachable as expected. See
  `findings.md §12` for the watchpoint design.

- **No competing X11/ollama processes** if measuring with the
  watchpoint enabled. x86 has only 4 hardware debug registers and
  ollama / Xorg consume them. Not relevant for the disabled-watchpoint
  case (which the demos use), but worth knowing.

- **Run on the H100 test host over SSH.** Local runs (workstation) produce 100%
  ENOGPU because no H100 is present.

---

## 13. What we *did not* need (vs libcuda)

For people accustomed to libcuda's stack, the absences below are
striking. None of these are required to launch a Hopper compute
kernel that does `*dst = token`:

| Not used | What libcuda does with it | Why we don't need it |
|---|---|---|
| `KEPLER_GRAPHICS_CONTEXT` (class 0xa297) | Allocated and PROMOTE_CTX'd before any compute work | Hopper's GR engine implicitly initialises per-context state at HOPPER_COMPUTE_A construction |
| `NV2080_CTRL_GPU_PROMOTE_CTX` | Per-channel context promotion | See above |
| `NV04_MAP_MEMORY_DMA` (escape 0x2e) | Unused on Hopper / driver 610 (libcuda goes through UVM for everything) | mc's compute channel uses dma_ch's NV50_MEMORY_VIRTUAL carrier, which IS one DMA-mapped object — but that's already there for the d2h path |
| `LOAD_MME_INSTRUCTION_RAM` × 39 dwords | libcuda preloads its MME bytecode at channel init | NVK ships zero MME bytecode for Hopper compute |
| `SET_CWD_REF_COUNTER` × 64 | Per-SM credit-counter ramp from 0x001c803f down to 0x001c8000 | NVK never writes this method |
| `SET_QMD_VERSION = 0x00010040` | Tells the front-end the QMD format | The QMD's own `QMD_MAJOR_VERSION` / `QMD_MINOR_VERSION` fields convey this |
| `SET_SPA_VERSION = 0x00000900` | "Shader Plus Architecture" version | NVK skips it |
| Texture/sampler pool methods | Bind a texture descriptor heap | We use no textures |
| The "runtime helper QMD #2" | libcuda launches *two* QMDs per `cudaLaunchKernel` — user kernel + a runtime prologue/epilogue with embedded SASS | Our kernel's SASS does its own prologue/epilogue; nothing to call back into |
| The 132-dword libcuda CB0 image | Bindless heap descriptors, signature dwords, internal self-pointers | Our SASS reads only `c[0x0][0x208/0x210/0x214/0x218]` — three dwords for args, one zero for `desc[]` raw mode |

Lesson: **a "minimum viable Hopper compute launch" is *much* smaller
than what a real CUDA runtime emits.** Most of libcuda's per-channel
setup is plumbing for features (managed memory, multi-stream
scheduling, NVCC-emitted MME-callable functions, CUDA Graphs, ...)
that a hand-rolled launcher doesn't need.

---

## 14. Quick-reference: pushbuffer methods used

For lookup. All methods are emitted on subch 1 (`NVA06F_SUBCHANNEL_COMPUTE`).

### Per-channel begin-compute (one-time)

| Method | Address | Header |
|---|---|---|
| `SET_OBJECT(HOPPER_COMPUTE_A)` | `0x0000` | `clc86f.h` |
| `INVALIDATE_SKED_CACHES(0)` | `0x0298` | `clb1c0.h` |
| `INVALIDATE_SAMPLER_CACHE_NO_WFI(LINES_ALL=0)` | `0x1424` | `cla0c0.h` |
| `INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI(LINES_ALL=0)` | `0x0244` | `cla0c0.h` |
| `SET_SHADER_SHARED_MEMORY_WINDOW_A/B` | `0x02a0`/`0x02a4` | `clc7c0.h` |
| `SET_SHADER_LOCAL_MEMORY_WINDOW_A/B` | `0x07b0`/`0x07b4` | `clc7c0.h` |

### Per-launch

| Method | Address | Header |
|---|---|---|
| `SEND_PCAS_A(qmd_va >> 8)` | `0x02b4` | `clc7c0.h` |
| `SEND_SIGNALING_PCAS2_B(0x0a)` | `0x02c0` | `clc7c0.h` |
| `SET_REPORT_SEMAPHORE_A/B/C/D` | `0x1b00..0x1b0c` | `clc7c0.h` |

### Files in this branch

| File | Role |
|---|---|
| `reverse/mc/mc_compute_qmd.h` | Public API for QMD setters + SASS bytes |
| `reverse/mc/mc_compute_qmd.c` | Implementation of QMD setters; embedded SASS (256 bytes) |
| `reverse/mc/mc_core.c::mc_compute_channel_init` | Compute channel allocation + buffer mapping |
| `reverse/mc/mc_compute_demo.c::mc_write_compute_setup_methods` | Begin-compute pushbuffer |
| `reverse/mc/mc_compute_demo.c::mc_write_compute_launch_methods` | Per-launch pushbuffer |
| `reverse/mc/mc_compute_demo.c::mc_compute_doorbell_kernel` | Top-level launch API |
| `reverse/mc/mc_compute_demo.c::mc_memcpy_gpu_doorbell_sm` | End-to-end chain (SM rings BAR1 doorbell) |
| `reverse/tests/mc/mc_compute_dbell_demo.c` | SM-writes-sysmem-cell test |
| `reverse/tests/mc/mc_compute_dbell_chain_demo.c` | SM-rings-BAR1-doorbell chain test |

### External ground truth referenced

| Source | Path | Used as |
|---|---|---|
| Mesa NVK QMD | Mesa's `src/nouveau/compiler/nak/qmd.rs` | Authoritative recipe — `Qmd4_0::fill_qmd` is what mc follows field-for-field |
| Mesa NVK dispatch | Mesa's `src/nouveau/vulkan/nvk_cmd_dispatch.c` | Begin-compute pushbuffer recipe + SHARED/LOCAL memory window constants |
| Mesa NVK QMD V04 header | Mesa's generated `nvidia/classes/clcbc0qmd.h` under `src/nouveau/headers/` | Bit-position macros, equivalent to the driver's `clcbc0qmd.h` |
| Driver SDK QMD V04 | `clcbc0qmd.h` — **not present in this tree** (the only QMD class header here is `cla0c0qmd.h`, for Kepler); NVK's copy above is the usable reference | `NVCBC0_QMDV04_00_*` macro names |
| Driver SDK Hopper compute methods | `src/common/sdk/nvidia/inc/class/clc7c0.h` | NVC7C0_SEND_PCAS_A et al. |

---

*Status: the SM-writes-sysmem-cell and SM-rings-BAR1-doorbell chains both
PASS on H100 PCIe.*
