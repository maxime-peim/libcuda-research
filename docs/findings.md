# GPU DMA Tracing and the `mc` D2H Transfer — Findings

> The `mc` implementation has moved to
> [`libmc`](https://github.com/maxime-peim/libmc). This research log links to
> current `libmc` source where it discusses the implementation.

This document summarises everything learned across all sessions: architecture,
driver ABI, tracing methodology, and the complete working implementation of
the `mc` library.

**Companion documents.** The full documentation set, in recommended reading
order for someone new to the domain (the
[`libmc` compute-launch reference](https://github.com/maxime-peim/libmc/blob/main/docs/compute_kernel_launch.md)
sits outside this ladder):

**Foundations (read if you don't already know these concepts):**
1. `docs/host_gpu_communication_primer.md` — how CPUs and GPUs talk in general:
   PCIe, BARs, MMIO, DMA, doorbells, command queues, cache coherency, IOMMU.
   Not NVIDIA-specific. Start here if you're new to host-device programming.
2. `docs/nvidia_software_stack.md` — the NVIDIA stack specifically: libcuda,
   nvidia.ko, nvidia-uvm.ko, GSP firmware, GSP-RPC, the RM object model.
   Explains why there are ~5 layers of software between an app and the GPU.
3. `docs/gpu_compute_model.md` — the GPU hardware: SMs, Copy Engines, PBDMA,
   channels, subchannels, TSGs, runlists, MMU, RAMIN/RAMFC/USERD.

**mc specifics:**
4. [`libmc/docs/mc_architecture.md`](https://github.com/maxime-peim/libmc/blob/main/docs/mc_architecture.md) — the *software* architecture: ioctl layering, kernel code paths, phase-by-phase setup walkthrough.
5. `docs/gpfifo_pushbuffer_reference.md` — the *data-structure* reference:
   bit-exact formats for GPFIFO entries, method headers, NVC8B5 methods, USERD,
   and the end-to-end component interaction timeline.

**Tracing:**
6. `docs/tracing_cuda.md` — how to capture and read an end-to-end timeline of
   any CUDA (or `mc_demo`) run: the kernel instrumentation, the userspace
   tools, and what each output file contains.
7. `docs/reference/trace-format.md` — the `mc1` kernel trace record format:
   grammar, the `mc_trace` category mask, and the complete event catalogue.
8. This document (`findings.md`) — research log and the bring-up history.

Section numbers here are stable anchors: material that moved out of this
document leaves its number behind as a pointer, and deleted sections leave a
gap rather than renumbering everything after them.

---

## 1. Architecture — How a D2H DMA Transfer Actually Works

### The three-layer model

```
CPU userspace (libcudart.so / mc)
  │  writes method stream into pushbuffer (mmap'd)
  │  writes GP entry into GPFIFO ring (mmap'd)
  │  MMIO-writes GPPut in USERD page (mmap'd)  ← no syscall
  ▼
GPU hardware
  │  PBDMA reads GPFIFO entry → finds pushbuffer VA
  │  PBDMA parses NVC8B5 method stream
  │  Copy Engine executes: HBM read → PCIe TLPs → host DRAM write
  │  CE writes semaphore on completion
  ▼
CPU userspace polls semaphore → cudaMemcpy returns
```

**Key insight**: after the setup phase, the entire D2H hot path has
**zero kernel involvement and zero GSP RPCs**. The CUDA runtime submits
work by writing ~80 bytes to mmap'd memory and one MMIO register.

### What the kernel driver (RM) and GSP-RM actually do

RM/GSP handle *infrastructure*, not work submission:

- Allocate RM object handles (client, device, subdevice, TSG, channel)
- Set up the GPU MMU page tables so the channel can reach HBM and host DRAM
- Allocate GPFIFO ring and pushbuffer memory
- Map GPFIFO, pushbuffer, and USERD doorbell page into userspace via BAR1/mmap
- Return a work-submit token the runtime writes to trigger the CE scheduler

After setup: the CE channel exists as a hardware state machine. All work
submission is pure userspace → GPU without any software in between.

### Why `nvWriteGpEntry` in `nvidia-push.c` never fires for CUDA

`nvidia-push.c` is compiled only into `nvidia-modeset.ko` (the display
driver), not into `nvidia.ko`. CUDA does its own push-buffer submission
entirely from userspace (`libcuda.so`). The kernel is not involved in
the per-transfer GPFIFO submission.

---

## 2. The NVC8B5 Method Stream (Hopper CE, class 0xC8B5)

A D2H transfer of N bytes requires **18 dwords = 72 bytes** (including the mandatory
SET_OBJECT header), because the copy and its completion signal are two separate
`LAUNCH_DMA`s — which is what H100 libcuda does. All methods must target **subchannel 4** (NVA06F_SUBCHANNEL_COPY_ENGINE).
Using subchannel 0 (the default) causes Xid 32 "invalid pushbuffer stream".

The code below is the current form in
[`libmc/mc/mc_submit.c`](https://github.com/maxime-peim/libmc/blob/main/mc/mc_submit.c),
after the DRF-macro refactor.
Every shift and mask is sourced from the NVIDIA SDK headers (`clc36f.h`, `clc8b5.h`,
`cla06fsubch.h`) rather than hand-rolled, so a future bit-layout change propagates
automatically. The "raw hex" form we used during bring-up is shown further below
for archaeological reference.

```c
// Method header built from clc36f.h bit fields — not a hand-rolled constant:
//   NVC36F_DMA_INCR_OPCODE      [31:29] = NVC36F_DMA_INCR_OPCODE_VALUE (=1)
//   NVC36F_DMA_INCR_COUNT       [28:16]
//   NVC36F_DMA_INCR_SUBCHANNEL  [15:13]
//   NVC36F_DMA_INCR_ADDRESS     [11:0] = method_addr >> 2
#define INCR_HEADER_SUB(method, count, subch)              \
  (DRF_DEF(C36F, _DMA_INCR, _OPCODE, _VALUE)               \
   | DRF_NUM(C36F, _DMA_INCR, _COUNT, (count))             \
   | DRF_NUM(C36F, _DMA_INCR, _SUBCHANNEL, (subch))        \
   | DRF_NUM(C36F, _DMA_INCR, _ADDRESS, (method) >> 2))
#define INCR_HEADER(method, count) \
  INCR_HEADER_SUB((method), (count), NVA06F_SUBCHANNEL_COPY_ENGINE)

// 0. Bind CE engine to subchannel 4 — REQUIRED as first method
pb[0] = INCR_HEADER(NVC86F_SET_OBJECT, 1);
pb[1] = HOPPER_DMA_COPY_A;

// 1–2. Source address (UPPER field is 25 bits at [24:0] holding src_va[56:32])
pb[2] = INCR_HEADER(NVC8B5_OFFSET_IN_UPPER, 2);
pb[3] = DRF_NUM(C8B5, _OFFSET_IN_UPPER, _UPPER, (NvU32)(src_va >> 32));
pb[4] = (NvU32)src_va;

// 3–4. Destination address
pb[5] = INCR_HEADER(NVC8B5_OFFSET_OUT_UPPER, 2);
pb[6] = DRF_NUM(C8B5, _OFFSET_OUT_UPPER, _UPPER, (NvU32)(dst_va >> 32));
pb[7] = (NvU32)dst_va;

// 5. Transfer size (32-bit dword → max 4 GiB per single LAUNCH_DMA)
pb[8] = INCR_HEADER(NVC8B5_LINE_LENGTH_IN, 1);
pb[9] = nbytes;

// 6. Launch 1 — the copy alone.  Each flag comes from clc8b5.h with its
// shift baked in.  SEMAPHORE_TYPE is field 4:3 and 0 means "none", so it
// is simply omitted.  Composite = 0x182.
pb[10] = INCR_HEADER(NVC8B5_LAUNCH_DMA, 1);
pb[11] =
    DRF_DEF(C8B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NON_PIPELINED)
  | DRF_DEF(C8B5, _LAUNCH_DMA, _FLUSH_ENABLE, _FALSE)
  | DRF_DEF(C8B5, _LAUNCH_DMA, _SRC_MEMORY_LAYOUT, _PITCH)
  | DRF_DEF(C8B5, _LAUNCH_DMA, _DST_MEMORY_LAYOUT, _PITCH);

// 7–9. Semaphore address + payload, set AFTER the copy launch
pb[12] = INCR_HEADER(NVC8B5_SET_SEMAPHORE_A, 3);
pb[13] = DRF_NUM(C8B5, _SET_SEMAPHORE_A, _UPPER, (NvU32)(sema_va >> 32));
pb[14] = (NvU32)sema_va;
pb[15] = sema_payload;  // completion value

// 10. Launch 2 — releases the semaphore, moves no data.  Composite = 0xc.
pb[16] = INCR_HEADER(NVC8B5_LAUNCH_DMA, 1);
pb[17] =
    DRF_DEF(C8B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NONE)
  | DRF_DEF(C8B5, _LAUNCH_DMA, _FLUSH_ENABLE, _TRUE)
  | DRF_DEF(C8B5, _LAUNCH_DMA, _SEMAPHORE_TYPE, _RELEASE_ONE_WORD_SEMAPHORE);
```

A single fused launch also works — `0x18e` = `NON_PIPELINED | FLUSH_ENABLE |
RELEASE_ONE_WORD_SEMAPHORE | SRC_PITCH | DST_PITCH` — and that is the value Yan et al.
report from an A40 trace (§7).  H100 libcuda does not use it.  `mc`'s SM-authored
kernel still does, because it fits in 16 dwords
([`libmc/mc/kernels/sm_owner.cu`](https://github.com/maxime-peim/libmc/blob/main/mc/kernels/sm_owner.cu)).

All addresses are GPU VAs from UVM (= CPU VAs on Hopper under UVM unification).

### GPFIFO entry format (from `clc86f.h`)

Each GP entry is 8 bytes. Current encoding uses DRF_NUM/DRF_DEF against the
SDK field definitions (`NVC86F_GP_ENTRY0_GET = 31:2`, etc.):

```c
// Normal entry (replaces raw-hex version used during bring-up):
entry0 = DRF_NUM(C86F, _GP_ENTRY0, _GET, (NvU32)(pb_va >> 2))
       | DRF_DEF(C86F, _GP_ENTRY0, _FETCH, _UNCONDITIONAL);
entry1 = DRF_NUM(C86F, _GP_ENTRY1, _GET_HI, (NvU32)(pb_va >> 32))
       | DRF_NUM(C86F, _GP_ENTRY1, _LENGTH, length_dwords)
       | DRF_DEF(C86F, _GP_ENTRY1, _LEVEL, _MAIN)
       | DRF_DEF(C86F, _GP_ENTRY1, _SYNC, _PROCEED);
```

**Important subtlety about DRF_NUM.** `NVC86F_GP_ENTRY0_GET` is defined as bits
`31:2` of ENTRY0 — a 30-bit field at positions [31:2] holding the *semantic
value* `pb_va[31:2]`. DRF_NUM takes the semantic value (pre-shifted by 2) and
performs the field placement itself, so we pass `pb_va >> 2`, not `pb_va`:

```c
// CORRECT — semantic value is pb_va[31:2]:
entry0 = DRF_NUM(C86F, _GP_ENTRY0, _GET, pb_va >> 2);

// WRONG — would place pb_va[29:0] at bits [31:2], producing a 4x-shifted addr:
entry0 = DRF_NUM(C86F, _GP_ENTRY0, _GET, pb_va);
```

During bring-up (before the DRF refactor) the equivalent raw-hex form was:

```c
// The raw-hex form, written by hand instead of with DRF macros:
entry0 = (uint32_t)(pb_va & 0xFFFFFFFC);   // pb_va[31:2] masked in-place
entry1 = ((pb_va >> 32) & 0xff) | (length_dwords << 10);

// The very first version had yet another bug — pre-shifting without masking:
//   entry0 = pb_va >> 2;     // ← shifted va[31:2] to [29:0] = 4-byte-off → Xid 31
```

All three forms produce the same output bits on a correctly-aligned 4-byte VA;
but DRF_NUM is self-documenting against the SDK and makes the semantic-value
contract explicit.

### Extended-base for high VAs (Hopper, UVM allocations above 40-bit range)

UVM routinely allocates buffers at VAs like `0x76baa0e2c000` (43 bits).
ENTRY1[7:0] can only encode 8 bits above bit 32, giving max 40-bit addressing.
For higher VAs, write a `SET_PB_SEGMENT_EXTENDED_BASE` control entry first:

```c
NvU32 ext_base = (NvU32)(pb_va >> 40);  // pb_va[56:40] — 17 bits
if (ext_base) {
    gpfifo_ring[idx*2 + 0] =
        DRF_NUM(C86F, _GP_ENTRY0, _PB_EXTENDED_BASE_OPERAND, ext_base);
    gpfifo_ring[idx*2 + 1] =
        DRF_DEF(C86F, _GP_ENTRY1, _OPCODE, _SET_PB_SEGMENT_EXTENDED_BASE);
    idx++; gp_put++;
}
// then write normal GP entry
```

The GPFIFO ring GPPut is advanced by writing the new index to USERD offset 0x8c.
**On Hopper, writing GPPut alone is insufficient** — the VF doorbell must also
be written (see §8 below, and §12 for the watchpoint that proves it).

---

## 3. Tracing Infrastructure

Moved.  What the kernel instrumentation is, which files carry `MC_TRACE`
sites, and what each userspace tool does now live in
`docs/tracing_cuda.md`; the record grammar and the event catalogue are in
`docs/reference/trace-format.md`.

---

## 4. Observed Trace Structure (a CUDA D2H benchmark on H100 PCIe)

Source capture: `reverse/tools/trace_cuda.sh reverse/bin/cuda_reference --size 256M --iters 11`
on H100 PCIe, driver 610.43.02, `mc_trace` at its default mask.  One H2D copy
seeds the buffer, eleven timed D2H copies follow — twelve `cudaMemcpy` calls.
The capture is checked in at `reverse/traces/cuda_reference/`; every figure
below is reproducible from it with
`python3 reverse/tools/phase_census.py --trace reverse/traces/cuda_reference`.

Phases are cut at the libcuda-level brackets pbcap records (first
`cudaMemcpy.enter` .. last `cudaMemcpy.exit`), not by wall-clock guesswork.

### Three phases

**Counts were byte-identical across 5 repeat captures.**  Only wall times move,
and those are inflated by `strace`, so timings should be taken from untraced
runs; event counts are unaffected by the instrumentation.

| | setup | 12 × cudaMemcpy | teardown | total |
|---|---|---|---|---|
| RM ioctls | 361 | **0** | 3 | 364 |
| of which `RM_CONTROL` | 191 | **0** | 1 | 192 |
| GSP RPCs | 237 | **0** | 183 | 420 |
| UVM ioctls | 87 | **0** | 2 | 89 |
| doorbells | 193 | 12 | 0 | 205 |
| pushbuffer submissions | 193 | 12 | 0 | 205 |
| distinct control sub-commands | 67 | 0 | 1 | 68 |

Teardown is RPC-driven rather than ioctl-driven: the 183 teardown RPCs are
issued from fd-close paths, not from `NV_ESC_RM_FREE` ioctls (there are only a
handful of those in the whole run).

### Setup cost does not scale with the work

A separate run of the same program doing **three** copies instead of twelve
costs 365 RM ioctls and 420 GSP RPCs, with an identical RPC breakdown (that
capture is not checked in).  The twelve-copy capture in
`reverse/traces/cuda_reference/` costs 364 and 420.  The object graph is built once and then reused; the per-copy
marginal cost at the driver level is zero.

### Decoded GSP RPC distribution (420 RPCs total)

| Function | Count | Meaning |
|---|---|---|
| `FREE` | 151 | teardown — one per object GSP knows about |
| `GSP_RM_ALLOC` | 122 | object allocations forwarded to GSP |
| `GSP_RM_CONTROL` | 118 | query/configure GPU state |
| `DUP_OBJECT` | 29 | cross-client handle duplication |

122 allocations + 29 duplications = the 151 objects later freed.

Captures taken as the *first* run after a module load additionally carry one-time
GSP bootstrap RPCs (`GSP_SET_SYSTEM_INFO`, `SET_REGISTRY`, `GET_GSP_STATIC_INFO`,
`INIT_GSP_TRACE_CRASH_BUFFER`, `UNLOADING_GUEST_DRIVER`) which belong to the
driver's lifetime, not the traced process.  Run with persistence mode enabled, or
discard the first capture, before quoting per-process figures.

### Decoded RM_CONTROL sub-commands (192 ioctls, 68 unique)

118 of the 192 were forwarded to GSP as `GSP_RM_CONTROL`; the rest were answered
host-side.  Key groups (from `src/common/sdk/nvidia/inc/ctrl/ctrl2080/`):

- **GPU info**: `GPU_GET_INFO_V2`, `GPU_GET_NAME_STRING`, `MC_GET_ARCH_INFO`,
  `GPU_GET_GID_INFO`, `BUS_GET_PCI_INFO`
- **CE info**: `CE_GET_ALL_CAPS` — available copy engines and their PCE masks
- **FIFO**: `GPFIFO_GET_WORK_SUBMIT_TOKEN`, `GPFIFO_SCHEDULE`, `SET_TIMESLICE`
- **Memory**: `FB_GET_INFO_V2`, `FB_GET_FS_INFO`
- **GR topology**: `GR_GET_TPC_MASK`, `GR_GET_GPC_MASK`, `GR_GET_CTX_BUFFER_SIZE`

**Notable**: zero transfer-related control commands.  Every `cudaMemcpy` action is
invisible from the RM/GSP layer.

### Ring geometry

The GPFIFO ring is **1024 entries**.  A pushbuffer slot is reused only after
~1022 submissions on that channel — measured at `--size 64K --iters 3000`: 3003
submissions on the copy channel, 2043 distinct pushbuffer addresses, first reuse
at submission #1023.  Below that rate nothing is recycled at all.

---

## 5. Driver ABI — Verified Facts for `mc`

### RM escape codes (from `nv_escape.h`)

| Escape | Value | Struct | Usage |
|---|---|---|---|
| `NV_ESC_RM_ALLOC` | `0x2b` | `NVOS64_PARAMETERS` | All object allocations |
| `NV_ESC_RM_CONTROL` | `0x2a` | `NVOS54_PARAMETERS` | Runtime queries |
| `NV_ESC_RM_FREE` | `0x29` | `NVOS00_PARAMETERS` | Free any handle |
| `NV_ESC_RM_MAP_MEMORY` | `0x4e` | `nv_ioctl_nvos33_parameters_with_fd` | Map memory to CPU VA |
| `NV_ESC_RM_ALLOC_MEMORY` | `0x27` | `nv_ioctl_nvos02_parameters_with_fd` | Allocate system memory |
| `NV_ESC_REGISTER_FD` | `0xc9` | `nv_ioctl_register_fd_t` | Associate ctl_fd with GPU fd |

### NVOS21 vs NVOS64 — critical distinction

**CUDA always uses `NVOS64_PARAMETERS` (size=48 bytes) for `NV_ESC_RM_ALLOC`**.
`NVOS21_PARAMETERS` (size=32 bytes) triggers the non-access API path in
`escape.c` which does NOT properly forward channel allocations to GSP-RM
on Hopper, causing `NV_ERR_INVALID_OBJECT_PARENT` failures.

`NVOS64_PARAMETERS` adds: `pRightsRequested`, `paramsSize`, `flags`.
Set all to 0 for basic usage.

### Object hierarchy

```
NV01_ROOT (0x0/0x41)         ← hRoot = hParent = hNew = h_client
  └─ NV01_DEVICE_0 (0x80)    ← parent = h_client
       └─ NV20_SUBDEVICE_0 (0x2080)    ← parent = h_device
       └─ NV01_MEMORY_LOCAL_USER (0x40) ← vidmem, parent = h_device
       └─ KEPLER_CHANNEL_GROUP_A (0xa06c)  ← TSG, parent = h_device
            └─ HOPPER_CHANNEL_GPFIFO_A (0xc86f) ← channel
                 └─ HOPPER_DMA_COPY_A (0xc8b5)  ← CE engine object
```

### NV_MEMORY_ALLOCATION_PARAMS for vidmem

From `NV_MEMORY_ALLOCATION_PARAMS` in `nvos.h`:
- `mp.owner = h_client`  — must be the client handle (not 0)
- `mp.attr` uses bits at positions `LOCATION[26:25]` and `PHYSICALITY[28:27]`
  — NOT bits[1:0] and [3:2] as one might expect from the field names alone
- `VIDMEM = 0 << 25`, `CONTIGUOUS = 2 << 27`
- Flags: `NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED`
- Parent: `h_device` (not `h_subdevice`)

### NV_ESC_RM_MAP_MEMORY pattern (verified from d2h.strace)

```
1. fd_N = open("/dev/nvidia0")        // fresh fd per mapping
2. ioctl(ctl_fd, 0x4e, {p.fd=fd_N})  // create mmap context in kernel
3. mmap(pLinearAddress,               // MAP_FIXED at address RM chose
        len, prot,
        MAP_SHARED|MAP_FIXED,
        fd_N, 0)                      // offset is ALWAYS 0
```

- Each mapping needs its own fresh `/dev/nvidia0` fd
- The mmap offset is always 0 — physical address is encoded in the kernel
  mmap context created by `rm_create_mmap_context()`, not in the offset
- `pLinearAddress` from RM is the target CPU VA to use with `MAP_FIXED`
- Vidmem (HBM) maps use BAR1 offset `~0x1fc0xxxxxxxx`

### CUDA init before channel allocation

CUDA issues **58 `RM_CONTROL` calls before any memory allocation**
(`cudaMalloc`). These configure: GPU capabilities, VA space defaults,
BUS topology, GR topology, CE capabilities, TIMER correlation.

Without these, channel allocation fails with `NV_ERR_INVALID_OBJECT_PARENT`
because GSP-RM requires device VA space to be initialised before
accepting channel objects.

### Host memory — UVM required on Hopper + CUDA 13

`cudaHostAlloc` on modern CUDA uses the **legacy** `NV_ESC_RM_ALLOC_MEMORY`
(escape 0x27, on `/dev/nvidia0`) to allocate and pin the pages, and then
UVM ioctls on `/dev/nvidia-uvm` to give the GPU a mapping:
`UVM_CREATE_EXTERNAL_RANGE + UVM_MAP_EXTERNAL_ALLOCATION`.  That is
exactly the sequence `mc_malloc_host` performs.  Measured 2026-08-18 from
`reverse/traces/cuda_reference` and confirmed at the PTE level: the
resulting GPU mappings are bit-identical between libcuda and mc (4 KiB
pages, attribute bits 0x68d on every PTE).  Note the `rm/alloc`
tracepoint only covers the modern `NV_ESC_RM_ALLOC` (0x2b), so these
allocations show `class=None` in decoded output — unresolved, not absent.

Without `nvidia-uvm.ko`, `cudaGetDeviceProperties` fails immediately
— the entire CUDA runtime requires UVM for device init on driver 610.

---

## 6. `mc` — WORKING — Complete Bring-Up Checklist

**Status: end-to-end D2H transfer working on H100 PCIe.**

*See §13 for the VA-pool fix that brought reliability to 1000/1000 at
N=1000, and §11 for the current bandwidth figures.  This §6 remains as
the original bring-up record.*

The full setup sequence verified to work on the H100 test host (H100 PCIe, driver 610.43.02).

Note: the size and layout numbers in this table reflect the
2026-05-02 bring-up.  Current code uses a 4 GiB VA-pool layout, a 2
MiB shared `gpu_ctl` allocation (GPFIFO + USERD), d_buf is CE-staged
with no CPU alias, pushbuffer is 1 MiB, and the active doorbell is
the BAR1 variant of HOPPER_USERMODE_A.  See §13 for the rework.

| Step | Status | Key detail |
|---|---|---|
| Open /dev/nvidiactl + /dev/nvidia0 | ✅ | + NV_ESC_REGISTER_FD |
| Allocate client/device/subdevice | ✅ | NVOS64 only |
| Alloc d_buf, h_buf (transfer_size, default 256 MiB each) | ✅ | NV01_MEMORY_LOCAL_USER + NV01_MEMORY_SYSTEM |
| Alloc gpu_ctl (2 MiB HBM, holds GPFIFO at offset 0 + USERD at offset 0x2000), pushbuffer (1 MiB sysmem), sema (4 KiB sysmem), staging (2 MiB sysmem) | ✅ | mix of vidmem/sysmem; gpu_ctl is shared GPFIFO+USERD per §13 |
| Alloc FERMI_VASPACE_A with IS_EXTERNALLY_OWNED flag | ✅ | mandatory for UVM |
| Open /dev/nvidia-uvm twice (primary + secondary fd) | ✅ | secondary for mm_struct hold |
| UVM_INITIALIZE (primary fd) + UVM_MM_INITIALIZE (secondary fd) | ✅ | in that order |
| Fetch GPU UUID via NV2080_CTRL_CMD_GPU_GET_GID_INFO | ✅ | FORMAT_BINARY |
| UVM_REGISTER_GPU | ✅ | phys UUID → instance UUID |
| UVM_REGISTER_GPU_VASPACE | ✅ | requires IS_EXTERNALLY_OWNED vaspace |
| uvm_map_buffer × 6 (d_buf, gpu_ctl, pb, h_buf, sema, staging) | ✅ | d_buf via plain uvm_map_buffer (PROT_NONE); the other five via uvm_map_buffer_at anchored to VA-pool slots (Paper F1, see §13.2) |
| Alloc TSG (KEPLER_CHANNEL_GROUP_A) with engineType=<non-GRCE LCE>, hVASpace | ✅ | parent = h_device |
| Alloc CE channel (HOPPER_CHANNEL_GPFIFO_A) | ✅ | parent = h_tsg, hVASpace=0, engineType matches TSG |
| Alloc CE object (HOPPER_DMA_COPY_A) | ✅ | parent = h_channel, NVB0B5_ALLOCATION_PARAMETERS with VERSION_1 + engineType |
| Get work-submit token (NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN) | ✅ | |
| Alloc HOPPER_USERMODE_A × 2 under subdevice | ✅ | BAR0 variant (bBar1Mapping=NV_FALSE) + BAR1 variant (bBar1Mapping=NV_TRUE) — the BAR1 one carries the active doorbell (see §13) |
| mmap HOPPER_USERMODE_A BAR1 variant (64 KiB) → vf_doorbell = cpu + 0x90 | ✅ | BAR1 NV_VIRTUAL_FUNCTION_DOORBELL |
| UVM_REGISTER_CHANNEL | ✅ | mandatory before SCHEDULE on externally-owned vaspace |
| NVA06F_CTRL_CMD_GPFIFO_SCHEDULE (bEnable=1) | ✅ | must come after REGISTER_CHANNEL |
| Map GPFIFO/USERD via rm_map_memory_at (BAR1 CPU alias into VA pool) | ✅ | for CPU writes |
| Sema is sysmem, allocated straight into the VA pool | ✅ | CPU polls it directly, no BAR1 alias — see the comment in [`libmc/mc/mc_core.c`](https://github.com/maxime-peim/libmc/blob/main/mc/mc_core.c) for why |
| CPU fills `staging` with FILL_PATTERN; CE-fills d_buf from staging | ✅ | d_buf has NO BAR1 alias — CE does the copy (see §13) |
| Write NVC8B5 method stream to pushbuffer | ✅ | 18 dwords, subch=4, SET_OBJECT first, split launches |
| Write GP entry to GPFIFO ring (+ ext-base entry if VA > 40-bit) | ✅ | entry0 = va & 0xFFFFFFFC |
| Write GPPut to USERD + write token to BAR1 VF doorbell (+0x90) | ✅ | both writes needed |
| Poll sema → expected payload | ✅ | 0.30 ms for 4 MiB transfer (pre-fix bring-up measurement; default 256 MiB transfers run ~18 ms) |

### Kernel crash bug (historical — fixed early)

A direct dereference of `pApi->pAllocParms` (a **userspace VA**) in
kernel context caused a page fault / kernel oops. Fixed by replacing
with `copy_from_user()` in `escape.c`. The correct kernel idiom:

```c
// WRONG — page fault:
NvU32 *w = (NvU32 *)(NvUPtr)pApi->pAllocParms;
w[0]; // kaboom

// CORRECT:
NvU32 buf[N];
copy_from_user(buf, (void __user *)(NvUPtr)pApi->pAllocParms, sizeof(buf));
buf[0]; // safe
```

---

## 7. Insights from "Revealing NVIDIA Closed-Source Driver Command Streams" (Yan et al., 2026)

This paper independently validates our architecture model and provides several key findings
directly applicable to `mc`.

### Better pushbuffer capture methodology
Rather than `copy_from_user` in escape.c, they install a **hardware watchpoint** on the
userspace USERD doorbell VA via a modified `nv_mmap`. When doorbell is written:
1. Watchpoint traps → kernel callback reads channel ID from doorbell value
2. Looks up `KernelChannel` → extracts USERD + RAMFC physical addresses
3. Reads `GP_PUT` + `GP_BASE` → computes new GPFIFO entry VA
4. Walks GPU MMU page tables → gets pushbuffer physical address
5. Maps physical pages to CPU VA → reads command stream intact
They use a **shadow RAM page** to buffer the doorbell value because VF doorbell reads always return 0.

### Finding 1 — GPU VA = CPU VA under UVM
GPU virtual addresses in pushbuffer commands ARE the process's user-space virtual addresses.
UVM maintains a unified address space — no separate translation step. This confirms why
our decode.py correctly predicts method stream bytes from CUDA malloc return values.

### Finding 2 — Asymmetric memory placement
- **GPFIFO ring**: GPU **VRAM** (CPU writes over PCIe BAR1, GPU reads locally)  
- **Pushbuffer**: **host RAM** (CPU writes locally, GPU reads over PCIe)

This explains the GPFIFO BAR1 offset (`0x1fc0xxxxxxxx`) we observed.  Note the pushbuffer
itself, being host RAM, does read back correctly from the CPU even though it is mapped
write-combine — what returns zero is the VF doorbell register, which has no backing
store at all (`tracing_cuda.md`, "`pbcap.c`, and two things it took a while
to get right"; and §12.1 below).

### Two DMA submission modes (cudaMemcpy size threshold ~24 KiB)
- **< 24 KiB**: Inline DMA — **compute engine**, source data embedded in pushbuffer payload
- **≥ 24 KiB**: Direct DMA — **copy engine**, `src_addr + dst_addr` in pushbuffer

### LAUNCH_DMA = 0x18e — an A40 form, not the H100 one
Their Listing 1 for a 64 MiB cudaMemcpyAsync shows a single fused
`LAUNCH_DMA = 0x18e`.  A captured H100 `cudaMemcpy` does **not** do this: libcuda
splits the work into `LAUNCH_DMA = 0x182` (the copy — no flush, no semaphore) followed
by a semaphore-only launch.  The individual flag encodings agree; what differs is
whether copy and release share one launch.  `mc` follows the H100 form on the host
path; its SM-authored kernel keeps the fused one to stay within 16 dwords.
Subchannel is 4 on both (NVA06F_SUBCHANNEL_COPY_ENGINE).

### Confirmed for mc
- Pushbuffer IS in host RAM (`NV01_MEMORY_SYSTEM`) ✅ confirmed.
- FERMI_VASPACE_A IS allocated and passed via TSG params ✅ confirmed.
- LAUNCH_DMA flag encodings ✅ match; but the paper's fused single launch is an A40 form — H100 libcuda splits copy from release, and `mc` follows H100 (see above).
- SET_OBJECT + subch=4 required: paper's trace shows `subch=4` — confirmed Xid 32
  without this, Xid 0 (success) with it.

---

## 8. Key Insight Summary

1. **The CE hot path has zero kernel involvement**: method stream write +
   GP entry write + USERD + VF doorbell = 96 bytes of userspace store instructions.

2. **GSP is an infrastructure manager, not a work submitter**: GSP sets
   up the channel so CE can run autonomously. GSP does not get involved
   per-transfer. Zero GSP RPCs in the d2h hot path.

3. **NVOS64 ≠ NVOS21**: CUDA always uses the 48-byte NVOS64 form of the
   alloc ioctl. Using the 32-byte NVOS21 form silently routes through a
   different code path that fails on Hopper + driver 610.

4. **`pAllocParms` is always a userspace pointer**: dereferencing it in
   kernel without `copy_from_user` causes an oops. This is a hard rule
   for any kernel instrumentation of RM.

5. **UVM is required on Hopper + CUDA 13**: there is no fallback to the
   classic RM host-memory-pinning path. CUDA uses zero `NV04_MAP_MEMORY_DMA`
   calls; all GPU VAs are established by UVM and are identical to CPU VAs
   (Yan et al. 2026 Finding 1).

6. **`rm_map_memory` needs a fresh fd per mapping**: each call to
   `NV_ESC_RM_MAP_MEMORY` consumes a fresh `/dev/nvidia0` fd via
   `rm_create_mmap_context()`. Reusing a fd for multiple mmaps fails.
   The mmap offset is always 0; the physical address is encoded in the
   kernel-side mmap context object.

7. **HOPPER_USERMODE_A's BAR1 variant carries the doorbell**: writing GPPut
   to USERD alone does NOT wake PBDMA. A write of the work-submit-token to
   the `HOPPER_USERMODE_A` mapping **+ 0x90** is also required, and the
   mapping libcuda and mc both ring is the BAR1 one (`bBar1Mapping=NV_TRUE`);
   see §12.6 for the 194/194-vs-0/194 evidence. The mapping-relative +0x90
   is the only offset worth memorising. As background, the underlying
   register is `NV_VIRTUAL_FUNCTION_DOORBELL`, at 0x30090 *within the VF
   register block* — BAR0+0xBB0090 on bare metal, since RM adds
   `virtualRegPhysOffset` (0xB80000). See
   `gpfifo_pushbuffer_reference.md §11`.

8. **GP entry0 is NOT shifted right by 2**: ENTRY0[31:2] stores va[31:2] at
   bit positions [31:2], not [29:0]. Write `entry0 = va & 0xFFFFFFFC`, not
   `entry0 = va >> 2`.

9. **High VAs need SET_PB_SEGMENT_EXTENDED_BASE**: UVM typically assigns
   buffer VAs above 40 bits. ENTRY1[7:0] covers only pb_va[39:32]. For
   pb_va[56:40], write a preceding GP entry with OPCODE=4 and
   operand = (pb_va >> 40) << 8 in entry0.

10. **CE methods require subchannel 4 and SET_OBJECT**: the first dword
    in every pushbuffer must be `INCR_HEADER(0, 1, subch=4)` followed by
    the class ID `0xC8B5`. Methods without this prefix reach an unbound
    subchannel → Xid 32.

---

## 9. UVM Subsystem — Key Structures and Sequence

The UVM subsystem (`kernel-open/nvidia-uvm/`) is the path CUDA uses for all
user-accessible GPU memory management on Hopper + driver 610.

### UVM ioctl encoding

**Raw integers** (no `_IOR/_IOW` packaging), unlike RM ioctls. All return 0 on
syscall success; status is always in `params.rmStatus`.

### Minimum-viable sequence

```
1. uvm_fd  = open("/dev/nvidia-uvm")
2. uvm_fdm = open("/dev/nvidia-uvm")
3. ioctl(uvm_fd,  UVM_INITIALIZE /* 0x30000001 */, {flags=0})
4. ioctl(uvm_fdm, UVM_MM_INITIALIZE /* 75 */,  {uvmFd=uvm_fd})
   → holds process mm_struct via secondary fd; must stay open
5. ioctl(uvm_fd,  UVM_REGISTER_GPU /* 37 */,   {gpu_uuid, rmCtrlFd=dev_fd, hClient})
   → on return, gpu_uuid updated to "instance UUID"
6. ioctl(uvm_fd,  UVM_REGISTER_GPU_VASPACE /* 25 */, {gpuUuid, rmCtrlFd, hClient, hVaSpace})
   → requires hVaSpace allocated with IS_EXTERNALLY_OWNED flag

   For each RM memory handle to expose at a GPU VA = CPU VA:
7.   cpu_va = mmap(NULL, size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
8.   ioctl(uvm_fd, UVM_CREATE_EXTERNAL_RANGE /* 73 */, {base=cpu_va, length=size})
9.   ioctl(uvm_fd, UVM_MAP_EXTERNAL_ALLOCATION /* 33 */,
           {base=cpu_va, length=size, offset=0, gpuAttributesCount=1,
            rmCtrlFd=dev_fd, hClient, hMemory=h_any_rm_mem_handle,
            perGpuAttributes[0]={gpuUuid=inst_uuid, mappingType=ReadWriteAtomic, rest=0}})

10. ioctl(uvm_fd, UVM_REGISTER_CHANNEL /* 27 */,
          {gpuUuid, rmCtrlFd=dev_fd, hClient, hChannel, base=any_reserved_va, length=4MiB})
    → mandatory before GPFIFO_SCHEDULE; internally: nvUvmInterfaceRetainChannel +
      UVM_CHANNEL_RETAINER alloc + bind_channel_resources
```

### Why UVM_REGISTER_CHANNEL is mandatory

`kchannelIsSchedulable_HAL()` (in kernel_channel.c) checks that the channel has been
registered with UVM via `UVM_CHANNEL_RETAINER` (class 0xC574) allocation.
This was added by "Bug 1737765: Prevent Externally Owned Channels from running unless
bound." Scheduling an un-registered channel returns `NV_ERR_INVALID_STATE` (0x40).

### Historical engine-type misclassification and current resolution

On Hopper, GR and GRCE can share runlist 0. `kchannelGetEngine_GM107()` determines
engine type by `runlistId → ENG_DESC → RM_ENGINE_TYPE`; because GR is first on
that shared runlist, a GRCE channel can be reported as GR. During bring-up this
made UVM try to load GR context buffers for the CE channel and return
`NV_ERR_INVALID_OBJECT` (0x31).

The original bring-up notes proposed changing `nvGpuOpsGetChannelEngineType()`
to prefer `pKernelChannel->engineType`. That functional patch is **not** in this
public driver tree and is not required by current `libmc`. The library queries
CE caps, selects a non-GRCE LCE, and passes that exact COPY engine consistently
to the TSG and channel, avoiding the shared GR runlist in userspace.

---

## 11. DRF-Macro Refactor and Measured Bandwidth

### SDK-DRF-based bit-field construction

The method stream and GP-entry construction were reworked to use the SDK's
`DRF_DEF` and `DRF_NUM` macros against the field definitions in `clc36f.h`,
`clc8b5.h`, `clc86f.h`, and `cla06fsubch.h`, so every shift and mask comes
from the SDK headers rather than hand-rolled constants. A future bit-layout
change propagates automatically; typos become compile errors.

**Subtlety about DRF_NUM**: the macro takes the *semantic value* of a field,
not the raw bits at those positions. For `NVC86F_GP_ENTRY0_GET = 31:2`, the
semantic value is `pb_va[31:2]` = `pb_va >> 2`, not `pb_va`:

```c
// CORRECT:
entry0 = DRF_NUM(C86F, _GP_ENTRY0, _GET, pb_va >> 2);

// WRONG — places pb_va[29:0] at bits [31:2], a 4x-shifted address → Xid 31:
entry0 = DRF_NUM(C86F, _GP_ENTRY0, _GET, pb_va);
```

Getting this wrong is what produced the Xid 31 described above. Reviewing
every DRF_NUM call site for the semantic-value contract is essential when
applying DRF to field encodings that span non-LSB positions.

### Measured bandwidth vs. transfer size

`mc_demo` takes `--size`, `--iters`, `--h2d` and `--wc`.
Measured on H100 PCIe (Gen5 x16, ~63 GB/s per direction theoretical) with
clocks left unlocked — `mc_init` requests boost the way libcuda does (§15),
and on this hardware locking them makes results worse rather than better.
Three independent launches per cell, with the arm order rotated between
launches so drift is shared rather than attributed; the iteration count is
scaled per size so every launch moves roughly 60 GiB, which keeps the
measurement clear of the warm-up transient described below. Median of the
per-launch mean throughput:

| Transfer | mc D2H | CUDA D2H | mc H2D | mc H2D `--wc` | SM-authored D2H |
|---|---:|---:|---:|---:|---:|
| 4 MiB   | 31.3 | 35.7 | 42.8 | 49.7 | 35.1 |
| 64 MiB  | 53.9 | 53.5 | 55.1 | 55.1 | 53.5 |
| 256 MiB | 55.1 | 52.9 | 55.1 | 55.2 | 54.9 |
| 1 GiB   | 55.5 | 55.4 | 55.5 | 55.5 | 55.4 |

Peak per-iteration throughput is flatter than the means: from 64 MiB up every
column reads 54.8–55.5, and even at 4 MiB the two H2D columns are identical at
51.6. Where a mean sits below its peak the gap is slow iterations, not a lower
ceiling — which is worth keeping in mind before reading a mean as a limit.

Five things this table says:

- **mc matches CUDA for D2H.** At 64 MiB, 256 MiB and 1 GiB the ratios are
  1.01, 1.04 and 1.00 — agreement within run-to-run spread, which on this box
  is itself a few percent. Read the 256 MiB CUDA cell with care: its three
  launches ranged 51.0–54.9, and with n=3 against a box whose slow mode is a
  discrete floor rather than a tail, the median is not a dependable point
  estimate — §15's independent 256 MiB comparison, taken over more samples,
  puts mc and CUDA 0.18 % apart. The pure-ioctl submission path adds no measurable per-transfer
  overhead — the claim Part 5 rests on, now measured rather than asserted. At
  4 MiB the two diverge in a way worth naming: mc's best iteration is faster
  (49.1 against 42.8 GB/s) while its mean is slower (31.3 against 35.7), so mc
  has the higher ceiling and CUDA the steadier floor.
- **Both directions reach the same plateau, ~55.5 GB/s at 1 GiB**, about 88 %
  of the Gen5 per-direction ceiling. Worth stating explicitly, because the
  opposite is easy to assume: D2H reads HBM and writes host DRAM, H2D does
  the reverse, and it would be reasonable to expect the two to differ. They
  do not. A measurement showing D2H far below H2D on this hardware is
  measuring idle clocks rather than the direction — the boost-clock request
  in §15 is what removes that. The absolute number is box-specific, so quote
  the CUDA parity rather than the figure.
- **`--wc` changes consistency, not throughput.** At 4 MiB the peak is
  identical to cached (51.6 both) while the mean is 16 % higher, because the
  cached arm has slow iterations the write-combined arm does not — spreads of
  42.1–51.1 against 49.2–51.1. From 64 MiB up the difference is inside
  measurement noise: 55.12 against 55.12 at 64 MiB, 55.10 against 55.17 at
  256 MiB, 55.46 against 55.46 at 1 GiB. The opt-in is therefore worth having for small,
  latency-sensitive staging and buys nothing on bulk transfers, while the
  ~370× read penalty applies at every size.
- **SM-authored submission costs nothing measurable** over host-authored
  (0.99–1.00 from 64 MiB up): a GPU thread writing its own pushbuffer and
  ringing the doorbell is as fast as the CPU doing it.
- **Small transfers are latency-bound**: 4 MiB sits well below plateau on
  every path, because the fixed doorbell-to-dispatch cost is a visible fraction
  of the total.

A methodology note that matters for reproduction: at default auto-boost the
GPU idles to 345 MHz between launches, so a short transfer can finish before
the clock has ramped and a launch can read at less than half its steady-state
rate. The library now requests boost clocks at init the way libcuda does
(§15), which removes the persistent low mode, but a sub-second ramp at the
head of a launch remains. The defence is iteration count rather than clock
locking, which on this hardware makes matters worse: scale iterations with
size so each launch moves enough bytes for the ramp to be a small fraction of
it. Measured the other way — 4 MiB at 100 iterations, some 0.4 GiB in total —
the cached arm reads 21.9 GB/s mean and 24.0 peak, against 42.0 and 51.6 for
the same arm at 20,000 iterations. Less than half the real rate, which is to
say the transient was the measurement.

Single-launch transfers above ~4 GiB are rejected: the CE's
`NVC8B5_LINE_LENGTH_IN` method is a 32-bit byte count, so transfers larger
than that would need to be split across multiple LAUNCH_DMAs.

---

## 12. Kernel-Side Hopper VF Doorbell Watchpoint (Yan et al. port, 2026-05-06)

This section documents the full port of Yan et al. §5.1–5.2
methodology into this tree, so the kernel (not libcuda, not `pbcap.c`)
observes every CUDA submission: which channel, which GPFIFO slot, which
pushbuffer VA/length.  Lives in `kernel-open/nvidia/nv-doorbell-watch.c`
+ `kernel-open/nvidia-dbell/` + a few surgical edits to `nv-mmap.c`,
`nv_gpu_ops.c`, and `rm-gpu-ops.c`.

### 12.1 Why userspace watchpoints don't work

`reverse/tools/pbcap.c` uses `mprotect(PROT_READ)` on the BAR0
doorbell page → SIGSEGV → trampoline via `TF` → SIGTRAP post-store.  The
TF-replayed store is the MMIO forward.  The paper's §3 architectural claim —
"userspace cannot win this race" — holds, and the reason is measurable
(2026-08-18, H100 PCIe, 610.43.02).  Two things are unrecoverable once the
handler regains control:

**The token.**  The VF doorbell dword at `+0x90` in the `HOPPER_USERMODE_A`
window has no backing store, so a read returns zero regardless of what was
written.  Without the token there is no `(chid, runlist)`, so no channel and
no ring to walk.  Read from userspace at three consecutive doorbells:

| mapping | `[+0x90]` |
|---|---|
| real BAR0 aperture (`nv_dbell_disable_intercept=1`) | `0x0`, `0x0`, `0x0` |
| kernel shadow page (`intercept=0`) | `0x00010005`, `0x00030003` |

The shadow-page values are genuine work-submit tokens and match this driver's
own `dbell/fire` records.  Diverting the mapping is therefore not a way to win
a race — it *materialises a value that otherwise exists nowhere readable*.

**The outstanding-work state.**  USERD keeps a producer/consumer index pair
(`+0x8c` / `+0x88`; USERD blocks sit at a 0x3000 stride in the BAR1 channel
region).  Sampled from userspace inside the SIGTRAP handler, immediately after
the doorbell store, across 3 runs × 60 channel observations: `GP_GET == GP_PUT`
**every time, zero pending**.  PBDMA drains the ring before the handler runs.

What is *not* the obstacle: the pushbuffer bytes themselves.  Those read back
correctly from userspace (measured in `tracing_cuda.md`, "`pbcap.c`, and two
things it took a while to get right"), and the ring gives a ~1022-submission
margin before a slot is recycled (§4), so a hook snapshotting after its own
synchronous `cudaMemcpy` cannot have been clobbered.  An earlier null result
here came from `PBCAP_MAX_BYTES`: the pushbuffer pool is 56 MiB against a
16 MiB default per-mapping cap, so the pool was never dumped.  A/B, replicated
N=4 — 16 MiB: 90 snapshots, 0 of the pool, 0 decodable; 128 MiB: 93 snapshots,
3 of the pool, 60 decodable.

Cost is an independent obstacle: arming the userspace watchpoint drops D2H
throughput to ~0.02 GB/s.

### 12.2 Architecture

```
nv_mmap()                         ↑ same userspace VMA, doorbell dword
  detects HOPPER_USERMODE_A      │ writes now land on WB shadow RAM.
  diverts to shadow_page          │
  arms DR0 on user_va+0x90        │
                                  │
Write to user_va+0x90 ──→ #DB trap (IRQ off, in-kernel)
                             │
                             ├─ decode token → (chid, runlist)
                             ├─ cache lookup (chid, runlist)
                             │    HIT  → read USERD.GP_PUT
                             │         → read GPFIFO entry[(gp_put-1) & (n-1)]
                             │         → emit mc1 dbell/* + pb/* records
                             │    MISS → enqueue nv_kthread_q worker
                             │         → worker calls rm_gpu_ops_dbell_resolve
                             │         → populates cache from RAMFC
                             └─ writel(token, real_BAR0)   ← inspect-then-forward
```

### 12.3 The modules

**`kernel-open/nvidia-dbell/nvidia_dbell.[ch]`** — GPL shim.
`register_user_hw_breakpoint` is `EXPORT_SYMBOL_GPL`; nvidia.ko is MIT-
licensed and can't link GPL-only symbols.  This ~80-line module wraps
register/unregister and re-exports as plain `EXPORT_SYMBOL` thunks
(`nv_dbell_bp_register`, `nv_dbell_bp_unregister`).  Conftest-gated at
`kernel-open/conftest.sh` via an inverted-signature compile probe for
`register_user_hw_breakpoint` (functions-category tests expect the probe
to FAIL compile when the symbol exists — see
`docs/findings.md §12.6` for the gotcha and
`kernel-open/nvidia-dbell/nvidia-dbell.Kbuild` for the build wiring).

**`kernel-open/nvidia/nv-doorbell-watch.[ch]`** — the core.
- 4-slot watch table (x86 only has DR0–DR3).
- `nv_dbell_intercept_mmap()` called from `nvidia_mmap_helper` when it
  sees a 64 KiB single-range device-node mapping (either the `IS_REG_OFFSET`
  BAR0 variant or the `IS_FB_OFFSET` BAR1 variant).
- Allocates a shadow page, `vm_insert_page`s it over the whole 64 KiB
  VMA, then calls `nv_dbell_bp_register` to arm an x86 hw breakpoint.
- Writes land on the shadow page; DR0 fires `nv_dbell_db_handler()` in
  trap context (IRQ-disabled) BEFORE any MMIO propagates.
- Handler decodes, emits `mc1` records via `MC_TRACE` (whose transport,
  `nv_trace_printf` = `ftrace_vprintk`, is trap-safe by construction —
  see §12.7), and `writel(token, real_bar0_iomap)` — so the GPU actually
  sees the submission.
- 80-entry per-(chid, runlist) channel cache.  FREE → PENDING → RESOLVED
  | FAILED state machine, atomic transitions, no locks in the trap
  handler.
- BAR1 tracker: a separate 16-slot table keyed on VMA that records
  `(phys_start, phys_end, user_va_start, user_va_end, kernel_va)` for
  every FB device-node mapping that went through `nv_io_remap_page_range`.
  The kernel VA comes from a kernel-side `ioremap_wc` we add; the user
  VA range comes from `vma->vm_start/end`.  This table is what lets us
  translate GPU VAs to kernel VAs (Paper Finding 1).

**`src/nvidia/src/kernel/rmapi/nv_gpu_ops.c`** — the RM-side resolver
`nvGpuOpsDbellResolveChannel(pGpu, chid, runlist, out_*)`.  Takes the
GPU lock, walks `pKernelFifo → chidMgr → KernelChannel`, returns the
channel's USERD info (kva, phys, size, addrspace) plus GPFIFO info
(gpu_va, entries).  The `(gpu_va, entries)` pair comes from a small
`(pGpu, runlist, chid) → (gpFifoOffset, gpFifoEntries)` lookup table
populated at `kchannelConstruct_IMPL` time from the client-supplied
`NV_CHANNEL_ALLOC_PARAMS` values — bit-identical to what RAMFC
encodes (see `kfifoWriteRamfcGpfifo_GH100`: GP_BASE/GP_BASE_HI store
`NvU64_{LO,HI}32(gpFifoOffset)`, GP_INFO.LIMIT2 stores
`log2(gpFifoEntries)`).  The first-cut implementation read those
values via `memdescMap(pRamfcDesc, ...)`; §12.9 explains why that
approach was replaced.

**`src/nvidia/src/kernel/gpu/fifo/kernel_channel.c`** — registers
each channel with the lookup table.  `kchannelConstruct_IMPL` calls
`nvGpuOpsDbellGpfifoRegister(pGpu, runlistId, ChID, gpFifoOffset,
gpFifoEntries)` in its cleanup block when `status == NV_OK` and
`gpFifoEntries > 0`.  `kchannelDestruct_IMPL` calls
`nvGpuOpsDbellGpfifoUnregister(pGpu, runlistId, ChID)` at the top,
before any teardown — so a late doorbell on this chid can't get a
stale offset.  Both calls happen under the RM device GPU lock,
same as the resolver's lookup.

### 12.4 The ftrace output format

Moved.  The `mc1` grammar, the `dbell/*` + `pb/*` event fields, and a
worked example of the records one `cudaMemcpy` produces are in
`docs/reference/trace-format.md`.

### 12.5 Layout discoveries — CUDA's per-channel BAR1 slot

Live sweep of CUDA's `/dev/nvidia0 rw-s 2 MiB` mapping after 20 channels
had been allocated revealed a clean layout:

| Channel chid | BAR1 offset | Content |
|---|---|---|
| 10 | 0x00000 | GPFIFO ring (1024 × 8 B = 8 KiB) |
| 10 | 0x02000 | HopperAControlGPFifo (USERD) |
| 11 | 0x03000 | GPFIFO ring |
| 11 | 0x05000 | USERD |
| ... | ... | per-channel 12 KiB slot, stride 0x3000 |
| 29 | 0x39000 | GPFIFO ring |
| 29 | 0x3b000 | USERD |

Each channel's 12 KiB = `[8 KiB GPFIFO ring][4 KiB page with USERD at
start + padding]`.  This layout is NOT a Hopper hardware constraint —
libcuda chose it.  Per-channel `subMemOffset` values observed from the
sub-memdescs:
- USERD: `0x2000 + (chid - 10) * 0x3000`.
- GPFIFO ring GPU VA: `mapping_user_va_base + (chid - 10) * 0x3000`.

The pushbuffer itself is **not** in this 2 MiB mapping — the mapping is
fully consumed by 20 × 12 KiB = 240 KiB of slots, and the 2 MiB
allocation leaves ~1.75 MiB unused (probably reserved for more channels
if CUDA scales up).  Pushbuffer is in sysmem, mmap'd through
`/dev/nvidiactl`.  To read pushbuffer bytes from the kernel we'd need a
separate tracker for sysmem pushbuffer pages — not implemented yet.

### 12.6 Gotchas encountered and documented

These are the kind of details that would cost a fresh reader a full
debug cycle if re-derived:

- **`conftest.sh` "functions" probe is inverted.**  The probe code must
  INTENTIONALLY fail to compile when the symbol exists.  Call with zero
  args against a real multi-arg function.  And the consuming `.c` file
  MUST `#include "conftest.h"` — the generated `NV_*_PRESENT` macros
  are not in `ccflags-y`.

- **the H100 test host is a VMware VM with H100 in PCIe passthrough.**
  `systemd-detect-virt` → `vmware`, but the H100 itself is at BDF
  `0000:23:00.0` with real BAR sizes (BAR0 16 MiB @ `0x1fe002000000`,
  BAR1 128 GiB @ `0x1fc000000000`).  RM / kernel traces log **guest**
  physical addresses; `/proc/PID/pagemap` reports **host** physical
  (passthrough IOMMU translation).  **Compare in GPA space; don't try
  pagemap-based BAR detection.**

- **libcuda's doorbell path is BAR1 HOPPER_USERMODE_A, not BAR0.**
  Libcuda allocates TWO `HOPPER_USERMODE_A` objects per context: one
  with `bBar1Mapping=NV_FALSE` (BAR0 variant) and one with
  `bBar1Mapping=NV_TRUE` (BAR1 GMMU variant).  All doorbell writes go
  to the BAR1 one — proven by 194/194 captured events on the BAR1 slot
  vs 0/194 on the BAR0 slot.  mc allocates both variants too, and
  rings the BAR1 one.  Our hook keys on size=64 KiB + numRanges=1 so
  it covers both.  The doorbell dword is at `+0x90` in either variant.

- **4 DR slots are scarce.**  Xorg and ollama each can claim one or
  two BAR0/BAR1 usermode mappings as soon as they start, exhausting
  DR0–DR3 before our test process gets any.  **Always
  `systemctl stop lightdm ollama` before running kernel-watchpoint
  tests.**

- **libcuda uses worker threads.**  `register_user_hw_breakpoint` binds
  to a specific `task_struct`.  We install one BP per thread in
  `current->group_leader`'s thread list at arm time.  Threads created
  AFTER arm time are invisible to our watchpoint — in practice
  libcuda's workers exist before any `cudaMemcpy` so this hasn't been
  an issue.

- **GPU VA ≠ GPU-physical VA ≠ CPU-visible BAR1 address.**  Earlier
  attempts at the BAR1-tracker approach tried
  `nv->fb->cpu_address + memdescGetPhysAddr(AT_GPU)` and landed 528 MiB
  off.  The `AT_GPU` value is a vidmem byte offset
  (what the GPU's own MMU uses internally); the CPU-visible BAR1
  address of USERD is the BAR1-aperture offset RM chose via
  `kbusMapFbApertureSingle`, totally unrelated numerically.  The clean
  path turned out to be: `pUserdMemDesc->subMemOffset` is offset inside
  the per-process BAR1 mapping; combined with `vma->vm_start` captured
  at `nv_dbell_bar1_track_add` time, this gives a direct kernel VA.

### 12.7 Trap-safety of the trace transport

`MC_TRACE` is a mask test — a plain global load and an AND — around
`nv_trace_printf`, which is a thin wrapper on `ftrace_vprintk`
(`kernel-open/nvidia/os-interface.c`).  Ftrace's per-CPU ring buffer uses
`raw_spin_lock_irqsave` only: no kmalloc, no sleeping locks, no per-CPU
variable migration hazards.  That is what makes it legal to call from the
`#DB` handler context (IRQ-off).  Empirically we have captured 174
`pb/submit` records + 20 `dbell/cache state=resolved` records + several
dozen `dbell/armed` / `dbell/release` lines per CUDA round-trip with zero
ftrace warnings, zero WARN_ONs, zero BUGs, and full 14+ GB/s bandwidth.

### 12.8 Live measurement (a 4 MiB CUDA round-trip)

Record counts from the capture, by `mc1` event:

```
dbell/armed                        = 2    (BAR0 + BAR1 HOPPER_USERMODE_A VMAs)
dbell/bar1_track state=add         = 1    (the 2 MiB /dev/nvidia0 rw-s mapping)
dbell/cache   state=reserve        = 20   (one per channel)
dbell/cache   state=resolved       = 20   (all channels resolved successfully)
dbell/cache   state=failed         = 0
dbell/cache   state=invalidate     = 20   (one per BAR1 mapping torn down at exit)
dbell/gpfifo_lookup result=table   = 20   (all GPFIFO lookups hit the table)
dbell/gpfifo_lookup result=miss    = 0
dbell/gpfifo_lookup result=not_in_table = 0
dbell/gp_put                       = 174  (read USERD.GP_PUT 174 times)
pb/submit                          = 174  (decoded GPFIFO entry 174 times)
```

Stress-tested post-fix on 2026-05-06 PM: 20× plain runs + 20×
LD_PRELOAD'd 4 MiB CUDA round-trip runs back-to-back,
**40/40 PASS, 0 WARN/BUG/vfree in dmesg, 0 crash traces**.
Run-to-run variance of ±2 on the `pb/submit` count is normal (we have
observed 172–174); the important invariants — zero `state=failed`, zero
`result=miss`, zero `result=not_in_table` — held every run.

`mc_demo --size 64M --iters 2` still passes verification.  Peak
bandwidth rose by roughly 2.5x after the memdescMap path was removed
from the resolver (see §12.9); the earlier number reflected the
overhead of the BAR2-aperture setup that memdescMap triggered per
doorbell.  (Both figures were taken on the older Gen4 test box these
measurements predate — see §11 for the current Gen5 numbers.)

(Historically, this section also noted that "our channel uses
UVM-allocated GPFIFO in sysmem, which our BAR1 tracker doesn't
cover — so the GPFIFO-lookup returns NULL and PB events don't fire
for it."  That was accurate at the time but **no longer applies**.  After the Paper-F1 VA-pool fix on 2026-05-12 [§13],
mc's GPFIFO now lives at offset 0 inside a 2 MiB HBM
`gpu_ctl` allocation whose BAR1 CPU alias is MAP_FIXED into the VA
pool — so `bar1_track_add` records it at `nvidia_mmap` time and
the #DB handler resolves it correctly.  `trace_cuda.sh` on `mc_demo`
now emits 2 `pb/submit` + 2 `pb/bytes` per round-trip, full parity with
libcuda.  See §13.4 and §13.9 dead-end #11.)

### 12.9 Superseded: `memdescMap` on RAMFC → table lookup

The feature's first two iterations read GP_BASE/GP_BASE_HI/GP_INFO
from `pKernelChannel->pFifoHalData[sub]->pRamfcDesc` via
`memdescMap(...)`.  Both iterations had problems; the third
(current) approach avoids touching RAMFC entirely.

**Iteration 1 — unconditional memdescMap.**  Broke mc with a
first-submit CE copy timeout.  The map/unmap cycle appears to take
BAR2 PTE resources RM was actively using on the freshly-created
channel.

**Iteration 2 — persistent-kva-first with a memdescMap fallback.**
Restored mc (its sysmem RAMFC has a persistent kernel mapping)
but surfaced a different bug on CUDA.  On Hopper + GSP-client,
`memdescGetKernelMapping(pRamfcDesc)` returns NULL (GSP firmware
writes RAMFC server-side; the host never CPU-maps the page), so
every CUDA channel took the fallback path.  That path routes:

```
memdescMap
  → kbusMapFbApertureSingle_IMPL
  → kbusMapFbAperture_GM107
  → kbusGetStaticFbAperture_TU102           (kern_bus_tu102.c:1027)
    → portMemAllocNonPaged(pMemArea->pRanges)    (:1126)
    → _kbusUpdateStaticBar1VAMapping_TU102
        → returns NV_ERR_NOT_SUPPORTED because page size < 2 MiB
    → error path: portMemFree(pMemArea->pRanges)  (:1141/1146)
        → os_free_mem → vfree → WARN "nonexistent vm area"
                              → WARN "bad address"
```

RAMFC is a 0x200-byte FBMEM memdesc; the static-BAR1-aperture path
is designed for ≥ 2 MiB mappings (`kern_bus_tu102.c:767-769`) and
its error-exit `portMemFree` tries to `vfree()` a pointer that was
never `vmalloc`'d.  Observed 44 paired WARNs per
4 MiB CUDA round-trip run (2 × 20 CUDA channels).  Not a
crash — a WARN — but real and ours.

**Iteration 3 (current) — lookup table, no RAMFC access.**  The
values we want (`gpFifoOffset`, `gpFifoEntries`) are passed in by
the client at channel alloc time via `NV_CHANNEL_ALLOC_PARAMS`, and
are literally what `kfifoWriteRamfcGpfifo_GH100` writes into
GP_BASE/GP_BASE_HI/GP_INFO.  So we stash them at
`kchannelConstruct_IMPL` in a 4096-slot linear-probed hash table
keyed on `(pGpu, runlist, chid)`, and the resolver's
`_dbellGpfifoLookup` returns them in O(load factor) without any
MMIO or BAR-aperture setup.  On miss (channels that pre-existed
module load, or an overflow that 4096 slots makes practically
impossible), the resolver returns `entries = 0`; the `#DB` handler
already treats `entries = 0` as "skip PB decode" (the
`cache->gpfifo_entries > 0` guard in `nv_dbell_db_handler`, and the
matching guard in `nv_dbell_resolve_fn` before it maps the ring), so the
`dbell/*` records still emit and the forward-write to real BAR0 still
happens — only `pb/submit` is silently dropped.

Diagnostic: `mc1 dbell/gpfifo_lookup result=table chid=N gpu_va=0x…
entries=N` on hit, `mc1 dbell/gpfifo_lookup result=not_in_table chid=N
runlist=N` on miss.
No `tmp_map` marker appears, because the memdescMap path it reported
on is gone.

**Coupled fix in kernel-open.**  Removing the slow memdescMap path
made the resolver substantially faster, which exposed a pre-existing
use-after-free in `g_dbell_cache`.  The cache stores `userd_kva` and
`gpfifo_kva` pointers into the BAR1 ioremap; but
`nv_dbell_bar1_track_remove` did `iounmap` without evicting the
cache.  A late doorbell fired during process teardown (LD_PRELOAD'd
CUDA) would then dereference freed memory (`CR2: ...208c` inside
`nv_dbell_db_handler+0x130`, faulting on USERD offset `+0x8c =
GPPut`).  The fix: a new `nv_dbell_cache_invalidate_range(kva_base,
size)` walks `g_dbell_cache`, zeroes `userd_kva`/`gpfifo_kva` of
any entry that points into the region, and moves the slot to FREE
before `iounmap` runs.  The trap-side reader already has an
`smp_rmb()` + NULL check on `cache->userd_kva` (in
`nv_dbell_db_handler`, on the `NV_DBELL_CACHE_RESOLVED` branch) which
closes the observable race —
a reader that saw state==RESOLVED before invalidation either reads
the stale kva while it's still mapped, or reads NULL after we zero
it.  No deref of freed memory either way.  Stress test post-fix:
40/40 back-to-back runs (plain + LD_PRELOAD), zero faults, zero
WARNs.

### 12.11 What was open at the time, and where it landed

The two items below were open when §12 was written; both were
subsequently implemented, and this subsection is kept to record the
sequence rather than the state.

- **Pushbuffer byte readout.**  Then: the submission record emitted
  `(pb_va, pb_len)` but not the method-stream bytes.  Now: the `#DB` handler
  also emits `mc1 pb/bytes`, chunked, and the parser reassembles it —
  `address_atlas.py` decodes the full NVC8B5 stream per submission.
- **Offline parsing of the pushbuffer records.**  Then: the lines
  were ftrace-emitted but the tools didn't read them.  Now:
  `strace_diff.py` parses them into `pb` / `pb_bytes` events; the
  shipped capture resolves 112 of 112 submissions with zero misses.
- **UVM-mapped GPFIFO tracking** for mc-style paths (GPFIFO not
  in a BAR1 mapping) — still open.

## 13. VA-Pool Fix for Intermittent D2H Failures (2026-05-12)

For several weeks the H2D → D2H round-trip run had a ~25–34 % failure
rate at `--size 128M` on H100 PCIe.  Two visible modes, both running
on byte-identical ioctl/GSP-RPC traces versus passing runs of the same
build:

- **Mode A** — semaphore-write timeout.  PBDMA consumed the H2D
  submission (`GPPut = GPGet = 0x2`) but the release-one-word
  semaphore never hit its expected value.  100 % of Mode A cases
  hung in the H2D stage of roundtrip, not the D2H.
- **Mode B** — verify-FAILED, with the corrupt readback value split
  between an invariant `0x20018000` (84 %, looks like a leaked
  MMU/PTE-adjacent word) and the literal `GARBLE_PATTERN`
  (`0xcafebabe`, 16 %, = silent H2D no-op).

Earlier sessions had ruled out PTE-alias between d_buf and the
pushbuffer (kernel PTE_HDR/PTE_ROW tracepoints showed zero physical
page overlap), every single-knob alloc-flag variation on d_buf
(PHYSICALITY, ALIGNMENT_FORCE, MEMORY_HANDLE_PROVIDED,
PERSISTENT_VIDMEM), the non-GRCE LCE switch, and the structural
reorder that matched libcuda's allocation ordering.  All left the
rate unchanged or made it worse.

### 13.1 Diagnosis via libcuda strace differential

The session's unlock was a careful read of libcuda's strace from a
`trace_cuda.sh` capture of a 128 MiB CUDA round-trip.  Two
calls early in the process stood out:

```
mmap(0x200000000, 4297064448, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    = 0x200000000
...
mmap(0x200600000, 58720256, PROT_READ|PROT_WRITE,
     MAP_SHARED|MAP_FIXED, 18, 0) = 0x200600000
```

libcuda reserves a 4 GiB PROT_NONE anonymous window at VA
`0x200000000` up front, then MAP_FIXED-mmaps every file-backed RM
memory (including the 56 MiB sysmem pushbuffer pool at offset
`0x600000` = VA `0x200600000`, fd 18 from `/dev/nvidiactl`) inside
that window.  The subsequent `UVM_CREATE_EXTERNAL_RANGE` + 
`UVM_MAP_EXTERNAL_ALLOCATION` calls use those same VAs.  Every
UVM-mapped allocation therefore satisfies Paper Finding 1 (GPU VA
== user VA) by construction: the CPU alias, the UVM range, and the
GPU VA that the channel's RAMFC records as `GP_BASE` are all the
same address.

We had been violating F1 systematically.  `uvm_map_buffer()`
reserved a fresh anonymous `PROT_NONE` window at an ASLR-random
address (we observed `0x7235daa00000`-class VAs) and passed that as
the UVM GPU-VA base.  `rm_map_memory()` and `rm_alloc_sysmem()`
meanwhile each mmap'd their file-backed mapping at whatever address
RM's `pLinearAddress` / `pMemory` out-parameter returned — also
kernel-picked.  Every UVM-mapped allocation had **two** CPU VAs:
one for CPU access, a different one used as the UVM "GPU VA"
handle.  That's an F1 violation, and empirically it wedged the
GPU ~25 % of the time.

### 13.2 Fix — libcuda-style VA-pool

Three helpers replace the old interfaces (see `libmc`'s
[`mc_vaspace.c`](https://github.com/maxime-peim/libmc/blob/main/mc/mc_vaspace.c),
[`mc_rm.c`](https://github.com/maxime-peim/libmc/blob/main/mc/mc_rm.c), and
[`mc_uvm.c`](https://github.com/maxime-peim/libmc/blob/main/mc/mc_uvm.c)):

- **`va_pool_init()`** — `mmap(VA_POOL_BASE=0x200000000, 4 GiB,
  PROT_NONE, MAP_FIXED_NOREPLACE)` at process init.  Bump allocator
  in `va_pool_reserve(size)`, 2 MiB-aligned (UVM's native page
  size).
- **`rm_alloc_sysmem_at(want_va)` / `rm_map_memory_at(want_va)`** —
  drop-in callers of `NV_ESC_RM_ALLOC_MEMORY` / `NV_ESC_RM_MAP_MEMORY`
  that MAP_FIXED the file-backed VMA at `want_va` instead of at RM's
  suggestion.  The kernel's mmap context lives on the fd, not on
  `pMemory`/`pLinearAddress`, so passing an arbitrary VA works — the
  ioctl fills pMemory as OUT but we ignore that value on the mmap
  side.  Verified: `p.params.pMemory` is out-only
  (`rmapi_deprecated_allocmemory.c:164` zeroes `*pAddress` before
  `RmAlloc`, then fills it).
- **`uvm_map_buffer_at(cpu_va)`** — `UVM_CREATE_EXTERNAL_RANGE` +
  `UVM_MAP_EXTERNAL_ALLOCATION` at a caller-supplied base, no
  PROT_NONE reservation.  Callers pass the same pool VA that
  `rm_alloc_sysmem_at` / `rm_map_memory_at` returned.

Call-site pattern (pseudocode):

```c
void *slot = va_pool_reserve(size, "h_buf");
h_h_mem = rm_alloc_sysmem_at(..., slot, &h_ptr);  // MAP_FIXED into slot
h_gpu_va = uvm_map_buffer_at(..., h_h_mem, h_ptr, size, "h_buf");
```

Every allocation with a CPU alias (h_buf, pushbuffer, staging,
gpu_ctl's BAR1 alias, and the sema) goes through the pool.  (At the time
of this fix the semaphore was HBM with a BAR1 alias; it has since moved
to sysmem with a direct CPU pointer — see the comment on the sema
allocation in [`libmc/mc/mc_core.c`](https://github.com/maxime-peim/libmc/blob/main/mc/mc_core.c)
— but it still lives in the pool, so
the invariant is unchanged.)
`d_buf` has no CPU alias — it is CE-filled from `staging`, never
touched by the CPU — so it stays on the plain `uvm_map_buffer()`
PROT_NONE path outside the pool.

### 13.3 Measured rate impact

| Binary                  | N     | Pass  | Mode A | 0x20018000 | Garble |
|-------------------------|-------|-------|--------|------------|--------|
| Pre-fix                 | 100   | 66    | 15     | 16         | 3      |
| gpu_ctl + pb anchor     | 100   | 83    | 1      | 6          | 10     |
| Full VA-pool            | 100   | 100   | 0      | 0          | 0      |
| Full VA-pool            | 1000  | 1000  | 0      | 0          | 0      |

The intermediate "gpu_ctl + pb anchor only" measurement is notable:
it moved overt Mode A nearly to zero (15 → 1) but *increased* the
silent garble variant (3 → 10).  Partial F1 conformance converted
hard hangs into silent no-ops.  Full conformance drove both to
zero.  Why the GPU/GSP pipeline wedges on split-VA bookkeeping we
still do not know mechanistically — the most plausible shape is
that some PBDMA or GSP cache keys on the `GP_BASE`/user-VA pair
and gets confused when they disagree.  We have the behavioural
evidence; the precise micro-architectural cause would need vendor
internals to confirm.

### 13.4 Secondary: tooling symmetry restored

An initial implementation of the pool used `mremap(MREMAP_FIXED)` to
relocate RM-picked VMAs into the pool after the fact.  Functionally
correct (1000/1000), but broke `pb/bytes` decoding on
mc traces: `bar1_track_add()` / `sysmem_track_add()` in
`nv-doorbell-watch.c` record `vma->vm_start` at the time
`nvidia_mmap` is called, and don't follow the VMA when mremap
moves it.  Trackers then carried stale VAs and
`nv_dbell_bar1_gpu_va_to_kva()` missed.

Fixed purely in userspace by eliminating mremap entirely: the
`_at()` helpers described above land each mapping at its final VA on
the *initial* mmap, so bar1/sysmem trackers record the correct VA
the first time.  No kernel change needed.  Post-fix, a
`trace_cuda.sh` capture of a 128 MiB `mc` H2D → D2H round-trip
produces 2 `pb/submit` / 2 `pb/bytes` / 0 `pb/bytes_miss` per run with the decoded
method stream matching libcuda's shape exactly (OFFSET_IN_VA at
`0x200xxxxxxx`, OFFSET_OUT_VA at d_buf, `LINE_LENGTH=0x08000000`,
`SET_SEMAPHORE_PAYLOAD=1`/`2`, and the split `LAUNCH_DMA` pair).

### 13.5 Files changed

- `libmc/mc/mc_vaspace.c`, `libmc/mc/mc_rm.c`, `libmc/mc/mc_uvm.c` —
  VA pool (init/reserve/clear), `rm_alloc_sysmem_at`,
  `rm_map_memory_at`, `uvm_map_buffer_at`,
  pool-routed allocations for every UVM-mapped buffer with a CPU
  alias.  Net `+ ~350` lines including comments.  Old PB filter
  bump: `PB_SIZE` 64 KiB → 1 MiB to pass `nv-mmap.c`'s 256-page
  sysmem_track threshold.
- `reverse/tools/strace_diff.py` — two-line bug-fix on
  `map_bar1` tuple unpack (was `(handle, len, flags)`, actual
  tuple is `(handle, len, flags, off, fd)`).  Unblocks
  `--handle-diff` and `--handle-history` modes that were
  crashing on ftrace input.

Kernel NOT modified in this fix.

### 13.6 Refutations and supersessions

- Does NOT contradict the PTE-alias refutation (§13.9, row 1).
  Physical pages still do not overlap.  The F1 violation operates
  in GPU-VA address space, not in PTE-content space.
- Supersedes the "position-dependent failures" hypothesis from the
  same day's morning session.  "First run always passes, failures
  onset at run 6–8" was in fact a side effect of
  GSP/driver state that had not yet been contaminated by the F1
  violation's cumulative effects.  With F1 respected, every run
  is a "first run."
- Supersedes all prior alloc-flag experiments.  These were all asking
  "what RM alloc flag do we need?" when the question should have been
  "what VA layout does the GPU need?".

### 13.7 Methodological note

The technique that worked was scientific-debugging differential
against libcuda as ground truth, with libcuda's strace read
carefully enough to notice the up-front 4 GiB PROT_NONE reservation
at a fixed low address.  That structural choice does not surface in
an ioctl count summary or a handle-role table — it only shows up
in the target address of the very first `mmap()` in the trace.
Prior sessions had read the same strace and produced summaries
like "libcuda has 29 UVM ranges vs our 6" and "libcuda maintains
an ~20-entry staging pool" — both true and both irrelevant to this
bug.  The bug was orthogonal: a VA-layout convention that applies
whether the program has 1 channel or 20.

### 13.8 What's still open

- **`address_atlas.py` role inference** for pool-allocated handles: the
  post-pass correlator doesn't yet link pool-placed VMAs to their
  `hMemory`, so some `methods.txt` entries show `class=None` even
  though the underlying method-stream decode is correct.
  Cosmetic only.
- **Understanding the GPU-side wedge mechanism.**  We know F1
  violation reproduces a 25–34 % failure rate and F1 conformance
  eliminates it at N=1000.  We do not know which PBDMA / GSP /
  MMU component specifically mis-behaves on split-VA bookkeeping.
  Would require vendor-internal instrumentation (GSP trace, PBDMA
  state dump) to nail down.  (A *different* GPU-side coherency
  question — the SM-poll-loop visibility gap on FB-resident
  semaphores — was diagnosed and fixed on 2026-05-31; see §14.1.
  The two are unrelated bugs but share the lesson that
  cross-engine memory-visibility on Hopper is not always what PTX
  scope modifiers suggest it is.)

### 13.9 Dead-end attempts and lessons learned

The Paper-F1 fix took roughly three weeks of wall-clock debugging
across multiple sessions before the real cause was identified.  We
burned that time on hypotheses that turned out to be wrong.  This
subsection catalogs them so future debugging sessions do not re-run
the same experiments.  Every entry records the evidence that
refuted it.

| # | Hypothesis | Why we tried it | Refuting evidence | Lesson |
|---|------------|-----------------|-------------------|--------|
| 1 | **PTE aliasing** — d_buf and pushbuffer share a physical page in the GPU MMU | `got 0x20018000` looked like a leaked MMU word; `--debug-readback` hit rate ~15 % suggested a page-granularity alias | Added the kernel `mmu/pte_hdr` + `pte/row` records logging every PTE value; captured pass + Mode A + Mode B at 128 MiB.  Zero phys-page overlap between any handle pair in any run. | Physical-page aliasing was the wrong layer.  The problem lived in GPU-VA bookkeeping, not PTE content. |
| 2 | **`PHYSICALITY_CONTIGUOUS` vs `ALLOW_NONCONTIGUOUS` on d_buf** | Strace-diff showed libcuda sets `ALLOW_NONCONTIGUOUS` on its bulk vidmem; ours set `CONTIGUOUS` | N=100 after flipping: 63/100 pass — identical to pre-change baseline. | Alloc-flag knobs that control physical-page bookkeeping are the wrong layer.  The bug was in VA layout. |
| 3 | **Full alloc-flag bundle** — `ALIGNMENT_FORCE \| MEMORY_HANDLE_PROVIDED \| PERSISTENT_VIDMEM` on d_buf | Remaining alloc-flag differences vs libcuda | N=100 after adding: 63/100 pass — identical to pre-change AND to the ALLOW_NONCONTIGUOUS-only measurement.  Three independent N=100 runs landing on 63 is the kind of replication that should be suspicious. | Once you've ruled a layer out once, don't try more flags in that layer.  Move up a layer. |
| 4 | **Drop d_buf's BAR1 alias, CE-stage fill from sysmem** | Hypothesis: the BAR1 CPU alias on d_buf was the source of split-VA coherency problems | Rate went 15 % → 37 % fail.  New "Mode A" variant appeared: staging-fill CE hit a timeout at offset 0. | Removing one split-VA allocation without fixing the root cause can make things worse by exercising new timing-sensitive code paths.  (Ironically, the CE-staged fill became part of the final solution once F1 was respected everywhere else.) |
| 5 | **Non-GRCE LCE selection** | libcuda prefers a non-GRCE LCE; ours was on the GRCE | Rate went 7 % → 15 % fail.  Same failure signature, doubled occurrence. | "libcuda does X" is necessary but not sufficient justification.  You also need a mechanistic argument for why the difference would matter.  Without that, you're pattern-matching, not reasoning. |
| 6 | **BAR1 doorbell alone** (`bBar1Mapping=NV_TRUE` on HOPPER_USERMODE_A) | libcuda writes its doorbell through the BAR1 variant | Rate regressed 75 % → 65 % pass at N=100.  Combining with FBMEM USERD later returned to 75 % baseline. | The intermittent failure rate was orthogonal to doorbell-aperture choice.  The regressions were WC-doorbell × cacheable-USERD coherency side effects, not the real bug. |
| 7 | **Structural reorder** — USERMODE_A before channel, UVM split in two passes, channel before bulk UVM maps | Matched libcuda's allocation ordering | N=30 at 128 MiB: ~13 % fail, same Mode A / B signatures.  Kept on structural-hygiene grounds; rate unchanged. | Allocation _ordering_ wasn't the issue.  Allocation _addressing_ was. |
| 8 | **Position-dependent failures** — "first run always passes, failures onset at runs 6–8" | Observed empirically across two reboot trials, 10/10 first-runs passed | The correlation was real but the causal story was wrong.  Once F1 was respected (VA-pool fix) every run became a "first run."  The accumulating state was actually GPU/GSP state slowly corrupted by the F1 violation itself, not a separate aging mechanism. | When a correlation reproduces reliably, resist the urge to reify it into a causal story.  A stable correlation can itself be a symptom of the deeper bug. |
| 9 | **Emulate libcuda's CREATE_RANGE_GROUP** | libcuda issues 8× `UVM_CREATE_RANGE_GROUP`; ours issues 0 | Before implementing: checked ftrace — libcuda creates 8 groups but never follows up with `UVM_SET_RANGE_GROUP`.  Empty range groups have no TLB / invalidation effect.  Emulating would have been noise. | "libcuda does X" is not actionable until you check "does libcuda _use_ X".  Allocation without use is an ignorable artifact. |
| 10 | **Partial F1 anchoring without a pool** (reverted) | Once we understood the F1 requirement, tried to anchor h_buf/sema/staging's UVM GPU VA to their already-mmap'd sysmem / BAR1 CPU VAs directly | Intermittent `UVM_CREATE_EXTERNAL_RANGE NV_ERR_IN_USE` (0x68): kernel-picked sysmem VAs sometimes collided with UVM's internal reserved range.  Larger allocations failed more often. | You cannot retro-fit Paper F1 onto allocations whose VA was picked by an allocator that doesn't know UVM's reserved window.  You have to pre-reserve the pool and force subsequent mappings to consume from it. |
| 11 | **Initial mremap-based VA-pool implementation** (superseded in-place) | First cut at the VA pool used `mremap(MREMAP_FIXED)` to relocate already-created VMAs into the pool | Functionally correct (1000/1000).  But broke kernel-watchpoint `pb/bytes` decoding because `bar1_track_add` / `sysmem_track_add` record `vma->vm_start` at nv-mmap time and don't follow mremap moves.  Trackers carried stale VAs; `nv_dbell_bar1_gpu_va_to_kva` missed. | Eliminate mremap entirely.  Land the VMA at its final VA on the _initial_ mmap via `MAP_FIXED` at a caller-supplied target — which is what the `_at()` helpers in the final fix do.  Kernel-side tracking that records VA at map-time is a hard constraint userspace has to respect. |

#### Meta-lessons for the next person

Patterns that emerged across the 11 dead ends:

1. **Ask which layer you're poking before you poke it.**  Before
   proposing an alloc-flag change, ask: "does this touch the
   allocation's _address_, or just its physical-page bookkeeping?"
   If the latter, it's probably the wrong layer.  We spent
   ~three sessions on dead-ends 1–3 because we kept working at
   the flags layer when the bug lived in VA addressing.

2. **"Matches libcuda" is a necessary but not sufficient
   justification.**  Dead ends 5, 6, 9 all started from "our code
   differs from libcuda here".  Sometimes that difference matters
   (the final fix!), sometimes it's irrelevant (empty range
   groups).  The filter is: check both that libcuda does X AND
   that libcuda _uses_ X to affect behavior.  Allocation without
   use is an artifact; behavioral difference without mechanism is
   pattern-matching.

3. **Small structural improvements that don't move the rate are
   worth keeping but should not be confused with "the fix".**
   Dead ends 5, 7 are the clearest example: kept on
   structural-parity grounds, bug rate unchanged.  Two common
   failure modes here were (a) "we changed something, the rate
   bounced up to 75 %, so this is the fix" (no — N=100 noise
   floor is ±5), and (b) "we matched libcuda, so the rest of the
   bug must be elsewhere" (survivor bias: we stopped investigating
   libcuda-matching candidates).

4. **N < 100 rate measurements lie.**  Three separate prior
   sessions produced false positives from N=20 runs that
   evaporated at N=100 (the clearest example: a 20/20 "fix" that
   was sample-size noise).  Current rule: every rate claim is N ≥ 100; confirmation at N =
   1000 for anything approaching 100 %.

5. **A stable correlation can be a symptom, not a signal.**  The
   "first run always passes" observation in dead-end 8 was
   reproducible across two reboot trials and across two
   independent observers.  It was real.  But the causal story we
   built around it (state accumulates across process launches →
   some aging mechanism → find the aging mechanism) was wrong.
   The correlation was a downstream side effect of the F1
   violation slowly corrupting GPU/GSP state run-over-run.  When
   a correlation reproduces cleanly, keep looking for simpler
   upstream causes before you commit to the correlation's
   implied mechanism.

6. **Kernel-side tracking constrains userspace-side choices.**
   Dead end 11 (mremap) is a good example: userspace-only fixes
   interact with kernel-side observers whose contracts you have
   to respect.  Specifically, `nv-doorbell-watch.c`'s trackers
   record `vma->vm_start` at registration — that's a hard VMA
   -identity constraint we inherit.  Any userspace VMA relocation
   story has to preserve the VMA whose `vm_start` was recorded,
   which in practice means "don't relocate; land at the final VA
   from the start".

7. **The real unlock was reading libcuda's strace for _structural
   anchors_, not ioctl counts.**  Prior sessions had produced
   summaries like "libcuda has 29 UVM ranges vs our 6" and
   "libcuda maintains an ~20-entry staging pool" — both true and
   both irrelevant to this bug.  The signal was in the _target
   address_ of the very first `mmap()` in libcuda's strace:
   `mmap(0x200000000, 4 GiB, PROT_NONE, MAP_ANONYMOUS)`.  That's
   a structural choice, not an ioctl-count choice; it doesn't
   surface in summaries or handle-role tables.  When reading
   traces, explicitly scan for: up-front reservations, fixed
   addresses, MAP_FIXED versus kernel-picked VAs, and the
   relative ordering of "reserve VA" versus "populate VA"
   calls.  These structural anchors are where invariants live.

These lessons are not specific to this bug.  They are general
patterns for debugging intermittent GPU-driver bugs where the
"known-good" reference is a closed-source binary (libcuda) and the
differential-against-ground-truth methodology is the main tool.

## 14. Fully GPU-Resident Control Plane (mc FB carrier, 2026-05-29 to 2026-05-31)

### 14.1 Goal and result

Up to mid-2026-05 the mc library (then under `reverse/mc/`, now in `libmc/mc/`) had two VAS
arms: `MC_VAS_UVM` and `MC_VAS_SYSMEM_CARRIER`.  Both still required
PCIe traffic on the SM-authored hot path: the SM's pushbuffer /
GPFIFO entry / USERD GPPut writes went to sysmem (BAR1-aliased) and
crossed PCIe per dword.  The motivating experiment for this work
was a third arm — `MC_VAS_FB_CARRIER` — where the channel resources
(PB / GPFIFO / USERD, and at the time an SM-polled DMA semaphore) were
to live in **FB (HBM)**.  What ships keeps the host-polled release
semaphore in sysmem; only the resources the SM itself writes moved to
FB.  The SM
authors the entire submission protocol as FB↔GPU-L2 traffic; only
the BAR1 doorbell ring at the end is an MMIO write.

Result, end-to-end on H100 PCIe: `mc_carrier_demo --fb
--size 1M` PASSes 1000/1000 with median **189.4 µs / 5.54 GB/s** for
a 1 MiB SM-authored D2H (vs 273.4 µs / 3.84 GB/s pre-cleanup; the
30 % improvement is from removing the host's redundant BAR1
re-read of the FB sema after submission; the host still polls a
sysmem release sema — see §14.4).  N=1000 stability sweep, post-cleanup N=50 sweep, sysmem
regression sweep all clean.

### 14.2 The actual bug — and the two layers of misdirection

The first attempt at FB-carrier wedged: USERD GPPut stayed at 0
from the host's view, the SM-side DMA-sema poll timed out, no Xid.
Two compounding misdirections sent the diagnosis off twice before
the real cause emerged.

**Misdirection 1 — the carrier-class hypothesis (2026-05-28).**
Initial theory was that `NV50_MEMORY_VIRTUAL` (class 0x50A0) was
re-routing the GMMU PTE through its own typed-heap-allocated FB
backing, so the SM's MMU walk for USERD landed on a different
physical page than `memdescGetPhysAddr(AT_GPU)` reported to PBDMA.
We added three permanent kernel records to test this —
`mmu/pte_src_decision`, `mmu/intermap_call`, `mmu/virtmem_backing`
(see §14.5).  The traces showed `chosen_pte0 == src_pte0` for every
NVOS46, and `has_heap=0` on every NV50 carrier libcuda allocates.
Hypothesis falsified.

We then *also* tried flipping our carriers to `NV01_MEMORY_VIRTUAL`
(class 0x70) on the theory that NV01 is "cleaner."  That hit
`NV_ERR_NO_MEMORY` from `dmaAllocMapping_GM107` and was reverted.
libcuda itself uses NV50_MEMORY_VIRTUAL with `has_heap=0` for its
non-UVM carriers; the morally correct refactor was to drop our
single-bump-allocator design and allocate a fresh NV50 carrier per
source `hMemory` with `dmaOffset=0`.  See §14.3.

**Misdirection 2 — the cache-flush hypothesis (2026-05-31 morning).**
After the carrier refactor every PTE was correctly routed but the
wedge persisted.  A strong-sounding theory suggested itself: SM
`st.relaxed.sys.global.u32` to FB USERD
commits to GPC L2, but PBDMA's USERD fetcher reads through HOST/XAL
and doesn't snoop GPC-L2 dirty lines.  We have direct evidence
this is a real concern in tree:
`kbusFlushPcieForBar0Doorbell_GH100`
(`kbusFlushPcieForBar0Doorbell_GH100` in
`src/nvidia/src/kernel/gpu/bus/arch/hopper/kern_bus_gh100.c`)
calls `kbusFlush_HAL(BUS_FLUSH_VIDEO_MEMORY)` *before every BAR0
doorbell* on Hopper, and `kbusSendSysmembarSingle_GH100` (`:2966`)
issues an explicit `NV_XAL_EP_ZB_UFLUSH_FB_FLUSH` token-counter
handshake to drain HSHUB write-coalescing.  The proposed fix was
an SM-issued write to a HOPPER_USERMODE_A MEMBAR register
mirroring the host-side flush.

We added a first round of probes — host-side BAR1 reflection re-read
after SM timeout, plus an SM-side self-readback of USERD GPPut via
`ld.acquire.sys.global.u32` — to test the theory.  The probes
**falsified it** instead.  USERD GPPut DID reach FB (post-100 ms
host BAR1 read sees `1`).  PBDMA DID run the CE op (the DMA-sema
release fired, host BAR1 sema_actual=1).  The SM saw its own USERD
write through its own MMU walk (self-readback returns the expected
value).

**The actual cause — the SM's *read* path, not its writes.**  The
SM kernel's poll loop on the FB-resident DMA semaphore was
`while (budget != 0) { if (ldg_u32_sys(sema_ptr) == expected) break; ... }`.
The compiler lowered `ld.relaxed.sys.global.u32` to
`LDG.E.STRONG.SYS`.  The first iteration cached the FB sema's
current value (zero) in the SM's L1, and every subsequent
iteration **re-served that L1 line** instead of fetching from L2
where PBDMA's release would eventually appear.  The loop spun
forever on stale L1 even though FB held the fresh value.  The fix
is `__threadfence_system();` + `ld.global.cg.u32` (cache-global,
bypass L1) per iteration; this lowers to `MEMBAR.SC.SYS` +
`LDG.E.STRONG.GPU` and reliably observes PBDMA's release in tens
of microseconds.

The SM observes the release in ~133k SM-clock ticks (~67 µs at
~2 GHz), p99 within ~1 % of median, max ever seen ~79 µs over an
N=1000 sweep.  The 4-GiB-tick budget (~2 s) used during diagnosis
was 30,000× the worst observation; the production cap is
2^28 ≈ 268M iterations as defence against a wedged victim.

The PTX-level lesson: **`.acquire.sys` and `.relaxed.sys` are
memory-ordering modifiers, not cache-class modifiers.**  They
order subsequent ops after the load but do not invalidate the L1
line the load itself hits.  `.cg` is the cache-class modifier
that actually bypasses L1.  Cross-engine spin-wait on Hopper
needs both: `__threadfence_system();` for visibility, `.cg` for
freshness.

### 14.3 Per-resource carrier refactor (2026-05-29)

While diagnosing misdirection 1 we captured libcuda's actual
carrier shape via the new `non_uvm_ledger.py` consumer (§14.5).
For each non-UVM channel libcuda allocates a fresh
`NV50_MEMORY_VIRTUAL` per source `hMemory`, sized to the source,
mapped with `NV04_MAP_MEMORY_DMA(dmaOffset=0)` so RM picks the
GPU VA from the FERMI VAS's heap.  All `has_heap=0`.  No
single-mega-carrier-with-bump-cursor design anywhere in libcuda's
trace.

We refactored `mc_vaspace.c` to match: deleted
`mc_va_space_carve` and the `vas->h_virt` / `virt_base` /
`virt_size` / `virt_cursor` shared bump state; added
`mc_va_space_dma_map_resource` (allocate fresh NV50 +
NVOS46 with `dmaOffset=0`) and `mc_va_space_release_carrier`
(NVOS47 + free, called at channel teardown before the source
`hMemory` is freed); the VAS now owns a `carriers[64]` table.  All
five carrier-using tests (sysmem path) PASS post-refactor.

The refactor surfaced misdirection 2 (the visibility bug) but
didn't cause it — the per-resource shape was the right move
regardless, since it matches libcuda's recipe and removes a layer
of disagreement between our path and hers.

### 14.4 Hopper PCIe egress paths are not pairwise-ordered

A second architectural finding emerged from cleanup of the
diagnostic plumbing.  After fixing the SM poll-loop, we initially
kept a host-side check: `mc_sm_owner_submit` would re-read the
DMA sema via the BAR1 reflection after the compute kernel
returned, to verify PBDMA's release.  This re-check **raced**: it
saw `sema_actual=0` immediately after `mc_channel_submit`
returned MC_OK, and the diagnostic block several hundred µs later
saw `sema_actual=1`.

The mechanism: PBDMA's compute-sema release and an SM
`stg_u64_sys` to a sysmem cell both reach the host through the
GPU's host interface, but **through different egress queues**.
The compute-sema (which `mc_channel_submit` polls) becomes
host-visible before the SM's writeback to a separate sysmem cell
has drained.  No `clflush` saved this — `clflush` invalidates the
*CPU's* cache; the value isn't yet in main memory.  Only a
retry-poll waiting tens-of-µs for the SM's egress-queue to drain
worked.

Generalisation: **on Hopper PCIe, "compute-sema visible" does NOT
imply "all SM sysmem writes drained."**  Any host code reading
SM-written sysmem immediately after a compute-sema fires must
either (a) `clflush` + retry-poll the read with a small ms-scale
ceiling, or (b) keep verification on the GPU side and use the
compute-sema purely as a "kernel exited" signal.

We tried (b) for mc's FB-carrier path and abandoned it: an
in-kernel `.cg` spin on an FB-resident sema proved unreliable,
because LTC-line aliasing from the SM's own earlier FB stores in
the same kernel can mask PBDMA's later release-store.

What ships is a third option — move the release semaphore itself
out of FB.  The FB-carrier victim channel owns a dedicated
**sysmem** sema cell, and the host polls it directly as ground
truth; `mc_channel_submit` returning is *not* a completion proof,
since PBDMA may not have processed the newly-published GPFIFO
entry yet (see the routing-rule comment in `mc_sm_owner.c`).  The
SM kernel's `.cg` poll loop is still compiled in but disabled —
the host passes `dma_sema_poll_va = 0`.

What did get removed was the host's redundant BAR1 *re-read* of
the FB sema after submission; dropping those retries took median
latency at 1 MiB from 273 µs to 189 µs (~30 %).

### 14.5 Shipped tooling

Moved.  The eight `mmu/*` and `fifo/*` records this work made permanent,
and `non_uvm_ledger.py` — the per-channel ledger built on them, and the
artifact that surfaced libcuda's per-resource carrier shape — are
documented in `docs/tracing_cuda.md`.

### 14.6 Refuted hypotheses

Worth recording so future work doesn't re-derive them:

1. **"NV50_MEMORY_VIRTUAL re-routes the PTE through carrier-local
   FB pages."**  Falsified: every NV50 carrier in tree (libcuda
   *and* mc post-refactor) shows `has_heap=0` per
   `mmu/virtmem_backing`; `chosen_pte0 == src_pte0` per
   `mmu/pte_src_decision`.

2. **"Switching to `NV01_MEMORY_VIRTUAL` (class 0x70) avoids the
   PTE-rerouting issue."**  Hit `NV_ERR_NO_MEMORY` from
   `dmaAllocMapping_GM107` and was reverted.  libcuda doesn't use
   NV01 for non-UVM channel carriers anyway; the right answer was
   to match libcuda's NV50-with-`has_heap=0`, per-resource shape.

3. **"GPC L2 holds the SM's USERD write; need an XAL UFLUSH from
   the SM via HOPPER_USERMODE_A MEMBAR."**  Architecturally
   plausible (mirrors `kbusFlushPcieForBar0Doorbell_GH100`'s
   pre-doorbell flush), but the SM's writes were never the
   problem — the bug was on the SM's *read* path (stale L1 on
   the poll-loop `LDG`).  We never needed a USERMODE MEMBAR
   register; PTX `.cg` cache-class on the load was sufficient.

The general meta-lesson, mirroring §13.9 lesson 7: a theory can
have a strong architectural argument *and* verifiable in-tree
precedent and still be wrong about the case in front of you.  That
combination is not a substitute for a probe.  We added the first
round of probes before committing to the proposed fix; they took
~30 minutes and saved us from chasing a HOPPER_USERMODE_A MEMBAR
offset that doesn't exist on the open-tree BAR1 page.

## 15. The Missing PERF_BOOST: Why `mc` Ran at Idle Clocks (2026-08-18)

For weeks `mc_demo` sustained ~37–40 GB/s D2H in loops where
`cuda_reference` sustained 54.7, on the same box, in interleaved runs —
while `mc`'s *peak* iteration matched CUDA exactly.  Everything
structural had been eliminated by measurement: command stream (split
launches + sysmem semaphore, §2 and §7), engine assignment, PCIe
link state, host DRAM, instrumentation, and finally the memory map
itself (GPU-MMU PTEs bit-identical to CUDA's for both buffers — same
page sizes, same batching, attribute bits `0x68d` host / `0x681`
device on every PTE).

### The diagnosis

Per-iteration timing (100 iters × 4 interleaved rounds, 256 MiB D2H)
broke the mystery open.  CUDA's 400 iterations all fall in
4.87–4.95 ms.  `mc`'s are **trimodal** — and discrete modes mean clock
states, not protocol overhead:

| mode | count | rate |
|---|---|---|
| 4.91 ms | 52 | 54.7 GB/s — CUDA speed |
| 7.13 ms | 344 | 37.6 GB/s |
| 10.7 ms | 4 | 25.1 GB/s |

Sampling `nvidia-smi` clocks at 100 ms during the runs settled it: the
SM clock sat at **345 MHz (idle)** for the whole `mc` run and at
**1755 MHz (max boost)** for the whole CUDA run.  The Copy Engine is
clocked from the same domain, so at idle clocks it cannot feed PCIe
Gen5.  A copy-only workload never drives utilisation high enough for
the auto-boost governor to ramp on its own.

Why does CUDA get boost?  The `cuda_reference` capture contains exactly
one `NV2080_CTRL_CMD_PERF_BOOST` (`0x2080200a`) control on the
subdevice during init.  An LD_PRELOAD snoop on the live call read the
params: `flags = 0x12` (`CMD_BOOST_TO_MAX | CUDA_YES`),
`duration = 0xffffffff` (infinite — the RM clears the boost when the
client is freed).  libcuda asks for boost clocks once per context and
holds them for the client's lifetime; `mc` never asked.

### The fix and the result

`rm_perf_boost()` in `mc_rm.c` issues the identical control from
`mc_init`, right after subdevice creation.  Best-effort: on failure it
logs and continues, since the library still works at idle clocks —
just slower.  Verified paired, 100 iters × 4 interleaved rounds:

| | before | after |
|---|---|---|
| `mc` D2H Mean | 36.8–39.8 GB/s | **54.8 GB/s** (CUDA: 54.7) |
| `mc` p50 iteration | 7.13 ms | **4.89 ms** (CUDA: 4.90) |
| `mc` H2D Mean | ~45 | **55.4** |
| SM-authored D2H | ~38 | **53.1** |

The only slow iterations left are a handful at the head of the first
run after idle — the boost request takes a few hundred ms to land, and
CUDA hides the same ramp inside its ~500 ms init.  The full 51-test
matrix still passes.

### What this retroactively explains

- **The ~25 GB/s "floor mode"** chased across two boxes: 10.7 ms per
  256 MiB *is* 25.1 GB/s.  The floor was the 345 MHz idle-clock state
  (54.4 / 24.9 = 2.19, matching the 10.68 / 4.91 mode ratio).
- **`mc`'s session-to-session ceiling variance** (25 / 37.7 / 44.9 /
  54.5 while CUDA never moved): `mc` was riding whatever clock state
  previous GPU activity left behind; CUDA always requested its own.
- **Why short runs sometimes matched CUDA**: residual boost from an
  earlier CUDA run, not anything `mc` did.

One caution for anyone repeating the diagnosis on a cloud box: this was
a passthrough VM (whole GPU, MIG off, no other tenants possible on the
silicon), so clock policy was genuinely ours.  What *is* shared behind
a passthrough GPU is the host's PCIe fabric and DRAM; episodes where
*both* programs floor together remain consistent with neighbour traffic
and cannot be settled from inside the guest.
