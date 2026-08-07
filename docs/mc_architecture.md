# mc — Architecture Deep-Dive

**Hand-rolled GPU DMA transfer using only raw NVIDIA driver ioctls**

This document is the "missing manual" for `reverse/mc/` and the `mc_demo`
program built on it. It describes
the complete software and hardware path from `main()` to a completed PCIe DMA
transfer: every ioctl, every kernel code path, every hardware register, and every
bug encountered along the way. It is grounded entirely in the open-gpu-kernel-modules
source tree (driver 610.43.02, Hopper / H100 PCIe, kernel 6.8).

**Companion doc:** For the bit-exact formats of GPFIFO entries, method headers,
NVC8B5 methods, USERD, VF doorbell, and the end-to-end component interaction
timeline, see `docs/gpfifo_pushbuffer_reference.md`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Model](#2-system-model)
3. [RM Object Hierarchy Setup](#3-rm-object-hierarchy-setup)
4. [UVM Setup and Buffer Mapping](#4-uvm-setup-and-buffer-mapping)
5. [Channel Setup and Scheduling](#5-channel-setup-and-scheduling)
6. [VF Doorbell Mapping](#6-vf-doorbell-mapping)
7. [Work Submission (the hot path)](#7-work-submission-the-hot-path)
8. [GP Entry Extended-Base Encoding](#8-gp-entry-extended-base-encoding)
9. [The NVC8B5 Method Stream](#9-the-nvc8b5-method-stream)
10. [Semaphore Completion and Bandwidth](#10-semaphore-completion-and-bandwidth)
11. [Kernel Instrumentation Used](#11-kernel-instrumentation-used)
12. [Complete Bug Log](#12-complete-bug-log)
13. [Quick-Reference: ioctl Table](#13-quick-reference-ioctl-table)

---

## 1. Overview

`mc_demo` performs a GPU→host DMA copy (default 256 MiB,
configurable via `--size`) using only:

- `/dev/nvidiactl` — RM control fd
- `/dev/nvidia0` — RM GPU fd
- `/dev/nvidia-uvm` — UVM fd (for virtual address management)

It requires **no CUDA runtime**, **no libcuda.so**, and **no OpenCL** — just the
kernel driver and standard C. Once setup is complete (~500 ms of ioctls), each
subsequent transfer executes as a sequence of pure userspace memory writes (MMIO)
with zero kernel involvement.

**Achieved throughput:** on H100 PCIe (Gen5 x16), ~55 GB/s in both
directions at 256 MiB — about 87 % of the per-direction ceiling, and
within a percent of a CUDA reference measured the same way.  The
absolute plateau is box-dependent (an earlier box capped D2H at
~38 GB/s for CUDA too); the portable claim is the parity.  Reaching it
requires the boost request `mc_init` now makes — `docs/findings.md §15`.
Full table and method in `docs/findings.md §11`.  Reliability at `--size 128M`: 1000/1000 passes
at N=1000 after the VA-pool fix described in §3.7 and §12 bug #12
(pre-fix was ~66/100; see `docs/findings.md §13`).

---

## 2. System Model

```
┌───────────────────────────────────────────────────────────────────────────┐
│ CPU (userspace process: mc_demo)                                          │
│                                                                           │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────────────┐   │
│  │  pushbuffer  │    │  GPFIFO ring   │    │  USERD (gpu_ctl +0x2000) │   │
│  │  (host RAM)  │    │  (HBM via BAR1,│    │  (HBM via BAR1 alias)    │   │
│  │  CPU writes  │    │   in gpu_ctl)  │    │  GPPut @ +0x8c           │   │
│  │  methods     │    │  CPU writes    │    │  GPGet @ +0x88           │   │
│  │              │    │  GP entries    │    │                          │   │
│  └──────────────┘    └────────────────┘    └──────────────────────────┘   │
│          │                    │                          │                │
│          │            writes GP entry        MMIO write GPPut             │
│          │                    │                          │                │
│  ┌────────────────────────────┴──────────────────────────┴─────────────┐  │
│  │  VF doorbell (HOPPER_USERMODE_A BAR1 mapping, offset +0x90)         │  │
│  │  CPU writes work-submit-token → wakes host scheduler                │  │
│  │  (0x30090 is VF-block relative, not BAR0; via the BAR1 aperture,    │  │
│  │   matching libcuda's submission path)                               │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
          │                    │                          │
          ▼                    ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ GPU (H100 PCIe)                                                            │
│                                                                            │
│  PBDMA reads GPFIFO entry (from HBM) → finds pushbuffer VA                 │
│  PBDMA reads pushbuffer (from host RAM over PCIe) → parses NVC8B5 methods  │
│  CE executes: HBM read → PCIe TLPs → host DRAM write                       │
│  CE writes semaphore payload (4 bytes) → host DRAM (sema is sysmem)        │
└────────────────────────────────────────────────────────────────────────────┘

Buffer memory locations (all CPU-aliased buffers MAP_FIXED into a
4 GiB PROT_NONE VA pool at VA 0x200000000 — see §3.7):
  gpu_ctl      → 2 MiB HBM (NV01_MEMORY_LOCAL_USER) — holds GPFIFO
                 ring at offset 0 and USERD at offset 0x2000; BAR1
                 CPU alias lives in the VA pool, UVM-anchored there.
  Pushbuffer   → 1 MiB host RAM (NV01_MEMORY_SYSTEM), MAP_FIXED in pool.
  d_buf (src)  → transfer_size HBM (NV01_MEMORY_LOCAL_USER).  NO CPU
                 alias — CE-staged fill from `staging`.  UVM-mapped at
                 an anonymous PROT_NONE VA outside the pool (F1 not
                 required: no other path resolves d_buf's VA).
  h_buf (dst)  → transfer_size host RAM (NV01_MEMORY_SYSTEM), MAP_FIXED
                 in pool.
  sema         → 4 KiB host RAM (NV01_MEMORY_SYSTEM), MAP_FIXED in pool.
                 Sysmem, not FB, so the CPU's spin is a local cached read
                 rather than a BAR1 round-trip competing with the transfer.
  staging      → 2 MiB host RAM (NV01_MEMORY_SYSTEM), MAP_FIXED in pool.
                 CPU fills with FILL_PATTERN; CE copies to d_buf in
                 chunks.
```

**Key property (Yan et al. 2026, Finding 1):** With UVM active, GPU VAs = CPU VAs.
The method stream addresses the CE writes are *the same numbers* the CPU sees in its
virtual address space. No separate GPU→CPU VA translation step.

---

## 3. RM Object Hierarchy Setup

### 3.1 Opening device files

```c
ctl_fd = open("/dev/nvidiactl", O_RDWR);   // control fd for all RM ioctls
dev_fd = open("/dev/nvidia0",   O_RDWR);   // GPU-specific fd
```

Then `NV_ESC_REGISTER_FD` (0xc9) links `dev_fd` to `ctl_fd` so RM can find the
client context when dev_fd is referenced:

```c
nv_ioctl_register_fd_t reg = { .ctl_fd = ctl_fd };
ioctl(dev_fd, _IOWR('F', NV_ESC_REGISTER_FD, nv_ioctl_register_fd_t), &reg);
```

Source: the `NV_ESC_REGISTER_FD` handling in
`src/nvidia/arch/nvalloc/unix/src/escape.c` handles
REGISTER_FD, associating the two file structures in the per-device client
tracking table; it is reached through `nvidia_ioctl` in
`kernel-open/nvidia/nv.c`.  There is no `nv-frontend.c` in 610 — the
separate frontend module was removed upstream some releases earlier.

### 3.2 Alloc ioctl — NVOS64 is mandatory

All object allocations go through `NV_ESC_RM_ALLOC` (0x2b) with **`NVOS64_PARAMETERS`**
(48 bytes), NOT the 32-byte `NVOS21_PARAMETERS`.

```c
NVOS64_PARAMETERS p = {
    .hRoot         = h_client,
    .hObjectParent = parent,
    .hObjectNew    = next_handle(),
    .hClass        = hclass,
    .pAllocParms   = (NvP64)(uintptr_t)alloc_params,
    .pRightsRequested = 0,
    .paramsSize    = 0,
    .flags         = 0,
};
ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_ALLOC, NVOS64_PARAMETERS), &p);
```

Why NVOS64 specifically: the `case NV_ESC_RM_ALLOC` arm of `RmIoctl()` in `escape.c`
selects the path purely by payload size — `NvBool bAccessApi = (dataSize ==
sizeof(NVOS64_PARAMETERS));`. A 48-byte NVOS64 payload makes that TRUE, routing through
`Nv04AllocWithAccessSecInfo()` which properly propagates the security context to
GSP-RM on Hopper. NVOS21 goes through a legacy path that fails with
`NV_ERR_INVALID_OBJECT_PARENT` for channel allocs.

### 3.3 Object allocation sequence

| Order | Class (ID) | Parent | Purpose |
|---|---|---|---|
| 1 | `NV01_ROOT` (0x41) | self | RM client root — all other handles hang here |
| 2 | `NV01_DEVICE_0` (0x80) | h_client | Device object; params: `deviceId=0` |
| 3 | `NV20_SUBDEVICE_0` (0x2080) | h_device | Subdevice (GPU die); params: `subDeviceId=0` |
| 4 | `HOPPER_USERMODE_A` (0xc661) × 2 | h_subdevice | Two variants: BAR0 (`bBar1Mapping=NV_FALSE`, structural parity with libcuda, unused for writes) and BAR1 (`bBar1Mapping=NV_TRUE`, carries the active VF doorbell at offset 0x90). |
| 5 | `NV01_MEMORY_LOCAL_USER` (0x40) | h_device | d_buf — `transfer_size` HBM (default 256 MiB; DMA source) |
| 6 | `NV01_MEMORY_SYSTEM` (0x3e) | h_device | h_buf — `transfer_size` host DRAM (default 256 MiB; DMA dest), via nvos02 |
| 7 | `NV01_MEMORY_LOCAL_USER` | h_device | gpu_ctl — **2 MiB HBM shared region**; holds GPFIFO ring (512 × 8 B = 4 KiB) at offset 0 and USERD (4 KiB) at offset `USERD_OFFSET = 0x2000`.  USERD is aliased via `h_userd_mem = h_gpu_ctl_mem`. |
| 8 | `NV01_MEMORY_SYSTEM` | h_device | Pushbuffer — 1 MiB host RAM (PB_SIZE, bumped from 64 KiB so nv-mmap's 256-page sysmem_track threshold admits it) |
| 9 | `NV01_MEMORY_SYSTEM` | h_device | Semaphore — 4 KiB host RAM (where libcuda puts it; the host polls it directly) |
| 10 | `NV01_MEMORY_SYSTEM` | h_device | Staging — 2 MiB host RAM (STAGING_SIZE); CPU fills, CE copies to d_buf |
| 11 | `FERMI_VASPACE_A` (0x90f1) | h_device | VA space for channel (IS_EXTERNALLY_OWNED) |
| 12 | `KEPLER_CHANNEL_GROUP_A` (0xa06c) | h_device | TSG; `engineType=<non-GRCE LCE>`, `hVASpace=h_vaspace` |
| 13 | `HOPPER_CHANNEL_GPFIFO_A` (0xc86f) | h_tsg | CE channel (see §5.1) |
| 14 | `HOPPER_DMA_COPY_A` (0xc8b5) | h_channel | CE engine object; `NVB0B5_ALLOCATION_PARAMETERS.version=_V1` + matching engineType |

Allocation ordering matches libcuda's: `HOPPER_USERMODE_A` variants and
all memory allocations are issued _before_ the channel, so the channel
constructor sees the memory handles already live.

### 3.4 `NV01_MEMORY_LOCAL_USER` — vidmem allocation

Vidmem uses `NV_MEMORY_ALLOCATION_PARAMS` (from `nvos.h`). Critical bitfield positions
**are NOT at [1:0] and [3:2]** — they use DRF-shifted positions:

```c
NV_MEMORY_ALLOCATION_PARAMS mp = {};
mp.owner = h_client;               // must be client handle, not 0
mp.type  = NVOS32_TYPE_IMAGE;
mp.flags = NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT |
           NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED;
mp.attr  = (NVOS32_ATTR_LOCATION_VIDMEM       << 25) |  // bits [26:25] = 0
           (NVOS32_ATTR_PHYSICALITY_CONTIGUOUS << 27);   // bits [28:27] = 2
mp.size  = size;
mp.alignment = 0x10000;  // 64 KiB
```

Parent must be `h_device` (not `h_subdevice`). RM returns the HBM physical address in
`mp.address` — this is a physical offset, not a GPU VA. It becomes a GPU VA only after
UVM maps it (see [UVM Setup and Buffer Mapping](#4-uvm-setup-and-buffer-mapping)).

### 3.5 `NV01_MEMORY_SYSTEM` — sysmem allocation via nvos02

Uses `NV_ESC_RM_ALLOC_MEMORY` (0x27) with `nv_ioctl_nvos02_parameters_with_fd`. The
alloc + mmap happen atomically from the kernel side:

```c
nv_ioctl_nvos02_parameters_with_fd p = {};
p.params.hClass  = NV01_MEMORY_SYSTEM;
p.params.flags   = (1u << 4)   // PHYSICALITY = NONCONTIGUOUS
                 | (2u << 12); // COHERENCY = WRITE_COMBINE
p.params.limit   = size - 1;
p.fd             = alloc_fd;   // MUST be a fresh nvidiactl fd
ioctl(ioctl_fd2, _IOWR('F', NV_ESC_RM_ALLOC_MEMORY, ...), &p);
// Then: land the VMA at a caller-chosen VA (a pre-reserved slot
// inside the VA pool — see §3.7).  The fd-based MAP_FIXED picks up
// the physical backing from alloc_fd's mmap context, so the target
// VA is free for us to choose; we DO NOT use p.params.pMemory (which
// is RM's suggestion, and which we ignore to satisfy Paper F1).
mmap(want_va, size, PROT_READ|PROT_WRITE,
     MAP_SHARED|MAP_FIXED, alloc_fd, 0);
```

Three subtle requirements:
- `p.fd` must be a **nvidiactl** fd (not nvidia0), because `rm_create_mmap_context()`
  uses `nv_get_ctl_state()`.
- `ioctl_fd2` (the ioctl target) must be a **nvidia0** fd (NV_ACTUAL_DEVICE_ONLY).
- mmap offset is **always 0**; the physical mapping is encoded in the mmap context.

`p.params.pMemory` is an OUT parameter — RM's suggested CPU VA.
Using it directly as the mmap target is the tempting mistake: it
leaves UVM free to pick its own GPU VA, producing a split-VA bug
that cost ~25 % of runs before it was understood (`findings.md §13`).
Instead `rm_alloc_sysmem_at(want_va, …)` passes a caller-chosen pool slot
instead, and `uvm_map_buffer_at(want_va, …)` anchors the UVM range
at the same VA so GPU VA == CPU VA (Paper F1).

### 3.6 `FERMI_VASPACE_A` — the right flags

The VA space must be allocated with `IS_EXTERNALLY_OWNED`:

```c
NV_VASPACE_ALLOCATION_PARAMETERS vp = {
    .index = NV_VASPACE_ALLOCATION_INDEX_GPU_NEW,
    .flags = NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
};
```

Without this flag, `nvGpuOpsDupAddressSpace` (called by UVM's
`UVM_REGISTER_GPU_VASPACE`) checks `vaspaceIsExternallyOwned()` and returns
`NV_ERR_INVALID_FLAGS` (0x29). The flag is translated from alloc params to the
internal `VASPACE_FLAGS_IS_EXTERNALLY_OWNED` by the
`NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED` test in `vaspace_api.c`.

### 3.7 VA pool reservation (libcuda pattern, Paper Finding 1)

Immediately after `REGISTER_FD` and before any RM allocation, `mc_init`
reserves a 4 GiB PROT_NONE anonymous window at `0x200000000`:

```c
#define VA_POOL_BASE  0x200000000ULL
#define VA_POOL_SIZE  (4ULL << 30)
mmap((void *)VA_POOL_BASE, VA_POOL_SIZE, PROT_NONE,
     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
```

Every subsequent sysmem alloc (`rm_alloc_sysmem_at`) and BAR1 alias
(`rm_map_memory_at`) takes a pre-reserved slot inside this window as its
`MAP_FIXED` target; `uvm_map_buffer_at(cpu_va, ...)` then passes the
same VA to `UVM_CREATE_EXTERNAL_RANGE`.  The net effect is **GPU VA ==
user VA** for every UVM-mapped allocation that also has a CPU alias
(Paper Finding 1 — Yan et al. 2026): pushbuffer, h_buf, gpu_ctl
(GPFIFO+USERD's BAR1 alias), the semaphore (sysmem, mapped straight
into the pool — no BAR1 alias), and staging all share a single VA
across CPU, UVM, and RAMFC.

Why this matters: letting `uvm_map_buffer()` pick an ASLR-random
PROT_NONE VA for the UVM GPU VA while `rm_alloc_sysmem`/`rm_map_memory`
mmaps the CPU alias at an unrelated kernel-picked VA gives every
UVM-mapped buffer two
CPU VAs for the same underlying allocation.  Empirically the GPU
wedged ~25 % of the time on this split-VA bookkeeping — see §12 (bug
#12) and `docs/findings.md §13` for the 34 → 0 / 1000 rate measurement
and the suspected cause.

`d_buf` has no CPU alias (CE-staged fill from `staging`; CPU never
reads or writes d_buf) and so stays on the plain `uvm_map_buffer()`
path outside the pool — F1 is only needed for buffers whose CPU VA
is exposed to kernel-side tracking or to the GPU through any channel
other than PTE lookup.

This matches libcuda's observed pattern exactly: a single
`mmap(0x200000000, 4 GiB, PROT_NONE, MAP_ANONYMOUS)` is the first VM
reservation in a libcuda strace, followed by `MAP_FIXED` file-backed
mappings at offsets inside it (e.g. the 56 MiB sysmem pushbuffer
pool at `0x200600000`, fd pointing at `/dev/nvidiactl`).

---

## 4. UVM Setup and Buffer Mapping

### 4.1 Why UVM is necessary

CUDA issues **zero** `NV04_MAP_MEMORY_DMA` (escape 0x2e) calls over a complete
CUDA benchmark run (verified from strace: 0/120 allocs use this escape). Yan et al. 2026
Finding 1 explains: "UVM address unification allows the driver to emit CPU virtual
addresses directly into pushbuffer commands." There is no separate GPU→CPU VA
translation. The UVM kernel module maintains identity mapping between CPU and GPU VAs.

Attempting to use `MAP_MEMORY_DMA` with `{hDma=FERMI_VASPACE_A, hMemory=vidmem}`
fails with `NV_ERR_INVALID_OBJECT_HANDLE` (0x33) from `serverInterMap_Prologue`
— open-RM has no inter-map policy wired between those two classes in resserv.

### 4.2 UVM ioctl numbering

UVM's `uvm_ioctl()` in `kernel-open/nvidia-uvm/uvm.c` uses **raw integer cmd values**
(NOT the `_IOR/_IOW` encoding used by nvidia.ko ioctls). So:

```c
ioctl(uvm_fd, UVM_INITIALIZE,               &params); // cmd = 0x30000001
ioctl(uvm_fd, UVM_REGISTER_GPU,             &params); // cmd = 37
ioctl(uvm_fd, UVM_REGISTER_GPU_VASPACE,     &params); // cmd = 25
ioctl(uvm_fd, UVM_CREATE_EXTERNAL_RANGE,    &params); // cmd = 73
ioctl(uvm_fd, UVM_MAP_EXTERNAL_ALLOCATION,  &params); // cmd = 33
ioctl(uvm_fd, UVM_REGISTER_CHANNEL,         &params); // cmd = 27
```

All return 0 on syscall success; real status is always in `params.rmStatus`.

### 4.3 UVM initialization sequence

```
uvm_fd    = open("/dev/nvidia-uvm")    // primary fd
uvm_fd_mm = open("/dev/nvidia-uvm")    // SECONDARY fd — holds process mm_struct
```

**UVM_INITIALIZE** (on `uvm_fd`): initializes the UVM VA space. Must be first.

**UVM_MM_INITIALIZE** (on `uvm_fd_mm`, with `uvmFd = uvm_fd`): binds the process
mm_struct to the UVM context. Without this, `uvm_va_space_mm_or_current_retain()`
returns NULL and `UVM_REGISTER_GPU_VASPACE` fails with
`NV_ERR_PAGE_TABLE_NOT_AVAIL` (0x5d) because `uvm_va_space_register_gpu_va_space()`
(in `uvm_va_space.c`) cannot obtain the mm.

**Critical**: `uvm_fd_mm` must stay open for the lifetime of the program. The kernel
reference on the mm is held through this fd; closing it triggers `uvm_mm_release()`.
On platforms that don't need this secondary fd, the ioctl returns
`NV_WARN_NOTHING_TO_DO` (0x00010006) and the fd can be closed.

### 4.4 GPU + VA-space registration

**UVM_REGISTER_GPU** (cmd=37): establishes which RM client + GPU we represent.

```c
UVM_REGISTER_GPU_PARAMS p = {
    .gpu_uuid    = { physical UUID from NV2080_CTRL_CMD_GPU_GET_GID_INFO },
    .rmCtrlFd    = dev_fd,    // /dev/nvidia0 fd
    .hClient     = h_client,
    .hSmcPartRef = 0,         // non-MIG
};
// On return, p.gpu_uuid is the "instance UUID" (same as phys on non-MIG)
```

**Get GPU UUID** via subdevice control:
```c
NV2080_CTRL_GPU_GET_GID_INFO_PARAMS p = {
    .flags = NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY  // = 2
};
rm_control(ctl_fd, h_client, h_subdevice, NV2080_CTRL_CMD_GPU_GET_GID_INFO, &p, sizeof(p));
// p.data[0..15] = 16-byte UUID
```

**UVM_REGISTER_GPU_VASPACE** (cmd=25): ties our `FERMI_VASPACE_A` into UVM's
page-table management.

```c
UVM_REGISTER_GPU_VASPACE_PARAMS p = {
    .gpuUuid  = instance_uuid,
    .rmCtrlFd = dev_fd,
    .hClient  = h_client,
    .hVaSpace = h_vaspace,
};
```

Internally, this calls `nvUvmInterfaceDupAddressSpace()` which issues a `DUP_OBJECT`
GSP RPC and calls `uvm_gpu_va_space_set_page_dir()` to set up Hopper's page table
structures.

### 4.5 Per-buffer mapping

Two helpers cover the two cases: buffers with a CPU alias (which must
satisfy Paper F1, see §3.7) and buffers without.

**With CPU alias — `uvm_map_buffer_at(cpu_va, ...)`** — called for
pushbuffer, h_buf, gpu_ctl (GPFIFO + USERD's BAR1 alias), the sema (a
plain sysmem allocation with a direct CPU pointer, no BAR1 alias), and
staging.  The caller has already placed a `MAP_FIXED`
file-backed mapping at `cpu_va` inside the VA pool (via
`rm_alloc_sysmem_at` or `rm_map_memory_at`), so the UVM range is
anchored to the same VA:

```c
UVM_CREATE_EXTERNAL_RANGE_PARAMS cr = { .base = cpu_va, .length = size };
ioctl(uvm_fd, UVM_CREATE_EXTERNAL_RANGE, &cr);

UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *mp = calloc(1, sizeof(*mp));
mp->base   = cpu_va;
mp->length = size;
mp->hMemory = h_mem;
mp->rmCtrlFd = dev_fd; mp->hClient = h_client;
mp->gpuAttributesCount = 1;
mp->perGpuAttributes[0] = { .gpuUuid = instance_uuid,
                            .gpuMappingType = UvmGpuMappingTypeReadWriteAtomic,
                            .gpuCachingType = UvmGpuCachingTypeDefault,
                            .gpuFormatType = UvmGpuFormatTypeDefault,
                            .gpuElementBits = UvmGpuFormatElementBitsDefault,
                            .gpuCompressionType = UvmGpuCompressionTypeDefault };
ioctl(uvm_fd, UVM_MAP_EXTERNAL_ALLOCATION, mp);
// gpu_va == cpu_va — Paper F1 invariant holds.
```

**Without CPU alias — `uvm_map_buffer()`** — called only for `d_buf`.
Reserves a fresh anonymous `PROT_NONE` window at an ASLR-picked VA and
uses that as the UVM GPU VA.  Safe because nothing outside the channel
dereferences `d_buf`'s VA; the CPU never touches d_buf.

| Buffer | Type | CPU alias in pool? | GPU VA == CPU VA? |
|---|---|---|---|
| GPFIFO ring (in `gpu_ctl`) | NV01_MEMORY_LOCAL_USER (HBM) | ✅ BAR1 via `rm_map_memory_at` | ✅ |
| Pushbuffer | NV01_MEMORY_SYSTEM | ✅ sysmem via `rm_alloc_sysmem_at` | ✅ |
| d_buf (src) | NV01_MEMORY_LOCAL_USER (HBM) | ❌ no alias (CE-staged) | N/A |
| h_buf (dst) | NV01_MEMORY_SYSTEM | ✅ sysmem via `rm_alloc_sysmem_at` | ✅ |
| Semaphore | NV01_MEMORY_SYSTEM | ✅ sysmem via `rm_alloc_sysmem_at` (plain CPU pointer, no BAR1) | ✅ |
| Staging | NV01_MEMORY_SYSTEM | ✅ sysmem via `rm_alloc_sysmem_at` | ✅ |

Only `gpu_ctl` must be mapped before channel allocation because
`gpFifoOffset` in `NV_CHANNEL_ALLOC_PARAMS` must be a valid GPU VA.
The other buffers are UVM-mapped in a second pass after channel
construct, matching libcuda's channel-before-buffer-pool ordering.

---

## 5. Channel Setup and Scheduling

### 5.1 `NV_CHANNEL_ALLOC_PARAMS` — correct fields

```c
NV_CHANNEL_ALLOC_PARAMS chan_params = {
    .gpFifoOffset     = gpfifo_uvm_va,   // UVM-mapped GPU VA of GPFIFO ring
    .gpFifoEntries    = 512,             // must be power-of-two, >= 4
    .engineType       = lce_engine_type,  // first non-GRCE LCE, probed at run time
    .hUserdMemory[0]  = h_userd_mem,    // client-allocated USERD handle
    .userdOffset[0]   = 0x2000,         // page offset within USERD allocation
};
```

**Parent must be the TSG** (`h_tsg`, not `h_device`). TSG-parented channel alloc
requires `hVASpace = 0` — the VA space flows through TSG params. If `hVASpace != 0`,
RM rejects the allocation in `kchannelConstruct_IMPL` (`kernel_channel.c`).

**engineType must be explicit**: if left 0, `kfifoGetDefaultRunlist_HAL` for COPY0
returns runlistId=0 on Hopper (shared GR/CE runlist). Later, UVM's engine-type
lookup via `kchannelGetEngine_GM107` calls `kfifoEngineInfoXlate(RUNLIST→ENG_DESC)`,
which returns the FIRST engine on runlist 0 — which is GR. UVM then classifies
the channel as GR and tries to load GR context buffers, failing with
`NV_ERR_INVALID_OBJECT` (0x31). Explicitly passing `COPY0` avoids this.

### 5.2 `UVM_REGISTER_CHANNEL` — mandatory, not optional

After channel alloc, call `UVM_REGISTER_CHANNEL` (cmd=27) BEFORE scheduling:

```c
UVM_REGISTER_CHANNEL_PARAMS p = {
    .gpuUuid  = instance_uuid,
    .rmCtrlFd = dev_fd,
    .hClient  = h_client,
    .hChannel = h_channel,
    .base     = 0x7f0000000000ULL,  // channel-reserved VA window (any large VA)
    .length   = 0x400000ULL,
};
ioctl(uvm_fd, UVM_REGISTER_CHANNEL, &p);
```

This is **not optional** for externally-owned VA spaces. `kchannelIsSchedulable_HAL`
(the "Bug 1737765" check in `kernel_channel.c`) prevents externally owned channels
from running unless bound.  Without UVM_REGISTER_CHANNEL, `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE`
returns `NV_ERR_INVALID_STATE` (0x40).

UVM_REGISTER_CHANNEL internally calls `nvUvmInterfaceRetainChannel()` which:
- Allocates a `UVM_CHANNEL_RETAINER` object (class 0xC574) to hold a refcount on
  the channel's CHID.
- Retrieves work-submit token, CHRAM register offsets, VF doorbell pointer.
- Calls `bind_channel_resources()` which is what finally satisfies
  `kchannelIsSchedulable_HAL`.

### 5.3 Scheduling the channel

```c
NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS sched = { .bEnable = NV_TRUE };
rm_control(ctl_fd, h_client, h_channel,
           NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,   // = 0xa06f0103
           &sched, sizeof(sched));
```

This calls `kchannelCtrlCmdGpFifoSchedule_IMPL` → GSP RPC (func=76 `GSP_RM_CONTROL`)
which adds the channel to the PBDMA runlist. After this, PBDMA will poll the
channel's GPFIFO when it sees a doorbell ring.

### 5.4 Asking for boost clocks

One control that is easy to miss and costs a third of the achievable
bandwidth if skipped.  `mc_init` issues it once, on the subdevice, right
after the subdevice is allocated:

```c
NV2080_CTRL_PERF_BOOST_PARAMS boost = {
    .flags    = DRF_DEF(2080, _CTRL_PERF_BOOST_FLAGS, _CMD,  _BOOST_TO_MAX)
              | DRF_DEF(2080, _CTRL_PERF_BOOST_FLAGS, _CUDA, _YES),   // = 0x12
    .duration = NV2080_CTRL_PERF_BOOST_DURATION_INFINITE,             // 0xffffffff
};
rm_control(ctl_fd, h_client, h_subdevice, NV2080_CTRL_CMD_PERF_BOOST,
           &boost, sizeof(boost));
```

This mirrors libcuda exactly — it makes the same call once during context
creation, with the same flags, and the RM drops the boost when the client
is freed.  Without it a copy-only workload never drives utilisation high
enough for the auto-boost governor to ramp, so the GPU sits at its idle
SM clock (345 MHz on H100 PCIe).  The Copy Engine is clocked from that
domain, so the effect is a hard throughput ceiling around two thirds of
the link rate, with per-iteration latencies stepping between discrete
clock states rather than varying smoothly.  `rm_perf_boost` treats a
rejection as non-fatal — the library still works, just slower.  Full
diagnosis in `findings.md §15`.

---

## 6. VF Doorbell Mapping

### 6.1 Why USERD GPPut alone is insufficient

Writing `userd->GPPut` advances the producer index that PBDMA periodically reads
from USERD. But PBDMA does not continuously poll it — it needs a "wake" signal.
The wake signal on Hopper is a **BAR0 register write** with the work-submit token.

CUDA (and UVM internally) performs this write to `NV_VIRTUAL_FUNCTION_DOORBELL`
(`0x30090` relative to the VF register block, so BAR0 + `0xBB0090` on bare metal;
reached in practice at `+0x90` inside the USERMODE window — see
`gpfifo_pushbuffer_reference.md §11`) with the channel's work-submit token. Without
this write, PBDMA ignores our GPPut update and GPGet stays at 0.

### 6.2 HOPPER_USERMODE_A mapping

Allocate a `HOPPER_USERMODE_A` (class 0xC661) object under the subdevice:

```c
/* Two instances, as libcuda allocates: a BAR0 one for structural parity
 * (never written after setup) and the BAR1 one that actually carries the
 * doorbell.  §12.6 of findings.md has the 194/194-vs-0/194 evidence. */
NV_HOPPER_USERMODE_A_PARAMS um_bar0 = {
    .bBar1Mapping = NV_FALSE,  // BAR0 (non-privileged VF region)
    .bPriv        = NV_FALSE,  // regular VF page (not PRIV page)
};
NV_HOPPER_USERMODE_A_PARAMS um_bar1 = {
    .bBar1Mapping = NV_TRUE,   // BAR1 — this is the one we ring
    .bPriv        = NV_FALSE,
};
h_usermode_bar0 = rm_alloc(ctl_fd, h_client, h_subdevice, HOPPER_USERMODE_A, &um_bar0);
h_usermode_bar1 = rm_alloc(ctl_fd, h_client, h_subdevice, HOPPER_USERMODE_A, &um_bar1);
usermode_bar1_cpu = rm_map_memory(ctl_fd, "/dev/nvidia0", h_client, h_subdevice,
                                  h_usermode_bar1, 0, 0x10000, 0);
vf_doorbell = (volatile uint32_t *)(usermode_bar1_cpu + 0x90);
```

This maps a 64 KB region of BAR0's VF non-privileged window
(`NV_VIRTUAL_FUNCTION = 0x0003FFFF:0x00030000` in Hopper's `dev_vm.h`).
The doorbell register sits at offset `0x90` within this region.

Both `0x30090` (the register) and `0x30000` (the window base) are *relative to the VF
register block*, on Ampere and Hopper alike, so the doorbell is at `+0x90` inside the
mapping on both. On bare metal the block is itself based at `0xB80000` in BAR0, making
the absolute address BAR0 + `0xBB0090`; `0x30090` on its own is the SR-IOV guest's view.
Nothing depends on the absolute number, because the window base and the register shift
together. See `gpfifo_pushbuffer_reference.md §11`.

---

## 7. Work Submission (the hot path)

After all setup, each D2H transfer is submitted like this. **No syscalls**:

```c
// 1. Write method stream into pushbuffer (host RAM, CPU-local write)
uint32_t *pb_end = mc_write_transfer_methods(pb_cpu, d_gpu_va, h_gpu_va, nbytes,
                                      sema_gpu_va, 1);
uint32_t copy_dwords = pb_end - pb_cpu;
_mm_sfence();

// 2. Write GP entry into GPFIFO ring
//    (HBM via BAR1 mapping; write is a PCIe posted TLP)
write_gp_entry(gpfifo_ring, gp_put, GPFIFO_ENTRIES, pb_gpu_va, 0, copy_dwords * 4);
gp_put += entries_used;  // may be 1 or 2 (ext-base + normal)
_mm_sfence();

// 3. Write GPPut to USERD + write token to VF doorbell
userd->GPPut = gp_put;
_mm_sfence();
*vf_doorbell = token;    // NV_VIRTUAL_FUNCTION_DOORBELL via HOPPER_USERMODE_A
_mm_sfence();

// 4. Poll semaphore (CE writes it on completion)
while (*sema_ptr != 1) { /* spin */ }
```

The sfence calls are critical on x86: writes to write-combine (WC) memory regions
(GPFIFO, USERD, BAR0) are buffered by the CPU write-combine hardware. sfence
flushes the WC buffers to the PCIe bus before we proceed.

---

## 8. GP Entry Extended-Base Encoding

Hopper GP entries encode the pushbuffer VA across two 32-bit words — ENTRY0 holds
`pb_va[31:2]` in place plus the FETCH opcode, ENTRY1 holds `pb_va[39:32]`, the
length in dwords, and the SYNC bit. The bit-exact field breakdown, with the
`NVC86F_*` symbol names, is in `gpfifo_pushbuffer_reference.md §4`.

**Important (common bug):** ENTRY0[31:2] stores va[31:2] at bit positions [31:2],
NOT shifted right by 2. So:

```c
// WRONG (shifts va[31:2] down to bit positions [29:0]):
entry0 = (uint32_t)(pb_va & 0xFFFFFFFF) >> 2;

// CORRECT (stores va[31:2] at bit positions [31:2]):
entry0 = (uint32_t)(pb_va & 0xFFFFFFFC);
```

This error causes PBDMA to read the pushbuffer at a 4-byte-shifted address,
producing an MMU fault.

### Extended-base for VAs > 40 bits

UVM typically places buffers above 40-bit VA range (e.g., `0x76baa0e2c000` = 43 bits).
Since ENTRY1[7:0] can only encode pb_va[39:32] (8 bits for max 40-bit addressing),
high VAs need a preceding `SET_PB_SEGMENT_EXTENDED_BASE` GP entry:

```c
if (ext_base != 0) {
    // ext_base = pb_va[56:40] (17 bits, placed at ENTRY0[24:8])
    gpfifo_ring[idx * 2 + 0] = (ext_base & 0x1ffff) << 8;
    gpfifo_ring[idx * 2 + 1] = 0x4;   // OPCODE_SET_PB_SEGMENT_EXTENDED_BASE
}
// then write normal GP entry
```

The `0x4` opcode at ENTRY1[7:0] triggers the `SET_PB_SEGMENT_EXTENDED_BASE` command.
The 17-bit operand at ENTRY0[24:8] provides pb_va[56:40]. PBDMA then uses
`(ext_base << 40) | (entry1[7:0] << 32) | (entry0 & 0xFFFFFFFC)` as the full VA.

---

## 9. The NVC8B5 Method Stream

### Complete stream for a D2H transfer (18 dwords = 72 bytes including SET_OBJECT)

A D2H copy is 18 dwords in seven method groups.  The copy and its completion signal
are two separate `LAUNCH_DMA`s, which is what H100 libcuda emits:

1. `SET_OBJECT` — bind HOPPER_DMA_COPY_A (0xC8B5) to subchannel 4 (CE methods
   require subch 4)
2. `OFFSET_IN_UPPER/LOWER` — source GPU VA (device HBM)
3. `OFFSET_OUT_UPPER/LOWER` — destination GPU VA (host DRAM)
4. `LINE_LENGTH_IN` — byte count
5. `LAUNCH_DMA` — the copy, value `0x00000182` (NON_PIPELINED | SRC/DST PITCH;
   SRC/DST TYPE = VIRTUAL).  No flush and no semaphore: `SEMAPHORE_TYPE` is field
   4:3 and 0 means none, so it is omitted
6. `SET_SEMAPHORE_A/B/PAYLOAD` — release-semaphore VA and payload.  The cell lives
   in sysmem, where libcuda also puts it
7. `LAUNCH_DMA` — the release, value `0x0000000c` (DATA_TRANSFER_TYPE = NONE |
   FLUSH_ENABLE | RELEASE_ONE_WORD).  Moves no data; only flushes and signals

A single fused launch (`0x0000018e`) also works and is what Yan et al. 2026
Listing 1 reports from an **A40** trace; H100 libcuda does not use it.  `mc`'s
SM-authored kernel keeps the fused form because it fits in 16 dwords.

For the bit-exact dword layout and per-method addresses, see
`gpfifo_pushbuffer_reference.md §6` and `§9`.  For the actual construction mc
emits — DRF macros throughout, per the project convention (never hand-rolled
shifts) — see `findings.md §2` and `reverse/mc/mc_submit.c`.

All addresses passed to the CE (OFFSET_IN/OUT, SET_SEMAPHORE) are **GPU VAs** as
established by UVM — which on Hopper happen to be identical to the CPU VAs.

---

## 10. Semaphore Completion and Bandwidth

The CE writes `sema_payload` to `sema_va` as a 32-bit word when all data has been
committed to the destination. We poll:

```c
volatile uint32_t *sema_ptr = sema_cpu;   // plain sysmem pointer
while (*sema_ptr != 1) { /* tight spin */ }
```

**The semaphore is in host DRAM** (`NV01_MEMORY_SYSTEM`, cached), because
that is where libcuda puts it — libcuda's `SET_SEMAPHORE_VA` resolves to
a sysmem allocation.  The CPU therefore polls a normal cached pointer,
MAP_FIXED into the VA pool like every other CPU-aliased buffer (§3.7);
there is no BAR1 alias in this path.  Polling an HBM cell instead would
turn every spin into a PCIe read competing with the transfer it is
waiting on.  Ordering still holds: the release launch carries
`FLUSH_ENABLE` with `FLUSH_TYPE=SYS`, and for a sysmem destination PCIe
posted-write ordering from a single Requester ID keeps the release
behind the data.

**Measured result on H100 PCIe** (Gen5 x16, clocks unlocked with the
library requesting boost at init; full table and method in
`findings.md §11`):

| Transfer | mc D2H | CUDA D2H | mc H2D |
|---|---:|---:|---:|
| 4 MiB   | 31.3 | 35.7 | 42.8 |
| 256 MiB | 55.1 | 52.9 | 55.1 |
| 1 GiB   | 55.5 | 55.4 | 55.5 |

Fixed per-submission overhead (~15 µs) is about 11 % of the ~134 µs a
4 MiB copy takes — visible, but no longer dominant; larger transfers
amortize it out and reach the plateau.

PCIe Gen5 x16 carries ~63 GB/s per direction after encoding — and it is
full-duplex, so a single one-way copy is bounded by that, not by the
~126 GB/s aggregate.  Both directions reach ~55.5 GB/s at 1 GiB, about
88 % of that per-direction ceiling, and they converge: a measurement
showing D2H far below H2D on this hardware is measuring idle clocks
rather than the direction, which is what the boost-clock request
described in `findings.md §15` removes.  The absolute plateau is
box-specific, so the portable result is the one that does not move: for
device-to-host, mc matches CUDA within run-to-run spread on the same box.
There is no CUDA host-to-device measurement here to compare against.

---

## 11. Kernel Instrumentation Used

The instrumentation API is `MC_TRACE(category, "event", "k=%u …", …)` from
`src/common/sdk/nvidia/inc/mc-trace.h`.  Each call emits one self-describing
record:

```
mc1 <category>/<event> key=value key=value ...
```

`nv_trace_printf` (`kernel-open/nvidia/os-interface.c`) is only the transport
underneath it — a wrapper on `ftrace_vprintk`, which is the va_list-accepting
counterpart of the `trace_printk` macro.  All output goes to
`/sys/kernel/debug/tracing/trace`, not to dmesg.

Every site is gated on `nv_trace_mask & MC_TRACE_CAT_<category>`.  The mask is
the writable `nvidia.ko` module parameter `mc_trace`, defaulting to every
category except `pte`:

```bash
cat /sys/module/nvidia/parameters/mc_trace
echo 0x3ff | sudo tee /sys/module/nvidia/parameters/mc_trace
```

Nineteen files across the RM core, the kernel-open modules and nvidia-modeset
carry call sites.  The ones that matter for the `mc` bring-up path:

| File | Events | Reveals |
|---|---|---|
| `escape.c` | `rm/ioctl`, `rm/alloc`, `rm/control`, `rm/free`, `rm/map_memory`, `body/alloc_*` | handle creates, control commands, and the allocation parameters themselves |
| `vgpu/rpc.c` | `gsp/rpc_tx` | what reaches GSP, and when |
| `kernel_channel.c` | `fifo/chan_construct`, `fifo/userd_rpc` | channel parameter validation, engine/runlist assignment |
| `kernel_channel_gm107.c` | `fifo/chan_get_engine` | runlistId → engine mapping |
| `nv_gpu_ops.c` | `fifo/verify_channel`, `fifo/retain_channel`, `fifo/userd_bind` | the UVM channel-retention path |
| `uvm_va_space.c` | `uvm/create_gpu_va_space`, `uvm/register_gpu_va_space` | UVM VA-space registration |
| `rs_server.c` | `rm/inter_map` | `serverInterMap` resource lookups, MAP_MEMORY_DMA policy checks |
| `nv-doorbell-watch.c` | `dbell/*`, `pb/submit`, `pb/bytes` | the doorbell watchpoint (§12 of `findings.md`) |

The grammar, the category bits and the complete 59-event catalogue are in
`docs/reference/trace-format.md`; how to capture and read a trace is in
`docs/tracing_cuda.md`.

---

## 12. Complete Bug Log

Twelve distinct issues encountered and resolved, in order of discovery:

| # | Bug | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 1 | Missing GPFIFO_SCHEDULE | GPGet=0 forever after doorbell | Channel not on runlist | Add `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE` RM control |
| 2 | MAP_MEMORY_DMA approach | 0x33 INVALID_OBJECT_HANDLE | open-RM has no inter-map policy for FERMI_VASPACE_A + NV01_MEMORY_LOCAL_USER | Drop MAP_MEMORY_DMA; use UVM path |
| 3 | VASPACE missing IS_EXTERNALLY_OWNED | UVM_REGISTER_GPU_VASPACE → 0x29 INVALID_FLAGS | `nvGpuOpsDupAddressSpace` checks `vaspaceIsExternallyOwned()` | Add `NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED` |
| 4 | Missing UVM_MM_INITIALIZE | UVM_REGISTER_GPU_VASPACE → 0x5d PAGE_TABLE_NOT_AVAIL | `uvm_va_space_mm_or_current_retain()` returns NULL without secondary fd | Open second `/dev/nvidia-uvm` fd, call UVM_MM_INITIALIZE |
| 5 | UVM_REGISTER_CHANNEL not called | GPFIFO_SCHEDULE → 0x40 INVALID_STATE | "Bug 1737765" in `kchannelIsSchedulable_HAL`: externally-owned channel must be UVM-registered | Add UVM_REGISTER_CHANNEL before SCHEDULE |
| 6 | Kernel runlist-based engine lookup wrong | `channelEngineType=1(GR)` for CE channel | `kchannelGetEngine_GM107` uses runlistId → first engine, which is GR on shared runlist | Patch `nvGpuOpsGetChannelEngineType` to use `pKernelChannel->engineType` directly |
| 7 | Missing VF doorbell write | GPGet=0 forever even with SCHEDULE | USERD GPPut alone doesn't wake PBDMA scheduler | Add HOPPER_USERMODE_A mapping + write token to +0x90 |
| 8 | No SET_PB_SEGMENT_EXTENDED_BASE | Xid 31 MMU fault at truncated VA | ENTRY1[7:0] only encodes pb_va[39:32]; UVM VAs are 43-bit | Add extended-base GP entry before normal entry |
| 9 | Wrong ext-base shift (34 vs 40) | Xid 31 at wrong address | ext_base was `va >> 34`; should be `va >> 40` | Change shift to 40 |
| 10 | entry0 shift error | Xid 31 MMU fault (4-byte shift) | `entry0 = va >> 2` puts va[31:2] at [29:0]; should be at [31:2] | Change to `entry0 = va & 0xFFFFFFFC` |
| 11 | Subchannel 0, no SET_OBJECT | Xid 32 invalid pushbuffer stream | CE methods must target subchannel 4; no engine bound at subch 0 | Add `INCR_HEADER_SUB(0, 1, 4)` + `HOPPER_DMA_COPY_A` as first method |
| 12 | Paper-F1 violation — split VAs per UVM-mapped buffer | ~25–34 % failure rate at N=100: H2D sem-timeout (Mode A, GPPut=GPGet=2) + verify-FAILED with invariant `0x20018000` or `0xcafebabe` garble (Mode B) | `uvm_map_buffer()` used an ASLR-picked PROT_NONE anonymous VA as UVM GPU VA; `rm_alloc_sysmem`/`rm_map_memory` mmap'd the CPU alias at a different kernel-picked VA.  Two CPU VAs aliased one physical backing; something in the GPU/GSP pipeline mis-behaves on the split-VA bookkeeping. | Reserve a 4 GiB PROT_NONE VA pool at `0x200000000` up front (see §3.7); add `rm_alloc_sysmem_at` / `rm_map_memory_at` / `uvm_map_buffer_at` that MAP_FIXED into the pool and pass the same VA to UVM.  N=1000 after fix: 1000/1000 pass.  Details in `docs/findings.md §13`. |

### Note on the kernel-side doorbell watchpoint (2026-05-12 update)

The kernel-side Hopper doorbell watchpoint added in this fork (see
`docs/findings.md §12`) now fires end-to-end on `mc` traces as
well as libcuda traces.  With the Paper-F1 fix (§3.7, bug #12 above),
the GPFIFO ring's GPU VA equals its CPU VA inside the VA pool, which
is registered with `bar1_track_add` during the initial MAP_FIXED
mmap.  `nv_dbell_bar1_gpu_va_to_kva` resolves it; the #DB handler
reads the GPFIFO entry and emits `mc1 pb/submit` + `mc1 pb/bytes`
for every submission.  A `trace_cuda.sh` capture of
`mc_demo --size 128M` produces matching submit / bytes / 0 miss
per run with the decoded method stream matching libcuda's shape
exactly (OFFSET_IN_VA, OFFSET_OUT_VA, `LINE_LENGTH=0x08000000`,
`SET_SEMAPHORE_PAYLOAD=1`/`2`, and the split `LAUNCH_DMA` pair).  This was the
"coverage gap" noted in the 2026-05-06 revision of this document —
it was closed as a userspace-only side effect of the bug-#12 fix.
No kernel change was needed.

---

## 13. Quick-Reference: ioctl Table

| Operation | Escape / cmd | Struct | Target fd |
|---|---|---|---|
| RM alloc | 0x2b (NV_ESC_RM_ALLOC) | NVOS64_PARAMETERS | ctl_fd |
| RM control | 0x2a (NV_ESC_RM_CONTROL) | NVOS54_PARAMETERS | ctl_fd |
| RM sysmem alloc | 0x27 (NV_ESC_RM_ALLOC_MEMORY) | nvos02_with_fd | dev_fd (nvidia0) |
| RM map to CPU | 0x4e (NV_ESC_RM_MAP_MEMORY) | nvos33_with_fd | ctl_fd |
| RM associate fd | 0xc9 (NV_ESC_REGISTER_FD) | nv_ioctl_register_fd_t | dev_fd |
| UVM init | 0x30000001 (UVM_INITIALIZE) | UVM_INITIALIZE_PARAMS | uvm_fd |
| UVM mm init | 75 (UVM_MM_INITIALIZE) | UVM_MM_INITIALIZE_PARAMS | uvm_fd_mm |
| UVM GPU reg | 37 (UVM_REGISTER_GPU) | UVM_REGISTER_GPU_PARAMS | uvm_fd |
| UVM vaspace reg | 25 (UVM_REGISTER_GPU_VASPACE) | UVM_REGISTER_GPU_VASPACE_PARAMS | uvm_fd |
| UVM buf reserve | 73 (UVM_CREATE_EXTERNAL_RANGE) | UVM_CREATE_EXTERNAL_RANGE_PARAMS | uvm_fd |
| UVM buf map | 33 (UVM_MAP_EXTERNAL_ALLOCATION) | UVM_MAP_EXTERNAL_ALLOCATION_PARAMS | uvm_fd |
| UVM chan reg | 27 (UVM_REGISTER_CHANNEL) | UVM_REGISTER_CHANNEL_PARAMS | uvm_fd |

---

*Document maintained alongside `reverse/mc/`.*
*Driver: 610.43.02 (open-gpu-kernel-modules). Platform: H100 PCIe, kernel 6.8.*
