# The NVIDIA Software Stack: From libcuda to GSP Firmware

**A guide to the layers of software involved in running CUDA on a modern NVIDIA
GPU, and why each layer exists.**

This document assumes you have read or are familiar with the concepts in
`host_gpu_communication_primer.md` — PCIe, BARs, MMIO, DMA, doorbells, command
queues, and the two-MMU model. Here we specialize those concepts to NVIDIA's
actual software layering: who runs where, who talks to whom, and what each
component owns.

By the end of this document you'll understand the trace output from mc
("RM ioctl: cmd=0x2b", "GSP RPC TX sync: func=103 (GSP_RM_ALLOC)", etc.) and
where each operation sits in the overall stack.

---

## Table of Contents

1. [The big picture: five layers](#1-the-big-picture-five-layers)
2. [libcuda.so — the userspace runtime](#2-libcudaso--the-userspace-runtime)
3. [The device files: /dev/nvidia*](#3-the-device-files-devnvidia)
4. [nvidia.ko — the Resource Manager kernel module](#4-nvidiako--the-resource-manager-kernel-module)
5. [nvidia-uvm.ko — Unified Virtual Memory](#5-nvidia-uvmko--unified-virtual-memory)
6. [GSP firmware — the GPU System Processor](#6-gsp-firmware--the-gpu-system-processor)
7. [The GSP-RPC mechanism](#7-the-gsp-rpc-mechanism)
8. [The RM object model](#8-the-rm-object-model)
9. [Why this architecture](#9-why-this-architecture)
10. [What each layer does for a cudaMemcpy](#10-what-each-layer-does-for-a-cudamemcpy)
11. [The open vs closed source split](#11-the-open-vs-closed-source-split)
12. [Other modules: modeset, drm, peermem](#12-other-modules-modeset-drm-peermem)
13. [Debugging across layers](#13-debugging-across-layers)

---

## 1. The big picture: five layers

A running CUDA process on a modern NVIDIA GPU spans five layers of software,
located in three protection domains and two processors:

```
                                                       Runs on:
┌────────────────────────────────────────────────┐
│ CUDA application (userspace, unprivileged)     │   CPU (user ring 3)
│   ./my_cuda_program                            │
│   Calls cudaMemcpy, cudaMalloc, ...            │
└──────────────────┬─────────────────────────────┘
                   │ C function calls
                   ▼
┌────────────────────────────────────────────────┐
│ CUDA Runtime + Driver (libcudart.so,           │   CPU (user ring 3)
│                       libcuda.so)              │
│   Translates CUDA API into device-specific     │
│   commands (pushbuffer, method stream).        │
│   Writes GPFIFO, rings doorbell.               │
└──────────────────┬─────────────────────────────┘
                   │ ioctl(fd, RM_ALLOC, ...) etc.
                   │ direct MMIO writes for doorbells
                   ▼
┌────────────────────────────────────────────────┐
│ nvidia.ko + nvidia-uvm.ko (RM kernel driver)   │   CPU (kernel ring 0)
│   Manages object hierarchy, client sessions,   │
│   memory allocations, page tables.             │
│   Translates RM API into GSP RPCs.             │
└──────────────────┬─────────────────────────────┘
                   │ GSP RPC (shared-memory ring)
                   ▼
┌────────────────────────────────────────────────┐
│ GSP firmware                                   │   GPU (on-die RISC-V)
│   Owns physical GPU state.                     │
│   Programs hardware registers, manages         │
│   runlists, handles faults.                    │
└──────────────────┬─────────────────────────────┘
                   │ direct register writes,
                   │ DMA engine programming
                   ▼
┌────────────────────────────────────────────────┐
│ GPU hardware                                   │   GPU (hardware state machines)
│   PBDMA, CE, SM, MMU, memory controller, ...   │
└────────────────────────────────────────────────┘
```

Two processors (CPU x86 cores; GPU has its own RISC-V for GSP). Three
protection domains (userspace, CPU kernel, GSP).

Each layer has a well-defined interface upward and downward:

| Layer | Upward interface | Downward interface |
|---|---|---|
| Application | CUDA Runtime API | C function calls |
| libcuda | CUDA Driver API | ioctls + MMIO |
| nvidia.ko | Ioctl surface | GSP RPC |
| GSP firmware | RPC message set | Hardware register writes |
| Hardware | Memory-mapped registers | Physics |

Understanding where a problem lives is essential for debugging. A
`cudaGetLastError() == cudaErrorInvalidValue` could originate in any layer —
the error message alone doesn't tell you where.

---

## 2. libcuda.so — the userspace runtime

### What it is

`libcuda.so` (full name `libcuda.so.610.43.02` on our system) is the
**closed-source** NVIDIA userspace driver. It lives in `/usr/lib/x86_64-linux-gnu/`
or similar, is loaded by every CUDA application, and implements:

- **The CUDA Driver API** (`cuCtxCreate`, `cuMemAlloc`, `cuMemcpy*`,
  `cuLaunchKernel`, etc. — the low-level API).
- **The CUDA Runtime API** (`cudaMalloc`, `cudaMemcpy`, etc.) is implemented
  by `libcudart.so`, which in turn calls into `libcuda.so`.

### What it does

libcuda is where most of CUDA's complexity lives:

- **Command stream encoding**: translates high-level calls like
  `cudaMemcpy(dst, src, n, D2H)` into NVC8B5 method streams (as we
  documented in `gpfifo_pushbuffer_reference.md`).
- **Memory management**: maintains allocators on top of RM's raw memory
  objects. `cudaMalloc` goes through a user-space slab allocator before ever
  reaching the kernel.
- **Stream and event ordering**: implements CUDA streams (queues of
  GPU work), events (synchronization primitives), and the associated
  happens-before relationships, all using GPFIFO submission and semaphore
  polling.
- **Kernel launch**: builds the SET_CTX, SET_OBJECT, and SEND_SIGNALING
  sequences that launch a compute kernel on an SM.
- **UVM policy decisions**: which memory should be paged-in to GPU vs.
  stay on host, prefetch heuristics, etc.

### Why it's large

libcuda is large — about 110 MB on the 610.43.02 driver this tree pins.
(The ~30 MB figure sometimes quoted is the 32-bit compatibility copy.)
The size reflects:
- Per-architecture optimized code paths (Turing, Ampere, Hopper, Blackwell).
- Compute-kernel support (PTX JIT, cubin loading, launch metadata).
- Graph APIs (CUDA Graphs).
- Profiling hooks (NVTX).
- Interop APIs (OpenGL, DirectX, Vulkan).
- Backwards compat for many CUDA versions.

### The driver hot path for mc

mc deliberately **bypasses libcuda entirely**. We do the work libcuda
would do — build the pushbuffer, write GPFIFO, ring the doorbell — directly
from our own C code. This is only practical for a minimal demo; for real
applications, the complexity of libcuda is well worth having.

### libcuda ↔ kernel interface

libcuda interacts with the kernel through three mechanisms:

1. **Ioctls on /dev/nvidia* files** (see §3). Most interaction with the kernel
   is through these, using the RM escape codes defined in `nv-ioctl-numbers.h`.
2. **Direct MMIO writes** (to BARs that are mmap'd by the kernel driver into
   libcuda's userspace). The doorbell, GPFIFO ring (via BAR1), USERD, and
   HOPPER_USERMODE_A mappings are all userspace MMIO.
3. **Shared memory** (USERD for GPPut/GPGet exchange).

That libcuda can access BAR-mapped device registers directly from userspace
(without an ioctl per access) is **the** key performance optimization —
without it, each command submission would pay syscall overhead.

---

## 3. The device files: /dev/nvidia*

The kernel driver exposes several device files, each with a distinct
purpose:

### `/dev/nvidiactl` — the RM control interface

Opened once per process. This is the fd through which most RM ioctls are
issued (RM_ALLOC, RM_CONTROL, RM_FREE, MAP_MEMORY). It provides a
**per-process client context** — each opener gets a separate session.

### `/dev/nvidia0`, `/dev/nvidia1`, ... — per-GPU handles

One per physical GPU (or MIG partition). Used for:
- `NV_ESC_REGISTER_FD` to associate the process's RM context with a
  specific GPU.
- Device-specific ioctls like `NV_ESC_RM_ALLOC_MEMORY` (which allocates
  OS-managed memory and needs device context).
- Memory mappings: each mmap of GPU memory consumes a fresh `/dev/nvidia0`
  fd (the kernel uses the fd identity to track mappings; see
  `rm_create_mmap_context`, declared in
  `src/nvidia/arch/nvalloc/unix/include/osapi.h`; `nv-mmap.c` only consumes
  the `nv_alloc_mapping_context_t` it produces).

### `/dev/nvidia-uvm` — the UVM interface

Used for UVM ioctls: `UVM_INITIALIZE`, `UVM_REGISTER_GPU`,
`UVM_MAP_EXTERNAL_ALLOCATION`, etc. Separate device because UVM is a
separate kernel module (`nvidia-uvm.ko`).

### `/dev/nvidia-modeset`

For display driver interactions. Not relevant for compute-only workloads.

### `/dev/nvidia-drm`

For DRM (Direct Rendering Manager) integration — graphics APIs like OpenGL,
Vulkan. Not relevant for compute.

### `/dev/nvidia-uvm-tools`

Used for UVM debug/tracing. Read-only.

### Why so many files?

Each device file corresponds to a distinct kernel module or driver subsystem.
The kernel uses `open()`'s result (a file descriptor) as a per-session
handle — it can attach per-process state to each open file. This is a clean
way to handle multi-GPU systems and heterogeneous capabilities.

For a basic compute app, you need: `/dev/nvidiactl`, `/dev/nvidia0`, and
`/dev/nvidia-uvm`. That's what mc opens.

---

## 4. nvidia.ko — the Resource Manager kernel module

### What it is

`nvidia.ko` is the core kernel driver — the "RM" (Resource Manager). It's a
**partially open-source** module: most of the core object-management logic is
open (in the open-gpu-kernel-modules source tree under `src/nvidia/`), and a
small hardware abstraction layer is binary (pre-compiled chip-specific
HALs shipped as `.bin` files, linked at module-build time).

RM is ~10 MB of code. It's large because it implements:

- **The ioctl dispatch** (`escape.c`) — entry point from userspace.
- **The resource-server framework** (resserv, `rs_server.c`) — a generic
  object-management library with classes, parents, ref-counting, locking.
- **Driver-specific object constructors** — one file per class
  (channel, memory, vaspace, TSG, etc.) implementing per-class allocation
  and lifecycle logic.
- **Platform layer** (`kernel-open/nvidia/*`) — Linux-specific glue
  (file ops, mmap, interrupts, DMA APIs).
- **GSP-RPC plumbing** (`vgpu/rpc.c`) — the mechanism to send requests
  to GSP firmware and receive responses.

### Entry points: the escape codes

All ioctls on `/dev/nvidiactl` and `/dev/nvidia0` land in
`src/nvidia/arch/nvalloc/unix/src/escape.c`, which dispatches on the ioctl
command number (the "escape code").  Key escapes, defined in
`src/nvidia/arch/nvalloc/unix/include/nv_escape.h` (except
`NV_ESC_REGISTER_FD`, which is in `nv-ioctl-numbers.h`):

| Escape | Value | Purpose |
|---|---|---|
| `NV_ESC_RM_ALLOC` | 0x2b | Allocate an RM object (channel, memory, etc.) |
| `NV_ESC_RM_CONTROL` | 0x2a | Execute a control call on an object |
| `NV_ESC_RM_FREE` | 0x29 | Free an RM object |
| `NV_ESC_RM_ALLOC_MEMORY` | 0x27 | Allocate OS-managed (sysmem) memory |
| `NV_ESC_RM_MAP_MEMORY` | 0x4e | Map memory into CPU address space |
| `NV_ESC_RM_MAP_MEMORY_DMA` | 0x57 | Map memory into a GPU VA space (legacy) |
| `NV_ESC_REGISTER_FD` | 0xc9 | Associate nvidia0 fd with ctl_fd |

### What RM actually does

RM's job is to **manage the graph of allocated objects** and **forward
hardware-touching operations to GSP**. It doesn't program hardware directly
on a GSP-enabled GPU (which is all modern GPUs from Turing+).

For example, allocating a channel:

1. Userspace calls `ioctl(ctl_fd, NV_ESC_RM_ALLOC, {class=0xC86F, ...})`.
2. RM's escape handler creates an object in the resserv tree (a graph of
   `RsResourceRef` nodes).
3. RM's channel constructor (`kchannelConstruct_IMPL`) allocates the
   RAMIN/RAMFC/USERD backing memory via the memory manager.
4. RM serializes an `rpc_alloc_channel_dma_v1F_04` message and sends it to
   GSP via the shared-memory RPC ring.
5. GSP firmware does the actual hardware programming (allocating a hardware
   channel slot, setting up the runlist entry, configuring the PBDMA).
6. GSP responds with success/failure; RM records it in the object and returns
   to userspace.

Control calls are similar: userspace requests a specific operation
(`NVA06F_CTRL_CMD_GPFIFO_SCHEDULE`), RM validates the client and object,
forwards to GSP, GSP programs the runlist, returns status.

### The resource-server (resserv) framework

Underneath the driver-specific logic is a generic object-management library.
This provides:

- **Handles**: every object has a 32-bit opaque handle.
- **Parent/child relationships**: objects form a tree; freeing a parent
  automatically frees children.
- **Reference counting**: objects are retained/released across boundaries
  (including across processes for shared memory).
- **Locking**: API-level and GPU-level locks protect concurrent access.
- **Client sessions**: each opener of `/dev/nvidiactl` gets an `RsClient`
  that scopes its objects.

The resserv framework is shared with OpenCL, display, and vGPU — it's a
reusable core. The object classes that use it are driver-specific.

### Memory management inside RM

RM has its own memory managers:

- **Page heap (pmm)**: manages HBM (video memory) in large blocks.
- **Context-buffer pool**: per-channel backing store (RAMIN, RAMFC).
- **MMU module**: GPU page tables.
- **OS-descriptor layer**: registers CPU-pinned host pages with the GPU MMU.

These are all in `src/nvidia/src/kernel/gpu/mem_mgr/`.

---

## 5. nvidia-uvm.ko — Unified Virtual Memory

### What it is

`nvidia-uvm.ko` is a **separate kernel module** that implements the UVM
subsystem. It depends on `nvidia.ko` (it calls into RM via an exported
interface) but lives in its own module for modularity.

Source: `kernel-open/nvidia-uvm/` — mostly open-source (unlike the partially-
binary RM), though it shares the `RmApi` header contract with RM.

### What it does

UVM manages the **process's unified virtual address space**. This means:

- **GPU VA = CPU VA**: UVM installs GPU MMU page tables at the same VAs the
  process uses in its CPU VA space.
- **On-demand paging**: UVM can migrate pages between CPU and GPU memory
  transparently (for managed memory). Not used by mc, but this is
  the feature that makes `cudaMallocManaged` work.
- **External-range registration**: UVM can also expose pre-allocated RM
  memory objects at specific VAs without migration (what mc uses via
  `UVM_MAP_EXTERNAL_ALLOCATION`).

### Two interfaces: external and managed

**External** (what mc uses): the memory is owned by RM; UVM just maps
it into the GPU MMU at a CPU-visible VA. No page migration. This is used for
buffers whose lifetime is explicitly managed (channel structures, method
buffers, UVM-registered CUDA allocations).

**Managed** (what `cudaMallocManaged` uses): the memory is owned by UVM; UVM
decides page-by-page whether it lives in CPU or GPU memory, migrating on
access. Requires page-fault handling; complex.

### UVM's own objects

UVM introduces its own object types inside RM's namespace:
- `UVM_CHANNEL_RETAINER` (0xC574) — an RM object UVM allocates to pin a
  channel so the channel can't go away while UVM tracks it.
- Various VA-space management objects internal to UVM.

### Why UVM needs to register the channel

See `mc_architecture.md §5.2` on `UVM_REGISTER_CHANNEL`: this is how UVM
learns about client-allocated channels. UVM needs to:
- Track the channel's instance memory for page-fault handling.
- Prevent the channel from being scheduled until UVM has set up the necessary
  page tables.

Without UVM_REGISTER_CHANNEL, `kchannelIsSchedulable_HAL()` returns false,
and `GPFIFO_SCHEDULE` fails with INVALID_STATE (this is "Bug 1737765" in the
comments).

### UVM's ioctl surface

UVM's ioctls go through `/dev/nvidia-uvm`, not `/dev/nvidiactl`. They're
dispatched by `uvm_ioctl()` in `kernel-open/nvidia-uvm/uvm.c`. Unlike RM's
ioctls, UVM's use **raw integer command numbers** (not `_IOR/_IOW`-encoded).

Key UVM ioctls (we saw these in mc):

| Cmd | Name | Purpose |
|---|---|---|
| 0x30000001 | `UVM_INITIALIZE` | Initial handshake per fd |
| 37 | `UVM_REGISTER_GPU` | Associate a client's GPU with UVM |
| 25 | `UVM_REGISTER_GPU_VASPACE` | Tie an RM VA space into UVM |
| 73 | `UVM_CREATE_EXTERNAL_RANGE` | Reserve a UVM-owned CPU VA range |
| 33 | `UVM_MAP_EXTERNAL_ALLOCATION` | Map an RM memory object at that VA |
| 27 | `UVM_REGISTER_CHANNEL` | Register an RM channel with UVM |
| 75 | `UVM_MM_INITIALIZE` | Bind process mm_struct (secondary fd) |

### Why UVM is a separate module

- **Licensing / distribution**: UVM is fully open; some RM core is not.
- **Upgrades**: UVM can be updated independently.
- **Optionality**: though in practice UVM is always loaded.
- **Subsystem isolation**: UVM has very different locking/memory-management
  semantics from RM core; separating them keeps code clearer.

---

## 6. GSP firmware — the GPU System Processor

### What it is

Starting with Turing (2018) and mandated from Hopper (2022), NVIDIA GPUs
include a dedicated on-die RISC-V processor called the **GSP** (GPU System
Processor). GSP runs NVIDIA-signed firmware that owns most of the
hardware-management logic previously done by the host kernel driver.

The firmware binary is large — about 84 MB for `gsp_ga10x.bin`, which is what
an H100 loads — and is shipped by NVIDIA as
`/lib/firmware/nvidia/<version>/gsp_<arch>.bin`. The kernel driver loads it
into GSP's private memory at GPU initialization.

### Why GSP exists

The move from "host kernel driver owns hardware" to "GSP firmware owns
hardware" was motivated by:

- **Security**: GPUs are large, complex state machines. Reducing the attack
  surface of the host kernel driver (by removing hardware-touching code)
  lowers the risk of kernel-level exploits.
- **Virtualization**: running the hardware-management code on the GPU itself
  makes vGPU / SR-IOV much cleaner — each VM gets a thin RM in its kernel
  that RPCs to a shared GSP.
- **Confidential Computing**: Hopper's CC mode requires that the host kernel
  not have direct hardware access (the GPU's own firmware enforces the
  trust boundary).
- **Driver uniformity**: GSP firmware is the same across Linux, Windows,
  FreeBSD; only the thin RM on each host OS needs porting.

### What GSP owns

- **Physical memory management**: HBM allocation, fragmentation.
- **MMU and page tables**: installs PDEs/PTEs, handles MMU faults.
- **Channel dispatch**: manages runlists, handles preemption, context
  switching.
- **Power/clock management**: DVFS, thermal throttling, idle-state
  transitions.
- **Fault handling**: when the MMU or an engine faults, GSP is the first
  responder; it may retry, or report to host RM.
- **Firmware loading for other chip processors**: PMU, FECS, GPCCS (the
  graphics front-end microcontrollers) all load firmware that GSP manages.

### What GSP does NOT own

- **Userspace interaction**: RM on the host still handles ioctls, client
  sessions, and object-handle-to-hardware-state mapping.
- **Performance-critical paths**: the CPU-submitted doorbell goes directly
  to the host scheduler hardware (bypassing GSP) for latency reasons.
- **UVM's page-fault handling policy**: UVM on the host decides where to
  page memory; GSP just executes the mapping updates.

### How GSP is structured internally

GSP firmware is a reasonably full real-time OS. It has:
- A scheduler.
- Interrupt handlers.
- An RPC receive loop (consuming messages from the host).
- Per-chip modules mirroring the host RM's structure.

Source for GSP firmware is **not public**. The host side of the GSP
interface (the RPC message formats, enumerations, struct layouts) *is*
public — in `src/nvidia/inc/kernel/vgpu/` and the generated
`g_rpc-structures.h`, `rpc_global_enums.h`.

---

## 7. The GSP-RPC mechanism

### The communication channel

The host RM and GSP firmware communicate via **RPC messages** exchanged
through a shared memory ring buffer. The ring lives in a region of HBM
that's accessible to both:

- Host RM issues a write to the ring (a PCIe write via BAR1).
- GSP receives an interrupt or polls the ring for new messages.
- GSP processes the message and writes a response back to the ring.
- Host RM reads the response.

This is the mechanism behind the `mc1 gsp/rpc_tx mode=sync func=103
name=GSP_RM_ALLOC len=64` records in our ftrace output.  Only the transmit
side is instrumented — there is no matching receive event.

### Message format

Each RPC message has:
- A function ID (an enum like `NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC` = 103).
- A parameter blob (struct-specific to the function).
- Metadata (transaction ID, direction, etc.).

The function IDs and struct definitions are auto-generated from a FINN
(NVIDIA's IDL) specification. The open source tree ships the generated
C headers under `src/nvidia/generated/`.

### Synchronous vs asynchronous RPCs

- **Synchronous**: host RM sends a message and waits for response. Most RPCs
  are sync.
- **Asynchronous**: host fires a message and doesn't wait. Used for
  notifications and background operations.

### RPC function examples seen in mc traces

| Func ID | Name | Meaning |
|---|---|---|
| 10 | FREE | Free an object |
| 21 | DUP_OBJECT | Duplicate a handle into another client/session |
| 76 | GSP_RM_CONTROL | Generic control call forwarded to GSP |
| 103 | GSP_RM_ALLOC | Generic object allocation forwarded to GSP |

These appear in the trace on every object allocation and many control calls.

### The nvidia-push.c anomaly

Not every "thing that looks like driver work" goes through GSP. mc
(and CUDA) submit GPFIFO work **entirely from userspace without any RPC**:
the pushbuffer is a shared structure, GPFIFO is a shared ring, USERD is
shared memory, and the VF doorbell is direct MMIO. Setting up the channel
requires many GSP RPCs; using it requires zero.

This is a deliberate design: RPCs have high latency (ms-scale overhead per
call due to ring-buffer polling and IPI wake-ups). Work submission has to
be µs-scale. So GSP is involved only in setup/teardown, not per-submission.

---

## 8. The RM object model

RM organizes everything into a tree of reference-counted **objects**, each
with a stable 32-bit handle. Understanding this model is essential for
reading the ioctl traces and allocation patterns.

### Objects, classes, and handles

- **Class**: a type identifier (a 32-bit constant like `NV01_ROOT = 0x41`,
  `NV01_DEVICE_0 = 0x80`, `HOPPER_CHANNEL_GPFIFO_A = 0xC86F`). Defines the
  object's behavior.
- **Handle**: a 32-bit integer chosen by the client at allocation time
  (mc uses `g_next_handle++` starting at `0xf0000000`).
- **Parent**: another handle (every object except ROOT has a parent).

### The hierarchy for a compute app

```
NV01_ROOT (0x41)                         client session, parent = itself
├── NV01_DEVICE_0 (0x80)                 device instance
│   ├── NV20_SUBDEVICE_0 (0x2080)        subdevice (GPU die)
│   │   └── HOPPER_USERMODE_A (0xC661)   VF doorbell window (BAR1 variant)
│   ├── NV01_MEMORY_LOCAL_USER (0x40)    HBM allocation (one per alloc)
│   ├── NV01_MEMORY_SYSTEM (0x3E)        host DRAM allocation
│   ├── FERMI_VASPACE_A (0x90F1)         GPU VA space
│   └── KEPLER_CHANNEL_GROUP_A (0xA06C)  TSG (channel group)
│       └── HOPPER_CHANNEL_GPFIFO_A (0xC86F)   channel
│           └── HOPPER_DMA_COPY_A (0xC8B5)     CE engine object
```

Each object's parent restricts where it can be allocated. You cannot allocate
a channel directly under a device — it must go under a TSG. You cannot
allocate a CE engine object under a subdevice — it must go under a channel.
These parent/child rules are defined in
`src/nvidia/src/kernel/rmapi/resource_list.h`.

### Why handles are 32-bit, not pointers

Handles are opaque integers because:
- They're stable across process boundaries (can be marshalled over RPC, UDS).
- They're safe to expose to userspace (no pointer leakage).
- They can be used across PID namespaces.
- They provide natural indirection for lifetime management.

### Control calls vs allocations

- **`RM_ALLOC`**: creates a new object with a specific class.
- **`RM_FREE`**: destroys an object (and recursively its children).
- **`RM_CONTROL`**: invokes a specific named operation on an existing
  object, identified by a command ID like `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE =
  0xa06f0103`.

The control command's high byte identifies the target class's interface
(here: `0xA06F` = KEPLER_CHANNEL_GPFIFO_A interface), and the low 16 bits
are the specific operation. This is how a single RM_CONTROL ioctl dispatches
to thousands of different operations.

---

## 9. Why this architecture

The 5-layer structure wasn't designed all at once — it evolved. Knowing the
history helps make sense of oddities.

### Generation 1: "fat kernel driver" (pre-Turing)

In early CUDA (~2007) through Pascal (2016), the entire architecture was:

```
App → libcuda → /dev/nvidia → nvidia.ko → Hardware
```

The kernel driver did everything: object management, hardware programming,
DMA setup, interrupt handling. It was ~5 MB of mostly-closed-source code.

Problems:
- The kernel driver was huge and performance-critical; bugs were serious.
- Ports to new hardware required large changes in kernel code.
- Virtualization was bolted on awkwardly (vGPU).

### Generation 2: "GSP offload" (Turing onwards)

Starting with Turing (~2018, partially) and mandatory from Hopper (~2022),
NVIDIA introduced GSP — an on-die RISC-V core — and moved most hardware-
management code to GSP firmware:

```
App → libcuda → /dev/nvidia → nvidia.ko (thin) → GSP firmware → Hardware
```

The host kernel driver became mostly a shim: validate client requests,
forward to GSP via RPC, return results. This allowed:
- Open-sourcing the host kernel driver (the "open-gpu-kernel-modules" repo
  from 2022).
- Cleaner virtualization (vGPU became a thin RM per guest + shared GSP).
- Confidential Computing (the firmware runs in a protected domain).

### Generation 3: UVM integration (Pascal onwards)

UVM (`cudaMallocManaged`) was added around Pascal (2016) and became
architecturally required for Hopper+. This added `nvidia-uvm.ko` as a
separate kernel module.

### The result: layered but ugly

Each generation added layers without fully retiring earlier interfaces. So
we have:
- Legacy paths (direct kernel-driver hardware programming) for pre-Turing
  chips.
- GSP paths for Turing+.
- UVM paths for modern CUDA.
- vGPU paths for virtualization.

All coexist in the same code base. When reading the source, you'll see lots
of conditionals like `if (IS_GSP_CLIENT(pGpu))` and `if (IS_VIRTUAL(pGpu))`.

### Why mc looks "hand-rolled"

mc exists because replicating even a basic cudaMemcpy from scratch
requires understanding all five layers:
- Userspace: command stream encoding (§9 of gpfifo_pushbuffer_reference.md).
- Runtime: GPFIFO submission, doorbell ringing.
- Kernel (RM): object hierarchy setup.
- Kernel (UVM): VA space management.
- GSP: hardware programming (indirect, via RPCs).
- Hardware: PBDMA, CE, MMU.

Every mc bug we hit was a layer-boundary misunderstanding (e.g., "RM's
VA space must be IS_EXTERNALLY_OWNED for UVM to dup it" — cross-layer
constraint).

---

## 10. What each layer does for a cudaMemcpy

Let's trace a `cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost)` through every
layer:

### Userspace (application)

```c
cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
```

Calls into `libcudart.so`.

### libcudart

`libcudart.so` is a thin C++ library. It:
1. Looks up the current CUDA context from thread-local storage.
2. Calls the corresponding Driver API function: `cuMemcpyDtoH(dst, src, n)`.

### libcuda (the Driver API)

Inside `cuMemcpyDtoH`:
1. Validates args (null pointers, valid stream, etc.).
2. Looks up the CUDA stream and its associated GPFIFO channel.
3. Decides: is `src` a managed UVM pointer? If so, may need to migrate the
   page. Is `dst` pinned? If not, may need to stage via an internal buffer.
4. Builds the NVC8B5 method stream into the channel's pushbuffer:
   `SET_OBJECT` → `OFFSET_IN/OUT` → `LINE_LENGTH_IN` → `SET_SEMAPHORE` →
   `LAUNCH_DMA`. This is exactly the stream mc constructs.
5. Writes a GPFIFO entry pointing at the method stream.
6. Writes `GPPut` to USERD.
7. Writes the work-submit token to the VF doorbell — `+0x90` inside the
   USERMODE window (`NV_VIRTUAL_FUNCTION_DOORBELL` is `0x30090` relative to
   the VF register block, not a BAR0 offset; see
   `gpfifo_pushbuffer_reference.md §11`).
8. If the call is synchronous (default `cudaMemcpy`), spins on the
   completion semaphore; if async (`cudaMemcpyAsync`), returns immediately
   and the semaphore wait happens in `cudaStreamSynchronize`.

**Zero ioctls. Zero kernel entries. Pure userspace MMIO.**

### nvidia.ko and GSP

During the hot path: **nothing**. The setup (channel creation, MMU mapping)
happened at `cudaMalloc` / `cudaHostAlloc` / `cudaStreamCreate` time and is
cached.

### The GPU hardware

1. BAR0 write hits the host scheduler.
2. Host scheduler dispatches the channel on a PBDMA engine.
3. PBDMA reads GPPut from USERD.
4. PBDMA reads the GPFIFO entry from the ring.
5. PBDMA reads the pushbuffer via PCIe.
6. PBDMA decodes the method stream, delivering NVC8B5 methods to the CE on
   subchannel 4.
7. CE reads from src_va (HBM local), writes to dst_va (host DRAM via PCIe).
8. CE flushes and writes the semaphore.

### libcuda returns

Semaphore flip is observed. `cuMemcpyDtoH` returns success. `libcudart`
returns. `cudaMemcpy` returns. Application continues.

### Total time for 4 MiB on H100 PCIe: ~134 µs

That is the measured end-to-end figure — 4 MiB at the 31.3 GB/s of
`findings.md §11`.  About ~119 µs of it is data in flight; the remaining ~15 µs is
submission plus completion overhead.  That split is an estimate — the
134 µs is what was measured.

---

## 11. The open vs closed source split

`nvidia-open` (open-gpu-kernel-modules) is the official open-source build.
Its structure:

```
open-gpu-kernel-modules/
├── kernel-open/            Kernel-facing (open):
│   ├── nvidia/             nvidia.ko source (the "thin RM" layer)
│   ├── nvidia-uvm/         nvidia-uvm.ko source (fully open)
│   ├── nvidia-drm/         DRM integration
│   ├── nvidia-modeset/     Modeset module (partially open)
│   ├── nvidia-peermem/     PeerMem module for RDMA
│   └── nvidia-dbell/       (this fork) GPL shim for hw_breakpoint API,
│                           enables the Hopper doorbell watchpoint — see
│                           docs/findings.md §12 and the ~80-line
│                           nvidia-dbell.ko module it produces
├── src/                    Shared (open) RM logic:
│   ├── nvidia/             The "thick RM" — object classes, resserv
│   │   ├── src/            Implementation per module
│   │   ├── inc/            Internal headers
│   │   ├── arch/           Hardware-specific hooks (also has binary HALs)
│   │   └── generated/      Auto-generated from FINN + NVOC
│   └── common/sdk/         Public SDK headers (class IDs, ctrl commands)
└── (pre-built) nv-kernel.o  Binary HAL blob, linked into nvidia.ko
```

### What's open

- All of `nvidia-uvm.ko`.
- All of the "thin RM" (kernel-open/nvidia/): ioctl dispatch, platform
  layer, mmap handling.
- All of the "thick RM" (src/nvidia/): object lifecycle, resserv, GSP-RPC
  marshalling, memory management, class constructors.
- All SDK headers.

### What's closed

- **GSP firmware** (`gsp_*.bin`): ships as binary, signed by NVIDIA. Not in
  the repo.
- **libcuda.so**: the userspace driver. Completely closed.
- **Hardware-specific HAL blobs** (`nv-kernel.o_binary`): a small portion of
  chip-specific hardware programming code. Linked into nvidia.ko at build
  time.

### Why the split

- Performance: libcuda's command-stream encoding and JIT compilation is
  proprietary IP.
- Security: GSP firmware handles Confidential Computing; the signing chain
  matters.
- Hardware: the HAL blobs contain some internal register layouts NVIDIA
  hasn't documented.

### The research angle

Yan et al. 2026 explicitly targets the closed-source piece: **libcuda's
command stream encoding**. By intercepting the pushbuffer right before the
doorbell ring, they reconstruct what libcuda emits without having its
source — from inside the kernel, using a diverted mapping and a hardware
breakpoint.  That is what `nvidia-dbell.ko` does in this tree; `pbcap`
is the userspace shim that observes libcuda's *calls*, which their §3
shows cannot win the submission race.

mc is the inverse: instead of observing libcuda, we **replicate it**.
By reading the open-source kernel module to understand the ABI and then
building the pushbuffer ourselves, we produce a working CE submission
without ever loading libcuda.

---

## 12. Other modules: modeset, drm, peermem

For completeness, the other NVIDIA kernel modules:

### `nvidia-modeset.ko`

The display driver. Handles monitors, resolutions, mode transitions. Depends
on `nvidia.ko`.

Relevant for our work: historically, `nvidia-push.c` (the pushbuffer helper)
was compiled into nvidia-modeset.ko, not nvidia.ko. This is why mc had
to build the method stream from scratch — the nvidia-modeset helpers aren't
accessible without display device context.

### `nvidia-drm.ko`

Integrates NVIDIA GPUs with Linux's DRM (Direct Rendering Manager)
framework, which is what graphics APIs (OpenGL, Vulkan) use. Exposes
`/dev/dri/card0`, `/dev/dri/renderD128`, etc.

Not relevant for compute-only workloads.

### `nvidia-peermem.ko`

Enables **GPUDirect RDMA**: lets a NIC DMA directly to/from GPU HBM without
bouncing through host RAM. Provides a small kernel API that RDMA drivers
(e.g., MLNX OFED for Mellanox NICs) call to pin GPU memory for DMA.

Relevant for HPC and ML training clusters but not for our single-node work.

### `/dev/nvidia-uvm-tools`

A debug/instrumentation interface for UVM.  Not a module of its own — the
node is served from inside `nvidia-uvm.ko`.

### `nvidia-dbell.ko` (this fork only)

A small (~80 LOC) GPL-licensed shim module introduced by this research
fork. It wraps the kernel's `register_user_hw_breakpoint` /
`unregister_hw_breakpoint` APIs — which are exported
`EXPORT_SYMBOL_GPL` and therefore unreachable from the MIT-licensed
`nvidia.ko` — and re-exports them as plain `EXPORT_SYMBOL` thunks
`nv_dbell_bp_register` / `nv_dbell_bp_unregister`.

`nvidia.ko` loads `nvidia-dbell.ko` as a dependency (via `depmod`) and
uses the thunks from `kernel-open/nvidia/nv-doorbell-watch.c` to arm
x86 hardware watchpoints on the userspace-visible Hopper
`HOPPER_USERMODE_A` doorbell page. Required for the kernel-side
doorbell-watchpoint work; see `findings.md §12` for the full design.

Not present in upstream NVIDIA's open-gpu-kernel-modules.

---

## 13. Debugging across layers

### If a CUDA error surfaces, where should you look?

| Error surface | Likely layer |
|---|---|
| App returns `cudaErrorInvalidValue` | libcuda arg validation |
| `CUDA_ERROR_OUT_OF_MEMORY` | libcuda allocator, or RM heap exhausted |
| Process hangs in `cudaDeviceSynchronize` | GPU is hung; check semaphore state |
| Xid error in dmesg | GPU hardware or firmware fault (see Xid number) |
| "RM status: 0x…" in dmesg | RM object-management error |
| Kernel oops in nvidia.ko | RM kernel bug (copy_from_user missing, etc.) |
| GSP RPC timeout | GSP firmware hang or crash |

### Xid codes

Xid is NVIDIA's error-reporting convention for hardware-detected faults. The
kernel driver reports Xid errors to dmesg with a number that identifies the
class:

- Xid 13: Graphics SM exception.
- Xid 31: MMU fault (see our mc bring-up — this was our frequent
  failure mode before fixing GP entry encoding).
- Xid 32: Invalid or corrupted pushbuffer stream (our subchannel-0 bug).
- Xid 79: GPU has fallen off the bus (usually thermal or power).

Full list: `/proc/driver/nvidia/xid.txt` (if enabled) or NVIDIA's public
Xid documentation.

### Tracing

For deep debugging:
1. **ftrace with the `mc1` records** (the instrumentation this fork adds
   through `MC_TRACE`): see every RM ioctl and GSP RPC in real time.
   Categories are selected at runtime through the `mc_trace` module
   parameter; the record format is in `docs/reference/trace-format.md`.
2. **`strace`**: see every ioctl from userspace.
3. **`dmesg`**: hardware errors, driver warnings, Xid reports.
4. **NVIDIA Nsight Systems**: high-level CUDA profiler.
5. **GSP trace log**: on debug builds, GSP firmware can write traces to an
   HBM region readable by the host. Not enabled by default.

### Cross-layer debugging example

From mc's history: we saw `UVM_REGISTER_CHANNEL` fail with
`NV_ERR_INVALID_OBJECT (0x31)`.

Investigation path:
1. libcuda not involved (mc bypasses it).
2. RM ioctl (UVM_REGISTER_CHANNEL) — but which sub-step fails?
3. Added an `MC_TRACE(fifo, "retain_channel", …)` site in
   `nvGpuOpsRetainChannel` (the RM-side handler).
4. Saw `mc1 fifo/retain_channel step=retain_resources_ret status=0x31`.
5. Added a site at `_nvGpuOpsRetainChannelResources` entry.
6. Saw `mc1 fifo/retain_channel_resources step=enter channel_engine_type=1`
   — GR, but we allocated a CE channel!
7. Traced `nvGpuOpsGetChannelEngineType` → used `kchannelGetEngine_GM107`
   → used `kchannelGetRunlistId` → returned 0 (shared runlist) →
   first engine on runlist = GR → misclassified.
8. Fix: patch RM to use `pKernelChannel->engineType` directly.

This bug involved userspace ABI assumption (RM treats `chan_params.engineType`
as optional), RM code (misclassification logic), and GSP (sets up the
runlist sharing in the first place). Debugging required tracing across
layers.

---

## Further reading

- **open-gpu-kernel-modules/README.md** — the canonical starting point.
- **`src/nvidia/arch/nvalloc/unix/src/escape.c`** — the ioctl entry dispatcher.
- **`src/nvidia/src/kernel/vgpu/rpc.c`** — the GSP RPC mechanism.
- **`kernel-open/nvidia-uvm/uvm.c`** — the UVM ioctl entry point.
- **NVIDIA's CUDA C++ Programming Guide** — libcuda/libcudart semantics.
- **NVIDIA's "GSP Transition" blog posts** — motivation for the GSP move.

*Driver: open-gpu-kernel-modules 610.43.02.  Some
details differ on vGPU, MIG-partitioned, or Confidential Computing setups.*
