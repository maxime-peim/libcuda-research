# The GPU Compute Model: What's Inside an H100

**A hardware-focused description of how a modern NVIDIA GPU is organized:
the engines (SM, CE, NVDEC, etc.), the memory system, the front-end
command-processing hardware, and the hardware scheduling primitives.**

This document explains what the GPU *is* as a machine, not what software you
use to drive it. Its goal is to demystify the terms that appear in
documentation and traces ("SM", "PBDMA", "runlist", "TSG", "subchannel",
"CE", "COPY0", "GR0") by mapping each to an actual piece of silicon.

After reading this, the command-stream details in
`gpfifo_pushbuffer_reference.md` should feel motivated rather than arbitrary:
*every* field of the GPFIFO entry and every NVC8B5 method exists because
some hardware state machine reads it and takes an action.

Prerequisites: `host_gpu_communication_primer.md` (PCIe, MMU, DMA) is
helpful but not required; `nvidia_software_stack.md` is complementary.

---

## Table of Contents

1. [The GPU as a multi-processor system](#1-the-gpu-as-a-multi-processor-system)
2. [Physical organization: an H100 at a glance](#2-physical-organization-an-h100-at-a-glance)
3. [Memory hierarchy](#3-memory-hierarchy)
4. [The engine zoo](#4-the-engine-zoo)
5. [SMs and the compute hierarchy](#5-sms-and-the-compute-hierarchy)
6. [Copy Engines — the DMA workhorses](#6-copy-engines--the-dma-workhorses)
7. [Channels — the hardware context](#7-channels--the-hardware-context)
8. [PBDMA — the front-end for each channel](#8-pbdma--the-front-end-for-each-channel)
9. [Subchannels — multiplexing engines within a channel](#9-subchannels--multiplexing-engines-within-a-channel)
10. [TSGs — time-slice groups](#10-tsgs--time-slice-groups)
11. [Runlists — how channels are scheduled onto engines](#11-runlists--how-channels-are-scheduled-onto-engines)
12. [The host scheduler and doorbell dispatch](#12-the-host-scheduler-and-doorbell-dispatch)
13. [The GPU MMU and FERMI_VASPACE_A](#13-the-gpu-mmu-and-fermi_vaspace_a)
14. [Instance memory: RAMIN, RAMFC, USERD](#14-instance-memory-ramin-ramfc-userd)
15. [Method spaces and engine classes](#15-method-spaces-and-engine-classes)
16. [Faults and preemption](#16-faults-and-preemption)
17. [Putting it together: a CE transfer in hardware terms](#17-putting-it-together-a-ce-transfer-in-hardware-terms)

---

## 1. The GPU as a multi-processor system

An H100 is a system-on-chip. It contains, roughly:

- **Compute**: 114 Streaming Multiprocessors on the PCIe SKU (132 on SXM,
  144 on the full die), each with 128 CUDA cores,
  4 tensor cores, L1 cache, shared memory, register file.
- **Specialized engines**: 20 Copy Engines (CEs), hardware video encoders
  (NVENC), decoders (NVDEC), JPEG decoders (OFA), optional RT cores
  (not on H100), security engine (SEC2).
- **Memory**: 80 GB HBM2e in 5 stacks, ~2 TB/s aggregate bandwidth, managed
  by an on-die memory controller with multiple channels.
- **Interconnect**: PCIe Gen5 x16 for host; NVLink for GPU↔GPU (H100 has
  NVLink 4 with 18 links, ~900 GB/s aggregate).
- **Front-end**: a set of hardware units that parse pushbuffers, schedule
  work, and dispatch methods to engines.
- **System processor**: the GSP (RISC-V core running NVIDIA firmware), as
  discussed in `nvidia_software_stack.md`.
- **An L2 cache** shared across all engines: ~50 MB on H100.

The GPU's internal architecture is a **multi-processor system with shared
memory**, where "processors" includes both programmable cores (SMs) and
fixed-function engines (CE, NVDEC, etc.).

### Why fixed-function engines exist

A DMA-copy workload on an SM uses ~0% of its compute capability. Dedicating
an SM to move bytes would be wasteful. The CE is a purpose-built state
machine that moves bytes at line rate (PCIe or HBM-bound) using ~1/100th
the silicon of an SM. It frees SMs to compute.

Similar reasoning applies to video encoding (NVENC) and decoding (NVDEC):
these are hot paths on modern workloads and deserve dedicated hardware.

### Why engines are independently addressable

Because SMs and CEs are separate hardware, they can run **in parallel**: an
SM can compute while a CE concurrently transfers data. CUDA exposes this
via streams — work submitted to different streams can execute on different
engines concurrently.

Each engine is reached by submitting to a **channel bound to that engine**.
The channel-to-engine binding is what makes "this GPFIFO submission runs on
CE1" a well-defined statement.

---

## 2. Physical organization: an H100 at a glance

Block diagram of an H100 GPU (simplified):

```
┌─────────────────────────────────────────────────────────────────────┐
│                         H100 GPU package                            │
│                                                                     │
│  ┌───────────────────┐   ┌───────────────────┐                      │
│  │  GPC0             │   │  GPC1     ...     │       8 GPCs         │
│  │  ┌──────────────┐ │   │                   │       (Graphics      │
│  │  │ TPC0         │ │   │                   │       Processing     │
│  │  │ ┌──┐ ┌──┐    │ │   │                   │       Clusters)      │
│  │  │ │SM│ │SM│ 2× │ │   │                   │                      │
│  │  │ └──┘ └──┘    │ │   │                   │                      │
│  │  └──────────────┘ │   │                   │       Each GPC has:  │
│  │  ┌──────────────┐ │   │                   │       - 9 TPCs       │
│  │  │ TPC1 ...     │ │   │                   │       - 1 Raster     │
│  │  └──────────────┘ │   │                   │         engine       │
│  └───────────────────┘   └───────────────────┘                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────┐                │
│  │  L2 cache (~50 MB, shared across all engines)   │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │ Memory     │ │ Memory     │ │ Memory     │  5 HBM controllers    │
│  │ controller │ │ controller │ │ controller │                       │
│  │ + PHY      │ │ + PHY      │ │ + PHY      │                       │
│  └────────────┘ └────────────┘ └────────────┘                       │
│         │              │              │                             │
│  ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴──────┐                       │
│  │ HBM2e stk0  │ │ HBM2e stk1 │ │ HBM2e stk2│  ... (5 stacks, 80 GB)│
│  └─────────────┘ └────────────┘ └───────────┘                       │
│                                                                     │
│  ┌────────────────────────────────────────────┐                     │
│  │  Host Interface & Front-End                │                     │
│  │  - PCIe Gen5 x16                           │                     │
│  │  - NVLink4 ×18                             │                     │
│  │  - Host scheduler (runlist, doorbells)     │                     │
│  │  - PBDMA (per-channel front-end) ×N        │                     │
│  │  - 20× Copy Engine                         │                     │
│  │  - NVENC, NVDEC, OFA engines               │                     │
│  │  - GSP (RISC-V, firmware-controlled)       │                     │
│  └────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key numbers on H100 PCIe

| Component | Count / size |
|---|---|
| SMs | 114 (PCIe SKU; 132 on SXM, 144 on the full die) |
| CUDA cores | 14,592 (16,896 in SXM SKU) |
| Copy engines | Up to 20 |
| L2 cache | ~50 MB |
| HBM2e capacity | 80 GB |
| HBM2e bandwidth | ~2 TB/s |
| PCIe link | Gen5 x16 (~63 GB/s per direction, full-duplex) |
| NVLink | 18 links × 50 GB/s = 900 GB/s (SXM; the PCIe SKU this project uses has no NVLink fabric) |
| Tensor core perf | ~756 TFLOPS FP16, ~1513 TFLOPS FP8 (PCIe SKU; the widely-quoted 989/1979 are SXM) |

### Differences across SKUs

- **H100 PCIe**: 80 GB HBM2e (~2 TB/s), PCIe Gen5 x16, 350 W. What mc runs on.
- **H100 SXM**: 80 GB HBM3 (~3.35 TB/s), NVLink, 700 W.
- **H200**: H100 + more HBM (141 GB) + higher bandwidth HBM3e.
- **Grace Hopper (GH200)**: H100 + Grace ARM CPU on-package with NVLink-C2C
  for cache-coherent CPU↔GPU access.

For understanding GPFIFO-level work submission, all variants are similar.
The numeric addresses (e.g., `NV_VIRTUAL_FUNCTION_DOORBELL = 0x30090`,
relative to the VF register block) are identical across H100/H200/GH200.

---

## 3. Memory hierarchy

From the perspective of a workload running on the GPU:

```
┌─────────────────────────────────────────────────────────┐
│ SM's register file (~256 KB/SM)                          │ ~1 ns, 32k regs
├─────────────────────────────────────────────────────────┤
│ Shared memory / L1 (~192 KB/SM combined)                │ ~1–2 ns
├─────────────────────────────────────────────────────────┤
│ L2 cache (~50 MB, shared)                               │ ~100 ns
├─────────────────────────────────────────────────────────┤
│ HBM2e (80 GB on package)                                 │ ~300–500 ns, 2 TB/s
├═════════════════════════════════════════════════════════┤
│ ——— PCIe boundary ———                                    │
├─────────────────────────────────────────────────────────┤
│ Host DRAM (hundreds of GB)                               │ ~500 ns–1 µs via PCIe
│                                                          │  ~63 GB/s per direction
└─────────────────────────────────────────────────────────┘
```

### Implications

- **HBM is fast and local**: data in HBM is accessed at ~2 TB/s aggregate by
  any GPU client. CE reads src from HBM at close-to-HBM-bandwidth.
- **Host DRAM is much slower**: every access crosses PCIe. Bandwidth capped at
  ~63 GB/s per direction (Gen5 x16), and measured copies land below that —
  ~55 GB/s at boost clocks (box-dependent; see `findings.md §11, §15`);
  latency ~10× HBM.
- **L2 is a big deal**: 50 MB can cache entire working sets for many
  workloads. For bandwidth-bound operations, L2 hits are ~30× faster than
  HBM misses.

### What does memory residency actually look like?

Each physical page in HBM or host DRAM has a fixed physical address. A
"GPU virtual address" is a name that the GPU MMU resolves to a physical
address. The MMU page tables encode *which* physical location (HBM vs host
DRAM) backs each VA.

For mc:
- `d_buf` (source): physically in HBM. GPU MMU walks → HBM physical address.
- `h_buf` (destination): physically in host DRAM. GPU MMU walks → host phys
  address (with aperture bit set).
- `pushbuffer`: physically in host DRAM, pinned. GPU MMU → host.
- `GPFIFO ring`: physically in HBM. GPU MMU → HBM.
- `sema`: physically in host DRAM, pinned. GPU MMU → host.

### Write-combine, caching, coherency

The SMs, CEs, and L2 all participate in the GPU's internal coherency
protocol. The GPU presents a mostly-coherent view to its own engines.

The GPU is *not* in the CPU's coherency domain (on conventional x86 PCIe —
Grace-Hopper's NVLink-C2C changes this). This means CPU-GPU shared data must
go through PCIe with explicit fence/flush, as covered in the primer.

---

## 4. The engine zoo

An engine is a hardware unit that consumes commands and does work. H100 has
many kinds:

| Engine | Count | Purpose |
|---|---|---|
| SM (Streaming Multiprocessor) | 114 | General-purpose compute (CUDA kernels) |
| CE (Copy Engine) | Up to 20 | DMA transfers |
| NVDEC | 7 | H.264/H.265/AV1 video decode |
| NVENC | 3 | H.264/H.265 video encode |
| OFA | 1 | Optical Flow Accelerator (motion estimation) |
| SEC2 | 1 | Security / Confidential Computing |
| GSP | 1 | Firmware-controlled system management |

Graphics engines (GR) are not typically used on datacenter GPUs like H100
for compute workloads, but the hardware is present for graphics SKUs.

### Engines and channels

A **channel** is bound to exactly one engine type. So:
- A "graphics channel" can submit to GR (SMs in graphics mode).
- A "compute channel" can submit to SMs in compute mode.
- A "copy channel" can submit to a CE.
- etc.

The engine type is specified at channel-allocation time.  `mc` does not
hard-code one: it picks the first non-GRCE logical copy engine the GPU
reports (`pick_non_grce_lce` in `reverse/mc/mc_core.c`), because the GRCE
instances are shared with graphics contexts.  One channel, one engine.

### Why there are multiple CE instances

Multiple CEs allow **concurrent DMA transfers**. Common patterns:
- Bidirectional transfers: CE1 does H2D while CE2 does D2H.
- Multi-stream: each CUDA stream gets its own CE channel, up to the
  hardware limit.

On H100 with up to 20 CEs, you can saturate PCIe + multiple NVLinks in
parallel.

### Reference: RM engine type enum

From `src/nvidia/inc/kernel/gpu/gpu_engine_type.h`:

```
RM_ENGINE_TYPE_NULL   = 0
RM_ENGINE_TYPE_GR0    = 1    ← graphics (SM graphics mode)
RM_ENGINE_TYPE_GR1    = 2
...
RM_ENGINE_TYPE_GR7    = 8
RM_ENGINE_TYPE_COPY0  = 9    ← often a GRCE; mc skips these
RM_ENGINE_TYPE_COPY1  = 10
...
RM_ENGINE_TYPE_COPY19 = 28
RM_ENGINE_TYPE_NVDEC0 = 29
...
```

---

## 5. SMs and the compute hierarchy

SMs (Streaming Multiprocessors) are the GPU's programmable compute cores.
They're mostly beyond scope for the copy path (which uses only CEs), but some
vocabulary is useful.

### Hierarchy

```
GPU
└── GPC (Graphics Processing Cluster, ~8 per H100)
    └── TPC (Texture/Thread Processing Cluster, 9 per GPC)
        └── SM (Streaming Multiprocessor, 2 per TPC)
            └── Warp (32 threads, executed in lockstep)
                └── Thread
```

### What's inside an SM

An SM contains:
- A register file (256 KB, partitioned across active threads).
- 128 CUDA cores (32-bit FP/INT ALUs).
- 4 Tensor Cores (matrix-multiply accelerators).
- Load/store units.
- Shared memory + L1 cache (192 KB, reconfigurable split).
- A warp scheduler that issues instructions from active warps.

### The warp execution model

Threads are grouped into **warps** of 32. All threads in a warp execute the
same instruction at the same cycle (SIMT — Single Instruction, Multiple
Threads). If threads diverge (branch differently), the hardware masks off
inactive lanes and re-converges later.

Relevance to the copy path: **none**. CEs don't use SMs. We don't launch any
CUDA kernels. But knowing that SMs exist and are entirely separate from the
DMA path is important.

### Compute channels and graphics channels

A compute channel (subchannel 1) submits to SMs in "compute mode" where the
full tensor-core and CUDA-core feature set is available. A graphics channel
(subchannel 0) submits to SMs in "graphics mode" with a pipeline stage
(vertex shader → fragment shader). Same physical SMs, different feature
modes.

---

## 6. Copy Engines — the DMA workhorses

The Copy Engine is the hardware that actually moves bytes for any DMA-based
transfer. It is the engine that `mc_memcpy` targets.

### What a CE does

A CE is a state machine that, given a source address, destination address,
and byte count, reads from source and writes to destination at line rate.
That's essentially its entire job.

Bells and whistles beyond the basic copy:
- **Transforms**: can optionally scale, convert data types (e.g., FP32 →
  FP16), or perform simple arithmetic during copy.
- **2D and 3D transfers**: pitched/tiled surfaces for graphics.
- **Block-linear layout support**: for textures and framebuffers.
- **Atomic writes**: some variants support atomic RMW at the destination.
- **Multiple peer accesses**: can copy between two peer GPUs over NVLink
  without staging through host.

For mc, we use only the most basic capability: 1D pitch-mode copy
from VIRTUAL source to VIRTUAL destination with semaphore release on
completion.

### The CE's "methods"

A CE exposes its programming interface as a set of **methods** —
addressable registers in a method space. See `gpfifo_pushbuffer_reference.md`
§9 for the complete list used by mc:

```
0x240  SET_SEMAPHORE_A     (sema_va high)
0x244  SET_SEMAPHORE_B     (sema_va low)
0x248  SET_SEMAPHORE_PAYLOAD
0x300  LAUNCH_DMA          (fires the transfer)
0x400  OFFSET_IN_UPPER     (src_va high)
0x404  OFFSET_IN_LOWER     (src_va low)
0x408  OFFSET_OUT_UPPER    (dst_va high)
0x40C  OFFSET_OUT_LOWER    (dst_va low)
0x418  LINE_LENGTH_IN      (byte count for 1D)
```

These are read by the CE's front-end when PBDMA delivers them. Writing a
method is literally equivalent to writing a register inside the CE.

### CE classes

NVIDIA versions the CE by GPU generation:

| Class | Generation |
|---|---|
| `KEPLER_DMA_COPY_A` (0xA0B5) | Kepler |
| `MAXWELL_DMA_COPY_A` (0xB0B5) | Maxwell |
| `PASCAL_DMA_COPY_A` (0xC0B5) | Pascal |
| `TURING_DMA_COPY_A` (0xC5B5) | Turing |
| `AMPERE_DMA_COPY_A/B` (0xC6B5/0xC7B5) | Ampere |
| **HOPPER_DMA_COPY_A (0xC8B5)** | **Hopper (what mc uses)** |
| `BLACKWELL_DMA_COPY_A` (0xC9B5) | Blackwell |

The method space is mostly stable across generations with additions for
newer features. The LAUNCH_DMA flag encodings mc uses work on most
Kepler+; what varies is whether the copy and its semaphore release share
one launch (the fused 0x18e form) or are split into two, as H100 libcuda
does and mc's host path follows.

### CE ≠ PCIe

Common confusion: the CE is not "the PCIe engine." The CE is a
general-purpose DMA controller that can move bytes anywhere the MMU points.
For D2H transfers, it happens to be programmed with a dst_va that resolves
to host DRAM, so the outgoing writes go over PCIe. For HBM-to-HBM transfers,
the same CE would never touch PCIe.

---

## 7. Channels — the hardware context

A **channel** is the fundamental unit of GPU work isolation. Think of it
as analogous to a process in an OS: each channel has its own address
space, its own command queue, its own execution context.

### What's in a channel

A channel is a data structure (in GPU memory) plus associated hardware
state, representing:

- **A GPFIFO ring**: the command queue for this channel.
- **A pushbuffer** (or multiple; the GPFIFO points at them).
- **USERD page**: the CPU-visible producer/consumer indices.
- **Instance memory (RAMIN)**: backing store for channel registers.
- **RAMFC**: saved/restored PBDMA register state on context switches.
- **A GPU VA space** (FERMI_VASPACE_A): the page tables this channel uses.
- **An engine binding**: which engine this channel submits to (CE, GR, etc.).
- **Per-subchannel object bindings**: what class is bound to each of
  8 subchannels (see §9).

### Why channels exist

The GPU is a shared resource. Multiple processes (or multiple CUDA contexts
within a process) must use it without interfering. Channels are the
hardware-enforced isolation primitive:

- Each channel has its own page tables → process A can't read process B's
  GPU memory.
- Each channel has its own command queue → process A's submissions don't
  interleave with process B's.
- Each channel has its own doorbell slot → MMIO targets are per-channel.

### Channel IDs

Each channel gets a hardware **Channel ID** (CHID) when allocated — a
small integer (on H100 up to thousands of channels). The CHID identifies
the channel in:
- The runlist (what the host scheduler sees).
- MMU faults (the fault reports which CHID faulted).
- PBDMA internal state.

mc never uses the CHID directly; RM and the work-submit token
abstract it. But in debug messages (Xid reports), the CHID is the primary
identifier of the faulting channel.

### Channel classes on Hopper

```
HOPPER_CHANNEL_GPFIFO_A (0xC86F) — the class for all Hopper channels
```

All channels are "GPFIFO channels" (channels that consume GPFIFO entries).
The engine they submit to is determined by the `engineType` field at
allocation time, not by the class.

---

## 8. PBDMA — the front-end for each channel

PBDMA (Pushbuffer DMA) is the per-channel hardware front-end. It's the
component that actually reads GPFIFO entries, fetches pushbuffers, decodes
method headers, and delivers methods to the downstream engine.

### What PBDMA does, cycle by cycle

When a channel is scheduled:
1. PBDMA loads the channel's RAMFC-saved state (GP_BASE, GP_PUT, GP_GET,
   internal method pointers).
2. PBDMA reads `GP_PUT` from USERD (on Hopper, local HBM read; on
   pre-Hopper where USERD lives in sysmem, a PCIe read to host DRAM).
3. If `GP_PUT > GP_GET`, there's work. PBDMA reads the GPFIFO entry at
   index `GP_GET` (local HBM read, since GPFIFO lives in HBM).
4. PBDMA decodes the entry: pb_va, length, any OPCODE (extended-base or
   NOP).
5. PBDMA issues reads for the pushbuffer bytes (PCIe reads if pushbuffer is
   in host DRAM).
6. PBDMA parses the method stream: header-data-data-header-data...
7. Each method is routed to the appropriate subchannel's bound engine.
8. When all methods are consumed, PBDMA updates `GP_GET`, writes it to
   USERD, and checks for more work.
9. If no more work, PBDMA saves state to RAMFC and yields.

### PBDMA's state

PBDMA is essentially a small processor with dedicated registers:
- Current pushbuffer pointer (PB_CURRENT).
- Current pushbuffer remaining count.
- Current subchannel.
- Current method address (incremented for INCR-type headers).
- Various flags (wait-on-sync, subroutine depth, etc.).

These are saved to RAMFC on context switch and restored when the channel
is rescheduled.

### PBDMA is a hardware unit, not a process

Importantly, PBDMA isn't "running GSP firmware." It's a fixed-function
hardware state machine. Its behavior is defined by the silicon, not by
firmware. This is why the pushbuffer format is fixed and stable across
GPUs — PBDMA would need silicon changes to support a new format.

### Per-channel vs shared PBDMA hardware

A GPU has a finite number of PBDMA engines (NVIDIA doesn't publish the
exact count, but it's smaller than the max channel count). Channels are
time-sliced onto PBDMA engines — multiple channels share each PBDMA
engine, with context switches between them.

This is why a channel can be "scheduled" (on the runlist) or
"unscheduled" (dormant). Scheduling is the act of time-slicing the
channel onto a PBDMA engine.

### Relationship to the host scheduler

PBDMA handles **one channel's** command consumption. The **host scheduler**
is a separate piece of hardware that decides which channel each PBDMA
engine runs. The host scheduler consumes the runlist (a list of schedulable
channels) and rotates channels onto PBDMA.

---

## 9. Subchannels — multiplexing engines within a channel

A single channel can potentially talk to multiple engines — e.g., a
graphics channel might need to drive the 3D engine, the 2D engine, and I2M
(inline-to-memory) all within one command stream. To support this, each
channel has **8 subchannels**, and each method header specifies which
subchannel it targets.

### Subchannel bindings

At any point in time, each of a channel's 8 subchannels is either:
- **Unbound** (no object installed).
- **Bound to a specific engine class** (e.g., subch 4 bound to
  HOPPER_DMA_COPY_A).

The binding is set by a `SET_OBJECT` method sent to that subchannel. Once
bound, subsequent methods to that subchannel are routed to the bound
engine.

### Conventional subchannel assignments

From `src/common/sdk/nvidia/inc/class/cla06fsubch.h`:

| Subch | Engine |
|---|---|
| 0 | 3D (graphics) |
| 1 | COMPUTE (SMs in compute mode) |
| 2 | I2M (inline-to-memory, small inline DMA) |
| 3 | 2D (2D graphics blit) |
| **4** | **COPY_ENGINE (CE)** |
| 5–7 | reserved |

The assignments are conventions enforced by software; the hardware accepts
any `SET_OBJECT` binding on any subchannel. But tooling, documentation, and
reference implementations assume these conventions.

### Why subchannel 4 matters for mc

`mc_memcpy` is a pure-CE workload. **Every** method in its pushbuffer targets
subchannel 4. The first method is `SET_OBJECT(0xC8B5)` on subchannel 4,
binding HOPPER_DMA_COPY_A to that subchannel. Every subsequent method
(OFFSET_IN, LINE_LENGTH_IN, LAUNCH_DMA, etc.) goes to subchannel 4 and is
routed by PBDMA to the CE.

### What happens if you target subchannel 0

PBDMA receives a method addressed to subchannel 0 but no object is bound
there. PBDMA reports this as a fatal error (Xid 32, "invalid pushbuffer
stream") and the channel is terminated.

This was one of the bugs in mc bring-up (bug #11 in the log): using
subchannel 0 (the default from an uninitialized `INCR_HEADER` macro)
caused Xid 32 until we added the subchannel parameter to the macro.

### How CUDA uses subchannels

A CUDA "compute channel" typically uses:
- Subchannel 1: COMPUTE engine (for CUDA kernels).
- Subchannel 4: CE (for cudaMemcpy and friends).
- Subchannel 2: I2M (for small inline transfers that piggyback on SM work).

A single channel can interleave compute and DMA work across subchannels —
the CPU builds a pushbuffer that contains both kernel launches and memory
copies, and PBDMA routes each method to the correct engine.

---

## 10. TSGs — time-slice groups

A **TSG** (Time-Slice Group) is a group of channels that share a
context-switch time slice. When the scheduler switches away from the TSG,
all its channels are paused; when it switches back, they all resume.

### Why TSGs exist

Two use cases:

1. **Compute + graphics interop**: a graphics channel and a compute channel
   might share state. You don't want the scheduler context-switching
   between them (thrashing context). Group them in a TSG.

2. **Subcontexts**: one process may want multiple isolated execution
   contexts that never run concurrently. TSGs enforce that only one member
   runs at a time, while sharing VA space and certain resources.

### TSGs on modern GPUs

Starting with Volta, TSGs became mandatory for all compute channels —
**every channel must be a member of some TSG**, even if the TSG has only
one channel. This is a scheduler simplification: the scheduler only deals
with TSGs, not individual channels.

For mc: we allocate a TSG (`KEPLER_CHANNEL_GROUP_A` = 0xA06C, the
generic class for all TSGs since Kepler) with `engineType` set to the
first non-GRCE logical copy engine the GPU reports, and our CE channel
is a member of that TSG.

### What's shared in a TSG

- **VA space**: all channels in a TSG share `FERMI_VASPACE_A`. This is why
  our TSG has `hVASpace = h_vaspace` but our channel has `hVASpace = 0`
  (inherited from TSG).
- **Engine type**: all channels in a TSG target the same engine class.
- **Context save/restore overhead**: when the scheduler switches away, all
  channels' PBDMA state is saved together.

### TSGs and runlists

A TSG occupies one or more entries on a runlist. When the scheduler reaches
a TSG's entry, it may run multiple channels from that TSG during the
time slice (via the per-engine PBDMA time-slicing).

---

## 11. Runlists — how channels are scheduled onto engines

A **runlist** is a list of TSGs (and channels) scheduled to run on a
specific engine. Each engine type has its own runlist.

### Runlist structure

On Hopper, a runlist is a linked or array structure in GPU memory containing
entries like:

```
entry: { TSG_id, CHID, priority, time_slice_us, ... }
```

The GPU's host scheduler walks the runlist and dispatches each entry onto
the engine's PBDMA in order. When a time slice expires or a channel yields,
the scheduler moves to the next entry.

### Runlist IDs

Each engine has a unique runlist ID. The mapping from engine to runlist is
set at GPU init time by GSP firmware. A channel's runlist ID indicates
which runlist it lives on.

### Why runlist IDs can be shared

On Hopper, **GR and CE can share runlist 0** — this was the source of
mc's "channel classified as GR" bug. When GR and CE are on the same
runlist, the first engine you find when walking the runlist mapping is GR
(arbitrary but consistent). Our patch in `nvGpuOpsGetChannelEngineType`
works around this by using the per-channel engineType field instead.

### Scheduling a channel

Adding a channel to a runlist is what `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE`
does. Before SCHEDULE, a channel exists but is not on any runlist — PBDMA
never sees it. After SCHEDULE (`bEnable=TRUE`), the channel's TSG is placed
on the appropriate engine's runlist, and the scheduler will dispatch it when
doorbells arrive or time-slice rotations occur.

### Removing a channel

`NVA06F_CTRL_CMD_GPFIFO_SCHEDULE` with `bEnable=FALSE` removes the channel
from the runlist. Channel state is preserved, but no further work executes.

### Multi-GPU and runlists

On multi-GPU systems, each GPU has its own independent runlists. Channels
are bound to a specific GPU at allocation time and cannot migrate.

---

## 12. The host scheduler and doorbell dispatch

The **host scheduler** is the GPU hardware component that manages runlists,
time-slicing, and doorbell dispatch. It's part of the front-end, separate
from per-channel PBDMAs.

### Doorbell dispatch

When the CPU writes to a VF doorbell register, the host scheduler:
1. Decodes the token written.
2. Identifies the target channel (by CHID, encoded in the token).
3. Verifies the channel is on a runlist.
4. Notifies the corresponding PBDMA to begin processing.

The doorbell is essentially an **interrupt replacement**: rather than the
CPU raising an IPI that travels through the GPU's interrupt handler, the
CPU writes a register that the host scheduler checks directly. This is
much faster.

### Runlist dispatch (no doorbell)

Even without doorbells, the host scheduler periodically checks runlist
entries for pending work (via time-slice rotation). A doorbell just
accelerates the wake-up; eventually the scheduler would have run the
channel anyway.

### Why the doorbell matters for latency

If you rely on time-slice rotation alone, a new submission might wait
milliseconds for the scheduler to rotate to your channel. With a doorbell,
the scheduler immediately prioritizes your channel, reducing wake-up to
microseconds.

For high-throughput, high-submission-rate workloads (like CUDA), this is
essential. For bulk transfers, it's not critical but still beneficial.

### The VF doorbell (NV_VIRTUAL_FUNCTION_DOORBELL)

On Hopper, the doorbell register is at `0x30090` within the virtual-function
register block (see `dev_vm.h`) — *not* at BAR0 + 0x30090. On bare metal the
block is based at `0xB80000`, so the absolute address is BAR0 + `0xBB0090`;
`0x30090` alone is the SR-IOV guest's view. Userspace uses neither: it writes
`+0x90` inside the USERMODE window (`gpfifo_pushbuffer_reference.md §11`).
It's 32-bit: you write the 32-bit work-submit-token.

The token is assigned by RM/GSP when the channel is allocated and retrieved
via `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN`. It encodes the
CHID plus some scheduler metadata.

mc's final hot-path step is exactly this:
```c
*vf_doorbell = work_submit_token;   // wake host scheduler
```

### Priority and preemption

Modern GPUs support **channel priorities**: high-priority channels
preempt running low-priority ones (instruction-level on SMs; at method
boundaries on CEs). This is configured via control calls on the TSG
(`NVA06F_CTRL_CMD_SET_TIMESLICE` and similar).

mc uses default priority — single-tenant workloads don't care about
scheduling policy.

---

## 13. The GPU MMU and FERMI_VASPACE_A

The GPU's MMU translates GPU virtual addresses to physical addresses. Its
structure is similar to the CPU MMU (multi-level page tables), but with
GPU-specific features: multiple page sizes, multiple apertures (HBM /
sysmem / peer), compression metadata.

### FERMI_VASPACE_A

`FERMI_VASPACE_A` (class 0x90F1, name inherited from Fermi but used through
Hopper and beyond) is an RM object representing a page-table tree. Multiple
channels can share one VA space (via a TSG); they see the same mappings.

Allocating a FERMI_VASPACE_A doesn't allocate all page tables at once;
it allocates the root (PDE0) and fills in intermediate levels on demand.

### IS_EXTERNALLY_OWNED

When the VA space is allocated with `IS_EXTERNALLY_OWNED`, ownership of
page-table updates transfers to an "external" entity — for mc, that
entity is UVM. Without this flag, RM refuses to let UVM install PTEs into
the VA space.

### Page walk during a DMA

When the CE reads from `src_va = 0x76baa0800000`:
1. The CE's MMU tries to translate `src_va`.
2. TLB miss → walk the page tables.
3. Load PDE0 from RAMIN (or cache).
4. Follow it to PDE1, PDE2, PDE3, finally PTE.
5. PTE has physical address + aperture + permissions.
6. Issue read to the correct physical subsystem (HBM controller for HBM
   aperture; PCIe master for sysmem aperture).

TLB caches translations to avoid walks on every access. A page-table walk
on a TLB miss takes ~30–100 ns (sequential memory loads).

### Page sizes

Hopper supports: 4 KB, 64 KB, 2 MB, 512 MB. Larger pages reduce TLB
pressure but increase alignment requirements. UVM picks page sizes based
on allocation patterns.

### Page faults

If the MMU finds no valid PTE for an access, it raises a **MMU fault**.
The fault propagates up:
1. GPU MMU reports to the fault handler.
2. Fault handler (in GSP) logs the fault.
3. GSP RPC reports to host RM.
4. RM reports Xid 31 via dmesg.

mc hit Xid 31 many times during bring-up — each time, the fault
address told us exactly what the CE was trying to read. E.g., fault at
`0xe3_1e704000` when pb_va should have been `0x72e3...` revealed that
the extended-base entry was missing (bug #8).

### ATS (Address Translation Services)

Hopper supports ATS/NVLink-ATS for cache-coherent GPU↔CPU page-table
sharing on certain platforms (Grace-Hopper, or with PCIe ATS extensions).
Our H100 PCIe setup does not use ATS. UVM must handle the case where ATS
is supported but not enabled (the `uvm_va_space_ats_enabled` helper in
`uvm_va_space.h`, consulted when a GPU VA space is created —
`uvm_va_space.c`, `create_gpu_va_space`).

---

## 14. Instance memory: RAMIN, RAMFC, USERD

Each channel has several small backing structures in GPU memory:

### RAMIN — the channel's instance memory

RAMIN is a per-channel block in HBM containing the channel's persistent
state: PBDMA registers (saved across context switches), MMU root pointer
(PDE0 address for the VA space), engine-specific state.

Size: a few KB per channel.

Allocated by RM during `kchannelAllocMem_GM107` — you can see this in
traces as `instmem alloc`.

### RAMFC — FIFO context memory

A subregion of RAMIN specifically for saving PBDMA's register state on
context switches. When the scheduler switches away from a channel, PBDMA
saves its working registers to RAMFC; when switching back, RAMFC is
restored to PBDMA.

Defined by `DRF_SIZE(NV_RAMIN_RAMFC) / 8` bytes.

### USERD — the user-accessible control page

USERD is where the CPU-GPU-shared indices (GPGet, GPPut) live.

On Hopper, USERD is placed in **FBMEM** (HBM), not host DRAM.  PBDMA
reads GPPut locally from HBM; the CPU writes GPPut through a BAR1
CPU alias of the FBMEM allocation (write-combining on x86, one
PCIe posted-write per `sfence`).  Earlier GPUs (Ampere and before)
often placed USERD in sysmem, and pedagogy about USERD from that
era still circulates — but the open-RM 595 driver on Hopper always
puts USERD in FBMEM for both libcuda and mc.

USERD size is 512 bytes per channel; each slot must be 4 KiB
aligned.  libcuda packs multiple channels' USERD slots inside a
2 MiB BAR1-apertured FBMEM region.  mc's CE channel
reuses that pattern: `h_gpu_ctl_mem` is a 2 MiB `NV01_MEMORY_LOCAL_USER`
(HBM) allocation with the GPFIFO ring at offset 0 and USERD at
offset `0x2000`; `chan_params.hUserdMemory[0] = h_gpu_ctl_mem` and
`userdOffset[0] = 0x2000`.

RM allocates and formats USERD during channel construction. The per-channel
USERD is exposed to userspace via the channel's mmap.

### The memory-instance flow

```
RAMIN ────┬──── PBDMA saved registers (RAMFC sub-region)
          ├──── MMU root pointer (PDE0 address)
          └──── Engine-specific context (GR pool, CE state, ...)

USERD ───── GPGet / GPPut (exposed to CPU)
```

---

## 15. Method spaces and engine classes

Every engine class defines a **method space** — a set of addressable
registers that software writes to program the engine. The method address
is 13 bits (after `>> 2`) in the GPFIFO header, giving 8192 possible method
slots per subchannel.

### Method space for HOPPER_DMA_COPY_A (class 0xC8B5)

From `src/common/sdk/nvidia/inc/class/clc8b5.h`:

| Range | Purpose |
|---|---|
| 0x000–0x0FF | Common Kepler-channel methods (SET_OBJECT, NOP, WFI) |
| 0x100–0x1FF | Submission control (WAIT_FOR_IDLE, etc.) |
| 0x240–0x24F | Semaphore setup |
| 0x260–0x26F | Physical-mode source/dst |
| 0x300 | LAUNCH_DMA (the go button) |
| 0x400–0x41F | Addresses (OFFSET_IN, OFFSET_OUT, PITCH_IN, PITCH_OUT, LINE_LENGTH_IN) |
| 0x500–0x5FF | 2D/tile/block-linear methods |
| 0x600+ | Advanced features (REMAP, COMPRESS) |

mc uses just a handful (§9 of gpfifo_pushbuffer_reference.md).

### Method space for COMPUTE (CUDA kernel launch)

A compute channel's method space is much larger and more complex, including
methods for:
- Setting up the kernel launch parameters (grid/block dimensions, shared
  memory size).
- Loading the kernel code (cubin → SM instruction cache).
- Configuring register counts, texture bindings, etc.

See `clcXXXX.h` headers for the compute class (e.g., `HOPPER_COMPUTE_A`
= 0xCBC0).

### Backward compatibility

Method spaces are approximately stable across generations. A Turing CE's
method stream will mostly execute on a Hopper CE (with newer feature
fields ignored or set to 0). This is why the Yan et al. paper's A40
(Ampere) pushbuffer traces are meaningful for Hopper comparison.

---

## 16. Faults and preemption

### MMU faults (Xid 31)

When the GPU MMU fails to translate a VA, it raises a fault. The fault
reaches GSP, which decides how to handle it:

- **Fatal fault**: channel is terminated, Xid 31 reported to host.
- **Replayable fault** (with ATS/UVM paging): GSP asks the host to page
  the missing memory in, then retries the access.

mc uses non-replayable MMU — any fault is fatal.

### CE/engine faults (Xid 32 for invalid pushbuffer, Xid 13 for SM exception)

Engine-internal errors:
- **Xid 32**: invalid pushbuffer stream. Caused by malformed method headers,
  unbound subchannel, out-of-range method addresses. We hit this with the
  subchannel-0 bug.
- **Xid 13**: SM exception (e.g., out-of-bounds global memory access from
  a CUDA kernel).
- **Xid 12, 14, etc.**: various engine-specific errors.

### Preemption

Modern GPUs support preemption at fine granularity: SMs can preempt mid-
kernel (instruction-boundary preemption) and CEs can preempt mid-transfer
(method-boundary preemption). This enables fair scheduling across channels.

For mc, we don't care — we're single-tenant on the GPU.

### Recovery

After a fatal fault, the channel is RC'd ("Robust Channel" recovery —
NVIDIA's terminology for clean channel termination). RC'd channels cannot
be resumed; the process must allocate a new channel.

If the fault was in GSP or the host scheduler itself (very rare), the whole
GPU may RC (`gpuRC`), requiring a reset.

---

## 17. Putting it together: a CE transfer in hardware terms

Revisiting the mc D2H transfer with all the hardware vocabulary in
hand:

### Setup phase (one-time)

1. **Allocate a FERMI_VASPACE_A** with `IS_EXTERNALLY_OWNED`.
   *(Creates a GPU MMU page-table tree in HBM.)*

2. **Allocate HBM for the src buffer (d_buf) and the shared `gpu_ctl`
   region (GPFIFO + USERD) via NV01_MEMORY_LOCAL_USER; the release
   semaphore is sysmem (`NV01_MEMORY_SYSTEM`), as libcuda does it.**
   *(RM's memory manager reserves HBM pages.  USERD lives at offset
   0x2000 inside the 2 MiB `gpu_ctl` allocation on Hopper.)*

3. **Allocate host DRAM for dst buffer (h_buf), pushbuffer, and
   staging via NV01_MEMORY_SYSTEM.**
   *(RM pins host pages and registers them with the IOMMU.)*

4. **Register all buffers with UVM.**
   *(UVM installs GPU MMU PTEs at specified CPU VAs. Both HBM and sysmem
   pages are mapped into the VA space.)*

5. **Allocate a TSG** with `engineType` = the first non-GRCE LCE.
   *(Creates a scheduling group for the CE.)*

6. **Allocate a channel (HOPPER_CHANNEL_GPFIFO_A)** parented on the TSG.
   *(Allocates RAMIN, RAMFC, USERD; allocates a CHID; creates an entry in
   the channel table.)*

7. **Allocate a CE engine object (HOPPER_DMA_COPY_A)** parented on the
   channel.
   *(Binds the channel's subchannel 4 to a CE instance.)*

8. **UVM_REGISTER_CHANNEL** and **GPFIFO_SCHEDULE(bEnable=TRUE)**.
   *(Inserts the channel's TSG into that engine's runlist. PBDMA will now see
   this channel when the host scheduler dispatches it.)*

9. **Allocate and map HOPPER_USERMODE_A** for BAR0 VF doorbell access.

### Hot path (per submission)

10. **CPU writes NVC8B5 method stream into pushbuffer**.
    *(WC writes to host DRAM; sfence flushes WC buffers; pushbuffer bytes
    are now resident in DRAM, visible via PCIe.)*

11. **CPU writes GP entry(ies) into GPFIFO ring**.
    *(WC writes through BAR1 to HBM; each write is a PCIe MWr TLP routed to
    HBM; sfence flushes WC buffers; GPFIFO bytes are now in HBM.)*

12. **CPU advances GPPut in USERD**.
    *(WC write through the BAR1 CPU alias of `gpu_ctl` (HBM); sfence.)*

13. **CPU writes work-submit-token to HOPPER_USERMODE_A + 0x90**.
    *(WC write to the BAR1-apertured VF doorbell register; sfence;
    PCIe MWr TLP reaches the GPU's host scheduler.  The underlying
    `NV_VIRTUAL_FUNCTION_DOORBELL` register sits at 0x30090 in the VF
    register block (BAR0 + 0xBB0090 on bare metal), but libcuda and mc
    access it via BAR1 through `HOPPER_USERMODE_A` with
    `bBar1Mapping=NV_TRUE`, at +0x90 into the mapping.)*

14. **Host scheduler receives doorbell**.
    *(Decodes CHID from token, confirms channel is on the runlist,
    notifies a PBDMA engine.)*

15. **PBDMA activates**:
    - Loads RAMFC state (PB_CURRENT, etc.) if channel was dormant.
    - Reads GPPut from USERD (on Hopper, local HBM read since USERD
      is in FBMEM; on pre-Hopper, a PCIe read to host DRAM).
    - Reads GPFIFO entry(ies) from HBM (local read).
    - If extended-base entry present: latches the extended base.
    - Reads pushbuffer from host DRAM (PCIe read; reads typically 64 B
      or 128 B at a time).

16. **PBDMA decodes methods**:
    - Sees SET_OBJECT → binds subchannel 4 to HOPPER_DMA_COPY_A.
    - Sees OFFSET_IN_UPPER/LOWER → routes to CE subchannel 4 → CE latches
      `src_va`.
    - Similarly for OFFSET_OUT, LINE_LENGTH_IN, SET_SEMAPHORE_*, LAUNCH_DMA.

17. **CE executes the transfer**:
    - CE's MMU resolves src_va → HBM physical address.
    - CE's MMU resolves dst_va → host DRAM physical address (via IOMMU).
    - CE's DMA engine issues HBM reads + PCIe writes, pipelining them.
    - Data flows src (HBM) → CE buffers → PCIe writer → PCIe TLPs → host
      memory controller → DRAM.

18. **CE flushes outstanding writes** (because FLUSH_ENABLE=TRUE):
    - CE waits for PCIe write acknowledgments.
    - All bytes are confirmed in host DRAM.

19. **CE writes sema payload** (because SEMAPHORE_TYPE = ONE_WORD_RELEASE):
    - CE issues a 4-byte write to sema_va, which is host DRAM.
    - The posted write crosses PCIe and commits in host memory.

20. **PBDMA completes the method stream**, updates GPGet in USERD, and
    yields to the scheduler.

21. **CPU polls sema_ptr** (a cached sysmem cell). The read is local and
    costs nothing on the bus; it returns the new value and the poll loop
    exits.

22. **CPU reads h_buf**. Data is already in host DRAM (step 18's flush
    snooped CPU caches). The read returns freshly transferred data.

Total: ~134 µs for 4 MiB. 31.3 GB/s effective throughput.

### Hardware units that were active

- **Host interface**: received BAR0 write; routed GPFIFO and pushbuffer
  accesses.
- **Host scheduler**: decoded doorbell; dispatched channel.
- **PBDMA (one instance)**: fetched GPFIFO + pushbuffer; decoded methods;
  routed to CE.
- **CE (one instance of 20)**: actual byte movement.
- **GPU MMU (shared across engines)**: two translations for the DMA
  (src and dst VAs), plus translations for PBDMA's pushbuffer read.
- **L2 cache**: pushbuffer reads and pushbuffer itself pass through L2.
- **HBM controller**: served reads from src.
- **PCIe controller**: carried pushbuffer reads and dst writes.
- **IOMMU**: translated dst physical addresses for PCIe writes.

**Not active**: SMs, all other CEs, NVDEC, NVENC, GR front-end. A 4 MiB D2H
exercises ~1% of the GPU's functional units.

---

## Further reading

- **NVIDIA H100 Whitepaper**: architectural overview, engine counts,
  performance. Public.
- **NVIDIA's "CUDA Hardware Implementation" documentation**: method spaces,
  engine internals. Partially public.
- **`src/common/sdk/nvidia/inc/class/*.h`** in open-gpu-kernel-modules:
  authoritative method-space definitions.
- **`src/common/inc/swref/published/hopper/gh100/*.h`**: per-chip register
  layouts (doorbells, MMU settings, runlist structures). Partially public.
- **`docs/gpfifo_pushbuffer_reference.md`**: the exact formats used on top
  of this hardware model.

*Focus: Hopper H100 with the CE and channel model.  Other architectures
(Ampere, Turing, Blackwell) share most of this but with different class IDs
and engine counts.*
