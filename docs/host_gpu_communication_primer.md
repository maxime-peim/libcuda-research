# Host–GPU Communication: A Primer

**For readers who know operating systems, C, and PCIe basics but have never peered
inside how a CPU actually tells a GPU to do something.**

This document explains the foundational concepts underlying *all* GPU programming,
not just NVIDIA's. It focuses on the **"what and why"** of host-GPU interaction: the
physical communication channels (PCIe BARs, DMA, doorbells), the memory models
(separate address spaces, cache coherency, write-combine memory), and the design
patterns (command queues, asynchronous completion, ring buffers) that these
low-level facts force on every high-level API.

After reading this, `mc_architecture.md` and `gpfifo_pushbuffer_reference.md`
should feel like natural specializations of general principles, not arbitrary
NVIDIA arcana.

---

## Table of Contents

1. [The fundamental fact: GPUs are separate computers](#1-the-fundamental-fact-gpus-are-separate-computers)
2. [The PCIe bus as a communication medium](#2-the-pcie-bus-as-a-communication-medium)
3. [BARs: the memory-mapped window into a device](#3-bars-the-memory-mapped-window-into-a-device)
4. [MMIO: how the CPU pokes the GPU](#4-mmio-how-the-cpu-pokes-the-gpu)
5. [DMA: how the GPU accesses host memory](#5-dma-how-the-gpu-accesses-host-memory)
6. [Doorbells: the minimal-latency wakeup pattern](#6-doorbells-the-minimal-latency-wakeup-pattern)
7. [Command queues: why GPUs don't take instructions one-at-a-time](#7-command-queues-why-gpus-dont-take-instructions-one-at-a-time)
8. [Producer–consumer indices and shared memory](#8-producerconsumer-indices-and-shared-memory)
9. [Semaphores and async completion](#9-semaphores-and-async-completion)
10. [Two MMUs, two address spaces](#10-two-mmus-two-address-spaces)
11. [Cache coherency on x86 with PCIe devices](#11-cache-coherency-on-x86-with-pcie-devices)
12. [Write-Combine memory and the sfence discipline](#12-write-combine-memory-and-the-sfence-discipline)
13. [The IOMMU](#13-the-iommu)
14. [Putting it all together: the canonical submission pattern](#14-putting-it-all-together-the-canonical-submission-pattern)
15. [A concrete walk-through: one CPU → GPU transfer](#15-a-concrete-walk-through-one-cpu--gpu-transfer)
16. [Common misconceptions](#16-common-misconceptions)

---

## 1. The fundamental fact: GPUs are separate computers

A modern discrete GPU (like an H100, A100, RTX 4090) is not an accelerator in the
sense that an FPU coprocessor is an accelerator. It is a **separate computer**
connected to the host via a bus. It has:

- Its own processor cores (tens of thousands of CUDA cores, or hundreds of SMs).
- Its own memory (HBM or GDDR: 80 GB on H100, 24 GB on RTX 4090).
- Its own memory-management unit (a GPU MMU with page tables distinct from the CPU's).
- Its own firmware processor (on recent NVIDIA GPUs, a RISC-V core running GSP firmware).
- Its own clock domains, power management, interrupt sources.

Everything you want the GPU to do has to cross the **PCIe bus** — a serialized
packet-switched interconnect with well-defined latencies (~microseconds) and
bandwidths (~32 GB/s per direction for Gen4 x16, ~63 GB/s for Gen5 x16).
You cannot just
"call a function on the GPU" the way you can call a library function; every
interaction is an I/O operation.

### Why this is counter-intuitive

From a CUDA programmer's perspective:

```c
cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
```

looks like a synchronous function call — you pass pointers, it returns when done.
In reality:

- `dst` is a CPU virtual address (your process's address space).
- `src` is a "CUDA device pointer" — a number that was returned by `cudaMalloc`
  and is meaningful only to the GPU.
- The `cudaMemcpy` call builds a **command packet** describing the transfer,
  writes it into a **queue** shared between CPU and GPU, writes a 4-byte
  MMIO value to *wake the GPU*, then polls a **completion flag** in memory
  until the GPU sets it.

None of the bytes move until the GPU's Copy Engine reads the command, walks
its own MMU to resolve `src` to a physical HBM page, issues PCIe DMA writes to
the host, and flips the completion flag.

All three primitives — command packets, MMIO wakeup, completion polling — are
dictated by the physics of a separate computer over PCIe. They are not
NVIDIA-specific; the same patterns show up in AMD GPUs, NICs, NVMe SSDs, and
FPGAs. What differs is the encoding and the naming.

### The three laws of host-device communication

1. **Nothing crosses the bus unless someone asks.** Every interaction is either a
   CPU-initiated MMIO (small, synchronous, limited in size), a device-initiated
   DMA (large, asynchronous, arbitrary size), or a notification (interrupt or
   polled flag).
2. **Bus transactions have fixed latency overhead.** Each MMIO round-trip is
   ~1 µs even on a modern system; each DMA setup has a similar fixed cost. This
   forces batching.
3. **The two sides have independent memory orderings.** The CPU may have cached a
   write, the device may have buffered a read. Visibility has to be explicitly
   coordinated with fences and cache management.

Every architectural quirk we discuss below is a consequence of these three laws.

---

## 2. The PCIe bus as a communication medium

PCIe (PCI Express) is the bus connecting the CPU's root complex to peripheral
devices. It's packet-switched, full-duplex, and layered (physical → data link →
transaction).

### Topology

```
                    CPU
                     │
                     │ QPI / UPI
                     │
             ┌───────┴────────┐
             │  Root Complex  │
             │  (IOMMU lives  │
             │   here on x86) │
             └───┬──────┬─────┘
                 │      │
     ┌───────────┘      └──────────┐
     │ PCIe Gen5 x16                │ PCIe Gen4 x4
     ▼                              ▼
  ┌─────┐                       ┌──────┐
  │ GPU │                       │ NIC  │
  │(H100)                       │(etc.)│
  └─────┘                       └──────┘
```

The GPU sits on one PCIe link (typically x16 — 16 lanes). Every CPU↔GPU byte
travels these 16 lanes. An H100 PCIe card trains a Gen5 x16 link: 32 GT/s per
lane, 128b/130b encoded, which is about **63 GB/s in each direction**.

PCIe is full-duplex, so those two directions are independent — a host-to-device
copy and a device-to-host copy can run at full speed simultaneously. That also
means a *one-way* copy is bounded by ~63 GB/s, not by the ~126 GB/s figure you
get from adding the directions together. Quoting the aggregate as though a
single transfer could reach it is the most common way to mis-state this.

### Transaction Layer Packets (TLPs)

At the transaction layer, PCIe communicates via **TLPs** — memory read, memory
write, configuration, completion, and message packets. From a software
perspective, you care about three:

- **Memory Write TLP (MWr)**: written by one agent, targeting a physical
  address on the other side. Used for CPU→GPU register writes, and for GPU→CPU
  DMA writes.
- **Memory Read TLP (MRd)**: a request; the target returns the data in a
  **Completion TLP (Cpl)**. Round-trip latency ~500 ns–1 µs.
- **Posted vs. Non-Posted**: MWr is *posted* (fire-and-forget, no completion
  expected). MRd is *non-posted* (the initiator waits for the Cpl).

### What this means for software

- **Writes are cheap** (one TLP each, no round-trip).
- **Reads are expensive** (full round-trip, ~1 µs on top of any bus contention).
- **A sequence of writes to a device register is much faster than alternating
  writes-and-reads.**
- **The GPU reading from host DRAM is the same cost as the CPU reading from
  GPU DRAM** — both directions cross PCIe.

This is why command submission is always structured as "CPU writes a batch of
commands, GPU reads them later," not "CPU and GPU ping-pong."

---

## 3. BARs: the memory-mapped window into a device

When a PCIe device is powered on and enumerated, the OS assigns it a set of
**Base Address Registers** (BARs) — regions of physical address space that,
when read or written by the CPU, are routed as PCIe TLPs to the device.

A GPU typically exposes multiple BARs. The ones that matter:

### BAR0: device registers (small, ~16 MB)

BAR0 is a window onto the GPU's **control registers** — hardware-defined
register addresses that configure the GPU's operation. Writing to BAR0
directly programs hardware: writing `0x1` to some register might set a clock
divider; writing to another might fire an interrupt on the GPU.

On NVIDIA GPUs, BAR0 contains:
- Clock/power management registers.
- Interrupt masks and status.
- **The VF_DOORBELL register** (`0x30090` within the virtual-function
  register block, not a BAR0 offset — see §6): write a channel token here to
  dispatch work on that channel.
- Chip-ID, PMU commands, many other control surfaces.

BAR0 is typically 16 MB — small, because it's meant for registers, not bulk data.

### BAR1: aperture into GPU memory (large)

BAR1 is a window onto the GPU's **memory** (HBM/GDDR). Writing to an address
in BAR1 actually writes a byte of GPU memory; reading returns a byte of GPU
memory. This lets the CPU access GPU memory without the GPU's help.

BAR1 size varies — historically it was 256 MB (small, forcing staged copies
for large transfers), but **Resizable BAR (ReBAR)** allows BAR1 to be as large
as the full GPU memory size (80 GB on H100 with ReBAR enabled). ReBAR is what
enables "the whole GPU is CPU-visible" — a performance feature for gaming and
a correctness feature for some unified-memory scenarios.

### BAR2, BAR3, etc.

Other BARs expose additional functionality (typically PRIV registers, SR-IOV
virtual functions, NVLink management). Not relevant to basic D2H operations.

### What's in a BAR is device-defined

The PCIe spec just says "BARs exist and are assigned address ranges." What
those ranges *mean* is entirely a function of the device. For NVIDIA:
`dev_vm.h` in the open-gpu-kernel-modules source defines BAR0's register map.
For the GPU to interpret a write correctly, the CPU must know and match this
per-GPU register map.

---

## 4. MMIO: how the CPU pokes the GPU

**Memory-Mapped I/O** is the primary mechanism by which the CPU controls the
GPU. The CPU performs a normal load or store instruction to a physical address
that happens to fall inside a PCIe BAR; the chipset recognizes this, converts
the memory operation into a PCIe TLP, and routes it to the device.

From C code, MMIO looks exactly like a memory access:

```c
volatile uint32_t *bar0 = mmap(/* the BAR0 physical region */);
bar0[DOORBELL_OFFSET / 4] = channel_token;   // this is a PCIe Memory Write TLP
```

But the semantics are drastically different from a normal memory access:

### MMIO semantics

| Concern | Normal memory | MMIO |
|---|---|---|
| Cached? | Usually yes (WB) | Usually no (UC or WC) |
| Reordered by CPU? | Per TSO rules | Yes unless fenced |
| Visible to other agents immediately? | After coherency settles | Variable — may sit in WC buffers |
| Read consistency | Load returns last stored value | Read may trigger a device operation; may return zero forever |
| Store consistency | Store is durable in RAM | Store may be lost if device rejects, or may trigger a device action |

### Why MMIO is slow for bulk data

Each MMIO write is one PCIe TLP. A 4-byte store emits a 4-byte write TLP (plus
framing overhead). To move 1 MB via MMIO would be 262,144 separate TLPs, each
paying bus arbitration overhead. Even a Gen5 x16 link would take milliseconds
instead of the microseconds a DMA would take.

**Rule of thumb**: MMIO is for **control** (writes to registers, setting up
state), never for **data** (moving buffers). Data movement is DMA's job.

### Why MMIO reads are even slower

An MMIO read is a PCIe Memory Read TLP — the CPU stalls until the device
responds with a Completion TLP. This round-trip is ~1 µs minimum, during which
the CPU cannot execute dependent instructions. A single MMIO read is ~1000×
slower than an L1 cache hit.

**Rule**: Avoid MMIO reads in hot paths. Use them only for one-shot
setup/query operations.

---

## 5. DMA: how the GPU accesses host memory

**Direct Memory Access** is the inverse of MMIO: the device, not the CPU,
initiates the bus transaction. The GPU issues PCIe TLPs targeting host
physical addresses, and the host's chipset routes them to DRAM.

DMA is how bulk data gets moved. A single DMA engine can sustain near-line-rate
PCIe bandwidth (~63 GB/s per direction for Gen5 x16) by streaming large bursts.

### How the GPU knows where to write

The GPU has its own MMU. When its Copy Engine executes a command like
"copy from address X to address Y," both X and Y are **GPU virtual
addresses**. The GPU's MMU walks page tables (managed by the kernel driver
and UVM) to resolve each VA into a physical address:

- If the physical address is in HBM, the DMA stays on-chip.
- If the physical address is in host DRAM, the DMA goes over PCIe.

The host DRAM physical addresses must have been pre-pinned by the OS so they
can't be paged out during the DMA. This is why CUDA `cudaHostAlloc` exists —
it pre-pins host pages and returns VAs that the GPU MMU can map.

### DMA vs MMIO: who initiates

| | MMIO | DMA |
|---|---|---|
| Initiator | CPU | Device |
| Direction | CPU ↔ device registers or device memory (via BAR) | Device ↔ host DRAM |
| Throughput | ~1 GB/s upper bound | ~55 GB/s measured, ~63 GB/s link ceiling |
| Setup cost | Zero (just a store instruction) | ~µs (command descriptor + doorbell) |
| Granularity | 4, 8, 16, 32, 64 bytes | Arbitrary byte counts |
| Synchronous? | Writes yes (posted); reads yes (blocking) | No — async, requires completion signal |

### Who has mastership?

PCIe allows devices to be **bus masters** — able to initiate transactions
rather than only responding. A GPU is a bus master. This is why DMA works:
the GPU can issue arbitrary memory writes to the host.

This also means the GPU can write anywhere in host DRAM if not restricted.
The **IOMMU** (see §13) exists to sandbox this.

---

## 6. Doorbells: the minimal-latency wakeup pattern

We've established:
- MMIO is fast for small messages but not for data.
- DMA is fast for bulk data but requires setup.
- The GPU has to know when new work is available.

The pattern that emerges naturally: **the CPU prepares a command buffer in
memory, writes a 4-byte "doorbell" MMIO value to the GPU to signal
availability, and the GPU DMAs in the command buffer and executes it.**

A doorbell is just a special register on the device that, when written,
triggers some internal processing. The value written usually identifies
*which* queue has new work.

### Why a doorbell is necessary

A GPU could continuously poll a shared memory location to see if new work
has arrived — but polling consumes power and memory bandwidth. Instead, the
device is *passive* until the CPU explicitly wakes it.

On NVIDIA Hopper, the doorbell is the register `NV_VIRTUAL_FUNCTION_DOORBELL`
(`dev_vm.h`), which sits at offset **`0x30090` inside the GPU's
*virtual-function register block***. Note that is not a BAR0 offset: on bare
metal the block itself is based at `0xB80000`, so the register really lands at
BAR0 + `0xBB0090`. In practice nobody uses either absolute address — userspace
writes **`+0x90`** inside the 64 KiB USERMODE window the driver maps for it,
and that offset holds on bare metal, in a VM, and through either BAR (full
derivation in `gpfifo_pushbuffer_reference.md §11`). Writing a channel's
**work-submit token** (a 32-bit opaque identifier obtained via an RM control
call) to this register tells the GPU's host scheduler "channel X has work;
please run it." The scheduler then dispatches the channel's PBDMA, which
starts reading the GPFIFO.

### Doorbell write is a single TLP

```c
*(volatile uint32_t *)doorbell_va = token;   // one PCIe MWr TLP
```

Single-instruction cost to the CPU. The GPU receives the TLP, decodes the
token, and wakes the corresponding internal hardware in ~hundreds of
nanoseconds.

### Why there are multiple doorbells

A GPU with hundreds of active channels cannot have a single shared doorbell
(all queues would contend). Modern GPUs have large doorbell register banks —
on Hopper, each channel has its own doorbell slot in the VF region. The token
encodes the channel ID.

---

## 7. Command queues: why GPUs don't take instructions one-at-a-time

If every GPU instruction required its own MMIO write + round-trip ACK, GPUs
would be bottlenecked on CPU submission overhead. Instead, GPUs consume work
from **command queues** (called "pushbuffers" on NVIDIA, "command buffers"
generically).

### The command queue pattern

A command queue is a chunk of memory (host or device) that both CPU and GPU
agree on. The CPU writes a sequence of encoded commands (an instruction stream
for the GPU front-end). The GPU reads the stream and executes each command.

An 18-dword (72-byte) method stream that performs a 4 MiB DMA is *far* cheaper
than 18 separate MMIO operations: the CPU writes 72 bytes locally (cached,
coalesced, nanoseconds of cost), then a single doorbell wakes the GPU, which
reads those bytes in one PCIe transaction and executes them.

### Why there's a *ring* structure

A command queue is typically a ring buffer: a fixed-size circular array of
commands (or of pointers-to-commands, which is what NVIDIA's GPFIFO is).
Ring buffers support:

- **Streaming submission**: CPU can keep writing new commands behind the
  GPU's current position.
- **Bounded memory**: the queue doesn't grow unboundedly.
- **Natural backpressure**: if the CPU fills the ring, it has to wait for
  the GPU to consume some.

### On NVIDIA: two-level queuing

NVIDIA splits the command queue into two levels:

- **GPFIFO ring**: a short array of 8-byte pointers, each pointing to a
  pushbuffer region. This is the structure that the producer/consumer
  indices (GPPut/GPGet) track.
- **Pushbuffer**: a larger byte-addressable region holding the actual method
  stream. Each GPFIFO entry says "the next command is N dwords at
  pb_va + offset."

This split lets short control sequences share pushbuffer storage (one
pushbuffer can hold many method streams referenced by many GPFIFO entries).

### Analogy: CPU branch-target buffer

Think of the GPFIFO like a queue of "jump targets" — each entry points to a
block of commands. The CPU can compose large command sequences out of reusable
pushbuffer fragments, and the GPU processes them in FIFO order.

---

## 8. Producer–consumer indices and shared memory

A ring buffer requires synchronization between producer (CPU) and consumer
(GPU). The simplest synchronization: two integer indices in shared memory.

### GPPut (producer index) and GPGet (consumer index)

- **GPPut**: written by CPU, read by GPU. Indicates "I have produced up to
  here, not including this slot."
- **GPGet**: written by GPU, read by CPU. Indicates "I have consumed up to
  here, not including this slot."

Ring is:
- **Empty** when GPPut == GPGet
- **Full** when GPPut + 1 == GPGet (mod ring_size)
- **Non-empty** otherwise; valid entries are at indices [GPGet, GPPut)

### Where do the indices live?

The indices have to be *shared* memory — both sides read and write them.
NVIDIA places them in **USERD**, a small per-channel memory region.  The
physical home of USERD has varied by GPU generation: on Ampere and older
GPUs it was often placed in host DRAM so the CPU writes are local and
the GPU pays a PCIe read.  On Hopper (H100), the driver places USERD
in **FBMEM** (HBM): the GPU reads locally, and the CPU writes through
a BAR1 CPU alias (write-combining).  This note uses the Hopper/FBMEM
layout throughout because that's what mc and libcuda actually do
on H100 — but the host-DRAM variant is still a valid mental model for
older hardware.

```
USERD (Hopper: FBMEM, 512 B per channel slot; pre-Hopper: host DRAM):
  offset 0x88: GPGet  ← GPU writes, CPU reads
  offset 0x8c: GPPut  ← CPU writes, GPU reads
```

### Why this works without atomic instructions

Because there is exactly one writer for each index, and each index is a single
32-bit word, atomic RMW operations are not needed. The only requirement is
that updates are *visible* to the other side — which means appropriate memory
fencing and avoiding caching on the CPU side.

### The `volatile` qualifier

In C, accesses to USERD must use `volatile` to force the compiler to emit
loads/stores rather than caching values in registers:

```c
volatile HopperAControlGPFifo *userd = ...;

userd->GPPut = new_index;       // actual store is emitted
uint32_t seen = userd->GPGet;   // actual load is emitted every call
```

Without `volatile`, a loop reading `userd->GPGet` to wait for the GPU to
advance would be optimized to a load-once-and-spin — an infinite loop.

### Interaction with caching

On x86, USERD is typically mapped as **Write-Combine (WC)** memory (see §12).
This means CPU writes are buffered but reads are uncached (each load is a
bus transaction). The WC buffer drains on `sfence` or when it fills.

---

## 9. Semaphores and async completion

A DMA is *asynchronous* — after the CPU writes the doorbell, the GPU processes
the command on its own timeline. How does the CPU know when it's done?

### Option 1: Interrupts

The GPU can raise a PCIe interrupt on completion. The OS's interrupt handler
runs, wakes a thread, which then knows the transfer is complete. This has:

- Fixed overhead (~10–50 µs for interrupt dispatch + thread wakeup).
- CPU efficiency (core is free during the wait).
- Complexity (interrupt plumbing, affinity, etc.).

Interrupts are used by CUDA internally for `cudaStreamSynchronize` when blocking.

### Option 2: Polling a memory flag (semaphore)

Alternatively, the CPU tells the GPU "when you're done, please write the value
1 to memory address X." The CPU then spins reading address X until it sees 1.

This has:

- Near-zero latency detection (~1 µs — a single memory read).
- Higher CPU overhead (busy-waits).
- No OS complexity.

NVIDIA's CE `LAUNCH_DMA` command takes `SET_SEMAPHORE_A/B/PAYLOAD` parameters
for exactly this pattern: the CE writes the 4-byte payload to the semaphore
VA when the transfer is complete.

### Why both exist

- Short, latency-sensitive transfers: poll the semaphore (mc does this).
- Long transfers or when CPU has useful work: use interrupts.
- CUDA exposes both via the Events API: `cudaEventQuery` is polling,
  `cudaEventSynchronize` can block on an interrupt.

### The FLUSH_ENABLE flag

One subtle detail: **the CE must not write the semaphore until it has
confirmed that all DMA writes have committed to the destination**. Otherwise,
the CPU could see the semaphore flip and then read stale destination data
(the writes might still be in PCIe transit).

`LAUNCH_DMA_FLUSH_ENABLE = TRUE` instructs the CE to flush all outstanding
PCIe writes before writing the semaphore. Setting this is *mandatory* for
correctness on D2H transfers.

### Where the semaphore physically lives

The semaphore must be accessible to both GPU (to write it) and CPU (to poll
it). Options:

- **In HBM**: GPU writes locally, CPU reads over PCIe (BAR1 alias). Writing
  is fast, CPU polling has PCIe latency per check (~1 µs).
- **In host DRAM**: GPU writes over PCIe, CPU reads from cache. Writing is
  slower (one PCIe write), polling is free after the cache line arrives.
  This is what mc uses, and what libcuda does.

For rapid submit–wait cycles, host DRAM is usually preferred. For occasional
waits, HBM is fine. CUDA chooses based on access patterns.

---

## 10. Two MMUs, two address spaces

This is the concept that trips up people coming from a CPU-only background.
**The GPU has its own MMU that walks its own page tables**, entirely separate
from the CPU's MMU.

### Why it's separate

The GPU needs to translate VAs to physical addresses at DMA time. The CPU's
MMU is *not* involved because the DMA doesn't go through the CPU. The GPU
needs its own translation machinery.

### Identity vs non-identity mapping

If CPU VAs and GPU VAs were unrelated, the driver would have to translate
every pointer the CPU emits. This is what happens in older CUDA models (the
"CUDA device pointer" is a number in a completely separate namespace).

**UVM (Unified Virtual Memory)** changes this: UVM installs GPU MMU page
tables such that a GPU VA *equals* the CPU VA. When the CPU writes
`src_va = 0x76baa0800000` into a command stream, the GPU's MMU walks its
tables at that same numeric address and finds the correct physical HBM (or
host DRAM) page.

This dramatically simplifies the programming model: you can dereference the
same pointer on both sides.

### The cost of UVM

UVM's identity mapping isn't free:
- Page tables have to be kept in sync (if the CPU remaps a page, the GPU MMU
  must see the new mapping).
- Page faults on one side have to be handled (if GPU accesses a page that
  isn't mapped yet, a fault propagates back to the kernel).
- There's kernel overhead to manage the coherency.

For pure host-GPU DMA (no on-demand paging), UVM just sets up the mappings
once at buffer-registration time and never touches them again. This is what
`UVM_MAP_EXTERNAL_ALLOCATION` does: it installs GPU PTEs at a CPU-reserved
VA range, and from then on the GPU can access the buffer at that VA without
further kernel involvement.

### GPU MMU page size

The GPU MMU typically supports page sizes of 4 KiB, 64 KiB, 128 KiB, and 2 MiB
(varies by generation). On Hopper, FERMI_VASPACE_A defaults to 64 KiB big
pages. This is why RM alignments are often 64 KiB (`mp.alignment = 0x10000`
in mc).

### Mismatches and implications

Because the two MMUs are independent:
- **Page faults are independent.** A CPU page fault doesn't bother the GPU
  directly.
- **Permissions are independent.** A page could be writable from CPU, read-
  only from GPU.
- **Protection is strong.** A GPU fault on an unmapped VA produces a specific
  Xid (31 on Hopper) that the driver reports to the kernel.

### Device memory vs host memory, from the GPU's view

From the GPU MMU's perspective, every page is just a physical address with
an "aperture" (HBM / host DRAM / peer GPU). The MMU has bits in each PTE
distinguishing these so the GPU knows whether to route the access internally
or over PCIe.

---

## 11. Cache coherency on x86 with PCIe devices

x86 CPUs maintain **cache coherency** among themselves via MESI-like
protocols. But PCIe devices are **not** in the coherency domain on most
current systems (there are exceptions — see CXL below). This means:

### Device-initiated writes to host DRAM

When the GPU DMA-writes to host DRAM:
- The write is a PCIe Memory Write TLP to the physical address.
- The root complex delivers it to the memory controller.
- **The CPU's caches are invalidated** as part of the write (on x86, PCIe
  writes snoop the CPU caches, so a subsequent CPU read sees the new value).

This is called **"PCIe coherency"** or **"I/O coherency."** It's only
partial: the GPU's view doesn't participate in cache coherency. The GPU has
its own caches (L1, L2), which are completely separate from the CPU's.

### CPU-initiated writes to host DRAM

When the CPU writes to host DRAM:
- The write lands in the CPU's write buffer / L1, eventually the L2/L3.
- **The write is NOT automatically visible to the GPU's caches.** The GPU
  doesn't snoop the CPU's caches.
- For the GPU to see the CPU's write, either the CPU must evict the cache
  line, OR the memory must be mapped as **uncached** or **write-combining**
  so there's no CPU cache involvement.

This is why the USERD page, GPFIFO ring BAR1 alias, and doorbell BAR0 region
are all mapped as Write-Combine: the CPU never caches these pages, so writes
bypass caches and go directly to the memory (for WC it's actually to a WC
buffer first, then drained).

### Cache-coherent PCIe (CXL, CC-IX)

Newer interconnects (CXL, CC-IX, NVLink-C2C) *do* put the device in the
coherency domain. On those systems, a CPU write is immediately visible to
the device without fencing. But on conventional x86-PCIe, it's not.

This affects mc nowhere (we're using WC mappings everywhere critical)
but is important context if you're reading about newer CXL-based GPUs.

### What happens for host DMA buffers (`h_buf`)

For mc's `h_buf` (the destination of the D2H copy):
- The CPU allocated it via `NV01_MEMORY_SYSTEM` and the kernel pinned the
  host pages.
- The GPU DMA-writes data into these pages.
- The PCIe write invalidates any CPU cache lines covering those pages.
- When the CPU reads from `h_buf` after the semaphore flips, it pulls fresh
  data from DRAM.

This "works" because PCIe writes snoop CPU caches. **If you were reading the
destination buffer from CPU *during* the DMA, you'd see partial state** —
some cache lines updated, others not. This is why the semaphore and
FLUSH_ENABLE are critical: you must not read the destination until the GPU
guarantees all writes have committed.

---

## 12. Write-Combine memory and the sfence discipline

Write-Combine (WC) is an x86 memory caching mode designed for streaming
writes to device memory (originally intended for framebuffers). It's what
makes CPU→GPU MMIO fast.

### How WC works

- CPU writes to a WC-mapped address are buffered in a **WC buffer** in the
  core (typically one 64-byte buffer per core).
- Writes to adjacent addresses within a 64-byte line coalesce into a single
  buffer entry.
- When the buffer fills, or when a serializing operation executes, the buffer
  is flushed to the bus as a single TLP (or burst).

This means 16 consecutive 4-byte writes to a WC region can become **one**
64-byte PCIe TLP instead of 16 separate 4-byte TLPs — a ~16× efficiency
improvement.

### Why reads from WC are weird

WC buffers are write-only. A load from a WC address *bypasses* the buffer and
reads from memory directly. Depending on the device, this may return zeros,
or stale values, or the actual value — it's implementation-defined.

**Rule**: don't read back from WC memory expecting to see your recent writes.
If you need to verify a write landed, use a read from a different memory type
(a cell in HBM or in cached sysmem, say) or trust the fencing discipline.

### When WC buffers drain

The WC buffer is flushed on:
- **Any serializing instruction**: `sfence`, `mfence`, `lfence` (sfence only
  fences stores, which is what you want here), CPUID, certain syscalls.
- **Buffer full**: 64 bytes written to adjacent addresses.
- **Cache line eviction** / implicit for some operations.

In practice, you should not rely on "natural" draining — always place an
explicit `sfence` before any operation that requires prior WC writes to be
visible.

### The fencing contract for mc

```c
// 1. Write method stream to pushbuffer (WC host DRAM)
*pb++ = header;  ...  *pb++ = payload;
_mm_sfence();   // flush pushbuffer writes

// 2. Write GP entry to GPFIFO ring (WC BAR1)
gpfifo_ring[idx*2 + 0] = entry0;
gpfifo_ring[idx*2 + 1] = entry1;
_mm_sfence();   // flush GP entry writes

// 3. Write GPPut to USERD (WC BAR1 alias → HBM on Hopper)
userd->GPPut = new_gp_put;
_mm_sfence();   // flush GPPut

// 4. Write doorbell to BAR1 VF-doorbell register (WC MMIO)
*vf_doorbell = token;
_mm_sfence();   // flush doorbell write
```

Each `sfence` ensures that by the time the next operation starts, the prior
operation's WC buffer has been flushed to the bus, so the GPU will observe
writes in the correct causal order.

### What goes wrong without sfence

If you omit the fence between (2) and (3), the CPU might drain the GPPut
write *before* the GP entry write. PBDMA sees GPPut=N+1, reads GPFIFO slot N
(which is stale or garbage), and produces Xid 32 "invalid pushbuffer stream."

The failure is **intermittent** because WC buffers often happen to drain in
program order — but not reliably.

### Compiler barriers

`_mm_sfence()` is a compiler intrinsic from `<emmintrin.h>`: it emits the
single x86 `sfence` instruction *and* tells GCC/Clang not to reorder
reads/writes across it. Without that compiler-side ordering, the generated
instruction stream might schedule USERD writes before GPFIFO writes even
though they appear in source order. So one call gives you both a CPU-level
fence (the sfence µop drains the WC buffer) and a compiler-level fence
(the intrinsic's implicit memory clobber).

The equivalent inline asm — `__asm__ volatile("sfence" ::: "memory")` —
makes the two halves explicit (the literal `sfence` instruction plus the
`"memory"` clobber). Either form is fine; the intrinsic is shorter and
matches what the codebase actually uses.

---

## 13. The IOMMU

The **IOMMU** (Input-Output Memory Management Unit) is a hardware unit in
the PCIe root complex that intercepts and translates DMA addresses from
devices before they reach DRAM.

### What the IOMMU does

Without an IOMMU:
- The GPU is a bus master; it can DMA-write to any host physical address.
- A compromised GPU or driver could DMA-read `/etc/shadow` out of the kernel.
- Virtualization is impossible — the GPU doesn't know about guest address
  spaces.

With an IOMMU:
- Each device has its own page table (an **IOVA** space — I/O Virtual
  Address).
- The device issues accesses using IOVAs; the IOMMU translates to physical
  addresses via the device's page table.
- The kernel controls the page table, so it can restrict which physical
  pages the device can reach.
- For virtualization, guest physical addresses are translated by the IOMMU
  to host physical addresses — the device can DMA directly into guest memory.

### IOMMU modes

- **Passthrough (pt) mode**: IOMMU just does 1:1 translation (IOVA == PA).
  Near-zero overhead, no protection. Linux boot option: `iommu=pt`.
- **Translation mode**: full IOVA translation. Some overhead per-TLP (IOTLB
  lookup). Required for virtualization.
- **Disabled**: IOMMU turned off. Linux boot option: `iommu=off`. Dangerous.

For GPU compute, pt mode is typically best (performance). For
virtualization/containerization scenarios, translation mode is required.

### Why it matters for mc

- **Performance**: IOMMU translation adds latency per TLP. Measurable in
  small-transfer benchmarks (~10% slowdown).
- **Correctness**: if IOMMU is mis-configured, DMA can silently fail (GPU
  writes go nowhere, or to the wrong addresses).
- **Pinning**: host pages destined for GPU DMA must be pinned AND registered
  in the IOMMU page table. This is what `NV01_MEMORY_SYSTEM_OS_DESCRIPTOR`
  (classic path) or UVM (modern path) does.

### SMMU (ARM equivalent)

On ARM systems (e.g., Grace-Hopper), the IOMMU is called SMMU but does the
same thing. The interfaces differ but the concepts are identical.

---

## 14. Putting it all together: the canonical submission pattern

Every modern GPU (NVIDIA, AMD, Intel) and accelerator (NICs, FPGAs) uses
variants of the same pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                         SETUP (once per context)                │
├─────────────────────────────────────────────────────────────────┤
│ 1. Kernel driver allocates:                                     │
│    - a command queue (ring buffer) in shared memory             │
│    - producer and consumer indices in shared memory             │
│    - a doorbell register mapping (MMIO)                         │
│    - per-buffer GPU MMU page tables                              │
│ 2. Userspace mmaps:                                              │
│    - the command queue (write access)                            │
│    - the indices (read/write)                                    │
│    - the doorbell (write-only MMIO)                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     HOT PATH (per submission)                   │
├─────────────────────────────────────────────────────────────────┤
│ 1. Userspace writes an encoded command into the queue           │
│    (WC host memory write; sfence)                                │
│ 2. Userspace advances the producer index in shared memory       │
│    (WC write; sfence)                                            │
│ 3. Userspace writes the doorbell MMIO register                   │
│    (single MWr TLP; sfence)                                      │
│ 4. Device's front-end wakes, reads the producer index,           │
│    DMAs the command from the queue                               │
│ 5. Device decodes and dispatches to execution units              │
│ 6. Execution completes; device writes a completion semaphore     │
│    (or raises an interrupt)                                      │
│ 7. Userspace polls the semaphore (or sleeps on the interrupt)    │
│    until completion is signaled                                  │
└─────────────────────────────────────────────────────────────────┘
```

This is the model from kernel drivers to CUDA to Vulkan to NVMe. The naming
differs (NVIDIA: GPFIFO/pushbuffer/method stream; AMD: command buffer/PM4
packets; NVMe: submission/completion queues; Vulkan: VkCommandBuffer/
VkQueueSubmit) but the structure is identical.

### Why this pattern is optimal

- **Amortizes MMIO cost**: one doorbell write per *batch* of commands, not
  per command.
- **Exploits DMA bandwidth**: commands and data both travel via DMA-friendly
  routes (large TLPs, not MMIO).
- **Decouples CPU and GPU timing**: CPU can pipeline many submissions ahead;
  GPU consumes at its own rate.
- **Scales to many queues**: each context has its own queue, indices, and
  doorbell; contention is only at the bus level.

### The limits

- **Latency is bounded below by bus round-trips**: each submission is ~µs
  minimum.
- **Short submissions don't amortize setup**: if you want to submit
  100 ns of work per call, this model is wasteful.
- **Correctness requires careful memory ordering**: WC fencing, semaphore
  flush-before-release, etc.

---

## 15. A concrete walk-through: one CPU → GPU transfer

Let's trace a single `mc_demo` D2H transfer through every concept above.
This is what happens after setup is done:

### t = 0 µs — CPU begins submission

The CPU has already:
- Allocated the pushbuffer (host DRAM, WC-mapped, MAP_FIXED into the
  VA pool at 0x200000000+).
- Allocated `gpu_ctl`: a 2 MiB HBM region holding GPFIFO at offset 0
  and USERD at offset 0x2000, with a BAR1 CPU alias MAP_FIXED into
  the same VA pool.
- Allocated HOPPER_USERMODE_A (BAR1 variant) for the VF doorbell
  (+0x90 of its 64 KiB mapping; MMIO, WC).
- Registered all buffers with UVM so the GPU MMU can find them.

The CPU writes the 72-byte NVC8B5 method stream into the pushbuffer:

```c
pb[0]  = INCR_HEADER(SET_OBJECT, 1);     pb[1]  = 0xC8B5;
pb[2]  = INCR_HEADER(OFFSET_IN_UPPER, 2); pb[3]  = src_hi; pb[4]  = src_lo;
...
pb[10] = INCR_HEADER(LAUNCH_DMA, 1);     pb[11] = 0x182;   /* the copy   */
...
pb[16] = INCR_HEADER(LAUNCH_DMA, 1);     pb[17] = 0x00c;   /* the release */
```

Each write is a CPU store to a WC buffer. Writes coalesce within 64-byte
regions. Total CPU work: ~50 ns.

### t = 50 ns — sfence #1

```c
_mm_sfence();
```

Flushes the pushbuffer WC buffer to host DRAM. The 72 bytes now live in DRAM
(though not yet visible to the GPU — it hasn't looked).

### t = 60 ns — write GPFIFO entry

Computes entry0 and entry1 from pb_va (with extended-base if needed) and
writes to `gpfifo_ring[idx*2..idx*2+1]`. These are writes to an HBM region
mapped into CPU address space via BAR1 — each write generates a PCIe MWr TLP
targeting BAR1 of the GPU, which the GPU's memory controller routes to HBM.

### t = 110 ns — sfence #2

Flushes any pending GP entry TLPs.

### t = 120 ns — advance GPPut

```c
userd->GPPut = new_gp_put;
```

One WC write to the USERD page (HBM on Hopper, via the BAR1 alias
of `h_gpu_ctl_mem`).

### t = 130 ns — sfence #3

Flushes GPPut to DRAM.

### t = 140 ns — ring the doorbell

```c
*vf_doorbell = work_submit_token;
```

A 4-byte MMIO write to the VF doorbell (`+0x90` inside the USERMODE
window). This emits a PCIe MWr TLP. The TLP
travels across the x16 link to the GPU's host interface in ~200 ns.

### t = 350 ns — sfence #4, CPU begins polling

```c
_mm_sfence();
while (*sema_ptr != 1) { /* spin */ }
```

Each read of `*sema_ptr` is a local read of a cached sysmem cell, so the spin
costs nothing on the bus. That is why the semaphore is in host memory rather
than HBM: from an HBM cell every spin would be a PCIe MRd of roughly 1 µs,
competing with the transfer it is waiting on. The CPU spins until the GPU
writes the sema.

### t = ~350 ns — GPU host scheduler receives doorbell

The GPU's host scheduler sees the MMIO write (at the BAR1-apertured
VF doorbell, `+0x90` into the USERMODE window), decodes the
token, identifies channel X, consults the runlist, and schedules
channel X onto a PBDMA engine.

### t = ~1 µs — PBDMA starts fetching

PBDMA reads `USERD.GPPut` (local HBM read on Hopper since USERD is
in FBMEM; a few hundred ns).  Sees GPPut > GPGet.  Reads the GPFIFO
ring entry at GPGet (local HBM read; GPFIFO lives at offset 0 inside
the same FBMEM `gpu_ctl` region as USERD).  Decodes: pb_va = X,
length = 18 dwords.

### t = ~2 µs — PBDMA reads pushbuffer

PBDMA issues a 72-byte PCIe MRd to pb_va (the GPU's MMU translates pb_va to
the host DRAM physical address). The host's memory controller returns the
cache-line to PBDMA. That round trip is latency-bound, not bandwidth-bound:
about 200 ns, essentially independent of link generation.

### t = ~2.5 µs — PBDMA decodes methods

PBDMA reads:
- dword 0: INCR header, subch 4, method 0x0 (SET_OBJECT), count 1.
- dword 1: 0xC8B5 → bind CE object to subchannel 4.
- dword 2: INCR header, method 0x400 (OFFSET_IN_UPPER), count 2.
- dword 3: src_va[56:32] → CE register write.
- dword 4: src_va[31:0] → CE register write.
- ...
- dword 10: INCR header, method 0x300 (LAUNCH_DMA), count 1.
- dword 11: 0x182 → the copy launch (no flush, no semaphore).
- dwords 12-15: SET_SEMAPHORE_A/B/PAYLOAD.
- dword 16: INCR header, method 0x300 (LAUNCH_DMA), count 1.
- dword 17: 0x00c → the release launch; moves no data.

Each method is delivered to the CE's register interface on subchannel 4.

### t = ~3 µs — CE begins DMA

The CE has its state latched (OFFSET_IN, OFFSET_OUT, LINE_LENGTH_IN,
SET_SEMAPHORE_A/B/PAYLOAD). LAUNCH_DMA fires the engine:

- CE issues reads from src_va (HBM) at full HBM bandwidth (~2 TB/s on the
  H100 PCIe SKU).
- CE issues writes to dst_va (host DRAM via PCIe) — this is the
  bottleneck; measured ~55 GB/s at 256 MiB at boost clocks (the CE is
  clocked with the SM domain, so an idle-clocked GPU moves data far
  slower — `findings.md §15`).
- The DMA pipelines reads and writes; read rate is throttled to match
  write rate.

### t = ~3 µs to ~122 µs — data transfer in flight

mc measures 31.3 GB/s wall-clock for a 4 MiB copy (`findings.md §11`), so the
whole call is 4 MiB / 31.3 GB/s ≈ 134 µs end to end.  Subtract the fixed
costs itemised below and roughly 130 µs of that is data actually moving.

During this time, the CE continuously issues read TLPs to HBM and write TLPs
to host DRAM. The CPU is spinning on the semaphore, which is a cached
sysmem cell — each read is local and costs nothing on the bus, so the poll
loop adds no PCIe traffic to compete with the transfer.

### t = ~122 µs — CE completes data, begins flush

`FLUSH_ENABLE=TRUE` causes the CE to wait for all outstanding PCIe writes to
commit. This adds a few microseconds for the last TLPs to flush through the
PCIe pipeline and for the root complex to acknowledge.

### t = ~132 µs — CE writes semaphore

The CE issues a 4-byte write to sema_va, which is host DRAM. The write
crosses PCIe as a posted write; from a single Requester ID, PCIe ordering
keeps it behind the data it is signalling.

### t = ~132 µs — CPU observes semaphore flip

The CPU's next poll reads the updated sema value.  The semaphore is a sysmem
cell, so the read is serviced from host DRAM — the CE's write invalidated the
line, and nothing crosses PCIe in the CPU's direction.  `*sema_ptr` now returns
1, the while loop exits.

### t = ~134 µs — CPU reads dst buffer

At this point, all data has been written to host DRAM and all PCIe writes
have snooped the CPU caches. The CPU can safely read `h_buf` and see the
correct data.

### Total elapsed: ~134 µs

Breakdown (only the ~134 µs end-to-end and the 31.3 GB/s it implies are
measured; the split below is an estimate):
- CPU submission: 350 ns (0.3%)
- GPU wake-up and fetch: ~2 µs (1.5%)
- Data transfer: ~119 µs (~89%)
- Flush, semaphore and completion: ~12 µs (~9%)
- CPU read-back start: free (already flushed)

At 4 MiB the few microseconds of fixed overhead are a visible slice; by
64 MiB they are around 1 % and the transfer is what is left.

---

## 16. Common misconceptions

### "cudaMemcpy is a function call that copies memory"

It's a *command submission* to a remote engine, followed by a *completion
wait*. The copy happens asynchronously on the GPU over PCIe.

### "The GPU is like a fast CPU, but for parallel work"

The GPU is an entire system — separate memory, separate MMU, separate clock
domains. You talk to it the way you'd talk to a remote server, except the
"network" is PCIe with ~µs latencies.

### "If the CPU writes to memory, the GPU sees it immediately"

Only if the memory is in a coherency domain that the GPU participates in
(unusual on x86 without CXL). Otherwise you need explicit fencing, and
often the memory has to be allocated with specific caching attributes (WC,
pinned) to work at all.

### "DMA is for large data and MMIO is for small data, and that's the only difference"

Larger issue: MMIO is for **commands**, DMA is for **data flow**. You never
use MMIO to transfer a buffer, even a small one — the design isn't about
size, it's about initiation direction (CPU→device vs device-initiated).

### "Pointers are pointers"

Under UVM, CPU VAs equal GPU VAs — but the pages they map to can be entirely
different (GPU VA X might map to HBM while CPU VA X maps to host DRAM, even
though the number is the same). The coincidence is by careful construction,
not by the architecture.

### "The GPU 'has' a pushbuffer the way a CPU has a stack"

The pushbuffer is a *shared data structure* the CPU writes and the GPU reads.
The GPU doesn't own it; it dips into it on demand. The GPU's own internal
execution state is entirely separate.

### "Doorbells are interrupts"

An interrupt is device → CPU. A doorbell is CPU → device. They're opposite
signals, used for opposite purposes (CPU waking device vs. device waking CPU).

### "Semaphores in GPU DMA are like POSIX semaphores"

No — they're just 4-byte memory locations polled by the CPU. No counting,
no blocking primitive, no kernel involvement. The name is suggestive but the
mechanism is just "write a value to an address and the other side reads it."

### "The kernel driver does the DMA"

On modern NVIDIA GPUs (and most accelerators), the kernel driver does almost
*nothing* on the data path. Setup is kernel; each individual transfer is
entirely userspace (pushbuffer write, doorbell write, semaphore poll). The
kernel's job is to map the necessary memory into userspace and to manage
page tables; the transfer itself bypasses it.

---

## Further reading

- **Section 6 of PCIe Base Specification** — TLP formats, memory ordering.
- **Intel SDM Vol. 3, Chapter 11** — Memory types on x86 (UC, WC, WB, etc.).
- **Linux kernel `Documentation/PCI/`** — how Linux sets up PCIe devices.
- **Linux kernel `Documentation/DMA-API.txt`** — the DMA API for drivers.
- **`linux-source/drivers/iommu/`** — IOMMU driver code.
- For NVIDIA specifics: `docs/nvidia_software_stack.md` and `docs/gpu_compute_model.md`.

*Target platforms: x86-64 Linux with a PCIe Gen4-or-newer discrete GPU.
Some details (coherency, IOMMU) vary on ARM, Power, or systems with
CXL/NVLink-C2C.*
