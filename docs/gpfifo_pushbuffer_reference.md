# GPFIFO, Pushbuffer, and Component Interaction Reference

**A detailed description of the data-structure formats and the choreography of
components that make a GPU DMA submission work on Hopper (H100).**

This document describes the *formats on the wire* — every byte the CPU writes
and every byte the GPU reads — and the *runtime choreography* — who reads what
when, what wakes whom, and how completion signals propagate. It complements
`mc_architecture.md` (which is more about the ioctl/setup layering) and
`findings.md` (which is the running research log).

All definitions are pulled from the header files shipped in
open-gpu-kernel-modules 610.43.02 (Hopper GH100 / H100 PCIe):

- `src/common/sdk/nvidia/inc/class/clc86f.h` — HOPPER_CHANNEL_GPFIFO_A (channel, USERD, GP entry)
- `src/common/sdk/nvidia/inc/class/clc8b5.h` — HOPPER_DMA_COPY_A (CE methods)
- `src/common/sdk/nvidia/inc/class/cla06fsubch.h` — subchannel assignments

---

## Table of Contents

1. [The cast of characters](#1-the-cast-of-characters)
2. [Memory types and locations](#2-memory-types-and-locations)
3. [The GPFIFO ring](#3-the-gpfifo-ring)
4. [GP entry format — bit-exact](#4-gp-entry-format--bit-exact)
5. [Extended-base GP entry](#5-extended-base-gp-entry)
6. [The pushbuffer and method stream](#6-the-pushbuffer-and-method-stream)
7. [Method header format — bit-exact](#7-method-header-format--bit-exact)
8. [Subchannels and the SET_OBJECT method](#8-subchannels-and-the-set_object-method)
9. [The NVC8B5 Copy Engine method set](#9-the-nvc8b5-copy-engine-method-set)
10. [USERD — the user-accessible doorbell page](#10-userd--the-user-accessible-doorbell-page)
11. [BAR0 VF doorbell register](#11-bar0-vf-doorbell-register)
12. [Semaphore format](#12-semaphore-format)
13. [Component interaction — end-to-end timeline](#13-component-interaction--end-to-end-timeline)
14. [Memory ordering and fence discipline](#14-memory-ordering-and-fence-discipline)
15. [What each component reads vs writes](#15-what-each-component-reads-vs-writes)

---

## 1. The cast of characters

For a single D2H DMA transfer, five software-visible data structures are in play.
Here are the sizes and physical homes we use in `mc`:

| Structure | Size | Physical home | Who writes | Who reads |
|---|---|---|---|---|
| GPFIFO ring | 4 KiB (512 × 8 B entries), at offset 0 of `gpu_ctl` | HBM (vidmem, part of a 2 MiB `NV01_MEMORY_LOCAL_USER` allocation shared with USERD) | CPU (via BAR1 alias) | GPU PBDMA (locally) |
| Pushbuffer | 1 MiB (`PB_SIZE`; bumped from 64 KiB so the kernel sysmem_track admits it at the 256-page threshold) | host DRAM (sysmem, WC) | CPU (directly, mapped MAP_FIXED into the VA pool at 0x200000000+) | GPU PBDMA (over PCIe) |
| USERD page | 4 KiB at offset `0x2000` of `gpu_ctl` | HBM (same 2 MiB `NV01_MEMORY_LOCAL_USER` allocation as GPFIFO; accessed by CPU via BAR1 alias) | CPU writes GPPut; GPU writes GPGet | CPU reads GPGet; PBDMA reads GPPut |
| Semaphore word | 4 B (within a 4 KiB sysmem alloc) | host DRAM (sysmem, cached — where libcuda puts it) | GPU Copy Engine (over PCIe) | CPU (plain cached pointer, MAP_FIXED into the VA pool) |
| VF doorbell | 4 B MMIO register at 0x30090 in the VF register block (= BAR0 + 0xBB0090 on bare metal; see §11).  Userspace reaches it via `HOPPER_USERMODE_A`, whose 64 KiB window can be carved from BAR0 **or** BAR1; libcuda and mc use the BAR1 alias on Hopper (`bBar1Mapping=NV_TRUE`).  Always at offset **+0x90** of the mapping, whichever BAR. | GPU VF register, BAR1-mapped in practice | CPU (MMIO) | GPU host scheduler |

And three active processing components on the GPU side:

| Component | Role |
|---|---|
| **PBDMA** (pushbuffer DMA) | Per-channel front-end. Reads GPFIFO entries, fetches pushbuffer bytes, decodes method headers, routes methods to the correct subchannel. |
| **Host scheduler** | Global front-end. Listens for VF doorbell writes and dispatches the corresponding PBDMA channel from the runlist. |
| **Copy Engine (CE)** | The DMA execution unit. Receives decoded methods from PBDMA on subchannel 4, programs internal state registers, executes memory transfers, writes the semaphore on completion. |

On Hopper, PBDMA and CE are separate hardware blocks connected by an internal
fabric. PBDMA handles all "work delivery" concerns (fetching, parsing,
sequencing). CE handles the actual data movement.

---

## 2. Memory types and locations

GPU memory accesses flow through several address spaces. Keeping these straight
is essential.

```
┌─────────────────────────────────────────────────────────────┐
│ CPU virtual address (process VA)                            │
│   After UVM_MAP_EXTERNAL_ALLOCATION: sysmem buffers have    │
│   direct CPU access; vidmem buffers are PROT_NONE at this   │
│   VA — CPU must use a separate BAR1 alias (see below).      │
└─────────────────────────────────────────────────────────────┘
         │                              │
         │ UVM identity mapping         │ separate rm_map_memory
         │ (same address on both sides) │ for vidmem CPU access
         ▼                              ▼
┌───────────────────────┐        ┌──────────────────────────┐
│ GPU virtual address   │        │ BAR1 CPU alias           │
│ (FERMI_VASPACE_A)     │        │ VA pool slot (0x2xxx…)   │
│   This is what gets   │        │   CPU-side window into   │
│   written into method │        │   HBM for vidmem         │
│   stream OFFSET_IN/   │        │   buffers (GPFIFO,       │
│   OUT and SET_SEM.    │        │   d_buf, sema).          │
└───────────────────────┘        └──────────────────────────┘
         │
         │ GPU MMU page walk
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Physical address:                                           │
│   - HBM: direct DRAM access by GPU clients                  │
│   - host DRAM: PCIe TLP over x16 Gen5 link                  │
└─────────────────────────────────────────────────────────────┘
```

**Under UVM** (which is what Hopper + CUDA 13 uses), the user-space driver
emits a single address — the CPU VA — and that same number reaches the GPU MMU
as the GPU VA. There is no per-access translation in software. This is why the
method stream works: the CPU writes `src_va = 0x76baa0800000` into the
pushbuffer, and the CE's MMU resolves `0x76baa0800000` against the channel's
`FERMI_VASPACE_A` page tables to find the HBM physical page.

**Without UVM** (the path we originally attempted), you would have to
allocate a GPU VA range explicitly via `NV04_MAP_MEMORY_DMA` and track it
separately from the CPU VA. This path is effectively deprecated in the open
driver for non-trivial buffer types and produces `NV_ERR_INVALID_OBJECT_HANDLE`
(0x33) for vidmem into FERMI_VASPACE_A on open-RM 595.

---

## 3. The GPFIFO ring

A GPFIFO is a fixed-size, power-of-two-length circular buffer of **8-byte
entries**. Each entry is a pointer-plus-metadata that tells PBDMA "go read
this many dwords from this pushbuffer address and execute them as methods."

```
GPFIFO ring (GPU VRAM, 512 × 8 B = 4 KiB in our case):

        ┌─────────────────────────────────────────┐
offset 0│ entry[0]    = {entry0_lo, entry1_hi}    │ ← oldest entry (first written)
        ├─────────────────────────────────────────┤
     8  │ entry[1]    = ...                       │
        ├─────────────────────────────────────────┤
     16 │ entry[2]    = ...                       │
        ├─────────────────────────────────────────┤
        │    ...                                  │
        ├─────────────────────────────────────────┤
  4080  │ entry[510]  = ...                       │
        ├─────────────────────────────────────────┤
  4088  │ entry[511]  = ...                       │ ← wraps back to entry[0]
        └─────────────────────────────────────────┘
```

Two indices track the ring state:

- **GPPut** — producer index. CPU writes this. Points to the *next empty slot*
  (i.e., the slot that will be filled by the CPU's next write). Stored in the
  channel's USERD page at offset `0x8c`.
- **GPGet** — consumer index. GPU PBDMA writes this. Points to the *next entry
  the GPU will consume*. Stored in USERD at offset `0x88`.

Ring is empty when `GPGet == GPPut`. Ring is full when
`GPPut == (GPGet - 1) mod ring_size`.

### Why the ring lives in HBM

On Hopper, mc allocates the GPFIFO as `NV01_MEMORY_LOCAL_USER`
(vidmem). This matches the paper's Finding 2:
- **GPFIFO ring in VRAM**: CPU writes over PCIe BAR1 (4 KiB of remote writes
  per submission pattern), GPU PBDMA reads locally from HBM at full HBM speed.
- **Pushbuffer in host RAM**: CPU writes locally (cached or WC), GPU PBDMA
  reads over PCIe.

This asymmetric placement optimizes for the fact that the CPU writes tiny
(8-byte) entries to GPFIFO infrequently but the GPU reads from the pushbuffer
many dwords per submission.

---

## 4. GP entry format — bit-exact

From `clc86f.h`:

```
ENTRY0 (32 bits, bit positions [31:0]):
  [31:2] NVC86F_GP_ENTRY0_GET      — pb_va[31:2] stored IN-PLACE (not shifted)
  [0:0]  NVC86F_GP_ENTRY0_FETCH    — 0 = UNCONDITIONAL, 1 = CONDITIONAL

ENTRY1 (32 bits):
  [31]    NVC86F_GP_ENTRY1_SYNC    — 0 = PROCEED, 1 = WAIT (host-wait before fetch)
  [30:10] NVC86F_GP_ENTRY1_LENGTH  — pushbuffer length in dwords (21 bits → max 8 MiB)
  [9]     NVC86F_GP_ENTRY1_LEVEL   — 0 = MAIN, 1 = SUBROUTINE
  [7:0]   NVC86F_GP_ENTRY1_GET_HI  — pb_va[39:32]
```

The native address range encodable in a *single* GP entry is therefore
bit [39:0] = 40-bit VAs (1 TiB).

### Encoding a pushbuffer at GPU VA `0x76baa0e2c000`, length 18 dwords

Step 1 — decompose the VA:

```
pb_va                     = 0x0000_76ba_a0e2_c000
pb_va[63:40] (overflow)   = 0x0000_76                  ← needs extended-base
pb_va[39:32]              = 0xba                        ← fits in ENTRY1[7:0]
pb_va[31:2]  (in-place)   = pb_va & 0xFFFFFFFC = 0xa0e2_c000
pb_va[1:0]                = must be 0 (pushbuffers are 4-byte aligned)
```

Step 2 — encode the **normal** GP entry (if pb_va ≤ 40 bits):

```
entry0 = 0xa0e2_c000          // pb_va[31:2] at bits [31:2]; FETCH=0 at [1:0]
entry1 = 0x12 << 10            // length = 18 dwords, placed at bits [30:10]
       | 0x000000ba            // pb_va[39:32] = 0xba
       | (0 << 9)              // LEVEL = MAIN
       | (0 << 31)             // SYNC = PROCEED
       = 0x0000_48ba                     // (see computation below)

Numerically:
  length_field   = 18 << 10     = 0x0000_4800
  get_hi_field   = 0xba          = 0x0000_00ba
  entry1          = 0x0000_48ba
```

Step 3 — because pb_va has bits above 40 (the `0x76` byte), we actually need
TWO GP entries: first a `SET_PB_SEGMENT_EXTENDED_BASE` entry providing the
high bits, then the normal entry shown above. See §5 for the extended-base
format.

### The two common bugs (both caught in mc bring-up)

1. **Shifting entry0 right by 2** instead of masking. If you write
   `entry0 = (uint32_t)(pb_va >> 2)`, the bits that should sit at [31:2]
   end up at [29:0], and ENTRY0_FETCH at [1:0] ends up populated with the
   actual GET bits. PBDMA reads the pushbuffer at a 4× shifted-down VA,
   produces a garbage read, and triggers Xid 31 (MMU fault).

2. **Forgetting extended-base**. If pb_va has any bits above [39:0]
   (guaranteed under UVM, which places buffers high in the 48-bit range),
   a single normal GP entry silently drops the high bits, and PBDMA reads
   from a wildly wrong address. Also Xid 31.

---

## 5. Extended-base GP entry

On Hopper, GP entries can carry an OPCODE in `ENTRY1[7:0]` that reinterprets
the entry as a control command rather than a pushbuffer fetch:

```
NVC86F_GP_ENTRY1_OPCODE (bits [7:0] of ENTRY1):
  0x00 — NOP (no-op)
  0x01 — ILLEGAL (reserved; triggers error)
  0x02 — GP_CRC (GPFIFO CRC operation, unused in normal submission)
  0x03 — PB_CRC (pushbuffer CRC, unused in normal submission)
  0x04 — SET_PB_SEGMENT_EXTENDED_BASE
```

**Note the field overlap**: ENTRY1[7:0] is both `GET_HI` (for normal entries)
and `OPCODE` (for control entries). The GPU distinguishes the two cases by the
context: if the preceding entry configured an extended base, the next entry is
a normal fetch whose VA is combined with that base.

### `SET_PB_SEGMENT_EXTENDED_BASE` (OPCODE 0x4)

When ENTRY1[7:0] = 0x04:

```
ENTRY0 (32 bits):
  [24:8] NVC86F_GP_ENTRY0_PB_EXTENDED_BASE_OPERAND  — operand: pb_va[56:40]
  others — reserved/zero

ENTRY1:
  [7:0] = 0x04
  others — zero
```

The 17-bit operand at ENTRY0[24:8] represents the **high 17 bits of the VA**
starting at bit 40. When PBDMA subsequently executes a normal GP entry,
it reconstructs the full 57-bit VA as:

```
  full_va = (extended_base << 40)       // from SET_PB_SEGMENT_EXTENDED_BASE operand
          | ((entry1 & 0xFF) << 32)     // from normal entry's GET_HI
          | (entry0 & 0xFFFFFFFC)       // from normal entry's GET
```

### Complete 2-entry write for high VA

Pushbuffer at `pb_va = 0x76baa0e2c000`, length 18 dwords:

```c
// Entry 0: SET_PB_SEGMENT_EXTENDED_BASE
uint32_t ext_base = (pb_va >> 40) & 0x1FFFF;   // = 0x76

gpfifo_ring[idx*2 + 0] = (ext_base & 0x1FFFF) << 8;  // = 0x0000_7600
gpfifo_ring[idx*2 + 1] = 0x00000004;                  // OPCODE_SET_PB_SEGMENT_EXTENDED_BASE
idx++;

// Entry 1: normal GP entry
gpfifo_ring[idx*2 + 0] = (uint32_t)(pb_va & 0xFFFFFFFC);  // = 0xa0e2_c000
gpfifo_ring[idx*2 + 1] = ((pb_va >> 32) & 0xFF)            // = 0xba
                       | (18 << 10);                        // length
idx++;

// sfence before writing GPPut
_mm_sfence();

// Advance GPPut by 2 (two entries consumed)
userd->GPPut += 2;
```

---

## 6. The pushbuffer and method stream

The pushbuffer is a byte-addressable buffer in host RAM (for Hopper + CUDA)
containing a sequence of 32-bit "method packets." Each packet is either a
method *header* (describing what comes next) or *data* (the argument to the
preceding header).

### Physical layout

```
pushbuffer (host DRAM, 1 MiB, write-combine):

offset 0x0:  ┌──────────────┐
             │ method[0]    │ (32-bit header or data)
             ├──────────────┤
offset 0x4:  │ method[1]    │
             ├──────────────┤
offset 0x8:  │ method[2]    │
             ├──────────────┤
             │    ...       │
             │              │
             └──────────────┘

A GP entry tells PBDMA:
  start at offset 0 (pb_va)
  read N dwords (ENTRY1_LENGTH)
  execute them as a method stream
```

### High-level stream shape for a D2H

A D2H transfer method stream consists of exactly 18 dwords:

```
dword 0  : method header — SET_OBJECT, count=1, subch=4
dword 1  : payload        — HOPPER_DMA_COPY_A (class id 0xC8B5)
dword 2  : method header — OFFSET_IN_UPPER, count=2, subch=4
dword 3  : payload        — src_va[56:32] (25 bits)
dword 4  : payload        — src_va[31:0]
dword 5  : method header — OFFSET_OUT_UPPER, count=2, subch=4
dword 6  : payload        — dst_va[56:32]
dword 7  : payload        — dst_va[31:0]
dword 8  : method header — LINE_LENGTH_IN, count=1, subch=4
dword 9  : payload        — nbytes
dword 10 : method header — LAUNCH_DMA, count=1, subch=4
dword 11 : payload        — 0x182 (the copy; no flush, no semaphore — see §9)
dword 12 : method header — SET_SEMAPHORE_A, count=3, subch=4
dword 13 : payload        — sema_va[56:32]
dword 14 : payload        — sema_va[31:0]
dword 15 : payload        — sema completion value (e.g., 1)
dword 16 : method header — LAUNCH_DMA, count=1, subch=4
dword 17 : payload        — 0x00c (release only; moves no data — see §9)

Total: 18 dwords = 72 bytes per transfer.
```

The copy and the release are two separate `LAUNCH_DMA`s because that is what H100
libcuda emits.  A single fused launch (`0x18e`) is also valid — see §9.

### Why the length is fixed regardless of transfer size

All data motion (4 MiB, 64 MiB, whatever) is driven by one copy `LAUNCH_DMA`.
The command describes *what* to move (src_va, dst_va, nbytes) and the CE
hardware does the actual moving. The pushbuffer method stream is a *description*,
not the data.

---

## 7. Method header format — bit-exact

Every method header is a single 32-bit dword with this Kepler-compatible layout
(used by all Turing+ GPUs, including Hopper):

```
bits [31:29]  NVA06F_DMA_SEC_OP  (names verbatim from cla06f.h)
     000  GRP0_USE_TERT   ← defer to the tertiary op in [17:16]
     001  INC_METHOD      ← the common case; addr increments per data dword
     010  GRP2_USE_TERT
     011  NON_INC_METHOD
     100  IMMD_DATA_METHOD ← immediate data (no separate payload)
     101  ONE_INC
     110  RESERVED6
     111  END_PB_SEGMENT
     110+  reserved

bits [28:16]  count       — number of data dwords that follow
bits [15:13]  subchannel  — 4 for CE (NVA06F_SUBCHANNEL_COPY_ENGINE)
bits [12:0]   method addr — method register address ÷ 4
```

### `INCR` header encoder (what mc uses)

```c
#define INCR_HEADER_SUB(method_addr, count, subch)  \
    ((1u << 29)                    /* type = INCR */             \
   | ((count) << 16)               /* count at [28:16] */        \
   | (((subch) & 0x7) << 13)       /* subchannel at [15:13] */   \
   | (((method_addr) >> 2) & 0x1FFF))  /* addr[14:2] at [12:0] */

// For CE, subch is always 4:
#define INCR_HEADER(m, c) INCR_HEADER_SUB((m), (c), 4)
```

### Example decoding

Take the SET_OBJECT header mc actually emits.

For SET_OBJECT (method=0x0, count=1, subch=4):
```
INCR_HEADER_SUB(0, 1, 4)
  = (1 << 29)             = 0x2000_0000
  | (1 << 16)             = 0x0000_0001_0000 = 0x0001_0000
  | (4 << 13)             = 0x0000_8000
  | (0 >> 2) & 0x1FFF     = 0x0000_0000
  = 0x2001_8000
```

So the very first dword of every CE pushbuffer in mc is `0x20018000`,
followed by `0x0000C8B5` (the HOPPER_DMA_COPY_A class ID) as the payload.

### Why `INCR`

With `type=INCR` and `count=N`, PBDMA increments the method address by 4 bytes
after each of the N data dwords. This is convenient for writing sequential
registers like `OFFSET_IN_UPPER` (0x400) followed by `OFFSET_IN_LOWER` (0x404):
one INCR header with count=2 is enough. The alternative `NON_INCR` re-writes
the same register N times (used for streaming data).

---

## 8. Subchannels and the SET_OBJECT method

A PBDMA channel carries up to 8 **subchannels** (bits [15:13] of the method
header). Each subchannel binds one "object" — typically an engine object like
CE, Graphics, or SEC2.

On Hopper + Kepler-style fifo:

```
from src/common/sdk/nvidia/inc/class/cla06fsubch.h:
  NVA06F_SUBCHANNEL_3D           = 0
  NVA06F_SUBCHANNEL_COMPUTE      = 1
  NVA06F_SUBCHANNEL_I2M          = 2
  NVA06F_SUBCHANNEL_2D           = 3
  NVA06F_SUBCHANNEL_COPY_ENGINE  = 4   ← the one we use
  // ...
```

For a CE transfer, all methods must go through subchannel 4. If any method
header carries subch=0 (the implicit default if not set), PBDMA sees a method
addressed to an unbound subchannel and reports Xid 32 "invalid or corrupted
pushbuffer stream."

### SET_OBJECT — binding the class to the subchannel

Method `NVC86F_SET_OBJECT` lives at method address `0x0`. Its payload is the
class ID of the object to bind. For HOPPER_DMA_COPY_A:

```
header : INCR_HEADER_SUB(0x0, count=1, subch=4)  = 0x2001_8000
payload: 0x0000_C8B5                              // HOPPER_DMA_COPY_A class ID
```

After PBDMA consumes these two dwords, subchannel 4 on the channel is bound to
a HOPPER_DMA_COPY_A instance, and subsequent methods addressed to subch=4 are
routed to the CE.

**SET_OBJECT must be the first method in every pushbuffer** unless the channel
is known to have been bound previously and not torn down. In practice, mc
issues SET_OBJECT in every pushbuffer to be safe — the overhead is trivial
(2 dwords per transfer).

---

## 9. The NVC8B5 Copy Engine method set

The CE has a large method space (hundreds of registers) but a D2H transfer uses
just five of them, plus LAUNCH_DMA to fire the transfer.

### Source and destination addresses (VIRTUAL mode)

```
NVC8B5_OFFSET_IN_UPPER     = 0x400   // src_va[56:32], 25-bit field
NVC8B5_OFFSET_IN_LOWER     = 0x404   // src_va[31:0]
NVC8B5_OFFSET_OUT_UPPER    = 0x408   // dst_va[56:32]
NVC8B5_OFFSET_OUT_LOWER    = 0x40C   // dst_va[31:0]
```

`UPPER` fields are **25 bits wide** (bits [24:0] of the method payload), covering
bits [56:32] of the VA. A 57-bit VA (= 128 PiB) exceeds Hopper's 48-bit physical
and 48-bit virtual address limits, so the full field is never used; but all 25
bits must be written as zeros in the unused positions.

For a VIRTUAL address (which is what UVM gives us), the CE's MMU walks the
channel's `FERMI_VASPACE_A` page tables to translate the VA.

Alternative: `NVC8B5_SET_SRC_PHYS_MODE` (0x260) and `NVC8B5_SET_DST_PHYS_MODE`
(0x264) switch a side to physical addressing with an explicit aperture
(LOCAL_FB / COHERENT_SYSMEM / NONCOHERENT_SYSMEM / PEERMEM). mc uses
VIRTUAL on both sides so these methods are not emitted.

### Transfer size

```
NVC8B5_LINE_LENGTH_IN = 0x418   // payload: nbytes (for 1D) or pitch (for 2D)
```

For a 1D transfer, LINE_LENGTH_IN is the total byte count. For 2D, combine
with `LINE_COUNT` and `PITCH_IN/OUT`. mc is 1D only and sets
`MULTI_LINE_ENABLE=FALSE` in LAUNCH_DMA.

### Semaphore configuration

```
NVC8B5_SET_SEMAPHORE_A       = 0x240   // payload: sema_va[56:32]
NVC8B5_SET_SEMAPHORE_B       = 0x244   // payload: sema_va[31:0]
NVC8B5_SET_SEMAPHORE_PAYLOAD = 0x248   // payload: 32-bit value to write
```

Written in a single `INCR` block with count=3 targeting method 0x240 (PBDMA
increments the method address by 4 for each data dword, covering 0x240, 0x244,
0x248 in order).

### LAUNCH_DMA — the fire-and-forget trigger

```
NVC8B5_LAUNCH_DMA = 0x300   // the payload encodes all launch behavior
```

Its bit-fields (subset we set in mc):

```
LAUNCH_DMA_DATA_TRANSFER_TYPE         [1:0]    0=NONE, 1=PIPELINED, 2=NON_PIPELINED
LAUNCH_DMA_FLUSH_ENABLE               [2:2]    0=FALSE, 1=TRUE (flush writes before sema)
LAUNCH_DMA_SEMAPHORE_TYPE             [4:3]    0=NONE, 1=RELEASE_ONE_WORD, 2=RELEASE_WITH_TIMESTAMP
LAUNCH_DMA_SRC_MEMORY_LAYOUT          [7:7]    0=BLOCKLINEAR, 1=PITCH
LAUNCH_DMA_DST_MEMORY_LAYOUT          [8:8]    0=BLOCKLINEAR, 1=PITCH
LAUNCH_DMA_MULTI_LINE_ENABLE          [9:9]    0=FALSE (1D), 1=TRUE (2D)
LAUNCH_DMA_REMAP_ENABLE               [10:10]  0=FALSE, 1=TRUE
LAUNCH_DMA_SRC_TYPE                   [12:12]  0=VIRTUAL, 1=PHYSICAL
LAUNCH_DMA_DST_TYPE                   [13:13]  0=VIRTUAL, 1=PHYSICAL
LAUNCH_DMA_FLUSH_TYPE                 [25:25]  0=SYS, 1=GL
```

H100 libcuda uses **two** launches, and so does `mc`'s host path.

Launch 1 — the copy, payload = 0x00000182:

```
bit 1   (TRANSFER_TYPE=NON_PIPELINED)    = 2
bit 7   (SRC_MEMORY_LAYOUT=PITCH)        = 1<<7  = 0x080
bit 8   (DST_MEMORY_LAYOUT=PITCH)        = 1<<8  = 0x100
        FLUSH_ENABLE=FALSE, SEMAPHORE_TYPE=0 (none), everything else 0
                                           total = 0x182
```

Launch 2 — the release, payload = 0x0000000c:

```
bits 1:0 (TRANSFER_TYPE=NONE)            = 0
bit 2   (FLUSH_ENABLE=TRUE)              = 1<<2  = 0x004
bit 3   (SEMAPHORE_TYPE=ONE_WORD_RELEASE)= 1<<3  = 0x008
        (everything else 0)
                                           total = 0x00c
```

The **fused** single-launch form, payload = 0x0000018e, is also valid and is what
Yan et al. 2026 Listing 1 reports from an A40 trace.  `mc`'s SM-authored kernel
still emits it because it fits in 16 dwords:

```
bit 1   (TRANSFER_TYPE=NON_PIPELINED)    = 2
bit 2   (FLUSH_ENABLE=TRUE)              = 1<<2  = 0x004
bit 3   (SEMAPHORE_TYPE=ONE_WORD_RELEASE)= 1<<3  = 0x008
bit 7   (SRC_MEMORY_LAYOUT=PITCH)        = 1<<7  = 0x080
bit 8   (DST_MEMORY_LAYOUT=PITCH)        = 1<<8  = 0x100
                                           total = 0x18e
```

**Implications of each flag:**

- `NON_PIPELINED`: the CE drains previously-started work before starting this
  one. Safer for debugging (serializes). `PIPELINED` overlaps this transfer
  with the previous one and is what you want for throughput.
- `FLUSH_ENABLE=TRUE`: CE waits for all DMA writes to commit before writing
  the semaphore. Without this, the semaphore can arrive before the data,
  breaking the completion contract.
- `SEMAPHORE_TYPE=ONE_WORD_RELEASE`: after the flush, CE writes the 4-byte
  payload to sema_va. The alternative `_WITH_TIMESTAMP` writes 16 bytes
  (payload + 64-bit GPU timestamp).
- `SRC_MEMORY_LAYOUT=PITCH` and `DST_MEMORY_LAYOUT=PITCH`: linear buffers,
  not block-linear (the tiled format used for graphics surfaces).
- `SRC_TYPE=VIRTUAL` and `DST_TYPE=VIRTUAL` (both 0 by default): addresses are
  VAs, not physical.

LAUNCH_DMA is *side-effecting*: once PBDMA delivers it to the CE, the CE
latches all previously-written state (OFFSET_IN, OFFSET_OUT, LINE_LENGTH_IN,
SET_SEMAPHORE_*) and begins executing. LAUNCH_DMA is always the last method
in a D2H pushbuffer.

---

## 10. USERD — the user-accessible doorbell page

USERD is the per-channel "user registers" page exposed to userspace. It's the
mechanism through which the CPU and GPU exchange producer/consumer indices
without kernel involvement.

### Layout (HopperAControlGPFifo, `clc86f.h`)

```
offset  size  field
─────── ───── ──────────────────────────────────────────────────
0x00    0x40  Ignored00         (reserved padding)
0x40    0x04  Put               (method submission put, legacy)
0x44    0x04  Get               (method submission get, legacy)
0x48    0x04  Reference         (reference value for sync, read-only)
0x4c    0x04  PutHi             (high-order Put bits)
0x50    0x08  Ignored01
0x58    0x04  TopLevelGet
0x5c    0x04  TopLevelGetHi
0x60    0x04  GetHi
0x64    0x1c  Ignored02
0x80    0x04  Ignored03         (used to be engine yield)
0x84    0x04  Ignored04
0x88    0x04  GPGet             ← GPU writes; CPU reads to see consumer index
0x8c    0x04  GPPut             ← CPU writes; PBDMA reads for producer index
0x90    0x170 Ignored05
─────── ─────
0x200   total size (one USERD slot is 512 bytes; libcuda packs its USERD slots
        into BAR1-mapped vidmem pages 3 at a time per channel.  mc takes
        a simpler approach: its USERD sits at `userdOffset=0x2000` inside a
        monolithic 2 MiB HBM `gpu_ctl` allocation that also holds the GPFIFO
        ring at offset 0.  Both layouts satisfy the HW constraint that each
        USERD slot is 4 KiB aligned.
```

### The `volatile` contract

Both GPPut and GPGet are concurrent between CPU and GPU:
- GPPut is written by the CPU, read asynchronously by the GPU (PBDMA).
- GPGet is written by the GPU (PBDMA), read asynchronously by the CPU.

In C, access must go through a `volatile` pointer to prevent the compiler from
caching the value in a register:

```c
volatile HopperAControlGPFifo *userd = ...;
userd->GPPut = new_gp_put;       // visible producer advance
uint32_t seen = userd->GPGet;    // current consumer position (may lag or race)
```

On x86, writes to USERD are architected as Write-Combine memory. The write may
sit in a CPU WC buffer until either the buffer fills or an `sfence` instruction
executes. PBDMA does not see the write until the WC buffer is flushed.

### USERD placement on Hopper — FBMEM, shared with GPFIFO

On Hopper, RM places USERD in FBMEM (HBM) for both mc and libcuda.
mc allocates USERD as part of `h_gpu_ctl_mem`, a 2 MiB
`NV01_MEMORY_LOCAL_USER` (HBM / vidmem) allocation that also holds the
GPFIFO ring: GPFIFO at offset 0, USERD at offset `USERD_OFFSET = 0x2000`.
The CPU reaches USERD through a BAR1 CPU alias of that 2 MiB region,
obtained from `rm_map_memory_at()` and placed inside the VA pool
(§13 of `findings.md`).  PBDMA reads GPPut locally from HBM.  The
write-combine aperture of the BAR1 alias keeps the CPU's GPPut write
bounded to one PCIe posted-write per `sfence`.

---

## 11. BAR0 VF doorbell register

On Hopper (and Ampere), the USERD GPPut update is **informative but not
activating**. PBDMA does not continuously poll USERD; it waits to be told
"this channel has work." That signal comes from an MMIO write to a BAR0
register.

### The register

```
NV_VIRTUAL_FUNCTION_DOORBELL = 0x30090   (relative to the VF register block)

Writeable 32-bit register. Writing a work-submit-token to this register
wakes the host scheduler, which then dispatches the corresponding channel
onto a PBDMA engine.
```

> **`0x30090` is not a BAR0 offset on bare metal.** `dev_vm.h` describes the
> *virtual function's* view of the register space. RM reaches the register as
> `GPU_GET_VREG_OFFSET(pGpu, NV_VIRTUAL_FUNCTION_DOORBELL)` —
> i.e. `pGpu->sriovState.virtualRegPhysOffset + 0x30090`
> (in `nv_gpu_ops.c`, macro `GPU_GET_VREG_OFFSET` in `g_gpu_access_nvoc.h`). And
> `gpuGetVirtRegPhysOffset_TU102` (`kern_gpu_tu102.c:94`) returns:
>
> | Context | `virtualRegPhysOffset` | Doorbell in BAR0 |
> |---|---|---|
> | Bare metal — every non-Tegra chip, incl. GH100 / GA10x | `DRF_BASE(NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET)` = `0xB80000` | **`0xBB0090`** |
> | SR-IOV guest VF | `0` | `0x30090` |
> | Tegra (`T234D`/`T239D`/`T264D`) | `0` | `0x30090` |
>
> Note the function reads counterintuitively — it returns **0 for the guest**
> and non-zero on the host — and the HAL table (`g_gpu_nvoc.c:1471`) routes
> only Tegra to the `return 0` variant. Sanity check:
> `NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET = 0x00BBFFFF:0x00B80000`, and
> `0xB80000 + 0x30000 = 0xBB0000` — exactly the top 64 KiB of that range.
>
> Earlier revisions of this document and `mc`'s comments all quoted
> `BAR0 + 0x30090` as an absolute address. That is the guest's number.
> No code was ever affected, because every consumer uses the `+0x90`
> mapping-relative offset below.

The *work-submit-token* is a per-channel identifier returned by the RM control
`NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN`. It encodes the channel ID and
some internal scheduling metadata; the exact bit layout is
implementation-defined but the value is 32 bits.

### HOPPER_USERMODE_A — the userspace handle

Userspace can't access BAR0 directly. The `HOPPER_USERMODE_A` class (0xC661)
is a kernel-managed object that, when allocated **under a subdevice** (so it is
per-GPU, not per-channel) and mapped, exposes a 64 KiB window onto the
non-privileged VF register block. The window base is
`GPU_GET_VREG_OFFSET(pGpu, DRF_BASE(NV_VIRTUAL_FUNCTION))`
(`kern_gpu_tu102.c:290`, reached via `kfifoGetUsermodeMapInfo_HAL`) and its
size is `DRF_SIZE(NVC361)` = 64 KiB.

**The doorbell is always at `+0x90` within that mapping**, and this is the only
offset worth memorising. The window base and the register pick up the *same*
`virtualRegPhysOffset`, so the difference is invariably
`0x30090 - 0x30000 = 0x90` — on bare metal or in a guest, and regardless of
whether the window is carved from BAR0 or BAR1.

```c
NV_HOPPER_USERMODE_A_PARAMS um_params = {
    /* libcuda — and mc today — pass NV_TRUE here on Hopper, which returns
     * a BAR1 alias of the same VF register block (and is what makes the
     * page GPU-mappable, so an SM can ring the doorbell itself).
     * AMPERE_USERMODE_A ignores this flag and always returns BAR0.
     * The doorbell is at +0x90 in every one of those cases. */
    .bBar1Mapping = NV_TRUE,   // BAR1 — the mapping that carries the doorbell
    .bPriv        = NV_FALSE,   // non-privileged VF page (not the PRIV region)
};
h_usermode = rm_alloc(ctl_fd, h_client, h_subdevice, HOPPER_USERMODE_A, &um_params);
usermode_cpu = rm_map_memory(ctl_fd, "/dev/nvidia0", h_client, h_subdevice,
                              h_usermode, 0, 0x10000, 0);
volatile uint32_t *vf_doorbell = (volatile uint32_t *)(usermode_cpu + 0x90);
```

### The ring-the-doorbell sequence

```c
userd->GPPut = new_gp_put;              // 1. advance producer index
_mm_sfence();                            // 2. flush WC to host RAM
*vf_doorbell = work_submit_token;        // 3. wake host scheduler (MMIO)
_mm_sfence();                            // 4. flush MMIO store
```

Both writes are needed. Writing only GPPut leaves PBDMA unaware of the new
work. Writing only the doorbell without advancing GPPut causes PBDMA to wake
up, consult GPPut, see no new work, and go back to sleep.

---

## 12. Semaphore format

The semaphore is just a 32-bit aligned memory word the CE writes on completion.
No special format.

### One-word release

When `LAUNCH_DMA_SEMAPHORE_TYPE = RELEASE_ONE_WORD_SEMAPHORE (=1)`, the CE
writes exactly the 4-byte `SET_SEMAPHORE_PAYLOAD` value to the address given by
`SET_SEMAPHORE_A/B`. No timestamp, no extra metadata.

```
[sema_va + 0]:  CE writes sema_payload  (4 bytes)
```

### With timestamp

When `LAUNCH_DMA_SEMAPHORE_TYPE = RELEASE_WITH_TIMESTAMP (=2)`, the CE writes
16 bytes:

```
[sema_va + 0]:  CE writes sema_payload  (4 bytes)
[sema_va + 4]:  (padding, typically 0)  (4 bytes)
[sema_va + 8]:  CE writes GPU timestamp (8 bytes, low 64 bits of engine timer)
```

The timestamp is in GPU-timer ticks (convertible to real time via
`NV2080_CTRL_TIMER_GET_TIME` or derived from `GPU_CPU_TIME_CORRELATION_INFO`).

### CPU access

mc places the semaphore in host RAM (sysmem), matching libcuda — whose
`SET_SEMAPHORE_VA` resolves to an `NV01_MEMORY_SYSTEM` allocation. The CPU
therefore polls a plain cached pointer, so each spin is a local read that
costs nothing on the bus; the line is invalidated when the CE's write
lands:

```c
volatile uint32_t *sema = ...;  // plain sysmem pointer
while (*sema != expected_payload) {
    /* tight spin; may yield or sleep in higher-level code */
}
```

Placing it in HBM instead would put the cell behind a BAR1 alias, turning
every spin into a PCIe read competing with the very transfer it is waiting
on. Ordering is preserved either way: the release launch carries
`FLUSH_ENABLE` with `FLUSH_TYPE=SYS`, and for a sysmem destination PCIe
posted-write ordering from a single Requester ID keeps the release behind
the data.

---

## 13. Component interaction — end-to-end timeline

This section walks through *who does what* for a single D2H submission,
after all setup is complete.

```
Time →

CPU:  [write pushbuffer]
      [write GP entry(ies) into GPFIFO ring via BAR1]
      [sfence]
      [write GPPut into USERD page]
      [sfence]
      [write work-submit-token to BAR0 VF doorbell]         ← kick
      [sfence]
      [begin polling sema]
          |
          |<─────────── (~µs to ~ms) ───────────>|
          |                                     |
          | waiting for GPU...                  | sees sema==1, polling exits
          v                                     ^
                                                |
GPU:  [host scheduler receives BAR0 write]      |
      [looks up channel by work-submit-token]   |
      [adds channel to PBDMA runlist]           |
           |                                    |
           v                                    |
      [PBDMA fetches channel state from RAMFC]  |
      [PBDMA reads GPGet/GPPut from USERD]      |
           |                                    |
           v                                    |
      [PBDMA reads GP entry from GPFIFO ring]   |
      [decodes: pb_va = ..., length = ...]      |
      (if extended-base entry: latches base     |
       and reads the next normal entry)         |
           |                                    |
           v                                    |
      [PBDMA issues PCIe read for pushbuffer]   |
      [PCIe RX: pushbuffer bytes arrive]        |
           |                                    |
           v                                    |
      [PBDMA parses method headers]             |
      [routes SET_OBJECT to subch 4 binding]    |
      [routes NVC8B5 methods to CE subch 4]     |
           |                                    |
           v                                    |
      [CE latches src/dst/size/sema from        |
       methods]                                 |
      [CE MMU walks src_va → HBM phys addr]     |
      [CE MMU walks dst_va → host phys addr     |
       (via IOMMU if enabled)]                  |
           |                                    |
           v                                    |
      [CE issues HBM reads (local)]             |
      [CE issues PCIe writes for dst]           |
      [data flows out over PCIe × transfer_time]|
           |                                    |
           v                                    |
      [FLUSH_ENABLE=TRUE: CE waits for all      |
       PCIe writes to commit to host DRAM]      |
           |                                    |
           v                                    |
      [CE writes sema_payload to sema_va]       |
      [sysmem write commits; CPU's next         |
       cached poll observes it]                 |
           |                                    |
           v                                    |
      [PBDMA updates USERD.GPGet via            |
       host-side state]                         |
           |                                    |
           +────────────────────────────────────┘
                          signal propagation complete
```

### Phase breakdown for timing

1. **CPU submission phase** — ~100 ns for a 72-byte pushbuffer + GPFIFO + USERD +
   doorbell on a modern x86. Dominated by sfence cost and BAR0 MMIO latency.
2. **GPU wake-up** — ~1–5 µs for the host scheduler to observe the BAR0 write
   and queue the channel.
3. **PBDMA fetch and parse** — ~1–3 µs for the first pushbuffer read over PCIe
   + method decode.
4. **CE setup** — ~100 ns for register latching.
5. **Data transfer** — (nbytes / PCIe_bandwidth).  A 4 MiB copy measures
   ~134 µs end to end (4 × 2²⁰ B ÷ 31.3 × 10⁹ B/s ≈ 134.0 µs), of which
   ~120 µs is data movement once the fixed costs above and below are taken
   out.
6. **Flush + sema write** — ~1 µs for write completion and sema visibility.

Total for 4 MiB on H100 PCIe: ~134 µs, dominated by the transfer itself (Appendix B walks the same scenario and lands at the same figure).

---

## 14. Memory ordering and fence discipline

x86 PCIe/MMIO writes are ordered in a way that is usually-but-not-always what
you want. The following rules govern mc's fences.

### Write-Combine (WC) memory

USERD, GPFIFO (BAR1), and the VF doorbell (BAR1 on Hopper) are all mapped Write-Combine on x86.
WC means:

- Writes to adjacent addresses within a cache line may coalesce in a CPU WC buffer.
- WC buffers drain to the bus on: sfence, mfence, serializing instructions, or
  when the buffer fills.
- Read-after-write on WC memory returns *implementation-defined* values — often
  zeros or garbage. **Do not expect to read back the bytes you just wrote.**
- Reads from WC are effectively uncached: every load is a bus transaction.

### Required fences

```
1. CPU writes method stream to pushbuffer (host DRAM, WC)
   sfence here: [implicit WC ordering]
2. CPU writes GP entry to GPFIFO ring (HBM via BAR1, WC)
   sfence required:
     ensures (1) is visible before (2) so PBDMA doesn't read the GP entry
     and try to fetch a pushbuffer that hasn't arrived yet.
3. CPU writes GPPut to USERD (HBM via BAR1 alias on Hopper, WC)
   sfence required:
     ensures (2) is visible before (3) so PBDMA doesn't see the advance
     and try to read a GPFIFO slot that hasn't been written yet.
4. CPU writes work-submit-token to VF doorbell (MMIO)
   sfence required:
     ensures (3) is visible before (4) so the GPU host scheduler doesn't
     wake and observe a stale GPPut.
```

Omitting any of these sfences produces *intermittent* failures: usually the
WC buffers happen to drain in order, but under load (cache pressure,
interrupts) an out-of-order drain causes PBDMA to fetch from a not-yet-written
pushbuffer. The result is a garbage method stream → Xid 32.

### Read-side

The semaphore poll reads a cached sysmem cell, so the reads stay local until
the CE's write invalidates the line — no fence is needed on the read side, but
the compiler's `volatile` qualifier is critical to force a real load every
iteration.

---

## 15. What each component reads vs writes

A cross-reference that makes the interactions unambiguous.

### Pushbuffer

| Component | Operation | Reason |
|---|---|---|
| CPU | Write | builds method stream |
| PBDMA | Read over PCIe | fetches method headers/data for decoding |

### GPFIFO ring (HBM)

| Component | Operation | Reason |
|---|---|---|
| CPU | Write via BAR1 | inserts GP entries |
| PBDMA | Read locally (HBM) | fetches GP entries for decoding |

### USERD page (HBM on Hopper, via BAR1 alias)

| Component | Field | Operation | Reason |
|---|---|---|---|
| CPU | GPPut | Write via BAR1 alias | advance producer index |
| CPU | GPGet | Read via BAR1 alias | observe consumer progress (optional) |
| PBDMA | GPPut | Read locally (HBM) | learn where work ends |
| PBDMA | GPGet | Write locally (HBM) | advance consumer after consumption |

(On pre-Hopper GPUs where USERD is in sysmem, the PBDMA reads/writes
become PCIe MRd/MWr; on Hopper they stay on-GPU.)

### BAR0 VF doorbell

| Component | Operation | Reason |
|---|---|---|
| CPU | Write MMIO with work-submit-token | wake host scheduler |
| GPU host scheduler | Read register internally | extract token, look up channel, dispatch |
| CPU | Read | **NOT SUPPORTED**: VF doorbell reads as zero on Hopper (per Yan et al. 2026 §5; requires a shadow page to intercept). |

### CE source (d_buf in HBM)

| Component | Operation | Reason |
|---|---|---|
| CPU | No access | `d_buf` has no CPU alias at all — see §13 |
| CE | Write from `staging` | CE-staged fill puts the test pattern in HBM |
| CE | Read locally (HBM) | fetches bytes to DMA to destination |

### CE destination (h_buf in host DRAM)

| Component | Operation | Reason |
|---|---|---|
| CPU | Read | verify transfer after completion |
| CE | Write over PCIe | actual DMA output |

### Semaphore (sysmem)

| Component | Operation | Reason |
|---|---|---|
| CPU | Write directly (once, zero before each transfer) | reset for reuse |
| CPU | Read directly (tight poll, cached) | detect completion |
| CE | Write over PCIe (sysmem) | signal completion with configured payload |

### GPU MMU page tables (managed by UVM + GSP)

| Component | Operation | Reason |
|---|---|---|
| UVM (kernel) | Writes during `UVM_MAP_EXTERNAL_ALLOCATION` | installs PDE/PTE entries for src_va, dst_va, pb_va, gpfifo_va, sema_va |
| CE MMU | Walks during DMA | translates VAs to physical addresses |
| PBDMA MMU | Walks during pushbuffer fetch | translates pb_va to host DRAM physical address |

These MMU walks are hardware-automatic once UVM installs the page tables —
no software is involved during the hot path.

---

## Appendix A — Bit-field reference tables (exact header values)

### GP entry (NVC86F)

| Symbol | Location | Width | Purpose |
|---|---|---|---|
| `NVC86F_GP_ENTRY__SIZE` | — | 8 bytes | total entry size |
| `NVC86F_GP_ENTRY0_FETCH` | entry0[0:0] | 1 | 0=UNCONDITIONAL, 1=CONDITIONAL |
| `NVC86F_GP_ENTRY0_GET` | entry0[31:2] | 30 | pb_va[31:2] in-place |
| `NVC86F_GP_ENTRY0_OPERAND` | entry0[31:0] | 32 | generic 32-bit operand (for OPCODE entries) |
| `NVC86F_GP_ENTRY0_PB_EXTENDED_BASE_OPERAND` | entry0[24:8] | 17 | pb_va[56:40] for SET_PB_SEGMENT_EXTENDED_BASE |
| `NVC86F_GP_ENTRY1_GET_HI` | entry1[7:0] | 8 | pb_va[39:32] (normal entries) |
| `NVC86F_GP_ENTRY1_OPCODE` | entry1[7:0] | 8 | opcode for control entries (overlaps GET_HI) |
| `NVC86F_GP_ENTRY1_LEVEL` | entry1[9:9] | 1 | 0=MAIN, 1=SUBROUTINE |
| `NVC86F_GP_ENTRY1_LENGTH` | entry1[30:10] | 21 | pushbuffer length in dwords |
| `NVC86F_GP_ENTRY1_SYNC` | entry1[31:31] | 1 | 0=PROCEED, 1=WAIT |

### OPCODE values (entry1[7:0])

| Value | Name |
|---|---|
| 0x00 | NOP |
| 0x01 | ILLEGAL |
| 0x02 | GP_CRC |
| 0x03 | PB_CRC |
| 0x04 | SET_PB_SEGMENT_EXTENDED_BASE |

### Method header (Kepler-style, all Turing+)

| bits | meaning |
|---|---|
| [31:29] | `SEC_OP` (1=INC_METHOD, 3=NON_INC_METHOD, 4=IMMD_DATA_METHOD, 7=END_PB_SEGMENT) |
| [28:16] | count (up to 8191) |
| [15:13] | subchannel (0–7) |
| [12:0] | method_address >> 2 |

### Subchannel conventional assignment (from cla06fsubch.h)

| Subch | Name |
|---|---|
| 0 | 3D |
| 1 | COMPUTE |
| 2 | I2M |
| 3 | 2D |
| **4** | **COPY_ENGINE** |
| 6+ | reserved |

### NVC8B5 CE methods used by mc

| Method | Address | Count | Payload |
|---|---|---|---|
| `SET_OBJECT` | 0x000 | 1 | class ID = 0xC8B5 |
| `OFFSET_IN_UPPER/_LOWER` | 0x400 | 2 | src_va[56:32], src_va[31:0] |
| `OFFSET_OUT_UPPER/_LOWER` | 0x408 | 2 | dst_va[56:32], dst_va[31:0] |
| `LINE_LENGTH_IN` | 0x418 | 1 | nbytes |
| `SET_SEMAPHORE_A/_B/_PAYLOAD` | 0x240 | 3 | sema_va[56:32], sema_va[31:0], payload |
| `LAUNCH_DMA` | 0x300 | 1 | flags — emitted twice: 0x182 (copy) then 0x00c (release) |

### LAUNCH_DMA payload bits (subset)

| bits | field | value we use |
|---|---|---|
| [1:0] | DATA_TRANSFER_TYPE | 2 (NON_PIPELINED) |
| [2] | FLUSH_ENABLE | 1 (TRUE) |
| [4:3] | SEMAPHORE_TYPE | 1 (RELEASE_ONE_WORD) |
| [7] | SRC_MEMORY_LAYOUT | 1 (PITCH) |
| [8] | DST_MEMORY_LAYOUT | 1 (PITCH) |
| [12] | SRC_TYPE | 0 (VIRTUAL) |
| [13] | DST_TYPE | 0 (VIRTUAL) |
| [25] | FLUSH_TYPE | 0 (SYS) |

Composite value: 0x0000018e.

### USERD layout (HopperAControlGPFifo, clc86f.h)

| Offset | Field | Direction |
|---|---|---|
| 0x88 | GPGet | GPU writes, CPU reads |
| 0x8c | GPPut | CPU writes, PBDMA reads |

### VF doorbell

| Offset | Name | Semantic |
|---|---|---|
| 0x30090 (VF-block relative) | NV_VIRTUAL_FUNCTION_DOORBELL | write work-submit-token to kick PBDMA |
| BAR0 + 0xBB0090 | same register, bare metal | `virtualRegPhysOffset` (0xB80000) + 0x30090 — see §11 |
| 0x30090 - 0x30000 = 0x90 | offset within HOPPER_USERMODE_A mapping | CPU VA for doorbell |

---

## Appendix B — Visual timing reference

A single transfer in numbers (H100 PCIe, 4 MiB D2H, measured in mc_demo):

```
CPU submission time         100 ns  ─┐
GPU wake-up latency        3000 ns   ├─ fixed overhead ~4 µs
PBDMA fetch + decode       1000 ns  ─┘

Data transfer (4 MiB, ~134 µs end-to-end less overhead) ~120 µs

Flush + sema write         ~1 µs  ─┐
Sema visibility to CPU     ~1 µs   ├─ completion overhead ~10 µs
CPU poll latency          ~few µs ─┘

─────────────────────────────────────────────────────
Total:                     ~134 µs = 0.13 ms

Matches the 31.3 GB/s D2H observed at 4 MiB per `findings.md §11`
bandwidth table — 4 MiB in 134 µs is 31.3 GB/s.  Throughput climbs to
~54 GB/s at 64 MiB and ~55 GB/s from 256 MiB up, once the fixed
per-transfer overhead amortizes out.
```

Fixed overhead (~15 µs) is a visible fraction at 4 MiB — about 11 % of the ~134 µs
total.  Larger transfers amortize it away: at 64 MiB it is about 1 %.  Measured D2H
climbs to ~54 GB/s at 64 MiB and ~55.5 GB/s at 1 GiB, which is about 88 % of the
~63 GB/s per-direction Gen5 ceiling; it matches H2D from 256 MiB up and sits within
a couple of percent of it at 64 MiB.  See `findings.md §11`.

---

*All bit-fields verified against
`src/common/sdk/nvidia/inc/class/clc86f.h` and `clc8b5.h` from
open-gpu-kernel-modules 610.43.02.*

---

## Appendix C — Empirical per-channel BAR1 slot layout used by libcuda

Added as part of the kernel-side doorbell watchpoint work (see
`findings.md §12`). On a 4 MiB CUDA round-trip run, libcuda
creates ONE 2 MiB `/dev/nvidia0 rw-s` BAR1 mapping per context and
packs all 20 channels' per-channel state into it as consecutive 12 KiB
slots. Observed directly from kernel-side `ioremap_wc` + sweep + RAMFC
read:

```
libcuda's 2 MiB BAR1 mapping
─────────────────────────────────────
│ chid=10  GPFIFO ring (1024 × 8 B)   │ ← RAMFC GP_BASE for chid=10 points here
│           8 KiB                     │   GPU VA = VMA_BASE + 0x0000
├─────────────────────────────────────┤
│ chid=10  HopperAControlGPFifo (USERD) │ ← pUserdMemDesc.subMemOffset = 0x2000
│           0x200 bytes + 0xE00 pad    │
├─────────────────────────────────────┤ ← +0x3000  (12 KiB per channel)
│ chid=11  GPFIFO ring    (8 KiB)      │
├─────────────────────────────────────┤
│ chid=11  USERD          (4 KiB page) │
├─────────────────────────────────────┤
    ... stride 0x3000 per channel ...
├─────────────────────────────────────┤
│ chid=29  GPFIFO ring                │
├─────────────────────────────────────┤
│ chid=29  USERD                      │ ← last observed at offset 0x3b000
├─────────────────────────────────────┤
│ unused (~1.76 MiB, probably reserved)│
└─────────────────────────────────────┘
```

Key facts:

- GPFIFO ring entry count for every channel is **1024**, encoded as
  `NV_PBDMA_GP_INFO_LIMIT2 = 10` in RAMFC (1 << 10 entries).
- Per-channel stride is exactly **0x3000** (12 KiB = 3 × 4 KiB pages).
- USERD starts **0x2000 bytes** after its channel's GPFIFO ring start
  — i.e., right after the 8 KiB ring ends.
- Pushbuffer is **NOT** in this 2 MiB mapping. The 240 KiB of slots
  leave the remaining ~1.76 MiB unused inside this VMA. Pushbuffer
  bytes live in sysmem reached through a different libcuda mapping
  (likely one of the 27 `/dev/nvidiactl` sysmem VMAs per process).

**Paper Finding 1 (GPU VA == user VA under UVM) verified for both
USERD and GPFIFO**: `memdescGetPhysAddrs(AT_GPU)` on a sub-memdesc
hands back a vidmem byte offset unrelated to this layout, but
`pUserdMemDesc->subMemOffset` + `vma->vm_start` (captured at mmap
time) gives the correct kernel VA via `user_va_start + subMemOffset`.
Similarly, RAMFC's `NV_RAMFC_GP_BASE_HI | NV_RAMFC_GP_BASE` is a GPU
VA that equals the userspace VA libcuda mapped — readable via
`vma->vm_start + (gpu_va - user_va_start)`.

libcuda packs 20+ channels this way, each in its own ~12 KiB BAR1
slot inside a 2 MiB `/dev/nvidia0 rw-s` mapping.  mc uses the
same FBMEM-shared-with-GPFIFO layout for **its single channel** but
in a monolithic 2 MiB `gpu_ctl` allocation (GPFIFO at offset 0,
USERD at offset 0x2000) rather than libcuda's multi-slot packing.
The BAR1 CPU alias of `gpu_ctl` is placed in the VA pool at
`0x200000000+` so GPU VA == CPU VA (Paper F1, see `findings.md §13`).
This BAR1-apertured FBMEM layout applies specifically to CUDA 13 +
driver 610 + Hopper — earlier GPUs put USERD in sysmem.

This layout was **not** declared anywhere in the public SDK — it was
reverse-engineered by instrumenting the kernel resolver. Future CUDA
versions may shuffle the slot order or insert padding; the safe
invariant is `pUserdMemDesc->subMemOffset` which comes straight from
RM and tracks whatever libcuda actually did.
