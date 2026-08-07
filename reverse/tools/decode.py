#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
decode.py — NVC8B5 (Hopper DMA Copy Engine) method stream encoder / decoder.

Usage modes:
  1) Encode (generate expected bytes from parameters):
       python3 decode.py encode <src_va_hex> <dst_va_hex> <size_bytes> [sema_va_hex [sema_payload]]
       e.g.: python3 decode.py encode 0x76ed0c000000 0x76ecfc000000 268435456

  2) Decode (disassemble raw bytes from a binary file or hex dump):
       python3 decode.py decode <file.bin> [offset_hex]

  3) Diff (decode the changed region between pre/post snapshot pair):
       python3 decode.py diff <pre.bin> <post.bin>

Background:
  On Hopper (H100), cudaMemcpy D2H builds a small method stream — 60 bytes
  for libcuda, 72 for mc, in both cases split across two LAUNCH_DMAs (the
  copy, then a semaphore-only release; see
  docs/gpfifo_pushbuffer_reference.md §9) —
  in the CUDA runtime's pushbuffer and submits it to a CE (Copy Engine) channel
  via a 2-word GPFIFO entry + MMIO doorbell write.  The method stream uses the
  HOPPER_DMA_COPY_A (0xC8B5) class.  The pushbuffer lives in WC (write-combine)
  host memory; despite the mapping, a CPU read does return what was written —
  verified against the kernel watchpoint's own read of the same addresses.
  What cannot be read back is the VF doorbell register itself: it has no
  backing store, so a read returns zero whatever was written (see
  "Rebuilding CUDA From Scratch" Part 4).

  This script:
    - Encodes the expected method stream given src/dst VAs and transfer size,
    - Decodes arbitrary 32-bit word sequences as NVC8B5 method headers + data,
    - Can diff pre/post snapshot pairs to find the changed region.
"""

import struct
import sys
import os

# ── NVC86F GPFIFO channel method header encoding ───────────────────────────
# On Hopper the PBDMA parses the Kepler-style method header:
#   bits [31:29] = type: 1=INCR (incremental), 3=INCR_ONCE?, 4=ONE (non-incr)
#                        5=IMMD (inline data in header), 7=IMMD2
#   bits [28:16] = count (for INCR) or inline value (for IMMD)
#   bits [15:13] = subchannel (4 for CE — NVA06F_SUBCHANNEL_COPY_ENGINE)
#   bits [12:0]  = method_address >> 2
METHOD_TYPE_INCR = 0x1   # incremental: N data dwords follow, methods at addr, addr+4, addr+8...
METHOD_TYPE_ONE  = 0x3   # non-incremental: N dwords all go to same method
METHOD_TYPE_IMMD = 0x5   # inline: data in bits[28:16], no following dwords

def make_header(method_addr, count, type_=METHOD_TYPE_INCR, subchan=0):
    assert method_addr & 3 == 0, f"method_addr {method_addr:#x} must be 4-byte aligned"
    return (type_ << 29) | (count << 16) | (subchan << 13) | (method_addr >> 2)

def decode_header(word):
    type_   = (word >> 29) & 0x7
    count   = (word >> 16) & 0x1fff
    subchan = (word >> 13) & 0x7
    maddr   = (word & 0x1fff) << 2
    return type_, count, subchan, maddr

# ── NVC8B5 (HOPPER_DMA_COPY_A) method register offsets ────────────────────
# From src/common/sdk/nvidia/inc/class/clc8b5.h
NVC8B5_SET_SEMAPHORE_A           = 0x0240   # sema VA [56:32]
NVC8B5_SET_SEMAPHORE_B           = 0x0244   # sema VA [31:0]
NVC8B5_SET_SEMAPHORE_PAYLOAD     = 0x0248   # value written on completion
NVC8B5_SET_SRC_PHYS_MODE         = 0x0260   # 0=LOCAL_FB, 1=COHERENT_SYSMEM
NVC8B5_SET_DST_PHYS_MODE         = 0x0264
NVC8B5_LAUNCH_DMA                = 0x0300   # trigger — must be last
NVC8B5_OFFSET_IN_UPPER           = 0x0400   # src VA [56:32]
NVC8B5_OFFSET_IN_LOWER           = 0x0404   # src VA [31:0]
NVC8B5_OFFSET_OUT_UPPER          = 0x0408   # dst VA [56:32]
NVC8B5_OFFSET_OUT_LOWER          = 0x040c   # dst VA [31:0]
NVC8B5_LINE_LENGTH_IN            = 0x0418   # transfer size in bytes (1D mode)

METHOD_NAMES = {
    NVC8B5_SET_SEMAPHORE_A:       "SET_SEMAPHORE_A",
    NVC8B5_SET_SEMAPHORE_B:       "SET_SEMAPHORE_B",
    NVC8B5_SET_SEMAPHORE_PAYLOAD: "SET_SEMAPHORE_PAYLOAD",
    NVC8B5_SET_SRC_PHYS_MODE:     "SET_SRC_PHYS_MODE",
    NVC8B5_SET_DST_PHYS_MODE:     "SET_DST_PHYS_MODE",
    NVC8B5_LAUNCH_DMA:            "LAUNCH_DMA",
    NVC8B5_OFFSET_IN_UPPER:       "OFFSET_IN_UPPER",
    NVC8B5_OFFSET_IN_LOWER:       "OFFSET_IN_LOWER",
    NVC8B5_OFFSET_OUT_UPPER:      "OFFSET_OUT_UPPER",
    NVC8B5_OFFSET_OUT_LOWER:      "OFFSET_OUT_LOWER",
    NVC8B5_LINE_LENGTH_IN:        "LINE_LENGTH_IN",
}

# LAUNCH_DMA flag decoding (bits from clc8b5.h)
def decode_launch_dma(val):
    transfer_type = val & 0x3          # bits [1:0]
    flush_en      = (val >> 2) & 0x1   # bit 2
    sema_type     = (val >> 3) & 0x3   # bits [4:3]
    src_layout    = (val >> 7) & 0x1   # bit 7: 0=blocklinear, 1=pitch
    dst_layout    = (val >> 8) & 0x1   # bit 8
    multi_line    = (val >> 9) & 0x1   # bit 9
    src_type      = (val >> 12) & 0x1  # bit 12: 0=VIRTUAL, 1=PHYSICAL
    dst_type      = (val >> 13) & 0x1  # bit 13
    flush_type    = (val >> 25) & 0x1  # bit 25: 0=SYS, 1=GL

    xfer_names = {0: "NONE", 1: "PIPELINED", 2: "NON_PIPELINED"}
    sema_names = {0: "NONE", 1: "RELEASE_ONE_WORD_SEMAPHORE",
                  2: "RELEASE_SEMAPHORE_WITH_TIMESTAMP", 3: "?"}
    parts = [
        f"DATA_TRANSFER_TYPE={xfer_names.get(transfer_type, str(transfer_type))}",
        f"FLUSH_ENABLE={'TRUE' if flush_en else 'FALSE'}",
        f"SEMAPHORE_TYPE={sema_names.get(sema_type, str(sema_type))}",
        f"SRC_MEMORY_LAYOUT={'PITCH' if src_layout else 'BLOCKLINEAR'}",
        f"DST_MEMORY_LAYOUT={'PITCH' if dst_layout else 'BLOCKLINEAR'}",
        f"MULTI_LINE_ENABLE={'TRUE' if multi_line else 'FALSE'}",
        f"SRC_TYPE={'PHYSICAL' if src_type else 'VIRTUAL'}",
        f"DST_TYPE={'PHYSICAL' if dst_type else 'VIRTUAL'}",
        f"FLUSH_TYPE={'GL' if flush_type else 'SYS'}",
    ]
    return " | ".join(parts)

# ── GP entry encoding (NVC86F) ─────────────────────────────────────────────
# From src/common/sdk/nvidia/inc/class/clc86f.h lines 168-189
# ENTRY0: bits[31:2] = pushbuffer_base >> 2, bit[0] = fetch
# ENTRY1: bits[7:0] = base_hi (bits[39:32] of 40-bit PB addr),
#         bits[30:10] = length in dwords, bit[31] = sync

def encode_gp_entry(pb_gpu_va, length_bytes):
    """Return (entry0, entry1) 32-bit words for a GPFIFO entry."""
    assert length_bytes % 4 == 0
    length_dwords = length_bytes // 4
    entry0 = (pb_gpu_va & 0xfffffffc) >> 2   # bits[31:2] of low 32 bits
    # Hopper uses bit extension: base_hi in ENTRY1[7:0] covers VA[39:32]
    # For VAs in the 48-bit user range, bits[63:32] fit in a wider field
    # ENTRY1 format: [7:0]=GET_HI (va[39:32]), [30:10]=LENGTH, [31]=SYNC
    base_hi = (pb_gpu_va >> 32) & 0xff
    entry1  = (base_hi & 0xff) | ((length_dwords & 0x1fffff) << 10)
    return entry0, entry1

def decode_gp_entry(e0, e1):
    base_lo     = (e0 & 0xffffffff) << 2   # this is the address, not shifted
    base_hi     = e1 & 0xff
    length_dw   = (e1 >> 10) & 0x1fffff
    sync        = (e1 >> 31) & 1
    pb_va       = (base_hi << 32) | (e0 << 2)   # reconstructed VA
    return pb_va, length_dw * 4, sync


# ── Method stream encoder ──────────────────────────────────────────────────

def encode_d2h_method_stream(src_va, dst_va, nbytes,
                              sema_va=None, sema_payload=1):
    """
    Build the NVC8B5 method stream for a 1D device-to-host DMA.

    NOTE: this encoder emits the *fused* single-LAUNCH_DMA form (copy +
    flush + semaphore release in one launch).  It is a convenience for
    round-tripping the decoder, not a reproduction of what libcuda or mc
    put on the wire — both of those split the work across two launches.

    Returns a bytes object (little-endian 32-bit dwords).

    The expected method sequence for a synchronous D2H transfer:
      1. OFFSET_IN_UPPER + OFFSET_IN_LOWER  — source device VA
      2. OFFSET_OUT_UPPER + OFFSET_OUT_LOWER — destination host VA
      3. LINE_LENGTH_IN                      — byte count
      4. SET_SEMAPHORE_A + _B + _PAYLOAD     — completion notification
      5. LAUNCH_DMA                          — trigger (must be last)
    """
    dwords = []

    def INCR(method, *values):
        dwords.append(make_header(method, len(values), METHOD_TYPE_INCR))
        dwords.extend(values)

    # Source: device HBM VA
    INCR(NVC8B5_OFFSET_IN_UPPER,
         (src_va >> 32) & 0x01ffffff,   # 25-bit field
         src_va & 0xffffffff)

    # Destination: pinned host VA
    INCR(NVC8B5_OFFSET_OUT_UPPER,
         (dst_va >> 32) & 0x01ffffff,
         dst_va & 0xffffffff)

    # Transfer size (1D mode: LINE_LENGTH_IN = byte count)
    INCR(NVC8B5_LINE_LENGTH_IN,
         nbytes & 0xffffffff)

    # Completion semaphore
    if sema_va is not None:
        INCR(NVC8B5_SET_SEMAPHORE_A,
             (sema_va >> 32) & 0x01ffffff,
             sema_va & 0xffffffff,
             sema_payload & 0xffffffff)

    # LAUNCH_DMA — trigger the copy engine
    # Flags for a standard virtual-address D2H non-pipelined transfer:
    #   DATA_TRANSFER_TYPE = NON_PIPELINED (2)
    #   FLUSH_ENABLE = TRUE (1 << 2)
    #   SEMAPHORE_TYPE = RELEASE_ONE_WORD_SEMAPHORE (1 << 3), if sema given
    #   SRC_MEMORY_LAYOUT = PITCH (1 << 7)
    #   DST_MEMORY_LAYOUT = PITCH (1 << 8)
    #   MULTI_LINE_ENABLE = FALSE (0 << 9)
    #   SRC_TYPE = VIRTUAL (0 << 12)
    #   DST_TYPE = VIRTUAL (0 << 13)
    launch_flags  = 2           # NON_PIPELINED
    launch_flags |= 1 << 2      # FLUSH_ENABLE = TRUE
    if sema_va is not None:
        launch_flags |= 1 << 3  # RELEASE_ONE_WORD_SEMAPHORE
    launch_flags |= 1 << 7      # SRC_MEMORY_LAYOUT = PITCH
    launch_flags |= 1 << 8      # DST_MEMORY_LAYOUT = PITCH
    # SRC_TYPE and DST_TYPE = 0 → VIRTUAL (no flags needed)

    INCR(NVC8B5_LAUNCH_DMA, launch_flags)

    return struct.pack(f"<{len(dwords)}I", *dwords)


# ── Method stream decoder / disassembler ──────────────────────────────────

def disassemble(data, base_offset=0, indent=""):
    """
    Disassemble bytes as a sequence of NVC86F-encoded method headers + data.
    Prints annotated listing.  Returns a list of (method_addr, value) pairs.
    """
    results = []
    i = 0
    while i + 4 <= len(data):
        word = struct.unpack_from("<I", data, i)[0]
        file_off = base_offset + i

        mtype, count, subchan, maddr = decode_header(word)
        mname = METHOD_NAMES.get(maddr, f"method_0x{maddr:04x}")

        type_names = {0: "NON_INCR", 1: "INCR", 3: "ONE", 4: "IMMD2", 5: "IMMD", 7: "IMMD3"}
        typename = type_names.get(mtype, f"type{mtype}")

        if mtype == METHOD_TYPE_INCR and 0 < count <= 32 and maddr < 0x1000:
            # Looks like a valid incremental method header
            print(f"{indent}+{file_off:#08x}  HEADER [{typename}] method=0x{maddr:04x} "
                  f"({mname}) count={count} subchan={subchan}")
            for j in range(count):
                if i + 4 + j * 4 + 4 > len(data):
                    print(f"{indent}  [truncated]")
                    break
                dval = struct.unpack_from("<I", data, i + 4 + j * 4)[0]
                cur_maddr = maddr + j * 4
                cur_mname = METHOD_NAMES.get(cur_maddr, f"method_0x{cur_maddr:04x}")
                annotation = _annotate(cur_maddr, dval)
                print(f"{indent}  +{base_offset+i+4+j*4:#08x}  {cur_mname:30s} = {dval:#010x}"
                      + (f"  ({annotation})" if annotation else ""))
                results.append((cur_maddr, dval))
            i += 4 + count * 4
        else:
            # Not a recognizable header — show raw dword
            print(f"{indent}+{file_off:#08x}  raw={word:#010x}")
            i += 4

    return results

def _annotate(maddr, val):
    """Return a human-readable annotation for a method's data value."""
    if maddr == NVC8B5_LAUNCH_DMA:
        return decode_launch_dma(val)
    if maddr in (NVC8B5_OFFSET_IN_UPPER, NVC8B5_OFFSET_OUT_UPPER,
                 NVC8B5_SET_SEMAPHORE_A):
        return f"VA bits [56:32] = 0x{val:07x}"
    if maddr == NVC8B5_OFFSET_IN_LOWER:
        return "src VA lo 32 bits"
    if maddr == NVC8B5_OFFSET_OUT_LOWER:
        return "dst VA lo 32 bits"
    if maddr == NVC8B5_LINE_LENGTH_IN:
        return f"{val} bytes = {val // (1024*1024)} MiB" if val >= 1024*1024 else f"{val} bytes"
    if maddr == NVC8B5_SET_SEMAPHORE_PAYLOAD:
        return f"completion value = {val}"
    return ""


# ── CLI entry points ───────────────────────────────────────────────────────

def cmd_encode(args):
    if len(args) < 3:
        print("Usage: decode.py encode <src_va_hex> <dst_va_hex> <size_bytes> "
              "[sema_va_hex [sema_payload]]")
        sys.exit(1)
    src_va   = int(args[0], 16)
    dst_va   = int(args[1], 16)
    nbytes   = int(args[2], 0)
    sema_va  = int(args[3], 16) if len(args) > 3 else None
    sema_pay = int(args[4], 0)  if len(args) > 4 else 1

    data = encode_d2h_method_stream(src_va, dst_va, nbytes, sema_va, sema_pay)

    print(f"D2H method stream for:")
    print(f"  src  = {src_va:#016x}  (device HBM)")
    print(f"  dst  = {dst_va:#016x}  (pinned host)")
    print(f"  size = {nbytes} bytes ({nbytes//(1024*1024)} MiB)")
    if sema_va:
        print(f"  sema = {sema_va:#016x}  payload={sema_pay}")
    print(f"  total= {len(data)} bytes ({len(data)//4} dwords)\n")
    print("Disassembly:")
    results = disassemble(data, indent="  ")

    print(f"\nRaw hex ({len(data)} bytes):")
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hexstr = " ".join(f"{b:02x}" for b in chunk)
        print(f"  {i:04x}:  {hexstr}")

    # Also encode the corresponding GP entry
    pb_gpu_va = 0xDEADBEEF00000000  # placeholder — actual VA known at runtime
    e0, e1 = encode_gp_entry(pb_gpu_va, len(data))
    print(f"\nGP entry (assuming pushbuffer GPU VA = <runtime>):")
    print(f"  ENTRY0 = (pb_gpu_va & 0xfffffffc) >> 2  [bits 31:2 of low 32]")
    print(f"  ENTRY1 = (pb_gpu_va[39:32] & 0xff) | ({len(data)//4} << 10)")
    print(f"         = 0x????00 | 0x{(len(data)//4) << 10:08x}")


def cmd_decode(args):
    if len(args) < 1:
        print("Usage: decode.py decode <file.bin> [offset_hex]")
        sys.exit(1)
    path   = args[0]
    offset = int(args[1], 16) if len(args) > 1 else 0
    data   = open(path, "rb").read()
    chunk  = data[offset:]
    print(f"Disassembling {path}  offset={offset:#x}  ({len(chunk)} bytes remaining)\n")
    disassemble(chunk, base_offset=offset)


def cmd_diff(args):
    if len(args) < 2:
        print("Usage: decode.py diff <pre.bin> <post.bin>")
        sys.exit(1)
    pre  = open(args[0], "rb").read()
    post = open(args[1], "rb").read()
    if len(pre) != len(post):
        print(f"Warning: files differ in size ({len(pre)} vs {len(post)})")
    size = min(len(pre), len(post))

    # Find changed dword ranges
    changed_offsets = [i for i in range(0, size, 4) if pre[i:i+4] != post[i:i+4]]
    if not changed_offsets:
        print("No changes found between pre and post snapshots.")
        return

    print(f"Found {len(changed_offsets)} changed dword(s):\n")

    # Group into contiguous runs
    runs = []
    rs = changed_offsets[0]
    rp = changed_offsets[0]
    for o in changed_offsets[1:]:
        if o == rp + 4:
            rp = o
        else:
            runs.append((rs, rp + 4))
            rs = rp = o
    runs.append((rs, rp + 4))

    for run_start, run_end in runs:
        print(f"Changed region: +{run_start:#08x} .. +{run_end-1:#08x} ({run_end-run_start} bytes)")
        print("  PRE  (content before cudaMemcpy):")
        disassemble(pre[run_start:run_end], base_offset=run_start, indent="    ")
        print("  POST (content after cudaMemcpy):")
        disassemble(post[run_start:run_end], base_offset=run_start, indent="    ")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd  = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "encode":
        cmd_encode(args)
    elif cmd == "decode":
        cmd_decode(args)
    elif cmd == "diff":
        cmd_diff(args)
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: encode, decode, diff")
        sys.exit(1)


if __name__ == "__main__":
    main()
