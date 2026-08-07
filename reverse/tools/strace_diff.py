#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
strace_diff.py — compare the NVIDIA-driver ioctl sequences of two
captures (e.g., CUDA vs. mc_demo) and highlight where they diverge.

Background:
  mc reproduces cudaMemcpy at the raw ioctl level without the CUDA
  runtime.  Comparing its ioctl sequence against CUDA's on the same
  workload reveals what setup CUDA does that mc doesn't (or vice
  versa).  This is the "why doesn't CUDA hit the bug?" investigation
  tool.

Input formats:

  strace output (default):  -f -e trace=ioctl,openat,mmap,munmap,close
      Each ioctl is opaque beyond the ioctl cmd byte — the param struct
      content is not visible.  Fine for "which ioctls are issued" but
      blind to "which handle is which buffer."

  ftrace text (--input-format ftrace):  /sys/kernel/debug/tracing/trace
      Richer: our kernel-side mc1 records dump hclient, hmemory,
      class, size and flags for rm/alloc, rm/map_memory and
      uvm/map_external.  This lets us infer each handle's role
      (d_buf / h_buf / pb / gpfifo / sema / staging / usermode / …)
      and annotate every ioctl line with the role it targets.

Decoding:
  Each strace ioctl line looks like
      <pid> ioctl(<fd>, _IOC(<dir>, <type>, <cmd>, <size>), <arg>) = <ret>
  We track which /dev/nvidia* path each fd points at (via openat) and
  then decode cmd based on path:

    /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-caps/*
        type = 0x46, cmd = RM escape byte (NV_ESC_RM_ALLOC=0x2b,
        NV_ESC_RM_CONTROL=0x2a, NV_ESC_RM_MAP_MEMORY=0x4e, etc.)

    /dev/nvidia-uvm
        type = 0, cmd = UVM_IOCTL_BASE(i) = i (on Linux), so e.g.
        UVM_MAP_EXTERNAL_ALLOCATION = 33 = 0x21.
        UVM_INITIALIZE is the one exception: it uses a raw 0x30000001.

Usage:
  strace_diff.py <cuda.strace> <mc_demo.strace>            # aligned diff
  strace_diff.py --summary <cuda.strace> <mc_demo.strace>  # counts only
  strace_diff.py --only <cuda.strace> <mc_demo.strace>     # list ioctls
                                                              unique to
                                                              each side
  strace_diff.py --input-format ftrace --roles \\
      <cuda.ftrace> <mc_demo.ftrace>                        # per-handle
                                                              role table
  strace_diff.py --input-format ftrace --missing-ioctls \\
      <cuda.ftrace> <mc_demo.ftrace>                        # list
                                                              ioctls
                                                              issued by
                                                              only one
                                                              side with
                                                              decoded
                                                              params
                                                              where
                                                              available
  strace_diff.py --input-format ftrace --handle-history \\
      <cuda.ftrace> <mc_demo.ftrace>                        # per-handle
                                                              timeline of
                                                              events with
                                                              decoded
                                                              NVOS32/NVOS33
                                                              fields
  strace_diff.py --input-format ftrace --handle-diff \\
      [--match-role <role>] \\
      <cuda.ftrace> <mc_demo.ftrace>                        # role-matched
                                                              side-by-side
                                                              parameter
                                                              diff
"""

import argparse
import re
import sys
from collections import Counter, namedtuple
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ── cmd/escape tables ──────────────────────────────────────────────────────

# RM escapes — sourced directly from two headers so the names match what
# the kernel actually dispatches on.  DO NOT hand-edit values; verify
# against the headers and paste the grep output.
#
# src/nvidia/arch/nvalloc/unix/include/nv_escape.h (RM object/mem escapes)
RM_ESCAPES: Dict[int, str] = {
    0x27: "NV_ESC_RM_ALLOC_MEMORY",
    0x28: "NV_ESC_RM_ALLOC_OBJECT",
    0x29: "NV_ESC_RM_FREE",
    0x2A: "NV_ESC_RM_CONTROL",
    0x2B: "NV_ESC_RM_ALLOC",
    0x34: "NV_ESC_RM_DUP_OBJECT",
    0x35: "NV_ESC_RM_SHARE",
    0x39: "NV_ESC_RM_I2C_ACCESS",
    0x41: "NV_ESC_RM_IDLE_CHANNELS",
    0x4A: "NV_ESC_RM_VID_HEAP_CONTROL",
    0x4D: "NV_ESC_RM_ACCESS_REGISTRY",
    0x4E: "NV_ESC_RM_MAP_MEMORY",
    0x4F: "NV_ESC_RM_UNMAP_MEMORY",
    0x52: "NV_ESC_RM_GET_EVENT_DATA",
    0x54: "NV_ESC_RM_ALLOC_CONTEXT_DMA2",
    0x56: "NV_ESC_RM_ADD_VBLANK_CALLBACK",
    0x57: "NV_ESC_RM_MAP_MEMORY_DMA",
    0x58: "NV_ESC_RM_UNMAP_MEMORY_DMA",
    0x59: "NV_ESC_RM_BIND_CONTEXT_DMA",
    0x5C: "NV_ESC_RM_EXPORT_OBJECT_TO_FD",
    0x5D: "NV_ESC_RM_IMPORT_OBJECT_FROM_FD",
    0x5E: "NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO",
    0x5F: "NV_ESC_RM_LOCKLESS_DIAGNOSTIC",
    # kernel-open/common/inc/nv-ioctl-numbers.h (NV_IOCTL_BASE = 200 = 0xC8).
    0xC8: "NV_ESC_CARD_INFO",               # base + 0
    0xC9: "NV_ESC_REGISTER_FD",              # base + 1
    0xCE: "NV_ESC_ALLOC_OS_EVENT",           # base + 6
    0xCF: "NV_ESC_FREE_OS_EVENT",            # base + 7
    0xD1: "NV_ESC_STATUS_CODE",              # base + 9
    0xD2: "NV_ESC_CHECK_VERSION_STR",        # base + 10
    0xD3: "NV_ESC_IOCTL_XFER_CMD",           # base + 11
    0xD4: "NV_ESC_ATTACH_GPUS_TO_FD",        # base + 12
    0xD5: "NV_ESC_QUERY_DEVICE_INTR",        # base + 13
    0xD6: "NV_ESC_SYS_PARAMS",               # base + 14
    # base + 15 (0xD7) and +16 (0xD8) are not in the header at 610.43.02 —
    # leave undecoded if hit.  If the strace diff flags them, chase it down.
    0xD9: "NV_ESC_EXPORT_TO_DMABUF_FD",      # base + 17
    0xDA: "NV_ESC_WAIT_OPEN_COMPLETE",       # base + 18
}

# UVM cmds (kernel-open/nvidia-uvm/uvm_ioctl.h).  UVM_IOCTL_BASE(i) = i
# on Linux, so the decimal cmd number in _IOC(...) is the same as the
# #define value in the header.
UVM_CMDS: Dict[int, str] = {
    # INITIALIZE (0x30000001) is handled specially; see parse_ioctl_line.
    1:  "UVM_RESERVE_VA",
    2:  "UVM_RELEASE_VA",
    3:  "UVM_REGION_COMMIT",
    4:  "UVM_REGION_DECOMMIT",
    5:  "UVM_REGION_SET_STREAM",
    6:  "UVM_SET_STREAM_RUNNING",
    7:  "UVM_SET_STREAM_STOPPED",
    8:  "UVM_MIGRATE_LEGACY",
    23: "UVM_CREATE_RANGE_GROUP",
    24: "UVM_DESTROY_RANGE_GROUP",
    25: "UVM_REGISTER_GPU_VASPACE",
    26: "UVM_UNREGISTER_GPU_VASPACE",
    27: "UVM_REGISTER_CHANNEL",
    28: "UVM_UNREGISTER_CHANNEL",
    29: "UVM_ENABLE_PEER_ACCESS",
    30: "UVM_DISABLE_PEER_ACCESS",
    31: "UVM_SET_RANGE_GROUP",
    32: "UVM_MAP_EXTERNAL_ALLOCATION_V1",  # deprecated variant
    33: "UVM_MAP_EXTERNAL_ALLOCATION",
    34: "UVM_FREE",
    35: "UVM_MEM_MAP",
    37: "UVM_REGISTER_GPU",
    38: "UVM_UNREGISTER_GPU",
    39: "UVM_PAGEABLE_MEM_ACCESS",
    40: "UVM_SET_PREFERRED_LOCATION",
    41: "UVM_UNSET_PREFERRED_LOCATION",
    42: "UVM_ENABLE_READ_DUPLICATION",
    43: "UVM_DISABLE_READ_DUPLICATION",
    44: "UVM_SET_ACCESSED_BY",
    45: "UVM_UNSET_ACCESSED_BY",
    49: "UVM_MIGRATE",
    50: "UVM_MIGRATE_RANGE_GROUP",
    51: "UVM_TOOLS_READ_PROCESS_MEMORY",
    52: "UVM_TOOLS_WRITE_PROCESS_MEMORY",
    53: "UVM_MAP_DYNAMIC_PARALLELISM_REGION",
    54: "UVM_UNMAP_DYNAMIC_PARALLELISM_REGION",
    55: "UVM_ALLOC_SEMAPHORE_POOL",
    56: "UVM_PAGEABLE_MEM_ACCESS_ON_GPU",
    57: "UVM_DISABLE_SYSTEM_WIDE_ATOMICS",
    58: "UVM_ENABLE_SYSTEM_WIDE_ATOMICS",
    60: "UVM_TOOLS_INIT_EVENT_TRACKER",
    61: "UVM_TOOLS_SET_NOTIFICATION_THRESHOLD",
    62: "UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS",
    63: "UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS",
    65: "UVM_MAP_EXTERNAL_SPARSE",
    66: "UVM_UNMAP_EXTERNAL",
    67: "UVM_TOOLS_FLUSH_EVENTS",
    68: "UVM_ALLOC_DEVICE_P2P",
    69: "UVM_CLEAR_ALL_PAGES_ACCESSED_BY_SM",
    70: "UVM_POPULATE_PAGEABLE",
    71: "UVM_VALIDATE_VA_RANGE",
    72: "UVM_CREATE_EXTERNAL_RANGE_V1",
    73: "UVM_CREATE_EXTERNAL_RANGE",
    74: "UVM_MM_INITIALIZE_V1",
    75: "UVM_MM_INITIALIZE",
    76: "UVM_IS_8_SUPPORTED",
}

UVM_INITIALIZE = 0x30000001


# ── NVOS32 / NVOS33 flag decoders ──────────────────────────────────────────
#
# The ftrace alloc-params body contains the NVOS32 flag word at dword
# index 2 (offset +0x08 in NV_MEMORY_ALLOCATION_PARAMS) and attr/attr2
# DRF-encoded words at dwords 6 and 7 (offsets +0x18, +0x1C).  NVOS33
# MAP_MEMORY flags is a single dword captured directly on the
# MAP_MEMORY tracepoint.  The decoders below turn those raw dwords
# into human-readable strings for the --handle-history / --handle-diff
# output modes.
#
# All symbol tables are sourced from src/common/sdk/nvidia/inc/nvos.h.
# DO NOT hand-edit — paste from the header and the name maps will
# survive driver-version bumps.

# NVOS32 single-bit alloc flags (NVOS32_ALLOC_FLAGS_* in nvos.h).  Bits that are
# aliased across multiple meanings (0x02000000 = KERNEL_MAPPING_MAP
# OR MAXIMIZE_ADDRESS_SPACE) get the first name only — aliases are
# documented as TODO in the header.  If a future trace surfaces one
# of the aliased bits and we need to tell which meaning applies, we
# can reintroduce context-dependent naming.
NVOS32_ALLOC_FLAGS: List[Tuple[int, str]] = [
    (0x00000001, "IGNORE_BANK_PLACEMENT"),
    (0x00000002, "FORCE_MEM_GROWS_UP"),
    (0x00000004, "FORCE_MEM_GROWS_DOWN"),
    (0x00000008, "FORCE_ALIGN_HOST_PAGE"),
    (0x00000010, "FIXED_ADDRESS_ALLOCATE"),
    (0x00000020, "BANK_HINT"),
    (0x00000040, "BANK_FORCE"),
    (0x00000080, "ALIGNMENT_HINT"),
    (0x00000100, "ALIGNMENT_FORCE"),
    (0x00000200, "BANK_GROW_DOWN"),
    (0x00000400, "LAZY"),
    (0x00000800, "FORCE_REVERSE_ALLOC"),
    (0x00001000, "NO_SCANOUT"),
    (0x00002000, "PITCH_FORCE"),
    (0x00004000, "MEMORY_HANDLE_PROVIDED"),
    (0x00008000, "MAP_NOT_REQUIRED"),
    (0x00010000, "PERSISTENT_VIDMEM"),
    (0x00020000, "USE_BEGIN_END"),
    (0x00040000, "TURBO_CIPHER_ENCRYPTED"),
    (0x00080000, "VIRTUAL"),
    (0x00100000, "FORCE_INTERNAL_INDEX"),
    (0x00200000, "ZCULL_COVG_SPECIFIED"),
    (0x00400000, "EXTERNALLY_MANAGED"),
    (0x00800000, "FORCE_DEDICATED_PDE"),
    (0x01000000, "PROTECTED"),
    (0x02000000, "KERNEL_MAPPING_MAP"),       # aliased: MAXIMIZE_ADDRESS_SPACE
    (0x04000000, "SPARSE"),                    # aliased: USER_READ_ONLY
    (0x08000000, "DEVICE_READ_ONLY"),
    (0x10000000, "SKIP_RESOURCE_ALLOC"),
    (0x20000000, "PREFER_PTES_IN_SYSMEMORY"),
    (0x40000000, "SKIP_ALIGN_PAD"),            # aliased: WPR1
    (0x80000000, "ZCULL_DONT_ALLOCATE_SHARED_1X"),  # aliased: WPR2
]


def decode_nvos32_flags(flags: int) -> str:
    """Render NVOS32_ALLOC_FLAGS_* bitmask as a '|'-joined name list."""
    if flags == 0:
        return "0"
    names = [name for bit, name in NVOS32_ALLOC_FLAGS if flags & bit]
    known_mask = 0
    for bit, _ in NVOS32_ALLOC_FLAGS:
        known_mask |= bit
    unknown = flags & ~known_mask
    if unknown:
        names.append(f"0x{unknown:08x}?")
    return "|".join(names)


# NVOS32 ATTR DRF fields (NVOS32_ATTR_* in nvos.h).  Each entry is
# (hi, lo, name, {value: value_name}).  Values that default to 0 are
# skipped from output unless explicitly non-default.
NVOS32_ATTR_FIELDS: List[Tuple[int, int, str, Dict[int, str]]] = [
    (2, 0, "DEPTH", {
        0: "UNKNOWN", 1: "8", 2: "16", 3: "24",
        4: "32", 5: "64", 6: "128",
    }),
    (3, 3, "COMPR_COVG", {0: "DEFAULT", 1: "PROVIDED"}),
    (7, 4, "AA_SAMPLES", {}),  # not relevant to our workloads
    (9, 8, "GPU_CACHE_SNOOPABLE", {
        0: "MAPPING", 1: "OFF", 2: "ON", 3: "INVALID",
    }),
    (11, 10, "ZCULL", {
        0: "NONE", 1: "REQUIRED", 2: "ANY", 3: "SHARED",
    }),
    (13, 12, "COMPR", {
        0: "NONE", 1: "REQUIRED", 2: "ANY", 3: "DISABLE_PLC_ANY",
    }),
    (14, 14, "ALLOCATE_FROM_RESERVED_HEAP", {0: "NO", 1: "YES"}),
    (17, 16, "FORMAT", {
        0: "PITCH", 1: "SWIZZLED", 2: "BLOCK_LINEAR",
    }),
    (18, 18, "Z_TYPE", {0: "FIXED", 1: "FLOAT"}),
    (21, 19, "ZS_PACKING", {}),
    (22, 22, "TILED", {}),
    (24, 23, "PAGE_SIZE", {
        0: "DEFAULT", 1: "4KB", 2: "BIG", 3: "HUGE",
    }),
    (26, 25, "LOCATION", {
        0: "VIDMEM", 1: "PCI", 3: "ANY",
    }),
    (28, 27, "PHYSICALITY", {
        0: "DEFAULT", 1: "NONCONTIGUOUS",
        2: "CONTIGUOUS", 3: "ALLOW_NONCONTIGUOUS",
    }),
    (31, 29, "COHERENCY", {
        0: "UNCACHED", 1: "CACHED", 2: "WRITE_COMBINE",
        3: "WRITE_THROUGH", 4: "WRITE_PROTECT",
        5: "WRITE_BACK",
    }),
]

# NVOS32 ATTR2 DRF fields (NVOS32_ATTR2_* in nvos.h).  Only the fields we expect to
# see set on typical memory allocations.
NVOS32_ATTR2_FIELDS: List[Tuple[int, int, str, Dict[int, str]]] = [
    (1, 0, "ZBC", {
        0: "DEFAULT", 1: "PREFER_NO_ZBC",
        2: "PREFER_ZBC", 3: "REQUIRE_ONLY_ZBC",
    }),
    (3, 2, "GPU_CACHEABLE", {
        0: "DEFAULT", 1: "YES", 2: "NO", 3: "INVALID",
    }),
    (5, 4, "P2P_GPU_CACHEABLE", {
        0: "DEFAULT", 1: "YES", 2: "NO",
    }),
]


def _drf_extract(word: int, hi: int, lo: int) -> int:
    mask = (1 << (hi - lo + 1)) - 1
    return (word >> lo) & mask


def decode_drf_fields(
    word: int, fields: List[Tuple[int, int, str, Dict[int, str]]]
) -> str:
    """Decode a DRF-encoded dword into 'FIELD=NAME FIELD=0xN ...' text.

    Skips fields whose value is 0 *and* whose 0-value maps to DEFAULT —
    otherwise output balloons.  Always shows fields with a non-DEFAULT
    known value.
    """
    parts = []
    for hi, lo, name, values in fields:
        v = _drf_extract(word, hi, lo)
        if v == 0 and values.get(0) in (None, "DEFAULT", "NONE", "UNKNOWN",
                                         "VIDMEM", "PITCH", "FIXED"):
            continue  # skip zero-valued defaults
        if v in values:
            parts.append(f"{name}={values[v]}")
        else:
            parts.append(f"{name}=0x{v:x}")
    return " ".join(parts) if parts else "(defaults)"


def decode_nvos32_attr(attr: int) -> str:
    return decode_drf_fields(attr, NVOS32_ATTR_FIELDS)


def decode_nvos32_attr2(attr2: int) -> str:
    return decode_drf_fields(attr2, NVOS32_ATTR2_FIELDS)


# NVOS33 MAP_MEMORY flags (NVOS33_FLAGS_* in nvos.h).  CACHING_TYPE is a DRF
# field at bits 25:23 INSIDE the flags dword; it is NOT a separate
# field.  Other flags are single-bit or 2-bit DRF.  This decoder
# surfaces every field; non-default values get highlighted.
NVOS33_FLAGS_FIELDS: List[Tuple[int, int, str, Dict[int, str]]] = [
    (1, 0, "ACCESS", {
        0: "READ_WRITE", 1: "READ_ONLY", 2: "WRITE_ONLY",
    }),
    (4, 4, "PERSISTENT", {0: "DISABLE", 1: "ENABLE"}),
    (8, 8, "SKIP_SIZE_CHECK", {0: "DISABLE", 1: "ENABLE"}),
    (14, 14, "MEM_SPACE", {0: "CLIENT", 1: "USER"}),
    (16, 15, "MAPPING", {
        0: "DEFAULT", 1: "DIRECT", 2: "REFLECTED",
    }),
    (17, 17, "FIFO_MAPPING", {0: "DEFAULT", 1: "ENABLE"}),
    (18, 18, "MAP_FIXED", {0: "DISABLE", 1: "ENABLE"}),
    (19, 19, "RESERVE_ON_UNMAP", {0: "DISABLE", 1: "ENABLE"}),
    (21, 20, "BUS", {
        0: "DEFAULT", 1: "COHERENT_LINK", 2: "PCIE",
    }),
    (22, 22, "OS_DESCRIPTOR", {0: "DISABLE", 1: "ENABLE"}),
    (25, 23, "CACHING_TYPE", {
        0: "CACHED", 1: "UNCACHED", 2: "WRITECOMBINED",
        5: "WRITEBACK", 6: "DEFAULT", 7: "UNCACHED_WEAK",
    }),
    (26, 26, "ALLOW_MAPPING_ON_HCC", {0: "NO", 1: "YES"}),
]


def decode_nvos33_flags(flags: int) -> str:
    parts = []
    # Legacy single-bit: NV04_MAP_MEMORY_FLAGS_USER (nvos.h)
    if flags & 0x00004000:
        parts.append("LEGACY_USER")
    for hi, lo, name, values in NVOS33_FLAGS_FIELDS:
        v = _drf_extract(flags, hi, lo)
        # Skip uninteresting 0-valued defaults to reduce noise.
        if v == 0 and values.get(0) in (
            "READ_WRITE", "DISABLE", "CLIENT", "DEFAULT", "CACHED", "NO",
        ):
            continue
        if v in values:
            parts.append(f"{name}={values[v]}")
        else:
            parts.append(f"{name}=0x{v:x}")
    return " ".join(parts) if parts else "(defaults)"


# UVM_POPULATE_PAGEABLE flags (UVM_POPULATE_PAGEABLE_FLAG_* in uvm_ioctl.h).
UVM_POPULATE_PAGEABLE_FLAGS: List[Tuple[int, str]] = [
    (0x00000001, "ALLOW_MANAGED"),
    (0x00000002, "SKIP_PROT_CHECK"),
    (0x00000004, "ALLOW_SPECIAL"),
]


def decode_uvm_populate_pageable_flags(flags: int) -> str:
    if flags == 0:
        return "0"
    names = [n for bit, n in UVM_POPULATE_PAGEABLE_FLAGS if flags & bit]
    known = 0
    for bit, _ in UVM_POPULATE_PAGEABLE_FLAGS:
        known |= bit
    unknown = flags & ~known
    if unknown:
        names.append(f"0x{unknown:x}?")
    return "|".join(names)


# ── Per-UVM-cmd body decoders ──────────────────────────────────────────────
#
# Each decoder takes the accumulated dwords from a uvm_ioctl event and
# returns a dict of {field_name -> human_string}.  The decoder only
# inspects the dwords within its cmd's known struct size — bytes past
# that are arbitrary kernel-side scratch (the uvm/ioctl site in uvm.c
# dumps a fixed 512-byte prefix regardless of the actual params size).
#
# When a cmd has no decoder here, the body is not decoded — use
# --handle-history to view raw dwords.

def _u64_from_dwords(dwords: List[int], idx: int) -> int:
    """Read a little-endian u64 starting at dword `idx`."""
    if idx + 1 >= len(dwords):
        return 0
    return (dwords[idx + 1] << 32) | dwords[idx]


def decode_uvm_populate_pageable(dwords: List[int]) -> Dict[str, str]:
    """UVM_POPULATE_PAGEABLE_PARAMS (uvm_ioctl.h):
       {NvU64 base, NvU64 length, NvU32 flags, NV_STATUS rmStatus}
       = 24 bytes = 6 dwords.
    """
    base = _u64_from_dwords(dwords, 0)
    length = _u64_from_dwords(dwords, 2)
    flags = dwords[4] if len(dwords) > 4 else 0
    return {
        "base": f"0x{base:x}",
        "length": f"0x{length:x}",
        "flags": f"0x{flags:x} ({decode_uvm_populate_pageable_flags(flags)})",
    }


def decode_uvm_map_external(dwords: List[int]) -> Dict[str, str]:
    """UVM_MAP_EXTERNAL_ALLOCATION_PARAMS — layout per uvm_ioctl.h:
       {NvU64 base, NvU64 length, NvU64 offset,
        UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS],
        NvU64 gpuAttributesCount, NvS32 rmCtrlFd,
        NvU32 hClient, NvU32 hMemory, NV_STATUS rmStatus}.
       Only the leading fields are decoded.  Dwords 6..9 are bytes 24..39,
       i.e. perGpuAttributes[0].gpuUuid — gpuUuid is the first member of
       UvmGpuMappingAttributes (uvm_types.h), so those 16 bytes are a real
       UUID.  The later scalars sit past perGpuAttributes and are far
       beyond the 512-byte prefix the kernel dumps."""
    base = _u64_from_dwords(dwords, 0)
    length = _u64_from_dwords(dwords, 2)
    offset = _u64_from_dwords(dwords, 4)
    uuid_bytes = []
    for i in range(6, 10):
        if i < len(dwords):
            uuid_bytes.append(dwords[i])
    uuid_str = " ".join(f"{d:08x}" for d in uuid_bytes) if uuid_bytes else "?"
    return {
        "base": f"0x{base:x}",
        "length": f"0x{length:x}",
        "offset": f"0x{offset:x}",
        "uuid": uuid_str,
    }


def decode_uvm_create_external_range(dwords: List[int]) -> Dict[str, str]:
    """UVM_CREATE_EXTERNAL_RANGE_PARAMS: {NvU64 base, NvU64 length, ...}"""
    return {
        "base": f"0x{_u64_from_dwords(dwords, 0):x}",
        "length": f"0x{_u64_from_dwords(dwords, 2):x}",
    }


# Map UVM cmd id -> decoder function.  Only cmds with non-trivial
# params get a decoder; low-signal ones are intentionally skipped.
UVM_CMD_DECODERS: Dict[int, callable] = {
    33: decode_uvm_map_external,            # UVM_MAP_EXTERNAL_ALLOCATION
    70: decode_uvm_populate_pageable,       # UVM_POPULATE_PAGEABLE
    73: decode_uvm_create_external_range,   # UVM_CREATE_EXTERNAL_RANGE
    72: decode_uvm_create_external_range,   # UVM_CREATE_EXTERNAL_RANGE_V1 (same layout)
    66: decode_uvm_create_external_range,   # UVM_UNMAP_EXTERNAL (same base/length prefix)
}


def decode_uvm_cmd(cmd: int, dwords: List[int]) -> Optional[Dict[str, str]]:
    fn = UVM_CMD_DECODERS.get(cmd)
    if fn is None:
        return None
    try:
        return fn(dwords)
    except Exception as exc:
        return {"decode_error": str(exc)}


# ── strace parser ──────────────────────────────────────────────────────────

# ts_ns is an optional trailing field carrying a CLOCK_MONOTONIC nanosecond
# timestamp.  When present it lets timeline_merge.py align events from
# ftrace + strace + pbcap-ndjson onto a single sorted timeline.  Existing
# callers that don't set it (strace parser paths, anywhere that hasn't been
# upgraded yet) get None.  defaults= attaches to the *trailing* fields, so
# ordering of existing fields is unchanged.
Event = namedtuple("Event", ["line_no", "pid", "kind", "info", "ts_ns"],
                   defaults=(None,))

# Matches, e.g.,
#   18292 openat(AT_FDCWD, "/dev/nvidia-uvm", O_RDWR|O_CLOEXEC) = 9
# or with strace -ttt:
#   1778055205.625509 18292 openat(AT_FDCWD, "/dev/nvidia-uvm", ...) = 9
# and
#   18292 ioctl(9, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2b, 0x30), 0x...) = 0
#   18292 ioctl(9, _IOC(_IOC_NONE, 0, 0x21, 0), 0x...) = 0
# and
#   18292 close(9) = 0
#
# The optional `(?P<ts>...)` group captures strace -ttt's leading
# CLOCK_REALTIME timestamp (seconds.microseconds).  When present,
# timeline_merge.py translates these to CLOCK_MONOTONIC ns using
# pbcap.init's anchor.
#
# strace output formatting depends on flags:
#   strace -ttt                  →  "1778055894.365652 pid openat(...)"
#   strace -ttt -f               →  "pid 1778055894.365652 openat(...)"
#   (the -f form prepends the pid for thread-group tracking, then ts)
#
# To handle both, the pid is required and the ts is optional AND can
# appear on either side.  We allow "[TS ]PID" OR "PID TS" patterns.
_TS_PID_PID_TS = (
    r"(?:"
    r"(?P<ts>\d+\.\d+)\s+(?P<pid>\d+)"     # ts then pid (no -f)
    r"|"
    r"(?P<pid2>\d+)\s+(?P<ts2>\d+\.\d+)"  # pid then ts (-f mode)
    r"|"
    r"(?P<pid3>\d+)"                       # pid only (no -ttt)
    r")"
)
OPENAT_RE = re.compile(
    r"^" + _TS_PID_PID_TS +
    r"\s+openat\([^,]*,\s*\"(?P<path>[^\"]+)\",[^)]*\)\s*=\s*(?P<fd>\d+)"
)
CLOSE_RE = re.compile(
    r"^" + _TS_PID_PID_TS + r"\s+close\((?P<fd>\d+)\)\s*=")
IOCTL_RE = re.compile(
    r"^" + _TS_PID_PID_TS +
    r"\s+ioctl\((?P<fd>\d+),\s*"
    r"(?:"
    r"_IOC\((?P<dir>[^,]+),\s*(?P<type>0x[0-9a-fA-F]+|\d+),\s*"
    r"(?P<cmd>0x[0-9a-fA-F]+|\d+),\s*(?P<size>0x[0-9a-fA-F]+|\d+)\)"
    r"|"
    r"(?P<named>[A-Z_][A-Z0-9_]*)"
    r")"
    r"[^)]*\)\s*=\s*(?P<ret>-?\d+)"
)
# mmap(addr, length, prot, flags, fd, offset) = ret
# addr may be NULL or a hex address; length is decimal or hex; fd is a
# decimal signed (-1 for MAP_ANONYMOUS); ret is the returned user VA.
MMAP_RE = re.compile(
    r"^" + _TS_PID_PID_TS +
    r"\s+mmap\(\s*(?P<addr>NULL|0x[0-9a-fA-F]+)\s*,\s*"
    r"(?P<length>\d+|0x[0-9a-fA-F]+)\s*,\s*"
    r"(?P<prot>[^,]+),\s*"
    r"(?P<flags>[^,]+),\s*"
    r"(?P<fd>-?\d+)\s*,\s*"
    r"(?P<off>\d+|0x[0-9a-fA-F]+)\s*\)"
    r"\s*=\s*(?P<ret>0x[0-9a-fA-F]+|-?\d+)"
)
MUNMAP_RE = re.compile(
    r"^" + _TS_PID_PID_TS +
    r"\s+munmap\(\s*(?P<addr>0x[0-9a-fA-F]+|\d+)\s*,\s*"
    r"(?P<length>\d+|0x[0-9a-fA-F]+)\s*\)"
    r"\s*=\s*(?P<ret>-?\d+)"
)


def _strace_pid(m) -> int:
    """Pick up the pid from whichever of the three alternation groups matched."""
    for k in ("pid", "pid2", "pid3"):
        v = m.group(k) if k in m.groupdict() else None
        if v:
            return int(v)
    return 0


def _strace_ts_ns(m) -> Optional[int]:
    """Extract CLOCK_REALTIME ts_ns from a strace -ttt line match.

    Returns None if the match has no `ts` group or it's empty (strace
    was invoked without -ttt).  Handles both orderings: `ts pid` (no -f)
    and `pid ts` (-f mode).  Callers keep these events in order by
    line_no when ts is absent.
    """
    for k in ("ts", "ts2"):
        v = m.group(k) if k in m.groupdict() else None
        if v:
            return int(float(v) * 1_000_000_000)
    return None


def _to_int(s: str) -> int:
    """Accept '0x2b' or decimal."""
    s = s.strip()
    return int(s, 16) if s.startswith(("0x", "0X")) else int(s)


def parse_strace(path: str) -> List[Event]:
    """
    Parse one strace file into a list of events.

    Returned events include every openat/close for /dev/nvidia*, and
    every NVIDIA-driver ioctl with its cmd decoded.  Non-NVIDIA syscalls
    (libc loading, anonymous mmap, TCGETS on stdout, etc.) are filtered
    out so the diff focuses on driver-level events.

    Event kinds:
      "open"  — info = (fd, path)
      "close" — info = fd
      "ioctl" — info = (fd, path, cmd_int, cmd_name, size, ret)
                ret is the kernel return value (negative for failure,
                None when the regex couldn't extract one — e.g. the
                bare-UVM_INITIALIZE fallback path).
                For named ioctls (TCGETS etc.) cmd_int is None and
                cmd_name is the symbolic name — these we filter.
      "mmap"  — info = (fd, path, addr, length, prot, flags, offset)
                Only emitted for mmap calls on an fd currently opened
                against a /dev/nvidia* path (so the ioctl-fd table acts
                as a filter).  addr is the RETURNED user VA, not the
                hint address argument.  Used by address_atlas.py to
                join (fd, length) against map_bar1 events and attach
                hMemory to the resulting cpu_range.
      "munmap"— info = (addr, length)
                Closes a cpu_range.
    """
    # Per-pid: fd -> path of the *last* /dev/nvidia* opened on it.
    # The strace fd table is not truly process-wide because we're
    # running single-process captures here, but tracking per-pid is
    # robust against any incidental -f fork.
    fd_table: Dict[Tuple[int, int], str] = {}

    events: List[Event] = []

    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            m = OPENAT_RE.match(line)
            if m:
                pid = _strace_pid(m)
                fd = int(m.group("fd"))
                p = m.group("path")
                if p.startswith("/dev/nvidia"):
                    fd_table[(pid, fd)] = p
                    events.append(
                        Event(line_no, pid, "open", (fd, p), _strace_ts_ns(m))
                    )
                continue

            m = CLOSE_RE.match(line)
            if m:
                pid = _strace_pid(m)
                fd = int(m.group("fd"))
                key = (pid, fd)
                if key in fd_table:
                    events.append(
                        Event(line_no, pid, "close", fd, _strace_ts_ns(m))
                    )
                    del fd_table[key]
                continue

            m = MMAP_RE.match(line)
            if m:
                pid = _strace_pid(m)
                fd = int(m.group("fd"))
                if fd < 0:
                    # MAP_ANONYMOUS — no backing file, irrelevant.
                    continue
                key = (pid, fd)
                path = fd_table.get(key)
                if path is None:
                    # mmap on a non-NVIDIA fd.  Skip.
                    continue
                ret = m.group("ret")
                try:
                    addr = int(ret, 16) if ret.startswith("0x") else int(ret)
                except ValueError:
                    continue
                if addr < 0:
                    # mmap returned an error code — no mapping created.
                    continue
                length = _to_int(m.group("length")) or 0
                offset = _to_int(m.group("off")) or 0
                events.append(
                    Event(line_no, pid, "mmap",
                          (fd, path, addr, length,
                           m.group("prot"), m.group("flags"), offset),
                          _strace_ts_ns(m))
                )
                continue

            m = MUNMAP_RE.match(line)
            if m:
                pid = _strace_pid(m)
                addr = _to_int(m.group("addr")) or 0
                length = _to_int(m.group("length")) or 0
                events.append(
                    Event(line_no, pid, "munmap",
                          (addr, length),
                          _strace_ts_ns(m))
                )
                continue

            m = IOCTL_RE.match(line)
            if m:
                pid = _strace_pid(m)
                fd = int(m.group("fd"))
                key = (pid, fd)
                if key not in fd_table:
                    # Not an NVIDIA-driver fd — skip (stdout TCGETS etc.).
                    continue
                path = fd_table[key]
                if m.group("named"):
                    # Named ioctl (TCGETS, FIOCLEX, …).  Ignore — we only
                    # care about the numeric cmds.
                    continue
                type_ = _to_int(m.group("type"))
                cmd = _to_int(m.group("cmd"))
                size = _to_int(m.group("size"))
                try:
                    ret = int(m.group("ret"))
                except (TypeError, ValueError):
                    ret = None

                # Decode cmd based on path.
                if path == "/dev/nvidia-uvm":
                    name = UVM_CMDS.get(cmd, f"UVM_0x{cmd:x}")
                elif type_ == 0x46:
                    name = RM_ESCAPES.get(cmd, f"ESC_0x{cmd:02x}")
                elif cmd == UVM_INITIALIZE & 0xFFFF and type_ == 0:
                    name = "UVM_INITIALIZE"
                else:
                    name = f"UNK(type=0x{type_:x} cmd=0x{cmd:x})"

                events.append(
                    Event(line_no, pid, "ioctl",
                          (fd, path, cmd, name, size, ret),
                          _strace_ts_ns(m))
                )
                continue

            # The UVM_INITIALIZE ioctl has a full 0x30000001 cmd that
            # strace may render as the full number or as _IOC().  Handle
            # the bare-number form too.  Accepts both orderings of the
            # optional -ttt timestamp (same as the main regexes).
            if "ioctl" in line and "0x30000001" in line:
                mm = re.match(
                    r"^" + _TS_PID_PID_TS +
                    r"\s+ioctl\((?P<fd>\d+),\s*(?P<arg>[^,]+),",
                    line,
                )
                if mm and "0x30000001" in mm.group("arg"):
                    pid = _strace_pid(mm)
                    fd = int(mm.group("fd"))
                    key = (pid, fd)
                    if key in fd_table:
                        events.append(
                            Event(
                                line_no, pid, "ioctl",
                                (fd, fd_table[key], UVM_INITIALIZE,
                                 "UVM_INITIALIZE", 0, None),
                                _strace_ts_ns(mm),
                            )
                        )
                continue

    return events


# ── ftrace parser ──────────────────────────────────────────────────────────
#
# Two layers.  FTRACE_LINE_RE below handles the outer one, the ftrace row
# envelope.  `nv_trace_printf` in that row is the ftrace event name the
# driver emits under — it is not the payload format.  Every payload is an
# mc1 record, whose grammar is described in the block further down and,
# authoritatively, in docs/reference/trace-format.md.
#
#     <task>-<pid>    [<cpu>] <flags>    <ts>: nv_trace_printf: <payload>

FTRACE_LINE_RE = re.compile(
    r"^\s*(?P<task>\S+)-(?P<pid>\d+)\s+\[(?P<cpu>\d+)\]\s+\S+\s+"
    r"(?P<ts>\d+\.\d+):\s+nv_trace_printf:\s*(?P<payload>.*)$"
)
# ── mc1 record parsing ──────────────────────────────────────────────────
#
# The kernel emits one self-describing record per ftrace line:
#
#     mc1 <category>/<event> key=value key=value ... [arr=[a,b,c]]
#
# (grammar: docs/reference/trace-format.md).  FTRACE_LINE_RE above peels
# the ftrace envelope; everything below turns the `mc1 …` payload into the
# Event(kind, info-tuple) shapes the downstream consumers expect, so
# timeline_merge._ftrace_info_to_dict and the three downstream tools
# (address_atlas.py, trace_section.py, non_uvm_ledger.py) are untouched.
#
# The dispatch is table-driven (MC1_SIMPLE) for the 1:1 positional events;
# the handful that aggregate (rm/alloc + body rows by id, uvm/ioctl + rows
# by id, pb/bytes chunks by seq) or that gate on a discriminant
# (rm/free entry-vs-ret, dbell/*_track state=add/remove) are handled
# explicitly.  Unknown events parse for free and are simply not emitted —
# nothing downstream consumes them.

MC1_RECORD_RE = re.compile(
    r"^mc1 (?P<cat>[a-z][a-z0-9_]*)/(?P<event>[a-z][a-z0-9_]*)(?: (?P<fields>.*))?$")

# key=value, where value is quoted / a [array] / bare (no unescaped space).
MC1_KV_RE = re.compile(
    r'([a-z][a-z0-9_]*)=("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\S+)')


def _mc1_fields(s: str) -> Dict[str, Any]:
    """Tokenise a record's field string into {key: value}.

    A bare/quoted value is a str; an [a,b,c] array is a list[str].  No
    numeric coercion happens here — callers coerce per field, because the
    same wire type (e.g. a %p pointer) is an int for some events and an
    opaque string for others."""
    out: Dict[str, Any] = {}
    for m in MC1_KV_RE.finditer(s):
        k, v = m.group(1), m.group(2)
        if v.startswith('"'):
            out[k] = v[1:-1].encode("ascii", "backslashreplace").decode("unicode_escape")
        elif v.startswith('['):
            inner = v[1:-1].strip()
            out[k] = [t.strip() for t in inner.split(",")] if inner else []
        else:
            out[k] = v
    return out


def _split_mc1(payload: str) -> Iterable[str]:
    """Yield each `mc1 …` record in one ftrace payload.

    The macro terminates every record with `\\n`, so one physical line
    holds exactly one record.  We still split on the `mc1 ` boundary so a
    run-together line (the historical missing-newline bug) degrades to
    every record parsed rather than all-but-the-first dropped."""
    starts = [m.start() for m in re.finditer(r"mc1 ", payload)]
    for a, b in zip(starts, starts[1:] + [len(payload)]):
        rec = payload[a:b].rstrip()
        if rec:
            yield rec


def _mc1_int(v: str) -> int:
    """int() over a bare mc1 value: 0x-hex, decimal, or signed decimal."""
    return int(v, 0)


# (category, event) -> (kind, ((wire_key, coerce), ...)).  coerce is _I
# (int via _mc1_int) or _S (raw string).  The tuple order is the info
# shape _ftrace_info_to_dict already unpacks for that kind — do not
# reorder without changing timeline_merge in lockstep.
_I, _S = "I", "S"
MC1_SIMPLE: Dict[tuple, tuple] = {
    ("rm", "control"): ("control", (
        ("cmd", _I), ("hclient", _I), ("hobject", _I), ("params_size", _I))),
    ("rm", "map_memory"): ("map_bar1", (
        ("hmemory", _I), ("length", _I), ("flags", _I), ("offset", _I), ("fd", _I))),
    ("mmu", "pte_src_decision"): ("pte_src_decision", (
        ("carrier_h", _I), ("carrier_class", _I), ("carrier_aspace", _I),
        ("src_h", _I), ("src_class", _I), ("src_aspace", _I),
        ("src_pte0", _I), ("chosen_pte0", _I))),
    ("mmu", "intermap_call"): ("intermap_call", (
        ("hclient", _I), ("hcarrier", _I), ("carrier_class", _I),
        ("hsrc", _I), ("src_class", _I), ("flags", _I),
        ("dma_offset", _I), ("length", _I))),
    ("mmu", "virtmem_backing"): ("virtmem_backing", (
        ("hmemory", _I), ("class", _I), ("va_size", _I), ("hvaspace", _I),
        ("aspace", _I), ("has_heap", _I), ("via", _S))),
    ("mmu", "gmmu_pte_phys"): ("gmmu_pte_phys", (
        ("va_lo", _I), ("va_hi", _I), ("pte0_phys", _I),
        ("pte_count", _I), ("page_size", _I), ("aperture", _I))),
    ("mmu", "bar1_reflect_phys"): ("bar1_reflect_phys", (
        ("at_gpu", _I), ("fb_aperture_off", _I), ("bar1_phys", _I), ("size", _I))),
    ("fifo", "userd_resolve"): ("userd_resolve", (
        ("hclient", _I), ("huserd", _I), ("userd_offset", _I),
        ("userd_addr", _I), ("address_space", _I), ("userd_size", _I))),
    ("fifo", "userd_rpc"): ("userd_rpc", (
        ("hchannel", _I), ("base", _I), ("size", _I),
        ("address_space", _I), ("cache_attrib", _I))),
    ("fifo", "userd_bind"): ("userd_bind", (
        ("retained", _S), ("resource_count", _I))),
    ("pb", "submit"): ("pb", (
        ("seq", _I), ("chid", _I), ("idx", _I), ("entry0", _I),
        ("entry1", _I), ("pb_va", _I), ("pb_len", _I))),
    ("pb", "bytes_miss"): ("pb_bytes_miss", (
        ("seq", _I), ("chid", _I), ("idx", _I), ("pb_va", _I), ("pb_len", _I))),
}


def parse_ftrace(path: str) -> List[Event]:
    """Parse one ftrace-text capture (mc1 format) into Event tuples.

    Produces the same Event(kind, info-tuple) shapes as parse_strace so the
    downstream aggregation / merge functions are format-agnostic.  The info
    tuples per kind are exactly what timeline_merge._ftrace_info_to_dict
    unpacks:

      alloc            (handle, cls, body_dict, root, parent)
      control          (ctrlcmd, hClient, hObject, paramsSize)
      map_bar1         (handle, length, flags, offset, fd)
      map_uvm          (handle, base, length, offset)
      free             (handle,)
      uvm_ioctl        (cmd, cmd_name, dwords)
      pb               (seq, chid, idx, entry0, entry1, pb_va, pb_len)
      pb_bytes         (seq, chid, idx, nbytes, hex_str)
      pb_bytes_miss    (seq, chid, idx, pb_va, pb_len)
      sysmem_track_add/remove, bar1_track_add/remove, and the eight
      non-UVM VAS/channel kinds — see MC1_SIMPLE and the explicit handlers.

    Synthetic "ioctl" events are emitted alongside rm/ioctl, uvm/ioctl and
    pb/submit so --summary / ioctl_counts account for them too.
    """

    events: List[Event] = []

    # Correlation-id accumulators.  rm/alloc and uvm/ioctl carry an id=
    # that their body rows repeat, so reassembly is order-independent; we
    # flush at EOF.  pb/bytes chunks key by seq and flush on completion.
    alloc_by_id: Dict[int, Dict[str, Any]] = {}
    uvm_by_id: Dict[int, Dict[str, Any]] = {}
    pb_bytes_acc: Dict[int, Dict[str, Any]] = {}

    def decode_alloc_body(cls: int, dwords: List[int]) -> Dict:
        """Decode the alloc-params body for the given class.

        Three struct layouts are in play:
        - NV_OS_DESC_MEMORY_ALLOCATION_PARAMS (nvos.h) — class 0x0070.
        - NV_CHANNEL_ALLOC_PARAMS (alloc_channel.h) — the *_CHANNEL_GPFIFO_A
          classes (0xc36f/0xc56f/0xc86f); binds channel → hVASpace,
          hUserdMemory[], gpFifoOffset, etc.
        - NV_MEMORY_ALLOCATION_PARAMS (nvos.h) — every other memory class.
        Dword index i is byte offset i*4."""
        if not dwords:
            return {}
        if cls == 0x0070:  # NV01_MEMORY_SYSTEM_OS_DESCRIPTOR
            body: Dict = {"_layout": "os_desc"}
            if len(dwords) > 0: body["type"] = dwords[0]
            if len(dwords) > 1: body["flags"] = dwords[1]
            if len(dwords) > 2: body["attr"] = dwords[2]
            if len(dwords) > 3: body["attr2"] = dwords[3]
            if len(dwords) > 7:
                body["limit"] = (dwords[7] << 32) | dwords[6]
            return body
        if cls in (0xc36f, 0xc56f, 0xc86f):
            # Dword indices below are offsetof()/4 on NV_CHANNEL_ALLOC_PARAMS,
            # not eyeballed from the struct: hHandleVASpace sits between
            # hVASpace and hUserdMemory[], and omitting it shifted every field
            # from dword 8 onward by one, which silently reported
            # hUserdMemory/userdOffset as 0 for every channel.
            body = {"_layout": "channel"}
            if len(dwords) >  3: body["gpFifoOffset"]   = (dwords[3] << 32) | dwords[2]
            if len(dwords) >  4: body["gpFifoEntries"]  = dwords[4]
            if len(dwords) >  5: body["flags"]          = dwords[5]
            if len(dwords) >  6: body["hContextShare"]  = dwords[6]
            if len(dwords) >  7: body["hVASpace"]       = dwords[7]
            if len(dwords) >  8: body["hHandleVASpace"] = dwords[8]
            if len(dwords) >  9: body["hUserdMemory"]   = dwords[9]
            if len(dwords) > 19: body["userdOffset"]    = (dwords[19] << 32) | dwords[18]
            if len(dwords) > 34: body["engineType"]     = dwords[34]
            if len(dwords) > 35: body["cid"]            = dwords[35]
            return body
        body = {"_layout": "main"}
        if len(dwords) > 0: body["owner"] = dwords[0]
        if len(dwords) > 1: body["type"] = dwords[1]
        if len(dwords) > 2: body["flags"] = dwords[2]
        if len(dwords) > 6: body["attr"] = dwords[6]
        if len(dwords) > 7: body["attr2"] = dwords[7]
        if len(dwords) > 8: body["format"] = dwords[8]
        if len(dwords) > 17:
            body["size"] = (dwords[17] << 32) | dwords[16]
        if len(dwords) > 28:
            body["internalflags"] = dwords[28]
        return body

    def place_dwords(dwords: List[int], idx: int, arr: List[str]) -> None:
        """Write the 4-dword `dw=[…]` row into `dwords` at dword index idx,
        zero-extending on the way (rows may skip ahead / arrive sparse)."""
        vals = [int(x, 0) for x in arr]
        while len(dwords) < idx + len(vals):
            dwords.append(0)
        for j, v in enumerate(vals):
            dwords[idx + j] = v

    def flush_alloc(a: Dict[str, Any]) -> None:
        body = decode_alloc_body(a["cls"], a["dwords"])
        events.append(Event(a["line_no"], a["pid"], "alloc",
                            (a["new"], a["cls"], body, a["root"], a["parent"]),
                            a["ts_ns"]))

    def flush_uvm(u: Dict[str, Any]) -> None:
        cmd = u["cmd"]
        if cmd == UVM_INITIALIZE:
            name = "UVM_INITIALIZE"
        else:
            name = UVM_CMDS.get(cmd, f"UVM_0x{cmd:x}")
        events.append(Event(u["line_no"], u["pid"], "uvm_ioctl",
                            (cmd, name, u["dwords"]), u["ts_ns"]))
        # Synthetic ioctl so ioctl_counts()/print_summary see the UVM cmd.
        events.append(Event(u["line_no"], u["pid"], "ioctl",
                            ("nvidia-uvm", name), u["ts_ns"]))

    with open(path) as f:
        for line_no, raw in enumerate(f, start=1):
            if raw.startswith("#"):
                continue
            m = FTRACE_LINE_RE.match(raw)
            if not m:
                continue
            pid = int(m.group("pid"))
            # trace_clock=mono: seconds-since-boot with ns fraction → int ns.
            ts_ns = int(float(m.group("ts")) * 1_000_000_000)

            for rec in _split_mc1(m.group("payload")):
                # A torn or truncated record (ftrace ring-buffer tail, a capture
                # read while still being written) must skip that record, not
                # abort the run — see _split_mc1's contract.  Missing key ->
                # KeyError; malformed number -> ValueError from _mc1_int.
                try:
                    rm = MC1_RECORD_RE.match(rec)
                    if not rm:
                        continue
                    key = (rm.group("cat"), rm.group("event"))
                    d = _mc1_fields(rm.group("fields") or "")

                    # ── correlation-id families (accumulate, flush later) ──
                    if key == ("rm", "alloc"):
                        # entry side carries new=; the result=ret side does not.
                        if "new" in d and "id" in d:
                            alloc_by_id[_mc1_int(d["id"])] = {
                                "line_no": line_no, "pid": pid, "ts_ns": ts_ns,
                                "cls": _mc1_int(d["hclass"]),
                                "root": _mc1_int(d["root"]),
                                "parent": _mc1_int(d["parent"]),
                                "new": _mc1_int(d["new"]),
                                "dwords": []}
                        continue
                    if key == ("body", "alloc_hdr"):
                        continue  # size/dwords count redundant; rows carry data
                    if key == ("body", "alloc_row"):
                        a = alloc_by_id.get(_mc1_int(d["id"]))
                        if a is not None and isinstance(d.get("dw"), list):
                            place_dwords(a["dwords"], _mc1_int(d["off"]) // 4, d["dw"])
                        continue
                    if key == ("uvm", "ioctl"):
                        if "id" in d and "cmd" in d:
                            uvm_by_id[_mc1_int(d["id"])] = {
                                "line_no": line_no, "pid": pid, "ts_ns": ts_ns,
                                "cmd": _mc1_int(d["cmd"]), "dwords": []}
                        continue
                    if key == ("body", "uvm_row"):
                        u = uvm_by_id.get(_mc1_int(d["id"]))
                        if u is not None and isinstance(d.get("dw"), list):
                            place_dwords(u["dwords"], _mc1_int(d["off"]) // 4, d["dw"])
                        continue
                    if key == ("pb", "bytes"):
                        seq = _mc1_int(d["seq"])
                        off = _mc1_int(d["off"])
                        acc = pb_bytes_acc.get(seq)
                        if acc is None:
                            acc = {"line_no": line_no, "pid": pid, "ts_ns": ts_ns,
                                   "chid": _mc1_int(d["chid"]),
                                   "idx": _mc1_int(d["idx"]),
                                   "nchunks": _mc1_int(d["nchunks"]),
                                   "chunks": {}}
                            pb_bytes_acc[seq] = acc
                        acc["chunks"][off] = d["hex"]
                        if len(acc["chunks"]) == acc["nchunks"]:
                            full = "".join(acc["chunks"][k] for k in sorted(acc["chunks"]))
                            events.append(Event(acc["line_no"], acc["pid"], "pb_bytes",
                                                (seq, acc["chid"], acc["idx"],
                                                 len(full) // 2, full), acc["ts_ns"]))
                            del pb_bytes_acc[seq]
                        continue

                    # ── discriminant-gated one-shot events ──
                    if key == ("rm", "ioctl"):
                        esc = _mc1_int(d["esc"])
                        name = RM_ESCAPES.get(esc, f"ESC_0x{esc:02x}")
                        events.append(Event(line_no, pid, "ioctl",
                                            ("nvidiactl", name), ts_ns))
                        continue
                    if key == ("rm", "free"):
                        if "result" not in d and "old" in d:  # entry side
                            events.append(Event(line_no, pid, "free",
                                                (_mc1_int(d["old"]),), ts_ns))
                        continue
                    if key == ("uvm", "map_external"):
                        if "base" in d:  # entry side (ret side has rm_status only)
                            events.append(Event(line_no, pid, "map_uvm",
                                                (_mc1_int(d["hmemory"]),
                                                 _mc1_int(d["base"]),
                                                 _mc1_int(d["length"]),
                                                 _mc1_int(d["offset"])), ts_ns))
                        continue
                    if key == ("dbell", "sysmem_track"):
                        st = d.get("state")
                        if st == "add":
                            events.append(Event(line_no, pid, "sysmem_track_add",
                                                (_mc1_int(d["slot"]),
                                                 _mc1_int(d["user_va_start"]),
                                                 _mc1_int(d["user_va_end"]),
                                                 _mc1_int(d["num_pages"]),
                                                 d["kva"]), ts_ns))
                        elif st == "remove":
                            events.append(Event(line_no, pid, "sysmem_track_remove",
                                                (d["kva"], _mc1_int(d["num_pages"])),
                                                ts_ns))
                        continue
                    if key == ("dbell", "bar1_track"):
                        st = d.get("state")
                        if st == "add":
                            events.append(Event(line_no, pid, "bar1_track_add",
                                                (_mc1_int(d["slot"]),
                                                 _mc1_int(d["phys"]),
                                                 _mc1_int(d["size"]),
                                                 d["kva"]), ts_ns))
                        elif st == "remove":
                            events.append(Event(line_no, pid, "bar1_track_remove",
                                                (_mc1_int(d["slot"]), d["kva"]), ts_ns))
                        continue

                    # ── table-driven 1:1 positional events ──
                    spec = MC1_SIMPLE.get(key)
                    if spec is not None:
                        kind, fields = spec
                        try:
                            info = tuple(_mc1_int(d[k]) if c == _I else d[k]
                                         for k, c in fields)
                        except KeyError:
                            continue  # malformed record — skip, don't crash
                        events.append(Event(line_no, pid, kind, info, ts_ns))
                        if kind == "pb":
                            events.append(Event(line_no, pid, "ioctl",
                                                ("kernel", "pb/submit"), ts_ns))
                        continue
                    # Unknown event: parsed cleanly, nothing downstream needs it.
                except (KeyError, ValueError):
                    continue

    # Flush correlation-id accumulators in first-seen order.
    for a in alloc_by_id.values():
        flush_alloc(a)
    for u in uvm_by_id.values():
        flush_uvm(u)

    return events


# ── Role inference ─────────────────────────────────────────────────────────

# RM object-class name map.  Covers the classes mc uses plus a
# handful of common ones in case they appear in CUDA traces.
RM_CLASSES: Dict[int, str] = {
    0x0000: "NV01_ROOT",
    0x0040: "NV01_MEMORY_LOCAL_USER",    # vidmem
    0x0041: "NV01_MEMORY_LOCAL_PRIVILEGED",
    0x003E: "NV01_MEMORY_SYSTEM",        # sysmem
    0x0070: "NV01_MEMORY_SYSTEM_OS_DESC",
    0x0079: "NV01_EVENT_OS_EVENT",
    0x0080: "NV01_DEVICE_0",
    0x00DE: "RM_USER_SHARED_DATA",
    0x2080: "NV20_SUBDEVICE_0",
    0x2081: "NV2081_BINAPI",
    0x503C: "NV50_THIRD_PARTY_P2P",
    0x50A0: "NV50_MEMORY_VIRTUAL",
    0x83DE: "GT200_DEBUGGER",
    0x9067: "FERMI_CONTEXT_SHARE_A",
    0x9072: "GF100_DISP_SW",
    0x90E6: "GF100_SUBDEVICE_MASTER",
    0x90F1: "FERMI_VASPACE_A",
    0xA06C: "KEPLER_CHANNEL_GROUP_A",
    0xC597: "TURING_A",                  # 3D class
    0xC661: "HOPPER_USERMODE_A",
    0xC7C0: "AMPERE_COMPUTE_B",
    0xC86F: "HOPPER_CHANNEL_GPFIFO_A",
    0xC8B5: "HOPPER_DMA_COPY_A",
    0xCB33: "NV_CONFIDENTIAL_COMPUTE",
    0xCBC0: "HOPPER_COMPUTE_A",
}

# Known size-to-role hints for memory classes — first match wins.
# Sizes chosen to match mc's allocations.  Kept conservative:
# when multiple buffers share a size (e.g., gpfifo and sema are both
# 4 KiB in our setup), we don't guess — the label just shows size and
# the user can disambiguate by allocation order / subsequent use.
_SIZE_HINTS_VIDMEM = [
    # Sizes that uniquely identify a role in our default config.  4 KiB
    # vidmem is ambiguous (gpfifo == sema == 4 KiB) so it's OMITTED.
    # No entries yet — keep table for easy future additions.
]
_SIZE_HINTS_SYSMEM = [
    (64 * 1024,         "pb"),
    (16 * 1024,         "userd"),
    (2 << 20,           "staging"),
]

# For large vidmem-uvm-only buffers, we guess "likely-d_buf" only at
# our specific test sizes (4M / 64M / 128M / 256M / 1G / 2G / 3G) to
# avoid spurious hits on CUDA-internal ~300-MiB allocations that
# aren't user buffers.
_KNOWN_D_BUF_SIZES = {
    4 << 20, 16 << 20, 64 << 20, 128 << 20, 256 << 20,
    1 << 30, 2 << 30, 3 << 30,
}


def _nearest_size_hint(size: int, table) -> Optional[str]:
    for sz, name in table:
        if size == sz:
            return name
    return None


def infer_roles(events: List[Event]) -> Dict[int, str]:
    """Pass 2: walk all events and assign a role label per RM handle.

    Returns {handle -> role_string}.  Role is a short human-readable
    label we use to annotate every subsequent trace line.
    """
    # Per-handle state derived from the event stream.
    attrs: Dict[int, Dict] = {}
    for e in events:
        if e.kind == "alloc":
            handle, cls, body, root, parent = e.info
            attrs.setdefault(handle, {})
            attrs[handle]["class"] = cls
            attrs[handle]["size"] = body.get("size")
            attrs[handle]["root"] = root
            attrs[handle]["parent"] = parent
        elif e.kind == "map_bar1":
            handle, length, _flags, _off, _fd = e.info
            attrs.setdefault(handle, {})
            attrs[handle]["bar1_mapped"] = True
            attrs[handle].setdefault("size", length)
        elif e.kind == "map_uvm":
            handle, _base, length, _off = e.info
            attrs.setdefault(handle, {})
            attrs[handle]["uvm_mapped"] = True
            # For sysmem allocs (escape 0x27, not traced in full), the
            # UVM MAP_EXTERNAL is often our first signal.  Use length
            # as the size hint.
            attrs[handle].setdefault("size", length)

    roles: Dict[int, str] = {}
    for handle, a in attrs.items():
        cls = a.get("class")
        size = a.get("size")
        bar1 = a.get("bar1_mapped", False)
        uvm = a.get("uvm_mapped", False)

        # Pass 2a: direct class-based labels for non-memory objects.
        if cls in (0x0000,):                           # NV01_ROOT
            roles[handle] = "client"
            continue
        if cls in (0x0080, 0x2080):                    # device / subdevice
            roles[handle] = RM_CLASSES[cls]
            continue
        if cls == 0x90F1:
            roles[handle] = "vaspace"
            continue
        if cls == 0xA06C:
            roles[handle] = "tsg"
            continue
        if cls == 0xC86F:
            roles[handle] = "channel"
            continue
        if cls == 0xC8B5:
            roles[handle] = "ce-object"
            continue
        if cls == 0xC661:
            roles[handle] = "usermode-doorbell"
            continue

        # Pass 2b: memory-class labels.  Distinguish by
        # (class, size, bar1, uvm).
        if cls == 0x0040 or (cls is None and size is not None and not bar1 and not uvm):
            # Vidmem.
            if bar1 and not uvm:
                base = "vidmem+bar1"
            elif bar1 and uvm:
                base = "vidmem+bar1+uvm"
            elif uvm:
                base = "vidmem-uvm-only"
            else:
                base = "vidmem"
            hint = _nearest_size_hint(size or 0, _SIZE_HINTS_VIDMEM)
            if (size is not None and not bar1
                    and size in _KNOWN_D_BUF_SIZES):
                hint = hint or "likely-d_buf"
            label = f"{base}[{size:#x}]" if size is not None else base
            if hint:
                label += f":{hint}"
            roles[handle] = label
            continue

        if cls == 0x003E:
            # Sysmem allocated via NV_ESC_RM_ALLOC (escape 0x2B).
            base = "sysmem"
            if bar1 and uvm:
                base = "sysmem+bar1+uvm"
            elif bar1:
                base = "sysmem+bar1"
            elif uvm:
                base = "sysmem+uvm"
            hint = _nearest_size_hint(size or 0, _SIZE_HINTS_SYSMEM)
            if (size is not None and not bar1
                    and size in _KNOWN_D_BUF_SIZES):
                hint = hint or "likely-h_buf"
            label = f"{base}[{size:#x}]" if size is not None else base
            if hint:
                label += f":{hint}"
            roles[handle] = label
            continue

        if cls is None and (bar1 or uvm):
            # Handle first seen via a map event — likely a sysmem
            # alloc via escape 0x27 (not traced with hClass).  Use
            # size to infer role.
            base = "mem"
            if bar1 and uvm:
                base = "mem+bar1+uvm"
            elif bar1:
                base = "mem+bar1"
            elif uvm:
                base = "mem+uvm"
            hint = _nearest_size_hint(size or 0, _SIZE_HINTS_SYSMEM)
            if (size is not None and not bar1
                    and size in _KNOWN_D_BUF_SIZES):
                hint = hint or "likely-h_buf"
            label = f"{base}[{size:#x}]" if size is not None else base
            if hint:
                label += f":{hint}"
            roles[handle] = label
            continue

        # Catch-all.
        if cls is not None:
            name = RM_CLASSES.get(cls, f"class_0x{cls:x}")
            roles[handle] = name
        else:
            roles[handle] = "unknown"

    return roles


def print_roles(path: str, events: List[Event], roles: Dict[int, str]) -> None:
    """Print a compact per-handle table: handle | role | size | flags."""
    label = path.split("/")[-1]
    print(f"== Role inference: {label} ==")
    # Preserve first-seen order for stable output.
    seen: List[int] = []
    order = {}
    for e in events:
        if e.kind == "alloc":
            h = e.info[0]
        elif e.kind == "map_bar1":
            h = e.info[0]
        elif e.kind == "map_uvm":
            h = e.info[0]
        else:
            continue
        if h not in order:
            order[h] = len(order)
            seen.append(h)
    for h in seen:
        print(f"  0x{h:08x}  {roles.get(h, 'unknown')}")
    print()


# ── Aggregation / diff ─────────────────────────────────────────────────────

def _ioctl_info(info):
    """Extract (path_family, cmd_name) from an ioctl Event.info tuple.

    strace-source events carry (fd, path, cmd_int, cmd_name, size, ret).
    ftrace-source events carry (path_family, cmd_name).  Normalize
    both so downstream helpers don't have to branch.
    """
    if len(info) == 2:                       # ftrace shape
        base, name = info
        return base, name
    # strace shape — accept any extra trailing fields for forward compat.
    _, path, _, name, *_ = info
    return path.replace("/dev/", ""), name


def ioctl_counts(events: List[Event]) -> Counter:
    c: Counter = Counter()
    for e in events:
        if e.kind == "ioctl":
            _, name = _ioctl_info(e.info)
            c[name] += 1
    return c


def ioctl_sequence(events: List[Event]) -> List[Tuple[str, str]]:
    """Flatten to (path_family, ioctl_name) pairs, dropping open/close.

    path_family is the /dev/nvidia* basename so we can group by fd class
    in the diff output.
    """
    seq: List[Tuple[str, str]] = []
    for e in events:
        if e.kind == "ioctl":
            base, name = _ioctl_info(e.info)
            seq.append((base, name))
    return seq


def lcs_diff(a: List[Tuple[str, str]], b: List[Tuple[str, str]]) -> List[Tuple[str, Tuple, Tuple]]:
    """Compute an edit-script alignment via LCS.

    Returns a list of (op, a_item, b_item) where op is one of:
      '='  common to both (a_item == b_item)
      '-'  in a but not b
      '+'  in b but not a
    """
    n, m = len(a), len(b)
    # LCS length table.  This is O(n*m); our sequences are ≤ ~400 so fine.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    script: List[Tuple[str, Tuple, Tuple]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            script.append(("=", a[i], b[j]))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            script.append(("-", a[i], ("", "")))
            i += 1
        else:
            script.append(("+", ("", ""), b[j]))
            j += 1
    while i < n:
        script.append(("-", a[i], ("", "")))
        i += 1
    while j < m:
        script.append(("+", ("", ""), b[j]))
        j += 1
    return script


# ── Output ─────────────────────────────────────────────────────────────────

def fmt_item(item: Tuple[str, str]) -> str:
    base, name = item
    if not name:
        return ""
    return f"{base}:{name}"


def print_summary(cap_a_path: str, cap_b_path: str,
                  counts_a: Counter, counts_b: Counter) -> None:
    """Per-ioctl count comparison."""
    all_names = sorted(set(counts_a) | set(counts_b))
    label_a = cap_a_path.split("/")[-1]
    label_b = cap_b_path.split("/")[-1]
    width = max(len(n) for n in all_names) + 2
    print(f"{'ioctl':<{width}} {label_a:>20} {label_b:>20}  diff")
    print("-" * (width + 46))
    for name in all_names:
        a = counts_a.get(name, 0)
        b = counts_b.get(name, 0)
        marker = ""
        if a and not b:
            marker = "  (A only)"
        elif b and not a:
            marker = "  (B only)"
        elif a != b:
            marker = f"  (Δ {b - a:+d})"
        print(f"{name:<{width}} {a:>20} {b:>20}{marker}")


def print_only(cap_a_path: str, cap_b_path: str,
               counts_a: Counter, counts_b: Counter) -> None:
    """List ioctls that appear only in one capture."""
    a_only = sorted(set(counts_a) - set(counts_b))
    b_only = sorted(set(counts_b) - set(counts_a))
    label_a = cap_a_path.split("/")[-1]
    label_b = cap_b_path.split("/")[-1]
    print(f"== Only in {label_a} ==")
    for n in a_only:
        print(f"  {n:<36} (×{counts_a[n]})")
    if not a_only:
        print("  (none)")
    print()
    print(f"== Only in {label_b} ==")
    for n in b_only:
        print(f"  {n:<36} (×{counts_b[n]})")
    if not b_only:
        print("  (none)")


def print_aligned_diff(cap_a_path: str, cap_b_path: str,
                       seq_a: List[Tuple[str, str]],
                       seq_b: List[Tuple[str, str]]) -> None:
    """Full LCS alignment with '-' / '+' / '=' markers."""
    script = lcs_diff(seq_a, seq_b)
    label_a = cap_a_path.split("/")[-1]
    label_b = cap_b_path.split("/")[-1]
    print(f"A = {label_a}")
    print(f"B = {label_b}")
    print()
    for op, ai, bi in script:
        if op == "=":
            # Keep common lines but dim them — they frame the divergences.
            print(f"    {fmt_item(ai)}")
        elif op == "-":
            print(f"  - {fmt_item(ai)}")
        else:
            print(f"  + {fmt_item(bi)}")


# ── Per-handle history ─────────────────────────────────────────────────────
#
# A handle history is a timeline of decoded events for one RM handle
# across the whole capture.  Used by --handle-history and --handle-diff
# to render "what happened to this handle" in human-readable form.

def build_handle_histories(events: List[Event]) -> Dict[int, List[Dict]]:
    """Walk events, group per handle into a list of decoded step dicts.

    Each step dict has a 'kind' key identifying what happened plus
    the decoded fields for that step.  Ordering is event-stream
    order (which matches ftrace timestamp order since parse_ftrace
    reads sequentially).
    """
    hist: Dict[int, List[Dict]] = {}

    def add(handle: int, step: Dict) -> None:
        hist.setdefault(handle, []).append(step)

    for e in events:
        if e.kind == "alloc":
            handle, cls, body, root, parent = e.info
            step = {
                "kind": "alloc",
                "line_no": e.line_no,
                "class": cls,
                "class_name": RM_CLASSES.get(cls, f"class_0x{cls:x}"),
                "root": root,
                "parent": parent,
                "body": body,
            }
            add(handle, step)
        elif e.kind == "map_bar1":
            handle, length, flags, _off, _fd = e.info
            add(handle, {
                "kind": "map_bar1",
                "line_no": e.line_no,
                "length": length,
                "flags": flags,
            })
        elif e.kind == "map_uvm":
            handle, base, length, offset = e.info
            add(handle, {
                "kind": "map_uvm",
                "line_no": e.line_no,
                "base": base,
                "length": length,
                "offset": offset,
            })
        elif e.kind == "control":
            ctrlcmd, hc, ho, ps = e.info
            # Control targets hObject — add to that handle's history
            # (and also to hClient's for completeness, so client-level
            # controls are visible).
            add(ho, {
                "kind": "control",
                "line_no": e.line_no,
                "cmd": ctrlcmd,
                "hClient": hc,
                "paramsSize": ps,
            })
        elif e.kind == "free":
            (handle,) = e.info
            add(handle, {
                "kind": "free",
                "line_no": e.line_no,
            })

    return hist


def format_step(step: Dict, indent: str = "  ") -> List[str]:
    """Render a single history step into one or more output lines."""
    out: List[str] = []
    kind = step["kind"]
    line_no = step["line_no"]
    if kind == "alloc":
        body = step["body"]
        cls_name = step["class_name"]
        out.append(f"{indent}[line {line_no:>6}] ALLOC     class={cls_name}")
        if "size" in body:
            out.append(f"{indent}                     size=0x{body['size']:x}")
        if "owner" in body:
            out.append(f"{indent}                     owner=0x{body['owner']:x} type={body['type']}")
        if "flags" in body:
            out.append(f"{indent}                     flags=0x{body['flags']:08x} "
                       f"({decode_nvos32_flags(body['flags'])})")
        if "attr" in body:
            out.append(f"{indent}                     attr=0x{body['attr']:08x} "
                       f"({decode_nvos32_attr(body['attr'])})")
        if "attr2" in body:
            out.append(f"{indent}                     attr2=0x{body['attr2']:08x} "
                       f"({decode_nvos32_attr2(body['attr2'])})")
        if "internalflags" in body and body["internalflags"] != 0:
            out.append(f"{indent}                     internalflags=0x{body['internalflags']:08x}")
    elif kind == "map_bar1":
        flags = step["flags"]
        out.append(f"{indent}[line {line_no:>6}] MAP_MEMORY (BAR1)  length=0x{step['length']:x}")
        out.append(f"{indent}                     flags=0x{flags:08x} ({decode_nvos33_flags(flags)})")
    elif kind == "map_uvm":
        out.append(f"{indent}[line {line_no:>6}] UVM_MAP_EXT  base=0x{step['base']:x} "
                   f"length=0x{step['length']:x} offset=0x{step['offset']:x}")
    elif kind == "control":
        out.append(f"{indent}[line {line_no:>6}] CONTROL   cmd=0x{step['cmd']:08x} "
                   f"paramsSize={step['paramsSize']}")
    elif kind == "free":
        out.append(f"{indent}[line {line_no:>6}] FREE")
    return out


# ── --missing-ioctls mode ──────────────────────────────────────────────────

def print_missing_ioctls(cap_a_path: str, cap_b_path: str,
                         events_a: List[Event], events_b: List[Event]) -> None:
    """List ioctls present in one capture but not the other.

    Goes beyond --only by annotating each missing ioctl with any
    decoded parameters we can see in the ftrace.  Currently most
    ioctls are opaque (just the cmd name), but for a handful of
    high-interest ones (UVM_POPULATE_PAGEABLE, NV_ESC_ALLOC_OS_EVENT,
    NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO) we note what the kernel
    tracepoints would need to expose to make them fully decodable.
    """
    counts_a = ioctl_counts(events_a)
    counts_b = ioctl_counts(events_b)
    a_only = sorted(set(counts_a) - set(counts_b))
    b_only = sorted(set(counts_b) - set(counts_a))
    label_a = cap_a_path.split("/")[-1]
    label_b = cap_b_path.split("/")[-1]

    # For decoded params: collect each uvm_ioctl event's decoded fields
    # by cmd name so we can show them inline per occurrence.
    def uvm_decoded_by_name(events: List[Event]) -> Dict[str, List[Dict[str, str]]]:
        out: Dict[str, List[Dict[str, str]]] = {}
        for e in events:
            if e.kind == "uvm_ioctl":
                cmd, name, dwords = e.info
                decoded = decode_uvm_cmd(cmd, dwords)
                if decoded is not None:
                    out.setdefault(name, []).append(decoded)
        return out

    decoded_a = uvm_decoded_by_name(events_a)
    decoded_b = uvm_decoded_by_name(events_b)

    # Notes: low-signal / not-yet-covered ioctls.  Items covered by the
    # UVM dispatcher tracepoint no longer need a note — their params
    # are decoded inline below.
    notes = {
        "NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO":
            "params body not traced yet — add escape.c case to "
            "dump NVOS56_PARAMETERS {hClient, hDevice, hMemory, "
            "pOldCpuAddress, pNewCpuAddress}",
        "NV_ESC_ALLOC_OS_EVENT":
            "params body not traced — add escape.c case to dump "
            "nv_ioctl_alloc_os_event_t {hClient, hDevice, fd}",
        "UVM_CREATE_RANGE_GROUP":
            "output-only params (rangeGroupId) — low-signal even with "
            "kernel tracepoint",
        "UVM_PAGEABLE_MEM_ACCESS":
            "output-only params — low-signal",
        "NV_ESC_SYS_PARAMS":
            "kernel-supplied sys params; unlikely to affect behavior",
        "NV_ESC_CARD_INFO":
            "device-enumeration query; unlikely to affect memory flow",
    }

    def emit(owner_label: str, names: List[str], counts: Counter,
             decoded: Dict[str, List[Dict[str, str]]]) -> None:
        print(f"== Only in {owner_label} ==")
        if not names:
            print("  (none)")
            return
        for n in names:
            count = counts[n]
            note = notes.get(n, "")
            print(f"  {n:<40} ×{count}")
            if note:
                print(f"    note: {note}")
            # Dump decoded params for each occurrence of this cmd.
            for i, d in enumerate(decoded.get(n, [])):
                fields = " ".join(f"{k}={v}" for k, v in d.items())
                print(f"    params[{i}]: {fields}")

    emit(label_a, a_only, counts_a, decoded_a)
    print()
    emit(label_b, b_only, counts_b, decoded_b)


# ── --handle-history mode ──────────────────────────────────────────────────

def print_handle_history(cap_path: str, events: List[Event]) -> None:
    """Print every handle's full decoded timeline."""
    hist = build_handle_histories(events)
    roles = infer_roles(events)
    label = cap_path.split("/")[-1]
    print(f"== Handle histories: {label} ==")
    # First-seen order (alloc or first map wins).
    order: Dict[int, int] = {}
    for e in events:
        if e.kind in ("alloc", "map_bar1", "map_uvm"):
            h = e.info[0]
            if h not in order:
                order[h] = len(order)
    for h in sorted(order, key=order.get):
        role = roles.get(h, "unknown")
        print(f"\nHandle 0x{h:08x}  ({role})")
        steps = hist.get(h, [])
        for step in steps:
            for line in format_step(step):
                print(line)


# ── --handle-diff mode ─────────────────────────────────────────────────────

def pick_handle_by_role(events: List[Event],
                        roles: Dict[int, str],
                        role_match: str) -> Optional[int]:
    """Find one handle whose role contains role_match substring.

    Matches on role string (e.g., "likely-d_buf", "likely-h_buf",
    "pb", "usermode-doorbell").  First match in first-seen order
    wins.  Returns None if no match.
    """
    seen: List[int] = []
    order: Dict[int, int] = {}
    for e in events:
        if e.kind in ("alloc", "map_bar1", "map_uvm"):
            h = e.info[0]
            if h not in order:
                order[h] = len(order)
                seen.append(h)
    for h in seen:
        r = roles.get(h, "")
        if role_match in r:
            return h
    return None


def _format_field_row(name: str, val_a: str, val_b: str,
                      col_w: int = 48) -> str:
    marker = "  " if val_a == val_b else "⚠ "
    return f"{marker}{name:<20} {val_a:<{col_w}} {val_b:<{col_w}}"


def print_handle_diff(cap_a_path: str, cap_b_path: str,
                       events_a: List[Event], events_b: List[Event],
                       match_role: Optional[str]) -> int:
    """Render side-by-side parameter diff for one matched role.

    Auto-picks likely-d_buf if --match-role omitted.  Returns 0 on
    success, 2 if no handle matches the role on either side.
    """
    roles_a = infer_roles(events_a)
    roles_b = infer_roles(events_b)
    target = match_role or "likely-d_buf"

    handle_a = pick_handle_by_role(events_a, roles_a, target)
    handle_b = pick_handle_by_role(events_b, roles_b, target)
    if handle_a is None or handle_b is None:
        print(f"error: could not find handle matching role '{target}' on "
              f"{'A' if handle_a is None else 'B'}", file=sys.stderr)
        print("available roles on A:", file=sys.stderr)
        for h, r in sorted(roles_a.items()):
            print(f"  0x{h:08x}  {r}", file=sys.stderr)
        print("available roles on B:", file=sys.stderr)
        for h, r in sorted(roles_b.items()):
            print(f"  0x{h:08x}  {r}", file=sys.stderr)
        return 2

    label_a = cap_a_path.split("/")[-1]
    label_b = cap_b_path.split("/")[-1]
    print(f"== Handle diff for role '{target}' ==")
    print(f"A = {label_a} handle 0x{handle_a:08x}")
    print(f"B = {label_b} handle 0x{handle_b:08x}")
    print()

    hist_a = build_handle_histories(events_a).get(handle_a, [])
    hist_b = build_handle_histories(events_b).get(handle_b, [])

    # Summarize each handle into a flat {field_name: string_value} map
    # that captures the most-recent value for each field across its
    # history.  Good enough for the common case where each handle has
    # one alloc and one/two mappings.
    def summarize(hist: List[Dict]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for step in hist:
            if step["kind"] == "alloc":
                body = step["body"]
                out["ALLOC class"] = step["class_name"]
                if "size" in body:
                    out["ALLOC size"] = f"0x{body['size']:x}"
                if "owner" in body:
                    out["ALLOC owner"] = f"0x{body['owner']:x}"
                if "type" in body:
                    out["ALLOC type"] = f"0x{body['type']:x}"
                if "flags" in body:
                    out["ALLOC flags"] = (f"0x{body['flags']:08x} "
                                          f"({decode_nvos32_flags(body['flags'])})")
                if "attr" in body:
                    out["ALLOC attr"] = (f"0x{body['attr']:08x} "
                                         f"({decode_nvos32_attr(body['attr'])})")
                if "attr2" in body:
                    out["ALLOC attr2"] = (f"0x{body['attr2']:08x} "
                                          f"({decode_nvos32_attr2(body['attr2'])})")
                if "internalflags" in body:
                    out["ALLOC internalflags"] = f"0x{body['internalflags']:08x}"
            elif step["kind"] == "map_bar1":
                out["MAP_MEMORY length"] = f"0x{step['length']:x}"
                out["MAP_MEMORY flags"] = (f"0x{step['flags']:08x} "
                                            f"({decode_nvos33_flags(step['flags'])})")
            elif step["kind"] == "map_uvm":
                out.setdefault("UVM_MAP count", "0")
                out["UVM_MAP count"] = str(int(out["UVM_MAP count"]) + 1)
                out["UVM_MAP last_length"] = f"0x{step['length']:x}"
                out["UVM_MAP last_offset"] = f"0x{step['offset']:x}"
            elif step["kind"] == "control":
                out.setdefault("CONTROL count", "0")
                out["CONTROL count"] = str(int(out["CONTROL count"]) + 1)
            elif step["kind"] == "free":
                out["FREE"] = "yes"
        return out

    sum_a = summarize(hist_a)
    sum_b = summarize(hist_b)

    all_fields: List[str] = []
    for k in list(sum_a) + list(sum_b):
        if k not in all_fields:
            all_fields.append(k)
    # Sort to group ALLOC/MAP_MEMORY/UVM_MAP/CONTROL together.
    all_fields.sort(key=lambda k: (
        0 if k.startswith("ALLOC") else
        1 if k.startswith("MAP_MEMORY") else
        2 if k.startswith("UVM_MAP") else
        3 if k.startswith("CONTROL") else
        4, k))

    col_w = max((len(sum_a.get(k, "—")) for k in all_fields), default=10)
    col_w = max(col_w, max((len(sum_b.get(k, "—")) for k in all_fields), default=10))
    col_w = min(col_w, 70)
    print(f"  {'field':<20} {'A':<{col_w}} {'B':<{col_w}}")
    print(f"  {'-' * 20} {'-' * col_w} {'-' * col_w}")
    for f in all_fields:
        va = sum_a.get(f, "—")
        vb = sum_b.get(f, "—")
        print(_format_field_row(f, va, vb, col_w=col_w))
    return 0


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Decode and diff NVIDIA-driver ioctl sequences from "
                    "two capture files (strace or ftrace)."
    )
    p.add_argument("cap_a", help="first capture (e.g., CUDA reference)")
    p.add_argument("cap_b", help="second capture (e.g., mc_demo)")
    p.add_argument("--input-format",
                   choices=("strace", "ftrace"), default="strace",
                   help="capture file format (default: strace).  "
                        "ftrace gives richer per-handle info from the "
                        "mc1 kernel records.")
    p.add_argument("--match-role", default=None,
                   help="role substring to match when --handle-diff is used "
                        "(default: 'likely-d_buf'). Examples: 'likely-d_buf', "
                        "'likely-h_buf', 'pb', 'usermode-doorbell'.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true",
                      help="per-ioctl count comparison only")
    mode.add_argument("--only", action="store_true",
                      help="list ioctls that appear only in one capture")
    mode.add_argument("--roles", action="store_true",
                      help="per-handle role table for each capture "
                           "(ftrace input only)")
    mode.add_argument("--missing-ioctls", action="store_true",
                      help="list ioctls issued by only one side, with "
                           "hints about what decoded params would "
                           "become visible with additional kernel "
                           "tracepoints (ftrace input only)")
    mode.add_argument("--handle-history", action="store_true",
                      help="per-handle timeline with decoded NVOS32/"
                           "NVOS33 fields (ftrace input only)")
    mode.add_argument("--handle-diff", action="store_true",
                      help="role-matched side-by-side parameter diff "
                           "(ftrace input only)")
    args = p.parse_args()

    try:
        if args.input_format == "strace":
            ev_a = parse_strace(args.cap_a)
            ev_b = parse_strace(args.cap_b)
        else:
            ev_a = parse_ftrace(args.cap_a)
            ev_b = parse_ftrace(args.cap_b)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # --roles / --handle-history / --handle-diff all rely on alloc-body
    # decoding from mc1 records, which only exist in ftrace.
    # --missing-ioctls works on either input; strace actually surfaces
    # more of them than ftrace does, because several escapes are answered
    # inside nvidia.ko's own ioctl switch and never reach RmIoctl, so no
    # rm/ioctl record is emitted for them.  (The uvm/ioctl site does fire
    # for every UVM cmd — it sits at the top of uvm_ioctl, ahead of the
    # switch.)
    needs_ftrace = (
        args.roles or args.handle_history or args.handle_diff
    )
    if needs_ftrace and args.input_format != "ftrace":
        print("error: this mode requires --input-format ftrace",
              file=sys.stderr)
        return 2

    if args.roles:
        print_roles(args.cap_a, ev_a, infer_roles(ev_a))
        print_roles(args.cap_b, ev_b, infer_roles(ev_b))
    elif args.missing_ioctls:
        print_missing_ioctls(args.cap_a, args.cap_b, ev_a, ev_b)
    elif args.handle_history:
        print_handle_history(args.cap_a, ev_a)
        print()
        print_handle_history(args.cap_b, ev_b)
    elif args.handle_diff:
        return print_handle_diff(args.cap_a, args.cap_b, ev_a, ev_b,
                                  args.match_role)
    elif args.summary:
        print_summary(args.cap_a, args.cap_b, ioctl_counts(ev_a), ioctl_counts(ev_b))
    elif args.only:
        print_only(args.cap_a, args.cap_b, ioctl_counts(ev_a), ioctl_counts(ev_b))
    else:
        print_aligned_diff(args.cap_a, args.cap_b,
                           ioctl_sequence(ev_a), ioctl_sequence(ev_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
