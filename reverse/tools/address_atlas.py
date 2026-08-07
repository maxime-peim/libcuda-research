#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
address_atlas.py — resolve every GPU VA in CUDA's pushbuffer method
streams to the concrete allocation it points at.

Pipeline:
  1. Walk merged.ndjson (produced by timeline_merge.py) and build three
     in-memory tables:
       A: allocations      — keyed by hMemory
       B: GPU VA ranges    — keyed by (base, length) from UVM_MAP_EXT
       C: CPU VA ranges    — keyed by (addr, length) from pbcap mmap
  2. For each doorbell event in the merged stream, find the pushbuffer
     snapshot at /tmp/tools/snap-<seq>-dbell-*.bin (a small write-only
     nvidia-uvm mapping).  Walk its bytes as an NVC8B5 method stream;
     for every method that carries an address operand (OFFSET_IN,
     OFFSET_OUT, SET_SEMAPHORE_A/B pairs, etc.), resolve the 57-bit
     GPU VA to Table B → hMemory → Table A.
  3. Emit:
       atlas.json       — the three tables + the per-doorbell resolved
                          methods, in JSON form.
       methods.txt      — human-readable per-method resolution across
                          all doorbell events.

Usage:
  python3 address_atlas.py \\
      --merged /tmp/merged.ndjson \\
      --pbcap-dir /tmp/pbcap \\
      --atlas-out /tmp/atlas.json \\
      --methods-out /tmp/methods.txt

The per-buffer role summary is produced by non_uvm_ledger.py, which consumes
atlas.json.
"""

import argparse
import glob
import json
import os
import re
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

# Import method name tables + header decoder from decode.py.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode import (  # noqa: E402
    decode_header,
    decode_launch_dma,
    METHOD_TYPE_INCR,
    METHOD_TYPE_ONE,
    METHOD_TYPE_IMMD,
    NVC8B5_LAUNCH_DMA,
    NVC8B5_LINE_LENGTH_IN,
    NVC8B5_OFFSET_IN_UPPER,
    NVC8B5_OFFSET_IN_LOWER,
    NVC8B5_OFFSET_OUT_UPPER,
    NVC8B5_OFFSET_OUT_LOWER,
    NVC8B5_SET_SEMAPHORE_A,
    NVC8B5_SET_SEMAPHORE_B,
    NVC8B5_SET_SEMAPHORE_PAYLOAD,
    NVC8B5_SET_SRC_PHYS_MODE,
    NVC8B5_SET_DST_PHYS_MODE,
)
# Auto-scanned catalog of all NVIDIA GPU classes + method registers.
# See class_catalog.py — ~25,800 method registers across ~87 classes,
# extracted from src/common/sdk/nvidia/inc/class/*.h.
from class_catalog import (  # noqa: E402
    CLASS_NAMES as _CATALOG_CLASS_NAMES,
    METHOD_NAMES as _CATALOG_METHOD_NAMES,
    class_name as _catalog_class_name,
    method_name as _catalog_method_name,
)

# Sentinel class id for "no SET_OBJECT seen on this subchannel yet".
# Channel-level methods (SEM_EXECUTE, WFI, MEM_OP_*, SET_OBJECT itself)
# live under NVC86F; our method_name() fallback handles that lookup.
_DEFAULT_CLASS = 0xC86F  # HOPPER_CHANNEL_GPFIFO_A

# CUDA's de-facto subchannel convention on Hopper.  libcuda typically
# issues SET_OBJECT on these bindings at channel initialization (before
# the kernel #DB watchpoint is armed, or via RM ioctls that bypass the
# pushbuffer we observe), so the per-memcpy pushbuffers we decode don't
# re-bind the classes — they assume these bindings.
#
# Verified empirically against the clc0c0.h method catalog:
#   seq=6103 on subch 1 hits 0x01b4 (LOAD_INLINE_DATA), 0x1b00..0x1b0c
#     (SET_REPORT_SEMAPHORE_A..D), 0x1698 (INVALIDATE_SHADER_CACHES_NO_WFI)
#     — all PASCAL_COMPUTE_A methods, so subch 1 → HOPPER_COMPUTE_A.
#   seq=6100 on subch 4 hits 0x0408 (OFFSET_OUT_UPPER), 0x0300 (LAUNCH_DMA),
#     0x0418 (LINE_LENGTH_IN) — all NVC8B5 methods, so subch 4 → HOPPER_DMA_COPY_A.
#
# Override at runtime via SET_OBJECT in the pushbuffer stream if a
# different class is bound mid-stream.
_SUBCH_DEFAULTS: Dict[int, int] = {
    0: 0xCB97,  # HOPPER_A (graphics/3D)
    1: 0xCBC0,  # HOPPER_COMPUTE_A
    2: 0xA140,  # KEPLER_INLINE_TO_MEMORY_B
    3: 0xCBC0,  # HOPPER_COMPUTE_A (secondary compute / P2P)
    4: 0xC8B5,  # HOPPER_DMA_COPY_A
    5: 0xC8B5,  # HOPPER_DMA_COPY_A (alt CE)
    6: 0xC8B5,  # HOPPER_DMA_COPY_A
    7: 0xC8B5,  # HOPPER_DMA_COPY_A
}

def _resolve_method(class_id: int, addr: int) -> str:
    """Return a printable method name for (class, addr).

    class_catalog.method_name handles the lookup, including walking
    inheritance chains (so NVC8B5 inherits from NVC3B5/NVC1B5/NVC0B5,
    HOPPER_COMPUTE_A inherits from NVC9C0/NVC6C0/.../NVC0C0) and
    falling back to NVC86F channel-level.  On miss we return
    "method_0x<addr>" — caller is expected to prefix with the class
    name to disambiguate.
    """
    name = _catalog_method_name(class_id, addr)
    if name is not None:
        return name
    return f"method_0x{addr:04x}"

# Address pair lookup: (UPPER_method_addr -> (LOWER_method_addr, role))
ADDRESS_PAIR_METHODS: Dict[int, Tuple[int, str]] = {
    NVC8B5_OFFSET_IN_UPPER:  (NVC8B5_OFFSET_IN_LOWER,  "CE_src"),
    NVC8B5_OFFSET_OUT_UPPER: (NVC8B5_OFFSET_OUT_LOWER, "CE_dst"),
    NVC8B5_SET_SEMAPHORE_A:  (NVC8B5_SET_SEMAPHORE_B,  "semaphore"),
}

# Classes worth naming in atlas output.  Names verified against the
# SDK headers under src/common/sdk/nvidia/inc/class/.  Entries
# prefixed "CUDA_INTERNAL_*" are classes that libcuda alloc'd heavily
# but don't have an obvious SDK header or are per-Hopper-context
# helpers — named by observed purpose when possible.
RM_CLASS_NAMES: Dict[int, str] = {
    0x0000: "NV01_ROOT",
    0x0040: "NV01_MEMORY_LOCAL_USER",
    0x0041: "NV01_MEMORY_LOCAL_PRIVILEGED",
    0x003E: "NV01_MEMORY_SYSTEM",
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


# ── Pass 1: scan merged.ndjson and build the three tables ──────────────────

class Atlas:
    """Per-capture address atlas.  Fields are JSON-serializable dicts."""
    def __init__(self) -> None:
        # hMemory (int) -> alloc info.  Closed by rm_free.
        self.allocs: Dict[int, Dict[str, Any]] = {}
        # List of GPU VA range dicts: {base, end, length, hMemory, offset,
        #                              ts_mapped, ts_unmapped}.
        # Not an interval tree yet — linear scan is fine for small N.
        self.gpu_ranges: List[Dict[str, Any]] = []
        # Same shape, CPU VA space, keyed on pbcap mmap events.
        self.cpu_ranges: List[Dict[str, Any]] = []
        # Per-doorbell pushbuffer-submission events from the kernel
        # watchpoint (ftrace kind="pb").  Each entry carries the decoded
        # GPFIFO fields plus a resolved GPU-VA range (if the pb_va falls
        # inside any tracked GPU VA).  Sorted by ts_ns.
        self.pb_events: List[Dict[str, Any]] = []
        # Per-submission method-stream bytes (ftrace kind="pb_bytes")
        # with decoded NVC8B5 methods and resolved address operands.
        self.pb_bytes_events: List[Dict[str, Any]] = []
        # Process clock anchor (captured from the first pbcap.init event).
        self.anchor: Optional[Dict[str, int]] = None
        # Per-fd history of open/close events — keyed by fd, value is a
        # chronologically-ordered list of {ts, event, path, pid}.  Used
        # by fd_path_at(fd, ts) to resolve which path an fd pointed at
        # at a given time (fds get reused, so a simple fd→path map
        # would lose information).  Populated from strace openat/close
        # and pbcap open/close events.
        self.fd_history: Dict[int, List[Dict[str, Any]]] = {}
        # Per-fd hClient association: the NV01_ROOT alloc registers a
        # root-client handle against the nvidiactl fd the alloc was
        # issued on.  Value: list of {ts, hClient} entries in order.
        # Used to attribute subsequent allocs on the same fd to the
        # right client.
        self.fd_to_hClient: Dict[int, List[Dict[str, Any]]] = {}
        # Short-lived queue for map_bar1 events awaiting correlation
        # with a subsequent pbcap mmap event.  NOT serialized to
        # atlas.json; drained during build.
        self.pending_rm_maps: List[Dict[str, Any]] = []

        # ── Non-UVM VAS / channel ledger ──────────────────────────────
        # Built from the alloc + virtmem_backing + intermap_call events
        # routed by strace_diff.parse_ftrace.  Lets non_uvm_ledger.py
        # answer "for channel X, which hMemory is USERD vs GPFIFO vs PB
        # vs sema, which carrier hosts each, and what alloc params drove
        # those carriers."
        #
        # channels: keyed by channel hMemory (an *_CHANNEL_GPFIFO_A handle).
        #   {hClient, hVASpace, hUserdMemory, userdOffset, gpFifoOffset,
        #    gpFifoEntries, engineType, ts_alloc, ts_free,
        #    carriers: [hCarrier, ...],
        #    resources: [{role, hSrc, src_class, length, hCarrier,
        #                 dmaOffset, ts_mapped}, ...]}
        self.channels: Dict[int, Dict[str, Any]] = {}
        # carriers: keyed by carrier hMemory (a NV01_MEMORY_VIRTUAL or
        # NV50_MEMORY_VIRTUAL handle).  Captured at virtmem_backing
        # time so subsequent intermap_call events can join carrier→VAS
        # without scanning self.allocs.
        #   {class, hVASpace, vaSize, has_heap, via, aspace, ts}
        self.carriers: Dict[int, Dict[str, Any]] = {}
        # NVOS46 events that didn't bind to a known channel via
        # hUserdMemory or hVASpace.  Same shape as channel.resources
        # entries, kept here so the human reader can see the bindings
        # that landed in shared/device-default VAS.
        self.unattributed_intermaps: List[Dict[str, Any]] = []


def _int_hex(v: Any) -> int:
    """Accept int, or '0x..' / decimal strings."""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith(("0x", "0X")):
            return int(s, 16)
        return int(s)
    return 0


# Role-inference table for per-channel resources, drawn from a
# libcuda ground-truth capture.
# (length, src_class) → role.  src_class names are the hex from the
# class headers (cl0040 NV01_MEMORY_LOCAL_USER, cl003e NV01_MEMORY_SYSTEM).
# A length of 0x200 + LOCAL_USER is USERD; 0x2000 + LOCAL_USER is GPFIFO.
# Anything in NV01_MEMORY_SYSTEM (sysmem) we tag SYSMEM unless the
# channel's hUserdMemory matches the source — in which case it's USERD
# in sysmem.
_NV01_MEMORY_LOCAL_USER = 0x40
_NV01_MEMORY_SYSTEM     = 0x3e


def _infer_resource_role(src_cls: int,
                         length: int,
                         channel: Optional[Dict[str, Any]],
                         src_h: int) -> str:
    """Best-effort role inference for an NVOS46 source hMemory.

    Cross-checks against the channel's NV_CHANNEL_ALLOC_PARAMS-derived
    hUserdMemory when available — that wins over length-based heuristics
    because the kernel told us literally which handle is USERD."""
    if channel is not None:
        if channel.get("hUserdMemory") and src_h == channel["hUserdMemory"]:
            return "USERD"
    if src_cls == _NV01_MEMORY_LOCAL_USER:
        if length == 0x200:
            return "USERD"
        if length == 0x2000:
            return "GPFIFO"
        return "FB"
    if src_cls == _NV01_MEMORY_SYSTEM:
        # Could be PB, sema, or USERD-in-sysmem; without a length
        # signature we just tag SYSMEM and let the consumer infer.
        return "SYSMEM"
    return f"class_0x{src_cls:x}"


def build_atlas(merged_path: str) -> Atlas:
    atlas = Atlas()
    with open(merged_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("ts_ns")
            src = ev.get("src")
            kind = ev.get("kind")
            data = ev.get("data") or {}

            if kind == "pbcap.init" and atlas.anchor is None:
                atlas.anchor = {
                    "mono_ns": int(data.get("mono_ns", 0)),
                    "real_ns": int(data.get("real_ns", 0)),
                    "pid": ev.get("pid"),
                }
                continue

            # alloc events come from ftrace's RM_ALLOC tracepoint via
            # strace_diff's deferred-emission path.
            if kind == "alloc":
                h = _int_hex(data.get("handle", 0))
                if h == 0:
                    continue
                cls = _int_hex(data.get("class", 0))
                atlas.allocs[h] = {
                    "handle":     f"0x{h:08x}",
                    "class":      cls,
                    "class_name": RM_CLASS_NAMES.get(cls, f"class_0x{cls:x}"),
                    "root":       data.get("root"),
                    "parent":     data.get("parent"),
                    "body":       data.get("body", {}),
                    "ts_alloc":   ts,
                    "ts_free":    None,
                }
                # Seed channels[] entry for *_CHANNEL_GPFIFO_A allocs.
                # body carries hVASpace, hUserdMemory, userdOffset,
                # gpFifoOffset, gpFifoEntries (decoded by
                # strace_diff.decode_alloc_body for class 0xc36f /
                # 0xc56f / 0xc86f).  Resources are filled in lazily as
                # virtmem_backing + intermap_call events arrive.
                if cls in (0xc36f, 0xc56f, 0xc86f):
                    body = data.get("body") or {}
                    # _jsonify_dict in timeline_merge.py renders any
                    # int >0xFFFFF as a "0x..." string, so values like
                    # gpFifoOffset and userdOffset arrive as strings
                    # here.  _int_hex normalizes both forms.
                    atlas.channels[h] = {
                        "handle":          f"0x{h:08x}",
                        "class":           cls,
                        "class_name":      RM_CLASS_NAMES.get(cls, f"class_0x{cls:x}"),
                        "hClient":         data.get("root"),
                        "hVASpace":        _int_hex(body.get("hVASpace", 0)),
                        "hUserdMemory":    _int_hex(body.get("hUserdMemory", 0)),
                        "userdOffset":     _int_hex(body.get("userdOffset", 0)),
                        "gpFifoOffset":    _int_hex(body.get("gpFifoOffset", 0)),
                        "gpFifoEntries":   _int_hex(body.get("gpFifoEntries", 0)),
                        "engineType":      _int_hex(body.get("engineType", 0)),
                        "cid":             _int_hex(body.get("cid", 0)),
                        "ts_alloc":        ts,
                        "ts_free":         None,
                        "carriers":        [],
                        "resources":       [],
                    }

                # Seed fd_to_hClient when an NV01_ROOT (class 0) alloc
                # registers a new client.  We can't tell from the RM
                # tracepoint which fd the alloc was issued on — the RM
                # events don't carry fd.  So we tentatively attribute
                # the alloc to any currently-open /dev/nvidiactl fd
                # known at this ts; the post-pass refines via temporal
                # proximity with the strace ioctl that carried RM_ALLOC.
                if cls == 0 and data.get("parent") == data.get("root"):
                    atlas.fd_to_hClient.setdefault("pending", []).append({
                        "ts":      ts,
                        "hClient": h,
                    })
                continue

            if kind == "free":
                h = _int_hex(data.get("handle", 0))
                if h in atlas.allocs:
                    atlas.allocs[h]["ts_free"] = ts
                if h in atlas.channels:
                    atlas.channels[h]["ts_free"] = ts
                continue

            # ── Non-UVM ledger ingestion ──────────────────────────────
            # virtmem_backing: emitted at carrier construct time
            # (T3 in virt_mem_range.c / virtual_mem.c).  Records the
            # carrier handle, its class, and its hVASpace.  We use
            # this lookup table to join intermap_call carriers to
            # channels (both keyed on hVASpace).
            if kind == "virtmem_backing":
                car_h = _int_hex(data.get("hMemory", 0))
                if car_h == 0:
                    continue
                atlas.carriers[car_h] = {
                    "class":    _int_hex(data.get("class", 0)),
                    "hVASpace": _int_hex(data.get("hVASpace", 0)),
                    "vaSize":   _int_hex(data.get("vaSize", 0)),
                    "has_heap": bool(data.get("has_heap")),
                    "via":      data.get("via"),
                    "aspace":   _int_hex(data.get("aspace", 0)),
                    "ts":       ts,
                }
                continue

            # intermap_call: emitted at every NVOS46 dispatch (T2 in
            # virtmemMapTo_IMPL).  Each call binds an hSrc into a
            # carrier's VA range; the question is "which channel does
            # this binding belong to."  Two join strategies, in order:
            #
            # 1. hSrc == channel.hUserdMemory — the NV_CHANNEL_ALLOC_PARAMS
            #    field tells us literally which hMemory is the channel's
            #    USERD.  This binding is unambiguously that channel's
            #    USERD map, regardless of hVASpace.
            #
            # 2. carrier.hVASpace == channel.hVASpace — works for
            #    libcuda contexts that allocate a dedicated FERMI_VASPACE_A
            #    per channel (the morning's 0xcaf00007 ground truth).
            #    Skipped when hVASpace=0, which means "device-default
            #    VAS" — every channel under that device shares it, so
            #    hVASpace alone can't disambiguate, and we fall back to
            #    treating the binding as "this client's, channel
            #    ambiguous" via the post-pass below.
            #
            # Bindings that join to no specific channel are recorded in
            # atlas.unattributed_intermaps, not silently dropped — the
            # post-pass and human can still inspect them.
            if kind == "intermap_call":
                car_h = _int_hex(data.get("hCarrier", 0))
                src_h = _int_hex(data.get("hSrc", 0))
                src_cls = _int_hex(data.get("src_class", 0))
                length = _int_hex(data.get("length", 0))
                dma_off = _int_hex(data.get("dmaOffset", 0))
                car = atlas.carriers.get(car_h)
                car_vas = car.get("hVASpace", 0) if car else 0

                # Strategy 1: USERD identity match.
                target_ch: Optional[int] = None
                for ch_h, ch in atlas.channels.items():
                    if (ch.get("hUserdMemory")
                            and ch["hUserdMemory"] == src_h
                            and ch.get("ts_free") is None):
                        target_ch = ch_h
                        break

                # Strategy 2: dedicated-VAS match (only meaningful
                # when hVASpace != 0).
                if target_ch is None and car_vas != 0:
                    for ch_h, ch in atlas.channels.items():
                        if (ch.get("hVASpace") == car_vas
                                and ch.get("ts_free") is None):
                            target_ch = ch_h
                            break

                role = _infer_resource_role(src_cls, length,
                                            atlas.channels.get(target_ch or 0),
                                            src_h)
                resource = {
                    "role":      role,
                    "hSrc":      src_h,
                    "src_class": src_cls,
                    "length":    length,
                    "hCarrier":  car_h,
                    "dmaOffset": dma_off,
                    "ts_mapped": ts,
                }
                if target_ch is not None:
                    ch = atlas.channels[target_ch]
                    if car_h not in ch["carriers"]:
                        ch["carriers"].append(car_h)
                    ch["resources"].append(resource)
                else:
                    atlas.unattributed_intermaps.append(resource)
                continue

            # UVM-mapped GPU VA range.
            if kind == "map_uvm":
                h = _int_hex(data.get("handle", 0))
                base = _int_hex(data.get("base", 0))
                length = int(data.get("length") or 0)
                offset = int(data.get("offset") or 0)
                if base == 0 or length == 0:
                    continue
                atlas.gpu_ranges.append({
                    "base":      base,
                    "end":       base + length,
                    "length":    length,
                    "offset":    offset,
                    "hMemory":   h,
                    "ts_mapped": ts,
                    "ts_unmapped": None,
                })
                continue

            # RM_MAP_MEMORY — the kernel-side half of a map+mmap pair.
            # Stash in pending_rm_maps keyed by (fd, length) so the
            # subsequent mmap(fd, length, ...) syscall (from strace or
            # pbcap) can join this RM handle onto its resulting user VA.
            # The mmap event ingestion below scans pending_rm_maps
            # newest-first to attach hMemory.
            if kind == "map_bar1":
                h = _int_hex(data.get("handle", 0))
                length = int(data.get("length") or 0)
                fd = data.get("fd")
                if fd is None or length == 0:
                    continue
                atlas.pending_rm_maps.append({
                    "hMemory": h,
                    "length":  length,
                    "flags":   _int_hex(data.get("flags", 0)),
                    "offset":  int(data.get("offset") or 0),
                    "fd":      int(fd),
                    "ts":      ts,
                })
                # Bound the queue — mmap arrives within microseconds
                # of its RM_MAP_MEMORY, so 256 slots is ~100x slack.
                if len(atlas.pending_rm_maps) > 256:
                    atlas.pending_rm_maps.pop(0)
                continue

            # Kernel sysmem tracker add event — the hooked
            # nvidia_mmap_sysmem path emits a tracepoint with the
            # userspace VMA range of each sysmem-backed mapping
            # >= 256 pages.  Under UVM (Paper Finding 1) this range
            # is also the GPU VA range, so we register it as a
            # gpu_range for pb_va resolution.  This covers the
            # libcuda-style path where the pushbuffer lives in a
            # sysmem region made GPU-visible via RM MAP_MEMORY +
            # mmap rather than UVM_MAP_EXTERNAL_ALLOCATION.
            if kind == "sysmem_track_add":
                base   = _int_hex(data.get("base", 0))
                end    = _int_hex(data.get("end", 0))
                length = int(data.get("length") or 0)
                if base == 0 or length == 0 or end <= base:
                    continue
                atlas.gpu_ranges.append({
                    "base":      base,
                    "end":       end,
                    "length":    length,
                    "offset":    0,
                    "hMemory":   0,                 # no RM handle assoc
                    "source":    "kernel_sysmem_track",
                    "kva":       data.get("kva"),
                    "slot":      data.get("slot"),
                    "ts_mapped": ts,
                    "ts_unmapped": None,
                })
                continue

            # Kernel sysmem tracker remove — mark the matching
            # gpu_range as unmapped so _resolve_gpu_va's ts-range
            # gating works correctly.  We match by kva (unique).
            if kind == "sysmem_track_remove":
                kva = data.get("kva")
                for r in atlas.gpu_ranges:
                    if (r.get("source") == "kernel_sysmem_track"
                        and r.get("kva") == kva
                        and r.get("ts_unmapped") is None):
                        r["ts_unmapped"] = ts
                        break
                continue

            # Kernel BAR1 tracker — symmetric to sysmem_track above
            # but keys on phys address (no user_va in the tracepoint).
            # The user_va is derived post-hoc in the canonicalization
            # post-pass by matching (length + ts) against a cpu_range
            # that carries a known hMemory.  Stored here with
            # base=phys as a placeholder so queries can still find it
            # by phys; _resolve_gpu_va won't hit these until user_va
            # is backfilled.
            if kind == "bar1_track_add":
                phys = _int_hex(data.get("phys", 0))
                size = int(data.get("size") or 0)
                if phys == 0 or size == 0:
                    continue
                atlas.gpu_ranges.append({
                    "base":      phys,
                    "end":       phys + size,
                    "length":    size,
                    "offset":    0,
                    "hMemory":   0,   # backfilled in post-pass
                    "source":    "kernel_bar1_track",
                    "kva":       data.get("kva"),
                    "slot":      data.get("slot"),
                    "phys":      phys,
                    "user_va_start": 0,  # backfilled in post-pass
                    "user_va_end":   0,  # backfilled in post-pass
                    "ts_mapped": ts,
                    "ts_unmapped": None,
                })
                continue

            if kind == "bar1_track_remove":
                kva = data.get("kva")
                for r in atlas.gpu_ranges:
                    if (r.get("source") == "kernel_bar1_track"
                        and r.get("kva") == kva
                        and r.get("ts_unmapped") is None):
                        r["ts_unmapped"] = ts
                        break
                continue

            # mmap — userspace VA space.  Accepts events from BOTH
            # pbcap (LD_PRELOAD hook) and strace (raw syscall trace);
            # strace covers any mmap libc/libcuda doesn't route through
            # its CUDA-API hooks (i.e., most nvidia-driver mmaps).
            # Every mmap on an NVIDIA fd is joined against the
            # pending_rm_maps queue on (fd, length) to attach the owning
            # hMemory (and matched entries are popped from the queue).
            if kind == "mmap":
                addr = _int_hex(data.get("addr", 0))
                length = int(data.get("length") or 0)
                if addr == 0 or length == 0:
                    continue
                fd_v = data.get("fd")
                fd = int(fd_v) if fd_v is not None else None

                # Scan pending_rm_maps newest-first for a matching
                # (fd, length) pair.  Pop on match (single-use).
                hMemory = 0
                map_flags = None
                if fd is not None:
                    for i in range(len(atlas.pending_rm_maps) - 1, -1, -1):
                        p = atlas.pending_rm_maps[i]
                        if p["fd"] == fd and p["length"] == length:
                            hMemory = p["hMemory"]
                            map_flags = p["flags"]
                            atlas.pending_rm_maps.pop(i)
                            break

                entry = {
                    "addr":   addr,
                    "end":    addr + length,
                    "length": length,
                    "fd":     fd,
                    "path":   data.get("path"),
                    "offset": _int_hex(data.get("offset", 0)),
                    "prot":   data.get("prot"),
                    "ts_mapped":   ts,
                    "ts_unmapped": None,
                    "src":    src,   # "pbcap" or "strace" (origin of mmap)
                }
                if hMemory:
                    entry["hMemory"]   = hMemory
                    entry["map_flags"] = map_flags
                atlas.cpu_ranges.append(entry)
                continue

            if kind == "munmap":
                addr = _int_hex(data.get("addr", 0))
                for r in atlas.cpu_ranges:
                    if r["addr"] == addr and r["ts_unmapped"] is None:
                        r["ts_unmapped"] = ts
                        break
                continue

            # open/openat — record in per-fd history so the post-pass can
            # attribute any ioctl or mmap on this fd to a path.  Accepts
            # events from both strace (openat) and pbcap (open hooks).
            if kind == "open":
                fd_v = data.get("fd")
                if fd_v is None:
                    continue
                fd = int(fd_v)
                atlas.fd_history.setdefault(fd, []).append({
                    "ts":    ts,
                    "event": "open",
                    "path":  data.get("path"),
                    "pid":   ev.get("pid"),
                    "src":   src,
                })
                continue

            if kind == "close":
                fd_v = data.get("fd") if isinstance(data, dict) else data
                if fd_v is None:
                    continue
                try:
                    fd = int(fd_v)
                except (TypeError, ValueError):
                    continue
                atlas.fd_history.setdefault(fd, []).append({
                    "ts":    ts,
                    "event": "close",
                    "path":  None,
                    "pid":   ev.get("pid"),
                    "src":   src,
                })
                continue

            # Per-doorbell pushbuffer submission (kernel #DB handler).
            # We resolve pb_va immediately so the atlas carries a
            # "hMemory=X class=Y" label alongside the raw fields.  If
            # pb_va isn't inside any tracked GPU VA range at ts, the
            # resolution is None — locate_pushbuffer.py classifies those.
            if kind == "pb":
                pb_va  = _int_hex(data.get("pb_va", 0))
                pb_len = int(data.get("pb_len") or 0)
                resolved = _resolve_gpu_va(atlas, pb_va, ts)
                atlas.pb_events.append({
                    "ts_ns":  ts,
                    "pid":    ev.get("pid"),
                    "seq":    int(data.get("seq") or 0),
                    "chid":   int(data.get("chid") or 0),
                    "idx":    int(data.get("idx") or 0),
                    "entry0": data.get("entry0"),
                    "entry1": data.get("entry1"),
                    "pb_va":  f"0x{pb_va:x}",
                    "pb_len": pb_len,
                    "resolved": resolved,  # None or {hMemory, class, …}
                })
                continue

            # Pushbuffer method-stream bytes (reassembled from the
            # kernel's chunked PB_BYTES: tracepoints).  Decode as an
            # NVC8B5 method stream, fold address-pair methods into
            # single VA entries, and resolve each VA against
            # atlas.gpu_ranges.  All the heavy lifting reuses
            # decode_pushbuffer / fold_address_pairs / _resolve_gpu_va
            # — the same helpers walk_doorbells uses for pbcap
            # snapshots.
            if kind == "pb_bytes":
                seq    = int(data.get("seq") or 0)
                chid   = int(data.get("chid") or 0)
                idx    = int(data.get("idx") or 0)
                nbytes = int(data.get("nbytes") or 0)
                hx     = data.get("hex") or ""
                try:
                    raw = bytes.fromhex(hx)
                except ValueError:
                    raw = b""
                methods_raw = decode_pushbuffer(raw)
                methods = fold_address_pairs(methods_raw)
                decoded: List[Dict[str, Any]] = []
                for m in methods:
                    # Common per-entry fields — class/subchan context.
                    common = {
                        "class_id":   m.get("class_id"),
                        "class_name": m.get("class_name"),
                        "subchan":    m.get("subchan"),
                    }
                    if "va" in m:
                        r = _resolve_gpu_va(atlas, m["va"], ts)
                        decoded.append({
                            **common,
                            "method":   m["method_name"],
                            "role":     m.get("role"),
                            "va":       f"0x{m['va']:x}",
                            "resolved": r,
                        })
                    elif "imm" in m:
                        decoded.append({
                            **common,
                            "method": m["method_name"],
                            "type":   m.get("type"),
                            "imm":    m.get("imm"),
                        })
                    else:
                        d = m.get("data")
                        entry = {
                            **common,
                            "method": m["method_name"],
                            "type":   m.get("type"),
                        }
                        if d is not None:
                            entry["data"] = f"0x{d:08x}"
                            if m["method_name"] == "LAUNCH_DMA":
                                entry["launch_dma"] = decode_launch_dma(d)
                        # SET_OBJECT carries bound_class_id /
                        # bound_class_name — surface those.
                        if "bound_class_id" in m:
                            entry["bound_class_id"]   = m["bound_class_id"]
                            entry["bound_class_name"] = m["bound_class_name"]
                        decoded.append(entry)
                atlas.pb_bytes_events.append({
                    "ts_ns":  ts,
                    "pid":    ev.get("pid"),
                    "seq":    seq,
                    "chid":   chid,
                    "idx":    idx,
                    "nbytes": nbytes,
                    "methods": decoded,
                })
                continue

    # ── Post-pass 1: cpu_range-driven hMemory + user_va backfill ──
    #
    # Tracker gpu_ranges (kernel_sysmem_track, kernel_bar1_track) carry
    # user_va/phys + size but no RM handle.  cpu_ranges (Edit 3) now
    # carry hMemory via the pending_rm_maps join.  For each tracker
    # range, find a cpu_range with matching `length` and temporally-
    # adjacent `ts_mapped` (within 10 ms) AND hMemory!=0 — that's a
    # strong match.  Backfill hMemory; for BAR1, additionally backfill
    # user_va_start/end so _resolve_gpu_va finds it by user VA too.
    WINDOW_NS = 10_000_000  # 10 ms
    for r in atlas.gpu_ranges:
        src_name = r.get("source")
        if src_name not in ("kernel_sysmem_track", "kernel_bar1_track"):
            continue
        if r.get("hMemory"):
            continue
        rng_len = r["length"]
        rng_ts  = r["ts_mapped"]
        best   = None
        best_dt = None
        for c in atlas.cpu_ranges:
            if c.get("length") != rng_len:
                continue
            if not c.get("hMemory"):
                continue
            c_ts = c.get("ts_mapped")
            if c_ts is None:
                continue
            dt = abs(rng_ts - c_ts)
            if dt > WINDOW_NS:
                continue
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best    = c
        if best is not None:
            r["hMemory"] = best["hMemory"]
            a = atlas.allocs.get(best["hMemory"])
            if a:
                r["class_name"] = a.get("class_name")
            # BAR1 tracker had placeholder user_va=0; fill from cpu_range.
            if src_name == "kernel_bar1_track":
                r["user_va_start"] = best["addr"]
                r["user_va_end"]   = best["end"]

    # ── Post-pass 2 (fallback): heuristic sysmem backfill for tracker ──
    # ──                entries that DIDN'T match a cpu_range         ──
    #
    # (e.g., UVM-mapped sysmem that doesn't go through the pbcap mmap
    # hook, or captures where the mmap event was lost.)  Same size+ts
    # heuristic we shipped earlier today.
    for r in atlas.gpu_ranges:
        if r.get("source") != "kernel_sysmem_track":
            continue
        if r.get("hMemory"):
            continue
        rng_len  = r["length"]
        rng_ts   = r["ts_mapped"]
        best_h   = 0
        best_cls = None
        best_dt  = None
        for a in atlas.allocs.values():
            if a.get("class_name") != "NV01_MEMORY_SYSTEM":
                continue
            body = a.get("body") or {}
            sz = body.get("size", 0)
            if isinstance(sz, str):
                sz = _int_hex(sz)
            PAGE = 0x1000
            sz_aligned = (sz + PAGE - 1) & ~(PAGE - 1)
            if sz_aligned != rng_len:
                continue
            ts_alloc = a.get("ts_alloc")
            if ts_alloc is None or ts_alloc > rng_ts:
                continue
            dt = rng_ts - ts_alloc
            if best_dt is None or dt < best_dt:
                best_dt  = dt
                best_h   = _int_hex(a.get("handle", 0))
                best_cls = a.get("class_name")
        if best_h:
            r["hMemory"] = best_h
            r["class_name"] = best_cls

    # ── Post-pass 3: per-handle events[] chronological timeline ──
    #
    # For each hMemory, build a sorted list of every event that
    # touched it.  Populated by re-walking merged.ndjson and stamping
    # events onto allocations[handle]["events"].  Matches:
    #   - alloc (ts_alloc from the allocation itself)
    #   - map_bar1 (via pending_rm_maps history — since we popped, we
    #     instead rebuild from cpu_ranges which carry hMemory + ts)
    #   - mmap (cpu_ranges with hMemory set)
    #   - map_uvm (gpu_ranges with source!=kernel_* and hMemory set)
    #   - control (RM_CONTROL events — we didn't store these yet;
    #     rewalk the merged.ndjson to collect per-handle control calls)
    #   - free (ts_free)
    #
    # This is a second linear pass over merged.ndjson — cheap, and
    # keeps the main-loop code short.  We emit events sorted by ts.
    for h, a in atlas.allocs.items():
        evs: List[Dict[str, Any]] = []
        if a.get("ts_alloc") is not None:
            evs.append({"ts": a["ts_alloc"], "event": "alloc",
                        "class": a.get("class_name")})
        if a.get("ts_free") is not None:
            evs.append({"ts": a["ts_free"], "event": "free"})
        a["events"] = evs  # placeholder; filled below

    for c in atlas.cpu_ranges:
        h = c.get("hMemory")
        if not h:
            continue
        a = atlas.allocs.get(h)
        if not a:
            continue
        a["events"].append({
            "ts":    c["ts_mapped"],
            "event": "mmap",
            "fd":    c.get("fd"),
            "path":  c.get("path"),
            "addr":  f"0x{c['addr']:x}",
            "length": c["length"],
        })
        if c.get("ts_unmapped") is not None:
            a["events"].append({
                "ts":    c["ts_unmapped"],
                "event": "munmap",
                "addr":  f"0x{c['addr']:x}",
            })

    for g in atlas.gpu_ranges:
        h = g.get("hMemory")
        if not h:
            continue
        a = atlas.allocs.get(h)
        if not a:
            continue
        a["events"].append({
            "ts":     g["ts_mapped"],
            "event":  "gpu_map",
            "source": g.get("source", "uvm"),
            "base":   f"0x{g['base']:x}",
            "length": g["length"],
        })
        if g.get("ts_unmapped") is not None:
            a["events"].append({
                "ts":     g["ts_unmapped"],
                "event":  "gpu_unmap",
                "source": g.get("source", "uvm"),
                "base":   f"0x{g['base']:x}",
            })

    # Control calls and other ioctls targeting a specific handle —
    # walk merged.ndjson once more to pick them up.
    with open(merged_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("kind")
            data = ev.get("data") or {}
            ts = ev.get("ts_ns")
            if kind == "control":
                h = _int_hex(data.get("hObject", 0))
                a = atlas.allocs.get(h)
                if a:
                    a["events"].append({
                        "ts":    ts,
                        "event": "control",
                        "ctrlcmd": data.get("ctrlcmd"),
                        "hClient": data.get("hClient"),
                        "paramsSize": data.get("paramsSize"),
                    })

    # Sort per-handle events by ts.
    for a in atlas.allocs.values():
        a["events"].sort(key=lambda e: e.get("ts") or 0)

    # ── Post-pass 4: re-resolve pb_events + per-method resolves ──
    #
    # Eager resolves during event ingestion happened against un-patched
    # gpu_ranges (kernel_sysmem_track hMemory=0, kernel_bar1_track
    # user_va=0).  Redo them now that the post-passes above are done.
    for pe in atlas.pb_events:
        va = _int_hex(pe.get("pb_va", 0))
        ts = pe.get("ts_ns")
        pe["resolved"] = _resolve_gpu_va(atlas, va, ts)
    for pbe in atlas.pb_bytes_events:
        ts = pbe.get("ts_ns")
        for m in pbe.get("methods") or []:
            if "va" not in m:
                continue
            va_s = m.get("va")
            va_i = _int_hex(va_s) if isinstance(va_s, str) else int(va_s or 0)
            m["resolved"] = _resolve_gpu_va(atlas, va_i, ts)

    return atlas


# ── Pass 2: walk a pushbuffer snapshot and decode method stream ────────────

def _resolve_gpu_va(atlas: Atlas, va: int, ts_ns: Optional[int]) -> Optional[Dict[str, Any]]:
    """Find the GPU VA range containing `va` at time `ts_ns`.  Returns a
    dict with range + alloc info, or None if unresolved."""
    for r in atlas.gpu_ranges:
        if va < r["base"] or va >= r["end"]:
            continue
        # Range was mapped before this timestamp, and if unmapped, after
        # — or ts_ns is None meaning we can't check.
        if ts_ns is not None:
            if r["ts_mapped"] is not None and r["ts_mapped"] > ts_ns:
                continue
            if r["ts_unmapped"] is not None and r["ts_unmapped"] < ts_ns:
                continue
        h = r["hMemory"]
        alloc = atlas.allocs.get(h)
        return {
            "gpu_va":   f"0x{va:x}",
            "offset":   va - r["base"],
            "base":     f"0x{r['base']:x}",
            "length":   r["length"],
            "hMemory":  f"0x{h:08x}",
            "class":    alloc.get("class_name") if alloc else None,
            "size":     (alloc or {}).get("body", {}).get("size"),
        }
    return None


def is_valid_method_stream(data: bytes, min_valid: int = 3) -> bool:
    """Quick validity check: does this buffer start with at least
    `min_valid` recognizable NVC8B5 method headers?

    A header is "recognizable" if (a) type is one of INCR(1), ONE(3),
    or IMMD(5), and (b) the method address decodes to one of the known
    NVC8B5 methods (LAUNCH_DMA, OFFSET_IN_UPPER/LOWER, etc.).  Random
    heap/gc/struct content will rarely pass this test."""
    if len(data) < 16:
        return False
    words = struct.unpack_from(f"<{min(64, len(data) // 4)}I", data)
    valid = 0
    i = 0
    while i < len(words) and valid < min_valid:
        hdr = words[i]
        if hdr == 0:
            break  # end of stream
        type_, count, _, maddr = decode_header(hdr)
        if type_ not in (1, 3, 5):
            return False
        # Validity probe: any known NVC8B5 or channel-level (NVC86F)
        # method register at this address.  Random heap bytes rarely
        # satisfy this when combined with the type-field check above.
        if ((0xC8B5, maddr) not in _CATALOG_METHOD_NAMES and
                (0xC86F, maddr) not in _CATALOG_METHOD_NAMES):
            return False
        valid += 1
        # Skip past the data dwords for INCR/ONE; IMMD has none.
        if type_ in (1, 3):
            i += 1 + count
        else:
            i += 1
    return valid >= min_valid


def decode_pushbuffer(data: bytes) -> List[Dict[str, Any]]:
    """Parse an NVC86F-encoded method stream from a pushbuffer snapshot.
    Returns a list of method dicts with full per-method classification:
    {method_addr, method_name, class_id, class_name, subchan, type,
     (data | imm)}.

    Subchannel dispatch: the channel binds a class to each subchannel
    via NVC86F_SET_OBJECT (method_addr=0, class_id in data[15:0]).
    We track per-subchannel bindings so methods on subch 4 dispatch
    against HOPPER_DMA_COPY_A, subch 1 against HOPPER_COMPUTE_A, etc.
    Channel-level methods (WFI, SEM_EXECUTE, MEM_OP_*) on any subch
    fall back to NVC86F via _resolve_method.

    Address-pair folding (UPPER+LOWER → 57-bit VA) happens in
    fold_address_pairs below, not here, because folding needs to
    inspect adjacent entries.

    Stops on the first all-zero header (trailing padding)."""
    if len(data) < 4:
        return []
    words = struct.unpack_from(f"<{len(data) // 4}I", data)
    out: List[Dict[str, Any]] = []
    i = 0
    n = len(words)
    # Per-channel state: subchannel → class_id.  Seeded with CUDA's
    # de-facto subchannel convention (see _SUBCH_DEFAULTS); updated
    # dynamically by SET_OBJECT (method_addr==0) if the stream binds
    # something different.
    subch_class: Dict[int, int] = dict(_SUBCH_DEFAULTS)

    while i < n:
        hdr = words[i]
        if hdr == 0:
            # All-zero — assume end of method stream.
            break
        type_, count, subchan, maddr = decode_header(hdr)
        i += 1
        # Effective class for this header's target subchannel.  If we
        # haven't seen a SET_OBJECT for this subch, treat methods as
        # channel-level (NVC86F) — _resolve_method also falls through
        # to NVC86F as a final step anyway.
        eff_class = subch_class.get(subchan, _DEFAULT_CLASS)
        eff_class_name = _catalog_class_name(eff_class)

        def _entry(addr: int, **extra) -> Dict[str, Any]:
            return {
                "method_addr": addr,
                "method_name": _resolve_method(eff_class, addr),
                "class_id":    eff_class,
                "class_name":  eff_class_name,
                "subchan":     subchan,
                **extra,
            }

        if type_ == METHOD_TYPE_IMMD:
            # Inline: single immediate value in the header.
            imm = (hdr >> 16) & 0x1fff
            out.append(_entry(maddr, type="IMMD", imm=imm))
            continue

        # INCR or ONE: `count` data dwords follow.
        dwords: List[int] = []
        for _ in range(count):
            if i >= n:
                break
            dwords.append(words[i])
            i += 1

        # Emit one entry per data dword for INCR (each targets
        # maddr + 4*k).  For ONE, all go to maddr.
        for k, dw in enumerate(dwords):
            this_addr = maddr + (4 * k if type_ == METHOD_TYPE_INCR else 0)
            this_class = subch_class.get(subchan, _DEFAULT_CLASS)
            this_class_name = _catalog_class_name(this_class)
            entry = {
                "method_addr": this_addr,
                "method_name": _resolve_method(this_class, this_addr),
                "class_id":    this_class,
                "class_name":  this_class_name,
                "subchan":     subchan,
                "type":        "INCR" if type_ == METHOD_TYPE_INCR else "ONE",
                "data":        dw,
            }
            # SET_OBJECT (method_addr 0 on NVC86F): bind the subchannel
            # to the class in data[15:0].  Annotate the entry so the
            # renderer can show "class=<id> <name>", and update the
            # subch_class map so subsequent methods in this stream
            # dispatch correctly.
            if this_addr == 0:
                bound_class = dw & 0xFFFF
                subch_class[subchan] = bound_class
                entry["bound_class_id"]   = bound_class
                entry["bound_class_name"] = _catalog_class_name(bound_class)
                # Re-resolve the method name against the NEW binding
                # for the output, since SET_OBJECT is itself on NVC86F
                # but from this point on we're dispatching the bound
                # class — keep the "SET_OBJECT" label intact though.
                entry["method_name"] = "SET_OBJECT"
                entry["class_id"]    = _DEFAULT_CLASS
                entry["class_name"]  = _catalog_class_name(_DEFAULT_CLASS)
            out.append(entry)
    return out


def fold_address_pairs(methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine adjacent *_UPPER and *_LOWER methods into a single 57-bit
    VA entry.  Order matters: the UPPER variant always immediately
    precedes the LOWER variant in a well-formed method stream.

    Preserves class_id / class_name / subchan from the UPPER entry on
    the folded result so downstream renderers can still show which
    class the address pair was issued against."""
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(methods):
        m = methods[i]
        if (m["method_addr"] in ADDRESS_PAIR_METHODS
                and i + 1 < len(methods)
                and methods[i + 1]["method_addr"] == ADDRESS_PAIR_METHODS[m["method_addr"]][0]):
            upper = m.get("data", 0)
            lower = methods[i + 1].get("data", 0)
            va = (upper << 32) | lower
            role = ADDRESS_PAIR_METHODS[m["method_addr"]][1]
            # Derive the folded method name from whichever suffix the
            # UPPER register uses (UPPER, _A for SET_SEMAPHORE_A+B).
            up_name = m["method_name"]
            if "_UPPER" in up_name:
                folded_name = up_name.replace("_UPPER", "_VA")
            elif up_name.endswith("_A"):
                folded_name = up_name[:-2] + "_VA"
            else:
                folded_name = up_name + "_VA"
            out.append({
                "method_name": folded_name,
                "method_pair": [m["method_name"], methods[i + 1]["method_name"]],
                "role":        role,
                "va":          va,
                "class_id":    m.get("class_id"),
                "class_name":  m.get("class_name"),
                "subchan":     m.get("subchan"),
            })
            i += 2
            continue
        out.append(m)
        i += 1
    return out


# ── Driver: walk merged timeline for doorbell events, resolve each ─────────

_SNAP_ANY_RE = re.compile(
    r"^snap-(\d+)-dbell-(?P<tag>[a-z0-9-]+)-(?P<addr>0x[0-9a-f]+)-len(?P<len>[0-9a-f]+)\.bin$"
)


def find_pushbuffer_candidates(pbcap_dir: str, seq: int) -> List[str]:
    """Return every doorbell snapshot for this seq.  We do not pre-filter
    by size because CUDA may embed its method stream at a non-zero
    offset within a larger staging-pool mapping.  The validator in
    find_method_stream_in_data scans each snapshot for a recognizable
    NVC8B5 method stream; we pick whichever one contains the best
    match."""
    pattern = os.path.join(pbcap_dir, f"snap-{seq:05d}-dbell-*.bin")
    all_snaps = glob.glob(pattern)
    candidates: List[Tuple[int, str]] = []
    for p in all_snaps:
        name = os.path.basename(p)
        m = _SNAP_ANY_RE.match(name)
        if not m:
            continue
        length = int(m.group("len"), 16)
        candidates.append((length, p))
    candidates.sort()
    return [p for _, p in candidates]


def find_method_stream_in_data(data: bytes) -> int:
    """Scan `data` at 4-byte-aligned offsets for the start of a
    recognizable NVC8B5 method stream.  Returns the byte offset of the
    first valid stream found, or -1 if none.

    This is how we cope with CUDA embedding its pushbuffer method
    bytes at a non-zero offset inside a larger mapping (e.g., inside
    a 2 MiB staging buffer): the method stream is a short (<256 dwords)
    block of well-structured headers at some offset we don't know
    a priori.
    """
    if len(data) < 64:
        return -1
    # Scan up to the first 64 KiB of the mapping — CUDA's command
    # streams live near the start of pushbuffer regions.
    max_scan = min(len(data), 64 * 1024)
    # Look for runs of at least 3 valid consecutive method headers at
    # 4-byte-aligned starts.
    for off in range(0, max_scan - 64, 4):
        if is_valid_method_stream(data[off:]):
            return off
    return -1


def walk_doorbells(merged_path: str, atlas: Atlas,
                   pbcap_dir: str) -> List[Dict[str, Any]]:
    """For each doorbell event in merged.ndjson, find and decode the
    matching pushbuffer snapshot and resolve every VA."""
    results: List[Dict[str, Any]] = []
    with open(merged_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") != "doorbell":
                continue
            seq = ev.get("data", {}).get("seq")
            if seq is None:
                continue
            candidates = find_pushbuffer_candidates(pbcap_dir, int(seq))
            dbell: Dict[str, Any] = {
                "seq":      int(seq),
                "ts_ns":    ev.get("ts_ns"),
                "bar0":     ev.get("data", {}).get("bar0"),
                "token":    ev.get("data", {}).get("token"),
                "tid":      ev.get("tid"),
                "snapshots_considered": len(candidates),
                "methods":  [],
                "resolved": [],
                "unresolved": [],
            }
            # Try every candidate snapshot; pick the one with the most
            # decoded methods (ties broken by smaller file first —
            # candidates are already sorted ascending by size).  BAR0
            # pages decode to 0 methods (that aperture has no backing
            # store, so it reads back as zero), and
            # bookkeeping pages likewise have no method stream — so the
            # real pushbuffer wins on method count.
            best_methods: List[Dict[str, Any]] = []
            best_path: Optional[str] = None
            best_off: int = 0
            for snap_path in candidates:
                try:
                    with open(snap_path, "rb") as sf:
                        data = sf.read()
                except OSError:
                    continue
                # Find the offset where a valid method stream begins.
                off = find_method_stream_in_data(data)
                if off < 0:
                    continue
                methods = decode_pushbuffer(data[off:])
                if len(methods) > len(best_methods):
                    best_methods = methods
                    best_path = snap_path
                    best_off = off
            if best_path is not None:
                dbell["pushbuffer_offset"] = f"0x{best_off:x}"
            if best_path:
                dbell["pushbuffer_snapshot"] = os.path.basename(best_path)
                methods = fold_address_pairs(best_methods)
                for m in methods:
                    if "va" in m:
                        resolved = _resolve_gpu_va(atlas, m["va"], dbell["ts_ns"])
                        entry = {
                            "method":   m["method_name"],
                            "role":     m.get("role"),
                            "va":       f"0x{m['va']:x}",
                            "resolved": resolved,
                        }
                        (dbell["resolved"] if resolved else
                         dbell["unresolved"]).append(entry)
                        dbell["methods"].append(entry)
                    else:
                        dbell["methods"].append({
                            "method": m["method_name"],
                            "type":   m.get("type"),
                            "data":   m.get("data"),
                            "imm":    m.get("imm"),
                        })
            results.append(dbell)
    return results


# ── Rendering ──────────────────────────────────────────────────────────────

def _pb_event_key(p: Dict[str, Any]) -> Tuple:
    """Key for pb_event run-length collapsing.  Consecutive events
    that share (chid, hMemory, pb_len) get folded together."""
    r = p.get("resolved") or {}
    return (p.get("chid"), r.get("hMemory"), p.get("pb_len"))


def write_pb_events_txt(f, pb_events: List[Dict[str, Any]],
                        cudaMemcpys: Optional[List[Dict[str, Any]]] = None,
                        t0: Optional[int] = None) -> None:
    """Append a PB-EVENTS section showing one line per kernel-side
    pushbuffer submission (from mc1 pb/submit ftrace records), keyed by
    seq.  Each line carries (chid, idx, pb_va, pb_len) plus a resolved
    hMemory/class/size if pb_va lands inside any atlas.gpu_ranges
    interval at that timestamp.  When resolved is None, the line shows
    "UNRESOLVED".

    Consecutive events with the same (chid, hMemory, pb_len) and
    monotonically-increasing seq are collapsed into a single summary
    line of the form:

      seq=A..B (×N) chid=C pb_len=L → hMemory=H class=K offset=F..L sizeMiB=S

    This trades line-by-line detail for structural legibility — a
    long CE fill loop collapses from 60 near-identical rows to one
    line.  The full raw data remains in atlas.json's pb_events.

    If `cudaMemcpys` is provided, pb_events that fall inside a
    cudaMemcpy.enter/exit interval are bracketed with header/footer
    lines showing the direction, size, and relative timestamp (E)."""
    if not pb_events:
        return
    n_resolved = sum(1 for p in pb_events if p["resolved"] is not None)
    f.write(f"\n=== PB_EVENTS: {len(pb_events)} doorbells, "
            f"{n_resolved} resolved, "
            f"{len(pb_events) - n_resolved} unresolved ===\n")

    def _format_one(p: Dict[str, Any]) -> str:
        """Single-event line (used when a run has length 1)."""
        r = p["resolved"]
        head = (f"seq={p['seq']:>4} chid={p['chid']:>3} idx={p['idx']:>3} "
                f"pb_va={p['pb_va']} pb_len={p['pb_len']}")
        if r is None:
            return f"{head}  → UNRESOLVED (pb_va not in any gpu_range)\n"
        sz = r.get("size")
        sz_int: Optional[int] = None
        if isinstance(sz, int):
            sz_int = sz
        elif isinstance(sz, str) and sz.startswith(("0x", "0X")):
            try: sz_int = int(sz, 16)
            except ValueError: pass
        sz_str = (f"sizeMiB={sz_int/(1024*1024):.2f}"
                  if sz_int is not None else "sizeMiB=?")
        return (f"{head}  → hMemory={r['hMemory']} "
                f"class={r['class']} offset={r['offset']} {sz_str}\n")

    def _format_run(run: List[Dict[str, Any]]) -> str:
        """Collapsed line for a run of ≥2 consecutive same-key events."""
        first = run[0]; last = run[-1]
        r = first["resolved"] or {}
        sz = r.get("size")
        sz_int: Optional[int] = None
        if isinstance(sz, int): sz_int = sz
        elif isinstance(sz, str) and sz.startswith(("0x", "0X")):
            try: sz_int = int(sz, 16)
            except ValueError: pass
        sz_str = (f"sizeMiB={sz_int/(1024*1024):.2f}"
                  if sz_int is not None else "sizeMiB=?")
        off_lo = r.get("offset", 0)
        off_hi = (last["resolved"] or {}).get("offset", 0)
        idx_lo = first["idx"]; idx_hi = last["idx"]
        head = (f"seq={first['seq']:>4}..{last['seq']:<4}"
                f" (×{len(run):<3})"
                f" chid={first['chid']:>3}"
                f" idx={idx_lo}..{idx_hi}"
                f" pb_len={first['pb_len']}")
        if r:
            return (f"{head}  → hMemory={r.get('hMemory')} "
                    f"class={r.get('class')} "
                    f"offset={off_lo}..{off_hi} {sz_str}\n")
        return f"{head}  → UNRESOLVED\n"

    # cudaMemcpy bracketing helpers (E).
    sorted_memcpys = sorted(cudaMemcpys or [],
                            key=lambda m: m.get("ts_enter") or 0)

    def _rel(ts: Optional[int]) -> str:
        if ts is None or t0 is None:
            return "t+?"
        return f"t+{(ts - t0)/1e9:.4f}s"

    def _fmt_bytes(n: Optional[int]) -> str:
        if not n: return "?"
        if n >= 1024*1024:
            return f"{n/(1024*1024):.1f} MiB"
        if n >= 1024:
            return f"{n/1024:.1f} KiB"
        return f"{n} B"

    # State tracking: which memcpy is currently open (by index into
    # sorted_memcpys).  We open/close as pb_event timestamps cross
    # each memcpy's [ts_enter, ts_exit] interval.
    active_memcpy: Optional[int] = None

    def _maybe_open_close(ts: Optional[int]):
        nonlocal active_memcpy
        if ts is None:
            return
        # Close an active memcpy if we've moved past its ts_exit.
        if (active_memcpy is not None
                and ts > (sorted_memcpys[active_memcpy].get("ts_exit") or 0)):
            m = sorted_memcpys[active_memcpy]
            f.write(f"──── /cudaMemcpy {m['direction']} "
                    f"{_fmt_bytes(m['nbytes'])} @ {_rel(m.get('ts_exit'))} "
                    f"────\n")
            active_memcpy = None
        # Open a memcpy if ts is inside one and nothing is active.
        if active_memcpy is None:
            for i, m in enumerate(sorted_memcpys):
                enter = m.get("ts_enter")
                exit_ = m.get("ts_exit")
                if enter is None or exit_ is None:
                    continue
                if enter <= ts <= exit_:
                    f.write(f"\n──── cudaMemcpy {m['direction']} "
                            f"{_fmt_bytes(m['nbytes'])} @ {_rel(enter)} ────\n")
                    active_memcpy = i
                    break

    # Group consecutive events by (chid, hMemory, pb_len).  A new key
    # OR a non-monotonic seq breaks the run.  Runs of length 1 print
    # with the original one-event format for maximal backward-compat.
    #
    # Bracket ordering (critical): flush any in-progress run BEFORE
    # calling _maybe_open_close.  Reason: the active run's FIRST
    # event's ts was used (earlier) to decide whether a bracket is
    # open; emitting the run AFTER a newly-opened bracket would
    # pull pb_events whose ts is BEFORE the bracket's enter into the
    # bracket.  Always: first render the previous run (if any), THEN
    # re-evaluate bracket state based on the NEW event's ts.
    run: List[Dict[str, Any]] = []
    run_key = None
    for p in pb_events:
        k = _pb_event_key(p)
        if (run_key is not None
                and k == run_key
                and p["seq"] == run[-1]["seq"] + 1):
            # Still part of the active run — no need to revisit
            # bracket state (all run members share the same ts-region
            # for bracket purposes, within µs of each other).
            run.append(p)
            continue
        # Run broke.  Flush the previous run first, so its output
        # settles OUTSIDE any bracket opened for the new event.
        if run:
            f.write(_format_run(run) if len(run) >= 2 else _format_one(run[0]))
        # Now evaluate bracket state based on the NEW event.
        _maybe_open_close(p.get("ts_ns"))
        run = [p]
        run_key = k
    if run:
        f.write(_format_run(run) if len(run) >= 2 else _format_one(run[0]))
    # Final close of any trailing active memcpy.
    if active_memcpy is not None:
        m = sorted_memcpys[active_memcpy]
        f.write(f"──── /cudaMemcpy {m['direction']} "
                f"{_fmt_bytes(m['nbytes'])} @ {_rel(m.get('ts_exit'))} ────\n")


def _format_method_label(m: Dict[str, Any]) -> str:
    """Render a per-method label as <CLASS>/<METHOD>.  If the method
    name came from the unknown-fallback branch (method_0xNNNN),
    reformat as <CLASS>::method_0xNNNN so the reader sees the class
    alongside.  When class context is missing, fall back to bare
    method."""
    name = m.get("method") or "?"
    class_name = m.get("class_name")
    if not class_name:
        return name
    if name.startswith("method_0x"):
        # Unknown method — emphasize with double-colon so it's visually
        # distinct from the CLASS/METHOD form for named methods.
        return f"{class_name}::{name}"
    return f"{class_name}/{name}"


def write_pb_bytes_txt(f, pb_bytes_events: List[Dict[str, Any]]) -> None:
    """Per-submission decoded method stream from the kernel's PB_BYTES:
    tracepoints.  For each event emits a header + one line per method,
    showing address-pair operands resolved against atlas.gpu_ranges.

    Each method line is rendered as <CLASS>/<METHOD> (named) or
    <CLASS>::method_0xNNNN (unknown method on a known class).
    Subchannel is shown in a column so you can tell which engine
    the method targets (CE vs. compute)."""
    if not pb_bytes_events:
        return
    total_methods = sum(len(p["methods"]) for p in pb_bytes_events)
    f.write(f"\n=== PB_BYTES_DECODE: {len(pb_bytes_events)} submissions, "
            f"{total_methods} decoded methods ===\n")
    def _emit_one_method(m: Dict[str, Any]) -> None:
        label = _format_method_label(m)
        subch = m.get("subchan")
        sch = f"sch{subch}" if subch is not None else "sch?"
        if "va" in m:
            r = m.get("resolved")
            if r:
                f.write(f"  [{sch}] {label:<48} role={m.get('role','-'):<10} "
                        f"va={m['va']} → hMemory={r['hMemory']} "
                        f"class={r['class']} offset={r['offset']}\n")
            else:
                f.write(f"  [{sch}] {label:<48} role={m.get('role','-'):<10} "
                        f"va={m['va']} → UNRESOLVED\n")
        elif "imm" in m:
            f.write(f"  [{sch}] {label:<48} imm=0x{m['imm']:x}\n")
        elif "launch_dma" in m:
            f.write(f"  [{sch}] {label:<48} data={m['data']} "
                    f"({m['launch_dma']})\n")
        elif "bound_class_id" in m:
            # SET_OBJECT: emphasize the binding.
            f.write(f"  [{sch}] {label:<48} class=0x{m['bound_class_id']:04x} "
                    f"({m['bound_class_name']})\n")
        elif "data" in m:
            f.write(f"  [{sch}] {label:<48} data={m['data']}\n")
        else:
            f.write(f"  [{sch}] {label}\n")

    def _method_run_key(m: Dict[str, Any]) -> Tuple:
        """Consecutive identical methods (same method_name + subchan,
        both carrying only 'data') are collapsible — typically
        LOAD_INLINE_DATA streams that upload compute SASS."""
        if not ("data" in m and "va" not in m and "imm" not in m
                and "launch_dma" not in m and "bound_class_id" not in m):
            return None
        return (m.get("method"), m.get("subchan"))

    for p in pb_bytes_events:
        f.write(f"\n--- seq={p['seq']} chid={p['chid']} idx={p['idx']} "
                f"nbytes={p['nbytes']} ---\n")
        methods = p["methods"]
        i = 0
        while i < len(methods):
            m = methods[i]
            key = _method_run_key(m)
            if key is None:
                _emit_one_method(m)
                i += 1
                continue
            # Scan forward while key matches.
            j = i + 1
            while j < len(methods) and _method_run_key(methods[j]) == key:
                j += 1
            run_len = j - i
            if run_len == 1:
                _emit_one_method(m)
            else:
                # Collapse: show a summary line + the first 3 / last 1
                # dword as previews so anomalies in the stream are
                # still visible.
                run = methods[i:j]
                label = _format_method_label(m)
                subch = m.get("subchan")
                sch = f"sch{subch}" if subch is not None else "sch?"
                previews_head = ", ".join(r["data"] for r in run[:3])
                previews_tail = run[-1]["data"] if run_len > 4 else ""
                preview = (f"[{previews_head}, ..., {previews_tail}]"
                           if previews_tail else f"[{previews_head}]")
                total_bytes = run_len * 4
                f.write(f"  [{sch}] {label:<48} ×{run_len:<4}"
                        f" ({total_bytes} bytes) {preview}\n")
            i = j


_CUDA_MEMCPY_KIND = {
    0: "H2H", 1: "H2D", 2: "D2H", 3: "D2D",
    4: "default",
}


def _load_cudamemcpy_events(merged_path: str) -> List[Dict[str, Any]]:
    """Extract pbcap cudaMemcpy.enter/exit pairs into a list of
    {ts_enter, ts_exit, direction, nbytes, src, dst}.  Used by E
    (bracket per-doorbell sections) and F (summary line).  Safe to
    call on captures without pbcap data."""
    events: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    try:
        with open(merged_path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = ev.get("kind")
                data = ev.get("data") or {}
                if k == "cudaMemcpy.enter":
                    # pbcap emits 'kind' as the cudaMemcpyKind integer
                    # and 'count' as nbytes.  Handle both the old
                    # field names too in case of shim skew.
                    dir_int = data.get("kind")
                    direction = (_CUDA_MEMCPY_KIND.get(dir_int, f"kind={dir_int}")
                                 if isinstance(dir_int, int)
                                 else (data.get("direction") or "?"))
                    pending = {
                        "ts_enter":  ev.get("ts_ns"),
                        "direction": direction,
                        "nbytes":    data.get("count") or data.get("nbytes"),
                        "src":       data.get("src"),
                        "dst":       data.get("dst"),
                    }
                elif k == "cudaMemcpy.exit" and pending is not None:
                    pending["ts_exit"] = ev.get("ts_ns")
                    events.append(pending)
                    pending = None
    except OSError:
        pass
    return events


def _compute_summary(atlas: Atlas,
                     pb_events: Optional[List[Dict[str, Any]]] = None,
                     cudaMemcpys: Optional[List[Dict[str, Any]]] = None
                     ) -> Dict[str, Any]:
    """Build a compact summary of the capture from atlas state.
    Returned dict is JSON-serializable (embedded in atlas.json by G)
    and rendered as a human banner by F.  Kept deliberately small —
    the detail tables remain the source of truth."""
    allocs = atlas.allocs
    # Duration (monotonic) from the earliest alloc/event to the last.
    ts_all: List[int] = []
    for a in allocs.values():
        if a.get("ts_alloc") is not None:
            ts_all.append(a["ts_alloc"])
        if a.get("ts_free") is not None:
            ts_all.append(a["ts_free"])
    for e in atlas.pb_events:
        if e.get("ts_ns"):
            ts_all.append(e["ts_ns"])
    duration_ns = (max(ts_all) - min(ts_all)) if ts_all else 0

    # Channel / TSG counts.
    tsgs = sum(1 for a in allocs.values()
               if a.get("class_name") == "KEPLER_CHANNEL_GROUP_A")
    channels = sum(1 for a in allocs.values()
                   if a.get("class_name") == "HOPPER_CHANNEL_GPFIFO_A")
    ce_objs = sum(1 for a in allocs.values()
                  if a.get("class_name") == "HOPPER_DMA_COPY_A")
    compute_objs = sum(1 for a in allocs.values()
                       if a.get("class_name") == "HOPPER_COMPUTE_A")

    # Memory totals.
    vidmem_bytes = 0
    sysmem_bytes = 0
    for a in allocs.values():
        body = a.get("body") or {}
        sz = body.get("size", 0)
        if isinstance(sz, str):
            sz = _int_hex(sz)
        cls = a.get("class_name") or ""
        if cls == "NV01_MEMORY_LOCAL_USER":
            vidmem_bytes += sz
        elif cls == "NV01_MEMORY_SYSTEM":
            sysmem_bytes += sz

    # Per-channel doorbell counts, biggest transfers.
    per_chan: Dict[int, int] = {}
    if pb_events:
        for e in pb_events:
            c = e.get("chid")
            if c is not None:
                per_chan[c] = per_chan.get(c, 0) + 1
    top_chans = sorted(per_chan.items(), key=lambda kv: -kv[1])[:5]

    # fd inventory.
    fd_paths: Dict[str, int] = {}
    for fd, evs in (atlas.fd_history or {}).items():
        for ev in evs:
            if ev.get("event") == "open" and ev.get("path"):
                p = ev["path"]
                fd_paths[p] = fd_paths.get(p, 0) + 1

    return {
        "t0_ns":        min(ts_all) if ts_all else None,
        "duration_ns":  duration_ns,
        "duration_ms":  duration_ns / 1e6,
        "tsgs":         tsgs,
        "channels":     channels,
        "ce_objects":   ce_objs,
        "compute_objs": compute_objs,
        "fd_count":     len(atlas.fd_history or {}),
        "fd_paths":     fd_paths,
        "allocations":  len(allocs),
        "vidmem_bytes": vidmem_bytes,
        "sysmem_bytes": sysmem_bytes,
        "doorbells":    len(pb_events or []),
        "doorbells_top_channels": [{"chid": c, "count": n}
                                   for c, n in top_chans],
        "cudaMemcpys":  len(cudaMemcpys or []),
    }


def _write_summary_banner(f, summary: Dict[str, Any]) -> None:
    """F: render the summary dict as a tight human banner."""
    f.write("=" * 78 + "\n")
    f.write("CAPTURE SUMMARY\n")
    f.write("=" * 78 + "\n")
    if summary.get("t0_ns"):
        f.write(f"  t+0 anchor        : {summary['t0_ns']} ns "
                f"(monotonic; all t+ timestamps below are relative to this)\n")
    f.write(f"  duration          : {summary['duration_ms']:.1f} ms "
            f"({summary['duration_ns']} ns)\n")
    fd_paths = summary.get("fd_paths") or {}
    fd_str = ", ".join(f"{p}×{n}" for p, n in fd_paths.items())
    f.write(f"  fds opened        : {summary['fd_count']} "
            f"({fd_str})\n")
    f.write(f"  channels / TSGs   : {summary['channels']} channels in "
            f"{summary['tsgs']} TSGs; "
            f"{summary['ce_objects']} CE + {summary['compute_objs']} compute objects\n")
    f.write(f"  allocations       : {summary['allocations']} handles; "
            f"vidmem {summary['vidmem_bytes']/1024/1024:.1f} MiB, "
            f"sysmem {summary['sysmem_bytes']/1024/1024:.1f} MiB\n")
    f.write(f"  doorbells         : {summary['doorbells']} total\n")
    for row in summary.get("doorbells_top_channels", []):
        f.write(f"    chid={row['chid']:>3}: {row['count']} submissions\n")
    if summary.get("cudaMemcpys"):
        f.write(f"  cudaMemcpys       : {summary['cudaMemcpys']}\n")
    f.write("\n")


def _fmt_alloc_row(a: Dict[str, Any], indent: str = "") -> str:
    """Render one allocation as a human-readable line.  Shared by the
    tree view (B) and any future flat dumps."""
    events = a.get("events") or []
    n_mmap = sum(1 for e in events if e.get("event") == "mmap")
    n_gpu  = sum(1 for e in events if e.get("event") == "gpu_map")
    n_ctl  = sum(1 for e in events if e.get("event") == "control")
    body = a.get("body") or {}
    sz = body.get("size", 0)
    if isinstance(sz, str):
        sz = _int_hex(sz)
    cls_name = a.get("class_name") or "?"
    is_mem = (cls_name.startswith("NV01_MEMORY_")
              or cls_name == "HOPPER_USERMODE_A")
    if is_mem and sz >= 1024*1024:
        sz_str = f"sz={sz/1024/1024:.1f}M"
    elif is_mem and sz > 0:
        sz_str = f"sz={sz}"
    else:
        sz_str = "sz=-"
    trail = f" mmap={n_mmap} gpu_map={n_gpu} ctrl={n_ctl}"
    return (f"{indent}{a['handle']} {cls_name:<28} "
            f"{sz_str:<12}{trail}\n")


def _write_allocation_tree(f, allocs: Dict[int, Dict[str, Any]]) -> None:
    """Render allocations as a hierarchy rather than a flat list.

    Grouping rules (first match wins):
      1. CHANNEL GROUPS — each KEPLER_CHANNEL_GROUP_A with its
         children (channels, per-channel CE + compute objects,
         context-shares) and the channel group's immediate memory
         children (gpu_ctl region, USERD, pushbuffer pool, etc).
      2. DEVICE-LEVEL MEMORY — allocations parented on the device or
         subdevice that aren't inside a channel group (user buffers
         like d_buf/h_buf, VA spaces, etc).
      3. OTHER — everything else (client, RM-internal helpers).

    Relationships come from the alloc.parent field captured by
    RM_ALLOC tracepoints.  Any allocation whose parent isn't itself
    in the table is considered a "top-level" entry under its section.
    """
    # Build parent → [child] index.
    children_of: Dict[int, List[int]] = {}
    for h, a in allocs.items():
        parent = _int_hex(a.get("parent"))
        children_of.setdefault(parent, []).append(h)
    # Stable order — by ts_alloc within each parent.
    for p, kids in children_of.items():
        kids.sort(key=lambda k: allocs[k].get("ts_alloc") or 0)

    # Find TSGs (root of each channel group).
    tsgs = [h for h, a in allocs.items()
            if a.get("class_name") == "KEPLER_CHANNEL_GROUP_A"]
    tsgs.sort(key=lambda h: allocs[h].get("ts_alloc") or 0)

    # Classes whose parent is a TSG (channels), a channel (CE / compute
    # objects, context shares), or the device (TSG-shared memory) —
    # the tree walker pulls in each of these.
    def _collect_subtree(root: int, seen: set) -> List[int]:
        out = []
        stack = [root]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
            for c in children_of.get(n, []):
                stack.append(c)
        return out

    visited: set = set()

    # ── Section 1: CHANNEL GROUPS ──
    if tsgs:
        f.write(f"allocations ({len(allocs)} handles, grouped by channel group):\n")
        for tsg in tsgs:
            # Collect every descendant of this TSG (but don't mark
            # them visited yet — we do that as we emit each one).
            subtree_set: set = set()
            stack = [tsg]
            while stack:
                n = stack.pop()
                if n in subtree_set:
                    continue
                subtree_set.add(n)
                for c in children_of.get(n, []):
                    stack.append(c)

            channels = sorted(
                (h for h in subtree_set
                 if allocs[h].get("class_name") == "HOPPER_CHANNEL_GPFIFO_A"),
                key=lambda k: allocs[k].get("ts_alloc") or 0)
            tsg_a = allocs[tsg]
            f.write(f"\n  ┌─ TSG {tsg_a['handle']} KEPLER_CHANNEL_GROUP_A "
                    f"({len(channels)} channels)\n")
            visited.add(tsg)

            # Pre-compute TSG-level leftover (context shares, events
            # attached directly to the TSG rather than through a
            # channel's compute-object).  To avoid double-counting,
            # collect the FULL descendant set of each channel (not
            # just direct kids) and subtract it from the TSG subtree.
            in_channel_subtree: set = set()
            for ch in channels:
                stk = [ch]
                while stk:
                    n = stk.pop()
                    if n in in_channel_subtree:
                        continue
                    in_channel_subtree.add(n)
                    for c in children_of.get(n, []):
                        if c in subtree_set:
                            stk.append(c)
            leftover_hs = sorted(
                (h for h in subtree_set
                 if h != tsg
                 and h not in channels
                 and h not in in_channel_subtree),
                key=lambda k: allocs[k].get("ts_alloc") or 0)

            total_top_kids = len(channels) + len(leftover_hs)
            emitted_top_kids = 0

            def _top_branch(is_last: bool) -> str:
                return "  │  └─ " if is_last else "  │  ├─ "

            def _top_vert(is_last: bool) -> str:
                """Continuation spacer under a top-level kid."""
                return "  │     " if is_last else "  │  │  "

            # Channels in alloc order.  For each, emit the channel
            # plus its direct descendants (CE obj, compute obj, etc).
            for ch in channels:
                emitted_top_kids += 1
                is_last_top = (emitted_top_kids == total_top_kids)
                f.write(_fmt_alloc_row(allocs[ch], indent=_top_branch(is_last_top)))
                visited.add(ch)
                kids = sorted(
                    (h for h in children_of.get(ch, []) if h in subtree_set),
                    key=lambda k: allocs[k].get("ts_alloc") or 0)
                for ki, kid in enumerate(kids):
                    kid_last = (ki == len(kids) - 1)
                    grand_kids = sorted(
                        (g for g in children_of.get(kid, [])
                         if g in subtree_set and g not in visited),
                        key=lambda k: allocs[k].get("ts_alloc") or 0)
                    kid_branch = "└─ " if kid_last and not grand_kids else "├─ "
                    f.write(_fmt_alloc_row(allocs[kid],
                                           indent=_top_vert(is_last_top) + kid_branch))
                    visited.add(kid)
                    # Recurse one level (e.g. channel → CE → {inputs}).
                    for gi, grand in enumerate(grand_kids):
                        grand_last = (gi == len(grand_kids) - 1)
                        g_branch = "└─ " if grand_last else "├─ "
                        # Under the kid, use "│  " for continuation
                        # (unless kid is the last kid under this top).
                        kid_cont = "   " if kid_last else "│  "
                        f.write(_fmt_alloc_row(
                            allocs[grand],
                            indent=_top_vert(is_last_top) + kid_cont + g_branch))
                        visited.add(grand)

            # TSG-direct leftover (FERMI_CONTEXT_SHARE_A etc).
            for li, lh in enumerate(leftover_hs):
                emitted_top_kids += 1
                is_last_top = (emitted_top_kids == total_top_kids)
                f.write(_fmt_alloc_row(allocs[lh], indent=_top_branch(is_last_top)))
                visited.add(lh)

            f.write(f"  └─ end TSG {tsg_a['handle']}\n")
        f.write("\n")

    # ── Section 2: DEVICE-LEVEL MEMORY (not inside a channel group) ──
    dev_mems = [h for h, a in allocs.items()
                if h not in visited
                and (a.get("class_name") or "").startswith(
                    ("NV01_MEMORY_", "NV50_MEMORY_"))]
    if dev_mems:
        f.write("device-level memory (not inside a channel group):\n")
        for h in sorted(dev_mems, key=lambda k: allocs[k].get("ts_alloc") or 0):
            f.write(_fmt_alloc_row(allocs[h], indent="  "))
            visited.add(h)
        f.write("\n")

    # ── Section 3: OTHER / RM-internal ──
    other = [h for h in allocs if h not in visited]
    if other:
        f.write(f"other ({len(other)} handles — client, VA space, internal):\n")
        for h in sorted(other, key=lambda k: allocs[k].get("ts_alloc") or 0):
            f.write(_fmt_alloc_row(allocs[h], indent="  "))
        f.write("\n")


def write_methods_txt(out_path: str, doorbells: List[Dict[str, Any]],
                      pb_events: Optional[List[Dict[str, Any]]] = None,
                      pb_bytes_events: Optional[List[Dict[str, Any]]] = None,
                      atlas: Optional[Atlas] = None,
                      cudaMemcpys: Optional[List[Dict[str, Any]]] = None) -> None:
    """Human-readable per-method dump across all doorbells.  Optionally
    prepends a PROCESS MAP banner summarizing fd_history + per-handle
    lifetimes (requires atlas).  Optionally appends a PB_EVENTS section
    at the end summarizing the kernel-side pb_va resolutions, and a
    PB_BYTES_DECODE section with full method-stream decode."""
    # t0 for relative timestamps (D).  The earliest event anchors
    # "t+0.000s"; everything downstream is rendered as t+<s>s.
    # Chosen over raw ns because 3628_537_538_456 is unreadable and
    # comparing two events is an arithmetic exercise.  Absolute ts
    # still lives in atlas.json for machine consumers.
    t0: Optional[int] = None
    if atlas is not None:
        ts_pool: List[int] = []
        for a in atlas.allocs.values():
            if a.get("ts_alloc") is not None:
                ts_pool.append(a["ts_alloc"])
        for e in atlas.pb_events:
            if e.get("ts_ns"):
                ts_pool.append(e["ts_ns"])
        if ts_pool:
            t0 = min(ts_pool)

    def _rel(ts: Optional[int]) -> str:
        if ts is None or t0 is None:
            return "t+?"
        dt = (ts - t0) / 1e9
        return f"t+{dt:.4f}s"

    with open(out_path, "w") as f:
        # ── CAPTURE SUMMARY (F) ──
        if atlas is not None:
            summary = _compute_summary(atlas, pb_events, cudaMemcpys)
            _write_summary_banner(f, summary)

        # ── PROCESS MAP banner ──
        if atlas is not None:
            f.write("=" * 78 + "\n")
            f.write("PROCESS MAP — fds, allocations, mappings\n")
            f.write("=" * 78 + "\n\n")
            # fd_history
            if atlas.fd_history:
                f.write("fd → path history (chronological):\n")
                for fd in sorted(atlas.fd_history.keys(),
                                 key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else 999):
                    evs = atlas.fd_history[fd]
                    paths = [e.get("path") for e in evs
                             if e.get("event") == "open" and e.get("path")]
                    last_path = paths[-1] if paths else "?"
                    opens = sum(1 for e in evs if e.get("event") == "open")
                    closes = sum(1 for e in evs if e.get("event") == "close")
                    f.write(f"  fd={fd:>3} {last_path:<20} "
                            f"opens={opens} closes={closes}\n")
                f.write("\n")
            # ── Per-handle grouped tree ──
            #
            # Group allocations by their parent-child relationship so
            # each channel group (TSG + its channels + per-channel CE
            # + compute objects + context shares + USERD-ish memory)
            # appears as one cluster.  Free-standing memory and
            # RM-internal objects go into separate sections.
            allocs = atlas.allocs if isinstance(atlas.allocs, dict) else {}
            if allocs:
                _write_allocation_tree(f, allocs)
            f.write("=" * 78 + "\n")
            f.write("per-doorbell method decode\n")
            f.write("=" * 78 + "\n\n")

        def _emit_door_method(m: Dict[str, Any]) -> None:
            if "va" in m:
                r = m.get("resolved")
                if r:
                    f.write(f"  {m['method']:<20} role={m.get('role'):<10} "
                            f"va={m['va']} → hMemory={r['hMemory']} "
                            f"class={r['class']} offset={r['offset']}\n")
                else:
                    f.write(f"  {m['method']:<20} role={m.get('role'):<10} "
                            f"va={m['va']} → UNRESOLVED\n")
            elif "data" in m:
                data = m["data"]
                if m["method"] == "LAUNCH_DMA":
                    f.write(f"  {m['method']:<20} "
                            f"data=0x{data:08x} ({decode_launch_dma(data)})\n")
                else:
                    f.write(f"  {m['method']:<20} "
                            f"data=0x{data:08x}\n")
            elif "imm" in m:
                f.write(f"  {m['method']:<20} imm=0x{m['imm']:x}\n")
            else:
                f.write(f"  {m['method']:<20}\n")

        def _door_run_key(m: Dict[str, Any]) -> Optional[Tuple]:
            """Same collapsing rule as pb_bytes: consecutive data-only
            methods with the same name — typically LOAD_INLINE_DATA
            uploads of compute SASS — fold into one summary line."""
            if ("va" in m or "imm" in m or "bound_class_id" in m
                    or m.get("method") == "LAUNCH_DMA"):
                return None
            if "data" not in m:
                return None
            return (m.get("method"),)

        for d in doorbells:
            snap = d.get("pushbuffer_snapshot") or "(no pushbuffer snap)"
            f.write(f"=== doorbell seq={d['seq']} {_rel(d.get('ts_ns'))} "
                    f"tid={d.get('tid')} bar0={d.get('bar0')} "
                    f"snapshot={snap} ===\n")
            if not d["methods"]:
                f.write("  (empty method stream)\n\n")
                continue
            methods = d["methods"]
            i = 0
            while i < len(methods):
                m = methods[i]
                key = _door_run_key(m)
                if key is None:
                    _emit_door_method(m)
                    i += 1
                    continue
                j = i + 1
                while j < len(methods) and _door_run_key(methods[j]) == key:
                    j += 1
                run_len = j - i
                if run_len == 1:
                    _emit_door_method(m)
                else:
                    run = methods[i:j]
                    data_vals = [f"0x{r['data']:08x}" for r in run]
                    previews_head = ", ".join(data_vals[:3])
                    previews_tail = data_vals[-1] if run_len > 4 else ""
                    preview = (f"[{previews_head}, ..., {previews_tail}]"
                               if previews_tail else f"[{previews_head}]")
                    total_bytes = run_len * 4
                    f.write(f"  {m['method']:<20} ×{run_len:<4}"
                            f" ({total_bytes} bytes) {preview}\n")
                i = j
            f.write("\n")
        if pb_events:
            write_pb_events_txt(f, pb_events, cudaMemcpys, t0)
        if pb_bytes_events:
            write_pb_bytes_txt(f, pb_bytes_events)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build VA→allocation atlas from merged timeline + "
                    "pbcap snapshots, decode each doorbell's pushbuffer.")
    p.add_argument("--merged", required=True,
                   help="merged.ndjson from timeline_merge.py")
    p.add_argument("--pbcap-dir", required=True,
                   help="pbcap output directory (for snap-*.bin files)")
    p.add_argument("--atlas-out", required=True,
                   help="atlas.json output path")
    p.add_argument("--methods-out", required=True,
                   help="methods.txt human-readable per-method dump")
    args = p.parse_args()

    atlas = build_atlas(args.merged)
    print(f"atlas: {len(atlas.allocs)} allocs, "
          f"{len(atlas.gpu_ranges)} GPU ranges, "
          f"{len(atlas.cpu_ranges)} CPU ranges, "
          f"{len(atlas.pb_events)} pb events, "
          f"{len(atlas.pb_bytes_events)} pb_bytes submissions",
          file=sys.stderr)

    doorbells = walk_doorbells(args.merged, atlas, args.pbcap_dir)
    print(f"doorbells: {len(doorbells)} processed; "
          f"{sum(len(d['resolved']) for d in doorbells)} VAs resolved, "
          f"{sum(len(d['unresolved']) for d in doorbells)} unresolved",
          file=sys.stderr)

    # pb_events is driven by the kernel-side #DB handler's decoded
    # GPFIFO entry.  Print how often the resolver finds a matching GPU
    # VA range, so a glance says whether the pushbuffer lives on a
    # mapping path the trackers already cover ("sysmem vs UVM?").
    n_resolved = sum(1 for p in atlas.pb_events if p["resolved"] is not None)
    print(f"pb_events: {len(atlas.pb_events)} total, "
          f"{n_resolved} resolved via atlas.gpu_ranges, "
          f"{len(atlas.pb_events) - n_resolved} unresolved",
          file=sys.stderr)

    # Extract cudaMemcpy pairs once; used by E (methods.txt brackets)
    # and F (summary line).
    cudaMemcpys = _load_cudamemcpy_events(args.merged)

    # Pre-compute the summary so G (atlas.json) and F (methods.txt
    # banner) share the same numbers.
    summary = _compute_summary(atlas, atlas.pb_events, cudaMemcpys)

    # Serialize atlas (tables + doorbell results + summary).
    out = {
        "summary":         summary,              # G
        "anchor":          atlas.anchor,
        "allocations":     atlas.allocs,
        "gpu_ranges":      atlas.gpu_ranges,
        "cpu_ranges":      atlas.cpu_ranges,
        "fd_history":      atlas.fd_history,
        "pb_events":       atlas.pb_events,
        "pb_bytes_events": atlas.pb_bytes_events,
        "cudaMemcpys":     cudaMemcpys,
        "doorbells":       doorbells,
        # Non-UVM channel ledger (consumed by non_uvm_ledger.py).
        "channels":            atlas.channels,
        "carriers":            atlas.carriers,
        "unattributed_intermaps": atlas.unattributed_intermaps,
    }
    # Convert int keys/values to hex strings for JSON friendliness.
    with open(args.atlas_out, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: f"0x{x:x}" if isinstance(x, int) else str(x))
    write_methods_txt(args.methods_out, doorbells,
                      atlas.pb_events, atlas.pb_bytes_events,
                      atlas=atlas,
                      cudaMemcpys=cudaMemcpys)
    print(f"wrote {args.atlas_out} + {args.methods_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
