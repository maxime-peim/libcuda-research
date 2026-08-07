#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""non_uvm_ledger.py — per-channel non-UVM allocation report.

Reads a merged.ndjson timeline (produced by timeline_merge.py from a
trace_cuda.sh capture) and emits a Markdown / JSON report grouped by
*_CHANNEL_GPFIFO_A handle, listing for each channel:

  - hClient, hVASpace, channel-class, ts_alloc..ts_free
  - the carriers (NV50_MEMORY_VIRTUAL or NV01_MEMORY_VIRTUAL) bound to
    its hVASpace, with their NVOS32 alloc params if known
  - per-resource NVOS46 maps (USERD, GPFIFO, FB-source, sysmem-source)
    with src hMemory, src_class, length, dmaOffset, ts

The report is the "what does libcuda actually do for non-UVM channels"
artifact that the FB-carrier work needed and didn't have.  Compare two
captures (e.g. cuda_reference vs mc_carrier_demo) by running the script
twice with --json and diffing the outputs.

The script imports address_atlas as a library and reuses its build_atlas
+ NVOS32 decoders — there is no NDJSON parsing logic here.

Usage:
    sudo ./trace_cuda.sh ./bin/cuda_init      # produces merged.ndjson
    python3 non_uvm_ledger.py merged.ndjson [--json | --summary] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, TextIO

# Allow `import address_atlas` and friends from this directory.
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from address_atlas import (  # noqa: E402
    Atlas, build_atlas, RM_CLASS_NAMES, _int_hex,
)
from strace_diff import (  # noqa: E402
    decode_nvos32_attr, decode_nvos32_attr2, decode_nvos32_flags,
)


def _as_hex(v: Any, width: int = 8) -> str:
    n = _int_hex(v) if not isinstance(v, int) else v
    return f"0x{n:0{width}x}"


def _class_name(cls: int) -> str:
    return RM_CLASS_NAMES.get(cls, f"class_0x{cls:x}")


def _carrier_alloc(atlas: Atlas, h_carrier: int) -> Optional[Dict[str, Any]]:
    """Look up the carrier's NVOS32 alloc record (if it was emitted as
    a tracked alloc event — NV50_MEMORY_VIRTUAL goes through that
    path, NV01_MEMORY_VIRTUAL does too via vmrangeConstruct → still
    fires the RM ALLOC tracepoint)."""
    return atlas.allocs.get(h_carrier)


def _format_alloc_params(alloc: Dict[str, Any]) -> str:
    body = alloc.get("body") or {}
    parts: List[str] = []
    if "size" in body:
        parts.append(f"size=0x{int(body['size']):x}")
    if "owner" in body:
        parts.append(f"owner=0x{int(body['owner']):x}")
    if "type" in body:
        parts.append(f"type={int(body['type'])}")
    if "flags" in body:
        flags = int(body['flags'])
        parts.append(f"flags=0x{flags:08x} ({decode_nvos32_flags(flags)})")
    if "attr" in body:
        attr = int(body['attr'])
        parts.append(f"ATTR=0x{attr:08x} ({decode_nvos32_attr(attr)})")
    if "attr2" in body:
        attr2 = int(body['attr2'])
        parts.append(f"ATTR2=0x{attr2:08x} ({decode_nvos32_attr2(attr2)})")
    return "  ".join(parts) if parts else "(no body decoded)"


def render_markdown(atlas: Atlas, out: TextIO) -> None:
    """Per-channel Markdown report.

    Sections in order: a one-line header summary, then one section per
    channel (sorted by ts_alloc), then unattributed NVOS46 carrier
    maps that didn't join to a channel via hUserdMemory or hVASpace
    (these are typical for libcuda's internal scratch carriers and
    for kernel-internal channel-USERD bindings whose NVOS46 fires
    inside RM rather than on a userspace ioctl path)."""
    n_channels = len(atlas.channels)
    n_carriers = len(atlas.carriers)
    n_unattributed = len(atlas.unattributed_intermaps)
    out.write(f"# Non-UVM channel ledger\n\n")
    out.write(f"channels={n_channels}, carriers={n_carriers}, "
              f"unattributed_intermaps={n_unattributed}\n\n")
    if n_channels == 0 and n_unattributed == 0:
        out.write("(no *_CHANNEL_GPFIFO_A allocations or NVOS46 maps "
                  "seen — was tracing enabled?)\n")
        return

    sorted_channels = sorted(atlas.channels.items(),
                             key=lambda kv: kv[1].get("ts_alloc") or 0)
    for h_ch, ch in sorted_channels:
        out.write(f"## Channel {ch.get('handle')} ({ch.get('class_name')})\n")
        out.write(f"  hClient       : {ch.get('hClient')}\n")
        out.write(f"  hVASpace      : {_as_hex(ch.get('hVASpace', 0))}\n")
        out.write(f"  hUserdMemory  : {_as_hex(ch.get('hUserdMemory', 0))} "
                  f"(offset={int(ch.get('userdOffset') or 0):#x})\n")
        out.write(f"  gpFifoOffset  : {int(ch.get('gpFifoOffset') or 0):#x} "
                  f"({int(ch.get('gpFifoEntries') or 0)} entries)\n")
        out.write(f"  engineType    : {int(ch.get('engineType') or 0)}\n")
        out.write(f"  cid           : {int(ch.get('cid') or 0)}\n")
        out.write(f"  Lifetime      : {ch.get('ts_alloc')} → {ch.get('ts_free')}\n\n")

        resources = ch.get("resources") or []
        if resources:
            out.write("  Resources (NVOS46 maps into this channel's VAS):\n\n")
            out.write("    role     hSrc        src_class                          "
                      "length    hCarrier     dmaOffset\n")
            for r in resources:
                src_h = _as_hex(r.get("hSrc", 0))
                src_cls = int(r.get("src_class", 0))
                src_cls_name = _class_name(src_cls)
                length = int(r.get("length") or 0)
                car_h = _as_hex(r.get("hCarrier", 0))
                dma = int(r.get("dmaOffset") or 0)
                role = r.get("role", "?")
                out.write(f"    {role:<8} {src_h}  {src_cls_name:<34} "
                          f"0x{length:<8x} {car_h}  0x{dma:x}\n")
            out.write("\n")
        else:
            out.write("  Resources: (none recorded)\n\n")

        carriers = ch.get("carriers") or []
        if carriers:
            out.write("  Carriers (alloc params):\n\n")
            for car_h in carriers:
                car_record = atlas.carriers.get(car_h, {})
                car_alloc = _carrier_alloc(atlas, car_h)
                car_class = car_record.get("class", 0)
                out.write(f"    {_as_hex(car_h)}  class={_class_name(car_class)} "
                          f"vaSize=0x{int(car_record.get('vaSize') or 0):x} "
                          f"has_heap={car_record.get('has_heap')} "
                          f"via={car_record.get('via')}\n")
                if car_alloc is not None:
                    out.write(f"        {_format_alloc_params(car_alloc)}\n")
                else:
                    out.write(f"        (alloc params not in capture window)\n")
            out.write("\n")

    # Unattributed carrier maps: NVOS46 events that didn't bind to any
    # channel via hUserdMemory or via a dedicated hVASpace match.  Two
    # common reasons:
    #   1) libcuda allocates scratch carriers (small sysmem + BAR1
    #      reflection) for use without a channel binding — config /
    #      compute-context buffers.
    #   2) Channels created via kchannelConstruct → RM internally binds
    #      USERD/GPFIFO via paths that don't go through virtmemMapTo_IMPL
    #      (T2's site) on the userspace ioctl path.  Those bindings
    #      are not visible in this NVOS46 stream.
    # Either way, surfacing them lets the human reader see what
    # libcuda's userspace path is doing in addition to the channel
    # maps captured above.
    if atlas.unattributed_intermaps:
        out.write(f"## Unattributed NVOS46 maps ({n_unattributed})\n\n")
        out.write("Carrier maps that didn't join to a known channel via "
                  "hUserdMemory or dedicated hVASpace.\n\n")
        out.write("    role     hSrc        src_class                          "
                  "length      hCarrier     dmaOffset\n")
        for r in atlas.unattributed_intermaps:
            src_h = _as_hex(r.get("hSrc", 0))
            src_cls = int(r.get("src_class", 0))
            src_cls_name = _class_name(src_cls)
            length = int(r.get("length") or 0)
            car_h = _as_hex(r.get("hCarrier", 0))
            dma = int(r.get("dmaOffset") or 0)
            role = r.get("role", "?")
            out.write(f"    {role:<8} {src_h}  {src_cls_name:<34} "
                      f"0x{length:<10x} {car_h}  0x{dma:x}\n")
        out.write("\n")


def render_json(atlas: Atlas, out: TextIO) -> None:
    """Machine-readable per-channel ledger.  Keys mirror the Markdown
    sections so two captures can be diffed structurally with `jq`."""
    sorted_channels = sorted(atlas.channels.items(),
                             key=lambda kv: kv[1].get("ts_alloc") or 0)
    payload: List[Dict[str, Any]] = []
    for h_ch, ch in sorted_channels:
        carriers_out: List[Dict[str, Any]] = []
        for car_h in ch.get("carriers") or []:
            car_record = atlas.carriers.get(car_h, {})
            car_alloc = _carrier_alloc(atlas, car_h)
            carriers_out.append({
                "handle":     _as_hex(car_h),
                "class":      _as_hex(car_record.get("class", 0), width=4),
                "class_name": _class_name(car_record.get("class", 0)),
                "vaSize":     int(car_record.get("vaSize") or 0),
                "has_heap":   bool(car_record.get("has_heap")),
                "via":        car_record.get("via"),
                "alloc_body": (car_alloc or {}).get("body") or {},
            })
        resources_out: List[Dict[str, Any]] = []
        for r in ch.get("resources") or []:
            resources_out.append({
                "role":       r.get("role"),
                "hSrc":       _as_hex(r.get("hSrc", 0)),
                "src_class":  _as_hex(r.get("src_class", 0), width=4),
                "src_class_name": _class_name(int(r.get("src_class", 0))),
                "length":     int(r.get("length") or 0),
                "hCarrier":   _as_hex(r.get("hCarrier", 0)),
                "dmaOffset":  int(r.get("dmaOffset") or 0),
                "ts_mapped":  r.get("ts_mapped"),
            })
        payload.append({
            "handle":         ch.get("handle"),
            "class":          _as_hex(ch.get("class", 0), width=4),
            "class_name":     ch.get("class_name"),
            "hClient":        ch.get("hClient"),
            "hVASpace":       _as_hex(ch.get("hVASpace", 0)),
            "hUserdMemory":   _as_hex(ch.get("hUserdMemory", 0)),
            "userdOffset":    int(ch.get("userdOffset") or 0),
            "gpFifoOffset":   int(ch.get("gpFifoOffset") or 0),
            "gpFifoEntries":  int(ch.get("gpFifoEntries") or 0),
            "engineType":     int(ch.get("engineType") or 0),
            "cid":            int(ch.get("cid") or 0),
            "ts_alloc":       ch.get("ts_alloc"),
            "ts_free":        ch.get("ts_free"),
            "carriers":       carriers_out,
            "resources":      resources_out,
        })
    json.dump({"channels": payload}, out, indent=2)
    out.write("\n")


def render_summary(atlas: Atlas, out: TextIO) -> None:
    """One-line-per-channel cross-table view, useful for at-a-glance
    comparisons across captures."""
    out.write(f"channels={len(atlas.channels)}  carriers={len(atlas.carriers)}\n")
    out.write(f"{'handle':<12} {'class':<28} {'hVASpace':<12} "
              f"{'#carriers':<10} {'#resources':<11} {'roles':<30}\n")
    sorted_channels = sorted(atlas.channels.items(),
                             key=lambda kv: kv[1].get("ts_alloc") or 0)
    for h_ch, ch in sorted_channels:
        roles = sorted({r.get("role", "?")
                        for r in (ch.get("resources") or [])})
        out.write(f"{ch.get('handle'):<12} "
                  f"{ch.get('class_name', '?'):<28} "
                  f"{_as_hex(ch.get('hVASpace', 0)):<12} "
                  f"{len(ch.get('carriers') or []):<10} "
                  f"{len(ch.get('resources') or []):<11} "
                  f"{','.join(roles):<30}\n")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged",
                    help="path to merged.ndjson (from timeline_merge.py)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true",
                      help="emit machine-readable JSON per channel")
    mode.add_argument("--summary", action="store_true",
                      help="emit a one-line-per-channel table")
    ap.add_argument("--out", default=None,
                    help="output file (default: stdout)")
    args = ap.parse_args(argv)

    atlas = build_atlas(args.merged)
    out = open(args.out, "w") if args.out else sys.stdout
    try:
        if args.json:
            render_json(atlas, out)
        elif args.summary:
            render_summary(atlas, out)
        else:
            render_markdown(atlas, out)
    finally:
        if args.out:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
