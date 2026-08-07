#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
locate_pushbuffer.py — classify every kernel-observed pb_va against
known CPU-side mappings, so we can decide where libcuda's pushbuffer
bytes actually live before hooking the kernel for byte readout.

Inputs
------
  --atlas      atlas.json from address_atlas.py.  Carries
               allocations, gpu_ranges (UVM-mapped), cpu_ranges
               (from pbcap's glibc mmap hook), and pb_events.
  --pbcap-dir  pbcap's output directory.  Contains snap-*-<tag>-<VA>-
               len<N>.bin files whose filenames encode the VA + length
               of every nvidia-backed mapping pbcap saw at each
               snapshot moment.  This catches libcuda's direct-syscall
               mmaps that the glibc hook misses (the "paths=… snapshot
               0/0" note from timeline_merge).

Output
------
  Text report to stdout showing, per pb_va cluster:
    - where it falls (uvm gpu_range / pbcap cpu_range / snap-file /
      unresolved)
    - backing fd (/dev/nvidia0 / /dev/nvidiactl / /dev/nvidia-uvm /
      anon)
    - hMemory + class when the atlas has it

Says which mapping path a pushbuffer came in on — the sysmem mmap
hook or the UVM one — when the atlas alone cannot resolve it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# pbcap snapshot filename format:
#   snap-<phase-idx>-<kind>-<source>-<addr_hex>-len<len_hex>.bin
# where <source> is "nvidia0", "nvidiactl", "nvidia-uvm", or similar
# (derived from the /dev/nvidia<N> path minus "/dev/").  We care about
# <source>, <addr_hex>, and <len_hex>.
SNAP_RE = re.compile(
    r"^snap-(?P<phase_idx>\d+)-(?P<phase>[a-z]+)-"
    r"(?P<source>[a-zA-Z0-9_-]+)-"
    r"0x(?P<addr>[0-9a-f]+)-len(?P<length>[0-9a-f]+)\.bin$"
)


def scan_snapshots(pbcap_dir: str) -> List[Dict[str, Any]]:
    """Return one entry per unique (source, addr, length) triple.  Many
    snapshots typically cover the same VA range across different
    phases/doorbells; we deduplicate."""
    if not os.path.isdir(pbcap_dir):
        return []
    seen: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for name in os.listdir(pbcap_dir):
        m = SNAP_RE.match(name)
        if not m:
            continue
        source = m.group("source")
        addr = int(m.group("addr"), 16)
        length = int(m.group("length"), 16)
        key = (source, addr, length)
        if key not in seen:
            seen[key] = {
                "source": source,
                "addr":   addr,
                "end":    addr + length,
                "length": length,
            }
    return sorted(seen.values(), key=lambda e: (e["source"], e["addr"]))


def find_in_ranges(va: int, ranges: List[Dict[str, Any]],
                    base_key: str = "base",
                    end_key:  str = "end") -> Optional[Dict[str, Any]]:
    for r in ranges:
        b = r.get(base_key)
        e = r.get(end_key)
        if isinstance(b, str) and b.startswith(("0x", "0X")):
            b = int(b, 16)
        if isinstance(e, str) and e.startswith(("0x", "0X")):
            e = int(e, 16)
        if b is None or e is None:
            continue
        if b <= va < e:
            return r
    return None


def classify(va: int, atlas: Dict[str, Any],
             snaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide which bucket a pb_va falls in.  Order: UVM atlas ranges,
    pbcap CPU ranges (glibc hook), snapshot filenames (catches
    direct-syscall mmaps too)."""
    # UVM-mapped (atlas.gpu_ranges comes from UVM MAP_EXTERNAL events).
    gpu_ranges = atlas.get("gpu_ranges") or []
    r = find_in_ranges(va, [{"base": rr["base"], "end": rr["end"], **rr}
                             for rr in gpu_ranges])
    if r is not None:
        h = r.get("hMemory")
        # hMemory may serialize as raw int (through JSON default=lambda)
        # when the stored value is an int.  Normalize to 0xNNNNNNNN.
        h_str = f"0x{h:08x}" if isinstance(h, int) else str(h)
        return {"bucket": "uvm_range",
                "hMemory": h_str,
                "base": r.get("base"), "length": r.get("length"),
                "class": _alloc_class(atlas, h)}

    # pbcap CPU VA hook.  Keys are "addr" + "end".
    cpu_ranges = atlas.get("cpu_ranges") or []
    r = find_in_ranges(va, cpu_ranges, base_key="addr")
    if r is not None:
        return {"bucket": "pbcap_cpu_range",
                "path": r.get("path"), "fd": r.get("fd"),
                "addr": r.get("addr"), "length": r.get("length")}

    # Snapshot filenames — catches direct-syscall mmaps missed by the
    # pbcap glibc hook.
    r = find_in_ranges(va, snaps, base_key="addr")
    if r is not None:
        return {"bucket": "snap_file",
                "source": r["source"],
                "addr":   f"0x{r['addr']:x}",
                "length": r["length"]}

    return {"bucket": "unresolved"}


def _alloc_class(atlas: Dict[str, Any], hMemory: Any) -> Optional[str]:
    if hMemory is None:
        return None
    allocs = atlas.get("allocations") or {}
    # atlas.allocations key formats observed across Atlas + json.dump paths:
    #   - Live Atlas: int keys.
    #   - After json.dump(default=...): int keys become decimal strings
    #     (since int serializes natively via json, the default= lambda
    #     that would hex-stringify is never invoked on raw ints).
    # Build all candidate keys and try each.
    candidates: List[Any] = [hMemory]
    if isinstance(hMemory, int):
        candidates += [str(hMemory), f"0x{hMemory:x}", f"0x{hMemory:08x}"]
    elif isinstance(hMemory, str):
        s = hMemory.strip()
        if s.startswith(("0x", "0X")):
            try:
                n = int(s, 16)
                candidates += [n, str(n)]
            except ValueError: pass
        else:
            try:
                n = int(s)
                candidates += [n, f"0x{n:x}", f"0x{n:08x}"]
            except ValueError: pass
    for k in candidates:
        entry = allocs.get(k)
        if entry is not None:
            return entry.get("class_name") or entry.get("class")
    return None


def _normalize_pb_va(va: Any) -> int:
    if isinstance(va, int):
        return va
    s = str(va).strip()
    if s.startswith(("0x", "0X")):
        return int(s, 16)
    return int(s)


def summarize(pb_events: List[Dict[str, Any]],
              atlas: Dict[str, Any],
              snaps: List[Dict[str, Any]]) -> str:
    """Group by (bucket, source/path/hMemory) and count."""
    # Per-doorbell classification.
    per_event: List[Dict[str, Any]] = []
    for p in pb_events:
        va = _normalize_pb_va(p["pb_va"])
        c = classify(va, atlas, snaps)
        per_event.append({
            "seq":   p["seq"],
            "chid":  p["chid"],
            "pb_va": p["pb_va"],
            "pb_len": p["pb_len"],
            **c,
        })

    def _as_hex(v: Any) -> str:
        if isinstance(v, int):
            return f"0x{v:x}"
        return str(v)

    # Aggregate.
    buckets = Counter(p["bucket"] for p in per_event)
    by_detail: Dict[str, Counter] = defaultdict(Counter)
    for p in per_event:
        if p["bucket"] == "uvm_range":
            k = (f"uvm hMemory={p.get('hMemory')} "
                 f"class={p.get('class')} base={_as_hex(p.get('base'))}")
        elif p["bucket"] == "pbcap_cpu_range":
            k = (f"pbcap_cpu path={p.get('path')} "
                 f"addr={_as_hex(p.get('addr'))}")
        elif p["bucket"] == "snap_file":
            k = (f"snap source={p.get('source')} "
                 f"addr={_as_hex(p.get('addr'))}")
        else:
            k = "unresolved"
        by_detail[p["bucket"]][k] += 1

    # Render.
    out: List[str] = []
    out.append(f"pb_events scanned: {len(per_event)}")
    out.append("")
    out.append("Bucket counts:")
    for b, n in buckets.most_common():
        out.append(f"  {b:<20} {n}")
    out.append("")
    for b, detail in by_detail.items():
        out.append(f"--- {b} ---")
        for k, n in detail.most_common():
            out.append(f"  {n:>5}  {k}")
        out.append("")

    # Also show the 5 most frequent unresolved pb_va values — these
    # are the ones that inform where the pushbuffer actually lives.
    unresolved = [p for p in per_event if p["bucket"] == "unresolved"]
    if unresolved:
        out.append("--- unresolved pb_va samples (up to 10) ---")
        seen_va = Counter(p["pb_va"] for p in unresolved)
        for va, n in seen_va.most_common(10):
            # Find the nearest snapshot+offset to help narrow down
            # where it COULD be.
            va_int = _normalize_pb_va(va)
            nearest = None
            nearest_delta = None
            for s in snaps:
                if s["addr"] <= va_int < s["end"]:
                    nearest = s
                    nearest_delta = va_int - s["addr"]
                    break
            if nearest is None:
                # Nearest below.
                below = [s for s in snaps if s["addr"] <= va_int]
                if below:
                    nearest = max(below, key=lambda s: s["addr"])
                    nearest_delta = va_int - nearest["addr"]
            hint = (f" nearest-snap {nearest['source']}@0x{nearest['addr']:x}"
                    f" +0x{nearest_delta:x}"
                    if nearest else " (no nearby snapshot)")
            out.append(f"  {n:>5}  pb_va={va}{hint}")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Classify pb_va values from atlas.pb_events against "
                    "known CPU-side mappings to decide sysmem vs UVM hook.")
    p.add_argument("--atlas", required=True,
                   help="atlas.json from address_atlas.py")
    p.add_argument("--pbcap-dir", required=True,
                   help="pbcap output directory (for snap-*.bin files)")
    args = p.parse_args()

    with open(args.atlas) as f:
        atlas = json.load(f)
    pb_events = atlas.get("pb_events") or []
    if not pb_events:
        print("no pb_events in atlas — did the capture include the "
              "kernel watchpoint mc1 pb/submit records?")
        return 2

    snaps = scan_snapshots(args.pbcap_dir)
    print(summarize(pb_events, atlas, snaps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
