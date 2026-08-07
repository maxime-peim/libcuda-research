#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
timeline_merge.py — merge ftrace + strace + pbcap NDJSON into one
sorted timeline (NDJSON, CLOCK_MONOTONIC ns axis).

Sources:
  --pbcap-ndjson   /tmp/pbcap/timeline.ndjson (produced by libpbcap.so)
                   Already CLOCK_MONOTONIC ns; first event is
                   `pbcap.init` carrying both mono_ns and real_ns
                   anchors taken back-to-back — used to translate
                   strace's CLOCK_REALTIME -ttt timestamps.
  --ftrace         /tmp/ftrace.txt
                   Assume trace_clock=mono (set via
                   `echo mono > /sys/kernel/debug/tracing/trace_clock`
                   before capture).  Timestamps then match pbcap's
                   mono_ns directly.  Parsed via strace_diff.parse_ftrace.
  --strace         /tmp/strace.log (optional)
                   Captured with `strace -ttt -T -f ...` → CLOCK_REALTIME
                   ns.  Translated to CLOCK_MONOTONIC via the pbcap.init
                   anchor.  Parsed via strace_diff.parse_strace.

Output: --out merged.ndjson — one JSON object per line sorted by ts_ns,
with a `src` field ("pbcap"/"ftrace"/"strace"), `kind`, and `data`.

Usage:
  python3 timeline_merge.py \\
      --pbcap-ndjson /tmp/tools/timeline.ndjson \\
      --ftrace /tmp/ftrace.txt \\
      [--strace /tmp/strace.log] \\
      --out /tmp/merged.ndjson \\
      [--filter kind=doorbell,cudaMemcpy.enter,cudaMemcpy.exit] \\
      [--window-around seq=5 --window-events=50]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Reuse the parsers from strace_diff.py rather than duplicating them.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from strace_diff import parse_ftrace, parse_strace  # noqa: E402


def load_pbcap_ndjson(path: str) -> Tuple[List[Dict], Optional[int], Optional[int]]:
    """Load pbcap's timeline.ndjson.

    Returns (events, mono_anchor_ns, real_anchor_ns).  The anchors come
    from the first `pbcap.init` event; used to translate strace's
    CLOCK_REALTIME timestamps to CLOCK_MONOTONIC.  If no init event is
    present, anchors are None and strace translation will fall back to
    line-order-only (no merging into the monotonic axis).
    """
    events: List[Dict] = []
    mono_anchor: Optional[int] = None
    real_anchor: Optional[int] = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate occasional partial/torn writes from the
                # signal-handler path.  Skip them.
                continue
            events.append(ev)
            if (mono_anchor is None
                    and ev.get("kind") == "pbcap.init"
                    and "mono_ns" in ev.get("data", {})
                    and "real_ns" in ev.get("data", {})):
                mono_anchor = int(ev["data"]["mono_ns"])
                real_anchor = int(ev["data"]["real_ns"])
    return events, mono_anchor, real_anchor


def normalize_ftrace(events: List) -> Iterable[Dict]:
    """strace_diff.parse_ftrace returns Event namedtuples with ts_ns
    already in CLOCK_MONOTONIC ns (when trace_clock=mono).  Project
    each one into our NDJSON shape."""
    for e in events:
        if e.ts_ns is None:
            continue
        # Drop the per-handle-history-only body events (pte_row etc.)
        # that aren't useful in the merged timeline?  Actually keep
        # them — they're rare and caller can filter.
        yield {
            "ts_ns": int(e.ts_ns),
            "src":   "ftrace",
            "pid":   e.pid,
            "kind":  e.kind,
            "data":  _ftrace_info_to_dict(e.kind, e.info),
            "line":  e.line_no,
        }


def _ftrace_info_to_dict(kind: str, info) -> Dict:
    """Project the strace_diff Event.info tuple into a JSON-safe dict.

    The info shape varies by kind (see parse_ftrace's docstring).  We
    translate the common kinds and fall back to a generic tuple dump
    for unusual ones.
    """
    if kind == "alloc":
        # (handle, cls, body_dict, root, parent)
        handle, cls, body, root, parent = info
        return {
            "handle": f"0x{handle:08x}",
            "class":  f"0x{cls:04x}",
            "root":   f"0x{root:08x}",
            "parent": f"0x{parent:08x}",
            "body":   _jsonify_dict(body),
        }
    if kind == "map_bar1":
        # parse_ftrace now emits (handle, length, flags, offset, fd).
        # fd is the join key with the subsequent pbcap mmap event that
        # actually produces the user VA for this RM-mapped region.
        handle, length, flags, offset, fd = info
        return {"handle": f"0x{handle:08x}", "length": length,
                "flags":  f"0x{flags:08x}",
                "offset": offset,
                "fd":     fd}
    if kind == "map_uvm":
        handle, base, length, off = info
        return {"handle": f"0x{handle:08x}",
                "base":   f"0x{base:x}",
                "length": length,
                "offset": off}
    if kind == "free":
        (handle,) = info
        return {"handle": f"0x{handle:08x}"}
    if kind == "control":
        ctrlcmd, hc, ho, ps = info
        return {"ctrlcmd":  f"0x{ctrlcmd:08x}",
                "hClient":  f"0x{hc:08x}",
                "hObject":  f"0x{ho:08x}",
                "paramsSize": ps}
    if kind == "uvm_ioctl":
        cmd, name, dwords = info
        # Keep the first 16 dwords only — enough for any known UVM
        # cmd's body; full-body is in the original ftrace file.
        return {"cmd": cmd, "cmd_name": name,
                "dwords": [f"0x{d:08x}" for d in dwords[:16]]}
    if kind == "pb":
        # (seq, chid, idx, entry0, entry1, pb_va, pb_len) from the
        # kernel-side #DB handler.  See strace_diff's MC1_SIMPLE entry
        # for ("pb", "submit").
        seq, chid, idx, e0, e1, pb_va, pb_len = info
        return {"seq": seq, "chid": chid, "idx": idx,
                "entry0": f"0x{e0:08x}", "entry1": f"0x{e1:08x}",
                "pb_va":  f"0x{pb_va:x}", "pb_len": pb_len}
    if kind == "pb_bytes":
        # (seq, chid, idx, nbytes, hex_str) — reassembled method-stream
        # bytes, one per (chid, idx) submission.  hex is lowercase hex
        # with 2 chars per byte.
        seq, chid, idx, nbytes, hex_str = info
        return {"seq": seq, "chid": chid, "idx": idx,
                "nbytes": nbytes, "hex": hex_str}
    if kind == "pb_bytes_miss":
        seq, chid, idx, pb_va, pb_len = info
        return {"seq": seq, "chid": chid, "idx": idx,
                "pb_va": f"0x{pb_va:x}", "pb_len": pb_len}
    if kind == "sysmem_track_add":
        # (slot, va_start, va_end, pages, kva) from the kernel
        # nvidia_mmap_sysmem hook.  Doubles as a gpu_range for
        # pb_va resolution under UVM (Paper Finding 1).
        slot, va_start, va_end, pages, kva = info
        return {"slot": slot,
                "base":   f"0x{va_start:x}",
                "end":    f"0x{va_end:x}",
                "length": va_end - va_start,
                "pages":  pages,
                "kva":    kva}
    if kind == "sysmem_track_remove":
        kva, pages = info
        return {"kva": kva, "pages": pages}
    if kind == "bar1_track_add":
        # (slot, phys, size, kva) from nv_dbell_bar1_track_add.  phys
        # lives in the BAR1 aperture (typically 0x1fc...); user_va is
        # derived in the atlas post-pass via cpu_range correlation.
        slot, phys, size, kva = info
        return {"slot": slot,
                "phys":   f"0x{phys:x}",
                "size":   size,
                "kva":    kva}
    if kind == "bar1_track_remove":
        slot, kva = info
        return {"slot": slot, "kva": kva}
    if kind == "ioctl":
        # Two shapes: ftrace synthesizes (path_family, cmd_name);
        # strace emits (fd, path, cmd_int, cmd_name, size, ret).
        if len(info) == 2:
            path_family, cmd_name = info
            return {"path_family": path_family, "cmd_name": cmd_name}
        fd, path, cmd_int, cmd_name, size, *rest = info
        ret = rest[0] if rest else None
        out = {"fd": fd, "path": path,
               "cmd": f"0x{cmd_int:x}" if cmd_int is not None else None,
               "cmd_name": cmd_name, "size": size}
        if ret is not None:
            out["ret"] = ret
        return out
    if kind == "open":
        fd, path = info
        return {"fd": fd, "path": path}
    if kind == "close":
        return {"fd": info}
    if kind == "mmap":
        # strace-sourced: (fd, path, addr, length, prot, flags, offset).
        # addr is the RETURNED user VA.  address_atlas.py joins this
        # against map_bar1 events on (fd, length) to attach hMemory.
        fd, path, addr, length, prot, flags, offset = info
        return {"fd":     fd,
                "path":   path,
                "addr":   f"0x{addr:x}",
                "length": length,
                "prot":   prot,
                "flags":  flags,
                "offset": offset}
    if kind == "munmap":
        addr, length = info
        return {"addr": f"0x{addr:x}", "length": length}
    # ── Non-UVM VAS / channel observability projections ─────────────
    # Each kind matches a parse_ftrace handler for one TRACE line; see
    # strace_diff.MC1_SIMPLE.  Hex strings preserve the kernel emission
    # exactly; class ids stay 4 hex digits to match the address atlas.
    if kind == "pte_src_decision":
        ch, cc, ca, sh, sc, sa, sp, cp = info
        return {"carrier_h":     f"0x{ch:08x}",
                "carrier_class": f"0x{cc:04x}",
                "carrier_aspace": ca,
                "src_h":         f"0x{sh:08x}",
                "src_class":     f"0x{sc:04x}",
                "src_aspace":    sa,
                "src_pte0":      f"0x{sp:x}",
                "chosen_pte0":   f"0x{cp:x}"}
    if kind == "intermap_call":
        hc, car, cc, src, sc, fl, dma, ln = info
        return {"hClient":       f"0x{hc:08x}",
                "hCarrier":      f"0x{car:08x}",
                "carrier_class": f"0x{cc:04x}",
                "hSrc":          f"0x{src:08x}",
                "src_class":     f"0x{sc:04x}",
                "flags":         f"0x{fl:08x}",
                "dmaOffset":     f"0x{dma:x}",
                "length":        ln}
    if kind == "virtmem_backing":
        hm, cl, vs, vas, asp, heap, via = info
        return {"hMemory":  f"0x{hm:08x}",
                "class":    f"0x{cl:04x}",
                "vaSize":   vs,
                "hVASpace": f"0x{vas:08x}",
                "aspace":   asp,
                "has_heap": bool(heap),
                "via":      via}
    if kind == "gmmu_pte_phys":
        lo, hi, pp, pc, ps, ap = info
        return {"vaLo":      f"0x{lo:x}",
                "vaHi":      f"0x{hi:x}",
                "pte0_phys": f"0x{pp:x}",
                "pteCount":  pc,
                "pageSize":  f"0x{ps:x}",
                "aperture":  ap}
    if kind == "userd_resolve":
        hc, hu, uo, ua, asp, us = info
        return {"hClient":      f"0x{hc:08x}",
                "hUserd":       f"0x{hu:08x}",
                "userdOffset":  f"0x{uo:x}",
                "userdAddr":    f"0x{ua:x}",
                "addressSpace": asp,
                "userdSize":    us}
    if kind == "userd_rpc":
        hch, base, sz, asp, ca = info
        return {"hChannel":     f"0x{hch:08x}",
                "base":         f"0x{base:x}",
                "size":         sz,
                "addressSpace": asp,
                "cacheAttrib":  ca}
    if kind == "userd_bind":
        rc, rsc = info
        return {"retained": rc, "resourceCount": rsc}
    if kind == "bar1_reflect_phys":
        gpu, fbo, phys, sz = info
        return {"at_gpu":          f"0x{gpu:x}",
                "fb_aperture_off": f"0x{fbo:x}",
                "bar1_phys":       f"0x{phys:x}",
                "size":            sz}
    # Fallback: stringify.
    return {"_raw": str(info)}


def _jsonify_dict(d: Dict) -> Dict:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, int):
            out[k] = f"0x{v:x}" if v > 0xFFFFF else v
        else:
            out[k] = v
    return out


def normalize_strace(events: List, real_anchor: Optional[int],
                     mono_anchor: Optional[int]) -> Iterable[Dict]:
    """Translate strace events from CLOCK_REALTIME to CLOCK_MONOTONIC
    using the anchor delta.  Events without a timestamp (strace run
    without -ttt) are skipped from the merged timeline."""
    if real_anchor is None or mono_anchor is None:
        return
    delta = real_anchor - mono_anchor
    for e in events:
        if e.ts_ns is None:
            continue
        mono_ts = int(e.ts_ns) - delta
        yield {
            "ts_ns": mono_ts,
            "src":   "strace",
            "pid":   e.pid,
            "kind":  e.kind,
            "data":  _ftrace_info_to_dict(e.kind, e.info),
            "line":  e.line_no,
        }


def normalize_pbcap(events: List[Dict]) -> Iterable[Dict]:
    """Pass through pbcap NDJSON (already CLOCK_MONOTONIC).  We
    preserve `pid`, `tid`, `kind`, `data` fields and tag `src=pbcap`."""
    for ev in events:
        if "ts_ns" not in ev:
            continue
        out: Dict[str, Any] = {
            "ts_ns": int(ev["ts_ns"]),
            "src":   "pbcap",
            "pid":   ev.get("pid"),
            "kind":  ev.get("kind", "unknown"),
            "data":  ev.get("data", {}),
        }
        if "tid" in ev:
            out["tid"] = ev["tid"]
        yield out


def merge_all(pbcap: Iterable[Dict],
              ftrace: Iterable[Dict],
              strace: Iterable[Dict]) -> List[Dict]:
    """Merge three already-sorted (by ts_ns, per-source) iterables into
    one sorted list.  Simple sort is fine for our scale (100Ks of
    events at most)."""
    all_events = list(pbcap) + list(ftrace) + list(strace)
    all_events.sort(key=lambda e: e["ts_ns"])
    return all_events


def apply_filter(events: List[Dict], kinds: Optional[List[str]]) -> List[Dict]:
    if not kinds:
        return events
    # Glob-ish: allow trailing .* for simple matching.
    wildcards: List[str] = []
    exacts: set = set()
    for k in kinds:
        if k.endswith(".*"):
            wildcards.append(k[:-1])  # keep trailing "."
        else:
            exacts.add(k)
    out: List[Dict] = []
    for e in events:
        k = e["kind"]
        if k in exacts:
            out.append(e)
            continue
        for w in wildcards:
            if k.startswith(w):
                out.append(e)
                break
    return out


def apply_window(events: List[Dict], key: str, value: str,
                 n_events: int) -> List[Dict]:
    """Keep the n_events events before and after the first event whose
    data[key] == value.  Used to zoom around a specific doorbell seq
    etc."""
    anchor_idx: Optional[int] = None
    for i, e in enumerate(events):
        d = e.get("data") or {}
        # Allow both int and string match.
        v = d.get(key)
        if v is None:
            continue
        if str(v) == str(value) or v == value:
            anchor_idx = i
            break
    if anchor_idx is None:
        return events
    lo = max(0, anchor_idx - n_events)
    hi = min(len(events), anchor_idx + n_events + 1)
    return events[lo:hi]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge ftrace + strace + pbcap NDJSON into one "
                    "CLOCK_MONOTONIC-sorted timeline (NDJSON)."
    )
    p.add_argument("--pbcap-ndjson", required=True,
                   help="pbcap timeline.ndjson path (source of anchor)")
    p.add_argument("--ftrace",
                   help="ftrace text (trace_clock=mono required for alignment)")
    p.add_argument("--strace",
                   help="strace -ttt -T output (optional; requires anchor)")
    p.add_argument("--out", required=True,
                   help="merged NDJSON output path")
    p.add_argument("--filter",
                   help="comma-separated kinds to keep; append .* for prefix "
                        "match (e.g., cudaMemcpy.*,doorbell)")
    p.add_argument("--window-around",
                   help="focus on events around a specific data field: "
                        "KEY=VALUE (e.g., seq=5)")
    p.add_argument("--window-events", type=int, default=50,
                   help="events to include on each side of --window-around")
    args = p.parse_args()

    pbcap_events, mono_anchor, real_anchor = load_pbcap_ndjson(args.pbcap_ndjson)
    print(f"pbcap: {len(pbcap_events)} events, "
          f"anchor mono_ns={mono_anchor} real_ns={real_anchor}",
          file=sys.stderr)

    ftrace_events: List = []
    if args.ftrace:
        ftrace_events = parse_ftrace(args.ftrace)
        print(f"ftrace: {len(ftrace_events)} events", file=sys.stderr)

    strace_events: List = []
    if args.strace:
        strace_events = parse_strace(args.strace)
        with_ts = sum(1 for e in strace_events if e.ts_ns is not None)
        print(f"strace: {len(strace_events)} events ({with_ts} with ts_ns)",
              file=sys.stderr)
        if with_ts > 0 and real_anchor is None:
            print("WARNING: strace has timestamps but pbcap has no anchor; "
                  "strace events will be DROPPED from merged timeline.",
                  file=sys.stderr)

    merged = merge_all(
        normalize_pbcap(pbcap_events),
        normalize_ftrace(ftrace_events),
        normalize_strace(strace_events, real_anchor, mono_anchor),
    )

    if args.filter:
        merged = apply_filter(merged, [k.strip() for k in args.filter.split(",")])

    if args.window_around:
        if "=" not in args.window_around:
            print("error: --window-around expects KEY=VALUE",
                  file=sys.stderr)
            return 2
        key, value = args.window_around.split("=", 1)
        merged = apply_window(merged, key, value, args.window_events)

    with open(args.out, "w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    print(f"merged: {len(merged)} events → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
