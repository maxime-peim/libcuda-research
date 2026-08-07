#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
trace_section.py — slice a merged.ndjson timeline by one or more
libcuda function names and render the events that fell inside each
call.

Background:
  pbcap.c brackets each hooked CUDA Runtime call with two NDJSON lines
  (`<fn>.enter` carrying args, `<fn>.exit` carrying return code).
  timeline_merge.py forwards them verbatim into merged.ndjson alongside
  every ftrace, strace, and pbcap event in the run, all on one
  CLOCK_MONOTONIC ns axis.

  This tool windows merged.ndjson on a (`<fn>.enter`, `<fn>.exit`) pair
  and prints every event in the window in chronological order, one per
  line, with a relative `+<offset>µs` marker.  It's the natural answer
  to "what does cudaHostRegister do in addition to cudaInitDevice?":
  point it at the cuda_host_register trace with `--section
  cudaHostRegister`, and the output IS the answer — no cross-trace
  subtraction, no role-matching, no nondeterminism handling.

Output denoising:
  By default each ioctl appears as ONE logical event even though it
  shows up in three NDJSON rows (`strace ioctl X`, `ftrace ioctl X`,
  optional `ftrace uvm_ioctl X` with the decoded params).  The fold
  is keyed on (cmd_name, fd) within a 100 µs window.  Pass `--unfold`
  to keep the raw stream-per-event rendering.

  When the fold absorbs a `ftrace uvm_ioctl` row that carries 16
  decoded body dwords, the UVM decoder from strace_diff.py renders
  the structured params inline (e.g. for UVM_MAP_EXTERNAL_ALLOCATION:
  base + length + offset + hMemory).

  Handle references (alloc/free/map_uvm/map_bar1/uvm payload) are
  resolved against atlas.json's allocations table to tag each handle
  with its class_name and (if applicable) byte size.

Usage:
  # <dir> is whatever trace_cuda.sh printed as its "output dir",
  # e.g. /tmp/trace-cuda_reference-20260809-015207
  trace_section.py --trace <dir> --section cudaMemcpy
  trace_section.py --trace <dir> \\
                   --section cudaInitDevice,cudaMalloc,cudaMemcpy
  trace_section.py --trace <dir> \\
                   --section cudaMemcpy --occurrence 0 --unfold

Inputs:
  --trace <dir>         a trace_cuda.sh output directory containing
                        merged.ndjson (atlas.json optional)
  --section a[,b[,c]]   one or more function-name prefixes; each window
                        is rendered back-to-back.  Missing sections warn
                        on stderr and skip.
  --occurrence N        which call to slice when a section appears more
                        than once (default 0 = first); applies to all
                        listed sections
  --unfold              don't fold strace/ftrace/uvm_ioctl pairs into
                        single logical events (debugging aid)
  --out <file>          write to file; default stdout
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Reuse strace_diff.py's UVM payload decoders so we render the same
# structured fields as --handle-history without duplicating tables.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from strace_diff import decode_uvm_cmd  # noqa: E402


# ── inputs ────────────────────────────────────────────────────────────────

def load_merged(path: str) -> List[Dict]:
    events: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_atlas(trace_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load atlas.json's allocations table → {handle_int: alloc_record}.

    Atlas keys are decimal integers (handles); we keep the same shape
    so callers can do `atlas.get(int(handle_str, 16))` cheaply.
    Returns {} if atlas.json is missing or malformed.
    """
    path = os.path.join(trace_dir, "atlas.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    allocs = doc.get("allocations") or {}
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in allocs.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


# ── windowing ─────────────────────────────────────────────────────────────

def find_window(events: List[Dict], section: str, occurrence: int
                ) -> Optional[Tuple[Dict, Dict, List[Dict]]]:
    """Locate the Nth `<section>.enter` / `<section>.exit` pair.  Returns
    (enter_event, exit_event, [events between them inclusive]) or None
    when the section / occurrence isn't present.
    """
    enter_kind = f"{section}.enter"
    exit_kind  = f"{section}.exit"
    enters: List[int] = []
    exits:  List[int] = []
    depth  = 0
    pending_enter: Optional[int] = None

    for i, ev in enumerate(events):
        k = ev.get("kind")
        if k == enter_kind:
            if depth == 0:
                pending_enter = i
            depth += 1
        elif k == exit_kind:
            if depth > 0:
                depth -= 1
                if depth == 0 and pending_enter is not None:
                    enters.append(pending_enter)
                    exits.append(i)
                    pending_enter = None

    if not enters or occurrence >= len(enters):
        return None
    e_idx = enters[occurrence]
    x_idx = exits[occurrence]
    return events[e_idx], events[x_idx], events[e_idx:x_idx + 1]


# ── handle resolution ─────────────────────────────────────────────────────

def _atlas_lookup(atlas: Dict[int, Dict], handle_str: str) -> str:
    """Render `handle_str` (e.g. "0x5c000019") with its atlas role tag,
    or just the handle itself when not resolvable.

    Examples:
        "0x5c000019"                              → if not in atlas
        "0x5c000019(NV01_MEMORY_SYSTEM, 56MiB)"   → resolved with size
        "0x5c000003(NV20_SUBDEVICE_0)"            → resolved without size
    """
    if not handle_str or not isinstance(handle_str, str):
        return str(handle_str)
    try:
        h = int(handle_str, 16)
    except ValueError:
        return handle_str
    rec = atlas.get(h)
    if not rec:
        return handle_str
    cls = rec.get("class_name") or "?"
    body = rec.get("body") or {}
    sz   = body.get("size")
    # Class objects (Device, Subdevice, Channel, ...) carry a `size` field
    # too but it isn't a byte count — only memory classes have meaningful
    # sizes.  Use the class_name prefix as a heuristic: NV01_MEMORY_*,
    # NV01_*_DESCRIPTOR, etc. carry byte sizes.  Anything else suppresses.
    looks_like_memory = (
        isinstance(sz, int) and sz > 0 and
        ("MEMORY" in cls or "DESCRIPTOR" in cls or cls.startswith("NV01_"))
    )
    if looks_like_memory:
        return f"{handle_str}({cls}, {_human_bytes(sz)})"
    return f"{handle_str}({cls})"


def _human_bytes(n: int) -> str:
    if n >= 1 << 30 and n % (1 << 30) == 0:
        return f"{n >> 30}GiB"
    if n >= 1 << 20 and n % (1 << 20) == 0:
        return f"{n >> 20}MiB"
    if n >= 1 << 10 and n % (1 << 10) == 0:
        return f"{n >> 10}KiB"
    if n >= 1 << 30:
        return f"{n / (1<<30):.1f}GiB"
    if n >= 1 << 20:
        return f"{n / (1<<20):.1f}MiB"
    if n >= 1 << 10:
        return f"{n / (1<<10):.1f}KiB"
    return f"{n}B"


# ── folding pass ──────────────────────────────────────────────────────────
#
# A logical ioctl event ⊆ {strace.ioctl, ftrace.ioctl, ftrace.uvm_ioctl}
# all carrying the same cmd_name within FOLD_WINDOW_NS, with the same fd
# on the strace side.  The strace row carries fd + path + ret; the
# ftrace.ioctl row is redundant; the ftrace.uvm_ioctl row carries the
# decoded body dwords.  We synthesize one row that combines whatever's
# present.

FOLD_WINDOW_NS = 100_000  # 100 µs


def _is_ioctl_like(ev: Dict) -> bool:
    src = ev.get("src")
    kind = ev.get("kind")
    return (
        (src == "strace" and kind == "ioctl") or
        (src == "ftrace" and kind in ("ioctl", "uvm_ioctl"))
    )


def fold_ioctl_pairs(events: List[Dict]) -> List[Dict]:
    """Produce a new list where (strace ioctl X, ftrace ioctl X,
    ftrace uvm_ioctl X) clusters become one synthetic event of kind
    `_folded_ioctl` carrying every detail.

    Non-ioctl events pass through untouched.  The synthetic event keeps
    the earliest ts_ns and aggregates `data` from each component:
        {
          "cmd_name": ...,
          "strace": {"fd": ..., "path": ..., "size": ..., "ret": ...},
          "ftrace_ioctl": True | absent,
          "uvm": {"cmd": ..., "dwords": [...]}  # if present
        }
    """
    out: List[Dict] = []
    i = 0
    n = len(events)
    while i < n:
        ev = events[i]
        if not _is_ioctl_like(ev):
            out.append(ev)
            i += 1
            continue
        cmd_name = (ev.get("data") or {}).get("cmd_name")
        if not cmd_name:
            out.append(ev)
            i += 1
            continue

        # Collect adjacent rows within the window that share cmd_name
        # AND don't double up any (src, kind) — repeating e.g.
        # `strace ioctl X` means a NEW logical call started, not that
        # the same call has two strace rows.  Stop the cluster there.
        cluster: List[Tuple[int, Dict]] = [(i, ev)]
        seen_src_kind = {(ev.get("src"), ev.get("kind"))}
        j = i + 1
        t_first = ev["ts_ns"]
        while j < n:
            cand = events[j]
            if cand["ts_ns"] - t_first > FOLD_WINDOW_NS:
                break
            if not _is_ioctl_like(cand):
                break
            cd = cand.get("data") or {}
            if cd.get("cmd_name") != cmd_name:
                break
            sk = (cand.get("src"), cand.get("kind"))
            if sk in seen_src_kind:
                break
            seen_src_kind.add(sk)
            cluster.append((j, cand))
            j += 1

        if len(cluster) == 1:
            out.append(ev)
            i += 1
            continue

        # Build the synthetic event from the cluster.
        synthetic: Dict[str, Any] = {
            "ts_ns": cluster[0][1]["ts_ns"],
            "src":   "folded",
            "pid":   ev.get("pid"),
            "kind":  "_folded_ioctl",
            "data":  {"cmd_name": cmd_name},
        }
        components: List[str] = []
        for _, c in cluster:
            cd = c.get("data") or {}
            if c.get("src") == "strace" and c.get("kind") == "ioctl":
                synthetic["data"]["strace"] = {
                    "fd":   cd.get("fd"),
                    "path": cd.get("path"),
                    "size": cd.get("size"),
                    "ret":  cd.get("ret"),
                }
                components.append("strace")
            elif c.get("src") == "ftrace" and c.get("kind") == "ioctl":
                synthetic["data"]["ftrace_ioctl"] = True
                components.append("ftrace")
            elif c.get("src") == "ftrace" and c.get("kind") == "uvm_ioctl":
                synthetic["data"]["uvm"] = {
                    "cmd":    cd.get("cmd"),
                    "dwords": cd.get("dwords") or [],
                }
                components.append("uvm")
        synthetic["data"]["_components"] = components
        out.append(synthetic)
        i = cluster[-1][0] + 1
    return out


# ── per-event rendering ───────────────────────────────────────────────────

def _short_data(d: Dict, keys: List[str]) -> str:
    parts = []
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            v = d[k]
            if isinstance(v, dict):
                parts.append(f"{k}={json.dumps(v, separators=(',', ':'))}")
            else:
                parts.append(f"{k}={v}")
    return " ".join(parts)


def _render_uvm_payload(uvm: Dict, atlas: Dict[int, Dict]) -> str:
    """Decode the uvm_ioctl body via strace_diff.decode_uvm_cmd, with
    handle resolution applied to any `handle`-shaped fields.
    """
    cmd = uvm.get("cmd")
    raw_dwords = uvm.get("dwords") or []
    try:
        dwords = [int(s, 16) if isinstance(s, str) else int(s)
                  for s in raw_dwords]
    except (TypeError, ValueError):
        return ""
    if cmd is None:
        return ""
    decoded = decode_uvm_cmd(int(cmd), dwords)
    if not decoded:
        return ""
    # Resolve known handle fields against the atlas.
    HANDLE_KEYS = {"hMemory"}
    parts = []
    for k, v in decoded.items():
        if k in HANDLE_KEYS:
            v = _atlas_lookup(atlas, str(v))
        parts.append(f"{k}={v}")
    return " ".join(parts)


def render_folded_ioctl(ev: Dict, atlas: Dict[int, Dict]) -> str:
    d = ev.get("data") or {}
    cmd = d.get("cmd_name", "?")
    components = "+".join(d.get("_components") or []) or "?"
    line = f"ioctl {cmd} [{components}]"

    s = d.get("strace") or {}
    fd   = s.get("fd")
    path = s.get("path")
    size = s.get("size")
    ret  = s.get("ret")

    bits: List[str] = []
    if fd is not None:
        bits.append(f"fd={fd}")
    if path:
        bits.append(f"path={path}")
    if size not in (None, 0, ""):
        bits.append(f"sz={size}")
    if ret is not None:
        bits.append(f"ret={ret}")
    if bits:
        line += " " + " ".join(bits)

    # UVM body payload, rendered after all the bookkeeping fields.
    uvm = d.get("uvm")
    if uvm:
        payload = _render_uvm_payload(uvm, atlas)
        if payload:
            line += f"  {{{payload}}}"
    return line


def render_event(ev: Dict, atlas: Dict[int, Dict]) -> str:
    src  = ev.get("src", "?")
    kind = ev.get("kind", "?")
    data = ev.get("data") or {}

    if src == "folded" and kind == "_folded_ioctl":
        return render_folded_ioctl(ev, atlas)

    if src == "strace":
        if kind == "ioctl":
            cmd  = data.get("cmd_name") or data.get("cmd") or "?"
            path = data.get("path", "?")
            fd   = data.get("fd", "?")
            sz   = data.get("size", "")
            ret  = data.get("ret")
            sz_s  = f" sz={sz}" if sz != "" else ""
            ret_s = f" ret={ret}" if ret is not None else ""
            return f"strace ioctl {cmd}{sz_s}{ret_s} fd={fd} path={path}"
        if kind == "open":
            return f"strace open fd={data.get('fd','?')} ← {data.get('path','?')} flags={data.get('flags','?')}"
        if kind == "openat":
            return f"strace openat fd={data.get('fd','?')} ← {data.get('path','?')} flags={data.get('flags','?')}"
        if kind == "close":
            return f"strace close fd={data.get('fd','?')}"
        if kind == "mmap":
            return ("strace mmap "
                    f"addr={data.get('addr','?')} "
                    f"length={data.get('length','?')} "
                    f"prot={data.get('prot','?')} "
                    f"flags={data.get('flags','?')} "
                    f"fd={data.get('fd','?')} "
                    f"offset={data.get('offset','?')}")
        if kind == "munmap":
            return f"strace munmap addr={data.get('addr','?')} length={data.get('length','?')}"
        return f"strace {kind} {_short_data(data, sorted(data.keys()))}"

    if src == "ftrace":
        if kind == "alloc":
            handle = _atlas_lookup(atlas, data.get("handle", ""))
            cls    = data.get("class", "?")
            body   = data.get("body") or {}
            extras = _short_data(body, ["size", "flags", "attr", "attr2"])
            return f"ftrace alloc handle={handle} class={cls} {extras}".rstrip()
        if kind == "free":
            handle = _atlas_lookup(atlas, data.get("handle", ""))
            return f"ftrace free handle={handle}"
        if kind == "control":
            return ("ftrace control "
                    f"ctrlcmd={data.get('ctrlcmd','?')} "
                    f"hClient={_atlas_lookup(atlas, data.get('hClient',''))} "
                    f"hObject={_atlas_lookup(atlas, data.get('hObject',''))} "
                    f"paramsSize={data.get('paramsSize','?')}")
        if kind == "ioctl":
            cmd = data.get("cmd_name") or data.get("cmd") or "?"
            return f"ftrace ioctl {cmd} {_short_data(data, ['fd','path','size'])}".rstrip()
        if kind == "uvm_ioctl":
            cmd = data.get("cmd_name") or data.get("cmd") or "?"
            payload = _render_uvm_payload(data, atlas)
            payload_s = f"  {{{payload}}}" if payload else ""
            return f"ftrace uvm_ioctl {cmd}{payload_s}".rstrip()
        if kind == "map_bar1":
            handle = _atlas_lookup(atlas, data.get("handle", ""))
            base = _short_data(data, ["length", "flags", "offset", "fd"])
            return f"ftrace map_bar1 handle={handle} {base}".rstrip()
        if kind == "map_uvm":
            handle = _atlas_lookup(atlas, data.get("handle", ""))
            base = _short_data(data, ["length", "flags", "offset", "fd"])
            return f"ftrace map_uvm handle={handle} {base}".rstrip()
        if kind == "pb":
            return ("ftrace pb "
                    + _short_data(data, ["chid", "idx", "pb_va", "pb_len",
                                          "seq"]))
        if kind == "pb_bytes":
            return ("ftrace pb_bytes "
                    + _short_data(data, ["chid", "idx", "off", "chunk",
                                          "seq"]))
        if kind in ("sysmem_track_add", "sysmem_track_remove",
                    "bar1_track_add",   "bar1_track_remove"):
            return f"ftrace {kind} {_short_data(data, sorted(data.keys()))}"
        return f"ftrace {kind} {_short_data(data, sorted(data.keys()))}"

    if src == "pbcap":
        if kind.endswith(".enter") or kind.endswith(".exit"):
            return f"pbcap {kind} {_short_data(data, sorted(data.keys()))}"
        if kind == "open" or kind == "openat":
            return (f"pbcap {kind} fd={data.get('fd','?')} "
                    f"← {data.get('path','?')} "
                    f"flags={data.get('flags','?')}")
        if kind == "close":
            return f"pbcap close fd={data.get('fd','?')}"
        if kind == "mmap":
            return ("pbcap mmap "
                    + _short_data(data, ["addr", "length", "prot",
                                          "flags", "fd", "offset"]))
        if kind == "munmap":
            return ("pbcap munmap "
                    + _short_data(data, ["addr", "length"]))
        if kind == "pbcap.init":
            return f"pbcap init mono_ns={data.get('mono_ns','?')}"
        if kind in ("doorbell", "doorbell.arm"):
            return f"pbcap {kind} {_short_data(data, sorted(data.keys()))}"
        return f"pbcap {kind} {_short_data(data, sorted(data.keys()))}"

    return f"{src} {kind} {_short_data(data, sorted(data.keys()))}"


# ── report assembly ───────────────────────────────────────────────────────

def render_section(enter: Dict, exit_ev: Dict, events: List[Dict],
                   atlas: Dict[int, Dict], unfold: bool) -> str:
    t0 = enter["ts_ns"]
    duration_ns = exit_ev["ts_ns"] - t0
    enter_args  = json.dumps(enter.get("data") or {}, separators=(',', ':'))
    exit_ret    = json.dumps(exit_ev.get("data") or {}, separators=(',', ':'))
    section_name = enter["kind"][:-len(".enter")]

    rendered_events = events if unfold else fold_ioctl_pairs(events)

    tally: Counter = Counter()
    for ev in rendered_events:
        if ev is enter or ev is exit_ev:
            continue
        if ev.get("kind", "").endswith((".enter", ".exit")):
            continue
        if ev.get("src") == "folded":
            tally[("ioctl", (ev.get("data") or {}).get("cmd_name", "?"))] += 1
        else:
            tally[(ev.get("src", "?"), ev.get("kind", "?"))] += 1

    lines: List[str] = []
    lines.append(f"=== section: {section_name} ===")
    lines.append(f"  pid:        {enter.get('pid','?')}")
    lines.append(f"  duration:   {duration_ns/1000:.3f} µs ({duration_ns} ns)")
    lines.append(f"  events:     {len(rendered_events) - 2}  (excluding the enter/exit pair"
                 + (", folded" if not unfold else ", unfolded") + ")")
    lines.append(f"  args:       {enter_args}")
    lines.append(f"  returned:   {exit_ret}")
    lines.append("")
    lines.append("  ── timeline ──")
    for ev in rendered_events:
        off_us = (ev["ts_ns"] - t0) / 1000.0
        lines.append(f"  +{off_us:10.3f} µs  {render_event(ev, atlas)}")

    lines.append("")
    lines.append("  ── kind summary ──")
    if tally:
        for (a, b), n in sorted(tally.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"    {n:5d}  {a} {b}")
    else:
        lines.append("    (no events between enter and exit)")
    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Slice a merged.ndjson by a libcuda function name "
                    "and print the events bracketed by its "
                    "<fn>.enter / <fn>.exit pbcap markers.")
    ap.add_argument("--trace", required=True,
                    help="trace_cuda.sh output directory (must contain "
                         "merged.ndjson; atlas.json is consulted for "
                         "handle resolution if present)")
    ap.add_argument("--section", required=True,
                    help="libcuda function name(s) as hooked in pbcap.c "
                         "— comma-separated for multi-section concat "
                         "(e.g. cudaInitDevice,cudaHostRegister,"
                         "cudaHostUnregister)")
    ap.add_argument("--occurrence", type=int, default=0,
                    help="which call of each section to slice (default "
                         "0 = first); applied to every listed section")
    ap.add_argument("--unfold", action="store_true",
                    help="don't fold strace+ftrace+uvm_ioctl rows into "
                         "single logical events (debugging aid)")
    ap.add_argument("--out", default=None,
                    help="output file (default: stdout)")
    args = ap.parse_args(argv)

    merged_path = os.path.join(args.trace, "merged.ndjson")
    if not os.path.isfile(merged_path):
        raise SystemExit(
            f"trace_section: {merged_path} does not exist — did "
            f"trace_cuda.sh run successfully?")

    events  = load_merged(merged_path)
    atlas   = load_atlas(args.trace)
    sections = [s.strip() for s in args.section.split(",") if s.strip()]
    if not sections:
        raise SystemExit("trace_section: --section must list at least "
                         "one function name")

    parts: List[str] = []
    for sec in sections:
        win = find_window(events, sec, args.occurrence)
        if win is None:
            sys.stderr.write(
                f"trace_section: section {sec!r} (occurrence "
                f"{args.occurrence}) not found in trace — skipping\n")
            continue
        enter, exit_ev, window_events = win
        parts.append(render_section(enter, exit_ev, window_events,
                                    atlas, args.unfold))

    if not parts:
        raise SystemExit(
            "trace_section: no requested sections matched any "
            "<fn>.enter / <fn>.exit pair in the trace")

    report = "\n".join(parts)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
