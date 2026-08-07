#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""phase_census.py — split a capture into setup / transfer / teardown and count.

The question this answers is the one the whole tracing apparatus was built
for: *how much driver work does a cudaMemcpy actually cost?*  The answer is
easier to see as three numbers than as a timeline, so this tool cuts a
capture at the transfer window and counts kernel events on either side.

Phases are taken from the libcuda-level brackets pbcap records, not from
wall-clock guesses:

    A  setup     start of capture   .. first cudaMemcpy.enter
    B  transfer  first .enter       .. last  .exit
    C  teardown  last  .exit        .. end of capture

Counts come from ftrace.txt rather than merged.ndjson, because `gsp/rpc_tx`
has no consumer in the merge layer and never reaches the merged timeline.
ftrace is the authority for every mc1 record regardless.

Usage:
    phase_census.py --trace DIR [--json] [--out FILE]
"""

import argparse
import collections
import json
import os
import re
import sys

# "  cuda_reference-11211   [023] .....  1757.187586: nv_trace_printf: mc1 rm/ioctl cmd=0x2b ..."
FTRACE_RE = re.compile(
    r"^\s*\S+\s+\[\d+\]\s+\S+\s+(?P<ts>\d+\.\d+):\s+\S+:\s+"
    r"mc1 (?P<cat>[a-z][a-z0-9_]*)/(?P<event>[a-z][a-z0-9_]*)(?P<rest>.*)$"
)
KV_RE = re.compile(r"(\w+)=(\S+)")

# The events worth counting per phase, in report order.  Each is
# (record, human label).
COUNTED = [
    ("rm/ioctl",    "RM ioctls"),
    ("rm/control",  "  of which RM_CONTROL"),
    ("gsp/rpc_tx",  "GSP RPCs"),
    ("uvm/ioctl",   "UVM ioctls"),
    ("dbell/fire",  "doorbells"),
    ("pb/submit",   "pushbuffer submissions"),
]


def load_ftrace(path):
    """Yield (ts_ns, "cat/event", {k: v}) for every mc1 record."""
    with open(path, errors="replace") as fh:
        for line in fh:
            m = FTRACE_RE.match(line)
            if not m:
                continue
            ts_ns = int(round(float(m.group("ts")) * 1e9))
            rec = "%s/%s" % (m.group("cat"), m.group("event"))
            yield ts_ns, rec, dict(KV_RE.findall(m.group("rest")))


def transfer_window(pbcap_path):
    """Return (first cudaMemcpy.enter, last cudaMemcpy.exit) in mono ns."""
    enters, exits = [], []
    with open(pbcap_path, errors="replace") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind, ts = ev.get("kind"), ev.get("ts_ns")
            if ts is None:
                continue
            if kind == "cudaMemcpy.enter":
                enters.append(ts)
            elif kind == "cudaMemcpy.exit":
                exits.append(ts)
    if not enters or not exits:
        sys.exit("error: no cudaMemcpy.enter/.exit pair in %s — nothing to "
                 "split on" % pbcap_path)
    return min(enters), max(exits), len(enters)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="capture directory")
    ap.add_argument("--json", action="store_true", help="emit JSON, not a table")
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args()

    ftrace = os.path.join(args.trace, "ftrace.txt")
    pbcap = os.path.join(args.trace, "pbcap", "timeline.ndjson")
    for p in (ftrace, pbcap):
        if not os.path.exists(p):
            sys.exit("error: %s not found" % p)

    t_start, t_end, n_memcpy = transfer_window(pbcap)

    records = list(load_ftrace(ftrace))
    if not records:
        sys.exit("error: no mc1 records in %s" % ftrace)
    cap_first, cap_last = records[0][0], records[-1][0]

    def phase_of(ts):
        if ts < t_start:
            return "setup"
        if ts <= t_end:
            return "transfer"
        return "teardown"

    counts = {p: collections.Counter() for p in ("setup", "transfer", "teardown")}
    gsp_names = {p: collections.Counter() for p in ("setup", "transfer", "teardown")}
    ctrl_cmds = {p: set() for p in ("setup", "transfer", "teardown")}

    for ts, rec, kv in records:
        p = phase_of(ts)
        counts[p][rec] += 1
        if rec == "gsp/rpc_tx":
            gsp_names[p][kv.get("name", "?")] += 1
        elif rec == "rm/control":
            ctrl_cmds[p].add(kv.get("cmd", "?"))

    spans = {
        "setup":    (cap_first, t_start),
        "transfer": (t_start, t_end),
        "teardown": (t_end, cap_last),
    }

    result = {
        "trace": os.path.abspath(args.trace),
        "cudaMemcpy_calls": n_memcpy,
        "phases": {
            p: {
                "wall_ms": round((spans[p][1] - spans[p][0]) / 1e6, 2),
                "counts": {rec: counts[p][rec] for rec, _ in COUNTED},
                "gsp_rpc_by_name": dict(gsp_names[p].most_common()),
                "rm_control_distinct_cmds": len(ctrl_cmds[p]),
            }
            for p in ("setup", "transfer", "teardown")
        },
        "totals": {
            "counts": {rec: sum(counts[p][rec] for p in counts) for rec, _ in COUNTED},
            "gsp_rpc_by_name": dict(sum(gsp_names.values(), collections.Counter()).most_common()),
            "rm_control_distinct_cmds": len(set().union(*ctrl_cmds.values())),
            "wall_ms": round((cap_last - cap_first) / 1e6, 2),
        },
    }

    if args.json:
        text = json.dumps(result, indent=2)
    else:
        out = []
        out.append("capture: %s" % result["trace"])
        out.append("cudaMemcpy calls: %d" % n_memcpy)
        out.append("")
        out.append("%-26s %10s %10s %10s %10s"
                   % ("", "setup", "transfer", "teardown", "total"))
        out.append("%-26s %10s %10s %10s %10s"
                   % ("wall time (ms)",
                      result["phases"]["setup"]["wall_ms"],
                      result["phases"]["transfer"]["wall_ms"],
                      result["phases"]["teardown"]["wall_ms"],
                      result["totals"]["wall_ms"]))
        for rec, label in COUNTED:
            out.append("%-26s %10d %10d %10d %10d"
                       % (label,
                          counts["setup"][rec], counts["transfer"][rec],
                          counts["teardown"][rec], result["totals"]["counts"][rec]))
        out.append("%-26s %10d %10d %10d %10d"
                   % ("  distinct control cmds",
                      len(ctrl_cmds["setup"]), len(ctrl_cmds["transfer"]),
                      len(ctrl_cmds["teardown"]),
                      result["totals"]["rm_control_distinct_cmds"]))
        out.append("")
        out.append("GSP RPCs by name (whole capture):")
        for name, n in result["totals"]["gsp_rpc_by_name"].items():
            out.append("  %-20s %6d" % (name, n))
        text = "\n".join(out)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print("wrote %s" % args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
