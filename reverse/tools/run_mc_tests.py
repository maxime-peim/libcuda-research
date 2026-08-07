#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
run_mc_tests.py — sweep the mc test binaries across a parameter
matrix and report pass / fail per cell.

Covers every test in reverse/tests/mc/ except mc_host_register_bench,
which is a benchmark rather than a pass/fail test and is run by hand.

The mc test set has two shapes:

  - parameterised demos (--size / --iters / --h2d, sometimes --fb / --sysmem):
      mc_demo, mc_carrier_demo, mc_sm_owner_demo,
      mc_compute_dbell_chain_demo, mc_dbell_chain_demo
  - one-shot tests with no flags:
      mc_allocation_test, mc_compute_dbell_demo, mc_dbell_demo

For each test we exercise the cross product of relevant parameters
(small + big sizes, multiple iters, both directions, both carriers
where applicable) and a couple of stress points (>= the 512-entry
GPFIFO ring boundary so the wrap path is hit).

Usage
-----

  python3 reverse/tools/run_mc_tests.py
      [--bin-dir reverse/bin]
      [--quick]          (only one size per test, --iters 4)
      [--stress]         (also run --iters 600 to cross the GPFIFO wrap)
      [--filter PAT]     (only run tests whose binary basename matches PAT)
      [--verbose]        (stream every command's stdout/stderr live)

Exit code is 0 iff every cell passed.

The script does NOT build anything — point --bin-dir at a tree that
already has the binaries.  Default is `reverse/bin/` relative to the
repository root inferred from the script's own location.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# ─── matrix definition ────────────────────────────────────────────────────────


@dataclass
class Cell:
    """One concrete test invocation."""
    binary: str
    args: list[str]
    timeout_s: float = 60.0
    # Optional predicate run on stdout+stderr to confirm the test
    # actually did what we expected (some demos print PASS even when
    # internal verifies skipped).  Default: just check exit==0.
    extra_check: Callable[[str], bool] | None = None

    @property
    def label(self) -> str:
        return f"{self.binary} {' '.join(self.args)}".rstrip()


def _expect_pass_keyword(out: str) -> bool:
    """Most mc_* tests print 'PASS' on success."""
    return "PASS" in out


def _expect_no_fail_keyword(out: str) -> bool:
    """Some tests print only 'mc_init ok' / 'OK' on success."""
    return ("FAIL" not in out) and ("error" not in out.lower())


def build_matrix(quick: bool, stress: bool) -> list[Cell]:
    """Return the full sweep as a flat list of cells."""

    cells: list[Cell] = []

    # Sizes that exercise:
    #   - latency-bound regime (4 KiB)
    #   - mid-size where per-submission latency starts to amortize (1 MiB)
    #   - big enough to saturate (8 MiB / 64 MiB)
    sizes_full   = ["4K", "1M", "8M", "64M"]
    sizes_quick  = ["8M"]
    sizes        = sizes_quick if quick else sizes_full

    iters_default = "4"

    # ── parameterised carrier demo: mc_carrier_demo (the broadest one) ──
    # 4-way matrix: (carrier × direction).
    for carrier_flag in ["--sysmem", "--fb"]:
        for direction in ["", "--h2d"]:
            for size in sizes:
                args = [carrier_flag, "--size", size, "--iters", iters_default]
                if direction:
                    args.append(direction)
                cells.append(Cell(
                    binary="mc_carrier_demo",
                    args=args,
                    extra_check=_expect_pass_keyword,
                ))

    # ── parameterised host-side demos: mc_demo, mc_sm_owner_demo,
    #    mc_compute_dbell_chain_demo, mc_dbell_chain_demo ──
    # 2-way matrix: direction.
    for binary in [
        "mc_demo",
        "mc_sm_owner_demo",
        "mc_compute_dbell_chain_demo",
        "mc_dbell_chain_demo",
    ]:
        for direction in ["", "--h2d"]:
            for size in sizes:
                args = ["--size", size, "--iters", iters_default]
                if direction:
                    args.append(direction)
                cells.append(Cell(
                    binary=binary,
                    args=args,
                    extra_check=_expect_pass_keyword,
                ))

    # ── one-shot tests: just exercise that they still link/run ──
    for binary in ["mc_allocation_test", "mc_compute_dbell_demo", "mc_dbell_demo"]:
        cells.append(Cell(
            binary=binary,
            args=[],
            extra_check=_expect_no_fail_keyword,
        ))

    # ── stress: cross the 512-entry GPFIFO wrap ──
    # Two SM submissions per --h2d iter (timed + readback) → 600 iters
    # is ~1200 entries on the SM-author path, well past the wrap.
    if stress:
        for carrier_flag in ["--sysmem", "--fb"]:
            for direction in ["", "--h2d"]:
                args = [carrier_flag, "--size", "4K", "--iters", "600"]
                if direction:
                    args.append(direction)
                cells.append(Cell(
                    binary="mc_carrier_demo",
                    args=args,
                    timeout_s=120.0,
                    extra_check=_expect_pass_keyword,
                ))

    return cells


# ─── runner ───────────────────────────────────────────────────────────────────


@dataclass
class Result:
    cell: Cell
    rc: int
    duration_s: float
    stdout: str
    stderr: str
    bandwidth_gbps: float | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        if self.rc != 0:
            return False
        if self.cell.extra_check is None:
            return True
        return self.cell.extra_check(self.stdout + self.stderr)


# Matches both shapes the test binaries emit:
#   "  iter  3: H2D=  627.6 us (13.37 GB/s)"        ← carrier / dbell demos
#   "  Peak: 0.71 GB/s (0.01 ms)"                    ← mc_demo
_RE_BANDWIDTH = re.compile(r"([0-9]+\.[0-9]+)\s*GB/s")


def _peak_bandwidth(out: str) -> float | None:
    """Best bandwidth seen in the test's stdout, in GB/s."""
    matches = [float(m.group(1)) for m in _RE_BANDWIDTH.finditer(out)]
    return max(matches) if matches else None


def run_cell(cell: Cell, bin_dir: Path, verbose: bool) -> Result:
    bin_path = bin_dir / cell.binary
    if not bin_path.exists():
        return Result(cell, rc=127, duration_s=0.0, stdout="",
                      stderr=f"binary not found: {bin_path}",
                      note="missing")

    cmd: list[str] = [str(bin_path)]
    cmd.extend(cell.args)

    if verbose:
        print(f"  $ {' '.join(shlex.quote(x) for x in cmd)}")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cell.timeout_s,
        )
        rc = proc.returncode
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired as exc:
        return Result(
            cell,
            rc=124,
            duration_s=cell.timeout_s,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\n[TIMEOUT after {cell.timeout_s:.0f}s]",
            note="timeout",
        )

    duration = time.monotonic() - t0
    bw = _peak_bandwidth(out)
    return Result(cell, rc=rc, duration_s=duration,
                  stdout=out, stderr=err, bandwidth_gbps=bw)


# ─── reporting ────────────────────────────────────────────────────────────────


def fmt_result(r: Result) -> str:
    status = "PASS" if r.ok else "FAIL"
    bw = f" {r.bandwidth_gbps:5.2f} GB/s" if r.bandwidth_gbps is not None else ""
    note = f" [{r.note}]" if r.note else ""
    return f"  [{status}] {r.duration_s:6.2f}s{bw}{note:>10}  {r.cell.label}"


def print_failure_detail(r: Result) -> None:
    print(f"\n--- FAIL: {r.cell.label} (rc={r.rc}) ---")
    if r.stdout.strip():
        print("  stdout (last 20 lines):")
        for line in r.stdout.rstrip().splitlines()[-20:]:
            print(f"    {line}")
    if r.stderr.strip():
        print("  stderr (last 20 lines):")
        for line in r.stderr.rstrip().splitlines()[-20:]:
            print(f"    {line}")


# ─── main ─────────────────────────────────────────────────────────────────────


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", default=None,
                    help="directory containing test binaries "
                         "(default: <repo>/reverse/bin)")
    ap.add_argument("--quick", action="store_true",
                    help="only run one size per test, no stress")
    ap.add_argument("--stress", action="store_true",
                    help="also run iters=600 to cross the GPFIFO 512 wrap")
    ap.add_argument("--filter", default=None,
                    help="only run tests whose binary basename matches this regex")
    ap.add_argument("--verbose", action="store_true",
                    help="print every command before running it")
    args = ap.parse_args(argv)

    bin_dir = Path(args.bin_dir) if args.bin_dir else repo_root() / "reverse" / "bin"
    if not bin_dir.is_dir():
        print(f"error: bin dir does not exist: {bin_dir}", file=sys.stderr)
        return 2


    cells = build_matrix(quick=args.quick, stress=args.stress)

    if args.filter:
        pat = re.compile(args.filter)
        cells = [c for c in cells if pat.search(c.binary)]
        if not cells:
            print(f"no tests matched --filter {args.filter!r}", file=sys.stderr)
            return 2

    print(f"running {len(cells)} cells from {bin_dir}")
    if args.quick:   print("  (quick: one size per test)")
    if args.stress:  print("  (stress: includes iters=600 wrap test)")
    print()

    results: list[Result] = []
    for cell in cells:
        r = run_cell(cell, bin_dir, verbose=args.verbose)
        results.append(r)
        print(fmt_result(r), flush=True)

    # ── summary ──
    passed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print()
    print(f"summary: {len(passed)}/{len(results)} passed, "
          f"{len(failed)} failed, "
          f"total wall {sum(r.duration_s for r in results):.1f}s")

    if failed:
        for r in failed:
            print_failure_detail(r)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
