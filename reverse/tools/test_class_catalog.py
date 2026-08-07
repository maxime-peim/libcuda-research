#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""Sanity-check the auto-scanned class/method catalog.

Run directly (no pytest needed):
    python3 reverse/tools/test_class_catalog.py

Exits 0 on success, non-zero on first failure.  Intentionally minimal
— we just want a smoke alarm that the header-regex + filters still
produce the handful of mappings downstream tools (decode.py,
address_atlas.py) rely on.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from class_catalog import CLASS_NAMES, METHOD_NAMES, class_name, method_name  # noqa: E402


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _eq(actual, expected, label: str) -> None:
    if actual != expected:
        _fail(f"{label}: got {actual!r}, expected {expected!r}")


# ── class-ID → name table ─────────────────────────────────────────────
_eq(CLASS_NAMES.get(0xC8B5), "HOPPER_DMA_COPY_A",         "CLASS_NAMES[0xC8B5]")
_eq(CLASS_NAMES.get(0xC86F), "HOPPER_CHANNEL_GPFIFO_A",   "CLASS_NAMES[0xC86F]")
_eq(CLASS_NAMES.get(0xCBC0), "HOPPER_COMPUTE_A",          "CLASS_NAMES[0xCBC0]")
_eq(CLASS_NAMES.get(0xC661), "HOPPER_USERMODE_A",         "CLASS_NAMES[0xC661]")
_eq(CLASS_NAMES.get(0xC0C0), "PASCAL_COMPUTE_A",          "CLASS_NAMES[0xC0C0]")

# ── per-class method registers ─────────────────────────────────────────
# HOPPER_DMA_COPY_A
_eq(METHOD_NAMES.get((0xC8B5, 0x0240)), "SET_SEMAPHORE_A",    "NVC8B5.0x0240")
_eq(METHOD_NAMES.get((0xC8B5, 0x0300)), "LAUNCH_DMA",         "NVC8B5.0x0300")
_eq(METHOD_NAMES.get((0xC8B5, 0x0400)), "OFFSET_IN_UPPER",    "NVC8B5.0x0400")
_eq(METHOD_NAMES.get((0xC8B5, 0x0418)), "LINE_LENGTH_IN",     "NVC8B5.0x0418")
_eq(METHOD_NAMES.get((0xC8B5, 0x0700)), "SET_REMAP_CONST_A",  "NVC8B5.0x0700")

# HOPPER_CHANNEL_GPFIFO_A — SET_OBJECT is hardcoded for GPFIFO classes.
_eq(METHOD_NAMES.get((0xC86F, 0x0000)), "SET_OBJECT",         "NVC86F.0x0000")
_eq(METHOD_NAMES.get((0xC86F, 0x006c)), "SEM_EXECUTE",        "NVC86F.0x006c")
_eq(METHOD_NAMES.get((0xC86F, 0x0078)), "WFI",                "NVC86F.0x0078")

# PASCAL_COMPUTE_A — reused by Hopper at runtime; the header carries
# the method catalog.  These are the most-frequent undecoded methods
# in CUDA captures before this catalog was wired in.
_eq(METHOD_NAMES.get((0xC0C0, 0x01b4)), "LOAD_INLINE_DATA",       "NVC0C0.0x01b4")
_eq(METHOD_NAMES.get((0xC0C0, 0x1b00)), "SET_REPORT_SEMAPHORE_A", "NVC0C0.0x1b00")

# ── helper functions ──────────────────────────────────────────────────
_eq(class_name(0xC8B5), "HOPPER_DMA_COPY_A",       "class_name(0xC8B5)")
_eq(class_name(0xDEAD), "class_0xdead",            "class_name(0xDEAD) fallback")

_eq(method_name(0xC8B5, 0x0300), "LAUNCH_DMA",     "method_name NVC8B5/LAUNCH_DMA")
# Channel-level fallback: SEM_EXECUTE is looked up as NVC8B5.0x006c but
# isn't there → falls back to NVC86F.0x006c.
_eq(method_name(0xC8B5, 0x006c), "SEM_EXECUTE",    "method_name fallback to NVC86F")
_eq(method_name(0xC8B5, 0xdead),  None,            "method_name unknown")

# Ancestor-chain fallback: NVC8B5 (Hopper DMA-copy) doesn't define
# LINE_COUNT at 0x041c directly, but its ancestor NVC3B5 (Turing) does.
# The chain walk in method_name() should find it.
_eq(method_name(0xC8B5, 0x041c), "LINE_COUNT",
    "method_name ancestor-chain: NVC8B5 → NVC3B5/LINE_COUNT")

# Indexed-macro expansion: SET_MME_SHADOW_SCRATCH(i) at (0x3400 + i*4)
# from clc0c0.h — compute's per-slot scratchpad array.
_eq(method_name(0xC0C0, 0x3400), "SET_MME_SHADOW_SCRATCH_0",
    "method_name indexed-macro: NVC0C0/SET_MME_SHADOW_SCRATCH(0)")
_eq(method_name(0xC0C0, 0x3404), "SET_MME_SHADOW_SCRATCH_1",
    "method_name indexed-macro: NVC0C0/SET_MME_SHADOW_SCRATCH(1)")
_eq(method_name(0xC0C0, 0x3408), "SET_MME_SHADOW_SCRATCH_2",
    "method_name indexed-macro: NVC0C0/SET_MME_SHADOW_SCRATCH(2)")

# ── bulk sanity: the scanner hit the headers and didn't over-filter ───
# Expanded-macro entries inflate the count to ~26k; 100k is still a
# generous sanity cap that would catch a regex blow-up.
if not (500 < len(METHOD_NAMES) < 100000):
    _fail(f"METHOD_NAMES size {len(METHOD_NAMES)} out of sanity range")
if not (20 < len(CLASS_NAMES) < 500):
    _fail(f"CLASS_NAMES size {len(CLASS_NAMES)} out of sanity range")

print(f"OK: {len(CLASS_NAMES)} classes, {len(METHOD_NAMES)} method registers")
