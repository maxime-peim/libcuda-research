# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""
class_catalog.py — auto-extract GPU class-method catalog from NVIDIA's
open SDK headers for use by the pushbuffer decoder.

Scans every `src/common/sdk/nvidia/inc/class/*.h` header, pulls out the
`#define <CLASS_NAME> (0xNNNN)` class-id line at the top of each file
and the `#define NV<HEX>_<METHOD> (0xNNNN)` per-method registers below
it, and exposes two dicts:

    CLASS_NAMES:  Dict[int, str]              — class_id → human name
                                                  e.g. 0xC8B5 → "HOPPER_DMA_COPY_A"
    METHOD_NAMES: Dict[Tuple[int,int], str]   — (class_id, method_addr) → short name
                                                  e.g. (0xC8B5, 0x0300) → "LAUNCH_DMA"

The short name drops the `NV<HEX>_` prefix.  For every class whose
name contains CHANNEL_GPFIFO or _GPFIFO_, we hardcode
`(class_id, 0x0000) → "SET_OBJECT"` — all GPFIFO-family classes share
this reserved method and headers typically emit it as a "0-valued"
define that our filter otherwise drops as an enum.

The scan is cheap (~310 files, ~9.6 MB total text) so we do it eagerly
at import time.  On a slow filesystem this adds ~100 ms — if that
ever matters, add an mtime-keyed JSON cache under
`~/.cache/pbcap-tools/class_catalog.json`.

Filter heuristics for "is this #define a pushbuffer method register"
(value-based; name-based filters are fragile because of the
ENABLE/TRUE/FALSE/TYPE name-family variety):

    value != 0       (zero = reserved / SET_OBJECT; hardcoded separately)
    value % 4 == 0   (method-register addrs are 4-byte aligned)
    value >= 0x40    (everything smaller is an enum/bit-field value)

In practice this cleanly separates the ~23 method registers of
NVC8B5 from its ~100 `_TYPE_`/`_ENABLE_`/`_VALUE_` enum defines.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Dict, Optional, Tuple

# ── header-location search ─────────────────────────────────────────────

# Tools dir is reverse/tools/; repo root is three levels up.
_THIS = pathlib.Path(__file__).resolve()
_MAX_DEPTH = len(_THIS.parents) - 1
_DEFAULT_CLASSDIR = _THIS.parents[_MAX_DEPTH] / "src/common/sdk/nvidia/inc/class"

if not _DEFAULT_CLASSDIR.exists():
    for _ in range(_MAX_DEPTH + 1):
        _MAX_DEPTH -= 1
        _DEFAULT_CLASSDIR = (
            _THIS.parents[_MAX_DEPTH] / "src/common/sdk/nvidia/inc/class"
        )
        if _DEFAULT_CLASSDIR.exists():
            break
    else:
        sys.exit(1)

# ── regexes ────────────────────────────────────────────────────────────

# A top-of-header class-id line:
#   #define HOPPER_USERMODE_A       (0xc661)
#   #define HOPPER_CHANNEL_GPFIFO_A (0x0000C86F)
#   #define MAXWELL_COMPUTE_A       0xB0C0
#
# Reject NV<hex>_<tok> shapes (they're method defs) with a negative
# lookahead.  Accept the rest — then filter by value range.
_CLASS_ID_RE = re.compile(
    r"^#define\s+"
    r"(?P<name>(?!NV[A-Z0-9]+_)[A-Z][A-Z0-9_]*)\s+"
    r"\(?(?P<val>0x[0-9a-fA-F]+)\)?\s*$"
)

# A per-method register define:
#   #define NVC86F_SET_OBJECT                                          (0x00000000)
#   #define NVC8B5_LAUNCH_DMA                                          (0x00000300)
#   #define NVC0C0_LOAD_INLINE_DATA                                    (0x000001b4)
#   #define NVC6C0_SET_TRAP_HANDLER_A                                  0x25f8
_DEFINE_RE = re.compile(
    r"^#define\s+"
    r"(?P<name>NV[A-Z0-9]+_[A-Z0-9_]+)\s+"
    r"\(?(?P<val>0x[0-9a-fA-F]+)\)?"
    r"(?:\s*/\*.*\*/)?\s*$"
)

# Indexed-macro method define — one define that expands to a whole
# array of registers.  Example:
#   #define NVC6C0_SET_MME_SHADOW_SCRATCH(i)  (0x3400+(i)*4)
# We enumerate i in [0..MAX_INDEX_EXPAND) to cover every variant.
# Real arrays are bounded (Hopper compute has 128 MME_SHADOW_SCRATCH
# slots, vertex/etc up to 32) so 128 is a safe upper bound.
_INDEXED_DEFINE_RE = re.compile(
    r"^#define\s+"
    r"(?P<name>NV[A-Z0-9]+_[A-Z0-9_]+)\(i\)\s+"
    r"\(\s*(?P<base>0x[0-9a-fA-F]+)\s*\+\s*\(i\)\s*\*\s*(?P<stride>\d+)\s*\)"
    r"\s*$"
)
_MAX_INDEX_EXPAND = 128


# ── filters ────────────────────────────────────────────────────────────

def _looks_like_class_id(value: int) -> bool:
    """Class IDs are 2-byte values, typically 0x0040..0xFFFF."""
    return 0x40 <= value < 0x10000


def _is_method_register(value: int) -> bool:
    """Return True if this value is plausibly a pushbuffer method-register
    offset.  Filters out bit-field enum values, which are small and
    usually not 4-byte aligned.

    Value == 0 is rejected here because most class headers emit
    `NV<cls>_<METHOD>_<FIELD>_<VALUE_ZERO>` defines at 0; the special
    `SET_OBJECT` register at offset 0x0 on GPFIFO classes is
    hardcoded in `_scan_all` instead, keyed by class-name pattern.
    """
    if value == 0:
        return False
    if value & 0x3:
        return False
    if value < 0x40:
        return False
    return True


# ── scanner ────────────────────────────────────────────────────────────

def _scan_all(classdir: pathlib.Path
              ) -> Tuple[Dict[int, str], Dict[Tuple[int, int], str]]:
    """Walk every *.h in `classdir`, return (CLASS_NAMES, METHOD_NAMES)."""
    class_names: Dict[int, str] = {}
    methods: Dict[Tuple[int, int], str] = {}

    for h in sorted(classdir.glob("*.h")):
        try:
            text = h.read_text(errors="replace")
        except OSError:
            continue

        # First pass: class-id.  Consistently lives within the first
        # few #defines after the include guard (verified across 300+
        # headers).  Stop at the first NV<HEX>_<TOK> method-define we
        # see — by then we're out of the header preamble and
        # continuing would risk false-positives on deep-file enum
        # constants that happen to sit in the 0x40..0xFFFF range.
        class_id: Optional[int] = None
        for line in text.splitlines():
            # Short-circuit: a method-style define means the class-id
            # block is behind us.
            if _DEFINE_RE.match(line):
                break
            m = _CLASS_ID_RE.match(line)
            if not m:
                continue
            name = m.group("name")
            # Reject include guards + auto-gen bit-field names
            # (double-underscore is NVIDIA's convention for those).
            if name.startswith("_") or "__" in name:
                continue
            val = int(m.group("val"), 16)
            if _looks_like_class_id(val):
                class_id = val
                # Don't overwrite if a header was scanned already —
                # some class IDs are redefined across header variants.
                class_names.setdefault(class_id, name)
                break

        if class_id is None:
            continue

        # Second pass: method-register defines.
        for line in text.splitlines():
            # Regular scalar-method define.
            m = _DEFINE_RE.match(line)
            if m:
                val = int(m.group("val"), 16)
                if not _is_method_register(val):
                    continue
                full_name = m.group("name")
                short = full_name.split("_", 1)[1]
                key = (class_id, val)
                if key in methods and len(methods[key]) < len(short):
                    continue
                methods[key] = short
                continue

            # Indexed macro — expand to an array of methods.
            m = _INDEXED_DEFINE_RE.match(line)
            if m:
                base   = int(m.group("base"), 16)
                stride = int(m.group("stride"))
                if stride <= 0:
                    continue
                full_name = m.group("name")
                short = full_name.split("_", 1)[1]
                for idx in range(_MAX_INDEX_EXPAND):
                    addr = base + idx * stride
                    if not _is_method_register(addr):
                        break
                    key = (class_id, addr)
                    # Use a name that carries the index so "<NAME>(3)"
                    # renders as "NAME_3" in the decoder output.
                    idx_name = f"{short}_{idx}"
                    if key in methods and len(methods[key]) < len(idx_name):
                        continue
                    methods[key] = idx_name
                continue

    # Hardcode SET_OBJECT for GPFIFO-family classes.  Every one of
    # them has a reserved 0x0 entry for binding a class to a
    # subchannel, but the headers encode it as an all-zeros define
    # our filter rightly drops.
    for cid, name in class_names.items():
        if "CHANNEL_GPFIFO" in name or "_GPFIFO_" in name:
            methods[(cid, 0x0)] = "SET_OBJECT"

    return class_names, methods


# ── module-level data ──────────────────────────────────────────────────

CLASS_NAMES: Dict[int, str]
METHOD_NAMES: Dict[Tuple[int, int], str]
CLASS_NAMES, METHOD_NAMES = _scan_all(_DEFAULT_CLASSDIR)


# ── helpers ────────────────────────────────────────────────────────────

def class_name(class_id: int) -> str:
    """Return the symbolic name of a class, or 'class_0xNNNN' on miss."""
    return CLASS_NAMES.get(class_id, f"class_0x{class_id:04x}")


# Class-inheritance chains — NVIDIA ships a new class-ID per GPU arch
# (Pascal → Volta → Turing → Ampere → Ada → Hopper → Blackwell) but the
# method-register layout is mostly-additive: each new class inherits
# its predecessor's registers and adds new ones, but the predecessor's
# header is where the original register names live.  To resolve a
# method on a modern class like NVC8B5 (Hopper DMA-copy) we walk back
# through NVC3B5 → NVC1B5 → NVC0B5 until we find the register.
_INHERITANCE_CHAINS: Dict[int, Tuple[int, ...]] = {
    # DMA-copy family (class suffix _B5 on Kepler onwards).
    0xC8B5: (0xC8B5, 0xC7B5, 0xC6B5, 0xC5B5, 0xC3B5, 0xC1B5, 0xC0B5, 0xA0B5),
    # Compute family (_C0 suffix).  libcuda's de-facto HOPPER_COMPUTE_A.
    0xCBC0: (0xCBC0, 0xC9C0, 0xC7C0, 0xC6C0, 0xC5C0, 0xC3C0, 0xC0C0, 0xB0C0, 0xA1C0, 0xA0C0),
    # Pascal compute alone — called out for testing.
    0xC0C0: (0xC0C0, 0xB0C0, 0xA1C0, 0xA0C0),
    # Blackwell.
    0xCDC0: (0xCDC0, 0xCBC0, 0xC9C0, 0xC7C0, 0xC6C0, 0xC5C0, 0xC3C0, 0xC0C0, 0xB0C0, 0xA1C0, 0xA0C0),
    0xCEC0: (0xCEC0, 0xCDC0, 0xCBC0, 0xC9C0, 0xC7C0, 0xC6C0, 0xC5C0, 0xC3C0, 0xC0C0, 0xB0C0, 0xA1C0, 0xA0C0),
    # 3D graphics family (_97 suffix) — chain by arch.
    0xCB97: (0xCB97, 0xC997, 0xC797, 0xC697, 0xC597, 0xC397, 0xC097, 0xB197, 0xB097, 0xA197, 0xA097),
}


def method_name(class_id: int, addr: int) -> Optional[str]:
    """Return the short method name for (class_id, addr), else None.

    Lookup order:
    1. Exact (class_id, addr).
    2. Older classes in the same engine family (DMA-copy chain,
       compute chain, etc.).  NVIDIA's method catalog is mostly
       additive across arch generations, so register names carry
       forward but live in the header where they were introduced.
    3. Channel-level fallback (NVC86F) — SET_OBJECT / SEM_* / WFI /
       MEM_OP_* can appear on any subchannel.
    """
    if (class_id, addr) in METHOD_NAMES:
        return METHOD_NAMES[(class_id, addr)]
    for ancestor in _INHERITANCE_CHAINS.get(class_id, ())[1:]:
        if (ancestor, addr) in METHOD_NAMES:
            return METHOD_NAMES[(ancestor, addr)]
    if (0xC86F, addr) in METHOD_NAMES:
        return METHOD_NAMES[(0xC86F, addr)]
    return None


if __name__ == "__main__":
    print(f"CLASS DIR:    {_DEFAULT_CLASSDIR}")
    print(f"CLASS_NAMES:  {len(CLASS_NAMES)} classes")
    print(f"METHOD_NAMES: {len(METHOD_NAMES)} method registers")
    for cid in sorted(CLASS_NAMES):
        print(f"  0x{cid:04x} {CLASS_NAMES[cid]}")
