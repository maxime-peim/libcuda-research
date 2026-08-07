#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
"""find_bar1_pfn.py — scan a LIVE process's /proc/PID/{maps,pagemap}
for userspace VAs whose physical PFNs fall inside a given BAR1
physical address range.

Reads pagemap by seeking ONLY to the VMA ranges listed in maps — this
is critical because pagemap is sparse over the full 48-bit virtual
address space; copying it with `cp` allocates terabytes.  Always
operate on /proc/PID/pagemap directly, never a copy.

Usage: sudo find_bar1_pfn.py <PID> <pfn_lo_hex> <pfn_hi_hex>

Example (scan for USERD in BAR1 PFN range around 0x1fc001232):
    sudo find_bar1_pfn.py 4202 0x1fc000000 0x1fc100000
"""
import struct
import sys

PAGE_SIZE = 4096
PAGEMAP_ENTRY_SIZE = 8

def parse_maps(path):
    vmas = []
    with open(path) as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) < 5:
                continue
            rng, perms, offset, dev, inode = parts[:5]
            path_ = parts[5].strip() if len(parts) == 6 else ""
            a, b = rng.split("-")
            vmas.append((int(a, 16), int(b, 16), perms, path_))
    return vmas

def scan_vma(pagemap_fd, start, end, pfn_lo, pfn_hi):
    hits = []
    present_count = 0
    n_pages = (end - start) // PAGE_SIZE
    offset = (start // PAGE_SIZE) * PAGEMAP_ENTRY_SIZE
    pagemap_fd.seek(offset)
    # Read all entries at once — at most n_pages * 8 bytes, a 2 MiB VMA = 4 KiB.
    raw = pagemap_fd.read(n_pages * PAGEMAP_ENTRY_SIZE)
    for i in range(0, len(raw), PAGEMAP_ENTRY_SIZE):
        entry = struct.unpack_from("Q", raw, i)[0]
        present = (entry >> 63) & 1
        swapped = (entry >> 62) & 1
        if not present or swapped:
            continue
        present_count += 1
        pfn = entry & ((1 << 55) - 1)
        if pfn_lo <= pfn < pfn_hi:
            va = start + (i // PAGEMAP_ENTRY_SIZE) * PAGE_SIZE
            hits.append((va, pfn))
    return hits, present_count

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    pid, pfn_lo, pfn_hi = sys.argv[1:]
    pfn_lo = int(pfn_lo, 16)
    pfn_hi = int(pfn_hi, 16)
    vmas = parse_maps(f"/proc/{pid}/maps")
    print(f"scanning {len(vmas)} VMAs for PFNs in [0x{pfn_lo:x}, 0x{pfn_hi:x})")
    with open(f"/proc/{pid}/pagemap", "rb") as pmf:
        total_hits = 0
        for (start, end, perms, path) in vmas:
            hits, present = scan_vma(pmf, start, end, pfn_lo, pfn_hi)
            if hits:
                size_kb = (end - start) // 1024
                print(f"\n[{start:#x}-{end:#x}] {perms} {size_kb}KiB {path}")
                print(f"  (present_pages={present}, bar1_hits={len(hits)})")
                for (va, pfn) in hits[:16]:
                    print(f"  va=0x{va:x} pfn=0x{pfn:x} phys=0x{pfn << 12:x}")
                if len(hits) > 16:
                    print(f"  ... +{len(hits)-16} more")
                total_hits += len(hits)
        print(f"\n=== {total_hits} total hits in BAR1 range ===")

if __name__ == "__main__":
    main()
