#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
# trace_cuda.sh — capture + analyze one CUDA (or mc) run end-to-end.
#
# Produces an output directory with:
#   ftrace.txt       — kernel trace buffer (mc1 pb/*, dbell/*, rm/*, ...)
#   strace.log       — syscall trace (ioctl/mmap/munmap/openat/close)
#   pbcap/           — libpbcap snapshots + timeline.ndjson
#   merged.ndjson    — unified timeline (ftrace + strace + pbcap), ts sorted
#   atlas.json       — hMemory + VA tables + decoded pb_bytes
#   methods.txt      — human-readable per-submission method decode
#   dmesg.{pre,post} — kernel log snapshots bracketing the run
#
# Run this ON THE BOX where the instrumented nvidia.ko / nvidia-dbell.ko
# are loaded (the H100 test host, in this project).  LD_PRELOAD won't fire
# from ssh side; trace buffers only see kernel activity on the box
# running the executable.
#
# Requires:
#   - mc1-instrumented nvidia{,-dbell,-uvm}.ko loaded
#   - libpbcap.so built: (cd reverse && make libpbcap)
#   - passwordless sudo (ftrace, dmesg, LD_PRELOAD as root)
#   - python3 (for timeline_merge.py + address_atlas.py)
#
# Usage:
#   trace_cuda.sh [--output-dir DIR] [--timeout SEC] <executable> [args...]
#
# Examples:
#   trace_cuda.sh ./bin/cuda_reference --size 4M
#   trace_cuda.sh /path/to/libmc/bin/mc_demo --size 64M --iters 2
#   trace_cuda.sh --timeout 30 /usr/local/cuda/samples/.../bench

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVERSE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBPBCAP="$REVERSE_DIR/lib/libpbcap.so"

OUTPUT_DIR=""
TIMEOUT=300

print_usage() {
  # Extract the "Usage:" block from this script's header.  The block
  # is terminated by a blank line (a bare "#" without anything after),
  # which is the marker between the examples and the "set -euo" code.
  awk '
    /^# Usage:/        { inside = 1 }
    inside && /^[^#]/  { exit }        # first non-# line ends the block
    inside             { sub(/^# ?/, ""); print }
  ' "$0"
}

while [ $# -gt 0 ] && [[ "$1" == -* ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --timeout)    TIMEOUT="$2"; shift 2 ;;
        --help|-h)    print_usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ $# -lt 1 ]; then
    print_usage >&2
    exit 1
fi

EXE="$1"; shift

if [ ! -f "$LIBPBCAP" ]; then
    echo "error: libpbcap.so not found at $LIBPBCAP" >&2
    echo "       build it first: (cd $REVERSE_DIR && make libpbcap)" >&2
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    STAMP="$(date +%Y%m%d-%H%M%S)"
    EXE_NAME="$(basename "$EXE")"
    OUTPUT_DIR="/tmp/trace-$EXE_NAME-$STAMP"
fi

mkdir -p "$OUTPUT_DIR/pbcap"
chmod 777 "$OUTPUT_DIR/pbcap"

FTRACE="$OUTPUT_DIR/ftrace.txt"
STRACE="$OUTPUT_DIR/strace.log"
PBCAP="$OUTPUT_DIR/pbcap"
MERGED="$OUTPUT_DIR/merged.ndjson"
ATLAS="$OUTPUT_DIR/atlas.json"
METHODS="$OUTPUT_DIR/methods.txt"
DMESG_PRE="$OUTPUT_DIR/dmesg.pre"
DMESG_POST="$OUTPUT_DIR/dmesg.post"

# ─────────── pre-flight ───────────
# x86 has only four hardware debug registers (DR0..DR3), and the doorbell
# watchpoint needs one.  Anything already holding them all — a display
# manager, a GPU daemon — silently degrades the watchpoint to pass-through.
# Stopping services is destructive on someone else's machine, so it is
# opt-in: set TRACE_STOP_SERVICES to a space-separated list to have them
# stopped for the duration of the capture.
if [ -n "${TRACE_STOP_SERVICES:-}" ]; then
    echo ">>> Stopping ${TRACE_STOP_SERVICES} so DR0..DR3 HW breakpoints are free..."
    # Best-effort: failures here don't block the run.
    # shellcheck disable=SC2086
    sudo systemctl stop ${TRACE_STOP_SERVICES} 2>/dev/null || true
else
    echo ">>> NOTE: not stopping any services.  If 'mc1 dbell/*' events are"
    echo "    missing, something else holds the debug registers; re-run with"
    echo "    TRACE_STOP_SERVICES='lightdm ollama' (adjust to your box)."
fi

echo ">>> Setting ftrace clock=mono + buffer=16 MB/CPU + clearing buffer..."
sudo bash -c 'echo mono > /sys/kernel/debug/tracing/trace_clock'
# 16384 KB/CPU = 16 MB/CPU.  Linux default is ~1408 KB/CPU, which
# wraps during libcuda's boot-time RM-alloc param-body dumps (dozens
# of KB per alloc × dozens of allocs, all bound to one CPU), causing
# early events like the 56 MiB pushbuffer-pool's sysmem_track_add
# to be discarded before the measured workload even starts.  Without
# those add events methods.txt reports every pb_va as UNRESOLVED.
sudo bash -c 'echo 16384 > /sys/kernel/debug/tracing/buffer_size_kb'
sudo bash -c 'echo > /sys/kernel/debug/tracing/trace'
# Some environments ship with tracing_on=0 by default (ftrace daemons
# flip it, stale test harness leaves it off, etc).  Force-enable.
sudo bash -c 'echo 1 > /sys/kernel/debug/tracing/tracing_on'
# Pause briefly after clearing+enabling.  On busy boxes we've observed
# tracepoints written during the clear getting clobbered; a short
# settle window lets the ring buffer reach a stable empty state
# before the measured process starts writing into it.
sleep 1

echo ">>> Snapshotting dmesg (for before/after kernel-issue diff)..."
sudo dmesg > "$DMESG_PRE"

# ─────────── capture ───────────
echo ">>> Running: $EXE $*"
echo "    output dir: $OUTPUT_DIR"
echo "    timeout:    ${TIMEOUT}s"

# Don't fail-fast on non-zero exit — we still want the trace even if
# the bench returned an error, and `timeout` exits 124 when it kills.
set +e
# Forward PBCAP_* opt-in envs through sudo's env-strip + into strace's
# child.  By default pbcap's userspace watchpoint is OFF; set e.g.
# PBCAP_DBELL=1 ./trace_cuda.sh ... to re-enable the legacy path.
sudo timeout "$TIMEOUT" env \
    LD_PRELOAD="$LIBPBCAP" \
    PBCAP_DIR="$PBCAP" \
    PBCAP_DBELL="${PBCAP_DBELL:-}" \
    PBCAP_DBELL_SAMPLE="${PBCAP_DBELL_SAMPLE:-}" \
    PBCAP_DBELL_SYNC="${PBCAP_DBELL_SYNC:-}" \
    PBCAP_MEMCPY_SNAPSHOT="${PBCAP_MEMCPY_SNAPSHOT:-}" \
    PBCAP_VERBOSE="${PBCAP_VERBOSE:-}" \
    PBCAP_MAX_BYTES="${PBCAP_MAX_BYTES:-}" \
  strace -ttt -T -f -o "$STRACE" \
    -e trace=ioctl,mmap,munmap,openat,close \
    "$EXE" "$@"
RC=$?
set -e
echo "    exit code: $RC"

echo ">>> Dumping ftrace..."
sudo cat /sys/kernel/debug/tracing/trace > "$FTRACE"
sudo dmesg > "$DMESG_POST"

# ─────────── analysis ───────────
echo ">>> Merging timeline..."
python3 "$SCRIPT_DIR/timeline_merge.py" \
    --pbcap-ndjson "$PBCAP/timeline.ndjson" \
    --ftrace "$FTRACE" --strace "$STRACE" \
    --out "$MERGED"

echo ">>> Building atlas + decoding methods..."
python3 "$SCRIPT_DIR/address_atlas.py" \
    --merged "$MERGED" --pbcap-dir "$PBCAP" \
    --atlas-out "$ATLAS" --methods-out "$METHODS"

# ─────────── summary ───────────
# Count kernel issues NEW to this run (diff of dmesg.post minus dmesg.pre).
# diff -u | grep '^+' catches lines added.  Using a simple line-count diff
# is fine for these noisy sentinels.
new_dmesg() {
    diff --changed-group-format='%>' --unchanged-group-format='' \
         "$DMESG_PRE" "$DMESG_POST" 2>/dev/null || true
}
# grep -c exits 1 on "no match", so `|| echo 0` previously double-emitted
# (both grep's "0" and the fallback "0").  Wrap in an if instead.
safe_count() {
    local n
    n=$(grep -c "$1" "$2" 2>/dev/null) || n=0
    echo "$n"
}
VFREE_NEW=$(safe_count "Trying to vfree"     <(new_dmesg))
BUG_NEW=$(safe_count    "BUG\|Oops\|Call Trace" <(new_dmesg))

count() { safe_count "$1" "$FTRACE"; }

cat <<EOF

=== capture summary ===
  exit code:              $RC  $([ "$RC" -eq 124 ] && echo '(timed out)')
  output dir:             $OUTPUT_DIR
  ftrace:                 $(wc -l < "$FTRACE" | tr -d ' ') lines
  strace:                 $(wc -l < "$STRACE" | tr -d ' ') lines
  pbcap snapshots:        $(ls "$PBCAP"/snap-*.bin 2>/dev/null | wc -l | tr -d ' ')

  dbell events:           $(count 'mc1 dbell/')
  pb/submit:              $(count 'mc1 pb/submit')
  pb/bytes:               $(count 'mc1 pb/bytes ')
  pb/bytes_miss:          $(count 'mc1 pb/bytes_miss')
  sysmem_track add:       $(count 'mc1 dbell/sysmem_track state=add')
  bar1_track add:         $(count 'mc1 dbell/bar1_track state=add')

  cache pre_reserve:      $(count 'mc1 dbell/cache state=pre_reserve')
  cache resolved:         $(count 'mc1 dbell/cache state=resolved')
  cache retry:            $(count 'mc1 dbell/cache state=retry')
  cache failed:           $(count 'mc1 dbell/cache state=failed')
  cache pending_init:     $(count 'mc1 dbell/cache state=pending_init')

  new vfree WARNs:        $VFREE_NEW
  new BUG/Oops:           $BUG_NEW

  merged.ndjson:          $MERGED
  atlas.json:             $ATLAS
  methods.txt:            $METHODS
EOF
