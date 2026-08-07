# Tracing a CUDA (or mc_demo) run end-to-end

This doc walks through how to capture and inspect the complete kernel
+ userspace timeline of any CUDA executable running on a box with the
instrumented `nvidia.ko` / `nvidia-dbell.ko` loaded. One helper script
does steps 1–3 automatically; the rest of this doc explains how to
drive the pipeline by hand and lists the tools that are useful when
the automated path falls short.

---

## The one-liner

```bash
cd reverse
./tools/trace_cuda.sh ./bin/cuda_reference --size 4M
# ...or any other CUDA binary:
./tools/trace_cuda.sh ./bin/mc_demo --size 64M --iters 2
./tools/trace_cuda.sh --timeout 30 /opt/my-cuda-app arg1 arg2
```

The script creates a timestamped directory under `/tmp/trace-<exe>-<stamp>/`
containing everything you need. It ends by printing a summary showing
the per-kind event counts and whether any kernel WARNs / BUGs appeared
during the run.

### Output tree

```
/tmp/trace-cuda_reference-20260507-122712/
├── ftrace.txt        # kernel trace buffer (mc1 dbell/*, pb/*, rm/*, ...)
├── strace.log        # user-syscall trace (ioctl, mmap, munmap, openat, close)
├── pbcap/            # libpbcap snapshots + timeline.ndjson
│   ├── timeline.ndjson
│   └── snap-*.bin
├── merged.ndjson     # ftrace + strace + pbcap, sorted on CLOCK_MONOTONIC
├── atlas.json        # hMemory / VA tables + decoded pb_bytes
├── methods.txt       # human-readable per-submission method decode
├── dmesg.pre         # dmesg snapshot right before the run
└── dmesg.post        # dmesg snapshot right after the run
```

### What the summary looks like

```
=== capture summary ===
  exit code:              0
  output dir:             /tmp/trace-cuda_reference-20260818-112411
  ftrace:                 10601 lines
  strace:                 839 lines
  pbcap snapshots:        0

  dbell events:           612
  pb/submit:              205
  pb/bytes:               724
  pb/bytes_miss:          0
  sysmem_track add:       6
  bar1_track add:         1

  cache pre_reserve:      26
  cache resolved:         20
  cache retry:            6
  cache failed:           0
  cache pending_init:     0

  new vfree WARNs:        0
  new BUG/Oops:           0
```

The counters are `grep -c` over the `mc1` records in `ftrace.txt`, so they
are also the quickest sanity check on a capture.  If `pb/bytes_miss` is
non-zero, the sysmem tracker didn't have a kernel VA for some pb_va —
either the mapping was below the 256-page filter or the pushbuffer lives
on a mapping path we don't hook.  Run `locate_pushbuffer.py` to classify
the unresolved VAs (see below).  `cache failed` should be 0; `cache retry`
is benign (the resolver retried and succeeded).

`pbcap snapshots: 0` is the normal reading — the userspace snapshot paths are
opt-in (`PBCAP_MEMCPY_SNAPSHOT=1`, `PBCAP_DBELL=1`).  If you turn one on and
still get 0 regions dumped, raise `PBCAP_MAX_BYTES`: it caps each mapping at
16 MiB by default and the pushbuffer pool is 56 MiB, so at the default the one
region you actually wanted is skipped.

---

## Prerequisites

The script only works on a box where:

1. **The instrumented `nvidia.ko` + `nvidia-dbell.ko` modules are
   loaded.** Check with `lsmod | grep nvidia_dbell`. If missing, see
   `findings.md §12` for the build + install steps.
2. **libpbcap.so is built.** Run `(cd reverse && make libpbcap)`
   once. The script expects it at `reverse/lib/libpbcap.so`.
3. **Passwordless sudo is available** for the user who runs the
   script. The script needs sudo to write the ftrace control files,
   `LD_PRELOAD` as root, and read the dmesg buffer.
4. **DR0–DR3 debug registers aren't claimed by other processes.**
   x86 has exactly 4 HW breakpoints per CPU and Xorg / ollama /
   anything using HW watchpoints will grab them before the bench
   starts. The script stops `lightdm` and `ollama` best-effort;
   inspect `systemctl list-units --type=service | grep active` if
   `mc1 dbell/*` events don't appear.

In this project all of the above hold on the H100 test host. For a fresh
deployment see `findings.md §12`.

### Userspace vs. kernel doorbell watchpoint

libpbcap's `PBCAP_DBELL=1` path arms a userspace SIGSEGV+TF+SIGTRAP
watchpoint on the HOPPER_USERMODE_A mapping. That is this project's own
earlier attempt, not the paper's method: Yan et al. §5.1–§5.2 propose the
**kernel-side** watchpoint, and their §3 argues that a userspace approach
cannot win the submission race at all — which `findings.md §12.1` confirms
by measurement. It is **off by default** because the kernel module
`nvidia-dbell.ko` intercepts the same doorbell writes in a `#DB`
trap handler before PBDMA sees them (see `findings.md §12`), and
emits `mc1 pb/submit` / `mc1 pb/bytes` records that reach
`address_atlas.py` through `strace_diff.py` → `merged.ndjson`.
Running the userspace path on
top adds 50–100× wallclock overhead (it calls `snapshot_all` from
signal context on every doorbell — CUDA hammers the doorbell
hundreds of times per `cudaMemcpy` while polling) and cannot recover the
work-submit token or the ring state, so it yields strictly less
information. Export `PBCAP_DBELL=1` only to A/B-compare against the
kernel path, or to reproduce the negative result for yourself.
Similarly, the `snapshot_all("pre/post")` calls around each
`cudaMemcpy` are gated by `PBCAP_MEMCPY_SNAPSHOT=1` (default off) —
the kernel's `pb/bytes` records now carry the pushbuffer contents.

---

## Reading `methods.txt`

`methods.txt` has three sections:

### `=== doorbell ... ===` (from pbcap's LD_PRELOAD watchpoint)

```
=== doorbell seq=0 ts_ns=314815582563 tid=3491 bar0=0x77160af59000 \
    snapshot=(no pushbuffer snap) ===
  (empty method stream)
```

One block per doorbell pbcap observed from *userspace*. `snapshot=`
names the `/tmp/.../pbcap/snap-*.bin` file the script picked as the
most-likely pushbuffer; `(no pushbuffer snap)` means pbcap couldn't
capture the pushbuffer at that instant. This section is mostly empty
for modern captures (the kernel #DB handler is where the real method
bytes come from — see below).

### `=== PB_EVENTS: N doorbells, M resolved, K unresolved ===`

One line per GPU work submit, as observed by the **kernel** doorbell
handler:

```
seq=1846 chid=  3 idx=  0 pb_va=0x203600000 pb_len=32
    → hMemory=0x5c000019 class=NV01_MEMORY_SYSTEM offset=50331648 sizeMiB=56.00
```

Fields:

- `seq`: monotonic kernel event counter.
- `chid`, `idx`: channel + GPFIFO entry index the doorbell advanced.
- `pb_va`, `pb_len`: the pushbuffer VA + byte length read from the
  GPFIFO entry. Under UVM, `pb_va` equals libcuda's userspace VA
  (Paper Finding 1).
- `hMemory`, `class`, `offset`, `sizeMiB`: the allocation that
  contains `pb_va`, resolved against `atlas.gpu_ranges`. Shows
  `UNRESOLVED` if the VA doesn't land in any tracked range.

### `=== PB_BYTES_DECODE: N submissions, M decoded methods ===`

The meat. One block per unique `(chid, idx)` with the full method
stream, with each line showing `[sch<N>] <CLASS>/<METHOD>`:

```
--- seq=2040 chid=3 idx=1 nbytes=60 ---
  [sch4] HOPPER_DMA_COPY_A/OFFSET_IN_VA            role=CE_src     va=0x10009e00000
                                                     → hMemory=0x5c000096 class=NV01_MEMORY_LOCAL_USER offset=0
  [sch4] HOPPER_DMA_COPY_A/OFFSET_OUT_VA           role=CE_dst     va=0x776e20000000
                                                     → hMemory=0x5c000098 offset=0
  [sch4] HOPPER_DMA_COPY_A/LINE_LENGTH_IN          data=0x10000000
  [sch4] HOPPER_DMA_COPY_A/LAUNCH_DMA              data=0x00000182
                                                     (DATA_TRANSFER_TYPE=NON_PIPELINED | SEMAPHORE_TYPE=NONE | SRC_MEMORY_LAYOUT=PITCH | ...)
  [sch4] HOPPER_DMA_COPY_A/SET_SEMAPHORE_VA        role=semaphore  va=0x20440ff30
                                                     → hMemory=0x5c00001f class=NV01_MEMORY_SYSTEM offset=65328
  [sch4] HOPPER_DMA_COPY_A/SET_SEMAPHORE_PAYLOAD   data=0x00000002
  [sch4] HOPPER_DMA_COPY_A/LAUNCH_DMA              data=0x00000014
                                                     (DATA_TRANSFER_TYPE=NONE | SEMAPHORE_TYPE=RELEASE_SEMAPHORE_WITH_TIMESTAMP | ...)
```

That is one whole `cudaMemcpy` of 256 MiB: `0x10000000` bytes from a device
allocation to a host one, then a semaphore release.  Note the **two**
`LAUNCH_DMA`s — the first moves the data and carries `SEMAPHORE_TYPE=NONE`, the
second moves nothing (`DATA_TRANSFER_TYPE=NONE`) and exists only to release the
semaphore.  A compute submission on the same capture looks like this instead:

```
  [sch1] HOPPER_COMPUTE_A/SET_MME_SHADOW_SCRATCH_0        data=0x00000000
  [sch1] HOPPER_COMPUTE_A/SET_MME_SHADOW_SCRATCH_1        data=0x00030000
  [sch1] HOPPER_COMPUTE_A/SET_FALCON04                    data=0x0017e2ac
  [sch1] HOPPER_COMPUTE_A/SET_REPORT_SEMAPHORE_A          data=0x00000002
  [sch1] HOPPER_COMPUTE_A/SET_REPORT_SEMAPHORE_B          data=0x0440fff0
  [sch1] HOPPER_COMPUTE_A/SET_REPORT_SEMAPHORE_C          data=0x00000053
  [sch1] HOPPER_COMPUTE_A/SET_REPORT_SEMAPHORE_D          data=0x00000000
```

- **`[schN]`** is the subchannel tag.  Under CUDA convention:
  `sch1` → `HOPPER_COMPUTE_A`, `sch4` → `HOPPER_DMA_COPY_A`, plus
  `sch0` graphics / `sch2` I2M / etc.  libcuda binds these at
  channel init (before our watchpoint arms), so the decoder
  assumes the defaults from `_SUBCH_DEFAULTS` in
  `address_atlas.py`.  A `SET_OBJECT` in the stream updates the
  binding dynamically.
- **`<CLASS>/<METHOD>`** — class and method resolved against the
  auto-scanned catalog in `class_catalog.py`, which walks every
  `src/common/sdk/nvidia/inc/class/*.h` header at import time to
  build `CLASS_NAMES` (class-id → name) and `METHOD_NAMES`
  ((class-id, method-addr) → name).  Inheritance chains mean a
  Hopper method like `NVC8B5/LINE_COUNT` resolves via ancestor
  `NVC3B5/LINE_COUNT` if the modern header doesn't re-declare it.
- **`<CLASS>::method_0xNNNN`** — fallback format for methods
  whose names aren't in any open class header.  In practice this
  is rare after the inheritance-chain + indexed-macro logic; on
  a 4 MiB CUDA round-trip the count is 0.
- Each `OFFSET_*_VA` and `SET_SEMAPHORE_VA` is resolved against
  `atlas.gpu_ranges` so you get `hMemory` + `class` + byte-offset
  for free.
- `LAUNCH_DMA` flag dwords are decoded into human names by the
  `_annotate()` helper in `decode.py`.

---

## Reading `atlas.json`

Top-level keys:

| key | what's in it |
|-----|--------------|
| `allocations` | hMemory → `{class, class_name, body, ts_alloc, ts_free}`. Every RM alloc seen via ftrace. |
| `gpu_ranges` | `[{base, end, length, hMemory, ts_mapped, ts_unmapped}]` — one entry per UVM MAP_EXTERNAL. |
| `cpu_ranges` | same shape, from pbcap's mmap hook (glibc-intercepted only; direct syscalls are missed). |
| `pb_events` | one per doorbell with resolved hMemory. |
| `pb_bytes_events` | one per unique submission with decoded methods. |
| `doorbells` | pbcap's userspace-view of doorbells (seq, bar0, token). |

Useful `jq` queries:

```bash
# Which allocations does libcuda make? (class + size)
jq -r '.allocations | to_entries[] |
       "\(.value.class_name) sz=\(.value.body.size // "?")"' \
       /tmp/trace-*/atlas.json | sort | uniq -c | sort -rn

# Where does every pb_va land?
jq -r '.pb_events[] | .resolved |
       "\(.class) hMem=\(.hMemory)"' /tmp/trace-*/atlas.json |
       sort | uniq -c | sort -rn

# What method types does CUDA issue?
jq -r '.pb_bytes_events[].methods[].method' /tmp/trace-*/atlas.json |
       sort | uniq -c | sort -rn | head -10

# All OFFSET_IN VAs (source of each copy)
jq -r '.pb_bytes_events[].methods[] |
       select(.method == "OFFSET_IN_VA") |
       "\(.va) → \(.resolved.hMemory)"' /tmp/trace-*/atlas.json
```

---

## Reading a function-section report

`methods.txt` and `atlas.json` answer "what did the program do
end-to-end?".  When you want to ask *"what does this individual
libcuda call do?"* — e.g. how does `cudaHostRegister` differ from
`cudaInitDevice` in driver activity? — use `trace_section.py`.

### How it works

`pbcap.c` brackets each hooked CUDA Runtime call with two NDJSON
events: `<fn>.enter` (carrying the args) and `<fn>.exit` (carrying
the return code).  `timeline_merge.py` forwards them verbatim into
`merged.ndjson`, where they sit on the same `CLOCK_MONOTONIC` axis
as every ftrace, strace, and pbcap event in the run.

`trace_section.py` opens `merged.ndjson`, finds the Nth
`<fn>.enter` / `<fn>.exit` pair, and prints every event in the
window in chronological order with a `+<offset>µs` marker, plus a
trailing kind-count summary.

### Which functions are hooked

`pbcap.c` hooks:

- `cudaInitDevice`
- `cudaHostRegister`, `cudaHostUnregister`
- `cudaLaunchKernel`
- `cudaMemcpy`
- `cudaMalloc`, `cudaHostAlloc`

The first five — everything except `cudaMalloc` and `cudaHostAlloc` — bracket
their call with `<fn>.enter` + `<fn>.exit`, which is what
`trace_section.py` windows on.  `cudaMalloc` and `cudaHostAlloc` emit a single
post-call event instead, so they cannot be sliced — asking for them warns and
skips.

To trace a function not on this list, add a hook in `pbcap.c`
following the pattern at the end of the file (declare a
`real_<fn>` pointer in the table of `real_*` declarations, then append
a hook body emitting `<fn>.enter` + `<fn>.exit` around the
forwarded call).  Re-run `make libpbcap` and re-capture.

### Usage

```bash
python3 reverse/tools/trace_section.py \
    --trace reverse/traces/cuda_host_register \
    --section cudaHostRegister
```

Optional flags:

- `--section a,b,c` — comma-separated list renders each section's
  window back-to-back.  Sections that don't appear in the trace warn on
  stderr and skip; the tool only errors out when *no* section matches.
- `--occurrence N` — when a function is called more than once, pick
  the Nth top-level call (default 0 = first).  Nesting is depth-tracked
  so an inner hooked call doesn't break occurrence counting.  Applies
  to every listed section.
- `--unfold` — disable the `(strace ioctl + ftrace ioctl + ftrace
  uvm_ioctl)` fold and print every event verbatim.  Useful when a
  folded line looks suspicious or you want to inspect timing between
  the three sources.
- `--out <path>` — write to a file instead of stdout.

### Sample output

By default each ioctl is rendered as one logical event even though
ftrace + strace each see it independently and `ftrace uvm_ioctl`
provides decoded body params.  The fold groups them by `(cmd_name,
fd)` within a 100 µs window.  UVM body params are decoded inline,
and handle references are resolved against `atlas.json` when
present.  Pass `--unfold` for the raw stream-per-event view.

```
=== section: cudaHostRegister ===
  pid:        32269
  duration:   3713.177 µs (3713177 ns)
  events:     10  (excluding the enter/exit pair, folded)
  args:       {"ptr":"0x73c4699ff010","size":4194304,"flags":"0x0"}
  returned:   {"ret":0}

  ── timeline ──
  +     0.000 µs  pbcap cudaHostRegister.enter flags=0x0 ptr=0x73c4699ff010 size=4194304
  +   188.641 µs  ioctl NV_ESC_RM_ALLOC_MEMORY [strace+ftrace] fd=13 path=/dev/nvidia0 sz=56 ret=0
  +  2304.481 µs  ioctl UVM_CREATE_EXTERNAL_RANGE [strace+uvm+ftrace] fd=10 path=/dev/nvidia-uvm ret=0  {base=0x73c4699ff000 length=0x1000}
  +  2475.489 µs  ioctl UVM_MAP_EXTERNAL_ALLOCATION [strace+uvm+ftrace] fd=10 path=/dev/nvidia-uvm ret=0  {base=0x73c4699ff000 length=0x1000 offset=0x0 uuid=…}
  +  2537.551 µs  ftrace map_uvm handle=0x5c000096 length=4096 offset=0
  +  2749.409 µs  ioctl UVM_CREATE_EXTERNAL_RANGE [strace+uvm+ftrace] fd=10 path=/dev/nvidia-uvm ret=0  {base=0x73c469a00000 length=0x3ff000}
  +  2903.521 µs  ioctl UVM_MAP_EXTERNAL_ALLOCATION [strace+uvm+ftrace] fd=10 path=/dev/nvidia-uvm ret=0  {base=0x73c469a00000 length=0x3ff000 offset=0x1000 uuid=…}
  +  2962.551 µs  ftrace map_uvm handle=0x5c000096 length=4190208 offset=4096
  +  3195.361 µs  ioctl UVM_CREATE_EXTERNAL_RANGE [strace+uvm+ftrace] fd=10 path=/dev/nvidia-uvm ret=0  {base=0x73c469dff000 length=0x1000}
  +  3339.489 µs  ioctl UVM_MAP_EXTERNAL_ALLOCATION [strace+uvm+ftrace] fd=10 path=/dev/nvidia-uvm ret=0  {base=0x73c469dff000 length=0x1000 offset=0x400000 uuid=…}
  +  3395.551 µs  ftrace map_uvm handle=0x5c000096 length=4096 offset=4194304
  +  3713.177 µs  pbcap cudaHostRegister.exit ret=0

  ── kind summary ──
        3  ftrace map_uvm
        3  ioctl UVM_CREATE_EXTERNAL_RANGE
        3  ioctl UVM_MAP_EXTERNAL_ALLOCATION
        1  ioctl NV_ESC_RM_ALLOC_MEMORY
```

The `[strace+uvm+ftrace]` provenance tag on each folded line shows
which sources the row was synthesized from (and is a useful hint when
debugging a missing-component case via `--unfold`).

What this particular section reveals about `cudaHostRegister`:

- 1× `NV_ESC_RM_ALLOC_MEMORY` (legacy escape `0x27`, *not* the modern
  `NV_ESC_RM_ALLOC` `0x2b`) on `/dev/nvidia0` — registers the
  pre-existing CPU buffer in place rather than minting fresh sysmem.
- 3× `UVM_CREATE_EXTERNAL_RANGE` + `UVM_MAP_EXTERNAL_ALLOCATION` slicing
  the buffer into head + body + tail (4 KiB + 4 190 208 B + 4 KiB =
  exactly 4 MiB).  The 3-way split is libcuda's response to a
  page-unaligned input pointer.
- Zero GPU submissions inside the window — `cudaHostRegister` is pure
  driver setup; `mc1 dbell/*` and `pb/*` counts in the full trace are unchanged
  from `cudaInitDevice` alone.

`cudaHostUnregister` produces the matching teardown (3× `UVM_FREE` +
1× `NV_ESC_RM_FREE handle=0x5c000096`).  Symmetric in shape, ~12
events, ~2 ms.

### When to reach for this vs. methods.txt vs. atlas.json

| Question | Tool |
|----------|------|
| "What submission did the GPU just receive?" | `methods.txt` |
| "Which allocations and VA ranges exist in this run?" | `atlas.json` |
| "What did *this specific libcuda call* do?" | `trace_section.py` |
| "How does mc's setup differ from libcuda's?" | `strace_diff.py` |

`trace_section.py` complements rather than replaces the others — it's
a per-call slicer over the same `merged.ndjson` they're all built on.

---

## Doing it by hand

The script is a wrapper around four steps. Understanding them lets
you deviate (trace different syscalls, longer timeout, re-analyze an
old capture).

### 1. Pre-flight

```bash
sudo systemctl stop lightdm ollama 2>/dev/null     # free DR0..DR3
sudo bash -c 'echo mono > /sys/kernel/debug/tracing/trace_clock'
sudo bash -c 'echo > /sys/kernel/debug/tracing/trace'
sudo dmesg > /tmp/dmesg.pre
```

`trace_clock=mono` is mandatory: the Python merger aligns pbcap and
strace timestamps against the boot-relative monotonic clock. With the
default `local` clock the merged timeline will be off by seconds.

### 2. Capture

```bash
mkdir -p /tmp/pbcap && chmod 777 /tmp/pbcap
sudo LD_PRELOAD=$PWD/reverse/lib/libpbcap.so \
     PBCAP_DIR=/tmp/pbcap \
  strace -ttt -T -f -o /tmp/strace.log \
    -e trace=ioctl,mmap,munmap,openat,close \
    ./reverse/bin/cuda_reference --size 4M
sudo cat /sys/kernel/debug/tracing/trace > /tmp/ftrace.txt
sudo dmesg > /tmp/dmesg.post
```

Notes:

- `strace -f` is required to follow libcuda worker threads that do
  their own mmaps. Without `-f` you miss the pushbuffer-pool mmap.
- `-ttt` prints absolute CLOCK_REALTIME microseconds; the Python
  merger converts those to CLOCK_MONOTONIC via the anchor pbcap
  writes at `pbcap.init`.
- Add / remove syscalls in `-e trace=...` as needed. `ioctl` alone
  catches every driver call; `mmap`/`munmap` catches memory
  layout. Watch out for `getpid` — a busy CUDA run does thousands.
- `sudo cat /sys/kernel/debug/tracing/trace` dumps the whole buffer.
  For very long runs use `trace_pipe` and process streaming output
  (not covered here — `trace` is simpler for < 10 s runs).

### 3. Merge + decode

```bash
python3 reverse/tools/timeline_merge.py \
    --pbcap-ndjson /tmp/pbcap/timeline.ndjson \
    --ftrace /tmp/ftrace.txt --strace /tmp/strace.log \
    --out /tmp/merged.ndjson

python3 reverse/tools/address_atlas.py \
    --merged /tmp/merged.ndjson --pbcap-dir /tmp/pbcap \
    --atlas-out /tmp/atlas.json --methods-out /tmp/methods.txt
```

Both scripts are idempotent — you can re-run them on a saved capture
without re-capturing.

### 4. Diagnose + compare

```bash
# Classify unresolved pb_va values:
python3 reverse/tools/locate_pushbuffer.py \
    --atlas /tmp/atlas.json --pbcap-dir /tmp/pbcap

# Diff two captures (e.g., mc_demo vs CUDA):
python3 reverse/tools/strace_diff.py \
    --input-format ftrace \
    /tmp/ftrace.cuda.txt /tmp/ftrace.mc_demo.txt --summary

# Slice the timeline to one libcuda call (see "Reading a function-section
# report" above):
python3 reverse/tools/trace_section.py \
    --trace /tmp/trace-cuda_host_register-* \
    --section cudaHostRegister
```

---

## Tool reference

### The kernel side

The instrumentation is `MC_TRACE(category, "event", "k=%u …", …)` from
`src/common/sdk/nvidia/inc/mc-trace.h`.  Nineteen files across the RM
core, the kernel-open modules and nvidia-modeset carry call sites; the
record grammar, the category set and the complete event catalogue are
in `reference/trace-format.md`.

| Where | What it reports |
|-------|-----------------|
| `src/nvidia/arch/nvalloc/unix/src/escape.c` | `rm/ioctl`, `rm/alloc`, `rm/control`, `rm/free`, `rm/map_memory`, `rm/update_mapping_info`, plus the `copy_from_user`-safe `body/alloc_hdr` + `body/alloc_row` dump of the NVOS64 allocation parameters (512-byte cap) |
| `src/nvidia/src/kernel/vgpu/rpc.c` | `gsp/rpc_tx` from `_issueRpcAndWait` and `_issueRpcAsync` — transmit side only, there is no `gsp/rpc_rx`.  `_rpcFunctionName()` decodes the function number via an X-macro re-include of `rpc_global_enums.h`, so the name table cannot go stale |
| `src/nvidia/src/kernel/gpu/fifo/` | `fifo/chan_construct`, `fifo/userd_resolve`, `fifo/userd_rpc`, `fifo/chan_get_engine` |
| `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c` | `fifo/verify_channel`, `fifo/retain_channel`, `fifo/retain_channel_resources`, `fifo/channel_engine_type`, `fifo/userd_bind`, and the RM half of `dbell/resolve` + `dbell/gpfifo_*` |
| `src/nvidia/src/kernel/{gpu/,}mem_mgr/` | the `mmu/*` records — PTE sources, inter-map calls, virtual-memory backing, BAR1 reflection |
| `src/nvidia/src/libraries/resserv/src/rs_server.c` | `rm/inter_map`, the resserv handle-validation path |
| `src/nvidia/src/kernel/gpu/mem_mgr/context_dma.c` | `rm/ctxdma_construct` |
| `src/common/unix/nvidia-push/src/nvidia-push.c` | `fifo/gp_entry_write` — this compiles into nvidia-modeset.ko, and does not fire for CUDA |
| `kernel-open/nvidia/nv-mmap.c` | `mmap/any`, `mmap/bar0`, `mmap/ctl_peer` |
| `kernel-open/nvidia/nv-doorbell-watch.c` | the whole `dbell/*` control plane plus `pb/submit` and `pb/bytes` |
| `kernel-open/nvidia-uvm/` | `uvm/*`, `mmap/uvm`, `mmu/pte_hdr`, `pte/row`, `body/uvm_row` |
| `kernel-open/nvidia/os-interface.c` | the transport: `nv_trace_printf` (a `ftrace_vprintk` wrapper — `trace_printk` is a variadic macro and cannot take a `va_list`), `nv_trace_next_id`, and the `nv_trace_mask` global, all `EXPORT_SYMBOL`ed so nvidia-uvm.ko and nvidia-modeset.ko can emit too |

Output goes to `/sys/kernel/debug/tracing/trace_pipe` (live) or
`/sys/kernel/debug/tracing/trace` (snapshot).  **Not** to dmesg.

Every site is gated on `nv_trace_mask & MC_TRACE_CAT_<category>`.  The mask
is the writable `nvidia.ko` module parameter `mc_trace`, default
`MC_TRACE_CAT_DEFAULT` = everything except `pte`:

```bash
cat /sys/module/nvidia/parameters/mc_trace
echo 0x3ff | sudo tee /sys/module/nvidia/parameters/mc_trace   # add pte
echo 0x1df | sudo tee /sys/module/nvidia/parameters/mc_trace   # drop pb, to
                                                               # measure trap cost
```

### The userspace tools

| Tool | Purpose |
|------|---------|
| `trace_cuda.sh` | one-shot capture + analyze for any executable |
| `timeline_merge.py` | merge ftrace + strace + pbcap NDJSON on CLOCK_MONOTONIC |
| `address_atlas.py` | build hMemory/VA tables + decode pb_bytes methods; also builds `Atlas.channels`, `Atlas.carriers`, `Atlas.unattributed_intermaps` for the non-UVM channel ledger |
| `non_uvm_ledger.py` | consume `merged.ndjson` and emit a per-channel non-UVM ledger (Markdown / JSON / summary).  Imports `address_atlas.py` as a library rather than re-parsing NDJSON |
| `trace_section.py` | window `merged.ndjson` on a `<fn>.enter`/`<fn>.exit` bracket pair and print every event inside the call (per-libcuda-function tracing) |
| `strace_diff.py` | compare / summarize / filter one or two traces (accepts both ftrace and strace).  Owns the `mc1` record parser |
| `phase_census.py` | per-phase event counts straight out of `ftrace.txt`; has its own `mc1` regex and does not go through `strace_diff.py` |
| `locate_pushbuffer.py` | classify pb_va values as uvm_range / sysmem / snap_file / unresolved |
| `class_catalog.py` | auto-extract class IDs + method-address names from `src/common/sdk/nvidia/inc/class/*.h` at import time |
| `find_bar1_pfn.py` | `/proc/PID/pagemap` scanner that reads only the VMA ranges listed in `/proc/PID/maps` — pagemap is sparse over the full 48-bit VA space, so reading it whole allocates terabytes |
| `decode.py encode` | generate the expected NVC8B5 method stream from src VA, dst VA, size and semaphore VA |
| `decode.py decode <bin>` | offline NVC8B5 disassembly of any byte blob |
| `decode.py diff <pre.bin> <post.bin>` | byte-level diff of two pbcap snapshots, decoding the changed regions |
| external: `nsys profile` | reference timeline for cross-checking (hides driver internals) |
| external: `cat /sys/kernel/debug/tracing/trace` | raw ftrace when the script's filters drop something you wanted |

`decode.py` predates the discovery that `SET_OBJECT` + subchannel 4 is
required, so its `encode` output is the 14-dword / 56-byte variant; the
canonical form mc emits is 18 dwords / 72 bytes including the `SET_OBJECT`
prefix and the split launches.  `address_atlas.py` is the current decode
path; `decode.py` is kept for one-off blobs.

### `pbcap.c`, and two things it took a while to get right

`reverse/tools/pbcap.c` is an `LD_PRELOAD` shim.  It hooks `open`/`openat`
to track nvidia device fds, `mmap`/`munmap` to track nvidia-backed
mappings, and a growing set of CUDA Runtime entry points to bracket each
call with `<fn>.enter` / `<fn>.exit` NDJSON events.

- `libcuda.so` opens `/dev/nvidia0` through raw `syscall()`, bypassing the
  glibc PLT, so the `open()` hook never sees it.  The fix is to scan
  `/proc/self/maps` at snapshot time and match on the `/dev/nvidia*`
  pathname.  Write-only (`-w-s`) channel mappings need a temporary
  `mprotect(PROT_READ|PROT_WRITE)` to be readable.
- The pushbuffer is mapped write-combine, which does **not** stop a CPU
  read.  Measured against the `#DB` handler's own read of the same
  addresses, a userspace read returned byte-identical data for 112 of 112
  submissions — zero all-zero, zero mismatches.  The BAR1 channel region
  reads back real GP entries too.  What genuinely reads as zero is the BAR0
  VF doorbell, which has no backing store (`findings.md §12.1`).  The
  snapshot path does need `PBCAP_MAX_BYTES` raised: the pushbuffer pool is
  56 MiB against a 16 MiB default per-mapping cap, so at the default the
  pool is skipped entirely.

### The records that reach `merged.ndjson`

`strace_diff.py` consumes 23 of the 59 emitted events.  Eight of those are
the permanent non-doorbell observability records — routed
`strace_diff.parse_ftrace` → `timeline_merge._ftrace_info_to_dict` →
`merged.ndjson`, and read from there by `address_atlas.py` and
`non_uvm_ledger.py`.  The field names below are the wire keys.

| Event | Emitted from | Fields |
|-------|--------------|--------|
| `mmu/pte_src_decision` | `dmaAllocMapping_GM107` (`virt_mem_allocator_gm107.c`) | `carrier_h carrier_class carrier_aspace src_h src_class src_aspace src_pte0 chosen_pte0` |
| `mmu/intermap_call` | `virtmemMapTo_IMPL` (`virtual_mem.c`) | `hclient hcarrier carrier_class hsrc src_class flags dma_offset length` |
| `mmu/virtmem_backing` | `virtmemConstruct_IMPL` + `vmrangeConstruct_IMPL` | `hmemory class va_size hvaspace aspace has_heap via` |
| `mmu/gmmu_pte_phys` | `dmaAllocMapping_GM107` | `va_lo va_hi pte0_phys pte_count page_size aperture` |
| `mmu/bar1_reflect_phys` | `memdescMap` (`mem_desc.c`) | `at_gpu fb_aperture_off bar1_phys size` |
| `fifo/userd_resolve` | `kchannelCreateUserdMemDesc_GV100` | `hclient huserd userd_offset userd_addr address_space userd_size` |
| `fifo/userd_rpc` | `_kchannelSendChannelAllocRpc` (`kernel_channel.c`) | `hchannel base size address_space cache_attrib` |
| `fifo/userd_bind` | `nvGpuOpsBindChannelResources` (`nv_gpu_ops.c`) | `retained resource_count` |

The first three were the instrumentation that mattered for the FB-carrier
diagnosis (`findings.md §14`); the rest are broader-purpose
channel-construction visibility.  All eight are in maskable categories
(`mmu`, `fifo`) — both on by default, both silenceable through `mc_trace`.

The other fifteen consumed events are `rm/ioctl`, `rm/control`, `rm/alloc`,
`rm/free`, `rm/map_memory`, `uvm/ioctl`, `uvm/map_external`,
`body/alloc_hdr`, `body/alloc_row`, `body/uvm_row`, `pb/submit`,
`pb/bytes`, `pb/bytes_miss`, `dbell/sysmem_track` and `dbell/bar1_track`.
The remaining 36 events are emitted and parse cleanly but have no
consumer; `reference/trace-format.md` marks which is which.

### The non-UVM channel ledger

`non_uvm_ledger.py` is the consumer built on top of the `mmu/*` records
above.  It reads `merged.ndjson` and emits a per-channel report of how a
channel's USERD and GPFIFO got their GPU VAs when they are *not* UVM-mapped
— which is the shape libcuda uses.  `address_atlas.py` grew three
structures for it: `Atlas.channels` (keyed on channel `hMemory`),
`Atlas.carriers` (keyed on `NV50_MEMORY_VIRTUAL` carrier `hMemory`), and
`Atlas.unattributed_intermaps` (NVOS46s that did not bind to a known
channel).

Two strategies join a carrier to a channel:

1. `hSrc == channel.hUserdMemory` — `NV_CHANNEL_ALLOC_PARAMS` says
   literally which `hMemory` is USERD.
2. `carrier.hVASpace == channel.hVASpace` — only meaningful when the
   channel has a dedicated VA space.

The ledger is what surfaced libcuda's per-resource carrier shape and
provided the before/after comparison for mc's per-resource refactor
(`findings.md §14.3`).

---

## Gotchas

- **DR0–DR3 exhaustion.** If `lsmod` shows `nvidia_dbell` loaded but
  the trace has 0 `mc1 dbell/*` records, HW breakpoint slots are
  taken. `sudo systemctl stop lightdm ollama` frees them; if a third
  process is claiming them, `cat /sys/kernel/debug/tracing/available_events
  | grep hw_breakpoint` and `perf list hw` can help identify it.
- **Stale kernel module.** If `uname -r` got upgraded but you haven't
  rebuilt, the loaded `nvidia.ko` might not have the sysmem tracker
  (or worse, might be an older kernel's ABI). Rebuild:
  `SYSSRC=/lib/modules/$(uname -r)/build make -j$(nproc) modules`
  then `sudo cp kernel-open/*.ko /lib/modules/$(uname -r)/kernel/drivers/video/`
  + `sudo depmod -a` + reboot.
- **Missing `strace -f`.** libcuda spawns worker threads; without `-f`
  the timeline merger can't correlate their mmaps and `pb_bytes`
  lookups fail.
- **Tracing overhead.** The `#DB` handler fires once per real GPU
  doorbell, reads ~64–4096 bytes per submission, emits chunked
  ftrace lines. We measured ~0% impact on PCIe bandwidth on H100
  — good for bench runs, but if you're profiling latency the
  microsecond-scale trap overhead shows up. Compare with + without
  the watchpoint to quantify.
- **Non-CUDA executables** work if they submit via the same RM/UVM
  path.  Since the 2026-05-12 VA-pool fix (`findings.md §13`),
  `mc_demo` produces full tooling parity with libcuda: a
  `trace_cuda.sh` capture of a 128 MiB H2D-seed-then-D2H round-trip
  emits `mc1 pb/submit` and `mc1 pb/bytes` for every
  submission (2 of each per round-trip, 0 `mc1 pb/bytes_miss`),
  with the method-stream decode matching libcuda's shape exactly
  (see `findings.md §13.4` for the captured example).

---

## See also

- `docs/findings.md §12` — kernel doorbell-watchpoint design, token
  decode, the trap-context invariants.
- `docs/findings.md §14` — non-UVM channel carrier investigation
  (FB-carrier resolution, per-resource refactor, refuted hypotheses).
- `docs/reference/trace-format.md` — the `mc1` record grammar, the
  category mask, and the full 59-event catalogue.
- `docs/gpfifo_pushbuffer_reference.md` — bit-exact GPFIFO entry
  layout and NVC8B5 method encoding referenced by the decoder.
- `docs/mc_architecture.md` — the raw-ioctl `mc` path that's
  being compared against libcuda throughout this work.
