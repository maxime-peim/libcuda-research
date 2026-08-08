# The `mc1` trace record format

Every line the instrumented driver emits into the ftrace buffer is one record in a
single, versioned, self-describing format. This document is the authority: the kernel
macro produces it, the userspace parser consumes it, and nothing else is valid.

## The grammar

```
record  := "mc1" SP event (SP field)* "\n"
event   := category "/" name
category := [a-z][a-z0-9_]*          one of the declared set below
name     := [a-z][a-z0-9_]*
field    := key "=" value
key     := [a-z][a-z0-9_]*
value   := bare | quoted | array
bare    := [^ "\[\]]+                 hex (0x…), decimal, identifiers, pointers
quoted  := '"' ( [^"\\] | '\' . )* '"'   anything with spaces, or free prose
array   := "[" (bare ("," bare)*)? "]"   at most one per record, and it must be last
```

### Invariants, and what breaks if one is violated

There is no linter. One of these is enforced by the compiler; the rest are
conventions kept by review, so it is worth writing down the failure mode of each.

| Invariant | Enforcement | Failure mode |
|---|---|---|
| the category is one of the declared ten | compiler | `MC_TRACE(foo, …)` expands to `MC_TRACE_CAT_foo`, which does not exist — build error |
| exactly one `\n`, and it is the last byte | `MC_TRACE` adds it | a record with an embedded `\n` becomes two ftrace lines; the second has no `mc1` sigil and every parser drops it |
| the record fits the kernel's `trace_printk` line buffer (~1 KiB) | convention | `ftrace_vprintk` truncates; the tail of the record is lost silently, and a truncated numeric value parses as a wrong number rather than an error. This is why `nv-doorbell-watch.c` emits `pb/bytes` in 64-byte (128 hex character) chunks instead of one line per submission |
| the record is ASCII | convention | `_mc1_fields` decodes quoted values through `unicode_escape`; non-ASCII bytes come back mangled |
| no duplicate keys | convention | `_mc1_fields` keeps the last occurrence and silently discards the earlier one |
| a bare value contains no unescaped space | convention | both parsers scan for `key=value` pairs rather than splitting the field string, so the offending value is silently truncated at the space while every field after it still parses — a wrong number rather than a visible error |
| at most one array field, and it is the last field | convention | tolerated by the current parser, but the array helpers (`MC_ARR4`, `MC_ARR8`) are written for a trailing array and nothing tests any other position |

### Why it is shaped this way

- **The version is the sigil `mc1`.** One greppable token that also anchors
  run-together recovery. Adding events, categories or keys is *not* a version bump —
  the grammar is self-describing, so the parser does not care. `mc2` is reserved for a
  grammar change (a new delimiter or structure), which tooling detects on line 1.
- **The category is the first half of the event name**, not a separate field. One
  token is the name, the category, and the mask selector at once, so the three can
  never disagree — and there is no syntactic room for a positional sub-verb or a
  multi-word name, so a whole class of naming drift is unrepresentable rather than
  merely discouraged.
- **Keys are looked up by name**, so the kernel may reorder or add fields without
  breaking any parser — field *order* is deliberately not part of the contract.
- **A value never contains an unescaped space** — it is bare, quoted, or a bracketed
  array — so splitting on whitespace is always correct.

## The categories

Selected at runtime by `nv_trace_mask`, exposed as the writable module parameter
`mc_trace` on `nvidia.ko` (`kernel-open/nvidia/os-interface.c`). Every `MC_TRACE` site
is gated on `nv_trace_mask & MC_TRACE_CAT_<cat>`; there are no unconditional sites.
The default is `MC_TRACE_CAT_DEFAULT` — everything except `pte`.

```bash
# read / change the mask at runtime (bit 9 = pte)
cat /sys/module/nvidia/parameters/mc_trace
echo 0x3ff | sudo tee /sys/module/nvidia/parameters/mc_trace   # everything on
echo 0x030 | sudo tee /sys/module/nvidia/parameters/mc_trace   # dbell + pb only
```

| bit | category | owns | default |
|---|---|---|---|
| 0 | `rm` | RM ioctl escape, resource server, context-DMA | on |
| 1 | `uvm` | the UVM driver | on |
| 2 | `fifo` | channel construct/retain, USERD, GPFIFO, the pushbuffer producer | on |
| 3 | `mmu` | GMMU, virtual memory, memory descriptors (one record per mapping) | on |
| 4 | `dbell` | the doorbell watchpoint control plane | on |
| 5 | `pb` | the pushbuffer hot path (`pb/submit`, `pb/bytes`) — split out so it can be silenced to measure trap overhead | on |
| 6 | `gsp` | the GSP RPC boundary | on |
| 7 | `mmap` | `nv-mmap.c` and the UVM mmap path | on |
| 8 | `body` | allocation-parameter dump rows — **do not mask this off**, see below | on |
| 9 | `pte` | `pte/row`, one record per eight PTEs — no consumer today | **off** |

Two of these are non-obvious and are called out so nobody "optimises" them away:

- **`body` looks like bulk noise and is not.** The `body/alloc_row` records carry the
  NVOS32/NVOS64 allocation parameters that `address_atlas.py` decodes; nothing else in
  the capture carries them. Masking `body` off silently empties the atlas. It is also
  the highest-volume category in a default capture — 6,676 of the 10,589 records in
  `reverse/traces/cuda_reference/ftrace.txt`, being 3,712 `body/alloc_row`, 2,848
  `body/uvm_row` and 116 `body/alloc_hdr` — which is what makes it a tempting thing
  to switch off.
- **`pb` is split from `dbell`** on purpose: silencing the pushbuffer firehose while
  keeping the rest of the doorbell control plane is how the trap handler's own
  overhead is measured.

`pte` is off by default because it is one record per eight PTEs over every mapping,
which dominates a capture. The checked-in capture at `reverse/traces/cuda_reference/`
was taken at the default mask and therefore contains zero `pte/row` records; there is
no measured share-of-capture figure for it in this tree.

## Records are an event stream, not a log

One logical operation gets **one event name**, and a `step=`, `result=` or `state=`
field distinguishes its phases. A function that reports progress from seven places
emits seven records that all share a name — `nvGpuOpsVerifyChannel` is
`fifo/verify_channel result=…`, `RetainChannel` is `fifo/retain_channel step=…`, the
doorbell resolver is `dbell/resolve state=…`, and channel construction reports its
fifteen breadcrumbs as `fifo/chan_construct step=…`.

Which discriminant key a given event uses is per-event and not derivable from the
name. `rm/inter_map` is the one event that uses both in the same event, `step=` for
progress and `result=` for outcomes; `fifo/retain_channel` looks similar but uses
`step=` throughout, including for its terminal `step=success`. The catalogue below
lists the actual key for each.

The emission points stay where they are; the records are not physically merged into
one call, which would mean restructuring returns in the RM core and inside the `#DB`
trap handler — not worth the risk for diagnostic output. The gain is that a capture is
queried by event class rather than by hunting for spellings.

The allocation-parameter dumps carry a kernel-emitted correlation `id=`
(`nv_trace_next_id()`, one `atomic_inc_return` per allocation, stamped on the header
and every body row), so the parser keeps no state — records are order-independent and
individually meaningful even in a truncated or interleaved capture.

---

## The event catalogue

Every event the driver emits, by category: 59 events across 135 `MC_TRACE` /
`MC_TRACE0` sites. The field list for each row is the union of what its sites emit;
where the sites differ, the row says which fields go with which discriminant value,
because assuming a field is always present is the common way to write a parser that
throws on a real capture.

`strace_diff.py` consumes 23 of the 59 — twelve through its `MC1_SIMPLE` table and
eleven through explicit handlers. The remaining 36 are emitted, parse cleanly per the
grammar, and are read by nothing but `grep`, `phase_census.py` and eyes. That is the
normal state, not a backlog: an event exists so a capture can answer a question, and
most questions are answered by reading the capture. The **consumed** column below
marks the 23.

`ret` is the universal suffix for the return/exit side of a call. `step=`, `result=`
and `state=` distinguish the phases of one operation that reports from several places.

### `rm` — resource manager escape path

| event | fields | consumed | notes |
|---|---|---|---|
| `rm/ioctl` | `cmd esc size` | yes | one per RM escape; `esc` is the `NV_ESC_*` code. Atlas backbone |
| `rm/control` | `cmd hclient hobject params_size` | yes | the CONTROL sub-command of an RM ioctl |
| `rm/alloc` | `id hclass root parent new` | yes | entry side. `id=` is the correlation key the `body/alloc_row` reassembly joins on — without it the atlas has no allocation parameters |
| `rm/alloc` (`result=ret`) | `result hclass status access` | — | exit side; carries no `id` |
| `rm/free` | `root parent old` | yes | entry side (distinguished by the absence of `result=`) |
| `rm/free` (`result=ret`) | `result old status` | — | |
| `rm/update_mapping_info` | `hclient hdevice hmemory old_cpu_addr new_cpu_addr` | — | |
| `rm/update_mapping_info` (`result=ret`) | `result hmemory status` | — | |
| `rm/inter_map` | `step=`/`result=` + `hclient hmapper hmappable hdevice` (enter), `hmapper status` (`result=mapper_fail`), `hmapper internal_class_id` (`step=mapper_ok`), `hmappable status` (`result=mappable_fail`), `hmappable internal_class_id` (`step=mappable_ok`), `hdevice status` (`result=device_fail`), `hdevice` (`result=ok`) | — | reports from seven points in `resserv`'s inter-map handle validation |
| `rm/actual_device_only_fail` | `flags` | — | |
| `rm/ctxdma_construct` | `hclass flags offset limit` | — | |
| `rm/map_memory` | `hclient hdevice hmemory offset length flags fd` | yes | BAR1/UVM double-map correlation; feeds `strace_diff.py`'s `map_bar1` kind (30 records in the checked-in capture) |

### `uvm` — the UVM driver

| event | fields | consumed | notes |
|---|---|---|---|
| `uvm/ioctl` | `id cmd` | yes | the UVM ioctl dispatcher. `cmd` is the UVM command number; `id=` joins the following `body/uvm_row` records |
| `uvm/map_external` | `hclient hmemory base length offset rm_ctrl_fd` | yes | entry side (distinguished by the presence of `base=`) |
| `uvm/map_external` (`result=ret`) | `result hmemory rm_status` | — | |
| `uvm/create_gpu_va_space` | `step=dup_call` + `user_client user_object rm_ctrl_fd`; `step=dup_ret` + `status`; `step=ats` + `ats_enabled uvm_ats_enabled ats_unset ats_supported pageable_mem_access` | — | |
| `uvm/register_gpu_va_space` | `step=enter`; `result=gpu_retain_fail`; `result=mm_retain_null`; `step=before_create`; `step=create_ret` + `status` | — | only the `create_ret` phase carries a field |
| `uvm/gpu_va_space_reject` | `reason` | — | `reason` is an identifier; the only value emitted is `ats_on_pageable_off` |

### `fifo` — channels, USERD, GPFIFO, pushbuffer producer

| event | fields | consumed | notes |
|---|---|---|---|
| `fifo/verify_channel` | `result=` + `hclient hkernel_channel session device vaspace` (`enter`), `status` (`client_fail`, `vaspace_fail`, `channel_fail`), `status hdevice` (`context_fail`), `channel_pvas expected` (`vas_mismatch`), nothing (`ok`) | — | reports from seven points; `result=` is the only discriminant — there is no `step=` here |
| `fifo/retain_channel` | `step=` + `channel_engine_type status` (`engine_type`), `session hchannel_parent btsg_channel hdup_tsg device_handle` (`alloc_retainer`), `status retained_handle` (`alloc_retainer_ret`), `hclient hkernel_channel` (`get_token`), `status token` (`get_token_ret`), nothing (`retain_resources`, `success`), `status` (`retain_resources_ret`) | — | reports from eight points |
| `fifo/retain_channel_resources` | `step=enter` + `channel_engine_type ce gr sec2`; `step=ce_path_done`; `step=not_ce_path` | — | |
| `fifo/channel_engine_type` | `rm_engine_type engine_type` | — | the engine-type workaround's own trace |
| `fifo/userd_resolve` | `hclient huserd userd_offset userd_addr address_space userd_size` | yes | `kchannelCreateUserdMemDesc_GV100` in `kernel_channel_gv100.c` |
| `fifo/userd_rpc` | `hchannel base size address_space cache_attrib` | yes | the USERD descriptor as sent to GSP |
| `fifo/userd_bind` | `retained resource_count` | yes | |
| `fifo/chan_construct` | `step=` + `hclient hparent hnew class` (`enter`), `herror_ctx blegacy chan_count` (`lock_acquired`), `status tsg_engine_type` (`set_legacy_mode`), `tsg_eng_type valid` (`per_runlist_chram`), `engine_type runlist_id tsg_engine_type param_engine_type` (`set_engine`), `chid status` (`before_instmem`), `status` (`after_instmem`, `after_alloc_hal`, `gsp_rpc_done`, `failed`), `is_gsp_client status` (`before_gsp_rpc`), nothing (`before_lock`, `check_engine_type`, `before_alloc_hal`, `before_vaspace_cache`) | — | fifteen sites; resolves the `kchannel`/`channel` near-collision |
| `fifo/chan_get_engine` | `hal runlist_id engine_type` | — | `hal=gm107` — the HAL variant is a field, not part of the name |
| `fifo/gp_entry_write` | `gp_put next_gp_put put_offset` | — | the only site outside `src/nvidia` and `kernel-open`: `src/common/unix/nvidia-push/src/nvidia-push.c`, which **compiles into nvidia-modeset.ko**. It does not fire for CUDA (see `findings.md §1`) |

### `mmu` — GMMU, virtual memory, descriptors

| event | fields | consumed | notes |
|---|---|---|---|
| `mmu/pte_hdr` | `hmem vatype map_off page_size n_ptes` | — | one per external mapping; stays on even though `pte/row` is off |
| `mmu/gmmu_pte_phys` | `va_lo va_hi pte0_phys pte_count page_size aperture` | yes | |
| `mmu/pte_src_decision` | `carrier_h carrier_class carrier_aspace src_h src_class src_aspace src_pte0 chosen_pte0` | yes | |
| `mmu/intermap_call` | `hclient hcarrier carrier_class hsrc src_class flags dma_offset length` | yes | distinct from `rm/inter_map`: this is `virtmemMapTo_IMPL`, that is `resserv` |
| `mmu/virtmem_backing` | `hmemory class va_size hvaspace aspace has_heap via` | yes | `via=` names the constructor — `virtmem_construct` or `vmrange_construct` |
| `mmu/bar1_reflect_phys` | `at_gpu fb_aperture_off bar1_phys size` | yes | |

### `dbell` — doorbell watchpoint control plane

Seventeen events. Most are in `kernel-open/nvidia/nv-doorbell-watch.c`; the
exceptions are `dbell/resolve` (ten sites in
`src/nvidia/src/kernel/rmapi/nv_gpu_ops.c`, one in `rm-gpu-ops.c` — none in
nv-doorbell-watch.c), `dbell/gpfifo_register` (nv_gpu_ops.c), and
`dbell/gpfifo_lookup`, which is split: `result=table` and `result=not_in_table` come
from nv_gpu_ops.c, `result=miss` from `nv_dbell_resolve_fn`, the deferred
`nv_kthread_q` resolver — not from the `#DB` handler itself.

| event | fields | consumed | notes |
|---|---|---|---|
| `dbell/armed` | `slot pid bps user_va bar0_pa` | — | the watchpoint is live on this VMA; `bps` is how many threads got a breakpoint.  `bar0_pa` is named historically: it carries whichever BAR the intercepted window sits on, which for the variant libcuda and mc ring is BAR1 |
| `dbell/slots_full` | `pass_through=1 pid bar0_pa` | — | all four `nv_dbell_ctx_t` slots busy; this mmap gets the real BAR0 page and is not watched |
| `dbell/ioremap_fail` | `bar0_pa` | — | `ioremap` of the real doorbell dword failed; arming aborts |
| `dbell/vm_insert_fail` | `addr ret` | — | `vm_insert_page` of the shadow page failed at this VA; arming aborts |
| `dbell/bp_register` | `tid ret` | — | emitted only on **failure** to install a breakpoint on one thread; the successful case is silent and is counted by `dbell/armed bps=` |
| `dbell/no_breakpoints` | *(none)* | — | zero breakpoints installed, so arming failed. The only `MC_TRACE0` in the tree |
| `dbell/release` | `slot count` | — | VMA teardown. `count` is `ctx->bp_count`, the number of doorbell writes this slot trapped — not a kernel address |
| `dbell/shadow` | `state=nonzero` + `slot off val`; `state=nonzero_truncated`; `state=all_zero` + `slot` | — | forensic dump of the 4 KiB shadow page at release, capped at 32 non-zero dwords. Answers "did userspace write anywhere other than +0x90?" |
| `dbell/bar1_track` | `state=add` + `slot phys size kva`; `state=remove` + `slot kva`; `state=table_full`; `state=ioremap_fail` + `phys size` | yes (`add`/`remove`) | the BAR1 FB-mapping tracker |
| `dbell/sysmem_track` | `state=add` + `slot user_va_start user_va_end num_pages kva`; `state=remove` + `kva num_pages`; `state=table_full`; `state=vmap_fail` + `user_va num_pages` | yes (`add`/`remove`) | the sysmem-mapping tracker. Note `state=add` carries a VA **range**, not a single `user_va` |
| `dbell/resolve` | `state=` + `nv` (`wrap_no_gpu`), `chid runlist pgpu` (`enter`), nothing (`no_fifo`), `runlist` (`no_chid_mgr`), `chid runlist` (`no_channel`), `chid sub` (`no_userd`), `chid pool_kva suboff kva` (`fbmem_pool`), `chid suboff` (`fbmem_no_pool`), `chid phys` (`sysmem`), `chid addrspace suboff` (`unknown`), `chid phys size addrspace` (`done`) | — | eleven sites; `state=` says which |
| `dbell/gpfifo_lookup` | `result=table` + `chid gpu_va entries`; `result=miss` + `chid gpu_va entries`; `result=not_in_table` + `chid runlist` | — | |
| `dbell/gpfifo_register` | `result=table_full chid runlist` | — | emitted only when `g_dbellGpfifoTable` has no free slot; the successful registration is silent |
| `dbell/fire` | `seq slot token chid runlist` | — | one per intercepted doorbell write. `chid`/`runlist` here are decoded from `token` |
| `dbell/gp_put` | `seq chid runlist gp_put` | — | GPPut as read out of USERD for this doorbell |
| `dbell/no_iomap` | `seq` | — | the doorbell fired but the slot has no `doorbell_iomap`; the write cannot be forwarded |
| `dbell/cache` | `state=` + `chid runlist reason` (`invalidate`), `chid runlist userd_kva gpfifo_kva entries` (`resolved`), `chid runlist phys addrspace` (`retry`), `chid runlist status phys addrspace` **or** `seq chid runlist` (`failed`), `chid runlist` (`pre_reserve`), `seq chid runlist` (`reserve`, `pending_init`, `pending_in_flight`) | — | the `g_dbell_cache` resolve-cache state machine; 72 records in the checked-in capture. `state=failed` is emitted from two sites with different field sets |

The two trackers are asymmetric on removal and it is worth knowing before writing a
parser: `dbell/bar1_track state=remove` carries `slot` and `kva`, while
`dbell/sysmem_track state=remove` carries `kva` and `num_pages` and **no** `slot`.
The sysmem remove path is keyed on the kernel VA rather than the slot index, so the
slot is not in scope at the emission point. Joining a sysmem add to its remove
therefore goes through `kva`, not `slot`.


### `pb` and `pte` — the two independently-maskable firehoses

Each is its own category (its own mask bit) precisely so it can be silenced without
touching anything else, which is only possible if the category *is* the first token of
the event name.

| event | fields | consumed | notes |
|---|---|---|---|
| `pb/submit` | `seq chid idx entry0 entry1 pb_va pb_len` | yes | one per GPFIFO entry published |
| `pb/bytes` | `seq chid idx chunk nchunks off hex` | yes | the method-stream bytes, 64 bytes (128 hex chars) per record; the parser reassembles the `nchunks` chunks by `seq` |
| `pb/bytes_miss` | `seq chid idx pb_va pb_len` | yes | the pushbuffer VA is not in the sysmem tracker, so its bytes could not be read |
| `pte/row` | `hmem idx ptes=[…]` | — | **off by default**; eight hex values as one array |

### `gsp`, `mmap`, `body`

| event | fields | consumed | notes |
|---|---|---|---|
| `gsp/rpc_tx` | `mode func name len` | — | `mode` is `sync` or `async`. Transmit side only — there is no `gsp/rpc_rx` |
| `mmap/any` | `pid comm ctl size access_start pgoff` | — | every `nvidia_mmap_helper` call with `vm_pgoff == 0`; `ctl=1` means `/dev/nvidiactl` |
| `mmap/bar0` | `pid comm pa size nr` | — | the register/BAR0 path; `nr` is `memArea.numRanges` |
| `mmap/ctl_peer` | `pid size pages first_phys` | — | the ctl-device peer path; no `comm` |
| `mmap/uvm` | `pid comm vm_start vm_end size vm_pgoff` | — | the UVM mmap path |
| `body/alloc_hdr` | `id size dwords` | yes (skipped) | `strace_diff.py` recognises it and discards it — the rows carry the data |
| `body/alloc_row` | `id off dw=[…]` | yes | four dwords per row, zero-padded to a full row so every record has one shape |
| `body/uvm_row` | `id off dw=[…]` | yes | same shape, joined to `uvm/ioctl` by `id` |

---

## Reading a capture

### The ftrace envelope

`nv_trace_printf` is `ftrace_vprintk`, so records land in the ftrace ring buffer, not
in dmesg. Each line the tracer prints wraps one record:

```
cuda_reference-31038   [014] d....  2929.680905: nv_trace_printf: mc1 pb/submit seq=1846 chid=3 idx=0 entry0=0x03600000 entry1=0x00002202 pb_va=0x203600000 pb_len=32
```

`task-pid`, `[cpu]`, the latency flags (`d....` = interrupts off, which is what a
record emitted from the `#DB` trap handler looks like), the timestamp, the tracer
function name, then the record. `strace_diff.py`'s `FTRACE_LINE_RE` peels exactly that
envelope and hands the `mc1 …` remainder to the record parser. The timestamp is
seconds-since-boot, so `trace_clock` must be `mono` for the merge with strace to line
up.

Read the buffer with `cat /sys/kernel/debug/tracing/trace` (snapshot) or
`trace_pipe` (live). `reverse/tools/trace_cuda.sh` does the setup, the run and the
snapshot in one step.

### One `cudaMemcpy`, end to end

The records a single 4 MiB `cudaMemcpy` produces, in the order they appear:

```
# at mmap time, once per HOPPER_USERMODE_A mmap
mc1 mmap/bar0 pid=31038 comm="cuda_reference" pa=0x6002bb0000 size=0x10000 nr=1
mc1 dbell/armed slot=0 pid=31038 bps=2 user_va=0x7770a2ed9000 bar0_pa=0x6002bb0000

# at BAR1 FB-mapping time, once per /dev/nvidia0 rw-s 2 MiB mmap
mc1 dbell/bar1_track state=add slot=0 phys=0x4021200000 size=0x200000 kva=00000000a432faf2

# at sysmem-mapping time, so the pushbuffer bytes can be read later
mc1 dbell/sysmem_track state=add slot=0 user_va_start=0x200600000 user_va_end=0x203e00000 num_pages=14336 kva=0000000063680cf9

# on the first doorbell for a channel, from the resolver kthread
mc1 dbell/resolve state=enter chid=3 runlist=1 pgpu=000000008ff3a28a
mc1 dbell/resolve state=fbmem_no_pool chid=3 suboff=0x0
mc1 dbell/gpfifo_lookup result=table chid=3 gpu_va=0x121010000 entries=1024
mc1 dbell/cache state=resolved chid=2 runlist=0 userd_kva=0x000000006ac3c7fb gpfifo_kva=0x00000000a432faf2 entries=1024

# per doorbell once the cache is warm, all from inside the #DB handler
mc1 dbell/fire seq=1846 slot=1 token=0x000a0003 chid=3 runlist=10
mc1 dbell/gp_put seq=1846 chid=3 runlist=10 gp_put=1
mc1 pb/submit seq=1846 chid=3 idx=0 entry0=0x03600000 entry1=0x00002202 pb_va=0x203600000 pb_len=32
mc1 pb/bytes seq=1846 chid=3 idx=0 chunk=0 nchunks=1 off=0 hex=00800120b5c800009080032002000000f0fe400401000000c080012014000000
```

How the doorbell-path fields are derived:

- `seq` — global monotonic counter across all doorbells and all processes. It is the
  join key for the four records of one submission.
- `chid`, `runlist` on `dbell/fire` — decoded from the 32-bit VF-doorbell token
  (`VECTOR[11:0]`, `RUNLIST_ID[22:16]`). The `dbell/resolve` records carry the RM-side
  chid/runlist, which is why the two can differ in one capture.
- `gp_put` — read from USERD at offset `+0x8c` (`HopperAControlGPFifo`, `clc86f.h`).
- `idx` = `(gp_put - 1) & (entries - 1)` — the GPFIFO slot this doorbell is
  publishing. `entries` is always a power of two.
- `pb_va` = `(entry1[7:0] << 32) | (entry0 & 0xfffffffc)` — pushbuffer GPU VA (under
  UVM also the userspace VA, per Paper F1).
- `pb_len` = `((entry1 >> 10) & 0x1fffff) * 4` — pushbuffer byte count for this
  submission.
- `hex` — the bytes at `pb_va`, read through the sysmem tracker's kernel VA. If the VA
  is not tracked, `pb/bytes_miss` is emitted instead and no bytes are recovered.

### Reassembling the body rows

`rm/alloc` and `uvm/ioctl` each stamp an `id=`, and their `body/*_row` records repeat
it. The chain from wire to atlas is:

1. `strace_diff.py` accumulates rows into a per-`id` dword array as it parses, and
   flushes each completed allocation as one `alloc` / `uvm_ioctl` event.
2. `timeline_merge.py` writes those events into `merged.ndjson` with their decoded
   parameter dicts.
3. `address_atlas.py` reads `merged.ndjson`. It never parses a trace line.

The `id=` join is what makes this order-independent: rows may interleave with any
other traffic, and a truncated capture loses only the allocations whose rows were cut.

### Who knows the wire format

Three things parse `mc1` text, and a fourth is coupled to the parse result:

| | reads | breaks on |
|---|---|---|
| `reverse/tools/strace_diff.py` | `ftrace.txt` | a changed key name in one of the 23 consumed events |
| `reverse/tools/phase_census.py` | `ftrace.txt`, with its own independent `mc1` regex | the record grammar, or a renamed event |
| `reverse/tools/trace_cuda.sh` | `ftrace.txt`, by `grep -c` on raw wire strings including `state=`-discriminated forms (`mc1 dbell/cache state=pre_reserve`, `mc1 dbell/sysmem_track state=add`, …) | a renamed `state=` value — silently, by reporting 0 |
| `reverse/tools/timeline_merge.py` | `strace_diff.parse_ftrace`'s output, not the text | a reordered info-tuple in `MC1_SIMPLE` |

The last row is the reason `MC1_SIMPLE`'s tuples say "do not reorder without changing
`timeline_merge` in lockstep". Everything above that layer — `address_atlas.py`,
`trace_section.py`, `non_uvm_ledger.py` — consumes event kinds and dict keys, which is
what keeps a format change from rippling further outward.

The `trace_cuda.sh` case is the one worth remembering: renaming a `state=` value does
not fail any parse and does not raise anything. The summary line at the end of a
capture just reads 0, and the capture looks like the driver stopped emitting.

---

## Adding an event

Three places, in this order:

1. **This file.** Pick the category and name first — the name is the contract, and a
   name invented at the call site is how a trace format drifts.
2. **The emission site.** `MC_TRACE(cat, "name", "k=%u …", …)`. The macro adds the
   sigil, the category prefix and the trailing newline, so a record cannot be
   malformed by a missed `\n`.
3. **The parser, if anything needs to consume it.** `reverse/tools/strace_diff.py`
   keys its dispatch on `(category, event)`; a 1:1 positional event is a row in
   `MC1_SIMPLE`, anything that aggregates or gates on a discriminant is an explicit
   handler. Events with no parser entry are read and ignored, so an event nothing
   consumes costs nothing but a line in the capture.

If the event replaces or renames an existing one, also grep `trace_cuda.sh` and
`phase_census.py` for the old spelling — neither imports `strace_diff.py`.
