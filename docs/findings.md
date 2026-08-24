# Experimental evidence and research log

This file keeps results that cannot be recovered from source comments alone:
capture provenance, measured rates, failure distributions, falsified
hypotheses, and open questions. Implementation details belong beside the code.

The reusable raw-driver library lives in
[`libmc`](https://github.com/maxime-peim/libmc). This repository contains the
instrumented 610.43.02 driver, trace tools, and checked-in captures used to
observe `libcuda` on H100 PCIe.

## Scope and evidence rules

Unless stated otherwise, measurements below were taken on Hopper/H100 PCIe
with driver 610.43.02. Absolute latency and bandwidth are machine-dependent;
protocol shapes and before/after results are the reusable evidence.

Claims are classified as:

- **source fact** when an in-tree header or implementation defines it;
- **trace observation** when the checked-in or named capture contains it;
- **experiment** when an A/B run changed one relevant variable;
- **inference** when the mechanism is consistent with evidence but not exposed
  by the public driver source.

The bit-exact GPFIFO and copy-method definitions are in
[`gpfifo_pushbuffer_reference.md`](gpfifo_pushbuffer_reference.md). Capture and
analysis procedures are in [`tracing_cuda.md`](tracing_cuda.md), and the `mc1`
wire format is in [`reference/trace-format.md`](reference/trace-format.md).

<a id="1-architecture--how-a-d2h-dma-transfer-actually-works"></a>
## Captured CUDA transfer path

The checked-in `reverse/traces/cuda_reference` capture contains one 256 MiB
H2D seed followed by eleven D2H copies. Its stable census is:

| Phase | RM ioctls | GSP RPCs |
|---|---:|---:|
| Setup | 361 | 237 |
| Twelve `cudaMemcpy` calls | 0 | 0 |
| Teardown | 3 | 183 |

The capture contains 205 resolved pushbuffer submissions, zero `pb/bytes`
misses, no failed doorbell-cache resolutions, and 10,589 well-formed `mc1`
records. Counts were byte-identical across five repeat captures. Byte-identity
holds for warm captures only: the first capture after a module load carries
additional one-time GSP bootstrap RPCs (`GSP_SET_SYSTEM_INFO`, `SET_REGISTRY`,
`GET_GSP_STATIC_INFO`, …) that belong to the driver's lifetime, not the traced
process — run with persistence mode enabled, or discard the first capture,
before quoting per-process figures. Wall-clock
times from these captures are not performance evidence because `strace` and
the tracing pipeline perturb them.

This directly shows the boundary: RM, UVM, and GSP establish objects,
mappings, and scheduling state; the transfer hot path is userspace stores to a
method stream, GPFIFO, USERD, and doorbell, followed by semaphore polling.
PBDMA fetches the commands and the copy engine executes them without a
per-transfer ioctl or GSP RPC.

The `fifo/gp_entry_write` instrumentation in `nvidia-push.c` does not fire for
CUDA because that file is part of `nvidia-modeset.ko`; `libcuda` authors its
own pushbuffers.

<a id="11-drf-macro-refactor-and-measured-bandwidth"></a>
## Transfer performance

The following table is the median of three launch means per cell. Each launch
moved roughly 60 GiB, and arm ordering was rotated to distribute machine
drift. `mc_init()` requested boost clocks; clocks were not locked.

| Transfer | libmc D2H | CUDA D2H | libmc H2D | libmc H2D WC | SM-authored D2H |
|---|---:|---:|---:|---:|---:|
| 4 MiB | 31.3 | 35.7 | 42.8 | 49.7 | 35.1 |
| 64 MiB | 53.9 | 53.5 | 55.1 | 55.1 | 53.5 |
| 256 MiB | 55.1 | 52.9 | 55.1 | 55.2 | 54.9 |
| 1 GiB | 55.5 | 55.4 | 55.5 | 55.5 | 55.4 |

Values are GB/s. From 64 MiB upward, libmc and CUDA agree within the box's
run-to-run spread, and SM-authored submission adds no measurable bulk-transfer
cost. The 4 MiB results remain latency-sensitive. Write-combined host memory
improved consistency at 4 MiB but did not change the bulk plateau; CPU reads
from that mapping remain extremely slow.

Short benchmarks can measure clock ramp rather than transfer capability. At
4 MiB, 100 iterations produced 21.9 GB/s mean and 24.0 peak, while 20,000
iterations produced 42.0 mean and 51.6 peak for the same H2D arm. Reproduction
should scale iteration count so startup is a small fraction of total bytes.

Single submissions above `UINT32_MAX` bytes are rejected because
`NVC8B5_LINE_LENGTH_IN` is a 32-bit byte count.

<a id="12-kernel-side-hopper-vf-doorbell-watchpoint-yan-et-al-port-2026-05-06"></a>
## Doorbell observation

The kernel watchpoint diverts a HOPPER_USERMODE_A mapping to a shadow page,
arms an x86 hardware breakpoint on its `+0x90` doorbell dword, resolves the
work-submit token in trap context, records the most recent GPFIFO entry, and
forwards the token to the real register. The implementation is in
`kernel-open/nvidia/nv-doorbell-watch.c` and the GPL export seam is in
`kernel-open/nvidia-dbell/`.

<a id="121-why-userspace-watchpoints-dont-work"></a>
### Why the userspace watchpoint is insufficient

Three runs observing 60 channel submissions each found:

- the real VF doorbell read back `0x0` after every store, while the kernel
  shadow contained valid work-submit tokens;
- USERD `GP_GET == GP_PUT` on all 180 post-store samples, so no pending ring
  entry remained when the userspace SIGTRAP handler ran;
- enabling the userspace watchpoint reduced D2H throughput to about
  0.02 GB/s.

Pushbuffer bytes themselves remained readable. A previous null result was a
capture-limit artifact: the pool was 56 MiB but `PBCAP_MAX_BYTES` defaulted to
16 MiB. Across four A/B runs, the 16 MiB limit captured 90 mappings but none of
the pool and decoded zero streams; a 128 MiB limit captured 93 mappings,
including three pool snapshots, and decoded 60 streams.

### Kernel-watchpoint validation

A 4 MiB CUDA round trip produced 174 `dbell/gp_put` and 174 `pb/submit`
records, resolved all 20 channels, and recorded zero cache failures or GPFIFO
lookup misses. Repeated stress was 40/40 successful across plain and
LD_PRELOAD runs, with no WARN, BUG, or crash in the kernel log. Submission
counts varied by two between launches; the zero-failure invariants did not.

Observed libcuda layout on this version was one 2 MiB BAR1 mapping containing
20 channel slots. Each slot was 0x3000 bytes: an 8 KiB, 1024-entry GPFIFO ring
followed by USERD at offset 0x2000. This is a libcuda allocation policy, not a
Hopper ABI guarantee.

<a id="126-gotchas-encountered-and-documented"></a>
### Reproduction constraints

- x86 provides only four DR watchpoint slots; display or inference processes
  may consume them before the test starts;
- breakpoints bind to existing threads, so late-created workers are not
  automatically covered;
- the observed doorbell path is the BAR1 HOPPER_USERMODE_A variant — 194/194
  captured doorbell events landed on the BAR1 slot vs 0/194 on BAR0 — although
  libcuda allocates both BAR0 and BAR1 objects; the doorbell dword is at
  `+0x90` in either variant;
- on the original VMware passthrough host, kernel traces used guest physical
  addresses while `/proc/PID/pagemap` exposed host physical addresses;
- GPU VA, GPU physical/FB offset, and CPU BAR1 aperture address must not be
  compared as if they were one address space;
- `kernel-open/conftest.sh` "functions" probes are inverted: probe code must
  intentionally fail to compile when the symbol exists (call a real multi-arg
  function with zero args), and consuming `.c` files must
  `#include "conftest.h"` — the generated `NV_*_PRESENT` macros are not in
  `ccflags-y`.

<a id="13-va-pool-fix-for-intermittent-d2h-failures-2026-05-12"></a>
## Paper-F1 VA identity

The original libmc path gave each CPU-visible RM allocation one CPU VA and a
different UVM GPU VA. At 128 MiB this produced a 25–34% failure rate with two
signatures: completion-semaphore timeouts after PBDMA consumed the H2D entry,
and silent H2D no-ops later exposed by D2H verification.

The libcuda strace showed the missing structural invariant: reserve a 4 GiB
`PROT_NONE` window at `0x200000000`, map RM memory into it with `MAP_FIXED`,
and register those same addresses as UVM external ranges. Thus CPU VA equals
GPU VA for every CPU-aliased UVM allocation.

| Configuration | Runs | Pass | Timeout | `0x20018000` | Garble |
|---|---:|---:|---:|---:|---:|
| Before fix | 100 | 66 | 15 | 16 | 3 |
| Pushbuffer/control anchored only | 100 | 83 | 1 | 6 | 10 |
| Full VA pool | 100 | 100 | 0 | 0 | 0 |
| Full VA pool | 1000 | 1000 | 0 | 0 | 0 |

Partial conformance converted many hard timeouts into silent corruption; only
the full identity layout removed both modes. The exact GPU/GSP mechanism is
not exposed by the available evidence. A cache or lookup keyed inconsistently
by the CPU/UVM address pair is plausible, but remains inference.

An initial `mremap(MREMAP_FIXED)` implementation also reached 1000/1000 but
broke trace decoding because kernel mapping trackers had recorded the original
VMA start. Landing mappings at their final VA on the first `mmap()` preserved
both correctness and observer identity.

### Refuted alternatives

- GPU PTE traces showed no physical-page alias between the pushbuffer and data
  allocation.
- Changing contiguity, persistence, alignment, or memory-handle flags did not
  improve the failure rate.
- Selecting a non-GRCE engine, switching doorbell aperture, and reordering RM
  allocations did not remove this failure mode.
- libcuda created range groups but did not use them to set policy; copying the
  unused calls would not affect mappings.
- Anchoring already-selected VMAs without reserving a pool collided with UVM's
  internal ranges and returned `NV_ERR_IN_USE`.

The durable debugging lesson was to compare structural address choices, not
only ioctl counts or flag bundles. Rate claims used at least 100 trials and
candidate fixes used 1000-trial confirmation.

<a id="14-fully-gpu-resident-control-plane-mc-fb-carrier-2026-05-29-to-2026-05-31"></a>
## FB-carrier and cross-engine ordering

The FB-carrier experiment moved the SM-authored channel's pushbuffer, GPFIFO,
and USERD from sysmem to HBM. The SM then performs its setup writes within GPU
memory and only the final BAR1 doorbell crosses PCIe.

At 1 MiB, the final path passed 1000/1000 and measured 189.4 us / 5.54 GB/s,
versus 273.4 us / 3.84 GB/s before cleanup. The improvement came from removing
a redundant host BAR1 reread; the release semaphore used by the final design
is sysmem-resident and polled by the host.

Two investigations mattered:

1. Instrumentation showed libcuda uses a fresh `NV50_MEMORY_VIRTUAL` carrier
   per source memory object, with no carrier-local heap and `dmaOffset=0`.
   Matching that shape simplified attribution but did not fix the wedge.
2. The SM's poll used a system-scope load that repeatedly hit a stale L1 line.
   `__threadfence_system()` plus a `.cg` load bypassed L1 and observed the
   copy-engine release. Over 1000 trials, observation was around 133,000
   SM-clock ticks, p99 within roughly 1% of median, and maximum about 79 us.

The experiment also showed that visibility is not pairwise ordered across GPU
egress paths: observing a compute-channel semaphore on the CPU did not prove
that an independent SM write to sysmem had reached host memory. CPU cache
flushes could not reveal data that had not drained from the GPU. The shipped
path therefore polls the copy engine's own sysmem release semaphore instead of
using compute completion as a proxy.

### Refuted alternatives

- `NV50_MEMORY_VIRTUAL` did not redirect PTEs into hidden carrier backing;
  traces showed `has_heap=0` and the chosen/source PTEs matched.
- Switching carriers to `NV01_MEMORY_VIRTUAL` returned `NV_ERR_NO_MEMORY` and
  did not match libcuda's shape.
- The SM's writes were visible in FB and the copy engine ran; an extra
  USERMODE memory-barrier register was unnecessary. The stale observation was
  on the SM load path, not the publication path.

<a id="15-the-missing-perf_boost-why-mc-ran-at-idle-clocks-2026-08-18"></a>
## Performance boost

Before requesting boost, 400 interleaved 256 MiB D2H iterations were trimodal:

| Mode | Count | Throughput |
|---|---:|---:|
| 4.91 ms | 52 | 54.7 GB/s |
| 7.13 ms | 344 | 37.6 GB/s |
| 10.7 ms | 4 | 25.1 GB/s |

Clock sampling showed 345 MHz during libmc runs and 1755 MHz during CUDA runs.
The libcuda capture contained one `NV2080_CTRL_CMD_PERF_BOOST` on the
subdevice, with flags `CMD_BOOST_TO_MAX | CUDA_YES` and infinite duration.
Adding the same best-effort request to `mc_init()` changed paired results to:

| Measurement | Before | After |
|---|---:|---:|
| libmc D2H mean | 36.8–39.8 GB/s | 54.8 GB/s |
| libmc D2H p50 | 7.13 ms | 4.89 ms |
| CUDA D2H | 54.7 GB/s | 54.7 GB/s |
| libmc H2D | about 45 GB/s | 55.4 GB/s |
| SM-authored D2H | about 38 GB/s | 53.1 GB/s |

The full 51-test matrix passed. A few first iterations after idle remained
slow while clocks ramped. Session-to-session libmc ceilings previously
attributed to protocol differences were residual GPU clock state.

## Open questions

- Which internal PBDMA, GSP, or MMU structure makes split CPU/UVM VAs
  unreliable on this Hopper path? The behavioral result is strong; the
  microarchitectural mechanism remains unproven.
- `address_atlas.py` does not always infer roles for handles mapped into the VA
  pool, although method decoding is correct.
- The observed libcuda BAR1 channel packing and carrier choices may change in
  another driver release and must be remeasured rather than assumed.
- Cross-engine visibility beyond the tested SM, PBDMA, CE, and host mappings
  should not be generalized without a targeted experiment.

## Reproduction entry points

```sh
cd reverse
make
python3 tools/phase_census.py --trace traces/cuda_reference
python3 tools/trace_section.py --trace traces/cuda_reference --section cudaMemcpy
python3 tools/non_uvm_ledger.py traces/cuda_reference/merged.ndjson --summary
```

For a fresh capture and module requirements, follow
[`tracing_cuda.md`](tracing_cuda.md). For library integration tests, follow the
[`libmc` README](https://github.com/maxime-peim/libmc/blob/main/README.md).
