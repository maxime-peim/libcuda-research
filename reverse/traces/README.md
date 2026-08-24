# Sample captures

Two recorded runs of the tracing pipeline, checked in so the analysis tools work
without a GPU. Both were taken on an H100 PCIe (driver 610.43.02, this tree's
instrumented modules loaded, `mc1` trace format at its default mask).

| Capture | Command |
|---|---|
| `cuda_reference/` | `sudo tools/trace_cuda.sh ./bin/cuda_reference --size 256M --iters 11` |
| `cuda_host_register/` | `sudo tools/trace_cuda.sh ./bin/cuda_host_register` |

## Files

| File | What it is |
|---|---|
| `ftrace.txt` | the raw kernel trace buffer — `mc1 <category>/<event>` records |
| `strace.log` | `strace -ttt -T` of the same run |
| `pbcap/timeline.ndjson` | the LD_PRELOAD shim's libcuda-call timeline |
| `merged.ndjson` | the three sources folded onto one monotonic clock by `timeline_merge.py` |
| `atlas.json` | the VA→allocation atlas built by `address_atlas.py` |
| `methods.txt` | the decoded per-doorbell method streams |

## Try it without a GPU

```
python3 tools/phase_census.py  --trace traces/cuda_reference
python3 tools/trace_section.py --trace traces/cuda_reference --section cudaMemcpy
python3 tools/trace_section.py --trace traces/cuda_host_register --section cudaHostRegister
python3 tools/non_uvm_ledger.py traces/cuda_reference/merged.ndjson --summary
```

The last command reports 20 channels and 21 carrier-shaped mappings, but no
carrier linked to any channel. That is the result supported by this capture:
the mappings are unattributed. They may be scratch/global objects or internal
channel bindings that are not visible in the captured NVOS46 stream. Run it
against a `libmc` carrier-VA capture to see the per-channel resource columns
populated.

## What `cuda_reference/` contains

One 256 MiB host-to-device copy to seed the buffer, then eleven timed
device-to-host copies — twelve `cudaMemcpy` calls. 205 pushbuffer submissions, all
resolved against the atlas, zero `pb/bytes` misses, a clean doorbell census
(`cache failed: 0`), and 10 589 well-formed `mc1` records with none malformed.

The point of the capture is the phase split: setup costs 361 RM ioctls and 237 GSP
RPCs, the twelve copies cost **zero of each**, and teardown costs 3 and 183.
`phase_census.py` prints that table. The counts were byte-identical across five
repeat captures; wall times were not, and are inflated by `strace`, so timings
should come from untraced runs.

## What `cuda_host_register/` contains

A single `cudaHostRegister` of a 4 MiB `malloc` buffer, and its matching
`cudaHostUnregister`. It exists so the per-function slicing example is reproducible:
one legacy `NV_ESC_RM_ALLOC_MEMORY` escape, then three
`UVM_CREATE_EXTERNAL_RANGE` + `UVM_MAP_EXTERNAL_ALLOCATION` pairs splitting the
buffer 4 KiB / 4 190 208 B / 4 KiB, because the pointer is sixteen-byte aligned
rather than page aligned. No GPU submission happens inside the call.
