# reverse/ — libcuda research tools and probes

Everything here is research code added to NVIDIA's driver tree. It captures,
decodes, and compares what `libcuda` sends through the driver boundary.

The reusable raw-driver library developed from this evidence has moved to
[`libmc`](https://github.com/maxime-peim/libmc), together with its tests,
implementation docs, and pinned NVIDIA source dependency.

## Quick start

```sh
make
python3 tools/trace_section.py --trace traces/cuda_reference --section cudaMemcpy
```

`make` builds the CUDA reference programs and `lib/libpbcap.so`; it requires a
CUDA toolkit. The Python decoders can work directly from the checked-in sample
capture and do not require a GPU.

The capture pipeline itself needs the instrumented driver built from the parent
repository. See [`../docs/tracing_cuda.md`](../docs/tracing_cuda.md) for setup
and usage.

## Tools

| Tool | What it does |
|---|---|
| `pbcap.c` | `LD_PRELOAD` shim recording what `libcuda` asks the kernel for |
| `trace_cuda.sh` | captures a whole run: ftrace + strace + pbcap, merged |
| `trace_section.py` | windows a capture down to a single libcuda call |
| `decode.py` | decodes and encodes NVC8B5 method streams |
| `strace_diff.py` | aligns two captures and reports where they diverge |
| `address_atlas.py` | maps GPU virtual addresses back to allocations |
| `phase_census.py` | splits a capture into setup, transfer, and teardown phases |

## Layout

```text
tests/cuda/  CUDA workloads used as references and trace targets
tools/       tracing and decoding tools
traces/      checked-in sample captures
```
