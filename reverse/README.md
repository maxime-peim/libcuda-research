# reverse/ — the research code

Everything in this directory is additions to NVIDIA's driver tree, not part
of it. Two halves:

- **`mc/`** — a small CUDA-runtime-like library that talks to the GPU
  through raw driver ioctls. No CUDA runtime, no `libcuda.so`.
- **`tools/`** — the tracing and decoding toolchain that was used to work
  out what `libcuda` does, which is how `mc/` came to exist.

Plus `tests/` (programs built on both) and `traces/` (sample captures).

---

## Quick start

```bash
cd reverse
make libmc mc-all        # the library and every test — no CUDA toolchain needed
sudo ./bin/mc_demo --size 256M --iters 5
```

Expected on an H100 PCIe:

```
mc_demo: size=268435456 bytes (256 MiB), iters=5, dir=D2H
D2H 256 MiB x 5 iters:
  Peak: 54.95 GB/s (4.88 ms)
  Mean: 54.59 GB/s (4.92 ms)
PASS: verification
```

The very first run after the GPU has been idle can read a little lower —
`mc_init` requests boost clocks the way libcuda does, and the ramp takes
a moment to land.

`make help` lists every target. `make` alone also builds the CUDA-based
reference programs, which need `nvcc`.

**Root is required** — `UVM_REGISTER_CHANNEL` needs `CAP_SYS_ADMIN`.

**The tools need the driver built from this tree; `mc` itself does not.**
The `mc1` tracing the tools consume only exists here. The library runs on a
stock driver of the same version — the whole test matrix passes against the
610.43.02 modules from NVIDIA's own installer. Building from this tree is
still the configuration everything was developed against, and note that the
doorbell watchpoint it adds is armed by default and must be turned off
(`nv_dbell_disable_intercept=1`) for the GPU-initiated paths. See the
top-level `README.md` for how to build and load it.

---

## `mc/` — the library

The core of `mc/mc.h` is ten functions: `mc_init`, `mc_fini`,
`mc_malloc_device`, `mc_malloc_host`, `mc_malloc_host_wc`,
`mc_host_register`, `mc_host_unregister`, `mc_free`, `mc_gpu_va`,
`mc_memcpy`. The rest of the header is demo entry points for the
GPU-initiated paths.

A copy is described by two things: which VA space the buffers live in,
and who authors the submission.

| VA space | Channel resources live in |
|---|---|
| `MC_VAS_UVM` | UVM-managed, GPU VA == CPU VA |
| `MC_VAS_SYSMEM_CARRIER` | host memory, non-UVM |
| `MC_VAS_FB_CARRIER` | HBM — only the doorbell and the sysmem release sema cross PCIe |

| Agent | Who writes the submission |
|---|---|
| `MC_XFER_HOST` | the CPU |
| `MC_XFER_SM` | one GPU thread, from device code |

`mc/README.md` documents the API properly. `../docs/mc_architecture.md` is
the level below that: every ioctl, every kernel path, and the bug log.

**Hopper (H100) only.** SM-authored submission needs a GPU-mappable
doorbell page that, as far as this work established, only Hopper provides.

---

## `tools/` — tracing and decoding

| Tool | What it does |
|---|---|
| `pbcap.c` | `LD_PRELOAD` shim recording what `libcuda` asks the kernel for |
| `trace_cuda.sh` | captures a whole run: ftrace + strace + pbcap, merged |
| `trace_section.py` | windows a capture down to a single libcuda call |
| `decode.py` | decodes (and encodes) NVC8B5 method streams |
| `strace_diff.py` | aligns two captures and reports where they diverge |
| `address_atlas.py` | maps GPU virtual addresses back to allocations |
| `phase_census.py` | splits a capture into setup / transfer / teardown and counts driver calls |
| `run_mc_tests.py` | runs every `mc` test across the VA-space × agent matrix |

There is a sample capture checked in, so this works without a GPU:

```bash
python3 tools/trace_section.py --trace traces/cuda_reference --section cudaMemcpy
```

`../docs/tracing_cuda.md` explains the pipeline and how to read its output.

---

## Layout

```
mc/          the library (one translation unit per topic)
mc/kernels/  sm_owner.cu — the kernel that authors its own submission
tests/mc/    programs built on the library
tests/cuda/  the same workloads through CUDA, for comparison
tools/       tracing and decoding
traces/      sample captures
```
