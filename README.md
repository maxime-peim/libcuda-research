# A research fork of NVIDIA's open GPU kernel modules

This is a fork of [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules),
pinned at **610.43.02**, with two things added:

1. **Instrumentation** — tracing through the resource manager and UVM, plus a
   kernel-side watchpoint on the GPU's doorbell register, so it is possible to
   see what `libcuda` actually asks the driver to do.
2. **`mc`** — a small CUDA-runtime-like library built from what that showed.
   It moves bytes between host and GPU using only raw driver ioctls: no CUDA
   runtime, no `libcuda.so` — and for device-to-host it matches a CUDA
   reference doing the same copy, close to the PCIe Gen5 line rate.

It exists to answer one question in as much detail as possible: **what actually
happens when you call `cudaMemcpy`?**

This fork is not affiliated with NVIDIA and is not supported by them. It is
research code — it is not a driver you should install on a machine you care
about.

> Written up as a seven-part series, *Rebuilding CUDA From Scratch*, starting
> with [Part 1. Your GPU is a Separate Computer](https://medium.com/ai-infrastructure/part-1-your-gpu-is-a-separate-computer-d029ce9f3d77).

NVIDIA's own README for the driver is preserved at
[`README.nvidia.md`](README.nvidia.md) — read that for anything about building
or running the driver as a driver.

---

## The short version

A `cudaMemcpy` is, underneath, four writes and a wait:

1. Write a sequence of *methods* — a small command program — into a buffer in
   host memory that the GPU can read.
2. Write an entry into a ring buffer pointing at that program.
3. Advance a "producer" index the GPU polls.
4. Write a token to a doorbell register over PCIe.

Then wait for a semaphore to change. The GPU does the rest by itself. On an
H100 that is a few hundred nanoseconds of CPU work to start a transfer that
takes hundreds of microseconds.

Getting those four writes exactly right — the bit layouts, the object graph the
driver requires first, the address-space invariant that makes it reliable — is
what this repository is about.

---

## What's here

| Path | What it is |
|---|---|
| `reverse/mc/` | the library: RM object setup, UVM mapping, submission, compute |
| `reverse/tests/` | programs built on it, plus CUDA equivalents to compare against |
| `reverse/tools/` | the tracing toolchain — `LD_PRELOAD` shim, decoder, timeline merger |
| `reverse/traces/` | sample captures, so the tools work without a GPU |
| `kernel-open/nvidia/nv-doorbell-watch.c` | the kernel-side doorbell watchpoint |
| `kernel-open/nvidia-dbell/` | a GPL shim, needed for a licensing reason (see below) |
| `docs/` | the documentation set — see the reading order below |

Everything else is upstream NVIDIA code. To see only what this fork changed:

```bash
git remote add upstream https://github.com/NVIDIA/open-gpu-kernel-modules.git
git fetch upstream --tags
git diff 610.43.02 --stat
```

---

## Reading order

The docs are meant to be read in this order if the area is new to you. The
first three are background and are not specific to this fork.

1. **`docs/host_gpu_communication_primer.md`** — PCIe, BARs, MMIO, DMA,
   doorbells, command queues, cache coherency. Not NVIDIA-specific.
2. **`docs/nvidia_software_stack.md`** — the layers: `libcudart`, `libcuda`,
   `nvidia.ko`, `nvidia-uvm.ko`, GSP firmware, and the RM object model.
3. **`docs/gpu_compute_model.md`** — the hardware: SMs, Copy Engines, PBDMA,
   channels, TSGs, runlists, the MMU.
4. **`docs/mc_architecture.md`** — the deep dive: every ioctl, every kernel
   path, and the full bug log from bring-up.
5. **`docs/gpfifo_pushbuffer_reference.md`** — bit-exact formats. GPFIFO
   entries, method headers, NVC8B5 methods, USERD, the doorbell.
6. **`docs/compute_kernel_launch.md`** — the compute path: QMD construction,
   and the chain where a GPU thread rings the doorbell itself.
7. **`docs/tracing_cuda.md`** — how to capture and read a trace of any CUDA
   program: the kernel instrumentation, the tools, and what each output file
   contains.
8. **`docs/reference/trace-format.md`** — the `mc1` kernel trace record
   format: the grammar, the category mask, and the full event catalogue.
9. **`docs/findings.md`** — the research log. Long, and the place where
   anything unclear elsewhere is probably explained.

---

## Running it

**You need an H100 (Hopper) and root.** Everything here was developed and
tested against the driver built from this tree, and that is the configuration
to reproduce. The tracing the tools consume only exists here, and the doorbell
watchpoint is armed by default — it diverts the userspace doorbell mapping to
a shadow page, so the driver does not behave identically to a stock one. Turn
it off with `nv_dbell_disable_intercept=1` if you want it out of the way.

```bash
make modules -j"$(nproc)"      # build the kernel modules
# install and load them — see README.nvidia.md
cd reverse
make libmc mc-all
sudo ./bin/mc_demo --size 256M --iters 5
```

Expected: `PASS: verification` and, on an H100 PCIe, around 55 GB/s
device-to-host at that size — within a percent of what `cudaMemcpy`
reaches on the same box.  See `docs/findings.md §11` for the full table
and `§15` for the clock-boost request this depends on.

The tracing tools do **not** need a GPU. There is a capture checked in:

```bash
cd reverse
python3 tools/trace_section.py --trace traces/cuda_reference --section cudaMemcpy
```

That prints every kernel and syscall event inside one `cudaMemcpy` call, on a
single clock. `tools/phase_census.py --trace traces/cuda_reference` is the other
first look worth taking: it splits the run into setup, transfer and teardown and
counts driver calls in each, which is how the copies turn out to cost none.

### Hardware and software

| | |
|---|---|
| GPU | Hopper — H100 PCIe. Class IDs `0xC86F`, `0xC8B5`, `0xC661` |
| Driver | this tree, based on 610.43.02 |
| OS | Linux, kernel 6.x (developed on 6.8) |
| Privileges | root |

SM-authored submission is Hopper-only, and that appears to be a hardware
limit rather than an unfinished port: it needs a GPU-mappable doorbell page
that other architectures do not expose.

---

## Two things worth knowing before you read the diff

**`nvidia-dbell` is a licensing seam, not a design one.** The doorbell
watchpoint needs `register_user_hw_breakpoint()`, which the kernel exports
`EXPORT_SYMBOL_GPL`. This tree's `nvidia.ko` is declared
`MODULE_LICENSE("Dual MIT/GPL")`, which the kernel accepts for GPL-only
symbols — so strictly it could link the call directly. The shim keeps that
GPL-only dependency in a module that is plainly GPL instead, and mirrors
the constraint the closed driver (`MODULE_LICENSE("NVIDIA")`) would
actually face: it could not link the symbol at all.

**The instrumentation is tracing, not logic.** Every `MC_TRACE` site
reads its inputs through `copy_from_user` where they might come from
userspace. Dereferencing a userspace pointer in kernel context is an oops,
and that lesson was learned the direct way.

---

## Credit

The kernel-side doorbell watchpoint is a port of the technique in Yan et al.
(2026), *Revealing NVIDIA Closed-Source Driver Command Streams for CPU–GPU
Runtime Behavior Insight*, §5.1–5.2. The QMD construction follows Mesa's NVK,
which is the readable reference for that structure.

## Licence

Upstream code stays under its original terms — dual MIT/GPL-2.0, see
[`COPYING`](COPYING). Files added by this fork carry their own SPDX headers:
MIT for userspace, GPL-2.0 for the kernel modules that need it. See
[`NOTICE`](NOTICE).
