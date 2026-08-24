# A research fork of NVIDIA's open GPU kernel modules

This is a fork of [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules),
pinned at **610.43.02**, with instrumentation added:

1. **Instrumentation** — tracing through the resource manager and UVM, plus a
   kernel-side watchpoint on the GPU's doorbell register, so it is possible to
   see what `libcuda` actually asks the driver to do.

The CUDA-runtime-like library built from those observations now lives in the
separate [`libmc`](https://github.com/maxime-peim/libmc) repository. Keeping it
separate lets each library revision pin the exact official NVIDIA driver source
whose private RM/UVM ABI it targets, while this repository remains focused on
instrumentation and reproducible evidence.

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
| [`libmc`](https://github.com/maxime-peim/libmc) | the library, its raw-driver implementation, tests, and pinned header dependency |
| `reverse/tests/cuda/` | CUDA comparison and instrumentation workloads |
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

The documentation now stays specific to this project. For general PCIe, GPU,
or CUDA background, use a current architecture reference rather than a second
hand primer in this repository.

1. **[`libmc/docs/mc_architecture.md`](https://github.com/maxime-peim/libmc/blob/main/docs/mc_architecture.md)** — concise context, VA-space, channel, and submission architecture.
2. **`docs/gpfifo_pushbuffer_reference.md`** — bit-exact formats. GPFIFO
   entries, method headers, NVC8B5 methods, USERD, the doorbell.
3. **[`libmc/docs/compute_kernel_launch.md`](https://github.com/maxime-peim/libmc/blob/main/docs/compute_kernel_launch.md)** — the Hopper compute and SM-authored copy chain.
4. **`docs/tracing_cuda.md`** — how to capture and read a trace of any CUDA
   program: the kernel instrumentation, the tools, and what each output file
   contains.
5. **`docs/reference/trace-format.md`** — the `mc1` kernel trace record
   format: the grammar, the category mask, and the full event catalogue.
6. **`docs/findings.md`** — compact experimental evidence: capture counts,
   measured rates, falsified hypotheses, and open questions.

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
make
```

The decoder and analysis tools do **not** need a GPU because a capture is
checked in. To build and run the raw-driver implementation instead, clone
[`libmc`](https://github.com/maxime-peim/libmc) with its submodule and follow
that repository's README.

For a first look at the sample capture:

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
