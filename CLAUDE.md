# CLAUDE.md — orientation for AI assistants and new contributors

Read `README.md` first; it explains what this repository is. This file covers
the things you would otherwise re-learn the hard way.

This is a research fork of NVIDIA's open GPU kernel modules, pinned at
610.43.02. Almost everything is upstream. The fork's own code is:

- `reverse/tools/` — the tracing toolchain
- `reverse/tests/cuda/` — CUDA reference and instrumentation programs
- `kernel-open/nvidia/nv-doorbell-watch.c` + `kernel-open/nvidia-dbell/`
- `MC_TRACE` instrumentation across 18 upstream files plus the fork's own
  doorbell watchpoint, emitting the `mc1`
  record format documented in `docs/reference/trace-format.md`
- `docs/`

The CUDA-runtime-like implementation, its tests, implementation docs, and
maintainer invariants live in the sibling `libmc` repository:
https://github.com/maxime-peim/libmc. Source-code changes implementing RM/UVM
and command-stream behavior belong there.

Scope is **Hopper (H100)** and the material covered by the seven-part series
*Rebuilding CUDA From Scratch*, published on Medium. Work on other
architectures, and a demand-faulting extension to UVM, exist in the private
research tree and are deliberately not here.

---

## Instrumentation conventions that are deliberate

If something looks redundant, check here before simplifying it.

- **Kernel instrumentation is tracing, never dereferencing userspace pointers.**
  Every `MC_TRACE` site reads its inputs through `copy_from_user` if they
  might come from userspace. Direct dereference in kernel context is an oops.

---

## Before you change things

- **Touching `nv_gpu_ops.c`?** It carries `nvGpuOpsDbellResolveChannel` and the
  `g_dbellGpfifoTable` that the watchpoint depends on. Signature changes ripple
  into `nv_gpu_ops.h`, `rm-gpu-ops.c`, `kernel_channel.c` and
  `nv-doorbell-watch.c`. Note the GPFIFO parameters are stashed at
  channel-construct time on purpose — reading them back from RAMFC trips a
  `vfree` bug in the static-aperture path.

- **Testing the watchpoint?** x86 has only four hardware debug registers, and
  on a desktop Xorg and friends claim them first. Stop them before measuring or
  the watchpoint silently degrades to pass-through.

- **Adding docs?** Cross-reference them in `README.md` so they stay findable.

- **Changing the library or its demos?** Work in `libmc` and follow that
  repository's `CLAUDE.md`; it contains the RM/UVM, command-stream, VA-pool,
  and generated-SASS invariants.

---

## Verifying claims

Prose in this tree has been wrong before, in ways that propagated. Two
documented cases: `libcuda.so` was described as ~30 MB when it is ~110 MB, and
the VF doorbell offset — `+0x90` inside the USERMODE mapping is the invariant;
`0x30090` is VF-block-relative, not a BAR0 offset on bare metal (see
`docs/gpfifo_pushbuffer_reference.md` §11).

So: check claims against the file, the header, or the struct — not against
another `.md`. `ls -l`, `readelf`, or read the source. If you find a stale
figure, grep the whole tree, because it will be in more places than you expect.

---

## When in doubt

`docs/findings.md` is the research log and links out to everything else. If
something seems contradictory, check there before re-deriving it.
