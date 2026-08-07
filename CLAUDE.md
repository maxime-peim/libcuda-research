# CLAUDE.md — orientation for AI assistants and new contributors

Read `README.md` first; it explains what this repository is. This file covers
the things you would otherwise re-learn the hard way.

This is a research fork of NVIDIA's open GPU kernel modules, pinned at
610.43.02. Almost everything is upstream. The fork's own code is:

- `reverse/mc/` — a CUDA-runtime-like library over raw driver ioctls
- `reverse/tools/` — the tracing toolchain
- `reverse/tests/` — programs built on both
- `kernel-open/nvidia/nv-doorbell-watch.c` + `kernel-open/nvidia-dbell/`
- `MC_TRACE` instrumentation across 18 upstream files plus the fork's own
  doorbell watchpoint, emitting the `mc1`
  record format documented in `docs/reference/trace-format.md`
- `docs/`

Scope is **Hopper (H100)** and the material covered by the seven-part series
*Rebuilding CUDA From Scratch*, published on Medium. Work on other
architectures, and a demand-faulting extension to UVM, exist in the private
research tree and are deliberately not here.

---

## Conventions that are deliberate

If something looks redundant, check here before simplifying it.

- **NVOS64, not NVOS21, for every `NV_ESC_RM_ALLOC`.** NVOS21 fails channel
  allocation on GSP clients. The mechanism was never established — the
  explanation in `docs/findings.md §5` is not supported by readable source —
  but the failure reproduces, so the rule stands. Do not "fix" this in either
  direction without testing on real hardware.

- **`DRF_DEF` / `DRF_NUM`, never hand-rolled shifts**, for every bit-field in
  the method stream and GP entries. One subtlety: `DRF_NUM` takes the
  *semantic* value of the field. For `GP_ENTRY0_GET = 31:2`, pass `pb_va >> 2`,
  not `pb_va`. See `docs/gpfifo_pushbuffer_reference.md`.

- **`NvU64_fmtx`, not `PRIx64`.** `NvU64` is `unsigned long long`; on LP64
  Linux `PRIx64` expands to `"lx"`, which warns.

- **Size constants use `ULL`.** `4u * 1024u * 1024u * 1024u` silently
  overflows to 0.

- **UVM path only.** `NV04_MAP_MEMORY_DMA` is not used for UVM-mapped buffers.
  Real CUDA on Hopper makes zero such calls. Use `uvm_map_buffer()`, or
  `uvm_map_buffer_at()` for anything with a CPU alias.

- **Paper F1 — GPU VA == user VA for any UVM-mapped buffer with a CPU alias.**
  A 4 GiB `PROT_NONE` pool is reserved at `0x200000000` up front; every such
  allocation lands inside it with `MAP_FIXED` and is handed the same address as
  its UVM external range. Violating this wedges the GPU on roughly a quarter of
  runs. See `docs/findings.md §13`.

- **`FERMI_VASPACE_A` with `IS_EXTERNALLY_OWNED`** is mandatory; without it UVM
  rejects the VA space with `NV_ERR_INVALID_FLAGS`.

- **Two writes to submit work**, not one: advance USERD GPPut, then write the
  work-submit token to the VF doorbell. Either alone hangs the channel forever.

- **The doorbell is at `+0x90` inside the USERMODE mapping.** `0x30090` is
  VF-block-relative, and is *not* a BAR0 offset on bare metal (there it is
  `BAR0 + 0xBB0090`). `+0x90` is the invariant. This has been got wrong more
  than once — `docs/gpfifo_pushbuffer_reference.md §11` has the derivation.

- **Kernel instrumentation is tracing, never dereferencing userspace pointers.**
  Every `MC_TRACE` site reads its inputs through `copy_from_user` if they
  might come from userspace. Direct dereference in kernel context is an oops.

---

## Before you change things

- **New ioctl call?** `docs/findings.md §5` has the ABI conventions — size
  embedded in the command, fresh fd per mapping, `ctl_fd` vs `dev_fd` routing.

- **Changing bit-field encoding?** Read `docs/gpfifo_pushbuffer_reference.md`
  §4–§9 first. Wrong encodings produce Xid 31 or Xid 32, usually with a useful
  pointer in `dmesg`.

- **Touching `nv_gpu_ops.c`?** It carries `nvGpuOpsDbellResolveChannel` and the
  `g_dbellGpfifoTable` that the watchpoint depends on. Signature changes ripple
  into `nv_gpu_ops.h`, `rm-gpu-ops.c`, `kernel_channel.c` and
  `nv-doorbell-watch.c`. Note the GPFIFO parameters are stashed at
  channel-construct time on purpose — reading them back from RAMFC trips a
  `vfree` bug in the static-aperture path.

- **Testing the watchpoint?** x86 has only four hardware debug registers, and
  on a desktop Xorg and friends claim them first. Stop them before measuring or
  the watchpoint silently degrades to pass-through.

- **Changing `sm_owner.cu`?** The build asserts on the generated SASS — at
  least 19 system-scope strong stores, one GPU-scope load, one system-scope
  barrier. If those assertions fire, `nvcc` changed its lowering; that is a real
  finding, not a build annoyance. Do not weaken the assertion to make it pass.

- **Adding docs?** Cross-reference them in `README.md` so they stay findable.

---

## Verifying claims

Prose in this tree has been wrong before, in ways that propagated. Two
documented cases: `libcuda.so` was described as ~30 MB when it is ~110 MB, and
the doorbell address above.

So: check claims against the file, the header, or the struct — not against
another `.md`. `ls -l`, `readelf`, or read the source. If you find a stale
figure, grep the whole tree, because it will be in more places than you expect.

---

## When in doubt

`docs/findings.md` is the research log and links out to everything else. If
something seems contradictory, check there before re-deriving it.
