# Contributing

This is an unaffiliated research fork (see `NOTICE`). It is not NVIDIA's
repository, and NVIDIA's contribution process does not apply here — their
guidance is preserved verbatim at [`CONTRIBUTING.nvidia.md`](CONTRIBUTING.nvidia.md)
if you came looking for it.

## What this fork wants

The fork exists to answer one question — what does a `cudaMemcpy` actually do
between the CPU and the GPU — and to keep the answer reproducible. So the
contributions that help most are the ones that make a claim checkable:

- **Corrections.** If a number, a bit-field, or a described mechanism is wrong,
  say so and cite the primary source (an SDK header in this tree, a capture, a
  measurement). Nearly every error found so far has been a claim that drifted
  away from its source rather than one that was never checked.
- **Reproductions on other hardware.** Everything here was exercised on one
  configuration (H100 PCIe, Linux 6.8). A report that something does or does
  not reproduce elsewhere is genuinely useful, especially with a capture.
- **Tooling that widens what can be observed** — see `docs/tracing_cuda.md`.

Changes to the vendored NVIDIA sources under `src/` and `kernel-open/` are
deliberately kept minimal: tracing instrumentation and the doorbell
watchpoint. A functional driver change needs evidence that the question cannot
be answered in `reverse/` or addressed in userspace.

## Practicalities

- Open an issue. There is no CLA, no template, and no triage funnel.
- Security-relevant reports: read `SECURITY.md` first — where a report should
  go depends on whether it reproduces against NVIDIA's driver or only here.
- The local gates in `reverse/` should pass: `make` builds the research tools
  and CUDA trace subjects. The `libmc` build, unit tests, and H100 integration
  matrix live in the separate
  [`libmc`](https://github.com/maxime-peim/libmc) repository.
- Documentation is part of the work, not an afterthought. If a change makes a
  statement in `docs/` false, fix the statement in the same change.
