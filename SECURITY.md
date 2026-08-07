# Security

**This is an unaffiliated research fork** (see `NOTICE`). Where a report should
go depends on what it is about.

Anything that reproduces on NVIDIA's own `open-gpu-kernel-modules`, or on the
proprietary driver, should go to NVIDIA through the process below. This fork
has no standing to handle those and forwarding them would only add latency.

Anything specific to what this fork adds — `reverse/`,
`kernel-open/nvidia-dbell/`, the doorbell watchpoint in
`kernel-open/nvidia/nv-doorbell-watch.c`, or the `mc1` tracing — should be
raised as an issue here, **not** sent to NVIDIA PSIRT.

It is worth being blunt about what this code is. The doorbell watchpoint
diverts a userspace mapping to a shadow page and installs a hardware
breakpoint; `mc` drives a GPU channel directly from userspace with hand-built
pushbuffers. Neither is written to be robust against a hostile caller, and
neither belongs on a machine that matters. Treat the whole fork as research
code rather than something with a security boundary.

---

## Reporting a vulnerability in an NVIDIA product

To report a potential security vulnerability in any NVIDIA product, please use either:
* This web form: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html), or
* Send email to: [NVIDIA PSIRT](mailto:psirt@nvidia.com)

**OEM Partners should contact their NVIDIA Customer Program Manager**

If reporting a potential vulnerability via email, please encrypt it using NVIDIA’s public PGP key ([see PGP Key page](https://www.nvidia.com/en-us/security/pgp-key/)) and include the following information:
* Product/Driver name and version/branch that contains the vulnerability
* Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
* Instructions to reproduce the vulnerability
* Proof-of-concept or exploit code
* Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

See https://www.nvidia.com/en-us/security/ for past NVIDIA Security Bulletins and Notices.
