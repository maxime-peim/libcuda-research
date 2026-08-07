# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# reverse/Makefile — builds every example artefact:
#
#   bin/mc_demo          — D2H transfer on top of libmc, no CUDA library.
#   bin/cuda_reference   — CUDA-runtime analogue of mc_demo (needs nvcc).
#   bin/cuda_mmap_snooper — paused CUDA app for external /proc inspection (needs nvcc).
#   bin/cuda_init        — minimal cudaInitDevice probe (needs nvcc).
#   bin/cuda_host_register — cudaHostRegister exercise (needs nvcc).
#   lib/libmc.so         — the mc library itself (see mc/mc.h).
#   lib/libpbcap.so      — LD_PRELOAD shim for capturing CUDA pushbuffer state
#                          (needs the CUDA headers, but not nvcc).
#
# New .cu files dropped in tests/cuda/ and new .c files in tests/mc/
# are picked up automatically — no Makefile edits required.
#
# Quick start:
#
#   make                   — same as `make all` (builds everything).
#   make cuda-all          — every tests/cuda/*.cu binary.
#   make mc-all            — every tests/mc/*.c binary.
#   make cuda_reference    — CUDA reference benchmark.
#   make mc_demo           — libmc demo (no CUDA toolchain needed).
#   make libmc             — library only.
#   make libpbcap          — pbcap shim.
#   make help              — print this summary.
#   make clean             — remove every produced artefact.
#
# Override toolchain or flags from the command line:
#
#   make CC=clang mc_demo
#   make NVCC=/opt/cuda/bin/nvcc cuda_reference
#   make NVCC_ARCH=sm_90 cuda_reference       # sm_90 = Hopper; sm_80 = Ampere
#   make CFLAGS='-O0 -g -fsanitize=address' mc_demo

# ── Toolchain ──────────────────────────────────────────────────────────────
CC        ?= gcc
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUOBJDUMP ?= $(shell command -v cuobjdump 2>/dev/null || echo /usr/local/cuda/bin/cuobjdump)
NVCC_ARCH ?= native
CUDA_INC  ?= /usr/local/cuda/include

REPO_ROOT := $(realpath ..)

# ── Flags ──────────────────────────────────────────────────────────────────
# Warnings on, but allow unused helpers: the sources keep a few reference
# implementations around as documentation.
CFLAGS       ?= -O2 -g -Wall -Wno-unused-function
# libmc talks to the driver directly, so it needs the in-tree SDK headers
# (class definitions, NVOS structs, UVM ioctl definitions).
SDK_INC      := -I$(REPO_ROOT)/src/common/sdk/nvidia/inc \
                -I$(REPO_ROOT)/src/common/sdk/nvidia/inc/class \
                -I$(REPO_ROOT)/src/common/sdk/nvidia/inc/alloc \
                -I$(REPO_ROOT)/src/nvidia/arch/nvalloc/unix/include \
                -I$(REPO_ROOT)/src/common/inc \
                -I$(REPO_ROOT)/src/nvidia/inc \
                -I$(REPO_ROOT)/kernel-open/nvidia-uvm \
                -I$(REPO_ROOT)/kernel-open/common/inc

# cuda_reference + cuda_mmap_snooper: --cudart=shared links libcudart.so
# dynamically so LD_PRELOAD tools (like pbcap) can interpose cuda* entry
# points.
NVCC_FLAGS   ?= -O2 -std=c++17 -arch=$(NVCC_ARCH) --cudart=shared

# libpbcap: shared library; -ldl for dlsym, -lpthread for atomic hooks.
PBCAP_CFLAGS ?= -O2 -g -Wall -fPIC
PBCAP_LDLIBS ?= -ldl -lpthread

# sm_owner kernel build flags.  -O2 is required for nvcc to fold our
# DRF-style header constants into immediate operands; -O0 leaves them
# as runtime register loads.  sm_90 is pinned explicitly because the
# Hopper-specific .sys store scope is sm_90+.
NVCC_KERNEL_FLAGS ?= -arch=sm_90 --cubin -O2 -lineinfo -Imc

# ── Artefacts ──────────────────────────────────────────────────────────────
# Auto-discovered from source directories — add a file, it builds automatically.
CUDA_SRCS      := $(wildcard tests/cuda/*.cu)
CUDA_BINS      := $(patsubst tests/cuda/%.cu,bin/%,$(CUDA_SRCS))

# Auto-discover test programs.  Files ending in _helpers.c are
# shared-helper translation units (mc_test_helpers.c etc.), NOT
# stand-alone tests — exclude them from binary discovery and link
# them into every test instead.
MC_ALL_SRCS := $(wildcard tests/mc/*.c)
MC_HELPERS  := $(filter %_helpers.c,$(MC_ALL_SRCS))
MC_SRCS     := $(filter-out %_helpers.c,$(MC_ALL_SRCS))
MC_BINS     := $(patsubst tests/mc/%.c,bin/%,$(MC_SRCS))

PBCAP        := lib/libpbcap.so
LIBMC := lib/libmc.so

BINS := $(CUDA_BINS) $(MC_BINS)
LIBS := $(PBCAP) $(LIBMC)

.DEFAULT_GOAL := all
.PHONY: all bins libs cuda-all mc-all clean help libpbcap libmc
.PHONY: $(notdir $(BINS))

all: $(BINS) $(LIBS)
bins: $(BINS)
libs: $(LIBS)
cuda-all:      $(CUDA_BINS)
mc-all: $(MC_BINS)

# Friendly aliases without the directory prefix (auto-generated).
$(foreach b,$(BINS),$(eval $(notdir $b): $b))

libpbcap:     $(PBCAP)
libmc: $(LIBMC)

# ── Output directories (order-only prereqs so parallel builds don't race) ─
bin lib:
	mkdir -p $@

# ── Build rules ────────────────────────────────────────────────────────────

# Generic rule for every tests/mc/*.c — links libmc
# and pulls in the test-side helper translation units (mc_test_helpers.c
# etc.).  No SDK headers needed; just public mc.h + the helper
# headers.  Future tests added to tests/mc/ pick up the helpers
# automatically.
bin/%: tests/mc/%.c $(MC_HELPERS) $(LIBMC) | bin
	$(CC) $(CFLAGS) -Imc -Itests/mc -o $@ \
	    $< $(MC_HELPERS) -L./lib -lmc \
	    -Wl,-rpath,'$$ORIGIN/../lib'

# Generic rule for every tests/cuda/*.cu.
bin/%: tests/cuda/%.cu | bin
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(PBCAP): tools/pbcap.c | lib
	$(CC) $(PBCAP_CFLAGS) -shared -I$(CUDA_INC) -o $@ $< $(PBCAP_LDLIBS)

# libmc: the ioctl + submission machinery behind a small
# CUDA-runtime-like public API (see reverse/mc/mc.h).  Needs the SDK
# headers, so it uses SDK_INC; compiled -fPIC and linked -shared.
# mc_internal.h is a prerequisite only for rebuild tracking — public
# consumers include mc.h and nothing else.
#
# Sources (one TU per topical band):
#   mc_rm.c            — RM ioctl wrappers (alloc/free/map/control + typed
#                        object allocators).  Hosts mc_debug() too.
#   mc_uvm.c           — UVM ioctl wrappers (initialize, register/unregister,
#                        map/unmap external allocation, register channel).
#   mc_submit.c        — method-stream + GPFIFO entry builders +
#                        mc_channel_arm/submit/poll_sema + drain_channel.
#   mc_vaspace.c       — VA pool (Paper-F1 anchor) + mc_va_space_* helpers.
#   mc_core.c          — lifecycle: per-kind channel init/fini + mc_init/
#                        mc_fini + mc_malloc/mc_free + mc_memcpy + the two
#                        CE-doorbell demos.
#   mc_compute_qmd.c   — Hopper QMD V04 builder + embedded SASS.
#   mc_compute_demo.c  — HOPPER_COMPUTE_A demos (mc_compute_doorbell_kernel,
#                        mc_memcpy_d2h_gpu_doorbell_sm, mc_compute_get_scratch).
$(LIBMC): mc/mc_core.c mc/mc_rm.c mc/mc_uvm.c mc/mc_submit.c \
                 mc/mc_vaspace.c mc/mc_compute_qmd.c \
                 mc/mc_compute_demo.c \
                 mc/mc.h mc/mc_internal.h \
                 mc/mc_compute_qmd.h | lib
	$(CC) $(CFLAGS) -fPIC $(SDK_INC) -Imc -shared -o $@ \
	    mc/mc_core.c mc/mc_rm.c mc/mc_uvm.c mc/mc_submit.c \
	    mc/mc_vaspace.c mc/mc_compute_qmd.c \
	    mc/mc_compute_demo.c

# ── sm_owner kernel: .cu -> .cubin -> .text bytes -> generated .h ─────────
#
# The sm_owner kernel (mc/kernels/sm_owner.cu) is a single-thread Hopper
# kernel that authors an entire CE-channel submission from the SM:
# pushbuffer methods, GPFIFO entry, USERD GPPut advance, BAR1 doorbell.
# libmc embeds it as a hex array (mc_sm_owner_kernel_sass[]) consumed
# by mc_sm_owner.c.
#
# The build is two-stage: nvcc compiles the .cu to a cubin, then the
# .text.sm_owner_kernel bytes are pulled out into a raw binary and xxd -i
# wraps them into a C-array source file that we transform into a header.
# The extraction is done by hand with readelf + dd rather than objcopy, for
# the reason given below.
#
# We commit the generated header (mc/mc_sm_owner_sass.h) so a fresh
# checkout builds without nvcc on the host.  The header is regenerated
# whenever sm_owner.cu changes; reviewers see the SASS-bytes diff in
# git, which is the only honest review surface for hand-managed SASS.
mc/mc_sm_owner_sass.h: mc/kernels/sm_owner.cu
	@echo "  NVCC    mc/kernels/sm_owner.cu (-> cubin)"
	$(NVCC) $(NVCC_KERNEL_FLAGS) -o mc/kernels/sm_owner.cubin $<
	@echo "  EXTRACT .text.sm_owner_kernel"
	@# objcopy's BFD doesn't have a backend for NVIDIA CUDA ELFs
	@# (e_machine = 190 / 0xBE), so we parse the ELF section table
	@# with readelf -SW (wide format, no name truncation) and dd
	@# the bytes out by file offset/size.  Columns 5/6 are hex
	@# offset/size without 0x prefix.
	@OFF=$$(readelf -SW mc/kernels/sm_owner.cubin                         \
	        | awk '/\.text\.sm_owner_kernel/ {print "0x"$$5; exit}');     \
	 SZ=$$(readelf  -SW mc/kernels/sm_owner.cubin                         \
	        | awk '/\.text\.sm_owner_kernel/ {print "0x"$$6; exit}');     \
	 if [ -z "$$OFF" ] || [ -z "$$SZ" ]; then                             \
	   echo "ERROR: could not locate .text.sm_owner_kernel in cubin" >&2; \
	   exit 1;                                                            \
	 fi;                                                                  \
	 dd if=mc/kernels/sm_owner.cubin of=mc/kernels/sm_owner.text.bin      \
	    bs=1 skip=$$(printf '%d' $$OFF) count=$$(printf '%d' $$SZ) status=none
	@echo "  GEN     $@"
	@(  echo '/*';                                                        \
	    echo ' * mc_sm_owner_sass.h — auto-generated by reverse/Makefile';\
	    echo ' *';                                                        \
	    echo ' * Source: mc/kernels/sm_owner.cu';                         \
	    echo ' * Compiled with: $(NVCC) $(NVCC_KERNEL_FLAGS)';            \
	    echo ' *';                                                        \
	    echo ' * DO NOT EDIT BY HAND.  To regenerate, modify sm_owner.cu';\
	    echo ' * and run `make mc/mc_sm_owner_sass.h` from reverse/.';    \
	    echo ' */';                                                       \
	    echo '#ifndef MC_SM_OWNER_SASS_H';                                \
	    echo '#define MC_SM_OWNER_SASS_H';                                \
	    echo '';                                                          \
	    echo '#include <stdint.h>';                                       \
	    echo '';                                                          \
	    SIZE=$$(stat -c '%s' mc/kernels/sm_owner.text.bin);               \
	    REGS=$$($(CUOBJDUMP) --dump-resource-usage mc/kernels/sm_owner.cubin \
	            2>&1                                                      \
	            | sed -n 's/.*REG:\([0-9]*\).*/\1/p' | head -1);          \
	    echo "/* SASS byte length and register count, harvested from cubin. */"; \
	    echo "#define MC_SM_OWNER_SASS_LEN ((uint32_t)$$SIZE)";          \
	    echo "#define MC_SM_OWNER_REGS     ((uint32_t)$$REGS)";          \
	    echo '';                                                          \
	    echo "static const uint8_t mc_sm_owner_kernel_sass[] = {";       \
	    xxd -i < mc/kernels/sm_owner.text.bin;                            \
	    echo "};";                                                        \
	    echo '';                                                          \
	    echo "#endif /* MC_SM_OWNER_SASS_H */";                           \
	) > $@
	@rm -f mc/kernels/sm_owner.text.bin
	@echo "  OK      $@ ($$(stat -c '%s' $@) bytes)"

# Convenience target so `make sm_owner` does the right thing.
.PHONY: sm_owner
sm_owner: mc/mc_sm_owner_sass.h


# ── Utility ────────────────────────────────────────────────────────────────
clean:
	rm -rf bin lib

help:
	@echo "Targets:"
	@echo "  all              build everything (bins + libs)"
	@echo "  cuda-all         build every tests/cuda/*.cu binary"
	@echo "  mc-all           build every tests/mc/*.c binary"
	@echo "  bins             build everything under bin/"
	@echo "  libs             build everything under lib/"
	@echo "  libpbcap         build $(PBCAP) — requires CUDA headers"
	@echo "  libmc     build $(LIBMC) — the mc library"
	@echo "  clean            remove bin/ and lib/ directories"
	@echo "  help             this message"
	@echo
	@echo "Discovered CUDA binaries (cuda-all):"
	@$(foreach b,$(CUDA_BINS),echo "  $(notdir $b)";)
	@echo "Discovered mc binaries (mc-all):"
	@$(foreach b,$(MC_BINS),echo "  $(notdir $b)";)
	@echo
	@echo "Each binary can also be built by name, e.g. 'make mc_demo'."
	@echo
	@echo "Overridable variables:"
	@echo "  CC         C compiler          (default: $(CC))"
	@echo "  CFLAGS     C compile flags     (default: $(CFLAGS))"
	@echo "  NVCC       CUDA compiler       (default: $(NVCC))"
	@echo "  NVCC_ARCH  GPU arch for nvcc   (default: $(NVCC_ARCH))"
	@echo "  CUDA_INC   CUDA header dir     (default: $(CUDA_INC))"
