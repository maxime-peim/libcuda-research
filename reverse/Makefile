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
