# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build the CUDA reference programs and the pbcap tracing shim.
# The standalone raw-driver library and its tests live at:
#   https://github.com/maxime-peim/libmc

CC        ?= gcc
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
NVCC_ARCH ?= native
CUDA_INC  ?= /usr/local/cuda/include

NVCC_FLAGS   ?= -O2 -std=c++17 -arch=$(NVCC_ARCH) --cudart=shared
PBCAP_CFLAGS ?= -O2 -g -Wall -fPIC
PBCAP_LDLIBS ?= -ldl -lpthread

CUDA_SRCS := $(wildcard tests/cuda/*.cu)
CUDA_BINS := $(patsubst tests/cuda/%.cu,bin/%,$(CUDA_SRCS))
PBCAP     := lib/libpbcap.so

BINS := $(CUDA_BINS)
LIBS := $(PBCAP)

.DEFAULT_GOAL := all
.PHONY: all bins libs cuda-all clean help libpbcap
.PHONY: $(notdir $(BINS))

all: $(BINS) $(LIBS)
bins: $(BINS)
libs: $(LIBS)
cuda-all: $(CUDA_BINS)
libpbcap: $(PBCAP)

$(foreach b,$(BINS),$(eval $(notdir $b): $b))

bin lib:
	mkdir -p $@

bin/%: tests/cuda/%.cu | bin
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(PBCAP): tools/pbcap.c | lib
	$(CC) $(PBCAP_CFLAGS) -shared -I$(CUDA_INC) -o $@ $< $(PBCAP_LDLIBS)

clean:
	rm -rf bin lib

help:
	@echo "Targets:"
	@echo "  all          build CUDA probes and libpbcap"
	@echo "  cuda-all     build every tests/cuda/*.cu binary"
	@echo "  bins         build everything under bin/"
	@echo "  libs         build everything under lib/"
	@echo "  libpbcap     build $(PBCAP)"
	@echo "  clean        remove bin/ and lib/"
	@echo "  help         this message"
	@echo
	@echo "Discovered CUDA binaries:"
	@$(foreach b,$(CUDA_BINS),echo "  $(notdir $b)";)
	@echo
	@echo "Each binary can also be built by name, e.g. 'make cuda_reference'."
	@echo
	@echo "Overridable variables:"
	@echo "  CC         C compiler        (default: $(CC))"
	@echo "  NVCC       CUDA compiler     (default: $(NVCC))"
	@echo "  NVCC_ARCH  GPU architecture  (default: $(NVCC_ARCH))"
	@echo "  CUDA_INC   CUDA header dir   (default: $(CUDA_INC))"
