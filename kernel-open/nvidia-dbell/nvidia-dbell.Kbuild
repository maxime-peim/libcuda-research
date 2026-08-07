# SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
# SPDX-License-Identifier: GPL-2.0
###########################################################################
# Kbuild fragment for nvidia-dbell.ko — GPL shim around the kernel's
# hardware-breakpoint API.  See nvidia-dbell/nvidia_dbell.h for rationale.
###########################################################################

NVIDIA_DBELL_SOURCES =
NVIDIA_DBELL_SOURCES += nvidia-dbell/nvidia_dbell.c

NVIDIA_DBELL_OBJECTS = $(patsubst %.c,%.o,$(NVIDIA_DBELL_SOURCES))

obj-m += nvidia-dbell.o
nvidia-dbell-y := $(NVIDIA_DBELL_OBJECTS)

NVIDIA_DBELL_KO = nvidia-dbell/nvidia-dbell.ko

NV_KERNEL_MODULE_TARGETS += $(NVIDIA_DBELL_KO)

#
# CFLAGS: include our own header directory and the shared common/inc
# for NV_VERSION_STRING.
#
NVIDIA_DBELL_CFLAGS =
NVIDIA_DBELL_CFLAGS += -I$(src)/nvidia-dbell
NVIDIA_DBELL_CFLAGS += -I$(src)/common/inc

$(call ASSIGN_PER_OBJ_CFLAGS, $(NVIDIA_DBELL_OBJECTS), $(NVIDIA_DBELL_CFLAGS))

#
# Conftest: gate on the presence of register_user_hw_breakpoint in the
# running kernel.  If absent, nvidia_dbell.c compiles to -ENOSYS stubs.
#
NV_OBJECTS_DEPEND_ON_CONFTEST += $(NVIDIA_DBELL_OBJECTS)

NV_CONFTEST_FUNCTION_COMPILE_TESTS += register_user_hw_breakpoint
