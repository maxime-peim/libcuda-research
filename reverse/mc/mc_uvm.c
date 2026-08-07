/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_uvm.c — UVM ioctl wrappers (UVM_INITIALIZE, UVM_REGISTER_GPU,
 * UVM_REGISTER_GPU_VASPACE, UVM_REGISTER_CHANNEL, UVM_MAP_EXTERNAL_*,
 * UVM_UNREGISTER_*, UVM_FREE).
 *
 * UVM ioctls differ from RM ioctls in two ways:
 *   - Raw integer command numbers (the UVM dispatcher in
 *     kernel-open/nvidia-uvm/uvm.c switches on cmd directly; there is
 *     no _IO{R,W,WR} size/type encoding).  Pass the integer from
 *     uvm_ioctl.h directly as cmd.
 *   - Status is in params.rmStatus (NV_STATUS at the end of every UVM
 *     param struct), not in the syscall return.  ioctl() returns 0 on
 *     syscall success regardless of UVM-level errors — always check
 *     rmStatus.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvCpuUuid.h"
#include "nvos.h"
#include "uvm_linux_ioctl.h"
#include "uvm_ioctl.h"
#include "uvm_types.h"
#include "nv_uvm_user_types.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"

#include "mc_internal.h"

/*
 * Fetch the 16-byte physical GPU UUID.  UVM_REGISTER_GPU requires it as
 * the device identity.  We use NV2080_CTRL_CMD_GPU_GET_GID_INFO
 * (0x2080014a) on the subdevice handle, with flags =
 * FORMAT_BINARY(2)|TYPE_SHA1(0)|MODE_GPU(0) so RM returns the raw UUID
 * bytes rather than an ASCII representation.
 */
static int rm_get_gpu_uuid(int ctl_fd, NvHandle h_client, NvHandle h_subdevice,
                           NvU8 out_uuid[NV_UUID_LEN])
{
  NV2080_CTRL_GPU_GET_GID_INFO_PARAMS p = {};
  p.flags = NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY;
  if (rm_control(ctl_fd, h_client, h_subdevice,
                 NV2080_CTRL_CMD_GPU_GET_GID_INFO, &p, sizeof(p)) != 0)
    return -1;
  if (p.length != NV_UUID_LEN)
  {
    ERROR_LOG("unexpected UUID length %u (expected %d)", p.length, NV_UUID_LEN);
    return -1;
  }
  memcpy(out_uuid, p.data, NV_UUID_LEN);
  return 0;
}
/*
 * UVM_INITIALIZE — must be the first ioctl on the primary /dev/nvidia-uvm
 * fd.  Its command number 0x30000001 is special (outside the small
 * UVM_IOCTL_BASE range) because it's the one-time initialization step
 * that precedes the regular dispatcher.
 */
static int uvm_initialize(int uvm_fd)
{
  UVM_INITIALIZE_PARAMS p = {};
  long                  r = ioctl(uvm_fd, UVM_INITIALIZE, &p);
  if (r != 0 || p.rmStatus != NV_OK)
  {
    ERROR_LOG("UVM_INITIALIZE r=%ld rmStatus=0x%x", r, p.rmStatus);
    return -1;
  }
  return 0;
}

/*
 * UVM_MM_INITIALIZE — issued on a SECONDARY /dev/nvidia-uvm fd.  Its
 * only job is to have the kernel hold a reference on the calling
 * process's mm_struct via the file descriptor, so on process exit the
 * kernel calls uvm_mm_release() (triggered by fd close) BEFORE
 * uvm_release() runs — giving UVM a chance to drain outstanding faults
 * while the mm is still valid.
 *
 * Without it, uvm_va_space_mm_or_current_retain() inside
 * UVM_REGISTER_GPU_VASPACE returns NULL, causing that ioctl to fail
 * with NV_ERR_PAGE_TABLE_NOT_AVAIL (0x5d).
 *
 * Some platforms don't need it and reply NV_WARN_NOTHING_TO_DO (0x10006)
 * — on those we close the secondary fd immediately.  Everywhere else
 * we keep it open (intentionally leaked) for the lifetime of the process.
 */
static int uvm_mm_initialize(int uvm_fd)
{
  UVM_MM_INITIALIZE_PARAMS p = {};
  int                      uvm_fd_mm;
  long                     r;

  uvm_fd_mm = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
  if (uvm_fd_mm < 0)
  {
    ERROR_LOG("open(/dev/nvidia-uvm) for MM: %s", strerror(errno));
    return -1;
  }
  p.uvmFd = uvm_fd;
  r       = ioctl(uvm_fd_mm, UVM_MM_INITIALIZE, &p);
  if (r != 0)
  {
    ERROR_LOG("UVM_MM_INITIALIZE syscall: %s", strerror(errno));
    close(uvm_fd_mm);
    return -1;
  }
  if (p.rmStatus == NV_OK)
  {
    /* intentionally leaked — holds mm ref until process exit */
    return 0;
  }
  if (p.rmStatus == NV_WARN_NOTHING_TO_DO)
  {
    close(uvm_fd_mm);
    return 0;
  }
  ERROR_LOG("UVM_MM_INITIALIZE rmStatus=0x%x", p.rmStatus);
  close(uvm_fd_mm);
  return -1;
}

/*
 * UVM_REGISTER_GPU — the gpu_uuid field is in/out: on entry we pass the
 * physical UUID; on return UVM has written back the instance UUID
 * (identical to the physical UUID on non-MIG GPUs, can differ on MIG).
 * The instance UUID is what every subsequent UVM call expects.
 */
static int uvm_register_gpu(int uvm_fd, int dev_fd, NvHandle h_client,
                            const NvU8 phys_uuid[NV_UUID_LEN],
                            NvU8       out_inst_uuid[NV_UUID_LEN])
{
  UVM_REGISTER_GPU_PARAMS p = {};
  memcpy(p.gpu_uuid.uuid, phys_uuid, NV_UUID_LEN);
  p.rmCtrlFd    = dev_fd;
  p.hClient     = h_client;
  p.hSmcPartRef = 0;

  if (ioctl(uvm_fd, UVM_REGISTER_GPU, &p) != 0 || p.rmStatus != NV_OK)
  {
    ERROR_LOG("UVM_REGISTER_GPU rmStatus=0x%x", p.rmStatus);
    return -1;
  }
  memcpy(out_inst_uuid, p.gpu_uuid.uuid, NV_UUID_LEN);
  return 0;
}

/*
 * UVM_REGISTER_GPU_VASPACE — hands our RM-allocated h_vaspace (with the
 * IS_EXTERNALLY_OWNED flag) to UVM so UVM can install GPU-MMU PTEs into
 * its page tables.  Must run after UVM_REGISTER_GPU.
 */
static int uvm_register_gpu_vaspace(int uvm_fd, int dev_fd, NvHandle h_client,
                                    const NvU8 inst_uuid[NV_UUID_LEN],
                                    NvHandle   h_vaspace)
{
  UVM_REGISTER_GPU_VASPACE_PARAMS p = {};
  memcpy(p.gpuUuid.uuid, inst_uuid, NV_UUID_LEN);
  p.rmCtrlFd = dev_fd;
  p.hClient  = h_client;
  p.hVaSpace = h_vaspace;

  if (ioctl(uvm_fd, UVM_REGISTER_GPU_VASPACE, &p) != 0 || p.rmStatus != NV_OK)
  {
    ERROR_LOG("UVM_REGISTER_GPU_VASPACE rmStatus=0x%x", p.rmStatus);
    return -1;
  }
  return 0;
}

/*
 * Minimum-viable UVM setup.  Opens /dev/nvidia-uvm, performs the four
 * ioctls needed to make our RM client + GPU + VA space known to UVM,
 * and returns the primary UVM fd plus the GPU instance UUID for every
 * subsequent UVM_MAP_EXTERNAL_ALLOCATION / UVM_REGISTER_CHANNEL call.
 *
 * Must run AFTER rm_alloc_vaspace() (IS_EXTERNALLY_OWNED is required)
 * and BEFORE any uvm_map_buffer call.  The primary fd must stay open
 * for the lifetime of the process; the secondary fd allocated inside
 * uvm_mm_initialize is also retained (intentionally leaked — it holds
 * the mm_struct reference until process exit).
 */
int uvm_setup(int ctl_fd, int dev_fd, NvHandle h_client,
                     NvHandle h_subdevice, NvHandle h_vaspace,
                     NvU8 out_inst_uuid[NV_UUID_LEN], int *out_uvm_fd)
{
  NvU8 phys_uuid[NV_UUID_LEN];
  int  uvm_fd;

  uvm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
  if (uvm_fd < 0)
  {
    ERROR_LOG("open(/dev/nvidia-uvm): %s", strerror(errno));
    return -1;
  }
  if (uvm_initialize(uvm_fd) != 0) goto fail;
  if (uvm_mm_initialize(uvm_fd) != 0) goto fail;
  if (rm_get_gpu_uuid(ctl_fd, h_client, h_subdevice, phys_uuid) != 0)
    goto fail;
  if (uvm_register_gpu(uvm_fd, dev_fd, h_client, phys_uuid, out_inst_uuid) != 0)
    goto fail;
  if (uvm_register_gpu_vaspace(uvm_fd, dev_fd, h_client, out_inst_uuid,
                               h_vaspace) != 0)
    goto fail;

  *out_uvm_fd = uvm_fd;
  return 0;

fail:
  close(uvm_fd);
  return -1;
}

/*
 * Declare an already-reserved CPU VA range as UVM-external — UVM claims
 * authority over page-table entries for that range.  The companion
 * uvm_map_external_allocation() then installs actual PTEs.
 */
static int uvm_create_external_range(int uvm_fd, NvU64 base, NvU64 size,
                                     const char *label)
{
  UVM_CREATE_EXTERNAL_RANGE_PARAMS p = {};
  p.base   = base;
  p.length = size;
  if (ioctl(uvm_fd, UVM_CREATE_EXTERNAL_RANGE, &p) != 0 || p.rmStatus != NV_OK)
  {
    ERROR_LOG("CREATE_EXTERNAL_RANGE(%s) rmStatus=0x%x", label, p.rmStatus);
    return -1;
  }
  return 0;
}

/*
 * Install GPU MMU PTEs at [base, base+size) pointing to hMemory.
 *
 * UVM_MAP_EXTERNAL_ALLOCATION_PARAMS embeds a 256-slot perGpuAttributes
 * array (UVM_MAX_GPUS) — ~9 KiB total.  We heap-allocate via calloc to
 * keep stack usage bounded.  gpuAttributesCount = 1 means "configure
 * only slot [0]"; the remaining 255 slots are ignored.  Our only GPU
 * goes in slot 0 with Default caching/format/compression.
 */
static int uvm_map_external_allocation(int uvm_fd, int dev_fd, NvHandle h_client,
                                       const NvU8 inst_uuid[NV_UUID_LEN],
                                       NvHandle hMemory, NvU64 base,
                                       NvU64 size, NvU64 offset,
                                       const char *label)
{
  UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *p;
  UvmGpuMappingAttributes            *attr;
  NV_STATUS                           st;
  long                                r;

  p = calloc(1, sizeof(*p));
  if (p == NULL) { ERROR_LOG("calloc MAP_EXTERNAL"); return -1; }

  p->base               = base;
  p->length             = size;
  p->offset             = offset;
  p->gpuAttributesCount = 1;
  p->rmCtrlFd           = dev_fd;
  p->hClient            = h_client;
  p->hMemory            = hMemory;

  attr = &p->perGpuAttributes[0];
  memcpy(attr->gpuUuid.uuid, inst_uuid, NV_UUID_LEN);
  attr->gpuMappingType     = UvmGpuMappingTypeReadWriteAtomic;
  attr->gpuCachingType     = UvmGpuCachingTypeDefault;
  attr->gpuFormatType      = UvmGpuFormatTypeDefault;
  attr->gpuElementBits     = UvmGpuFormatElementBitsDefault;
  attr->gpuCompressionType = UvmGpuCompressionTypeDefault;

  r  = ioctl(uvm_fd, UVM_MAP_EXTERNAL_ALLOCATION, p);
  st = p->rmStatus;
  free(p);
  if (r != 0 || st != NV_OK)
  {
    ERROR_LOG("MAP_EXTERNAL_ALLOCATION(%s) r=%ld rmStatus=0x%x", label, r, st);
    return -1;
  }
  return 0;
}

int uvm_map_buffer_range_at(int uvm_fd, int dev_fd, NvHandle h_client,
                            const NvU8 inst_uuid[NV_UUID_LEN],
                            NvHandle hMemory, NvU64 base, NvU64 size,
                            NvU64 offset, const char *label)
{
  if (uvm_create_external_range(uvm_fd, base, size, label) != 0)
    return -1;
  if (uvm_map_external_allocation(uvm_fd, dev_fd, h_client, inst_uuid, hMemory,
                                  base, size, offset, label) != 0)
  {
    uvm_free_range(uvm_fd, base, label);
    return -1;
  }
  return 0;
}

/*
 * Install an RM memory object into the GPU's VA space via UVM, returning
 * the GPU VA at which the channel can access the memory.  Used ONLY for
 * buffers WITHOUT a CPU alias — the GPU VA is picked by the kernel
 * (via MAP_ANONYMOUS/PROT_NONE) and is unrelated to any CPU VA the
 * caller might have.
 *
 * Three-step sequence:
 *   a. Reserve a CPU VA range via MAP_ANONYMOUS/PROT_NONE.  UVM will
 *      claim the range as externally managed; PROT_NONE makes any
 *      stray CPU access fault immediately — the caller shouldn't be
 *      touching this VA.
 *   b. UVM_CREATE_EXTERNAL_RANGE declares the range UVM-owned.
 *   c. UVM_MAP_EXTERNAL_ALLOCATION installs GPU MMU PTEs at the range
 *      pointing to the pages that back hMemory.
 *
 * This is mc_malloc_device's mapping path — no Paper-F1 constraint
 * applies because the CPU never references the returned VA.
 */
NvU64 uvm_map_buffer(int uvm_fd, int dev_fd, NvHandle h_client,
                            const NvU8 inst_uuid[NV_UUID_LEN], NvHandle hMemory,
                            NvU64 size, const char *label)
{
  void *base;
  NvU64 gpu_va;

  base = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED) { ERROR_LOG("uvm_map(%s) mmap: %s", label, strerror(errno)); return 0; }
  gpu_va = (NvU64)(uintptr_t)base;

  if (uvm_create_external_range(uvm_fd, gpu_va, size, label) != 0) return 0;
  if (uvm_map_external_allocation(uvm_fd, dev_fd, h_client, inst_uuid, hMemory,
                                  gpu_va, size, 0, label) != 0)
    return 0;
  return gpu_va;
}

/*
 * Variant of uvm_map_buffer() that anchors the UVM range at a caller-
 * supplied CPU VA — used for every buffer with a CPU alias.  The caller
 * is expected to have placed a MAP_FIXED file-backed mapping at cpu_va
 * already (via rm_alloc_sysmem_at or rm_map_memory_at into the VA
 * pool), so the resulting mapping satisfies Paper F1: GPU VA == user VA.
 *
 * Paper F1 matters structurally: the kernel-side doorbell watchpoint's
 * bar1_track / sysmem_track entries record the tracked VMA's
 * user_va_start, and UVM_CREATE_EXTERNAL_RANGE uses that same VA as
 * its GPU-VA base — a GPFIFO-entry-recorded pb_va translates back to a
 * kernel VA for method-stream decoding.  It also matters for
 * correctness: a UVM-mapped CPU-aliased buffer whose GPU VA ≠ CPU VA
 * wedges the GPU ~25% of the time on H100 PCIe
 * (see docs/mc_architecture.md §3.7 and §12 bug #12).
 */
NvU64 uvm_map_buffer_at(int uvm_fd, int dev_fd, NvHandle h_client,
                               const NvU8 inst_uuid[NV_UUID_LEN],
                               NvHandle hMemory, void *cpu_va, NvU64 size,
                               const char *label)
{
  NvU64 gpu_va = (NvU64)(uintptr_t)cpu_va;
  if (uvm_create_external_range(uvm_fd, gpu_va, size, label) != 0) return 0;
  if (uvm_map_external_allocation(uvm_fd, dev_fd, h_client, inst_uuid, hMemory,
                                  gpu_va, size, 0, label) != 0)
    return 0;
  return gpu_va;
}

/*
 * UVM_REGISTER_CHANNEL — mandatory before NVA06F_CTRL_CMD_GPFIFO_SCHEDULE
 * on externally-owned VA spaces (see "Bug 1737765" in
 * kernel-open/nvidia-uvm/uvm_user_channel.c).
 *
 * base/length reserves a VA window in UVM that user mappings must not
 * overlap.  We pick 4 MiB at 0x7f0000000000 — well above the UVM-
 * internal region and outside the VA pool at 0x200000000.  Only the
 * UVM channel (role MC_ROLE_UVM_CE) is UVM-registered, so this reservation
 * applies once per mc_ctx; the carrier DMA + compute channels never call
 * UVM_REGISTER_CHANNEL.
 */
int uvm_register_channel(int uvm_fd, int dev_fd, NvHandle h_client,
                                const NvU8 inst_uuid[NV_UUID_LEN],
                                NvHandle   h_channel)
{
  UVM_REGISTER_CHANNEL_PARAMS p = {};
  memcpy(p.gpuUuid.uuid, inst_uuid, NV_UUID_LEN);
  p.rmCtrlFd = dev_fd;
  p.hClient  = h_client;
  p.hChannel = h_channel;
  p.base     = MC_UVM_CHANNEL_BASE;
  p.length   = MC_UVM_CHANNEL_LENGTH;

  if (ioctl(uvm_fd, UVM_REGISTER_CHANNEL, &p) != 0 || p.rmStatus != NV_OK)
  {
    ERROR_LOG("UVM_REGISTER_CHANNEL rmStatus=0x%x", p.rmStatus);
    return -1;
  }
  return 0;
}

/*
 * UVM_UNREGISTER_CHANNEL (cmd 28).  Counterpart to uvm_register_channel.
 * Must run BEFORE the RM-side channel free; otherwise UVM still holds a
 * dup of the channel object and the free races.
 */
void uvm_unregister_channel(int uvm_fd, NvHandle h_client,
                                   NvHandle h_channel)
{
  UVM_UNREGISTER_CHANNEL_PARAMS p = {};
  p.hClient  = h_client;
  p.hChannel = h_channel;
  if (ioctl(uvm_fd, UVM_UNREGISTER_CHANNEL, &p) != 0 || p.rmStatus != NV_OK)
    WARN_LOG("UVM_UNREGISTER_CHANNEL rmStatus=0x%x", p.rmStatus);
}

/*
 * UVM_UNMAP_EXTERNAL (cmd 66).  Counterpart to the UVM_MAP_EXTERNAL_
 * ALLOCATION side of uvm_map_buffer{,_at}.  There is no explicit
 * counterpart to UVM_CREATE_EXTERNAL_RANGE — the range is reaped when
 * the VA space is unregistered.  Unmapping each buffer explicitly gives
 * UVM an unambiguous point to flush GPU MMU PTEs for that range.
 */
void uvm_unmap_buffer(int uvm_fd, const NvU8 inst_uuid[NV_UUID_LEN],
                             NvU64 base, NvU64 length, const char *label)
{
  UVM_UNMAP_EXTERNAL_PARAMS p = {};
  p.base   = base;
  p.length = length;
  memcpy(p.gpuUuid.uuid, inst_uuid, NV_UUID_LEN);
  if (ioctl(uvm_fd, UVM_UNMAP_EXTERNAL, &p) != 0 || p.rmStatus != NV_OK)
    WARN_LOG("UVM_UNMAP_EXTERNAL(%s) rmStatus=0x%x", label, p.rmStatus);
}

/*
 * UVM_FREE (cmd 34).  cudaHostUnregister uses this for each
 * CREATE_EXTERNAL_RANGE segment it made for a registered host pointer.
 * Unlike UVM_UNMAP_EXTERNAL it takes only the range base.
 */
void uvm_free_range(int uvm_fd, NvU64 base, const char *label)
{
  UVM_FREE_PARAMS p = {};
  p.base = base;
  if (ioctl(uvm_fd, UVM_FREE, &p) != 0 || p.rmStatus != NV_OK)
    WARN_LOG("UVM_FREE(%s base=0x%llx) rmStatus=0x%x", label,
             (unsigned long long)base, p.rmStatus);
}

/*
 * UVM_UNREGISTER_GPU_VASPACE (cmd 26).  Counterpart to
 * UVM_REGISTER_GPU_VASPACE.  Detaches our externally-owned
 * FERMI_VASPACE_A from UVM, flushing any remaining GPU MMU PTEs UVM
 * had cached for this address space.
 */
void uvm_unregister_gpu_vaspace(int        uvm_fd,
                                       const NvU8 inst_uuid[NV_UUID_LEN])
{
  UVM_UNREGISTER_GPU_VASPACE_PARAMS p = {};
  memcpy(p.gpuUuid.uuid, inst_uuid, NV_UUID_LEN);
  if (ioctl(uvm_fd, UVM_UNREGISTER_GPU_VASPACE, &p) != 0 || p.rmStatus != NV_OK)
    WARN_LOG("UVM_UNREGISTER_GPU_VASPACE rmStatus=0x%x", p.rmStatus);
}

/*
 * UVM_UNREGISTER_GPU (cmd 38).  Counterpart to uvm_register_gpu.  Note
 * the param struct field is `gpu_uuid` on this side (different from
 * UVM_UNMAP_EXTERNAL / UVM_UNREGISTER_GPU_VASPACE's `gpuUuid` — upstream
 * naming inconsistency, preserved here).
 */
void uvm_unregister_gpu(int uvm_fd, const NvU8 inst_uuid[NV_UUID_LEN])
{
  UVM_UNREGISTER_GPU_PARAMS p = {};
  memcpy(p.gpu_uuid.uuid, inst_uuid, NV_UUID_LEN);
  if (ioctl(uvm_fd, UVM_UNREGISTER_GPU, &p) != 0 || p.rmStatus != NV_OK)
    WARN_LOG("UVM_UNREGISTER_GPU rmStatus=0x%x", p.rmStatus);
}
