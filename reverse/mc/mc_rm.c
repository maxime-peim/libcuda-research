/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc_rm.c — RM ioctl wrappers (NV_ESC_RM_ALLOC / FREE / CONTROL /
 * MAP_MEMORY / MAP_MEMORY_DMA / REGISTER_FD).
 *
 * Two layers:
 *   1. Generic alloc / free / map / control wrappers used to talk to
 *      /dev/nvidiactl + /dev/nvidia0.
 *   2. Typed object allocators (rm_alloc_root, rm_alloc_device,
 *      rm_alloc_subdevice, rm_alloc_usermode, rm_alloc_tsg,
 *      rm_alloc_channel, rm_alloc_ce, rm_alloc_compute) that encode
 *      the fields and invariants that matter for one RM class each.
 *
 * Also hosts mc_debug() — the verbose-logging gate used by the log
 * macros in mc_internal.h.  Defined here because mc_rm.c is always
 * part of the libmc.so link.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "nvos.h"
#include "nv-ioctl.h"
#include "nv-ioctl-numbers.h"
#include "nv_escape.h"
#include "nv-unix-nvos-params-wrappers.h"
#include "class/cl0000.h"        /* NV01_ROOT */
#include "class/cl0080.h"        /* NV01_DEVICE_0 */
#include "class/cl2080.h"        /* NV20_SUBDEVICE_0 */
#include "class/cl0040.h"        /* NV01_MEMORY_LOCAL_USER */
#include "class/cl003e.h"        /* NV01_MEMORY_SYSTEM */
#include "class/cl0071.h"        /* NV01_MEMORY_SYSTEM_OS_DESCRIPTOR */
#include "class/cla06c.h"        /* KEPLER_CHANNEL_GROUP_A */
#include "class/cl90f1.h"        /* FERMI_VASPACE_A */
#include "class/clc86f.h"        /* HOPPER_CHANNEL_GPFIFO_A */
#include "class/clc8b5.h"        /* HOPPER_DMA_COPY_A */
#include "class/clb0b5sw.h"      /* NVB0B5_ALLOCATION_PARAMETERS */
#include "class/clcbc0.h"        /* HOPPER_COMPUTE_A */
#include "class/clc661.h"        /* HOPPER_USERMODE_A */
#include "class/cl50a0.h"        /* NV50_MEMORY_VIRTUAL */
#include "alloc/alloc_channel.h" /* NV_CHANNEL_ALLOC_PARAMS */
#include "ctrl/ctrl2080/ctrl2080ce.h"
#include "ctrl/ctrl2080/ctrl2080perf.h"
#include "ctrl/ctrlc36f.h"
#include "ctrl/ctrla06f/ctrla06fgpfifo.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"

#include "mc_internal.h"

/* ── Logging gate ─────────────────────────────────────────────────────────
 * Library logging is stderr-only, uncoloured, and prefixed "[mc]".
 * Set MC_VERBOSE=1 in the environment to enable DEBUG output.
 */
static int mc_debug_enabled = -1;

int mc_debug(void)
{
  if (mc_debug_enabled == -1)
  {
    const char *v      = getenv("MC_VERBOSE");
    mc_debug_enabled   = (v != NULL && v[0] != '\0' && v[0] != '0');
  }
  return mc_debug_enabled;
}

/* ── RM helpers ──────────────────────────────────────────────────────────
 *
 * These wrap the RM escape ioctls.  They are return-coded rather than
 * abort-on-failure, because this is a library and a library has no
 * business killing its caller's process.
 *
 * Where a comment names a file and a check in the in-tree RM kernel
 * source, that is the cross-check or invariant being relied on.  Symbols
 * are named rather than line numbers, which drift between releases.
 */

/* Handle values are picked by the client (RM doesn't assign them); we
 * start at 0xf0000000 to stay clear of kernel-reserved low ranges.  A
 * handle space is process-local, so we never collide with other clients. */
static NvHandle g_next_handle = 0xf0000000;

static NvHandle next_handle(void) { return g_next_handle++; }
/*
 * rm_free_handle — issue NV_ESC_RM_FREE on one handle in a teardown-
 * tolerant way.  Logs WARN on any failure and returns so subsequent
 * teardown steps still run: aborting halfway through cleanup leaves
 * worse kernel-side state than best-effort full teardown, so we must
 * NOT abort here.
 *
 * Params (per NVOS00_PARAMETERS in src/common/sdk/nvidia/inc/nvos.h):
 *   hRoot         = client handle (NV01_ROOT).
 *   hObjectParent = parent in the RM object tree (e.g. h_device for a
 *                   memory handle, h_client for the device/subdevice).
 *   hObjectOld    = the handle being freed.
 * When freeing the client itself, all three equal h_client.
 *
 * `label` is for log output only — descriptive strings ("h_channel",
 * "h_client") make the teardown trace readable.
 */
void rm_free_handle(int ctl_fd, NvHandle h_root, NvHandle h_parent,
                           NvHandle h_target, const char *label)
{
  NVOS00_PARAMETERS p = {};
  long              r;

  p.hRoot         = h_root;
  p.hObjectParent = h_parent;
  p.hObjectOld    = h_target;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_FREE & 0xff, NVOS00_PARAMETERS), &p);
  if (r != 0)
  {
    WARN_LOG("rm_free(%s=0x%x) syscall failed: %s", label, h_target,
             strerror(errno));
    return;
  }
  if (p.status != NV_OK)
  {
    WARN_LOG("rm_free(%s=0x%x) rmStatus=0x%x", label, h_target,
             (unsigned)p.status);
    return;
  }
  DEBUG_LOG("rm_free(%s=0x%x) ok", label, h_target);
}

/*
 * Allocate an RM object using NV_ESC_RM_ALLOC (escape 0x2b).
 *
 * NV_ESC_RM_ALLOC accepts either NVOS21_PARAMETERS (32 B) or
 * NVOS64_PARAMETERS (48 B); we always use NVOS64, matching what libcuda
 * does.  NVOS64 routes through Nv04AllocWithAccessSecInfo() in escape.c,
 * which properly propagates the security context to GSP-RM; NVOS21 goes
 * through a legacy path that fails channel allocations with
 * NV_ERR_INVALID_OBJECT_PARENT (0x1f) on GSP clients.
 */
NvHandle rm_alloc(int ctl_fd, NvHandle root, NvHandle parent,
                         NvU32 hclass, void *alloc_params)
{
  NVOS64_PARAMETERS p = {};
  long              r;

  p.hRoot            = root;
  p.hObjectParent    = parent;
  p.hObjectNew       = next_handle();
  p.hClass           = hclass;
  p.pAllocParms      = (NvP64)(uintptr_t)alloc_params;
  p.pRightsRequested = 0;
  p.paramsSize       = 0;
  p.flags            = 0;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_ALLOC & 0xff, NVOS64_PARAMETERS), &p);
  if (r != 0)
  {
    ERROR_LOG("rm_alloc(class=0x%x) syscall failed: %s", hclass, strerror(errno));
    return 0;
  }
  if (p.status != NV_OK)
  {
    ERROR_LOG("rm_alloc(class=0x%x) rmStatus=0x%x", hclass, p.status);
    return 0;
  }
  return p.hObjectNew;
}

/*
 * Allocate GPU vidmem (HBM) using NV01_MEMORY_LOCAL_USER (class 0x40).
 *
 * Alloc params = NV_MEMORY_ALLOCATION_PARAMS (the flat struct in nvos.h,
 * NOT the legacy nested NVOS32_PARAMETERS).  RM writes the HBM physical
 * address back into mp.address on success; we stash it in *out_addr but
 * the channel/CE actually addresses vidmem via the GPU VA set up by UVM
 * on top of this handle.
 *
 * The flag bundle matches what libcuda sets for its 128-MiB device
 * buffers — ALLOW_NONCONTIGUOUS lets RM back the allocation with non-
 * contiguous physical pages; ALIGNMENT_FORCE pins mp.alignment;
 * MEMORY_HANDLE_PROVIDED tells RM we supply p.hObjectNew;
 * PERSISTENT_VIDMEM preserves the allocation across device reset.  A
 * single-flag deviation from the whole bundle measurably changed the
 * failure-rate distribution during debugging, so treat the bundle as
 * atomic.
 *
 * Parent MUST be the device handle (not the subdevice); HBM is
 * allocated at device granularity in the RM object tree.
 */
NvHandle rm_alloc_vidmem(int ctl_fd, NvHandle root, NvHandle device,
                                NvU64 size, NvU64 *out_addr)
{
  NV_MEMORY_ALLOCATION_PARAMS mp = {};
  NVOS64_PARAMETERS           p  = {};
  long                        r;

  mp.owner = root;
  mp.type  = NVOS32_TYPE_IMAGE;
  mp.flags = NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT
             | NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED
             | NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE
             | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED
             | NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM;
  mp.attr = DRF_DEF(OS32, _ATTR, _LOCATION, _VIDMEM)
            | DRF_DEF(OS32, _ATTR, _PHYSICALITY, _ALLOW_NONCONTIGUOUS);
  mp.size      = size;
  mp.alignment = MC_VIDMEM_ALIGN_BYTES;

  p.hRoot         = root;
  p.hObjectParent = device;
  p.hObjectNew    = next_handle();
  p.hClass        = NV01_MEMORY_LOCAL_USER;
  p.pAllocParms   = (NvP64)(uintptr_t)&mp;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_ALLOC & 0xff, NVOS64_PARAMETERS), &p);
  if (r != 0 || p.status != NV_OK)
  {
    ERROR_LOG("rm_alloc_vidmem failed: r=%ld status=0x%x", r, p.status);
    return 0;
  }
  if (out_addr)
    *out_addr = (NvU64)(NvUPtr)mp.address;
  return p.hObjectNew;
}

/*
 * REGISTER_FD links an nvidia0 fd to an nvidiactl fd so RM can find the
 * client context when an ioctl arrives on the device fd.  Every nvidia0
 * fd that ioctls will flow through must be registered once; the link
 * lives on the kernel-side file private until either fd closes.
 */
int rm_register_client_fd(int ctl_fd, int dev_fd)
{
  nv_ioctl_register_fd_t reg = { .ctl_fd = ctl_fd };
  long                   r;

  r = ioctl(dev_fd,
            _IOWR('F', NV_ESC_REGISTER_FD & 0xff, nv_ioctl_register_fd_t),
            &reg);
  if (r != 0)
  {
    ERROR_LOG("REGISTER_FD: %s", strerror(errno));
    return -1;
  }
  return 0;
}

/*
 * Allocate + pin host DRAM via NV01_MEMORY_SYSTEM (class 0x3e) using
 * NV_ESC_RM_ALLOC_MEMORY (escape 0x27) + nv_ioctl_nvos02_parameters_with_fd.
 * This helper both allocates the memory and mmaps it into userspace in a
 * single flow; *out_cpu_ptr receives the CPU VA.  Under UVM the same CPU
 * VA also becomes the GPU VA once the buffer goes through
 * UVM_MAP_EXTERNAL_ALLOCATION.
 *
 * Flag encoding (NVOS02_FLAGS bit positions from nvos.h):
 *   PHYSICALITY bits [7:4]   = NONCONTIGUOUS (1)
 *   COHERENCY   bits [15:12] = WRITE_COMBINE (2)
 *
 * Fd discipline (getting this wrong fails silently):
 *   - The ioctl goes to a REGISTER_FD'd nvidia0 fd (escape is marked
 *     NV_ACTUAL_DEVICE_ONLY in escape.c).
 *   - p.fd (the alloc fd) must be nvidiactl, because rm_create_mmap_
 *     context uses nv_get_ctl_state() to register the kernel mmap
 *     context.  We therefore open two fresh fds per allocation.
 *
 * If `want_addr != NULL`, mmap lands the VMA there with MAP_FIXED — this
 * is the libcuda pattern used to anchor the CPU alias inside the VA
 * pool (Paper F1 invariant: GPU VA == CPU VA).
 */
static NvHandle rm_alloc_sysmem_coh_at(int ctl_fd, int dev_fd, NvHandle root,
                                   NvHandle device, NvU64 size, void *want_addr,
                                   NvU32 coherency, void **out_cpu_ptr)
{
  nv_ioctl_nvos02_parameters_with_fd p = {};
  int                                alloc_fd, ioctl_fd;
  void                              *addr;
  int                                mmap_flags;
  long                               r;

  (void)dev_fd; /* kept in the signature for symmetry with the other
                 * rm_* wrappers, which do route through the device fd */

  p.params.hRoot         = root;
  p.params.hObjectParent = device;
  p.params.hObjectNew    = next_handle();
  p.params.hClass        = NV01_MEMORY_SYSTEM;
  p.params.flags         = DRF_DEF(OS02, _FLAGS, _PHYSICALITY, _NONCONTIGUOUS)
                   | DRF_NUM(OS02, _FLAGS, _COHERENCY, coherency);
  p.params.limit = size - 1;

  alloc_fd = open(MC_CONTROL_DEV_PATH, O_RDWR | O_CLOEXEC);
  ioctl_fd = open(MC_DEVICE_DEV_PATH, O_RDWR | O_CLOEXEC);
  if (alloc_fd < 0 || ioctl_fd < 0)
  {
    ERROR_LOG("rm_alloc_sysmem_at: open fds: %s", strerror(errno));
    if (alloc_fd >= 0) close(alloc_fd);
    if (ioctl_fd >= 0) close(ioctl_fd);
    return 0;
  }
  p.fd = alloc_fd;

  if (rm_register_client_fd(ctl_fd, ioctl_fd) != 0)
  {
    close(alloc_fd);
    close(ioctl_fd);
    return 0;
  }

  r = ioctl(ioctl_fd,
            _IOWR('F', NV_ESC_RM_ALLOC_MEMORY & 0xff,
                  nv_ioctl_nvos02_parameters_with_fd),
            &p);
  close(ioctl_fd);
  if (r != 0 || p.params.status != NV_OK)
  {
    ERROR_LOG("rm_alloc_sysmem_at failed: r=%ld status=0x%x", r,
              p.params.status);
    close(alloc_fd);
    return 0;
  }

  mmap_flags = MAP_SHARED;
  if (want_addr)
    mmap_flags |= MAP_FIXED;

  addr = mmap(want_addr, size, PROT_READ | PROT_WRITE, mmap_flags, alloc_fd, 0);
  if (addr == MAP_FAILED)
  {
    ERROR_LOG("sysmem mmap failed: %s", strerror(errno));
    /* Roll back the RM allocation too — the object exists at this point,
     * and the caller only learns "failed" (handle 0), so nothing else can
     * free it.  Mirrors the ioctl-failure path above. */
    rm_free_handle(ctl_fd, root, device, p.params.hObjectNew,
                   "sysmem_mmap_rollback");
    close(alloc_fd);
    return 0;
  }
  if (out_cpu_ptr)
    *out_cpu_ptr = addr;
  return p.params.hObjectNew;
}

/*
 * Allocate host memory that the CPU will actually read.
 *
 * COHERENCY_CACHED leaves the userspace PTE at the kernel's default
 * write-back type, so loads hit the cache hierarchy normally.  This is the
 * right default for anything a user program touches: on H100 the same
 * buffer reads at ~11.8 GB/s cached versus ~32 MB/s write-combined — a
 * ~360x difference on the read side — and cached is also the faster of the
 * two for sequential stores, because full-cache-line writes to pinned
 * pages beat write-combining.
 *
 * Note the ABI's default is NOT this: NVOS02_FLAGS_COHERENCY_UNCACHED is
 * 0, so an allocation that simply omits the field gets UC-.  The value has
 * to be asked for explicitly.  (WRITE_BACK is accepted too, but RM folds
 * CACHED / WRITE_THROUGH / WRITE_PROTECT / WRITE_BACK onto one internal
 * NV_MEMORY_CACHED, so the two are indistinguishable past that point.)
 */
NvHandle rm_alloc_sysmem_at(int ctl_fd, int dev_fd, NvHandle root,
                                   NvHandle device, NvU64 size, void *want_addr,
                                   void **out_cpu_ptr)
{
  return rm_alloc_sysmem_coh_at(ctl_fd, dev_fd, root, device, size, want_addr,
                                NVOS02_FLAGS_COHERENCY_CACHED, out_cpu_ptr);
}

/*
 * Write-combined variant, for memory the host writes and the GPU reads.
 *
 * Used for the control plane — pushbuffers, GPFIFO rings, USERD, release
 * semaphores, the compute QMD / CB0 / SASS images.  Those are produced by
 * the CPU in whole-buffer streaming stores and then consumed by the GPU;
 * WC suits that shape and keeps the pages out of the CPU's cache, so no
 * flush discipline is needed to make the GPU's view current.
 *
 * Do not reach for this for user data.  Anything the host reads back
 * belongs in rm_alloc_sysmem_at.
 */
NvHandle rm_alloc_sysmem_wc_at(int ctl_fd, int dev_fd, NvHandle root,
                                      NvHandle device, NvU64 size, void *want_addr,
                                      void **out_cpu_ptr)
{
  return rm_alloc_sysmem_coh_at(ctl_fd, dev_fd, root, device, size, want_addr,
                                NVOS02_FLAGS_COHERENCY_WRITE_COMBINE,
                                out_cpu_ptr);
}

/*
 * Register caller-owned host pages as an RM OS descriptor using
 * NV01_MEMORY_SYSTEM_OS_DESCRIPTOR (class 0x71).  This is the RM half of
 * cudaHostRegister for malloc memory: the driver pins/locks the already
 * existing user pages and returns an hMemory handle, but mc does not
 * mmap or otherwise allocate a new CPU alias.
 *
 * `page_base` and `page_covered_size` must describe the page-aligned range
 * covering the user's [ptr, ptr+n) interval.  UVM will later map sub-ranges
 * of this descriptor at CPU VA == GPU VA with offsets from this page_base.
 */
NvHandle rm_register_user_memory(int ctl_fd, int dev_fd, NvHandle root,
                                 NvHandle device, void *page_base,
                                 NvU64 page_covered_size)
{
  nv_ioctl_nvos02_parameters_with_fd p = {};
  int                                alloc_fd, ioctl_fd;
  long                               r;

  (void)dev_fd; /* kept for signature symmetry with rm_alloc_sysmem_at */

  if (page_base == NULL || page_covered_size == 0)
    return 0;

  p.params.hRoot         = root;
  p.params.hObjectParent = device;
  p.params.hObjectNew    = next_handle();
  p.params.hClass        = NV01_MEMORY_SYSTEM_OS_DESCRIPTOR;
  p.params.flags         = DRF_DEF(OS02, _FLAGS, _LOCATION, _PCI)
                   | DRF_DEF(OS02, _FLAGS, _MAPPING, _NO_MAP)
                   | DRF_DEF(OS02, _FLAGS, _PHYSICALITY, _NONCONTIGUOUS)
                   | DRF_DEF(OS02, _FLAGS, _COHERENCY, _WRITE_BACK);
  p.params.pMemory = (NvP64)(uintptr_t)page_base;
  p.params.limit   = page_covered_size - 1;

  alloc_fd = open(MC_CONTROL_DEV_PATH, O_RDWR | O_CLOEXEC);
  ioctl_fd = open(MC_DEVICE_DEV_PATH, O_RDWR | O_CLOEXEC);
  if (alloc_fd < 0 || ioctl_fd < 0)
  {
    ERROR_LOG("rm_register_user_memory: open fds: %s", strerror(errno));
    if (alloc_fd >= 0) close(alloc_fd);
    if (ioctl_fd >= 0) close(ioctl_fd);
    return 0;
  }
  p.fd = alloc_fd;

  if (rm_register_client_fd(ctl_fd, ioctl_fd) != 0)
  {
    close(alloc_fd);
    close(ioctl_fd);
    return 0;
  }

  r = ioctl(ioctl_fd,
            _IOWR('F', NV_ESC_RM_ALLOC_MEMORY & 0xff,
                  nv_ioctl_nvos02_parameters_with_fd),
            &p);
  close(ioctl_fd);
  close(alloc_fd);
  if (r != 0 || p.params.status != NV_OK)
  {
    ERROR_LOG("rm_register_user_memory failed: r=%ld status=0x%x base=%p size=0x%llx",
              r, p.params.status, page_base,
              (unsigned long long)page_covered_size);
    return 0;
  }

  DEBUG_LOG("rm_register_user_memory: h=0x%x base=%p size=0x%llx",
            p.params.hObjectNew, page_base,
            (unsigned long long)page_covered_size);
  return p.params.hObjectNew;
}

/*
 * Allocate a FERMI_VASPACE_A (class 0x90f1) GPU VA space object.
 *
 * This VA space becomes the channel's GPU MMU page-table root.  Buffers
 * that the channel accesses (pushbuffer, src/dst, sema, GPFIFO ring) are
 * mapped into it by UVM.
 *
 * The IS_EXTERNALLY_OWNED flag is REQUIRED for UVM compatibility.  It
 * tells RM that an external driver (UVM) will manage the page tables.
 * Without this flag, UVM_REGISTER_GPU_VASPACE — which internally calls
 * nvGpuOpsDupAddressSpace — fails with NV_ERR_INVALID_FLAGS (0x29)
 * because nvGpuOpsDupAddressSpace in nv_gpu_ops.c rejects VA spaces that aren't externally-
 * owned.
 *
 * index = GPU_NEW requests a fresh VA space (vs. inheriting one).
 */
NvHandle rm_alloc_vaspace(int ctl_fd, NvHandle root, NvHandle device)
{
  NV_VASPACE_ALLOCATION_PARAMETERS vp = {};
  NVOS64_PARAMETERS                p  = {};
  long                             r;

  vp.index = NV_VASPACE_ALLOCATION_INDEX_GPU_NEW;
  vp.flags = NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED;

  p.hRoot         = root;
  p.hObjectParent = device;
  p.hObjectNew    = next_handle();
  p.hClass        = FERMI_VASPACE_A;
  p.pAllocParms   = (NvP64)(uintptr_t)&vp;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_ALLOC & 0xff, NVOS64_PARAMETERS), &p);
  if (r != 0 || p.status != NV_OK)
  {
    ERROR_LOG("rm_alloc_vaspace failed: r=%ld status=0x%x", r, p.status);
    return 0;
  }
  return p.hObjectNew;
}

/*
 * Variant of rm_alloc_vaspace that drops IS_EXTERNALLY_OWNED.  Used by
 * the second (DMA-only) channel that doesn't talk to UVM.
 *
 * Why a separate VAS: NV04_MAP_MEMORY_DMA (and the underlying
 * gvaspaceAlloc_IMPL in gpu_vaspace.c) refuses to allocate VA on
 * any externally-owned VAS — even with NVOS46_FLAGS_DMA_OFFSET_FIXED.
 * The UVM channel's VAS must be externally-owned (UVM contract); the
 * carrier VAS used by DMA + compute channels must NOT be (DMA-map
 * contract).  Two VAS kinds, no conflict.
 *
 * Note: NV04_MAP_MEMORY_DMA actually expects `hDma` to be an
 * NV50_MEMORY_VIRTUAL handle (with hVASpace pointing here), NOT this
 * VAS handle directly.  In the open tree, VaSpaceApi has no
 * `mapTo` override, so passing the VAS handle as `hDma` returns
 * NV_ERR_INVALID_OBJECT_HANDLE (resMapTo_IMPL default).  See
 * rm_alloc_virtual_memory() below for the carrier allocation.
 */
NvHandle rm_alloc_vaspace_dma(int ctl_fd, NvHandle root,
                                     NvHandle device)
{
  NV_VASPACE_ALLOCATION_PARAMETERS vp = {};
  NVOS64_PARAMETERS                p  = {};
  long                             r;

  vp.index = NV_VASPACE_ALLOCATION_INDEX_GPU_NEW;
  vp.flags = 0; /* deliberately NOT IS_EXTERNALLY_OWNED */

  p.hRoot         = root;
  p.hObjectParent = device;
  p.hObjectNew    = next_handle();
  p.hClass        = FERMI_VASPACE_A;
  p.pAllocParms   = (NvP64)(uintptr_t)&vp;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_ALLOC & 0xff, NVOS64_PARAMETERS), &p);
  if (r != 0 || p.status != NV_OK)
  {
    ERROR_LOG("rm_alloc_vaspace_dma failed: r=%ld status=0x%x", r, p.status);
    return 0;
  }
  return p.hObjectNew;
}

/*
 * Allocate an NV50_MEMORY_VIRTUAL "carrier" inside the given VAS.
 *
 * This is the open-tree mapper-side handle for NV04_MAP_MEMORY_DMA.
 * VaSpaceApi itself has no resMapTo override (resMapTo_IMPL returns
 * NV_ERR_INVALID_OBJECT_HANDLE / 0x33), but VirtualMemory does
 * (virtmemMapTo_IMPL in virtual_mem.c).  So MAP_MEMORY_DMA's
 * `hDma` argument must be an NV50_MEMORY_VIRTUAL handle whose
 * `hVASpace` points at our actual VAS — a carrier-of-VAS.
 *
`mc` allocates one carrier per mapped resource, sized to that resource,
 * which is the shape libcuda uses (`dmaOffset=0`, RM-chosen GPU VA).
 * An earlier design shared a single oversized carrier across every
 * mapping; it did not match libcuda and wedged the FB-resident control
 * plane, so `size` here is the caller's resource size.
 */
NvHandle rm_alloc_virtual_memory(int ctl_fd, NvHandle root,
                                        NvHandle device, NvHandle hvaspace,
                                        NvU64 size, NvU64 *out_gpu_va_base)
{
  NV_MEMORY_ALLOCATION_PARAMS mp = {};
  NVOS64_PARAMETERS           p  = {};
  long                        r;

  mp.owner = root;
  mp.type  = NVOS32_TYPE_IMAGE;
  mp.flags = NVOS32_ALLOC_FLAGS_VIRTUAL
             | NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT
             | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED;
  mp.attr  = DRF_DEF(OS32, _ATTR, _LOCATION, _PCI)
             | DRF_DEF(OS32, _ATTR, _PHYSICALITY, _NONCONTIGUOUS)
             | DRF_DEF(OS32, _ATTR, _PAGE_SIZE, _DEFAULT);
  mp.size      = size;
  mp.alignment = 0x10000ULL;       /* 64 KiB */
  mp.hVASpace  = hvaspace;

  p.hRoot         = root;
  p.hObjectParent = device;
  p.hObjectNew    = next_handle();
  p.hClass        = NV50_MEMORY_VIRTUAL;
  p.pAllocParms   = (NvP64)(uintptr_t)&mp;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_ALLOC & 0xff, NVOS64_PARAMETERS), &p);
  if (r != 0 || p.status != NV_OK)
  {
    ERROR_LOG("rm_alloc_virtual_memory failed: r=%ld status=0x%x", r, p.status);
    return 0;
  }

  /* mp.offset is [IN/OUT] — RM writes back the chosen GPU VA base. */
  if (out_gpu_va_base)
    *out_gpu_va_base = mp.offset;
  return p.hObjectNew;
}

/*
 * Install a GPU MMU PTE that maps `hmemory` at a RM-chosen GPU VA
 * inside the VAS that backs `hvirt`.  Returns the GPU VA
 * (p.dmaOffset).  Used by the carrier VAS for everything: gpfifo+userd,
 * pushbuffer, semaphore, and crucially the BAR1 doorbell page (whose
 * hMemory is HOPPER_USERMODE_A — RM substitutes pBar1VF via the
 * usrmodeGetMemInterMapParams_IMPL override, so the resulting PTE
 * resolves GPU-VA → real BAR1).
 *
 * `hvirt` must be an NV50_MEMORY_VIRTUAL allocation (see
 * rm_alloc_virtual_memory).  Passing a raw VAS handle here returns
 * 0x33 (INVALID_OBJECT_HANDLE) because VaSpaceApi has no resMapTo
 * override — the inter-map dispatch falls through to the default.
 *
 * flags = 0 lets RM allocate the VA itself; we have no Paper-F1
 * invariant to satisfy here (no UVM, no CPU alias for the GPU VA).
 */
NvU64 rm_map_memory_dma(int ctl_fd, NvHandle client, NvHandle device,
                               NvHandle hvirt, NvHandle hmemory,
                               NvU64 offset, NvU64 length, NvU64 want_gpu_va)
{
  NVOS46_PARAMETERS p = {};
  long              r;

  p.hClient = client;
  p.hDevice = device;   /* device handle, NOT subdevice */
  p.hDma    = hvirt;    /* NV50_MEMORY_VIRTUAL carrier */
  p.hMemory = hmemory;
  p.offset  = offset;
  p.length  = length;
  /* When want_gpu_va != 0, set DMA_OFFSET_FIXED_TRUE and supply
   * `dmaOffset` as the requested GPU VA.  Forces RM to skip its
   * heap allocator (which empirically returns 0x1f INVALID_ARGUMENT
   * after the first map on this VAS) and just install the PTE at
   * the requested address inside the VirtualMemory carrier. */
  if (want_gpu_va != 0)
  {
    p.flags     = DRF_DEF(OS46, _FLAGS, _DMA_OFFSET_FIXED, _TRUE);
    p.dmaOffset = want_gpu_va;
  }
  else
  {
    p.flags = 0;
  }

  r = ioctl(ctl_fd,
            _IOWR('F', NV_ESC_RM_MAP_MEMORY_DMA & 0xff, NVOS46_PARAMETERS),
            &p);
  if (r != 0 || p.status != NV_OK)
  {
    ERROR_LOG("rm_map_memory_dma(hMemory=0x%x) failed: r=%ld status=0x%x",
              hmemory, r, p.status);
    return 0;
  }
  return p.dmaOffset;
}

/*
 * Counterpart to rm_map_memory_dma — issues NV04_UNMAP_MEMORY_DMA
 * (escape 0x58 / NVOS47).  Best-effort: returns void, just WARNs on
 * failure since teardown can't usefully recover.
 */
void rm_unmap_memory_dma(int ctl_fd, NvHandle client, NvHandle device,
                                NvHandle hvirt, NvHandle hmemory,
                                NvU64 dma_offset)
{
  NVOS47_PARAMETERS p = {};
  long              r;

  p.hClient   = client;
  p.hDevice   = device;
  p.hDma      = hvirt;
  p.hMemory   = hmemory;
  p.flags     = 0;
  p.dmaOffset = dma_offset;

  r = ioctl(ctl_fd,
            _IOWR('F', NV_ESC_RM_UNMAP_MEMORY_DMA & 0xff, NVOS47_PARAMETERS),
            &p);
  if (r != 0 || p.status != NV_OK)
    WARN_LOG("rm_unmap_memory_dma(hMemory=0x%x) r=%ld status=0x%x", hmemory, r,
             p.status);
}

/*
 * Map a memory object into userspace using NV_ESC_RM_MAP_MEMORY (escape
 * 0x4e).
 *
 * Verified pattern from real-CUDA strace:
 *   1. Open a fresh /dev/nvidia0 fd dedicated to this one mapping.
 *   2. ioctl(ctl_fd, esc=0x4e, {p.fd = fresh_fd, ...})
 *   3. mmap(pLinearAddress, length, prot, MAP_SHARED|MAP_FIXED, fresh_fd, 0)
 *      - offset is ALWAYS 0 (the physical mapping is encoded in the
 *        kernel mmap context attached to fresh_fd).
 *      - fresh_fd is "consumed" by nvidia_mmap and must not be reused.
 *
 * If `want_va != NULL`, mmap lands the VMA there (libcuda pattern for
 * anchoring CPU aliases into the VA pool — Paper F1).
 */
void *rm_map_memory_at(int ctl_fd, const char *dev_path, NvHandle client,
                              NvHandle device, NvHandle hmem, NvU64 offset,
                              NvU64 length, NvU32 flags, void *want_addr)
{
  nv_ioctl_nvos33_parameters_with_fd p = {};
  void                              *addr;
  int                                map_fd;
  int                                mmap_flags;
  long                               r;

  map_fd = open(dev_path, O_RDWR | O_CLOEXEC);
  if (map_fd < 0)
  {
    ERROR_LOG("rm_map_memory: open(%s): %s", dev_path, strerror(errno));
    return NULL;
  }

  p.params.hClient = client;
  p.params.hDevice = device;
  p.params.hMemory = hmem;
  p.params.offset  = offset;
  p.params.length  = length;
  p.params.flags   = flags;
  p.fd             = map_fd;

  r = ioctl(ctl_fd,
            _IOWR('F', NV_ESC_RM_MAP_MEMORY & 0xff,
                  nv_ioctl_nvos33_parameters_with_fd),
            &p);
  if (r != 0 || p.params.status != NV_OK)
  {
    ERROR_LOG("rm_map_memory: client=0x%x device=0x%x mem=0x%x len=%zu "
              "status=0x%x",
              client, device, hmem, (size_t)length, p.params.status);
    close(map_fd);
    return NULL;
  }

  mmap_flags = MAP_SHARED;
  if (want_addr)
    mmap_flags |= MAP_FIXED;

  addr = mmap(want_addr, length, PROT_READ | PROT_WRITE, mmap_flags, map_fd, 0);
  if (addr == MAP_FAILED)
  {
    ERROR_LOG("mmap(MAP_FIXED, %p, len=%zu): %s", want_addr, (size_t)length,
              strerror(errno));
    return NULL;
  }
  return addr;
}

/* Allocate a vidmem region and return both an hMemory + a host-visible
 * BAR1-aliased CPU mapping of it.  Used by the FB-carrier VAS to back
 * channel resources (PB / GPFIFO / USERD / sema) with FB pages while
 * still letting the host read/write via BAR1 for one-time setup
 * pushbuffer writes and failure-path diagnostics.
 *
 * Backing path:
 *   - rm_alloc_vidmem allocates the FB region (NV01_MEMORY_LOCAL_USER).
 *   - rm_map_memory_at with NVOS33_FLAGS_MAPPING_REFLECTED produces a
 *     BAR1 alias the kernel kbus layer routes through pBar1VF.
 *
 * The host-visible CPU pointer's load/store traffic on the hot path is
 * minimal: PBDMA reads PB/GPFIFO/USERD through the GPU MMU PTE, not
 * BAR1, and the SM-author kernel writes them through the same path.
 * BAR1 is only used at bring-up (host-built SET_OBJECT pushbuffer) and
 * for diagnostics.
 *
 * If `want_va` is non-NULL, MAP_FIXED anchors the BAR1 mmap at that
 * address (Paper-F1 VA pool placement).  Returns the new vidmem
 * hMemory on success and stores the BAR1 CPU VA in *out_cpu_va.  On
 * failure returns 0; the vidmem allocation is rolled back. */
NvHandle rm_alloc_vidmem_bar1_at(int ctl_fd, const char *dev_path,
                                 NvHandle client, NvHandle device,
                                 NvU64 size, void *want_va,
                                 void **out_cpu_va)
{
  NvHandle h_mem;
  void    *cpu;

  if (out_cpu_va == NULL) return 0;

  h_mem = rm_alloc_vidmem(ctl_fd, client, device, size, NULL);
  if (h_mem == 0) return 0;

  cpu = rm_map_memory_at(ctl_fd, dev_path, client, device, h_mem,
                         /*offset=*/0, size,
                         DRF_NUM(OS33, _FLAGS, _MAPPING,
                                 NVOS33_FLAGS_MAPPING_REFLECTED),
                         want_va);
  if (cpu == NULL)
  {
    /* Roll back: the vidmem alloc lives independently of the mapping
     * so we can free it directly without an unmap step. */
    rm_free_handle(ctl_fd, client, device, h_mem,
                   "rm_alloc_vidmem_bar1_at_rollback");
    return 0;
  }

  *out_cpu_va = cpu;
  return h_mem;
}

/* Convenience wrapper: rm_map_memory without pre-reserved want_va — RM's
 * pLinearAddress is used as the mmap target.  Used for the two
 * HOPPER_USERMODE_A register windows, which are the only mappings that
 * do not need the Paper-F1 identity VA (no GPU-side alias of them
 * exists); every buffer with a GPU alias goes through rm_map_memory_at. */
void *rm_map_memory(int ctl_fd, const char *dev_path,
                                     NvHandle client, NvHandle device,
                                     NvHandle hmem, NvU64 offset, NvU64 length,
                                     NvU32 flags)
{
  return rm_map_memory_at(ctl_fd, dev_path, client, device, hmem, offset,
                          length, flags, NULL);
}

/* Issue a subdevice/channel control call (NV_ESC_RM_CONTROL, escape 0x2a).
 * Returns 0 on success, -1 on syscall error or non-OK RM status. */
int rm_control(int ctl_fd, NvHandle client, NvHandle object, NvU32 cmd,
                      void *params, NvU32 params_size)
{
  NVOS54_PARAMETERS p = {};
  long              r;

  p.hClient    = client;
  p.hObject    = object;
  p.cmd        = cmd;
  p.params     = (NvP64)(uintptr_t)params;
  p.paramsSize = params_size;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_CONTROL & 0xff, NVOS54_PARAMETERS),
            &p);
  if (r != 0)
  {
    ERROR_LOG("rm_control(cmd=0x%x) syscall: %s", cmd, strerror(errno));
    return -1;
  }
  if (p.status != NV_OK)
  {
    ERROR_LOG("rm_control(cmd=0x%x) rmStatus=0x%x", cmd, p.status);
    return -1;
  }
  return 0;
}

/*
 * Pick a Logical Copy Engine (LCE) whose runlist isn't shared with GR.
 *
 * Hopper exposes several LCEs.  Some (the "GRCEs") share a runlist with
 * the graphics engine; channels on those get misclassified by the
 * upstream runlistId → first-engine lookup, which picks GR first on
 * runlist 0.  UVM_REGISTER_CHANNEL then rejects the channel with
 * NV_ERR_INVALID_OBJECT (0x31).
 *
 * Picking the first non-GRCE LCE is what libcuda does on H100 PCIe and
 * keeps the upstream engine-classification path unambiguous.  Must be
 * called AFTER allocating the subdevice and BEFORE allocating the TSG
 * (whose engineType we then set to the returned value).  The same value
 * must flow into the CE object alloc too — cross-checked at
 * the TSG engine-Id cross-check in kernel_channel.c.
 *
 * Returns NV2080_ENGINE_TYPE_COPY0 + <lce_index>, or (NvU32)-1 if no
 * non-GRCE LCE is present (not expected on Hopper).
 */
NvU32 pick_non_grce_lce(int ctl_fd, NvHandle h_client,
                               NvHandle h_subdevice)
{
  NV2080_CTRL_CE_GET_ALL_CAPS_PARAMS caps = {};
  NvU32                              lce;
  NvU32                              first_non_grce = (NvU32)-1;

  if (rm_control(ctl_fd, h_client, h_subdevice, NV2080_CTRL_CMD_CE_GET_ALL_CAPS,
                 &caps, sizeof(caps)) != 0)
    return (NvU32)-1;

  for (lce = 0; lce < NV2080_CTRL_MAX_CES; lce++)
  {
    NvBool grce;
    if (!((caps.present >> lce) & 1))
      continue;
    grce = !!NV2080_CTRL_CE_GET_CAP(caps.capsTbl[lce],
                                    NV2080_CTRL_CE_CAPS_CE_GRCE);
    if (!grce && first_non_grce == (NvU32)-1)
      first_non_grce = lce;
  }

  if (first_non_grce == (NvU32)-1)
  {
    ERROR_LOG("no non-GRCE LCE present (present=0x%llx)",
              (unsigned long long)caps.present);
    return (NvU32)-1;
  }
  return NV2080_ENGINE_TYPE_COPY0 + first_non_grce;
}

/* ── RM object type helpers ────────────────────────────────────────────────
 *
 * Thin typed wrappers around rm_alloc() that each encode the fields and
 * invariants that matter for one RM class.  Callers don't have to remember
 * the struct layout or the kernel cross-checks; they get a function per
 * object type and the wrapper sets everything up correctly.  If an
 * invariant moves or a field name changes upstream, a single wrapper edit
 * fixes every call site.
 */

/*
 * Allocate the RM client root (NV01_ROOT).  This is a special alloc:
 * hRoot = hObjectParent = hObjectNew = h_client all refer to the same
 * handle; RM builds the self-referential client object.  We pick
 * h_client up front from next_handle() (since it has to be known
 * before the alloc) and pass it into all three slots.
 */
NvHandle rm_alloc_root(int ctl_fd)
{
  NV0000_ALLOC_PARAMETERS ap = {};
  NVOS64_PARAMETERS       p  = {};
  long                    r;
  NvHandle                h_client = next_handle();

  ap.hClient      = h_client;
  p.hRoot         = h_client;
  p.hObjectParent = h_client;
  p.hObjectNew    = h_client;
  p.hClass        = NV01_ROOT;
  p.pAllocParms   = (NvP64)(uintptr_t)&ap;

  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_ALLOC & 0xff, NVOS64_PARAMETERS), &p);
  if (r != 0 || p.status != NV_OK)
  {
    ERROR_LOG("NV01_ROOT alloc failed: syscall=%ld status=0x%x", r, p.status);
    return 0;
  }
  return h_client;
}

/* Allocate the NV01_DEVICE_0 handle (deviceId=0 — our only GPU). */
NvHandle rm_alloc_device(int ctl_fd, NvHandle root)
{
  NV0080_ALLOC_PARAMETERS dev_ap = { .deviceId = 0 };
  return rm_alloc(ctl_fd, root, root, NV01_DEVICE_0, &dev_ap);
}

/* Allocate the NV20_SUBDEVICE_0 handle (subDeviceId=0 — non-MIG). */
NvHandle rm_alloc_subdevice(int ctl_fd, NvHandle root, NvHandle device)
{
  NV2080_ALLOC_PARAMETERS sub_ap = { .subDeviceId = 0 };
  return rm_alloc(ctl_fd, root, device, NV20_SUBDEVICE_0, &sub_ap);
}

/*
 * Ask the RM to hold the GPU at boost clocks for the life of this
 * client — NV2080_CTRL_CMD_PERF_BOOST on the subdevice, exactly the
 * call libcuda makes once during context creation (snooped on H100:
 * flags=0x12 = CMD_BOOST_TO_MAX | CUDA_YES, duration=INFINITE; the RM
 * clears the boost when the client is freed).
 *
 * Without it the GPU stays at its idle SM clock (345 MHz on H100 PCIe)
 * for a pure-copy workload, and the Copy Engine — clocked from the same
 * domain — moves data at ~2/3 of the PCIe line rate: measured 7.13 ms
 * vs 4.91 ms per 256 MiB D2H, in discrete per-iteration modes.  The
 * copies themselves never pull utilisation high enough for the
 * auto-boost governor to ramp, so the floor is stable, not a warm-up.
 *
 * Best-effort by design: on failure we log and carry on, since the
 * library works at idle clocks — just slower.
 */
void rm_perf_boost(int ctl_fd, NvHandle client, NvHandle subdevice)
{
  NV2080_CTRL_PERF_BOOST_PARAMS boost = {
    .flags    = DRF_DEF(2080, _CTRL_PERF_BOOST_FLAGS, _CMD, _BOOST_TO_MAX)
              | DRF_DEF(2080, _CTRL_PERF_BOOST_FLAGS, _CUDA, _YES),
    .duration = NV2080_CTRL_PERF_BOOST_DURATION_INFINITE,
  };
  if (rm_control(ctl_fd, client, subdevice, NV2080_CTRL_CMD_PERF_BOOST,
                 &boost, sizeof(boost)) != 0)
    ERROR_LOG("perf boost request failed; continuing at default clocks");
}

/*
 * Allocate HOPPER_USERMODE_A (class 0xC661) — a 64 KiB register window
 * that exposes the VF doorbell to userspace.  Two variants exist:
 *   - bar1 == false: BAR0 mapping (pa=0x1fe...).  Allocated for
 *     structural parity with libcuda (which allocates both) but never
 *     written to after setup.
 *   - bar1 == true:  BAR1 mapping (pa=0x1fc...).  The actual doorbell.
 *     libcuda writes 194/194 doorbells here; we do the same.
 * Both have identical VF_DOORBELL offset (+0x90 within the window).
 */
NvHandle rm_alloc_usermode(int ctl_fd, NvHandle root, NvHandle subdevice,
                                  bool bar1)
{
  NV_HOPPER_USERMODE_A_PARAMS um_params = {
    .bBar1Mapping = bar1,
    .bPriv        = NV_FALSE,
  };
  return rm_alloc(ctl_fd, root, subdevice, HOPPER_USERMODE_A, &um_params);
}

/*
 * Allocate a KEPLER_CHANNEL_GROUP_A (TSG, class 0xA06C).  Every channel
 * on Volta+ must be a TSG member; the TSG carries the engine type and
 * VA space that the channel inherits.  engine_type MUST match the CE
 * object's engineType at CE-alloc time (the TSG engine-Id
 * cross-check in kernel_channel.c), so we plumb the same value
 * through pick_non_grce_lce → rm_alloc_tsg → rm_alloc_channel →
 * rm_alloc_ce.
 */
NvHandle rm_alloc_tsg(int ctl_fd, NvHandle root, NvHandle device,
                             NvU32 engine_type, NvHandle vaspace)
{
  NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS tsg_params = {
    .engineType = engine_type,
    .hVASpace   = vaspace,
  };
  return rm_alloc(ctl_fd, root, device, KEPLER_CHANNEL_GROUP_A, &tsg_params);
}

/*
 * Allocate a HOPPER_CHANNEL_GPFIFO_A (class 0xC86F) parented on a TSG.
 * Two invariants:
 *   - engine_type must match the TSG's engineType (the TSG engine-Id
 *     cross-check in kernel_channel.c).
 *   - hVASpace and hHandleVASpace MUST be zero when the parent is a
 *     TSG — the channel inherits the TSG's VA space.  kernel_channel.c
 *     rejects non-zero channel-level VAS fields for TSG-parented
 *     channels.  We never set them.
 * gp_fifo_offset is the GPU VA of the GPFIFO ring (under UVM, equals
 * the CPU VA — Paper F1).  h_userd + userd_offset identify the
 * channel's USERD slot inside a caller-provided memory handle (the
 * library packs GPFIFO at offset 0 and USERD at +MC_USERD_OFFSET in
 * one gpfifo+userd alloc).
 */
NvHandle rm_alloc_channel(int ctl_fd, NvHandle root, NvHandle tsg,
                                 NvU64 gp_fifo_offset, NvU32 gp_fifo_entries,
                                 NvU32 engine_type, NvHandle h_userd,
                                 NvU64 userd_offset)
{
  NV_CHANNEL_ALLOC_PARAMS chan_params = {
    .gpFifoOffset    = gp_fifo_offset,
    .gpFifoEntries   = gp_fifo_entries,
    .engineType      = engine_type,
    .hUserdMemory[0] = h_userd,
    .userdOffset[0]  = userd_offset,
    .hVASpace        = 0,
    .hHandleVASpace  = 0,
  };
  return rm_alloc(ctl_fd, root, tsg, HOPPER_CHANNEL_GPFIFO_A, &chan_params);
}

/*
 * Allocate a HOPPER_DMA_COPY_A (CE engine object, class 0xC8B5)
 * parented on a channel.  Passing NULL alloc params on Hopper silently
 * resets runlistId to 0 (GR-shared); we explicitly pass
 * NVB0B5_ALLOCATION_PARAMETERS with VERSION_1 + engine_type matching
 * the TSG/channel.  libcuda does this too on H100 PCIe.
 */
NvHandle rm_alloc_ce(int ctl_fd, NvHandle root, NvHandle channel,
                            NvU32 engine_type)
{
  NVB0B5_ALLOCATION_PARAMETERS ce_params = {
    .version    = NVB0B5_ALLOCATION_PARAMETERS_VERSION_1,
    .engineType = engine_type,
  };
  return rm_alloc(ctl_fd, root, channel, HOPPER_DMA_COPY_A, &ce_params);
}

/*
 * Allocate a HOPPER_COMPUTE_A (compute engine object, class 0xCBC0)
 * parented on a channel.  resource_list.h marks NV_GR_ALLOCATION_PARAMETERS
 * as RS_OPTIONAL so we can pass NULL — but libcuda always passes a
 * filled-in struct (version=2, size=sizeof, flags=0).  Match that.
 */
NvHandle rm_alloc_compute(int ctl_fd, NvHandle root, NvHandle channel)
{
  NV_GR_ALLOCATION_PARAMETERS gr_params = {
    .version = 2,
    .flags   = 0,
    .size    = sizeof(NV_GR_ALLOCATION_PARAMETERS),
    .caps    = 0,
  };
  return rm_alloc(ctl_fd, root, channel, HOPPER_COMPUTE_A, &gr_params);
}

/*
 * NVA06F_CTRL_CMD_GPFIFO_SCHEDULE with bEnable=1 places the channel's
 * TSG on the engine's runlist.  After this call, PBDMA will dispatch
 * the channel once the VF doorbell wakes the scheduler.
 */
int rm_gpfifo_schedule(int ctl_fd, NvHandle h_client, NvHandle h_channel)
{
  NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS sched = { .bEnable = NV_TRUE };
  return rm_control(ctl_fd, h_client, h_channel,
                    NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, &sched, sizeof(sched));
}

/*
 * Counterpart to rm_gpfifo_schedule — bEnable=FALSE removes the channel
 * from the runlist.  Required before freeing the channel handle,
 * otherwise RM client teardown may race with in-flight GPU work.
 *
 * Teardown-tolerant: inlines the RM_CONTROL ioctl rather than calling
 * rm_control() (which returns -1 on failure but WE want to keep going
 * even on non-OK status) and WARNs on error.
 */
void rm_channel_disable(int ctl_fd, NvHandle h_client, NvHandle h_channel)
{
  NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS sched = {};
  NVOS54_PARAMETERS                  p     = {};
  long                               r;
  sched.bEnable = NV_FALSE;
  p.hClient     = h_client;
  p.hObject     = h_channel;
  p.cmd         = NVA06F_CTRL_CMD_GPFIFO_SCHEDULE;
  p.params      = (NvP64)(uintptr_t)&sched;
  p.paramsSize  = sizeof(sched);
  r = ioctl(ctl_fd, _IOWR('F', NV_ESC_RM_CONTROL & 0xff, NVOS54_PARAMETERS),
            &p);
  if (r != 0 || p.status != NV_OK)
    WARN_LOG("rm_channel_disable rmStatus=0x%x", p.status);
}
