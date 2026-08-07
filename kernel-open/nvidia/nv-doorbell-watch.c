/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Kernel-side Hopper VF_DOORBELL watchpoint.
 *
 * The mechanism: install a shadow page in place of the real VF
 * window (BAR1 for the variant libcuda and mc actually ring; the hook
 * keys on size + numRanges so it covers the BAR0 variant too), arm an
 * x86 hardware watchpoint via the nvidia-dbell.ko GPL
 * shim, and in the #DB handler walk KernelChannel -> USERD -> GPFIFO ->
 * pushbuffer, emit the decoded submission, then forward the write to
 * the real doorbell MMIO.
 *
 * Key invariant: on every doorbell write we must ALWAYS end by
 * writing the original value to the real doorbell MMIO, or the GPU hangs.
 */

#include "os-interface.h"
#include "nv-linux.h"
#include "conftest.h"
#include "nv-doorbell-watch.h"
#include "mc-trace.h"
#include "nv-kthread-q.h"

#include <linux/perf_event.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/sched/signal.h>  /* for_each_thread */

/*
 * The global deferred-work kthread queue lives in nv.c.  It has no
 * header-level extern; declare locally.  nv_kthread_q_t is provided by
 * nv-kthread-q.h (included above).
 */
extern nv_kthread_q_t nv_kthread_q;

/*
 * Forward-declare just the one RM-side helper we call rather than
 * including rm-gpu-ops.h.  That header declares interfaces in terms of
 * UVM types (UvmCslContext, UVM_ACCESS_BITS_DUMP_MODE) that are not
 * visible in this translation unit, so including it genuinely fails to
 * compile — verified, not assumed.
 *
 * The cost is that this declaration and the one in rm-gpu-ops.h are
 * maintained separately and nothing makes the compiler compare them.
 * Both must match the definition in
 * src/nvidia/arch/nvalloc/unix/src/rm-gpu-ops.c — change all three
 * together.
 *
 * Returns USERD phys+size+addrspace; we do the phys→kva mapping
 * locally via the BAR1 tracker.
 */
extern NV_STATUS NV_API_CALL
rm_gpu_ops_dbell_resolve_channel(nvidia_stack_t *sp,
                                 nv_state_t *pNv,
                                 NvU32 chid,
                                 NvU32 runlist,
                                 void **out_userd_kva,
                                 NvU64 *out_userd_phys,
                                 NvU64 *out_userd_size,
                                 NvU32 *out_addrspace,
                                 NvU64 *out_gpfifo_gpu_va,
                                 NvU32 *out_gpfifo_entries);

/* Forward declaration — defined by the GPL shim module nvidia-dbell.ko,
 * exported as plain EXPORT_SYMBOL so this MIT file can link. */
typedef void (*nv_dbell_overflow_fn)(struct perf_event *bp,
                                     struct perf_sample_data *data,
                                     struct pt_regs *regs);

extern int nv_dbell_bp_register(struct task_struct *task,
                                unsigned long user_va,
                                nv_dbell_overflow_fn cb,
                                void *ctx,
                                struct perf_event **out_event);

extern void nv_dbell_bp_unregister(struct perf_event *event);

/*
 * Per-VMA context.  Hung off vma->vm_private_data with `tag` as the
 * first field so the release path can safely distinguish us from the
 * existing nv_alloc_t* usage (nv_alloc_t has its own layout and is
 * only used for sysmem control-node mappings — device-node BAR0
 * mappings normally leave vm_private_data == NULL).
 */
/*
 * CUDA's libcuda uses multiple worker threads and submits doorbell
 * writes from threads OTHER than the one that called mmap.
 * register_user_hw_breakpoint binds to a single task_struct, so to
 * catch cross-thread writes we install one BP per thread in the
 * thread-group at arm time.
 */
#define NV_DBELL_MAX_BP_PER_CTX  64

typedef struct nv_dbell_ctx_s {
    unsigned long        tag;            /* == NV_DBELL_VMA_TAG */
    unsigned int         slot_idx;
    struct perf_event   *bp_events[NV_DBELL_MAX_BP_PER_CTX];
    unsigned int         bp_event_count;
    nv_state_t          *nv;
    struct page         *shadow_page;    /* alloc_page() — single 4 KiB page */
    void                *shadow_kva;     /* page_address(shadow_page) */
    unsigned long        user_va;        /* userspace base of the VMA */
    void __iomem        *doorbell_iomap; /* real BAR0 +NV_DBELL_OFFSET */
    struct task_struct  *owner;
    atomic_t             bp_count;       /* total doorbell events observed */
} nv_dbell_ctx_t;

/*
 * Global slot table.  x86 HW breakpoints are scarce (DR0..DR3), so we
 * cap the number of concurrently watched mappings at 4 and serve
 * additional mmaps with a pass-through (normal BAR0 remap).
 */
typedef struct nv_dbell_slot_s {
    atomic_t         claimed;    /* 0=free, 1=in-use; CAS to claim */
    nv_dbell_ctx_t  *ctx;        /* only valid while claimed==1 */
} nv_dbell_slot_t;

static nv_dbell_slot_t g_dbell_slots[NV_DBELL_SLOTS];

/* Atomic monotonic counter for trace sequence numbers. */
static atomic_t g_dbell_seq = ATOMIC_INIT(0);

/*
 * Research-mode kill-switch.  Set to 1 (e.g. via
 * `modprobe nvidia nv_dbell_disable_intercept=1` or
 * `echo 1 > /sys/module/nvidia/parameters/nv_dbell_disable_intercept`)
 * to make nv_dbell_intercept_mmap return 0 unconditionally — userspace
 * mmaps of HOPPER_USERMODE_A then reach REAL BAR0/BAR1, no shadow page,
 * no watchpoint armed.  Required when measuring whether a GPU-issued
 * MMIO write actually reaches the doorbell, since the shadow path
 * absorbs GPU writes too (the HW debug regs only see CPU MMU
 * dereferences).  Default 0 — opt-in only.
 */
int nv_dbell_disable_intercept = 0;
module_param(nv_dbell_disable_intercept, int, 0644);
MODULE_PARM_DESC(nv_dbell_disable_intercept,
    "If non-zero, nv_dbell_intercept_mmap is a no-op so HOPPER_USERMODE_A "
    "mmaps land on real BAR0/BAR1 instead of a shadow RAM page (research).");

/* ------------------------------------------------------------------ */
/* BAR1 mapping tracker                                               */
/* ------------------------------------------------------------------ */

/*
 * Global table of BAR1/FB device-node mmaps we've ioremapped a kernel
 * shadow for.  Populated from nvidia_mmap_helper() via
 * nv_dbell_bar1_track_add(); torn down from nvidia_vma_release() via
 * nv_dbell_bar1_track_remove().
 *
 * Used by the kthread resolver to translate USERD's FBMEM phys address
 * into a kernel VA without needing to call memdescMap (which would set
 * up a second BAR2 mapping that RM also manages).
 *
 * Sized generously: one 2 MiB mapping per process, plus the 64 KiB
 * HOPPER_USERMODE_A mappings we also pass through.  16 entries is plenty
 * for a typical desktop session.
 */
#define NV_DBELL_BAR1_TABLE_SIZE  16

typedef struct nv_dbell_bar1_entry_s {
    /* vma pointer is the primary key (one entry per VMA).  NULL = free. */
    struct vm_area_struct *vma;
    NvU64                  phys_start;
    NvU64                  phys_end;   /* exclusive */
    NvU64                  user_va_start;  /* vma->vm_start at track-add; used
                                            * to translate GPU VA (= user VA
                                            * under UVM per Paper Finding 1)
                                            * to kernel VA for GPFIFO reads. */
    NvU64                  user_va_end;    /* exclusive */
    void __iomem          *kernel_va;  /* ioremap result; NULL in a free slot */
} nv_dbell_bar1_entry_t;

static nv_dbell_bar1_entry_t g_bar1_tracked[NV_DBELL_BAR1_TABLE_SIZE];

/*
 * Spinlock, not mutex: nv_dbell_bar1_lookup_kva can be called from
 * softirq/workqueue context and must not sleep.  The table is small and
 * scans are fast so spinning is fine.
 */
static DEFINE_SPINLOCK(g_bar1_lock);

int nv_dbell_bar1_track_add(struct vm_area_struct *vma,
                            NvU64 phys_start, NvU64 phys_size)
{
    unsigned int i, slot = (unsigned int)-1;
    void __iomem *kva;
    unsigned long flags;

    /* ioremap is sleepable — do it OUTSIDE the spinlock.  Cache-attribute
     * choice: WC matches how BAR1 vidmem apertures are typically mapped
     * on Hopper, and the kernel only reads from it (READ_ONCE on USERD
     * GP_PUT), so cache pollution is not a concern. */
    kva = ioremap_wc(phys_start, phys_size);
    if (kva == NULL)
    {
        /* Fall back to uncached; ioremap_wc can fail on some kernel
         * configurations. */
        kva = ioremap(phys_start, phys_size);
        if (kva == NULL)
        {
            MC_TRACE(dbell, "bar1_track", "state=ioremap_fail phys=0x%llx size=0x%llx",
                            (unsigned long long)phys_start,
                            (unsigned long long)phys_size);
            return -ENOMEM;
        }
    }

    spin_lock_irqsave(&g_bar1_lock, flags);
    for (i = 0; i < NV_DBELL_BAR1_TABLE_SIZE; i++)
    {
        if (g_bar1_tracked[i].vma == NULL)
        {
            slot = i;
            g_bar1_tracked[i].vma           = vma;
            g_bar1_tracked[i].phys_start    = phys_start;
            g_bar1_tracked[i].phys_end      = phys_start + phys_size;
            g_bar1_tracked[i].user_va_start = (NvU64)vma->vm_start;
            g_bar1_tracked[i].user_va_end   = (NvU64)vma->vm_end;
            g_bar1_tracked[i].kernel_va     = kva;
            break;
        }
    }
    spin_unlock_irqrestore(&g_bar1_lock, flags);

    if (slot == (unsigned int)-1)
    {
        /* Table full; undo the ioremap. */
        iounmap(kva);
        MC_TRACE(dbell, "bar1_track", "state=table_full");
        return -ENOSPC;
    }

    MC_TRACE(dbell, "bar1_track", "state=add slot=%u phys=0x%llx "
                    "size=0x%llx kva=%p",
                    slot,
                    (unsigned long long)phys_start,
                    (unsigned long long)phys_size, kva);
    return 0;
}

/* Forward declaration — defined further down alongside the cache. */
static void nv_dbell_cache_invalidate_range(void *kva_base, size_t size);

void nv_dbell_bar1_track_remove(struct vm_area_struct *vma)
{
    unsigned int i;
    void __iomem *kva_to_unmap = NULL;
    size_t         iomap_size  = 0;
    unsigned long  flags;

    spin_lock_irqsave(&g_bar1_lock, flags);
    for (i = 0; i < NV_DBELL_BAR1_TABLE_SIZE; i++)
    {
        if (g_bar1_tracked[i].vma == vma)
        {
            kva_to_unmap = g_bar1_tracked[i].kernel_va;
            iomap_size   = (size_t)(g_bar1_tracked[i].phys_end -
                                    g_bar1_tracked[i].phys_start);
            g_bar1_tracked[i].vma           = NULL;
            g_bar1_tracked[i].phys_start    = 0;
            g_bar1_tracked[i].phys_end      = 0;
            g_bar1_tracked[i].user_va_start = 0;
            g_bar1_tracked[i].user_va_end   = 0;
            g_bar1_tracked[i].kernel_va     = NULL;
            break;
        }
    }
    spin_unlock_irqrestore(&g_bar1_lock, flags);

    if (kva_to_unmap != NULL)
    {
        /*
         * Invalidate any cache entries whose resolved userd_kva or
         * gpfifo_kva points into the region we're about to iounmap,
         * or a later #DB trap on the cached (chid, runlist) will
         * dereference freed memory.  This mirrors the lifetime
         * coupling between BAR1 VMA teardown and RM-side channel
         * destruct.
         */
        nv_dbell_cache_invalidate_range((void *)kva_to_unmap, iomap_size);
        iounmap(kva_to_unmap);
        MC_TRACE(dbell, "bar1_track", "state=remove slot=%u kva=%p", i, kva_to_unmap);
    }
}

void *nv_dbell_bar1_lookup_kva(NvU64 phys, NvU64 size)
{
    unsigned int i;
    void *result = NULL;
    unsigned long flags;

    spin_lock_irqsave(&g_bar1_lock, flags);
    for (i = 0; i < NV_DBELL_BAR1_TABLE_SIZE; i++)
    {
        nv_dbell_bar1_entry_t *e = &g_bar1_tracked[i];
        if (e->vma == NULL)
            continue;
        if (phys >= e->phys_start && phys + size <= e->phys_end)
        {
            result = (u8 __force *)e->kernel_va + (phys - e->phys_start);
            break;
        }
    }
    spin_unlock_irqrestore(&g_bar1_lock, flags);

    return result;
}

void *nv_dbell_bar1_offset_kva(NvU64 off, NvU64 size)
{
    unsigned int i;
    void *result = NULL;
    unsigned long flags;
    NvU64 best_size = 0;

    /*
     * Pick the LARGEST tracked entry whose size >= off + size.
     * On our workloads there is exactly one "big" BAR1 mapping per
     * process (the 2 MiB USERD+GPFIFO+pushbuffer region) plus two
     * small 64 KiB HOPPER_USERMODE_A doorbell pages.  We want the
     * 2 MiB one.
     */
    spin_lock_irqsave(&g_bar1_lock, flags);
    for (i = 0; i < NV_DBELL_BAR1_TABLE_SIZE; i++)
    {
        nv_dbell_bar1_entry_t *e = &g_bar1_tracked[i];
        NvU64 entry_size;
        if (e->vma == NULL || e->kernel_va == NULL)
            continue;
        entry_size = e->phys_end - e->phys_start;
        if (entry_size < off + size)
            continue;
        if (entry_size > best_size)
        {
            best_size = entry_size;
            result = (u8 __force *)e->kernel_va + off;
        }
    }
    spin_unlock_irqrestore(&g_bar1_lock, flags);

    return result;
}

/*
 * Translate a GPU VA to a kernel VA using the userspace VA range
 * captured at track_add time.  Under UVM (Paper Finding 1) a pushbuffer
 * command's GPU VA equals the process userspace VA.  The GPFIFO ring's
 * own GPU VA (from RAMFC's GP_BASE) also follows this rule — verified
 * empirically by the TRACE Gpfifo diagnostic.
 *
 * For the 2 MiB per-process BAR1 mapping: libcuda mmaps it at user VA
 * `VSTART`, and then the GPU VA `gp_base` that RM writes into RAMFC
 * equals VSTART for the first channel's GPFIFO ring.  So
 *     offset = gpu_va - VSTART
 *     kernel_va = ioremap_base + offset
 *
 * Returns NULL if no tracked entry covers the full [gpu_va, gpu_va+size).
 */
void *nv_dbell_bar1_gpu_va_to_kva(NvU64 gpu_va, NvU64 size)
{
    unsigned int i;
    void *result = NULL;
    unsigned long flags;

    spin_lock_irqsave(&g_bar1_lock, flags);
    for (i = 0; i < NV_DBELL_BAR1_TABLE_SIZE; i++)
    {
        nv_dbell_bar1_entry_t *e = &g_bar1_tracked[i];
        if (e->vma == NULL || e->kernel_va == NULL)
            continue;
        if (e->user_va_start == 0 || e->user_va_end == 0)
            continue;
        if (gpu_va < e->user_va_start)
            continue;
        if (gpu_va + size > e->user_va_end)
            continue;
        result = (u8 __force *)e->kernel_va + (gpu_va - e->user_va_start);
        break;
    }
    spin_unlock_irqrestore(&g_bar1_lock, flags);

    return result;
}

/* ------------------------------------------------------------------ */
/* Sysmem mapping tracker — pushbuffer-bytes readout path             */
/* ------------------------------------------------------------------ */

/*
 * libcuda's pushbuffer pool is an NV01_MEMORY_SYSTEM allocation
 * CPU-mapped via NVOS33 + MAP_FIXED mmap of /dev/nvidiactl at the same
 * userspace VA it will later be GPU-mapped to via UVM_MAP_EXTERNAL
 * (Paper Finding 1: GPU VA == user VA under UVM).
 *
 * We vmap the underlying pages at track_add time so the #DB handler
 * can read method-stream bytes at (kernel_va + pb_va - user_va_start).
 *
 * Sized small: typical CUDA process has O(10) sysmem mmaps that pass
 * the "large enough to be interesting" filter in nv-mmap.c.
 */
#define NV_DBELL_SYSMEM_TABLE_SIZE   8

typedef struct nv_dbell_sysmem_entry_s {
    struct vm_area_struct *vma;
    NvU64  user_va_start;
    NvU64  user_va_end;
    void  *kernel_va;      /* vmap() result; NULL in a free slot */
    NvU32  num_pages;      /* for nv_vm_unmap_pages */
} nv_dbell_sysmem_entry_t;

static nv_dbell_sysmem_entry_t g_sysmem_tracked[NV_DBELL_SYSMEM_TABLE_SIZE];
static DEFINE_SPINLOCK(g_sysmem_lock);

int nv_dbell_sysmem_track_add(struct vm_area_struct *vma,
                              nvidia_pte_t *page_table,
                              NvU64 page_index,
                              NvU64 num_pages)
{
    unsigned int i, slot = (unsigned int)-1;
    struct page **pages;
    NvUPtr kva;
    unsigned long flags;
    NvU64 j;

    if (vma == NULL || page_table == NULL || num_pages == 0)
        return -EINVAL;

    /* Cap: vmap() eats VA space and the tracker only has 8 slots. */
    if (num_pages > (256u * 1024u))  /* 1 GiB @ 4 KiB */
        return -E2BIG;

    /*
     * Build a struct page* array to pass to nv_vm_map_pages().
     * NV_GET_PAGE_STRUCT(phys) == pfn_to_page(phys >> PAGE_SHIFT).
     * kmalloc is sleepable — we're in an mmap syscall here, not the
     * #DB trap handler.
     */
    pages = kmalloc_array((size_t)num_pages, sizeof(struct page *),
                          GFP_KERNEL);
    if (pages == NULL)
        return -ENOMEM;

    for (j = 0; j < num_pages; j++)
        pages[j] = NV_GET_PAGE_STRUCT(page_table[page_index + j].phys_addr);

    /*
     * cached=NV_TRUE so the kernel VA reads match the CPU's view —
     * libcuda writes method bytes into this mapping via normal stores,
     * which land in the CPU cache.  unencrypted=NV_FALSE: we're not
     * dealing with confidential-compute carveout here.
     */
    kva = nv_vm_map_pages(pages, (NvU32)num_pages, NV_TRUE, NV_FALSE);
    kfree(pages);
    if (kva == 0)
    {
        MC_TRACE(dbell, "sysmem_track", "state=vmap_fail user_va=0x%llx num_pages=%llu",
                        (unsigned long long)vma->vm_start,
                        (unsigned long long)num_pages);
        return -ENOMEM;
    }

    spin_lock_irqsave(&g_sysmem_lock, flags);
    for (i = 0; i < NV_DBELL_SYSMEM_TABLE_SIZE; i++)
    {
        if (g_sysmem_tracked[i].vma == NULL)
        {
            slot = i;
            g_sysmem_tracked[i].vma           = vma;
            g_sysmem_tracked[i].user_va_start = (NvU64)vma->vm_start;
            g_sysmem_tracked[i].user_va_end   = (NvU64)vma->vm_end;
            g_sysmem_tracked[i].kernel_va     = (void *)kva;
            g_sysmem_tracked[i].num_pages     = (NvU32)num_pages;
            break;
        }
    }
    spin_unlock_irqrestore(&g_sysmem_lock, flags);

    if (slot == (unsigned int)-1)
    {
        nv_vm_unmap_pages(kva, (NvU32)num_pages);
        MC_TRACE(dbell, "sysmem_track", "state=table_full");
        return -ENOSPC;
    }

    MC_TRACE(dbell, "sysmem_track", "state=add slot=%u "
                    "user_va_start=0x%llx user_va_end=0x%llx num_pages=%llu kva=%p",
                    slot,
                    (unsigned long long)vma->vm_start,
                    (unsigned long long)vma->vm_end,
                    (unsigned long long)num_pages,
                    (void *)kva);
    return 0;
}

void nv_dbell_sysmem_track_remove(struct vm_area_struct *vma)
{
    unsigned int i;
    NvUPtr kva_to_unmap = 0;
    NvU32  pages_to_unmap = 0;
    unsigned long flags;

    spin_lock_irqsave(&g_sysmem_lock, flags);
    for (i = 0; i < NV_DBELL_SYSMEM_TABLE_SIZE; i++)
    {
        if (g_sysmem_tracked[i].vma == vma)
        {
            kva_to_unmap   = (NvUPtr)g_sysmem_tracked[i].kernel_va;
            pages_to_unmap = g_sysmem_tracked[i].num_pages;
            g_sysmem_tracked[i].vma           = NULL;
            g_sysmem_tracked[i].user_va_start = 0;
            g_sysmem_tracked[i].user_va_end   = 0;
            g_sysmem_tracked[i].kernel_va     = NULL;
            g_sysmem_tracked[i].num_pages     = 0;
            break;
        }
    }
    spin_unlock_irqrestore(&g_sysmem_lock, flags);

    if (kva_to_unmap != 0)
    {
        /* vunmap sleeps — must be OUTSIDE the spinlock.  OK because
         * vma_release runs in munmap syscall context. */
        nv_vm_unmap_pages(kva_to_unmap, pages_to_unmap);
        MC_TRACE(dbell, "sysmem_track", "state=remove kva=%p num_pages=%u",
                        (void *)kva_to_unmap, pages_to_unmap);
    }
}

void *nv_dbell_sysmem_gpu_va_to_kva(NvU64 gpu_va, NvU64 size)
{
    unsigned int i;
    void *result = NULL;
    unsigned long flags;

    spin_lock_irqsave(&g_sysmem_lock, flags);
    for (i = 0; i < NV_DBELL_SYSMEM_TABLE_SIZE; i++)
    {
        nv_dbell_sysmem_entry_t *e = &g_sysmem_tracked[i];
        if (e->vma == NULL || e->kernel_va == NULL)
            continue;
        if (e->user_va_start == 0 || e->user_va_end == 0)
            continue;
        if (gpu_va < e->user_va_start)
            continue;
        if (gpu_va + size > e->user_va_end)
            continue;
        result = (u8 *)e->kernel_va + (gpu_va - e->user_va_start);
        break;
    }
    spin_unlock_irqrestore(&g_sysmem_lock, flags);

    return result;
}

/* ------------------------------------------------------------------ */
/* Slot management                                                    */
/* ------------------------------------------------------------------ */

static int nv_dbell_claim_slot(nv_dbell_ctx_t *ctx)
{
    unsigned int i;
    for (i = 0; i < NV_DBELL_SLOTS; i++)
    {
        if (atomic_cmpxchg(&g_dbell_slots[i].claimed, 0, 1) == 0)
        {
            g_dbell_slots[i].ctx = ctx;
            ctx->slot_idx = i;
            return (int)i;
        }
    }
    return -1;
}

static void nv_dbell_release_slot(nv_dbell_ctx_t *ctx)
{
    unsigned int i = ctx->slot_idx;
    if (i < NV_DBELL_SLOTS)
    {
        g_dbell_slots[i].ctx = NULL;
        atomic_set(&g_dbell_slots[i].claimed, 0);
    }
}

/* ------------------------------------------------------------------ */
/* Per-(chid, runlist) USERD cache                                    */
/* ------------------------------------------------------------------ */

/*
 * CUDA distributes its ~20 channels across 4+ runlists.  A single VMA's
 * doorbell writes select the target channel via the token bit-field
 * (chid = token & 0xfff, runlist = (token >> 16) & 0x7f).  We cache
 * USERD kernel mappings per (chid, runlist) so the trap handler can
 * read GPPut with zero locks.
 *
 * 80 entries is enough for CUDA's ~20 channels × 4 runlists with slack.
 * Lookup is a linear scan of up to 80 u64s — noise in #DB context.
 */
#define NV_DBELL_CACHE_SIZE   80

#define NV_DBELL_CACHE_FREE      0u  /* empty slot */
#define NV_DBELL_CACHE_PENDING   1u  /* worker enqueued, resolution in flight */
#define NV_DBELL_CACHE_RESOLVED  2u  /* userd_kva valid, trap handler may read */
#define NV_DBELL_CACHE_FAILED    3u  /* RM returned error; negative cache */

typedef struct nv_dbell_cache_entry_s {
    atomic_t   state;          /* NV_DBELL_CACHE_* */
    NvU32      chid;
    NvU32      runlist;
    void      *userd_kva;      /* written once before state is moved to RESOLVED */
    void      *gpfifo_kva;     /* GPFIFO ring kernel VA; NULL if resolution partial */
    NvU32      gpfifo_entries; /* 0 if unknown */
    nv_state_t *nv;            /* GPU that owns this channel */
    /*
     * Dedup cursor for PB_BYTES emission: libcuda pokes the doorbell
     * ~85× per logical submission while spin-polling; each poke reads
     * the same GPFIFO entry at idx=(gp_put-1).  Emit pushbuffer bytes
     * only when (idx, pb_va) changes vs. the previously-emitted pair.
     * Single-writer (the #DB handler serialized by cache slot), so
     * plain stores with WRITE_ONCE suffice — no atomics needed.
     * Sentinel: last_pb_bytes_idx = 0xFFFFFFFF at slot init means
     * "never emitted", always trips on first real doorbell.
     */
    NvU32      last_pb_bytes_idx;
    NvU64      last_pb_bytes_va;
} nv_dbell_cache_entry_t;

static nv_dbell_cache_entry_t g_dbell_cache[NV_DBELL_CACHE_SIZE];

/*
 * Lockless lookup: linear scan.  The trap handler calls this; entries
 * transition FREE → PENDING → (RESOLVED | FAILED) monotonically with
 * atomic state writes, so a concurrent reader sees either the old state
 * or the new state, never a tear.
 *
 * Returns the entry pointer regardless of state; caller inspects
 * entry->state.  Returns NULL if no entry matches.
 */
static nv_dbell_cache_entry_t *
nv_dbell_cache_find(NvU32 chid, NvU32 runlist)
{
    unsigned int i;
    for (i = 0; i < NV_DBELL_CACHE_SIZE; i++)
    {
        nv_dbell_cache_entry_t *e = &g_dbell_cache[i];
        if (atomic_read(&e->state) == NV_DBELL_CACHE_FREE)
            continue;
        if (e->chid == chid && e->runlist == runlist)
            return e;
    }
    return NULL;
}

/*
 * Reserve a cache slot for this (chid, runlist).  Returns the entry
 * pointer if we transitioned FREE→PENDING (caller owns enqueueing the
 * worker), or NULL if another CPU beat us to it (worker already in
 * flight or cache full).  Atomic; safe to call from #DB context.
 */
static nv_dbell_cache_entry_t *
nv_dbell_cache_reserve(NvU32 chid, NvU32 runlist, nv_state_t *nv)
{
    unsigned int i;
    for (i = 0; i < NV_DBELL_CACHE_SIZE; i++)
    {
        nv_dbell_cache_entry_t *e = &g_dbell_cache[i];
        if (atomic_cmpxchg(&e->state,
                           NV_DBELL_CACHE_FREE,
                           NV_DBELL_CACHE_PENDING) == NV_DBELL_CACHE_FREE)
        {
            /* We own this slot now. */
            e->chid               = chid;
            e->runlist            = runlist;
            e->nv                 = nv;
            e->userd_kva          = NULL;
            e->gpfifo_kva         = NULL;
            e->gpfifo_entries     = 0;
            e->last_pb_bytes_idx  = 0xFFFFFFFFu;  /* "never emitted" sentinel */
            e->last_pb_bytes_va   = 0;
            smp_wmb();
            return e;
        }
    }
    return NULL;
}

/*
 * Invalidate any cache entries whose userd_kva or gpfifo_kva points
 * into [kva_base, kva_base + size).  Called from
 * nv_dbell_bar1_track_remove just before iounmap so a subsequent
 * doorbell trap doesn't dereference freed iomap memory.
 *
 * Moves the slot to FREE so the next doorbell on that (chid, runlist)
 * re-reserves and re-resolves.  If the channel is gone by then,
 * the resolver will fail gracefully and the cache entry will be
 * marked FAILED.
 *
 * We don't hold g_bar1_lock here (caller already dropped it) and we
 * don't need any additional sync: the trap handler reads
 * e->userd_kva only after observing state == RESOLVED, and we move
 * state away from RESOLVED atomically here before any iounmap.
 */
static void
nv_dbell_cache_invalidate_range(void *kva_base, size_t size)
{
    unsigned int i;
    u8 *lo = (u8 *)kva_base;
    u8 *hi = lo + size;

    if (kva_base == NULL || size == 0)
        return;

    for (i = 0; i < NV_DBELL_CACHE_SIZE; i++)
    {
        nv_dbell_cache_entry_t *e = &g_dbell_cache[i];
        u8 *u;
        u8 *g;

        /* Snapshot under the state-observation guarantee: we care
         * about entries already RESOLVED (userd_kva valid). */
        if (atomic_read(&e->state) != NV_DBELL_CACHE_RESOLVED)
            continue;

        u = (u8 *)e->userd_kva;
        g = (u8 *)e->gpfifo_kva;

        if ((u != NULL && u >= lo && u < hi) ||
            (g != NULL && g >= lo && g < hi))
        {
            /* Park the slot FREE so the next trap reserves fresh.
             * Write state AFTER zeroing the stale pointers, so a
             * concurrent reader that got past the state check
             * reads NULL and skips the deref. */
            e->userd_kva         = NULL;
            e->gpfifo_kva        = NULL;
            e->gpfifo_entries    = 0;
            e->last_pb_bytes_idx = 0xFFFFFFFFu;
            e->last_pb_bytes_va  = 0;
            smp_wmb();
            atomic_set(&e->state, NV_DBELL_CACHE_FREE);
            MC_TRACE(dbell, "cache", "state=invalidate chid=%u runlist=%u reason=bar1_iounmap",
                            e->chid, e->runlist);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Resolver work item                                                 */
/* ------------------------------------------------------------------ */

typedef struct nv_dbell_resolve_work_s {
    nv_kthread_q_item_t     qitem;
    nv_dbell_cache_entry_t *entry;
    nvidia_stack_t         *sp;
} nv_dbell_resolve_work_t;

/*
 * Deferred resolution.  Runs in the nv_kthread_q worker context (can
 * sleep, can take RM locks).  Calls into MIT-side rm_gpu_ops to walk
 * kfifo for USERD's phys+size+addrspace, then resolves phys → kernel
 * VA via the BAR1 tracker: no memdescMap and no BAR2 PTE churn, because
 * it reuses CUDA's own 2 MiB BAR1 mapping, which already contains USERD.
 *
 * For sysmem USERDs (mc's carrier channels) we fall back to phys_to_virt
 * since sysmem phys → kernel VA is just the direct map.
 */
static void
nv_dbell_resolve_fn(void *arg)
{
    nv_dbell_resolve_work_t *work = (nv_dbell_resolve_work_t *)arg;
    nv_dbell_cache_entry_t *e = work->entry;
    void *userd_kva        = NULL;
    void *gpfifo_kva       = NULL;
    NvU64 userd_phys       = 0;
    NvU64 userd_size       = 0;
    NvU32 addrspace        = 0;
    NvU64 gpfifo_gpu_va    = 0;
    NvU32 gpfifo_entries   = 0;
    NV_STATUS status;

    status = rm_gpu_ops_dbell_resolve_channel(work->sp,
                                              e->nv,
                                              e->chid,
                                              e->runlist,
                                              &userd_kva,
                                              &userd_phys,
                                              &userd_size,
                                              &addrspace,
                                              &gpfifo_gpu_va,
                                              &gpfifo_entries);

    /*
     * FBMEM USERD: if RM's preallocated pool had userd_kva, it's set.
     * Otherwise RM set userd_phys = subOff (offset within the
     * per-process BAR1 mapping); look up via the tracker.
     */
    if (status == NV_OK && userd_kva == NULL && addrspace == 2 /* FBMEM */
        && userd_size > 0)
    {
        userd_kva = nv_dbell_bar1_offset_kva(userd_phys, userd_size);
    }

    /* Sysmem fallback (mc carrier channels) — phys_to_virt over direct map. */
    if (status == NV_OK && userd_kva == NULL && addrspace == 1 /* SYSMEM */
        && userd_phys != 0)
    {
        userd_kva = phys_to_virt((phys_addr_t)userd_phys);
    }

    /*
     * Resolve GPFIFO ring kernel VA.  Under UVM the GPU VA that RAMFC
     * encodes equals the userspace VA inside libcuda's 2 MiB /dev/nvidia0
     * mapping — Paper Finding 1, the property user_va_start relies on.
     * Translate via the user_va range captured at bar1_track_add time.
     */
    if (status == NV_OK && gpfifo_gpu_va != 0 && gpfifo_entries > 0)
    {
        NvU64 ring_bytes = (NvU64)gpfifo_entries * 8;  /* 8 B per entry */
        gpfifo_kva = nv_dbell_bar1_gpu_va_to_kva(gpfifo_gpu_va, ring_bytes);
        if (gpfifo_kva == NULL)
        {
            MC_TRACE(dbell, "gpfifo_lookup", "result=miss chid=%u gpu_va=0x%llx entries=%u",
                            e->chid,
                            (unsigned long long)gpfifo_gpu_va,
                            gpfifo_entries);
        }
    }

    if (userd_kva != NULL)
    {
        e->userd_kva       = userd_kva;
        e->gpfifo_kva      = gpfifo_kva;      /* may be NULL if GPFIFO lookup failed */
        e->gpfifo_entries  = gpfifo_entries;
        smp_wmb();
        atomic_set(&e->state, NV_DBELL_CACHE_RESOLVED);
        MC_TRACE(dbell, "cache", "state=resolved chid=%u runlist=%u "
                    "userd_kva=0x%p gpfifo_kva=0x%p entries=%u",
                        e->chid, e->runlist,
                        userd_kva, gpfifo_kva, gpfifo_entries);
    }
    else if (status == NV_OK)
    {
        /*
         * "Resolve succeeded but userd_kva couldn't be reached."  Two
         * known cases, both from the register-time pre-resolve path (called
         * during kchannelConstruct):
         *   1. userd_phys == 0 — RM hasn't finished populating the
         *      channel's USERD memdesc yet.
         *   2. userd_phys != 0 but nv_dbell_bar1_offset_kva returned
         *      NULL — the BAR1 tracker hasn't yet registered the
         *      userspace BAR1 mapping that contains this USERD (the
         *      user hasn't called rm_map_memory on the gpu_ctl handle
         *      yet, or will do it later).
         * In both cases the right behavior is "roll back to FREE so
         * the #DB handler retries on the first doorbell" — by then
         * all the setup the resolver depends on will be in place.
         * FAILED is reserved for "RM said no such channel."
         */
        smp_wmb();
        atomic_set(&e->state, NV_DBELL_CACHE_FREE);
        MC_TRACE(dbell, "cache", "state=retry chid=%u runlist=%u phys=0x%llx addrspace=%u",
                        e->chid, e->runlist,
                        (unsigned long long)userd_phys, addrspace);
    }
    else
    {
        atomic_set(&e->state, NV_DBELL_CACHE_FAILED);
        MC_TRACE(dbell, "cache", "state=failed chid=%u runlist=%u status=0x%x phys=0x%llx addrspace=%u",
                        e->chid, e->runlist, status,
                        (unsigned long long)userd_phys, addrspace);
    }

    nv_kmem_cache_free_stack(work->sp);
    kfree(work);
}

/*
 * Enqueue a resolution work item.  Caller must own the reserved cache
 * entry (state == PENDING).  Must NOT be called from atomic context —
 * allocates memory.
 *
 * But our #DB handler IS atomic.  Solution: reserve the entry
 * atomically, then SKIP the enqueue if we can't alloc/alloc_stack.  The
 * entry stays in PENDING; the next doorbell on the same (chid, runlist)
 * will go down the enqueue path again.  That's a small amount of extra
 * churn but correctness-preserving.
 */
static void
nv_dbell_try_enqueue_resolve(nv_dbell_cache_entry_t *e)
{
    nv_dbell_resolve_work_t *work;
    nvidia_stack_t *sp = NULL;
    int rc;

    /*
     * GFP_ATOMIC: we may be called from the #DB handler itself.  If
     * allocation fails, roll the slot back to FREE so a subsequent
     * doorbell can try again.
     */
    work = kmalloc(sizeof(*work), GFP_ATOMIC);
    if (work == NULL)
        goto fail;

    if (nv_kmem_cache_alloc_stack_atomic(&sp) != 0)
      goto fail;

    work->entry = e;
    work->sp    = sp;
    nv_kthread_q_item_init(&work->qitem, nv_dbell_resolve_fn, work);

    rc = nv_kthread_q_schedule_q_item(&nv_kthread_q, &work->qitem);
    if (rc == 0)
    {
        /* Already in the queue for some reason — unusual but harmless. */
        nv_kmem_cache_free_stack(sp);
        kfree(work);
        goto fail;
    }

    return;

fail:
    if (sp != NULL)
        nv_kmem_cache_free_stack(sp);
    kfree(work);
    /* Roll back so the next doorbell re-enqueues. */
    atomic_set(&e->state, NV_DBELL_CACHE_FREE);
}

/*
 * nv_dbell_cache_pre_resolve — enqueue a resolve at channel register time.
 *
 * Motivation: the #DB handler-triggered path reserves a cache entry only
 * when the first doorbell for a given (chid, runlist) fires.  That first
 * doorbell emits `cache state=pending_init` and no `pb/submit` record
 * because the
 * async resolver hasn't populated the entry yet.  For libcuda workloads
 * that's ~20 missed first-submission PB decodes per --size 128M run (one
 * per channel).  Those submissions typically carry channel-init methods
 * (SET_OBJECT, etc.) that are the most informative about channel state.
 *
 * Fix: call this function from nvGpuOpsDbellGpfifoRegister (which runs
 * during kchannelConstruct, BEFORE any doorbell can fire on the new
 * channel).  We reserve a cache slot and enqueue the resolver kthread.
 * By the time the user's first doorbell lands, the cache entry is
 * RESOLVED (or FAILED; either way, no more pending_init).
 *
 * Locking: called with GPU locks held.  Only uses GFP_ATOMIC allocations
 * and nv_kthread_q_schedule_q_item (non-blocking); safe under locks.
 * The resolver kthread runs AFTER the caller releases locks because
 * Linux's scheduler doesn't preempt into a worker thread while locks
 * are held in non-preemptible sections.
 *
 * If the entry already exists (PENDING or RESOLVED), return without doing
 * anything — the existing lifecycle owns it.  The #DB-handler fallback
 * path remains for channels that somehow don't go through register
 * (shouldn't happen in current code paths, but preserves correctness).
 */
void NV_API_CALL
nv_dbell_cache_pre_resolve(NvU32 chid, NvU32 runlist, void *nv_opaque)
{
    nv_state_t *nv = (nv_state_t *)nv_opaque;
    nv_dbell_cache_entry_t *e;

    if (nv == NULL)
        return;

    /* If an entry already exists for this (chid, runlist), don't touch it. */
    e = nv_dbell_cache_find(chid, runlist);
    if (e != NULL)
        return;

    /* Atomically claim a FREE slot. */
    e = nv_dbell_cache_reserve(chid, runlist, nv);
    if (e == NULL)
    {
        /* Cache full — not fatal.  The #DB handler will retry at
         * first-doorbell time and produce the usual pending_init path. */
        return;
    }

    MC_TRACE(dbell, "cache", "state=pre_reserve chid=%u runlist=%u", chid, runlist);
    nv_dbell_try_enqueue_resolve(e);
}

/* ------------------------------------------------------------------ */
/* #DB handler — runs in x86 trap context, atomic                     */
/* ------------------------------------------------------------------ */

static void nv_dbell_db_handler(struct perf_event *bp,
                                struct perf_sample_data *data,
                                struct pt_regs *regs)
{
    nv_dbell_ctx_t *ctx = (nv_dbell_ctx_t *)bp->overflow_handler_context;
    u32 token;
    u32 chid, runlist;
    unsigned int seq;
    nv_dbell_cache_entry_t *cache;

    (void)data;
    (void)regs;

    if (ctx == NULL || ctx->shadow_kva == NULL)
        return;

    /* READ_ONCE: the shadow store may coalesce in the CPU store buffer;
     * we want whatever value is currently visible at the dword. */
    token = READ_ONCE(*(volatile u32 *)((u8 *)ctx->shadow_kva + NV_DBELL_OFFSET));

    /* Hopper token layout: VECTOR=[11:0]=chid, RUNLIST_ID=[22:16].
     * Bit positions match NV_CTRL_VF_DOORBELL_* from dev_ctrl.h. */
    chid    = token & 0xfffu;
    runlist = (token >> 16) & 0x7fu;

    seq = atomic_inc_return(&g_dbell_seq);
    atomic_inc(&ctx->bp_count);

    MC_TRACE(dbell, "fire", "seq=%u slot=%u token=0x%08x chid=%u runlist=%u",
                    seq, ctx->slot_idx, token, chid, runlist);

    /*
     * Channel cache lookup + USERD GP_PUT read.
     *
     * Three states:
     *   FREE/not-found → reserve + enqueue resolver, emit pending.
     *   PENDING        → resolver already in flight; emit pending.
     *   RESOLVED       → read GPPut at USERD + 0x8c and emit.
     *   FAILED         → previous resolve attempt failed; emit and
     *                    don't re-enqueue (negative cache).
     */
    cache = nv_dbell_cache_find(chid, runlist);
    if (cache == NULL)
    {
        cache = nv_dbell_cache_reserve(chid, runlist, ctx->nv);
        if (cache != NULL)
        {
            MC_TRACE(dbell, "cache", "state=reserve seq=%u chid=%u runlist=%u", seq, chid, runlist);
            nv_dbell_try_enqueue_resolve(cache);
        }
        /* Either way: emit pending and forward. */
        MC_TRACE(dbell, "cache", "state=pending_init seq=%u chid=%u runlist=%u", seq, chid, runlist);
    }
    else
    {
        u32 state = atomic_read(&cache->state);
        if (state == NV_DBELL_CACHE_RESOLVED)
        {
            /*
             * USERD is a sysmem page kernel-mapped by RM at channel
             * alloc time and stable thereafter.  READ_ONCE the GPPut
             * dword at the HopperAControlGPFifo offset 0x8c (clc86f.h).
             */
            smp_rmb();
            if (cache->userd_kva != NULL)
            {
                u32 gp_put = READ_ONCE(
                    *(volatile u32 *)((u8 *)cache->userd_kva + 0x8c));
                MC_TRACE(dbell, "gp_put", "seq=%u chid=%u runlist=%u gp_put=%u", seq, chid, runlist, gp_put);

                /*
                 * GPFIFO entry decode.  Read the entry at (gp_put - 1)
                 * mod entries — the most recent submission that this
                 * doorbell is publishing.  Each entry is 8 bytes:
                 *   entry0[31:2]  = pb_va[31:2]     (GP_ENTRY0_GET)
                 *   entry1[7:0]   = pb_va[39:32]    (GP_ENTRY1_GET_HI)
                 *   entry1[30:10] = pb length in dwords (GP_ENTRY1_LENGTH)
                 * Bit positions per clc86f.h.  entries is always a
                 * power of two (1 << LIMIT2), so modulo is a mask.
                 */
                if (cache->gpfifo_kva != NULL && cache->gpfifo_entries > 0)
                {
                    u32 entries_mask = cache->gpfifo_entries - 1u;
                    u32 idx          = (gp_put - 1u) & entries_mask;
                    volatile u32 *ring = (volatile u32 *)cache->gpfifo_kva;
                    u32 e0 = READ_ONCE(ring[idx * 2u]);
                    u32 e1 = READ_ONCE(ring[idx * 2u + 1u]);
                    u64 pb_va_lo   = (u64)(e0 & 0xFFFFFFFCu);         /* bits 31:2 << 2 */
                    u64 pb_va_hi   = (u64)(e1 & 0xFFu) << 32;         /* bits 7:0 */
                    u64 pb_va      = pb_va_hi | pb_va_lo;
                    u32 pb_dwords  = (e1 >> 10) & 0x1FFFFFu;          /* bits 30:10 */
                    u32 pb_bytes   = pb_dwords * 4u;

                    MC_TRACE(pb, "submit", "seq=%u chid=%u idx=%u "
                                    "entry0=0x%08x entry1=0x%08x "
                                    "pb_va=0x%llx pb_len=%u",
                                    seq, chid, idx, e0, e1,
                                    (unsigned long long)pb_va, pb_bytes);

                    /*
                     * Pushbuffer-bytes readout.
                     *
                     * libcuda writes the doorbell many times per logical
                     * submission (spin-polling); measured ~85× for a 4M
                     * roundtrip.  Each fire reads the same GPFIFO entry,
                     * so the same (chid, idx) tuple shows up repeatedly.
                     * Emit bytes only when the tuple CHANGES, keyed on
                     * the per-channel cache entry.  Saves ~95% of
                     * trap-handler work + ftrace bandwidth.
                     *
                     * The cache entry is per-(chid, runlist); we extend
                     * it with a one-slot "last idx for which we emitted
                     * bytes" counter to skip repeat fires.
                     */
                    if (READ_ONCE(cache->last_pb_bytes_idx) != idx ||
                        cache->last_pb_bytes_va != pb_va)
                    {
                        WRITE_ONCE(cache->last_pb_bytes_idx, idx);
                        cache->last_pb_bytes_va = pb_va;

                        /* Reach into the sysmem tracker for the kva. */
                        if (pb_bytes > 0 && pb_bytes <= 4096u)
                        {
                            void *pb_kva =
                                nv_dbell_sysmem_gpu_va_to_kva(pb_va,
                                                              (u64)pb_bytes);
                            if (pb_kva != NULL)
                            {
                                /*
                                 * Emit bytes in chunks of 64 bytes
                                 * (128 hex chars).  One dword = 4 bytes
                                 * = 8 hex chars → 16 dwords per line
                                 * leaves room for prefix + ts under the
                                 * ~1 KiB ftrace line limit.
                                 */
                                const u32 CHUNK = 64u;
                                u32 chunks_total =
                                    (pb_bytes + CHUNK - 1u) / CHUNK;
                                u32 off;
                                u32 ck = 0;
                                char hex[129];  /* 64 bytes * 2 + NUL */

                                for (off = 0; off < pb_bytes; off += CHUNK)
                                {
                                    u32 this_len =
                                        (pb_bytes - off) < CHUNK
                                        ? (pb_bytes - off) : CHUNK;
                                    u32 k;

                                    /* Hex-encode directly; no allocs. */
                                    for (k = 0; k < this_len; k++)
                                    {
                                        u8 b = READ_ONCE(
                                            ((volatile u8 *)pb_kva)[off + k]);
                                        static const char hd[] = "0123456789abcdef";
                                        hex[k * 2]     = hd[(b >> 4) & 0xf];
                                        hex[k * 2 + 1] = hd[b & 0xf];
                                    }
                                    hex[this_len * 2] = '\0';

                                    MC_TRACE(pb, "bytes", "seq=%u chid=%u "
                                        "idx=%u chunk=%u nchunks=%u off=%u hex=%s",
                                        seq, chid, idx,
                                        ck, chunks_total, off, hex);
                                    ck++;
                                }
                            }
                            else
                            {
                                MC_TRACE(pb, "bytes_miss", "seq=%u chid=%u "
                                    "idx=%u pb_va=0x%llx pb_len=%u",
                                    seq, chid, idx,
                                    (unsigned long long)pb_va, pb_bytes);
                            }
                        }
                    }
                }
            }
        }
        else if (state == NV_DBELL_CACHE_PENDING)
        {
            MC_TRACE(dbell, "cache", "state=pending_in_flight seq=%u chid=%u runlist=%u", seq, chid, runlist);
        }
        else  /* FAILED */
        {
            MC_TRACE(dbell, "cache", "state=failed seq=%u chid=%u runlist=%u", seq, chid, runlist);
        }
    }

    /* ALWAYS forward to real BAR0, even on error paths — the trap must
     * never cost the GPU a doorbell, or the channel hangs forever. */
    if (ctx->doorbell_iomap != NULL)
        writel(token, ctx->doorbell_iomap);
    else
        MC_TRACE(dbell, "no_iomap", "seq=%u", seq);
}

/*
 * Unregister every hardware breakpoint this context installed.  Safe on a
 * partially-armed context: slots past bp_event_count were never written, and
 * unset slots inside it are NULL.
 */
static void nv_dbell_unregister_bps(nv_dbell_ctx_t *ctx)
{
    unsigned int i;

    for (i = 0; i < ctx->bp_event_count; i++)
    {
        if (ctx->bp_events[i] != NULL)
        {
            nv_dbell_bp_unregister(ctx->bp_events[i]);
            ctx->bp_events[i] = NULL;
        }
    }
    ctx->bp_event_count = 0;
}

/* ------------------------------------------------------------------ */
/* VMA lifecycle                                                      */
/* ------------------------------------------------------------------ */

int nv_dbell_is_watched_vma(struct vm_area_struct *vma)
{
    nv_dbell_ctx_t *ctx = (nv_dbell_ctx_t *)NV_VMA_PRIVATE(vma);
    return (ctx != NULL && ctx->tag == NV_DBELL_VMA_TAG);
}

void nv_dbell_vma_release(struct vm_area_struct *vma)
{
    nv_dbell_ctx_t *ctx = (nv_dbell_ctx_t *)NV_VMA_PRIVATE(vma);
    unsigned int off;

    if (ctx == NULL || ctx->tag != NV_DBELL_VMA_TAG)
        return;

    MC_TRACE(dbell, "release", "slot=%u count=%d", ctx->slot_idx, atomic_read(&ctx->bp_count));

    /*
     * Forensic dump: walk the 4 KiB shadow page dword-by-dword; log any
     * non-zero offsets.  This tells us whether userspace wrote ANYWHERE
     * on the page — useful when the HW watchpoint at +0x90 produces
     * zero events (i.e. userspace wrote to a different dword).
     */
    if (ctx->shadow_kva != NULL)
    {
        volatile u32 *p = (volatile u32 *)ctx->shadow_kva;
        unsigned int nonzero = 0;
        for (off = 0; off < PAGE_SIZE / 4; off++)
        {
            u32 v = p[off];
            if (v != 0)
            {
                MC_TRACE(dbell, "shadow", "state=nonzero slot=%u off=0x%x val=0x%08x",
                                ctx->slot_idx, off * 4, v);
                if (++nonzero >= 32)
                {
                    MC_TRACE(dbell, "shadow", "state=nonzero_truncated");
                    break;
                }
            }
        }
        if (nonzero == 0)
            MC_TRACE(dbell, "shadow", "state=all_zero slot=%u", ctx->slot_idx);
    }

    nv_dbell_unregister_bps(ctx);

    if (ctx->doorbell_iomap != NULL)
    {
        iounmap(ctx->doorbell_iomap);
        ctx->doorbell_iomap = NULL;
    }

    nv_dbell_release_slot(ctx);

    if (ctx->shadow_page != NULL)
    {
        __free_page(ctx->shadow_page);
        ctx->shadow_page = NULL;
        ctx->shadow_kva  = NULL;
    }

    NV_VMA_PRIVATE(vma) = NULL;
    kfree(ctx);
}

/* ------------------------------------------------------------------ */
/* mmap interception                                                  */
/* ------------------------------------------------------------------ */

/*
 * Arm HW watchpoints on the userspace VA + 0x90 for every existing thread in
 * the process (thread group).  register_user_hw_breakpoint binds to a specific
 * task_struct, so a breakpoint installed on the main thread cannot see writes
 * from worker threads.  libcuda uses worker threads, so single-thread binding
 * misses every submission.
 *
 * Limitation: threads created AFTER arm time are not covered.  For CUDA that is
 * usually fine, because libcuda's worker threads are created early, well before
 * any cudaMemcpy.
 *
 * Returns 0 once at least one breakpoint is installed, -ENOMEM if none could be.
 */
static int nv_dbell_arm_threads(nv_dbell_ctx_t *ctx, struct vm_area_struct *vma)
{
    struct task_struct *leader = current->group_leader;
    struct task_struct *t;
    unsigned int installed = 0;

    rcu_read_lock();
    for_each_thread(leader, t)
    {
        struct perf_event *ev = NULL;
        int ret;

        if (installed >= NV_DBELL_MAX_BP_PER_CTX)
            break;
        get_task_struct(t);
        rcu_read_unlock();

        ret = nv_dbell_bp_register(t,
                                   vma->vm_start + NV_DBELL_OFFSET,
                                   nv_dbell_db_handler,
                                   ctx,
                                   &ev);
        if (ret == 0 && ev != NULL)
        {
            ctx->bp_events[installed++] = ev;
        }
        else
        {
            MC_TRACE(dbell, "bp_register", "tid=%d ret=%d", t->pid, ret);
        }

        put_task_struct(t);
        rcu_read_lock();
    }
    rcu_read_unlock();

    ctx->bp_event_count = installed;

    if (installed == 0)
    {
        MC_TRACE0(dbell, "no_breakpoints");
        return -ENOMEM;
    }
    return 0;
}

int nv_dbell_intercept_mmap(nv_state_t *nv,
                            struct vm_area_struct *vma,
                            nv_alloc_mapping_context_t *mmap_context)
{
    nv_dbell_ctx_t *ctx = NULL;
    int slot, ret;
    NvU64 bar_pa;
    unsigned long addr;

    /* Research kill-switch: skip diversion entirely so HOPPER_USERMODE_A
     * mmaps reach real BAR0/BAR1.  Required for measuring GPU-issued
     * doorbell writes — the shadow path absorbs them since x86 HW
     * debug regs only catch CPU MMU dereferences. */
    if (nv_dbell_disable_intercept)
        return 0;

    /* Only intercept the 64 KiB HOPPER_USERMODE_A window. */
    if (NV_VMA_SIZE(vma) != NV_DBELL_REGION_SIZE ||
        mmap_context->memArea.numRanges != 1)
    {
        return 0;
    }

    /* could be BAR0 or BAR1 on Hopper */
    bar_pa = mmap_context->memArea.pRanges[0].start;

    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (ctx == NULL)
        return -ENOMEM;

    ctx->tag     = NV_DBELL_VMA_TAG;
    ctx->nv      = nv;
    ctx->user_va = vma->vm_start;
    ctx->owner   = current;
    atomic_set(&ctx->bp_count, 0);

    slot = nv_dbell_claim_slot(ctx);
    if (slot < 0)
    {
        /* All 4 slots busy; fall through to normal BAR0 remap. */
        MC_TRACE(dbell, "slots_full", "pass_through=1 pid=%d bar0_pa=0x%llx",
            current->pid, (unsigned long long)bar_pa);
        kfree(ctx);
        return 0;
    }

    /* Allocate the shadow page.  GFP_KERNEL is fine — we're in
     * nvidia_mmap_helper which runs from a syscall context. */
    ctx->shadow_page = alloc_page(GFP_KERNEL | __GFP_ZERO);
    if (ctx->shadow_page == NULL)
    {
        ret = -ENOMEM;
        goto fail;
    }
    ctx->shadow_kva = page_address(ctx->shadow_page);

    /* Remap the real VF doorbell dword so the #DB handler can forward
     * writes.  ioremap must happen in sleepable context — do it here. */
    ctx->doorbell_iomap = ioremap(bar_pa + NV_DBELL_OFFSET, 4);
    if (ctx->doorbell_iomap == NULL)
    {
      MC_TRACE(dbell, "ioremap_fail", "bar0_pa=0x%llx", (unsigned long long)bar_pa);
      ret = -ENOMEM;
      goto fail;
    }

    /* Install the shadow page into the userspace VMA.  vm_insert_page
     * takes a reference on the page and sets up PTEs covering the full
     * 64 KiB region by mapping the SAME shadow page for every offset.
     * That's fine: libcuda only writes dword +0x90, and any stray
     * write elsewhere in the window hits the shadow harmlessly.
     *
     * We need VM_MIXEDMAP so vm_insert_page is accepted. */
    nv_vm_flags_set(vma, VM_MIXEDMAP | VM_DONTEXPAND | VM_DONTCOPY);

    for (addr = vma->vm_start; addr < vma->vm_end; addr += PAGE_SIZE)
    {
        ret = vm_insert_page(vma, addr, ctx->shadow_page);
        if (ret != 0)
        {
            MC_TRACE(dbell, "vm_insert_fail", "addr=0x%lx ret=%d", addr, ret);
            goto fail;
        }
    }

    ret = nv_dbell_arm_threads(ctx, vma);
    if (ret != 0)
        goto fail;

    NV_VMA_PRIVATE(vma) = ctx;

    MC_TRACE(dbell, "armed", "slot=%u pid=%d bps=%u "
                    "user_va=0x%lx bar0_pa=0x%llx",
                    ctx->slot_idx, current->pid, ctx->bp_event_count,
                    ctx->user_va, (unsigned long long)bar_pa);

    return 1;

fail:
    /* Release everything we acquired so far. */
    nv_dbell_unregister_bps(ctx);
    if (ctx->doorbell_iomap != NULL)
        iounmap(ctx->doorbell_iomap);
    nv_dbell_release_slot(ctx);
    if (ctx->shadow_page != NULL)
        __free_page(ctx->shadow_page);
    kfree(ctx);
    return ret;
}
