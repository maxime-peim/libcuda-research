// SPDX-License-Identifier: GPL-2.0
/*
 * nvidia-dbell: GPL-licensed shim that re-exports the kernel's
 * hardware-breakpoint API as EXPORT_SYMBOL thunks for nvidia.ko.
 *
 * Background: register_user_hw_breakpoint()/unregister_hw_breakpoint()
 * are exported EXPORT_SYMBOL_GPL.  This tree's nvidia.ko declares
 * MODULE_LICENSE("Dual MIT/GPL") (nv.c), which the kernel treats as
 * GPL-compatible, so strictly it could link the symbols directly.  The
 * shim keeps the GPL-only-API dependency in a module that is plainly
 * GPL-2.0 instead of inside the dual-licensed driver — and mirrors the
 * constraint the closed driver (MODULE_LICENSE("NVIDIA"), not
 * GPL-compatible) would actually face: for it, any direct call would
 * fail with "Unknown symbol register_user_hw_breakpoint".
 *
 * This module is GPL, so it may freely call the GPL-only symbols, and
 * then re-expose them via plain EXPORT_SYMBOL wrappers.  nvidia.ko
 * links against nv_dbell_bp_{register,unregister}() instead.
 *
 * Yan et al. §5.1 motivates the approach; see
 * "Rebuilding CUDA From Scratch" Part 4.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/sched.h>

#include "conftest.h"

#if defined(NV_REGISTER_USER_HW_BREAKPOINT_PRESENT) && defined(CONFIG_HAVE_HW_BREAKPOINT)
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#define NV_DBELL_HW_BREAKPOINT_USABLE 1
#endif

#include "nvidia_dbell.h"

int nv_dbell_bp_register(struct task_struct *task,
                         unsigned long user_va,
                         nv_dbell_overflow_fn cb,
                         void *ctx,
                         struct perf_event **out_event)
{
#if defined(NV_DBELL_HW_BREAKPOINT_USABLE)
    struct perf_event_attr attr;
    struct perf_event *event;

    if (task == NULL || cb == NULL || out_event == NULL)
        return -EINVAL;

    hw_breakpoint_init(&attr);
    attr.bp_addr = user_va;
    attr.bp_len  = HW_BREAKPOINT_LEN_4;
    attr.bp_type = HW_BREAKPOINT_W;

    /*
     * register_user_hw_breakpoint binds the breakpoint to `task`.  The
     * overflow handler runs synchronously in #DB trap context (on x86)
     * whenever `task` writes to `user_va`.
     */
    event = register_user_hw_breakpoint(&attr,
                                        (perf_overflow_handler_t)cb,
                                        ctx, task);
    if (IS_ERR(event))
        return PTR_ERR(event);

    *out_event = event;
    return 0;
#else
    (void)task; (void)user_va; (void)cb; (void)ctx; (void)out_event;
    return -ENOSYS;
#endif
}
EXPORT_SYMBOL(nv_dbell_bp_register);

void nv_dbell_bp_unregister(struct perf_event *event)
{
#if defined(NV_DBELL_HW_BREAKPOINT_USABLE)
    if (event != NULL && !IS_ERR(event))
        unregister_hw_breakpoint(event);
#else
    (void)event;
#endif
}
EXPORT_SYMBOL(nv_dbell_bp_unregister);

static int __init nvidia_dbell_init(void)
{
    pr_info("nvidia-dbell: loaded (hw_breakpoint %s)\n",
#if defined(NV_DBELL_HW_BREAKPOINT_USABLE)
            "present"
#else
            "unavailable, thunks return -ENOSYS"
#endif
           );
    return 0;
}

static void __exit nvidia_dbell_exit(void)
{
    pr_info("nvidia-dbell: unloaded\n");
}

module_init(nvidia_dbell_init);
module_exit(nvidia_dbell_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Maxime Peim");
MODULE_DESCRIPTION("GPL thunks exposing hw_breakpoint API to nvidia.ko");
MODULE_VERSION(NV_VERSION_STRING);
