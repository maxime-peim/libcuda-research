/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * nvidia-dbell: GPL shim exposing the x86 hardware-breakpoint API to
 * nvidia.ko.  register_user_hw_breakpoint is exported EXPORT_SYMBOL_GPL.
 * This tree's nvidia.ko is MODULE_LICENSE("Dual MIT/GPL"), which the
 * kernel treats as GPL-compatible, so it could link the symbol directly;
 * the shim keeps the GPL-only dependency in a module that is plainly
 * GPL-2.0.  See nvidia_dbell.c for the full reasoning.
 *
 * See "Rebuilding CUDA From Scratch" Part 4 for the broader
 * design (Hopper VF doorbell watchpoint, after Yan et al. §5.1).
 */

#ifndef _NVIDIA_DBELL_H_
#define _NVIDIA_DBELL_H_

#include <linux/types.h>

struct perf_event;
struct perf_event_attr;
struct perf_sample_data;
struct pt_regs;
struct task_struct;

/* overflow handler invoked from #DB trap context */
typedef void (*nv_dbell_overflow_fn)(struct perf_event *,
                                     struct perf_sample_data *,
                                     struct pt_regs *);

/*
 * Register an x86 write-watchpoint on `user_va` in the caller's task.
 * Width is 4 bytes (the VF doorbell dword) and the watchpoint fires on
 * writes only; reads are not trapped.  On success,
 * *out_event is set and ownership stays with the shim — caller must
 * unregister via nv_dbell_bp_unregister().
 *
 * Safe to call from process context only.  Returns 0 on success or a
 * negative errno (-ENOSYS if the running kernel lacks hw_breakpoint
 * support).
 */
int nv_dbell_bp_register(struct task_struct *task,
                         unsigned long user_va,
                         nv_dbell_overflow_fn cb,
                         void *ctx,
                         struct perf_event **out_event);

/*
 * Unregister a breakpoint previously returned by nv_dbell_bp_register.
 * Safe to call from any context that may sleep.
 */
void nv_dbell_bp_unregister(struct perf_event *event);

#endif /* _NVIDIA_DBELL_H_ */
