/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * mc-trace.h — the single definition of the mc1 kernel trace record.
 *
 * Every instrumented site in this fork emits through MC_TRACE(), which
 * produces one self-terminated record in the format:
 *
 *     mc1 <category>/<event> key=value key=value ...\n
 *
 * The grammar, the category set, and the full event name table are in
 * docs/reference/trace-format.md — that document is the authority; this
 * header and the userspace parser both implement it.
 *
 * Location: src/common/sdk/nvidia/inc/ is on the include path of all three
 * build domains that emit trace records — the RM core (nv-kernel.o), the
 * kernel-open modules, and nvidia-modeset (nvidia-push.c) — so one header
 * serves them all.
 */
#ifndef MC_TRACE_H
#define MC_TRACE_H

#include "nvtypes.h"                  /* NvU32, NvU64 */
#include "nv-kernel-interface-api.h"  /* NV_API_CALL — not always defined at include point */

/* Sigil.  One greppable token; also the run-together recovery anchor.
 * Bumping it (mc2) signals a grammar change tooling detects on line 1. */
#define MC_SIGIL "mc1"

/* Categories.  The bare token (rm, uvm, ...) is BOTH the mask selector and
 * the emitted category, so the two cannot drift.  Default-on is everything
 * except pte (the page-table-entry firehose: very high volume, and no
 * consumer in reverse/tools today).  Turn it on with mc_trace=0x3ff. */
#define foreach_mc_trace_cat                                                   \
  _(0, rm)                                                                     \
  _(1, uvm)                                                                    \
  _(2, fifo)                                                                   \
  _(3, mmu)                                                                    \
  _(4, dbell)                                                                  \
  _(5, pb)                                                                     \
  _(6, gsp)                                                                    \
  _(7, mmap)                                                                   \
  _(8, body)                                                                   \
  _(9, pte)

enum {
#define _(b, n) MC_TRACE_CAT_##n = (1u << (b)),
  foreach_mc_trace_cat
#undef _
      MC_TRACE_CAT_DEFAULT = (0x1ffu), /* all bits 0..8; pte (bit 9) off */
};

/* The emit primitive and the runtime mask.  Both are defined in
 * kernel-open/nvidia/os-interface.c and EXPORT_SYMBOL'd, so nvidia-uvm.ko
 * and nvidia-modeset.ko (which both already depend on nvidia.ko) can reach
 * them.  Same cross-module data-symbol pattern as os_page_size. */
void NV_API_CALL nv_trace_printf(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));
extern NvU32 nv_trace_mask;

/* Monotonic correlation id.  Stamped as id= on a multi-record operation's
 * header and every one of its body rows, so the parser reassembles them by
 * id rather than by adjacency — order-independent, and correct even in a
 * truncated or interleaved capture.  atomic under the hood. */
NvU32 NV_API_CALL nv_trace_next_id(void);

/* Event WITH fields.  The leading space before fmt and the trailing \n are
 * added here, so the missing-newline class of bug cannot recur.  The mask
 * test is a plain global load and an AND — no allocation, no locking — so
 * this is safe in the #DB doorbell-trap context (interrupts off, raw
 * spinlock only). */
#define MC_TRACE(cat, event, fmt, ...)                                     \
    do {                                                                   \
        if (nv_trace_mask & MC_TRACE_CAT_##cat)                            \
            nv_trace_printf(MC_SIGIL " " #cat "/" event " " fmt "\n",      \
                            ##__VA_ARGS__);                                \
    } while (0)

/* Event with NO fields (avoids a dangling trailing space). */
#define MC_TRACE0(cat, event)                                              \
    do {                                                                   \
        if (nv_trace_mask & MC_TRACE_CAT_##cat)                            \
            nv_trace_printf(MC_SIGIL " " #cat "/" event "\n");             \
    } while (0)

/* Array-value helpers for the genuinely-positional cases (pte/row's eight
 * PTEs, the body dump rows).  Pure macro expansion — no buffer, no
 * allocation — so they stay within the trap-safety envelope.  The array is
 * always the last field of a record. */
#define MC_ARR8   "[%#llx,%#llx,%#llx,%#llx,%#llx,%#llx,%#llx,%#llx]"
#define MC_ARR8V(p)                                                        \
    (unsigned long long)(p)[0], (unsigned long long)(p)[1],                \
    (unsigned long long)(p)[2], (unsigned long long)(p)[3],                \
    (unsigned long long)(p)[4], (unsigned long long)(p)[5],                \
    (unsigned long long)(p)[6], (unsigned long long)(p)[7]

#define MC_ARR4   "[%#010x,%#010x,%#010x,%#010x]"
#define MC_ARR4V(b, i) (b)[(i)], (b)[(i)+1], (b)[(i)+2], (b)[(i)+3]

#endif /* MC_TRACE_H */
