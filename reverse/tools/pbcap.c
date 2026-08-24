/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * pbcap.c — pushbuffer snapshot shim for cudaMemcpy + GPU-doorbell watchpoint
 *
 * LD_PRELOAD library that intercepts mmap/munmap/open/openat/cudaMemcpy and
 * (when enabled) traps on every write to the Hopper VF doorbell so we
 * can dump pushbuffer contents at each GPU work-submit.
 *
 * The doorbell watchpoint uses mprotect(PROT_READ) on HOPPER_USERMODE_A
 * mappings + a SIGSEGV handler that sets TF for single-step + a SIGTRAP
 * handler that reads the work-submit-token and dumps state.  USERD writes
 * (on a separate sysmem page) don't trap; only BAR0 writes do.
 *
 * NOTE (2026-05-07): the userspace watchpoint + cudaMemcpy snapshot
 * paths are now OFF by default.  The kernel module nvidia-dbell.ko (see
 * "Rebuilding CUDA From Scratch" Part 4 and
 * kernel-open/nvidia/nv-doorbell-watch.c) does
 * the same interception in the #DB trap handler BEFORE PBDMA can
 * consume the submission, and emits pb/submit + pb/bytes via ftrace.
 * Running the userspace paths on top adds 50-100× wallclock overhead
 * for no additional information.  Re-enable for A/B comparison or
 * legacy reproduction via PBCAP_DBELL=1 and PBCAP_MEMCPY_SNAPSHOT=1.
 *
 * Env vars:
 *   PBCAP_DIR              output directory (default: /tmp/pbcap)
 *   PBCAP_MAX_BYTES        per-mapping snapshot cap (default: 16 MiB)
 *   PBCAP_VERBOSE          1 to print per-hook lines on stderr
 *   PBCAP_DBELL            1 to arm the userspace doorbell watchpoint
 *                          (default: 0 — kernel-side is active in
 *                          modern builds; see docs/tracing_cuda.md)
 *   PBCAP_DBELL_SAMPLE     snapshot every Nth doorbell (default: 1)
 *   PBCAP_DBELL_SYNC       1=snapshot in SIGTRAP context, 0=worker thread
 *   PBCAP_MEMCPY_SNAPSHOT  1 to snapshot every mapping before+after
 *                          each cudaMemcpy (default: 0)
 *
 * Build:  make
 * Use:    LD_PRELOAD=$PWD/libpbcap.so ./bin/cuda_reference
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <emmintrin.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

/* ── configuration ───────────────────────────────────────────────────────── */

#define MAX_FDS           128
#define MAX_MAPPINGS      256
#define DEFAULT_MAX_BYTES (16u * 1024u * 1024u)

static const char *g_outdir    = "/tmp/pbcap";
static size_t      g_max_bytes = DEFAULT_MAX_BYTES;
static int         g_verbose   = 0;
/*
 * Default OFF: the kernel-side watchpoint (nvidia-dbell.ko) supersedes
 * this path and runs at ~zero cost.  This userspace watchpoint is our
 * own earlier attempt, not Yan et al.'s method — their §5.1–§5.2 is the
 * kernel-side one, and their §3 argues userspace cannot win this race
 * (docs/findings.md §12.1 measures why).  Export PBCAP_DBELL=1 to A/B
 * the paths or to reproduce that negative result.
 */
static int      g_dbell_arm    = 0;
static unsigned g_dbell_sample = 1; /* snapshot every Nth doorbell */
static int      g_dbell_sync =
    1; /* 1=sync-in-handler (beats GPU), 0=worker-thread */
/*
 * Per-cudaMemcpy snapshot of every nvidia-backed mapping — used to be
 * the only way to capture pushbuffer content, now redundant because
 * the kernel emits pb/bytes with full method streams.  Default
 * OFF; set PBCAP_MEMCPY_SNAPSHOT=1 to restore.
 */
static int g_memcpy_snapshot = 0;

/* Hopper doorbell facts — see libmc/mc/mc_submit.c and
 * docs/gpfifo_pushbuffer_reference.md §10-11.
 * HOPPER_USERMODE_A (class 0xC661) is an mmap of the non-privileged VF
 * window: a 64 KiB range whose +0x90 dword is the VF_DOORBELL.  libcuda
 * and mc allocate both the BAR0 and the BAR1 variant and ring the BAR1
 * one; the +0x90 offset is the same either way (findings.md §12.6).
 * Ringing it with a 32-bit work-submit-token wakes the GPU host
 * scheduler after USERD GPPut has been advanced.  The mapping is
 * always 64 KiB on this driver version. */
#define HOPPER_USERMODE_BAR0_LEN 65536u
#define HOPPER_VF_DOORBELL_OFF   0x90u

/* ── fd table: which fds are /dev/nvidia* ───────────────────────────────────
 */

struct fd_info
{
  int  fd;       /* -1 = empty slot */
  char name[64]; /* e.g. "/dev/nvidia0" */
};
static struct fd_info  g_fds[MAX_FDS];
static pthread_mutex_t g_fds_mu = PTHREAD_MUTEX_INITIALIZER;

static int fd_is_nvidia(const char *path)
{
  if (!path)
    return 0;
  if (strncmp(path, "/dev/nvidia", 11) != 0)
    return 0;
  /* accept /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm, etc. */
  return 1;
}

static void fd_record(int fd, const char *path)
{
  if (fd < 0)
    return;
  pthread_mutex_lock(&g_fds_mu);
  for (int i = 0; i < MAX_FDS; i++)
  {
    if (g_fds[i].fd == -1)
    {
      g_fds[i].fd = fd;
      snprintf(g_fds[i].name, sizeof(g_fds[i].name), "%s", path);
      break;
    }
  }
  pthread_mutex_unlock(&g_fds_mu);
}

static void fd_forget(int fd)
{
  if (fd < 0)
    return;
  pthread_mutex_lock(&g_fds_mu);
  for (int i = 0; i < MAX_FDS; i++)
  {
    if (g_fds[i].fd == fd)
    {
      g_fds[i].fd      = -1;
      g_fds[i].name[0] = '\0';
      break;
    }
  }
  pthread_mutex_unlock(&g_fds_mu);
}

static const char *fd_lookup(int fd)
{
  const char *name = NULL;
  pthread_mutex_lock(&g_fds_mu);
  for (int i = 0; i < MAX_FDS; i++)
  {
    if (g_fds[i].fd == fd)
    {
      name = g_fds[i].name;
      break;
    }
  }
  pthread_mutex_unlock(&g_fds_mu);
  return name;
}

/* ── mapping table: track active nvidia-backed mmaps ─────────────────────── */

enum map_kind
{
  MAP_KIND_OTHER = 0,
  MAP_KIND_USERMODE_BAR0, /* HOPPER_USERMODE_A — arm the doorbell watchpoint */
};

struct map_info
{
  void         *addr; /* NULL = empty slot */
  size_t        length;
  int           fd; /* original fd (for label) */
  char          fdname[64];
  off_t         offset;
  int           prot;
  enum map_kind kind;
};
static struct map_info g_maps[MAX_MAPPINGS];
static pthread_mutex_t g_maps_mu = PTHREAD_MUTEX_INITIALIZER;

/* Classify a freshly-created nvidia-backed mapping.  Heuristic: the Hopper
 * BAR0 VF window is exactly 64 KiB, writable, and mmap'd on /dev/nvidia0.
 * Everything else (pushbuffer, USERD, gpfifo ring, semaphores, ...) is
 * left as MAP_KIND_OTHER and captured via the existing cudaMemcpy
 * snapshot path. */
static enum map_kind classify_map(const char *fdname, size_t length, int prot)
{
  if (!fdname)
    return MAP_KIND_OTHER;
  if (length == HOPPER_USERMODE_BAR0_LEN && (prot & PROT_WRITE)
      && strstr(fdname, "/dev/nvidia0"))
    return MAP_KIND_USERMODE_BAR0;
  return MAP_KIND_OTHER;
}

/* Forward decls — watchpoint lives below but map_record needs to arm. */
static void dbell_arm_mapping(void *addr, size_t length);

static void map_record(void *addr, size_t length, int fd, off_t offset,
                       int prot)
{
  const char *name = fd_lookup(fd);
  if (!name)
    return; /* not an nvidia fd — ignore */
  enum map_kind kind = classify_map(name, length, prot);
  pthread_mutex_lock(&g_maps_mu);
  for (int i = 0; i < MAX_MAPPINGS; i++)
  {
    if (g_maps[i].addr == NULL)
    {
      g_maps[i].length = length;
      g_maps[i].fd     = fd;
      g_maps[i].offset = offset;
      g_maps[i].prot   = prot;
      g_maps[i].kind   = kind;
      snprintf(g_maps[i].fdname, sizeof(g_maps[i].fdname), "%s", name);
      /* Publish addr LAST with release semantics so the signal-safe
       * reader sees fully-populated metadata when addr != NULL. */
      __atomic_store_n(&g_maps[i].addr, addr, __ATOMIC_RELEASE);
      break;
    }
  }
  pthread_mutex_unlock(&g_maps_mu);
  /* Arm after releasing the lock — the signal handler may take it. */
  if (kind == MAP_KIND_USERMODE_BAR0 && g_dbell_arm)
    dbell_arm_mapping(addr, length);
}

static void map_forget(void *addr, size_t length)
{
  pthread_mutex_lock(&g_maps_mu);
  for (int i = 0; i < MAX_MAPPINGS; i++)
  {
    if (g_maps[i].addr == addr && g_maps[i].length == length)
    {
      g_maps[i].addr = NULL;
      break;
    }
  }
  pthread_mutex_unlock(&g_maps_mu);
}

/* Lookup a mapping by fault address.  Returns the USERMODE_BAR0 map_info
 * whose range contains `fault`, or NULL if none matches.  Must be
 * signal-safe — uses only atomic table reads, no mutex.  The g_maps
 * table is updated under a mutex; readers see a possibly-stale snapshot
 * which is fine because we only care about mappings that have completed
 * arming before any fault can arrive there. */
static struct map_info *dbell_find_mapping(void *fault)
{
  uintptr_t f = (uintptr_t)fault;
  for (int i = 0; i < MAX_MAPPINGS; i++)
  {
    void *a = __atomic_load_n(&g_maps[i].addr, __ATOMIC_ACQUIRE);
    if (!a)
      continue;
    if (g_maps[i].kind != MAP_KIND_USERMODE_BAR0)
      continue;
    uintptr_t s = (uintptr_t)a;
    if (f >= s && f < s + g_maps[i].length)
      return &g_maps[i];
  }
  return NULL;
}

/* ── timestamps + NDJSON timeline emission ──────────────────────────────── */
/*
 * The timeline.ndjson log is one line per event with CLOCK_MONOTONIC ns.
 * All writes go through write(2) on a pre-opened fd so they are
 * async-signal-safe (the doorbell watchpoint's SIGTRAP handler calls
 * `emit_event` — fprintf / malloc in a signal handler would deadlock).
 * A clock anchor (CLOCK_MONOTONIC + CLOCK_REALTIME, captured back-to-back
 * at pbcap init) is emitted as the first event so offline merger tools
 * can translate strace's wall-clock `-ttt` timestamps into the same
 * monotonic timebase.
 */

static int g_timeline_fd = -1;

static inline uint64_t mono_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint64_t real_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline pid_t raw_gettid(void)
{
  return (pid_t)syscall(SYS_gettid);
}

/* Write up to len bytes; retry on EINTR.  Signal-safe. */
static void safe_write(int fd, const char *buf, size_t len)
{
  if (fd < 0)
    return;
  while (len > 0)
  {
    ssize_t n = write(fd, buf, len);
    if (n > 0)
    {
      buf += (size_t)n;
      len -= (size_t)n;
    }
    else if (n < 0 && errno == EINTR)
    {
      continue;
    }
    else
    {
      break; /* short or failed — can't report from signal handler */
    }
  }
}

/*
 * Emit a timeline event.  `data_json` is a raw JSON object fragment
 * (e.g., `"dst":"0x7f...","src":"0x...","count":131072`) — caller is
 * responsible for valid JSON syntax.  Kept signal-safe: no malloc, no
 * floating point, no fprintf.
 */
static void emit_event(const char *kind, const char *data_json)
{
  if (g_timeline_fd < 0)
    return;
  /* 512-byte line cap.  Larger events need to split their payload
   * (none currently do). */
  char  line[512];
  int   n;
  pid_t pid = getpid();
  pid_t tid = raw_gettid();
  if (data_json && *data_json)
    n = snprintf(line, sizeof(line),
                 "{\"ts_ns\":%" PRIu64
                 ",\"src\":\"pbcap\",\"pid\":%d,\"tid\":%d,"
                 "\"kind\":\"%s\",\"data\":{%s}}\n",
                 mono_ns(), (int)pid, (int)tid, kind, data_json);
  else
    n = snprintf(line, sizeof(line),
                 "{\"ts_ns\":%" PRIu64
                 ",\"src\":\"pbcap\",\"pid\":%d,\"tid\":%d,"
                 "\"kind\":\"%s\",\"data\":{}}\n",
                 mono_ns(), (int)pid, (int)tid, kind);
  if (n < 0)
    return;
  if ((size_t)n >= sizeof(line))
    n = (int)sizeof(line) - 1;
  safe_write(g_timeline_fd, line, (size_t)n);
}

/*
 * Format a JSON payload and emit it.  emit_event still does the write; this
 * only saves every hook from declaring its own scratch buffer.  256 bytes is
 * the widest payload any hook produces, and emit_event caps the assembled line
 * at 512 regardless.
 *
 * Not for use from the SIGTRAP handler.  emit_event is async-signal-safe by
 * construction — write(2) on a pre-opened fd, no malloc — and the doorbell path
 * keeps calling it directly with an already-formatted buffer rather than
 * widening that surface with vsnprintf.
 */
static void emit_eventf(const char *kind, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void emit_eventf(const char *kind, const char *fmt, ...)
{
  char    data[256];
  va_list ap;

  va_start(ap, fmt);
  vsnprintf(data, sizeof(data), fmt, ap);
  va_end(ap);
  emit_event(kind, data);
}

/* ── real-function pointers (resolved lazily with dlsym RTLD_NEXT) ──────── */

static int (*real_open)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_close)(int);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_munmap)(void *, size_t);
static cudaError_t (*real_cudaMemcpy)(void *, const void *, size_t,
                                      enum cudaMemcpyKind);
static cudaError_t (*real_cudaHostAlloc)(void **, size_t, unsigned int);
static cudaError_t (*real_cudaMalloc)(void **, size_t);
static cudaError_t (*real_cudaInitDevice)(int, unsigned int, unsigned int);
static cudaError_t (*real_cudaHostRegister)(void *, size_t, unsigned int);
static cudaError_t (*real_cudaHostUnregister)(void *);
static cudaError_t (*real_cudaLaunchKernel)(const void *, dim3, dim3, void **,
                                            size_t, cudaStream_t);

#define RESOLVE(fn)                                                       \
  do                                                                      \
  {                                                                       \
    if (!real_##fn)                                                       \
    {                                                                     \
      real_##fn = dlsym(RTLD_NEXT, #fn);                                  \
      if (!real_##fn)                                                     \
      {                                                                   \
        fprintf(stderr, "pbcap: dlsym(%s) failed: %s\n", #fn, dlerror()); \
      }                                                                   \
    }                                                                     \
  } while (0)

/* ── one-time init ───────────────────────────────────────────────────────── */

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

/* Forward decl for watchpoint installer. */
static void dbell_install_handlers(void);

static void do_init(void)
{
  const char *e;
  if ((e = getenv("PBCAP_DIR")) && *e)
    g_outdir = e;
  /* Every getenv() below tests *e as well, because trace_cuda.sh forwards
   * the whole PBCAP_* set through sudo unconditionally and passes an empty
   * string for the ones the caller did not set.  An empty string is a
   * present variable, so testing the pointer alone would parse "" as 0 --
   * for PBCAP_MAX_BYTES that silently caps every mapping at zero bytes and
   * snapshotting produces nothing at all, with no diagnostic. */
  if ((e = getenv("PBCAP_MAX_BYTES")) && *e)
    g_max_bytes = (size_t)strtoull(e, NULL, 0);
  if ((e = getenv("PBCAP_VERBOSE")) && *e)
    g_verbose = atoi(e);
  if ((e = getenv("PBCAP_DBELL")) && *e)
    g_dbell_arm = atoi(e);
  if ((e = getenv("PBCAP_DBELL_SAMPLE")) && *e)
    g_dbell_sample = (unsigned)strtoul(e, NULL, 0);
  if (g_dbell_sample == 0)
    g_dbell_sample = 1;
  if ((e = getenv("PBCAP_DBELL_SYNC")) && *e)
    g_dbell_sync = atoi(e);
  if ((e = getenv("PBCAP_MEMCPY_SNAPSHOT")) && *e)
    g_memcpy_snapshot = atoi(e);

  for (int i = 0; i < MAX_FDS; i++)
    g_fds[i].fd = -1;
  for (int i = 0; i < MAX_MAPPINGS; i++)
    g_maps[i].addr = NULL;

  /* best-effort mkdir; if it already exists, fine */
  mkdir(g_outdir, 0755);

  RESOLVE(open);
  RESOLVE(openat);
  RESOLVE(close);
  RESOLVE(mmap);
  RESOLVE(munmap);

  /* Open the NDJSON timeline log.  O_APPEND so multi-process (fork)
   * captures stay coherent.  Use plain open, not our hook, so it
   * doesn't recurse. */
  char tlpath[512];
  snprintf(tlpath, sizeof(tlpath), "%s/timeline.ndjson", g_outdir);
  g_timeline_fd =
      real_open
          ? real_open(tlpath, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644)
          : open(tlpath, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
  if (g_timeline_fd < 0 && g_verbose)
    fprintf(stderr, "pbcap: timeline open(%s) failed: %s\n", tlpath,
            strerror(errno));

  /* Emit clock anchor so offline tools can translate strace's CLOCK_REALTIME
   * `-ttt` timestamps to our CLOCK_MONOTONIC timebase.  The two clock_gettime
   * calls are back-to-back; delta is nanoseconds — negligible for the
   * ms-scale correlation we need. */
  uint64_t mono = mono_ns();
  uint64_t real = real_ns();
  char     anchor[256];
  int      na = snprintf(anchor, sizeof(anchor),
                         "\"mono_ns\":%" PRIu64 ",\"real_ns\":%" PRIu64, mono, real);
  if (na > 0)
    emit_event("pbcap.init", anchor);

  if (g_dbell_arm)
    dbell_install_handlers();

  if (g_verbose)
    fprintf(stderr,
            "pbcap: initialized, outdir=%s max_bytes=%zu dbell=%d sample=%u\n",
            g_outdir, g_max_bytes, g_dbell_arm, g_dbell_sample);
}

#define INIT() pthread_once(&g_init_once, do_init)

/* ── snapshot engine ─────────────────────────────────────────────────────── */

static _Atomic unsigned g_snap_idx = 0;

/*
 * A per-mapping entry built from /proc/self/maps scanning.
 * Distinct from map_info which requires our open() hook to have fired.
 */
struct snap_entry
{
  void  *addr;
  size_t length;
  int    readable; /* 1 if originally r--s/rw-s; 0 if -w-s (needs mprotect) */
  char   tag[64];  /* "nvidia0", "nvidiactl", "nvidia-uvm", etc. */
};

/*
 * Scan /proc/self/maps and collect every small mapping backed by a
 * /dev/nvidia* file.
 *
 * Why not filter by device major 0xc3: on Linux, /proc/self/maps reports the
 * *filesystem* device (devtmpfs = 00:05) for character device mappings, not
 * the char device's own major:minor.  The pathname field is the reliable
 * signal.
 *
 * Why include write-only (-w-s) mappings: the pushbuffer and GPFIFO ring are
 * mapped PROT_WRITE only by the nvidia driver to prevent accidental CPU reads
 * (reads from WC memory stall on a PCIe round-trip).  We temporarily add
 * PROT_READ via mprotect() for the snapshot, then restore.  This is safe: the
 * GPU's address spaces are independent and unaffected by CPU-side VMA flags.
 *
 * Returns the number of entries populated into `out` (≤ max_out).
 */
static int scan_nvidia_maps(struct snap_entry *out, int max_out)
{
  FILE *f = fopen("/proc/self/maps", "r");
  if (!f)
    return 0;

  int  n = 0;
  char line[512];
  while (n < max_out && fgets(line, sizeof(line), f))
  {
    uintptr_t     start, end;
    char          perms[8], offset_str[32], dev_str[16];
    unsigned long inode;
    char          pathname[256] = "";

    /* format: start-end perms offset dev inode [pathname] */
    int fields = sscanf(line, "%lx-%lx %7s %31s %15s %lu %255s",
                        (unsigned long *)&start, (unsigned long *)&end, perms,
                        offset_str, dev_str, &inode, pathname);
    if (fields < 6)
      continue;

    /* match any mapping backed by /dev/nvidia* */
    if (!strstr(pathname, "/dev/nvidia"))
      continue;

    size_t length = end - start;
    if (length == 0 || length > g_max_bytes)
      continue;

    /* record both readable (r--s/rw-s) and write-only (-w-s) mappings;
     * we handle the -w-s case with mprotect in snapshot_all() */
    int readable  = (perms[0] == 'r');
    int writeable = (perms[1] == 'w');
    (void)writeable; /* used implicitly in snapshot logic */

    out[n].addr     = (void *)start;
    out[n].length   = length;
    out[n].readable = readable;

    const char *p = strrchr(pathname, '/');
    snprintf(out[n].tag, sizeof(out[n].tag), "%s", p ? p + 1 : pathname);
    n++;
  }
  fclose(f);
  return n;
}

/*
 * Dump one snapshot of every small NVIDIA-backed mapping.
 * Primary detection: /proc/self/maps scan by device major 0xc3.
 * Supplement: fd-tracking table g_maps[] (adds labels for UVM mappings
 * whose open() went through glibc and was caught by our hook).
 *
 * Files land in ${PBCAP_DIR}/snap-<idx>-<phase>-<tag>-<addr>-len<N>.bin
 */
static void snapshot_all(const char *phase, unsigned idx)
{
  struct snap_entry entries[MAX_MAPPINGS];
  int               n = scan_nvidia_maps(entries, MAX_MAPPINGS);

  /*
   * Supplement with any fd-tracked mappings not already found by the scan
   * (e.g. the UVM mapping opened via glibc whose fd we did catch).
   * These are always treated as readable since we tracked them via mmap().
   */
  pthread_mutex_lock(&g_maps_mu);
  for (int i = 0; i < MAX_MAPPINGS && n < MAX_MAPPINGS; i++)
  {
    if (!g_maps[i].addr || g_maps[i].length > g_max_bytes)
      continue;
    int dup = 0;
    for (int j = 0; j < n; j++)
    {
      if (entries[j].addr == g_maps[i].addr)
      {
        dup = 1;
        break;
      }
    }
    if (dup)
      continue;
    entries[n].addr     = g_maps[i].addr;
    entries[n].length   = g_maps[i].length;
    entries[n].readable = 1;
    const char *name    = g_maps[i].fdname;
    const char *p       = strrchr(name, '/');
    snprintf(entries[n].tag, sizeof(entries[n].tag), "%s", p ? p + 1 : name);
    n++;
  }
  pthread_mutex_unlock(&g_maps_mu);

  if (g_verbose)
    fprintf(stderr, "pbcap: snap-%05u-%s: %d mappings to dump\n", idx, phase,
            n);

  for (int i = 0; i < n; i++)
  {
    char path[512];
    snprintf(path, sizeof(path), "%s/snap-%05u-%s-%s-%p-len%zx.bin", g_outdir,
             idx, phase, entries[i].tag, entries[i].addr, entries[i].length);

    FILE *f = fopen(path, "wb");
    if (!f)
    {
      if (g_verbose)
        fprintf(stderr, "pbcap: fopen(%s) failed: %s\n", path, strerror(errno));
      continue;
    }

    /*
     * For write-only (-w-s) mappings (pushbuffer, GPFIFO ring), temporarily
     * add PROT_READ so fwrite() can load the bytes without SIGBUS.
     * The GPU's address spaces are independent of CPU VMA flags — this is
     * safe and only adds a brief read-permission window on these pages.
     */
    int did_mprotect = 0;
    if (!entries[i].readable)
    {
      if (mprotect(entries[i].addr, entries[i].length, PROT_READ | PROT_WRITE)
          == 0)
      {
        did_mprotect = 1;
      }
      else
      {
        if (g_verbose)
          fprintf(stderr, "pbcap: mprotect(%p, %zu) failed: %s — skipping\n",
                  entries[i].addr, entries[i].length, strerror(errno));
        fclose(f);
        continue;
      }
    }

    size_t wrote = fwrite(entries[i].addr, 1, entries[i].length, f);
    fclose(f);

    /* restore original write-only protection */
    if (did_mprotect)
      mprotect(entries[i].addr, entries[i].length, PROT_WRITE);

    if (g_verbose)
      fprintf(stderr, "pbcap: snap-%05u-%s %s %p len=%zu wrote=%zu%s\n", idx,
              phase, entries[i].tag, entries[i].addr, entries[i].length, wrote,
              did_mprotect ? " (mprotected)" : "");
  }
}

/* ── signal-safe snapshot path (no fopen/malloc/fprintf) ─────────────────── */
/*
 * The async worker-thread path loses the race: by the time it dequeues a
 * snapshot request, the GPU's PBDMA has already fetched the pushbuffer
 * and CUDA may have overwritten it.  To capture the pushbuffer bytes at
 * the moment of submission, we need to run the snapshot synchronously
 * inside the SIGTRAP handler — which constrains us to async-signal-safe
 * primitives (no fopen/fprintf/malloc).  This implementation uses only
 * open(2)/read(2)/write(2)/close(2)/mprotect(2).
 */

/* Read /proc/self/maps into a caller-supplied buffer using only
 * signal-safe syscalls.  Returns bytes read, or 0 on failure. */
static size_t proc_maps_read_sigsafe(char *buf, size_t buflen)
{
  int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return 0;
  size_t total = 0;
  while (total + 1 < buflen)
  {
    ssize_t n = read(fd, buf + total, buflen - 1 - total);
    if (n > 0)
      total += (size_t)n;
    else if (n == 0)
      break;
    else if (errno == EINTR)
      continue;
    else
      break;
  }
  close(fd);
  if (total < buflen)
    buf[total] = '\0';
  else
    buf[buflen - 1] = '\0';
  return total;
}

/* Parse one /proc/self/maps line into {start, end, perms[4], pathname}.
 * Signal-safe: no strtok/sscanf (sscanf is NOT signal-safe on glibc —
 * uses locale).  Returns 1 if the line was an nvidia-backed mapping
 * that fits size criteria, 0 otherwise.  `line` is mutated in place. */
static int parse_maps_line_sigsafe(char *line, size_t *out_start,
                                   size_t *out_end, char perms[4],
                                   const char **out_path)
{
  /* Format: "start-end perms offset dev inode pathname\n" */
  char *p = line;
  /* start */
  size_t v = 0;
  while (*p && *p != '-')
  {
    char c = *p++;
    int  d = (c >= '0' && c <= '9')   ? c - '0'
             : (c >= 'a' && c <= 'f') ? c - 'a' + 10
             : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                                      : -1;
    if (d < 0)
      return 0;
    v = (v << 4) | (size_t)d;
  }
  if (*p != '-')
    return 0;
  *out_start = v;
  p++;
  /* end */
  v = 0;
  while (*p && *p != ' ')
  {
    char c = *p++;
    int  d = (c >= '0' && c <= '9')   ? c - '0'
             : (c >= 'a' && c <= 'f') ? c - 'a' + 10
             : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                                      : -1;
    if (d < 0)
      return 0;
    v = (v << 4) | (size_t)d;
  }
  if (*p != ' ')
    return 0;
  *out_end = v;
  p++;
  /* perms: 4 chars */
  if (!*p || !p[1] || !p[2] || !p[3])
    return 0;
  perms[0] = p[0];
  perms[1] = p[1];
  perms[2] = p[2];
  perms[3] = p[3];
  p += 4;
  /* skip " offset dev inode " — exactly 3 whitespace-separated tokens
   * between perms and pathname.  Off-by-one here ate the pathname and
   * was the reason sync snapshots produced 0 files. */
  for (int i = 0; i < 3; i++)
  {
    while (*p == ' ')
      p++;
    while (*p && *p != ' ' && *p != '\n')
      p++;
  }
  while (*p == ' ')
    p++;
  *out_path = p;
  /* strip trailing newline from pathname */
  char *e = p;
  while (*e && *e != '\n')
    e++;
  *e = '\0';
  return 1;
}

/* Write a buffer to a file by path, signal-safe.  Truncates/creates.
 * Returns 0 on success, -1 on any failure (silent — we can't printf
 * from here). */
static int write_file_sigsafe(const char *path, const void *data, size_t len)
{
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0)
    return -1;
  const char *p = (const char *)data;
  while (len > 0)
  {
    ssize_t n = write(fd, p, len);
    if (n > 0)
    {
      p += n;
      len -= (size_t)n;
    }
    else if (n < 0 && errno == EINTR)
    {
      continue;
    }
    else
    {
      close(fd);
      return -1;
    }
  }
  close(fd);
  return 0;
}

/* Format a positive unsigned into a fixed-width zero-padded base-10
 * string.  signal-safe.  Writes at most `width` chars into `out` (no
 * terminator).  Returns pointer past the last written byte. */
static char *u32_to_fixed(char *out, unsigned v, int width)
{
  char tmp[16];
  int  n = 0;
  if (v == 0)
    tmp[n++] = '0';
  while (v > 0 && n < (int)sizeof(tmp))
  {
    tmp[n++] = '0' + (v % 10);
    v /= 10;
  }
  /* left-pad with '0' up to `width` */
  while (n < width)
    tmp[n++] = '0';
  /* reverse into `out` */
  for (int i = n - 1; i >= 0; i--)
    *out++ = tmp[i];
  return out;
}

/* Format a uintptr_t as lowercase hex with no "0x" prefix.  signal-safe. */
static char *uptr_to_hex(char *out, uintptr_t v)
{
  char        tmp[20];
  int         n = 0;
  const char *d = "0123456789abcdef";
  if (v == 0)
    tmp[n++] = '0';
  while (v > 0 && n < (int)sizeof(tmp))
  {
    tmp[n++] = d[v & 0xf];
    v >>= 4;
  }
  for (int i = n - 1; i >= 0; i--)
    *out++ = tmp[i];
  return out;
}

/*
 * Synchronous snapshot called from the SIGTRAP handler, before mprotect
 * re-arms the BAR0 page.  Walks /proc/self/maps in the handler (safe:
 * open/read/close are signal-safe), writes
 * snap-<idx>-dbell-<tag>-<addr>-len<L>.bin files via open/write/close.
 *
 * This is THE critical path for catching the pushbuffer with integrity:
 * PBDMA has been notified (the store retired during TF single-step) but
 * the GPU's actual fetch of the pushbuffer takes a nonzero time window;
 * if this runs fast enough (<1 µs per mapping for the small ones CUDA
 * cares about), we beat the GPU.
 */
static void snapshot_all_sigsafe(unsigned idx)
{
  /* Static buffer for /proc/self/maps — 256 KiB is enough for thousands
   * of mappings.  Single-threaded access from the signal handler; the
   * only reentrancy concern is nested signals, which are blocked by
   * default within the handler's signal mask. */
  static char maps_buf[256 * 1024];
  size_t      mlen = proc_maps_read_sigsafe(maps_buf, sizeof(maps_buf));
  if (mlen == 0)
    return;

  /* Walk lines in-place, parsing each and dumping nvidia-backed
   * mappings whose length is <= g_max_bytes. */
  char *line = maps_buf;
  while (line < maps_buf + mlen && *line)
  {
    char *nl = line;
    while (nl < maps_buf + mlen && *nl != '\n')
      nl++;
    if (nl >= maps_buf + mlen)
      break;
    *nl = '\0'; /* terminate this line */

    size_t      start = 0, end = 0;
    char        perms[4];
    const char *path = NULL;
    if (parse_maps_line_sigsafe(line, &start, &end, perms, &path) && path
        && *path)
    {
      /* Match /dev/nvidia* in pathname. */
      const char *match = path;
      int         is_nv = 0;
      while (*match)
      {
        if (match[0] == '/' && match[1] == 'd' && match[2] == 'e'
            && match[3] == 'v' && match[4] == '/' && match[5] == 'n'
            && match[6] == 'v' && match[7] == 'i' && match[8] == 'd'
            && match[9] == 'i' && match[10] == 'a')
        {
          is_nv = 1;
          break;
        }
        match++;
      }
      size_t length = end - start;
      if (is_nv && length > 0 && length <= g_max_bytes)
      {
        /* Extract the short tag (basename after last '/') for filename. */
        const char *tag = match;
        /* Advance to basename: move past "/dev/" */
        tag += 5; /* "/dev/" */
        /* In-memory bytes are tag like "nvidia0", "nvidiactl", "nvidia-uvm".
         * Truncate at any trailing whitespace/newline. */
        int tag_len = 0;
        while (tag[tag_len] && tag[tag_len] != ' ' && tag[tag_len] != '\n'
               && tag_len < 32)
          tag_len++;

        /* Build output path:
         *   <g_outdir>/snap-<idx:05>-dbell-<tag>-<addr>-len<hex>.bin
         * All signal-safe concat. */
        char        outpath[512];
        char       *q  = outpath;
        const char *od = g_outdir;
        while (*od)
          *q++ = *od++;
        const char *lit = "/snap-";
        while (*lit)
          *q++ = *lit++;
        q   = u32_to_fixed(q, idx, 5);
        lit = "-dbell-";
        while (*lit)
          *q++ = *lit++;
        for (int i = 0; i < tag_len; i++)
          *q++ = tag[i];
        lit = "-0x";
        while (*lit)
          *q++ = *lit++;
        q   = uptr_to_hex(q, start);
        lit = "-len";
        while (*lit)
          *q++ = *lit++;
        q   = uptr_to_hex(q, length);
        lit = ".bin";
        while (*lit)
          *q++ = *lit++;
        *q = '\0';

        /* Handle write-only mappings: temporarily add PROT_READ.  We
         * must NOT do this on our BAR0 page (it's currently PROT_READ
         * already — already armed).  Detect by perms[1] == '-'. */
        int did_mprotect = 0;
        if (perms[1] == '-' && perms[0] != 'r')
        {
          if (mprotect((void *)start, length, PROT_READ | PROT_WRITE) == 0)
            did_mprotect = 1;
          else
            continue; /* skip this mapping */
        }

        write_file_sigsafe(outpath, (const void *)start, length);

        if (did_mprotect)
          mprotect((void *)start, length, PROT_WRITE);
      }
    }

    line = nl + 1;
  }
}

/* ── doorbell watchpoint ─────────────────────────────────────────────────── */
/*
 * Mechanism:
 *   1. At mmap() of a HOPPER_USERMODE_A 64-KiB page, mprotect(PROT_READ)
 *      to strip write permission.
 *   2. SIGSEGV handler catches the first store attempt.  Sets TF in
 *      ucontext EFLAGS, mprotects back to PROT_READ|PROT_WRITE, returns.
 *      Kernel replays the store.
 *   3. SIGTRAP handler runs immediately after the retired store (via TF).
 *      Clears TF, mfences, reads the doorbell dword to capture the
 *      work-submit-token, emits a timeline event, enqueues a snapshot
 *      request, re-mprotects to PROT_READ.
 *   4. A worker thread drains the snapshot queue.  Deferring the actual
 *      fopen/fprintf/fwrite out of signal-handler context is required
 *      because those functions are not async-signal-safe and would
 *      deadlock if CUDA held a malloc lock when the doorbell fired.
 *
 * USERD writes (on a different page) don't trap because only
 * MAP_KIND_USERMODE_BAR0 mappings are mprotected.  On Hopper the submit
 * sequence is "USERD GPPut advance; VF doorbell write" (see mc_submit.c)
 * — we trap exactly once per submit, at the doorbell, which is what
 * the user asked for.
 */

static _Atomic unsigned g_dbell_seq = 0;
static _Thread_local void
    *g_dbell_pending; /* BAR0 base; set by SEGV, used by TRAP */

/* Single-producer-single-consumer queue of snapshot requests posted from
 * the SIGTRAP handler to the worker thread.  Signal handler writes one
 * byte per request (the seq encoded as a string-line into a tiny pipe)
 * because pipes are async-signal-safe and give us a wait mechanism on
 * the consumer side. */
static int         g_dbell_pipe_r = -1;
static int         g_dbell_pipe_w = -1;
static pthread_t   g_dbell_worker;
static _Atomic int g_dbell_worker_run = 0;

static void *dbell_worker_fn(void *arg)
{
  (void)arg;
  /* Drain the pipe; each read returns one or more seq lines terminated
   * by '\n'.  Call the existing snapshot_all (which uses malloc/fopen
   * — safe here, we're a normal thread). */
  char   buf[512];
  char   partial[64];
  size_t partial_len = 0;
  while (__atomic_load_n(&g_dbell_worker_run, __ATOMIC_ACQUIRE))
  {
    ssize_t n = read(g_dbell_pipe_r, buf, sizeof(buf));
    if (n <= 0)
      break;
    for (ssize_t i = 0; i < n; i++)
    {
      char c = buf[i];
      if (c == '\n')
      {
        partial[partial_len] = '\0';
        unsigned seq         = (unsigned)strtoul(partial, NULL, 10);
        snapshot_all("dbell", seq);
        partial_len = 0;
      }
      else if (partial_len + 1 < sizeof(partial))
      {
        partial[partial_len++] = c;
      }
    }
  }
  return NULL;
}

static void sig_segv(int sig, siginfo_t *si, void *ucv)
{
  (void)sig;
  struct map_info *m = dbell_find_mapping(si->si_addr);
  if (!m)
  {
    /* Not one of ours — re-raise with default handler so we don't mask
     * real crashes.  signal(SIGSEGV, SIG_DFL) + return lets the faulting
     * instruction re-execute and dump core as usual. */
    struct sigaction sa = { 0 };
    sa.sa_handler       = SIG_DFL;
    sigaction(SIGSEGV, &sa, NULL);
    return;
  }
  /* Step 2: un-arm the page, set TF, stash the mapping for the TRAP handler. */
  ucontext_t *uc = (ucontext_t *)ucv;
  mprotect(m->addr, m->length, PROT_READ | PROT_WRITE);
  g_dbell_pending = m->addr;
  uc->uc_mcontext.gregs[REG_EFL] |= 0x100; /* TF — single-step */
}

static void sig_trap(int sig, siginfo_t *si, void *ucv)
{
  (void)sig;
  (void)si;
  ucontext_t *uc = (ucontext_t *)ucv;
  /* Clear TF. */
  uc->uc_mcontext.gregs[REG_EFL] &= ~0x100;
  void *base      = g_dbell_pending;
  g_dbell_pending = NULL;
  if (!base)
    return;
  /* Read the doorbell dword for reference.  NOTE: BAR0 is mapped
   * write-combine on Linux; CPU reads return undefined (typically
   * zero) values.  The real token can be recovered offline by
   * correlating the doorbell timestamp with
   * NV2080_CTRL_CMD_GPU_GET_WORK_SUBMIT_TOKEN ioctl replies seen in ftrace
   * (each channel has exactly one token, so with N channels there are at most N
   * possible values).  Future work: decode the x86 store instruction at the
   * faulting RIP to extract the token directly from ucontext registers. */
  _mm_mfence();
  volatile uint32_t *dbell =
      (volatile uint32_t *)((char *)base + HOPPER_VF_DOORBELL_OFF);
  uint32_t token = *dbell;

  unsigned seq = atomic_fetch_add(&g_dbell_seq, 1);

  /* Emit timeline event.  Signal-safe: snprintf + write. */
  char data[160];
  int  n = snprintf(data, sizeof(data),
                    "\"seq\":%u,\"bar0\":\"0x%" PRIxPTR "\","
                     "\"token\":\"0x%08x\"",
                    seq, (uintptr_t)base, token);
  (void)n;
  emit_event("doorbell", data);

  /* Snapshot.  Sync (in-handler) beats the GPU's PBDMA fetch by running
   * BEFORE the re-arm mprotect and BEFORE any malloc/fopen can
   * preempt.  Async (worker-thread via pipe) misses the pushbuffer
   * contents because the GPU has already consumed + CUDA may have
   * refilled.  Env PBCAP_DBELL_SYNC=0 to force the legacy worker path
   * for A/B testing. */
  if ((seq % g_dbell_sample) == 0)
  {
    if (g_dbell_sync)
    {
      snapshot_all_sigsafe(seq);
    }
    else if (g_dbell_pipe_w >= 0)
    {
      char line[32];
      int  ln = snprintf(line, sizeof(line), "%u\n", seq);
      if (ln > 0)
        safe_write(g_dbell_pipe_w, line, (size_t)ln);
    }
  }

  /* Re-arm. */
  struct map_info *m = dbell_find_mapping(base);
  if (m)
    mprotect(m->addr, m->length, PROT_READ);
}

/* Called from map_record when a HOPPER_USERMODE_A mapping is observed. */
static void dbell_arm_mapping(void *addr, size_t length)
{
  if (mprotect(addr, length, PROT_READ) != 0)
  {
    if (g_verbose)
      fprintf(stderr, "pbcap: dbell arm mprotect(%p,%zu,R) failed: %s\n", addr,
              length, strerror(errno));
    return;
  }
  char data[128];
  int  n =
      snprintf(data, sizeof(data), "\"bar0\":\"0x%" PRIxPTR "\",\"length\":%zu",
               (uintptr_t)addr, length);
  (void)n;
  emit_event("doorbell.arm", data);
  if (g_verbose)
    fprintf(stderr, "pbcap: dbell armed on %p (len=%zu)\n", addr, length);
}

/*
 * Scan /proc/self/maps for any 64-KiB /dev/nvidia0 mappings we haven't
 * already armed.  Needed because libcuda appears to call the mmap
 * syscall directly (bypassing our glibc mmap hook), so map_record never
 * fires for these HOPPER_USERMODE_A pages.
 *
 * Call this periodically — at least once before any cudaMemcpy so the
 * doorbell watchpoint is armed by the time CUDA submits work.  Idempotent:
 * re-arming an already-armed page is a no-op (mprotect R -> R).
 */
static void dbell_scan_and_arm(void)
{
  if (!g_dbell_arm)
    return;
  FILE *f = fopen("/proc/self/maps", "r");
  if (!f)
    return;
  char line[512];
  while (fgets(line, sizeof(line), f))
  {
    uintptr_t     start, end;
    char          perms[8], offset_str[32], dev_str[16];
    unsigned long inode;
    char          pathname[256] = "";
    int           fields = sscanf(line, "%lx-%lx %7s %31s %15s %lu %255s",
                                  (unsigned long *)&start, (unsigned long *)&end, perms,
                                  offset_str, dev_str, &inode, pathname);
    if (fields < 6)
      continue;
    if (!strstr(pathname, "/dev/nvidia0"))
      continue;
    size_t length = end - start;
    if (length != HOPPER_USERMODE_BAR0_LEN)
      continue;
    if (!(perms[1] == 'w'))
      continue;

    /* Already tracked? */
    int already = 0;
    pthread_mutex_lock(&g_maps_mu);
    for (int i = 0; i < MAX_MAPPINGS; i++)
    {
      if (g_maps[i].addr == (void *)start
          && g_maps[i].kind == MAP_KIND_USERMODE_BAR0)
      {
        already = 1;
        break;
      }
    }
    if (!already)
    {
      /* Synthesize a map_info entry so the signal handler can find it. */
      for (int i = 0; i < MAX_MAPPINGS; i++)
      {
        if (g_maps[i].addr == NULL)
        {
          g_maps[i].length = length;
          g_maps[i].fd     = -1;
          g_maps[i].offset = 0;
          g_maps[i].prot   = PROT_READ | PROT_WRITE;
          g_maps[i].kind   = MAP_KIND_USERMODE_BAR0;
          snprintf(g_maps[i].fdname, sizeof(g_maps[i].fdname), "/dev/nvidia0");
          __atomic_store_n(&g_maps[i].addr, (void *)start, __ATOMIC_RELEASE);
          break;
        }
      }
    }
    pthread_mutex_unlock(&g_maps_mu);

    if (!already)
      dbell_arm_mapping((void *)start, length);
  }
  fclose(f);
}

static void dbell_install_handlers(void)
{
  /* Open a nonblocking-write, blocking-read pipe for the handler→worker
   * queue.  O_DIRECT on pipes = packet mode (Linux-only); we don't need
   * that — ordinary stream is fine since each line is short. */
  int fds[2];
  if (pipe(fds) != 0)
  {
    if (g_verbose)
      fprintf(stderr, "pbcap: pipe() failed: %s\n", strerror(errno));
    return;
  }
  g_dbell_pipe_r = fds[0];
  g_dbell_pipe_w = fds[1];
  /* Writer side: non-blocking so the signal handler never stalls if the
   * pipe is momentarily full (would drop snapshot events but not crash). */
  int wfl = fcntl(g_dbell_pipe_w, F_GETFL, 0);
  if (wfl >= 0)
    fcntl(g_dbell_pipe_w, F_SETFL, wfl | O_NONBLOCK);

  __atomic_store_n(&g_dbell_worker_run, 1, __ATOMIC_RELEASE);
  if (pthread_create(&g_dbell_worker, NULL, dbell_worker_fn, NULL) != 0)
  {
    if (g_verbose)
      fprintf(stderr, "pbcap: worker thread create failed\n");
    close(g_dbell_pipe_r);
    close(g_dbell_pipe_w);
    g_dbell_pipe_r = g_dbell_pipe_w = -1;
    return;
  }

  /* Install handlers on an alt stack so we survive stack pressure in
   * CUDA threads.  glibc 2.34+ made SIGSTKSZ a runtime expression (not a
   * constant), so use a generously-sized static buffer that covers any
   * practical SIGSTKSZ value. */
  static char altstack[64 * 1024];
  stack_t     ss = { .ss_sp    = altstack,
                     .ss_size  = sizeof(altstack),
                     .ss_flags = 0 };
  sigaltstack(&ss, NULL);

  struct sigaction sa = { 0 };
  sa.sa_flags         = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
  sa.sa_sigaction     = sig_segv;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, NULL);

  struct sigaction sat = { 0 };
  sat.sa_flags         = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
  sat.sa_sigaction     = sig_trap;
  sigemptyset(&sat.sa_mask);
  sigaction(SIGTRAP, &sat, NULL);
}

/* ── hooked libc/syscalls ────────────────────────────────────────────────── */

int open(const char *pathname, int flags, ...)
{
  INIT();
  mode_t mode = 0;
  if (flags & O_CREAT)
  {
    va_list ap;
    va_start(ap, flags);
    mode = va_arg(ap, mode_t);
    va_end(ap);
  }
  int fd = real_open(pathname, flags, mode);
  if (fd >= 0 && fd_is_nvidia(pathname))
  {
    fd_record(fd, pathname);
    char data[256];
    int  n = snprintf(data, sizeof(data),
                      "\"fd\":%d,\"path\":\"%s\",\"flags\":\"0x%x\"", fd,
                      pathname, flags);
    (void)n;
    emit_event("open", data);
    if (g_verbose)
      fprintf(stderr, "pbcap: open(%s) -> %d\n", pathname, fd);
  }
  return fd;
}

int openat(int dirfd, const char *pathname, int flags, ...)
{
  INIT();
  mode_t mode = 0;
  if (flags & O_CREAT)
  {
    va_list ap;
    va_start(ap, flags);
    mode = va_arg(ap, mode_t);
    va_end(ap);
  }
  int fd = real_openat(dirfd, pathname, flags, mode);
  if (fd >= 0 && fd_is_nvidia(pathname))
  {
    fd_record(fd, pathname);
    char data[256];
    int  n = snprintf(data, sizeof(data),
                      "\"fd\":%d,\"path\":\"%s\",\"flags\":\"0x%x\"", fd,
                      pathname, flags);
    (void)n;
    emit_event("openat", data);
    if (g_verbose)
      fprintf(stderr, "pbcap: openat(%s) -> %d\n", pathname, fd);
  }
  return fd;
}

int close(int fd)
{
  INIT();
  const char *name = fd_lookup(fd);
  if (name)
  {
    char data[128];
    int n = snprintf(data, sizeof(data), "\"fd\":%d,\"path\":\"%s\"", fd, name);
    (void)n;
    emit_event("close", data);
  }
  fd_forget(fd);
  return real_close(fd);
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
  INIT();
  void *p = real_mmap(addr, length, prot, flags, fd, offset);
  if (p != MAP_FAILED && fd >= 0)
  {
    const char *name = fd_lookup(fd);
    if (name)
    {
      map_record(p, length, fd, offset, prot);
      char data[320];
      int  n =
          snprintf(data, sizeof(data),
                   "\"fd\":%d,\"path\":\"%s\",\"addr\":\"0x%" PRIxPTR "\","
                   "\"length\":%zu,\"prot\":\"0x%x\",\"flags\":\"0x%x\","
                   "\"offset\":\"0x%lx\"",
                   fd, name, (uintptr_t)p, length, prot, flags, (long)offset);
      (void)n;
      emit_event("mmap", data);
      if (g_verbose)
        fprintf(stderr,
                "pbcap: mmap(fd=%d=%s len=%zu prot=0x%x off=0x%lx) -> %p\n", fd,
                name, length, prot, (long)offset, p);
    }
  }
  return p;
}

int munmap(void *addr, size_t length)
{
  INIT();
  char data[128];
  int  n =
      snprintf(data, sizeof(data), "\"addr\":\"0x%" PRIxPTR "\",\"length\":%zu",
               (uintptr_t)addr, length);
  (void)n;
  emit_event("munmap", data);
  map_forget(addr, length);
  return real_munmap(addr, length);
}

/* ── hooked CUDA entry points ────────────────────────────────────────────── */

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind)
{
  INIT();
  if (!real_cudaMemcpy)
    RESOLVE(cudaMemcpy);

  /* libcuda calls mmap via the syscall directly, bypassing our glibc
   * hook.  Scan /proc/self/maps here to catch any HOPPER_USERMODE_A
   * mappings created since the last cudaMemcpy — arming the watchpoint
   * on new ones.  Cheap enough to run per memcpy. */
  dbell_scan_and_arm();

  unsigned idx = atomic_fetch_add(&g_snap_idx, 1);
  if (g_verbose)
    fprintf(stderr, "pbcap: cudaMemcpy[%u] dst=%p src=%p n=%zu kind=%d\n", idx,
            dst, src, count, (int)kind);

  emit_eventf("cudaMemcpy.enter",
              "\"idx\":%u,\"dst\":\"0x%" PRIxPTR "\","
              "\"src\":\"0x%" PRIxPTR "\",\"count\":%zu,\"kind\":%d",
              idx, (uintptr_t)dst, (uintptr_t)src, count, (int)kind);

  if (g_memcpy_snapshot)
    snapshot_all("pre", idx);
  cudaError_t r = real_cudaMemcpy(dst, src, count, kind);
  if (g_memcpy_snapshot)
    snapshot_all("post", idx);

  emit_eventf("cudaMemcpy.exit", "\"idx\":%u,\"ret\":%d", idx, (int)r);

  /* record a tiny text manifest with args so the offline decoder knows
   * dst/src VAs and direction without guessing */
  char path[512];
  snprintf(path, sizeof(path), "%s/snap-%05u-meta.txt", g_outdir, idx);
  FILE *f = fopen(path, "w");
  if (f)
  {
    fprintf(f, "dst=%p src=%p count=%zu kind=%d ret=%d\n", dst, src, count,
            (int)kind, (int)r);
    fclose(f);
  }
  return r;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
  INIT();
  if (!real_cudaHostAlloc)
    RESOLVE(cudaHostAlloc);
  cudaError_t r = real_cudaHostAlloc(pHost, size, flags);
  if (r == cudaSuccess)
  {
    char data[192];
    int  n =
        snprintf(data, sizeof(data),
                 "\"size\":%zu,\"flags\":\"0x%x\",\"ptr\":\"0x%" PRIxPTR "\"",
                 size, flags, (uintptr_t)*pHost);
    (void)n;
    emit_event("cudaHostAlloc", data);
    if (g_verbose)
      fprintf(stderr, "pbcap: cudaHostAlloc(%zu) -> %p\n", size, *pHost);
  }
  return r;
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
  INIT();
  if (!real_cudaMalloc)
    RESOLVE(cudaMalloc);
  cudaError_t r = real_cudaMalloc(devPtr, size);
  if (r == cudaSuccess)
  {
    char data[192];
    int  n =
        snprintf(data, sizeof(data), "\"size\":%zu,\"ptr\":\"0x%" PRIxPTR "\"",
                 size, (uintptr_t)*devPtr);
    (void)n;
    emit_event("cudaMalloc", data);
    if (g_verbose)
      fprintf(stderr, "pbcap: cudaMalloc(%zu) -> %p\n", size, *devPtr);
  }
  return r;
}

/* ── per-call entry/exit markers for individual libcuda functions ─────────
 *
 * Each of these hooks brackets a single CUDA Runtime call with two
 * NDJSON lines: `<fn>.enter` (args) and `<fn>.exit` (return code).
 * timeline_merge.py forwards them verbatim into merged.ndjson; the
 * trace_section.py helper then windows merged.ndjson on the bracket
 * pair to show every event (ftrace + strace + pbcap) that fell inside
 * the call.  This is how we answer "what does cudaHostRegister do in
 * addition to cudaInitDevice?" without a cross-trace differ.
 */

cudaError_t cudaInitDevice(int device, unsigned int deviceFlags,
                           unsigned int flags)
{
  INIT();
  if (!real_cudaInitDevice)
    RESOLVE(cudaInitDevice);

  emit_eventf("cudaInitDevice.enter",
              "\"device\":%d,\"deviceFlags\":\"0x%x\",\"flags\":\"0x%x\"",
              device, deviceFlags, flags);

  cudaError_t r = real_cudaInitDevice(device, deviceFlags, flags);

  emit_eventf("cudaInitDevice.exit", "\"ret\":%d", (int)r);
  return r;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
  INIT();
  if (!real_cudaHostRegister)
    RESOLVE(cudaHostRegister);

  emit_eventf("cudaHostRegister.enter",
              "\"ptr\":\"0x%" PRIxPTR "\",\"size\":%zu,\"flags\":\"0x%x\"",
              (uintptr_t)ptr, size, flags);

  cudaError_t r = real_cudaHostRegister(ptr, size, flags);

  emit_eventf("cudaHostRegister.exit", "\"ret\":%d", (int)r);
  return r;
}

cudaError_t cudaHostUnregister(void *ptr)
{
  INIT();
  if (!real_cudaHostUnregister)
    RESOLVE(cudaHostUnregister);

  emit_eventf("cudaHostUnregister.enter", "\"ptr\":\"0x%" PRIxPTR "\"",
              (uintptr_t)ptr);

  cudaError_t r = real_cudaHostUnregister(ptr);

  emit_eventf("cudaHostUnregister.exit", "\"ret\":%d", (int)r);
  return r;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream)
{
  INIT();
  if (!real_cudaLaunchKernel)
    RESOLVE(cudaLaunchKernel);

  emit_eventf("cudaLaunchKernel.enter",
              "\"func ptr\":\"0x%" PRIxPTR
              "\",\"gridDim\":{\"x\":%u,\"y\":%u,\"z\":%u},\"blockDim\":"
              "{\"x\":%u,\"y\":%u,\"z\":%u},\"args\":\"0x%" PRIxPTR
              "\",\"sharedMem\":%zu",
              (uintptr_t)func, gridDim.x, gridDim.y, gridDim.z, blockDim.x,
              blockDim.y, blockDim.z, (uintptr_t)args, sharedMem);

  cudaError_t r =
      real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

  emit_eventf("cudaLaunchKernel.exit", "\"ret\":%d", (int)r);
  return r;
}
