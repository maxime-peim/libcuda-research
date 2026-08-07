/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * cuda_mmap_snooper.cu — minimal CUDA program that establishes full
 * driver state (cuInit, cudaMalloc, cudaMemcpy), then blocks reading
 * stdin so an external observer can snapshot /proc/PID/maps and
 * /proc/PID/pagemap while libcuda's mappings are live.
 *
 * Usage (from shell, NOT interactive): pipe a newline in to release:
 *   { sleep 2; echo; } | sudo ./bin/cuda_mmap_snooper > /dev/null &
 *   PID=$!
 *   sudo cp /proc/$PID/maps   /tmp/snooper.maps
 *   sudo python3 ../tools/find_bar1_pfn.py $PID   # never `cp` pagemap:
 *                                                 # it is sparse over the full
 *                                                 # 48-bit VA space
 *   wait $PID
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e));       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main(void)
{
    const size_t N = 4 * 1024 * 1024;  /* 4 MiB — matches the size used in historical cuda_reference 4M traces */

    uint32_t *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, N));

    uint32_t *h_buf = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_buf, N, cudaHostAllocDefault));

    /* One round-trip so every USERD/GPFIFO/pushbuffer is exercised. */
    for (size_t i = 0; i < N / 4; ++i) h_buf[i] = 0xdeadbeefu;
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    fprintf(stderr, "snooper: ready pid=%d — send newline to stdin to exit\n",
            (int)getpid());

    /* Block until stdin produces a byte.  Observer snapshots proc files
     * here and then sends a newline. */
    char buf;
    ssize_t n = read(0, &buf, 1);
    (void)n;

    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaFree(d_buf));
    return 0;
}
