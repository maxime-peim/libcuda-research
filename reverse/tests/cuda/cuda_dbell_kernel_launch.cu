/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * cuda_dbell_kernel_launch.cu — CUDA reference run for the same SASS
 * kernel mc's compute path will execute (mc_doorbell_kernel:
 * a single store of `token` to `*dst`, exit).  Capture this with
 * trace_cuda.sh to get an authoritative libcuda QMD/CB0 layout we
 * can use as a template in mc_compute_qmd.c.
 *
 * Same kernel signature → same SASS → same per-launch field set in
 * the captured QMD.  PROGRAM_ADDRESS will point to wherever libcuda
 * uploads the SASS; CB0 will contain the (dst, token) pair at the
 * Hopper ABI offsets (0x210 / 0x218).
 *
 * Run on the H100 test host:
 *   sudo PBCAP_DBELL=0 reverse/tools/trace_cuda.sh ./bin/cuda_dbell_kernel_launch
 *
 * The output will land in /tmp/trace-cuda_dbell_kernel_launch-DATE/ .
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                \
  do                                                                    \
  {                                                                     \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess)                                              \
    {                                                                   \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                                  \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

extern "C" __global__
void mc_doorbell_kernel(volatile unsigned int *dst, unsigned int token)
{
  *dst = token;
}

int main(void)
{
  dim3 gridDim  = { 1, 1, 1 };
  dim3 blockDim = { 1, 1, 1 };
  unsigned int *d_cell = nullptr;

  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  CUDA_CHECK(cudaMalloc((void **)&d_cell, sizeof(unsigned int)));

  unsigned int token = 0xdeadbeef;
  void *args[] = { &d_cell, &token };
  CUDA_CHECK(cudaLaunchKernel((const void *)mc_doorbell_kernel,
                              gridDim, blockDim, args, 0, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned int host_cell = 0;
  CUDA_CHECK(cudaMemcpy(&host_cell, d_cell, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  printf("Reference run: cell read back as 0x%08x (expected 0xdeadbeef)\n",
         host_cell);

  cudaFree(d_cell);
  return host_cell == token ? 0 : 1;
}
