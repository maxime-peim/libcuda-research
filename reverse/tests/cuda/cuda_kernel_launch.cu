/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * cuda_kernel_launch.cu — the smallest possible CUDA kernel launch,
 * kept as a trace subject.  Capturing it under trace_cuda.sh isolates
 * what libcuda does for a launch with no memory traffic of its own,
 * which is the baseline the QMD work in docs/compute_kernel_launch.md
 * is read against.  Its stdout is not the point; the capture is.
 *
 *   sudo reverse/tools/trace_cuda.sh ./bin/cuda_kernel_launch
 */
#include <cuda_runtime.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

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

__global__ void helloFromGPU()
{ /* This kernel is launched on the device */
  printf("Hello World from GPU!\n");
}

int main(int argc, char *argv[])
{
  dim3 gridDim = { 1, 1, 1 }, blockDim = { 1, 1, 1 };
  CUDA_CHECK(cudaInitDevice(0, 0, 0));

  cudaLaunchKernel(helloFromGPU, gridDim, blockDim, NULL);

  return 0;
}
