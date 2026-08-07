/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Cisco and/or its affiliates.
 * SPDX-License-Identifier: MIT
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

int main(int argc, char *argv[])
{
  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  return 0;
}
