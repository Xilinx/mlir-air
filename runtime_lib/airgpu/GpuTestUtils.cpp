//===- GpuTestUtils.cpp - Custom GPU test utilities for mlir-air ---------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Custom GPU runtime utilities for testing matmul and other operations.
// These functions supplement the standard libmlir_rocm_runtime.so from LLVM.
//
// Usage:
//   mlir-runner --shared-libs=libmlir_rocm_runtime.so,libairgpu.so output.mlir
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "hip/hip_runtime.h"

#define HIP_REPORT_IF_ERROR(expr)                                              \
  [](hipError_t result) {                                                      \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

/// Initialize matrices for matmul testing.
/// Matrix A is filled with sequential values (1, 2, 3, ...).
/// Matrix B is initialized as an identity matrix.
extern "C" void mgpuInit(float *A, float *B, int64_t N, int64_t M) {
  int64_t i = 1;
  for (int64_t y = 0; y < N; y++) {
    for (int64_t x = 0; x < M; x++) {
      A[M * y + x] = static_cast<float>(i);
      B[M * y + x] = (x == y) ? 1.0f : 0.0f;
      i++;
    }
  }
}

/// Verify matmul output matches expected results.
/// Compares device output against host reference with epsilon tolerance.
extern "C" void mgpuCheckOutput(float *device, float *hostA, float *hostB,
                                int64_t N, int64_t M) {
  float epsilon = 1e-3f; // Relaxed tolerance for GPU floating point
  bool mismatch = false;

  for (int64_t i = 0; i < N && !mismatch; i++) {
    for (int64_t j = 0; j < M && !mismatch; j++) {
      float expected = hostA[i * M + j];
      float actual = device[i * M + j];
      float diff = std::fabs(expected - actual);

      // Use relative error for large values, absolute for small
      float relError =
          (std::fabs(expected) > 1.0f) ? diff / std::fabs(expected) : diff;

      if (relError > epsilon) {
        std::cout << "Mismatch at (" << i << ", " << j << "): " << "expected "
                  << expected << " != actual " << actual << " (diff: " << diff
                  << ")" << std::endl;
        mismatch = true;
      }
    }
  }

  if (!mismatch) {
    std::cout << "Output Matched!\n";
  }
}

/// Measure elapsed time between two GPU events.
/// Wraps hipEventElapsedTime and prints the result.
extern "C" int32_t mgpuEventElapsedTime(float *ms, hipEvent_t start,
                                        hipEvent_t stop) {
  hipError_t result = hipEventElapsedTime(ms, start, stop);
  if (result == hipSuccess) {
    printf("Elapsed time: %.3f ms\n", *ms);
  } else {
    HIP_REPORT_IF_ERROR(result);
  }
  return static_cast<int32_t>(result);
}
