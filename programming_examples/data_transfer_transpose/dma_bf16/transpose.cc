// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Simple matrix transpose kernel for bf16 (uint16_t).
// Transposes an M x N row-major matrix to N x M row-major.
// Uses scalar element access (not VSHUFFLE-optimized).

#include <cstdint>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

using DTYPE = uint16_t;

extern "C" {

void transpose_bf16(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  for (unsigned i = 0; i < DIM_M; i++) {
    for (unsigned j = 0; j < DIM_N; j++) {
      out_ptr[j * DIM_M + i] = in_ptr[i * DIM_N + j];
    }
  }
}

} // extern "C"
