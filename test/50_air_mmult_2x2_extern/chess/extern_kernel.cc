//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 1
#define NOCPP
#define __AIE_ARCH__ 10

#include "aie_api/aie.hpp"

#define L1_M 32
#define L1_N 32
#define L1_K 32

extern "C" {
void mmult_acc_intrinsic(int32_t *in0, int32_t *in1, int32_t *out0, unsigned M,
                         unsigned K, unsigned N) {
  for (unsigned r = 0; r < M; ++r) {
    for (unsigned c = 0; c < N; ++c) {
      int32_t acc = 0;
      for (unsigned k = 0; k < K; ++k) {
        unsigned i0 = r * K + k;
        unsigned i1 = k * N + c;
        acc += ((int32_t)in0[i0]) * ((int32_t)in1[i1]);
      }
      unsigned o0 = r * N + c;
      out0[o0] += acc;
    }
  }
}

void extern_kernel(int32_t *restrict A, int32_t *restrict B,
                   int32_t *restrict C) {
  mmult_acc_intrinsic(A, B, C, L1_M, L1_K, L1_M);
}
}