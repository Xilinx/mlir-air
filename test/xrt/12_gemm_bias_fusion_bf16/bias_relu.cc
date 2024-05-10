//===- bias_relu.cc ---------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef BIAS_RELU_CC
#define BIAS_RELU_CC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void bias_relu_scalar_2x2(const T_in *__restrict pA, const T_in *__restrict pB,
                          T_out *__restrict pC) {
  for (unsigned i = 0; i < rowA; i++) {
    for (unsigned j = 0; j < colB; j++) {
      for (unsigned m = 0; m < r; m++) {
        for (unsigned n = 0; n < t; n++) {
          T_out temp =
              pA[n + m * t + j * r * t + i * r * t * colB] + pB[n + i * t];
          if (temp < (T_out)0)
            temp = (T_out)0;
          pC[n + m * t + j * r * t + i * r * t * colB] = temp;
        }
      }
    }
  }
}

template <unsigned m, unsigned k, unsigned n>
void bias_relu_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                               const bfloat16 *__restrict pB,
                               bfloat16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return bias_relu_scalar_2x2<bfloat16, bfloat16, m / r, k / s, n / t, r, s, t>(
      pA, pB, pC);
}

// TODO:
// template <typename T, int M, int N, int r>
// void bias_relu_vectorized(T *__restrict c) {
// }

#endif
