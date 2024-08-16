//===- kernel.cpp -----------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T, int M, int N> void zero_scalar(T *__restrict c) {
  for (int i = 0; i < M * N; i++) {
    c[i] = 0.0f;
  }
}

template <typename T, int M, int N, int r>
void zero_vectorized(T *__restrict c) {
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c + r < c_end; c += r) {
    aie::store_v(c, zeros);
  }
  // Do a scalar write for any remainder not divisible by vector instruction
  // size r
  for (; c < c_end; c++) {
    *c = 0;
  }
}

template <typename T_in, typename T_out, int M, int K, int N>
void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      T_out running_sum = 0.0f;
      for (int i = 0; i < K; i++) {
        running_sum += a[row * K + i] * b[i * N + col];
      }
      c[row * N + col] += running_sum;
    }
  }
  event1();
}

extern "C" {

void linalg_fill_bf16_view64x64xbf16as2(bfloat16 *d) {
  zero_vectorized<bfloat16, 64, 64, 32>(d);
}

void linalg_fill_f32_view64x64xf32as2(float *d) {
  zero_scalar<float, 64, 64>(d);
}

void linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xbf16as2(
    bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matmul_scalar<bfloat16, bfloat16, 64, 64, 64>(a, b, c);
}

void linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(
    bfloat16 *a, bfloat16 *b, float *c) {
  matmul_scalar<bfloat16, float, 64, 64, 64>(a, b, c);
}
}
