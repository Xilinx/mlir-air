//===- zero.cc --------------------------------------------000---*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_CC
#define ZERO_CC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T, int M, int N>
void zero_scalar(T *__restrict c) {
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

template <typename T, int M, int N, int r>
void zero_vectorized(int offset, T *__restrict c) {
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  T *__restrict c_start = c + offset;
  T *__restrict c_end = c_start + M * N;
  for (; c_start + r < c_end; c_start += r) {
    aie::store_v(c_start, zeros);
  }
  // Do a scalar write for any remainder not divisible by vector instruction
  // size r
  for (; c_start < c_end; c_start++) {
    *c_start = 0;
  }
}

#endif
