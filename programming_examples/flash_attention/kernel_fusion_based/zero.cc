//===- zero.cc --------------------------------------------------*- C++ -*-===//
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

template <typename T, int M, int N, int r>
void neg_inf_vectorized(T *__restrict c) {
  uint16_t neg_infinity = (uint16_t)0xff80;
  T *T_neg_infinity = (T *)&neg_infinity;
  const aie::vector<T, r> neg_infs = aie::broadcast<T, r>(*T_neg_infinity);
  const T *__restrict c_end = c + M * N;
  for (; c + r < c_end; c += r) {
    aie::store_v(c, neg_infs);
  }
  // Do a scalar write for any remainder not divisible by vector instruction
  // size r
  for (; c < c_end; c++) {
    *c = *T_neg_infinity;
  }
}

#endif
