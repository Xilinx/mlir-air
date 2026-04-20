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
  // Use bf16 lowest (0xff7f ≈ -3.39e38) instead of -inf (0xff80) to avoid NaN
  // on AIE2P: max(NaN, -inf) returns NaN, but max(NaN, lowest) also returns NaN
  // — the real fix is that lowest - lowest = 0, not NaN, avoiding the issue.
  uint16_t lowest_u16 = (uint16_t)0xff7f;
  T *T_lowest = (T *)&lowest_u16;
  const aie::vector<T, r> lowest_vec = aie::broadcast<T, r>(*T_lowest);
  const T *__restrict c_end = c + M * N;
  for (; c + r < c_end; c += r) {
    aie::store_v(c, lowest_vec);
  }
  for (; c < c_end; c++) {
    *c = *T_lowest;
  }
}

#endif
