//===- conv.cc --------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void conv_vectorized(const T_in *__restrict pA, const T_in *__restrict pB,
                     T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in>;

  static_assert(r == 4);
  static_assert(s == 8);
  static_assert(t == 8);
  static_assert(MMUL::size_A == 32);
  static_assert(MMUL::size_B == 64);

  event0();

  aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC);
  MMUL C00(acc_C00);
  const T_in *__restrict pA1 = pA;
  const T_in *__restrict pB1 = pB;
  for (unsigned i = 0; i < colA; i += 1)
    chess_prepare_for_pipelining chess_loop_range(18, ) {
      aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
      aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
      C00.mac(A0, B0);
      pA1 += MMUL::size_A;
      pB1 += MMUL::size_B;
    }
  aie::store_v(pC, C00.template to_vector<T_out>());

  event1();
}

template <unsigned m, unsigned k, unsigned n>
void conv_vectorized_4x8x8_i8_i32(const int8 *__restrict pA,
                                  const int8 *__restrict pB,
                                  int *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m % r == 0);
  static_assert(k % s == 0);
  static_assert(n % t == 0);
  return conv_vectorized<int8, int, m / r, k / s, n / t, r, s, t>(pA, pB, pC);
}

extern "C" {

void linalg_conv_1d_nwc_wcf_view1x4x8xi8as2_view1x8x8xi8as2_view1x4x8xi32as2(
    int8 *a_in, int8 *b_in, int *c_out) {
  conv_vectorized_4x8x8_i8_i32<4, 8, 8>(a_in, b_in, c_out);
}
void linalg_fill_i32_view1x1x4x8xi32as2(int *c_out) {
  zero_vectorized<int, 4, 8, 32>(c_out);
}

} // extern "C"
