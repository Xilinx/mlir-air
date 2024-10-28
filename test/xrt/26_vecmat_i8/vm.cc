//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
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

template <typename T_in, typename T_out, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
void vecmat_vectorized(const T_in *__restrict pA, const T_in *__restrict pB,
                       T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accfloat>;

  event0();

  T_out *__restrict pC1 = pC;

  for (unsigned j = 0; j < colB; j += 2)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      const T_in *__restrict pA1 = pA;
      const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
      const T_in *__restrict pB2 = pB + ((j + 1)) * colA * MMUL::size_B;
      aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
      pA1 += MMUL::size_A;
      aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B;
      aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
      pB2 += MMUL::size_B;

      aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
      aie::vector<T_out, MMUL::size_C> acc_C01 =
          aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);

      MMUL C00(acc_C00);
      MMUL C01(acc_C01);

      C00.mac(A0, B0);
      C01.mac(A0, B1);

      for (unsigned i = 1; i < colA; ++i)
        chess_prepare_for_pipelining chess_loop_range(7, ) {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;
          C00.mac(A0, B0);
          C01.mac(A0, B1);
        }

      aie::store_v(pC1, C00.template to_vector<T_out>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<T_out>());
      pC1 += MMUL::size_C;
    }

  event1();
}

template <unsigned m, unsigned k, unsigned n>
void vecmat_vectorized_1x16x8_i8_i32(const int8 *__restrict pA,
                                     const int8 *__restrict pB,
                                     int *__restrict pC) {
  constexpr int r = 1;
  constexpr int s = 16;
  constexpr int t = 8;
  static_assert(m == 1);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return vecmat_vectorized<int8, int, k / s, n / t, r, s, t>(pA, pB, pC);
}

extern "C" {

#define combos(X) X(int8, i8, int, i32, 1, 16, 8)

#define vecmat_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void vecmat_##mlir_type_in##_##mlir_type_out(ctype_in *a_in, ctype_in *b_in, \
                                               ctype_out *c_out) {             \
    vecmat_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        1, 128, 128>(a_in, b_in, c_out);                                       \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void linalg_fill_i32_view16x8xi32as2(ctype_out *c_out) {                     \
    zero_vectorized<ctype_out, 64, 64, 32>(c_out);                             \
  }

combos(vecmat_vectorized_c_func) combos(zero_vectorized_c_func)

} // extern "C"
