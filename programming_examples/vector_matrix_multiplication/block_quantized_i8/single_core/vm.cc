//===- vm.cc ----------------------------------------------000---*- C++ -*-===//
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

template <typename T_in, typename T_out, typename T_accum, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t, unsigned GS>
void vecmat_vectorized(const T_in *__restrict pA, const T_out *__restrict pA_s,
                       const T_in *__restrict pB, const T_out *__restrict pB_s,
                       T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, acc32>;

  static_assert(MMUL::size_A == 16);
  static_assert(MMUL::size_B == 128);
  static_assert(MMUL::size_C == 8);

  constexpr int GS_factor = GS / s;

  event0();

  T_out *__restrict pC1 = pC;

  for (unsigned j = 0; j < colB; j += 2)
    chess_prepare_for_pipelining chess_loop_range(3, ) {
      const T_in *__restrict pA1 = pA;
      const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
      const T_in *__restrict pB2 = pB + ((j + 1)) * colA * MMUL::size_B;
      aie::vector<T_in, MMUL::size_A> A0;
      aie::vector<T_in, MMUL::size_B> B0;
      aie::vector<T_in, MMUL::size_B> B1;
      aie::vector<T_accum, MMUL::size_C> acc_C00 =
          aie::zeros<T_accum, MMUL::size_C>();
      aie::vector<T_accum, MMUL::size_C> acc_C01 =
          aie::zeros<T_accum, MMUL::size_C>();

      aie::accum<accfloat, MMUL::size_C> accfloat_C00 =
          vector_cast<accfloat, aie::vector<T_out, MMUL::size_C>>(
              aie::load_v<MMUL::size_C>(pC1));
      aie::accum<accfloat, MMUL::size_C> accfloat_C01 =
          vector_cast<accfloat, aie::vector<T_out, MMUL::size_C>>(
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C));

      aie::vector<T_out, MMUL::size_C> BS0;
      aie::vector<T_out, MMUL::size_C> BS1;
      const T_out *__restrict pB_s1 =
          pB_s + (j) * (colA / GS_factor) * MMUL::size_C;
      const T_out *__restrict pB_s2 =
          pB_s + (j + 1) * (colA / GS_factor) * MMUL::size_C;

      MMUL C00(acc_C00);
      MMUL C01(acc_C01);

      for (unsigned i = 0; i < colA / GS_factor; i++)
        chess_prepare_for_pipelining chess_loop_range(3, ) {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;
          C00.mac(A0, B0);
          C01.mac(A0, B1);
          A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;
          C00.mac(A0, B0);
          C01.mac(A0, B1);

          BS0 = aie::load_v<MMUL::size_C>(pB_s1);
          pB_s1 += MMUL::size_C;
          accfloat_C00 = mac(accfloat_C00,
                             to_float<float, T_accum, MMUL::size_C>(
                                 C00.template to_vector<T_accum>()),
                             mul(pA_s[i], BS0).template to_vector<float>());
          BS1 = aie::load_v<MMUL::size_C>(pB_s2);
          pB_s2 += MMUL::size_C;
          accfloat_C01 = mac(accfloat_C01,
                             to_float<float, T_accum, MMUL::size_C>(
                                 C01.template to_vector<T_accum>()),
                             mul(pA_s[i], BS1).template to_vector<float>());

          C00.zero = true;
          C01.zero = true;
        }

      aie::store_v(pC1, accfloat_C00.template to_vector<float>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, accfloat_C01.template to_vector<float>());
      pC1 += MMUL::size_C;
    }

  event1();
}

template <unsigned m, unsigned k, unsigned n>
void vecmat_vectorized_1x16x8_i8_f32_i32_32(const int8 *__restrict pA,
                                            const float *__restrict pA_s,
                                            const int8 *__restrict pB,
                                            const float *__restrict pB_s,
                                            float *__restrict pC) {
  constexpr int r = 1;
  constexpr int s = 16;
  constexpr int t = 8;
  constexpr int GS = 32;
  static_assert(m == 1);
  static_assert(k % s == 0 && k / s > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return vecmat_vectorized<int8, float, int, k / s, n / t, r, s, t, GS>(
      pA, pA_s, pB, pB_s, pC);
}

extern "C" {

#define combos(X) X(int8, i8, float, f32, int, i32, 32, 1, 16, 8)

#define vecmat_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,                                          \
                                 mlir_type_out, ctype_accum, mlir_type_accum,                                \
                                 group_size, r, s, t)                                                        \
  void                                                                                                       \
      vecmat_##mlir_type_in##_##mlir_type_out##_##mlir_type_accum##_##group_size(                            \
          ctype_in *a_in, ctype_out *a_s, ctype_in *b_in, ctype_out *b_s,                                    \
          ctype_out *c_out) {                                                                                \
    vecmat_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out##_##mlir_type_accum##_##group_size< \
        1, 96, 48>(a_in, a_s, b_in, b_s, c_out);                                                             \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, ctype_accum, mlir_type_accum,    \
                               group_size, r, s, t)                            \
  void linalg_fill_i32_view16x8xi32as2(ctype_out *c_out) {                     \
    zero_vectorized<ctype_out, 1, 48, 16>(c_out);                              \
  }

combos(vecmat_vectorized_c_func) combos(zero_vectorized_c_func)

} // extern "C"
