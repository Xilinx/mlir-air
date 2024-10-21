//===- conv.cc --------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
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

template <typename T_in, typename T_out, unsigned kerCol, unsigned kerRow,
          unsigned chanTile, unsigned imgWidth, unsigned r, unsigned s,
          unsigned t>
void conv_vectorized(const T_in *__restrict pA, const T_in *__restrict pB,
                     T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in>;

  static_assert(r == 4);
  static_assert(s == 8);
  static_assert(t == 8);

  static_assert(imgWidth == r + kerRow - 1); // stride 1

  event0();

  aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC);
  MMUL C00(acc_C00);

  for (unsigned i = 0; i < kerCol; i++)
    chess_loop_range(3, ) {

      for (unsigned z = 0; z < chanTile; z += 4)
        chess_prepare_for_pipelining chess_loop_range(4, ) {

          const T_in *__restrict pA1 =
              pA + i * chanTile * imgWidth * s + z * imgWidth * s;
          const T_in *__restrict pB1 =
              pB + i * MMUL::size_B * chanTile * kerRow + z * MMUL::size_B;

          aie::vector<T_in, 64> A0_0 = aie::load_v<64>(pA1);
          aie::vector<T_in, 64> A0_1 = aie::load_v<64>(pA1 + 64);
          aie::vector<T_in, 64> A0_2 = aie::load_v<64>(pA1 + 128);

          // z = 0
          aie::vector<T_in, 32> A0 = extract_v32int8(A0_0, 0);
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(shift(A0_0, A0_1, 8), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(shift(A0_0, A0_1, 16), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          // z = 1
          pB1 = pB + i * MMUL::size_B * chanTile * kerRow + z * MMUL::size_B +
                MMUL::size_B;
          A0 = extract_v32int8(shift(A0_0, A0_1, 48), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(shift(A0_0, A0_1, 56), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(A0_1, 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          // z = 2
          pB1 = pB + i * MMUL::size_B * chanTile * kerRow + z * MMUL::size_B +
                2 * MMUL::size_B;
          A0 = extract_v32int8(shift(A0_1, A0_2, 32), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(shift(A0_1, A0_2, 40), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(shift(A0_1, A0_2, 48), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          // z = 3
          pB1 = pB + i * MMUL::size_B * chanTile * kerRow + z * MMUL::size_B +
                3 * MMUL::size_B;
          A0 = extract_v32int8(shift(A0_2, undef_v64int8(), 16), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(shift(A0_2, undef_v64int8(), 24), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);

          A0 = extract_v32int8(shift(A0_2, undef_v64int8(), 32), 0);
          B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * chanTile;
          C00.mac(A0, B0);
        }
    }

  aie::store_v(pC, C00.template to_vector<T_out>());

  event1();
}

template <unsigned kerCol, unsigned kerRow, unsigned inputChan,
          unsigned inputWidth>
void conv_vectorized_3x3x32x6_i8_i32(const int8 *__restrict pA,
                                     const int8 *__restrict pB,
                                     int *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(inputChan % s == 0);
  return conv_vectorized<int8, int, kerCol, kerRow, inputChan / s, inputWidth,
                         r, s, t>(pA, pB, pC);
}

extern "C" {

void conv(int8 *a_in, int8 *b_in, int *c_out) {
  conv_vectorized_3x3x32x6_i8_i32<3, 3, 32, 6>(a_in, b_in, c_out);
}
void linalg_fill_i32_view1x1x4x1x8xi32as2(int *c_out) {
  zero_vectorized<int, 4, 8, 32>(c_out);
}

} // extern "C"
