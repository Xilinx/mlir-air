//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in, typename T_out, int rowA, int colA, int colB>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      c[row * colB + col] += running_sum;
    }
  }
  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_4x4(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;
      T_out *__restrict pC3 = pC + ((z + 2)) * MMUL::size_C;
      T_out *__restrict pC4 = pC + ((z + 3)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + ((z + 2)) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + ((z + 3)) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
          const T_in *__restrict pB2 = pB + ((j + 1)) * colA * MMUL::size_B;
          const T_in *__restrict pB3 = pB + ((j + 2)) * colA * MMUL::size_B;
          const T_in *__restrict pB4 = pB + ((j + 3)) * colA * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A3 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB3);
          pB3 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B3 = aie::load_v<MMUL::size_B>(pB4);
          pB4 += MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C02 =
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C03 =
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C12 =
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C13 =
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C22 =
              aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C23 =
              aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C32 =
              aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C33 =
              aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C * rowA);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C02(acc_C02);
          MMUL C03(acc_C03);

          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C12(acc_C12);
          MMUL C13(acc_C13);

          MMUL C20(acc_C20);
          MMUL C21(acc_C21);
          MMUL C22(acc_C22);
          MMUL C23(acc_C23);

          MMUL C30(acc_C30);
          MMUL C31(acc_C31);
          MMUL C32(acc_C32);
          MMUL C33(acc_C33);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          C02.mac(A0, B2);
          C03.mac(A0, B3);
          C12.mac(A1, B2);
          C13.mac(A1, B3);

          C20.mac(A2, B0);
          C21.mac(A2, B1);
          C30.mac(A3, B0);
          C31.mac(A3, B1);

          C22.mac(A2, B2);
          C23.mac(A2, B3);
          C32.mac(A3, B2);
          C33.mac(A3, B3);

          for (unsigned i = 1; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              A2 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += rowA * MMUL::size_A;
              A3 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += rowA * MMUL::size_A;

              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;
              B2 = aie::load_v<MMUL::size_B>(pB3);
              pB3 += MMUL::size_B;
              B3 = aie::load_v<MMUL::size_B>(pB4);
              pB4 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);

              C02.mac(A0, B2);
              C03.mac(A0, B3);
              C12.mac(A1, B2);
              C13.mac(A1, B3);

              C20.mac(A2, B0);
              C21.mac(A2, B1);
              C30.mac(A3, B0);
              C31.mac(A3, B1);

              C22.mac(A2, B2);
              C23.mac(A2, B3);
              C32.mac(A3, B2);
              C33.mac(A3, B3);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C02.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C03.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;

          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C12.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C13.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;

          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C22.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C23.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;

          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C32.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C33.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
        }
    }

  event1();
}

// bf16 MatMul kernel definion with bf16 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {

  // After extensive experimentation, the 4x8x4 aie::mmul size was found to be
  // optimal for AIE2, in combination with the 4x4 mmul expanded kernel
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;

  // Since the kernel has been expanded 4 times for both A ('m' dimension) and B
  // ('n' dimension), the following assertions veirify this even division for
  // the single AIE MatMul dimensionality Notice that 'k' dimension is not
  // spatially expanded.
  static_assert(m % (4 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (4 * t) == 0); // 'n' dimension

  return matmul_vectorized_4x4<bfloat16, bfloat16, (m / r), (k / s), (n / t), r,
                               s, t>(pA, pB, pC);
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 128
#define DIM_M_DIV_4 32
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#ifndef DIM_N
#define DIM_N 64
#define DIM_N_DIV_4 16
#endif

#ifndef combos
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)
#endif

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void op_has_no_registered_library_name(ctype_in *a_in, ctype_in *b_in,       \
                                         ctype_out *c_out) {                   \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);                               \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(a_in, b_in,        \
                                                            c_out);            \
  }

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define MAKE_LINALG_FILL_NAME(mlir_in, mlir_out, N_div_4, M_div_4)             \
  CAT(CAT(CAT(CAT(CAT(CAT(CAT(CAT(linalg_fill_, mlir_out), _view1x1x),         \
                          N_div_4),                                            \
                      x),                                                      \
                  M_div_4),                                                    \
              x4x4x),                                                          \
          mlir_out),                                                           \
      as2)

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void MAKE_LINALG_FILL_NAME(mlir_type_in, mlir_type_out, DIM_N_DIV_4,         \
                             DIM_M_DIV_4)(ctype_out * c_out) {                 \
    zero_vectorized<ctype_out, DIM_M, DIM_N, 32>(c_out);                       \
  }

combos(matmul_vectorized_c_func) combos(matmul_scalar_c_func)
    combos(zero_vectorized_c_func)

} // extern "C"
