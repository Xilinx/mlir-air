//===- ffn_kernels.cc - FFN SwiGLU decode kernels --------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Combined kernels for FFN SwiGLU decode:
//   1. matvec_vectorized_bf16_bf16 — matrix-vector multiply (from mv.cc)
//   2. zero_vectorized_bf16 — zero fill output
//   3. swiglu_bf16 — SiLU(gate) * up element-wise
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

// ============================================================
// Zero fill
// ============================================================
template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 256 / (sizeof(T) * 8);
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

// ============================================================
// Matrix-vector multiply (from mlir-aie aie_kernels/aie2/mv.cc)
// A[M,K] in 32-bit-word-transposed layout × b[K] → c[M]
// ============================================================
#include "aie_kernels/aie_kernel_utils.h"

#ifndef DIM_M
#define DIM_M 128
#endif
#ifndef DIM_K
#define DIM_K 128
#endif

template <typename T_in, typename T_out, typename T_acc, unsigned m, unsigned k,
          unsigned r, unsigned s>
void matvec_vectorized(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(m % r == 0 && k % 2 == 0);
  static_assert(s == 8);
  static_assert(k % s == 0);

  T_in *__restrict a_ptr = a;
  T_in *__restrict b_ptr = b;

  for (int col = 0; col < k; col += 8) {
    aie::vector<T_in, 8> b_vec = aie::load_v<8>(b_ptr);
    T_out *__restrict c_ptr = c;
    AIE_LOOP_MIN_ITERATION_COUNT(m / r)
    for (int row = 0; row < m; row += r) {
      aie::accum<T_acc, r> c_acc_in;
      c_acc_in.from_vector(aie::load_v<r>(c_ptr));

      const aie::vector<T_in, 2 * r> a_vec_0 = aie::load_v<2 * r>(a_ptr);
      const aie::vector<T_in, 2 * r> a_vec_1 =
          aie::load_v<2 * r>(a_ptr + 2 * m);
      const aie::vector<T_in, 2 * r> a_vec_2 =
          aie::load_v<2 * r>(a_ptr + 4 * m);
      const aie::vector<T_in, 2 * r> a_vec_3 =
          aie::load_v<2 * r>(a_ptr + 6 * m);

      const aie::vector<T_in, r> a_vec_0_0 = aie::filter_even(a_vec_0);
      const aie::vector<T_in, r> a_vec_0_1 = aie::filter_odd(a_vec_0);
      const aie::vector<T_in, r> a_vec_1_0 = aie::filter_even(a_vec_1);
      const aie::vector<T_in, r> a_vec_1_1 = aie::filter_odd(a_vec_1);
      const aie::vector<T_in, r> a_vec_2_0 = aie::filter_even(a_vec_2);
      const aie::vector<T_in, r> a_vec_2_1 = aie::filter_odd(a_vec_2);
      const aie::vector<T_in, r> a_vec_3_0 = aie::filter_even(a_vec_3);
      const aie::vector<T_in, r> a_vec_3_1 = aie::filter_odd(a_vec_3);

      auto c_acc_out = aie::accumulate<r>(
          c_acc_in, b_vec, 0, a_vec_0_0, a_vec_0_1, a_vec_1_0, a_vec_1_1,
          a_vec_2_0, a_vec_2_1, a_vec_3_0, a_vec_3_1);

      aie::store_v(c_ptr, c_acc_out.template to_vector<T_out>());
      a_ptr += 2 * r;
      c_ptr += r;
    }
    a_ptr += 6 * m;
    b_ptr += s;
  }
}

// ============================================================
// SwiGLU: output[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x * sigmoid(x) = x * 0.5 * (tanh(x/2) + 1)
// Uses tanh-based sigmoid identity (avoids exp/div in bf16).
// Requires n % 8 == 0.
// ============================================================

// ============================================================
// Extern C functions
// ============================================================
extern "C" {

void matvec_vectorized_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in,
                                 bfloat16 *c_out) {
  matvec_vectorized<bfloat16, bfloat16, accfloat, DIM_M, DIM_K, 16, 8>(
      a_in, b_in, c_out);
}

void zero_vectorized_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, DIM_M, 1>(c_out);
}

void swiglu_bf16(bfloat16 *gate, bfloat16 *up, bfloat16 *out, int32_t n) {
  // SwiGLU(gate, up) = SiLU(gate) * up
  // SiLU(x) = x * sigmoid(x) = x * 0.5 * (1 + tanh(x/2))
  constexpr int VecLen = 8;
  aie::vector<bfloat16, VecLen> half_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.5f);
  aie::vector<bfloat16, VecLen> one_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)1.0f);

  for (int i = 0; i < n; i += VecLen) {
    aie::vector<bfloat16, VecLen> g = aie::load_v<VecLen>(gate + i);
    aie::vector<bfloat16, VecLen> u = aie::load_v<VecLen>(up + i);

    // sigmoid(g) = 0.5 * (1 + tanh(g/2))
    aie::vector<bfloat16, VecLen> g_half = aie::mul(g, half_vec);
    aie::accum<accfloat, VecLen> tanh_in;
    tanh_in.from_vector(g_half);
    aie::vector<bfloat16, VecLen> tanh_val =
        aie::tanh<bfloat16>(tanh_in.to_vector<float>());
    aie::vector<bfloat16, VecLen> one_plus_tanh = aie::add(one_vec, tanh_val);
    aie::vector<bfloat16, VecLen> sigmoid = aie::mul(half_vec, one_plus_tanh);
    // SiLU = gate * sigmoid
    aie::vector<bfloat16, VecLen> silu = aie::mul(g, sigmoid);
    // SwiGLU = SiLU(gate) * up
    aie::vector<bfloat16, VecLen> result = aie::mul(silu, u);

    aie::store_v(out + i, result);
  }
}

} // extern "C"
