//===- ffn_kernels.cc - FFN SwiGLU prefill kernels --------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Combined kernels for FFN SwiGLU prefill:
//   1. matmul_bf16 — matrix multiply C[M,N] += A[M,K] @ B[K,N] (row-major)
//   2. zero_vectorized_bf16 — zero fill output buffer
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
template <typename T, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 256 / (sizeof(T) * 8);
  static_assert(N % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + N;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

// ============================================================
// Simple row-major matrix multiply
// C[M,N] += A[M,K] @ B[K,N]
// Implemented as a straightforward scalar triple loop.
// ============================================================

#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_K
#define DIM_K 128
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

template <typename T, unsigned M, unsigned K, unsigned N>
void matmul_row_major(const T *__restrict A, const T *__restrict B,
                      T *__restrict C) {
  // Simple scalar matmul: C[M,N] += A[M,K] @ B[K,N]
  // A, B, C are row-major.
  for (unsigned i = 0; i < M; i++) {
    for (unsigned j = 0; j < N; j++) {
      float sum = 0.0f;
      for (unsigned kk = 0; kk < K; kk++) {
        sum += (float)A[i * K + kk] * (float)B[kk * N + j];
      }
      C[i * N + j] += (T)sum;
    }
  }
}

// ============================================================
// SwiGLU: output[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x * sigmoid(x) = x * 0.5 * (tanh(x/2) + 1)
// ============================================================

// ============================================================
// Extern C functions
// ============================================================
extern "C" {

void matmul_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  matmul_row_major<bfloat16, DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);
}

void zero_vectorized_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, DIM_M * DIM_N>(c_out);
}

void swiglu_bf16(bfloat16 *gate, bfloat16 *up, bfloat16 *out, int32_t n) {
  constexpr int VecLen = 8;
  aie::vector<bfloat16, VecLen> half_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.5f);
  aie::vector<bfloat16, VecLen> one_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)1.0f);

  for (int i = 0; i < n; i += VecLen) {
    aie::vector<bfloat16, VecLen> g = aie::load_v<VecLen>(gate + i);
    aie::vector<bfloat16, VecLen> u = aie::load_v<VecLen>(up + i);

    aie::vector<bfloat16, VecLen> g_half = aie::mul(g, half_vec);
    aie::accum<accfloat, VecLen> tanh_in;
    tanh_in.from_vector(g_half);
    aie::vector<bfloat16, VecLen> tanh_val =
        aie::tanh<bfloat16>(tanh_in.to_vector<float>());
    aie::vector<bfloat16, VecLen> one_plus_tanh = aie::add(one_vec, tanh_val);
    aie::vector<bfloat16, VecLen> sigmoid = aie::mul(half_vec, one_plus_tanh);
    aie::vector<bfloat16, VecLen> silu = aie::mul(g, sigmoid);
    aie::vector<bfloat16, VecLen> result = aie::mul(silu, u);

    aie::store_v(out + i, result);
  }
}

} // extern "C"
