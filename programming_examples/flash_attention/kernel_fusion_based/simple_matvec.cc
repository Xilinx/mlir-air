// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// simple_matvec_bf16: y[m] = sum_k W[m, k] * x[k] for bf16 inputs/output.
// Simple vectorized matvec for the 5-all V1a decode_block test —
// keeps the gemv_herd compute self-contained (no rms_qkv_rope kernel
// dependency) at the cost of being slower than the production
// matvec_vectorized_bf16_bf16 kernel. Only used at toy dims (N<=64,
// M<=384) where perf isn't load-bearing.

#include <aie_api/aie.hpp>
#include <stdint.h>

template <int K_VAL>
static inline void matvec_kernel_bf16(const bfloat16 *restrict W,
                                      const bfloat16 *restrict x,
                                      bfloat16 *restrict y, int32_t M) {
  constexpr int VL = 16;
  static_assert(K_VAL % VL == 0, "K must be a multiple of vector width");
  constexpr int K_VECS = K_VAL / VL;

  // Pre-load x into vector registers (small K, fits).
  ::aie::vector<bfloat16, VL> x_vec[K_VECS];
  for (int i = 0; i < K_VECS; i++)
    x_vec[i] = ::aie::load_v<VL>(x + i * VL);

  for (int m = 0; m < M; m++) {
    ::aie::accum<accfloat, VL> acc;
    acc = ::aie::zeros<accfloat, VL>();
    for (int i = 0; i < K_VECS; i++) {
      auto w = ::aie::load_v<VL>(W + m * K_VAL + i * VL);
      acc = ::aie::mac(acc, w, x_vec[i]);
    }
    // Reduce VL-wide accumulator to a scalar bf16.
    ::aie::vector<float, VL> sum_vec = acc.template to_vector<float>();
    float sum = 0.0f;
    for (int i = 0; i < VL; i++)
      sum += sum_vec[i];
    y[m] = (bfloat16)sum;
  }
}

extern "C" {
// W: [M, K] row-major bf16
// x: [K] bf16
// y: [M] bf16
//
// Three identically-bodied entrypoints with distinct symbols so the
// caller can declare an external_func per memref shape (Q rows: [256],
// K rows: [64], V rows: [64]) and have the
// llvm.emit_c_interface wrapper match each ABI.
static inline void run(bfloat16 *W, bfloat16 *x, bfloat16 *y, int32_t M,
                       int32_t K) {
  event0();
  if (K == 64)
    matvec_kernel_bf16<64>(W, x, y, M);
  event1();
}
void simple_matvec_bf16(bfloat16 *W, bfloat16 *x, bfloat16 *y, int32_t M,
                        int32_t K) {
  run(W, x, y, M, K);
}
void simple_matvec_bf16_k(bfloat16 *W, bfloat16 *x, bfloat16 *y, int32_t M,
                          int32_t K) {
  run(W, x, y, M, K);
}
void simple_matvec_bf16_v(bfloat16 *W, bfloat16 *x, bfloat16 *y, int32_t M,
                          int32_t K) {
  run(W, x, y, M, K);
}
}
