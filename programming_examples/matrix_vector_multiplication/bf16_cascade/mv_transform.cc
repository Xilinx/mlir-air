// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Vectorized bf16 matvec kernel with f32 accumulation for AIE2P.
// Used by the transform-dialect-based cascade matvec flow.
//
// Replaces the scalar linalg.matvec that Peano would otherwise compile
// with bf16 accumulation (losing precision for large K).
//
// Interface matches linalg.matvec semantics:
//   C[m] += A[m, k] * B[k]   (accumulate into existing C)

#include <aie_api/aie.hpp>

static constexpr uint32_t VEC_SIZE = 32;

extern "C" {

// Matvec with f32 internal accumulation.
// A is [M x K] row-major, B is [K], C is [M] (bf16, accumulated).
// M and K are passed as i32 arguments (matching linalg calling convention).
void matvec_bf16_f32acc(int32_t M, int32_t K,
                        const bfloat16 *__restrict A,
                        const bfloat16 *__restrict B,
                        bfloat16 *__restrict C) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  uint32_t m = static_cast<uint32_t>(M);
  uint32_t k = static_cast<uint32_t>(K);

  for (uint32_t row = 0; row < m; ++row) {
    aie::accum<accfloat, VEC_SIZE> acc = aie::zeros<accfloat, VEC_SIZE>();
    const bfloat16 *a_row = A + row * k;
    for (uint32_t j = 0; j < k; j += VEC_SIZE) {
      aie::vector<bfloat16, VEC_SIZE> a_vec = aie::load_v<VEC_SIZE>(a_row + j);
      aie::vector<bfloat16, VEC_SIZE> b_vec = aie::load_v<VEC_SIZE>(B + j);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    float sum = aie::reduce_add(acc.template to_vector<float>());
    // Accumulate into existing C value (linalg.matvec semantics: C += A*B)
    C[row] = static_cast<bfloat16>(static_cast<float>(C[row]) + sum);
  }
}

} // extern "C"
