//===- mv.cc ------------------------------------------------------*- C++
//-*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Matrix-vector multiplication kernel for AIE.
// Ported from IRON generic/mv.cc.
//
// C[M] = A[M,K] @ B[K]
//
// The kernel processes `m` output rows per call. Each row computes the dot
// product of one row of A with the full vector B, accumulating in accfloat
// and reducing to a scalar bfloat16 result.
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

/*
Matrix-vector multiplication kernel (vectorized)

 - m: Number of output rows == number of rows in the input matrix chunk
 - k: Number of columns in the input matrix == length of the input vector
 - a: Pointer to the input matrix, stored in row-major order
 - b: Pointer to the input vector
 - c: Pointer to the output vector
 - r: Vector size; data from the matrix and vector will be loaded in and
      processed in chunks of this size
*/
template <uint32_t r>
void matvec_vectorized(uint32_t m, uint32_t k, const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *c_end = c + m;
  const bfloat16 *b_end = b + k;
  for (; c < c_end; c++) {
    aie::accum acc = aie::zeros<accfloat, r>();
    for (const bfloat16 *__restrict b_cur = b; b_cur < b_end;
         b_cur += r, a += r) {
      aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    *c =
        static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
  }
}

extern "C" {

#ifndef DIM_M_OUTPUT
#define DIM_M_OUTPUT 2048
#endif

/* The row_offset parameter offsets writes into c: c_out += row_offset.
 * This allows multiple kernel calls to fill different parts of the output
 * buffer without pointer arithmetic in the calling MLIR code. */

void matvec_vectorized_bf16_bf16(uint32_t m, uint32_t k, uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_vectorized<64>(m, k, a_in, b_in, c_out);
}

void linalg_fill_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, DIM_M_OUTPUT, 1, 32>(c_out);
}

} // extern "C"
