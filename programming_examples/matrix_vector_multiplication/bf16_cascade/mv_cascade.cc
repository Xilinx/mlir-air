//===- mv_cascade.cc ------------------------------------------------*- C++
//-*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Cascade-based matrix-vector multiplication kernel for AIE.
//
// C[M] = A[M,K] @ B[K]
//
// The K dimension is split across N_CASCADE tiles in a column.  Each tile
// computes partial dot products for its K-chunk.  The cascade channel
// transfers are handled by the compiler (via air.channel.put/get on cascade
// channels), NOT by explicit put_mcd/get_scd calls in the kernel.
//
// Three kernel variants:
//   - First tile:  compute partial sums → write to scratch buffer
//   - Middle tile: read upstream partial sums from recv buffer, add own
//                  partial dot products → write to scratch buffer
//   - Last tile:   read upstream partial sums from recv buffer, add own
//                  partial dot products → convert to bf16 → write to output
//
// Partial sums use float to preserve precision across the cascade pipeline.
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

// Vector width for bf16 MAC operations.
static constexpr uint32_t VEC_SIZE = 64;

// ---------------------------------------------------------------------------
// Compute partial dot products for `m` output rows over `k` columns.
// Accumulates in accfloat, reduces each row to a single float.
// Results are written to `partial[0..m-1]`.
// ---------------------------------------------------------------------------
static inline void matvec_partial_dot(uint32_t m, uint32_t k,
                                      const bfloat16 *__restrict a,
                                      const bfloat16 *__restrict b,
                                      float *__restrict partial) {
  for (uint32_t row = 0; row < m; ++row) {
    aie::accum<accfloat, VEC_SIZE> acc = aie::zeros<accfloat, VEC_SIZE>();
    const bfloat16 *a_row = a + row * k;
    for (uint32_t j = 0; j < k; j += VEC_SIZE) {
      aie::vector<bfloat16, VEC_SIZE> a_vec = aie::load_v<VEC_SIZE>(a_row + j);
      aie::vector<bfloat16, VEC_SIZE> b_vec = aie::load_v<VEC_SIZE>(b + j);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    partial[row] = aie::reduce_add(acc.template to_vector<float>());
  }
}

extern "C" {

#ifndef DIM_M_OUTPUT
#define DIM_M_OUTPUT 4
#endif

// ---------------------------------------------------------------------------
// First tile in the cascade column (ty == N_CASCADE-1, northernmost).
//
// Computes partial dot products and writes them to the scratch buffer.
// The caller (AIR code) then does ChannelPut to send via cascade.
//
// Arguments:
//   m       - number of output rows to process
//   k       - number of K elements for this tile's chunk
//   a_in    - pointer to A tile [m x k] in L1 (row-major)
//   b_in    - pointer to B slice [k] in L1
//   scratch - pointer to scratch buffer [tile_m] floats in L1
// ---------------------------------------------------------------------------
void matvec_cascade_first_bf16(uint32_t m, uint32_t k,
                               const bfloat16 *__restrict a_in,
                               const bfloat16 *__restrict b_in,
                               float *__restrict scratch) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  matvec_partial_dot(m, k, a_in, b_in, scratch);
}

// ---------------------------------------------------------------------------
// Middle tile in the cascade column (1 <= ty <= N_CASCADE-2).
//
// Reads upstream partial sums from recv buffer (filled by ChannelGet),
// computes own partial dot products, adds them together, and writes
// the accumulated result to the scratch buffer.
// The caller then does ChannelPut on scratch to forward via cascade.
//
// Arguments:
//   m       - number of output rows
//   k       - K elements for this tile's chunk
//   a_in    - A tile [m x k]
//   b_in    - B slice [k]
//   recv    - received upstream partial sums [tile_m] floats (from ChannelGet)
//   scratch - output buffer [tile_m] floats (for ChannelPut)
// ---------------------------------------------------------------------------
void matvec_cascade_middle_bf16(uint32_t m, uint32_t k,
                                const bfloat16 *__restrict a_in,
                                const bfloat16 *__restrict b_in,
                                const float *__restrict recv,
                                float *__restrict scratch) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (uint32_t row = 0; row < m; ++row) {
    aie::accum<accfloat, VEC_SIZE> acc = aie::zeros<accfloat, VEC_SIZE>();
    const bfloat16 *a_row = a_in + row * k;
    for (uint32_t j = 0; j < k; j += VEC_SIZE) {
      aie::vector<bfloat16, VEC_SIZE> a_vec = aie::load_v<VEC_SIZE>(a_row + j);
      aie::vector<bfloat16, VEC_SIZE> b_vec = aie::load_v<VEC_SIZE>(b_in + j);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    scratch[row] = recv[row] + aie::reduce_add(acc.template to_vector<float>());
  }
}

// ---------------------------------------------------------------------------
// Last tile in the cascade column (ty == 0, southernmost).
//
// Reads upstream partial sums from recv buffer (filled by ChannelGet),
// computes own partial dot products, adds them together, converts the
// final float sum to bf16, and writes to the output buffer.
//
// Arguments:
//   m          - number of output rows
//   k          - K elements for this tile's chunk
//   row_offset - offset into c_out for multi-call patterns
//   a_in       - A tile [m x k]
//   b_in       - B slice [k]
//   recv       - received upstream partial sums [tile_m] floats
//   c_out      - output buffer [tile_m] bf16
// ---------------------------------------------------------------------------
void matvec_cascade_last_bf16(uint32_t m, uint32_t k, uint32_t row_offset,
                              const bfloat16 *__restrict a_in,
                              const bfloat16 *__restrict b_in,
                              const float *__restrict recv,
                              bfloat16 *__restrict c_out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  c_out += row_offset;
  for (uint32_t row = 0; row < m; ++row) {
    aie::accum<accfloat, VEC_SIZE> acc = aie::zeros<accfloat, VEC_SIZE>();
    const bfloat16 *a_row = a_in + row * k;
    for (uint32_t j = 0; j < k; j += VEC_SIZE) {
      aie::vector<bfloat16, VEC_SIZE> a_vec = aie::load_v<VEC_SIZE>(a_row + j);
      aie::vector<bfloat16, VEC_SIZE> b_vec = aie::load_v<VEC_SIZE>(b_in + j);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    float total = recv[row] + aie::reduce_add(acc.template to_vector<float>());
    c_out[row] = static_cast<bfloat16>(total);
  }
}

// Zero-fill for output buffer.
void linalg_fill_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, DIM_M_OUTPUT, 1, 32>(c_out);
}

} // extern "C"
