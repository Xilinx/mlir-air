//===- attn_decode_npu2.cc --------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Decode-phase flash attention kernels for AIE2P (NPU2).
//
// Uses matvec-style (mv.cc) QK computation since group_size is too small for
// 8x8 tiled matmul.  Flat row-major buffer layout throughout.
//
// Compile-time parameters (pass with -D):
//   group_size : number of Q heads per KV head (GQA ratio), e.g. 4
//   lkp        : K/V chunk size, e.g. 64
//   dk         : key head dimension, e.g. 64
//   dv         : value head dimension, e.g. 64
//
// Buffer layouts (all flat row-major):
//   Q        [group_size, dk]
//   K        [lkp, dk]
//   V        [lkp, dv]
//   scores   [group_size, lkp]
//   output   [group_size, dv]
//   max_vec  [max_sum_buf_size]   (first group_size entries used)
//   sum_vec  [max_sum_buf_size]   (first group_size entries used)
//
// max_sum_buf_size must be >= group_size and >= 32 (one 512-bit cascade word).
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

// Default values if not provided by Makefile
#ifndef group_size
#define group_size 4
#endif

#ifndef lkp
#define lkp 64
#endif

#ifndef dk
#define dk 64
#endif

#ifndef dv
#define dv 64
#endif

// Cascade alignment: 512-bit bus = 32 bfloat16 elements.
// max/sum buffers are padded to this size.
#ifndef max_sum_buf_size
#define max_sum_buf_size 32
#endif

// Scale: log2e / sqrt(dk) for attention score normalization.
// Using the same constexpr pattern as attn_npu2.cc.
#include <cmath>

constexpr double decode_constexpr_sqrt_dk = (dk == 64)    ? 8.0
                                            : (dk == 128) ? 11.313708498984761
                                            : (dk == 256) ? 16.0
                                            : (dk == 512) ? 22.627416997969522
                                                          : 8.0;
static_assert(dk == 64 || dk == 128 || dk == 256 || dk == 512,
              "Unsupported dk value: update decode_constexpr_sqrt_dk");

#define decode_log2e (1.44269504089 / decode_constexpr_sqrt_dk)

// Bf16 special values
static const uint16_t bf16_lowest_u16 = (uint16_t)0xff7f;  // ≈ -3.39e38
static const uint16_t bf16_neg_inf_u16 = (uint16_t)0xff80; // -inf

//===========================================================================
// Internal: matvec C[m] = A[m,k] @ b[k]  (A row-major, b a vector)
// Accumulates in accfloat and reduces each row to a scalar bfloat16 result.
//===========================================================================
template <uint32_t r>
static void matvec_decode(uint32_t m, uint32_t k, const bfloat16 *__restrict a,
                          const bfloat16 *__restrict b,
                          bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *b_end = b + k;
  for (uint32_t row = 0; row < m; row++, c++) {
    aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
    const bfloat16 *b_cur = b;
    const bfloat16 *a_cur = a + row * k;
    for (; b_cur < b_end; b_cur += r, a_cur += r) {
      aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a_cur);
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    *c =
        static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
  }
}

extern "C" {

#ifdef ROUND_CONV_EVEN
#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)
#else
#define SET_ROUNDING() /* no-op */
#endif

// ============================================================
// Initialization
// ============================================================

void decode_zero_output(bfloat16 *out) {
  SET_ROUNDING();
  zero_vectorized<bfloat16, group_size, dv, 32>(out);
}

void decode_zero_sum(bfloat16 *sum) {
  SET_ROUNDING();
  zero_vectorized<bfloat16, max_sum_buf_size, 1, 32>(sum);
}

void decode_neg_inf_max(bfloat16 *max_v) {
  SET_ROUNDING();
  neg_inf_vectorized<bfloat16, max_sum_buf_size, 1, 32>(max_v);
}

// ============================================================
// QK score computation (matvec-style)
// ============================================================

// For each head h in [0, group_size):
//   scores[h, i] = K[i, :] . Q[h, :]   for i in [0, lkp)
// K is [lkp, dk] row-major, Q is [group_size, dk] row-major.
// scores is [group_size, lkp] row-major.
void compute_qk_scores_bf16(bfloat16 *K, bfloat16 *Q, bfloat16 *scores) {
  SET_ROUNDING();
  for (int h = 0; h < group_size; h++) {
    matvec_decode<64>(lkp, dk, K, Q + h * dk, scores + h * lkp);
  }
}

// ============================================================
// Decode causal mask
// ============================================================

// Set scores[h, i] = -inf for ALL heads h when (chunk_start + i) > current_pos.
// TODO(perf): Replace scalar conditional with aie::sel vectorized mask:
//   aie::mask<lkp> m = aie::lt(position_vec, aie::broadcast(current_pos));
//   scores_vec = aie::sel(neg_inf_vec, scores_vec, m);
// This eliminates lkp scalar branches per invocation (lkp=64, group_size=4
// → 256 scalar comparisons per call vs. one vectorized mask operation).
void apply_decode_mask(bfloat16 *scores, int32_t chunk_start,
                       int32_t current_pos) {
  SET_ROUNDING();
  bfloat16 neg_inf_val = *(const bfloat16 *)&bf16_neg_inf_u16;
  for (int i = 0; i < lkp; i++) {
    if (chunk_start + i > current_pos) {
      for (int h = 0; h < group_size; h++) {
        scores[h * lkp + i] = neg_inf_val;
      }
    }
  }
}

// ============================================================
// Online softmax
// ============================================================

// Compute per-head row-max of scores[group_size, lkp].
// score_max[h] = max(scores[h, :]).
// Does NOT consider any running max from previous chunks.
void decode_softmax_max(bfloat16 *scores, bfloat16 *score_max) {
  SET_ROUNDING();
  constexpr int VecLen = 16;
  bfloat16 lowest_val = *(const bfloat16 *)&bf16_lowest_u16;
  for (int h = 0; h < group_size; h++) {
    bfloat16 *sc_h = scores + h * lkp;
    aie::vector<bfloat16, VecLen> max_v =
        aie::broadcast<bfloat16, VecLen>(lowest_val);
    for (int i = 0; i < lkp; i += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        max_v = aie::max(max_v, aie::load_v<VecLen>(sc_h + i));
      }
    // Reduce 16 → 1
    aie::vector<bfloat16, 8> r0 = max_v.extract<8>(0);
    aie::vector<bfloat16, 8> r1 = max_v.extract<8>(1);
    r0 = aie::max(r0, r1);
    score_max[h] = aie::reduce_max(r0);
  }
}

// In-place online softmax update:
//   combined_max[h] = max(max_vec[h], score_max[h])
//   rescale[h]      = exp(max_vec[h] - combined_max[h])   [for old output]
//   scores[h, i]    = exp(scores[h, i] - combined_max[h]) [in-place]
//   max_vec[h]      = combined_max[h]                      [running max
//   updated]
void decode_softmax_exp(bfloat16 *scores, bfloat16 *max_vec,
                        bfloat16 *score_max, bfloat16 *rescale) {
  SET_ROUNDING();
  constexpr int VecLen = 16;
  bfloat16 lowest_val = *(const bfloat16 *)&bf16_lowest_u16;
  aie::vector<bfloat16, VecLen> log2e_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)decode_log2e);
  aie::vector<bfloat16, VecLen> lowest_vec =
      aie::broadcast<bfloat16, VecLen>(lowest_val);

  for (int h = 0; h < group_size; h++) {
    float om = (float)max_vec[h];
    float nm = (float)score_max[h];
    float combined = (om > nm) ? om : nm;

    // Rescale factor for old accumulated output: exp(old_max - new_max)
    float diff_rescale_f = om - combined;
    bfloat16 diff_rescale = (bfloat16)diff_rescale_f;
    // Clamp to lowest before exp to avoid -inf input to exp2
    if ((float)diff_rescale < (float)lowest_val)
      diff_rescale = lowest_val;
    {
      aie::vector<bfloat16, 8> d8 = aie::broadcast<bfloat16, 8>(diff_rescale);
      aie::vector<bfloat16, 8> l8 =
          aie::broadcast<bfloat16, 8>((bfloat16)decode_log2e);
      aie::vector<bfloat16, 8> r8 =
          aie::exp2<bfloat16>(aie::mul(d8, l8).to_vector<float>());
      rescale[h] = r8.get(0);
    }

    // Update running max
    max_vec[h] = (bfloat16)combined;

    // Normalize scores: scores[h,i] = exp(scores[h,i] - combined)
    bfloat16 *sc_h = scores + h * lkp;
    aie::vector<bfloat16, VecLen> max_bcast =
        aie::broadcast<bfloat16, VecLen>((bfloat16)combined);
    for (int i = 0; i < lkp; i += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(sc_h + i);
        v = aie::sub(v, max_bcast);
        v = aie::max(v, lowest_vec); // clamp before exp
        v = aie::exp2<bfloat16>(aie::mul(v, log2e_vec).to_vector<float>());
        aie::store_v(sc_h + i, v);
      }
  }
}

// Update running sum: sum[h] = sum[h] * rescale[h] + row_sum(scores[h, :])
// scores[h, :] must already contain exp-normalized values (after
// decode_softmax_exp).
void decode_softmax_sum(bfloat16 *scores, bfloat16 *rescale, bfloat16 *sum) {
  SET_ROUNDING();
  constexpr int VecLen = 16;
  for (int h = 0; h < group_size; h++) {
    bfloat16 *sc_h = scores + h * lkp;
    aie::accum<accfloat, VecLen> row_acc = aie::zeros<accfloat, VecLen>();
    for (int i = 0; i < lkp; i += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        row_acc = aie::add(row_acc, aie::load_v<VecLen>(sc_h + i));
      }
    // Reduce 16 → scalar
    aie::vector<float, VecLen> sum_f = row_acc.to_vector<float>();
    aie::vector<float, 8> r0 = sum_f.extract<8>(0);
    aie::vector<float, 8> r1 = sum_f.extract<8>(1);
    r0 = aie::add(r0, r1);
    float row_sum = aie::reduce_add(r0);
    // update: sum = sum * rescale + row_sum
    sum[h] = (bfloat16)((float)sum[h] * (float)rescale[h] + row_sum);
  }
}

// Rescale output: output[h, :] *= rescale[h]
void decode_rescale_output(bfloat16 *rescale, bfloat16 *out) {
  SET_ROUNDING();
  constexpr int VecLen = 16;
  for (int h = 0; h < group_size; h++) {
    bfloat16 *out_h = out + h * dv;
    aie::vector<bfloat16, VecLen> r_bcast =
        aie::broadcast<bfloat16, VecLen>(rescale[h]);
    for (int j = 0; j < dv; j += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(out_h + j);
        aie::accum<accfloat, VecLen> acc = aie::mul(v, r_bcast);
        aie::store_v(out_h + j, acc.to_vector<bfloat16>());
      }
  }
}

// Accumulate PV: output[h, :] += sum_i scores[h, i] * V[i, :]
// V is [lkp, dv] row-major.  scores[h, :] must hold exp-normalized weights.
// Uses scalar-broadcast mac: for each position i, broadcast scores[h,i] and
// multiply with V[i, j:j+VecLen] vector chunk.
void compute_pv_output_bf16(bfloat16 *scores, bfloat16 *V, bfloat16 *out) {
  SET_ROUNDING();
  constexpr int VecLen = 16;
  bfloat16 one_bf16 = (bfloat16)1.0f;
  for (int h = 0; h < group_size; h++) {
    bfloat16 *out_h = out + h * dv;
    bfloat16 *sc_h = scores + h * lkp;
    for (int j = 0; j < dv; j += VecLen) {
      // Load existing output into accfloat (multiply by 1.0 to convert)
      aie::vector<bfloat16, VecLen> existing = aie::load_v<VecLen>(out_h + j);
      aie::vector<bfloat16, VecLen> one_vec =
          aie::broadcast<bfloat16, VecLen>(one_bf16);
      aie::accum<accfloat, VecLen> acc = aie::mul(existing, one_vec);
      // Scalar-broadcast mac over K/V positions.
      // TODO(perf): Add chess_prepare_for_pipelining pragma here (this is the
      // most compute-intensive inner loop) once exp2 vector usage (lines above)
      // is confirmed compatible with pipelining on AIE2P.
      for (int i = 0; i < lkp; i++) {
        aie::vector<bfloat16, VecLen> s_bcast =
            aie::broadcast<bfloat16, VecLen>(sc_h[i]);
        aie::vector<bfloat16, VecLen> v_row =
            aie::load_v<VecLen>(V + i * dv + j);
        acc = aie::mac(acc, s_bcast, v_row);
      }
      aie::store_v(out_h + j, acc.to_vector<bfloat16>());
    }
  }
}

// Final division: output[h, :] /= sum[h]
void decode_div_output(bfloat16 *sum, bfloat16 *out) {
  SET_ROUNDING();
  constexpr int VecLen = 16;
  for (int h = 0; h < group_size; h++) {
    bfloat16 *out_h = out + h * dv;
    bfloat16 inv_sum = (bfloat16)(1.0f / (float)sum[h]);
    aie::vector<bfloat16, VecLen> inv_bcast =
        aie::broadcast<bfloat16, VecLen>(inv_sum);
    for (int j = 0; j < dv; j += VecLen)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(out_h + j);
        aie::accum<accfloat, VecLen> acc = aie::mul(v, inv_bcast);
        aie::store_v(out_h + j, acc.to_vector<bfloat16>());
      }
  }
}

// ============================================================
// Cascade merge helpers
// ============================================================

// Element-wise max update: max_local[h] = max(max_local[h], max_recv[h])
// Only the first group_size entries are updated.
void decode_cascade_merge_max(bfloat16 *max_recv, bfloat16 *max_local) {
  SET_ROUNDING();
  for (int h = 0; h < group_size; h++) {
    float a = (float)max_local[h];
    float b = (float)max_recv[h];
    max_local[h] = (bfloat16)(a > b ? a : b);
  }
}

// Compute rescale factor: rescale[h] = exp(old_max[h] - new_max[h])
// old_max and new_max have group_size meaningful entries.
void decode_compute_rescale(bfloat16 *old_max, bfloat16 *new_max,
                            bfloat16 *rescale) {
  SET_ROUNDING();
  bfloat16 lowest_val = *(const bfloat16 *)&bf16_lowest_u16;
  for (int h = 0; h < group_size; h++) {
    float diff = (float)old_max[h] - (float)new_max[h];
    bfloat16 diff_bf16 = (bfloat16)diff;
    if ((float)diff_bf16 < (float)lowest_val)
      diff_bf16 = lowest_val;
    aie::vector<bfloat16, 8> d8 = aie::broadcast<bfloat16, 8>(diff_bf16);
    aie::vector<bfloat16, 8> l8 =
        aie::broadcast<bfloat16, 8>((bfloat16)decode_log2e);
    aie::vector<bfloat16, 8> r8 =
        aie::exp2<bfloat16>(aie::mul(d8, l8).to_vector<float>());
    rescale[h] = r8.get(0);
  }
}

// Element-wise add: out_dst[h, :] += out_src[h, :]
// Both are [group_size, dv] row-major.
void decode_add_output(bfloat16 *out_src, bfloat16 *out_dst) {
  SET_ROUNDING();
  constexpr int VecLen = 32;
  constexpr int num_elems = group_size * dv;
  bfloat16 *__restrict ps = out_src;
  bfloat16 *__restrict pd = out_dst;
  for (int i = 0; i < num_elems; i += VecLen)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      aie::vector<bfloat16, VecLen> vs = aie::load_v<VecLen>(ps);
      aie::vector<bfloat16, VecLen> vd = aie::load_v<VecLen>(pd);
      aie::accum<accfloat, VecLen> acc(vd);
      acc = aie::add(acc, vs);
      aie::store_v(pd, acc.to_vector<bfloat16>());
      ps += VecLen;
      pd += VecLen;
    }
}

// Add sums: sum_dst[h] += sum_src[h]  for h in [0, group_size)
void decode_add_sum(bfloat16 *sum_src, bfloat16 *sum_dst) {
  SET_ROUNDING();
  for (int h = 0; h < group_size; h++) {
    sum_dst[h] = (bfloat16)((float)sum_dst[h] + (float)sum_src[h]);
  }
}

// Rescale sum in-place: sum[h] *= rescale[h]  for h in [0, group_size)
void decode_rescale_sum(bfloat16 *rescale, bfloat16 *sum) {
  SET_ROUNDING();
  for (int h = 0; h < group_size; h++) {
    sum[h] = (bfloat16)((float)sum[h] * (float)rescale[h]);
  }
}

// Copy max/sum vector: dst[h] = src[h]  for h in [0, group_size)
void decode_copy_max_sum(bfloat16 *src, bfloat16 *dst) {
  SET_ROUNDING();
  for (int h = 0; h < group_size; h++) {
    dst[h] = src[h];
  }
}

} // extern "C"
