//===- layer_norm_rows.cc ------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Affine LayerNorm over LN_ROWS rows of LN_N bf16 elements on one AIE2P tile.
//
// The arithmetic follows the air.api builder in layer_norm.py:
//
//   mean = bf16(sum_f32(x) / N)
//   d    = x - mean                       (bf16 vector)
//   var  = sum_f32(bf16(d * d)) / N
//   rstd = bf16(rsqrt(var + eps))
//   y    = bf16(bf16(d * rstd) * weight + bias)
//
// with the last multiply-add as one accumulate, i.e. one rounding, which is
// what the vector codegen of the air.api version does. The output is not
// bit-identical to it (the fp32 sums are reduced in a different order, so a few
// rows round differently), but it is as accurate: cosine against the f32
// reference is the same to 6 digits, and the largest difference is one bf16
// ulp.
//
// Why it is a C++ kernel: the air.api version spends ~2700 cycles per row, all
// of it latency, on a strictly serial chain (reduce through an L1 scratch
// buffer, a second reduce, then the epilogue). Here 32 lanes are used, and
// LN_GROUP rows are processed together so their independent chains overlap.
//
// Tuning notes (NPU2, M=3072 rows over 8 tiles): LN_ROWS == LN_GROUP == 4 is
// the best point. Larger groups spill accumulators to the stack and get slower;
// LN_ROWS = LN_GROUP = 8 does not even run.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef LN_N
#define LN_N 768
#endif
#ifndef LN_ROWS
#define LN_ROWS 4
#endif
#ifndef LN_GROUP
#define LN_GROUP LN_ROWS
#endif

constexpr int VL = 32;
constexpr int NV = LN_N / VL;
static_assert(LN_N % VL == 0, "N must be a multiple of 32");
static_assert(LN_ROWS % LN_GROUP == 0, "rows must be a multiple of the group");

using V = aie::vector<bfloat16, VL>;

static inline float hsum(const aie::accum<accfloat, VL> &a) {
  aie::vector<float, VL> v = a.to_vector<float>();
  aie::vector<float, 16> s = aie::add(v.extract<16>(0), v.extract<16>(1));
  return aie::reduce_add(s);
}

extern "C" {

// x, y: [LN_ROWS, LN_N] bf16, row-major. param: weight[LN_N] || bias[LN_N].
void ln_rows_bf16(const bfloat16 *__restrict x,
                  const bfloat16 *__restrict param, bfloat16 *__restrict y) {
  const bfloat16 *w = param;
  const bfloat16 *b = param + LN_N;
  const float inv_n = 1.0f / (float)LN_N;
  const float eps = 1e-5f;

  for (int g = 0; g < LN_ROWS; g += LN_GROUP) {
    const bfloat16 *xr[LN_GROUP];
    bfloat16 *yr[LN_GROUP];
#pragma clang loop unroll(full)
    for (int r = 0; r < LN_GROUP; r++) {
      xr[r] = x + (g + r) * LN_N;
      yr[r] = y + (g + r) * LN_N;
    }

    // Pass 1: per-lane f32 sums, one independent chain per row.
    aie::accum<accfloat, VL> s[LN_GROUP];
#pragma clang loop unroll(full)
    for (int r = 0; r < LN_GROUP; r++)
      s[r] = aie::zeros<accfloat, VL>();
    for (int c = 0; c < NV; c++)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
#pragma clang loop unroll(full)
        for (int r = 0; r < LN_GROUP; r++)
          s[r] = aie::add(s[r], aie::load_v<VL>(xr[r] + c * VL));
      }
    V mean_v[LN_GROUP];
#pragma clang loop unroll(full)
    for (int r = 0; r < LN_GROUP; r++) {
      bfloat16 m = (bfloat16)(hsum(s[r]) * inv_n);
      mean_v[r] = aie::broadcast<bfloat16, VL>(m);
    }

    // Pass 2: d = x - mean (parked in y), variance accumulation.
    aie::accum<accfloat, VL> q[LN_GROUP];
#pragma clang loop unroll(full)
    for (int r = 0; r < LN_GROUP; r++)
      q[r] = aie::zeros<accfloat, VL>();
    for (int c = 0; c < NV; c++)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
#pragma clang loop unroll(full)
        for (int r = 0; r < LN_GROUP; r++) {
          V d = aie::sub(aie::load_v<VL>(xr[r] + c * VL), mean_v[r]);
          aie::store_v(yr[r] + c * VL, d);
          V sq = aie::mul(d, d).template to_vector<bfloat16>();
          q[r] = aie::add(q[r], sq);
        }
      }
    V rstd_v[LN_GROUP];
#pragma clang loop unroll(full)
    for (int r = 0; r < LN_GROUP; r++) {
      float rs = aie::invsqrt(hsum(q[r]) * inv_n + eps);
      rstd_v[r] = aie::broadcast<bfloat16, VL>((bfloat16)rs);
    }

    // Pass 3: y = bf16(bf16(d * rstd) * w + b).
    for (int c = 0; c < NV; c++)
      chess_prepare_for_pipelining chess_loop_range(4, ) {
        V wv = aie::load_v<VL>(w + c * VL);
        V bv = aie::load_v<VL>(b + c * VL);
#pragma clang loop unroll(full)
        for (int r = 0; r < LN_GROUP; r++) {
          V d = aie::load_v<VL>(yr[r] + c * VL);
          V t = aie::mul(d, rstd_v[r]).template to_vector<bfloat16>();
          aie::accum<accfloat, VL> o(bv);
          o = aie::mac(o, t, wv);
          aie::store_v(yr[r] + c * VL, o.template to_vector<bfloat16>());
        }
      }
  }
}

} // extern "C"
