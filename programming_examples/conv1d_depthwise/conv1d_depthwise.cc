//===- conv1d_depthwise.cc - causal depthwise 1-D conv kernel -*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Causal depthwise 1-D convolution with kernel size 3 — the convolution
// inside LFM2's `Lfm2ShortConv` operator.
//
//   y[t, c] = w[0, c] * x[t + 0, c]
//           + w[1, c] * x[t + 1, c]
//           + w[2, c] * x[t + 2, c]
//
// Depthwise: each channel `c` is convolved by its own 3 taps, with no
// cross-channel mixing. The channel axis is therefore the vectorization
// axis and is contiguous in both `x` and `y`.
//
// CAUSALITY IS EXPRESSED BY PRE-PADDING, NOT BY MASKING. `x` carries
// `ts + 2` rows while `y` carries `ts`: row `t` of `x` is the sample at
// original position `t - 2`. So `x` row `t` is the OLDEST sample feeding
// `y[t]` and pairs with tap 0 — matching PyTorch `nn.Conv1d`'s
// cross-correlation over a left-padded input (oldest-first tap order).
//
// The two leading rows of `x` are the conv state: zeros at the start of a
// sequence (prefill), or the carried tail of the previous chunk (decode).
// That makes prefill and decode the same kernel with a different pad.
//
// `w` is laid out TAP-MAJOR — shape (3, tc), so `w + j*tc` is a contiguous
// channel slice for tap `j`. HuggingFace stores the kernel channel-major as
// (C, 1, 3); the host transposes it once at load time so this inner loop can
// use unit-stride vector loads.
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

extern "C" {

void conv1d_depthwise_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y, int32_t ts,
                           int32_t tc) {
  constexpr int VecLen = 16;
  const int F = tc / VecLen;

  bfloat16 *__restrict pY = y;

  for (int t = 0; t < ts; t++) {
    const bfloat16 *__restrict p0 = x + (t + 0) * tc;
    const bfloat16 *__restrict p1 = x + (t + 1) * tc;
    const bfloat16 *__restrict p2 = x + (t + 2) * tc;
    const bfloat16 *__restrict pw0 = w + 0 * tc;
    const bfloat16 *__restrict pw1 = w + 1 * tc;
    const bfloat16 *__restrict pw2 = w + 2 * tc;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 0; i < F; i++) {
      aie::vector<bfloat16, VecLen> v0 = aie::load_v<VecLen>(p0);
      p0 += VecLen;
      aie::vector<bfloat16, VecLen> v1 = aie::load_v<VecLen>(p1);
      p1 += VecLen;
      aie::vector<bfloat16, VecLen> v2 = aie::load_v<VecLen>(p2);
      p2 += VecLen;
      aie::vector<bfloat16, VecLen> c0 = aie::load_v<VecLen>(pw0);
      pw0 += VecLen;
      aie::vector<bfloat16, VecLen> c1 = aie::load_v<VecLen>(pw1);
      pw1 += VecLen;
      aie::vector<bfloat16, VecLen> c2 = aie::load_v<VecLen>(pw2);
      pw2 += VecLen;

      // FP32 accumulate across the 3 taps (the GPU/HF standard: upcast the
      // bf16 products, accumulate in f32, round once on store). Expressed as
      // mul + 2x mac so it lowers to the vector FMA unit — deliberately NOT
      // a mulf->addf chain, which aievec rejects.
      aie::accum<accfloat, VecLen> acc = aie::mul(v0, c0);
      acc = aie::mac(acc, v1, c1);
      acc = aie::mac(acc, v2, c2);

      aie::store_v(pY, acc.to_vector<bfloat16>());
      pY += VecLen;
    }
  }
}

} // extern "C"
