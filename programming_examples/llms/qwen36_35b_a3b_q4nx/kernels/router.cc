// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Qwen3.6 MoE router, transcribed from FastFlowLM's
// FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding (router.cc, ab_proj.cc, sort.hpp,
// nonLinearFunction.h): the router GEMV into float32 logits, a partial
// selection of the TOP_K largest logits, then a float32 softmax over those
// alone (the renormalized top-k of a softmax over every expert).
//
// A router block is 32 experts x 256 inputs of bf16, column-major; FLM splits
// each block's rows between two cores, 16 apiece. The expert weight fetches
// FLM issues by control packet come from the compiler here.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"

namespace {

#ifndef MOE_N_EXPERTS
#define MOE_N_EXPERTS 256
#endif
constexpr int N_EXPERTS = MOE_N_EXPERTS;
constexpr int TOP_K = 8;
constexpr int BLOCK_ROWS = 32;
constexpr int BLOCK_COLS = 256;
constexpr int HALF = BLOCK_ROWS / 2;

// acc[0:16] += the 16 rows from `rows` of a column-major 32 x 256 block . x
inline void matvec_half(const bfloat16 *blk, int rows, const bfloat16 *x,
                        float *acc) {
  aie::accum<accfloat, HALF> a;
  a.from_vector(aie::load_v<HALF>(acc));
  const bfloat16 *col = blk + rows;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(BLOCK_COLS / 16, BLOCK_COLS / 16)
  for (int c = 0; c < BLOCK_COLS; c += 16) {
    aie::vector<bfloat16, 16> xv = aie::load_v<16>(x + c);
    aie::vector<bfloat16, HALF> w[16];
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < 16; k++)
      w[k] = aie::load_v<HALF>(col + (c + k) * BLOCK_ROWS);
    a = aie::accumulate<HALF>(a, xv, 0, w[0], w[1], w[2], w[3], w[4], w[5],
                              w[6], w[7], w[8], w[9], w[10], w[11], w[12],
                              w[13], w[14], w[15]);
  }
  aie::store_v(acc, a.template to_vector<float>());
}

} // namespace

extern "C" {

void router_zero(float *__restrict logits) {
  for (int i = 0; i < N_EXPERTS; i += 16)
    aie::store_v(logits + i, aie::zeros<float, 16>());
}

// Both halves of block (rb, cb) on one core.
void router_mvm(bfloat16 *__restrict blk, bfloat16 *__restrict x_full,
                float *__restrict logits, int rb, int cb) {
  for (int h = 0; h < 2; h++)
    matvec_half(blk, h * HALF, x_full + cb * BLOCK_COLS,
                logits + rb * BLOCK_ROWS + h * HALF);
}

// index[0:TOP_K] = the experts of the TOP_K largest logits, largest first,
// offset by base; score[0:TOP_K] = their softmax. FastFlowLM's selection: a
// swap can move a lower expert behind an equal logit. logits is reordered in
// place.
void router_topk(float *__restrict logits, int32_t *__restrict index,
                 float *__restrict score, int32_t base) {
  int idx[N_EXPERTS];
  for (int i = 0; i < N_EXPERTS; i++)
    idx[i] = i;
  for (int i = 0; i < TOP_K; i++) {
    int max_j = i;
    for (int j = i + 1; j < N_EXPERTS; j++)
      if (logits[j] > logits[max_j])
        max_j = j;
    if (max_j != i) {
      float td = logits[i];
      logits[i] = logits[max_j];
      logits[max_j] = td;
      int ti = idx[i];
      idx[i] = idx[max_j];
      idx[max_j] = ti;
    }
  }
  constexpr float log2_e = 1.44269504088896341f;
  // 2^-127 flushes to zero, so the lanes past TOP_K drop out of the sum
  aie::vector<float, 16> t = aie::broadcast<float, 16>(-127.0f);
  for (int i = 0; i < TOP_K; i++)
    t.set((logits[i] - logits[0]) * log2_e, i);
  aie::vector<bfloat16, 16> e = aie::exp2<bfloat16>(t);
  float sum = 0.0f;
  for (int i = 0; i < TOP_K; i++)
    sum += (float)e.get(i);
  const float inv = 1.0f / sum;
  for (int i = 0; i < TOP_K; i++) {
    index[i] = idx[i] + base;
    score[i] = (float)e.get(i) * inv;
  }
}

// router_topk of layer b of the run: its experts' rows start at
// a + N_EXPERTS b
void router_topk_at(float *__restrict logits, int32_t *__restrict index,
                    float *__restrict score, int32_t a, int32_t b) {
  router_topk(logits, index, score, a + N_EXPERTS * b);
}

// index[slot] = a + b: a row the core starts a transfer of
void router_slot(int32_t *__restrict index, int32_t slot, int32_t a,
                 int32_t b) {
  index[slot] = a + b;
}

} // extern "C"
