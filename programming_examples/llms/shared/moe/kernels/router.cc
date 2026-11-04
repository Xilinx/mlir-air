// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// MoE router, top-k and expert combine, transcribed from FastFlowLM's GPT-OSS
// FLM_Xclbin/GPTOSS/gpt_npu_bin/kernel/gptoss_moe_router_{master,slave}.cc and
// mvm_bf16.hpp. FLM splits the router GEMV between a master core (rows 0..15)
// and a slave core (rows 16..31) reading one shared block; router_mvm_half is
// that per-core step. The expert weight fetches FLM's slave issues by control
// packet come from the compiler here (a core-chosen L3 offset), so they are
// not in these kernels.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "lut_based_ops.h"
#include "sort.hpp"

namespace {

#ifndef MOE_N_EXPERTS
#define MOE_N_EXPERTS 32
#endif
#ifndef MOE_TOP_K
#define MOE_TOP_K 4
#endif

constexpr int NUM_OF_EXPERT = MOE_N_EXPERTS;
constexpr int TOP_K = MOE_TOP_K;
static_assert(NUM_OF_EXPERT % 16 == 0 && TOP_K <= 16);
constexpr int BLOCK_ROWS = NUM_OF_EXPERT; // a router block: every expert's row
constexpr int BLOCK_COLS = 64;            // EXPERT_BLOCK_COLS
constexpr int TILE_ROWS = 16;             // EXPERT_BLOCK_TILE_ROWS
constexpr int MVM_INTERLEAVE_SIZE = 128;

// FLM matvec_vectorized_Bf16_col_major<16, 64, 16>
inline void matvec_col_major_16x64(const bfloat16 *mat_in,
                                   const bfloat16 *vec_in,
                                   float *result_acc_ptr) {
  const bfloat16 *a_ptr = mat_in;
  aie::accum<accfloat, TILE_ROWS> res_acc;
  res_acc.from_vector(aie::load_v<TILE_ROWS>(result_acc_ptr));
  const bfloat16 *b_ptr = vec_in;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(BLOCK_COLS / 16, BLOCK_COLS / 16)
  for (int col_idx = 0; col_idx < BLOCK_COLS; col_idx += 16) {
    aie::vector<bfloat16, 16> b_vec = aie::load_v<16>(b_ptr);
    b_ptr += 16;
    aie::vector<bfloat16, TILE_ROWS> a[16];
    AIE_LOOP_UNROLL_FULL
    for (int c = 0; c < 16; c++) {
      a[c] = aie::load_v<TILE_ROWS>(a_ptr);
      a_ptr += TILE_ROWS;
    }
    res_acc = aie::accumulate<TILE_ROWS>(
        res_acc, b_vec, 0, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
        a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
  }
  aie::store_v(result_acc_ptr, res_acc.to_vector<float>());
}

} // namespace

extern "C" {

// acc[32] = router bias (FLM's master loads it from the first weight block).
void router_bias_init(bfloat16 *__restrict bias, float *__restrict acc) {
  for (int i = 0; i < BLOCK_ROWS; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(bias + i));
    aie::store_v(acc + i, a.to_vector<float>());
  }
}

// One core's half of a 32x64 router block (tiles of 16 rows, column-major):
// acc[half*16 .. +16] += block rows half*16.. . x_full[j*64 .. +64]
void router_mvm_half(bfloat16 *__restrict w_blk, bfloat16 *__restrict x_full,
                     float *__restrict acc, int j, int half) {
  matvec_col_major_16x64(w_blk + half * (BLOCK_ROWS / 2) * BLOCK_COLS,
                         x_full + j * BLOCK_COLS,
                         acc + half * (BLOCK_ROWS / 2));
}

// The whole block on one core.
void router_mvm(bfloat16 *__restrict w_blk, bfloat16 *__restrict x_full,
                float *__restrict acc, int j) {
  for (int t = 0; t < BLOCK_ROWS / TILE_ROWS; t++)
    matvec_col_major_16x64(w_blk + t * TILE_ROWS * BLOCK_COLS,
                           x_full + j * BLOCK_COLS, acc + t * TILE_ROWS);
}

void router_flush(float *__restrict acc, bfloat16 *__restrict logits) {
  for (int i = 0; i < BLOCK_ROWS; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(acc + i));
    aie::store_v(logits + i, a.to_vector<bfloat16>());
  }
}

// FLM select_top_k_with_index, split where it issues the gate/up weight
// fetches: router_sort, then the fetches, then router_softmax.

// Sorts logits[32] in place (largest first), their expert ids into index[32];
// base offsets the ids into a tensor holding several layers' experts.
void router_sort(bfloat16 *__restrict logits, int32_t *__restrict index,
                 int32_t base) {
  bubble_sort_largest_to_smallest_bf16_with_index<NUM_OF_EXPERT>(logits, index);
  for (int k = 0; k < TOP_K; k++)
    index[k] += base;
}

// top_k_data[0..TOP_K) = softmax of the TOP_K largest of the sorted logits.
//   top_k_data: 16 bf16
void router_softmax(bfloat16 *__restrict logits,
                    bfloat16 *__restrict top_k_data) {
  float exp_sum = 0.0;
  aie::vector<bfloat16, 16> top_16_vec = aie::load_v<16>(logits);
  bfloat16 max_value_vec = aie::reduce_max(top_16_vec);
  aie::vector<bfloat16, 16> sub_value = aie::sub(top_16_vec, max_value_vec);
  constexpr int min_clamp = -87.0f;
  constexpr int max_clamp = 88.0f;
  aie::vector<bfloat16, 16> top_16_vec_clamped =
      aie::clamp(sub_value, aie::broadcast<bfloat16, 16>(min_clamp),
                 aie::broadcast<bfloat16, 16>(max_clamp));
  aie::accum<accfloat, 16> exp_res_acc = getExpBf16(top_16_vec_clamped);
  aie::store_v(top_k_data, exp_res_acc.to_vector<bfloat16>());
  for (int l = 0; l < TOP_K; l++)
    exp_sum += top_k_data[l];

  // 1/sum as rsqrt(sum)^2, FLM's fast inverse square root
  float x2, divrms;
  const float threehalfs = 1.5F;
  uint32_t i_u32;
  x2 = exp_sum * 0.5F;
  divrms = exp_sum;
  i_u32 = *(uint32_t *)&divrms;
  i_u32 = 0x5f3759df - (i_u32 >> 1);
  divrms = *(float *)&i_u32;
  divrms = divrms * (threehalfs - (x2 * divrms * divrms));
  divrms = divrms * (threehalfs - (x2 * divrms * divrms));
  float inv_exp_sum = divrms * divrms;
  for (int i = 0; i < TOP_K; i++)
    top_k_data[i] = top_k_data[i] * inv_exp_sum;
}

// Weighted sum of the TOP_K experts' down projections for one chunk of
// MVM_INTERLEAVE_SIZE outputs; FLM stacks the experts' chunks back to back:
// expert k's values are at down[k*128 .. +128]. AIR loops over the chunks.
void moe_combine(bfloat16 *__restrict down, bfloat16 *__restrict top_k_data,
                 bfloat16 *__restrict out) {
  constexpr int vec_len = 16;
  AIE_LOOP_RANGE(MVM_INTERLEAVE_SIZE / vec_len, MVM_INTERLEAVE_SIZE / vec_len)
  for (int l = 0; l < MVM_INTERLEAVE_SIZE; l += vec_len) {
    aie::accum<accfloat, vec_len> acc_res = aie::zeros<accfloat, vec_len>();
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < TOP_K; k++) {
      aie::vector<bfloat16, vec_len> expert_down_val =
          aie::load_v<vec_len>(down + k * MVM_INTERLEAVE_SIZE + l);
      acc_res = aie::mac(acc_res, top_k_data[k], expert_down_val);
    }
    aie::store_v(out + l, acc_res.to_vector<bfloat16>());
  }
}

} // extern "C"
