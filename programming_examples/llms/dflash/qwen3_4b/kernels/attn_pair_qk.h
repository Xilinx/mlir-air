// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Included after the current QK implementation, reusing its softmax update.
static_assert(KV_HEADS_PER_CU == 2 && Q_HEADS_PER_GROUP == 4 &&
              Q_HEADS_PADDED_PER_CU == 8 && DH == 128);
static_assert(ATTN_QUERY_TILE >= 2 && ATTN_QUERY_TILE % 2 == 0);

static_assert(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16);
static inline __attribute__((always_inline))
aie::block_vector<bfp16ebs8,64> dflash_qk_native_key(const bf16 *keys) {
  aie::accum<accfloat,64> expanded=::mul_elem_64(aie::load_v<64>(keys),
      ::concat(::broadcast_one_to_v32bfloat16(),::broadcast_one_to_v32bfloat16()));
  return aie::block_vector<bfp16ebs8,64>(::to_v64bfp16ebs8(expanded));
}

static inline __attribute__((always_inline))
void dflash_qk_pair_raw(bf16 *__restrict queries, bf16 *__restrict keys,
                        bf16 *__restrict scores, int first_row, int local) {
  using MM = aie::mmul<8, 8, 8, bf16, bfp16ebs8, accfloat>;
  constexpr int query_width = Q_HEADS_PADDED_PER_CU * DH;
  #pragma clang loop unroll(disable)
  for (int group = 0; group < 2; ++group) {
    MM left(aie::zeros<bf16, 64>()), right(aie::zeros<bf16, 64>());
    #pragma clang loop unroll(disable)
    for (int d = 0; d < DH / 8; ++d) {
      auto a = aie::concat(
          aie::load_v<32>(queries + first_row * query_width + d * 64 + group * 32),
          aie::load_v<32>(queries + (first_row + 1) * query_width + d * 64 + group * 32));
      bf16 *k = keys + group * DH * 16 + d * 128;
      left.mac(a, aie::op_transpose(dflash_qk_native_key(k)));
      right.mac(a, aie::op_transpose(dflash_qk_native_key(k + 64)));
    }
    auto interleaved = aie::interleave_zip(left.template to_vector<bf16>(),
                                           right.template to_vector<bf16>(), 8);
    // Match the original BF16 conversion and scale before online softmax.
    // Store raw scores so the matrix accumulators die before the LUT work.
    auto first0 = aie::mul(aie::filter_even(interleaved.first, 32), (bf16)ATTN_SCALE)
                      .template to_vector<bf16>(0);
    auto first1 = aie::mul(aie::filter_odd(interleaved.first, 32), (bf16)ATTN_SCALE)
                      .template to_vector<bf16>(0);
    auto second0 = aie::mul(aie::filter_even(interleaved.second, 32), (bf16)ATTN_SCALE)
                       .template to_vector<bf16>(0);
    auto second1 = aie::mul(aie::filter_odd(interleaved.second, 32), (bf16)ATTN_SCALE)
                       .template to_vector<bf16>(0);
    aie::store_v(scores + local * SSZ_BLK + group * 64, first0);
    aie::store_v(scores + local * SSZ_BLK + group * 64 + 32, first1);
    aie::store_v(scores + (local + 1) * SSZ_BLK + group * 64, second0);
    aie::store_v(scores + (local + 1) * SSZ_BLK + group * 64 + 32, second1);
  }
}

static inline __attribute__((always_inline))
void dflash_qk_tile(bf16 *__restrict queries, bf16 *__restrict keys,
                     bf16 *__restrict maximum, float *__restrict correction,
                     bf16 *__restrict scores, int block, int length,
                     int first_row, int causal) {
  aie_round_nearest_even();
  #pragma clang loop unroll(disable)
  for (int local = 0; local < ATTN_QUERY_TILE; local += 2) {
    if (length + (local + 1) * causal - block * 16 > 0)
      dflash_qk_pair_raw(queries, keys, scores, first_row + local, local);
  }
  const auto neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
  const aie::vector<int, 16> positions(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
  #pragma clang loop unroll(disable)
  for (int local = 0; local < ATTN_QUERY_TILE; ++local) {
    bf16 *m = maximum + local * 16;
    float *c = correction + local * 8;
    bf16 *s = scores + local * SSZ_BLK;
    if (block == 0)
      aie::store_v(m, neg_inf);
    int valid = length + local * causal - block * 16;
    if (valid <= 0)
      continue;
    valid = valid < 16 ? valid : 16;
    auto mask = aie::le(positions, valid);
    bool first = block == 0;
    #pragma clang loop unroll(disable)
    for (int head = 0; head < 8; ++head) {
      auto raw = aie::load_v<16>(s + head * 16);
      auto probability = update(m + head, c + head, raw, mask, first);
      aie::store_v(s + head * 16, probability);
    }
    aie::store_v(reinterpret_cast<float *>(s + 128), aie::load_v<8>(c));
  }
}
