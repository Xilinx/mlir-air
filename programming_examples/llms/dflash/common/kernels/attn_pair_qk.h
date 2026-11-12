// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Included after the QK implementation, reusing its softmax update.
static_assert(KV_HEADS_PER_CU == 2 && Q_HEADS_PER_GROUP == 4 &&
              Q_HEADS_PADDED_PER_CU == 8 && DH == 128);
static_assert(ATTN_QUERY_TILE >= 2 && ATTN_QUERY_TILE % 2 == 0);

#ifndef ATTN_NATIVE_KEY
#define ATTN_NATIVE_KEY 1
#endif
static_assert(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16);
static inline __attribute__((always_inline)) aie::block_vector<bfp16ebs8, 64>
dflash_qk_native_key(const bf16 *keys) {
  aie::accum<accfloat, 64> expanded = ::mul_elem_64(
      aie::load_v<64>(keys), ::concat(::broadcast_one_to_v32bfloat16(),
                                      ::broadcast_one_to_v32bfloat16()));
  return aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(expanded));
}

static inline __attribute__((always_inline)) void
dflash_qk_pair_raw(bf16 *__restrict queries, bf16 *__restrict keys,
                   bf16 *__restrict scores, int first_row, int local) {
#if ATTN_NATIVE_KEY
  using MM = aie::mmul<8, 8, 8, bf16, bfp16ebs8, accfloat>;
#else
  using MM = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
#endif
  constexpr int query_width = Q_HEADS_PADDED_PER_CU * DH;
#pragma clang loop unroll(disable)
  for (int group = 0; group < 2; ++group) {
    MM left(aie::zeros<bf16, 64>()), right(aie::zeros<bf16, 64>());
#pragma clang loop unroll(disable)
    for (int d = 0; d < DH / 8; ++d) {
      auto a =
          aie::concat(aie::load_v<32>(queries + first_row * query_width +
                                      d * 64 + group * 32),
                      aie::load_v<32>(queries + (first_row + 1) * query_width +
                                      d * 64 + group * 32));
      bf16 *k = keys + group * DH * 16 + d * 128;
#if ATTN_NATIVE_KEY
      left.mac(a, aie::op_transpose(dflash_qk_native_key(k)));
      right.mac(a, aie::op_transpose(dflash_qk_native_key(k + 64)));
#else
      left.mac(a, aie::transpose(aie::load_v<64>(k), 8, 8));
      right.mac(a, aie::transpose(aie::load_v<64>(k + 64), 8, 8));
#endif
    }
    auto interleaved = aie::interleave_zip(left.template to_vector<bf16>(),
                                           right.template to_vector<bf16>(), 8);
    // Match the original BF16 conversion and scale before online softmax.
    // Store raw scores so the matrix accumulators die before the LUT work.
    auto first0 =
        aie::mul(aie::filter_even(interleaved.first, 32), (bf16)ATTN_SCALE)
            .template to_vector<bf16>(0);
    auto first1 =
        aie::mul(aie::filter_odd(interleaved.first, 32), (bf16)ATTN_SCALE)
            .template to_vector<bf16>(0);
    auto second0 =
        aie::mul(aie::filter_even(interleaved.second, 32), (bf16)ATTN_SCALE)
            .template to_vector<bf16>(0);
    auto second1 =
        aie::mul(aie::filter_odd(interleaved.second, 32), (bf16)ATTN_SCALE)
            .template to_vector<bf16>(0);
    aie::store_v(scores + local * SSZ_BLK + group * 64, first0);
    aie::store_v(scores + local * SSZ_BLK + group * 64 + 32, first1);
    aie::store_v(scores + (local + 1) * SSZ_BLK + group * 64, second0);
    aie::store_v(scores + (local + 1) * SSZ_BLK + group * 64 + 32, second1);
  }
}

static inline __attribute__((always_inline)) void
dflash_qk_tile(bf16 *__restrict queries, bf16 *__restrict keys,
               bf16 *__restrict maximum, float *__restrict correction,
               bf16 *__restrict scores, int block, int length, int first_row,
               int causal) {
  aie_round_nearest_even();
#pragma clang loop unroll(disable)
  for (int local = 0; local < ATTN_QUERY_TILE; local += 2) {
    if (length + (local + 1) * causal - block * 16 > 0)
      dflash_qk_pair_raw(queries, keys, scores, first_row + local, local);
  }
  const auto neg_inf = aie::broadcast<bf16, 16>(-0x1.FEp127f);
  const aie::vector<int, 16> positions(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16);
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
    // update() for all eight heads of the query in one pass: per head the
    // same operations, with the maxima and corrections kept in vectors.
    const auto lowest = aie::broadcast<bf16, 16>(-0x1.FEp127f);
    // update()'s clamp on the exponent
    const auto lo = aie::broadcast<bf16, 16>(-87.0f);
    const auto hi = aie::broadcast<bf16, 16>(88.0f);
    aie::vector<bf16, 128> raw = aie::load_v<128>(s);
    aie::vector<bf16, 128> masked;
#pragma clang loop unroll(full)
    for (int head = 0; head < 8; ++head)
      masked.insert(head,
                    aie::select(lowest, raw.template extract<16>(head), mask));
    // per half of four heads: [head][key] -> [key][head], then max over the
    // keys down to [2 keys][4 heads]
    auto half_max = [](aie::vector<bf16, 64> v) {
      aie::vector<bf16, 64> t = aie::transpose(v, 4, 16);
      aie::vector<bf16, 32> a =
          aie::max(t.template extract<32>(0), t.template extract<32>(1));
      aie::vector<bf16, 16> b =
          aie::max(a.template extract<16>(0), a.template extract<16>(1));
      return aie::max(b.template extract<8>(0), b.template extract<8>(1));
    };
    aie::vector<bf16, 16> both =
        aie::concat(half_max(masked.template extract<64>(0)),
                    half_max(masked.template extract<64>(1)));
    aie::vector<bf16, 8> rowmax =
        aie::max(aie::filter_even(both, 4), aie::filter_odd(both, 4));
    aie::vector<bf16, 8> old_m = aie::load_v<8>(m);
    aie::vector<bf16, 8> vmax = aie::max(rowmax, old_m);
    aie::vector<bf16, 16> corr =
        aie::clamp(aie::sub(old_m, vmax).template grow<16>(), lo, hi);
    aie::vector<float, 16> cf =
        aie::accum<accfloat, 16>(getExpBf16(corr)).template to_vector<float>();
    aie::store_v(m, vmax);
    aie::store_v(c, cf.template extract<8>(0));
    aie::store_v(reinterpret_cast<float *>(s + 128), cf.template extract<8>(0));
    // each head's max repeated over its 16 keys, [head][key]
    aie::vector<bf16, 64> per_head =
        aie::transpose(vmax.template grow_replicate<64>(), 8, 8);
    auto [heads03, heads47] = aie::interleave_zip(per_head, per_head, 8);
    aie::vector<bf16, 128> vm128 = aie::concat(heads03, heads47);
#pragma clang loop unroll(full)
    for (int head = 0; head < 8; ++head) {
      aie::vector<bf16, 16> d =
          aie::clamp(aie::sub(raw.template extract<16>(head),
                              vm128.template extract<16>(head)),
                     lo, hi);
      aie::vector<bf16, 16> e =
          aie::accum<accfloat, 16>(getExpBf16(d)).template to_vector<bf16>(0);
      aie::store_v(s + head * 16, aie::select((bf16)0, e, mask));
    }
  }
}
