// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Flash attention for one core's LQ query rows, one LK-key block per call:
// S = Q K^T, the causal or sliding-window mask, the online softmax and
// O += P V, all in one pass per 8-row block of queries.
//
// Layouts, in 8x8 blocks (row-major inside a block); bfp16 is the cores'
// native matmul operand, packed by the host (hostops.c kv_rec_bfp, q_bfp):
//   q   bfp16 [LQ/8][DH/8]  rows q,   cols d
//   k   bfp16 [DH/8][LK/8]  rows key, cols d
//   v   bfp16 [DH/8][LK/8]  rows d,   cols key  (the blocks transposed)
//   o   [DH/8][LQ/8]  rows q,   cols d   (bf16, scaled by 1/l at the end)
//   ml  running max and sum (attn_init)
// qb and kb are the q and key block indices (LQ == LK), w the window in
// blocks: a key block is visible to q block qb when qb - w <= kb <= qb.

#include <aie_api/aie.hpp>

#ifndef DH
#define DH 128
#endif
#define LQ 32
#define LK 32

static_assert(LQ == LK, "the masks compare q and key block indices");

using BV = aie::block_vector<bfp16ebs8, 64>;
using ACC = aie::accum<accfloat, 64>;
using VB = aie::vector<bfloat16, 64>;

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr float SCALE = LOG2E / (DH == 64    ? 8.0f
                                        : DH == 128 ? 11.313708498984761f
                                                    : 16.0f);
// the most negative finite bf16: a masked score, and the initial max
static const bfloat16 LOWEST = -3.3895313892515355e38f;

// col - row of each lane of an 8x8 block
alignas(aie::vector_decl_align) static const int16 cmr[64] = {
    0,  1,  2,  3,  4,  5,  6, 7, -1, 0,  1,  2,  3,  4,  5,  6,
    -2, -1, 0,  1,  2,  3,  4, 5, -3, -2, -1, 0,  1,  2,  3,  4,
    -4, -3, -2, -1, 0,  1,  2, 3, -5, -4, -3, -2, -1, 0,  1,  2,
    -6, -5, -4, -3, -2, -1, 0, 1, -7, -6, -5, -4, -3, -2, -1, 0};

// lane r * 8 + c of the result is v[r]
static inline VB rows8(aie::vector<bfloat16, 8> v) {
  auto a = aie::interleave_zip(v, v, 1);
  aie::vector<bfloat16, 16> b = aie::concat(a.first, a.second);
  auto c = aie::interleave_zip(b, b, 2);
  aie::vector<bfloat16, 32> d = aie::concat(c.first, c.second);
  auto e = aie::interleave_zip(d, d, 4);
  return aie::concat(e.first, e.second);
}

// the max of each row (8 lanes) of an 8x8 block
static inline aie::vector<bfloat16, 8> row_max(VB v) {
  VB t = aie::transpose(v, 8, 8);
  aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
  aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
  return aie::max(b.extract<8>(0), b.extract<8>(1));
}

static inline aie::mask<64> keep_mask(int edge, int nk, int mq,
                                      aie::vector<int16, 64> d) {
  if (edge == 1)
    return nk < mq   ? aie::mask<64>(true)
           : nk > mq ? aie::mask<64>(false)
                     : aie::le(d, int16(0));
  return nk > mq   ? aie::mask<64>(true)
         : nk < mq ? aie::mask<64>(false)
                   : aie::gt(d, int16(0));
}

static inline BV to_bfp(VB v) {
  ACC a;
  a.from_vector(v);
  return a.template to_vector<bfp16ebs8>();
}

// 8x8x8 on bfp16, B given as N x K (row n is column n of B): the core's
// native form
static inline ACC mm(BV a, BV b) { return ACC(::mul_8x8_8x8T(a, b)); }
static inline ACC mma(ACC c, BV a, BV b) {
  return ACC(::mac_8x8_8x8T_conf(a, b, c, 0, 0, 0));
}

static inline VB exp2_scaled(VB x, VB m) {
  return aie::exp2<bfloat16>(
      aie::mul(aie::sub(x, m), bfloat16(SCALE)).template to_vector<float>());
}

extern "C" {

// ml: per row block, the running max (bf16 [LQ]) then the running sum as
// 8x8 blocks with each row's sum in all 8 lanes (float [LQ / 8][64])
void attn_init(bfloat16 *o, float *ml) {
  for (int i = 0; i < LQ * DH; i += 64)
    aie::store_v(o + i, aie::zeros<bfloat16, 64>());
  aie::store_v((bfloat16 *)ml, aie::broadcast<bfloat16, LQ>(LOWEST));
  for (int i = 0; i < LQ * 8; i += 64)
    aie::store_v(ml + LQ / 2 + i, aie::zeros<float, 64>());
}

void attn_blk(bfloat16 *qb16, bfloat16 *k, bfloat16 *v, bfloat16 *o, float *ml,
              int32_t qb, int32_t kb, int32_t w) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  if (kb > qb || kb < qb - w)
    return;
  // 1: the diagonal block, keys past the query masked; 2: the oldest block
  // of the window, keys at or before query - window masked
  const int edge = kb == qb ? 1 : kb == qb - w ? 2 : 0;
  const aie::vector<int16, 64> d = aie::load_v<64>(cmr);
  const VB lowest = aie::broadcast<bfloat16, 64>(LOWEST);
  const BV ones = to_bfp(aie::broadcast<bfloat16, 64>(1.0f));
  bfloat16 *mrow = (bfloat16 *)ml;
  float *lrow = ml + LQ / 2;
  constexpr int QROW = (DH / 8) * 72; // bytes of one row block's q

  for (int mq = 0; mq < LQ / 8; mq++) {
    ACC c0, c1, c2, c3;
    {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> qs(
          (const bfp16ebs8 *)((const char *)qb16 + mq * QROW));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ks((bfp16ebs8 *)k);
      BV a, b0, b1, b2, b3;
      qs >> a;
      ks >> b0 >> b1 >> b2 >> b3;
      c0 = mm(a, b0);
      c1 = mm(a, b1);
      c2 = mm(a, b2);
      c3 = mm(a, b3);
      for (int kd = 1; kd < DH / 8; kd++) {
        qs >> a;
        ks >> b0 >> b1 >> b2 >> b3;
        c0 = mma(c0, a, b0);
        c1 = mma(c1, a, b1);
        c2 = mma(c2, a, b2);
        c3 = mma(c3, a, b3);
      }
    }
    VB s0 = c0.template to_vector<bfloat16>();
    VB s1 = c1.template to_vector<bfloat16>();
    VB s2 = c2.template to_vector<bfloat16>();
    VB s3 = c3.template to_vector<bfloat16>();
    if (edge) {
      s0 = aie::select(lowest, s0, keep_mask(edge, 0, mq, d));
      s1 = aie::select(lowest, s1, keep_mask(edge, 1, mq, d));
      s2 = aie::select(lowest, s2, keep_mask(edge, 2, mq, d));
      s3 = aie::select(lowest, s3, keep_mask(edge, 3, mq, d));
    }
    aie::vector<bfloat16, 8> m_old = aie::load_v<8>(mrow + mq * 8);
    aie::vector<bfloat16, 8> m_new =
        aie::max(m_old, row_max(aie::max(aie::max(s0, s1), aie::max(s2, s3))));
    aie::vector<bfloat16, 16> dm = aie::concat(m_old, m_new);
    aie::vector<bfloat16, 16> dn = aie::concat(m_new, m_new);
    aie::vector<bfloat16, 8> c =
        aie::exp2<bfloat16>(aie::mul(aie::sub(dm, dn), bfloat16(SCALE))
                                .template to_vector<float>())
            .template extract<8>(0);
    aie::store_v(mrow + mq * 8, m_new);
    VB mp = rows8(m_new);
    VB cp = rows8(c);
    VB p0 = exp2_scaled(s0, mp), p1 = exp2_scaled(s1, mp);
    VB p2 = exp2_scaled(s2, mp), p3 = exp2_scaled(s3, mp);
    if (edge) {
      const VB z = aie::zeros<bfloat16, 64>();
      p0 = aie::select(z, p0, keep_mask(edge, 0, mq, d));
      p1 = aie::select(z, p1, keep_mask(edge, 1, mq, d));
      p2 = aie::select(z, p2, keep_mask(edge, 2, mq, d));
      p3 = aie::select(z, p3, keep_mask(edge, 3, mq, d));
    }
    BV q0 = to_bfp(p0), q1 = to_bfp(p1), q2 = to_bfp(p2), q3 = to_bfp(p3);

    // l = l c + rowsum(P): P times a block of ones sums each row into all
    // eight of its lanes
    aie::accum<accfloat, 64> cpa;
    cpa.from_vector(cp);
    float *lp = lrow + mq * 64;
    ACC L = aie::mul(aie::load_v<64>(lp), cpa.template to_vector<float>());
    L = mma(mma(mma(mma(L, q0, ones), q1, ones), q2, ones), q3, ones);
    aie::store_v(lp, L.template to_vector<float>());

    // two output column blocks at a time: two independent accumulators
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> vs((bfp16ebs8 *)v);
    for (int nd = 0; nd < DH / 8; nd += 2) {
      bfloat16 *o0 = o + (nd * (LQ / 8) + mq) * 64, *o1 = o0 + (LQ / 8) * 64;
      BV b0, b1, b2, b3, e0, e1, e2, e3;
      vs >> b0 >> b1 >> b2 >> b3 >> e0 >> e1 >> e2 >> e3;
      ACC r0 = aie::mul(aie::load_v<64>(o0), cp);
      ACC r1 = aie::mul(aie::load_v<64>(o1), cp);
      r0 = mma(r0, q0, b0);
      r1 = mma(r1, q0, e0);
      r0 = mma(r0, q1, b1);
      r1 = mma(r1, q1, e1);
      r0 = mma(r0, q2, b2);
      r1 = mma(r1, q2, e2);
      r0 = mma(r0, q3, b3);
      r1 = mma(r1, q3, e3);
      aie::store_v(o0, r0.template to_vector<bfloat16>());
      aie::store_v(o1, r1.template to_vector<bfloat16>());
    }
  }
}

void attn_fin(bfloat16 *o, float *ml) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int mq = 0; mq < LQ / 8; mq++) {
    aie::accum<accfloat, 64> ia;
    ia.from_vector(aie::inv(aie::load_v<64>(ml + LQ / 2 + mq * 64)));
    VB r = ia.template to_vector<bfloat16>();
    for (int nd = 0; nd < DH / 8; nd++) {
      bfloat16 *op = o + (nd * (LQ / 8) + mq) * 64;
      aie::store_v(
          op, aie::mul(aie::load_v<64>(op), r).template to_vector<bfloat16>());
    }
  }
}

} // extern "C"
