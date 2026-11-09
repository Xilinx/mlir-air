// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The inputs of a Gated DeltaNet decode step, transcribed from FastFlowLM's
// FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/kernels (conv.cc, ab_proj.cc,
// nonLinearFunction.h), with their approximations: SiLU and the decay through
// the AIE2P exp2 instruction, softplus through FLM's linear-approximation
// table, sigmoid through tanh.
//
//   gdn_conv     the causal depthwise conv4 of 128 channels, then SiLU
//   gdn_l2norm   a head's q or k l2-normalized (q also scaled by 1/sqrt(128))
//   ab_*         alpha / beta: the [64, 2048] projection (32 x 256 column-major
//                blocks), then alpha = exp(A softplus(a + dt)), beta =
//                sigmoid(b)
//   gdn_pack     a value head's [k, alpha | v, beta | q] for the step cores

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"
#include "softplus_lut.h"

using bf16 = bfloat16;

namespace {

constexpr int HD = 128; // head dim, keys and values
constexpr int NK = 16, NV = 32;
constexpr int DQ = NK * HD, DKV = NK * HD, DV = NV * HD;
constexpr int SLOT = HD + 32;
constexpr int BLOCK_ROWS = 32, BLOCK_COLS = 256;
constexpr float SCALE = 0.08838834764831845f; // 1 / sqrt(128)
constexpr bf16 LOG2_E = 1.44269504089f;

// acc[0:16] += the 16 rows from `rows` of a column-major 32 x 256 block . x
inline void matvec_half(const bf16 *blk, int rows, const bf16 *x, float *acc) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(acc));
  const bf16 *col = blk + rows;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(BLOCK_COLS / 16, BLOCK_COLS / 16)
  for (int c = 0; c < BLOCK_COLS; c += 16) {
    aie::vector<bf16, 16> xv = aie::load_v<16>(x + c);
    aie::vector<bf16, 16> w[16];
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < 16; k++)
      w[k] = aie::load_v<16>(col + (c + k) * BLOCK_ROWS);
    a = aie::accumulate<16>(a, xv, 0, w[0], w[1], w[2], w[3], w[4], w[5], w[6],
                            w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14],
                            w[15]);
  }
  aie::store_v(acc, a.template to_vector<float>());
}

template <int N>
aie::vector<float, N> softplus(aie::vector<bf16, N> x) {
  using lut_type = aie::lut<4, float, bf16>;
  lut_type lut(64, (bf16 *)softplus_lut_ab, (bf16 *)softplus_lut_cd);
  aie::linear_approx<bf16, lut_type> approx(lut, -2, 32, 0);
  return approx.compute(x).template to_vector<float>();
}

// e^x as FLM's exp2_bf16_noLUT: clamp, times log2(e) in bf16, exp2
template <int N>
aie::vector<bf16, N> exp_bf16(aie::vector<bf16, N> x) {
  x = aie::clamp(x, aie::broadcast<bf16, N>(-87.0f),
                 aie::broadcast<bf16, N>(88.0f));
  aie::accum<accfloat, N> s = aie::mul(x, LOG2_E);
  return aie::exp2<bf16>(s.template to_vector<float>());
}

// (tanh(x / 2) + 1) / 2
template <int N>
aie::vector<float, N> sigmoid(aie::vector<float, N> x) {
  aie::vector<float, N> half = aie::broadcast<float, N>(0.5f);
  aie::accum<accfloat, N> xh = aie::mul(x, half);
  aie::vector<bf16, N> t = aie::tanh<bf16>(xh.template to_vector<float>());
  aie::accum<accfloat, N> ta;
  ta.from_vector(t);
  aie::vector<float, N> t1 =
      aie::add(ta.template to_vector<float>(), aie::broadcast<float, N>(1.0f));
  return aie::mul(t1, half);
}

} // namespace

extern "C" {

// y[128 c ..] = SiLU(sum_t w_t * x_t), wx = [x0, w0, x1, w1, x2, w2, x3, w3],
// each 128 channels, x0 the oldest input.
void gdn_conv(bf16 *__restrict wx, bf16 *__restrict y, int c) {
  y += c * HD;
  constexpr int V = 16;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(HD / V, HD / V)
  for (int i = 0; i < HD; i += V) {
    aie::accum<accfloat, V> acc =
        aie::mul(aie::load_v<V>(wx + i), aie::load_v<V>(wx + HD + i));
    AIE_LOOP_UNROLL_FULL
    for (int t = 1; t < 4; t++)
      acc = aie::mac(acc, aie::load_v<V>(wx + 2 * t * HD + i),
                     aie::load_v<V>(wx + (2 * t + 1) * HD + i));
    aie::vector<float, V> yf = acc.template to_vector<float>();
    aie::vector<bf16, V> yb = acc.template to_vector<bf16>();
    // SiLU(y) = y / (1 + 2^(-y log2 e))
    aie::vector<float, V> ys =
        aie::mul(yb, aie::broadcast<bf16, V>(-(bf16)LOG2_E));
    aie::accum<accfloat, V> den;
    den.from_vector(aie::exp2<bf16>(ys));
    den = aie::add(den, aie::broadcast<bf16, V>((bf16)1.0f));
    aie::vector<float, V> inv = aie::inv(den.template to_vector<float>());
    aie::accum<accfloat, V> s = aie::mul(inv, yf);
    aie::store_v(y + i, s.template to_vector<bf16>());
  }
}

// The same conv as the layer runs it: hx = [x0, x1, x2, w0, w1, w2, w3] from
// the cache, x3 the chunk's new input (chunk c % 4 of a 512-row round); the
// cache's next inputs ho = [x1, x2, x3].
void gdn_conv_cache(bf16 *__restrict hx, bf16 *__restrict rnd,
                    bf16 *__restrict y, bf16 *__restrict ho, int c) {
  const bf16 *x3 = rnd + (c % 4) * HD;
  y += c * HD;
  constexpr int V = 16;
  for (int i = 0; i < HD; i += V) {
    aie::vector<bf16, V> x0 = aie::load_v<V>(hx + i);
    aie::vector<bf16, V> x1 = aie::load_v<V>(hx + HD + i);
    aie::vector<bf16, V> x2 = aie::load_v<V>(hx + 2 * HD + i);
    aie::vector<bf16, V> xn = aie::load_v<V>(x3 + i);
    aie::accum<accfloat, V> acc = aie::mul(x0, aie::load_v<V>(hx + 3 * HD + i));
    acc = aie::mac(acc, x1, aie::load_v<V>(hx + 4 * HD + i));
    acc = aie::mac(acc, x2, aie::load_v<V>(hx + 5 * HD + i));
    acc = aie::mac(acc, xn, aie::load_v<V>(hx + 6 * HD + i));
    aie::vector<float, V> yf = acc.template to_vector<float>();
    aie::vector<bf16, V> yb = acc.template to_vector<bf16>();
    aie::vector<float, V> ys =
        aie::mul(yb, aie::broadcast<bf16, V>(-(bf16)LOG2_E));
    aie::accum<accfloat, V> den;
    den.from_vector(aie::exp2<bf16>(ys));
    den = aie::add(den, aie::broadcast<bf16, V>((bf16)1.0f));
    aie::vector<float, V> inv = aie::inv(den.template to_vector<float>());
    aie::accum<accfloat, V> s = aie::mul(inv, yf);
    aie::store_v(y + i, s.template to_vector<bf16>());
    aie::store_v(ho + i, x1);
    aie::store_v(ho + HD + i, x2);
    aie::store_v(ho + 2 * HD + i, xn);
  }
}

// y[128 c ..][0:128] /= ||.||, then *= 1/sqrt(128) when scale (q)
void gdn_l2norm(bf16 *__restrict y, int c, int scale) {
  y += c * HD;
  constexpr int V = 16;
  aie::accum<accfloat, V> s0 = aie::zeros<accfloat, V>();
  aie::accum<accfloat, V> s1 = aie::zeros<accfloat, V>();
  for (int i = 0; i < HD; i += 2 * V) {
    s0 = aie::mac_square(s0, aie::load_v<V>(y + i));
    s1 = aie::mac_square(s1, aie::load_v<V>(y + i + V));
  }
  float sum = aie::reduce_add(aie::add(s0.template to_vector<float>(),
                                       s1.template to_vector<float>())) +
              1e-6f;
  float inv = aie::invsqrt(aie::broadcast<float, V>(sum))[0];
  for (int i = 0; i < HD; i += V) {
    aie::accum<accfloat, V> a;
    a.from_vector(aie::load_v<V>(y + i));
    aie::vector<bf16, V> o = aie::mul(a.template to_vector<float>(), inv);
    if (scale)
      o = aie::mul(o, aie::broadcast<bf16, V>((bf16)SCALE));
    aie::store_v(y + i, o);
  }
}

void ab_zero(float *__restrict ab) {
  for (int i = 0; i < 2 * NV; i += 16)
    aie::store_v(ab + i, aie::zeros<float, 16>());
}

// ab[32 rb ..] += block (rb, cb) . x_full[256 cb ..]; rows 0..31 alpha's
// projection a, 32..63 beta's b
void ab_mvm(bf16 *__restrict blk, bf16 *__restrict x_full, float *__restrict ab,
            int rb, int cb) {
  for (int h = 0; h < 2; h++)
    matvec_half(blk, h * 16, x_full + cb * BLOCK_COLS,
                ab + rb * BLOCK_ROWS + h * 16);
}

// ab[0:32] = exp(A softplus(a + dt)), ab[32:64] = sigmoid(b);
// a_dt = [A (= -exp(A_log)) 32 | dt bias 32]
void ab_activate(float *__restrict ab, float *__restrict a_dt) {
  for (int i = 0; i < NV; i += 16) {
    aie::vector<float, 16> dt =
        aie::add(aie::load_v<16>(ab + i), aie::load_v<16>(a_dt + NV + i));
    aie::accum<accfloat, 16> da;
    da.from_vector(dt);
    aie::vector<bf16, 16> d = aie::clamp(da.template to_vector<bf16>(),
                                         aie::broadcast<bf16, 16>(-87.0f),
                                         aie::broadcast<bf16, 16>(88.0f));
    aie::vector<float, 16> g =
        aie::mul(softplus<16>(d), aie::load_v<16>(a_dt + i));
    aie::accum<accfloat, 16> ga;
    ga.from_vector(g);
    aie::vector<bf16, 16> gb = ga.template to_vector<bf16>();
    aie::accum<accfloat, 16> e;
    e.from_vector(exp_bf16<16>(gb));
    aie::store_v(ab + i, e.template to_vector<float>());
  }
  for (int i = NV; i < 2 * NV; i += 16)
    aie::store_v(ab + i, sigmoid<16>(aie::load_v<16>(ab + i)));
}

// kvq = [k_{h / 2}, alpha_h | v_h, beta_h | q_{h / 2}] of value head h, from
// conv = [q 2048 | k 2048 | v 4096]
void gdn_pack(bf16 *__restrict conv, float *__restrict ab, bf16 *__restrict kvq,
              int h) {
  const bf16 *q = conv + (h / 2) * HD;
  const bf16 *k = conv + DQ + (h / 2) * HD;
  const bf16 *v = conv + DQ + DKV + h * HD;
  for (int i = 0; i < HD; i += 16) {
    aie::store_v(kvq + i, aie::load_v<16>(k + i));
    aie::store_v(kvq + SLOT + i, aie::load_v<16>(v + i));
    aie::store_v(kvq + 2 * SLOT + i, aie::load_v<16>(q + i));
  }
  *(float *)(kvq + HD) = ab[h];
  *(float *)(kvq + SLOT + HD) = ab[NV + h];
}

} // extern "C"
