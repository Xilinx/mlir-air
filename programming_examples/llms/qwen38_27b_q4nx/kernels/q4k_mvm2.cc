// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Two tokens' Q4_K GEMV per weight block, for the speculative decode's verify
// pass: q4k_mvm.cc's block math, with the nibbles' dequantization and the
// scales and mins read once for both tokens.
//
// x holds the two tokens' rows back to back, [2][K]; acc is [token][scale
// path 32 | min path 32]; y is [16 + 2 * 32]: the packet header slot, then
// each token's rows.

#include "q4k_mvm.cc"

namespace {

// group sums of x as two bf16 limbs (hi, lo), as q4k_step takes them
inline void group_sums(const bf16 *x, bf16 *xs) {
  for (int g = 0; g < GROUPS; g++) {
    aie::accum<accfloat, GROUP> xa;
    xa.from_vector(aie::load_v<GROUP>(x + g * GROUP));
    float sum = aie::reduce_add(xa.template to_vector<float>());
    bf16 hi = (bf16)sum;
    xs[2 * g] = hi;
    xs[2 * g + 1] = (bf16)(sum - (float)hi);
  }
}

inline void q4k_block2(acc_t &s0, acc_t &m0, acc_t &s1, acc_t &m1,
                       const bf16 *x0, const bf16 *x1, const uint8_t *w,
                       const bf16 *xs0, const bf16 *xs1) {
  constexpr int NIBBLES = 128; // one 512-bit load
  constexpr int COLS_PER_LOAD = NIBBLES / ROWS;
  constexpr int LOADS = GROUP / COLS_PER_LOAD;
  const uint4 *qs = (const uint4 *)(w + QS_OFF);
  const uint8_t *sc = w;
  const uint8_t *mn = w + GROUPS * ROWS;
  // the scale path: the nibbles against both tokens. The min path, which
  // needs no nibbles, is a loop of its own below, so fewer accumulators are
  // live here (a 2 KB core stack takes no spills of them)
  acc_t as0, as1;
  as0.from_vector(aie::zeros<bf16, ROWS>());
  as1.from_vector(aie::zeros<bf16, ROWS>());
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(GROUPS, GROUPS)
  for (int g = 0; g < GROUPS; g++) {
    aie::vector<bf16, GROUP> xc0 = aie::load_v<GROUP>(x0 + g * GROUP);
    aie::vector<bf16, GROUP> xc1 = aie::load_v<GROUP>(x1 + g * GROUP);
    acc_t t0, t1;
    t0.from_vector(aie::zeros<bf16, ROWS>());
    t1.from_vector(aie::zeros<bf16, ROWS>());
    AIE_LOOP_UNROLL_FULL
    for (int l = 0; l < LOADS; l++) {
      aie::vector<uint4, NIBBLES> q_raw = aie::load_v<NIBBLES>(qs);
      qs += NIBBLES / 2; // a uint4 pointer steps a byte
      aie::accum<accfloat, NIBBLES> qa;
      qa.from_vector(aie::to_float(q_raw, 0));
      aie::vector<bf16, NIBBLES> q = qa.template to_vector<bf16>();
      AIE_LOOP_UNROLL_FULL
      for (int k = 0; k < COLS_PER_LOAD; k++) {
        aie::vector<bf16, ROWS> qk = q.template extract<ROWS>(k);
        t0 = aie::mac(t0, qk, xc0.get(l * COLS_PER_LOAD + k));
        t1 = aie::mac(t1, qk, xc1.get(l * COLS_PER_LOAD + k));
      }
    }
    acc_t s;
    s.from_vector(aie::to_float(aie::load_v<ROWS>(sc + g * ROWS), 0));
    aie::vector<bf16, ROWS> sv = to_bf16(s);
    as0 = mac_f32_bf16(as0, t0.template to_vector<float>(), sv);
    as1 = mac_f32_bf16(as1, t1.template to_vector<float>(), sv);
  }
  aie::vector<bf16, ROWS> S = aie::load_v<ROWS>((const bf16 *)(w + S_OFF));
  s0 = mac_f32_bf16(s0, as0.template to_vector<float>(), S);
  s1 = mac_f32_bf16(s1, as1.template to_vector<float>(), S);
  acc_t am0, am1;
  am0.from_vector(aie::zeros<bf16, ROWS>());
  am1.from_vector(aie::zeros<bf16, ROWS>());
  for (int g = 0; g < GROUPS; g++) {
    acc_t m;
    m.from_vector(aie::to_float(aie::load_v<ROWS>(mn + g * ROWS), 0));
    aie::vector<bf16, ROWS> mv = to_bf16(m);
    am0 = aie::mac(am0, mv, xs0[2 * g + 1]);
    am0 = aie::mac(am0, mv, xs0[2 * g]);
    am1 = aie::mac(am1, mv, xs1[2 * g + 1]);
    am1 = aie::mac(am1, mv, xs1[2 * g]);
  }
  aie::vector<bf16, ROWS> M = aie::load_v<ROWS>((const bf16 *)(w + M_OFF));
  m0 = mac_f32_bf16(m0, am0.template to_vector<float>(), M);
  m1 = mac_f32_bf16(m1, am1.template to_vector<float>(), M);
}

} // namespace

extern "C" {

// Block cb of a row block against both tokens' x[t][256 (cb - cb0) ..]: x
// holds the K range from column block cb0 on, K wide per token. cb == 0
// starts the row. acc has a slot per row group, rg the one this block adds
// to, so a projection whose K comes in parts keeps every row in progress.
void q4k_step2(bf16 *__restrict x, uint8_t *__restrict w, float *__restrict acc,
               int cb, int K, int cb0, int rg) {
  const bf16 *x0 = x + (cb - cb0) * COLS, *x1 = x0 + K;
  acc += 4 * ROWS * rg;
  alignas(aie::vector_decl_align) bf16 xs0[2 * GROUPS];
  alignas(aie::vector_decl_align) bf16 xs1[2 * GROUPS];
  group_sums(x0, xs0);
  group_sums(x1, xs1);
  acc_t s0, m0, s1, m1;
  if (cb == 0) {
    s0.from_vector(aie::zeros<bf16, ROWS>());
    m0.from_vector(aie::zeros<bf16, ROWS>());
    s1.from_vector(aie::zeros<bf16, ROWS>());
    m1.from_vector(aie::zeros<bf16, ROWS>());
  } else {
    s0.from_vector(aie::load_v<ROWS>(acc));
    m0.from_vector(aie::load_v<ROWS>(acc + ROWS));
    s1.from_vector(aie::load_v<ROWS>(acc + 2 * ROWS));
    m1.from_vector(aie::load_v<ROWS>(acc + 3 * ROWS));
  }
  q4k_block2(s0, m0, s1, m1, x0, x1, w, xs0, xs1);
  aie::store_v(acc, s0.template to_vector<float>());
  aie::store_v(acc + ROWS, m0.template to_vector<float>());
  aie::store_v(acc + 2 * ROWS, s1.template to_vector<float>());
  aie::store_v(acc + 3 * ROWS, m1.template to_vector<float>());
}

// Both tokens' row blocks into y + 16, token 0's then token 1's
void q4k_flush2(float *__restrict acc, bf16 *__restrict y, int rg) {
  acc += 4 * ROWS * rg;
  for (int t = 0; t < 2; t++) {
    acc_t c;
    c.from_vector(aie::load_v<ROWS>(acc + 2 * ROWS * t));
    c = aie::add(c, aie::load_v<ROWS>(acc + 2 * ROWS * t + ROWS));
    aie::vector<bf16, ROWS> v = to_bf16(c);
    aie::store_v(y + 16 + ROWS * t, v.template extract<16>(0));
    aie::store_v(y + 32 + ROWS * t, v.template extract<16>(1));
  }
}

} // extern "C"
