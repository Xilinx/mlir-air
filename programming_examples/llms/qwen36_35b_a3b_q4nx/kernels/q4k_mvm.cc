// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Q4_K GEMV on one 32 x 256 weight block of FastFlowLM's bundle, as stored,
// transcribed from FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding (q4_core.h,
// bf16_limb_math.hpp, proj_main.cc): w = S' s'_j q + M' m'_j, the group sum
// reduced against x in float32 before it meets s'_j, and every float32 term
// that reaches the output carried as two bf16 limbs.
//
// FLM keeps the scale path and the min path in registers across a row's K
// loop; here AIR calls one block at a time, so they wait in acc between
// calls: acc[0:32] the scale path, acc[32:64] the min path.
//
// Block: scales uint8 [8 groups][32 rows], mins the same, nibbles
// [256 columns][32 rows] (low nibble the even row), S bf16 [32], M bf16 [32].

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"

using bf16 = bfloat16;

namespace {

constexpr int ROWS = 32;  // rows of a block, all resident at once
constexpr int COLS = 256; // one super-block
constexpr int GROUP = 32;
constexpr int GROUPS = COLS / GROUP;
constexpr int QS_OFF = 2 * GROUPS * ROWS;
constexpr int S_OFF = QS_OFF + ROWS * COLS / 2;
constexpr int M_OFF = S_OFF + 2 * ROWS;

using acc_t = aie::accum<accfloat, ROWS>;

// accum -> bf16 16 lanes at a time: Peano's 32-lane conversion returns the
// upper half twice.
inline aie::vector<bf16, ROWS> to_bf16(acc_t a) {
  aie::vector<float, ROWS> f = a.template to_vector<float>();
  aie::accum<accfloat, 16> lo, hi;
  lo.from_vector(f.template extract<16>(0));
  hi.from_vector(f.template extract<16>(1));
  return aie::concat(lo.template to_vector<bf16>(),
                     hi.template to_vector<bf16>());
}

// acc + a * b with a float32 carried as two bf16 limbs.
inline acc_t mac_f32_bf16(acc_t acc, aie::vector<float, ROWS> a,
                          aie::vector<bf16, ROWS> b) {
  const aie::vector<bf16, ROWS> one = aie::broadcast<bf16, ROWS>((bf16)1.0f);
  acc_t r0;
  r0.from_vector(a);
  aie::vector<bf16, ROWS> l0 = to_bf16(r0);
  aie::vector<bf16, ROWS> l1 = to_bf16(aie::msc(r0, l0, one));
  acc = aie::mac(acc, b, l1);
  return aie::mac(acc, b, l0);
}

// One block into the running scale / min paths. xs holds sum_i x_i of each
// group as two bf16 limbs (hi, lo).
inline void q4k_block(acc_t &c_s, acc_t &c_m, const bf16 *x, const uint8_t *w,
                      const bf16 *xs) {
  constexpr int NIBBLES = 128; // one 512-bit load
  constexpr int COLS_PER_LOAD = NIBBLES / ROWS;
  constexpr int LOADS = GROUP / COLS_PER_LOAD;
  const uint4 *qs = (const uint4 *)(w + QS_OFF);
  const uint8_t *sc = w;
  const uint8_t *mn = w + GROUPS * ROWS;
  acc_t acc_s, acc_m;
  acc_s.from_vector(aie::zeros<bf16, ROWS>());
  acc_m.from_vector(aie::zeros<bf16, ROWS>());

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(GROUPS, GROUPS)
  for (int g = 0; g < GROUPS; g++) {
    aie::vector<bf16, GROUP> xc = aie::load_v<GROUP>(x + g * GROUP);
    acc_t t;
    t.from_vector(aie::zeros<bf16, ROWS>());
    AIE_LOOP_UNROLL_FULL
    for (int l = 0; l < LOADS; l++) {
      aie::vector<uint4, NIBBLES> q_raw = aie::load_v<NIBBLES>(qs);
      qs += NIBBLES / 2; // a uint4 pointer steps a byte
      aie::accum<accfloat, NIBBLES> qa;
      qa.from_vector(aie::to_float(q_raw, 0));
      aie::vector<bf16, NIBBLES> q = qa.template to_vector<bf16>();
      AIE_LOOP_UNROLL_FULL
      for (int k = 0; k < COLS_PER_LOAD; k++)
        t = aie::mac(t, q.template extract<ROWS>(k),
                     xc.get(l * COLS_PER_LOAD + k));
    }
    acc_t s;
    s.from_vector(aie::to_float(aie::load_v<ROWS>(sc + g * ROWS), 0));
    acc_s = mac_f32_bf16(acc_s, t.template to_vector<float>(), to_bf16(s));
    acc_t m;
    m.from_vector(aie::to_float(aie::load_v<ROWS>(mn + g * ROWS), 0));
    aie::vector<bf16, ROWS> mv = to_bf16(m);
    acc_m = aie::mac(acc_m, mv, xs[2 * g + 1]);
    acc_m = aie::mac(acc_m, mv, xs[2 * g]);
  }
  aie::vector<bf16, ROWS> S = aie::load_v<ROWS>((const bf16 *)(w + S_OFF));
  aie::vector<bf16, ROWS> M = aie::load_v<ROWS>((const bf16 *)(w + M_OFF));
  c_s = mac_f32_bf16(c_s, acc_s.template to_vector<float>(), S);
  c_m = mac_f32_bf16(c_m, acc_m.template to_vector<float>(), M);
}

} // namespace

extern "C" {

// Block cb of a row block against x_full[256 cb : 256 (cb + 1)]; cb == 0
// starts the row.
void q4k_step(bf16 *__restrict x_full, uint8_t *__restrict w,
              float *__restrict acc, int cb) {
  const bf16 *x = x_full + cb * COLS;
  alignas(aie::vector_decl_align) bf16 xs[2 * GROUPS];
  for (int g = 0; g < GROUPS; g++) {
    aie::accum<accfloat, GROUP> xa;
    xa.from_vector(aie::load_v<GROUP>(x + g * GROUP));
    float sum = aie::reduce_add(xa.template to_vector<float>());
    bf16 hi = (bf16)sum;
    xs[2 * g] = hi;
    xs[2 * g + 1] = (bf16)(sum - (float)hi);
  }
  acc_t c_s, c_m;
  if (cb == 0) {
    c_s.from_vector(aie::zeros<bf16, ROWS>());
    c_m.from_vector(aie::zeros<bf16, ROWS>());
  } else {
    c_s.from_vector(aie::load_v<ROWS>(acc));
    c_m.from_vector(aie::load_v<ROWS>(acc + ROWS));
  }
  q4k_block(c_s, c_m, x, w, xs);
  aie::store_v(acc, c_s.template to_vector<float>());
  aie::store_v(acc + ROWS, c_m.template to_vector<float>());
}

// The row block, scale path plus min path, as bf16 at y + 16 (after the
// packet header slot).
void q4k_flush(float *__restrict acc, bf16 *__restrict y) {
  acc_t c;
  c.from_vector(aie::load_v<ROWS>(acc));
  c = aie::add(c, aie::load_v<ROWS>(acc + ROWS));
  // y + 16 is only 32-byte aligned: store 16 lanes at a time
  aie::vector<bf16, ROWS> v = to_bf16(c);
  aie::store_v(y + 16, v.template extract<16>(0));
  aie::store_v(y + 32, v.template extract<16>(1));
}

} // extern "C"
