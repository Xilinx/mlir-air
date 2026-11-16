// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The LM head for two rows at once, the speculative decode's verify pass:
// Qwen3.6's LM head kernels (lm_head.cc), each Q8_0 weight block dequantized
// once and applied to both tokens' x.

#include "lm_head.cc"

namespace {
using q36_q8::acc_t;
using q36_q8::COLS;
using q36_q8::GROUP;
using q36_q8::GROUPS;
using q36_q8::ROWS;
} // namespace

extern "C" {

// xw = [x0 | x1 | w]: both rows normed in place
void lm_rms_norm2(bf16 *__restrict xw) {
  rms_norm_inplace(xw, xw + 2 * MODEL_DIM);
  rms_norm_inplace(xw + MODEL_DIM, xw + 2 * MODEL_DIM);
}

// q8_step on x = [2][MODEL_DIM]; acc [2][ROWS], token t's at acc + ROWS t
void q8_step2(bf16 *__restrict x_full, uint8_t *__restrict w,
              float *__restrict acc, int cb) {
  const bf16 *x0 = x_full + cb * COLS, *x1 = x0 + MODEL_DIM;
  const bf16 *scale = (const bf16 *)w;
  const int8_t *q = (const int8_t *)(w + 2 * GROUPS * ROWS);
  acc_t c0, c1;
  if (cb == 0) {
    c0 = aie::zeros<accfloat, ROWS>();
    c1 = aie::zeros<accfloat, ROWS>();
  } else {
    c0.from_vector(aie::load_v<ROWS>(acc));
    c1.from_vector(aie::load_v<ROWS>(acc + ROWS));
  }
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(GROUPS, GROUPS)
  for (int g = 0; g < GROUPS; g++) {
    aie::vector<bf16, GROUP> xa = aie::load_v<GROUP>(x0 + g * GROUP);
    aie::vector<bf16, GROUP> xb = aie::load_v<GROUP>(x1 + g * GROUP);
    acc_t t0 = aie::zeros<accfloat, ROWS>();
    acc_t t1 = aie::zeros<accfloat, ROWS>();
    AIE_LOOP_UNROLL(4)
    for (int k = 0; k < GROUP; k++) {
      acc_t qa;
      qa.from_vector(aie::to_float(aie::load_v<ROWS>(q), 0));
      q += ROWS;
      aie::vector<bf16, ROWS> qb = bf16_limb::to_bf16(qa);
      t0 = aie::mac(t0, qb, xa.get(k));
      t1 = aie::mac(t1, qb, xb.get(k));
    }
    aie::vector<bf16, ROWS> sc = aie::load_v<ROWS>(scale + g * ROWS);
    c0 = bf16_limb::mac_f32_bf16_low(c0, t0.template to_vector<float>(), sc);
    c1 = bf16_limb::mac_f32_bf16_low(c1, t1.template to_vector<float>(), sc);
  }
  aie::store_v(acc, c0.template to_vector<float>());
  aie::store_v(acc + ROWS, c1.template to_vector<float>());
}

// both rows' blocks as bf16 at y + 16: token 0's 32, then token 1's
void q8_flush2(float *__restrict acc, bf16 *__restrict y) {
  for (int t = 0; t < 2; t++) {
    acc_t c;
    c.from_vector(aie::load_v<ROWS>(acc + t * ROWS));
    aie::vector<bf16, ROWS> v = bf16_limb::to_bf16(c);
    aie::store_v(y + 16 + t * ROWS, v.template extract<16>(0));
    aie::store_v(y + 32 + t * ROWS, v.template extract<16>(1));
  }
}

} // extern "C"
