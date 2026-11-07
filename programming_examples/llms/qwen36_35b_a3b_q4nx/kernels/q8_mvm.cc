// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Q8_0 GEMV on one 32 x 256 weight block of FastFlowLM's bundle, as stored
// (the LM head), transcribed from FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/
// kernels/q8_core.h: each group of 32 columns reduced against x in float32,
// then scaled by the group's bf16 scale with the sum carried as two limbs.
//
// Block: scales bf16 [8 groups][32 rows], then int8 [256 columns][32 rows].

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"
#include "bf16_limb.hpp"

using bf16 = bfloat16;

namespace {

constexpr int ROWS = 32, COLS = 256, GROUP = 32, GROUPS = COLS / GROUP;
using acc_t = aie::accum<accfloat, ROWS>;

} // namespace

extern "C" {

// Block cb of a row block against x_full[256 cb ..]; cb == 0 starts the row.
void q8_step(bf16 *__restrict x_full, uint8_t *__restrict w,
             float *__restrict acc, int cb) {
  const bf16 *x = x_full + cb * COLS;
  const bf16 *scale = (const bf16 *)w;
  const int8_t *q = (const int8_t *)(w + 2 * GROUPS * ROWS);
  acc_t c;
  if (cb == 0)
    c = aie::zeros<accfloat, ROWS>();
  else
    c.from_vector(aie::load_v<ROWS>(acc));
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(GROUPS, GROUPS)
  for (int g = 0; g < GROUPS; g++) {
    aie::vector<bf16, GROUP> xc = aie::load_v<GROUP>(x + g * GROUP);
    acc_t t = aie::zeros<accfloat, ROWS>();
    AIE_LOOP_UNROLL(4)
    for (int k = 0; k < GROUP; k++) {
      acc_t qa;
      qa.from_vector(aie::to_float(aie::load_v<ROWS>(q), 0));
      q += ROWS;
      t = aie::mac(t, bf16_limb::to_bf16(qa), xc.get(k));
    }
    c = bf16_limb::mac_f32_bf16_low(c, t.template to_vector<float>(),
                                    aie::load_v<ROWS>(scale + g * ROWS));
  }
  aie::store_v(acc, c.template to_vector<float>());
}

// The row block as bf16 at y + 16 (after the packet header slot).
void q8_flush(float *__restrict acc, bf16 *__restrict y) {
  acc_t c;
  c.from_vector(aie::load_v<ROWS>(acc));
  aie::vector<bf16, ROWS> v = bf16_limb::to_bf16(c);
  aie::store_v(y + 16, v.template extract<16>(0));
  aie::store_v(y + 32, v.template extract<16>(1));
}

} // extern "C"
