// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The LM head's core kernels, as one object: FastFlowLM's final RMS norm
// (FLM_Xclbin/Qwen3_5/lm_head_npu_bin/kernel/qmm.cc), and the Q8_0 block GEMV.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"
#include "bf16_limb.hpp"

namespace q36_q8 {
#include "q8_mvm.cc"
} // namespace q36_q8

using bf16 = bfloat16;

// Peano's scalar float ops, as FLM spells them
#define LM_FADD(a, b) __addsf3((a), (b))
#define LM_FSUB(a, b) __subsf3((a), (b))
#define LM_FMUL(a, b) __mulsf3((a), (b))
extern "C" float __addsf3(float, float);
extern "C" float __subsf3(float, float);
extern "C" float __mulsf3(float, float);

// hidden size: Qwen3.6-35B-A3B's 2048 by default, Qwen3.8-27B builds with
// -DLM_DIM=5120
#ifndef LM_DIM
#define LM_DIM 2048
#endif

namespace {
constexpr int MODEL_DIM = LM_DIM;
} // namespace

extern "C" {

// xw = [x | w]: x <- x / rms(x) * w, in place.
void lm_rms_norm(bf16 *__restrict xw) {
  bf16 *x = xw, *w = xw + MODEL_DIM;
  constexpr int V = 16, F = MODEL_DIM / V;
  constexpr float one_over_D = 1.0f / (float)MODEL_DIM;
  aie::accum<accfloat, V> sq = aie::zeros<accfloat, V>();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(F, F)
  for (int i = 0; i < F; i++)
    sq = aie::mac_square(sq, aie::load_v<V>(x + V * i));
  float sum = aie::reduce_add(sq.template to_vector<float>());
  sum = LM_FMUL(sum, one_over_D);
  sum = LM_FADD(sum, 1e-6f);
  const float threehalfs = 1.5f;
  float x2 = LM_FMUL(sum, 0.5f);
  float r = sum;
  uint32_t u = *(uint32_t *)&r;
  u = 0x5f3759df - (u >> 1);
  r = *(float *)&u;
  r = LM_FMUL(r, LM_FSUB(threehalfs, LM_FMUL(LM_FMUL(x2, r), r)));
  r = LM_FMUL(r, LM_FSUB(threehalfs, LM_FMUL(LM_FMUL(x2, r), r)));
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(F, F)
  for (int i = 0; i < F; i++) {
    aie::vector<float, V> wx =
        aie::mul(aie::load_v<V>(x + V * i), aie::load_v<V>(w + V * i));
    aie::vector<bf16, V> o = aie::mul(wx, r);
    aie::store_v(x + V * i, o);
  }
}

} // extern "C"
