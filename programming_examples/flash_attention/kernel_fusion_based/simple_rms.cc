// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// simple_rms_bf16: y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
// Standard RMSNorm over a vector of N bf16 elements with bf16 weights.
// Used by 5-all V2 decode_block test.

#include <aie_api/aie.hpp>
#include <stdint.h>

// Quake-style fast inverse square root. AIE2P has no libm so we
// can't pull in sqrtf. Two Newton-Raphson iterations from the
// classic bit-magic seed give ~16 bits of mantissa precision —
// well beyond what bf16 needs.
static inline float fast_rsqrt(float x) {
  union {
    float f;
    int32_t i;
  } u;
  u.f = x;
  u.i = 0x5f3759df - (u.i >> 1);
  float y = u.f;
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

template <int N_VAL>
static inline void rms_kernel_bf16(const bfloat16 *restrict x,
                                   const bfloat16 *restrict w,
                                   bfloat16 *restrict y, float eps) {
  constexpr int VL = 16;
  static_assert(N_VAL % VL == 0, "N must be a multiple of vector width");
  constexpr int N_VECS = N_VAL / VL;

  // Pass 1: accumulate sum of squares.
  ::aie::accum<accfloat, VL> ssq = ::aie::zeros<accfloat, VL>();
  ::aie::vector<bfloat16, VL> x_vec[N_VECS];
  for (int i = 0; i < N_VECS; i++) {
    x_vec[i] = ::aie::load_v<VL>(x + i * VL);
    ssq = ::aie::mac(ssq, x_vec[i], x_vec[i]);
  }
  ::aie::vector<float, VL> ssq_v = ssq.template to_vector<float>();
  float total = 0.0f;
  for (int i = 0; i < VL; i++)
    total += ssq_v[i];
  float rstd = fast_rsqrt(total / (float)N_VAL + eps);

  // Pass 2: y = x * rstd * w (scalar fallback — small N, perf not load-
  // bearing for the V2 plumbing test).
  for (int i = 0; i < N_VAL; i++) {
    float xi = (float)x[i];
    float wi = (float)w[i];
    y[i] = (bfloat16)(xi * rstd * wi);
  }
}

extern "C" {
// x: [N] bf16 input vector
// w: [N] bf16 RMSNorm weights
// y: [N] bf16 output vector
void simple_rms_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y, int32_t N) {
  event0();
  if (N == 64)
    rms_kernel_bf16<64>(x, w, y, 1e-5f);
  event1();
}
}
