// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The residual stream and the MoE combine, transcribed from FastFlowLM's
// FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/kernels (rms_residual.cc, glu.cc,
// nonLinearFunction.h):
//
//   rms_norm      y = RMSNorm(x) * w (the weight holds the model's 1 + w)
//   residual_add  xbuf += x, y = xbuf
//   shared_gate   s *= sigmoid(h . w), the dot product rounded to bf16
//   glu_silu      y = SiLU(gate) * up, SiLU as x sigmoid(x), sigmoid via tanh
//   score_acc     acc += score * x, float32 accumulation
//   moe_out       y = s + bf16(acc): the gated shared expert plus the
//                 routed sum
//
// Every vector that becomes bf16 is 16 lanes wide: Peano's 32-lane
// conversion returns the upper half twice.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"

using bf16 = bfloat16;

namespace {

constexpr int V = 16;

aie::vector<bf16, V> sigmoid(aie::vector<bf16, V> x) {
  const aie::vector<bf16, V> half = aie::broadcast<bf16, V>(0.5f);
  aie::accum<accfloat, V> xh = aie::mul(x, half);
  aie::vector<bf16, V> t = aie::tanh<bf16>(xh.template to_vector<float>());
  t = aie::add(t, aie::broadcast<bf16, V>(1.0f));
  return aie::mul(t, half).template to_vector<bf16>();
}

} // namespace

extern "C" {

void rms_norm(bf16 *__restrict y, bf16 *__restrict x, bf16 *__restrict w,
              int n) {
  aie::accum<accfloat, V> s0 = aie::zeros<accfloat, V>();
  aie::accum<accfloat, V> s1 = aie::zeros<accfloat, V>();
  for (int i = 0; i < n; i += 2 * V) {
    s0 = aie::mac_square(s0, aie::load_v<V>(x + i));
    s1 = aie::mac_square(s1, aie::load_v<V>(x + i + V));
  }
  float sum = aie::reduce_add(
      aie::add(s0.template to_vector<float>(), s1.template to_vector<float>()));
  sum = sum / (float)n + 1e-6f;
  float inv = aie::invsqrt(aie::broadcast<float, V>(sum))[0];
  for (int i = 0; i < n; i += V) {
    aie::vector<float, V> wx =
        aie::mul(aie::load_v<V>(x + i), aie::load_v<V>(w + i));
    aie::store_v(y + i, aie::mul(wx, inv).template to_vector<bf16>());
  }
}

void residual_add(bf16 *__restrict y, bf16 *__restrict xbuf, bf16 *__restrict x,
                  int n) {
  for (int i = 0; i < n; i += V) {
    aie::vector<bf16, V> s =
        aie::add(aie::load_v<V>(xbuf + i), aie::load_v<V>(x + i));
    aie::store_v(xbuf + i, s);
    aie::store_v(y + i, s);
  }
}

void shared_gate(bf16 *__restrict h, bf16 *__restrict w, bf16 *__restrict s,
                 int n) {
  aie::accum<accfloat, V> acc = aie::zeros<accfloat, V>();
  for (int i = 0; i < n; i += V)
    acc = aie::mac(acc, aie::load_v<V>(h + i), aie::load_v<V>(w + i));
  bf16 g = (bf16)aie::reduce_add(acc.template to_vector<float>());
  aie::vector<bf16, V> sg = sigmoid(aie::broadcast<bf16, V>(g));
  for (int i = 0; i < n; i += V)
    aie::store_v(
        s + i, aie::mul(aie::load_v<V>(s + i), sg).template to_vector<bf16>());
}

void glu_silu(bf16 *__restrict y, bf16 *__restrict gate, bf16 *__restrict up,
              int n) {
  for (int i = 0; i < n; i += V) {
    aie::vector<bf16, V> g = aie::load_v<V>(gate + i);
    aie::vector<bf16, V> act =
        aie::mul(g, sigmoid(g)).template to_vector<bf16>();
    aie::store_v(
        y + i,
        aie::mul(act, aie::load_v<V>(up + i)).template to_vector<bf16>());
  }
}

// acc[0:n] += score[k] * x[0:n]; x is bf16-exact, so the score meets it as
// two bf16 limbs on the native MAC (AIE2P has no float32 vector multiply)
void score_acc(float *__restrict acc, bf16 *__restrict x,
               float *__restrict score, int k, int n) {
  const float sc = score[k];
  const bf16 hi = (bf16)sc;
  const bf16 lo = (bf16)(sc - (float)hi);
  for (int i = 0; i < n; i += V) {
    aie::accum<accfloat, V> a;
    a.from_vector(aie::load_v<V>(acc + i));
    aie::vector<bf16, V> xv = aie::load_v<V>(x + i);
    a = aie::mac(a, xv, lo);
    a = aie::mac(a, xv, hi);
    aie::store_v(acc + i, a.template to_vector<float>());
  }
}

void acc_zero(float *__restrict acc, int n) {
  for (int i = 0; i < n; i += V)
    aie::store_v(acc + i, aie::zeros<float, V>());
}

// shared_gate in two steps, so h can be sent on before s arrives:
// g[0] = h . w, then s *= sigmoid(g[0])
void shared_gate_dot(bf16 *__restrict h, bf16 *__restrict w, bf16 *__restrict g,
                     int n) {
  aie::accum<accfloat, V> acc = aie::zeros<accfloat, V>();
  for (int i = 0; i < n; i += V)
    acc = aie::mac(acc, aie::load_v<V>(h + i), aie::load_v<V>(w + i));
  g[0] = (bf16)aie::reduce_add(acc.template to_vector<float>());
}

void shared_gate_apply(bf16 *__restrict g, bf16 *__restrict s, int n) {
  aie::vector<bf16, V> sg = sigmoid(aie::broadcast<bf16, V>(g[0]));
  for (int i = 0; i < n; i += V)
    aie::store_v(
        s + i, aie::mul(aie::load_v<V>(s + i), sg).template to_vector<bf16>());
}

// y = bf16(acc): a model with no shared expert
void moe_out_routed(bf16 *__restrict y, float *__restrict acc, int n) {
  for (int i = 0; i < n; i += V) {
    aie::accum<accfloat, V> a;
    a.from_vector(aie::load_v<V>(acc + i));
    aie::store_v(y + i, a.template to_vector<bf16>());
  }
}

void moe_out(bf16 *__restrict y, bf16 *__restrict s, float *__restrict acc,
             int n) {
  for (int i = 0; i < n; i += V) {
    aie::accum<accfloat, V> a;
    a.from_vector(aie::load_v<V>(acc + i));
    aie::store_v(y + i,
                 aie::add(aie::load_v<V>(s + i), a.template to_vector<bf16>()));
  }
}

} // extern "C"
