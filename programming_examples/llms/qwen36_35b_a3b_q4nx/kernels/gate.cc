// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The mixer output's gate, transcribed from FastFlowLM's
// FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/kernels (gate.cc,
// delta_net_norm.cc): a full-attention layer's o times sigmoid(gate)
// (sigmoid through tanh), a Gated DeltaNet layer's per-head RMSNorm of o
// times SiLU(z) (SiLU through FLM's linear-approximation table).

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"
#include "silu_lut.h"

using bf16 = bfloat16;

namespace {

constexpr int HD = 128; // a Gated DeltaNet value head
// value heads: Qwen3.6-35B-A3B's 32 by default, Qwen3.8-27B builds with
// -DGDN_NV=48
#ifndef GDN_NV
#define GDN_NV 32
#endif

aie::vector<bf16, 16> sigmoid(aie::vector<bf16, 16> x) {
  const aie::vector<bf16, 16> half = aie::broadcast<bf16, 16>(0.5f);
  aie::accum<accfloat, 16> xh = aie::mul(x, half);
  aie::vector<bf16, 16> t = aie::tanh<bf16>(xh.template to_vector<float>());
  t = aie::add(t, aie::broadcast<bf16, 16>(1.0f));
  return aie::mul(t, half);
}

aie::vector<bf16, 16> silu(aie::vector<bf16, 16> x) {
  using lut_type = aie::lut<4, float, bf16>;
  lut_type lut(64, (bf16 *)silu_lut_ab, (bf16 *)silu_lut_cd);
  aie::linear_approx<bf16, lut_type> approx(lut, -2, 32, 0);
  return approx.compute(x).template to_vector<bf16>();
}

} // namespace

extern "C" {

// y[0:n] = sigmoid(gate) * o
void gate_sigmoid(bf16 *__restrict y, bf16 *__restrict gate, bf16 *__restrict o,
                  int n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(y + i, aie::mul(sigmoid(aie::load_v<16>(gate + i)),
                                 aie::load_v<16>(o + i))
                            .template to_vector<bf16>());
}

// y[0:n] = SiLU(z) * o
void gate_silu(bf16 *__restrict y, bf16 *__restrict z, bf16 *__restrict o,
               int n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(y + i,
                 aie::mul(silu(aie::load_v<16>(z + i)), aie::load_v<16>(o + i))
                     .template to_vector<bf16>());
}

// y = RMSNorm(o) * w per 128-wide head, n values
void dn_norm(bf16 *__restrict y, bf16 *__restrict o, bf16 *__restrict w,
             int n) {
  constexpr int V = 16;
  for (int h = 0; h < n; h += HD) {
    aie::accum<accfloat, V> s0 = aie::zeros<accfloat, V>();
    aie::accum<accfloat, V> s1 = aie::zeros<accfloat, V>();
    for (int i = 0; i < HD; i += 2 * V) {
      s0 = aie::mac_square(s0, aie::load_v<V>(o + h + i));
      s1 = aie::mac_square(s1, aie::load_v<V>(o + h + i + V));
    }
    float sum = aie::reduce_add(aie::add(s0.template to_vector<float>(),
                                         s1.template to_vector<float>()));
    sum = sum * (1.0f / HD) + 1e-6f;
    float inv = aie::invsqrt(aie::broadcast<float, V>(sum))[0];
    for (int i = 0; i < HD; i += V) {
      aie::vector<float, V> wx =
          aie::mul(aie::load_v<V>(o + h + i), aie::load_v<V>(w + i));
      aie::store_v(y + h + i, aie::mul(wx, inv).template to_vector<bf16>());
    }
  }
}

// dn_norm of the step cores' outputs as they arrive, o = [4 cores][NH heads]
// [32 values], into head order y = [NH heads][128].
void dn_norm_rhv(bf16 *__restrict y, bf16 *__restrict o, bf16 *__restrict w) {
  constexpr int V = 16, NH = GDN_NV, VC = 32, NC = HD / VC;
  for (int h = 0; h < NH; h++) {
    aie::accum<accfloat, V> s = aie::zeros<accfloat, V>();
    for (int r = 0; r < NC; r++)
      for (int i = 0; i < VC; i += V)
        s = aie::mac_square(s, aie::load_v<V>(o + r * NH * VC + h * VC + i));
    float sum = aie::reduce_add(s.template to_vector<float>());
    sum = sum * (1.0f / HD) + 1e-6f;
    float inv = aie::invsqrt(aie::broadcast<float, V>(sum))[0];
    for (int r = 0; r < NC; r++)
      for (int i = 0; i < VC; i += V) {
        aie::vector<float, V> wx =
            aie::mul(aie::load_v<V>(o + r * NH * VC + h * VC + i),
                     aie::load_v<V>(w + r * VC + i));
        aie::store_v(y + h * HD + r * VC + i,
                     aie::mul(wx, inv).template to_vector<bf16>());
      }
  }
}

// z[512 j ..] = round j of the z projection
void z_round(bf16 *__restrict z, bf16 *__restrict rnd, int j) {
  for (int i = 0; i < 512; i += 16)
    aie::store_v(z + 512 * j + i, aie::load_v<16>(rnd + i));
}

} // extern "C"
