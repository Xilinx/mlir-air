// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Elementary functions for the Gated DeltaNet kernels (gdn_step.cc,
// gdn_mix.cc). aie::exp2<bfloat16> on AIE2P is a coarse approximation (a few
// % off), so they do not use it. gm::*_v: FP32 on 16-lane float vectors, ~1e-7
// relative (exp: range reduction + degree-7 polynomial + exponent bits; 1 / d:
// bit-trick seed + three Newton steps); FP32 multiplies are emulated on AIE2P.
// gm::bf: the faster bf16-MAC sigmoid / SiLU, ~1e-3 relative.

#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace gm {

using VF = aie::vector<float, 16>;
using VI = aie::vector<int32_t, 16>;

static inline VF vmul(VF a, VF b) { return aie::mul(a, b).to_vector<float>(); }
static inline VF vmuls(VF a, float b) {
  return aie::mul(a, b).to_vector<float>();
}
static inline VF vadds(VF a, float b) {
  return aie::add(a, aie::broadcast<float, 16>(b));
}
static inline VF vbc(float b) { return aie::broadcast<float, 16>(b); }

static inline VF exp_v(VF x) {
  x = aie::max(aie::min(x, vbc(88.0f)), vbc(-87.0f));
  VF y = vmuls(x, 1.4426950408889634f);
  VI n = aie::to_fixed<int32_t>(y, 0); // round to nearest
  VF f = vmuls(aie::sub(y, aie::to_float<float>(n, 0)), 0.6931471805599453f);
  VF p = vadds(vmuls(f, 1.0f / 5040), 1.0f / 720);
  p = vadds(vmul(p, f), 1.0f / 120);
  p = vadds(vmul(p, f), 1.0f / 24);
  p = vadds(vmul(p, f), 1.0f / 6);
  p = vadds(vmul(p, f), 0.5f);
  p = vadds(vmul(p, f), 1.0f);
  p = vadds(vmul(p, f), 1.0f);
  VI e = aie::upshift(aie::add(n, aie::broadcast<int32_t, 16>(127)), 23);
  return vmul(p, aie::vector_cast<float>(e));
}

// 1 / d for d > 0 (normal): bit-trick seed, three Newton steps
static inline VF inv_v(VF d) {
  VI di = aie::vector_cast<int32_t>(d);
  VF r = aie::vector_cast<float>(
      aie::sub(aie::broadcast<int32_t, 16>(0x7EF311C3), di));
  for (int i = 0; i < 3; i++)
    r = vmul(r, aie::sub(vbc(2.0f), vmul(d, r)));
  return r;
}

static inline VF log1p_v(VF u) { // u in [0, 1]
  VF s = vmul(u, inv_v(vadds(u, 2.0f)));
  VF s2 = vmul(s, s);
  VF p = vadds(vmuls(s2, 1.0f / 15), 1.0f / 13);
  p = vadds(vmul(p, s2), 1.0f / 11);
  p = vadds(vmul(p, s2), 1.0f / 9);
  p = vadds(vmul(p, s2), 1.0f / 7);
  p = vadds(vmul(p, s2), 1.0f / 5);
  p = vadds(vmul(p, s2), 1.0f / 3);
  p = vadds(vmul(p, s2), 1.0f);
  return vmuls(vmul(s, p), 2.0f);
}

// sigmoid(x) = 1 / (1 + e) for x >= 0, e / (1 + e) for x < 0, e = exp(-|x|)
static inline VF sigmoid_v(VF x) {
  VF zero = vbc(0.0f);
  VF e = exp_v(aie::min(x, aie::sub(zero, x)));
  VF r = inv_v(vadds(e, 1.0f));
  return aie::select(r, vmul(e, r), aie::lt(x, zero));
}

static inline VF to_f(aie::vector<bfloat16, 16> v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.to_vector<float>();
}

// bf16(x * sigmoid(x)), x bf16: torch's silu on a bf16 tensor
static inline aie::vector<bfloat16, 16> silu_bf16(aie::vector<bfloat16, 16> x) {
  VF f = to_f(x);
  return aie::mul(f, sigmoid_v(f)).to_vector<bfloat16>();
}

// ---- bf16-MAC sigmoid / SiLU: t = 2^(-|x| log2 e) from round-to-nearest range
// reduction in the FP32 accumulator, a degree-4 polynomial on bf16 MACs and
// exponent bits; r = 1 / (1 + t) by two Newton steps; ~1e-3 relative ----
namespace bf {

constexpr int VL = 16;
using V = aie::vector<bfloat16, VL>;
using A = aie::accum<accfloat, VL>;

static inline A acc_of(float c) {
  A a;
  a.from_vector(aie::broadcast<float, VL>(c));
  return a;
}

// 2^y for y <= 0 held in an FP32 accumulator -> bf16 (relative error ~1e-3)
static inline V exp2_neg(A y) {
  const float magic = 12582912.0f; // 1.5 * 2^23
  VF yv = aie::max(y.to_vector<float>(), aie::broadcast<float, VL>(-126.0f));
  A ya;
  ya.from_vector(yv);
  A sa = aie::add(ya, aie::broadcast<float, VL>(magic));
  VF sv = sa.to_vector<float>();                         // magic + round(y)
  A na = aie::sub(sa, aie::broadcast<float, VL>(magic)); // round(y), exact
  A fa = aie::sub(ya, na.to_vector<float>()); // f in [-1/2, 1/2], exact
  V f = fa.to_vector<bfloat16>();
  aie::vector<int32, VL> nb = aie::vector_cast<int32>(sv);
  nb = aie::sub(nb, aie::broadcast<int32, VL>(0x4B400000 - 127));
  VF pn = aie::vector_cast<float>(aie::upshift(nb, 23));
  V p = aie::broadcast<bfloat16, VL>((bfloat16)0.0096181291f);
  p = aie::mac(acc_of(0.0555041087f), f, p).to_vector<bfloat16>();
  p = aie::mac(acc_of(0.2402265070f), f, p).to_vector<bfloat16>();
  p = aie::mac(acc_of(0.6931471806f), f, p).to_vector<bfloat16>();
  p = aie::mac(acc_of(1.0f), f, p).to_vector<bfloat16>();
  A pa;
  pa.from_vector(pn);
  return aie::mul(p, pa.to_vector<bfloat16>()).to_vector<bfloat16>();
}

static inline __attribute__((always_inline)) V sigmoid(V x) {
  const bfloat16 l2e_hi = (bfloat16)1.4426950408889634f;
  const bfloat16 l2e_lo = (bfloat16)(1.4426950408889634f - (float)l2e_hi);
  V ax = aie::abs(x);
  A m = aie::mul(ax, (bfloat16)(-(float)l2e_hi));
  m = aie::msc(m, ax, l2e_lo);
  V t = exp2_neg(m);
  A dacc = acc_of(1.0f);
  dacc = aie::add(dacc, t);
  V d = dacc.to_vector<bfloat16>();
  A r0a = aie::msc(acc_of(24.0f / 17.0f), d, (bfloat16)(8.0f / 17.0f));
  V r = r0a.to_vector<bfloat16>();
  for (int i = 0; i < 2; i++) {
    V e = aie::msc(acc_of(1.0f), d, r).to_vector<bfloat16>();
    A ra;
    ra.from_vector(r);
    r = aie::mac(ra, r, e).to_vector<bfloat16>();
  }
  V tr = aie::mul(t, r).to_vector<bfloat16>();
  return aie::select(r, tr, aie::lt(x, aie::zeros<bfloat16, VL>()));
}

// bf16(x * sigmoid(x))
static inline __attribute__((always_inline)) V silu(V x) {
  return aie::mul(x, sigmoid(x)).to_vector<bfloat16>();
}

} // namespace bf

} // namespace gm
