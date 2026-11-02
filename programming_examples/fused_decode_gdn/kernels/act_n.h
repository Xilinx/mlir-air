// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// FP32-accurate exp2 / sigmoid / SiLU on N-lane vectors. The core's
// FP32-accumulator MACs are 32 lanes wide, so the 32-lane form does twice the
// 16-lane work per instruction; the GLU and the Gated DeltaNet conv use it.

#pragma once
#include <aie_api/aie.hpp>

namespace actn {

template <int N>
struct L {
  using V = aie::vector<bfloat16, N>;
  using A = aie::accum<accfloat, N>;
  using VF = aie::vector<float, N>;
  struct S2 {
    V h, l;
  };

  static inline A acc_of(float c) {
    A a;
    a.from_vector(aie::broadcast<float, N>(c));
    return a;
  }

  static inline A acc_of(VF v) {
    A a;
    a.from_vector(v);
    return a;
  }

  static inline S2 split(A a) {
    S2 s;
    s.h = a.template to_vector<bfloat16>();
    s.l = aie::msc(a, s.h, (bfloat16)1.0f).template to_vector<bfloat16>();
    return s;
  }

  static inline A mac2(A acc, S2 a, S2 b) {
    acc = aie::mac(acc, a.h, b.h);
    acc = aie::mac(acc, a.h, b.l);
    return aie::mac(acc, a.l, b.h);
  }

  // 2^y, y <= 127 (below -126 it flushes to 0)
  static inline A exp2_acc(A y) {
    const float magic = 12582912.0f; // 1.5 * 2^23
    VF yv = aie::max(y.template to_vector<float>(),
                     aie::broadcast<float, N>(-126.0f));
    A ya = acc_of(yv);
    A sa = aie::add(ya, aie::broadcast<float, N>(magic));
    VF sv = sa.template to_vector<float>();               // magic + round(y)
    A na = aie::sub(sa, aie::broadcast<float, N>(magic)); // round(y), exact
    S2 f =
        split(aie::sub(ya, na.template to_vector<float>())); // f in [-1/2, 1/2]
    aie::vector<int32, N> nb = aie::vector_cast<int32>(sv);
    nb = aie::sub(nb, aie::broadcast<int32, N>(0x4B400000 - 127));
    V pn = acc_of(aie::vector_cast<float>(aie::upshift(nb, 23)))
               .template to_vector<bfloat16>();
    const float c[6] = {1.0f,          0.6931471806f, 0.2402265070f,
                        0.0555041087f, 0.0096181291f, 0.0013333558f};
    A p = acc_of(c[5]);
    for (int k = 4; k >= 0; k--)
      p = mac2(acc_of(c[k]), split(p), f);
    S2 ps = split(p);
    A t = aie::mul(ps.h, pn); // exact: pn is a power of two
    return aie::mac(t, ps.l, pn);
  }

  static inline A sigmoid_acc(V x) {
    const bfloat16 l2e_hi = (bfloat16)1.4426950408889634f;
    const bfloat16 l2e_lo = (bfloat16)(1.4426950408889634f - (float)l2e_hi);
    V ax = aie::abs(x);
    A m = aie::mul(ax, (bfloat16)(-(float)l2e_hi));
    m = aie::msc(m, ax, l2e_lo);
    A t = exp2_acc(m);
    S2 ts = split(t);
    S2 d = split(aie::add(acc_of(1.0f), t.template to_vector<float>()));
    S2 r =
        split(aie::msc(acc_of(24.0f / 17.0f), d.h, (bfloat16)(8.0f / 17.0f)));
    A ra;
    for (int i = 0; i < 2; i++) {
      A e = acc_of(1.0f);
      e = aie::msc(e, d.h, r.h);
      e = aie::msc(e, d.h, r.l);
      e = aie::msc(e, d.l, r.h);
      S2 es = split(e);
      ra.from_vector(r.h);
      ra = aie::add(ra, r.l);
      ra = mac2(ra, r, es);
      r = split(ra);
    }
    A tr = mac2(aie::zeros<accfloat, N>(), ts, r);
    VF neg = tr.template to_vector<float>(),
       pos = ra.template to_vector<float>();
    return acc_of(aie::select(pos, neg, aie::lt(x, aie::zeros<bfloat16, N>())));
  }

  // bf16(x * sigmoid(x)), one rounding
  static inline V silu(V x) {
    S2 s = split(sigmoid_acc(x));
    A y = aie::mul(x, s.h);
    return aie::mac(y, x, s.l).template to_vector<bfloat16>();
  }

  // split(sigmoid_acc(x)) of K independent vectors: the same arithmetic,
  // step for step, with the K chains' statements in lockstep (constant trip
  // loops, unrolled) so their latencies overlap: one sigmoid_acc() is a
  // single dependent chain of ~70 vector operations. (The results are bf16
  // pairs: arrays of accumulators passed through memory miscompute.)
  template <int K>
  static inline void sigmoid_k(const V *x, S2 *out) {
    const float magic = 12582912.0f; // 1.5 * 2^23
    const bfloat16 l2e_hi = (bfloat16)1.4426950408889634f;
    const bfloat16 l2e_lo = (bfloat16)(1.4426950408889634f - (float)l2e_hi);
    const float c[6] = {1.0f,          0.6931471806f, 0.2402265070f,
                        0.0555041087f, 0.0096181291f, 0.0013333558f};
    V ax[K];
    A y[K], t[K], p[K];
    S2 f[K];
    V pn[K];
#pragma unroll
    for (int j = 0; j < K; j++) {
      ax[j] = aie::abs(x[j]);
      A m = aie::mul(ax[j], (bfloat16)(-(float)l2e_hi));
      y[j] = aie::msc(m, ax[j], l2e_lo);
    }
    // exp2_acc
#pragma unroll
    for (int j = 0; j < K; j++) {
      VF yv = aie::max(y[j].template to_vector<float>(),
                       aie::broadcast<float, N>(-126.0f));
      A ya = acc_of(yv);
      A sa = aie::add(ya, aie::broadcast<float, N>(magic));
      VF sv = sa.template to_vector<float>();
      A na = aie::sub(sa, aie::broadcast<float, N>(magic));
      f[j] = split(aie::sub(ya, na.template to_vector<float>()));
      aie::vector<int32, N> nb = aie::vector_cast<int32>(sv);
      nb = aie::sub(nb, aie::broadcast<int32, N>(0x4B400000 - 127));
      pn[j] = acc_of(aie::vector_cast<float>(aie::upshift(nb, 23)))
                  .template to_vector<bfloat16>();
      p[j] = acc_of(c[5]);
    }
    for (int k = 4; k >= 0; k--) {
#pragma unroll
      for (int j = 0; j < K; j++)
        p[j] = mac2(acc_of(c[k]), split(p[j]), f[j]);
    }
#pragma unroll
    for (int j = 0; j < K; j++) {
      S2 ps = split(p[j]);
      A tt = aie::mul(ps.h, pn[j]);
      t[j] = aie::mac(tt, ps.l, pn[j]);
    }
    // sigmoid_acc
    S2 ts[K], d[K], r[K];
    A ra[K];
#pragma unroll
    for (int j = 0; j < K; j++) {
      ts[j] = split(t[j]);
      d[j] = split(aie::add(acc_of(1.0f), t[j].template to_vector<float>()));
      r[j] = split(
          aie::msc(acc_of(24.0f / 17.0f), d[j].h, (bfloat16)(8.0f / 17.0f)));
    }
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < K; j++) {
        A e = acc_of(1.0f);
        e = aie::msc(e, d[j].h, r[j].h);
        e = aie::msc(e, d[j].h, r[j].l);
        e = aie::msc(e, d[j].l, r[j].h);
        S2 es = split(e);
        ra[j].from_vector(r[j].h);
        ra[j] = aie::add(ra[j], r[j].l);
        ra[j] = mac2(ra[j], r[j], es);
        r[j] = split(ra[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < K; j++) {
      A tr = mac2(aie::zeros<accfloat, N>(), ts[j], r[j]);
      VF neg = tr.template to_vector<float>(),
         pos = ra[j].template to_vector<float>();
      out[j] = split(acc_of(
          aie::select(pos, neg, aie::lt(x[j], aie::zeros<bfloat16, N>()))));
    }
  }

  // silu of K independent vectors in place (silu(), K chains in lockstep)
  template <int K>
  static inline void silu_k(V *x) {
    S2 s[K];
    sigmoid_k<K>(x, s);
#pragma unroll
    for (int j = 0; j < K; j++) {
      const S2 &ss = s[j];
      A o = aie::mul(x[j], ss.h);
      x[j] = aie::mac(o, x[j], ss.l).template to_vector<bfloat16>();
    }
  }
};

} // namespace actn
