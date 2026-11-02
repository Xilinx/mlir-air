// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fused_decode_gdn's Gated DeltaNet herd (column 3, 4 cores): the delta-rule
// step of one token, one call per 4096-word input chunk, one 4096-word output
// chunk per call. A token's output stream is the next token's input stream
// (the host swaps the two buffers), so every chunk is written back in full:
// the X chunk with its history shifted, a state unit updated, a magic chunk
// unchanged.
//
// Per Gated DeltaNet layer a core runs its 4 heads in turn; per head an X
// chunk and the head's recurrent state in 8 units of 16 value columns. The
// call on the head's last unit also writes the head's gated output to go.
//
//   X   [(unused: the head's q k v | z | b a come in gx) | hist (3 x 384, the
//       raw lin rows of the 3 previous tokens) | taps (4 x 384, oldest first) |
//       -exp(A_log), dt_bias (f32) | . | norm weight (128)]
//       out: the same chunk with hist' (rows t-2, t-1, t) in place of hist
//   U   one state unit [hi | lo][64 k pairs][16 v][2]: the FP32 state as bf16
//       hi + lo, rows interleaved by pairs
//       out: the updated unit
// A chunk with the magic tail (the wave's first chunk, which resets the
// schedule, or a NOP that pads the wave to an even chunk count for the
// double-buffered stream) is copied through.
//
// Math: causal conv (FP32 accumulation, bf16 out) + SiLU (act_n.h); beta =
// bf16(sigmoid(b)) and decay = exp(-exp(A_log) softplus(a + dt_bias)), both
// from the mixer tile (gdn_mix.cc); l2-normalised q (times DK^-0.5) and k as
// bf16; the delta rule's dot products on the hi halves only. The gated
// RMSNorm: bf16(bf16(w bf16(o / rms(o))) * silu(z)).

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "act_n.h"
#include "aie_kernel_utils.h"
#include "gdn_math.h"

namespace {

constexpr int DK = 128, DV = 128, U = 16, NU = DV / U, HPC = 4;
constexpr int NCH = 3 * DK;
constexpr int OQKV = 0, OZ = 384, OBA = 512, OHIST = 544, OTAPS = 1696,
              OPRM = 3232, ONW = 3264;
constexpr int PER_HEAD = 1 + NU, PER_LAYER = HPC * PER_HEAD;
constexpr int GX_ROWS =
    544; // a head's mixer rows: q k v | z | b a (hi, lo) | pad
constexpr int OPRM2 =
    3236; // the core's first X chunk: its 4 heads' (-exp(A_log), dt_bias)
constexpr int VL = 16;

// ctx (bf16 [1024]): 0 qn, 128 kn, 256 v, 384 z, 512 norm w, 640 o, 768 gated o
// fs (f32 [16]): 0 beta, 1 decay

// 1 / sqrt(x) on a float vector (the bit-trick seed, three Newton steps): the
// cores emulate scalar FP32 multiplies in library calls, vector ones inline
static inline gm::VF rsqrt_v(gm::VF x) {
  gm::VI xi = aie::vector_cast<int32_t>(x);
  gm::VF y = aie::vector_cast<float>(
      aie::sub(aie::broadcast<int32_t, 16>(0x5f3759df), aie::downshift(xi, 1)));
  const gm::VF hx = gm::vmuls(x, 0.5f);
  for (int i = 0; i < 3; i++)
    y = gm::vmul(y, aie::sub(gm::vbc(1.5f), gm::vmul(hx, gm::vmul(y, y))));
  return y;
}

static inline float rsqrt_f(float x) {
  union {
    float f;
    uint32_t u;
  } c{x};
  c.u = 0x5f3759dfu - (c.u >> 1);
  float y = c.f;
  for (int i = 0; i < 3; i++)
    y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

static inline float ssq_bf16(const bfloat16 *x) {
  aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
  for (int i = 0; i < DK; i += VL) {
    aie::vector<bfloat16, VL> v = aie::load_v<VL>(x + i);
    acc = aie::mac(acc, v, v);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

// x * s on bf16 MACs (s split hi + lo; FP32 multiplies are emulated)
static inline void scale_to_bf16(const bfloat16 *x, float s, bfloat16 *dst) {
  const bfloat16 sh = (bfloat16)s, sl = (bfloat16)(s - (float)sh);
  for (int i = 0; i < DK; i += 32) {
    aie::vector<bfloat16, 32> v = aie::load_v<32>(x + i);
    aie::accum<accfloat, 32> a = aie::mul(v, sh);
    a = aie::mac(a, v, sl);
    aie::store_v(dst + i, a.to_vector<bfloat16>());
  }
}

static inline aie::vector<bfloat16, 16> tobf(gm::VF f) {
  aie::accum<accfloat, 16> a;
  a.from_vector(f);
  return a.template to_vector<bfloat16>();
}

__attribute__((noinline)) static void x_chunk(const bfloat16 *in,
                                              const bfloat16 *gx, bfloat16 *ctx,
                                              float *fs, bfloat16 *out, int j) {
  using namespace gm;
  // the chunk itself, then the new history (rows t-2, t-1 and this token's
  // raw q | k | v) over the old one
  for (int i = 0; i < 4096; i += 32)
    aie::store_v(out + i, aie::load_v<32>(in + i));
  for (int i = 0; i < 2 * NCH; i += 32)
    aie::store_v(out + OHIST + i, aie::load_v<32>(in + OHIST + NCH + i));
  for (int i = 0; i < NCH; i += 32)
    aie::store_v(out + OHIST + 2 * NCH + i, aie::load_v<32>(gx + OQKV + i));
  // conv (bf16 out) + SiLU, 32 channels at a time (act_n.h's FP32-accurate
  // SiLU on the accumulator's 32-lane MACs)
  alignas(64) bfloat16 cv[NCH];
  for (int ch = 0; ch < NCH; ch += 32) {
    aie::accum<accfloat, 32> a = aie::mul(aie::load_v<32>(in + OHIST + ch),
                                          aie::load_v<32>(in + OTAPS + ch));
    a = aie::mac(a, aie::load_v<32>(in + OHIST + NCH + ch),
                 aie::load_v<32>(in + OTAPS + NCH + ch));
    a = aie::mac(a, aie::load_v<32>(in + OHIST + 2 * NCH + ch),
                 aie::load_v<32>(in + OTAPS + 2 * NCH + ch));
    a = aie::mac(a, aie::load_v<32>(gx + OQKV + ch),
                 aie::load_v<32>(in + OTAPS + 3 * NCH + ch));
    aie::store_v(cv + ch, actn::L<32>::silu(a.to_vector<bfloat16>()));
  }
  bfloat16 *qn = ctx, *kn = ctx + DK, *v = ctx + 2 * DK;
  // l2-norm factors of k and q (q also times DK^-0.5) on one float vector
  gm::VF ss = aie::select(gm::vbc(ssq_bf16(cv + DK)), gm::vbc(ssq_bf16(cv)),
                          aie::mask<16>::from_uint32(0xFFFEu));
  gm::VF rs = rsqrt_v(gm::vadds(ss, 1e-6f));
  rs = gm::vmul(rs, aie::select(gm::vbc(1.0f), gm::vbc(0.08838834764831845f),
                                aie::mask<16>::from_uint32(0xFFFEu)));
  scale_to_bf16(cv + DK, rs.get(0), kn);
  scale_to_bf16(cv, rs.get(1), qn);
  for (int i = 0; i < DV; i += 32) {
    aie::store_v(v + i, aie::load_v<32>(cv + 2 * DK + i));
    aie::store_v(ctx + 384 + i, aie::load_v<32>(gx + OZ + i));
    aie::store_v(ctx + 512 + i, aie::load_v<32>(in + ONW + i));
  }
  // beta and decay, f32, computed by the mixer tile in the rows of b and a
  const float *g = (const float *)(gx + OBA);
  fs[0] = g[0];
  fs[1] = g[1];
}

// a row pair's per-lane multipliers in the interleaved unit layout (lane 2c + e
// is column c of row r + e): (x_r, x_r+1) as one 32-bit word, broadcast
static inline aie::vector<bfloat16, 32> pair_bc(const bfloat16 *x, int r) {
  return aie::broadcast<uint32_t, 16>(*(const uint32_t *)(x + r))
      .template cast_to<bfloat16>();
}

// the state update of one unit, a row pair per iteration: S' = decay S + k u^T
// (S as bf16 hi + lo, the products in the FP32 accumulator), and q^T S'. Its
// own function with restrict parameters and an exact trip count, so the
// modulo scheduler overlaps the row pairs' dependent MAC chains.
__attribute__((noinline)) static aie::accum<accfloat, 32>
update(const bfloat16 *__restrict Sh, const bfloat16 *__restrict Sl,
       bfloat16 *__restrict Oh, bfloat16 *__restrict Ol,
       const bfloat16 *__restrict kn, const bfloat16 *__restrict qn,
       aie::vector<bfloat16, 32> uv2, bfloat16 dh, bfloat16 dl) {
  const bfloat16 one = (bfloat16)1.0f;
  aie::accum<accfloat, 32> oa = aie::zeros<accfloat, 32>(), ob = oa;
  // four independent row pairs per iteration, interleaved by hand (the bundle
  // scheduler overlaps them; the loop is not modulo scheduled)
  for (int i = 0; i < DK; i += 8) {
    aie::vector<bfloat16, 32> h0 = aie::load_v<32>(Sh + (i + 0) * U);
    aie::vector<bfloat16, 32> h1 = aie::load_v<32>(Sh + (i + 2) * U);
    aie::vector<bfloat16, 32> h2 = aie::load_v<32>(Sh + (i + 4) * U);
    aie::vector<bfloat16, 32> h3 = aie::load_v<32>(Sh + (i + 6) * U);
    aie::vector<bfloat16, 32> l0 = aie::load_v<32>(Sl + (i + 0) * U);
    aie::vector<bfloat16, 32> l1 = aie::load_v<32>(Sl + (i + 2) * U);
    aie::vector<bfloat16, 32> l2 = aie::load_v<32>(Sl + (i + 4) * U);
    aie::vector<bfloat16, 32> l3 = aie::load_v<32>(Sl + (i + 6) * U);
    aie::accum<accfloat, 32> a0 = aie::mul(h0, dh);
    aie::accum<accfloat, 32> a1 = aie::mul(h1, dh);
    aie::accum<accfloat, 32> a2 = aie::mul(h2, dh);
    aie::accum<accfloat, 32> a3 = aie::mul(h3, dh);
    a0 = aie::mac(a0, l0, dh);
    a1 = aie::mac(a1, l1, dh);
    a2 = aie::mac(a2, l2, dh);
    a3 = aie::mac(a3, l3, dh);
    a0 = aie::mac(a0, h0, dl);
    a1 = aie::mac(a1, h1, dl);
    a2 = aie::mac(a2, h2, dl);
    a3 = aie::mac(a3, h3, dl);
    a0 = aie::mac(a0, uv2, pair_bc(kn, i + 0));
    a1 = aie::mac(a1, uv2, pair_bc(kn, i + 2));
    a2 = aie::mac(a2, uv2, pair_bc(kn, i + 4));
    a3 = aie::mac(a3, uv2, pair_bc(kn, i + 6));
    aie::vector<bfloat16, 32> n0 = a0.to_vector<bfloat16>();
    aie::vector<bfloat16, 32> n1 = a1.to_vector<bfloat16>();
    aie::vector<bfloat16, 32> n2 = a2.to_vector<bfloat16>();
    aie::vector<bfloat16, 32> n3 = a3.to_vector<bfloat16>();
    aie::store_v(Oh + (i + 0) * U, n0);
    aie::store_v(Ol + (i + 0) * U, aie::msc(a0, n0, one).to_vector<bfloat16>());
    aie::store_v(Oh + (i + 2) * U, n1);
    aie::store_v(Ol + (i + 2) * U, aie::msc(a1, n1, one).to_vector<bfloat16>());
    aie::store_v(Oh + (i + 4) * U, n2);
    aie::store_v(Ol + (i + 4) * U, aie::msc(a2, n2, one).to_vector<bfloat16>());
    aie::store_v(Oh + (i + 6) * U, n3);
    aie::store_v(Ol + (i + 6) * U, aie::msc(a3, n3, one).to_vector<bfloat16>());
    oa = aie::mac(oa, n0, pair_bc(qn, i + 0));
    ob = aie::mac(ob, n1, pair_bc(qn, i + 2));
    oa = aie::mac(oa, n2, pair_bc(qn, i + 4));
    ob = aie::mac(ob, n3, pair_bc(qn, i + 6));
  }
  oa = aie::add(oa, ob.to_vector<float>());
  return oa;
}

__attribute__((noinline)) static void
unit(const bfloat16 *in, bfloat16 *ctx, const float *fs, bfloat16 *out, int u) {
  // Two state rows at a time on 32 lanes (the FP32 accumulator's MAC width):
  // a unit holds its 128 x 16 hi and lo halves as [64 row pairs][16 columns]
  // [2 rows], so a pair is one 32-lane vector whose multipliers are one 32-bit
  // broadcast; restrict: the state's loads and the updated state's stores do
  // not alias.
  const bfloat16 *__restrict Sh = in, *__restrict Sl = in + DK * U;
  bfloat16 *__restrict Oh = out, *__restrict Ol = out + DK * U;
  const bfloat16 *qn = ctx, *kn = ctx + DK, *v = ctx + 2 * DK;
  const float beta = fs[0], decay = fs[1];
  const bfloat16 one = (bfloat16)1.0f;
  // kn^T S (hi half): two accumulators of row pairs
  aie::accum<accfloat, 32> k0 = aie::zeros<accfloat, 32>(), k1 = k0;
  for (int i = 0; i < DK; i += 4) {
    k0 = aie::mac(k0, aie::load_v<32>(Sh + i * U), pair_bc(kn, i));
    k1 = aie::mac(k1, aie::load_v<32>(Sh + (i + 2) * U), pair_bc(kn, i + 2));
  }
  aie::vector<float, 32> kf =
      aie::add(k0, k1.to_vector<float>()).to_vector<float>();
  aie::accum<accfloat, U> kv;
  kv.from_vector(aie::add(aie::filter_even(kf, 1), aie::filter_odd(kf, 1)));
  aie::accum<accfloat, U> va;
  va.from_vector(aie::load_v<U>(v + u * U));
  gm::VF d =
      aie::sub(va.to_vector<float>(),
               aie::mul(kv.to_vector<float>(), decay).to_vector<float>());
  aie::vector<bfloat16, U> uv = aie::mul(d, beta).to_vector<bfloat16>();
  const auto uz = aie::interleave_zip(uv, uv, 1);
  const aie::vector<bfloat16, 32> uv2 = aie::concat(uz.first, uz.second);
  const bfloat16 dh = (bfloat16)decay;
  const bfloat16 dl = (bfloat16)(decay - (float)dh);
  aie::accum<accfloat, 32> oa = update(Sh, Sl, Oh, Ol, kn, qn, uv2, dh, dl);
  aie::vector<float, 32> of = oa.to_vector<float>();
  aie::accum<accfloat, U> o16;
  o16.from_vector(aie::add(aie::filter_even(of, 1), aie::filter_odd(of, 1)));
  aie::store_v(ctx + 640 + u * U, o16.to_vector<bfloat16>());
}

// gated RMSNorm of the head's o -> ctx + 768
__attribute__((noinline, minsize)) static void gate_norm(bfloat16 *ctx) {
  using namespace gm;
  // bf16 MACs throughout (FP32 multiplies are emulated): o / rms(o) with the
  // factor split hi + lo, times w, times SiLU(z) on bf16 MACs (~1e-3, below
  // the bf16 rounding of the output)
  const bfloat16 *o = ctx + 640, *z = ctx + 384, *w = ctx + 512;
  const float r =
      rsqrt_v(gm::vadds(gm::vmuls(gm::vbc(ssq_bf16(o)), 1.0f / DV), 1e-6f))
          .get(0);
  const bfloat16 rh = (bfloat16)r, rl = (bfloat16)(r - (float)rh);
  for (int i = 0; i < DV; i += VL) {
    aie::vector<bfloat16, VL> ov = aie::load_v<VL>(o + i);
    aie::accum<accfloat, VL> a = aie::mul(ov, rh);
    a = aie::mac(a, ov, rl);
    aie::vector<bfloat16, VL> xn = a.to_vector<bfloat16>();
    aie::vector<bfloat16, VL> y =
        aie::mul(xn, aie::load_v<VL>(w + i)).to_vector<bfloat16>();
    aie::vector<bfloat16, VL> sz = gm::bf::silu(aie::load_v<VL>(z + i));
    aie::store_v(ctx + 768 + i, aie::mul(y, sz).to_vector<bfloat16>());
  }
}

} // namespace

extern "C" {

// in, out [4096] bf16; gx [4 x 544] the core's 4 heads' mixer rows (core c
// runs heads 4c..4c+3); go [4 x 128] the core's heads' gated outputs (to the
// mixer, once per wave); ctx [1024] bf16, fs [16] f32, st [16] int32 scratch
void gdn_step(bfloat16 *in, bfloat16 *gx, bfloat16 *go, bfloat16 *ctx,
              float *fs, int32_t *st, bfloat16 *out, int32_t core) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  // core | arm << 4: only a Gated DeltaNet wave (arm 1) runs the delta rule;
  // on an attention wave the call does nothing (that layer's slot is unused)
  const int arm = core >> 4;
  core &= 15;
  if (arm != 1)
    return;
  const uint16_t *mg = (const uint16_t *)in;
  if (mg[4092] == 0x7AB1 && mg[4093] == 0xC0DE && mg[4094] == 0x5EED) {
    if (mg[4095] == 0xFACE) { // the wave's first chunk: start over
      st[0] = 0;
      st[1] = 0;
    }
    for (int i = 0; i < 4096; i += 32) // or a NOP chunk; either is copied
      aie::store_v(out + i, aie::load_v<32>(in + i));
    return;
  }
  // st[0]: the chunk within the head (0 = X, 1..NU = state units), st[1]:
  // the head (counters, not k / PER_HEAD: the cores have no integer divide).
  // Bounded, so a stream that did not start with the reset chunk cannot index
  // past the core's buffers.
  int r = st[0];
  if ((unsigned)r > (unsigned)NU)
    r = 0;
  const int j = st[1] & (HPC - 1);
  if (r == 0) {
    x_chunk(in, gx + j * GX_ROWS, ctx, fs, out, j);
    st[0] = 1;
    return;
  }
  unit(in, ctx, fs, out, r - 1);
  if (r < NU) {
    st[0] = r + 1;
    return;
  }
  // the head's last unit: its gated output into go (the core's 4 heads, which
  // go on-chip to the mixer once per wave)
  gate_norm(ctx);
  for (int i = 0; i < DV; i += 32)
    aie::store_v(go + j * DV + i, aie::load_v<32>(ctx + 768 + i));
  st[0] = 0;
  st[1] = j + 1;
}

} // extern "C"
