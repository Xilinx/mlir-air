// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The 8-row attention block's kernels: the Q4_K array's projections, the
// front core's new K and V rows, and the units' attention, one KV head and
// its 8 query heads each, for 8 token rows at positions P .. P + 7.
//
// An egress round: [column][core][token][32 rows] bf16; column c of a K, V
// or q round is head c of its 4, core r its dims 32 r ...
// RW: cos [8 tokens][64] | sin [8][64] | q_norm [128] | k_norm [128].
// A unit's q: bfp16 tiles [token][16 dim tiles] of [8 heads][8 dims].
// K block: [16 dim tiles][16 keys][8]; V block: [2 key halves][16][8 keys][8].
// Scores: per token SSZ words, P [8 heads][16 keys] then 8 float corrections.
// y: f32 per token [17][8 heads][8], dim tiles then the denominators.

#include "q4k_b8.cc"

namespace {

constexpr int TOK = 8, DH = 128, QH = 8;
constexpr int CHUNK = KT * BLK;
constexpr int PAY = 16; // bf16 offset of an egress payload
constexpr int COS = 0, SIN = 512, QN = 1024, KN = 1152;
constexpr int SSZ = 160;
constexpr int YT = 17 * 64;    // a token's y
constexpr int HALF = 16 * BLK; // a K or V block's tiles of one key half
constexpr float SCALE = 0.08838834764831845f; // 1 / sqrt(DH)
constexpr float STALE = 8.0f;

// head c's dims of token t from a round, rows 32 r .. from core r
inline void gather(const bf16 *rnd, int c, int t, bf16 *x) {
  for (int r = 0; r < 4; r++)
    aie::store_v(x + 32 * r, aie::load_v<32>(rnd + (c * 4 + r) * 256 + t * 32));
}

// y = RoPE(RMSNorm(x) w), as the decode's qk_norm and apply_rope
inline void norm_rope(const bf16 *x, const bf16 *w, const bf16 *cs,
                      const bf16 *sn, bf16 *y) {
  alignas(aie::vector_decl_align) bf16 n[DH];
  aie::accum<accfloat, 16> ss = aie::zeros<accfloat, 16>();
  for (int i = 0; i < DH; i += 16)
    ss = aie::mac_square(ss, aie::load_v<16>(x + i));
  // scalar float is emulated here: keep its op count low
  const float d = aie::invsqrt(
      aie::reduce_add(ss.template to_vector<float>()) * (1.0f / DH) + 1e-6f);
  // x w d in bf16 limbs: no float vector multiply here
  const bf16 dh = (bf16)d, dl = (bf16)(d - (float)dh);
  for (int i = 0; i < DH; i += 16) {
    aie::accum<accfloat, 16> wx =
        aie::mul(aie::load_v<16>(x + i), aie::load_v<16>(w + i));
    aie::vector<bf16, 16> hi = wx.template to_vector<bf16>();
    aie::vector<bf16, 16> lo =
        aie::msc(wx, hi, bf16(1.0f)).template to_vector<bf16>();
    aie::accum<accfloat, 16> o = aie::mul(hi, dh);
    o = aie::mac(o, lo, dh);
    o = aie::mac(o, hi, dl);
    aie::store_v(n + i, o.template to_vector<bf16>());
  }
  for (int i = 0; i < DH / 2; i += 16) {
    auto a = aie::load_v<16>(n + i), b = aie::load_v<16>(n + DH / 2 + i);
    auto c = aie::load_v<16>(cs + i), s = aie::load_v<16>(sn + i);
    aie::accum<accfloat, 16> lo = aie::mul(a, c);
    lo = aie::msc(lo, b, s);
    aie::accum<accfloat, 16> hi = aie::mul(a, s);
    hi = aie::mac(hi, b, c);
    aie::store_v(y + i, lo.template to_vector<bf16>());
    aie::store_v(y + DH / 2 + i, hi.template to_vector<bf16>());
  }
}

// 64 bf16 as a bfp16 tile into a stream
inline void put_tile(OS &os, VB v) {
  ACC a;
  a.from_vector(v);
  os << a.template to_vector<bfp16ebs8>();
}

// exp(x) for x in [-87, 88] in bf16 MACs: 2^n 2^(y - n), y = x log2(e)
// (the DFlash kernels' dflash_exp). Expects round-to-nearest-even.
template <unsigned N>
inline aie::vector<bf16, N> exp_bf16(aie::vector<bf16, N> x) {
  constexpr float log2e_hi = 1.4453125f, log2e_lo = -0.00262451171875f;
  constexpr float round_bias = 12582912.0f;
  const auto ones = aie::broadcast<bf16, N>(1.0f);
  aie::accum<accfloat, N> biased = aie::mul(x, (bf16)log2e_hi);
  biased = aie::mac(biased, x, (bf16)log2e_lo);
  biased = aie::mac(biased, ones, (bf16)round_bias);
  aie::vector<int16, N> n = aie::filter_even(
      biased.template to_vector<float>().template cast_to<int16>());
  aie::accum<accfloat, N> minus_r = aie::mac(biased, ones, (bf16)-round_bias);
  minus_r = aie::msc(minus_r, x, (bf16)log2e_hi);
  minus_r = aie::msc(minus_r, x, (bf16)log2e_lo);
  aie::vector<bf16, N> r = minus_r.template to_vector<bf16>();
  aie::vector<bf16, N> r2 = aie::mul(r, r).template to_vector<bf16>();
  aie::vector<bf16, N> r3 = aie::mul(r2, r).template to_vector<bf16>();
  aie::accum<accfloat, N> p = aie::mul(r, (bf16)-0.69140625f);
  p = aie::mac(p, ones, (bf16)1.0f);
  p = aie::mac(p, r2, (bf16)0.2421875f);
  p = aie::msc(p, r3, (bf16)0.054931640625f);
  aie::vector<int16, N> bits =
      aie::add(p.template to_vector<bf16>().template cast_to<int16>(),
               aie::upshift(n, 7));
  return bits.template cast_to<bf16>();
}

// one token's online-softmax step over raw scores [8 heads][16 keys], the
// first `valid` keys live: P and the corrections into s, the maxima in m
inline void softmax_step(aie::vector<bf16, 128> raw, bf16 *m, bf16 *s,
                         int valid) {
  const aie::vector<int, 16> positions(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16);
  const aie::vector<int16, 32> pair_positions(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  valid = valid < 16 ? valid : 16;
  auto mask = aie::le(positions, valid);
  auto pair_mask = aie::le(pair_positions, (int16)valid);
  const auto lowest = aie::broadcast<bf16, 16>(-0x1.FEp127f);
  const auto lo = aie::broadcast<bf16, 16>(-87.0f);
  const auto hi = aie::broadcast<bf16, 16>(88.0f);
  aie::vector<bf16, 128> masked;
#pragma clang loop unroll(full)
  for (int head = 0; head < 8; ++head)
    masked.insert(head,
                  aie::select(lowest, raw.template extract<16>(head), mask));
  auto half_max = [](aie::vector<bf16, 64> v) {
    aie::vector<bf16, 64> t = aie::transpose(v, 4, 16);
    aie::vector<bf16, 32> a =
        aie::max(t.template extract<32>(0), t.template extract<32>(1));
    aie::vector<bf16, 16> b =
        aie::max(a.template extract<16>(0), a.template extract<16>(1));
    return aie::max(b.template extract<8>(0), b.template extract<8>(1));
  };
  aie::vector<bf16, 16> both =
      aie::concat(half_max(masked.template extract<64>(0)),
                  half_max(masked.template extract<64>(1)));
  aie::vector<bf16, 8> rowmax =
      aie::max(aie::filter_even(both, 4), aie::filter_odd(both, 4));
  aie::vector<bf16, 8> old_m = aie::load_v<8>(m);
  // a head's max moves only once a block passes it by STALE: P stays below
  // e^STALE, and most blocks then need no rescale
  aie::vector<bf16, 16> grow16 =
      aie::sub(rowmax.template grow<16>(), old_m.template grow<16>());
  aie::vector<bf16, 8> vmax =
      aie::select(old_m.template grow<16>(), rowmax.template grow<16>(),
                  aie::gt(grow16, aie::broadcast<bf16, 16>(STALE)))
          .template extract<8>(0);
  aie::vector<bf16, 16> corr =
      aie::clamp(aie::sub(old_m, vmax).template grow<16>(), lo, hi);
  aie::accum<accfloat, 16> cf_acc;
  cf_acc.from_vector(exp_bf16(corr));
  aie::vector<float, 16> cf = cf_acc.template to_vector<float>();
  aie::store_v(m, vmax);
  aie::store_v(reinterpret_cast<float *>(s + 128), cf.template extract<8>(0));
  aie::vector<bf16, 64> per_head =
      aie::transpose(vmax.template grow_replicate<64>(), 8, 8);
  auto [heads03, heads47] = aie::interleave_zip(per_head, per_head, 8);
  aie::vector<bf16, 128> vm128 = aie::concat(heads03, heads47);
#pragma clang loop unroll(full)
  for (int pair = 0; pair < 4; ++pair) {
    aie::vector<bf16, 32> d = aie::clamp(
        aie::sub(raw.template extract<32>(pair),
                 vm128.template extract<32>(pair)),
        lo.template grow_replicate<32>(), hi.template grow_replicate<32>());
    aie::store_v(s + pair * 32, aie::select((bf16)0, exp_bf16(d), pair_mask));
  }
}

// a V block as tiles [half][dim tile] of [8 dims][8 keys], keys from
// `valid` on zeroed: a masked value would share its tile's exponent
inline void v_tiles(const bf16 *vb, char *vf, int valid) {
  OS os((bfp16ebs8 *)vf);
  for (int h = 0; h < 2; h++) {
    const int live = valid - h * 8;
    const uint64_t bits = live >= 8   ? ~uint64_t{0}
                          : live <= 0 ? uint64_t{0}
                                      : (uint64_t{1} << (live * 8)) - 1;
    const auto keep = aie::mask<64>::from_uint64(bits);
    for (int d = 0; d < 16; d++) {
      VB v = aie::select(aie::zeros<bf16, 64>(),
                         aie::load_v<64>(vb + h * 1024 + d * 64), keep);
      put_tile(os, aie::transpose(v, 8, 8));
    }
  }
}

} // namespace

extern "C" {

// ---- array cores

void ab_zero(float *__restrict acc) { q4k_b8_zero(acc); }

// acc += chunk cb of x (bfp16 tiles) . W
void ab_block(uint8_t *__restrict x, uint8_t *__restrict w,
              float *__restrict acc, uint8_t *__restrict sp, int32_t cb) {
  q4k_tiles((const char *)x + cb * CHUNK, w, acc, (char *)sp);
}

// the row block into an egress payload, [token][32 rows]
void ab_flush(float *__restrict acc, bf16 *__restrict y) {
  q4k_b8_flush(acc, y + PAY);
}

// ---- front: the new K and V rows, kv [K, V][token][512]

void fr_k(bf16 *__restrict rnd, bf16 *__restrict rw, bf16 *__restrict kv) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  alignas(aie::vector_decl_align) bf16 x[DH];
  for (int t = 0; t < TOK; t++)
    for (int c = 0; c < 4; c++) {
      gather(rnd, c, t, x);
      norm_rope(x, rw + KN, rw + COS + 64 * t, rw + SIN + 64 * t,
                kv + t * 512 + c * DH);
    }
}

void fr_v(bf16 *__restrict rnd, bf16 *__restrict kv) {
  for (int t = 0; t < TOK; t++)
    for (int c = 0; c < 4; c++)
      gather(rnd, c, t, kv + TOK * 512 + t * 512 + c * DH);
}

// ---- unit qk core

// key block `block` against the 8 tokens; token t, at pos + t, sees keys
// up to it, or with causal 0 up to pos + 7. kf: 2 KB for the block's tiles
void qk_blk(uint8_t *__restrict qb, bf16 *__restrict kb, uint8_t *__restrict kf,
            bf16 *__restrict m, bf16 *__restrict s, int32_t block, int32_t pos,
            int32_t causal) {
  const int base = pos + 1 - block * 16;
  if (base + TOK - 1 <= 0)
    return;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  {
    OS os((bfp16ebs8 *)kf);
    for (int h = 0; h < 2; h++)
      for (int d = 0; d < 16; d++)
        put_tile(os, aie::load_v<64>(kb + d * 128 + h * 64));
  }
  for (int t = 0; t < TOK; t++) {
    const int valid = base + (causal ? t : TOK - 1);
    if (valid <= 0)
      continue;
    if (block == 0)
      aie::store_v(m + t * 16, aie::broadcast<bf16, 16>(-0x1.FEp127f));
    BS qa((const bfp16ebs8 *)(qb + t * 16 * BLK));
    BS kl((const bfp16ebs8 *)kf), kr((const bfp16ebs8 *)(kf + HALF));
    ACC l = aie::zeros<accfloat, 64>(), r = l;
    for (int d = 0; d < 16; d++) {
      BV A, Bl, Br;
      qa >> A;
      kl >> Bl;
      kr >> Br;
      l = mma(l, A, Bl);
      r = mma(r, A, Br);
    }
    auto z = aie::interleave_zip(l.template to_vector<bf16>(),
                                 r.template to_vector<bf16>(), 8);
    softmax_step(aie::concat(z.first, z.second), m + t * 16, s + t * SSZ,
                 valid);
  }
}

// ---- unit pv core

void pv_zero(float *__restrict y) {
  for (int i = 0; i < TOK * YT; i += 32)
    aie::store_v(y + i, aie::zeros<float, 32>());
}

// y = y c + P V for each live token; vf: 2 KB for the block's tiles
void pv_blk(bf16 *__restrict s, bf16 *__restrict vb, uint8_t *__restrict vf,
            float *__restrict y, int32_t block, int32_t pos, int32_t causal) {
  const int base = pos + 1 - block * 16;
  if (base + TOK - 1 <= 0)
    return;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const v2int32 ex = {EXP1, EXP1};
  const v64bfp16ebs8 ov = {aie::broadcast<int8, 64>(1), ex};
  const BV ones(ov);
  const bool full = base >= 16 || (!causal && base + TOK - 1 >= 16);
  if (full)
    v_tiles(vb, (char *)vf, 16);
  for (int t = 0; t < TOK; t++) {
    const int valid = base + (causal ? t : TOK - 1);
    if (valid <= 0)
      continue;
    if (!full)
      v_tiles(vb, (char *)vf, valid);
    const bf16 *st = s + t * SSZ;
    float *yt = y + t * YT;
    // P's key halves as [8 heads][8 keys]
    aie::vector<bf16, 128> p = aie::load_v<128>(st);
    BV a0, a1;
    {
      ACC a;
      a.from_vector(aie::filter_even(p, 8));
      a0 = a.template to_vector<bfp16ebs8>();
      a.from_vector(aie::filter_odd(p, 8));
      a1 = a.template to_vector<bfp16ebs8>();
    }
    // rescale only when a running max moved
    const float *cf = (const float *)(st + 128);
    bool moved = false;
    for (int h = 0; h < 8; h++)
      moved |= cf[h] < 1.0f;
    if (moved) {
      alignas(aie::vector_decl_align) bf16 chs[8], cls[8];
      for (int h = 0; h < 8; h++) {
        float c = cf[h];
        chs[h] = (bf16)c;
        cls[h] = (bf16)(c - (float)chs[h]);
      }
      const VB ch = rows8(aie::load_v<8>(chs)), cl = rows8(aie::load_v<8>(cls));
      for (int d = 0; d < 17; d++) {
        ACC a;
        a.from_vector(aie::load_v<64>(yt + d * 64));
        VB hi, lo;
        limbs(a, hi, lo);
        ACC n = aie::mul(hi, ch);
        n = aie::mac(n, lo, ch);
        n = aie::mac(n, hi, cl);
        aie::store_v(yt + d * 64, n.template to_vector<float>());
      }
    }
    BS vl((const bfp16ebs8 *)vf), vr((const bfp16ebs8 *)(vf + HALF));
    for (int d = 0; d < 16; d++) {
      BV Bl, Br;
      vl >> Bl;
      vr >> Br;
      ACC a;
      a.from_vector(aie::load_v<64>(yt + d * 64));
      a = mma(a, a0, Bl);
      a = mma(a, a1, Br);
      aie::store_v(yt + d * 64, a.template to_vector<float>());
    }
    ACC a;
    a.from_vector(aie::load_v<64>(yt + 16 * 64));
    a = mma(a, a0, ones);
    a = mma(a, a1, ones);
    aie::store_v(yt + 16 * 64, a.template to_vector<float>());
  }
}

// y / denominators as o_proj's x: bfp16 tiles [head][dim tile] of
// [8 tokens][8 dims], heads 4 hf ..
void pv_fin(float *__restrict y, uint8_t *__restrict ot, int32_t hf) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  alignas(aie::vector_decl_align) bf16 ih[QH][TOK], il[QH][TOK];
  for (int t = 0; t < TOK; t++)
    for (int h = 0; h < QH; h++) {
      float v = 1.0f / y[t * YT + 16 * 64 + h * 8];
      ih[h][t] = (bf16)v;
      il[h][t] = (bf16)(v - (float)ih[h][t]);
    }
  OS os((bfp16ebs8 *)ot);
  alignas(aie::vector_decl_align) float g[64];
  for (int h = 4 * hf; h < 4 * hf + 4; h++) {
    const VB vh = rows8(aie::load_v<8>(ih[h])),
             vl = rows8(aie::load_v<8>(il[h]));
    for (int d = 0; d < 16; d++) {
      for (int t = 0; t < TOK; t++)
        aie::store_v(g + t * 8, aie::load_v<8>(y + t * YT + d * 64 + h * 8));
      ACC a;
      a.from_vector(aie::load_v<64>(g));
      VB hi, lo;
      limbs(a, hi, lo);
      ACC n = aie::mul(hi, vh);
      n = aie::mac(n, lo, vh);
      n = aie::mac(n, hi, vl);
      os << n.template to_vector<bfp16ebs8>();
    }
  }
}

} // extern "C"
