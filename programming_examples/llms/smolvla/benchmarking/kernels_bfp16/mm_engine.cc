//===- mm_engine.cc - GEMM engine kernels: bfp16 GEMM + fused epilogues -*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// mm_bfp16.cc plus the epilogues gemm_engine.py folds into its GEMM jobs, so
// one object serves every phase of the engine herd (AIR links one object per
// core):
//   add_residual_blocked   acc += a residual tile that arrived on the A channel
//   zero_rows / sumsq_rows_blocked   per-row sum of squares of the A chunks
//                                    (RMSNorm statistics of the GEMM input)
//   rows_rstd              ss -> rsqrt(ss/RMS_K + eps)
//   rms_rope_blocked       row scale by rstd, then RoPE on column pairs, with
//                          the (cos, sin) table arriving on the A channel
//   f32_to_bf16_rms_swiglu drain: row scale by rstd, then SwiGLU, into one half
//                          of the output tile
//
// Layouts: acc (DIM_N/8, DIM_M/8, 8, 8) N-outer; A tile (DIM_M/8, DIM_K/8, 8, 8);
// every 8x8 block is row-major (m row, n or k column).
//
//===----------------------------------------------------------------------===//

#include "mm_bfp16.cc"

#ifndef RMS_K
#define RMS_K 960
#endif
#ifndef RMS_EPS
#define RMS_EPS 1e-5f
#endif
// Own-key attention geometry (gemm_engine.py OWN_*): head dim, q heads per kv
// group, kv groups, and q heads per DIM_K-pair (one arena tile). One key per
// tile row: only the 16-row build (the expert) has own-key jobs.
#if DIM_M == 16
#define OWN_HD 64
#define OWN_QPG 3
#define OWN_NKV 5
#define OWN_HPT (2 * DIM_K / OWN_HD)
#define OWN_VW 144
#define OWN_VR 25

// 8 rows of 8 bf16 (row stride `stride` elements) as one row-major 8x8 block.
static inline aie::vector<bfloat16, 64> load_block_8x8(const bfloat16 *p,
                                                       unsigned stride) {
  return aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + stride),
                     aie::load_v<8>(p + 2 * stride),
                     aie::load_v<8>(p + 3 * stride),
                     aie::load_v<8>(p + 4 * stride),
                     aie::load_v<8>(p + 5 * stride),
                     aie::load_v<8>(p + 6 * stride),
                     aie::load_v<8>(p + 7 * stride));
}

static inline aie::block_vector<bfp16ebs8, 64>
to_bfp16(const aie::vector<bfloat16, 64> &v) {
  aie::accum<accfloat, 64> a;
  a.from_vector(v);
  return a.to_vector<bfp16ebs8>();
}

// acc block (nb, mb) += A block (mb, kb) x B block (n rows by k columns).
static inline void mac_block(float *acc, const bfloat16 *a, unsigned nb,
                             unsigned mb, unsigned kb,
                             const aie::block_vector<bfp16ebs8, 64> &b) {
  constexpr unsigned MB = DIM_M / 8, KB = DIM_K / 8, BE = 64;
  float *pc = acc + (nb * MB + mb) * BE;
  aie::accum<accfloat, 64> c(aie::load_v<64>(pc));
  c = mac_8x8_8x8T(to_bfp16(aie::load_v<64>(a + (mb * KB + kb) * BE)), b, c);
  aie::store_v(pc, c.template to_vector<float>());
}

// Scores against the step's own keys, one DIM_K chunk kc of Q tile t (A). b is
// this core column's DIM_M keys of the K arena tile, row-major [DIM_M][2*DIM_K]
// over all kv groups (then padding). Output block nb = head * 2 + key half.
static void own_scores(bfloat16 *a, uint8_t *b_bytes, float *acc, unsigned kc,
                       unsigned t) {
  constexpr unsigned T = 8, MB = DIM_M / T, KB = DIM_K / T, W = 2 * DIM_K;
  static_assert(DIM_M == 16 && DIM_N == 2 * OWN_HPT * T, "16 keys x 5 heads");
  ::aie::set_rounding(round_mode);
  const bfloat16 *b = reinterpret_cast<const bfloat16 *>(b_bytes);
  for (unsigned kb = 0; kb < KB; kb++) {
    const unsigned d = kc * DIM_K + kb * T, j = d / OWN_HD;
    const unsigned g = (t * OWN_HPT + j) / OWN_QPG;
    for (unsigned kg = 0; kg < 2; kg++) {
      const auto bq =
          to_bfp16(load_block_8x8(b + kg * T * W + g * OWN_HD + d % OWN_HD, W));
      for (unsigned mb = 0; mb < MB; mb++)
        mac_block(acc, a, j * 2 + kg, mb, kb, bq);
    }
  }
}

// P_own . V for one DIM_K chunk kc of P_own (A: columns key group 2kc + kb/10,
// head, key half, as own_scores writes them). b is V of herd rows 2kc,
// 2kc+1 as [2][OWN_VR][OWN_VW] (rows past DIM_M are padding), columns col0..
// (gemm_engine.own_pv_col0). Core column `col`, output half `half`: blocks
// 0..NB/2 are P.V for output dims col*DIM_N + half*DIM_N/2 + .., blocks NB/2..
// the matching P.1 sums.
static void own_pv(bfloat16 *a, uint8_t *b_bytes, float *acc, unsigned kc,
                   unsigned t, unsigned col, unsigned half) {
  constexpr unsigned T = 8, MB = DIM_M / T, H = DIM_N / T / 2;
  constexpr unsigned W = OWN_VW, KEYS = DIM_M, TW = 2 * DIM_K;
  static_assert(DIM_M == 16 && DIM_K == 2 * OWN_HPT * KEYS, "2 key groups");
  ::aie::set_rounding(round_mode);
  const bfloat16 *b = reinterpret_cast<const bfloat16 *>(b_bytes);
  const unsigned o0 = col * DIM_N + half * (DIM_N / 2);
  unsigned col0 = (t * OWN_HPT + o0 / OWN_HD) / OWN_QPG * OWN_HD;
  if (col0 > TW - W)
    col0 = TW - W;
  const auto ones = to_bfp16(aie::broadcast<bfloat16, 64>((bfloat16)1.0f));
  for (unsigned nb = 0; nb < H; nb++) {
    const unsigned o = o0 + nb * T, j = o / OWN_HD;
    const unsigned vc = (t * OWN_HPT + j) / OWN_QPG * OWN_HD + o % OWN_HD - col0;
    for (unsigned cl = 0; cl < 2; cl++) {
      for (unsigned kg = 0; kg < 2; kg++) {
        const unsigned kb = cl * OWN_HPT * 2 + j * 2 + kg;
        const bfloat16 *pb = b + (cl * OWN_VR + kg * T) * W + vc;
        const auto bq = to_bfp16(aie::transpose(load_block_8x8(pb, W), 8, 8));
        for (unsigned mb = 0; mb < MB; mb++) {
          mac_block(acc, a, nb, mb, kb, bq);
          mac_block(acc, a, nb + H, mb, kb, ones);
        }
      }
    }
  }
}
#endif

extern "C" {

// Columns [half * DIM_N, half * DIM_N + DIM_N) of a DIM_M x DIM_K A tile, added
// in f32 to the accumulator.
void SYM(add_residual_blocked)(float *acc, bfloat16 *a, int32_t half) {
  constexpr unsigned T = 8, NB = DIM_N / T, MB = DIM_M / T, KB = DIM_K / T;
  constexpr unsigned BE = T * T, VW = 16;
  static_assert(DIM_K == 2 * DIM_N, "one A chunk holds two output tiles");
  const aie::vector<bfloat16, VW> one_v =
      aie::broadcast<bfloat16, VW>((bfloat16)1.0f);
  for (unsigned nb = 0; nb < NB; nb++) {
    for (unsigned mb = 0; mb < MB; mb++) {
      float *pc = acc + (nb * MB + mb) * BE;
      const bfloat16 *pa = a + (mb * KB + half * NB + nb) * BE;
      for (unsigned e = 0; e < BE; e += VW) {
        aie::accum<accfloat, VW> c(aie::load_v<VW>(pc + e));
        c = aie::mac(c, aie::load_v<VW>(pa + e), one_v);
        aie::store_v(pc + e, c.template to_vector<float>());
      }
    }
  }
}

void SYM(zero_rows)(float *ss) {
  for (unsigned i = 0; i < DIM_M; i++)
    ss[i] = 0.0f;
}

void SYM(sumsq_rows_blocked)(bfloat16 *a, float *ss) {
  constexpr unsigned T = 8, MB = DIM_M / T, KB = DIM_K / T, BE = T * T;
  constexpr unsigned VW = 16;
  for (unsigned mb = 0; mb < MB; mb++) {
    for (unsigned rp = 0; rp < T / 2; rp++) {
      aie::accum<accfloat, VW> s = aie::zeros<accfloat, VW>();
      for (unsigned kb = 0; kb < KB; kb++) {
        aie::vector<bfloat16, VW> v =
            aie::load_v<VW>(a + (mb * KB + kb) * BE + rp * VW);
        s = aie::mac(s, v, v);
      }
      aie::vector<float, VW> f = s.template to_vector<float>();
      float lo = 0.0f, hi = 0.0f;
      for (unsigned i = 0; i < T; i++) {
        lo += f[i];
        hi += f[i + T];
      }
      ss[mb * T + 2 * rp] += lo;
      ss[mb * T + 2 * rp + 1] += hi;
    }
  }
}

// Sum of squares -> rstd = rsqrt(ss / RMS_K + eps), in place.
// Vector form: the scalar loop is fully unrolled (2.6 KB) and pulls in a soft
// float multiply, and every core's program is reloaded on every launch.
void SYM(rows_rstd)(float *ss) {
  constexpr unsigned VW = 16;
  static_assert(DIM_M % VW == 0, "rows in whole vectors");
  const aie::vector<float, VW> inv_k = aie::broadcast<float, VW>(1.0f / RMS_K);
  const aie::vector<float, VW> eps = aie::broadcast<float, VW>(RMS_EPS);
  for (unsigned r = 0; r < DIM_M; r += VW) {
    aie::vector<float, VW> v =
        aie::mul(aie::load_v<VW>(ss + r), inv_k).template to_vector<float>();
    aie::store_v(ss + r, aie::invsqrt(aie::add(v, eps)));
  }
}

// RMSNorm row scale, then RoPE on adjacent column pairs: (a, b) -> (a cos - b sin,
// b cos + a sin), with (cos, sin) the matching column pair of the A tile's half
// `half` (the table rides the A channel like a residual). Head dims are stored
// pair-interleaved so each rotation stays inside one 8x8 block; (1, 0) pairs pass
// columns through unrotated.
void SYM(rms_rope_blocked)(float *acc, float *ss, bfloat16 *a, int32_t half) {
  constexpr unsigned T = 8, NB = DIM_N / T, MB = DIM_M / T, KB = DIM_K / T;
  constexpr unsigned BE = T * T, VW = 16;
  static_assert(DIM_K == 2 * DIM_N, "one A chunk holds two output tiles");
  const aie::mask<VW> hi_rows = aie::mask<VW>::from_uint32(0xFF00u);
  for (unsigned nb = 0; nb < NB; nb++) {
    for (unsigned mb = 0; mb < MB; mb++) {
      float *pc = acc + (nb * MB + mb) * BE;
      const bfloat16 *pa = a + (mb * KB + half * NB + nb) * BE;
      for (unsigned e = 0; e < BE; e += 2 * VW) {
        // Lanes of the two vectors: rows row .. row+3, 8 columns each.
        const unsigned row = mb * T + e / T;
        const aie::vector<float, VW> s0 =
            aie::select(aie::broadcast<float, VW>(ss[row]),
                        aie::broadcast<float, VW>(ss[row + 1]), hi_rows);
        const aie::vector<float, VW> s1 =
            aie::select(aie::broadcast<float, VW>(ss[row + 2]),
                        aie::broadcast<float, VW>(ss[row + 3]), hi_rows);
        aie::vector<float, VW> x0 =
            aie::mul(aie::load_v<VW>(pc + e), s0).template to_vector<float>();
        aie::vector<float, VW> x1 = aie::mul(aie::load_v<VW>(pc + e + VW), s1)
                                        .template to_vector<float>();
        auto [xa, xb] = aie::interleave_unzip(x0, x1, 1);
        aie::accum<accfloat, VW> t0, t1;
        t0.from_vector(aie::load_v<VW>(pa + e));
        t1.from_vector(aie::load_v<VW>(pa + e + VW));
        auto [co, si] = aie::interleave_unzip(t0.template to_vector<float>(),
                                              t1.template to_vector<float>(), 1);
        aie::vector<float, VW> oa =
            aie::sub(aie::mul(xa, co).template to_vector<float>(),
                     aie::mul(xb, si).template to_vector<float>());
        aie::vector<float, VW> ob =
            aie::add(aie::mul(xb, co).template to_vector<float>(),
                     aie::mul(xa, si).template to_vector<float>());
        auto [o0, o1] = aie::interleave_zip(oa, ob, 1);
        aie::store_v(pc + e, o0);
        aie::store_v(pc + e + VW, o1);
      }
    }
  }
}

// g and u are narrowed to bf16, scaled by the per-row rstd, then silu_and_mul's
// math, as f32_to_bf16_swiglu_mn.
void SYM(f32_to_bf16_rms_swiglu)(float *src, float *ss, bfloat16 *dst,
                                 int32_t half) {
  constexpr unsigned VW = 16, T = 8;
  constexpr unsigned NB = DIM_N / T, H = NB / 2, BE = DIM_M * T;
  static_assert(NB % 2 == 0, "tile_n must hold whole gate/up block pairs");
  const aie::vector<bfloat16, VW> half_v =
      aie::broadcast<bfloat16, VW>((bfloat16)0.5f);
  const aie::vector<bfloat16, VW> one_v =
      aie::broadcast<bfloat16, VW>((bfloat16)1.0f);
  const aie::mask<VW> hi_rows = aie::mask<VW>::from_uint32(0xFF00u);
  for (unsigned jb = 0; jb < H; jb++) {
    const float *pg = src + jb * BE;
    const float *pu = src + (jb + H) * BE;
    bfloat16 *pd = dst + (half * H + jb) * BE;
    for (unsigned e = 0; e < BE; e += VW) {
      // e = mb * 64 + r * 8 + t: lanes 0-7 are row mb*8+r, 8-15 the next row.
      const unsigned row = e / T;
      const aie::vector<bfloat16, VW> sv =
          aie::select(aie::broadcast<bfloat16, VW>((bfloat16)ss[row]),
                      aie::broadcast<bfloat16, VW>((bfloat16)ss[row + 1]),
                      hi_rows);
      ::aie::set_rounding(aie::rounding_mode::conv_even);
      aie::vector<bfloat16, VW> g =
          narrow_f32_to_bf16<VW>(aie::load_v<VW>(pg + e));
      aie::vector<bfloat16, VW> u =
          narrow_f32_to_bf16<VW>(aie::load_v<VW>(pu + e));
      g = aie::mul(g, sv).template to_vector<bfloat16>();
      u = aie::mul(u, sv).template to_vector<bfloat16>();
      ::aie::set_rounding(aie::rounding_mode::floor);
      aie::vector<bfloat16, VW> g_half = aie::mul(g, half_v);
      aie::accum<accfloat, VW> tanh_in;
      tanh_in.from_vector(g_half);
      aie::vector<bfloat16, VW> tanh_val =
          aie::tanh<bfloat16>(tanh_in.template to_vector<float>());
      aie::vector<bfloat16, VW> sigmoid =
          aie::mul(half_v, aie::add(one_v, tanh_val));
      aie::vector<bfloat16, VW> silu = aie::mul(g, sigmoid);
      aie::vector<bfloat16, VW> out = aie::mul(silu, u);
      aie::store_v(pd + e, out);
    }
  }
}

// Attention probabilities without the running max: exp of the (scaled, masked)
// scores, so every score must stay below ~88 (the exponent is clamped to
// [-126, 127] in base 2: masked scores give ~1e-38, not 0).
void SYM(f32_to_bf16_exp_mn)(float *src, bfloat16 *dst) {
  constexpr unsigned VW = 16, N = DIM_M * DIM_N;
  const aie::vector<float, VW> log2e = aie::broadcast<float, VW>(1.4426950409f);
  const aie::vector<float, VW> lo = aie::broadcast<float, VW>(-126.0f);
  const aie::vector<float, VW> hi = aie::broadcast<float, VW>(127.0f);
  for (unsigned e = 0; e < N; e += VW) {
    aie::vector<float, VW> x =
        aie::mul(aie::load_v<VW>(src + e), log2e).template to_vector<float>();
    aie::store_v(dst + e, aie::exp2<bfloat16>(aie::min(aie::max(x, lo), hi)));
  }
}

// Softmax normalisation as a paired drain (the SwiGLU pairing): the tile_n block
// holds tile_n/2 columns of P.V then tile_n/2 of the matching row sums P.1,
// and the quotient fills one half of the output tile.
void SYM(f32_to_bf16_div_mn)(float *src, bfloat16 *dst, int32_t half) {
  constexpr unsigned VW = 16, T = 8;
  constexpr unsigned NB = DIM_N / T, H = NB / 2, BE = DIM_M * T;
  static_assert(NB % 2 == 0, "tile_n must hold whole value/sum block pairs");
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned jb = 0; jb < H; jb++) {
    const float *pv = src + jb * BE;
    const float *ps = src + (jb + H) * BE;
    bfloat16 *pd = dst + (half * H + jb) * BE;
    for (unsigned e = 0; e < BE; e += VW) {
      aie::vector<float, VW> q =
          aie::mul(aie::load_v<VW>(pv + e), aie::inv(aie::load_v<VW>(ps + e)))
              .template to_vector<float>();
      aie::store_v(pd + e, narrow_f32_to_bf16<VW>(q));
    }
  }
}

#if DIM_M == 16
// GEMM step of a job with own-key tiles (gemm_engine Job.own): the packed
// matmul unless `own`, else kind 1 scores / kind 2 P.V against the own keys, on
// DIM_K chunk ch & 1 of the own step.
void SYM(matmul_engine_own)(bfloat16 *a, uint8_t *b, float *acc, int32_t kind,
                            int32_t own, int32_t ch, int32_t t, int32_t col,
                            int32_t half) {
  if (!own)
    matmul_bf16_x_bfp16_packed_f32(a, b, acc);
  else if (kind == 1)
    own_scores(a, b, acc, ch & 1, t);
  else
    own_pv(a, b, acc, ch & 1, t, col, half);
}
#endif

} // extern "C"
