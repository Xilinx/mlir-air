// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// One Q4_K weight block of the Qwen3-Coder bundle (32 rows x 256 columns, as
// the decode streams it) against 8 token rows. The decode's GEMV (q4k_mvm.cc)
// in matrix form: the nibbles go into the matmul as they are, exact integers,
// and the scale and min paths are applied per 32-column group after it:
//
//   y = S sum_g sc_g (x_g . q_g) + M sum_g m_g sum(x_g)
//
// Block: scales uint8 [8 groups][32 rows], mins the same, nibbles pre-tiled
// (q4k_layout.py) as [8 groups][4 col tiles][4 row tiles][8 rows][8 cols]
// (low nibble first), S bf16 [32], M bf16 [32].
// x:   bf16 [32 col tiles][8 tokens][8]
// acc: f32  [4 row tiles][8 tokens][8]

#pragma once

#include <aie_api/aie.hpp>
#include <cstdint>

namespace {

using bf16 = bfloat16;
using BV = aie::block_vector<bfp16ebs8, 64>;
using ACC = aie::accum<accfloat, 64>;
using VB = aie::vector<bf16, 64>;
using BS = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;
using OS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64>;

constexpr int ROWS = 32, GROUPS = 8, KT = 32, NT = 4;
constexpr int QS_OFF = 2 * GROUPS * ROWS;
constexpr int S_OFF = QS_OFF + ROWS * 256 / 2;
constexpr int BLK = 72; // bytes of a bfp16 8 x 8 tile
// a tile row m 2^(x - 133): with x = 133 the mantissas are the values
constexpr int EXP1 = 0x85858585;

inline ACC mma(ACC c, BV a, BV b) {
  return ACC(::mac_8x8_8x8T_conf(a, b, c, 0, 0, 0));
}

// lane r * 8 + c is v[r]
inline VB rows8(aie::vector<bf16, 8> v) {
  aie::vector<bf16, 16> b = aie::concat(v, v);
  aie::vector<bf16, 32> c = aie::concat(b, b);
  return aie::transpose(aie::concat(c, c), 8, 8);
}

// lane r * 8 + c is v[c]
inline VB cols8(aie::vector<bf16, 8> v) {
  aie::vector<bf16, 16> b = aie::concat(v, v);
  aie::vector<bf16, 32> c = aie::concat(b, b);
  return aie::concat(c, c);
}

// a float32 accumulator as two bf16 limbs
inline void limbs(ACC a, VB &hi, VB &lo) {
  hi = a.template to_vector<bf16>();
  lo = aie::msc(a, hi, bf16(1.0f)).template to_vector<bf16>();
}

// 32 uint8 (integers below 256, exact) as bf16, 16 lanes at a time: Peano's
// 32-lane conversion returns the upper half twice
inline aie::vector<bf16, 32> u8_bf16_32(const uint8_t *p) {
  aie::accum<accfloat, 16> lo, hi;
  lo.from_vector(aie::to_float(aie::load_v<16>(p), 0));
  hi.from_vector(aie::to_float(aie::load_v<16>(p + 16), 0));
  return aie::concat(lo.template to_vector<bf16>(),
                     hi.template to_vector<bf16>());
}

// sp[J] += limbs(p) * sc, sp[4 + J] += limbs(ps) * mn, row tile J's lanes
template <int J>
inline void fold(ACC p, VB psh, VB psl, aie::vector<bf16, 32> sc,
                 aie::vector<bf16, 32> mn, float *sp) {
  VB scj = cols8(sc.template extract<8>(J));
  VB mnj = cols8(mn.template extract<8>(J));
  VB ph, pl;
  limbs(p, ph, pl);
  ACC as, am;
  as.from_vector(aie::load_v<64>(sp + J * 64));
  as = aie::mac(as, ph, scj);
  as = aie::mac(as, pl, scj);
  aie::store_v(sp + J * 64, as.template to_vector<float>());
  am.from_vector(aie::load_v<64>(sp + (NT + J) * 64));
  am = aie::mac(am, psh, mnj);
  am = aie::mac(am, psl, mnj);
  aie::store_v(sp + (NT + J) * 64, am.template to_vector<float>());
}

// acc[J] += the scale path times S and the min path times M
template <int J>
inline void finish(const float *sp, const bf16 *S, const bf16 *M, float *acc) {
  ACC c;
  c.from_vector(aie::load_v<64>(acc + J * 64));
  VB Sj = cols8(aie::load_v<8>(S + 8 * J));
  VB Mj = cols8(aie::load_v<8>(M + 8 * J));
  ACC a;
  VB h, l;
  a.from_vector(aie::load_v<64>(sp + J * 64));
  limbs(a, h, l);
  c = aie::mac(c, h, Sj);
  c = aie::mac(c, l, Sj);
  a.from_vector(aie::load_v<64>(sp + (NT + J) * 64));
  limbs(a, h, l);
  c = aie::mac(c, h, Mj);
  c = aie::mac(c, l, Mj);
  aie::store_v(acc + J * 64, c.template to_vector<float>());
}

// row tiles J0 and J0 + 1 of one pass over the block (x . 1 too when J0 = 0):
// fewer live accumulators than all four at once, which the core would spill
template <int J0>
inline void q4k_pass(const char *xb, const uint8_t *w, float *sp) {
  const v2int32 ex = {EXP1, EXP1};
  // x . 1: each token's sum over a tile, in every lane of its row
  const v64bfp16ebs8 ov = {aie::broadcast<int8, 64>(1), ex};
  const BV ones(ov);
  BS xa((const bfp16ebs8 *)xb);
  // row tiles J0 and J0 + 1 of each col tile
  const uint8_t *q = w + QS_OFF + J0 * 32;
  for (int g = 0; g < GROUPS; g++)
    chess_prepare_for_pipelining {
      ACC pa = aie::zeros<accfloat, 64>(), pb = pa, ps = pa;
#pragma clang loop unroll(full)
      for (int k = 0; k < 4; k++) {
        aie::vector<int8, 128> u = aie::vector_cast<int8>(
            aie::unpack(aie::vector_cast<uint4>(aie::load_v<64>(q))));
        q += 128;
        BV A;
        xa >> A;
        v64bfp16ebs8 va = {u.template extract<64>(0), ex};
        v64bfp16ebs8 vb = {u.template extract<64>(1), ex};
        pa = mma(pa, A, BV(va));
        pb = mma(pb, A, BV(vb));
        if constexpr (J0 == 0)
          ps = mma(ps, A, ones);
      }
      aie::vector<bf16, 32> sc = u8_bf16_32(w + g * ROWS);
      aie::vector<bf16, 32> mn = u8_bf16_32(w + (GROUPS + g) * ROWS);
      VB psh, psl;
      if constexpr (J0 == 0) {
        limbs(ps, psh, psl);
        // the min path's x sums, for the second pass
        aie::store_v((bf16 *)(sp + 2 * NT * 64) + g * 128, psh);
        aie::store_v((bf16 *)(sp + 2 * NT * 64) + g * 128 + 64, psl);
      } else {
        psh = aie::load_v<64>((const bf16 *)(sp + 2 * NT * 64) + g * 128);
        psl = aie::load_v<64>((const bf16 *)(sp + 2 * NT * 64) + g * 128 + 64);
      }
      fold<J0>(pa, psh, psl, sc, mn, sp);
      fold<J0 + 1>(pb, psh, psl, sc, mn, sp);
    }
}

// acc += x . W over the block; xb: x as bfp16 tiles [col tile]; sp: 4 KB for
// the scale and min paths and the x sums.
inline void q4k_tiles(const char *xb, const uint8_t *w, float *acc, char *spc) {
  float *sp = (float *)spc;
  for (int i = 0; i < 2 * NT * 64; i += 32)
    aie::store_v(sp + i, aie::zeros<float, 32>());
  q4k_pass<0>(xb, w, sp);
  q4k_pass<2>(xb, w, sp);
  const bf16 *S = (const bf16 *)(w + S_OFF);
  const bf16 *M = S + ROWS;
  finish<0>(sp, S, M, acc);
  finish<1>(sp, S, M, acc);
  finish<2>(sp, S, M, acc);
  finish<3>(sp, S, M, acc);
}

} // namespace

extern "C" {

void q4k_b8_zero(float *__restrict acc) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < NT * 64; i += 32)
    aie::store_v(acc + i, aie::zeros<float, 32>());
}

// scratch: x in bfp16, then the nibble tiles
void q4k_b8_block(bf16 *__restrict x, uint8_t *__restrict w,
                  float *__restrict acc, uint8_t *__restrict scratch) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  char *xb = (char *)scratch;
  char *wb = xb + KT * BLK;
  {
    OS os((bfp16ebs8 *)xb);
    for (int i = 0; i < KT; i++) {
      ACC a;
      a.from_vector(aie::load_v<64>(x + i * 64));
      os << a.template to_vector<bfp16ebs8>();
    }
  }
  q4k_tiles(xb, w, acc, wb);
}

// y[token][32 rows] = acc in bf16
void q4k_b8_flush(float *__restrict acc, bf16 *__restrict y) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int t = 0; t < 8; t++)
    for (int j = 0; j < NT; j++) {
      aie::accum<accfloat, 8> a;
      a.from_vector(aie::load_v<8>(acc + (j * 8 + t) * 8));
      aie::store_v(y + t * 32 + j * 8, a.template to_vector<bf16>());
    }
}

} // extern "C"
