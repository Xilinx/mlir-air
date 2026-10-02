// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// int4 GEMV for the LM head: y[n] = sum_k w[n][k] x[k], dequantized in place.
//
// One packet = GV_N output rows x GV_K inputs, packed on the host
// (lm_gemv.pack):
//   uint8 q[GV_N/8][GV_K/8][8 k][8 n / 2]   two nibbles per byte, low first
//   bf16  scale[GV_K/32][GV_N]              per row, per 32-wide K group
//   bf16  mins [GV_K/32][GV_N]
// w = q * scale + min. Each 8x8 (k, n) tile is dequantized as in dq4.cc and
// multiplied in a 64-lane mac against x repeated across n, so no bf16 weight
// tile is stored; gv_flush sums the 8 k lanes of each n.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef GV_N
#define GV_N 64
#endif
#ifndef GV_K
#define GV_K 256
#endif
#ifndef GV_KTOT
#define GV_KTOT 1536
#endif
#define GV_GROUP 32

extern "C" {

// xexp[k * 8 + nn] = x[k]: x laid out for gv_acc's 8x8 tiles
void gv_expand(bfloat16 *__restrict x, bfloat16 *__restrict xexp) {
  for (unsigned k = 0; k < GV_KTOT; k += 8)
    for (unsigned kk = 0; kk < 8; kk++)
      aie::store_v(xexp + (k + kk) * 8, aie::broadcast<bfloat16, 8>(x[k + kk]));
}

void gv_zero(float *acc) {
  for (unsigned i = 0; i < GV_N * 8; i += 64)
    aie::store_v(acc + i, aie::zeros<float, 64>());
}

// acc [GV_N/8][64] f32 += dq(packet) . x[kc*GV_K .. +GV_K), with x in the
// gv_expand layout
void gv_acc(uint8_t *__restrict packed, bfloat16 *__restrict xexp,
            float *__restrict acc, int32_t kc) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned LANES = 64;
  constexpr unsigned Q_BYTES = GV_N * GV_K / 2;
  constexpr unsigned NG = GV_K / GV_GROUP;
  constexpr unsigned TILES_PER_G = GV_GROUP / 8;
  const bfloat16 *sc = (const bfloat16 *)(packed + Q_BYTES);
  const bfloat16 *mn = sc + NG * GV_N;
  const aie::vector<uint8, LANES> bias = aie::broadcast<uint8, LANES>(0x43);
  for (unsigned nb = 0; nb < GV_N / 8; nb++) {
    const uint4 *q = (const uint4 *)packed + nb * (GV_K / 8) * LANES / 2;
    const bfloat16 *xe = xexp + kc * GV_K * 8;
    aie::accum<accfloat, LANES> y;
    y.from_vector(aie::load_v<LANES>(acc + nb * LANES));
    for (unsigned g = 0; g < NG; g++) {
      aie::vector<bfloat16, 8> s8 = aie::load_v<8>(sc + g * GV_N + nb * 8);
      aie::vector<bfloat16, 8> m8 = aie::load_v<8>(mn + g * GV_N + nb * 8);
      aie::vector<bfloat16, 16> s16 = aie::concat(s8, s8);
      aie::vector<bfloat16, 16> m16 = aie::concat(m8, m8);
      aie::vector<bfloat16, 32> s32 = aie::concat(s16, s16);
      aie::vector<bfloat16, 32> m32 = aie::concat(m16, m16);
      aie::vector<bfloat16, LANES> sv = aie::concat(s32, s32);
      aie::vector<bfloat16, LANES> mv = aie::concat(m32, m32);
      aie::accum<accfloat, LANES> base;
      base.from_vector(mv);
      base = aie::msc(base, sv, bfloat16(128.0f));
      for (unsigned t = 0; t < TILES_PER_G; t++)
        chess_prepare_for_pipelining chess_loop_range(TILES_PER_G,
                                                      TILES_PER_G) {
          aie::vector<uint4, LANES> qr = aie::load_v<LANES>(q);
          q += LANES / 2;
          aie::vector<uint8, LANES> qb = aie::unpack(qr);
          auto z = aie::interleave_zip(qb, bias, 1);
          aie::vector<bfloat16, LANES> w =
              aie::vector_cast<bfloat16>(aie::concat(z.first, z.second));
          aie::vector<bfloat16, LANES> wd =
              aie::mac(base, w, sv).template to_vector<bfloat16>();
          // xe holds x[k] repeated over the 8 n lanes, so the tile's 64 lanes
          // line up with (kk, nn)
          aie::vector<bfloat16, LANES> xv = aie::load_v<LANES>(xe);
          xe += LANES;
          y = aie::mac(y, wd, xv);
        }
    }
    aie::store_v(acc + nb * LANES, y.template to_vector<float>());
  }
}

// out[j * GV_N + n] (bf16) = sum over the 8 k lanes of acc[nb][kk * 8 + nn]
void gv_flush(float *__restrict acc, bfloat16 *__restrict out_base, int32_t j) {
  bfloat16 *out = out_base + j * GV_N;
  for (unsigned nb = 0; nb < GV_N / 8; nb++) {
    aie::vector<float, 8> s = aie::zeros<float, 8>();
    for (unsigned kk = 0; kk < 8; kk++)
      s = aie::add(s, aie::load_v<8>(acc + nb * 64 + kk * 8));
    aie::accum<accfloat, 8> a;
    a.from_vector(s);
    aie::store_v(out + nb * 8, a.template to_vector<bfloat16>());
  }
}

} // extern "C"
