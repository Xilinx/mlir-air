// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// int4 weight tile -> the bf16 B tile mm_aie2p.cc's matmul consumes.
//
// One K step of one core's DQ_TN output columns, packed on the host
// (packing.pack_q4):
//   uint8 q[TN/8][TK/8][8 k][8 n / 2]   two nibbles per byte, low nibble first
//   bf16  scale[TK/32][TN]             per output column, per 32-wide K group
//   bf16  mins [TK/32][TN]
// w = q * scale + min (Q4NX's codec). Output: bf16 [TN/8][TK/8][8 k][8 n].
//
// uint4 -> bf16 without a conversion: the bf16 bit pattern 0x4300 | q is
// exactly 128 + q, so w = d (128 + q) + (m - 128 d), with the 128 d term
// folded into the per-group base.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DQ_TN
#define DQ_TN 64
#endif
#ifndef DQ_TK
#define DQ_TK 128
#endif
#define DQ_GROUP 32

extern "C" {

void dq4_bf16(uint8_t *__restrict packed, bfloat16 *__restrict out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned LANES = 64;
  constexpr unsigned Q_BYTES = DQ_TN * DQ_TK / 2;
  constexpr unsigned NG = DQ_TK / DQ_GROUP;
  const bfloat16 *sc = (const bfloat16 *)(packed + Q_BYTES);
  const bfloat16 *mn = sc + NG * DQ_TN;
  const aie::vector<uint8, 2 * LANES> bias =
      aie::broadcast<uint8, 2 * LANES>(0x43);
  const uint4 *q = (const uint4 *)packed;
  bfloat16 *o = out;
  // one flat loop over (column block, K group), the group's tiles unrolled
  // two at a time: one load and one zip give two tiles
#pragma clang loop min_iteration_count(DQ_TN / 8 * NG)
#pragma clang loop max_iteration_count(DQ_TN / 8 * NG)
  for (unsigned i = 0; i < DQ_TN / 8 * NG; i++) {
    unsigned nb = i / NG, g = i % NG;
    // lane kk * 8 + nn of a tile is column nn: the 8 scales repeat per k row
    aie::vector<bfloat16, 8> s8 = aie::load_v<8>(sc + g * DQ_TN + nb * 8);
    aie::vector<bfloat16, 8> m8 = aie::load_v<8>(mn + g * DQ_TN + nb * 8);
    aie::vector<bfloat16, 16> s16 = aie::concat(s8, s8);
    aie::vector<bfloat16, 16> m16 = aie::concat(m8, m8);
    aie::vector<bfloat16, 32> s32 = aie::concat(s16, s16);
    aie::vector<bfloat16, 32> m32 = aie::concat(m16, m16);
    aie::vector<bfloat16, LANES> sv = aie::concat(s32, s32);
    aie::vector<bfloat16, LANES> mv = aie::concat(m32, m32);
    aie::accum<accfloat, LANES> base;
    base.from_vector(mv);
    base = aie::msc(base, sv, bfloat16(128.0f));
#pragma clang loop unroll(full)
    for (unsigned t = 0; t < DQ_GROUP / 16; t++) {
      aie::vector<uint4, 2 * LANES> qr = aie::load_v<2 * LANES>(q);
      q += LANES;
      auto z = aie::interleave_zip(aie::unpack(qr), bias, 1);
      aie::store_v(o, aie::mac(base, aie::vector_cast<bfloat16>(z.first), sv)
                          .template to_vector<bfloat16>());
      aie::store_v(o + LANES,
                   aie::mac(base, aie::vector_cast<bfloat16>(z.second), sv)
                       .template to_vector<bfloat16>());
      o += 2 * LANES;
    }
  }
}

// bf16 weights in the same stream: a packet holds DQ_TN columns x 32 K of bf16
// in B-tile order [TN/8][4][8 k][8 n]; slot j is K rows 32j..32j+31
// (packing.pack_bf16).
void bf16_pkt(uint8_t *__restrict packed, bfloat16 *__restrict out, int32_t j) {
  constexpr unsigned SUB = 4 * 64;
  const bfloat16 *src = (const bfloat16 *)packed;
  for (unsigned nb = 0; nb < DQ_TN / 8; nb++)
    chess_prepare_for_pipelining {
      bfloat16 *dst = out + nb * (DQ_TK / 8) * 64 + j * SUB;
#pragma clang loop unroll(full)
      for (unsigned v = 0; v < SUB / 32; v++)
        aie::store_v(dst + v * 32, aie::load_v<32>(src + nb * SUB + v * 32));
    }
}

} // extern "C"
