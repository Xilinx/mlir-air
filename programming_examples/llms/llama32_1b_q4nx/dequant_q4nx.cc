//===- dequant_q4nx.cc - Q4NX uint4 -> bfloat16 weight dequant ------------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// On-device Q4NX dequant (the reference's dequant.xclbin mechanism, w = q*scale
// + min). Adapted from dequant_awq/dequant.cc: MIN (bf16) replaces the AWQ
// zero-point, and the dequant is  w = q*scale + min  (not (q - z)*s). One call
// dequantizes ONE weight row of DIM_K nibbles.
//
// Packed L1 row layout (Q4NX, group size DIM_GS):
//   [ Q   : DIM_K/2 bytes uint8 -- two uint4 per byte (low nibble = even k) ]
//   [ S   : DIM_K/DIM_GS bf16   -- per-group scale ]
//   [ MIN : DIM_K/DIM_GS bf16   -- per-group min ]
// Output: DIM_K bfloat16 row.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_K
#define DIM_K 2048
#endif
#ifndef DIM_GS
#define DIM_GS 32
#endif

static_assert(DIM_K % 2 == 0, "DIM_K must be even");
static_assert(DIM_K % DIM_GS == 0, "DIM_K must be a multiple of DIM_GS");
static_assert(DIM_GS % 32 == 0,
              "DIM_GS must be a multiple of 32 (vector width)");

template <unsigned k, unsigned gs, unsigned r = 32>
static void
dequant_q4nx_bf16_impl(uint8_t *__restrict q, bfloat16 *__restrict s,
                       bfloat16 *__restrict mn, bfloat16 *__restrict out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  static_assert(gs % r == 0, "group size must be multiple of inner vector r");
  constexpr unsigned NG = k / gs;
  constexpr unsigned NSUB = gs / r;

  for (unsigned g = 0; g < NG; g++) {
    bfloat16 sv = s[g];
    aie::vector<bfloat16, r> mnv = aie::broadcast<bfloat16, r>(mn[g]);
#pragma clang loop unroll(full)
    for (unsigned i = 0; i < NSUB; i++) {
      const unsigned base = g * gs + i * r;
      aie::vector<uint8, r / 2> pk = aie::load_v<r / 2>(q + base / 2);
      aie::vector<int8, r> w_i8 =
          pk.template cast_to<uint4>().template unpack_sign<int8>(false);
      aie::vector<bfloat16, r> w_bf16 = aie::to_float<bfloat16>(w_i8, 0);
      // w*scale + min : mac(min, w, scale) accumulates q*scale onto the min
      // base.
      aie::accum<accfloat, r> acc;
      acc.from_vector(mnv);
      acc = aie::mac(acc, w_bf16, sv);
      aie::store_v(out + base, acc.template to_vector<bfloat16>());
    }
  }
}

extern "C" {

void dequant_q4nx_bf16(uint8_t *__restrict packed, bfloat16 *__restrict out) {
  constexpr unsigned Q_BYTES = DIM_K / 2;
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * 2;
  uint8_t *q = packed;
  bfloat16 *s = reinterpret_cast<bfloat16 *>(packed + Q_BYTES);
  bfloat16 *mn = reinterpret_cast<bfloat16 *>(packed + Q_BYTES + S_BYTES);
  dequant_q4nx_bf16_impl<DIM_K, DIM_GS, 32>(q, s, mn, out);
}

} // extern "C"
