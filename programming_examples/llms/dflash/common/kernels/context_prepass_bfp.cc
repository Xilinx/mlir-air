// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Adapted from matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc:
// same BF16/BFP 8x8x8 primitive and FP32 partial sums, one row tile at B8.
#include "context_prepass.cc"

#ifndef BFP_PREPASS_HIDDEN
#define BFP_PREPASS_HIDDEN 2560
#endif

#ifndef BFP_PREPASS_K
#define BFP_PREPASS_K 512
#endif
#ifndef BFP_PREPASS_N
#define BFP_PREPASS_N 32
#endif

extern "C" void bfp_prepass_zero(float *c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < 8 * BFP_PREPASS_N; i += 16)
    aie::store_v(c + i, aie::zeros<float, 16>());
}

extern "C" void bfp_prepass_project(const bfloat16 *x, const bfloat16 *w,
                                    float *c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  // The example's A layout is [K/8, M/8, 8, 8]. Pack a single M tile.
  alignas(aie::vector_decl_align) bfloat16 packed[8 * BFP_PREPASS_K];
  for (int k = 0; k < BFP_PREPASS_K / 8; ++k)
    for (int row = 0; row < 8; ++row)
      aie::store_v(packed + k * 64 + row * 8,
                   aie::load_v<8>(x + row * BFP_PREPASS_K + k * 8));
  constexpr int kb = BFP_PREPASS_K / 8;
  for (int col = 0; col < BFP_PREPASS_N / 8; col += 4) {
    MMUL c0(aie::load_v<64>(c + (col + 0) * 64));
    MMUL c1(aie::load_v<64>(c + (col + 1) * 64));
    MMUL c2(aie::load_v<64>(c + (col + 2) * 64));
    MMUL c3(aie::load_v<64>(c + (col + 3) * 64));
    for (int k = 0; k < kb; ++k) {
      auto a = aie::load_v<64>(packed + k * 64);
      c0.mac(a, aie::load_v<64>(w + ((col + 0) * kb + k) * 64));
      c1.mac(a, aie::load_v<64>(w + ((col + 1) * kb + k) * 64));
      c2.mac(a, aie::load_v<64>(w + ((col + 2) * kb + k) * 64));
      c3.mac(a, aie::load_v<64>(w + ((col + 3) * kb + k) * 64));
    }
    aie::store_v(c + (col + 0) * 64, c0.to_vector<float>());
    aie::store_v(c + (col + 1) * 64, c1.to_vector<float>());
    aie::store_v(c + (col + 2) * 64, c2.to_vector<float>());
    aie::store_v(c + (col + 3) * 64, c3.to_vector<float>());
  }
}

extern "C" void bfp_prepass_flush(const float *c, float *row_major) {
  for (int row = 0; row < 8; ++row)
    for (int col = 0; col < BFP_PREPASS_N / 8; ++col)
      aie::store_v(row_major + row * BFP_PREPASS_N + col * 8,
                   aie::load_v<8>(c + col * 64 + row * 8));
}

extern "C" void bfp_prepass_norm(const float *x, const float *weight,
                                 bfloat16 *out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float scale = factor<BFP_PREPASS_HIDDEN>(x);
  for (int j = 0; j < BFP_PREPASS_HIDDEN; j += 16) {
    aie::vector<float, 16> normalized = aie::mul(aie::load_v<16>(x + j), scale);
    aie::accum<accfloat, 16> result =
        aie::mul(normalized, aie::load_v<16>(weight + j));
    aie::store_v(out + j, result.to_vector<bfloat16>());
  }
}
