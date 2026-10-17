// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Key-head RMSNorm and RoPE helpers for the BF16/BFP context prepass.
#include <aie_api/aie.hpp>
#include <cstdint>

static float inverse_sqrt(float v) {
  uint32_t bits = __builtin_bit_cast(uint32_t, v);
  float y = __builtin_bit_cast(float, 0x5f3759dfu - (bits >> 1));
  const float half = 0.5f * v;
  y = y * (1.5f - half * y * y);
  y = y * (1.5f - half * y * y);
  y = y * (1.5f - half * y * y);
  return y;
}

template <int N>
static float factor(const float *x) {
  aie::accum<accfloat, 16> sum = aie::zeros<accfloat>();
  for (int j = 0; j < N; j += 16) {
    auto value = aie::load_v<16>(x + j);
    sum = aie::mac(sum, value, value);
  }
  return inverse_sqrt(aie::reduce_add(sum.to_vector<float>()) / N + 1.0e-6f);
}

extern "C" {
void prepass_key(const float *key, const float *__restrict weight,
                 const float *__restrict lut, float *out) {
  const float scale = factor<128>(key);
  for (int j = 0; j < 64; j += 16) {
    aie::vector<float, 16> left = aie::mul(aie::load_v<16>(key + j), scale);
    aie::vector<float, 16> right =
        aie::mul(aie::load_v<16>(key + j + 64), scale);
    left = aie::mul(left, aie::load_v<16>(weight + j));
    right = aie::mul(right, aie::load_v<16>(weight + j + 64));
    auto cosine = aie::load_v<16>(lut + j);
    auto sine = aie::load_v<16>(lut + j + 64);
    aie::vector<float, 16> lc = aie::mul(left, cosine);
    aie::vector<float, 16> rs = aie::mul(right, sine);
    aie::vector<float, 16> rc = aie::mul(right, cosine);
    aie::vector<float, 16> ls = aie::mul(left, sine);
    aie::store_v(out + j, aie::sub(lc, rs));
    aie::store_v(out + j + 64, aie::add(rc, ls));
  }
}
}

extern "C" void prepass_key_heads(const float *key,
                                  const float *__restrict weight,
                                  const float *__restrict lut, float *out) {
  for (int head = 0; head < 8; ++head)
    prepass_key(key + head * 128, weight, lut, out + head * 128);
}

extern "C" void prepass_key_row(float *kv, const float *parameters) {
  for (int layer = 0; layer < 5; ++layer)
    prepass_key_heads(kv + layer * 2048, parameters + layer * 128,
                      parameters + 640, kv + layer * 1024);
}
