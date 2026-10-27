// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Adapted from fused_decode/kernels/rms_residual.cc.
#include <aie_api/aie.hpp>
#include <cstdint>

extern "C" void scaled_residual_bf16(bfloat16 *out, const bfloat16 *branch,
                                     const bfloat16 *residual, int32_t count,
                                     float multiplier) {
  for (int i = 0; i < count; i += 16) {
    auto a = aie::load_v<16>(branch + i);
    auto b = aie::load_v<16>(residual + i);
    aie::vector<float, 16> af = aie::mul(a, (bfloat16)1.0f).to_vector<float>();
    auto scaled = aie::mul(af, multiplier).to_vector<bfloat16>();
    aie::store_v(out + i, aie::add(scaled, b));
  }
}
