// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Phi LayerNorm, adapted from the AIR FP32-accumulating norm primitive.
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
extern "C" void phi_norm(bf16 *y, bf16 *x, bf16 *w, bf16 *b, int arm) {
  aie_round_nearest_even();
  aie::accum<accfloat, 16> sum = aie::zeros<accfloat, 16>();
  for (int i = 0; i < 2560; i += 16)
    sum = aie::mac(sum, aie::load_v<16>(x + i), (bf16)1.0f);
  float mean = aie::reduce_add(sum.to_vector<float>()) * (1.0f / 2560.0f);
  aie::accum<accfloat, 16> sq = aie::zeros<accfloat, 16>();
  for (int i = 0; i < 2560; i += 16) {
    auto z = aie::sub(
        aie::mul(aie::load_v<16>(x + i), (bf16)1.0f).to_vector<float>(), mean);
    sq = aie::mac(sq, z, z);
  }
  float variance =
      aie::reduce_add(sq.to_vector<float>()) * (1.0f / 2560.0f) + 1e-5f;
  union {
    float f;
    unsigned u;
  } inv;
  inv.f = variance;
  inv.u = 0x5f3759df - (inv.u >> 1);
  for (int j = 0; j < 3; j++)
    inv.f *= 1.5f - 0.5f * variance * inv.f * inv.f;
  for (int i = 0; i < 2560; i += 16) {
    auto z = aie::sub(
        aie::mul(aie::load_v<16>(x + i), (bf16)1.0f).to_vector<float>(), mean);
    auto n = aie::mul(z, inv.f).to_vector<float>();
    auto wf = aie::mul(aie::load_v<16>(w + i), (bf16)1.0f).to_vector<float>();
    auto bf = aie::mul(aie::load_v<16>(b + i), (bf16)1.0f).to_vector<float>();
    aie::accum<accfloat, 16> acc(bf);
    aie::store_v(y + i, aie::mac(acc, n, wf).to_vector<bf16>());
  }
  for (int i = 2560; i < 3072; i += 16)
    aie::store_v(y + i, aie::zeros<bf16, 16>());
  y[2560] = (bf16)1.0f; // Exact constant input for affine bias columns.
}
extern "C" void phi_merge(bf16 *y, bf16 *res, bf16 *attn, bf16 *ffn) {
  aie_round_nearest_even();
  for (int i = 0; i < 3072; i += 16) {
    auto branches =
        aie::add(aie::load_v<16>(attn + i), aie::load_v<16>(ffn + i));
    aie::store_v(y + i, aie::add(branches, aie::load_v<16>(res + i)));
  }
}
extern "C" void phi_save_attention(bf16 *saved, bf16 *received) {
  for (int i = 0; i < 3072; i += 16)
    aie::store_v(saved + i, aie::load_v<16>(received + i));
}
// Unused head arm remains linkable; production excludes all head waves.
extern "C" void rms_norm_aie(bf16 *y, bf16 *x, bf16 *w, int arm) {
  for (int i = 0; i < 3072; i++)
    y[i] = x[i];
}
extern "C" void residual_add_aie(bf16 *y, bf16 *x, bf16 *z) {
  for (int i = 0; i < 3072; i++)
    y[i] = (bf16)((float)x[i] + (float)z[i]);
}
extern "C" void phi_norm_packed(bf16 *y, bf16 *x, bf16 *w, int arm) {
  phi_norm(y, x, w, w + 3072, arm);
}
extern "C" void phi_merge_packed(bf16 *y, bf16 *x, bf16 *branches) {
  phi_merge(y, x, branches, branches + 3072);
}
