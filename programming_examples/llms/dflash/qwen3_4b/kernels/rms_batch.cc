// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fused_decode/kernels/rms_residual.cc"

// The same reduction and two-step reciprocal square root as upstream rms_norm.
// Keep the factor separate so a normalized chunk can be regenerated without
// retaining a second 40 KiB activation tensor on the residual tile.
static float batch_rsqrt(const bf16 *x) {
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat>();
  for (int j = 0; j < MODEL_DIM; j += 16)
    acc = aie::mac_square(acc, aie::load_v<16>(x + j));
  float sum = aie::reduce_add(acc.template to_vector<float>());
  sum = sum * (1.0f / MODEL_DIM) + 1.0e-6f;
  float half = sum * 0.5f;
  uint32_t bits = __builtin_bit_cast(uint32_t, sum);
  bits = 0x5f3759df - (bits >> 1);
  float inv = __builtin_bit_cast(float, bits);
  inv = inv * (1.5f - half * inv * inv);
  inv = inv * (1.5f - half * inv * inv);
  return inv;
}

extern "C" {
void rms_scale_row_aie(float *__restrict factors, bf16 *__restrict x, int row,
                       int arm) {
  (void)arm;
  aie_round_nearest_even();
  factors[row] = batch_rsqrt(x + row * MODEL_DIM);
}

void rms_chunk_aie(bf16 *__restrict y, bf16 *__restrict x, bf16 *__restrict w,
                   float *__restrict factors, int batch, int chunk, int width) {
  aie_round_nearest_even();
  for (int row = 0; row < batch; ++row) {
    for (int j = 0; j < width; j += 16) {
      auto xv = aie::load_v<16>(x + row * MODEL_DIM + chunk * width + j);
      auto wv = aie::load_v<16>(w + chunk * width + j);
      aie::vector<float, 16> product = aie::mul(xv, wv);
      aie::vector<bf16, 16> out = aie::mul(product, factors[row]);
      aie::store_v(y + row * width + j, out);
    }
  }
}

void residual_acc_row_aie(bf16 *__restrict residual, bf16 *__restrict update,
                          int row, int offset, int width) {
  aie_round_nearest_even();
  for (int j = 0; j < width; j += 16) {
    auto a = aie::load_v<16>(residual + row * MODEL_DIM + offset + j);
    auto b = aie::load_v<16>(update + row * width + j);
    aie::store_v(residual + row * MODEL_DIM + offset + j, aie::add(a, b));
  }
}
}
