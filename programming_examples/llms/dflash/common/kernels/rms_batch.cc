// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fused_decode/kernels/rms_residual.cc"

// The same reduction and two-step reciprocal square root as rms_residual.cc's
// rms_norm. Keep the factor separate so a normalized chunk can be regenerated
// without holding a second activation tensor on the residual tile.
static float batch_rsqrt(const bf16 *x) {
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat>();
  for (int j = 0; j < MODEL_DIM; j += 16)
    acc = aie::mac_square(acc, aie::load_v<16>(x + j));
  float sum = aie::reduce_add(acc.template to_vector<float>());
  sum = sum * (1.0f / MODEL_DIM) + RMS_NORM_EPS;
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

// y = x * w * factor, for at most eight rows. The FP32 product x * w is exact
// in two BF16 limbs and the factor is split into two as well, so the scaling
// is native BF16 macs; an FP32 vector-by-scalar multiply is emulated.
void rms_chunk_aie(bf16 *__restrict y, bf16 *__restrict x, bf16 *__restrict w,
                   float *__restrict factors, int batch, int chunk, int width) {
  aie_round_nearest_even();
  aie::accum<accfloat, 16> fa(aie::load_v<8>(factors).template grow<16>());
  aie::vector<bf16, 16> f_hi = fa.template to_vector<bf16>();
  aie::vector<bf16, 16> f_lo =
      aie::sub(fa, aie::accum<accfloat, 16>(f_hi)).template to_vector<bf16>();
  for (int row = 0; row < batch; ++row) {
    const bf16 r_hi = f_hi[row], r_lo = f_lo[row];
    const bf16 *xr = x + row * MODEL_DIM + chunk * width;
    const bf16 *wr = w + chunk * width;
    bf16 *yr = y + row * width;
    AIE_PREPARE_FOR_PIPELINING AIE_LOOP_MIN_ITERATION_COUNT(
        4) for (int j = 0; j < width; j += 32) {
      aie::accum<accfloat, 32> xw =
          aie::mul(aie::load_v<32>(xr + j), aie::load_v<32>(wr + j));
      aie::vector<bf16, 32> hi = xw.template to_vector<bf16>();
      aie::vector<bf16, 32> lo =
          aie::sub(xw, aie::accum<accfloat, 32>(hi)).template to_vector<bf16>();
      aie::accum<accfloat, 32> o = aie::mul(hi, r_hi);
      o = aie::mac(o, lo, r_hi);
      o = aie::mac(o, hi, r_lo);
      aie::store_v(yr + j, o.template to_vector<bf16>());
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
