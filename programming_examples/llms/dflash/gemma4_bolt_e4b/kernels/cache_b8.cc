// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The normalization cache core of the Gemma4 eight-token decoder. Its slab
// holds either a normalized input the projections re-read, or a sublayer's
// output, which it normalizes (Gemma4's post norms) chunk by chunk for the
// residual core.

#include "llms/dflash/common/kernels/norm_cache.cc"
#include "llms/dflash/common/kernels/rms_batch.cc"

// A sublayer's output arrives as it is produced, [chunk][row][CHUNK]: the
// post norm's factors and its normalized chunks in that layout.
constexpr int CHUNK = 512;
constexpr int NCHUNK = MODEL_DIM / CHUNK;

extern "C" {
void cache_scale_rows_aie(float *__restrict factors, bf16 *__restrict slab,
                          int rows) {
  aie_round_nearest_even();
  for (int row = 0; row < rows; ++row) {
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat>();
    for (int c = 0; c < NCHUNK; ++c)
      for (int j = 0; j < CHUNK; j += 16)
        acc = aie::mac_square(
            acc, aie::load_v<16>(slab + (c * rows + row) * CHUNK + j));
    float sum = aie::reduce_add(acc.template to_vector<float>());
    sum = sum * (1.0f / MODEL_DIM) + RMS_NORM_EPS;
    float half = sum * 0.5f;
    uint32_t bits = __builtin_bit_cast(uint32_t, sum);
    bits = 0x5f3759df - (bits >> 1);
    float inv = __builtin_bit_cast(float, bits);
    inv = inv * (1.5f - half * inv * inv);
    inv = inv * (1.5f - half * inv * inv);
    factors[row] = inv;
  }
}

// The chunk normalized, without the weight: the residual core applies it.
void cache_chunk_aie(bf16 *__restrict y, bf16 *__restrict slab,
                     float *__restrict factors, int rows, int chunk) {
  aie_round_nearest_even();
  for (int row = 0; row < rows; ++row)
    for (int j = 0; j < CHUNK; j += 16) {
      auto xv = aie::load_v<16>(slab + (chunk * rows + row) * CHUNK + j);
      aie::vector<float, 16> xf = aie::mul(xv, aie::broadcast<bf16, 16>(1.0f));
      aie::store_v(y + row * CHUNK + j,
                   aie::mul(xf, factors[row]).to_vector<bf16>());
    }
}
}
