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
// At most 16 rows. The factor is split into two BF16 limbs so the scaling is
// two native macs.
void cache_chunk_aie(bf16 *__restrict y, bf16 *__restrict slab,
                     float *__restrict factors, int rows, int chunk) {
  aie_round_nearest_even();
  // factors holds only `rows` values: pad a local copy for the vector load
  alignas(aie::vector_decl_align) float f[16] = {};
  for (int row = 0; row < rows; ++row)
    f[row] = factors[row];
  aie::accum<accfloat, 16> fa(aie::load_v<16>(f));
  aie::vector<bf16, 16> f_hi = fa.template to_vector<bf16>();
  aie::vector<bf16, 16> f_lo =
      aie::sub(fa, aie::accum<accfloat, 16>(f_hi)).template to_vector<bf16>();
  for (int row = 0; row < rows; ++row) {
    const bf16 r_hi = f_hi[row], r_lo = f_lo[row];
    const bf16 *xr = slab + (chunk * rows + row) * CHUNK;
    bf16 *yr = y + row * CHUNK;
    AIE_PREPARE_FOR_PIPELINING AIE_LOOP_MIN_ITERATION_COUNT(
        4) for (int j = 0; j < CHUNK; j += 32) {
      aie::vector<bf16, 32> xv = aie::load_v<32>(xr + j);
      aie::accum<accfloat, 32> o = aie::mul(xv, r_hi);
      o = aie::mac(o, xv, r_lo);
      aie::store_v(yr + j, o.template to_vector<bf16>());
    }
  }
}
}
