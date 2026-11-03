// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The residual core of the Gemma4 eight-token decoder: dflash's batched RMS
// and residual functions, plus the per-layer-embedding gate's input.

#include "llms/dflash/common/kernels/rms_batch.cc"

extern "C" {

// Chunk `chunk` of every row of x [rows][MODEL_DIM], unnormalized: the input
// of the per-layer-embedding gate.
void raw_chunk_aie(bf16 *__restrict y, bf16 *__restrict x, int rows, int chunk,
                   int width) {
  for (int row = 0; row < rows; ++row)
    for (int j = 0; j < width; j += 32)
      aie::store_v(y + row * width + j,
                   aie::load_v<32>(x + row * MODEL_DIM + chunk * width + j));
}
}

extern "C" {
// x[row][offset + j] += update[row][j] * w[offset + j]: a post-normed sublayer
// output (normalized by the cache core, weighted here).
void residual_acc_w_row_aie(bf16 *__restrict residual, bf16 *__restrict update,
                            bf16 *__restrict w, int row, int offset,
                            int width) {
  aie_round_nearest_even();
  for (int j = 0; j < width; j += 16) {
    auto a = aie::load_v<16>(residual + row * MODEL_DIM + offset + j);
    aie::accum<accfloat, 16> acc =
        aie::mul(aie::load_v<16>(update + row * width + j),
                 aie::load_v<16>(w + offset + j));
    acc = aie::add(acc, a);
    aie::store_v(residual + row * MODEL_DIM + offset + j,
                 acc.to_vector<bf16>());
  }
}
}
