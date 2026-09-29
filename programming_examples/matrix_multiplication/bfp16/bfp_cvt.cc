//===- bfp_cvt.cc - bfp16ebs8 accumulator -> bf16 / f32 drain -------------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// mm_bfp.cc accumulates C in bfp16ebs8. This converts that L1 accumulator to
// the example's chosen output type (bf16 or f32) on the way out.
//
// The block order is PRESERVED: both source and destination stay
// [m/8][n/8][8][8] sub-tile-major, so this is a straight contiguous stream and
// the L1 side of the drain DMA collapses to a single dimension. The
// de-blocking permute into row-major is left to that DMA, where it costs
// nothing extra.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

template <unsigned NELEM, typename T>
static void bfp16_to_float_blocks(const bfp16ebs8 *__restrict src,
                                  T *__restrict dst) {
  static_assert(NELEM % 64 == 0, "tile must be a whole number of 8x8 blocks");
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> in(src);
  for (unsigned i = 0; i < NELEM / 64; ++i) {
    // accum<accfloat,64> is constructible from a block_vector, the same
    // conversion mm_bfp.cc uses in the opposite direction.
    aie::accum<accfloat, 64> acc(in.pop());
    aie::store_v(dst + i * 64, acc.template to_vector<T>());
  }
}

// Zeroing the bfp16ebs8 accumulator. mm_bfp.cc used to export this as
// `zero_kernel` under a ZERO_ONLY macro; that path is gone from the reference
// kernel, so the example carries its own. Writing a zeroed accum as bfp16ebs8
// (rather than memset) keeps the shared-exponent block layout well-formed.
template <int M, int N>
static void zero_blocks(bfp16ebs8 *__restrict cOut) {
  static_assert(M * N % 64 == 0, "tile must be a whole number of 8x8 blocks");
  const aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(cOut);
  for (int i = 0; i < M * N / 64; ++i)
    out << acc.to_vector<bfp16ebs8>();
}

extern "C" {

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  zero_blocks<DIM_M, DIM_N>(cOut);
}

void bfp16_to_bf16_mn(uint8_t *src, bfloat16 *dst) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  bfp16_to_float_blocks<DIM_M * DIM_N, bfloat16>(
      reinterpret_cast<const bfp16ebs8 *>(src), dst);
}

void bfp16_to_f32_mn(uint8_t *src, float *dst) {
  bfp16_to_float_blocks<DIM_M * DIM_N, float>(
      reinterpret_cast<const bfp16ebs8 *>(src), dst);
}

} // extern "C"
