// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GPT-OSS prefill expert dequant, after FastFlowLM
// FLM_Xclbin/GPTOSS/gpt_dequant_mxfp4_npu_bin (dequant_Q4NX_block_32_128_16):
// one 2560-byte MXFP4 block, 32 rows by 128 columns, to bf16 through FLM's
// u4-pair lookup table and e8m0 scales. The output is laid out as the prefill
// GEMM reads its weights rather than as FLM's GEMM does.
//
// Block: bytes 0..127 the e8m0 scales [column group of 32][row]; bytes 512..
// two stripes of 16 rows, column after column, 8 bytes a column (row 2k in the
// low nibble of byte k, row 2k + 1 in the high one).
// Output [step = column / 32][row / 8][column % 32 / 8][column % 8][row % 8].

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "MXFP4.h"
#include "aie_kernel_utils.h"

namespace {

constexpr int ROWS = 32, COLS = 128, STRIPE = 16, DATA = 512;

} // namespace

extern "C" {

void dqp_block(uint8_t *__restrict blk, bfloat16 *__restrict out) {
  for (int s = 0; s < ROWS / STRIPE; s++) {
    const uint8_t *data = blk + DATA + s * (COLS * STRIPE / 2);
    for (int g = 0; g < COLS / 32; g++) {
      aie::vector<uint8_t, 16> e = aie::load_v<16>(blk + g * ROWS + s * STRIPE);
      aie::vector<bfloat16, 16> scale = mxfp4_scalar_to_bf16(e);
      for (int c = g * 32; c < g * 32 + 32; c += 2) {
        aie::vector<bfloat16, 32> v =
            mxfp4_to_bf16(aie::load_v<16>(data + c * (STRIPE / 2)));
        // constant halves: Peano miscompiles a loop-indexed extract
        aie::accum<accfloat, 16> w0 =
            aie::mul(scale, v.template extract<16>(0));
        aie::accum<accfloat, 16> w1 =
            aie::mul(scale, v.template extract<16>(1));
        aie::vector<bfloat16, 16> y0 = w0.template to_vector<bfloat16>();
        aie::vector<bfloat16, 16> y1 = w1.template to_vector<bfloat16>();
        // columns c and c + 1 sit side by side: one store per 8 rows
        bfloat16 *dst = out + (c / 32) * 1024 + (s * 2) * 256 +
                        ((c % 32) / 8) * 64 + (c % 8) * 8;
        aie::store_v(dst, aie::concat(y0.template extract<8>(0),
                                      y1.template extract<8>(0)));
        aie::store_v(dst + 256, aie::concat(y0.template extract<8>(1),
                                            y1.template extract<8>(1)));
      }
    }
  }
}

} // extern "C"
