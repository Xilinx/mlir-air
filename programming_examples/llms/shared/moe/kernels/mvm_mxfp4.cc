// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GPT-OSS expert GEMV on MXFP4 Q4NX blocks, transcribed from FastFlowLM
// FLM_Xclbin/GPTOSS/gpt_npu_bin/kernel/{mvm_MXFP4.hpp,proj.cc}. FLM's
// MVM_Q4NX_MFP4_block owns the loop and locks; here AIR owns them and calls one
// 32x128 block at a time.
//
// Block (2560 B): e8m0 scales [4 groups][32 rows], 384 B whose first 64 hold
// the 32 row biases (bf16, column block 0 only), then nibbles
// [2 stripes][64 byte cols][16 B].

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "MXFP4.h"
#include "aie_kernel_utils.h"

namespace {

constexpr int Q4NX_ROW = 32;
constexpr int Q4NX_COL = 128;
constexpr int Q4NX_COL_STRIDE = 16;
constexpr int DATA_OFFSET = Q4NX_ROW * (Q4NX_COL / 32) * 4;
constexpr int BIAS_OFFSET = Q4NX_ROW * (Q4NX_COL / 32);

// FLM mvm_MXFP4<32, 128, 16>: y_acc[32] += W(block) . x[128]
inline void mvm_mxfp4(const uint8_t *scalar, const uint8_t *data_block,
                      const bfloat16 *vec_in, float *result_acc_ptr) {
  const uint8_t *cur_block_ptr = data_block;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(Q4NX_ROW / Q4NX_COL_STRIDE, Q4NX_ROW / Q4NX_COL_STRIDE)
  for (int row_idx = 0; row_idx < Q4NX_ROW; row_idx += Q4NX_COL_STRIDE) {
    const bfloat16 *cur_vec_ptr = vec_in;
    aie::accum<accfloat, Q4NX_COL_STRIDE> res_acc;
    res_acc.from_vector(aie::load_v<Q4NX_COL_STRIDE>(result_acc_ptr + row_idx));
    const uint8_t *cur_scalar_ptr = scalar + row_idx;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(Q4NX_COL / 32, Q4NX_COL / 32)
    // Chess unrolls this and schedules across the copies; Peano leaves it
    // rolled and the MACs' latency goes unhidden (FLM's Peano port does this)
    AIE_LOOP_UNROLL(Q4NX_COL / 32)
    for (int col_idx = 0; col_idx < Q4NX_COL; col_idx += 32) {
      aie::vector<bfloat16, 32> in_vec_scalar = aie::load_v<32>(cur_vec_ptr);
      cur_vec_ptr += 32;
      aie::vector<uint8_t, Q4NX_COL_STRIDE> scalar_vec_in =
          aie::load_v<16>(cur_scalar_ptr);
      cur_scalar_ptr += Q4NX_ROW;
      aie::vector<bfloat16, Q4NX_COL_STRIDE> scalar_dequant =
          mxfp4_scalar_to_bf16(scalar_vec_in);

      aie::accum<accfloat, Q4NX_COL_STRIDE> temp_acc;
      temp_acc.from_vector(aie::zeros<float, 16>());
      AIE_LOOP_UNROLL_FULL
      for (int k = 0; k < 16; k++) {
        aie::vector<uint8_t, Q4NX_COL_STRIDE> block_vec_in =
            aie::load_v<Q4NX_COL_STRIDE>(cur_block_ptr);
        cur_block_ptr += Q4NX_COL_STRIDE;
        aie::vector<bfloat16, 32> data_dequant = mxfp4_to_bf16(block_vec_in);
        temp_acc = aie::mac(temp_acc, data_dequant.extract<16>(0),
                            in_vec_scalar.get(2 * k));
        temp_acc = aie::mac(temp_acc, data_dequant.extract<16>(1),
                            in_vec_scalar.get(2 * k + 1));
      }
      res_acc =
          aie::mac(res_acc, temp_acc.to_vector<bfloat16>(), scalar_dequant);
    }
    aie::store_v(result_acc_ptr + row_idx, res_acc.to_vector<float>());
  }
}

} // namespace

extern "C" {

// y_acc = the block's row biases (FLM starts each row block from them).
void mxfp4_bias_init(uint8_t *__restrict w, float *__restrict y_acc) {
  const bfloat16 *bias = (const bfloat16 *)(w + BIAS_OFFSET);
  for (int i = 0; i < Q4NX_ROW; i++)
    y_acc[i] = (float)bias[i];
}

void mxfp4_zero(float *__restrict y_acc) {
  for (int i = 0; i < Q4NX_ROW; i += 16)
    aie::store_v(y_acc + i, aie::zeros<float, 16>());
}

// y_acc[32] += block . x_blk[128]
void mxfp4_acc128(bfloat16 *__restrict x_blk, uint8_t *__restrict w,
                  float *__restrict y_acc) {
  mvm_mxfp4(w, w + DATA_OFFSET, x_blk, y_acc);
}

// y_acc[32] += block . x_full[j*128 .. +128], x resident for the whole GEMV
void mxfp4_acc128_x(bfloat16 *__restrict x_full, uint8_t *__restrict w,
                    float *__restrict y_acc, int j) {
  mvm_mxfp4(w, w + DATA_OFFSET, x_full + j * Q4NX_COL, y_acc);
}

void mxfp4_flush(float *__restrict y_acc, bfloat16 *__restrict y) {
  for (int i = 0; i < Q4NX_ROW; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(y_acc + i));
    aie::store_v(y + i, a.to_vector<bfloat16>());
  }
}

// The same into y[16..48): elements 14-15 carry the packet header the
// compiler writes for the put's destination.
void mxfp4_flush_hdr(float *__restrict y_acc, bfloat16 *__restrict y) {
  mxfp4_flush(y_acc, y + 16);
}

} // extern "C"
