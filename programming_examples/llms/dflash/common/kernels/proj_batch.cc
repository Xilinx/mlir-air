// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fused_decode/kernels/proj_qmm.cc"
#include "q4k_batch.h"

static_assert(PROJ_MM_BATCH == 8 || PROJ_MM_BATCH == 16);

extern "C" {
void proj_qmm_mm_zero(float *__restrict acc, int arm) {
  (void)arm;
  aie_round_nearest_even();
  zero_vectorized<float, PROJ_MM_BATCH * Q4NX_ROW_BLOCK_SIZE>(acc);
}

void proj_qmm_mm_acc(bf16 *__restrict x, bf16 *__restrict packed,
                     float *__restrict acc, bf16 *__restrict scratch) {
  aie_round_nearest_even();
  q4k_unpack_block<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(
      reinterpret_cast<const q4k_block_t *>(packed), scratch);
#if PROJ_MM_BATCH == 16
  q4k_mmul_b16_2x2<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(x, scratch, acc);
#else
  q4k_mmul_small<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE, 8>(x, scratch, acc);
#endif
}

} // extern "C"

// Keep the GEMM accumulator resident, but drain one token per short packet.
// Packet routing metadata is still generated from the AIR put's dest operand.
extern "C" void proj_qmm_mm_flush_token(float *__restrict acc,
                                        bf16 *__restrict packet, int token,
                                        int unused) {
  (void)unused;
  aie_round_nearest_even();
  constexpr int width = Q4NX_ROW_BLOCK_SIZE;
  alignas(aie::vector_decl_align) float row[width];
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < width / 8; ++j)
    aie::store_v(row + j * 8,
                 aie::load_v<8>(acc + j * PROJ_MM_BATCH * 8 + token * 8));
  copy_float_to_bf16<width>(packet + 16, row);
}
