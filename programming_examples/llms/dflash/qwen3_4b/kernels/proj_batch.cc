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
#if PROJ_MM_BATCH == 8
  q4k_mmul_small<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE, 8>(x, scratch, acc);
#elif defined(DFLASH_PROJECTION_2X2) && DFLASH_PROJECTION_2X2
  q4k_mmul_b16_2x2<Q4NX_ROW_BLOCK_SIZE,Q4NX_COL_BLOCK_SIZE>(x,scratch,acc);
#else
  q4k_mmul_groups<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE, PROJ_MM_BATCH>(x, scratch, acc);
#endif
}

// C is four [8 tokens, 8 output channels] tiles. The packet carries each
// token's 32 channels contiguously after the preserved 16-element header.
void proj_qmm_mm_flush_row(float *__restrict acc, bf16 *__restrict packet,
                           int role, int token_stride) {
  aie_round_nearest_even();
  constexpr int width = Q4NX_ROW_BLOCK_SIZE;
  alignas(aie::vector_decl_align) float row[width];
  for (int token = 0; token < PROJ_MM_BATCH; ++token) {
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < width / 8; ++j)
      aie::store_v(row + j * 8, aie::load_v<8>(acc + j * PROJ_MM_BATCH * 8 + token * 8));
    copy_float_to_bf16<width>(packet + 16 + (token * token_stride + role) * width, row);
  }
}
}

// Fixed circuit gathers preserve only the first core's routing word. The
// final output channel declares the matching packet IDs [31, 30, 29].
extern "C" void proj_qmm_mm_flush_routed(float *__restrict acc,
                                        bf16 *__restrict packet,
                                        int destination, int token_stride) {
  proj_qmm_mm_flush_row(acc, packet, 0, token_stride);
  reinterpret_cast<uint32_t *>(packet + 14)[0] = 31 - destination;
}

// Keep the GEMM accumulator resident, but drain one token per short packet.
// Packet routing metadata is still generated from the AIR put's dest operand.
extern "C" void proj_qmm_mm_flush_token(float *__restrict acc,
                                       bf16 *__restrict packet,
                                       int token, int unused) {
  (void)unused;
  aie_round_nearest_even();
  constexpr int width = Q4NX_ROW_BLOCK_SIZE;
  alignas(aie::vector_decl_align) float row[width];
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < width / 8; ++j)
    aie::store_v(row + j * 8, aie::load_v<8>(acc + j * PROJ_MM_BATCH * 8 + token * 8));
  copy_float_to_bf16<width>(packet + 16, row);
}
