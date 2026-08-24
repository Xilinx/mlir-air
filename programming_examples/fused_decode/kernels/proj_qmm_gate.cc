// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Runs BOTH projection paths on the same weights, in one launch, and hands both
// results back: the GEMV the decode ships today, once per token, and the
// batched matmul meant to replace it. proj_qmm_gate.py compares them.
//
// This is the comparison q4k_mm.h's header asked for and q4k_mm_gate.py does
// not make. That gate proves the batched kernel computes what numpy says it
// computes; it says nothing about whether that agrees with _qmm_q4k_bf16 on the
// same weights -- and it cannot, because the two kernels do genuinely different
// arithmetic. The GEMV factors the +min term out through a per-group reduction
// of the activation and never builds W; the batched path materializes
// w = q*scale + min and multiplies it. Same maths on paper, different roundings
// in different places, and only a device run settles how far apart they land.
//
// Doing both in ONE launch is the point: same L1, same packed weights, same
// activations, no host round trip in between. A difference is then attributable
// to the kernels and nothing else.
//
// proj_qmm.cc is #included rather than linked because a herd carries exactly
// one link_with object. That also guarantees the entry points under test are
// the engine's own source and not a copy.
#ifndef PROJ_MM_BATCH
#define PROJ_MM_BATCH 16
#endif
#ifndef GATE_NBLK
#define GATE_NBLK 1
#endif
#define GATE_BATCH PROJ_MM_BATCH

#include "proj_qmm.cc"

extern "C" {

// One token's col-block, pulled back out of the mmul A tile order into the
// plain [COL_BLOCK] the GEMV wants. The inverse of pack_A, restricted to a
// single (token, block).
//
// Both paths therefore read the SAME activation buffer, which is the point:
// there is no second copy that could differ, and no plain [BATCH][K] buffer
// resident at all. That is also what makes NBLK=2 fit -- holding both a plain
// and a tiled activation costs 16 KB at batch 16 and puts the design over L1,
// while this scratch is 512 bytes.
//
// If it were wrong the GEMV would disagree with exact fp32 by orders of
// magnitude rather than by a fraction of a percent, so it needs no gate of its
// own; the GEMV column of the report is its check.
static inline void detile_token(const bf16 *__restrict x_tile,
                                bf16 *__restrict x_blk, int t, int b) {
  constexpr int k = Q4NX_COL_BLOCK_SIZE;
  constexpr int rowA = GATE_BATCH / 8;
  const int z = t / 8, rr = t % 8;
  const bf16 *src = x_tile + b * (GATE_BATCH * k) + z * 64 + rr * 8;
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < k / 8; i++)
    aie::store_v(x_blk + i * 8, aie::load_v<8>(src + i * rowA * 64));
}

// GEMV reference: the shipping path, run once per token.
//   x_tile : [GATE_NBLK][GATE_BATCH x COL_BLOCK] in mmul A tile order
//   w      : GATE_NBLK packed q4k blocks
//   y_out  : [GATE_BATCH][ROW_BLOCK] bf16
//   y_acc  : ROW_BLOCK floats of scratch
//   rc     : reduce cache, GATE_NBLK*(COL_BLOCK/32) bf16
//   x_blk  : COL_BLOCK bf16 de-tile scratch
void proj_gate_gemv(bf16 *__restrict x_tile, bf16 *__restrict w,
                    bf16 *__restrict y_out, float *__restrict y_acc,
                    bf16 *__restrict rc, bf16 *__restrict x_blk) {
  constexpr int m = Q4NX_ROW_BLOCK_SIZE;
  for (int t = 0; t < GATE_BATCH; t++) {
    proj_qmm_zero(y_acc, 0);
    for (int b = 0; b < GATE_NBLK; b++) {
      detile_token(x_tile, x_blk, t, b);
      // fill = 1 every time: the reduce cache is keyed by col-block and the
      // activation changes per token, so it is stale the moment t advances.
      // In the engine that condition is `i == 0`, because there the row-block
      // loop is what repeats over a fixed activation.
      proj_qmm_acc256_c(x_blk, w + b * Q4K_BLOCK_BF16, y_acc, rc, b, 1);
    }
    copy_float_to_bf16<m>(y_out + t * m, y_acc);
  }
}

// Batched path: the same weights, all tokens at once.
//   x_tile : [GATE_NBLK][GATE_BATCH x COL_BLOCK] in mmul A tile order
//   y_acc  : GATE_BATCH*ROW_BLOCK floats
//   ws     : ROW_BLOCK*COL_BLOCK bf16 unpack scratch
void proj_gate_mm(bf16 *__restrict x_tile, bf16 *__restrict w,
                  bf16 *__restrict y_out, float *__restrict y_acc,
                  bf16 *__restrict ws) {
  constexpr int k = Q4NX_COL_BLOCK_SIZE;
  proj_qmm_mm_zero(y_acc, 0);
  for (int b = 0; b < GATE_NBLK; b++)
    proj_qmm_mm_acc(x_tile + b * (GATE_BATCH * k), w + b * Q4K_BLOCK_BF16,
                    y_acc, ws);
  // tok_stride = 1: one row-block per core here, so the packet is a plain
  // [token][ROW_BLOCK]. y_out is biased by -16 to cancel the header slot the
  // engine's packet carries and this test does not.
  proj_qmm_mm_flush_row(y_acc, y_out - 16, 0, 1);
}

} // extern "C"
