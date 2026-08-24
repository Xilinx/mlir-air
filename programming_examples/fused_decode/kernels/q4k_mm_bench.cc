// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Static-cost bench for the batched q4k matmul (q4k_mm.h). One exported symbol
// per measurement point; bench_q4k_mm.py compiles this, disassembles each
// symbol, and counts VLIW bundles.
//
// The three points that matter, per 32x256 weight block:
//   q4k_bench_unpack      unpack only              -> the INTERCEPT
//   q4k_bench_mm<BATCH>   unpack + mmul at BATCH   -> intercept + slope*BATCH
//   q4k_bench_mmul<BATCH> mmul only                -> the SLOPE, isolated
//
// The batch-1 baseline is the existing decode GEMV (_qmm_q4k_bf16 in q4_k.h),
// benched from the same script via proj_qmm.o.
//
// STATIC COST ONLY, NOT NUMERICALLY VALIDATED -- see the q4k_mm.h header.
#include "q4k_mm.h"

#ifndef BENCH_MROWS
#define BENCH_MROWS 32
#endif
#ifndef BENCH_KCOL
#define BENCH_KCOL 256
#endif
// Register blocking of the multiply: RB A tiles x CB B tiles held live across
// the contraction, so RB*CB accumulators. 2x2 is mm_aie2p's choice.
#ifndef BENCH_RB
#define BENCH_RB 2
#endif
#ifndef BENCH_CB
#define BENCH_CB 2
#endif

extern "C" {

// Intercept: unpack a 32x256 block, no multiply.
void q4k_bench_unpack(const q4k_block_t *A, bf16 *__restrict W) {
  q4k_unpack_block<BENCH_MROWS, BENCH_KCOL>(A, W);
}

// unpack + mmul, one point per batch.
#define BENCH_MM(B)                                                            \
  void q4k_bench_mm_b##B(const q4k_block_t *A, const bf16 *__restrict Bt,      \
                         float *__restrict C, bf16 *__restrict W) {            \
    q4k_mm_block<BENCH_MROWS, BENCH_KCOL, B>(A, Bt, C, W);                     \
  }                                                                            \
  void q4k_bench_mmul_b##B(const bf16 *__restrict W, const bf16 *__restrict Bt,\
                           float *__restrict C) {                              \
    /* Activations first: pA is Bt, pB is the unpacked W. Bundle counts are     \
       unaffected -- rowA/colA/colB come from the template arguments, not from  \
       which pointer feeds which slot, so every measurement taken with the      \
       operands the other way round still stands. */                           \
    q4k_mmul_any<BENCH_MROWS, BENCH_KCOL, B>(Bt, W, C);                        \
  }

// 4 and 8 use q4k_mmul_small (1x4 blocking, rowA = 1); 16 and 32 use the 2x2
// q4k_mmul. q4k_mmul_any picks, so the bench spans the whole usable range --
// which now starts at 4, because that is the block size the iteration model
// actually recommends (dflash_blocksize.py).
BENCH_MM(4)
BENCH_MM(8)
BENCH_MM(16)
BENCH_MM(32)

// Alternative register blocking, off by default: it roughly doubles the
// unrolled compile time and it already measured WORSE (see the q4k_mmul_2x4
// header). Build with -DBENCH_BLK24 to re-check.
#ifdef BENCH_BLK24
#define BENCH_MM24(B)                                                          \
  void q4k_bench_mmul24_b##B(const bf16 *__restrict W,                         \
                             const bf16 *__restrict Bt, float *__restrict C) { \
    q4k_mmul_2x4<BENCH_MROWS, BENCH_KCOL, B>(Bt, W, C);                        \
  }
BENCH_MM24(16)
BENCH_MM24(32)
#endif

// NCHUNK contraction chunks through one scratch, accumulated into one C.
// Compared against NCHUNK * (unpack + mmul) to see whether splitting the
// contraction costs anything -- see the q4k_mm_chunked header comment.
//
// ONE batch per build, chosen by BENCH_CHUNK_BATCH. The bodies here are
// always_inline'd and so are duplicated NCHUNK times rather than shared, which
// makes this by far the most expensive thing in the file: emitting both 16 and
// 32 put the compile past 25 minutes without finishing.
#ifdef BENCH_CHUNKS
#ifndef BENCH_CHUNK_BATCH
#define BENCH_CHUNK_BATCH 16
#endif
void q4k_bench_chunked(const q4k_block_t *A, const bf16 *__restrict Bt,
                       float *__restrict C, bf16 *__restrict W) {
  q4k_mm_chunked<BENCH_MROWS, BENCH_KCOL, BENCH_CHUNK_BATCH, BENCH_CHUNKS>(
      A, Bt, C, W);
}
#endif

} // extern "C"
