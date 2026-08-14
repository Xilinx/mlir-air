// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// AIR-lock-stripped variant of the reference proj_main's GEMV inner loop
// (q4_npu_eXpress / the reference GEMV), split into separate zero /
// accumulate / flush entry points so the float accumulator y_acc is used by ops
// OUTSIDE the col-block (j) loop -- this keeps AIR from sinking the accumulator
// alloc into the j-loop (which would re-allocate it per col-block and destroy
// the accumulation). The accumulator is a caller-provided L1 buffer scoped to
// the row-block (i) loop: zeroed once, accumulated over all col-blocks, flushed
// once. NO in-kernel _lock_acquire -- AIR owns sync.
//
// Math matches the proj_main kernel: w = q*scale + min, y[r] = sum_c
// w[r,c]*x[c].
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "model_spec.h"
#include "q4_k.h"

extern "C" {

// Zero the row-block accumulator (call once before the col-block loop).
void proj_qmm_zero(float *__restrict y_acc, int _arm) {
  (void)_arm; // per-token RTP arm-gate operand (kept alive so AIR emits the arm
              // lock)
  zero_vectorized<float, Q4NX_ROW_BLOCK_SIZE>(y_acc);
}

// PASSTHROUGH stand-in for proj_qmm_acc256: same signature/dataflow (reads
// x_blk and w so neither is DCE'd), but does NO GEMV -- just adds x_blk[0:32]
// into y_acc and touches w[0]. Used by the dataflow-isolation reproducer to
// prove the deadlock is independent of the GEMV compute.
void proj_qmm_pass256(bf16 *__restrict x_blk, bf16 *__restrict w,
                      float *__restrict y_acc) {
  volatile bf16 wkeep = w[0];
  (void)wkeep;
  for (int i = 0; i < Q4NX_ROW_BLOCK_SIZE; i++)
    y_acc[i] += (float)x_blk[i];
}

// Accumulate ONE q4k block (32 rows x 256 cols) into y_acc (pure accumulate;
// y_acc must be pre-zeroed by proj_qmm_zero). The full activation x is RESIDENT
// (sent once, like the reference's broadcast x); this call reads col-block j
// from it.
//   x_full : full activation, K bf16 (resident)
//   j      : col-block index (reads x_full + j*256)
//   w      : one q4k block, 2560 bf16
//   (scales[8][32]++mins[8][32]++qs[2][256][8]) y_acc  : caller-provided float
//   accumulator (32), read-modify-written
void proj_qmm_acc(bf16 *__restrict x_full, int j, bf16 *__restrict w,
                  float *__restrict y_acc) {
  constexpr int m = Q4NX_ROW_BLOCK_SIZE; // 32
  constexpr int k = Q4NX_COL_BLOCK_SIZE; // 256
  bf16 *x = x_full + j * k;
#ifndef Q4_0
  alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[k / 32]; // 8

  // per-group (32 cols) reduction of x, used for the +min term.
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_UNROLL_FULL
  for (int l = 0; l < k / 32; l++)
    AIE_LOOP_FLATTEN {
      b_col_reduce_add[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
    }

  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x, y_acc, b_col_reduce_add);
#else
  // Q4_0: scale-only (no +min term) -> no b_col_reduce, 3-arg form.
  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x, y_acc);
#endif
}

// the reference-streaming variant of proj_qmm_acc: x is ONE 256-element
// col-block (NOT the full resident activation). Matches the reference
// linear_proj_iD's x_ping/x_pong (a single COL_BLOCK pulled per (i,j) via
// ping-pong). Identical MAC; x_blk is x_full+j*256 already sliced by the DMA,
// so no j offset here.
//   x_blk : one col-block of the activation, 256 bf16
//   w     : one q4k block, 2560 bf16
//   y_acc : caller-provided float accumulator (32), read-modify-written
void proj_qmm_acc256(bf16 *__restrict x_blk, bf16 *__restrict w,
                     float *__restrict y_acc) {
  constexpr int m = Q4NX_ROW_BLOCK_SIZE; // 32
  constexpr int k = Q4NX_COL_BLOCK_SIZE; // 256
#ifndef Q4_0
  alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[k / 32]; // 8

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_UNROLL_FULL
  for (int l = 0; l < k / 32; l++)
    AIE_LOOP_FLATTEN {
      b_col_reduce_add[l] =
          bf16(aie::reduce_add(aie::load_v<32>(x_blk + 32 * l)));
    }

  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x_blk, y_acc, b_col_reduce_add);
#else
  // Q4_0: scale-only (no +min term) -> no b_col_reduce, 3-arg form.
  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x_blk, y_acc);
#endif
}

// CACHED-REDUCTION variant of proj_qmm_acc256 (the default; PROJ_RC_CACHE=0
// falls back to the plain one above).
//
// b_col_reduce_add is the per-32-group sum of the ACTIVATION slice, so it
// depends only on the col-block j -- NOT on the row-block i. proj_qmm_acc256
// recomputes it on every (i, j) block because its 8-element result is a
// call-local stack array with nowhere to live between calls. That reduction is
// 171 of the function's 177 bundles and all 64 of its vector stack accesses
// (measured: building with -DQ4_0, which drops the +min term, leaves 6 bundles
// and 0 spills), so the waste is ~171 bundles + ~4 KB of L1 spill traffic per
// block -- L1 traffic that contends with the DMA streaming the next weight
// block into the same tile.
//
// The reference proj_main does the same thing correctly: it keeps a persistent
// b_col_reduce_add[INTERMEDIATE_SIZE/Q4NX_GROUP_SIZE] in its caller frame,
// indexes it by j, and fills it under `if (i == 0)` ("special logic for i == 0,
// avoid recompute if it is repeat"). This is that, with the cache handed in by
// AIR because the AIR kernel is a leaf call and has no caller frame of its own.
//
// Redundancy removed, llama-3.2-1B (row-blocks/phase = I2P*PAIR_ROWS =
// [6,4,32,4] decode, 36 per lm-head wave): 9440 -> 952 reductions per token.
//
//   rc   : caller-owned cache, >= (max col-blocks)*(k/32) bf16, live across the
//          row-block loop and refilled at each new projection. AIR sizes it
//          RCACHE_LEN and pins its alloc at projection scope via
//          proj_qmm_rc_arm below.
//   j    : col-block index -- selects the slot
//   fill : nonzero on the projection's FIRST row-block (computes the slot),
//          zero afterwards (reuses it)
void proj_qmm_acc256_c(bf16 *__restrict x_blk, bf16 *__restrict w,
                       float *__restrict y_acc, bf16 *__restrict rc, int j,
                       int fill) {
  constexpr int m = Q4NX_ROW_BLOCK_SIZE; // 32
  constexpr int k = Q4NX_COL_BLOCK_SIZE; // 256
#ifndef Q4_0
  bf16 *slot = rc + j * (k / 32);
  if (fill) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_UNROLL_FULL
    for (int l = 0; l < k / 32; l++)
      AIE_LOOP_FLATTEN {
        slot[l] = bf16(aie::reduce_add(aie::load_v<32>(x_blk + 32 * l)));
      }
  }
  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x_blk, y_acc, slot);
#else
  (void)rc;
  (void)j;
  (void)fill;
  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x_blk, y_acc);
#endif
}

// Pin the reduce cache at PROJECTION scope. AIR sinks an alloc to the innermost
// region that uses it, and proj_qmm_acc256_c is the cache's only other user --
// which would sink it into the col-block loop and reset it every row-block,
// silently defeating the cache. One call per projection, outside both the
// row-block and col-block loops, keeps it alive across them. Same reason
// proj_qmm_zero/proj_qmm_flush exist as separate entry points for y_acc.
//
// The pin is an IR-level effect: AIR decides where to sink the alloc from the
// operands of this func.call, long before Peano sees the body. So the body only
// has to be a side effect that is not undefined. It must NOT read rc -- on the
// first call the buffer is uninitialized, and reading an indeterminate bf16 is
// UB the compiler is free to resolve by deleting the call outright -- and it
// must not write rc either, since slot 0 is live cache state. Storing the
// POINTER to a volatile satisfies both: a volatile store is a side effect the
// compiler must emit, and it never dereferences rc. (Inline asm is not an
// option here: the Peano AIE2P backend fails to translate it.)
void proj_qmm_rc_arm(bf16 *__restrict rc, int _arm) {
  bf16 *volatile keep = rc;
  (void)keep;
  (void)_arm;
}

// Flush the accumulator to bf16 output (call once after the col-block loop).
void proj_qmm_flush(float *__restrict y_acc, bf16 *__restrict y_out) {
  copy_float_to_bf16<Q4NX_ROW_BLOCK_SIZE>(y_out, y_acc);
}

// Convert row-block i's f32 accumulator to the bf16 payload of the egress
// packet. PAYLOAD ONLY -- this writes no routing header.
//
// Buffer layout is [hdr@14 | payload0@16 | payload1@16+ROW_BLOCK | ...] and the
// matching air.channel.put streams from offset 14, size 2 + nbi_pc*ROW_BLOCK.
// A core producing several row-blocks emits them as ONE packet with a single
// header at the front, so each row-block writes only its own slice and i says
// which.
//
// The header at element 14 used to be written here too, by a separate
// proj_qmm_flush_hdr taking the id as an argument -- an id the design also had
// to spell on the channel, in a second place, with nothing keeping the two in
// step. The compiler emits that store now, from the `dest` operand on the
// air.channel.put, so what is left is plain compute. proj_qmm_flush_hdr was
// exactly this function with i = 0 once the header write went away, and is
// gone.
void proj_qmm_flush_row(float *__restrict y_acc, bf16 *__restrict y_out,
                        int i) {
  copy_float_to_bf16<Q4NX_ROW_BLOCK_SIZE>(y_out + 16 + i * Q4NX_ROW_BLOCK_SIZE,
                                          y_acc);
}

// DEBUG: fill the resident activation X with a constant ON-CHIP, so the proj X
// need not be loaded from DDR via the shim (matching the reference's
// shim=weights-only dataflow). Used by MERGE_CONST_X to isolate the egress
// deadlock from the X-feed / attention-feedback path.
void proj_qmm_fill_x(bf16 *__restrict x, int n) {
  bf16 c = bf16(0.0625f);
  for (int i = 0; i < n; i++)
    x[i] = c;
}
}
