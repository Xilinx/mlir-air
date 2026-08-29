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
#ifdef PROJ_MM_BATCH
#include "q4k_mm.h" // batched projection only; see the bottom of this file
#endif

#ifdef PROJ_DELAY
static inline void proj_probe_delay() {
  volatile int s = 0;
  for (int i = 0; i < PROJ_DELAY; i++)
    s += i;
}
#endif

extern "C" {

// Zero the row-block accumulator (call once before the col-block loop).
void proj_qmm_zero(float *__restrict y_acc, int _arm) {
  aie_round_nearest_even();
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
  aie_round_nearest_even();
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
  aie_round_nearest_even();
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
  aie_round_nearest_even();
#ifdef PROJ_DELAY
  proj_probe_delay();
#endif
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
  aie_round_nearest_even();
#ifdef PROJ_DELAY
  proj_probe_delay();
#endif
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
  aie_round_nearest_even();
  bf16 *volatile keep = rc;
  (void)keep;
  (void)_arm;
}

// Flush the accumulator to bf16 output (call once after the col-block loop).
void proj_qmm_flush(float *__restrict y_acc, bf16 *__restrict y_out) {
  aie_round_nearest_even();
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
  aie_round_nearest_even();
  copy_float_to_bf16<Q4NX_ROW_BLOCK_SIZE>(y_out + 16 + i * Q4NX_ROW_BLOCK_SIZE,
                                          y_acc);
}

// DEBUG: fill the resident activation X with a constant ON-CHIP, so the proj X
// need not be loaded from DDR via the shim (matching the reference's
// shim=weights-only dataflow). Used by MERGE_CONST_X to isolate the egress
// deadlock from the X-feed / attention-feedback path.
void proj_qmm_fill_x(bf16 *__restrict x, int n) {
  aie_round_nearest_even();
  bf16 c = bf16(0.0625f);
  for (int i = 0; i < n; i++)
    x[i] = c;
}

// ---------------------------------------------------------------------------
// BATCHED PROJECTION (DFlash step 2). Behind -DPROJ_MM_BATCH=<B>, so a build
// that does not ask for it is unchanged and does not even parse q4k_mm.h.
//
// Same three entry points as the GEMV above, split the same way and for the
// same reason: y_acc has to live at row-block scope, so it is caller-provided
// and AIR must not sink it into the col-block loop.
//
// NO REDUCE CACHE, and that is the substantive difference. The GEMV never
// materializes W -- it factors the +min term out as min[r,g] * (sum of x over
// group g), which is the entire purpose of b_col_reduce_add and the rc / fill /
// proj_qmm_rc_arm machinery. q4k_unpack_block builds w = q*scale + min
// elementwise instead, because aie::mmul needs a materialized B operand. So the
// batched path drops the cache, the fill bookkeeping and the arm pin, and pays
// for it with a MROWS*KCOL bf16 scratch: 608 bytes of rc traded for 16 KB of
// scratch on qwen3-4b. That trade is the batched path's whole L1 cost.
// ---------------------------------------------------------------------------
#ifdef PROJ_MM_BATCH
void proj_qmm_mm_zero(float *__restrict y_acc, int _arm) {
  (void)_arm; // per-token RTP arm-gate operand, as proj_qmm_zero
  zero_vectorized<float, PROJ_MM_BATCH * Q4NX_ROW_BLOCK_SIZE>(y_acc);
}

// CRITICAL-PATH PROBE: burn PROJ_DELAY units on this core, per weight block.
//
// There is no way to time a core on this part. The AIE trace unit does not work
// here, and `get_cycles()` -- declared in Peano's aie2p_aie_api_compat.h -- has
// no implementation under Peano: it compiles to `jl #_Z10get_cyclesv` and links
// to an undefined symbol. Same shape as chess_storage(), and it would be the
// third time that trap was walked into. So the core is timed INDIRECTLY: give
// one role a known amount of extra work and see whether the dispatch notices.
//
// Sweeping PROJ_DELAY measures this core's SLACK. While the dispatch time is
// flat, the core was idle at least that long per block and is not the critical
// path; past the knee it grows 1:1 and the SLOPE calibrates the unit in
// cycles -- ms/unit / (blocks per layer * layers) -- so no absolute cycle
// counter is needed. The knee, in those calibrated units, is the slack.
//
// volatile forces the loop to survive -O2; nothing else here may be
// -- a delay the optimizer deletes reads as "this core has infinite slack".
//
// IT IS CALLED FROM BOTH THE BATCH-1 AND THE BATCHED PER-BLOCK ENTRY, and the
// definition sits above extern "C" so it can be. It used to sit inside the
// #ifdef PROJ_MM_BATCH block with its only call in proj_qmm_mm_acc, which made
// a batch-1 sweep flat to any delay -- not because the core had slack, but
// because build_template.sh omits -DPROJ_MM_BATCH at batch 1 and the code
// being timed was never compiled in. A flat sweep is the interesting answer
// here, so it is exactly the one worth being sure of.
// Accumulate ONE q4k block across all PROJ_MM_BATCH tokens.
//   x_tile : this col-block's activations for every token, in aie::mmul's A
//            TILE ORDER -- not a plain [BATCH][COL_BLOCK] buffer. See pack_A in
//            q4k_mm_gate.py for the exact order; the memtile owes a strided BD.
//   w      : one packed q4k block, Q4K_BLOCK_BF16 bf16
//   y_acc  : PROJ_MM_BATCH*ROW_BLOCK floats in mmul C tile order, accumulated
//   ws     : ROW_BLOCK*COL_BLOCK bf16 unpack scratch, overwritten every call
void proj_qmm_mm_acc(bf16 *__restrict x_tile, bf16 *__restrict w,
                     float *__restrict y_acc, bf16 *__restrict ws) {
#ifdef PROJ_DELAY
  proj_probe_delay();
#endif
#if defined(PROJ_MM_PROBE) && PROJ_MM_PROBE == 1
  // Diagnostic builds only: skip the mmul and ship the A OPERAND AS DELIVERED,
  // so the accumulator dump (PROJ_FLUSH_PROBE=4) reads out what the X feed
  // actually put in this core's L1 rather than what the arithmetic made of it.
  // Under RMS_CHUNK_PROBE=1 the answer is known exactly: pack_A's order at
  // BATCH 8 is x_tile[i*64 + rr*8 + ss] = X[token rr], so the dump must be
  // 8-float runs scaling 1,2,...,8 and repeating every 64.
  (void)w;
  (void)ws;
  for (int m = 0; m < PROJ_MM_BATCH * Q4NX_ROW_BLOCK_SIZE; m++)
    y_acc[m] = (float)x_tile[m];
#elif defined(PROJ_MM_PROBE) && PROJ_MM_PROBE == 2
  // TIMING ONLY -- split q4k_mm_block in half to see which half the core's
  // per-block time is in. 2 keeps the multiply and drops the dequantize, so
  // the mmul reads whatever the scratch already held; 3 keeps the dequantize
  // and drops the multiply, leaving y_acc at the zero proj_qmm_mm_zero wrote.
  // The results are garbage by construction and only the DISPATCH TIME means
  // anything.
  //
  // Worth doing only because the PROJ_DELAY sweep showed the projection core
  // has no slack at batch 8: with a knee at zero, core time removed shows up in
  // the dispatch 1:1, so this subtraction is a measurement rather than an
  // upper bound. Note 1 is NOT the third point of this partition -- its
  // 256-element scalar store loop is comparable in cost to what it replaces.
  q4k_mmul_any<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE, PROJ_MM_BATCH>(
      x_tile, ws, y_acc);
  (void)w;
#elif defined(PROJ_MM_PROBE) && PROJ_MM_PROBE == 3
  q4k_unpack_block<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(
      (const q4k_block_t *)w, ws);
  (void)x_tile;
  (void)y_acc;
#else
  q4k_mm_block<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE, PROJ_MM_BATCH>(
      (const q4k_block_t *)w, x_tile, y_acc, ws);
#endif
}

// Batched proj_qmm_flush_row: f32 accumulator -> the bf16 payload of the egress
// packet, for every token.
//
// This has to DE-TILE. y_acc comes out of aie::mmul in C tile order -- tile
// (z, j) at (j*rowA + z)*64, row-major [8 tokens][8 rows] inside -- so one
// token's 32 rows are four 8-float runs scattered 64 floats apart, not a
// contiguous 32. The GEMV had nothing to do here because its accumulator was
// already one row-block of one token.
//
// tok_stride is the number of ROW_BLOCKs between consecutive tokens in the
// output packet, i.e. how many row-blocks this core emits. Token-major, so the
// group memtile can gather one row-block across all tokens with a single extra
// BD dimension (sizes=[B, ROW_BLOCK], strides=[tok_stride*ROW_BLOCK, 1]).
// Passed rather than assumed: it is a property of the phase, not the kernel.
void proj_qmm_mm_flush_row(float *__restrict y_acc, bf16 *__restrict y_out,
                           int i, int tok_stride) {
  constexpr int RB = Q4NX_ROW_BLOCK_SIZE;
  constexpr int RA = PROJ_MM_BATCH / 8; // mmul rowA: token blocks
  constexpr int CB = RB / 8;            // mmul colB: row blocks within RB
  // The de-tiling below reads C tile (z, j) at (j*RA + z)*64 and token rr at
  // rr*8 within it. That is aie::mmul<8,8,8>'s layout, and it is also
  // q4k_mmul_small's at batch 8, where RA collapses to 1 -- the two agree
  // exactly there, which is why one formula serves 8, 16 and 32.
  //
  // It does NOT serve batch 4. q4k_mmul_any picks aie::mmul<4,8,8> there,
  // size_C is 32 rather than 64, and RA integer-divides to ZERO so every j
  // would read tile 0. q4k_mm.h is correct at batch 4 (q4k_mm_gate.py --batch 4
  // is bit-exact); only this de-tiling is not, and it would fail by returning a
  // plausible wrong answer rather than by crashing.
  static_assert(PROJ_MM_BATCH % 8 == 0,
                "proj_qmm_mm_flush_row de-tiles for aie::mmul<8,8,8>; batch 4 "
                "needs a size_C=32 variant");
  alignas(aie::vector_decl_align) float tmp[RB];

  // PROJ_FLUSH_PROBE, diagnostic builds only. The batched engine loses 8 of
  // every 64 elements of the LAST token's row on ONE egress round --
  // y_acc[248:256] at batch 8, the last vector this loop reads.
  //   1  run the tokens backwards. If the hole moves to token 0 it is about
  //      being written LAST (the egress reading before the store lands); if it
  //      stays on token 7 it is the token index or the accumulator.
  //      [measured: it STAYS. Not a write-order race.]
  //   2  store a marker instead of the last vector. If the marker reaches the
  //      KV cache, the WRITE lands and the accumulator was zero; if the hole is
  //      still a hole, the write itself is being lost.
  //      [measured: the marker LANDS, at exactly 24..31 of token 7's K.]
  //   3  label every element with the token, the position and the role, and
  //      read the labels out of the KV cache. Proves the addressing BELOW this
  //      point -- both gathers, the id-demux, the L2 transpose, rope's slice,
  //      the KV append. [measured: every label right, all 8 tokens.]
  //   4  ship y_acc RAW instead of de-tiling it, split into PROJ_MM_BATCH
  //      contiguous RB-float chunks and read with SCALAR loads so the dump
  //      cannot inherit whatever the vector path is doing. This is the one that
  //      ended the hunt: role 1's accumulator is a textbook mmul<8,8,8> C (4
  //      tiles of 64, token rr at rr*8, scaling 1..8 under RMS_CHUNK_PROBE)
  //      and role 0's is the same array SHIFTED LEFT BY ONE 8-FLOAT VECTOR with
  //      y_acc[248:256] never written -- a 32-byte-misaligned 512-bit store,
  //      not a token permutation. See l1_align.py and docs/DFlashFeasibility.md.
  for (int tt = 0; tt < PROJ_MM_BATCH; tt++) {
#if defined(PROJ_FLUSH_PROBE) && PROJ_FLUSH_PROBE == 1
    const int t = PROJ_MM_BATCH - 1 - tt;
#else
    const int t = tt;
#endif
    const int z = t / 8, rr = t % 8;
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < CB; j++)
      aie::store_v(tmp + j * 8,
                   aie::load_v<8>(y_acc + (j * RA + z) * 64 + rr * 8));
#if defined(PROJ_FLUSH_PROBE) && PROJ_FLUSH_PROBE == 4
    // Skip the de-tiling entirely and ship y_acc RAW, split into
    // PROJ_MM_BATCH contiguous RB-float chunks. Under RMS_CHUNK_PROBE=1 the
    // accumulator's true tile structure is known exactly -- tile j at j*64,
    // token rr at rr*8 inside it, scaled by (rr+1) -- so a raw dump says
    // whether the ACCUMULATOR is wrong or only the de-tiling that reads it.
    // Emitted slot t therefore carries y_acc[t*RB .. t*RB+RB), which for
    // RB=32 is half a C tile: slot 0 = tile 0 tokens 0-3, slot 1 = tile 0
    // tokens 4-7, slot 2 = tile 1 tokens 0-3, and so on.
    for (int p = 0; p < RB; p++)
      tmp[p] = y_acc[t * RB + p];
#elif defined(PROJ_FLUSH_PROBE) && PROJ_FLUSH_PROBE == 2
    if (t == PROJ_MM_BATCH - 1)
      aie::store_v(tmp + (CB - 1) * 8, aie::broadcast<float, 8>(0.125f));
#elif defined(PROJ_FLUSH_PROBE) && PROJ_FLUSH_PROBE == 3
    // Label every element with WHERE it came from, and read the labels back out
    // of the KV cache. V is copied through rope unrotated, so its labels
    // survive; K's do not. t*32 + p is 0..255, which bf16 holds exactly, and
    // the role goes in the sign.
    for (int p = 0; p < RB; p++)
      tmp[p] = (i ? -1.0f : 1.0f) * (float)(t * RB + p);
#endif
    copy_float_to_bf16<RB>(y_out + 16 + (t * tok_stride + i) * RB, tmp);
  }
}
#endif // PROJ_MM_BATCH
}
