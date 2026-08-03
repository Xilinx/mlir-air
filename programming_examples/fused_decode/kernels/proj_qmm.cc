// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// AIR-lock-stripped variant of the reference proj_main's GEMV inner loop
// (q4_npu_eXpress / proj_main.cc::linear_proj_iD), split into separate zero /
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
  alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[k / 32]; // 8

  // per-group (32 cols) reduction of x, used for the +min term.
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_UNROLL_FULL
  for (int l = 0; l < k / 32; l++)
    AIE_LOOP_FLATTEN {
      b_col_reduce_add[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
    }

  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x, y_acc, b_col_reduce_add);
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
  constexpr int m = Q4NX_ROW_BLOCK_SIZE;                             // 32
  constexpr int k = Q4NX_COL_BLOCK_SIZE;                             // 256
  alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[k / 32]; // 8

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_UNROLL_FULL
  for (int l = 0; l < k / 32; l++)
    AIE_LOOP_FLATTEN {
      b_col_reduce_add[l] =
          bf16(aie::reduce_add(aie::load_v<32>(x_blk + 32 * l)));
    }

  _qmm_q4k_bf16<m, k>((q4k_block_t *)w, x_blk, y_acc, b_col_reduce_add);
}

// Flush the accumulator to bf16 output (call once after the col-block loop).
void proj_qmm_flush(float *__restrict y_acc, bf16 *__restrict y_out) {
  copy_float_to_bf16<Q4NX_ROW_BLOCK_SIZE>(y_out, y_acc);
}

// the reference proj_main keep_pkt_header flush: write the packet routing id as
// a uint32 at element 14 (the packet HEADER word the stream switch routes by)
// and the payload at element 16, exactly like proj_main.cc
// (`*(uint32*)(y+14)=pkt_id; copy_float_to_bf16<m>(y+16, y_acc)`). The matching
// air.channel.put streams from offset 14 (2 header + ROW_BLOCK payload) on a
// {keep_pkt_header} channel so the kernel-written id (NOT a compiler-stamped
// filter) drives routing. y_out must be sized >= 16 + Q4NX_ROW_BLOCK_SIZE.
void proj_qmm_flush_hdr(float *__restrict y_acc, bf16 *__restrict y_out,
                        unsigned int pkt_id) {
  *reinterpret_cast<unsigned int *>(y_out + 14) = pkt_id;
  copy_float_to_bf16<Q4NX_ROW_BLOCK_SIZE>(y_out + 16, y_acc);
}

// Multi-row-block packet support (nbi_pc>1): a core that produces several
// row-blocks must emit them as ONE packet with a SINGLE header at the front
// (a packet carries one header word; a per-flow keep_pkt_header keeps exactly
// the offset-0 contribution). proj_qmm_flush_row writes row-block i's payload
// at y_payload + i*ROW_BLOCK (no header); proj_qmm_write_hdr writes just the
// 2-word packet id at the front. Layout: [hdr@14 | payload0@16 | payload1@16+RB
// | ...]; the matching put streams from offset 14, size 2 + nbi_pc*ROW_BLOCK.
void proj_qmm_flush_row(float *__restrict y_acc, bf16 *__restrict y_out,
                        int i) {
  // payload region starts at element 16 (matching proj_qmm_flush_hdr).
  copy_float_to_bf16<Q4NX_ROW_BLOCK_SIZE>(y_out + 16 + i * Q4NX_ROW_BLOCK_SIZE,
                                          y_acc);
}

void proj_qmm_write_hdr(bf16 *__restrict y_out, unsigned int pkt_id) {
  *reinterpret_cast<unsigned int *>(y_out + 14) = pkt_id;
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
