//===- attn_stages.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Stage kernels for the pipelined flash attention in attn_npu2.py, built from
// the kernel_fusion_based microkernels. A hand-off buffer is a [lqp * lkp]
// tile followed by two [lqp] row vectors: the rescale factor r at TAIL_R and
// the running row sum s at TAIL_S.
//
//===----------------------------------------------------------------------===//

#include "attn_npu2.cc"

#define TAIL_R (lqp * lkp)
#define TAIL_S (lqp * lkp + lqp)

extern "C" {

// Softmax stage: copy P and its r and s into the send buffer.
void sm_out(bfloat16 *g, bfloat16 *r, bfloat16 *s, bfloat16 *out) {
  copy_tile(g, out);
  vector_copy_32elems(TAIL_R, r, out);
  vector_copy_32elems(TAIL_S, s, out);
}

// 3-stage PV: gp = gp * r + P V.
void pv_step(bfloat16 *p, bfloat16 *v, bfloat16 *gp, bfloat16 *s) {
  mul_r_gp(p + TAIL_R, gp);
  matmul_g_b_bf16(p, v, gp);
  vector_copy_32elems(0, p + TAIL_S, s);
}

void pv_final(bfloat16 *s, bfloat16 *gp) { div_gp_sp(s, gp); }

// 4-stage PV: out = P V, passing r and s on.
void pv_fresh(bfloat16 *p, bfloat16 *v, bfloat16 *out) {
  zero_fill_gp_bf16(out);
  matmul_g_b_bf16(p, v, out);
  vector_copy_32elems(TAIL_R, p + TAIL_R, out);
  vector_copy_32elems(TAIL_S, p + TAIL_S, out);
}

// 4-stage rescale: acc = acc * r + P V.
void rs_step(bfloat16 *in, bfloat16 *acc, bfloat16 *s) {
  mul_r_gp(in + TAIL_R, acc);
  add_gp_g(in, acc);
  vector_copy_32elems(0, in + TAIL_S, s);
}

void rs_final(bfloat16 *s, bfloat16 *acc) { div_gp_sp(s, acc); }

} // extern "C"
