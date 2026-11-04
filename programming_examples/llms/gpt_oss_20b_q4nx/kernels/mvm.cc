// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The MVM cores' kernels: FLM's MVM cores run every projection of a GPT-OSS
// layer -- the Q4_1 attention projections through AIR's proj_qmm, the MXFP4
// experts through mvm_mxfp4 -- so both go in the one object a core links.
// Build with -DMODEL_TYPE=GPT_OSS_20B.

#include "mvm_mxfp4.cc"
#include "proj_qmm.cc"

extern "C" {

// One weight block of a round, cb its column block; the round's first block
// starts the accumulator. Lets a core loop over a phase's blocks in pairs.
void proj_qmm_step(bf16 *__restrict x_full, int cb, bf16 *__restrict w,
                   float *__restrict y_acc) {
  if (cb == 0)
    proj_qmm_zero(y_acc, 0);
  proj_qmm_acc(x_full, cb, w, y_acc);
}

void mxfp4_step(bfloat16 *__restrict x_full, uint8_t *__restrict w,
                float *__restrict y_acc, int cb) {
  if (cb == 0)
    mxfp4_bias_init(w, y_acc);
  mxfp4_acc128_x(x_full, w, y_acc, cb);
}

} // extern "C"
