// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Qwen3.6's delta-rule step (gdn_decode.cc) with a head's outputs placed at
// an offset: a step core's buffer holds both tokens' outputs.

#include "gdn_decode.cc"

extern "C" {

// gdn_step_h with head h's outputs at out + off + 32 h
void gdn_step_ho(float *__restrict s_in, float *__restrict s_out,
                 bfloat16 *__restrict kvq, bfloat16 *__restrict out, int chunk,
                 int h, int off) {
  gdn_step(s_in, s_out, kvq, out + off + h * VC, chunk);
}

} // extern "C"
