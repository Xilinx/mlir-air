// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GQA kernel: single-head attention with packed K/V buffer.
// Q[LQ,D] @ K[D,LK]^T -> softmax -> P[LQ,LK] @ V[LK,D] -> O[LQ,D]
// K and V are packed contiguously: kv_buf = [K_data | V_data]

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_LQ
#define DIM_LQ 16
#endif
#ifndef DIM_LK
#define DIM_LK 16
#endif
#ifndef DIM_D
#define DIM_D 16
#endif

extern "C" {

void attention_head_bf16(bfloat16 *__restrict Q_buf,
                         bfloat16 *__restrict KV_buf,
                         bfloat16 *__restrict O_buf) {
  // KV_buf layout: [K[D,LK] | V[LK,D]]
  bfloat16 *K_buf = KV_buf;
  bfloat16 *V_buf = KV_buf + DIM_D * DIM_LK;

  // Step 1: S[LQ,LK] = Q[LQ,D] @ K[D,LK]
  bfloat16 S[DIM_LQ * DIM_LK];
  for (unsigned i = 0; i < DIM_LQ; i++) {
    for (unsigned j = 0; j < DIM_LK; j++) {
      float sum = 0.0f;
      for (unsigned k = 0; k < DIM_D; k++) {
        sum += (float)Q_buf[i * DIM_D + k] * (float)K_buf[k * DIM_LK + j];
      }
      S[i * DIM_LK + j] = (bfloat16)sum;
    }
  }

  // Step 2: Row-wise softmax on S
  for (unsigned i = 0; i < DIM_LQ; i++) {
    // Find max
    float max_val = (float)S[i * DIM_LK];
    for (unsigned j = 1; j < DIM_LK; j++) {
      float v = (float)S[i * DIM_LK + j];
      if (v > max_val)
        max_val = v;
    }
    // Exp (Taylor approximation) and sum
    float sum = 0.0f;
    for (unsigned j = 0; j < DIM_LK; j++) {
      float diff = (float)S[i * DIM_LK + j] - max_val;
      if (diff < -10.0f)
        diff = -10.0f;
      float x2 = diff * diff;
      float x3 = x2 * diff;
      float e = 1.0f + diff + 0.5f * x2 + 0.16666667f * x3 +
                0.04166667f * x2 * x2 + 0.00833333f * x2 * x3 +
                0.00138889f * x3 * x3;
      if (e < 0.0f)
        e = 0.0f;
      S[i * DIM_LK + j] = (bfloat16)e;
      sum += e;
    }
    // Normalize
    float inv_sum = 1.0f / sum;
    for (unsigned j = 0; j < DIM_LK; j++) {
      S[i * DIM_LK + j] = (bfloat16)((float)S[i * DIM_LK + j] * inv_sum);
    }
  }

  // Step 3: O[LQ,D] = P[LQ,LK] @ V[LK,D]
  for (unsigned i = 0; i < DIM_LQ; i++) {
    for (unsigned j = 0; j < DIM_D; j++) {
      float sum = 0.0f;
      for (unsigned k = 0; k < DIM_LK; k++) {
        sum += (float)S[i * DIM_LK + k] * (float)V_buf[k * DIM_D + j];
      }
      O_buf[i * DIM_D + j] = (bfloat16)sum;
    }
  }
}

} // extern "C"
