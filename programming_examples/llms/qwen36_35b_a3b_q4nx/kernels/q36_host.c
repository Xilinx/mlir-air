// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Host side of Qwen3.6's fused prefill: Qwen3.5's Gated DeltaNet mixer at this
// model's head counts, and the experts' GLU.
#define NK 16
#define NV 32
#include "../../qwen35_0_8b_q4nx/kernels/q35_host.c"

// An expert's GLU into the down GEMM's A (bf16 tiles [HR][n/TK][TM][TK]):
// bf16(u * SiLU(g)), c bf16 [t, ldc] holding u and g in turns of 128 columns
void exp_glu_tile(const uint16_t *c, int ldc, int t, int n, int TM, int TK,
                  uint16_t *dst, int nt) {
  int ks = n / TK;
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int r = 0; r < t; r++) {
    for (int s = 0; s < ks; s++) {
      const uint16_t *u = c + (long)r * ldc + 2 * s * TK, *g = u + TK;
      uint16_t *d = dst + (((long)(r / TM) * ks + s) * TM + r % TM) * TK;
      for (int k = 0; k < TK; k++) {
        float gv = bf2f(g[k]);
        d[k] = f2bf(bf2f(u[k]) * gv * sigm(gv));
      }
    }
  }
}
