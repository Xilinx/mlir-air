// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The d3 prefill's host row ops that write the next GEMM's A straight in the
// device's bfp16 layout. Each is hostops' (or q35_host's) op followed by
// hostops' tiles_bfp, fused: the rows of an 8-row block are computed into a
// local bf16 buffer and converted from there, with the same arithmetic, so
// rows [0, t) come out bit for bit as the two-pass ops write them. Rows past
// t of the last 8-row block are converted as zeros.
#include "../../shared/fused_prefill/kernels/hostops.c"

// q35_host.c's SiLU, the same arithmetic (vexp: Cody-Waite reduction, degree-6
// Taylor)
static inline float d3_vexp(float x) {
  x = x < -87.f ? -87.f : (x > 88.f ? 88.f : x);
  float n = rintf(x * 1.44269504f);
  float r = x - n * 0.693145752f;
  r = r - n * 1.42860677e-6f;
  float p = 1.f / 720;
  p = p * r + 1.f / 120;
  p = p * r + 1.f / 24;
  p = p * r + 1.f / 6;
  p = p * r + 0.5f;
  p = p * r + 1.f;
  p = p * r + 1.f;
  int32_t e = ((int32_t)n + 127) << 23;
  float sc;
  memcpy(&sc, &e, 4);
  return p * sc;
}

static inline float d3_sigm(float x) { return 1.f / (1.f + d3_vexp(-x)); }

// The 8 rows of block rb, bf16 [8][K], into the A at dst (tiles_bfp's
// layout; tile (ty, s) at dst + (ty * dr + s * dk) tiles)
static void blk_bfp(const uint16_t *blk, int rb, int K, int TM, int TK,
                    uint8_t *dst, long dr, long dk) {
  long tile = 72L * (TM / 8) * (TK / 8);
  int nb = TM / 8, ty = rb / nb, b = rb % nb;
  for (int s = 0; s < K / TK; s++) {
    uint8_t *d = dst + (ty * dr + s * dk) * tile + 72L * (b % 2);
    for (int kb = 0; kb < TK / 8; kb++)
      bfp_block(blk + s * TK + kb * 8, K, 1,
                d + 144L * ((b / 2) * (TK / 8) + kb));
  }
}

// rms_tile + tiles_bfp: dst = bfp16(bf16(rms(x) * w)); x [t, n] f32
void d3_norm_bfp(const float *x, int t, int n, const float *w, float eps,
                 int TM, int TK, uint8_t *dst, long dr, long dk, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int rb = 0; rb < (t + 7) / 8; rb++) {
    uint16_t blk[8 * n];
    for (int i = 0; i < 8; i++) {
      int r = rb * 8 + i;
      uint16_t *o = blk + (long)i * n;
      if (r >= t) {
        memset(o, 0, sizeof(uint16_t) * n);
        continue;
      }
      const float *a = x + (long)r * n;
      float inv = 1.f / sqrtf(sumsq(a, n) / n + eps);
      for (int j = 0; j < n; j++)
        o[j] = f2bf(a[j] * inv * w[j]);
    }
    blk_bfp(blk, rb, n, TM, TK, dst, dr, dk);
  }
}

// add_rms_tile (no post norm) + tiles_bfp: x += c (c bf16 [t, ldc]); dst =
// bfp16(bf16(rms(x) * w))
void d3_add_norm_bfp(float *x, const uint16_t *c, int ldc, int t, int n,
                     const float *w, float eps, int TM, int TK, uint8_t *dst,
                     long dr, long dk, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int rb = 0; rb < (t + 7) / 8; rb++) {
    uint16_t blk[8 * n];
    for (int i = 0; i < 8; i++) {
      int r = rb * 8 + i;
      uint16_t *o = blk + (long)i * n;
      if (r >= t) {
        memset(o, 0, sizeof(uint16_t) * n);
        continue;
      }
      float *a = x + (long)r * n;
      const uint16_t *cr = c + (long)r * ldc;
      for (int j = 0; j < n; j++)
        a[j] += bf2f(cr[j]);
      float inv = 1.f / sqrtf(sumsq(a, n) / n + eps);
      for (int j = 0; j < n; j++)
        o[j] = f2bf(a[j] * inv * w[j]);
    }
    blk_bfp(blk, rb, n, TM, TK, dst, dr, dk);
  }
}

// swiglu_tile + tiles_bfp: dst = bfp16(bf16(bf16(SiLU(g)) * u)); c bf16 [t,
// ldc], g its columns [0, n), u [n, 2n)
void d3_swiglu_bfp(const uint16_t *c, int ldc, int t, int n, int TM, int TK,
                   uint8_t *dst, long dr, long dk, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int rb = 0; rb < (t + 7) / 8; rb++) {
    uint16_t blk[8 * n];
    for (int i = 0; i < 8; i++) {
      int r = rb * 8 + i;
      uint16_t *o = blk + (long)i * n;
      if (r >= t) {
        memset(o, 0, sizeof(uint16_t) * n);
        continue;
      }
      const uint16_t *g = c + (long)r * ldc, *u = g + n;
      for (int j = 0; j < n; j++) {
        float gv = bf2f(g[j]);
        o[j] = f2bf(bf2f(f2bf(gv * d3_sigm(gv))) * bf2f(u[j]));
      }
    }
    blk_bfp(blk, rb, n, TM, TK, dst, dr, dk);
  }
}

// tile_a + tiles_bfp: dst = bfp16(bf16(x)); x [t, K] f32
void d3_rows_bfp(const float *x, int t, int K, int TM, int TK, uint8_t *dst,
                 long dr, long dk, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int rb = 0; rb < (t + 7) / 8; rb++) {
    uint16_t blk[8 * K];
    for (int i = 0; i < 8; i++) {
      int r = rb * 8 + i;
      uint16_t *o = blk + (long)i * K;
      if (r >= t) {
        memset(o, 0, sizeof(uint16_t) * K);
        continue;
      }
      const float *a = x + (long)r * K;
      for (int j = 0; j < K; j++)
        o[j] = f2bf(a[j]);
    }
    blk_bfp(blk, rb, K, TM, TK, dst, dr, dk);
  }
}
