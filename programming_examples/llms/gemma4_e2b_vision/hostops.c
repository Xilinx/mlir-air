// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Host side of the vision tower between its NPU ops: norms, rope, the
// ClippableLinear clamps and products. Each op reads a GEMM's bf16 output,
// applying that GEMM's output clamp, and writes the next GEMM's bf16 input.
// Rows are split across NT threads.
#include <math.h>
#include <stdint.h>
#include <string.h>

#define NT 4

static inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u)
    return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static inline float bf2f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// out[r] = bf16(clip(x[r] / rms(x[r]) * w, lo, hi)) for r < valid, 0 after.
void rms_clip_bf16(const float *x, int rows, int valid, int n, const float *w,
                   float eps, float lo, float hi, uint16_t *out) {
#pragma omp parallel for schedule(static) num_threads(NT)
  for (int r = 0; r < rows; r++) {
    uint16_t *o = out + (long)r * n;
    if (r >= valid) {
      memset(o, 0, sizeof(uint16_t) * n);
      continue;
    }
    const float *a = x + (long)r * n;
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int j = 0; j < n; j++)
      s += a[j] * a[j];
    const float inv = 1.f / sqrtf(s / n + eps);
    for (int j = 0; j < n; j++)
      o[j] = f2bf(clampf(a[j] * inv * w[j], lo, hi));
  }
}

// h[r] += rms(clip(y[r], lo, hi)) * w for r < valid; h[r] = 0 after. y bf16.
void add_rms_clip(float *h, const uint16_t *y, int rows, int valid, int n,
                  const float *w, float eps, float lo, float hi) {
#pragma omp parallel for schedule(static) num_threads(NT)
  for (int r = 0; r < rows; r++) {
    float *o = h + (long)r * n;
    if (r >= valid) {
      memset(o, 0, sizeof(float) * n);
      continue;
    }
    const uint16_t *a = y + (long)r * n;
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int j = 0; j < n; j++) {
      float v = clampf(bf2f(a[j]), lo, hi);
      s += v * v;
    }
    const float inv = 1.f / sqrtf(s / n + eps);
    for (int j = 0; j < n; j++)
      o[j] += clampf(bf2f(a[j]), lo, hi) * inv * w[j];
  }
}

// Per-head clip + rmsnorm (w may be null) + optional 2-D rope of one head of
// dh values; each half of dh is a half-split rotation by its own position.
static inline void head(const uint16_t *a, int dh, float lo, float hi,
                        const float *w, float eps, const float *c,
                        const float *sv, float scale, float *t, float *o) {
  float s = 0.f;
  for (int j = 0; j < dh; j++) {
    t[j] = clampf(bf2f(a[j]), lo, hi);
    s += t[j] * t[j];
  }
  const float inv = 1.f / sqrtf(s / dh + eps);
  for (int j = 0; j < dh; j++)
    t[j] = w ? t[j] * inv * w[j] : t[j] * inv;
  if (!c) {
    for (int j = 0; j < dh; j++)
      o[j] = t[j] * scale;
    return;
  }
  const int per = dh / 2, h2 = per / 2;
  for (int p = 0; p < 2; p++) {
    const float *xp = t + p * per, *cp = c + p * per, *sp = sv + p * per;
    float *op = o + p * per;
    for (int j = 0; j < h2; j++) {
      op[j] = (xp[j] * cp[j] - xp[j + h2] * sp[j]) * scale;
      op[j + h2] = (xp[j + h2] * cp[j + h2] + xp[j] * sp[j + h2]) * scale;
    }
  }
}

// The fused q|k|v GEMM output [rows, 3*H*dh] (bf16) -> the attention's
// sequence-first inputs, each head widened from dh to dk columns:
//   Q [rows, H*dk]:  clip, q_norm, rope, * qscale; column dh is 1
//   K [kv_rows, H*dk]: clip, k_norm, rope; column dh is 0, or -big on rows
//                      >= valid so padding keys get a score of -big
//   V [kv_rows, H*dh]: clip, unscaled norm; 0 on rows >= valid
// bounds = {q_lo, q_hi, k_lo, k_hi, v_lo, v_hi}.
void qkv_prep(const uint16_t *qkv, int rows, int valid, int kv_rows, int H,
              int dh, int dk, const float *qn, const float *kn, float eps,
              const float *cs, const float *sn, const float *bounds,
              float qscale, float big, uint16_t *Q, uint16_t *K, uint16_t *V) {
  const int E = H * dh;
  const int top = rows > kv_rows ? rows : kv_rows;
#pragma omp parallel for schedule(static) num_threads(NT)
  for (int r = 0; r < top; r++) {
    float t[256], o[256];
    const uint16_t *a = qkv + (long)r * 3 * E;
    const float *c = cs + (long)r * dh, *sv = sn + (long)r * dh;
    for (int h = 0; h < H; h++) {
      if (r < rows) {
        uint16_t *q = Q + ((long)r * H + h) * dk;
        head(a + h * dh, dh, bounds[0], bounds[1], qn, eps, c, sv, qscale, t,
             o);
        for (int j = 0; j < dh; j++)
          q[j] = f2bf(o[j]);
        q[dh] = f2bf(1.f);
        memset(q + dh + 1, 0, sizeof(uint16_t) * (dk - dh - 1));
      }
      if (r < kv_rows) {
        uint16_t *k = K + ((long)r * H + h) * dk;
        uint16_t *v = V + ((long)r * H + h) * dh;
        memset(k + dh, 0, sizeof(uint16_t) * (dk - dh));
        if (r >= valid) {
          memset(k, 0, sizeof(uint16_t) * dh);
          memset(v, 0, sizeof(uint16_t) * dh);
          k[dh] = f2bf(-big);
          continue;
        }
        head(a + E + h * dh, dh, bounds[2], bounds[3], kn, eps, c, sv, 1.f, t,
             o);
        for (int j = 0; j < dh; j++)
          k[j] = f2bf(o[j]);
        head(a + 2 * E + h * dh, dh, bounds[4], bounds[5], NULL, eps, NULL,
             NULL, 1.f, t, o);
        for (int j = 0; j < dh; j++)
          v[j] = f2bf(o[j]);
      }
    }
  }
}

// out = bf16(clip(min(g, g_hi) * clip(u, u_lo, u_hi), lo, hi)), g and u bf16.
// g is the gate GEMM's GELU output, and g_hi = GELU(gate output_max) stands in
// for the gate's output clamp (see gemma4_vision_encoder.py).
void glu_clip(const uint16_t *g, const uint16_t *u, long count, float g_hi,
              float u_lo, float u_hi, float lo, float hi, uint16_t *out) {
#pragma omp parallel for schedule(static) num_threads(NT)
  for (long i = 0; i < count; i++) {
    float gv = bf2f(g[i]);
    gv = gv > g_hi ? g_hi : gv;
    out[i] = f2bf(clampf(gv * clampf(bf2f(u[i]), u_lo, u_hi), lo, hi));
  }
}

// x = bf16(clip(x)) in place, x bf16.
void clip_bf16(uint16_t *x, long count, float lo, float hi) {
#pragma omp parallel for schedule(static) num_threads(NT)
  for (long i = 0; i < count; i++)
    x[i] = f2bf(clampf(bf2f(x[i]), lo, hi));
}
