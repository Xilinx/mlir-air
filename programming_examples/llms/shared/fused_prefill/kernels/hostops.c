// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Host side of the fused prefill: conversions between float32 activations and
// the device's bf16 layouts, and the elementwise ops between GEMMs. Row loops
// run on NT threads; rows are independent, so results do not depend on NT.
#include <stdint.h>
#include <string.h>

#ifndef NT
#define NT 4
#endif

// round to nearest even; NaN not handled
static inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// x [t, K] f32 -> dst [HR][K/TK][TM][TK] bf16, the GEMM's A layout
void tile_a(const float *x, int t, int K, int TM, int TK, uint16_t *dst) {
  int ks = K / TK;
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < t; r++) {
    int ty = r / TM, i = r % TM;
    const float *src = x + (long)r * K;
    for (int s = 0; s < ks; s++) {
      uint16_t *d = dst + (((long)ty * ks + s) * TM + i) * TK;
      const float *xs = src + s * TK;
      for (int k = 0; k < TK; k++)
        d[k] = f2bf(xs[k]);
    }
  }
}

// src [rows, ld] bf16 -> dst [rows, n] f32
void bf16_to_f32(const uint16_t *src, int rows, int ld, int n, float *dst) {
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < rows; r++) {
    const uint16_t *s = src + (long)r * ld;
    float *d = dst + (long)r * n;
    for (int j = 0; j < n; j++) {
      uint32_t u = (uint32_t)s[j] << 16;
      memcpy(d + j, &u, 4);
    }
  }
}

#include <math.h>

// sum of squares, in vector lanes (an in-order float sum does not vectorize)
static inline float sumsq(const float *a, int n) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (int j = 0; j < n; j++)
    s += a[j] * a[j];
  return s;
}

// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * (w ? w : 1); rows of n
void rms(const float *x, int rows, int n, const float *w, float eps, float *y) {
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < rows; r++) {
    const float *a = x + (long)r * n;
    float *o = y + (long)r * n;
    float inv = 1.f / sqrtf(sumsq(a, n) / n + eps);
    if (w)
      for (int j = 0; j < n; j++)
        o[j] = a[j] * inv * w[j];
    else
      for (int j = 0; j < n; j++)
        o[j] = a[j] * inv;
  }
}

// y = res + rms(x) * w
void add_rms(const float *res, const float *x, int rows, int n, const float *w,
             float eps, float *y) {
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < rows; r++) {
    const float *a = x + (long)r * n, *b = res + (long)r * n;
    float *o = y + (long)r * n;
    float inv = 1.f / sqrtf(sumsq(a, n) / n + eps);
    for (int j = 0; j < n; j++)
      o[j] = b[j] + a[j] * inv * w[j];
  }
}

// x [T, H, dh] in place: half-split rotary on the first rot dims; cs/sn [T,
// rot/2]
void rope(float *x, int T, int H, int dh, int rot, const float *cs,
          const float *sn) {
  int h = rot / 2;
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int t = 0; t < T; t++)
    for (int k = 0; k < H; k++) {
      float *v = x + ((long)t * H + k) * dh;
      const float *c = cs + (long)t * h, *s = sn + (long)t * h;
      for (int i = 0; i < h; i++) {
        float a = v[i], b = v[i + h];
        v[i] = a * c[i] - b * s[i];
        v[i + h] = b * c[i] + a * s[i];
      }
    }
}

static inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// dst (tiled A, K = n) = bf16(g * u), g / u bf16 [t, ld*] (GLU: gate already
// GELU'd)
void glu_tile(const uint16_t *g, int ldg, const uint16_t *u, int ldu, int t,
              int n, int TM, int TK, uint16_t *dst) {
  int ks = n / TK;
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < t; r++) {
    int ty = r / TM, i = r % TM;
    for (int s = 0; s < ks; s++) {
      uint16_t *d = dst + (((long)ty * ks + s) * TM + i) * TK;
      const uint16_t *gs = g + (long)r * ldg + s * TK,
                     *us = u + (long)r * ldu + s * TK;
      for (int k = 0; k < TK; k++)
        d[k] = f2bf(bf2f(gs[k]) * bf2f(us[k]));
    }
  }
}

// dst (tiled A, K = n) = bf16(g * p), g bf16 [t, ldg], p f32 [t, n]
void mul_tile(const uint16_t *g, int ldg, const float *p, int t, int n, int TM,
              int TK, uint16_t *dst) {
  int ks = n / TK;
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < t; r++) {
    int ty = r / TM, i = r % TM;
    for (int s = 0; s < ks; s++) {
      uint16_t *d = dst + (((long)ty * ks + s) * TM + i) * TK;
      const uint16_t *gs = g + (long)r * ldg + s * TK;
      const float *ps = p + (long)r * n + s * TK;
      for (int k = 0; k < TK; k++)
        d[k] = f2bf(bf2f(gs[k]) * ps[k]);
    }
  }
}

// q [T, H, dh] f32 -> dst [H, M, dh] bf16 = bf16(q * scale), rows T..M zeroed
void q_pack(const float *q, int T, int H, int dh, int M, float scale,
            uint16_t *dst) {
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int k = 0; k < H; k++) {
    uint16_t *d = dst + (long)k * M * dh;
    for (int t = 0; t < T; t++) {
      const float *s = q + ((long)t * H + k) * dh;
      for (int i = 0; i < dh; i++)
        d[(long)t * dh + i] = f2bf(s[i] * scale);
    }
    memset(d + (long)T * dh, 0, (size_t)(M - T) * dh * 2);
  }
}

// o [H, M, dh] bf16 -> dst [T, H * dh] f32
void o_unpack(const uint16_t *o, int T, int H, int dh, int M, float *dst) {
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int t = 0; t < T; t++)
    for (int k = 0; k < H; k++) {
      const uint16_t *s = o + ((long)k * M + t) * dh;
      float *d = dst + ((long)t * H + k) * dh;
      for (int i = 0; i < dh; i++)
        d[i] = bf2f(s[i]);
    }
}
