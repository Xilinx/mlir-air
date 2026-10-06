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

// sum of squares in 16 fixed lanes: vectorizes, and the order does not depend
// on the pointer's alignment
static inline float sumsq(const float *a, int n) {
  float acc[16] = {0};
  int j = 0;
  for (; j + 16 <= n; j += 16)
    for (int l = 0; l < 16; l++)
      acc[l] += a[j + l] * a[j + l];
  float s = 0.f;
  for (int l = 0; l < 16; l++)
    s += acc[l];
  for (; j < n; j++)
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

// one row of the GEMM's A layout: row r of dst [HR][K/TK][TM][TK]
static inline uint16_t *a_row(uint16_t *dst, int r, int s, int ks, int TM,
                              int TK) {
  return dst + (((long)(r / TM) * ks + s) * TM + r % TM) * TK;
}

static void tile_row(const float *x, int K, int r, int TM, int TK,
                     uint16_t *dst) {
  int ks = K / TK;
  for (int s = 0; s < ks; s++) {
    uint16_t *d = a_row(dst, r, s, ks, TM, TK);
    for (int k = 0; k < TK; k++)
      d[k] = f2bf(x[s * TK + k]);
  }
}

// dst (tiled A, K = n) = bf16(rms(x) * w)
void rms_tile(const float *x, int t, int n, const float *w, float eps, int TM,
              int TK, uint16_t *dst) {
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < t; r++) {
    const float *a = x + (long)r * n;
    float row[n];
    float inv = 1.f / sqrtf(sumsq(a, n) / n + eps);
    for (int j = 0; j < n; j++)
      row[j] = a[j] * inv * w[j];
    tile_row(row, n, r, TM, TK, dst);
  }
}

// x += post ? rms(c) * post : c, c bf16 [t, ldc]; then, if w, dst (tiled A)
// = bf16(rms(x) * w)
void add_rms_tile(float *x, const uint16_t *c, int ldc, const float *post,
                  float eps, int t, int n, const float *w, int TM, int TK,
                  uint16_t *dst) {
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < t; r++) {
    float *a = x + (long)r * n;
    float row[n];
    const uint16_t *cr = c + (long)r * ldc;
    for (int j = 0; j < n; j++)
      row[j] = bf2f(cr[j]);
    if (post) {
      float inv = 1.f / sqrtf(sumsq(row, n) / n + eps);
      for (int j = 0; j < n; j++)
        a[j] += row[j] * inv * post[j];
    } else {
      for (int j = 0; j < n; j++)
        a[j] += row[j];
    }
    if (w) {
      float inv = 1.f / sqrtf(sumsq(a, n) / n + eps);
      for (int j = 0; j < n; j++)
        row[j] = a[j] * inv * w[j];
      tile_row(row, n, r, TM, TK, dst);
    }
  }
}

// nh heads of dh at column col0 of src bf16 [t, ld]: + bias, rms with norm
// (if given), half-split rotary on the first rot dims, * scale, to bf16
// dst[h * hs + r * rs + i]
void head_post(const uint16_t *src, int ld, int t, int col0, int nh, int dh,
               const float *bias, const float *norm, float eps, const float *cs,
               const float *sn, int rot, float scale, uint16_t *dst, long hs,
               long rs) {
  int hr = rot / 2;
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < t; r++) {
    float v[dh];
    for (int h = 0; h < nh; h++) {
      const uint16_t *s = src + (long)r * ld + col0 + h * dh;
      for (int i = 0; i < dh; i++)
        v[i] = bf2f(s[i]) + (bias ? bias[col0 + h * dh + i] : 0.f);
      if (norm) {
        float inv = 1.f / sqrtf(sumsq(v, dh) / dh + eps);
        for (int i = 0; i < dh; i++)
          v[i] = v[i] * inv * norm[i];
      }
      if (rot) {
        const float *c = cs + (long)r * hr, *sv = sn + (long)r * hr;
        for (int i = 0; i < hr; i++) {
          float a = v[i], b = v[i + hr];
          v[i] = a * c[i] - b * sv[i];
          v[i + hr] = b * c[i] + a * sv[i];
        }
      }
      uint16_t *d = dst + h * hs + r * rs;
      for (int i = 0; i < dh; i++)
        d[i] = f2bf(v[i] * scale);
    }
  }
}

// KV records of one head for rows [0, t): k / v bf16 [t, ld] at column c0;
// per lkp-row block, the K tile [lkp][dh] then V as dh/dvt tiles [lkp][dvt];
// rows past t are zero
void kv_rec(const uint16_t *k, const uint16_t *v, int ld, int c0, int t, int dh,
            int lkp, int dvt, uint16_t *dst) {
  int nb = (t + lkp - 1) / lkp, rec = 2 * lkp * dh;
  for (int b = 0; b < nb; b++) {
    uint16_t *kd = dst + (long)b * rec, *vd = kd + lkp * dh;
    for (int i = 0; i < lkp; i++) {
      int r = b * lkp + i;
      if (r < t) {
        memcpy(kd + i * dh, k + (long)r * ld + c0, dh * 2);
        for (int z = 0; z < dh / dvt; z++)
          memcpy(vd + ((long)z * lkp + i) * dvt,
                 v + (long)r * ld + c0 + z * dvt, dvt * 2);
      } else {
        memset(kd + i * dh, 0, dh * 2);
        for (int z = 0; z < dh / dvt; z++)
          memset(vd + ((long)z * lkp + i) * dvt, 0, dvt * 2);
      }
    }
  }
}

// o bf16 [nh, M, dh] -> heads [h0, h0 + nh) of dst (tiled A, K)
void o_tile(const uint16_t *o, int t, int nh, int dh, int M, int h0, int K,
            int TM, int TK, uint16_t *dst) {
  int ks = K / TK;
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int r = 0; r < t; r++)
    for (int h = 0; h < nh; h++)
      for (int i = 0; i < dh; i += TK < dh ? TK : dh) {
        int col = (h0 + h) * dh + i, n = TK < dh ? TK : dh;
        uint16_t *d = a_row(dst, r, col / TK, ks, TM, TK) + col % TK;
        memcpy(d, o + ((long)h * M + r) * dh + i, n * 2);
      }
}

// bfp16ebs8, the cores' native matmul operand: per 8 values one exponent byte
// (127 + the largest value's exponent) and 8 int8 mantissas, value * 2^(6 -
// exponent) rounded to even. A mantissa that rounds to 128 moves the group up
// an exponent, as the cores' own conversion does.
static inline void bfp_group(const uint16_t *src, long cs, uint8_t *dst) {
  uint32_t b[8], emax = 0; // biased exponents; 0 for zeros
  for (int i = 0; i < 8; i++) {
    b[i] = src[i * cs];
    uint32_t e = (b[i] >> 7) & 0xFF;
    emax = e > emax ? e : emax;
  }
  if (emax < 8) { // zeros and values too small to keep
    memset(dst, 0, 9);
    return;
  }
  float x[8], m[8], top = 0.f;
  // 2^(6 - (emax - 127)), from its bits
  uint32_t sb = (260u - emax) << 23;
  float sc;
  memcpy(&sc, &sb, 4);
  for (int i = 0; i < 8; i++) {
    uint32_t u = b[i] << 16;
    memcpy(x + i, &u, 4);
    m[i] = __builtin_rintf(x[i] * sc);
    float a = __builtin_fabsf(m[i]);
    top = a > top ? a : top;
  }
  if (top > 127.f) {
    emax++;
    sc *= 0.5f;
    for (int i = 0; i < 8; i++)
      m[i] = __builtin_rintf(x[i] * sc);
  }
  dst[0] = (uint8_t)emax;
  for (int i = 0; i < 8; i++)
    dst[1 + i] = (uint8_t)(int8_t)m[i];
}

// an 8x8 block, row g = src[g * rs + c * cs]: 72 bytes
static void bfp_block(const uint16_t *src, long rs, long cs, uint8_t *dst) {
  for (int g = 0; g < 8; g++)
    bfp_group(src + g * rs, cs, dst + 9 * g);
}

// kv_rec in bfp16: per lkp-row block the K blocks [dh/8][lkp/8] (rows key)
// then the V blocks [dh/8][lkp/8] transposed (rows d); rows past t are zero
void kv_rec_bfp(const uint16_t *k, const uint16_t *v, int ld, int c0, int t,
                int dh, int lkp, uint8_t *dst) {
  int nb = (t + lkp - 1) / lkp;
  long half = 72L * (dh / 8) * (lkp / 8);
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int b = 0; b < nb; b++) {
    uint16_t kt[lkp * dh], vt[lkp * dh];
    for (int i = 0; i < lkp; i++) {
      int r = b * lkp + i;
      if (r < t) {
        memcpy(kt + i * dh, k + (long)r * ld + c0, dh * 2);
        memcpy(vt + i * dh, v + (long)r * ld + c0, dh * 2);
      } else {
        memset(kt + i * dh, 0, dh * 2);
        memset(vt + i * dh, 0, dh * 2);
      }
    }
    uint8_t *kd_ = dst + b * 2 * half, *vd = kd_ + half;
    for (int x = 0; x < dh / 8; x++)
      for (int y = 0; y < lkp / 8; y++) {
        bfp_block(kt + y * 8 * dh + x * 8, dh, 1,
                  kd_ + 72 * (x * (lkp / 8) + y));
        bfp_block(vt + y * 8 * dh + x * 8, 1, dh,
                  vd + 72 * (x * (lkp / 8) + y));
      }
  }
}

// q bf16 [nh][M][dh] -> bfp16, per head per tq-row tile the blocks [tq/8][dh/8]
void q_bfp(const uint16_t *q, int nh, int M, int dh, int tq, uint8_t *dst) {
  int nt = M / tq, nblk = (tq / 8) * (dh / 8);
#pragma omp parallel for num_threads(NT) schedule(static)
  for (int ht = 0; ht < nh * nt; ht++) {
    const uint16_t *src = q + (long)ht * tq * dh;
    uint8_t *d = dst + 72L * nblk * ht;
    for (int mq = 0; mq < tq / 8; mq++)
      for (int x = 0; x < dh / 8; x++)
        bfp_block(src + mq * 8 * dh + x * 8, dh, 1,
                  d + 72 * (mq * (dh / 8) + x));
  }
}
