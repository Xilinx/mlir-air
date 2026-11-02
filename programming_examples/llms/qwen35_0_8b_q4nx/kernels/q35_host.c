// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Host side of Qwen3.5's fused prefill (shared/fused_prefill/qwen35.py) that
// shared/fused_prefill's hostops does not have: the Gated DeltaNet mixer of
// one 128-row chunk (causal conv1d + SiLU, q/k l2-norm, gates, the gated delta
// rule on an FP32 state, gated RMSNorm) and the attention's output gate. bf16
// roundings follow the HF BF16 model's casts.
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define NH 16   // linear-attention heads (q, k and v)
#define DK 128  // key / value head dim
#define NC 6144 // conv channels: q | k | v
#define NZ 2048 // z (output gate) width
#define TAPS 4
#define VB 32 // state columns per scan task

static inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static inline float rbf(float f) { return bf2f(f2bf(f)); }

// exp in vectorizable arithmetic (Cody-Waite reduction, degree-6 Taylor on
// |r| <= ln2 / 2: relative error ~2e-7), so the per-element SiLU / sigmoid
// loops vectorize without -ffast-math
static inline float vexp(float x) {
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

static inline float sigm(float x) { return 1.f / (1.f + vexp(-x)); }

// One chunk of a Gated DeltaNet layer.
//   lin  bf16 [t, ldl]: q | k | v (NC raw, before the conv), then z (NZ)
//   ba   bf16 [t, ldb]: b (NH), then a (NH)
//   hist f32 [TAPS - 1, NC]: the raw q | k | v rows before the chunk; on
//        return the chunk's last TAPS - 1 rows (earlier ones if t < 3)
//   conv f32 [TAPS, NC], nega f32 [NH] (-exp(A_log)), dtb f32 [NH],
//   normw f32 [DK] (the gated norm's weight)
//   state f32 [NH][DK][DK] (k, v), carried across chunks
//   u    f32 scratch [t + 3, NC], y f32 scratch [t, NC]
//   out  f32 [t, NZ]: the out_proj's input
void gdn_chunk(const uint16_t *lin, int ldl, const uint16_t *ba, int ldb, int t,
               float *hist, const float *conv, const float *nega,
               const float *dtb, const float *normw, float *state, float *u,
               float *y, float *out, int nt) {
  // X = [hist; chunk's raw q | k | v] as f32 (u's rows 0..t+2 hold X, then
  // are overwritten row by row by the conv output, which lags 3 rows)
  float *X = u; // [t + TAPS - 1, NC]
  memcpy(X, hist, sizeof(float) * (TAPS - 1) * NC);
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int r = 0; r < t; r++) {
    float *xr = X + (long)(r + TAPS - 1) * NC;
    const uint16_t *lr = lin + (long)r * ldl;
    for (int c = 0; c < NC; c++)
      xr[c] = bf2f(lr[c]);
  }
  memcpy(hist, X + (long)t * NC, sizeof(float) * (TAPS - 1) * NC);
  // the gates per row and head: beta = sigmoid(b), decay exp(g)
  float beta[t][NH], decay[t][NH];
  // conv1d + SiLU into y (rows r of the output from X rows r..r+3), q and k
  // l2-normalized (q * DK^-0.5); written to out-of-place rows of y
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int r = 0; r < t; r++) {
    float *ur = y + (long)r * NC;
    const float *x0 = X + (long)r * NC;
    for (int c = 0; c < NC; c++) {
      float s = 0.f;
      for (int j = 0; j < TAPS; j++)
        s += conv[(long)j * NC + c] * x0[(long)j * NC + c];
      s = rbf(s);
      ur[c] = rbf(s * sigm(s));
    }
    for (int h = 0; h < NH; h++) {
      beta[r][h] = rbf(sigm(bf2f(ba[(long)r * ldb + h])));
      float av = bf2f(ba[(long)r * ldb + NH + h]) + dtb[h];
      float sp = av > 20.f ? av : log1pf(expf(av));
      decay[r][h] = expf(nega[h] * sp);
    }
    for (int h = 0; h < 2 * NH; h++) { // q heads, then k heads
      float *v = ur + h * DK, ss = 0.f;
      for (int i = 0; i < DK; i++)
        ss += v[i] * v[i];
      float inv = 1.f / sqrtf(ss + 1e-6f);
      float sc = h < NH ? 1.f / sqrtf((float)DK) : 1.f;
      for (int i = 0; i < DK; i++)
        v[i] = rbf(v[i] * inv) * sc;
    }
  }
  u = y;
  // the delta rule: every value column of a head's state evolves on its own,
  // so a task is a head's VB columns (16 KB, in L1); raw o into out
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int task = 0; task < NH * (DK / VB); task++) {
    int h = task / (DK / VB), c0 = task % (DK / VB) * VB;
    _mm_setcsr(_mm_getcsr() | 0x8040); // flush denormals (decaying states)
    float S[DK][VB] __attribute__((aligned(64)));
    for (int a = 0; a < DK; a++)
      memcpy(S[a], state + ((long)h * DK + a) * DK + c0, sizeof(S[a]));
    for (int r = 0; r < t; r++) {
      const float *q = u + (long)r * NC + h * DK;
      const float *k = u + (long)r * NC + NH * DK + h * DK;
      const float *vv = u + (long)r * NC + 2 * NH * DK + h * DK + c0;
      float eg = decay[r][h], bt = beta[r][h];
      // AVX-512, VB = 32 columns in two registers; four partial sums over
      // the rows of S break the dependency chains
      __m512 ka[4][2], e = _mm512_set1_ps(eg);
      for (int j = 0; j < 4; j++)
        ka[j][0] = ka[j][1] = _mm512_setzero_ps();
      for (int a = 0; a < DK; a += 4)
        for (int j = 0; j < 4; j++) {
          __m512 kk = _mm512_set1_ps(k[a + j]);
          ka[j][0] = _mm512_fmadd_ps(_mm512_load_ps(S[a + j]), kk, ka[j][0]);
          ka[j][1] =
              _mm512_fmadd_ps(_mm512_load_ps(S[a + j] + 16), kk, ka[j][1]);
        }
      __m512 d[2], oa[4][2];
      for (int x = 0; x < 2; x++) {
        __m512 kv = _mm512_add_ps(_mm512_add_ps(ka[0][x], ka[1][x]),
                                  _mm512_add_ps(ka[2][x], ka[3][x]));
        d[x] = _mm512_mul_ps(
            _mm512_sub_ps(_mm512_loadu_ps(vv + 16 * x), _mm512_mul_ps(e, kv)),
            _mm512_set1_ps(bt));
      }
      for (int j = 0; j < 4; j++)
        oa[j][0] = oa[j][1] = _mm512_setzero_ps();
      for (int a = 0; a < DK; a += 4)
        for (int j = 0; j < 4; j++) {
          __m512 kk = _mm512_set1_ps(k[a + j]), qq = _mm512_set1_ps(q[a + j]);
          for (int x = 0; x < 2; x++) {
            __m512 sv = _mm512_fmadd_ps(_mm512_load_ps(S[a + j] + 16 * x), e,
                                        _mm512_mul_ps(kk, d[x]));
            _mm512_store_ps(S[a + j] + 16 * x, sv);
            oa[j][x] = _mm512_fmadd_ps(qq, sv, oa[j][x]);
          }
        }
      float o[VB];
      for (int x = 0; x < 2; x++)
        _mm512_storeu_ps(o + 16 * x,
                         _mm512_add_ps(_mm512_add_ps(oa[0][x], oa[1][x]),
                                       _mm512_add_ps(oa[2][x], oa[3][x])));
      memcpy(out + (long)r * NZ + h * DK + c0, o, sizeof(o));
    }
    for (int a = 0; a < DK; a++)
      memcpy(state + ((long)h * DK + a) * DK + c0, S[a], sizeof(S[a]));
  }
  // gated RMSNorm: bf16(rms(bf16(o))) * w * SiLU(z)
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int r = 0; r < t; r++)
    for (int h = 0; h < NH; h++) {
      float *o = out + (long)r * NZ + h * DK, ss = 0.f;
      for (int i = 0; i < DK; i++) {
        o[i] = rbf(o[i]);
        ss += o[i] * o[i];
      }
      float inv = 1.f / sqrtf(ss / DK + 1e-6f);
      const uint16_t *z = lin + (long)r * ldl + NC + h * DK;
      for (int i = 0; i < DK; i++) {
        float zi = bf2f(z[i]);
        o[i] = rbf(rbf(o[i] * inv) * normw[i]) * (zi * sigm(zi));
      }
    }
}

// offset of (i, j) in a tile of rows x cols as the attention cores write
// it, [cols / 8][rows / 8][8][8] (hostops.c's t8)
static inline long t8(int i, int j, int rows) {
  return ((long)(j >> 3) * (rows >> 3) + (i >> 3)) * 64 + (i & 7) * 8 + (j & 7);
}

// hostops.c's o_tile with the output gate: heads [h0, h0 + nh) of the GEMM's
// A (bf16 tiles [HR][K/TK][TM][TK]) = bf16(o * bf16(sigmoid(gate))),
// o [nh][M / tq] tiles of tq x dh as the attention cores write them,
// gate bf16 [t, ldg] from column (h0 * dh) on
void o_gate_tile(const uint16_t *o, int t, int nh, int dh, int M, int tq,
                 int h0, int K, int TM, int TK, const uint16_t *gate, int ldg,
                 uint16_t *dst, int nt) {
  int ks = K / TK;
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int r = 0; r < t; r++) {
    float sg[nh * dh];
    const uint16_t *g = gate + (long)r * ldg + (long)h0 * dh;
    for (int i = 0; i < nh * dh; i++)
      sg[i] = rbf(sigm(bf2f(g[i])));
    for (int h = 0; h < nh; h++) {
      const uint16_t *b = o + (long)h * M * dh + (long)(r / tq) * tq * dh;
      for (int i0 = 0; i0 < dh; i0 += 8) {
        int col = (h0 + h) * dh + i0;
        uint16_t *d = dst +
                      (((long)(r / TM) * ks + col / TK) * TM + r % TM) * TK +
                      col % TK;
        const uint16_t *src = b + t8(r % tq, i0, tq);
        const float *gs = sg + h * dh + i0;
        for (int i = 0; i < 8; i++)
          d[i] = f2bf(bf2f(src[i]) * gs[i]);
      }
    }
  }
}

// SwiGLU of the fused gate | up GEMM's output: the GEMM's A (bf16 tiles
// [HR][n/TK][TM][TK]) = bf16(bf16(SiLU(g)) * u), c bf16 [t, ldc] holding g
// in columns [0, n) and u in [n, 2n)
void swiglu_tile(const uint16_t *c, int ldc, int t, int n, int TM, int TK,
                 uint16_t *dst, int nt) {
  int ks = n / TK;
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int r = 0; r < t; r++) {
    const uint16_t *g = c + (long)r * ldc, *u = g + n;
    for (int s = 0; s < ks; s++) {
      uint16_t *d = dst + (((long)(r / TM) * ks + s) * TM + r % TM) * TK;
      for (int k = 0; k < TK; k++) {
        float gv = bf2f(g[s * TK + k]);
        d[k] = f2bf(rbf(gv * sigm(gv)) * bf2f(u[s * TK + k]));
      }
    }
  }
}
