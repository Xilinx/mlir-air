// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The RMS and GLU cores' kernels: Qwen3.6-35B-A3B's (moe_glu.cc, ffn.cc), and
// residual adds in place. residual_add also writes a copy of the sum, which
// would need a further [D] buffer, or a buffer the core has sent and would
// then write again within the same layer; residual_round adds a projection's
// rounds as they arrive, with no [D] buffer to assemble them in.

#include "moe_glu.cc"

extern "C" {

// x[512 j ..] += rnd: a dense phase's round j added in place, the rows its
// residual needs
void residual_round(bf16 *__restrict x, bf16 *__restrict rnd, int j) {
  x += NC * SL * j;
  for (int i = 0; i < NC * SL; i += V)
    aie::store_v(x + i,
                 aie::add(aie::load_v<V>(x + i), aie::load_v<V>(rnd + i)));
}

// xbuf += x
void residual_inplace(bf16 *__restrict xbuf, bf16 *__restrict x, int n) {
  for (int i = 0; i < n; i += V)
    aie::store_v(xbuf + i,
                 aie::add(aie::load_v<V>(xbuf + i), aie::load_v<V>(x + i)));
}

// Two tokens' rounds (the speculative decode's verify pass): a round is
// [column][core][token][32 rows], column c holding the 128-row group 4 j + c.
constexpr int TB = 32; // a core's rows of a block

// GLU round j of a K half: h[t][256 j ..] = SiLU(gate) * up of both slices
// [up 2j, gate 2j, up 2j+1, gate 2j+1], for tokens t = 0, 1; h is
// [2][half_len]
void glu2_round(bf16 *__restrict h, bf16 *__restrict rnd, int j, int half_len) {
  for (int t = 0; t < 2; t++)
    for (int s = 0; s < 2; s++)
      for (int r = 0; r < NC; r++) {
        const bf16 *u = rnd + (((2 * s) * NC + r) * 2 + t) * TB;
        const bf16 *g = rnd + (((2 * s + 1) * NC + r) * 2 + t) * TB;
        bf16 *y = h + t * half_len + 2 * SL * j + SL * s + TB * r;
        for (int i = 0; i < TB; i += V) {
          aie::vector<bf16, V> gv = aie::load_v<V>(g + i);
          aie::vector<bf16, V> act =
              aie::mul(gv, sigmoid(gv)).template to_vector<bf16>();
          aie::store_v(
              y + i,
              aie::mul(act, aie::load_v<V>(u + i)).template to_vector<bf16>());
        }
      }
}

// dst[t][512 j ..] = round j's rows of token t; dst is [2][n]
void copy2_round(bf16 *__restrict dst, bf16 *__restrict rnd, int j, int n) {
  for (int t = 0; t < 2; t++)
    for (int c = 0; c < NC; c++)
      for (int r = 0; r < NC; r++)
        for (int i = 0; i < TB; i += V)
          aie::store_v(dst + t * n + NC * SL * j + SL * c + TB * r + i,
                       aie::load_v<V>(rnd + ((c * NC + r) * 2 + t) * TB + i));
}

// y = rms_norm of token t of x [2][n] with norm weight k of w [2][n]
void rms_norm_tw(bf16 *__restrict y, bf16 *__restrict x, bf16 *__restrict w,
                 int n, int t, int k) {
  rms_norm(y, x + t * n, w + k * n, n);
}

// x[t][512 j ..] += round j's rows of token t, for both tokens of a
// two-token round; x is [2][n]
void residual_round2(bf16 *__restrict x, bf16 *__restrict rnd, int j, int n) {
  for (int t = 0; t < 2; t++)
    for (int c = 0; c < NC; c++)
      for (int r = 0; r < NC; r++) {
        bf16 *xa = x + t * n + NC * SL * j + SL * c + TB * r;
        const bf16 *ra = rnd + ((c * NC + r) * 2 + t) * TB;
        for (int i = 0; i < TB; i += V)
          aie::store_v(
              xa + i, aie::add(aie::load_v<V>(xa + i), aie::load_v<V>(ra + i)));
      }
}

// does nothing: a core whose loop calls no kernel compiles to a self-call,
// which the stack measurement takes for recursion
void core_tick(bf16 *__restrict x) { (void)x; }

// The verify pass's RMS core takes x and the norm weights a 1024-element
// round at a time, into one buffer: a channel that carries x once a wave and
// the weights each layer would need two buffers in a rotation they do not
// follow. x and y are [2][n], both tokens.

// x[1024 k ..] = rnd
void x_round(bf16 *__restrict x, bf16 *__restrict rnd, int k) {
  for (int i = 0; i < 1024; i += V)
    aie::store_v(x + 1024 * k + i, aie::load_v<V>(rnd + i));
}

// inv[t] = 1 / sqrt(mean(x[t]^2) + eps), as rms_norm takes it
void rms_inv2(bf16 *__restrict x, float *__restrict inv, int n) {
  for (int t = 0; t < 2; t++) {
    const bf16 *xt = x + t * n;
    aie::accum<accfloat, V> s0 = aie::zeros<accfloat, V>();
    aie::accum<accfloat, V> s1 = aie::zeros<accfloat, V>();
    for (int i = 0; i < n; i += 2 * V) {
      s0 = aie::mac_square(s0, aie::load_v<V>(xt + i));
      s1 = aie::mac_square(s1, aie::load_v<V>(xt + i + V));
    }
    float sum = aie::reduce_add(aie::add(s0.template to_vector<float>(),
                                         s1.template to_vector<float>()));
    sum = sum / (float)n + 1e-6f;
    inv[t] = aie::invsqrt(aie::broadcast<float, V>(sum))[0];
  }
}

// y[t][1024 k ..] = bf16(x[t] * w * inv[t]) over chunk k, both tokens, with
// the weight's chunk k in rnd, as rms_norm rounds it
void rms_apply2(bf16 *__restrict y, bf16 *__restrict x, float *__restrict inv,
                bf16 *__restrict rnd, int k, int n) {
  for (int t = 0; t < 2; t++) {
    const float it = inv[t];
    for (int i = 0; i < 1024; i += V) {
      int o = t * n + 1024 * k + i;
      aie::vector<float, V> wx =
          aie::mul(aie::load_v<V>(x + o), aie::load_v<V>(rnd + i));
      aie::store_v(y + o, aie::mul(wx, it).template to_vector<bf16>());
    }
  }
}

} // extern "C"
