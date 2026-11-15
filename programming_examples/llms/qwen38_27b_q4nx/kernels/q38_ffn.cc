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

} // extern "C"
