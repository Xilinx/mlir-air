// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The GLU core of the MoE block, after FastFlowLM's
// FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/kernels/glu.cc: it takes the GEMV
// array's rounds of 512 rows (4 columns x 128) and
//
//   - activates the shared expert's gate/up rounds ([up, gate, up, gate]
//     128 rows each) and assembles its down rows,
//   - activates the routed experts' rounds: a round is one 128-row slice of
//     up (or gate) of the 4 experts of a turn, column i the expert of rank i,
//   - adds score x down rows of each turn's experts into a float32 sum,
//
// then gates the shared expert and adds the routed sum (ffn.cc).

#include "ffn.cc"

namespace {

constexpr int SL = 128; // a column's rows of a round
constexpr int NC = 4;   // columns
constexpr int I = 512;  // an expert's intermediate width

void glu_n(bf16 *__restrict y, const bf16 *g, const bf16 *u) {
  for (int i = 0; i < SL; i += V) {
    aie::vector<bf16, V> gv = aie::load_v<V>(g + i);
    aie::vector<bf16, V> act =
        aie::mul(gv, sigmoid(gv)).template to_vector<bf16>();
    aie::store_v(
        y + i, aie::mul(act, aie::load_v<V>(u + i)).template to_vector<bf16>());
  }
}

} // namespace

extern "C" {

// Shared gate/up round j = [up 2j, gate 2j, up 2j+1, gate 2j+1]:
// hs[256 j ..] = SiLU(gate) * up for both slices.
void glu_shared_round(bf16 *__restrict hs, bf16 *__restrict rnd, int j) {
  for (int t = 0; t < 2; t++)
    glu_n(hs + 2 * SL * j + SL * t, rnd + 2 * SL * t + SL, rnd + 2 * SL * t);
}

// Experts' slice s: h[e][128 s ..] = SiLU(g[e]) * u[e], e the column.
void glu_expert_round(bf16 *__restrict h, bf16 *__restrict u,
                      bf16 *__restrict g, int s) {
  for (int e = 0; e < NC; e++)
    glu_n(h + e * I + s * SL, g + e * SL, u + e * SL);
}

// Down row group rg of the 4 experts of turn t:
// acc[128 rg ..] += score[4 t + e] * rnd[128 e ..]
void moe_acc_round(float *__restrict acc, bf16 *__restrict rnd,
                   float *__restrict score, int t, int rg) {
  for (int e = 0; e < NC; e++) {
    const float sc = score[NC * t + e];
    const bf16 hi = (bf16)sc;
    const bf16 lo = (bf16)(sc - (float)hi);
    float *a = acc + SL * rg;
    const bf16 *x = rnd + SL * e;
    for (int i = 0; i < SL; i += V) {
      aie::accum<accfloat, V> av;
      av.from_vector(aie::load_v<V>(a + i));
      aie::vector<bf16, V> xv = aie::load_v<V>(x + i);
      av = aie::mac(av, xv, lo);
      av = aie::mac(av, xv, hi);
      aie::store_v(a + i, av.template to_vector<float>());
    }
  }
}

// dst[512 j ..] = rnd: a dense phase's round j is its rows 512 j ..
void copy_round(bf16 *__restrict dst, bf16 *__restrict rnd, int j) {
  for (int i = 0; i < NC * SL; i += V)
    aie::store_v(dst + NC * SL * j + i, aie::load_v<V>(rnd + i));
}

// A round set aside, so the next one can land in the same buffer.
void copy_up(bf16 *__restrict dst, bf16 *__restrict rnd) {
  copy_round(dst, rnd, 0);
}

} // extern "C"
