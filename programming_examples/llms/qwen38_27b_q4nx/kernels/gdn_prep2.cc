// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Qwen3.6's Gated DeltaNet prep kernels (gdn_prep.cc) for two tokens, the
// speculative decode's verify pass. Two-token buffers are [2][n], token 1
// after token 0; a two-token round is [column][core][token][32 rows].

#include "gdn_prep.cc"

namespace {

constexpr int QKVN = DQ + DKV + DV; // conv channels: q | k | v
constexpr int TROWS = 32;           // a core's rows of a round

// y = SiLU(x0 w0 + x1 w1 + x2 w2 + x3 w3) over one 128-channel chunk
inline void conv_chunk(const bf16 *x0p, const bf16 *x1p, const bf16 *x2p,
                       const bf16 *x3p, const bf16 *w, bf16 *y) {
  constexpr int V = 16;
  for (int i = 0; i < HD; i += V) {
    aie::accum<accfloat, V> acc =
        aie::mul(aie::load_v<V>(x0p + i), aie::load_v<V>(w + i));
    acc = aie::mac(acc, aie::load_v<V>(x1p + i), aie::load_v<V>(w + HD + i));
    acc =
        aie::mac(acc, aie::load_v<V>(x2p + i), aie::load_v<V>(w + 2 * HD + i));
    acc =
        aie::mac(acc, aie::load_v<V>(x3p + i), aie::load_v<V>(w + 3 * HD + i));
    aie::vector<float, V> yf = acc.template to_vector<float>();
    aie::vector<bf16, V> yb = acc.template to_vector<bf16>();
    aie::vector<float, V> ys =
        aie::mul(yb, aie::broadcast<bf16, V>(-(bf16)LOG2_E));
    aie::accum<accfloat, V> den;
    den.from_vector(aie::exp2<bf16>(ys));
    den = aie::add(den, aie::broadcast<bf16, V>((bf16)1.0f));
    aie::vector<float, V> inv = aie::inv(den.template to_vector<float>());
    aie::accum<accfloat, V> s = aie::mul(inv, yf);
    aie::store_v(y + i, s.template to_vector<bf16>());
  }
}

} // namespace

extern "C" {

// gdn_conv_cache for two tokens: hx = [x0, x1, x2, w0, w1, w2, w3] from the
// cache, the tokens' new inputs a, b chunk c of a two-token round. Token 0 is
// conv(x0, x1, x2, a), token 1 conv(x1, x2, a, b), into y [2][QKVN]; the
// cache's next inputs ho = [x2, a, b]. The cache after token 0 alone, [x1, x2,
// a], the host takes from the input cache and ho.
void gdn_conv_cache2(bf16 *__restrict hx, bf16 *__restrict rnd,
                     bf16 *__restrict y, bf16 *__restrict ho, int c) {
  alignas(aie::vector_decl_align) bf16 a[HD];
  alignas(aie::vector_decl_align) bf16 b[HD];
  const int col = c % 4;
  for (int r = 0; r < 4; r++)
    for (int i = 0; i < TROWS; i += 16) {
      aie::store_v(a + r * TROWS + i,
                   aie::load_v<16>(rnd + ((col * 4 + r) * 2) * TROWS + i));
      aie::store_v(b + r * TROWS + i,
                   aie::load_v<16>(rnd + ((col * 4 + r) * 2 + 1) * TROWS + i));
    }
  const bf16 *w = hx + 3 * HD;
  conv_chunk(hx, hx + HD, hx + 2 * HD, a, w, y + c * HD);
  conv_chunk(hx + HD, hx + 2 * HD, a, b, w, y + QKVN + c * HD);
  for (int i = 0; i < HD; i += 16) {
    aie::store_v(ho + i, aie::load_v<16>(hx + 2 * HD + i));
    aie::store_v(ho + HD + i, aie::load_v<16>(a + i));
    aie::store_v(ho + 2 * HD + i, aie::load_v<16>(b + i));
  }
}

// gdn_l2norm of chunk c of token t of y [2][QKVN]
void gdn_l2norm_t(bf16 *__restrict y, int c, int scale, int t) {
  gdn_l2norm(y + t * QKVN, c, scale);
}

// ab [2][2 NV]
void ab_zero2(float *__restrict ab) {
  ab_zero(ab);
  ab_zero(ab + 2 * NV);
}

// ab_mvm of both tokens of x [2][n]
void ab_mvm2(bf16 *__restrict blk, bf16 *__restrict x, float *__restrict ab,
             int rb, int cb, int n) {
  ab_mvm(blk, x, ab, rb, cb);
  ab_mvm(blk, x + n, ab + 2 * NV, rb, cb);
}

void ab_activate2(float *__restrict ab, float *__restrict a_dt) {
  ab_activate(ab, a_dt);
  ab_activate(ab + 2 * NV, a_dt);
}

// gdn_pack of value head h of token t
void gdn_pack_t(bf16 *__restrict conv, float *__restrict ab,
                bf16 *__restrict kvq, int h, int t) {
  gdn_pack(conv + t * QKVN, ab + t * 2 * NV, kvq, h);
}

} // extern "C"
