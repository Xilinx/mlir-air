// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Gemma4 attention for an eight-token block on one shard: two tokens and the
// four query heads of one KV head, so the eight rows of an mmul<8,8,8> are
// (token, head) pairs, row = token * 4 + head.
//
// Q, K and V all arrive as [DH/8][row or key][8]: Q one frame of eight rows,
// K and V eight keys at a time. Each row keeps an online
// softmax state (running max, denominator) in float and its output
// accumulator as [DH/8][row][8] in float.
//
// A row of token t sees keys [visible + t - window, visible + t), `visible`
// being the first token's count and window 0 meaning no lower bound. BFP16
// shares one exponent between the eight keys of a V tile, so in a block that
// some row sees only part of, the rows of each token are computed with the
// keys it does not see zeroed first. A block is cut by the window's start or
// by the causal end, never both (the window is far wider than a block): the
// less restricted token goes first, so each zeroing only removes keys the
// next token does not see either.
//
// A sliding layer (window != 0) has its 256-wide heads spread into the
// 512-wide slots, q and k as [lo | 0 | hi | 0] and v as [v | 0], so its
// products skip the zero quarters and halves.
//
// The running max moves only when a score passes it by more than LAZY_MAX,
// so the output's rescale (float x float, emulated) is rare; probabilities
// then reach exp(LAZY_MAX) before the final division.

#include "aie_api/aie.hpp"
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

#ifndef G8_DH
#define G8_DH 512
#endif
#ifndef G8_SCALE
#define G8_SCALE 1.0f
#endif

constexpr int DH8 = G8_DH / 8;
constexpr float LAZY_MAX = 8.0f;

// d tile i of a sliding head's q and k: the two real quarters of the slot
static inline int qk_tile(int i, bool sliding) {
  return sliding ? (i & (DH8 / 4 - 1)) + (i / (DH8 / 4)) * (DH8 / 2) : i;
}
constexpr int ROWS = 8;
constexpr int KEYS = 8;
constexpr float NEG = -3.0e38f;

using MM = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;

static inline aie::vector<float, 16> exp16(aie::vector<float, 16> x) {
  // getExpBf16 takes bf16; clamp so masked entries stay finite.
  auto lo = aie::broadcast<float, 16>(-87.0f);
  x = aie::max(x, lo);
  aie::accum<accfloat, 16> a;
  a.from_vector(x);
  return aie::accum<accfloat, 16>(getExpBf16(a.to_vector<bf16>()))
      .to_vector<float>();
}

extern "C" {

// Keep this shard's Q frame; reset the softmax state.
void g8_start(bf16 *__restrict frame, bf16 *__restrict q, float *__restrict y,
              float *__restrict m, float *__restrict l) {
  for (int i = 0; i < DH8 * 64; i += 64)
    aie::store_v(q + i, aie::load_v<64>(frame + i));
  for (int i = 0; i < DH8 * 64; i += 16)
    aie::store_v(y + i, aie::zeros<float, 16>());
  for (int r = 0; r < ROWS; ++r) {
    m[r] = NEG;
    l[r] = 0.0f;
  }
}

// One block of eight keys, keys [8 * block, 8 * block + 8).
void g8_block(bf16 *__restrict q, bf16 *__restrict k, bf16 *__restrict v,
              float *__restrict y, float *__restrict m, float *__restrict l,
              int block, int visible, int window) {
  aie_round_nearest_even();
  const int base = block * KEYS;
  int seen[2], from[2];
  for (int t = 0; t < 2; ++t) {
    int s = visible + t - base;
    seen[t] = s < 0 ? 0 : (s > KEYS ? KEYS : s);
    int f = window ? visible + t - window - base : 0;
    from[t] = f < 0 ? 0 : (f > KEYS ? KEYS : f);
  }
  if (seen[1] == 0 || from[0] >= KEYS)
    return;
  const bool sliding = window != 0;
  const int nqk = sliding ? DH8 / 2 : DH8; // q/k d tiles with data
  const int npv = sliding ? DH8 / 2 : DH8; // v d tiles with data

  // S = Q K^T, rows x keys. K is used as BFP16 through the transpose
  // modifier, so no 8x8 shuffle per tile.
  aie::vector<float, 64> s;
  {
    using MK = aie::mmul<8, 8, 8, bf16, bfp16ebs8, accfloat>;
    const auto one = ::concat(::broadcast_one_to_v32bfloat16(),
                              ::broadcast_one_to_v32bfloat16());
    MK acc(aie::zeros<bf16, 64>());
#pragma clang loop unroll(disable)
    for (int i = 0; i < nqk; ++i) {
      const int d = qk_tile(i, sliding);
      aie::accum<accfloat, 64> kx =
          ::mul_elem_64(aie::load_v<64>(k + d * 64), one);
      aie::block_vector<bfp16ebs8, 64> kb(::to_v64bfp16ebs8(kx));
      acc.mac(aie::load_v<64>(q + d * 64), aie::op_transpose(kb));
    }
    s = aie::mul(acc.to_vector<float>(), G8_SCALE).to_vector<float>();
  }

  // Every row sees all eight keys (the usual block): no masks. Rows of the
  // later token see at least the earlier token's keys.
  const bool whole = seen[0] == KEYS && from[1] == 0;
  // Masks: element i is (row i / 8, key i % 8); rows 0..3 are token 0.
  // Built as numbers, not as a vector mask the backend cannot index.
  alignas(32) float off[64], keep[64];
  if (!whole) {
    for (int i = 0; i < 64; ++i) {
      const int kk = i % 8, t = i / 32;
      const bool on = kk >= from[t] && kk < seen[t];
      off[i] = on ? 0.0f : NEG;
      keep[i] = on ? 1.0f : 0.0f;
    }
    s = aie::add(s, aie::load_v<64>(off));
  }

  // Per-row max, correction and probabilities.
  alignas(32) float mnew[16], mold[16];
  bool same = true;
#pragma clang loop unroll(full)
  for (int r = 0; r < ROWS; ++r) {
    const float rm = aie::reduce_max(s.extract<8>(r));
    mold[r] = m[r];
    mnew[r] = rm > m[r] + LAZY_MAX ? rm : m[r];
    m[r] = mnew[r];
    same &= mnew[r] == mold[r];
  }
  aie::vector<float, 64> mv;
#pragma clang loop unroll(full)
  for (int r = 0; r < ROWS; ++r)
    mv.insert(r, aie::broadcast<float, 8>(mnew[r]));
  aie::vector<float, 64> e;
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i)
    e.insert(i, exp16(aie::sub(s.extract<16>(i), mv.extract<16>(i))));
  if (!whole)
    e = aie::mul(e, aie::load_v<64>(keep)).to_vector<float>();
  aie::vector<float, 64> cv;
  if (same) {
#pragma clang loop unroll(full)
    for (int r = 0; r < ROWS; ++r)
      l[r] += aie::reduce_add(e.extract<8>(r));
  } else {
    for (int r = ROWS; r < 16; ++r)
      mold[r] = mnew[r] = 0.0f;
    const auto corr =
        exp16(aie::sub(aie::load_v<16>(mold), aie::load_v<16>(mnew)));
#pragma clang loop unroll(full)
    for (int r = 0; r < ROWS; ++r) {
      l[r] = l[r] * corr.get(r) + aie::reduce_add(e.extract<8>(r));
      cv.insert(r, aie::broadcast<float, 8>(corr.get(r)));
    }
  }
  aie::vector<bf16, 64> p;
  {
    aie::accum<accfloat, 64> a;
    a.from_vector(e);
    p = a.to_vector<bf16>();
  }

  if (whole) {
    // V cannot take the transpose modifier (it arrives key-major), so P
    // stays BF16; the accumulator starts from the rescaled output.
    if (same) {
#pragma clang loop unroll_count(2)
      for (int d = 0; d < npv; ++d) {
        aie::accum<accfloat, 64> a0;
        a0.from_vector(aie::load_v<64>(y + d * 64));
        MM pv(a0);
        pv.mac(p, aie::load_v<64>(v + d * 64));
        aie::store_v(y + d * 64, pv.template to_vector<float>());
      }
    } else {
#pragma clang loop unroll_count(2)
      for (int d = 0; d < npv; ++d) {
        MM pv(aie::mul(cv, aie::load_v<64>(y + d * 64)));
        pv.mac(p, aie::load_v<64>(v + d * 64));
        aie::store_v(y + d * 64, pv.template to_vector<float>());
      }
    }
    return;
  }
  // A partial block: each token's rows with the keys it does not see zeroed,
  // the less restricted token first.
  const bool cut_low = from[1] > 0;
  for (int i = 0; i < 2; ++i) {
    const int t = cut_low ? i : 1 - i;
    for (int d = 0; d < npv; ++d)
      for (int j = 0; j < KEYS; ++j)
        if (j < from[t] || j >= seen[t])
          aie::store_v(v + d * 64 + j * 8, aie::zeros<bf16, 8>());
    const auto pt = t ? aie::concat(aie::zeros<bf16, 32>(), p.extract<32>(1))
                      : aie::concat(p.extract<32>(0), aie::zeros<bf16, 32>());
    for (int d = 0; d < npv; ++d) {
      MM pv(aie::zeros<bf16, 64>());
      pv.mac(pt, aie::load_v<64>(v + d * 64));
      float *yd = y + d * 64 + t * 32;
      auto old = same ? aie::load_v<32>(yd)
                      : aie::mul(cv.extract<32>(t), aie::load_v<32>(yd))
                            .to_vector<float>();
      aie::store_v(yd, aie::add(old, pv.to_vector<float>().extract<32>(t)));
    }
  }
}

// o [ROWS][DH] = y / l.
void g8_finish(float *__restrict y, float *__restrict l, bf16 *__restrict o) {
  for (int r = 0; r < ROWS; ++r) {
    const float inv = 1.0f / l[r];
    for (int d = 0; d < DH8; ++d) {
      auto x = aie::mul(aie::load_v<8>(y + d * 64 + r * 8), inv);
      aie::store_v(o + r * G8_DH + d * 8, x.to_vector<bf16>());
    }
  }
}
}
