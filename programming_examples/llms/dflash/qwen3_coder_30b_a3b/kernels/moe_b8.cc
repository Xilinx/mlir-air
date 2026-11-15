// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The 8-row MoE block's kernels. Each array column runs one expert a turn;
// a core holds row block r of every row group of it.
//
// A core's message buffer is [header | 8 chunks]: x as bfp16 tiles lands in
// the chunks, then an H message, [header | 3 chunks of its column's GLU
// output], over the front. Header (float words): [16 + 8 c + t] column c's
// score for token t this turn.
//
// A core's GLU output: per slice s, 4 tiles [8 tokens][8 h rows], rows
// 128 s + 32 r + 8 j of tile j.

#pragma once

#include "q4k_b8.cc"

namespace {

constexpr int HDR = 256;        // header bytes
constexpr int CHUNK = KT * BLK; // a 256-column chunk of 8 rows, bfp16
constexpr int HMSG = HDR + 3 * CHUNK;
constexpr int SLICES = 6; // 128-row slices of an expert's up
constexpr int PAY = 16;   // bf16 offset of an egress payload

inline aie::vector<bf16, 16> sigmoid(aie::vector<bf16, 16> x) {
  const aie::vector<bf16, 16> half = aie::broadcast<bf16, 16>(0.5f);
  aie::accum<accfloat, 16> xh = aie::mul(x, half);
  aie::vector<bf16, 16> t = aie::tanh<bf16>(xh.template to_vector<float>());
  t = aie::add(t, aie::broadcast<bf16, 16>(1.0f));
  return aie::mul(t, half).template to_vector<bf16>();
}

} // namespace

extern "C" {

// ---- array cores

// sc[0:8] = column c's scores
void moe_scores(uint8_t *__restrict msg, float *__restrict sc, int32_t c) {
  const float *s = (const float *)msg + 16 + 8 * c;
  for (int t = 0; t < 8; t++)
    sc[t] = s[t];
}

void moe_zero(float *__restrict acc, int32_t n) {
  for (int i = 0; i < n; i += 32)
    aie::store_v(acc + i, aie::zeros<float, 32>());
}

// the same, for the running sum
void moe_clear(float *__restrict moe, int32_t n) { moe_zero(moe, n); }

// acc[256 a ..] += chunk cb of x (bfp16 tiles, at msg + off) . W; sp: 4 KB
void moe_block(uint8_t *__restrict msg, uint8_t *__restrict w,
               float *__restrict acc, uint8_t *__restrict sp, int32_t cb,
               int32_t a, int32_t off) {
  q4k_tiles((const char *)msg + off + cb * CHUNK, w, acc + 256 * a, (char *)sp);
}

// the same, x from an H message
void moe_block_h(uint8_t *__restrict msg, uint8_t *__restrict w,
                 float *__restrict acc, uint8_t *__restrict sp, int32_t cb,
                 int32_t a, int32_t off) {
  moe_block(msg, w, acc, sp, cb, a, off);
}

// slice s of the GLU into an egress payload: 4 tiles of SiLU(gate) * up,
// up = acc[0:256], gate = acc[256:512], each [row tile][token][8]
void moe_glu(float *__restrict acc, bf16 *__restrict y) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  OS os((bfp16ebs8 *)(y + PAY));
  for (int j = 0; j < NT; j++) {
    alignas(aie::vector_decl_align) bf16 h[64];
    for (int i = 0; i < 64; i += 16) {
      aie::accum<accfloat, 16> ua, ga;
      ua.from_vector(aie::load_v<16>(acc + j * 64 + i));
      ga.from_vector(aie::load_v<16>(acc + 256 + j * 64 + i));
      aie::vector<bf16, 16> u = ua.template to_vector<bf16>();
      aie::vector<bf16, 16> g = ga.template to_vector<bf16>();
      aie::vector<bf16, 16> act =
          aie::mul(g, sigmoid(g)).template to_vector<bf16>();
      aie::store_v(h + i, aie::mul(act, u).template to_vector<bf16>());
    }
    ACC a;
    a.from_vector(aie::load_v<64>(h));
    os << a.template to_vector<bfp16ebs8>();
  }
}

// moe[256 rg ..] += score[token] * acc, both [row tile][token][8]; the
// score as two bf16 limbs, acc rounded to bf16 as the decode's down rows are
void moe_acc(float *__restrict moe, float *__restrict acc, float *__restrict sc,
             int32_t rg) {
  alignas(aie::vector_decl_align) bf16 hi[8], lo[8];
  for (int t = 0; t < 8; t++) {
    hi[t] = (bf16)sc[t];
    lo[t] = (bf16)(sc[t] - (float)hi[t]);
  }
  const VB sh = rows8(aie::load_v<8>(hi));
  const VB sl = rows8(aie::load_v<8>(lo));
  float *m = moe + 256 * rg;
  for (int j = 0; j < NT; j++) {
    ACC a;
    a.from_vector(aie::load_v<64>(acc + j * 64));
    VB v = a.template to_vector<bf16>();
    ACC c;
    c.from_vector(aie::load_v<64>(m + j * 64));
    c = aie::mac(c, v, sl);
    c = aie::mac(c, v, sh);
    aie::store_v(m + j * 64, c.template to_vector<float>());
  }
}

// row group rg of moe as bf16 after the header slot
void moe_out(float *__restrict moe, bf16 *__restrict y, int32_t rg) {
  for (int i = 0; i < 256; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(moe + 256 * rg + i));
    aie::store_v(y + PAY + i, a.template to_vector<bf16>());
  }
}

// ---- feeder

// the headers of the 4 H messages hm [4][HMSG]: turn t's scores. plan
// (int32 words): [8 + slot] the experts, [80 + 8 slot + t] (float) each
// slot's scores
void fd_header(uint8_t *__restrict hm, uint8_t *__restrict plan, int32_t t) {
  const float *ps = (const float *)plan + 80;
  for (int m = 0; m < 4; m++) {
    float *hs = (float *)(hm + m * HMSG) + 16;
    for (int c = 0; c < 4; c++)
      for (int k = 0; k < 8; k++)
        hs[8 * c + k] = ps[8 * (4 * t + c) + k];
  }
}

// dst[0:n] = src[0:n], n a multiple of 16 bytes
void fd_copy(uint8_t *__restrict src, uint8_t *__restrict dst, int32_t n) {
  for (int i = 0; i < n; i += 16)
    aie::store_v(dst + i, aie::load_v<16>(src + i));
}

// egress round s ([column][core][512 bytes], slice s's 4 tiles first) into
// the columns' H messages
void fd_scatter(bf16 *__restrict rnd, uint8_t *__restrict hm, int32_t s) {
  const uint8_t *src = (const uint8_t *)rnd;
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < 4; r++) {
      // the core's 4 tiles go together: rows 32 r .. of slice s
      const int tile = (s % 2) * 16 + r * 4;
      uint8_t *d = hm + c * HMSG + HDR + (s / 2) * CHUNK + tile * BLK;
      const uint8_t *q = src + (c * 4 + r) * 512;
      for (int i = 0; i < NT * BLK; i += 8)
        *(uint64_t *)(d + i) = *(const uint64_t *)(q + i);
    }
}

} // extern "C"
