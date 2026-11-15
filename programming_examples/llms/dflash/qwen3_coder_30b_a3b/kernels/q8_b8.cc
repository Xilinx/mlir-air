// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// One Q8_0 block of the LM head (32 rows x 256 columns) against 8 token rows:
// the int8 values go into the matmul as they are, and each 32-column group's
// row scales are applied after it.
//
// Block: scales bf16 [8 groups][32 rows], then int8 [256 columns][32 rows].

#include "q4k_b8.cc"

namespace {

constexpr int Q8_OFF = 2 * GROUPS * ROWS;

// acc[J] += limbs(p) * sc, row tile J's lanes
template <int J>
inline void scale_add(ACC p, aie::vector<bf16, 32> sc, float *acc) {
  VB s = cols8(sc.template extract<8>(J));
  VB h, l;
  limbs(p, h, l);
  ACC c;
  c.from_vector(aie::load_v<64>(acc + J * 64));
  c = aie::mac(c, h, s);
  c = aie::mac(c, l, s);
  aie::store_v(acc + J * 64, c.template to_vector<float>());
}

// row tiles J0 and J0 + 1 of the block
template <int J0>
inline void q8_pass(const char *xb, const uint8_t *w, float *acc) {
  const v2int32 ex = {EXP1, EXP1};
  BS xa((const bfp16ebs8 *)xb);
  const int8_t *q = (const int8_t *)(w + Q8_OFF);
  for (int g = 0; g < GROUPS; g++)
    chess_prepare_for_pipelining {
      ACC pa = aie::zeros<accfloat, 64>(), pb = pa;
#pragma clang loop unroll(full)
      for (int k = 0; k < 4; k++) {
        aie::vector<int8, 32> ta[2], tb[2];
#pragma clang loop unroll(full)
        for (int h = 0; h < 2; h++) {
          // 4 columns x 32 rows
          aie::vector<int8, 128> u = aie::load_v<128>(q);
          q += 128;
          auto e = aie::interleave_unzip(u.template extract<64>(0),
                                         u.template extract<64>(1), 8);
          auto a = aie::interleave_unzip(e.first.template extract<32>(0),
                                         e.first.template extract<32>(1), 8);
          auto b = aie::interleave_unzip(e.second.template extract<32>(0),
                                         e.second.template extract<32>(1), 8);
          ta[h] = J0 == 0 ? a.first : a.second;
          tb[h] = J0 == 0 ? b.first : b.second;
        }
        BV A;
        xa >> A;
        v64bfp16ebs8 va = {aie::transpose(aie::concat(ta[0], ta[1]), 8, 8), ex};
        v64bfp16ebs8 vb = {aie::transpose(aie::concat(tb[0], tb[1]), 8, 8), ex};
        pa = mma(pa, A, BV(va));
        pb = mma(pb, A, BV(vb));
      }
      aie::vector<bf16, 32> sc = aie::load_v<32>((const bf16 *)w + g * ROWS);
      scale_add<J0 == 0 ? 0 : 2>(pa, sc, acc);
      scale_add<J0 == 0 ? 1 : 3>(pb, sc, acc);
    }
}

} // namespace

extern "C" {

void q8_b8_zero(float *__restrict acc) { q4k_b8_zero(acc); }

// acc += chunk cb of x (bfp16 tiles) . W
void q8_b8_block(uint8_t *__restrict x, uint8_t *__restrict w,
                 float *__restrict acc, int32_t cb) {
  const char *xb = (const char *)x + cb * KT * BLK;
  q8_pass<0>(xb, w, acc);
  q8_pass<2>(xb, w, acc);
}

// the row block as bf16 [token][32 rows]
void q8_b8_flush(float *__restrict acc, bf16 *__restrict y) {
  q4k_b8_flush(acc, y);
}

} // extern "C"
