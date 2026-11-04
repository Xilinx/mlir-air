// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GPT-OSS decode attention with sinks, transcribed from FastFlowLM
// FLM_Xclbin/GPTOSS/gpt_npu_bin/kernel/{attn_qk.cc,attn_kv.cc}. One unit is 16
// query heads (two KV heads of eight) over 16-key blocks. FLM's attn_qk/attn_kv
// own the block loop and the locks; here AIR calls one block at a time. The
// sink of each head seeds its running max, and exp(sink - m) joins the
// softmax denominator in the final scale.
//
// q: 16 heads x 64, then the 16 sinks; FLM's qk core hands the kv core the
// sinks as a separate buffer. k, v blocks: 2 KV heads x 16 keys x 64.
// s block: 16x16 scores, then c[16] and m[16] as floats.

#include <stdint.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

typedef bfloat16 bf16;

namespace {

constexpr int DH = 64;
constexpr float scale = 0.125f;

inline aie::vector<bf16, 16>
update(float *m, float *c, aie::vector<bf16, 16> &out, aie::mask<16> &mask) {
  bf16 vm = aie::reduce_max(out);
  bf16 vmax = aie::max(vm, (bf16)*m);
  aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);

  constexpr int min_clamp = -87.0f;
  constexpr int max_clamp = 88.0f;
  aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
  aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp);

  aie::vector<bf16, 16> Vec = aie::clamp(Vecsub, min_value, max_value);
  aie::accum<accfloat, 16> Outvec = getExpBf16(Vec);
  Vec = Outvec.template to_vector<bf16>(0);
  Vec = aie::select((bf16)0, Vec, mask);

  bf16 correct = aie::sub((bf16)*m, vmax);
  aie::vector<bf16, 16> correct_vec = aie::broadcast<bf16, 16>(correct);
  correct_vec = aie::clamp(correct_vec, min_value, max_value);
  aie::accum<accfloat, 16> correct_acc = getExpBf16(correct_vec);
  aie::vector<float, 16> correct_float =
      correct_acc.template to_vector<float>();

  *m = (float)vmax;
  *c = (float)correct_float.get(0);
  return Vec;
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY,
              float *__restrict m, float *__restrict c, aie::mask<16> &mask) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  {
    bf16 *__restrict pQ1 = pQ;
    bf16 *__restrict pY1 = pY;
    bf16 *__restrict pK1 = pK;
    bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    for (unsigned i = 1; i < colQ; ++i) {
      A0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;

      B00 = aie::load_v<MMUL::size_B>(pK1);
      B0 = aie::transpose(B00, 8, 8);
      pK1 += MMUL::size_B * 2;
      B01 = aie::load_v<MMUL::size_B>(pK2);
      B1 = aie::transpose(B01, 8, 8);
      pK2 += MMUL::size_B * 2;

      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);

    aie::vector<bf16, 32> mout2, mout3, mout4, mout5;

    mout2 = aie::filter_even(mout0.first, 32);
    mout3 = aie::filter_odd(mout0.first, 32);
    mout4 = aie::filter_even(mout0.second, 32);
    mout5 = aie::filter_odd(mout0.second, 32);

    mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
    mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);
    mout4 = aie::mul(mout4, (bf16)scale).template to_vector<bf16>(0);
    mout5 = aie::mul(mout5, (bf16)scale).template to_vector<bf16>(0);

    auto out0 = aie::filter_even(mout2, 16);
    auto out1 = aie::filter_odd(mout2, 16);
    auto out2 = aie::filter_even(mout3, 16);
    auto out3 = aie::filter_odd(mout3, 16);
    auto out4 = aie::filter_even(mout4, 16);
    auto out5 = aie::filter_odd(mout4, 16);
    auto out6 = aie::filter_even(mout5, 16);
    auto out7 = aie::filter_odd(mout5, 16);

    aie::vector<bf16, 16> vec = update(m, c, out0, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 1, c + 1, out1, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 2, c + 2, out2, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 3, c + 3, out3, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 4, c + 4, out4, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 5, c + 5, out5, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 6, c + 6, out6, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 7, c + 7, out7, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
  }
  {
    bf16 *__restrict pQ1 = pQ + 8 * MMUL::size_A;
    bfloat16 *__restrict pY1 = pY + 8 * 16;
    bf16 *__restrict pK1 = pK + 8 * MMUL::size_B * 2;
    bf16 *__restrict pK2 = pK + 8 * MMUL::size_B * 2 + 1 * MMUL::size_B;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    for (unsigned i = 1; i < colQ; ++i) {
      A0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;

      B00 = aie::load_v<MMUL::size_B>(pK1);
      B0 = aie::transpose(B00, 8, 8);
      pK1 += MMUL::size_B * 2;
      B01 = aie::load_v<MMUL::size_B>(pK2);
      B1 = aie::transpose(B01, 8, 8);
      pK2 += MMUL::size_B * 2;

      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);

    aie::vector<bf16, 32> mout2, mout3, mout4, mout5;
    mout2 = aie::filter_even(mout0.first, 32);
    mout3 = aie::filter_odd(mout0.first, 32);
    mout4 = aie::filter_even(mout0.second, 32);
    mout5 = aie::filter_odd(mout0.second, 32);

    mout2 = aie::mul(mout2, (bf16)scale).template to_vector<bf16>(0);
    mout3 = aie::mul(mout3, (bf16)scale).template to_vector<bf16>(0);
    mout4 = aie::mul(mout4, (bf16)scale).template to_vector<bf16>(0);
    mout5 = aie::mul(mout5, (bf16)scale).template to_vector<bf16>(0);

    auto out0 = aie::filter_even(mout2, 16);
    auto out1 = aie::filter_odd(mout2, 16);
    auto out2 = aie::filter_even(mout3, 16);
    auto out3 = aie::filter_odd(mout3, 16);
    auto out4 = aie::filter_even(mout4, 16);
    auto out5 = aie::filter_odd(mout4, 16);
    auto out6 = aie::filter_even(mout5, 16);
    auto out7 = aie::filter_odd(mout5, 16);

    aie::vector<bf16, 16> vec = update(m + 8, c + 8, out0, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 9, c + 9, out1, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 10, c + 10, out2, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 11, c + 11, out3, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 12, c + 12, out4, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 13, c + 13, out5, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 14, c + 14, out6, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
    vec = update(m + 15, c + 15, out7, mask);
    aie::store_v(pY1, vec);
    pY1 += 16;
  }
}

void reorder_s(bf16 *__restrict pS) {
  aie::vector<bf16, 64> S0 = aie::load_v<64>(pS);
  aie::vector<bf16, 64> S1 = aie::load_v<64>(pS + 64);
  aie::vector<bf16, 64> S2 = aie::load_v<64>(pS + 128);
  aie::vector<bf16, 64> S3 = aie::load_v<64>(pS + 192);

  aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
  aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
  aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
  aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);
  aie::vector<bf16, 32> S20 = aie::filter_even(S2, 8);
  aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 8);
  aie::vector<bf16, 32> S30 = aie::filter_even(S3, 8);
  aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 8);

  S0 = aie::concat(S00, S10);
  S1 = aie::concat(S01, S11);
  S2 = aie::concat(S20, S30);
  S3 = aie::concat(S21, S31);

  aie::store_v(pS, S0);
  aie::store_v(pS + 64, S1);
  aie::store_v(pS + 128, S2);
  aie::store_v(pS + 192, S3);
}

void calculate_l(bf16 *__restrict pS, float *__restrict c,
                 float *__restrict l) {
  using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
  bf16 *__restrict pS1 = pS;

  aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1);

  aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

  aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>();
  MMUL C0(acc_C0);
  C0.mac(S0, ones);
  C0.mac(S1, ones);
  aie::vector<float, MMUL::size_C> sumf = C0.template to_vector<float>();
  auto sum03 = aie::filter_even(sumf, 4);
  auto sum02 = aie::filter_even(sum03, 2);
  auto sum01 = aie::filter_even(sum02, 1);

  aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>();
  MMUL C1(acc_C1);
  C1.mac(S2, ones);
  C1.mac(S3, ones);
  aie::vector<float, MMUL::size_C> sumf1 = C1.template to_vector<float>();
  auto sum13 = aie::filter_even(sumf1, 4);
  auto sum12 = aie::filter_even(sum13, 2);
  auto sum11 = aie::filter_even(sum12, 1);

  auto sum = aie::concat(sum01, sum11);
  aie::accum<accfloat, 16> sum_acc;
  sum_acc.from_vector(sum);

  aie::vector<float, 16> c_float = aie::load_v<16>(c);
  aie::vector<float, 16> l_float = aie::load_v<16>(l);

  aie::accum<accfloat, 16> l_acc;
  l_acc = mac_elem_16_accuracy_safe(l_float, c_float, sum_acc, 0, 0, 0);
  aie::store_v(l, l_acc.template to_vector<float>());
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV, float *__restrict pY,
             float *__restrict c) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bf16 *__restrict pS1 = pS;

  aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1);

  {
    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);
    aie::vector<float, 64> CORRECT =
        aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    bf16 *__restrict pV1 = pV;
    float *__restrict pY1 = pY;

    for (unsigned j = 0; j < colQ; j += 4) {
      // v0
      bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
      bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;
      bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
      bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

      aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
      pV01 += MMUL::size_B * colQ * 2;
      aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
      pV02 += MMUL::size_B * colQ * 2;
      aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
      pV03 += MMUL::size_B * colQ * 2;
      aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
      pV04 += MMUL::size_B * colQ * 2;

      aie::vector<float, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
      aie::vector<float, MMUL::size_C> acc_Y01 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
      aie::vector<float, MMUL::size_C> acc_Y02 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
      aie::vector<float, MMUL::size_C> acc_Y03 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

      aie::accum<accfloat, MMUL::size_C> ACC_Y00;
      aie::accum<accfloat, MMUL::size_C> ACC_Y01;
      aie::accum<accfloat, MMUL::size_C> ACC_Y02;
      aie::accum<accfloat, MMUL::size_C> ACC_Y03;
      ACC_Y00 = aie::mul(CORRECT, acc_Y00);
      ACC_Y01 = aie::mul(CORRECT, acc_Y01);
      ACC_Y02 = aie::mul(CORRECT, acc_Y02);
      ACC_Y03 = aie::mul(CORRECT, acc_Y03);

      MMUL Y00(ACC_Y00);
      MMUL Y01(ACC_Y01);
      MMUL Y02(ACC_Y02);
      MMUL Y03(ACC_Y03);

      Y00.mac(S0, V00);
      Y01.mac(S0, V01);
      Y02.mac(S0, V02);
      Y03.mac(S0, V03);

      V00 = aie::load_v<MMUL::size_B>(pV01);
      V01 = aie::load_v<MMUL::size_B>(pV02);
      V02 = aie::load_v<MMUL::size_B>(pV03);
      V03 = aie::load_v<MMUL::size_B>(pV04);

      Y00.mac(S1, V00);
      Y01.mac(S1, V01);
      Y02.mac(S1, V02);
      Y03.mac(S1, V03);

      aie::store_v(pY1, Y00.template to_vector<float>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y01.template to_vector<float>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y02.template to_vector<float>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y03.template to_vector<float>());
      pY1 += MMUL::size_C;
    }
  }

  {
    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[8]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[9]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[10]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[11]);
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[12]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[13]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[14]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[15]);
    aie::vector<float, 64> CORRECT =
        aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    bf16 *__restrict pV2 = pV + MMUL::size_B * colQ;
    float *__restrict pY2 = pY + MMUL::size_C * colQ;

    for (unsigned j = 0; j < colQ; j += 4) {
      // v1
      bf16 *__restrict pV11 = pV2 + (j + 0) * MMUL::size_B;
      bf16 *__restrict pV12 = pV2 + (j + 1) * MMUL::size_B;
      bf16 *__restrict pV13 = pV2 + (j + 2) * MMUL::size_B;
      bf16 *__restrict pV14 = pV2 + (j + 3) * MMUL::size_B;

      aie::vector<bf16, MMUL::size_B> V10 = aie::load_v<MMUL::size_B>(pV11);
      pV11 += MMUL::size_B * colQ * 2;
      aie::vector<bf16, MMUL::size_B> V11 = aie::load_v<MMUL::size_B>(pV12);
      pV12 += MMUL::size_B * colQ * 2;
      aie::vector<bf16, MMUL::size_B> V12 = aie::load_v<MMUL::size_B>(pV13);
      pV13 += MMUL::size_B * colQ * 2;
      aie::vector<bf16, MMUL::size_B> V13 = aie::load_v<MMUL::size_B>(pV14);
      pV14 += MMUL::size_B * colQ * 2;

      aie::vector<float, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);
      aie::vector<float, MMUL::size_C> acc_Y11 =
          aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);
      aie::vector<float, MMUL::size_C> acc_Y12 =
          aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C * 2);
      aie::vector<float, MMUL::size_C> acc_Y13 =
          aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C * 3);

      aie::accum<accfloat, MMUL::size_C> ACC_Y10;
      aie::accum<accfloat, MMUL::size_C> ACC_Y11;
      aie::accum<accfloat, MMUL::size_C> ACC_Y12;
      aie::accum<accfloat, MMUL::size_C> ACC_Y13;
      ACC_Y10 = aie::mul(CORRECT, acc_Y10);
      ACC_Y11 = aie::mul(CORRECT, acc_Y11);
      ACC_Y12 = aie::mul(CORRECT, acc_Y12);
      ACC_Y13 = aie::mul(CORRECT, acc_Y13);

      MMUL Y10(ACC_Y10);
      MMUL Y11(ACC_Y11);
      MMUL Y12(ACC_Y12);
      MMUL Y13(ACC_Y13);

      Y10.mac(S2, V10);
      Y11.mac(S2, V11);
      Y12.mac(S2, V12);
      Y13.mac(S2, V13);

      V10 = aie::load_v<MMUL::size_B>(pV11);
      V11 = aie::load_v<MMUL::size_B>(pV12);
      V12 = aie::load_v<MMUL::size_B>(pV13);
      V13 = aie::load_v<MMUL::size_B>(pV14);

      Y10.mac(S3, V10);
      Y11.mac(S3, V11);
      Y12.mac(S3, V12);
      Y13.mac(S3, V13);

      aie::store_v(pY2, Y10.template to_vector<float>());
      pY2 += MMUL::size_C;
      aie::store_v(pY2, Y11.template to_vector<float>());
      pY2 += MMUL::size_C;
      aie::store_v(pY2, Y12.template to_vector<float>());
      pY2 += MMUL::size_C;
      aie::store_v(pY2, Y13.template to_vector<float>());
      pY2 += MMUL::size_C;
    }
  }
}

template <unsigned dv>
void scale_div_aie(bf16 *o, float *a, float *l, bf16 *sink, float *m) {

  constexpr int vec_factor = 64;
  const int F = 8 * dv / vec_factor;

  aie::vector<bf16, 16> sink_vec = aie::load_v<16>(sink);
  aie::accum<accfloat, 16> sink_acc;
  sink_acc.from_vector(sink_vec);
  aie::vector<float, 16> sink_float = sink_acc.template to_vector<float>();
  aie::vector<float, 16> m_vec = aie::load_v<16>(m);
  aie::vector<float, 16> sink_sub_vec = aie::sub(sink_float, m_vec);

  constexpr int min_clamp = -87.0f;
  constexpr int max_clamp = 88.0f;
  aie::vector<float, 16> min_value = aie::broadcast<float, 16>(min_clamp);
  aie::vector<float, 16> max_value = aie::broadcast<float, 16>(max_clamp);

  sink_sub_vec = aie::clamp(sink_sub_vec, min_value, max_value);
  aie::accum<accfloat, 16> sink_sub_acc;
  sink_sub_acc.from_vector(sink_sub_vec);
  sink_sub_acc = getExpBf16(sink_sub_acc.template to_vector<bf16>());
  aie::vector<float, 16> sink_sub_float =
      sink_sub_acc.template to_vector<float>();

  {

    float *pA1 = a;
    bf16 *pO1 = o;
    bf16 l00 = (bf16)getInvBf16(l[0] + sink_sub_float.get(0));
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
    bf16 l01 = (bf16)getInvBf16(l[1] + sink_sub_float.get(1));
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
    bf16 l02 = (bf16)getInvBf16(l[2] + sink_sub_float.get(2));
    aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
    bf16 l03 = (bf16)getInvBf16(l[3] + sink_sub_float.get(3));
    aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);
    bf16 l04 = (bf16)getInvBf16(l[4] + sink_sub_float.get(4));
    aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l04);
    bf16 l05 = (bf16)getInvBf16(l[5] + sink_sub_float.get(5));
    aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l05);
    bf16 l06 = (bf16)getInvBf16(l[6] + sink_sub_float.get(6));
    aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l06);
    bf16 l07 = (bf16)getInvBf16(l[7] + sink_sub_float.get(7));
    aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l07);
    auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    for (int i = 0; i < F; i++) {
      aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      aie::accum<accfloat, vec_factor> A0_acc;
      A0_acc.from_vector(A0);
      aie::accum<accfloat, vec_factor> AL =
          aie::mul(A0_acc.template to_vector<bf16>(), L);
      aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
      aie::store_v(pO1, AL_bf16);
      pO1 += vec_factor;
      pA1 += vec_factor;
    }
  }

  {
    float *pA1 = a + 8 * dv;
    bf16 *pO1 = o + 8 * dv;
    bf16 l00 = (bf16)getInvBf16(l[8] + sink_sub_float.get(8));
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
    bf16 l01 = (bf16)getInvBf16(l[9] + sink_sub_float.get(9));
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
    bf16 l02 = (bf16)getInvBf16(l[10] + sink_sub_float.get(10));
    aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
    bf16 l03 = (bf16)getInvBf16(l[11] + sink_sub_float.get(11));
    aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);
    bf16 l04 = (bf16)getInvBf16(l[12] + sink_sub_float.get(12));
    aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l04);
    bf16 l05 = (bf16)getInvBf16(l[13] + sink_sub_float.get(13));
    aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l05);
    bf16 l06 = (bf16)getInvBf16(l[14] + sink_sub_float.get(14));
    aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l06);
    bf16 l07 = (bf16)getInvBf16(l[15] + sink_sub_float.get(15));
    aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l07);
    auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    for (int i = 0; i < F; i++) {
      aie::vector<float, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      aie::accum<accfloat, vec_factor> A0_acc;
      A0_acc.from_vector(A0);
      aie::accum<accfloat, vec_factor> AL =
          aie::mul(A0_acc.template to_vector<bf16>(), L);
      aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
      aie::store_v(pO1, AL_bf16);
      pO1 += vec_factor;
      pA1 += vec_factor;
    }
  }
}

} // namespace

extern "C" {

// FLM attn_qk's loop body for block blk, attending keys [L - window, L)
// (all L if window is 0); m_state/c_state persist across a query's blocks.
void gpt_attn_qk_blk(bf16 *__restrict q, bf16 *__restrict k_block,
                     float *__restrict m_state, float *__restrict c_state,
                     bf16 *__restrict s_block, int blk, int L, int window) {
  int lo = (window && L > window) ? L - window : 0;
  const aie::vector<int, 16> idx = aie::vector<int, 16>(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  if (blk == 0) {
    aie::accum<accfloat, 16> sink_acc;
    sink_acc.from_vector(aie::load_v<16>(q + 16 * DH));
    aie::store_v(m_state, sink_acc.to_vector<float>());
  }
  int i = L - blk * 16, j = lo - blk * 16;
  if (i <= 0 || j >= 16) {
    // no key attended: zero scores and no rescale leave the kv state as is
    for (int k = 0; k < 16 * 16; k += 32)
      aie::store_v(s_block + k, aie::zeros<bf16, 32>());
    aie::store_v(c_state, aie::broadcast<float, 16>(1.0f));
  } else {
    aie::mask<16> mask =
        aie::le(idx, (i < 16) ? i : 16) & aie::gt(idx, (j > 0) ? j : 0);
    _attn_qk<DH / 8, 8, 8, 8>(q, k_block, s_block, m_state, c_state, mask);
  }
  float *c = (float *)(s_block + 16 * 16);
  float *m = (float *)(s_block + 16 * 16 + 32);
  aie::store_v(c, aie::load_v<16>(c_state));
  aie::store_v(m, aie::load_v<16>(m_state));
}

// FLM attn_kv's loop body; y_state/l_state/m_last persist across the blocks.
void gpt_attn_kv_blk(bf16 *__restrict s_block, bf16 *__restrict v_block,
                     float *__restrict y_state, float *__restrict l_state,
                     float *__restrict m_last, int blk) {
  if (blk == 0) {
    zero_vectorized<float, 16 * DH>(y_state);
    aie::store_v(l_state, aie::broadcast<float, 16>(0));
  }
  float *c = (float *)(s_block + 16 * 16);
  float *m = (float *)(s_block + 16 * 16 + 32);
  reorder_s(s_block);
  calculate_l(s_block, c, l_state);
  attn_fv<DH / 8, 8, 8, 8>(s_block, v_block, y_state, c);
  aie::store_v(m_last, aie::load_v<16>(m));
}

// o[16 x 64] = y / (l + exp(sink - m)); sink: the 16 heads' sinks
void gpt_attn_kv_fin(float *__restrict y_state, float *__restrict l_state,
                     bf16 *__restrict sink, float *__restrict m_last,
                     bf16 *__restrict o) {
  scale_div_aie<DH>(o, y_state, l_state, sink, m_last);
}

} // extern "C"
