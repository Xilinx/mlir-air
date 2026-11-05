// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GPT-OSS prefill attention with sinks, transcribed from FastFlowLM
// FLM_Xclbin/GPTOSS/gpt_prefill_attn_v1_npu_bin/kernel/{attn.cc,attn_kernel.h}.
// A core owns 16 query rows of one head; keys come in 128-key blocks of eight
// 16-key steps, K steps then V steps. The head's sink seeds the running max,
// and exp(sink - m) joins the denominator at the end. FLM's attn() owns the
// block loop and the locks; here AIR calls one step at a time.
//
// q   [2][8][8][8]   16 rows x 64 in 8x8 blocks, then the sink (f32) at 1024
// k/v [8][2][8][8]   16 keys x 64: [dh block][key block][key][dh]
// s   [8][16][16]    one block's scores
// mm                 m [16][16], previous m [16], new m [16]
// cl                 c [16], l [16]
// y   [2][8][8][8]   f32; o as y, bf16

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

typedef bfloat16 bf16;

namespace {

constexpr int lq = 16, lk = 16, dh = 64;
constexpr float scale = 0.125f;
constexpr bf16 exp_scale = (bf16)1.4426950408889634f;
constexpr int PREV = lq * lk, NEW = PREV + lq;

alignas(aie::vector_decl_align) int idx_array[16] = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

inline aie::accum<accfloat, 16>
mac_elem_16_accuracy_safe(aie::vector<float, 16> a, aie::vector<float, 16> b,
                          aie::accum<accfloat, 16> acc) {
  return aie::add(acc, aie::mul(a, b));
}

void apply_softmax(bf16 *__restrict pS, bf16 *__restrict new_m_local) {
  for (int b = 0; b < 8; b++) {
    for (int q = 0; q < lq; q++) {
      aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS);
      aie::vector<bf16, 16> Vec = aie::sub(s_vec, *(new_m_local + q));
      aie::accum<accfloat, 16> Vec_acc = aie::mul(Vec, exp_scale);
      Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>());
      aie::store_v(pS, Vec);
      pS += 16;
    }
  }
}

void apply_mask_and_get_max(bf16 *__restrict pS, bf16 *__restrict m,
                            int pointer_block_inner_k,
                            int pointer_block_inner_q,
                            int pointer_block_inner_k_current) {
  aie::vector<int, 16> idx_vec = aie::load_v<16>(idx_array);
  aie::vector<int, lk> l_vec = aie::broadcast<int, lk>(pointer_block_inner_k);
  l_vec = aie::add(l_vec, idx_vec);
  aie::vector<int, lk> r_vec = aie::broadcast<int, lk>(pointer_block_inner_q);
  r_vec = aie::add(r_vec, idx_vec);

  for (int i = 0; i < lq; i++) {
    int current_idx = pointer_block_inner_k_current + 15 - i;
    aie::mask<lk> mask_l = aie::gt(current_idx, l_vec);
    aie::mask<lk> mask_r = aie::le(current_idx, r_vec);
    aie::vector<bf16, lk> In_vec = aie::load_v<lk>(pS);
    aie::vector<bf16, lk> s_vec =
        aie::select(bf16(-0x1.FEp127f), In_vec, mask_r);
    s_vec = aie::select(bf16(-0x1.FEp127f), s_vec, mask_l);

    aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m);
    aie::mask<lk> mask_m = aie::gt(s_vec, m_vec);
    aie::vector<bf16, lk> m_vec_new = aie::select(m_vec, s_vec, mask_m);
    aie::store_v(pS, s_vec);
    aie::store_v(m, m_vec_new);
    pS += lk;
    m += lk;
  }
}

void calculate_c(float *c, bf16 *prev_m_local, bf16 *new_m_local) {
  for (int i = 0; i < lq; i++) {
    bf16 inner_correct = aie::sub(*(prev_m_local + i), *(new_m_local + i));
    inner_correct = aie::mul(inner_correct, exp_scale);
    aie::vector<bf16, lk> correct_vec = aie::broadcast<bf16, lk>(inner_correct);
    aie::accum<accfloat, lk> correct_acc;
    correct_acc.from_vector(correct_vec);
    correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());
    *(c + i) = (float)correct_vec.get(0);
  }
}

void reorder_s(bf16 *__restrict pS) {
  for (int b = 0; b < 8; b++) {
    bf16 *pS1 = pS + b * 256;
    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
    aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
    aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
    aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
    aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
    aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);
    aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
    aie::vector<bf16, 32> S20 = aie::filter_even(S2, 8);
    aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 8);
    aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);
    aie::vector<bf16, 32> S30 = aie::filter_even(S3, 8);
    aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 8);

    S0 = aie::concat(S00, S10);
    S1 = aie::concat(S01, S11);
    S2 = aie::concat(S20, S30);
    S3 = aie::concat(S21, S31);

    aie::store_v(pS1, S0);
    aie::store_v(pS1 + 64, S1);
    aie::store_v(pS1 + 128, S2);
    aie::store_v(pS1 + 192, S3);
  }
}

void calculate_l(float *l, float *__restrict c, bf16 *__restrict pS) {
  using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
  aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
  aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>();
  aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>();
  MMUL C0(acc_C0);
  MMUL C1(acc_C1);

  for (int b = 0; b < 8; b++) {
    bf16 *pS1 = pS + b * 256;
    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
    aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
    aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
    aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);
    C0.mac(S0, Ones);
    C1.mac(S2, Ones);
    C0.mac(S1, Ones);
    C1.mac(S3, Ones);
  }

  auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
  auto sum00 = aie::filter_even(sum0, 2);
  auto sum01 = aie::filter_even(sum00, 1);
  auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
  auto sum10 = aie::filter_even(sum1, 2);
  auto sum11 = aie::filter_even(sum10, 1);
  auto sum2 = aie::concat(sum01, sum11);
  aie::accum<accfloat, 16> sum;
  sum.from_vector(sum2);

  aie::vector<float, 16> c_float32 = aie::load_v<16>(c);
  aie::vector<float, 16> l_float32 = aie::load_v<16>(l);
  aie::accum<accfloat, 16> l_out =
      mac_elem_16_accuracy_safe(l_float32, c_float32, sum);
  aie::store_v(l, l_out.template to_vector<float>());
}

void calculate_y(float *y, float *c) {
  for (int i = 0; i < lq / 8; i++) {
    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[i * 8]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[i * 8 + 1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[i * 8 + 2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[i * 8 + 3]);
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[i * 8 + 4]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[i * 8 + 5]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[i * 8 + 6]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[i * 8 + 7]);
    aie::vector<float, 64> CORRECT =
        aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    float *pY = y + i * 8 * dh;
    for (unsigned j = 0; j < dh / 8; j += 1) {
      aie::vector<float, 64> Y = aie::load_v<64>(pY);
      aie::accum<accfloat, 64> ACC_Y = aie::mul(CORRECT, Y);
      aie::store_v(pY, ACC_Y.template to_vector<float>());
      pY += 64;
    }
  }
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_fv(T_out *__restrict pY, T_in *__restrict pS, T_in *__restrict pV) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  aie::vector<T_in, 64> S0 = aie::load_v<64>(pS);
  aie::vector<T_in, 64> S1 = aie::load_v<64>(pS + 64);
  aie::vector<T_in, 64> S2 = aie::load_v<64>(pS + 128);
  aie::vector<T_in, 64> S3 = aie::load_v<64>(pS + 192);

  T_out *__restrict pY1 = pY;
  T_out *__restrict pY2 = pY + colB * MMUL::size_C;

  for (unsigned j = 0; j < colB; j += 2) {
    T_in *__restrict pV1 = pV + j * MMUL::size_B * colA;
    T_in *__restrict pV2 = pV + (j + 1) * MMUL::size_B * colA;

    aie::vector<T_in, MMUL::size_B> V0 = aie::load_v<MMUL::size_B>(pV1);
    pV1 += MMUL::size_B;
    aie::vector<T_in, MMUL::size_B> V1 = aie::load_v<MMUL::size_B>(pV2);
    pV2 += MMUL::size_B;

    MMUL Y00(aie::load_v<MMUL::size_C>(pY1));
    MMUL Y01(aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C));
    MMUL Y10(aie::load_v<MMUL::size_C>(pY2));
    MMUL Y11(aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C));

    Y00.mac(S0, V0);
    Y01.mac(S0, V1);
    Y10.mac(S2, V0);
    Y11.mac(S2, V1);

    V0 = aie::load_v<MMUL::size_B>(pV1);
    V1 = aie::load_v<MMUL::size_B>(pV2);

    Y00.mac(S1, V0);
    Y01.mac(S1, V1);
    Y10.mac(S3, V0);
    Y11.mac(S3, V1);

    aie::store_v(pY1, Y00.template to_vector<T_out>());
    pY1 += MMUL::size_C;
    aie::store_v(pY1, Y01.template to_vector<T_out>());
    pY1 += MMUL::size_C;
    aie::store_v(pY2, Y10.template to_vector<T_out>());
    pY2 += MMUL::size_C;
    aie::store_v(pY2, Y11.template to_vector<T_out>());
    pY2 += MMUL::size_C;
  }
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void _attn_qk(T_out *__restrict pS, T_in *__restrict pQ, T_in *__restrict pK) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  for (unsigned z = 0; z < rowA; z += 2) {
    T_out *__restrict pS1 = pS + (z * colB + 0) * MMUL::size_C;
    T_out *__restrict pS2 = pS + ((z + 1) * colB + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2) {
      const T_in *__restrict pQ1 = pQ + (z * colA + 0) * MMUL::size_A;
      const T_in *__restrict pQ2 = pQ + ((z + 1) * colA + 0) * MMUL::size_A;
      const T_in *__restrict pK1 = pK + (0 * colB + j) * MMUL::size_B;
      const T_in *__restrict pK2 = pK + (0 * colB + (j + 1)) * MMUL::size_B;

      MMUL C00(aie::zeros<T_out, MMUL::size_C>());
      MMUL C01(aie::zeros<T_out, MMUL::size_C>());
      MMUL C10(aie::zeros<T_out, MMUL::size_C>());
      MMUL C11(aie::zeros<T_out, MMUL::size_C>());

      for (unsigned i = 0; i < colA; ++i) {
        aie::vector<T_in, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
        pQ1 += MMUL::size_A;
        aie::vector<T_in, MMUL::size_A> Q1 = aie::load_v<MMUL::size_A>(pQ2);
        pQ2 += MMUL::size_A;
        aie::vector<T_in, MMUL::size_B> K0 =
            aie::transpose(aie::load_v<MMUL::size_B>(pK1), 8, 8);
        pK1 += MMUL::size_B * colB;
        aie::vector<T_in, MMUL::size_B> K1 =
            aie::transpose(aie::load_v<MMUL::size_B>(pK2), 8, 8);
        pK2 += MMUL::size_B * colB;

        C00.mac(Q0, K0);
        C01.mac(Q0, K1);
        C10.mac(Q1, K0);
        C11.mac(Q1, K1);
      }
      auto mout0 = aie::interleave_zip(C00.template to_vector<T_out>(),
                                       C01.template to_vector<T_out>(), 8);
      auto mout1 = aie::interleave_zip(C10.template to_vector<T_out>(),
                                       C11.template to_vector<T_out>(), 8);

      auto mout2 =
          aie::mul(mout0.first, (T_out)scale).template to_vector<T_out>(0);
      auto mout3 =
          aie::mul(mout0.second, (T_out)scale).template to_vector<T_out>(0);
      auto mout4 =
          aie::mul(mout1.first, (T_out)scale).template to_vector<T_out>(0);
      auto mout5 =
          aie::mul(mout1.second, (T_out)scale).template to_vector<T_out>(0);

      aie::store_v(pS1, mout2);
      pS1 += MMUL::size_C;
      aie::store_v(pS1, mout3);
      pS1 += MMUL::size_C;
      aie::store_v(pS2, mout4);
      pS2 += MMUL::size_C;
      aie::store_v(pS2, mout5);
      pS2 += MMUL::size_C;
    }
  }
}

inline void scale16(bf16 *o, float *y, aie::vector<bf16, 16> l) {
  aie::accum<accfloat, 16> Y;
  Y.from_vector(aie::load_v<16>(y));
  aie::accum<accfloat, 16> AL = aie::mul(Y.template to_vector<bf16>(), l);
  aie::store_v(o, AL.template to_vector<bf16>());
}

void scale_div_aie(bf16 *o, bf16 *l, float *y) {
  constexpr int vec_factor = 64;
  aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l[0]);
  aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l[1]);
  aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l[2]);
  aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l[3]);
  aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l[4]);
  aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l[5]);
  aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l[6]);
  aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l[7]);
  auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

  // 16 lanes at a time, with constant halves: a wider accfloat -> bf16
  // conversion and a loop-indexed extract both pick the wrong lanes
  scale16(o, y, LL00.template extract<16>(0));
  scale16(o + 16, y + 16, LL00.template extract<16>(1));
  scale16(o + 32, y + 32, LL00.template extract<16>(2));
  scale16(o + 48, y + 48, LL00.template extract<16>(3));
}

void offset_l(bf16 *l_out, float *l, bf16 *m, bf16 sink_value) {
  aie::vector<bf16, lq> m_vec = aie::load_v<lq>(m);
  aie::vector<bf16, lq> sink_vec = aie::broadcast<bf16, lq>(sink_value);
  aie::vector<bf16, lq> f_vec = aie::sub(sink_vec, m_vec);
  aie::accum<accfloat, lq> Vec_acc = aie::mul(f_vec, exp_scale);
  aie::vector<bf16, lq> Vec =
      aie::exp2<bf16>(Vec_acc.template to_vector<float>());
  aie::vector<float, lq> l_vec = aie::load_v<lq>(l);
  aie::accum<accfloat, lq> l_acc_vec;
  l_acc_vec.from_vector(l_vec);
  l_acc_vec = aie::add(l_acc_vec, Vec);
  aie::vector<bf16, lq> l_vec_out =
      aie::inv(l_acc_vec.template to_vector<bf16>());
  aie::store_v(l_out, l_vec_out);
}

inline bf16 sink_of(bf16 *q) { return (bf16)(*(float *)(q + lq * dh)); }

} // namespace

extern "C" {

// A head's first block: c = 1, l = 0, y = 0, running max = its sink.
void apf_init(bf16 *__restrict q, bf16 *__restrict mm, float *__restrict cl,
              float *__restrict y) {
  aie::store_v(cl, aie::broadcast<float, lq>(1.0f));
  aie::store_v(cl + lq, aie::zeros<float, lq>());
  for (int i = 0; i < lq * dh; i += 16)
    aie::store_v(y + i, aie::zeros<float, 16>());
  aie::store_v(mm + PREV, aie::broadcast<bf16, lq>(sink_of(q)));
}

// Step j of a block: scores of 16 keys, masked, into the running max. Query
// rows start at q0; the block's keys at k0 (step j's at k0 + 16 j); keys at or
// below q - window are masked.
void apf_qk(bf16 *__restrict q, bf16 *__restrict k, bf16 *__restrict s,
            bf16 *__restrict mm, int32_t j, int32_t q0, int32_t k0,
            int32_t window) {
  if (j == 0)
    for (int r = 0; r < lq; r++)
      aie::store_v(mm + r * lk, aie::broadcast<bf16, lk>(mm[PREV + r]));
  int inner_k = q0 - window > -16 ? q0 - window : -16;
  _attn_qk<bf16, bf16, lq / 8, dh / 8, lk / 8, 8, 8, 8>(s + j * lq * lk, q, k);
  apply_mask_and_get_max(s + j * lq * lk, mm, inner_k, q0, k0 + j * 16);
}

// After a block's K steps: the block's softmax and the rescale of l and y.
void apf_softmax(bf16 *__restrict s, bf16 *__restrict mm, float *__restrict cl,
                 float *__restrict y) {
  for (int r = 0; r < lq; r++) {
    aie::vector<bf16, lk> m_vec = aie::load_v<lk>(mm + r * lk);
    mm[NEW + r] = aie::reduce_max(m_vec);
  }
  apply_softmax(s, mm + NEW);
  calculate_c(cl, mm + PREV, mm + NEW);
  reorder_s(s);
  calculate_l(cl + lq, cl, s);
  calculate_y(y, cl);
  aie::store_v(mm + PREV, aie::load_v<lq>(mm + NEW));
}

// Step j of a block's V.
void apf_fv(bf16 *__restrict s, bf16 *__restrict v, float *__restrict y,
            int32_t j) {
  _attn_fv<bf16, float, lq / 8, lk / 8, dh / 8, 8, 8, 8>(y, s + j * lq * lk, v);
}

// o = y / (l + exp(sink - m)), in y's 8x8 blocks.
void apf_out(bf16 *__restrict q, bf16 *__restrict mm, float *__restrict cl,
             float *__restrict y, bf16 *__restrict o) {
  alignas(aie::vector_decl_align) bf16 l_bf16[lq];
  offset_l(l_bf16, cl + lq, mm + PREV, sink_of(q));
  for (int o_row = 0; o_row < lq / 8; o_row++)
    for (int o_col = 0; o_col < dh / 8; o_col++)
      scale_div_aie(o + (o_row * dh / 8 + o_col) * 64, l_bf16 + o_row * 8,
                    y + o_row * 8 * dh + o_col * 64);
}

} // extern "C"
