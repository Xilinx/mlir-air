//===- mha.cc ---------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "lut_based_ops.h"
#include "zero.cc"

template <typename T_in, typename T_out, unsigned k, unsigned n, unsigned s,
          unsigned t>
void vecmat_vectorized(int offset, T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(n % t == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 because that is the number of
                         // column vectors (a_vec_0_0..a_vec_3_1) we create
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value ||
                std::is_same<T_in, int16_t>::value);

  event0();
  T_in *__restrict a_ptr = a;
  T_in *__restrict b_ptr = b;

  T_out *__restrict c_ptr = c + offset; // reset to the first row of C output on
  // each outer loop tieration
  for (int col = 0; col < n; col += t) {

    const T_in *__restrict a_ptr1 = a_ptr;
    const T_in *__restrict b_ptr1 = b_ptr;
    for (int row = 0; row < k; row += 8)
      chess_loop_range(k / 8, ) {
        aie::vector<T_in, 8> a_vec = aie::load_v<8>(a_ptr1);
        a_ptr1 += 8;
        aie::accum<accfloat, t> c_acc_in;
        c_acc_in.from_vector(aie::load_v<t>(c_ptr));

        for (int i = 0; i < 8; i++) {
          const aie::vector<T_in, t> b_vec = aie::load_v<t>(b_ptr1);
          b_ptr1 += n;
          const aie::vector<T_in, t> s0 = aie::broadcast<T_in, t>(a_vec[i]);
          c_acc_in = mac(c_acc_in, s0, b_vec);
        }
        aie::store_v(c_ptr, c_acc_in.template to_vector<T_out>());
      }
    b_ptr += t;
    c_ptr += t;
  }
  event1();
}

template <unsigned N, unsigned n, bool isSine>
void sinf_cosf_poly_bf16(const bfloat16 *__restrict inputs,
                         bfloat16 *__restrict outputs) {
  constexpr float sin_poly_factors[4] = {
      -0x1.5555555555555p-3, 0x1.1111111110bb3p-7, -0x1.a01a019e83e5cp-13,
      0x1.71de3796cde01p-19};
  constexpr float cos_poly_factors[4] = {
      0x1.5555555555555p-5,   /* 0.0416667 */
      -0x1.6c16c16c16967p-10, /* -0.00138889 */
      0x1.A01A019F4EC91p-16,  /* 2.48016e-005 */
      -0x1.27E4FA17F667Bp-22  /* -2.75573e-007 */
  };
  const float twobypival = 0x1.45f306dc9c883p-1;
  const float piby2_1val = 0x1.921fb54400000p0;
  for (unsigned i = 0; i < N; i += n) {
    aie::vector<bfloat16, n> ux = aie::load_v<n>(inputs + i);
    aie::accum<accfloat, n> ux_acc, x_acc, r_acc, x3_acc, t_acc, output_acc;
    aie::accum<accfloat, n> poly_acs, acs0, acs2, out_acs;
    aie::accum<accfloat, n> poly_acc, acc0, acc2, out_acc;
    aie::accum<accfloat, n> zp_acc, z_acc, one_acc;

    aie::vector<bfloat16, n> x, r, r2, out_abs;
    aie::vector<bfloat16, n> s0, s1, s2, s3, c0, c1, c2, c3;
    aie::vector<bfloat16, n> twobypi, oneby2, negoneby2, piby2_1, one, zero_f,
        negone;
    aie::vector<int16_t, n> z, zeroint16, oneint16, twoint16, threeint16;

    // preparing vector & accum
    s0 = aie::broadcast<bfloat16, n>(sin_poly_factors[0]);
    s1 = aie::broadcast<bfloat16, n>(sin_poly_factors[1]);
    s2 = aie::broadcast<bfloat16, n>(sin_poly_factors[2]);
    s3 = aie::broadcast<bfloat16, n>(sin_poly_factors[3]);
    c0 = aie::broadcast<bfloat16, n>(cos_poly_factors[0]);
    c1 = aie::broadcast<bfloat16, n>(cos_poly_factors[1]);
    c2 = aie::broadcast<bfloat16, n>(cos_poly_factors[2]);
    c3 = aie::broadcast<bfloat16, n>(cos_poly_factors[3]);
    twobypi = aie::broadcast<bfloat16, n>(twobypival);
    piby2_1 = aie::broadcast<bfloat16, n>(piby2_1val);
    oneby2 = aie::broadcast<bfloat16, n>(0.5);
    negoneby2 = aie::broadcast<bfloat16, n>(-0.5);
    one = aie::broadcast<bfloat16, n>(1.0);
    negone = aie::broadcast<bfloat16, n>(-1.0);
    // preparing vector for z
    zeroint16 = aie::broadcast<int16, n>(0);
    oneint16 = aie::broadcast<int16, n>(1);
    twoint16 = aie::broadcast<int16, n>(2);
    threeint16 = aie::broadcast<int16, n>(3);
    // acc
    acs0.from_vector(s0);
    acs2.from_vector(s2);
    acc0.from_vector(c0);
    acc2.from_vector(c2);
    one_acc.from_vector(one);

#define POLY_EVAL_4_VECTOR(r, s0, s1, s2, s3, c0, c1, c2, c3, outs, outc, r2)  \
  {                                                                            \
    aie::accum<accfloat, n> t1s, t2s, t1c, t2c, r2_acc;                        \
    t1s = mac(s0, s1, r);                                                      \
    t2s = mac(s2, s3, r);                                                      \
    t1c = mac(c0, c1, r);                                                      \
    t2c = mac(c2, c3, r);                                                      \
    r2_acc = mul(r, r);                                                        \
    r2 = r2_acc.template to_vector<bfloat16>();                                \
    outs = mac(t1s, r2, t2s.template to_vector<bfloat16>());                   \
    outc = mac(t1c, r2, t2c.template to_vector<bfloat16>());                   \
  }

    zp_acc.from_vector(oneby2);
    ux_acc.from_vector(ux);
    zp_acc = mac(zp_acc, ux, twobypi);
    z = to_fixed<int16_t>(zp_acc.template to_vector<bfloat16>(), 0);
    x_acc = msc(ux_acc, piby2_1, to_float<bfloat16>(z));

    aie::mask<n> is_odd, is_neg;
    if (isSine) {
      is_odd = eq(bit_and(z, oneint16), oneint16); // select sin or cos
      is_neg =
          eq(bit_and(z, twoint16), twoint16); // select positive or negetive
    } else {
      is_odd = eq(bit_and(z, oneint16), zeroint16); // select sin or cos
      is_neg =
          eq(bit_xor(downshift(bit_and(z, twoint16), 1), bit_and(z, oneint16)),
             oneint16); // select positive or negetive
    }

    x = x_acc.template to_vector<bfloat16>();
    r_acc = mul(x, x);
    r = r_acc.template to_vector<bfloat16>();

    t_acc = mac(one_acc, negoneby2, r);

    POLY_EVAL_4_VECTOR(r, acs0, s1, acs2, s3, acc0, c1, acc2, c3, poly_acs,
                       poly_acc, r2);

    x3_acc = mul(r, x);
    out_acs = mac(x_acc, x3_acc.template to_vector<bfloat16>(),
                  poly_acs.template to_vector<bfloat16>());
    out_acc = mac(t_acc, r2, poly_acc.template to_vector<bfloat16>());

    out_abs = select(out_acs.template to_vector<bfloat16>(),
                     out_acc.template to_vector<bfloat16>(), is_odd);
    output_acc = aie::mul(negone, out_abs);
    bfloat16 *__restrict pOut = outputs + i;
    aie::store_v(
        pOut,
        select(out_abs, output_acc.template to_vector<bfloat16>(), is_neg));
  }
}

template <unsigned n>
void freq_pos_bf16_24(const int pos, bfloat16 *__restrict outputs) {

  alignas(aie::vector_decl_align) const bfloat16 freq[] = {
      0x1p0,          0x1.5cd25p-1,   0x1.db4c78p-2,  0x1.43d136p-2,
      0x1.b93a6cp-3,  0x1.2c9af4p-3,  0x1.99999ap-4,  0x1.170ea6p-4,
      0x1.7c3d2cp-5,  0x1.030dc6p-5,  0x1.60fb8ap-6,  0x1.e0f7eep-7,
      0x1.47ae14p-7,  0x1.be7dd8p-8,  0x1.3030fp-8,   0x1.9e7c72p-9,
      0x1.1a62d8p-9,  0x1.80c652p-10, 0x1.0624dep-10, 0x1.653176p-11,
      0x1.e6b4bcp-12, 0x1.4b96cep-12, 0x1.c3d114p-13, 0x1.33d1eap-13,
  };

  const bfloat16 *freq_ptr = freq;
  aie::vector<bfloat16, n> vec0 = aie::load_v<n>(freq_ptr);
  freq_ptr += n;
  aie::vector<bfloat16, n> vec1 = aie::load_v<n>(freq_ptr);
  freq_ptr += n;
  aie::vector<bfloat16, n> vec2 = aie::load_v<n>(freq_ptr);
  freq_ptr += n;
  aie::vector<bfloat16, n> vecPos =
      aie::broadcast<bfloat16, n>(aie::to_float<bfloat16>(pos));

  bfloat16 *__restrict pOut = outputs;
  aie::store_v(pOut, aie::mul(vecPos, vec0).template to_vector<bfloat16>());
  pOut += n;
  aie::store_v(pOut, aie::mul(vecPos, vec1).template to_vector<bfloat16>());
  pOut += n;
  aie::store_v(pOut, aie::mul(vecPos, vec2).template to_vector<bfloat16>());
  pOut += n;
}

template <unsigned N>
void shuffle_apply_rope_bf16_8(int offset, const bfloat16 *__restrict fcr,
                               const bfloat16 *__restrict fci,
                               bfloat16 *__restrict outputs) {

  constexpr unsigned n = 8;
  constexpr unsigned two_n = n * 2;

  const bfloat16 *__restrict pFcr = fcr;
  const bfloat16 *__restrict pFci = fci;
  for (unsigned i = 0; i < N; i += two_n) {
    const bfloat16 *__restrict pA1 = outputs + i + offset;
    aie::vector<bfloat16, n> v0Lo = aie::load_v<n>(pA1);
    pA1 += n;
    aie::vector<bfloat16, n> v0Hi = aie::load_v<n>(pA1);
    pA1 += n;
    aie::vector<bfloat16, two_n> v0 = aie::concat(v0Lo, v0Hi);

    aie::vector<bfloat16, n> zerosV8 = aie::zeros<bfloat16, n>();
    aie::vector<bfloat16, two_n> zerosV16 = aie::zeros<bfloat16, two_n>();
    aie::vector<bfloat16, two_n> v0Shuffled = extract_v16bfloat16(
        ::shuffle(aie::concat(zerosV16, v0), ::shuffle_T16_8x2), 1);
    aie::vector<bfloat16, n> v0ShuffledEven = extract_v8bfloat16(v0Shuffled, 0);
    aie::vector<bfloat16, n> v0ShuffledOdd = extract_v8bfloat16(v0Shuffled, 1);

    aie::vector<bfloat16, n> vFcr = aie::load_v<n>(pFcr);
    pFcr += n;
    aie::vector<bfloat16, n> vFci = aie::load_v<n>(pFci);
    pFci += n;
    aie::vector<bfloat16, n> vOutEven =
        aie::sub(aie::mul(v0ShuffledEven, vFcr), aie::mul(v0ShuffledOdd, vFci));
    aie::vector<bfloat16, n> vOutOdd =
        aie::add(aie::mul(v0ShuffledEven, vFci), aie::mul(v0ShuffledOdd, vFcr));
    aie::vector<bfloat16, two_n> vOutUnshuffled = extract_v16bfloat16(
        ::shuffle(aie::concat(zerosV8, zerosV8, vOutEven, vOutOdd),
                  ::shuffle_T16_2x8),
        1);
    aie::vector<bfloat16, n> vOutLo = extract_v8bfloat16(vOutUnshuffled, 0);
    aie::vector<bfloat16, n> vOutHi = extract_v8bfloat16(vOutUnshuffled, 1);

    bfloat16 *__restrict pOut = outputs + i + offset;
    aie::store_v(pOut, vOutLo);
    pOut += n;
    aie::store_v(pOut, vOutHi);
    pOut += n;
  }
}

template <unsigned VecLen, typename VecIteratorIn, typename VecIteratorOut>
float exp_bf16(int num_elems, VecIteratorIn &in, VecIteratorOut &out) {
  bfloat16 __aie_dm_resource_a *ilut_ab =
      (bfloat16 __aie_dm_resource_a *)softmax_ilut_ab;
  bfloat16 __aie_dm_resource_b *ilut_cd =
      (bfloat16 __aie_dm_resource_b *)softmax_ilut_cd;
  bfloat16 __aie_dm_resource_a *flut_ab =
      (bfloat16 __aie_dm_resource_a *)softmax_flut_ab;
  bfloat16 __aie_dm_resource_b *flut_cd =
      (bfloat16 __aie_dm_resource_b *)softmax_flut_cd;
  using lut_type = aie::lut<4, bfloat16, bfloat16>;
  const int LUT_elems = 256;
  const int step_i = 8;
  const int step_f = 0;

  constexpr int SM_SCALE_FAC =
      8; // Use 8-bit fractional part for LUTs when converting from bfloat16 to
         // int, adjust any input scale factor using this.

  const int elem_iters =
      num_elems / VecLen +
      (num_elems % VecLen != 0); // number of iterations need to be performed
  aie::vector<bfloat16, VecLen> I_val_vec, F_val_vec, res0, input_bf16;
  aie::accum<accfloat, VecLen> exp_val_accum;
  aie::accum<accfloat, VecLen> exp_val_accum_shift;
  exp_val_accum = aie::zeros<accfloat, VecLen>();
  // Maximum value computation
  bfloat16 max_value;
  aie::vector<bfloat16, VecLen> max_bfloat16;
  aie::accum<accfloat, VecLen> acc0, acc1, acc_res;
  aie::vector<int16, VecLen> input;
  aie::vector<int16, 2 * VecLen> input0;

  lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
  lut_type lut_f(LUT_elems, flut_ab, flut_cd);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_i(lut_i, step_i);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_f(lut_f, step_f);
  aie::accum<accfloat, VecLen> exp_val;

  auto input_max = in;
  uint16 neg_infinity = (uint16)0xff80;
  bfloat16 *bf_neg_infinity = (bfloat16 *)&neg_infinity;
  aie::vector<bfloat16, VecLen> max_vec =
      aie::broadcast<bfloat16, VecLen>((*bf_neg_infinity));
  aie::vector<bfloat16, VecLen> temp;
  for (int i = 0; i < elem_iters; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      temp = aie::load_v<VecLen>(input_max);
      max_vec = aie::max(max_vec, temp);
    }
  max_value = aie::reduce_max(max_vec);
  max_bfloat16 = aie::broadcast<bfloat16, VecLen>(max_value);

  for (int i = 0; i < elem_iters; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      aie::vector<bfloat16, VecLen> input_org = aie::load_v<VecLen>(in);
      in += VecLen;
      acc0.from_vector(input_org, 0);
      acc1.from_vector(max_bfloat16, 0);
      acc_res = sub(acc0, acc1);
      input_bf16 = to_v16bfloat16(acc_res);
      input0 = v32int16(bfloat16_to_int(input_bf16, SM_SCALE_FAC));
#ifndef SM_USE_MSB
      input = filter_even(input0);
#else
      input = filter_odd(input0);
#endif

      I_val_vec = lookup_i.fetch(input.template cast_to<uint16>());
      F_val_vec = lookup_f.fetch(input.template cast_to<uint16>());
      exp_val = aie::mul(I_val_vec, F_val_vec);
      exp_val_accum = add(exp_val_accum, exp_val);
      aie::store_v(out, exp_val.template to_vector<bfloat16>());
      out += VecLen;
    }
  // Variant not using emulated FP32 for the mul reduce, off by +/- 1 in final
  // result and 10 cycles slower
  aie::vector<float, VecLen> reduce = exp_val_accum.template to_vector<float>();
  float res = aie::reduce_add(reduce);
  return res;
}

bfloat16 __attribute__((always_inline)) compute_inv_as_bf16(float x) {
  unsigned int *B_x;
  unsigned int exp_mask = 0x7F800000;
  unsigned int mantissa_mask = 0x007FFFFF;
  unsigned int mantissa_Q = 0x00008000;
  unsigned char exponent, mantissa;
  unsigned inv_exponent;
  unsigned short inv_x_val;
  unsigned int B_Q;
  bfloat16 *inv_x;
  B_x = (unsigned int *)&x;
  B_Q = *B_x + mantissa_Q;
  exponent = (B_Q & exp_mask) >> 23;
  mantissa = (B_Q & mantissa_mask) >> 16;
  inv_exponent = (mantissa == 0) + (253 - exponent);
  inv_x_val = (inv_exponent << 7) + m_inv_lut[mantissa];
  inv_x = (bfloat16 *)&inv_x_val;
  return *inv_x;
}

extern "C" {

void sinf_bf16_24_8(const bfloat16 *__restrict inputs,
                    bfloat16 *__restrict outputs) {
  sinf_cosf_poly_bf16<24, 8, true>(inputs, outputs);
}
void cosf_bf16_24_8(const bfloat16 *__restrict inputs,
                    bfloat16 *__restrict outputs) {
  sinf_cosf_poly_bf16<24, 8, false>(inputs, outputs);
}

void freq_pos_bf16_24_8(const int pos, bfloat16 *__restrict outputs) {
  // Head size 48 (24 even + 24 odd indexed elements), vector size 8.
  freq_pos_bf16_24<8>(pos, outputs);
}

void shuffle_apply_rope_bf16_48(int offset, const bfloat16 *__restrict fcr,
                                const bfloat16 *__restrict fci,
                                bfloat16 *__restrict outputs) {
  shuffle_apply_rope_bf16_8<48>(offset, fcr, fci, outputs);
}

void roundI32ToBf16(const int *__restrict inputs, int offset,
                    bfloat16 *__restrict outputs) {
  const int *__restrict pIn = inputs;
  bfloat16 *__restrict pOut = outputs;
  pOut += offset;
  for (unsigned i = 0; i < 3; i++) {
    aie::store_v(pOut, to_float<bfloat16>(aie::load_v<16>(pIn)));
    pIn += 16;
    pOut += 16;
  }
}

void attn_1(const bfloat16 *__restrict q, const bfloat16 *__restrict k,
            const int pos, bfloat16 *__restrict outputs) {
  const bfloat16 *__restrict pQ = q;
  const bfloat16 *__restrict pK = k;
  float result = 0.0;
  for (unsigned i = 0; i < 48 / 16; i++) {
    aie::vector<bfloat16, 16> qvec = aie::load_v<16>(pQ);
    pQ += 16;
    aie::vector<bfloat16, 16> kvec = aie::load_v<16>(pK);
    pK += 16;
    aie::accum<accfloat, 16> output_acc = aie::mul(qvec, kvec);
    float res = aie::reduce_add(output_acc.to_vector<float>());
    result += res;
  }
  result *= 0x1.279a74p-3;
  outputs[pos] = (bfloat16)result;
}

void softmax_bf16(const bfloat16 *__restrict in, const int num_elems,
                  bfloat16 *__restrict out) {
  const bfloat16 *__restrict pIn = in;
  bfloat16 *__restrict pExp =
      out; // Reusing output buffer to buffer intermediate exp results.
  bfloat16 *__restrict pOut = out;
  zero_vectorized<bfloat16, 1, 256, 16>(0, out);
  float accum_exp_val = exp_bf16<16>(num_elems, pIn, pExp);
  bfloat16 accum_inv = compute_inv_as_bf16(accum_exp_val);
  for (unsigned i = 0; i < num_elems / 16 + (num_elems % 16 != 0); i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      aie::vector<bfloat16, 16> in_elems = aie::load_v<16>(pOut);
      aie::accum<accfloat, 16> out_vals = aie::mul(in_elems, accum_inv);
      aie::store_v(pOut, out_vals.template to_vector<bfloat16>());
      pOut += 16;
    }
}

void attn_2(const bfloat16 *__restrict attn, const bfloat16 *__restrict v,
            const int pos, bfloat16 *__restrict outputs) {
  const bfloat16 *__restrict pV = v;
  bfloat16 *__restrict pOut = outputs;
  aie::vector<bfloat16, 16> attnBcast = aie::broadcast<bfloat16, 16>(attn[pos]);
  for (unsigned j = 0; j < 48 / 16; j++) {
    aie::accum<accfloat, 16> acc;
    acc.from_vector(aie::load_v<16>(pOut));
    acc = mac(acc, attnBcast, aie::load_v<16>(pV));
    pV += 16;
    aie::store_v(pOut, acc.template to_vector<bfloat16>());
    pOut += 16;
  }
}

void vector_copy(const int offset, const bfloat16 *__restrict inputs,
                 bfloat16 *__restrict outputs) {
  const bfloat16 *__restrict pIn = inputs;
  bfloat16 *__restrict pOut = outputs + offset;
  for (unsigned j = 0; j < 48 / 16; j++) {
    aie::vector<bfloat16, 16> vec = aie::load_v<16>(pIn);
    pIn += 16;
    aie::store_v(pOut, vec);
    pOut += 16;
  }
}

#ifndef DIM_N
#define DIM_N 48
#endif

#ifndef DIM_K
#define DIM_K 96
#endif

#define combos(X) X(bfloat16, bf16, bfloat16, bf16)

#define vecmat_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out)                                \
  void vecmat_##mlir_type_in##_##mlir_type_out(                                \
      int offset, ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {          \
    vecmat_vectorized<ctype_in, ctype_out, DIM_K, DIM_N, 8, 16>(offset, a_in,  \
                                                                b_in, c_out);  \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out)                                  \
  void linalg_fill_##mlir_type_out(int offset, ctype_out *c_out) {             \
    zero_vectorized<ctype_out, 1, DIM_N, 32>(offset, c_out);                   \
  }

combos(vecmat_vectorized_c_func) combos(zero_vectorized_c_func)

} // extern "C"
