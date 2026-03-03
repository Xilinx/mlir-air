//===- rope.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
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
void freq_pos_bf16_24(int pos, bfloat16 *__restrict outputs) {

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

template <unsigned N, unsigned v>
void vector_copy_bf16(const bfloat16 *__restrict inputs,
                      bfloat16 *__restrict outputs) {
  const bfloat16 *__restrict pIn = inputs;
  bfloat16 *__restrict pOut = outputs;
  for (unsigned j = 0; j < N / v; j++) {
    aie::vector<bfloat16, v> vec = aie::load_v<v>(pIn);
    pIn += v;
    aie::store_v(pOut, vec);
    pOut += v;
  }
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

void vector_copy_bf16_192_16(const bfloat16 *__restrict inputs,
                             bfloat16 *__restrict outputs) {
  vector_copy_bf16<192, 16>(inputs, outputs);
}

} // extern "C"
