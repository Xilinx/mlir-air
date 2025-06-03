//===- sine_cosine.cc -------------------------------------000---*- C++ -*-===//
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

    zeroint16 = aie::broadcast<int16, n>(0);
    oneint16 = aie::broadcast<int16, n>(1);
    twoint16 = aie::broadcast<int16, n>(2);
    threeint16 = aie::broadcast<int16, n>(3);

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

    // sin x
    x = x_acc.template to_vector<bfloat16>();
    r_acc = mul(x, x);
    r = r_acc.template to_vector<bfloat16>();

    // cos x
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

extern "C" {

void sinf_bf16_24_8(const bfloat16 *__restrict inputs,
                    bfloat16 *__restrict outputs) {
  sinf_cosf_poly_bf16<24, 8, true>(inputs, outputs);
}
void cosf_bf16_24_8(const bfloat16 *__restrict inputs,
                    bfloat16 *__restrict outputs) {
  sinf_cosf_poly_bf16<24, 8, false>(inputs, outputs);
}

} // extern "C"
