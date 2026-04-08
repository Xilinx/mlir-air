//===- rope.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef __AIENGINE__
#define __AIENGINE__ 2
#endif
#define NOCPP
#ifndef __AIEARCH__
#define __AIEARCH__ 20
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

// On AIE2P, use native-width vectors (n=16) to avoid sub-native-width
// accfloat codegen issues in the Peano backend. On AIE2, use n=8.
#if __AIEARCH__ >= 21
static constexpr unsigned SINCOS_VEC = 16;
#else
static constexpr unsigned SINCOS_VEC = 8;
#endif

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
    aie::accum<accfloat, n> zp_acc, one_acc;

    aie::vector<bfloat16, n> x, r, r2, out_abs;
    aie::vector<bfloat16, n> s0, s1, s2, s3, c0, c1, c2, c3;
    aie::vector<bfloat16, n> twobypi, oneby2, negoneby2, piby2_1, one, negone;
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

    // Range reduction: z = round(ux * 2/pi), x = ux - z * pi/2.
    // Uses scalar rounding and quadrant detection to avoid vector int16
    // operations (bit_and, to_fixed) that crash the Peano backend.
    zp_acc.from_vector(oneby2);
    ux_acc.from_vector(ux);
    zp_acc = mac(zp_acc, ux, twobypi);
    aie::vector<bfloat16, n> zp_bf16 = zp_acc.template to_vector<bfloat16>();
    alignas(aie::vector_decl_align) bfloat16 zp_buf[n];
    aie::store_v(zp_buf, zp_bf16);
    alignas(aie::vector_decl_align) bfloat16 z_buf[n];
    aie::mask<n> is_odd, is_neg;
    {
      uint32_t odd_bits = 0, neg_bits = 0;
      for (unsigned j = 0; j < n; j++) {
        int zj = (int)(float)zp_buf[j];
        z_buf[j] = (bfloat16)(float)zj;
        if (isSine) {
          if (zj & 1)
            odd_bits |= (1u << j);
          if (zj & 2)
            neg_bits |= (1u << j);
        } else {
          if (!(zj & 1))
            odd_bits |= (1u << j);
          if (((zj >> 1) & 1) ^ (zj & 1))
            neg_bits |= (1u << j);
        }
      }
      is_odd = aie::mask<n>::from_uint32(odd_bits);
      is_neg = aie::mask<n>::from_uint32(neg_bits);
    }
    aie::vector<bfloat16, n> z_rounded = aie::load_v<n>(z_buf);
    x_acc = msc(ux_acc, piby2_1, z_rounded);

    x = x_acc.template to_vector<bfloat16>();
    r = aie::mul(x, x).template to_vector<bfloat16>();

    aie::vector<bfloat16, n> t_vec =
        mac(one_acc, negoneby2, r).template to_vector<bfloat16>();

    POLY_EVAL_4_VECTOR(r, acs0, s1, acs2, s3, acc0, c1, acc2, c3, poly_acs,
                       poly_acc, r2);

    aie::accum<accfloat, n> x_acc_fresh;
    x_acc_fresh.from_vector(x);
    aie::vector<bfloat16, n> x3 = aie::mul(r, x).template to_vector<bfloat16>();
    out_acs = mac(x_acc_fresh, x3, poly_acs.template to_vector<bfloat16>());
    aie::accum<accfloat, n> t_acc_fresh;
    t_acc_fresh.from_vector(t_vec);
    out_acc = mac(t_acc_fresh, r2, poly_acc.template to_vector<bfloat16>());

    out_abs = select(out_acs.template to_vector<bfloat16>(),
                     out_acc.template to_vector<bfloat16>(), is_odd);
    output_acc = aie::mul(negone, out_abs);
    bfloat16 *__restrict pOut = outputs + i;
    aie::store_v(
        pOut,
        select(out_abs, output_acc.template to_vector<bfloat16>(), is_neg));
  }
#undef POLY_EVAL_4_VECTOR
}

// Frequency table padded to 32 entries for n=16 processing on AIE2P.
// The extra 8 entries are zero (sin(0)=0, cos(0)=1 — correct but unused).
alignas(aie::vector_decl_align) static const bfloat16 freq_table[32] = {
    0x1p0,
    0x1.5cd25p-1,
    0x1.db4c78p-2,
    0x1.43d136p-2,
    0x1.b93a6cp-3,
    0x1.2c9af4p-3,
    0x1.99999ap-4,
    0x1.170ea6p-4,
    0x1.7c3d2cp-5,
    0x1.030dc6p-5,
    0x1.60fb8ap-6,
    0x1.e0f7eep-7,
    0x1.47ae14p-7,
    0x1.be7dd8p-8,
    0x1.3030fp-8,
    0x1.9e7c72p-9,
    0x1.1a62d8p-9,
    0x1.80c652p-10,
    0x1.0624dep-10,
    0x1.653176p-11,
    0x1.e6b4bcp-12,
    0x1.4b96cep-12,
    0x1.c3d114p-13,
    0x1.33d1eap-13,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
};

template <unsigned N, unsigned n>
void freq_pos_bf16(int pos, bfloat16 *__restrict outputs) {
  const bfloat16 *freq_ptr = freq_table;
  aie::vector<bfloat16, n> vecPos =
      aie::broadcast<bfloat16, n>(aie::to_float<bfloat16>(pos));
  bfloat16 *__restrict pOut = outputs;
  for (unsigned i = 0; i < N; i += n) {
    aie::vector<bfloat16, n> v = aie::load_v<n>(freq_ptr);
    freq_ptr += n;
    aie::store_v(pOut, aie::mul(vecPos, v).template to_vector<bfloat16>());
    pOut += n;
  }
}

// Apply RoPE rotation using n-wide vectors. On AIE2P, n=SINCOS_VEC=16
// ensures all accfloat operations use native-width registers.
// N must be divisible by 2*n. For head_size=48, the caller pads to 64.
template <unsigned N, unsigned n>
void shuffle_apply_rope(int offset, const bfloat16 *__restrict fcr,
                        const bfloat16 *__restrict fci,
                        bfloat16 *__restrict outputs) {

  constexpr unsigned two_n = n * 2;

  const bfloat16 *__restrict pFcr = fcr;
  const bfloat16 *__restrict pFci = fci;
  for (unsigned i = 0; i < N; i += two_n) {
    aie::vector<bfloat16, two_n> v0 = aie::load_v<two_n>(outputs + i + offset);

    aie::vector<bfloat16, n> v0Even = aie::filter_even(v0, 1);
    aie::vector<bfloat16, n> v0Odd = aie::filter_odd(v0, 1);

    aie::vector<bfloat16, n> vFcr = aie::load_v<n>(pFcr);
    pFcr += n;
    aie::vector<bfloat16, n> vFci = aie::load_v<n>(pFci);
    pFci += n;
    // out_even = even*cos - odd*sin, out_odd = even*sin + odd*cos
    aie::vector<bfloat16, n> vOutEven =
        msc(aie::mul(v0Even, vFcr), v0Odd, vFci).template to_vector<bfloat16>();
    aie::vector<bfloat16, n> vOutOdd =
        mac(aie::mul(v0Even, vFci), v0Odd, vFcr).template to_vector<bfloat16>();
    auto [vOutLo, vOutHi] = aie::interleave_zip(vOutEven, vOutOdd, 1);

    bfloat16 *__restrict pOut = outputs + i + offset;
    aie::store_v(pOut, vOutLo);
    pOut += n;
    aie::store_v(pOut, vOutHi);
    pOut += n;
  }
}

// Wrapper that pads head_size to a multiple of 2*SINCOS_VEC using a local
// work buffer, so ALL operations use native-width vectors on AIE2P.
template <unsigned head_size>
void shuffle_apply_rope_padded(int offset, const bfloat16 *__restrict fcr,
                               const bfloat16 *__restrict fci,
                               bfloat16 *__restrict outputs) {
  static_assert(head_size % 16 == 0,
                "head_size must be a multiple of 16 for vectorized copy");
  constexpr unsigned n = SINCOS_VEC;
  constexpr unsigned two_n = n * 2;
  // Round up head_size to next multiple of two_n
  constexpr unsigned padded = ((head_size + two_n - 1) / two_n) * two_n;

  // Copy data to padded work buffer (head_size elements + zero padding)
  alignas(aie::vector_decl_align) bfloat16 work[padded];
  for (unsigned j = 0; j < head_size; j += 16)
    aie::store_v(work + j, aie::load_v<16>(outputs + j + offset));
  for (unsigned j = head_size; j < padded; j += 16)
    aie::store_v(work + j, aie::zeros<bfloat16, 16>());

  // Copy sin/cos buffers to padded local arrays.
  // Source buffers are 32 elements (already zero-padded by sinf/cosf).
  // We need padded/2 elements; pad any excess with zeros.
  alignas(aie::vector_decl_align) bfloat16 fcr_pad[padded / 2];
  alignas(aie::vector_decl_align) bfloat16 fci_pad[padded / 2];
  for (unsigned j = 0; j < padded / 2; j += 16) {
    aie::store_v(fcr_pad + j, j < 32 ? aie::load_v<16>(fcr + j)
                                     : aie::zeros<bfloat16, 16>());
    aie::store_v(fci_pad + j, j < 32 ? aie::load_v<16>(fci + j)
                                     : aie::zeros<bfloat16, 16>());
  }

  // Process with native-width vectors
  shuffle_apply_rope<padded, n>(0, fcr_pad, fci_pad, work);

  // Copy results back (only head_size elements)
  for (unsigned j = 0; j < head_size; j += 16)
    aie::store_v(outputs + j + offset, aie::load_v<16>(work + j));
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

// sinf/cosf: on AIE2P use n=16 with N=32 (padded), on AIE2 use n=8 with N=24.
// The extern "C" wrappers handle padding internally so callers always pass
// 32-element buffers (extra 8 elements are unused padding on AIE2).
void sinf_bf16(const bfloat16 *__restrict inputs,
               bfloat16 *__restrict outputs) {
  sinf_cosf_poly_bf16<32, SINCOS_VEC, true>(inputs, outputs);
}
void cosf_bf16(const bfloat16 *__restrict inputs,
               bfloat16 *__restrict outputs) {
  sinf_cosf_poly_bf16<32, SINCOS_VEC, false>(inputs, outputs);
}

void freq_pos_bf16(const int pos, bfloat16 *__restrict outputs) {
  freq_pos_bf16<32, SINCOS_VEC>(pos, outputs);
}

void shuffle_apply_rope_bf16_48(int offset, const bfloat16 *__restrict fcr,
                                const bfloat16 *__restrict fci,
                                bfloat16 *__restrict outputs) {
#if __AIEARCH__ >= 21
  // AIE2P: use padded work buffer to ensure all ops use native-width n=16
  shuffle_apply_rope_padded<48>(offset, fcr, fci, outputs);
#else
  shuffle_apply_rope<48, 8>(offset, fcr, fci, outputs);
#endif
}

void vector_copy_bf16_144_16(const bfloat16 *__restrict inputs,
                             bfloat16 *__restrict outputs) {
  vector_copy_bf16<144, 16>(inputs, outputs);
}

} // extern "C"
