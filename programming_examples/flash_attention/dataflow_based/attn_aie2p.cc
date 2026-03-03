//===- attn_aie2p.cc - AIE2P flash attention kernels ------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
//
// AIE2P version: uses 8x8x8 matmul and exp2-based exponential.
// Helper functions use row-major data layout (matching dataflow_based
// channel tiling).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

#ifndef lqp
#define lqp 32
#endif
#ifndef lkp
#define lkp 96
#endif
#ifndef dk
#define dk 64
#endif
#ifndef dv
#define dv 64
#endif

// ============================================================
// Matmul: 2x2 mmul expansion for 8x8x8 AIE2P mmul
// ============================================================
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2) {
        const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
        const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
        const T_in *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;

        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);

        for (unsigned i = 0; i < colA; ++i) {
          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);
        }

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C * rowA;
      }
    }

  event1();
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t>(pA, pB, pC);
}

// ============================================================
// Exp: exp2-based, matching IRON's approach (no LUT needed)
// e^x = 2^(x * log2(e))
// ============================================================

__attribute__((always_inline)) aie::vector<bfloat16, 8>
getExpBf16_v8(aie::vector<bfloat16, 8> x) {
  constexpr float log2e_val = 1.44269504089f;
  aie::vector<bfloat16, 8> log2e_vec =
      aie::broadcast<bfloat16, 8>((bfloat16)log2e_val);
  aie::accum<accfloat, 8> exp_in = aie::mul(x, log2e_vec);
  return aie::exp2<bfloat16>(exp_in.to_vector<float>());
}

// ============================================================
// Extern C kernel functions — row-major data layout
// ============================================================
extern "C" {

void matmul_a_b_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *out) {
  matmul_vectorized_8x8x8_bf16_bf16<lqp, dk, lkp>(a_in, b_in, out);
}

void matmul_g_b_bf16(bfloat16 *g_in, bfloat16 *b_in, bfloat16 *out) {
  matmul_vectorized_8x8x8_bf16_bf16<lqp, lkp, dv>(g_in, b_in, out);
}

void zero_fill_gp_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, lqp, dv, 16>(c_out);
}

void zero_fill_sp_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, lqp, 1, 16>(c_out);
}

void zero_fill_g_bf16(bfloat16 *c_out) {
  zero_vectorized<bfloat16, lqp, lkp, 16>(c_out);
}

void neg_inf_fill_up_bf16(bfloat16 *c_out) {
  neg_inf_vectorized<bfloat16, lqp, 1, 16>(c_out);
}

void max_g_bf16(bfloat16 *in, bfloat16 *out) {
  constexpr int VecLen = 8;
  constexpr int num_elems = lkp;
  constexpr int num_rows = lqp;
  bfloat16 *__restrict pOut = out;
  uint16_t neg_infinity = (uint16_t)0xff80;
  bfloat16 *bf_neg_infinity = (bfloat16 *)&neg_infinity;
  aie::vector<bfloat16, VecLen> outputVec;
  for (int rowVec = 0; rowVec < num_rows; rowVec += VecLen) {
    for (int rowVecElem = 0; rowVecElem < VecLen; rowVecElem++) {
      aie::vector<bfloat16, VecLen> max_vec =
          aie::broadcast<bfloat16, VecLen>((*bf_neg_infinity));
      aie::vector<bfloat16, VecLen> temp;
      for (int32_t i = 0; i < num_elems; i += VecLen)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          temp =
              aie::load_v<VecLen>(in + i + (rowVec + rowVecElem) * num_elems);
          max_vec = aie::max(max_vec, temp);
        }
      bfloat16 max_value = aie::reduce_max(max_vec);
      outputVec[rowVecElem] = max_value;
    }
    aie::store_v(pOut, outputVec);
    pOut += VecLen;
  }
}

void maximum_up_u_bf16(bfloat16 *up, bfloat16 *u) {
  constexpr int VecLen = 8;
  constexpr int num_elems = lqp;
  bfloat16 *__restrict pu = u;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> up_temp = aie::load_v<VecLen>(up + i);
    aie::vector<bfloat16, VecLen> u_temp = aie::load_v<VecLen>(pu);
    u_temp = aie::max(up_temp, u_temp);
    aie::store_v(pu, u_temp);
    pu += VecLen;
  }
}

void exp_g_minus_u(bfloat16 *u, bfloat16 *g) {
  constexpr int VecLen = 8;
  constexpr int num_elems = lkp;
  constexpr int num_rows = lqp;
  bfloat16 *__restrict pG = g;
  // G <- G - u, then exp(G)
  for (int row = 0; row < num_rows; row++) {
    aie::vector<bfloat16, VecLen> u_bcast =
        aie::broadcast<bfloat16, VecLen>(*(u + row));
    for (int32_t i = 0; i < num_elems; i += VecLen) {
      aie::vector<bfloat16, VecLen> temp = aie::load_v<VecLen>(pG);
      temp = aie::sub(temp, u_bcast);
      temp = getExpBf16_v8(temp);
      aie::store_v(pG, temp);
      pG += VecLen;
    }
  }
}

void exp_up_minus_u(bfloat16 *up, bfloat16 *u, bfloat16 *r) {
  constexpr int VecLen = 8;
  constexpr int num_elems = lqp;
  bfloat16 *__restrict pr = r;
  bfloat16 *__restrict pu = u;
  bfloat16 *__restrict pup = up;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> uTemp = aie::load_v<VecLen>(pu);
    aie::vector<bfloat16, VecLen> upTemp = aie::load_v<VecLen>(pup);
    aie::vector<bfloat16, VecLen> rTemp = aie::sub(upTemp, uTemp);
    rTemp = getExpBf16_v8(rTemp);
    aie::store_v(pr, rTemp);
    pr += VecLen;
    pu += VecLen;
    pup += VecLen;
  }
}

void mul_r_gp(bfloat16 *r, bfloat16 *gp) {
  constexpr int VecLen = 8;
  constexpr int num_elems = dv;
  constexpr int num_rows = lqp;
  bfloat16 *__restrict pGp = gp;
  for (int row = 0; row < num_rows; row++) {
    aie::vector<bfloat16, VecLen> r_bcast =
        aie::broadcast<bfloat16, VecLen>(*(r + row));
    for (int32_t i = 0; i < num_elems; i += VecLen) {
      aie::vector<bfloat16, VecLen> temp = aie::load_v<VecLen>(pGp);
      temp = aie::mul(temp, r_bcast);
      aie::store_v(pGp, temp);
      pGp += VecLen;
    }
  }
}

void sum_g(bfloat16 *g, bfloat16 *s) {
  constexpr int VecLen = 8;
  constexpr int num_elems = lkp;
  constexpr int num_rows = lqp;
  bfloat16 *__restrict ps = s;
  for (int row = 0; row < num_rows; row++) {
    bfloat16 sum_value = 0.0;
    for (int32_t i = 0; i < num_elems; i += VecLen) {
      aie::vector<bfloat16, VecLen> temp =
          aie::load_v<VecLen>(g + row * num_elems + i);
      sum_value += aie::reduce_add(temp);
    }
    *(ps + row) = sum_value;
  }
}

void accum_sp_r_s(bfloat16 *sp, bfloat16 *r, bfloat16 *s) {
  constexpr int VecLen = 8;
  constexpr int num_elems = lqp;
  bfloat16 *__restrict pr = r;
  bfloat16 *__restrict ps = s;
  bfloat16 *__restrict psp = sp;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> rTemp = aie::load_v<VecLen>(pr);
    aie::vector<bfloat16, VecLen> spTemp = aie::load_v<VecLen>(psp);
    aie::accum<accfloat, VecLen> accTemp = aie::mul(rTemp, spTemp);
    accTemp = aie::add(accTemp, aie::load_v<VecLen>(ps));
    aie::store_v(ps, accTemp.to_vector<bfloat16>());
    pr += VecLen;
    ps += VecLen;
    psp += VecLen;
  }
}

void vector_copy_32elems(const int offset, const bfloat16 *__restrict inputs,
                         bfloat16 *__restrict outputs) {
  constexpr int VecLen = 8;
  constexpr int num_elems = lqp;
  const bfloat16 *__restrict pIn = inputs;
  bfloat16 *__restrict pOut = outputs + offset;
  for (unsigned j = 0; j < num_elems / VecLen; j++) {
    aie::vector<bfloat16, VecLen> vec = aie::load_v<VecLen>(pIn);
    pIn += VecLen;
    aie::store_v(pOut, vec);
    pOut += VecLen;
  }
}

void vector_copy_32x96elems(const int offset, const bfloat16 *__restrict inputs,
                            bfloat16 *__restrict outputs) {
  constexpr int VecLen = 32;
  constexpr int num_elems = lqp * lkp;
  const bfloat16 *__restrict pIn = inputs;
  bfloat16 *__restrict pOut = outputs + offset;
  for (unsigned j = 0; j < num_elems / VecLen; j++) {
    aie::vector<bfloat16, VecLen> vec = aie::load_v<VecLen>(pIn);
    pIn += VecLen;
    aie::store_v(pOut, vec);
    pOut += VecLen;
  }
}

void vector_accum_32x64elems(const bfloat16 *__restrict inputs,
                             bfloat16 *__restrict outputs) {
  constexpr int VecLen = 32;
  constexpr int num_elems = lqp * dv;
  const bfloat16 *__restrict pIn = inputs;
  bfloat16 *__restrict pOut = outputs;
  for (unsigned j = 0; j < num_elems / VecLen; j++) {
    aie::accum<accfloat, VecLen> accTemp;
    accTemp.from_vector(aie::load_v<VecLen>(pOut));
    accTemp = aie::add(accTemp, aie::load_v<VecLen>(pIn));
    pIn += VecLen;
    aie::store_v(pOut, accTemp.to_vector<bfloat16>());
    pOut += VecLen;
  }
}

void div_gp_sp(bfloat16 *sp, bfloat16 *gp) {
  constexpr int VecLen = 8;
  constexpr int num_elems = dv;
  constexpr int num_rows = lqp;
  bfloat16 *__restrict pGp = gp;
  for (int row = 0; row < num_rows; row++) {
    aie::vector<bfloat16, VecLen> sp_bcast =
        aie::broadcast<bfloat16, VecLen>(*(sp + row));
    for (int32_t i = 0; i < num_elems; i += VecLen) {
      aie::vector<bfloat16, VecLen> temp = aie::load_v<VecLen>(pGp);
      temp = aie::div(temp, sp_bcast);
      aie::store_v(pGp, temp);
      pGp += VecLen;
    }
  }
}

void add_gp_g(bfloat16 *gp, bfloat16 *g) {
  constexpr int VecLen = 32;
  constexpr int num_elems = lqp * dv;
  bfloat16 *__restrict gp_ptr = gp;
  bfloat16 *__restrict g_ptr = g;
  for (unsigned j = 0; j < num_elems / VecLen; j++) {
    aie::vector<bfloat16, VecLen> gp_vec = aie::load_v<VecLen>(gp_ptr);
    aie::vector<bfloat16, VecLen> g_vec = aie::load_v<VecLen>(g_ptr);
    auto accTemp = aie::add(gp_vec, g_vec);
    aie::store_v(g_ptr, accTemp);
    gp_ptr += VecLen;
    g_ptr += VecLen;
  }
}

} // extern "C"
