//===- attn.cc --------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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

// Default values if not provided by Makefile
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

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
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

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_A> A1 =
                  aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              aie::vector<T_in, MMUL::size_B> B1 =
                  aie::load_v<MMUL::size_B>(pB2);
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

// bf16 MatMul kernel definion with bf16 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {

  // After extensive experimentation, the 4x8x4 aie::mmul size was found to be
  // optimal for AIE2, in combination with the 4x4 mmul expanded kernel
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // Since the kernel has been expanded 4 times for both A ('m' dimension) and B
  // ('n' dimension), the following assertions verify this even division for
  // the single AIE MatMul dimensionality Notice that 'k' dimension is not
  // spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t>(pA, pB, pC);
}

#define log2e 1.44269504089

__attribute__((always_inline)) v8bfloat16 getExpBf16(v8bfloat16 x) {

  constexpr int VecLen = 8;

  // Calculate the e^(x) function as 2^(log2e * x)
  aie::vector<bfloat16, VecLen> input_bf16 = x;
  aie::accum<accfloat, VecLen> exp_in;
  aie::vector<bfloat16, VecLen> exp_val;
  aie::vector<bfloat16, VecLen> log2e_vec =
      aie::broadcast<bfloat16, VecLen>(log2e);

  exp_in = aie::mul(input_bf16, log2e_vec);
  exp_val = aie::exp2<bfloat16>(exp_in.to_vector<float>());
  return exp_val;
}

extern "C" {

void matmul_a_b_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *out) {
  // Buffer shapes:
  // A: [lqp, dk] = [32, 64]
  // B: [dk, lkp] = [64, 96]
  // Out: [lqp, lkp] = [32, 96]
  matmul_vectorized_8x8x8_bf16_bf16<lqp, dk, lkp>(a_in, b_in, out);
}

void matmul_g_b_bf16(bfloat16 *g_in, bfloat16 *b_in, bfloat16 *out) {
  // Buffer shapes:
  // G: [lqp, lkp] = [32, 96]
  // B: [lkp, dv] = [96, 64]
  // Out: [lqp, dv] = [32, 64]
  matmul_vectorized_8x8x8_bf16_bf16<lqp, lkp, dv>(g_in, b_in, out);
}

void zero_fill_gp_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, dv] = [32, 64]
  zero_vectorized<bfloat16, lqp, dv, 16>(c_out);
}

void zero_fill_sp_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, 1] = [32, 1]
  zero_vectorized<bfloat16, lqp, 1, 16>(c_out);
}

void zero_fill_g_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, lkp] = [32, 96]
  zero_vectorized<bfloat16, lqp, lkp, 16>(c_out);
}

void neg_inf_fill_up_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, 1] = [32, 1]
  neg_inf_vectorized<bfloat16, lqp, 1, 16>(c_out);
}

void max_g_bf16(bfloat16 *in, bfloat16 *out) {
  // u = np.max(G, axis=-1, keepdims=True)
  // Buffer shape:
  // Input: [lqp, lkp] = [32, 96]
  // Layout: [12x4x8x8xbf16]
  // Output: [lqp, 1] = [32, 1]
  constexpr int VecLen = 8;
  constexpr int num_elems = lkp;
  constexpr int num_rows = lqp;
  bfloat16 *__restrict pOut = out;
  uint16_t neg_infinity = (uint16_t)0xff80;
  bfloat16 *bf_neg_infinity = (bfloat16 *)&neg_infinity;
  aie::vector<bfloat16, VecLen> outputVec;
  for (int rowVec = 0; rowVec < num_rows; rowVec += VecLen) {
    // Unroll by 2 to process 2 rows simultaneously
    for (int rowVecElem = 0; rowVecElem < VecLen; rowVecElem += 2) {
      aie::vector<bfloat16, VecLen> max_vec0 =
          aie::broadcast<bfloat16, VecLen>((*bf_neg_infinity));
      aie::vector<bfloat16, VecLen> max_vec1 =
          aie::broadcast<bfloat16, VecLen>((*bf_neg_infinity));
      aie::vector<bfloat16, VecLen> temp0, temp1;
      int row0 = rowVec + rowVecElem;
      int row1 = rowVec + rowVecElem + 1;
      int row_block0 = row0 / VecLen;
      int row_in_block0 = row0 % VecLen;
      int row_block1 = row1 / VecLen;
      int row_in_block1 = row1 % VecLen;
      for (int32_t col_block = 0; col_block < num_elems / VecLen; col_block++)
        chess_prepare_for_pipelining chess_loop_range(12, ) {
          int offset0 = col_block * (num_rows * VecLen) +
                        row_block0 * (VecLen * VecLen) + row_in_block0 * VecLen;
          int offset1 = col_block * (num_rows * VecLen) +
                        row_block1 * (VecLen * VecLen) + row_in_block1 * VecLen;
          temp0 = aie::load_v<VecLen>(in + offset0);
          temp1 = aie::load_v<VecLen>(in + offset1);
          max_vec0 = aie::max(max_vec0, temp0);
          max_vec1 = aie::max(max_vec1, temp1);
        }
      outputVec[rowVecElem] = aie::reduce_max(max_vec0);
      outputVec[rowVecElem + 1] = aie::reduce_max(max_vec1);
    }
    aie::store_v(pOut, outputVec);
    pOut += VecLen;
  }
}

void maximum_up_u_bf16(bfloat16 *up, bfloat16 *u) {
  // u = np.maximum(u, up)
  // Buffer shape:
  // up: [lqp, 1] = [32, 1]
  // u: [lqp, 1] = [32, 1]
  constexpr int VecLen = 32;
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
  // G = np.exp(G - u)
  // Buffer shape:
  // u: [lqp, 1] = [32, 1]
  // G: [lqp, lkp] = [32, 96]
  // Layout: [12x4x8x8xbf16]
  constexpr int VecLen = 8;
  constexpr int num_elems = lkp;
  constexpr int num_rows = lqp;
  aie::vector<bfloat16, VecLen> uVec;

  // G <- exp(G - u) - fused subtraction and exponentiation with unrolled rows
  for (int rowVec = 0; rowVec < num_rows; rowVec += VecLen) {
    uVec = aie::load_v<VecLen>(u + rowVec);
    // Unroll by 2 to process 2 rows simultaneously
    for (int rowVecElem = 0; rowVecElem < VecLen; rowVecElem += 2) {
      aie::vector<bfloat16, VecLen> u_bcast0 =
          aie::broadcast<bfloat16, VecLen>(uVec[rowVecElem]);
      aie::vector<bfloat16, VecLen> u_bcast1 =
          aie::broadcast<bfloat16, VecLen>(uVec[rowVecElem + 1]);
      aie::vector<bfloat16, VecLen> temp0, temp1, exp_val0, exp_val1;
      int row0 = rowVec + rowVecElem;
      int row1 = rowVec + rowVecElem + 1;
      int row_block0 = row0 / VecLen;
      int row_in_block0 = row0 % VecLen;
      int row_block1 = row1 / VecLen;
      int row_in_block1 = row1 % VecLen;
      for (int32_t col_block = 0; col_block < num_elems / VecLen; col_block++)
        chess_prepare_for_pipelining chess_loop_range(12, ) {
          int offset0 = col_block * (num_rows * VecLen) +
                        row_block0 * (VecLen * VecLen) + row_in_block0 * VecLen;
          int offset1 = col_block * (num_rows * VecLen) +
                        row_block1 * (VecLen * VecLen) + row_in_block1 * VecLen;
          temp0 = aie::load_v<VecLen>(g + offset0);
          temp1 = aie::load_v<VecLen>(g + offset1);
          temp0 = aie::sub(temp0, u_bcast0);
          temp1 = aie::sub(temp1, u_bcast1);
          exp_val0 = getExpBf16(temp0);
          exp_val1 = getExpBf16(temp1);
          aie::store_v(g + offset0, exp_val0);
          aie::store_v(g + offset1, exp_val1);
        }
    }
  }
}

void exp_up_minus_u(bfloat16 *up, bfloat16 *u, bfloat16 *r) {
  // r = exp(up - u)
  // Buffer shape:
  // up: [lqp, 1] = [32, 1]
  // u: [lqp, 1] = [32, 1]
  // r: [lqp, 1] = [32, 1]
  constexpr int VecLen = 8;
  constexpr int num_elems = lqp;
  // r <- exp(up - u) - fused subtraction and exponentiation
  bfloat16 *__restrict pr = r;
  bfloat16 *__restrict pu = u;
  bfloat16 *__restrict pup = up;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> uTemp = aie::load_v<VecLen>(pu);
    aie::vector<bfloat16, VecLen> upTemp = aie::load_v<VecLen>(pup);
    aie::vector<bfloat16, VecLen> rTemp = aie::sub(upTemp, uTemp);
    aie::vector<bfloat16, VecLen> exp_val = getExpBf16(rTemp);
    aie::store_v(pr, exp_val);
    pr += VecLen;
    pu += VecLen;
    pup += VecLen;
  }
}

void mul_r_gp(bfloat16 *r, bfloat16 *gp) {
  // Gp = Gp * r
  // Buffer shape:
  // Gp: [lqp, dv] = [32, 64]
  // Layout: [8x4x8x8xbf16]
  // r: [lqp, 1] = [32, 1]
  constexpr int VecLen = 32;
  constexpr int num_elems = dv;
  constexpr int num_rows = lqp;
  aie::vector<bfloat16, VecLen> rVec;
  for (int rowVec = 0; rowVec < num_rows; rowVec += VecLen) {
    rVec = aie::load_v<VecLen>(r + rowVec);
    // Unroll by 2 to process 2 rows simultaneously
    for (int rowVecElem = 0; rowVecElem < VecLen; rowVecElem += 2) {
      aie::vector<bfloat16, VecLen> r_bcast0 =
          aie::broadcast<bfloat16, VecLen>(rVec[rowVecElem]);
      aie::vector<bfloat16, VecLen> r_bcast1 =
          aie::broadcast<bfloat16, VecLen>(rVec[rowVecElem + 1]);
      aie::vector<bfloat16, VecLen> temp0, temp1;
      int row0 = rowVec + rowVecElem;
      int row1 = rowVec + rowVecElem + 1;
      int row_block0 = row0 / VecLen;
      int row_in_block0 = row0 % VecLen;
      int row_block1 = row1 / VecLen;
      int row_in_block1 = row1 % VecLen;
      for (int32_t col_block = 0; col_block < num_elems / VecLen; col_block++)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          int offset0 = col_block * (num_rows * VecLen) +
                        row_block0 * (VecLen * VecLen) + row_in_block0 * VecLen;
          int offset1 = col_block * (num_rows * VecLen) +
                        row_block1 * (VecLen * VecLen) + row_in_block1 * VecLen;
          temp0 = aie::load_v<VecLen>(gp + offset0);
          temp1 = aie::load_v<VecLen>(gp + offset1);
          temp0 = aie::mul(temp0, r_bcast0);
          temp1 = aie::mul(temp1, r_bcast1);
          aie::store_v(gp + offset0, temp0);
          aie::store_v(gp + offset1, temp1);
        }
    }
  }
}

void sum_g(bfloat16 *g, bfloat16 *s) {
  // s = sum(G, axis=-1, keepdims=True)
  // Buffer shape:
  // G: [lqp, lkp] = [32, 96]
  // Layout: [12x4x8x8xbf16]
  // s: [lqp, 1] = [32, 1]
  constexpr int VecLen = 32;
  constexpr int num_elems = lkp;
  constexpr int num_rows = lqp;
  bfloat16 *__restrict ps = s;
  for (int rowVec = 0; rowVec < num_rows; rowVec += VecLen) {
    aie::vector<bfloat16, VecLen> sVec;
    for (int rowVecElem = 0; rowVecElem < VecLen; rowVecElem++) {
      aie::vector<bfloat16, VecLen> temp;
      bfloat16 sum_value = 0.0;
      int row = rowVec + rowVecElem;
      int row_block = row / VecLen;
      int row_in_block = row % VecLen;
      for (int32_t col_block = 0; col_block < num_elems / VecLen; col_block++)
        chess_prepare_for_pipelining chess_loop_range(12, ) {
          int offset = col_block * (num_rows * VecLen) +
                       row_block * (VecLen * VecLen) + row_in_block * VecLen;
          temp = aie::load_v<VecLen>(g + offset);
          sum_value += aie::reduce_add(temp);
        }
      sVec[rowVecElem] = sum_value;
    }
    aie::store_v(ps, sVec);
    ps += VecLen;
  }
}

void accum_sp_r_s(bfloat16 *sp, bfloat16 *r, bfloat16 *s) {
  // s += sp * r
  // Buffer shape:
  // sp: [lqp, 1] = [32, 1]
  // r: [lqp, 1] = [32, 1]
  // s: [lqp, 1] = [32, 1]
  constexpr int VecLen = 32;
  constexpr int num_elems = lqp;
  bfloat16 *__restrict pr = r;
  bfloat16 *__restrict ps = s;
  bfloat16 *__restrict psp = sp;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> rTemp = aie::load_v<VecLen>(pr);
    aie::vector<bfloat16, VecLen> spTemp = aie::load_v<VecLen>(psp);
    aie::accum<accfloat, VecLen> accTemp = aie::mul(rTemp, spTemp);
    accTemp = aie::add(accTemp, aie::load_v<VecLen>(ps));
    aie::vector<bfloat16, VecLen> sTemp = to_v32bfloat16(accTemp);
    aie::store_v(ps, sTemp);
    pr += VecLen;
    ps += VecLen;
    psp += VecLen;
  }
}

void vector_copy_32elems(const int offset, const bfloat16 *__restrict inputs,
                         bfloat16 *__restrict outputs) {
  constexpr int VecLen = 32;
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

void div_gp_sp(bfloat16 *sp, bfloat16 *gp) {
  // Gp = Gp / sp
  // Buffer shape:
  // Gp: [lqp, dv] = [32, 64]
  // Layout: [8x4x8x8xbf16]
  // sp: [lqp, 1] = [32, 1]
  constexpr int VecLen = 32;
  constexpr int num_elems = dv;
  constexpr int num_rows = lqp;
  aie::vector<bfloat16, VecLen> spVec;
  for (int rowVec = 0; rowVec < num_rows; rowVec += VecLen) {
    spVec = aie::load_v<VecLen>(sp + rowVec);
    // Unroll by 2 to process 2 rows simultaneously
    for (int rowVecElem = 0; rowVecElem < VecLen; rowVecElem += 2) {
      aie::vector<bfloat16, VecLen> sp_bcast0 =
          aie::broadcast<bfloat16, VecLen>(spVec[rowVecElem]);
      aie::vector<bfloat16, VecLen> sp_bcast1 =
          aie::broadcast<bfloat16, VecLen>(spVec[rowVecElem + 1]);
      aie::vector<bfloat16, VecLen> temp0, temp1;
      int row0 = rowVec + rowVecElem;
      int row1 = rowVec + rowVecElem + 1;
      int row_block0 = row0 / VecLen;
      int row_in_block0 = row0 % VecLen;
      int row_block1 = row1 / VecLen;
      int row_in_block1 = row1 % VecLen;
      for (int32_t col_block = 0; col_block < num_elems / VecLen; col_block++)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          int offset0 = col_block * (num_rows * VecLen) +
                        row_block0 * (VecLen * VecLen) + row_in_block0 * VecLen;
          int offset1 = col_block * (num_rows * VecLen) +
                        row_block1 * (VecLen * VecLen) + row_in_block1 * VecLen;
          temp0 = aie::load_v<VecLen>(gp + offset0);
          temp1 = aie::load_v<VecLen>(gp + offset1);
          temp0 = aie::div(temp0, sp_bcast0);
          temp1 = aie::div(temp1, sp_bcast1);
          aie::store_v(gp + offset0, temp0);
          aie::store_v(gp + offset1, temp1);
        }
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
