//===- attn_npu1.cc - Flash attention kernels for NPU1 (AIE2) ---*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
// NPU1 (AIE2) variant of kernel_fusion_based flash attention.
// Key differences from NPU2 (attn_npu2.cc):
//   - mmul<4,8,4> instead of mmul<8,8,8>
//   - LUT-based exp instead of aie::exp2
//   - Column-major 4x4 block tiling instead of 8x8
//   - aie::div instead of aie::inv
//   - scale_g_bf16: explicit 1/sqrt(dk) scaling after matmul
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

#include "lut_based_ops.h"
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

#ifndef dv_full
#define dv_full dv
#endif

#ifndef dk_full
#define dk_full dk
#endif

// ============================================================================
// Matmul template: 4x4 expansion with transpose_b control for AIE2 mmul<4,8,4>
// ============================================================================

// Column-major B matmul with compile-time transpose control.
// transpose_b: true  = apply aie::transpose before mac (K DMA: inner [n_in,
// k_in])
//              false = load B as-is, hardware mul_4x8_4x8T transposes (V DMA:
//              inner [k_in, n_in])
// A and C are always column-major tiled.
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool transpose_b = true>
static inline void matmul_vectorized_4x4(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;
      T_out *__restrict pC3 = pC + ((z + 2)) * MMUL::size_C;
      T_out *__restrict pC4 = pC + ((z + 3)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + ((z + 2)) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + ((z + 3)) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
          const T_in *__restrict pB2 = pB + ((j + 1)) * colA * MMUL::size_B;
          const T_in *__restrict pB3 = pB + ((j + 2)) * colA * MMUL::size_B;
          const T_in *__restrict pB4 = pB + ((j + 3)) * colA * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A3 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += rowA * MMUL::size_A;

          aie::vector<T_in, MMUL::size_B> B0, B1, B2, B3;
          if constexpr (transpose_b) {
            // K DMA k-major block layout: block (n=j, k=i) at i*colB+j.
            // Sub-tile elements are [n_in, k_in], transpose to [k_in, n_in].
            const T_in *__restrict pBk0 = pB + (0 * colB + j) * MMUL::size_B;
            const T_in *__restrict pBk1 =
                pB + (0 * colB + (j + 1)) * MMUL::size_B;
            const T_in *__restrict pBk2 =
                pB + (0 * colB + (j + 2)) * MMUL::size_B;
            const T_in *__restrict pBk3 =
                pB + (0 * colB + (j + 3)) * MMUL::size_B;
            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pBk0), t, s);
            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pBk1), t, s);
            B2 = aie::transpose(aie::load_v<MMUL::size_B>(pBk2), t, s);
            B3 = aie::transpose(aie::load_v<MMUL::size_B>(pBk3), t, s);
          } else {
            B0 = aie::load_v<MMUL::size_B>(pB1);
            B1 = aie::load_v<MMUL::size_B>(pB2);
            B2 = aie::load_v<MMUL::size_B>(pB3);
            B3 = aie::load_v<MMUL::size_B>(pB4);
          }
          pB1 += MMUL::size_B;
          pB2 += MMUL::size_B;
          pB3 += MMUL::size_B;
          pB4 += MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C02 =
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C03 =
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C12 =
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C13 =
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C22 =
              aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C23 =
              aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C32 =
              aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C33 =
              aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C * rowA);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C02(acc_C02);
          MMUL C03(acc_C03);

          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C12(acc_C12);
          MMUL C13(acc_C13);

          MMUL C20(acc_C20);
          MMUL C21(acc_C21);
          MMUL C22(acc_C22);
          MMUL C23(acc_C23);

          MMUL C30(acc_C30);
          MMUL C31(acc_C31);
          MMUL C32(acc_C32);
          MMUL C33(acc_C33);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          C02.mac(A0, B2);
          C03.mac(A0, B3);
          C12.mac(A1, B2);
          C13.mac(A1, B3);

          C20.mac(A2, B0);
          C21.mac(A2, B1);
          C30.mac(A3, B0);
          C31.mac(A3, B1);

          C22.mac(A2, B2);
          C23.mac(A2, B3);
          C32.mac(A3, B2);
          C33.mac(A3, B3);

          for (unsigned i = 1; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              A2 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += rowA * MMUL::size_A;
              A3 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += rowA * MMUL::size_A;

              if constexpr (transpose_b) {
                const T_in *__restrict pBk0 =
                    pB + (i * colB + j) * MMUL::size_B;
                const T_in *__restrict pBk1 =
                    pB + (i * colB + (j + 1)) * MMUL::size_B;
                const T_in *__restrict pBk2 =
                    pB + (i * colB + (j + 2)) * MMUL::size_B;
                const T_in *__restrict pBk3 =
                    pB + (i * colB + (j + 3)) * MMUL::size_B;
                B0 = aie::transpose(aie::load_v<MMUL::size_B>(pBk0), t, s);
                B1 = aie::transpose(aie::load_v<MMUL::size_B>(pBk1), t, s);
                B2 = aie::transpose(aie::load_v<MMUL::size_B>(pBk2), t, s);
                B3 = aie::transpose(aie::load_v<MMUL::size_B>(pBk3), t, s);
              } else {
                B0 = aie::load_v<MMUL::size_B>(pB1);
                B1 = aie::load_v<MMUL::size_B>(pB2);
                B2 = aie::load_v<MMUL::size_B>(pB3);
                B3 = aie::load_v<MMUL::size_B>(pB4);
              }
              pB1 += MMUL::size_B;
              pB2 += MMUL::size_B;
              pB3 += MMUL::size_B;
              pB4 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);

              C02.mac(A0, B2);
              C03.mac(A0, B3);
              C12.mac(A1, B2);
              C13.mac(A1, B3);

              C20.mac(A2, B0);
              C21.mac(A2, B1);
              C30.mac(A3, B0);
              C31.mac(A3, B1);

              C22.mac(A2, B2);
              C23.mac(A2, B3);
              C32.mac(A3, B2);
              C33.mac(A3, B3);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C02.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C03.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;

          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C12.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C13.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;

          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C22.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C23.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;

          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C32.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C33.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
        }
    }

  event1();
}

// bf16 MatMul kernel with bf16 outputs for AIE2 (4x8x4).
// transpose_b: controls whether B blocks are software-transposed before mac.
template <unsigned m, unsigned k, unsigned n, bool transpose_b = true>
static inline void
matmul_vectorized_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (4 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (4 * t) == 0); // 'n' dimension

  return matmul_vectorized_4x4<bfloat16, bfloat16, (m / r), (k / s), (n / t), r,
                               s, t, transpose_b>(pA, pB, pC);
}

// ============================================================================
// LUT-based exponential for AIE2 (no native exp2)
// ============================================================================

alignas(aie::vector_decl_align) extern int16 exp_ilut_ab[512];
alignas(aie::vector_decl_align) extern int16 exp_ilut_cd[512];
alignas(aie::vector_decl_align) extern int16 exp_flut_ab[512];
alignas(aie::vector_decl_align) extern int16 exp_flut_cd[512];

__attribute__((always_inline)) v16accfloat getExpBf16(v16bfloat16 x) {
  bfloat16 __aie_dm_resource_a *ilut_ab =
      (bfloat16 __aie_dm_resource_a *)exp_ilut_ab;
  bfloat16 __aie_dm_resource_b *ilut_cd =
      (bfloat16 __aie_dm_resource_b *)exp_ilut_cd;
  bfloat16 __aie_dm_resource_a *flut_ab =
      (bfloat16 __aie_dm_resource_a *)exp_flut_ab;
  bfloat16 __aie_dm_resource_b *flut_cd =
      (bfloat16 __aie_dm_resource_b *)exp_flut_cd;

  using lut_type = aie::lut<4, bfloat16, bfloat16>;
  const int LUT_elems = 256;
  const int step_i = 8;
  const int step_f = 0;

  lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
  lut_type lut_f(LUT_elems, flut_ab, flut_cd);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_i(lut_i, step_i);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_f(lut_f, step_f);

  aie::vector<bfloat16, 16> I_val_vec, F_val_vec;
  aie::accum<accfloat, 16> exp_val;
  aie::vector<bfloat16, 16> input_bf16 = x;

  // position of output decimal point = 8, making input become 8 bits, and for
  // LUT_elems = 256 lookup.
  aie::vector<int16, 32> input0 = v32int16(bfloat16_to_int(input_bf16, 8));
  aie::vector<int16, 16> input = aie::filter_even(input0);

  I_val_vec = lookup_i.fetch(input.cast_to<uint16>());
  F_val_vec = lookup_f.fetch(input.cast_to<uint16>());
  exp_val = aie::mul(I_val_vec, F_val_vec);
  return v16accfloat(exp_val);
}

// ============================================================================
// Scaling constant for 1/sqrt(dk_full)
// ============================================================================
#include <cmath>

static const double inv_sqrt_dk_val = 1.0 / sqrt((double)dk_full);

#define inv_sqrt_dk inv_sqrt_dk_val

// ============================================================================
// Kernel functions
// ============================================================================

extern "C" {

// Copy tile_size_q x dk elements from src to dst (single-pass vector copy)
void copy_tile(bfloat16 *src, bfloat16 *dst) {
  constexpr int VecLen = 16;
  constexpr int num_elems = lqp * dk;
  bfloat16 *__restrict ps = src;
  bfloat16 *__restrict pd = dst;
  for (unsigned j = 0; j < num_elems / VecLen; j++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(ps);
      aie::store_v(pd, v);
      ps += VecLen;
      pd += VecLen;
    }
}

void matmul_a_b_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *out) {
  // Buffer shapes:
  // A: [lqp, dk] (Q tile, column-major 4x4 tiled)
  // B: [lkp, dk] (K chunk, transpose per block)
  // Out: [lqp, lkp] (G matrix, column-major 4x4 tiled)
  matmul_vectorized_4x8x4_bf16_bf16<lqp, dk, lkp>(a_in, b_in, out);
}

void matmul_g_b_bf16(bfloat16 *g_in, bfloat16 *b_in, bfloat16 *out) {
  // Buffer shapes:
  // G: [lqp, lkp] (attention scores, column-major 4x4 tiled)
  // B: [lkp, dv] (V chunk, no software transpose)
  // Out: [lqp, dv] (attention output, column-major 4x4 tiled)
  //
  // G is in 4x4 column-major block layout (from QK matmul C output):
  //   Block [rb, cb] at g_in + rb * size_C + cb * rowA_C * size_C
  //   where size_C = r*t = 16, rowA_C = lqp/r = lqp/4.
  // But mmul<4,8,4> needs A in 4x8 block format (size_A = r*s = 32).
  // We load two adjacent column-blocks and interleave them into a 4x8 sub-tile.
  //
  // matmul: G[lqp, lkp] x V[lkp, dv] -> Out[lqp, dv]
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  constexpr unsigned rowA = lqp / r; // number of row-blocks of A/C
  constexpr unsigned colA = lkp / s; // number of k-blocks (A is 4x8)
  constexpr unsigned colB = dv / t;  // number of n-blocks of B/C
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;

  // 4x4 C-block layout parameters
  constexpr unsigned size_C_blk = r * t; // 16 elements per 4x4 block
  constexpr unsigned col_block_stride =
      rowA * size_C_blk; // stride between column-blocks = 16*16 = 256

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      bfloat16 *__restrict pC1 = out + (z)*MMUL::size_C;
      bfloat16 *__restrict pC2 = out + ((z + 1)) * MMUL::size_C;
      bfloat16 *__restrict pC3 = out + ((z + 2)) * MMUL::size_C;
      bfloat16 *__restrict pC4 = out + ((z + 3)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4) {
        const bfloat16 *__restrict pB1 = b_in + (j)*colA * MMUL::size_B;
        const bfloat16 *__restrict pB2 = b_in + ((j + 1)) * colA * MMUL::size_B;
        const bfloat16 *__restrict pB3 = b_in + ((j + 2)) * colA * MMUL::size_B;
        const bfloat16 *__restrict pB4 = b_in + ((j + 3)) * colA * MMUL::size_B;

        // Load A from 4x4 block format: read two 4x4 blocks, interleave to 4x8
        // For A sub-tile [z, i=0]: read C[rb=z, cb=0] and C[rb=z, cb=1]
        auto load_A_4x4 =
            [&](unsigned rb,
                unsigned kb) -> aie::vector<bfloat16, MMUL::size_A> {
          const bfloat16 *pLo =
              g_in + rb * size_C_blk + (2 * kb) * col_block_stride;
          const bfloat16 *pHi =
              g_in + rb * size_C_blk + (2 * kb + 1) * col_block_stride;
          aie::vector<bfloat16, 16> lo = aie::load_v<16>(pLo);
          aie::vector<bfloat16, 16> hi = aie::load_v<16>(pHi);
          // interleave_zip with step=4: takes alternating groups of 4 from lo,
          // hi lo = [r0c0..3 r1c0..3 r2c0..3 r3c0..3] hi = [r0c4..7 r1c4..7
          // r2c4..7 r3c4..7] result_lo = [r0c0..3 r0c4..7 r1c0..3 r1c4..7]
          // (rows 0-1, 8 cols) result_hi = [r2c0..3 r2c4..7 r3c0..3 r3c4..7]
          // (rows 2-3, 8 cols)
          auto [zlo, zhi] = aie::interleave_zip(lo, hi, 4);
          return aie::concat(zlo, zhi);
        };

        aie::vector<bfloat16, MMUL::size_A> A0 = load_A_4x4(z, 0);
        aie::vector<bfloat16, MMUL::size_A> A1 = load_A_4x4(z + 1, 0);
        aie::vector<bfloat16, MMUL::size_A> A2 = load_A_4x4(z + 2, 0);
        aie::vector<bfloat16, MMUL::size_A> A3 = load_A_4x4(z + 3, 0);

        aie::vector<bfloat16, MMUL::size_B> B0, B1, B2, B3;
        B0 = aie::load_v<MMUL::size_B>(pB1);
        B1 = aie::load_v<MMUL::size_B>(pB2);
        B2 = aie::load_v<MMUL::size_B>(pB3);
        B3 = aie::load_v<MMUL::size_B>(pB4);
        pB1 += MMUL::size_B;
        pB2 += MMUL::size_B;
        pB3 += MMUL::size_B;
        pB4 += MMUL::size_B;

        aie::vector<bfloat16, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<bfloat16, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C02 =
            aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C03 =
            aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C * rowA);

        aie::vector<bfloat16, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<bfloat16, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C12 =
            aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C13 =
            aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C * rowA);

        aie::vector<bfloat16, MMUL::size_C> acc_C20 =
            aie::load_v<MMUL::size_C>(pC3);
        aie::vector<bfloat16, MMUL::size_C> acc_C21 =
            aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C22 =
            aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C23 =
            aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C * rowA);

        aie::vector<bfloat16, MMUL::size_C> acc_C30 =
            aie::load_v<MMUL::size_C>(pC4);
        aie::vector<bfloat16, MMUL::size_C> acc_C31 =
            aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C32 =
            aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C * rowA);
        aie::vector<bfloat16, MMUL::size_C> acc_C33 =
            aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C * rowA);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C02(acc_C02);
        MMUL C03(acc_C03);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C12(acc_C12);
        MMUL C13(acc_C13);
        MMUL C20(acc_C20);
        MMUL C21(acc_C21);
        MMUL C22(acc_C22);
        MMUL C23(acc_C23);
        MMUL C30(acc_C30);
        MMUL C31(acc_C31);
        MMUL C32(acc_C32);
        MMUL C33(acc_C33);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
        C02.mac(A0, B2);
        C03.mac(A0, B3);
        C12.mac(A1, B2);
        C13.mac(A1, B3);
        C20.mac(A2, B0);
        C21.mac(A2, B1);
        C30.mac(A3, B0);
        C31.mac(A3, B1);
        C22.mac(A2, B2);
        C23.mac(A2, B3);
        C32.mac(A3, B2);
        C33.mac(A3, B3);

        for (unsigned i = 1; i < colA; ++i) {
          A0 = load_A_4x4(z, i);
          A1 = load_A_4x4(z + 1, i);
          A2 = load_A_4x4(z + 2, i);
          A3 = load_A_4x4(z + 3, i);

          B0 = aie::load_v<MMUL::size_B>(pB1);
          B1 = aie::load_v<MMUL::size_B>(pB2);
          B2 = aie::load_v<MMUL::size_B>(pB3);
          B3 = aie::load_v<MMUL::size_B>(pB4);
          pB1 += MMUL::size_B;
          pB2 += MMUL::size_B;
          pB3 += MMUL::size_B;
          pB4 += MMUL::size_B;

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);
          C02.mac(A0, B2);
          C03.mac(A0, B3);
          C12.mac(A1, B2);
          C13.mac(A1, B3);
          C20.mac(A2, B0);
          C21.mac(A2, B1);
          C30.mac(A3, B0);
          C31.mac(A3, B1);
          C22.mac(A2, B2);
          C23.mac(A2, B3);
          C32.mac(A3, B2);
          C33.mac(A3, B3);
        }

        aie::store_v(pC1, C00.template to_vector<bfloat16>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C01.template to_vector<bfloat16>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C02.template to_vector<bfloat16>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C03.template to_vector<bfloat16>());
        pC1 += MMUL::size_C * rowA;

        aie::store_v(pC2, C10.template to_vector<bfloat16>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C11.template to_vector<bfloat16>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C12.template to_vector<bfloat16>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C13.template to_vector<bfloat16>());
        pC2 += MMUL::size_C * rowA;

        aie::store_v(pC3, C20.template to_vector<bfloat16>());
        pC3 += MMUL::size_C * rowA;
        aie::store_v(pC3, C21.template to_vector<bfloat16>());
        pC3 += MMUL::size_C * rowA;
        aie::store_v(pC3, C22.template to_vector<bfloat16>());
        pC3 += MMUL::size_C * rowA;
        aie::store_v(pC3, C23.template to_vector<bfloat16>());
        pC3 += MMUL::size_C * rowA;

        aie::store_v(pC4, C30.template to_vector<bfloat16>());
        pC4 += MMUL::size_C * rowA;
        aie::store_v(pC4, C31.template to_vector<bfloat16>());
        pC4 += MMUL::size_C * rowA;
        aie::store_v(pC4, C32.template to_vector<bfloat16>());
        pC4 += MMUL::size_C * rowA;
        aie::store_v(pC4, C33.template to_vector<bfloat16>());
        pC4 += MMUL::size_C * rowA;
      }
    }

  event1();
}

void zero_fill_gp_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, dv]
  zero_vectorized<bfloat16, lqp, dv, 16>(c_out);
}

void zero_fill_sp_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, 1]
  zero_vectorized<bfloat16, lqp, 1, 16>(c_out);
}

void zero_fill_g_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, lkp]
  zero_vectorized<bfloat16, lqp, lkp, 16>(c_out);
}

void neg_inf_fill_up_bf16(bfloat16 *c_out) {
  // Buffer shape: [lqp, 1]
  neg_inf_vectorized<bfloat16, lqp, 1, 16>(c_out);
}

// Scale G by 1/sqrt(dk_full) in-place.
// G is column-major 4x4 block tiled: [lqp, lkp].
void scale_g_bf16(bfloat16 *g) {
  constexpr int VecLen = 16;
  constexpr int num_elems = lqp * lkp;
  bfloat16 scale_val = (bfloat16)inv_sqrt_dk;
  aie::vector<bfloat16, VecLen> scale_vec =
      aie::broadcast<bfloat16, VecLen>(scale_val);
  bfloat16 *__restrict pG = g;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(pG);
    aie::accum<accfloat, VecLen> acc = aie::mul(v, scale_vec);
    aie::store_v(pG, acc.to_vector<bfloat16>());
    pG += VecLen;
  }
}

// Row-wise max of G matrix.
// G is column-major 4x4 block tiled.
// VecLen=16 reads one full 4x4 block (4 rows x 4 cols).
// Within a 16-wide vector: elements [0..3]=row0, [4..7]=row1, [8..11]=row2,
// [12..15]=row3. Since aie::vector<bf16,4> is not supported on AIE2,
// use scalar element access for per-row reduction.
void max_g_bf16(bfloat16 *in, bfloat16 *out) {
  constexpr int VecLen = 16;
  constexpr int BlockSize = 16; // 4x4 block
  constexpr int ColsPerBlock = 4;
  constexpr int RowsPerBlock = 4;
  constexpr int col_blocks = lkp / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride =
      lqp * ColsPerBlock; // stride between column blocks

  // Use bf16 lowest (0xff7f) instead of -inf to avoid NaN propagation.
  uint16_t lowest_u16 = (uint16_t)0xff7f;
  bfloat16 lowest_val = *(bfloat16 *)&lowest_u16;

  bfloat16 *__restrict pOut = out;
  for (int rb = 0; rb < row_blocks; rb++) {
    aie::vector<bfloat16, VecLen> max_vec =
        aie::broadcast<bfloat16, VecLen>(lowest_val);
    int base = rb * BlockSize;
    for (int cb = 0; cb < col_blocks; cb++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bfloat16, VecLen> v =
            aie::load_v<VecLen>(in + base + cb * block_stride);
        max_vec = aie::max(max_vec, v);
      }
    // Extract per-row max via scalar access.
    // Row i occupies elements [i*4 .. i*4+3] in the 16-wide vector.
    for (int row = 0; row < RowsPerBlock; row++) {
      bfloat16 m = max_vec[row * ColsPerBlock];
      for (int c = 1; c < ColsPerBlock; c++) {
        bfloat16 val = max_vec[row * ColsPerBlock + c];
        if (val > m)
          m = val;
      }
      pOut[row] = m;
    }
    pOut += RowsPerBlock;
  }
}

void maximum_up_u_bf16(bfloat16 *up, bfloat16 *u) {
  // u = max(u, up)
  // Buffer shape: [lqp, 1]
  constexpr int VecLen = 16;
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

// G = exp(G - u) in-place. G is column-major 4x4 block tiled.
// VecLen=16 processes one full 4x4 block (4 rows x 4 cols).
// Uses LUT-based exp.
void exp_g_minus_u(bfloat16 *u, bfloat16 *g) {
  constexpr int VecLen = 16;
  constexpr int BlockSize = 16;
  constexpr int ColsPerBlock = 4;
  constexpr int RowsPerBlock = 4;
  constexpr int col_blocks = lkp / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride = lqp * ColsPerBlock;

  for (int rb = 0; rb < row_blocks; rb++) {
    // Build 16-wide u vector: 4 rows x 4 cols, each row's u broadcast to its
    // 4 column elements. Use scalar set since vector<bf16,4> not supported.
    int row_start = rb * RowsPerBlock;
    aie::vector<bfloat16, VecLen> u_vec = aie::zeros<bfloat16, VecLen>();
    for (int row = 0; row < RowsPerBlock; row++) {
      bfloat16 uval = u[row_start + row];
      for (int c = 0; c < ColsPerBlock; c++) {
        u_vec[row * ColsPerBlock + c] = uval;
      }
    }

    int base = rb * BlockSize;
    for (int cb = 0; cb < col_blocks; cb++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        int off = base + cb * block_stride;
        aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(g + off);
        v = aie::sub(v, u_vec);
        // LUT-based exp: getExpBf16 takes v16bfloat16, returns v16accfloat
        aie::vector<bfloat16, VecLen> exp_val = to_v16bfloat16(getExpBf16(v));
        aie::store_v(g + off, exp_val);
      }
  }
}

// r = exp(up - u). Uses LUT-based exp.
void exp_up_minus_u(bfloat16 *up, bfloat16 *u, bfloat16 *r) {
  constexpr int VecLen = 16;
  constexpr int num_elems = lqp;
  bfloat16 *__restrict pr = r;
  bfloat16 *__restrict pu = u;
  bfloat16 *__restrict pup = up;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> uTemp = aie::load_v<VecLen>(pu);
    aie::vector<bfloat16, VecLen> upTemp = aie::load_v<VecLen>(pup);
    aie::vector<bfloat16, VecLen> diff = aie::sub(upTemp, uTemp);
    // LUT-based exp
    aie::vector<bfloat16, VecLen> exp_val = to_v16bfloat16(getExpBf16(diff));
    aie::store_v(pr, exp_val);
    pr += VecLen;
    pu += VecLen;
    pup += VecLen;
  }
}

// Gp = Gp * r (per-row scaling).
// Gp is column-major 4x4 block tiled: [lqp, dv].
void mul_r_gp(bfloat16 *r, bfloat16 *gp) {
  constexpr int VecLen = 16;
  constexpr int BlockSize = 16; // 4x4 block
  constexpr int ColsPerBlock = 4;
  constexpr int RowsPerBlock = 4;
  constexpr int col_blocks = dv / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride =
      lqp * ColsPerBlock; // stride between column blocks

  for (int rb = 0; rb < row_blocks; rb++) {
    // Build 16-wide r vector: 4 rows x 4 cols, each row's r broadcast
    int row_start = rb * RowsPerBlock;
    aie::vector<bfloat16, VecLen> r_vec = aie::zeros<bfloat16, VecLen>();
    for (int row = 0; row < RowsPerBlock; row++) {
      bfloat16 rval = r[row_start + row];
      for (int c = 0; c < ColsPerBlock; c++) {
        r_vec[row * ColsPerBlock + c] = rval;
      }
    }

    int base = rb * BlockSize;
    for (int cb = 0; cb < col_blocks; cb++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        int off = base + cb * block_stride;
        aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(gp + off);
        aie::accum<accfloat, VecLen> acc = aie::mul(v, r_vec);
        aie::store_v(gp + off, acc.to_vector<bfloat16>());
      }
  }
}

// s = sum(G, axis=-1, keepdims=True).
// G is column-major 4x4 block tiled.
void sum_g(bfloat16 *g, bfloat16 *s) {
  constexpr int VecLen = 16;
  constexpr int BlockSize = 16;
  constexpr int ColsPerBlock = 4;
  constexpr int RowsPerBlock = 4;
  constexpr int col_blocks = lkp / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride = lqp * ColsPerBlock;

  bfloat16 *__restrict ps = s;
  for (int rb = 0; rb < row_blocks; rb++) {
    // Accumulate sum across column blocks for 4 rows
    aie::accum<accfloat, VecLen> sum_acc = aie::zeros<accfloat, VecLen>();
    int base = rb * BlockSize;
    for (int cb = 0; cb < col_blocks; cb++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bfloat16, VecLen> v =
            aie::load_v<VecLen>(g + base + cb * block_stride);
        sum_acc = aie::add(sum_acc, v);
      }
    // Reduce each 4-element row slice via scalar access.
    aie::vector<float, VecLen> sum_v = sum_acc.to_vector<float>();
    for (int row = 0; row < RowsPerBlock; row++) {
      float row_sum = 0.0f;
      for (int c = 0; c < ColsPerBlock; c++) {
        row_sum += sum_v[row * ColsPerBlock + c];
      }
      ps[row] = (bfloat16)row_sum;
    }
    ps += RowsPerBlock;
  }
}

void accum_sp_r_s(bfloat16 *sp, bfloat16 *r, bfloat16 *s) {
  // s += sp * r
  // Buffer shape: [lqp, 1]
  constexpr int VecLen = 16;
  constexpr int num_elems = lqp;
  bfloat16 *__restrict pr = r;
  bfloat16 *__restrict ps = s;
  bfloat16 *__restrict psp = sp;
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> rTemp = aie::load_v<VecLen>(pr);
    aie::vector<bfloat16, VecLen> spTemp = aie::load_v<VecLen>(psp);
    aie::accum<accfloat, VecLen> accTemp = aie::mul(rTemp, spTemp);
    accTemp = aie::add(accTemp, aie::load_v<VecLen>(ps));
    aie::vector<bfloat16, VecLen> sTemp = to_v16bfloat16(accTemp);
    aie::store_v(ps, sTemp);
    pr += VecLen;
    ps += VecLen;
    psp += VecLen;
  }
}

void vector_copy_32elems(const int offset, const bfloat16 *__restrict inputs,
                         bfloat16 *__restrict outputs) {
  constexpr int VecLen = 16;
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

// Gp = Gp / sp (per-row normalization).
// Gp is column-major 4x4 block tiled: [lqp, dv].
// Uses aie::div (AIE2-compatible, no aie::inv).
void div_gp_sp(bfloat16 *sp, bfloat16 *gp) {
  constexpr int VecLen = 16;
  constexpr int BlockSize = 16; // 4x4 block
  constexpr int ColsPerBlock = 4;
  constexpr int RowsPerBlock = 4;
  constexpr int col_blocks = dv / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride =
      lqp * ColsPerBlock; // stride between column blocks

  for (int rb = 0; rb < row_blocks; rb++) {
    // Build 16-wide sp vector via scalar access
    int row_start = rb * RowsPerBlock;
    aie::vector<bfloat16, VecLen> sp_vec = aie::zeros<bfloat16, VecLen>();
    for (int row = 0; row < RowsPerBlock; row++) {
      bfloat16 spval = sp[row_start + row];
      for (int c = 0; c < ColsPerBlock; c++) {
        sp_vec[row * ColsPerBlock + c] = spval;
      }
    }

    int base = rb * BlockSize;
    for (int cb = 0; cb < col_blocks; cb++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        int off = base + cb * block_stride;
        aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(gp + off);
        v = aie::div(v, sp_vec);
        aie::store_v(gp + off, v);
      }
  }
}

// Fused softmax: delegates to existing kernels.
// On return: up=new_max, sp=sum(exp(G)), r=rescale_factor, G=exp(G-max).
void fused_softmax(bfloat16 *g, bfloat16 *up, bfloat16 *sp, bfloat16 *r) {
  scale_g_bf16(g);
  max_g_bf16(g, r);
  maximum_up_u_bf16(up, r);
  exp_g_minus_u(r, g);
  exp_up_minus_u(up, r, sp);
  vector_copy_32elems(0, r, up);
  vector_copy_32elems(0, sp, r);
  sum_g(g, sp);
}

void add_gp_g(bfloat16 *gp, bfloat16 *g) {
  constexpr int VecLen = 16;
  constexpr int num_elems = lqp * dv;
  bfloat16 *__restrict gp_ptr = gp;
  bfloat16 *__restrict g_ptr = g;
  for (unsigned j = 0; j < num_elems / VecLen; j++) {
    aie::vector<bfloat16, VecLen> gp_vec = aie::load_v<VecLen>(gp_ptr);
    aie::vector<bfloat16, VecLen> g_vec = aie::load_v<VecLen>(g_ptr);
    aie::accum<accfloat, VecLen> acc(gp_vec);
    acc = aie::add(acc, g_vec);
    aie::store_v(g_ptr, acc.to_vector<bfloat16>());
    gp_ptr += VecLen;
    g_ptr += VecLen;
  }
}

// Apply causal mask to QK scores in-place.
// G is column-major 4x4 block tiled.
// Uses scalar access since aie::vector<bf16,4> is not supported on AIE2.
void apply_causal_mask(bfloat16 *g, int32_t q_block_idx, int32_t kv_block_idx) {
  uint16_t neg_inf_u16 = (uint16_t)0xff80;
  bfloat16 neg_inf_val = *(bfloat16 *)&neg_inf_u16;

  // 1. Block above diagonal: all masked -> fill with -inf
  if (kv_block_idx > q_block_idx) {
    constexpr int VecLen = 16;
    aie::vector<bfloat16, VecLen> neg_inf_vec =
        aie::broadcast<bfloat16, VecLen>(neg_inf_val);
    bfloat16 *p = g;
    for (int i = 0; i < lqp * lkp; i += VecLen) {
      aie::store_v(p, neg_inf_vec);
      p += VecLen;
    }
    return;
  }

  // 2. Block below diagonal: no masking needed
  if (kv_block_idx < q_block_idx) {
    return;
  }

  // 3. Diagonal block (kv_block_idx == q_block_idx):
  // Use scalar writes for per-element causal masking.
  constexpr int BlkDim = 4;

  for (int row = 0; row < lqp; row++) {
    int mask_start = row + 1;
    int row_blk = row / BlkDim;
    int row_in = row % BlkDim;

    for (int col_blk = 0; col_blk < lkp / BlkDim; col_blk++) {
      int col_start = col_blk * BlkDim;
      int off = col_blk * (lqp * BlkDim) + row_blk * (BlkDim * BlkDim) +
                row_in * BlkDim;

      if (col_start >= mask_start) {
        // Entire sub-row masked
        for (int c = 0; c < BlkDim; c++) {
          g[off + c] = neg_inf_val;
        }
      } else if (col_start + BlkDim > mask_start) {
        // Partial: mask columns >= mask_start
        for (int c = 0; c < BlkDim; c++) {
          if (col_start + c >= mask_start) {
            g[off + c] = neg_inf_val;
          }
        }
      }
      // else: unmasked, leave unchanged
    }
  }
}

} // extern "C"
