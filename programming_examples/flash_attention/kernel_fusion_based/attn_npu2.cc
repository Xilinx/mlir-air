//===- attn_npu2.cc -----------------------------------------------*- C++ -*-===//
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

#ifndef dv_full
#define dv_full dv
#endif

// Column-major B matmul with compile-time transpose control.
// transpose_b: true  = apply aie::transpose before mac (K DMA: inner [n_in,
// k_in])
//              false = load B as-is, hardware mul_8x8_8x8T transposes (V DMA:
//              inner [k_in, n_in])
// A and C are always column-major tiled.
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool transpose_b = true>
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

              aie::vector<T_in, MMUL::size_B> B0, B1;
              if constexpr (transpose_b) {
                // K DMA k-major block layout: block (n=j, k=i) at i*colB+j.
                // Sub-tile elements are [n_in, k_in], transpose to [k_in,
                // n_in].
                const T_in *__restrict pBk0 =
                    pB + (i * colB + j) * MMUL::size_B;
                const T_in *__restrict pBk1 =
                    pB + (i * colB + (j + 1)) * MMUL::size_B;
                B0 = aie::transpose(aie::load_v<MMUL::size_B>(pBk0), t, s);
                B1 = aie::transpose(aie::load_v<MMUL::size_B>(pBk1), t, s);
              } else {
                // V DMA inner layout is [k_in, n_in] — already correct for
                // hardware mul_8x8_8x8T, no software transpose needed.
                B0 = aie::load_v<MMUL::size_B>(pB1);
                B1 = aie::load_v<MMUL::size_B>(pB2);
              }
              pB1 += MMUL::size_B;
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

// bf16 MatMul kernel with bf16 outputs.
// transpose_b: controls whether B blocks are software-transposed before mac.
template <unsigned m, unsigned k, unsigned n, bool transpose_b = true>
static inline void
matmul_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t, transpose_b>(pA, pB, pC);
}

// Combined scale: log2e / sqrt(dk_full). Applies 1/sqrt(dk) inside softmax
// with accfloat precision, avoiding bf16 truncation of Q.
// dk is the tile dimension (= lkp), dk_full is the full key dimension.
// When dk_full == dk (default), sqrt(64) = 8.0 — no change.
#include <cmath>

#ifndef dk_full
#define dk_full dk
#endif

constexpr double constexpr_sqrt_dk = (dk_full == 64)    ? 8.0
                                     : (dk_full == 128) ? 11.313708498984761
                                     : (dk_full == 256) ? 16.0
                                     : (dk_full == 512) ? 22.627416997969522
                                                        : 8.0;
static_assert(dk_full == 64 || dk_full == 128 || dk_full == 256 ||
                  dk_full == 512,
              "Unsupported dk_full value: update constexpr_sqrt_dk");

#define log2e (1.44269504089 / constexpr_sqrt_dk)

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

// Set rounding mode at the start of every extern C function.
// Setting conv_even rounding in every softmax function; without this,
// softmax intermediates use the system default rounding mode,
// causing ~44% errors at val_range=4 due to rounding noise
// amplified by softmax's peaked distribution.
#ifdef ROUND_CONV_EVEN
#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)
#else
#define SET_ROUNDING() /* no-op */
#endif

// Copy tile_size_q×dk elements from src to dst (single-pass vector copy)
void copy_tile(bfloat16 *src, bfloat16 *dst) {
  SET_ROUNDING();
  constexpr int VecLen = 32;
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
  SET_ROUNDING();
  // Buffer shapes:
  // A: [lqp, dk] = [32, 64]
  // B: [lkp, dk] = [96, 64]  (K row-major, aie::transpose per block)
  // Out: [lqp, lkp] = [32, 96]
  matmul_vectorized_8x8x8_bf16_bf16<lqp, dk, lkp>(a_in, b_in, out);
}

void matmul_g_b_bf16(bfloat16 *g_in, bfloat16 *b_in, bfloat16 *out) {
  SET_ROUNDING();
  // Buffer shapes:
  // G: [lqp, lkp] = [32, 96]
  // B: [lkp, dv] = [96, 64]
  // Out: [lqp, dv] = [32, 64]
  // G@V: V DMA inner layout is [k_in, n_in], so NO software transpose needed.
  // The hardware mul_8x8_8x8T already transposes B internally.
  matmul_vectorized_8x8x8_bf16_bf16<lqp, lkp, dv, /*transpose_b=*/false>(
      g_in, b_in, out);
}

void zero_fill_gp_bf16(bfloat16 *c_out) {
  SET_ROUNDING();
  // Buffer shape: [lqp, dv] = [32, 64]
  zero_vectorized<bfloat16, lqp, dv, 32>(c_out);
}

void zero_fill_sp_bf16(bfloat16 *c_out) {
  SET_ROUNDING();
  // Buffer shape: [lqp, 1] = [32, 1]
  zero_vectorized<bfloat16, lqp, 1, 32>(c_out);
}

void zero_fill_g_bf16(bfloat16 *c_out) {
  SET_ROUNDING();
  // Buffer shape: [lqp, lkp] = [32, 96]
  zero_vectorized<bfloat16, lqp, lkp, 32>(c_out);
}

void neg_inf_fill_up_bf16(bfloat16 *c_out) {
  SET_ROUNDING();
  // Buffer shape: [lqp, 1] = [32, 1]
  neg_inf_vectorized<bfloat16, lqp, 1, 32>(c_out);
}

void max_g_bf16(bfloat16 *in, bfloat16 *out) {
  SET_ROUNDING();
  // u = np.max(G, axis=-1, keepdims=True)
  // G is in column-major 8x8 tiled layout.
  // Each block is 64 contiguous elements (8 rows × 8 cols).
  // VecLen=32 reads 4 rows at once (half a block).
  constexpr int VecLen = 32;
  constexpr int BlockSize = 64; // 8×8 block
  constexpr int ColsPerBlock = 8;
  constexpr int RowsPerBlock = 8;
  constexpr int col_blocks = lkp / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride =
      lqp * ColsPerBlock; // stride between column blocks

  // Use bf16 lowest (0xff7f) instead of -inf (0xff80) as initial max value.
  // For fully-masked rows (all -inf), max returns bf16_lowest > -inf,
  // avoiding NaN in exp(G - u) where G=-inf and u would be -inf.
  uint16_t lowest_u16 = (uint16_t)0xff7f;
  bfloat16 lowest_val = *(bfloat16 *)&lowest_u16;

  bfloat16 *__restrict pOut = out;
  for (int rb = 0; rb < row_blocks; rb++) {
    // Process 4 rows at a time (half block = 32 elements)
    for (int half = 0; half < 2; half++) {
      aie::vector<bfloat16, VecLen> max_vec =
          aie::broadcast<bfloat16, VecLen>(lowest_val);
      int base = rb * BlockSize + half * VecLen;
      for (int cb = 0; cb < col_blocks; cb++)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          aie::vector<bfloat16, VecLen> v =
              aie::load_v<VecLen>(in + base + cb * block_stride);
          max_vec = aie::max(max_vec, v);
        }
      // Extract per-row max from 32-wide vector (4 rows × 8 cols)
      aie::vector<bfloat16, 8> r0 = max_vec.extract<8>(0);
      aie::vector<bfloat16, 8> r1 = max_vec.extract<8>(1);
      aie::vector<bfloat16, 8> r2 = max_vec.extract<8>(2);
      aie::vector<bfloat16, 8> r3 = max_vec.extract<8>(3);
      pOut[half * 4 + 0] = aie::reduce_max(r0);
      pOut[half * 4 + 1] = aie::reduce_max(r1);
      pOut[half * 4 + 2] = aie::reduce_max(r2);
      pOut[half * 4 + 3] = aie::reduce_max(r3);
    }
    pOut += RowsPerBlock;
  }
}

void maximum_up_u_bf16(bfloat16 *up, bfloat16 *u) {
  SET_ROUNDING();
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
  SET_ROUNDING();
  // G = exp(G - u) in-place. G is column-major 8×8 tiled.
  // VecLen=32 processes 4 rows at once (half a block).
  // exp2 native width is 16, so split 30→2×16 for exp.
  // With bf16 lowest (not -inf), lowest - lowest = 0 (not NaN).
  constexpr int VecLen = 32;
  constexpr int BlockSize = 64;
  constexpr int ColsPerBlock = 8;
  constexpr int RowsPerBlock = 8;
  constexpr int col_blocks = lkp / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride = lqp * ColsPerBlock;

  uint16_t lowest_u16 = (uint16_t)0xff7f;
  bfloat16 lowest_val = *(bfloat16 *)&lowest_u16;
  aie::vector<bfloat16, 16> log2e_vec16 =
      aie::broadcast<bfloat16, 16>((bfloat16)log2e);
  aie::vector<bfloat16, VecLen> lowest_vec =
      aie::broadcast<bfloat16, VecLen>(lowest_val);

  for (int rb = 0; rb < row_blocks; rb++) {
    for (int half = 0; half < 2; half++) {
      // Build 32-wide u vector: 4 rows × 8 cols, each row broadcast
      int row_start = rb * RowsPerBlock + half * 4;
      aie::vector<bfloat16, 8> u0 = aie::broadcast<bfloat16, 8>(u[row_start]);
      aie::vector<bfloat16, 8> u1 =
          aie::broadcast<bfloat16, 8>(u[row_start + 1]);
      aie::vector<bfloat16, 8> u2 =
          aie::broadcast<bfloat16, 8>(u[row_start + 2]);
      aie::vector<bfloat16, 8> u3 =
          aie::broadcast<bfloat16, 8>(u[row_start + 3]);
      aie::vector<bfloat16, VecLen> u_vec;
      u_vec.insert(0, u0);
      u_vec.insert(1, u1);
      u_vec.insert(2, u2);
      u_vec.insert(3, u3);

      int base = rb * BlockSize + half * VecLen;
      for (int cb = 0; cb < col_blocks; cb++)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          int off = base + cb * block_stride;
          aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(g + off);
          v = aie::sub(v, u_vec);
          v = aie::max(v, lowest_vec);
          // exp2(log2e * v) — split into 2×16 for native exp2 width
          aie::vector<bfloat16, 16> lo = v.extract<16>(0);
          aie::vector<bfloat16, 16> hi = v.extract<16>(1);
          lo =
              aie::exp2<bfloat16>(aie::mul(lo, log2e_vec16).to_vector<float>());
          hi =
              aie::exp2<bfloat16>(aie::mul(hi, log2e_vec16).to_vector<float>());
          v.insert(0, lo);
          v.insert(1, hi);
          aie::store_v(g + off, v);
        }
    }
  }
}

void exp_up_minus_u(bfloat16 *up, bfloat16 *u, bfloat16 *r) {
  SET_ROUNDING();
  // r = exp(up - u) — VecLen=16 to match exp2 native width
  // With bf16 lowest (not -inf), lowest - lowest = 0 (not NaN).
  constexpr int VecLen = 16;
  constexpr int num_elems = lqp;
  uint16_t lowest_u16 = (uint16_t)0xff7f;
  bfloat16 lowest_val = *(bfloat16 *)&lowest_u16;
  aie::vector<bfloat16, VecLen> lowest_vec =
      aie::broadcast<bfloat16, VecLen>(lowest_val);
  bfloat16 *__restrict pr = r;
  bfloat16 *__restrict pu = u;
  bfloat16 *__restrict pup = up;
  aie::vector<bfloat16, VecLen> log2e_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)log2e);
  for (int i = 0; i < num_elems; i += VecLen) {
    aie::vector<bfloat16, VecLen> uTemp = aie::load_v<VecLen>(pu);
    aie::vector<bfloat16, VecLen> upTemp = aie::load_v<VecLen>(pup);
    aie::vector<bfloat16, VecLen> diff = aie::sub(upTemp, uTemp);
    // Clamp extreme negative values
    diff = aie::max(diff, lowest_vec);
    aie::vector<bfloat16, VecLen> exp_val =
        aie::exp2<bfloat16>(aie::mul(diff, log2e_vec).to_vector<float>());
    aie::store_v(pr, exp_val);
    pr += VecLen;
    pu += VecLen;
    pup += VecLen;
  }
}

void mul_r_gp(bfloat16 *r, bfloat16 *gp) {
  SET_ROUNDING();
  // Gp = Gp * r (per-row scaling)
  // Buffer shape: Gp: [lqp, dv], r: [lqp, 1]
  // Layout: column-major 8×8 block tiled (same as matmul output).
  // block(col_blk, row_blk) at offset col_blk * (lqp * 8) + row_blk * 64,
  // element within block at row_in * 8 + col_in.
  // VecLen=32 reads 4 rows × 8 cols (half a block).
  constexpr int VecLen = 32;
  constexpr int BlockSize = 64; // 8×8 block
  constexpr int ColsPerBlock = 8;
  constexpr int RowsPerBlock = 8;
  constexpr int col_blocks = dv / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride =
      lqp * ColsPerBlock; // stride between column blocks

  for (int rb = 0; rb < row_blocks; rb++) {
    for (int half = 0; half < 2; half++) {
      // Build 32-wide r vector: 4 rows × 8 cols, each row's r broadcast to 8
      int row_start = rb * RowsPerBlock + half * 4;
      aie::vector<bfloat16, 8> r0 = aie::broadcast<bfloat16, 8>(r[row_start]);
      aie::vector<bfloat16, 8> r1 =
          aie::broadcast<bfloat16, 8>(r[row_start + 1]);
      aie::vector<bfloat16, 8> r2 =
          aie::broadcast<bfloat16, 8>(r[row_start + 2]);
      aie::vector<bfloat16, 8> r3 =
          aie::broadcast<bfloat16, 8>(r[row_start + 3]);
      aie::vector<bfloat16, VecLen> r_vec;
      r_vec.insert(0, r0);
      r_vec.insert(1, r1);
      r_vec.insert(2, r2);
      r_vec.insert(3, r3);

      int base = rb * BlockSize + half * VecLen;
      for (int cb = 0; cb < col_blocks; cb++)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          int off = base + cb * block_stride;
          aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(gp + off);
          aie::accum<accfloat, VecLen> acc = aie::mul(v, r_vec);
          aie::store_v(gp + off, acc.to_vector<bfloat16>());
        }
    }
  }
}

void sum_g(bfloat16 *g, bfloat16 *s) {
  SET_ROUNDING();
  // s = sum(G, axis=-1, keepdims=True)
  // G is column-major 8×8 tiled. VecLen=32 loads 4 rows at once.
  constexpr int VecLen = 32;
  constexpr int BlockSize = 64;
  constexpr int ColsPerBlock = 8;
  constexpr int RowsPerBlock = 8;
  constexpr int col_blocks = lkp / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride = lqp * ColsPerBlock;

  bfloat16 *__restrict ps = s;
  for (int rb = 0; rb < row_blocks; rb++) {
    for (int half = 0; half < 2; half++) {
      // Accumulate sum across column blocks for 4 rows
      aie::accum<accfloat, VecLen> sum_acc = aie::zeros<accfloat, VecLen>();
      int base = rb * BlockSize + half * VecLen;
      for (int cb = 0; cb < col_blocks; cb++)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          aie::vector<bfloat16, VecLen> v =
              aie::load_v<VecLen>(g + base + cb * block_stride);
          sum_acc = aie::add(sum_acc, v);
        }
      // Reduce each 8-element row slice to get per-row sum.
      // Use f32 for reduce_add to preserve precision.
      aie::vector<float, VecLen> sum_v = sum_acc.to_vector<float>();
      aie::vector<float, 8> r0 = sum_v.extract<8>(0);
      aie::vector<float, 8> r1 = sum_v.extract<8>(1);
      aie::vector<float, 8> r2 = sum_v.extract<8>(2);
      aie::vector<float, 8> r3 = sum_v.extract<8>(3);
      ps[half * 4 + 0] = (bfloat16)aie::reduce_add(r0);
      ps[half * 4 + 1] = (bfloat16)aie::reduce_add(r1);
      ps[half * 4 + 2] = (bfloat16)aie::reduce_add(r2);
      ps[half * 4 + 3] = (bfloat16)aie::reduce_add(r3);
    }
    ps += RowsPerBlock;
  }
}

void accum_sp_r_s(bfloat16 *sp, bfloat16 *r, bfloat16 *s) {
  SET_ROUNDING();
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
  SET_ROUNDING();
  // Gp = Gp / sp (per-row normalization)
  // Buffer shape: Gp: [lqp, dv], sp: [lqp, 1]
  // Layout: column-major 8×8 block tiled (same as matmul output).
  // block(col_blk, row_blk) at offset col_blk * (lqp * 8) + row_blk * 64,
  // element within block at row_in * 8 + col_in.
  // VecLen=32 reads 4 rows × 8 cols (half a block).
  constexpr int VecLen = 32;
  constexpr int BlockSize = 64; // 8×8 block
  constexpr int ColsPerBlock = 8;
  constexpr int RowsPerBlock = 8;
  constexpr int col_blocks = dv / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride =
      lqp * ColsPerBlock; // stride between column blocks

  for (int rb = 0; rb < row_blocks; rb++) {
    for (int half = 0; half < 2; half++) {
      // Build 32-wide 1/sp vector: 4 rows × 8 cols, each row's inv(sp)
      // broadcast
      int row_start = rb * RowsPerBlock + half * 4;
      aie::vector<bfloat16, 8> sp0 = aie::broadcast<bfloat16, 8>(sp[row_start]);
      aie::vector<bfloat16, 8> sp1 =
          aie::broadcast<bfloat16, 8>(sp[row_start + 1]);
      aie::vector<bfloat16, 8> sp2 =
          aie::broadcast<bfloat16, 8>(sp[row_start + 2]);
      aie::vector<bfloat16, 8> sp3 =
          aie::broadcast<bfloat16, 8>(sp[row_start + 3]);
      aie::vector<bfloat16, VecLen> sp_vec;
      sp_vec.insert(0, sp0);
      sp_vec.insert(1, sp1);
      sp_vec.insert(2, sp2);
      sp_vec.insert(3, sp3);
      aie::vector<bfloat16, VecLen> sp_inv = aie::inv(sp_vec);

      int base = rb * BlockSize + half * VecLen;
      for (int cb = 0; cb < col_blocks; cb++)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          int off = base + cb * block_stride;
          aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(gp + off);
          aie::accum<accfloat, VecLen> acc = aie::mul(v, sp_inv);
          aie::store_v(gp + off, acc.to_vector<bfloat16>());
        }
    }
  }
}

// Fused softmax: delegates to existing optimized VecLen=32 kernels.
// On return: up=new_max, sp=sum(exp(G)), r=rescale_factor, G=exp(G-max).
void fused_softmax(bfloat16 *g, bfloat16 *up, bfloat16 *sp, bfloat16 *r) {
  SET_ROUNDING();
  max_g_bf16(g, r);
  maximum_up_u_bf16(up, r);
  exp_g_minus_u(r, g);
  exp_up_minus_u(up, r, sp);
  vector_copy_32elems(0, r, up);
  vector_copy_32elems(0, sp, r);
  sum_g(g, sp);
}

void add_gp_g(bfloat16 *gp, bfloat16 *g) {
  SET_ROUNDING();
  constexpr int VecLen = 32;
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

// Apply causal mask to QK scores in-place. Sets elements where
// global_kv_col > global_q_row to -inf in the tiled G buffer.
// G is in column-major 8×8 tiled layout: block(col_blk, row_blk) at
// offset col_blk * (lqp * 8) + row_blk * 64, element within block at
// row_in_blk * 8 + col_in_blk.
void apply_causal_mask(bfloat16 *g, int32_t q_block_idx, int32_t kv_block_idx) {
  SET_ROUNDING();
  uint16_t neg_inf_u16 = (uint16_t)0xff80;
  bfloat16 neg_inf_val = *(bfloat16 *)&neg_inf_u16;

  // 1. Block above diagonal: all masked -> fill with -inf
  if (kv_block_idx > q_block_idx) {
    constexpr int VecLen = 32;
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
  // Read-modify-write ALL 8-element row slices for EVERY row.
  // For unmasked blocks: read and write back unchanged.
  // For masked blocks: write mask value.
  // For partial blocks: read, select, write back.
  // This ensures EVERY position goes through a vector load+store cycle.
  constexpr int BlkDim = 8;
  aie::vector<bfloat16, BlkDim> mask_vec =
      aie::broadcast<bfloat16, BlkDim>(neg_inf_val);

  for (int row = 0; row < lqp; row++) {
    int mask_start = row + 1;
    int row_blk = row / BlkDim;
    int row_in = row % BlkDim;

    for (int col_blk = 0; col_blk < lkp / BlkDim; col_blk++) {
      int col_start = col_blk * BlkDim;
      int off = col_blk * (lqp * BlkDim) + row_blk * (BlkDim * BlkDim) +
                row_in * BlkDim;

      aie::vector<bfloat16, BlkDim> orig = aie::load_v<BlkDim>(g + off);

      if (mask_start >= lkp) {
        // Last row or beyond: no masking, write back unchanged
        aie::store_v(g + off, orig);
      } else if (col_start >= mask_start) {
        // Entire block masked
        aie::store_v(g + off, mask_vec);
      } else if (col_start + BlkDim > mask_start) {
        // Partial block
        uint32_t sel_bits = 0;
        for (int c = 0; c < BlkDim; c++) {
          if (col_start + c >= mask_start) {
            sel_bits |= (1u << c);
          }
        }
        aie::mask<BlkDim> sel(sel_bits);
        aie::store_v(g + off, aie::select(orig, mask_vec, sel));
      } else {
        // Unmasked block: write back unchanged
        aie::store_v(g + off, orig);
      }
    }
  }
}

} // extern "C"
