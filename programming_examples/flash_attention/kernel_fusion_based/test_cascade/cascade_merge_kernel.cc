//===- cascade_merge_kernel.cc - Cascade merge test kernels ------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
// Kernel functions for testing the flash attention cascade merge logic
// in isolation. Merge functions are copied verbatim from attn.cc.
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

#include "../zero.cc"

// Dimensions matching flash attention with tile_size_q=64, lkp=dk=dv=64
#ifndef lqp
#define lqp 64
#endif

#ifndef lkp
#define lkp 64
#endif

#ifndef dk
#define dk 64
#endif

#ifndef dv
#define dv 64
#endif

// Combined scale: log2e / sqrt(dk) — same as attn.cc
#include <cmath>
#define log2e (1.44269504089 / constexpr_sqrt_dk)
constexpr double constexpr_sqrt_dk = 8.0; // sqrt(64)

extern "C" {

#ifdef ROUND_CONV_EVEN
#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)
#else
#define SET_ROUNDING() /* no-op */
#endif

// ========================================================================
// Test-specific fill functions
// ========================================================================

// Fill Gp buffer [lqp * dv] with a uniform bf16 value
void fill_gp_uniform(bfloat16 *buf, int32_t val) {
  SET_ROUNDING();
  constexpr int VecLen = 32;
  constexpr int num_elems = lqp * dv;
  bfloat16 fill_val = (bfloat16)((float)val);
  aie::vector<bfloat16, VecLen> v =
      aie::broadcast<bfloat16, VecLen>(fill_val);
  bfloat16 *__restrict p = buf;
  for (int i = 0; i < num_elems / VecLen; i++)
    chess_prepare_for_pipelining {
      aie::store_v(p, v);
      p += VecLen;
    }
}

// Fill sp/up buffer [lqp] with a uniform bf16 value
void fill_sp_uniform(bfloat16 *buf, int32_t val) {
  SET_ROUNDING();
  constexpr int VecLen = 32;
  constexpr int num_elems = lqp;
  bfloat16 fill_val = (bfloat16)((float)val);
  aie::vector<bfloat16, VecLen> v =
      aie::broadcast<bfloat16, VecLen>(fill_val);
  bfloat16 *__restrict p = buf;
  for (int i = 0; i < num_elems / VecLen; i++)
    chess_prepare_for_pipelining {
      aie::store_v(p, v);
      p += VecLen;
    }
}

// ========================================================================
// Functions from attn.cc — copied verbatim
// ========================================================================

void zero_fill_gp_bf16(bfloat16 *c_out) {
  SET_ROUNDING();
  zero_vectorized<bfloat16, lqp, dv, 32>(c_out);
}

void zero_fill_sp_bf16(bfloat16 *c_out) {
  SET_ROUNDING();
  zero_vectorized<bfloat16, lqp, 1, 32>(c_out);
}

void neg_inf_fill_up_bf16(bfloat16 *c_out) {
  SET_ROUNDING();
  neg_inf_vectorized<bfloat16, lqp, 1, 32>(c_out);
}

void maximum_up_u_bf16(bfloat16 *up, bfloat16 *u) {
  SET_ROUNDING();
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

void exp_up_minus_u(bfloat16 *up, bfloat16 *u, bfloat16 *r) {
  SET_ROUNDING();
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
  constexpr int VecLen = 32;
  constexpr int BlockSize = 64;
  constexpr int ColsPerBlock = 8;
  constexpr int RowsPerBlock = 8;
  constexpr int col_blocks = dv / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride = lqp * ColsPerBlock;

  for (int rb = 0; rb < row_blocks; rb++) {
    for (int half = 0; half < 2; half++) {
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

void accum_sp_r_s(bfloat16 *sp, bfloat16 *r, bfloat16 *s) {
  SET_ROUNDING();
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
  constexpr int VecLen = 32;
  constexpr int BlockSize = 64;
  constexpr int ColsPerBlock = 8;
  constexpr int RowsPerBlock = 8;
  constexpr int col_blocks = dv / ColsPerBlock;
  constexpr int row_blocks = lqp / RowsPerBlock;
  constexpr int block_stride = lqp * ColsPerBlock;

  for (int rb = 0; rb < row_blocks; rb++) {
    for (int half = 0; half < 2; half++) {
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

} // extern "C"
