//===- extern_func.cc -------------------------------------------*- C++ -*-===//
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

#include "lut_based_ops.h"
#include <aie_api/aie.hpp>

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

  // if constexpr(maxsub_en == 1){
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

extern "C" {

void getExpBf16(const bfloat16 *__restrict in, bfloat16 *__restrict out) {
  const bfloat16 *__restrict pIn = in;
  bfloat16 *__restrict pExp = out;

  float accum_exp_val =
      exp_bf16<16>(16, pIn, pExp); // vector size 16, reduction length 256
}

} // extern "C"
