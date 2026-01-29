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

// Vectorized exponential function: takes float32 inputs/outputs, internally
// converts to bfloat16 for efficiency. Computes exp(x) using parallel table
// lookups with integer/fractional part decomposition. Lookup tables
// (softmax_ilut_ab/cd, softmax_flut_ab/cd) defined in lut_based_ops.h provide
// pre-computed exponential values.
//
// NOTE: This function assumes the input has already been normalized (max
// subtracted) by the calling pipeline. It only performs the exp() computation.
template <unsigned VecLen, typename VecIteratorIn, typename VecIteratorOut>
float exp_f32(int num_elems, VecIteratorIn &in, VecIteratorOut &out) {
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
  aie::vector<bfloat16, VecLen> I_val_vec, F_val_vec, input_bf16;
  aie::accum<accfloat, VecLen> exp_val_accum;
  exp_val_accum = aie::zeros<accfloat, VecLen>();
  aie::vector<int16, VecLen> input;
  aie::vector<int16, 2 * VecLen> input0;

  lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
  lut_type lut_f(LUT_elems, flut_ab, flut_cd);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_i(lut_i, step_i);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_f(lut_f, step_f);
  aie::accum<accfloat, VecLen> exp_val;

  // Main computation loop: convert to bf16, lookup exp values, store results
  for (int i = 0; i < elem_iters; i++) {
    // Load input (already normalized: x - max)
    aie::vector<float, VecLen> input_f32 = aie::load_v<VecLen>(in);
    in += VecLen;

    // Convert to bfloat16 for LUT indexing
    input_bf16 = to_v16bfloat16(input_f32);
    input0 = v32int16(bfloat16_to_int(input_bf16, SM_SCALE_FAC));
#ifndef SM_USE_MSB
    input = filter_even(input0);
#else
    input = filter_odd(input0);
#endif

    // Parallel LUT lookups for integer and fractional parts
    I_val_vec = lookup_i.fetch(input.template cast_to<uint16>());
    F_val_vec = lookup_f.fetch(input.template cast_to<uint16>());

    // exp(x) = exp(integer_part) * exp(fractional_part)
    exp_val = aie::mul(I_val_vec, F_val_vec);
    exp_val_accum = add(exp_val_accum, exp_val);

    // Store result as float32
    aie::store_v(out, exp_val.template to_vector<float>());
    out += VecLen;
  }

  // Return sum of all exp values (useful for softmax denominator)
  aie::vector<float, VecLen> reduce = exp_val_accum.template to_vector<float>();
  float res = aie::reduce_add(reduce);
  return res;
}

extern "C" {

void exp_vec16_f32(const float *__restrict in, float *__restrict out) {
  const float *__restrict pIn = in;
  float *__restrict pExp = out;

  float accum_exp_val =
      exp_f32<16>(256, pIn, pExp); // vector size 16, reduction length 256
}

} // extern "C"
