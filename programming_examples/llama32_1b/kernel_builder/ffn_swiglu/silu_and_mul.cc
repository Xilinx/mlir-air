//===- silu_and_mul.cc - SiLU + elementwise multiply kernel -*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Element-wise SiLU activation followed by multiply:
//   output[i] = SiLU(gate[i]) * up[i]
//   SiLU(x) = x * sigmoid(x) = x * 0.5 * (tanh(x/2) + 1)
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

extern "C" {

void silu_and_mul_bf16(bfloat16 *gate, bfloat16 *up, bfloat16 *out, int32_t n) {
  constexpr int VecLen = 16;
  aie::vector<bfloat16, VecLen> half_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.5f);
  aie::vector<bfloat16, VecLen> one_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)1.0f);

  bfloat16 *__restrict pG = gate;
  bfloat16 *__restrict pU = up;
  bfloat16 *__restrict pO = out;
  const int F = n / VecLen;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<bfloat16, VecLen> g = aie::load_v<VecLen>(pG);
    pG += VecLen;
    aie::vector<bfloat16, VecLen> u = aie::load_v<VecLen>(pU);
    pU += VecLen;

    aie::vector<bfloat16, VecLen> g_half = aie::mul(g, half_vec);
    aie::accum<accfloat, VecLen> tanh_in;
    tanh_in.from_vector(g_half);
    aie::vector<bfloat16, VecLen> tanh_val =
        aie::tanh<bfloat16>(tanh_in.to_vector<float>());
    aie::vector<bfloat16, VecLen> one_plus_tanh = aie::add(one_vec, tanh_val);
    aie::vector<bfloat16, VecLen> sigmoid = aie::mul(half_vec, one_plus_tanh);
    aie::vector<bfloat16, VecLen> silu = aie::mul(g, sigmoid);
    aie::vector<bfloat16, VecLen> result = aie::mul(silu, u);

    aie::store_v(pO, result);
    pO += VecLen;
  }
}

} // extern "C"
