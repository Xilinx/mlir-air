//===- gelu_and_mul.cc - GELU + elementwise multiply kernel -*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Element-wise GELU (tanh approximation) followed by multiply:
//   output[i] = GELU(gate[i]) * up[i]
//   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// This is `gelu_pytorch_tanh` -- the GLU activation Gemma3 uses where Llama
// uses SiLU. Structured exactly like silu_and_mul.cc (same signature, same
// vector loop) so the two are drop-in alternatives in a GLU stitcher.
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

void gelu_and_mul_bf16(bfloat16 *gate, bfloat16 *up, bfloat16 *out, int32_t n) {
  constexpr int VecLen = 16;
  aie::vector<bfloat16, VecLen> half_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.5f);
  aie::vector<bfloat16, VecLen> one_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)1.0f);
  // sqrt(2/pi) and the cubic coefficient of the tanh approximation.
  aie::vector<bfloat16, VecLen> c_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.7978845608f);
  aie::vector<bfloat16, VecLen> beta_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.044715f);

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

    // inner = sqrt(2/pi) * (g + 0.044715 * g^3)
    aie::vector<bfloat16, VecLen> g2 = aie::mul(g, g);
    aie::vector<bfloat16, VecLen> g3 = aie::mul(g2, g);
    aie::vector<bfloat16, VecLen> beta_g3 = aie::mul(beta_vec, g3);
    aie::vector<bfloat16, VecLen> poly = aie::add(g, beta_g3);
    aie::vector<bfloat16, VecLen> inner = aie::mul(c_vec, poly);

    aie::accum<accfloat, VecLen> tanh_in;
    tanh_in.from_vector(inner);
    aie::vector<bfloat16, VecLen> tanh_val =
        aie::tanh<bfloat16>(tanh_in.to_vector<float>());

    aie::vector<bfloat16, VecLen> one_plus_tanh = aie::add(one_vec, tanh_val);
    aie::vector<bfloat16, VecLen> g_half = aie::mul(half_vec, g);
    aie::vector<bfloat16, VecLen> gelu = aie::mul(g_half, one_plus_tanh);
    aie::vector<bfloat16, VecLen> result = aie::mul(gelu, u);

    aie::store_v(pO, result);
    pO += VecLen;
  }
}

} // extern "C"
