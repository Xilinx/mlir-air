//===- rope_halfsplit.cc - Half-split RoPE kernel for AIE2P -----*- C++ -*-===//
//
// Half-split Rotary Position Embedding matching HuggingFace Llama convention.
// Pairs (x[i], x[i + dim/2]) with rotation angle theta_i.
//
// LUT layout: [cos_0, cos_1, ..., cos_{half-1}, sin_0, sin_1, ...,
// sin_{half-1}]
//   (first half = cos values, second half = sin values)
//
// Rotation formula:
//   out[i]        = x[i] * cos[i] - x[i + half] * sin[i]
//   out[i + half] = x[i] * sin[i] + x[i + half] * cos[i]
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, int N>
void rope_halfsplit_kernel(const T *restrict input, const T *restrict lut,
                           T *restrict output, int32_t dims) {
  event0();

  const int half = dims / 2;

  for (int v = 0; v < half; v += N) {
    // Load first-half and second-half elements
    ::aie::vector<T, N> x1 = ::aie::load_v<N>(input + v);
    ::aie::vector<T, N> x2 = ::aie::load_v<N>(input + v + half);

    // Load cos and sin from concatenated LUT
    ::aie::vector<T, N> cos_v = ::aie::load_v<N>(lut + v);
    ::aie::vector<T, N> sin_v = ::aie::load_v<N>(lut + v + half);

    // out[i]        = x1[i] * cos[i] - x2[i] * sin[i]
    // out[i + half] = x1[i] * sin[i] + x2[i] * cos[i]
    ::aie::vector<T, N> out1 =
        ::aie::sub(::aie::mul(x1, cos_v), ::aie::mul(x2, sin_v));
    ::aie::vector<T, N> out2 =
        ::aie::add(::aie::mul(x1, sin_v), ::aie::mul(x2, cos_v));

    ::aie::store_v(output + v, out1);
    ::aie::store_v(output + v + half, out2);
  }
  event1();
}

extern "C" {
void rope(bfloat16 *input, bfloat16 *lut, bfloat16 *output, int32_t dims) {
  rope_halfsplit_kernel<bfloat16, 16>(input, lut, output, dims);
}
}
