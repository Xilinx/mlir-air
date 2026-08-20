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

// Partial rotary (Phi-4: partial_rotary_factor=0.75 -> rope_dims=96 of
// dims=128). Rotates the leading rope_dims (halves rope_dims/2 apart) and
// copies the trailing dims-rope_dims through untouched, matching HF Phi3's
//   q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
// The LUT row stays `dims` wide -- [cos(rope_dims/2) | sin(rope_dims/2) |
// unused] -- so the caller's DMA shapes and row offsets are identical to the
// full-rotary path.
template <typename T, int N>
void rope_partial_kernel(const T *restrict input, const T *restrict lut,
                         T *restrict output, int32_t dims, int32_t rope_dims) {
  event0();

  const int half = rope_dims / 2;

  for (int v = 0; v < half; v += N) {
    ::aie::vector<T, N> x1 = ::aie::load_v<N>(input + v);
    ::aie::vector<T, N> x2 = ::aie::load_v<N>(input + v + half);
    ::aie::vector<T, N> cos_v = ::aie::load_v<N>(lut + v);
    ::aie::vector<T, N> sin_v = ::aie::load_v<N>(lut + v + half);

    ::aie::vector<T, N> out1 =
        ::aie::sub(::aie::mul(x1, cos_v), ::aie::mul(x2, sin_v));
    ::aie::vector<T, N> out2 =
        ::aie::add(::aie::mul(x1, sin_v), ::aie::mul(x2, cos_v));

    ::aie::store_v(output + v, out1);
    ::aie::store_v(output + v + half, out2);
  }

  // Pass-through tail: dims-rope_dims elements copied verbatim.
  for (int v = rope_dims; v < dims; v += N) {
    ::aie::store_v(output + v, ::aie::load_v<N>(input + v));
  }
  event1();
}

extern "C" {
void rope(bfloat16 *input, bfloat16 *lut, bfloat16 *output, int32_t dims) {
  rope_halfsplit_kernel<bfloat16, 16>(input, lut, output, dims);
}

void rope_partial(bfloat16 *input, bfloat16 *lut, bfloat16 *output,
                  int32_t dims, int32_t rope_dims) {
  rope_partial_kernel<bfloat16, 16>(input, lut, output, dims, rope_dims);
}
}
