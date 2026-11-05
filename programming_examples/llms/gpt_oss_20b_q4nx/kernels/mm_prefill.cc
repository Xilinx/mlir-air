// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GPT-OSS prefill GEMM core, transcribed from FastFlowLM
// FLM_Xclbin/MM/host7_mt_variant_bias (mm_64_128_128_4_8_8 col-major,
// custom_mm_behavior.hpp): bf16 mmul two by two into an f32 tile, bias seeded
// into the tile, and the GPT-OSS output modes on the way out. FLM's kernel owns
// the tile loop and locks; here AIR calls one step at a time. FLM's 4x8x8 mmul
// is 8x8x8 here: the bfp16 emulation computes eight rows of A either way.
//
// Layouts (r, s, t = 8, 8, 8; KC = 32 rows of K per step):
//   a    [MM/r][KC/s][r][s]
//   b    [N/t][KC/s][s][t]
//   acc  [MM/r][N/t][r][t] f32
//   out  [r][N] row-major bf16, one row block per call

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

#ifndef MM_M
#define MM_M 64
#endif

namespace {

constexpr int R = 8, S = 8, T = 8, KC = 32, N = 128;
constexpr int MB = MM_M / R, KB = KC / S, NB = N / T;
using MMUL = aie::mmul<R, S, T, bfloat16, bfloat16, accauto>;

template <int V> aie::vector<bfloat16, V> sigmoid(aie::vector<bfloat16, V> x) {
  auto half = aie::broadcast<bfloat16, V>(0.5f);
  auto one = aie::broadcast<bfloat16, V>(1.0f);
  aie::accum<accfloat, V> xh = aie::mul(x, half);
  aie::vector<bfloat16, V> th =
      aie::tanh<bfloat16>(xh.template to_vector<float>());
  return aie::mul(aie::add(th, one), half);
}

// 0: cast. 1 (gate): clamp <= 7, x * sigmoid(1.702 x). 2 (up): clamp to
// [-7, 7], + 1.
template <int V>
aie::vector<bfloat16, V> output_mode(aie::vector<bfloat16, V> x, int mode) {
  auto lim = aie::broadcast<bfloat16, V>(7.0f);
  if (mode == 1) {
    x = aie::clamp(x, aie::broadcast<bfloat16, V>(-3.30e+38f), lim);
    auto k = aie::broadcast<bfloat16, V>(1.702f);
    return aie::mul(x, sigmoid<V>(aie::mul(x, k)));
  }
  if (mode == 2) {
    x = aie::clamp(x, aie::broadcast<bfloat16, V>(-7.0f), lim);
    return aie::add(x, aie::broadcast<bfloat16, V>(1.0f));
  }
  return x;
}

} // namespace

extern "C" {

// acc = bias broadcast down the rows, or 0
void mmp_init(bfloat16 *__restrict bias, float *__restrict acc,
              int32_t use_bias) {
  for (int j = 0; j < NB; j++) {
    aie::vector<float, T> b = aie::zeros<float, T>();
    if (use_bias) {
      aie::accum<accfloat, T> ba;
      ba.from_vector(aie::load_v<T>(bias + j * T));
      b = ba.template to_vector<float>();
    }
    aie::vector<float, R * T> rows = aie::concat(b, b, b, b, b, b, b, b);
    for (int i = 0; i < MB; i++)
      aie::store_v(acc + (i * NB + j) * R * T, rows);
  }
}

// acc += a * b over one KC-deep step
void mmp_step(bfloat16 *__restrict a, bfloat16 *__restrict b,
              float *__restrict acc) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MAX_ITERATION_COUNT(MB / 2)
  for (int z = 0; z < MB; z += 2) {
    float *__restrict c1 = acc + (z * NB) * MMUL::size_C;
    float *__restrict c2 = acc + ((z + 1) * NB) * MMUL::size_C;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MAX_ITERATION_COUNT(NB / 2)
    for (int j = 0; j < NB; j += 2)
      AIE_LOOP_FLATTEN {
        const bfloat16 *__restrict a1 = a + z * KB * MMUL::size_A;
        const bfloat16 *__restrict a2 = a1 + KB * MMUL::size_A;
        const bfloat16 *__restrict b1 = b + j * KB * MMUL::size_B;
        const bfloat16 *__restrict b2 = b1 + KB * MMUL::size_B;
        MMUL c00(aie::load_v<MMUL::size_C>(c1 + j * MMUL::size_C));
        MMUL c01(aie::load_v<MMUL::size_C>(c1 + (j + 1) * MMUL::size_C));
        MMUL c10(aie::load_v<MMUL::size_C>(c2 + j * MMUL::size_C));
        MMUL c11(aie::load_v<MMUL::size_C>(c2 + (j + 1) * MMUL::size_C));
        for (int i = 0; i < KB; i++) {
          auto a0 = aie::load_v<MMUL::size_A>(a1 + i * MMUL::size_A);
          auto a1v = aie::load_v<MMUL::size_A>(a2 + i * MMUL::size_A);
          auto b0 = aie::load_v<MMUL::size_B>(b1 + i * MMUL::size_B);
          auto b1v = aie::load_v<MMUL::size_B>(b2 + i * MMUL::size_B);
          c00.mac(a0, b0);
          c01.mac(a0, b1v);
          c10.mac(a1v, b0);
          c11.mac(a1v, b1v);
        }
        aie::store_v(c1 + j * MMUL::size_C, c00.template to_vector<float>());
        aie::store_v(c1 + (j + 1) * MMUL::size_C,
                     c01.template to_vector<float>());
        aie::store_v(c2 + j * MMUL::size_C, c10.template to_vector<float>());
        aie::store_v(c2 + (j + 1) * MMUL::size_C,
                     c11.template to_vector<float>());
      }
  }
}

// out = mode(acc row block rb), row-major
void mmp_out(float *__restrict acc, bfloat16 *__restrict out, int32_t rb,
             int32_t mode) {
  const float *src = acc + rb * NB * R * T;
  for (int j = 0; j < NB; j++) {
    // 16 lanes at a time: a wider accfloat -> bf16 conversion repeats its
    // upper half
    for (int h = 0; h < R * T / 16; h++) {
      aie::accum<accfloat, 16> v;
      v.from_vector(aie::load_v<16>(src + j * R * T + h * 16));
      aie::vector<bfloat16, 16> y =
          output_mode<16>(v.template to_vector<bfloat16>(), mode);
      aie::store_v(out + (2 * h) * N + j * T, y.template extract<T>(0));
      aie::store_v(out + (2 * h + 1) * N + j * T, y.template extract<T>(1));
    }
  }
}

} // extern "C"
