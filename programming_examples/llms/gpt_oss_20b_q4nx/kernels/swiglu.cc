// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GPT-OSS clamped SwiGLU, transcribed from FastFlowLM
// FLM_Xclbin/GPTOSS/gpt_npu_bin/kernel/gelu.cc (pseduo_gelu): gate clamped to
// <= 7, up to [-7, 7], y = (up + 1) * glu(gate) with glu(x) = x*sigmoid(1.702x)
// from a linear-approximation LUT. FLM's gelu() owns the slice loop and locks;
// here AIR calls one slice at a time.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

namespace {

constexpr float MLP_LIMIT = 7.0;
constexpr int GELU_SLICE = 2 * 32 * 4 * 4; // gate[512] then up[512]

template <int L>
void pseduo_gelu(bfloat16 *y, const bfloat16 *x) {
  const bfloat16 *gate_ptr = x;
  const bfloat16 *up_ptr = x + (L / 2);
  static_assert(L % 16 == 0);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < L / 2; i += 16) {
    aie::vector<bfloat16, 16> _gate_vec = aie::load_v<16>(gate_ptr + i);
    aie::vector<bfloat16, 16> _up_vec = aie::load_v<16>(up_ptr + i);
    aie::vector<bfloat16, 16> gate_max_vec =
        aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(MLP_LIMIT));
    aie::vector<bfloat16, 16> up_min_vec =
        aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(-MLP_LIMIT));
    aie::vector<bfloat16, 16> up_max_vec =
        aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(MLP_LIMIT));
    aie::vector<bfloat16, 16> gate_vec_clamp =
        aie::clamp(_gate_vec, _gate_vec, gate_max_vec);
    aie::vector<bfloat16, 16> up_vec_clamp =
        aie::clamp(_up_vec, up_min_vec, up_max_vec);
    aie::vector<bfloat16, 16> glu = getCustomSILU(gate_vec_clamp);
    aie::accum<accfloat, 16> glu_acc;
    glu_acc.from_vector(glu);
    glu_acc = aie::mac(glu_acc, up_vec_clamp, glu);
    aie::store_v(y + i, glu_acc.to_vector<bfloat16>());
  }
}

} // namespace

extern "C" {

// y[512] = swiglu(gate = x[0..512), up = x[512..1024))
void swiglu_slice(bfloat16 *__restrict x, bfloat16 *__restrict y) {
  pseduo_gelu<GELU_SLICE>(y, x);
}

} // extern "C"
