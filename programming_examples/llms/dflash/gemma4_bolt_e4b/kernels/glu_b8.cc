// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The GLU core of the Gemma4 eight-token decoder: dflash's batched GLU and
// LM-head drain, plus the per-layer embedding's GELU(gate) * per-layer input.

#include "llms/dflash/common/kernels/glu_batch.cc"

// y may be pli: each element is read before it is written.
extern "C" void ple_act_aie(bf16 *y, bf16 *__restrict gate, bf16 *pli, int n) {
  aie_round_nearest_even();
  for (int i = 0; i < n; i += 16) {
    aie::vector<bf16, 16> g = getActivationBf16(aie::load_v<16>(gate + i));
    aie::store_v(y + i,
                 aie::mul(g, aie::load_v<16>(pli + i)).to_vector<bf16>());
  }
}
