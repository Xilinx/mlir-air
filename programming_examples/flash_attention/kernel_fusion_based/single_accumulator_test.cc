// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Exercise the production template with independent carried-output contracts.
#include "attn_npu2.cc"

extern "C" void attention_contract(bfloat16 *input, bfloat16 *output,
                                   int mode) {
  SET_ROUNDING();
  constexpr int size = 32 * 32;
  for (int i = 0; i < size; i += 32)
    aie::store_v(output + i, aie::load_v<32>(input + 2 * size + i));
  auto *a = input;
  auto *b = input + size;
  auto *rows = input + 3 * size;
  if (mode == 0)
    matmul_vectorized_8x8x8_bf16_bf16<32, 32, 32>(a, b, output);
  else if (mode == 1)
    matmul_vectorized_8x8x8_bf16_bf16<32, 32, 32, true, true>(a, b, output);
  else if (mode == 2)
    matmul_vectorized_8x8x8_bf16_bf16<32, 32, 32, true, true, true>(a, b,
                                                                    output);
  else
    matmul_vectorized_8x8x8_bf16_bf16<32, 32, 32, false, false, false, true>(
        a, b, output, rows);
}
