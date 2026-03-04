// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// AWQ-style int4 → bfloat16 dequantization kernel.
// Unpacks int4 values from packed uint8 pairs, applies per-group
// scale and zero-point: output = (int4_val - zero_point) * scale.
//
// Input layout:
//   weights: uint8[N/2] — two int4 values packed per byte (low nibble first)
//   params:  bfloat16[2*N/GROUP_SIZE] — interleaved [scale0, zero0, scale1,
//   zero1, ...]
// Output:
//   output:  bfloat16[N] — dequantized values

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_N
#define DIM_N 1024
#endif

#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

extern "C" {

void dequant_int4_bf16(uint8_t *__restrict weights, bfloat16 *__restrict params,
                       bfloat16 *__restrict output) {
  for (unsigned i = 0; i < DIM_N; i += 2) {
    uint8_t packed = weights[i / 2];
    int low = packed & 0x0F;
    int high = (packed >> 4) & 0x0F;

    unsigned g_low = i / GROUP_SIZE;
    unsigned g_high = (i + 1) / GROUP_SIZE;

    float s_low = (float)params[2 * g_low];
    float z_low = (float)params[2 * g_low + 1];
    float s_high = (float)params[2 * g_high];
    float z_high = (float)params[2 * g_high + 1];

    output[i] = (bfloat16)(((float)low - z_low) * s_low);
    output[i + 1] = (bfloat16)(((float)high - z_high) * s_high);
  }
}

} // extern "C"
