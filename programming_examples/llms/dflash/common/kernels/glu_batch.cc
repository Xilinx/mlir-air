// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Eight-row 4B packets are token-major; 8B packets arrive in up/gate waves.
#ifndef GLU_BATCH_WAVE_LAYOUT
#define GLU_BATCH_WAVE_LAYOUT 0
#endif
#if GLU_BATCH_WAVE_LAYOUT
#include <aie_api/aie.hpp>
// aie::lut requires vector_decl_align (64 bytes on NPU2). The inherited
// chess_storage annotation did not survive this Peano build: the full graph's
// RTP word displaced .data to ...e004, whereas the standalone probe happened
// to place it at ...a000. Use the actual API requirement, not the older 32-byte
// vector annotation; ...e020 also fails the NPU2 requirement.
extern float activation_lut_ab[] __attribute__((aligned(aie::vector_decl_align)));
extern float activation_lut_cd[] __attribute__((aligned(aie::vector_decl_align)));
#include "fused_decode/kernels/glu.cc"

static_assert(GLU_SLICE == 1024);
// The core receives [up/gate wave, token, 512] linearly. A DMA scatter into
// [token,1024] would require an unencodable 256-word innermost wrap on NPU2.
extern "C" void glu_row_aie(bf16 *__restrict y, bf16 *__restrict x,
                             int row, int arm) {
  (void)arm;
  aie_round_nearest_even();
  const bf16 *up = x + row * 512;
  const bf16 *gate = x + 8 * 512 + row * 512;
  for (int i = 0; i < 512; i += 16) {
    aie::vector<bf16, 16> g = getActivationBf16(aie::load_v<16>(gate + i));
    aie::vector<bf16, 16> product = aie::mul(g, aie::load_v<16>(up + i));
    aie::store_v(y + row * 512 + i, product);
  }
}

extern "C" void glu_copy_aie(bf16 *__restrict y, bf16 *__restrict x,
                              int offset, int size, int arm) {
  (void)arm;
  aie_round_nearest_even();
  // Vocabulary has no activation: restore [token,1024] while draining two
  // 4096-element halves, preserving the existing host vocabulary layout.
  for (int j = 0; j < size; j += 16) {
    int linear = offset + j;
    int token = linear / 1024;
    int channel = linear % 1024;
    int source = (channel / 512) * (8 * 512) + token * 512 + channel % 512;
    aie::store_v(y + j, aie::load_v<16>(x + source));
  }
}

#else
#include "fused_decode/kernels/glu.cc"

extern "C" {
void glu_row_aie(bf16 *__restrict y, bf16 *__restrict x, int row, int arm) {
  (void)arm;
  aie_round_nearest_even();
  pseduo_glu<GLU_SLICE>(y + row * (GLU_SLICE / 2), x + row * GLU_SLICE);
}

void glu_copy_aie(bf16 *__restrict y, bf16 *__restrict x, int offset, int size,
                  int arm) {
  (void)arm;
  aie_round_nearest_even();
  for (int j = 0; j < size; j += 16)
    aie::store_v(y + j, aie::load_v<16>(x + offset + j));
}
}

#endif
