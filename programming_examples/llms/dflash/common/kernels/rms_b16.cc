// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rms_batch.cc"

// A single source frame contains sixteen activation rows and two gamma
// vectors. Every owner sees the frame and copies only its own rows.
static_assert(MODEL_DIM == 2560);
constexpr int OWNER_ROWS = 8;
constexpr int SOURCE_CHUNK = 512;
constexpr int OWNER_CHUNKS = OWNER_ROWS * MODEL_DIM / SOURCE_CHUNK;
constexpr int ACTIVATION_CHUNKS = 16 * MODEL_DIM / SOURCE_CHUNK;
constexpr int GAMMA_CHUNKS = MODEL_DIM / SOURCE_CHUNK;
extern "C" void owner_load_chunk(const bf16 *__restrict stage,
                                 bf16 *__restrict x, bf16 *__restrict w0,
                                 bf16 *__restrict w1, int chunk, int owner) {
  bf16 *dst;
  if (chunk < ACTIVATION_CHUNKS) {
    if (chunk / OWNER_CHUNKS != owner)
      return;
    dst = x + (chunk % OWNER_CHUNKS) * SOURCE_CHUNK;
  } else if (chunk < ACTIVATION_CHUNKS + GAMMA_CHUNKS) {
    dst = w0 + (chunk - ACTIVATION_CHUNKS) * SOURCE_CHUNK;
  } else {
    dst = w1 + (chunk - ACTIVATION_CHUNKS - GAMMA_CHUNKS) * SOURCE_CHUNK;
  }
  for (int i = 0; i < SOURCE_CHUNK; i += 32)
    aie::store_v(dst + i, aie::load_v<32>(stage + i));
}
