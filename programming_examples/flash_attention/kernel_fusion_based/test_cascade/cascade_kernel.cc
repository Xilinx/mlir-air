//===- cascade_kernel.cc - Cascade test kernels -----------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
// Minimal kernel functions for verifying AIE2P cascade data integrity.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef BUF_SIZE
#define BUF_SIZE 1024
#endif

extern "C" {

// Fill buffer with a constant value: buf[i] = (bfloat16)(float)val
void fill_pattern(bfloat16 *buf, int32_t val) {
  constexpr int VecLen = 32;
  bfloat16 fill_val = (bfloat16)((float)val);
  aie::vector<bfloat16, VecLen> v =
      aie::broadcast<bfloat16, VecLen>(fill_val);
  bfloat16 *__restrict p = buf;
  for (int i = 0; i < BUF_SIZE / VecLen; i++)
    chess_prepare_for_pipelining {
      aie::store_v(p, v);
      p += VecLen;
    }
}

// Add a constant to all elements: buf[i] += (bfloat16)(float)val
// Uses accfloat for precision.
void add_const_bf16(bfloat16 *buf, int32_t val) {
  constexpr int VecLen = 32;
  bfloat16 add_val = (bfloat16)((float)val);
  aie::vector<bfloat16, VecLen> add_vec =
      aie::broadcast<bfloat16, VecLen>(add_val);
  bfloat16 *__restrict p = buf;
  for (int i = 0; i < BUF_SIZE / VecLen; i++)
    chess_prepare_for_pipelining {
      aie::vector<bfloat16, VecLen> v = aie::load_v<VecLen>(p);
      aie::accum<accfloat, VecLen> acc(v);
      acc = aie::add(acc, add_vec);
      aie::store_v(p, acc.to_vector<bfloat16>());
      p += VecLen;
    }
}

// Zero-fill buffer
void zero_fill_cascade(bfloat16 *buf) {
  constexpr int VecLen = 32;
  aie::vector<bfloat16, VecLen> zero =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.0f);
  bfloat16 *__restrict p = buf;
  for (int i = 0; i < BUF_SIZE / VecLen; i++)
    chess_prepare_for_pipelining {
      aie::store_v(p, zero);
      p += VecLen;
    }
}

} // extern "C"
