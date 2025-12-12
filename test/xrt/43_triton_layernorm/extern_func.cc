//===- extern_func.cc -------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

extern "C" {

namespace c_api_wrapper {
v16accfloat getRsqrtBf16(v16bfloat16 x) {
  aie::vector<bfloat16, 16> vec = x;
  aie::vector<bfloat16, 16> result_vec = aie::invsqrt(vec);
  aie::accum<accfloat, 16> acc;
  acc.from_vector(result_vec, 0);
  return acc;
}

} // namespace c_api_wrapper

} // extern "C"
