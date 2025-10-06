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

#include "lut_based_ops.h"
#include <aie_api/aie.hpp>

extern "C" {

namespace c_api_wrapper {
v16accfloat getExpBf16(v16bfloat16 x) { return ::getExpBf16(x); }

} // namespace c_api_wrapper

} // extern "C"
