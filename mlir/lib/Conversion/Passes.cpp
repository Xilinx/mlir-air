//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Conversion/Passes.h.inc"
} // namespace

void xilinx::air::registerConversionPasses() {
  ::registerAIRConversionPasses();
}
