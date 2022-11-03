//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Transform/Passes.h.inc"
}

void xilinx::air::registerTransformPasses() { ::registerAIRTransformPasses(); }
