//===- AIRInterfaces.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Interfaces/AIRInterfaces.h"
#include "air/Interfaces/AIRBufferizationInterfaces.h"

#include "air/Dialect/AIR/AIRDialect.h"

using namespace mlir;

namespace xilinx {
namespace air {

void registerCodegenInterfaces(DialectRegistry &registry) {
  registerBufferizationInterfaces(registry);
}

} // end namespace air
} // end namespace xilinx
