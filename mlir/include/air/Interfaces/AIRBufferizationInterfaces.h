//===- AIRBufferizationInterfaces.h -----------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_INTERFACES_BUFFERIZATION_INTERFACES_H
#define AIR_INTERFACES_BUFFERIZATION_INTERFACES_H

#include "mlir/IR/Dialect.h"

using namespace mlir;

namespace xilinx {
namespace air {

// Register all interfaces needed for bufferization.
void registerBufferizationInterfaces(DialectRegistry &registry);

} // namespace air
} // namespace xilinx

#endif // AIR_INTERFACES_BUFFERIZATION_INTERFACES_H
