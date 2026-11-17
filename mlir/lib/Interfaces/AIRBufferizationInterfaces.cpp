//===- AIRBufferizationInterfaces.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Interfaces/AIRBufferizationInterfaces.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"

namespace xilinx {
namespace air {

void registerBufferizationInterfaces(DialectRegistry &registry) {
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  registry.insert<linalg::LinalgDialect>();
}

} // end namespace air
} // end namespace xilinx
