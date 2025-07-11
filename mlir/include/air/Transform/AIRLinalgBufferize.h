//===- AIRLinalgBufferize.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_LINALG_BUFFERIZE_H
#define AIR_LINALG_BUFFERIZE_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass>
createAIRresolveTensorOpOperandConflictsWithNewTensors();

} // namespace air
} // namespace xilinx

#endif // AIR_LINALG_BUFFERIZE_H
