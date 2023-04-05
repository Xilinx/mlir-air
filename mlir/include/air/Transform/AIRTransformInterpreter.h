//===- AIRTransformInterpreter.h --------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TRANSFORM_INTERPRETER_H
#define AIR_TRANSFORM_INTERPRETER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRTransformInterpreterPass();
mlir::LogicalResult runAIRTransform(mlir::ModuleOp transformModule,
                                    mlir::ModuleOp payloadModule);

} // namespace air
} // namespace xilinx

#endif // AIR_TRANSFORM_INTERPRETER_H
