//===- Transform.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air-c/Transform.h"

#include "air/Transform/AIRTransformInterpreter.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"

void runTransform(MlirModule transform_ir, MlirModule payload_ir) {
  auto transformModule = unwrap(transform_ir);
  auto payloadModule = unwrap(payload_ir);
  auto logicalResult =
      xilinx::air::runAIRTransform(transformModule, payloadModule);
  (void)logicalResult;
  return;
}
