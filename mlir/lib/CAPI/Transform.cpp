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

MlirLogicalResult runTransform(MlirModule transform_ir, MlirModule payload_ir) {
  return wrap(
      xilinx::air::runAIRTransform(unwrap(transform_ir), unwrap(payload_ir)));
}
