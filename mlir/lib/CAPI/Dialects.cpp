//===- Dialects.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air-c/Dialects.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIR, air, xilinx::air::airDialect)

//===---------------------------------------------------------------------===//
// AsyncTokenType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAIRAsyncTokenType(MlirType type) {
  return llvm::isa<xilinx::air::AsyncTokenType>(unwrap(type));
}

MlirType mlirAIRAsyncTokenTypeGet(MlirContext ctx) {
  return wrap(xilinx::air::AsyncTokenType::get(unwrap(ctx)));
}
