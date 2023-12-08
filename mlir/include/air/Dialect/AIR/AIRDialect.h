//===- AIRDialect.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIR_DIALECT_H
#define MLIR_AIR_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"

#include <map>

using namespace mlir;

namespace xilinx {
namespace air {

void registerAIRRtTranslations();

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  static constexpr StringLiteral name = "xilinx.air.async_token";
};

// Adds a `air.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);
// Erases a `air.async.token` at position index of the argument list.
void eraseAsyncDependency(Operation *op, unsigned index);

} // namespace air
} // namespace xilinx

#include "air/Dialect/AIR/AIRDialect.h.inc"
#include "air/Dialect/AIR/AIREnums.h.inc"
#include "air/Dialect/AIR/AIROpInterfaces.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIR.h.inc"

#endif
