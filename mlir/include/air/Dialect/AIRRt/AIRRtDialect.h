//===- AIRRtDialect.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_DIALECT_H
#define AIRRT_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace xilinx {
namespace airrt {
class TensorType : public Type::TypeBase<TensorType, Type, TypeStorage> {
public:
  using Base::Base;

  static TensorType get(MLIRContext *context) { return Base::get(context); }
  static constexpr StringLiteral name = "xilinx.airrt.tensor";
};

class EventType : public Type::TypeBase<EventType, Type, TypeStorage> {
public:
  using Base::Base;
  static constexpr StringLiteral name = "xilinx.airrt.event";
};

} // namespace airrt
} // namespace xilinx

#include "air/Dialect/AIRRt/AIRRtOpsDialect.h.inc"

#endif // AIRRT_DIALECT_H
