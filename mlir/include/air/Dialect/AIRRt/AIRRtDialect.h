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

using namespace mlir;

namespace xilinx {
namespace airrt {
class TensorType : public Type::TypeBase<TensorType, Type, TypeStorage> {
public:
  using Base::Base;

  static TensorType get(MLIRContext *context) { return Base::get(context); }
};

class EventType
    : public Type::TypeBase<EventType, Type, TypeStorage> {
public:
  using Base::Base;
};

} // namespace airrt
} // namespace xilinx

#include "air/Dialect/AIRRt/AIRRtOpsDialect.h.inc"

#endif // AIRRT_DIALECT_H
