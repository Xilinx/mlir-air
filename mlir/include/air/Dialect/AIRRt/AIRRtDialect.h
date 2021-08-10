// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
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

} // namespace airrt
} // namespace xilinx

#include "AIRRtOpsDialect.h.inc"

#endif // AIRRT_DIALECT_H
