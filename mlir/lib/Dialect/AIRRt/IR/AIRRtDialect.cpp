// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "AIRRtDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "AIRRtOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx::airrt;

void AIRRtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AIRRtOps.cpp.inc"
      >();
  addTypes<TensorType>();
}
