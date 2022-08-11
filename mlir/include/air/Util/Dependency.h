// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
//===- AIRDependencyUtils.h - AIR Loop tiling utilities ------------------------===//
//
// This header file defines utility functions that are commonly used in passes,
// primarily AIR dependency tracing passes.
//===-----------------------------------------------------------------------===//

#pragma once

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace xilinx {
namespace air {

bool areEqualIndices (mlir::Value index_0, mlir::Value index_1);
void traceDependentInductionVar (air::DmaMemcpyInterface async_op, SmallVector<Value, 1> &loop_dep_history, std::vector<Operation *> &op_history);
void traceDependentInductionVar (air::AsyncOpInterface async_op, SmallVector<Value, 1> &loop_dep_history, std::vector<Operation *> &op_history);
void eraseAsyncDependencyFromAsyncOp(xilinx::air::AsyncOpInterface op, Value token);

} // namespace air
} // namespace xilinx
