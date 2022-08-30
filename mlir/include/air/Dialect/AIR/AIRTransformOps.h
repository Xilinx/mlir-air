// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#ifndef MLIR_AIR_TRANSFORM_OPS_H
#define MLIR_AIR_TRANSFORM_OPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
class DialectRegistry;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace xilinx {
namespace air {
class PartitionOp;
}
}

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIRTransformOps.h.inc"

namespace xilinx {
namespace air {
void registerTransformDialectExtension(mlir::DialectRegistry &registry);
} // namespace air
} // namespace xilinx

#endif // MLIR_AIR_TRANSFORM_OPS_H
