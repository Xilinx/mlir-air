// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef AIR_UTIL_UTIL_H
#define AIR_UTIL_UTIL_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace xilinx {
namespace air {

void coalesceLoops(AffineForOp outer, AffineForOp inner);

void normalizeLoop(AffineForOp afo);

FuncOp getMangledFunction(ModuleOp module, std::string fnName, ArrayRef<Value> operands, ArrayRef<Type> retTys);

uint64_t getTensorVolume(const ShapedType ty);

uint64_t getTensorVolume(const Type ty);

}
}
#endif // AIR_UTIL_UTIL_H
