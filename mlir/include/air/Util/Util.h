// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef AIR_UTIL_UTIL_H
#define AIR_UTIL_UTIL_H

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace xilinx {
namespace air {

void coalesceLoops(AffineForOp outer, AffineForOp inner);

void normalizeLoop(AffineForOp afo);

func::FuncOp getMangledFunction(ModuleOp module, std::string fnName,
                                ArrayRef<Value> operands,
                                ArrayRef<Type> retTys);

uint64_t getTensorVolume(const ShapedType ty);

uint64_t getTensorVolume(const Type ty);

scf::ForOp getForRegionIterArgsOwner(Value val);

air::HerdLaunchOp getHerdLaunchTileIdOwner(Value val);

}
}
#endif // AIR_UTIL_UTIL_H
