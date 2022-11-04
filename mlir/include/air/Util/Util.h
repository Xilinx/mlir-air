//===- Util.h ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_UTIL_H
#define AIR_UTIL_UTIL_H

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

scf::ParallelOp getParallelRegionInitValsOwner(Operation *op, Value val);

air::HerdOp getHerdArgOwner(Value val);

air::HierarchyInterface getHierarchyArgOwner(Value val);

int getIdAttr(Operation *op);

void renumberDmaOps(func::FuncOp func, std::string mode = "herd");

std::string to_string(Operation *op);

struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

} // namespace air
} // namespace xilinx
#endif // AIR_UTIL_UTIL_H