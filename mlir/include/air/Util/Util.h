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

LogicalResult normalizeLoop(AffineForOp afo);

func::FuncOp getMangledFunction(ModuleOp module, std::string fnName,
                                ArrayRef<Value> operands,
                                ArrayRef<Type> retTys);

uint64_t getTensorVolume(const ShapedType ty);
uint64_t getTensorVolume(const Type ty);

// Get the parent scf.for op of an iter_arg
scf::ForOp getForRegionIterArgsOwner(Value val);
// Get the parent scf.parallel op of an init_val
scf::ParallelOp getParallelRegionInitValsOwner(Operation *op, Value val);
// Get the parent air.launch_herd op of a tile id
HerdOp getHerdArgOwner(Value val);
// Get the parent air.hierarchy op of a tile id
HierarchyInterface getHierarchyArgOwner(Value val);

// Erase a kernel operand from air.hierarchy op
void eraseAIRHierarchyOperand(HierarchyInterface op, unsigned index);

// Get operation's "id" attribute
int getIdAttr(Operation *op);

// Renumber the DMA ops. Mode can be within a herd or global
void renumberDmaOps(func::FuncOp func, std::string mode = "herd");

// Return op name as string
std::string to_string(Operation *op);
// Return memory space as string
std::string getMemorySpaceAsString(Value memref);

// Returns the first affine if op in block; nullptr otherwise
mlir::AffineIfOp getAffineIfInBlock(mlir::Block *block);
// Returns the first air.dma op in block; nullptr otherwise
DmaMemcpyNdOp getAIRDmaInBlock(mlir::Block *block);

// Get channel declaration through channel symbol
ChannelOp getChannelDeclarationThroughSymbol(ChannelInterface op);
// Get ChannelPutOp from ChannelOp
ChannelPutOp getChannelPutOpThroughSymbol(ChannelOp channel);
// Get ChannelGetOp from ChannelOp
ChannelGetOp getChannelGetOpThroughSymbol(ChannelOp channel);
// Get the other channel op through channel symbol
ChannelGetOp getTheOtherChannelOpThroughSymbol(ChannelPutOp put);
ChannelPutOp getTheOtherChannelOpThroughSymbol(ChannelGetOp get);

struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_UTIL_H
