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
std::string getElementTypeAsString(const mlir::Type ty);
unsigned getElementSizeInBytes(const mlir::Type ty);

// Get the parent scf.for op of an iter_arg
scf::ForOp getForRegionIterArgsOwner(Value val);
// Get the parent scf.parallel op of an init_val
scf::ParallelOp getParallelRegionInitValsOwner(Operation *op, Value val);
// Get the parent air.launch_herd op of a tile id
HerdOp getHerdArgOwner(Value val);
// Get the parent air.hierarchy op of a tile id
HierarchyInterface getHierarchyArgOwner(Value val);
// Get the scf parent op from scf.yield op
template <typename T> T getScfParentOpFromYieldOp(Operation *yield) {
  return dyn_cast_if_present<T>(yield->getParentOp());
}

// Erase a kernel operand from air.hierarchy op
void eraseAIRHierarchyOperand(HierarchyInterface op, unsigned index);

// Get operation's "id" attribute
int getIdAttr(Operation *op);

// Renumber the DMA ops. Mode can be within a herd or global
void renumberDmaOps(func::FuncOp func, std::string mode = "herd");
void renumberChannelOps(Block *region);
void renumberChannelOps(Block *region, std::map<int, int> &reverse_map);

// Return op name as string
std::string to_string(Operation *op);
std::string to_string(mlir::Type t);
// Return memory space as string
std::string getMemorySpaceAsString(Value memref);

// Returns the first affine if op in block; nullptr otherwise
mlir::AffineIfOp getAffineIfInBlock(mlir::Block *block);
// Returns the first air.dma op in block; nullptr otherwise
DmaMemcpyNdOp getAIRDmaInBlock(mlir::Block *block);

// Get channel declaration through channel symbol
ChannelOp getChannelDeclarationThroughSymbol(ChannelInterface op);
// Get ChannelPutOps from ChannelOp
std::vector<ChannelPutOp>
getChannelPutOpThroughSymbol(ChannelOp channel, Operation *scope = nullptr);
// Get ChannelGetOps from ChannelOp
std::vector<ChannelGetOp>
getChannelGetOpThroughSymbol(ChannelOp channel, Operation *scope = nullptr);
// Get the other channel op through channel symbol
std::vector<ChannelGetOp> getTheOtherChannelOpThroughSymbol(ChannelPutOp put);
std::vector<ChannelPutOp> getTheOtherChannelOpThroughSymbol(ChannelGetOp get);
void getSizesFromIntegerSet(MLIRContext *ctx, IntegerSet int_set,
                            SmallVector<int, 2> &lbs_int,
                            SmallVector<int, 2> &ubs_int);
// Get spatial sizes from spatial loop (scf.parallel or air.hierarchy)
void getSizesFromSpatialLoop(Operation *spatial_loop,
                             SmallVector<int, 2> &lbs_spatial,
                             SmallVector<int, 2> &ubs_spatial);
// Get else sizes from affine.if. Assumption: rectangular input, then and else
// sizes only
void getElseSizesFromAffineIf(SmallVector<int, 2> &lbs_in,
                              SmallVector<int, 2> &ubs_in,
                              SmallVector<int, 2> &lbs_then,
                              SmallVector<int, 2> &ubs_then);
// Walk affine.if then and else blocks and check if current core lies in
// condition
bool positionHitsAffineIfCondition(Operation *op,
                                   std::vector<unsigned> position);
bool positionHitsAffineIfCondition(Operation *op, Operation *spatial_loop,
                                   std::vector<Operation *> affine_if_nest,
                                   std::vector<unsigned> position);
Operation *
getAffineIfNestAndSpatialLoopFromOp(Operation *op,
                                    std::vector<Operation *> &affine_if_nest,
                                    Operation *&spatial_loop);

struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

// Check if an operand of an operation is read or write access
char checkOpOperandReadOrWrite(mlir::OpOperand &op_operand);
char checkOpOperandReadOrWrite(Value op_operand, Operation *owner);

// Convert a vector of SSA returned from arith::ConstantIndexOp into a vector of
// uints
std::vector<unsigned>
convertVecOfConstIndexToVecOfUInt(SmallVector<Value> svec);

// Get iterator corresponding to a position in a multi-dimensional vector
unsigned getIteratorFromMDVector(std::vector<unsigned> dims,
                                 std::vector<unsigned> position);
// Get coordinates corresponding to a position in a multi-dimensional vector
// from an iterator
std::vector<unsigned> getMDVectorFromIterator(std::vector<unsigned> dims,
                                              unsigned iter);

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_UTIL_H
