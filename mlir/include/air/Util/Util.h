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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace xilinx {
namespace air {

LogicalResult normalizeLoop(affine::AffineForOp afo);

func::FuncOp getMangledFunction(ModuleOp module, std::string fnName,
                                ArrayRef<Value> operands,
                                ArrayRef<Type> retTys);

uint64_t getTensorVolume(const ShapedType ty);
uint64_t getTensorVolume(const Type ty);
SmallVector<int> getTensorShape(const ShapedType ty);
SmallVector<int> getTensorShape(const Type ty);
std::string getElementTypeAsString(const mlir::Type ty);
uint64_t getElementSizeInBytes(const mlir::Type ty);

// Get the parent scf.for op of an iter_arg
scf::ForOp getForRegionIterArgsOwner(Value val);
// Get the parent scf.parallel op of an init_val
scf::ParallelOp getParallelRegionInitValsOwner(Operation *op, Value val);
// Get the parent air.launch_herd op of a tile id
HerdOp getHerdArgOwner(Value val);
// Get the parent air.hierarchy op of a tile id
HierarchyInterface getHierarchyArgOwner(Value val);
// Get the scf parent op from scf.yield op
template <typename T>
T getScfParentOpFromYieldOp(Operation *yield) {
  return dyn_cast_if_present<T>(yield->getParentOp());
}

std::optional<int64_t> getStaticScfForTripCountAsInt(scf::ForOp for_op);
std::optional<int64_t>
getStaticAffineForTripCountAsInt(affine::AffineForOp for_op);

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
// Return type name as string
std::string to_string(mlir::Type t);

// Generate a new unique channel name
std::string createChannelName(Operation *scope);
// Return memory space as string
std::string getMemorySpaceAsString(Value memref);

// Returns the first affine if op in block; nullptr otherwise
affine::AffineIfOp getAffineIfInBlock(mlir::Block *block);
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
std::vector<air::ChannelInterface>
getTheOtherChannelOpThroughSymbol(air::ChannelInterface op);
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

// Evaluate the condition lower and upper boundaries that the specified op is
// hitting, from an affine if nest. Assumes a rectangular condition bound
// region.
SmallVector<std::pair<int, int>> getRectangularConditionBoundsThroughAffineIfs(
    Operation *op, Operation *spatial_loop,
    std::vector<Operation *> affine_if_nest);

// Evaluate the integer value of affine set expression if the only symbolic
// identifier is replaced with zero
int evaluateSymbolEqualityInSet(AffineExpr c, MLIRContext *ctx);

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

// Recursively trace back in defining ops
void getDefiningOpsToOperands(Operation *op, SmallVector<Operation *> &def_ops);

// Fold perfectly nested parent loops into wraps and strides list
LogicalResult foldForLoopNestAsExtendedSizesAndStrides(
    OpBuilder builder, Operation *for_op, Operation *channel_op,
    SmallVector<Value> &offsets, SmallVector<Value> &wraps,
    SmallVector<Value> &strides, Value memref);

// Canonicalize wrap and stride lists, by removing redundant dimensions.
LogicalResult canonicalizeWrapAndStrideList(OpBuilder builder,
                                            SmallVector<Value> &offsets,
                                            SmallVector<Value> &sizes,
                                            SmallVector<Value> &strides,
                                            int memref_volume);

// If wrap-and-stride lists are empty, populate them with default data access
// layout (contiguous, row-major).
void populateDefaultWrapsAndStrides(OpBuilder builder, Value memref,
                                    SmallVector<Value> &offsets,
                                    SmallVector<Value> &wraps,
                                    SmallVector<Value> &strides);

// Check if the wraps and strides imply the default (contiguous, row-major) data
// access pattern.
bool isDefaultDataAccessPattern(SmallVector<Value> memcpy_sizes,
                                SmallVector<Value> memcpy_strides,
                                Value memref);
// Get the memref size along a given dimension, that the access pattern actually
// covers.
SmallVector<int64_t>
getEffectiveMemrefSizeFromAccessPattern(SmallVector<int> memref_shape,
                                        SmallVector<Value> sizes,
                                        SmallVector<Value> strides);

// Get the overall data access pattern from air.channel ops which access the
// memref.
std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
writeAccessPattern(air::ChannelInterface chanOp);
std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
writeAccessPattern(memref::SubViewOp subview);
std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
writeAccessPattern(mlir::vector::TransferReadOp readOp);
std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
writeAccessPattern(mlir::vector::TransferWriteOp writeOp);
SmallVector<int64_t>
getDataAccessShapeFromMemcpyOp(Value memref,
                               SmallVector<air::ChannelInterface> chanUsers);
SmallVector<int64_t>
getDataAccessShapeFromMemcpyOp(Value memref,
                               SmallVector<air::ChannelInterface> chanOps);
SmallVector<int64_t> getDataAccessShapeFromMemcpyOp(
    Value memref,
    SmallVector<
        std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>>
        patterns);
SmallVector<int64_t>
getDataAccessShapeFromMemcpyOp(Value memref, SmallVector<Operation *> users);

// Update strides after memref shrinkage. Assuming there is only one dimension
// being shrunk.
SmallVector<int>
getUpdatedStridesAfterShrinkage(SmallVector<int> old_memref_shape,
                                SmallVector<int64_t> new_memref_shape,
                                SmallVector<Value> strides);
// Update offsets after memref shrinkage.
SmallVector<int>
getUpdatedOffsetsAfterShrinkage(SmallVector<int> old_memref_shape,
                                SmallVector<int64_t> new_memref_shape,
                                SmallVector<Value> offsets);

// Given a dimension on wrap-and-stride list, infer the dimension on memref that
// this pattern spans completely on.
std::optional<int> getMemrefDimFromOffsetDim(int dimOnOffset,
                                             SmallVector<Value> offsets,
                                             SmallVector<Value> strides,
                                             SmallVector<int> memrefShape);

// Given a dimension on memref shape, infer the dimension on wrap-and-stride
// list that spans on this memref dimension.
std::optional<int> getOffsetDimFromMemrefDim(int dimOnMemref,
                                             SmallVector<Value> strides,
                                             SmallVector<int> memrefShape);

// Evaluate the affine expression of affine map on a sparse vector of constant
// ints.
std::optional<int64_t> evaluateConstantsInMap(
    AffineMap map, SmallVector<std::optional<int64_t>> symbolInputs,
    SmallVector<std::optional<int64_t>> dimInputs, MLIRContext *ctx);

// Extend the lookupOrDefault method to operate on a vector of values.
Value lookupOrDefaultRange(Value v, IRMapping &remap);
SmallVector<Value> lookupOrDefaultRange(SmallVectorImpl<Value> &vec,
                                        IRMapping &remap);
SmallVector<Value> lookupOrDefaultRange(OperandRange vec, IRMapping &remap);

// Extend isPure method to operate on air.execute.
bool isPure(Operation *op);

// Return if the given block contains N ops which are impure and aren't async
// wait ops (such as air.wait_all).
bool hasNImpureOps(Block *block, unsigned N);

// Return if the given block contains N ops or not, not counting the block's
// terminator.
bool hasNElements(Block *block, unsigned N);

// Get backward slice to a vector of values, within a specified region.
void getBackwardSliceInRegion(OpBuilder builder, Region *region,
                              SmallVectorImpl<Value> &vals,
                              SetVector<Operation *> &backwardSlices);

// Buffer all allocations of memref directly within the func op's body into the
// func op's arguments.
void populateBufferMemrefToFuncArgsPattern(RewritePatternSet &patterns);

// Find a common region that contains all ops, or ancestors of ops, until a
// specified region.
Region *findCommonRegionContainingAllAncestors(SmallVector<Operation *> ops,
                                               Operation *until = nullptr);

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_UTIL_H
