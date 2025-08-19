//===- AIRBufferizationInterfaces.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Interfaces/AIRBufferizationInterfaces.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace xilinx {
namespace air {

namespace {

/// Normalize a (possibly partial) permutation `perm` to a permutation over a
/// rank-`rank` space.
///
/// Intuition:
///   * If `perm` already covers the whole rank (size == rank), return the
///     identity permutation. The caller will typically use `perm` elsewhere to
///     order inner tiles; here we only need an "outer" identity.
///   * If `perm` is partial (size < rank), first list all missing dimensions
///     in ascending order, then append `perm`. This yields a full permutation
///     where untouched (outer) dims stay in-order and the explicitly permuted
///     (inner) dims are placed at the end in the specified order.
///   * If `perm.size() > rank`, this is an invalid request.
static SmallVector<int64_t>
getPackUnpackNormalizedPerm(int rank, ArrayRef<int64_t> perm) {
  if (rank == (int)perm.size()) {
    // Full-rank input -> return identity over the whole rank.
    SmallVector<int64_t> vec;
    for (auto [index, value] : llvm::enumerate(perm))
      vec.push_back(index);
    return vec;
  } else if (rank > (int)perm.size()) {
    // Partial input -> infer and prepend the missing dims.
    SmallVector<int64_t> vec;
    for (auto i : llvm::seq<unsigned>(0, rank)) {
      if (llvm::any_of(perm, [i](int64_t elem) { return elem == i; }))
        continue;
      vec.push_back(i);
    }
    // Append the explicitly permuted dims.
    vec.insert(vec.end(), perm.begin(), perm.end());
    return vec;
  } else {
    // More entries than rank is not meaningful.
    assert(false &&
           "expected output permutation list's rank must not be less than the "
           "original permutation list");
    return SmallVector<int64_t>{};
  }
}

/// Compute the permutation that maps a packed (strip-mined) shape back to the
/// "pre-permuted" strip-mined shape, i.e., before any outer or inner
/// permutations were applied.
static SmallVector<int64_t>
getPackUnpackStripMinedPerm(ArrayRef<int64_t> shape,
                            ArrayRef<int64_t> innerDimsPos,
                            ArrayRef<int64_t> outerDimsPerm) {
  int64_t numPackedDims = innerDimsPos.size();
  int64_t packedRank = shape.size();

  // Indices of the "last N" dims that hold the packed tiles.
  auto lastDims = llvm::to_vector(
      llvm::seq<int64_t>(packedRank - numPackedDims, packedRank));

  // Derive insert/outer positions metadata for strip-mining.
  PackingMetadata packingMetadata =
      computePackingMetadata(packedRank, innerDimsPos);

  // (a) Permute last dims into requested inner positions.
  SmallVector<int64_t> innerPositionsPerm = computePermutationVector(
      packedRank, lastDims, packingMetadata.insertPositions);

  // (b) Optionally permute outer dims according to `outerDimsPerm`.
  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  if (!outerDimsPerm.empty())
    applyPermutationToVector(outerPos, outerDimsPerm);
  SmallVector<int64_t> outerPositionPerm = computePermutationVector(
      packedRank, packingMetadata.outerPositions, outerPos);

  // Compose outer perm over inner perm to obtain the final mapping.
  SmallVector<int64_t> packedToStripMinedShapePerm = innerPositionsPerm;
  applyPermutationToVector(packedToStripMinedShapePerm, outerPositionPerm);

  return packedToStripMinedShapePerm;
}

/// Result bundle for lowering Pack/UnPack:
///   - a `memref.transpose` on the (sub)tile
///   - a `xilinx.air.dma_memcpy_nd` that performs the copy
struct LowerPackUnPackResult {
  memref::TransposeOp transposeOp;
  xilinx::air::DmaMemcpyNdOp dmaOp;
};

/// Lower a `linalg.pack` into:
///   1) (Optional) expand_shape from source to strip-mined shape,
///   2) transpose to line up inner tile order,
///   3) DMA copy into the destination buffer.
///
/// Requires static inner tile sizes; dynamic tiles would need an
/// `memref.expand_shape` with dynamic sizes (NYI here).
FailureOr<LowerPackUnPackResult> lowerPack(RewriterBase &rewriter, Value source,
                                           Value dest, linalg::PackOp packOp) {
  // 1. Filter out unsupported cases.
  MemRefType inputMemRef = dyn_cast<MemRefType>(source.getType());
  if (!inputMemRef)
    return rewriter.notifyMatchFailure(packOp, "source isn't of MemRefType");
  MemRefType outputMemRef = dyn_cast<MemRefType>(dest.getType());
  if (!outputMemRef)
    return rewriter.notifyMatchFailure(packOp, "dest isn't of MemRefType");

  // Only support static inner tiles for now.
  if (llvm::any_of(packOp.getStaticInnerTiles(),
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return rewriter.notifyMatchFailure(
        packOp,
        "non-static shape NYI, needs a more powerful memref.expand_shape op");
  }

  Location loc = packOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(packOp);

  auto innerDimsPos = packOp.getInnerDimsPos();
  auto destShape = outputMemRef.getShape();
  SmallVector<int64_t> transpPerm = {};
  Value tile = nullptr;

  // If any inner dim in the destination is not a degenerate (1), we need to
  // explicitly materialize strip-mined shape and transpose.
  if (llvm::any_of(innerDimsPos, [destShape](int64_t index) {
        return destShape[index] != 1;
      })) {
    // (1) Compute permutation from packed -> strip-mined (pre-permutation) shape.
    PackingMetadata packingMetadata =
        computePackingMetadata(outputMemRef.getRank(), innerDimsPos);

    SmallVector<int64_t> packedToStripMinedShapePerm =
        getPackUnpackStripMinedPerm(outputMemRef.getShape(), innerDimsPos,
                                    packOp.getOuterDimsPerm());

    // (2) Build the strip-mined shape by permuting the packed shape.
    SmallVector<int64_t> stripMinedShape(outputMemRef.getShape());
    applyPermutationToVector(stripMinedShape, packedToStripMinedShapePerm);

    // (3) Expand from the padded result to the strip-mined shape.
    //     `reassociations` come from the packing metadata.
    tile = rewriter.create<memref::ExpandShapeOp>(
        loc, stripMinedShape, source, packingMetadata.reassociations);

    // (4) Later, transpose strip-mined -> packed (inverse of step 1).
    transpPerm = invertPermutationVector(packedToStripMinedShapePerm);
  } else {
    // Degenerate inner dims (all ones) → we can transpose directly.
    tile = source;
    transpPerm =
        getPackUnpackNormalizedPerm(inputMemRef.getRank(), innerDimsPos);
  }

  // Create the transpose op with the computed permutation.
  memref::TransposeOp transposeOp = rewriter.create<memref::TransposeOp>(
      loc, tile,
      AffineMapAttr::get(
          AffineMap::getPermutationMap(transpPerm, packOp->getContext())));

  // Inject a DMA copy from the transposed tile into the destination buffer.
  SmallVector<Value, 2> emptyVec;
  xilinx::air::DmaMemcpyNdOp dmaOp =
      rewriter.create<xilinx::air::DmaMemcpyNdOp>(
          loc, SmallVector<Type, 1>{}, emptyVec, dest, emptyVec, emptyVec,
          emptyVec, transposeOp.getResult(), emptyVec, emptyVec, emptyVec);

  return LowerPackUnPackResult{transposeOp, dmaOp};
}

/// Lower a `linalg.unpack` into:
///   1) (Optional) extract a subview of the packed tile,
///   2) transpose back from packed order,
///   3) DMA copy into the (unpacked) destination buffer.
///
/// For degenerate inner dims in the source (all ones), we can directly extract
/// a tile via `memref.subview`; otherwise we interpret the layout as
/// strip-mined and compute the inverse permutations accordingly.
FailureOr<LowerPackUnPackResult> lowerUnPack(RewriterBase &rewriter,
                                             Value source, Value dest,
                                             linalg::UnPackOp unPackOp) {
  Location loc = unPackOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(unPackOp);

  // Validate source/dest types.
  MemRefType inputMemRef = dyn_cast<MemRefType>(source.getType());
  if (!inputMemRef)
    return rewriter.notifyMatchFailure(unPackOp, "source isn't of MemRefType");
  MemRefType outputMemRef = dyn_cast<MemRefType>(dest.getType());
  if (!outputMemRef)
    return rewriter.notifyMatchFailure(unPackOp, "dest isn't of MemRefType");

  ArrayRef<int64_t> srcShape = inputMemRef.getShape();
  ArrayRef<int64_t> innerDimsPos = unPackOp.getInnerDimsPos();

  SmallVector<int64_t> perm = {};
  Value tile = nullptr;

  // If any inner dim in the *source* is not 1, we will transpose using the
  // strip-mined-to-packed permutation.
  if (llvm::any_of(innerDimsPos, [srcShape](int64_t index) {
        return srcShape[index] != 1;
      })) {
    int64_t packedRank = inputMemRef.getRank();

    // Derive packing metadata; used for the permutation (not for expand here).
    PackingMetadata packingMetadata =
        computePackingMetadata(packedRank, innerDimsPos);

    // Compute permutation from packed -> strip-mined shape.
    perm = getPackUnpackStripMinedPerm(inputMemRef.getShape(), innerDimsPos,
                                       unPackOp.getOuterDimsPerm());

    tile = source; // No subview needed in this case.
  } else {
    // Degenerate inner dims: extract the full inner tile with a subview, then
    // permute it so that inner tile dims match the expected order.
    int64_t srcRank = inputMemRef.getRank();
    int64_t destRank = outputMemRef.getRank();

    // Sanity: outer dims of result should be 1 (degenerate).
    if (llvm::any_of(innerDimsPos, [srcShape](int64_t index) {
          return srcShape[index] != 1;
        })) {
      return rewriter.notifyMatchFailure(
          unPackOp,
          "require the tiled outer dimensions of the result are all 1s");
    }

    // Build subview to read a tile from the packed source.
    Location loc = unPackOp.getLoc();
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        unPackOp.getDimAndTileMapping();
    Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
    Attribute oneIdxAttr = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> readOffsets(srcRank, zeroIdxAttr);
    SmallVector<OpFoldResult> readStrides(srcRank, oneIdxAttr);
    SmallVector<OpFoldResult> readSizes;
    SmallVector<int64_t> readShape;
    SmallVector<Value> dynamicDims;

    // Outer portion: keep sizes from src (except tiled dims -> size 1).
    for (auto i : llvm::seq<unsigned>(0, destRank)) {
      if (dimAndTileMapping.count(i)) {
        readSizes.push_back(oneIdxAttr);
        continue;
      }

      if (ShapedType::isDynamic(srcShape[i])) {
        return rewriter.notifyMatchFailure(
            unPackOp,
            "Support for dynamic input shapes is not implemented.");
      } else {
        readSizes.push_back(rewriter.getIndexAttr(srcShape[i]));
      }
      readShape.push_back(srcShape[i]);
    }

    // Append inner tile sizes (mixedTiles can contain attrs/values).
    auto mixedTiles = unPackOp.getMixedTiles();
    readSizes.append(mixedTiles.begin(), mixedTiles.end());

    // Explicitly compute the subview result type to keep inner tile dims even
    // if some sizes are 1 (we want to represent the whole inner tile).
    auto tileShape = srcShape.drop_front(destRank);
    readShape.append(tileShape.begin(), tileShape.end());
    auto readType =
        cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            readShape, inputMemRef, readOffsets, readSizes, readStrides));

    // Materialize the tile extraction.
    tile = rewriter.create<memref::SubViewOp>(
        loc, readType, source, readOffsets, readSizes, readStrides);

    // Compute permutation that aligns inner tile dims with `innerDimsPos`.
    perm = getPackUnpackNormalizedPerm(readType.getRank(), innerDimsPos);
    perm = invertPermutationVector(perm);
  }

  // Transpose packedShape -> stripMinedShape (or vice versa depending on path).
  memref::TransposeOp transposeOp = rewriter.create<memref::TransposeOp>(
      loc, tile,
      AffineMapAttr::get(
          AffineMap::getPermutationMap(perm, unPackOp->getContext())));

  // DMA copy into destination.
  SmallVector<Value, 2> emptyVec;
  xilinx::air::DmaMemcpyNdOp dmaOp =
      rewriter.create<xilinx::air::DmaMemcpyNdOp>(
          loc, SmallVector<Type, 1>{}, emptyVec, dest, emptyVec, emptyVec,
          emptyVec, transposeOp.getResult(), emptyVec, emptyVec, emptyVec);

  return LowerPackUnPackResult{transposeOp, dmaOp};
}

} // namespace

} // end namespace air
} // end namespace xilinx

namespace xilinx {
namespace air {

/// Helper for Pack/UnPack bufferization: retrieve the source and destination
/// buffers that the op should read/write.
template <typename OpTy>
static FailureOr<std::pair<Value, Value>> getSourceAndDestFromPackUnPackOp(
    RewriterBase &rewriter, OpTy op,
    const bufferization::BufferizationOptions &options,
    const bufferization::BufferizationState &state) {
  static_assert(llvm::is_one_of<OpTy, linalg::PackOp, linalg::UnPackOp>::value);
  Value source;
  auto maybeBuffer = getBuffer(rewriter, op.getSource(), options, state);
  if (failed(maybeBuffer))
    return failure();
  source = *maybeBuffer;

  // For destination-style ops, the result aliases exactly one init operand.
  Value dest;
  bufferization::AnalysisState analysisState(options);
  bufferization::AliasingOpOperandList aliasingOpOperands =
      analysisState.getAliasingOpOperands(op->getOpResult(0));
  assert(aliasingOpOperands.getNumAliases() == 1 && "expected 1 OpOperand");
  FailureOr<Value> resultBuffer = getBuffer(
      rewriter, aliasingOpOperands.getAliases().front().opOperand->get(),
      options, state);
  if (failed(resultBuffer))
    return failure();
  dest = *resultBuffer;
  return std::make_pair(source, dest);
}

/// Bufferize a `linalg.pack` by lowering it to transpose + DMA and replacing
/// the op with the destination buffer SSA value.
static LogicalResult
bufferizePackOp(RewriterBase &rewriter, linalg::PackOp op,
                const bufferization::BufferizationOptions &options,
                const bufferization::BufferizationState &state) {
  // Take a guard to restore insertion point after rewrites.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Resolve source/dest buffers.
  auto maybeSrcAndDest =
      getSourceAndDestFromPackUnPackOp(rewriter, op, options, state);
  if (failed(maybeSrcAndDest))
    return failure();
  auto [source, dest] = *maybeSrcAndDest;

  // Lower to IR (transpose + DMA).
  rewriter.setInsertionPoint(op);
  if (failed(lowerPack(rewriter, source, dest, op)))
    return failure();

  // Replace the tensor result with the destination buffer value.
  bufferization::replaceOpWithBufferizedValues(rewriter, op, dest);

  return success();
}

/// Bufferize a `linalg.unpack` by lowering it to (optional subview) + transpose
/// + DMA and replacing the op with the destination buffer SSA value.
static LogicalResult
bufferizeUnPackOp(RewriterBase &rewriter, linalg::UnPackOp op,
                  const bufferization::BufferizationOptions &options,
                  const bufferization::BufferizationState &state) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Resolve source/dest buffers.
  auto maybeSrcAndDest =
      getSourceAndDestFromPackUnPackOp(rewriter, op, options, state);
  if (failed(maybeSrcAndDest))
    return failure();
  auto [source, dest] = *maybeSrcAndDest;

  // Lower to IR (subview if needed) + transpose + DMA.
  rewriter.setInsertionPoint(op);
  if (failed(lowerUnPack(rewriter, source, dest, op)))
    return failure();

  // Replace the tensor result with the destination buffer value.
  bufferization::replaceOpWithBufferizedValues(rewriter, op, dest);

  return success();
}

/// External BufferizableOpInterface model for linalg.{Pack,UnPack}.
/// Declares read/write behavior and implements bufferization entry points.
template <typename OpTy>
struct PackUnPackOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          PackUnPackOpInterface<OpTy>, OpTy> {

  /// Both ops read their input tensor/buffer.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return true;
  }

  /// Writes occur on "init" operands (destination-style ops).
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    // Operand is written to if it has an aliasing OpResult.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  /// For a given result, return the aliasing init operand (destination).
  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const bufferization::AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return {dpsOp.getDpsInitOperand(opResult.getResultNumber())};
  }

  /// Value-level aliasing: the i-th "out" tensor may alias the i-th result.
  SmallVector<OpResult>
  getAliasingValue(Operation *op, OpOperand &opOperand,
                   const bufferization::AnalysisState &state) const {
    auto dspOp = cast<DestinationStyleOpInterface>(op);
    if (dspOp.isDpsInit(&opOperand))
      return {dspOp.getTiedOpResult(&opOperand)};
    return {};
  }

  /// Extended aliasing information with buffer relation semantics.
  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    auto dspOp = cast<DestinationStyleOpInterface>(op);
    if (dspOp.isDpsInit(&opOperand))
      return {bufferization::AliasingValue(
          dspOp.getTiedOpResult(&opOperand),
          bufferization::BufferRelation::Equivalent,
          /*isDefinite=*/false)};
    return {};
  }

  /// The result buffer is equivalent to the init buffer (destination-style).
  bufferization::BufferRelation
  bufferRelation(Operation *op, OpResult opResult,
                 const bufferization::AnalysisState &state) const {
    return bufferization::BufferRelation::Equivalent;
  }

  /// Bufferize by delegating to the specific lowerings above.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .template Case<linalg::PackOp>([&](auto pack) {
          return bufferizePackOp(rewriter, pack, options, state);
        })
        .template Case<linalg::UnPackOp>([&](auto unpack) {
          return bufferizeUnPackOp(rewriter, unpack, options, state);
        })
        .Default([](auto) { return failure(); });
  }
};

/// Registers external bufferization models for linalg Pack/UnPack.
void registerBufferizationInterfaces(DialectRegistry &registry) {
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  registry.insert<linalg::LinalgDialect>();
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::PackOp::attachInterface<PackUnPackOpInterface<linalg::PackOp>>(
        *ctx);
    linalg::UnPackOp::attachInterface<PackUnPackOpInterface<linalg::UnPackOp>>(
        *ctx);
  });
}

} // end namespace air
} // end namespace xilinx
