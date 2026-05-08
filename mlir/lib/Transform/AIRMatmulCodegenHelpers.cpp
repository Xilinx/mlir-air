//===- AIRMatmulCodegenHelpers.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulCodegenHelpers.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace xilinx {
namespace air {

//===----------------------------------------------------------------------===//
// Pure predicates / utilities. Only those needed by helpers landed so far
// are defined; others arrive as their consuming runFoo functions land.
//===----------------------------------------------------------------------===//

// True if two index values are semantically the same: direct SSA equality,
// or their defining ops match per `air::isEquivalentTo` (which also accepts
// distinct constant SSAs that fold to the same int value).
static bool areEquivalentIndices(Value idx1, Value idx2) {
  if (idx1 == idx2)
    return true;
  Operation *def1 = idx1.getDefiningOp();
  Operation *def2 = idx2.getDefiningOp();
  if (!def1 || !def2)
    return false;
  return xilinx::air::isEquivalentTo(def1, def2);
}

bool areIdenticalReads(vector::TransferReadOp read1,
                       vector::TransferReadOp read2) {
  if (read1.getBase() != read2.getBase())
    return false;
  if (read1.getIndices().size() != read2.getIndices().size())
    return false;
  for (auto [idx1, idx2] : llvm::zip(read1.getIndices(), read2.getIndices())) {
    if (!areEquivalentIndices(idx1, idx2))
      return false;
  }
  auto vec1Ty = llvm::cast<VectorType>(read1.getVector().getType());
  auto vec2Ty = llvm::cast<VectorType>(read2.getVector().getType());
  return vec1Ty == vec2Ty;
}

bool dependsOnLoopIV(Value val, Value loopIV) {
  if (val == loopIV)
    return true;
  if (auto affineOp = val.getDefiningOp<affine::AffineApplyOp>()) {
    for (Value operand : affineOp.getMapOperands())
      if (dependsOnLoopIV(operand, loopIV))
        return true;
  }
  if (auto defOp = val.getDefiningOp()) {
    for (Value operand : defOp->getOperands())
      if (dependsOnLoopIV(operand, loopIV))
        return true;
  }
  return false;
}

bool hasWritesBetweenReads(vector::TransferReadOp firstRead,
                           vector::TransferReadOp secondRead) {
  Value sourceMemref = firstRead.getBase();

  Block *block = firstRead->getBlock();
  if (block != secondRead->getBlock())
    return true; // Conservative: different blocks, assume writes.

  auto firstIt = firstRead->getIterator();
  auto secondIt = secondRead->getIterator();
  for (auto it = ++firstIt; it != secondIt; ++it) {
    Operation *op = &(*it);

    auto memInterface = dyn_cast_if_present<MemoryEffectOpInterface>(op);
    if (!memInterface) {
      // Conservative: if effects can't be queried and op may recurse into
      // nested regions with writes, assume a write.
      if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      return true;
    }

    SmallVector<MemoryEffects::EffectInstance> effects;
    memInterface.getEffects(effects);
    for (auto &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      Value effectValue = effect.getValue();
      if (!effectValue)
        return true;
      if (effectValue == sourceMemref)
        return true;
      if (auto subview = effectValue.getDefiningOp<memref::SubViewOp>())
        if (subview.getSource() == sourceMemref)
          return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// runFoldUnitExtentDimsOnFunc
//===----------------------------------------------------------------------===//

LogicalResult runFoldUnitExtentDimsOnFunc(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();

  RewritePatternSet foldPatterns(ctx);
  linalg::ControlDropUnitDims options;
  // LLVM 23's collapseValue rejects memrefs with non-identity layouts (strided
  // memrefs from subview ops). Override collapseFn to use rank-reducing
  // memref.subview for strided memrefs, allowing the fold to handle linalg ops
  // with subview outputs inside air.herd regions.
  options.collapseFn =
      [](RewriterBase &rewriter, Location loc, Value operand,
         ArrayRef<int64_t> targetShape,
         ArrayRef<ReassociationIndices> reassociation,
         const linalg::ControlDropUnitDims &control) -> FailureOr<Value> {
    if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
      if (!memrefType.getLayout().isIdentity()) {
        return memref::SubViewOp::rankReduceIfNeeded(rewriter, loc, operand,
                                                     targetShape);
      }
      MemRefLayoutAttrInterface layout;
      auto targetType =
          MemRefType::get(targetShape, memrefType.getElementType(), layout,
                          memrefType.getMemorySpace());
      return memref::CollapseShapeOp::create(rewriter, loc, targetType, operand,
                                             reassociation)
          .getResult();
    }
    return failure();
  };
  linalg::populateFoldUnitExtentDimsPatterns(foldPatterns, options);
  (void)applyPatternsGreedily(funcOp, std::move(foldPatterns));
  return success();
}

//===----------------------------------------------------------------------===//
// runEliminateRedundantVectorTransfers
//===----------------------------------------------------------------------===//

int runEliminateRedundantVectorTransfers(Operation *target,
                                         RewriterBase &rewriter) {
  SmallVector<vector::TransferReadOp> transferReads;
  target->walk(
      [&](vector::TransferReadOp readOp) { transferReads.push_back(readOp); });

  llvm::SmallDenseSet<Operation *> eliminated;
  int eliminatedCount = 0;
  for (size_t i = 0; i < transferReads.size(); ++i) {
    if (eliminated.contains(transferReads[i]))
      continue;
    for (size_t j = i + 1; j < transferReads.size(); ++j) {
      if (eliminated.contains(transferReads[j]))
        continue;
      vector::TransferReadOp firstRead = transferReads[i];
      vector::TransferReadOp secondRead = transferReads[j];
      if (!areIdenticalReads(firstRead, secondRead))
        continue;
      if (hasWritesBetweenReads(firstRead, secondRead))
        continue;
      rewriter.replaceAllUsesWith(secondRead.getResult(),
                                  firstRead.getResult());
      rewriter.eraseOp(secondRead);
      eliminated.insert(secondRead);
      ++eliminatedCount;
    }
  }
  return eliminatedCount;
}

//===----------------------------------------------------------------------===//
// runFlattenForIterArgs
//===----------------------------------------------------------------------===//

FailureOr<scf::ForOp> runFlattenForIterArgs(scf::ForOp forOp,
                                            RewriterBase &rewriter) {
  Location loc = forOp.getLoc();

  // Collect vector-typed iter_args.
  SmallVector<unsigned> vectorIterArgIndices;
  SmallVector<VectorType> originalVectorTypes;
  SmallVector<VectorType> flattenedVectorTypes;
  for (auto [idx, iterArg] : llvm::enumerate(forOp.getInitArgs())) {
    if (auto vecType = dyn_cast_if_present<VectorType>(iterArg.getType())) {
      vectorIterArgIndices.push_back(idx);
      originalVectorTypes.push_back(vecType);
      int64_t numElements = vecType.getNumElements();
      flattenedVectorTypes.push_back(
          VectorType::get({numElements}, vecType.getElementType()));
    }
  }

  if (vectorIterArgIndices.empty())
    return forOp;

  // Step 1: insert shape_cast before the loop to flatten init values.
  rewriter.setInsertionPoint(forOp);
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  for (auto [idx, vecIdx] : llvm::enumerate(vectorIterArgIndices)) {
    Value initArg = forOp.getInitArgs()[vecIdx];
    auto shapeCast = vector::ShapeCastOp::create(
        rewriter, loc, flattenedVectorTypes[idx], initArg);
    newInitArgs[vecIdx] = shapeCast.getResult();
  }

  // Step 2: build new result types.
  SmallVector<Type> newResultTypes;
  for (auto [idx, resultType] : llvm::enumerate(forOp.getResultTypes())) {
    auto it = llvm::find(vectorIterArgIndices, idx);
    if (it != vectorIterArgIndices.end()) {
      size_t vecIdx = std::distance(vectorIterArgIndices.begin(), it);
      newResultTypes.push_back(flattenedVectorTypes[vecIdx]);
    } else {
      newResultTypes.push_back(resultType);
    }
  }

  // Step 3: create new scf.for with flattened iter_args.
  auto newForOp =
      scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), newInitArgs);

  // Step 4: clone the body, inserting shape_cast back to original shape for
  // vector iter_args inside the loop.
  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();
  rewriter.setInsertionPointToStart(newBody);
  IRMapping mapping;
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
  for (auto [idx, vecIdx] : llvm::enumerate(vectorIterArgIndices)) {
    BlockArgument newArg = newBody->getArgument(vecIdx + 1);
    auto shapeCast = vector::ShapeCastOp::create(
        rewriter, loc, originalVectorTypes[idx], newArg);
    mapping.map(oldBody->getArgument(vecIdx + 1), shapeCast.getResult());
  }
  for (auto [idx, arg] :
       llvm::enumerate(oldBody->getArguments().drop_front(1))) {
    if (llvm::find(vectorIterArgIndices, idx) == vectorIterArgIndices.end())
      mapping.map(arg, newBody->getArgument(idx + 1));
  }
  for (Operation &op : oldBody->without_terminator())
    rewriter.clone(op, mapping);

  // Step 5: rebuild yield, flattening vector values.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (auto [idx, yieldValue] : llvm::enumerate(oldYield.getOperands())) {
    auto it = llvm::find(vectorIterArgIndices, idx);
    if (it != vectorIterArgIndices.end()) {
      size_t vecIdx = std::distance(vectorIterArgIndices.begin(), it);
      Value mappedValue = mapping.lookup(yieldValue);
      auto shapeCast = vector::ShapeCastOp::create(
          rewriter, loc, flattenedVectorTypes[vecIdx], mappedValue);
      newYieldOperands.push_back(shapeCast.getResult());
    } else {
      newYieldOperands.push_back(mapping.lookup(yieldValue));
    }
  }
  scf::YieldOp::create(rewriter, loc, newYieldOperands);

  // Step 6: insert shape_cast back after the loop and replace uses.
  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> finalResults;
  for (auto [idx, result] : llvm::enumerate(newForOp.getResults())) {
    auto it = llvm::find(vectorIterArgIndices, idx);
    if (it != vectorIterArgIndices.end()) {
      size_t vecIdx = std::distance(vectorIterArgIndices.begin(), it);
      auto shapeCast = vector::ShapeCastOp::create(
          rewriter, loc, originalVectorTypes[vecIdx], result);
      finalResults.push_back(shapeCast.getResult());
    } else {
      finalResults.push_back(result);
    }
  }
  rewriter.replaceOp(forOp, finalResults);
  return newForOp;
}

//===----------------------------------------------------------------------===//
// runHoistLoopInvariantTransfers
//===----------------------------------------------------------------------===//

Value cloneOpAndOperands(Operation *op, Value loopIV, scf::ForOp loopOp,
                         RewriterBase &rewriter, IRMapping &mapping) {
  if (!op->getResults().empty())
    if (mapping.contains(op->getResult(0)))
      return mapping.lookup(op->getResult(0));

  for (Value operand : op->getOperands()) {
    if (operand == loopIV)
      continue;
    if (mapping.contains(operand))
      continue;
    if (isa<BlockArgument>(operand) && operand != loopIV)
      continue; // Outer-loop block args still in scope.
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;
    if (!loopOp->isAncestor(defOp))
      continue; // Defined outside the loop, already in scope.
    if (!dependsOnLoopIV(operand, loopIV)) {
      Value clonedOperand =
          cloneOpAndOperands(defOp, loopIV, loopOp, rewriter, mapping);
      mapping.map(operand, clonedOperand);
    }
  }

  Operation *cloned = rewriter.clone(*op, mapping);
  if (cloned->getResults().empty())
    return nullptr;
  return cloned->getResult(0);
}

namespace {

/// Hoist a single transfer_read/transfer_write pair out of `loopOp`. The
/// read is cloned before the loop, the write is cloned after the loop, and
/// the accumulator value flows through a new iter_arg.
FailureOr<scf::ForOp> hoistTransferPairFromLoop(vector::TransferReadOp readOp,
                                                vector::TransferWriteOp writeOp,
                                                scf::ForOp loopOp,
                                                RewriterBase &rewriter) {
  Value loopIV = loopOp.getInductionVar();

  rewriter.setInsertionPoint(loopOp);
  IRMapping readMapping;
  Value clonedReadResult =
      cloneOpAndOperands(readOp, loopIV, loopOp, rewriter, readMapping);

  Value writeVector = writeOp.getVector();
  auto yieldValuesFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
    BlockArgument readIterArg = newBbArgs.back();
    rewriter.replaceAllUsesWith(readOp.getResult(), readIterArg);
    return {writeVector};
  };

  FailureOr<LoopLikeOpInterface> newLoopResult =
      cast<LoopLikeOpInterface>(loopOp.getOperation())
          .replaceWithAdditionalYields(rewriter, ValueRange{clonedReadResult},
                                       /*replaceInitOperandUsesInLoop=*/true,
                                       yieldValuesFn);
  if (failed(newLoopResult))
    return failure();

  auto newLoop = cast<scf::ForOp>(newLoopResult->getOperation());
  rewriter.eraseOp(readOp);

  Value valueToWrite = newLoop.getResults().back();
  IRMapping writeMapping;
  writeMapping.map(writeVector, valueToWrite);
  rewriter.setInsertionPointAfter(newLoop);

  for (Value index : writeOp.getIndices()) {
    Operation *defOp = index.getDefiningOp();
    if (!defOp || dependsOnLoopIV(index, loopIV))
      continue;
    if (!newLoop->isProperAncestor(defOp))
      continue;
    if (!writeMapping.contains(index)) {
      Value clonedIndex =
          cloneOpAndOperands(defOp, loopIV, newLoop, rewriter, writeMapping);
      if (clonedIndex)
        writeMapping.map(index, clonedIndex);
    }
  }

  rewriter.clone(*writeOp.getOperation(), writeMapping);
  rewriter.eraseOp(writeOp);
  return newLoop;
}

} // namespace

FailureOr<scf::ForOp> runHoistLoopInvariantTransfers(Operation *scopeOp,
                                                     scf::ForOp loopOp,
                                                     RewriterBase &rewriter) {
  if (!scopeOp->isProperAncestor(loopOp))
    return loopOp->emitError("loop must be inside the scope operation");

  scf::ForOp currentLoop = loopOp;
  while (true) {
    Value loopIV = currentLoop.getInductionVar();
    vector::TransferWriteOp foundWrite = nullptr;
    vector::TransferReadOp foundRead = nullptr;

    currentLoop->walk([&](vector::TransferWriteOp writeOp) {
      if (foundWrite)
        return;
      if (writeOp->getParentOfType<scf::ForOp>() != currentLoop)
        return;
      for (Value index : writeOp.getIndices())
        if (dependsOnLoopIV(index, loopIV))
          return;

      currentLoop->walk([&](vector::TransferReadOp readOp) {
        if (foundRead)
          return;
        if (readOp->getParentOfType<scf::ForOp>() != currentLoop)
          return;
        if (readOp.getBase() != writeOp.getBase())
          return;
        for (Value index : readOp.getIndices())
          if (dependsOnLoopIV(index, loopIV))
            return;
        if (readOp.getIndices().size() != writeOp.getIndices().size())
          return;
        for (auto [ri, wi] :
             llvm::zip(readOp.getIndices(), writeOp.getIndices()))
          if (!areEquivalentIndices(ri, wi))
            return;
        foundRead = readOp;
      });
      if (foundRead)
        foundWrite = writeOp;
    });

    if (!foundWrite || !foundRead)
      break;

    FailureOr<scf::ForOp> newLoop =
        hoistTransferPairFromLoop(foundRead, foundWrite, currentLoop, rewriter);
    if (failed(newLoop))
      return currentLoop->emitError("failed to hoist transfer pair");
    currentLoop = *newLoop;
  }

  return currentLoop;
}

//===----------------------------------------------------------------------===//
// runHoistVectorTransferPointers
//===----------------------------------------------------------------------===//

LogicalResult runHoistVectorTransferPointers(scf::ForOp forOp,
                                             RewriterBase &rewriter) {
  Value loopIV = forOp.getInductionVar();
  Location loc = forOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);

  struct TransferOpInfo {
    Operation *op;
    Value base;
    MemRefType memrefType;
    VectorType vectorType;
    SmallVector<Value> indices;
    int64_t constantStride;
    bool hasIVDependentIndices;
  };

  SmallVector<TransferOpInfo> transferOps;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto transferOp = dyn_cast_if_present<VectorTransferOpInterface>(&op);
    if (!transferOp)
      continue;
    Value base = transferOp.getBase();
    auto memrefType = dyn_cast_if_present<MemRefType>(base.getType());
    if (!memrefType)
      continue;
    VectorType vectorType;
    if (auto readOp = dyn_cast_if_present<vector::TransferReadOp>(&op)) {
      vectorType = readOp.getVectorType();
    } else if (auto writeOp =
                   dyn_cast_if_present<vector::TransferWriteOp>(&op)) {
      vectorType = writeOp.getVectorType();
    } else {
      continue;
    }
    SmallVector<Value> indices(transferOp.getIndices().begin(),
                               transferOp.getIndices().end());
    bool hasIVDependentIndices = false;
    int64_t constantStride = 0;
    for (size_t dimIdx = 0; dimIdx < indices.size(); ++dimIdx) {
      Value idx = indices[dimIdx];
      if (dependsOnLoopIV(idx, loopIV)) {
        hasIVDependentIndices = true;
        int64_t dimStride = 1;
        for (size_t j = dimIdx + 1;
             j < static_cast<size_t>(memrefType.getRank()); ++j)
          dimStride *= memrefType.getShape()[j];
        // Assumes IV coefficient is 1 (index = IV or IV+const). This is the
        // total stride increment per loop iteration.
        constantStride += dimStride;
      }
    }
    transferOps.push_back({&op, base, memrefType, vectorType, indices,
                           constantStride, hasIVDependentIndices});
  }

  // Prepare iter_args (one base pointer per IV-dependent transfer).
  SmallVector<Value> newInitArgs;
  SmallVector<Value> flatMemrefs;
  for (const auto &info : transferOps) {
    if (!info.hasIVDependentIndices)
      continue;
    rewriter.setInsertionPoint(forOp);
    Value flatMemref = info.base;
    if (info.memrefType.getRank() > 1) {
      int64_t totalSize = 1;
      for (int64_t dim : info.memrefType.getShape()) {
        if (dim == ShapedType::kDynamic)
          return forOp->emitError("dynamic memref shapes not supported");
        totalSize *= dim;
      }
      MemRefType flatMemrefType =
          MemRefType::get({totalSize}, info.memrefType.getElementType(),
                          AffineMap(), info.memrefType.getMemorySpace());
      SmallVector<ReassociationIndices> reassociation;
      ReassociationIndices allDims;
      for (size_t i = 0; i < static_cast<size_t>(info.memrefType.getRank());
           ++i)
        allDims.push_back(i);
      reassociation.push_back(allDims);
      flatMemref = memref::CollapseShapeOp::create(
          rewriter, loc, flatMemrefType, info.base, reassociation);
    }
    flatMemrefs.push_back(flatMemref);

    int64_t rank = info.memrefType.getRank();
    AffineExpr linearExpr = rewriter.getAffineConstantExpr(0);
    int64_t stride = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
      linearExpr = linearExpr + rewriter.getAffineDimExpr(i) * stride;
      if (i > 0)
        stride *= info.memrefType.getShape()[i];
    }
    auto linearMap = AffineMap::get(rank, 0, linearExpr);

    SmallVector<Value> baseIndices;
    IRMapping indexMapping;
    for (Value idx : info.indices) {
      if (!dependsOnLoopIV(idx, loopIV)) {
        if (auto defOp = idx.getDefiningOp()) {
          Value clonedIdx =
              cloneOpAndOperands(defOp, loopIV, forOp, rewriter, indexMapping);
          baseIndices.push_back(clonedIdx ? clonedIdx : idx);
        } else {
          baseIndices.push_back(idx);
        }
      } else {
        baseIndices.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
      }
    }
    Value basePointer =
        affine::AffineApplyOp::create(rewriter, loc, linearMap, baseIndices);
    newInitArgs.push_back(basePointer);
  }

  // No IV-dependent transfers: rewrite each transfer to a 1D form using a
  // freshly-computed pointer per use, no iter_arg needed.
  if (newInitArgs.empty()) {
    for (const auto &info : transferOps) {
      rewriter.setInsertionPoint(info.op);
      int64_t numElements = info.vectorType.getNumElements();
      VectorType flatVectorType =
          VectorType::get({numElements}, info.vectorType.getElementType());

      rewriter.setInsertionPoint(forOp);
      Value flatMemref = info.base;
      if (info.memrefType.getRank() > 1) {
        int64_t totalSize = 1;
        for (int64_t dim : info.memrefType.getShape())
          totalSize *= dim;
        MemRefType flatMemrefType =
            MemRefType::get({totalSize}, info.memrefType.getElementType(),
                            AffineMap(), info.memrefType.getMemorySpace());
        SmallVector<ReassociationIndices> reassociation;
        ReassociationIndices allDims;
        for (size_t i = 0; i < static_cast<size_t>(info.memrefType.getRank());
             ++i)
          allDims.push_back(i);
        reassociation.push_back(allDims);
        flatMemref = memref::CollapseShapeOp::create(
            rewriter, loc, flatMemrefType, info.base, reassociation);
      }

      int64_t rank = info.memrefType.getRank();
      AffineExpr linearExpr = rewriter.getAffineConstantExpr(0);
      int64_t stride = 1;
      for (int64_t i = rank - 1; i >= 0; --i) {
        linearExpr = linearExpr + rewriter.getAffineDimExpr(i) * stride;
        if (i > 0)
          stride *= info.memrefType.getShape()[i];
      }
      auto linearMap = AffineMap::get(rank, 0, linearExpr);

      rewriter.setInsertionPoint(info.op);
      Value currentPointer =
          affine::AffineApplyOp::create(rewriter, loc, linearMap, info.indices);

      AffineMap identityMap1D = AffineMap::get(
          1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());
      auto inBoundsAttr = rewriter.getBoolArrayAttr({true});

      if (auto readOp = dyn_cast_if_present<vector::TransferReadOp>(info.op)) {
        Value flatRead = vector::TransferReadOp::create(
            rewriter, loc, flatVectorType, flatMemref,
            ValueRange{currentPointer}, AffineMapAttr::get(identityMap1D),
            readOp.getPadding(), /*mask=*/Value(), inBoundsAttr);
        Value shapedRead = vector::ShapeCastOp::create(
            rewriter, loc, info.vectorType, flatRead);
        rewriter.replaceOp(readOp, shapedRead);
      } else if (auto writeOp =
                     dyn_cast_if_present<vector::TransferWriteOp>(info.op)) {
        Value flatValue = vector::ShapeCastOp::create(
            rewriter, loc, flatVectorType, writeOp.getVector());
        rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
            writeOp, flatValue, flatMemref, ValueRange{currentPointer},
            AffineMapAttr::get(identityMap1D), /*mask=*/Value(), inBoundsAttr);
      }
    }
    return success();
  }

  // IV-dependent transfers: thread base pointers as iter_args, advance by
  // constant stride per iteration.
  auto yieldValuesFn =
      [&](OpBuilder &b, Location yieldLoc,
          ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
    SmallVector<Value> yieldValues;
    size_t iterArgIdx = 0;
    for (size_t i = 0; i < transferOps.size(); ++i) {
      const auto &info = transferOps[i];
      if (!info.hasIVDependentIndices)
        continue;
      BlockArgument ptrIterArg =
          newBbArgs[newBbArgs.size() - newInitArgs.size() + iterArgIdx];
      Value flatMemref = flatMemrefs[iterArgIdx];

      int64_t numElements = info.vectorType.getNumElements();
      VectorType flatVectorType =
          VectorType::get({numElements}, info.vectorType.getElementType());
      b.setInsertionPoint(info.op);
      AffineMap identityMap1D =
          AffineMap::get(1, 0, b.getAffineDimExpr(0), b.getContext());
      auto inBoundsAttr = b.getBoolArrayAttr({true});

      if (auto readOp = dyn_cast_if_present<vector::TransferReadOp>(info.op)) {
        Value flatRead = vector::TransferReadOp::create(
            b, loc, flatVectorType, flatMemref, ValueRange{ptrIterArg},
            AffineMapAttr::get(identityMap1D), readOp.getPadding(),
            /*mask=*/Value(), inBoundsAttr);
        Value shapedRead =
            vector::ShapeCastOp::create(b, loc, info.vectorType, flatRead);
        rewriter.replaceOp(readOp, shapedRead);
      } else if (auto writeOp =
                     dyn_cast_if_present<vector::TransferWriteOp>(info.op)) {
        Value flatValue = vector::ShapeCastOp::create(b, loc, flatVectorType,
                                                      writeOp.getVector());
        rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
            writeOp, flatValue, flatMemref, ValueRange{ptrIterArg},
            AffineMapAttr::get(identityMap1D), /*mask=*/Value(), inBoundsAttr);
      }

      Value strideConst =
          arith::ConstantIndexOp::create(b, yieldLoc, info.constantStride);
      Value nextPtr =
          arith::AddIOp::create(b, yieldLoc, ptrIterArg, strideConst);
      yieldValues.push_back(nextPtr);
      ++iterArgIdx;
    }
    return yieldValues;
  };

  FailureOr<LoopLikeOpInterface> newLoopResult =
      cast<LoopLikeOpInterface>(forOp.getOperation())
          .replaceWithAdditionalYields(rewriter, newInitArgs,
                                       /*replaceInitOperandUsesInLoop=*/true,
                                       yieldValuesFn);
  if (failed(newLoopResult))
    return forOp->emitError("failed to add pointer iter_args to loop");
  return success();
}

//===----------------------------------------------------------------------===//
// runHoistCastPair
//===----------------------------------------------------------------------===//

FailureOr<scf::ForOp> runHoistCastPair(Operation *extensionOp,
                                       Operation *truncationOp,
                                       scf::ForOp loopOp,
                                       RewriterBase &rewriter) {
  Value extensionInput, extensionOutput;
  Value truncationInput, truncationOutput;
  bool isFloatingPoint = false;

  if (auto extsiOp = dyn_cast_if_present<arith::ExtSIOp>(extensionOp)) {
    extensionInput = extsiOp.getIn();
    extensionOutput = extsiOp.getOut();
    auto trunciOp = dyn_cast_if_present<arith::TruncIOp>(truncationOp);
    if (!trunciOp)
      return extensionOp->emitError(
          "arith.extsi must be paired with arith.trunci");
    truncationInput = trunciOp.getIn();
    truncationOutput = trunciOp.getOut();
  } else if (auto extuiOp = dyn_cast_if_present<arith::ExtUIOp>(extensionOp)) {
    extensionInput = extuiOp.getIn();
    extensionOutput = extuiOp.getOut();
    auto trunciOp = dyn_cast_if_present<arith::TruncIOp>(truncationOp);
    if (!trunciOp)
      return extensionOp->emitError(
          "arith.extui must be paired with arith.trunci");
    truncationInput = trunciOp.getIn();
    truncationOutput = trunciOp.getOut();
  } else if (auto extfOp = dyn_cast_if_present<arith::ExtFOp>(extensionOp)) {
    extensionInput = extfOp.getIn();
    extensionOutput = extfOp.getOut();
    auto truncfOp = dyn_cast_if_present<arith::TruncFOp>(truncationOp);
    if (!truncfOp)
      return extensionOp->emitError(
          "arith.extf must be paired with arith.truncf");
    truncationInput = truncfOp.getIn();
    truncationOutput = truncfOp.getOut();
    isFloatingPoint = true;
  } else {
    return extensionOp->emitError(
        "extension operation must be arith.extsi, arith.extui, or arith.extf");
  }

  if (!loopOp->isProperAncestor(extensionOp) ||
      !loopOp->isProperAncestor(truncationOp))
    return loopOp->emitError(
        "extension and truncation operations must be inside the loop");

  // Find which iter_arg the extension operates on (directly or via shape_cast).
  BlockArgument iterArg = nullptr;
  int64_t iterArgIndex = -1;
  vector::ShapeCastOp shapeCastBeforeExtension = nullptr;
  if (auto blockArg = dyn_cast_if_present<BlockArgument>(extensionInput)) {
    if (blockArg.getOwner() == loopOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      iterArg = blockArg;
      iterArgIndex = blockArg.getArgNumber() - 1;
    }
  } else if (auto shapeCastOp =
                 extensionInput.getDefiningOp<vector::ShapeCastOp>()) {
    Value src = shapeCastOp.getSource();
    if (auto blockArg = dyn_cast_if_present<BlockArgument>(src)) {
      if (blockArg.getOwner() == loopOp.getBody() &&
          blockArg.getArgNumber() > 0) {
        iterArg = blockArg;
        iterArgIndex = blockArg.getArgNumber() - 1;
        shapeCastBeforeExtension = shapeCastOp;
      }
    }
  }
  if (!iterArg)
    return extensionOp->emitError("extension must operate on a loop iter_arg "
                                  "(directly or via shape_cast)");

  // The yielded value must come from the truncation (possibly via shape_cast)
  // and feed the same iter_arg position.
  vector::ShapeCastOp shapeCastAfterTruncation = nullptr;
  auto yieldOp = cast<scf::YieldOp>(loopOp.getBody()->getTerminator());
  bool truncationIsYielded = false;
  int64_t yieldIndex = -1;
  for (auto [idx, yieldValue] : llvm::enumerate(yieldOp.getOperands())) {
    if (yieldValue == truncationOutput) {
      truncationIsYielded = true;
      yieldIndex = idx;
      break;
    } else if (auto shapeCast =
                   yieldValue.getDefiningOp<vector::ShapeCastOp>()) {
      if (shapeCast.getSource() == truncationOutput) {
        truncationIsYielded = true;
        yieldIndex = idx;
        shapeCastAfterTruncation = shapeCast;
        break;
      }
    }
  }
  if (!truncationIsYielded || yieldIndex != iterArgIndex)
    return loopOp->emitError("truncation result must be yielded at the same "
                             "position as the extension iter_arg");

  Location loc = loopOp.getLoc();

  // Step 1: extend the init value before the loop.
  rewriter.setInsertionPoint(loopOp);
  Value initValue = loopOp.getInitArgs()[iterArgIndex];
  Type wideElemType =
      cast<VectorType>(extensionOutput.getType()).getElementType();
  Type wideInitType = VectorType::get(
      cast<VectorType>(initValue.getType()).getShape(), wideElemType);
  Value extendedInit;
  if (isFloatingPoint)
    extendedInit =
        arith::ExtFOp::create(rewriter, loc, wideInitType, initValue);
  else if (isa<arith::ExtSIOp>(extensionOp))
    extendedInit =
        arith::ExtSIOp::create(rewriter, loc, wideInitType, initValue);
  else
    extendedInit =
        arith::ExtUIOp::create(rewriter, loc, wideInitType, initValue);

  // Step 2: build new loop with the wide iter_arg.
  SmallVector<Value> newInitArgs(loopOp.getInitArgs().begin(),
                                 loopOp.getInitArgs().end());
  newInitArgs[iterArgIndex] = extendedInit;
  auto newLoopOp =
      scf::ForOp::create(rewriter, loc, loopOp.getLowerBound(),
                         loopOp.getUpperBound(), loopOp.getStep(), newInitArgs);

  // Step 3: clone the loop body, adjusting types as needed.
  Block *oldBody = loopOp.getBody();
  Block *newBody = newLoopOp.getBody();
  rewriter.setInsertionPointToStart(newBody);
  IRMapping mapping;
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
  for (auto [idx, oldArg] :
       llvm::enumerate(oldBody->getArguments().drop_front(1)))
    mapping.map(oldArg, newBody->getArgument(idx + 1));

  for (Operation &op : oldBody->without_terminator()) {
    if (&op == extensionOp) {
      if (!shapeCastBeforeExtension) {
        // No shape_cast: extension result becomes the wide iter_arg directly.
        mapping.map(extensionOutput, newBody->getArgument(iterArgIndex + 1));
      }
      continue;
    }
    if (&op == truncationOp)
      continue; // Yield handled below.
    if (shapeCastBeforeExtension &&
        &op == shapeCastBeforeExtension.getOperation()) {
      auto narrowVecType =
          cast<VectorType>(shapeCastBeforeExtension.getResult().getType());
      auto wideVecType =
          VectorType::get(narrowVecType.getShape(), wideElemType);
      Value mappedSource = mapping.lookup(shapeCastBeforeExtension.getSource());
      auto newShapeCast =
          vector::ShapeCastOp::create(rewriter, loc, wideVecType, mappedSource);
      mapping.map(shapeCastBeforeExtension.getResult(),
                  newShapeCast.getResult());
      mapping.map(extensionOutput, newShapeCast.getResult());
      continue;
    }
    if (shapeCastAfterTruncation &&
        &op == shapeCastAfterTruncation.getOperation())
      continue; // Handled in yield processing.
    rewriter.clone(op, mapping);
  }

  // Step 4: build new yield with the wide value.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (auto [idx, yieldValue] : llvm::enumerate(oldYield.getOperands())) {
    if ((int64_t)idx == iterArgIndex) {
      Value wideValue = mapping.lookup(truncationInput);
      if (shapeCastAfterTruncation) {
        auto narrowVecType =
            cast<VectorType>(shapeCastAfterTruncation.getResult().getType());
        auto wideVecType =
            VectorType::get(narrowVecType.getShape(), wideElemType);
        auto newShapeCast =
            vector::ShapeCastOp::create(rewriter, loc, wideVecType, wideValue);
        newYieldOperands.push_back(newShapeCast.getResult());
      } else {
        newYieldOperands.push_back(wideValue);
      }
    } else {
      newYieldOperands.push_back(mapping.lookup(yieldValue));
    }
  }
  scf::YieldOp::create(rewriter, loc, newYieldOperands);

  // Step 5: truncate the wide loop result back to narrow type.
  rewriter.setInsertionPointAfter(newLoopOp);
  Value wideResult = newLoopOp.getResults()[iterArgIndex];
  auto narrowElemType =
      cast<VectorType>(loopOp.getInitArgs()[iterArgIndex].getType())
          .getElementType();
  auto narrowResultType = VectorType::get(
      cast<VectorType>(wideResult.getType()).getShape(), narrowElemType);
  Value narrowResult;
  if (isFloatingPoint)
    narrowResult =
        arith::TruncFOp::create(rewriter, loc, narrowResultType, wideResult);
  else
    narrowResult =
        arith::TruncIOp::create(rewriter, loc, narrowResultType, wideResult);

  // Step 6: replace uses of the old loop.
  SmallVector<Value> finalResults;
  for (auto [idx, result] : llvm::enumerate(newLoopOp.getResults())) {
    if ((int64_t)idx == iterArgIndex)
      finalResults.push_back(narrowResult);
    else
      finalResults.push_back(result);
  }
  rewriter.replaceOp(loopOp, finalResults);
  return newLoopOp;
}

} // namespace air
} // namespace xilinx
