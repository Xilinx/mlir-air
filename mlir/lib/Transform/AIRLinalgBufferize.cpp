//===- AIRLinalgBufferize.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRLinalgBufferize.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Util/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "air-linalg-bufferize"

using namespace mlir;

namespace xilinx {
namespace air {

LogicalResult resolveTensorOpOperandConflictsWithNewTensors(
    Operation *op, RewriterBase &rewriter,
    const bufferization::AnalysisState &analysisState,
    const bufferization::BufferizationState &bufferizationState) {
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<OpOperand *> outOfPlaceOpOperands;
  DenseSet<Value> readValues;

  // Find all OpOperands that leads to a conflict arising from read-write buffer
  // reuse.
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();
    if (!llvm::isa<TensorType>(operandType))
      continue;
    if (analysisState.isInPlace(opOperand))
      continue;
    if (llvm::isa<UnrankedTensorType>(operandType))
      return op->emitError("copying of unranked tensors is not implemented");

    if (analysisState.bufferizesToMemoryWrite(opOperand) &&
        readValues.contains(opOperand.get())) {
      outOfPlaceOpOperands.push_back(&opOperand);
    }
    if (analysisState.bufferizesToMemoryRead(opOperand))
      readValues.insert(opOperand.get());
  }

  // Insert copies of OpOperands.
  rewriter.setInsertionPoint(op);
  for (OpOperand *opOperand : outOfPlaceOpOperands) {
    FailureOr<Value> copy = allocateTensorForShapedValue(
        rewriter, op->getLoc(), opOperand->get(), analysisState.getOptions(),
        bufferizationState, /*copy*/ false);
    if (failed(copy))
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { opOperand->set(*copy); });
  }

  return success();
}

class AIRresolveTensorOpOperandConflictsWithNewTensors
    : public air::impl::AIRresolveTensorOpOperandConflictsWithNewTensorsBase<
          AIRresolveTensorOpOperandConflictsWithNewTensors> {

public:
  AIRresolveTensorOpOperandConflictsWithNewTensors() = default;
  AIRresolveTensorOpOperandConflictsWithNewTensors(
      const AIRresolveTensorOpOperandConflictsWithNewTensors &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    air::airDialect>();
  }

  void runOnFunction(func::FuncOp f) {
    bufferization::OneShotBufferizationOptions options;
    bufferization::BufferizationState bufferizationState;
    bufferization::OneShotAnalysisState analysisState(f, options);
    SmallVector<Operation *> worklist;
    f.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (options.isOpAllowed(op) && bufferization::hasTensorSemantics(op))
        worklist.push_back(op);
    });
    IRRewriter rewriter(f.getContext());
    for (unsigned i = 0; i < worklist.size(); ++i) {
      Operation *nextOp = worklist[i];
      // Skip ops that are not bufferizable or not allowed.
      auto bufferizableOp = options.dynCastBufferizableOp(nextOp);
      if (!bufferizableOp)
        continue;
      // Skip ops that no longer have tensor semantics.
      if (!bufferization::hasTensorSemantics(nextOp))
        continue;
      // Check for unsupported unstructured control flow.
      if (!bufferizableOp.supportsUnstructuredControlFlow())
        for (Region &r : nextOp->getRegions())
          if (r.getBlocks().size() > 1) {
            nextOp->emitOpError(
                "op or BufferizableOpInterface implementation does not support "
                "unstructured control flow, but at least one region has "
                "multiple "
                "blocks");
            signalPassFailure();
          }
      // Resolve conflicts for op's operands and results.
      LLVM_DEBUG(llvm::outs()
                 << "//===-------------------------------------------===//\n"
                 << "IR after resolving conflicts: " << nextOp->getName()
                 << "\n");
      rewriter.setInsertionPoint(nextOp);
      if (failed(resolveTensorOpOperandConflictsWithNewTensors(
              nextOp, rewriter, analysisState, bufferizationState))) {
        LLVM_DEBUG(
            llvm::outs()
            << "failed to resolve conflicts\n"
            << "//===-------------------------------------------===//\n");
        nextOp->emitError("failed to resolve operand conflicts for op");
        signalPassFailure();
      }
    }
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOnFunction(f);
  }

private:
};

} // namespace air
} // namespace xilinx

//===----------------------------------------------------------------------===//
// AIRHoistStaticAllocOp
//===----------------------------------------------------------------------===//

/// Returns true if a particular use of an allocation can be replaced with a
/// `memref.subview` without violating typing constraints.
///
/// Safe cases:
///   * linalg ops (buffer semantics): will accept subviews
///   * memref.dealloc: deallocating a subview of an entry allocation is valid
///   * memref.store: stores accept subviews
///   * memref.subview: composing subviews is fine
///
/// Non-trivial / unsafe (hence excluded here):
///   * scf.yield / func.return: the yielded/returned value type must match the
///     function / region type exactly; a subview type would differ.
///
static bool isUseReplaceableWithSubview(OpOperand &use) {
  Operation *user = use.getOwner();
  return isa<linalg::LinalgOp, memref::DeallocOp, memref::StoreOp,
             memref::SubViewOp>(user);
}

template <typename AllocLikeOpType>
std::optional<Value> hoistOneStaticallyBoundAllocation(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment,
    std::optional<vector::VscaleRange> vscaleRange) {
  // Encode alignment as an attribute if provided. (Some alloc-like ops accept
  // alignment attrs; we thread it through unchanged.)
  IntegerAttr alignmentAttr =
      alignment ? builder.getI64IntegerAttr(alignment.value()) : nullptr;

  // Fast path: if there are no dynamic sizes at all, create the same allocation
  // in the function entry block and return it. No subview is needed because the
  // type is identical.
  if (dynamicSizes.empty()) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    Value allocation =
        AllocLikeOpType::create(builder, loc, allocLikeType, alignmentAttr);
    // For memref.alloc, also insert a dealloc in the entry block terminator
    // block to preserve semantics (leaks avoided).
    if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
      builder.setInsertionPoint(
          funcOp.getFunctionBody().front().getTerminator());
      memref::DeallocOp::create(builder, loc, allocation);
    }
    return allocation;
  }

  // General path: we will create an entry-block allocation with the same
  // *static* shape as `allocLikeType`, then produce a subview with the
  // original shape to substitute the original value. This keeps users' types
  // unchanged even when the new allocation uses canonicalized sizes.
  SmallVector<OpFoldResult> allocSizes;
  SmallVector<OpFoldResult> subviewSizes;
  allocSizes.reserve(allocLikeType.getRank());
  subviewSizes.reserve(allocLikeType.getRank());

  Value allocation;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());

    // Materialize sizes for each dimension. Only static sizes are supported in
    // this implementation. If any dimension is dynamic, we currently bail out.
    for (auto dimSize : allocLikeType.getShape()) {
      if (!ShapedType::isDynamic(dimSize)) {
        auto dimSizeAttr = builder.getIndexAttr(dimSize);
        allocSizes.push_back(dimSizeAttr);
        subviewSizes.push_back(dimSizeAttr);
        continue;
      }
      // Dynamic shaped allocLike unsupported (NYI).
      return nullptr;
    }

    SmallVector<int64_t> staticShape;
    SmallVector<Value> dynamicSizes;
    dispatchIndexOpFoldResults(allocSizes, dynamicSizes, staticShape);
    auto allocationType = allocLikeType.clone(staticShape);

    allocation = AllocLikeOpType::create(builder, loc, allocationType,
                                         dynamicSizes, alignmentAttr);
  }

  // Create a subview that exactly matches the original requested type.
  // Offsets = 0, Strides = 1, Sizes = original shape.
  SmallVector<OpFoldResult> offsets(allocLikeType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(allocLikeType.getRank(),
                                    builder.getIndexAttr(1));
  Value subviewOp = memref::SubViewOp::create(builder, loc, allocation, offsets,
                                              subviewSizes, strides);

  // Some consumers (e.g., another subview) may require the *exact* original
  // memref type. If the subview result type does not match, insert an explicit
  // memref.cast back to `allocLikeType` to satisfy verifier and users.
  if (subviewOp.getType() != allocLikeType) {
    subviewOp = memref::CastOp::create(builder, loc, allocLikeType, subviewOp);
  }

  // As above, insert a dealloc at function end.
  if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
    builder.setInsertionPoint(funcOp.getFunctionBody().front().getTerminator());
    memref::DeallocOp::create(builder, loc, allocation);
  }

  return subviewOp;
}

template <typename AllocLikeOpType>
std::optional<Value> hoistOneStaticallyBoundAllocation(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder,
    AllocLikeOpType allocLikeOp,
    std::optional<vector::VscaleRange> vscaleRange) {
  // Convenience overload: set insertion point to the original alloc-like op
  // and forward its properties to the main hoisting routine.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(allocLikeOp);
  return hoistOneStaticallyBoundAllocation<AllocLikeOpType>(
      funcOp, builder, allocLikeOp.getLoc(), allocLikeOp.getType(),
      allocLikeOp.getDynamicSizes(), allocLikeOp.getAlignment(), vscaleRange);
}

template <typename AllocLikeOpType>
void hoistStaticallyBoundAllocationsInFunc(
    RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
    std::optional<vector::VscaleRange> vscaleRange = std::nullopt) {
  SmallVector<AllocLikeOpType> allocLikeOps;

  // Collect candidate alloc-like ops that are not already in the entry block
  // and whose uses are safe to rewrite (or have no dynamic sizes).
  funcOp.walk([&](AllocLikeOpType allocLikeOp) {
    if (allocLikeOp->getBlock() == &funcOp.getFunctionBody().front())
      return;
    if (allocLikeOp.getDynamicSizes().empty()) {
      allocLikeOps.push_back(allocLikeOp);
      return;
    }
    // All uses must tolerate replacement by a subview.
    if (llvm::all_of(allocLikeOp->getUses(), [](OpOperand &use) {
          return isUseReplaceableWithSubview(use);
        })) {
      allocLikeOps.push_back(allocLikeOp);
      return;
    }
  });

  // Hoist each candidate and replace all uses with the hoisted value.
  for (auto allocLikeOp : allocLikeOps) {
    // Track and remove any deallocs tied to the original allocation; the new
    // hoisted allocation installs its own dealloc in the entry block.
    SmallVector<memref::DeallocOp> deallocOps;
    for (Operation *user : allocLikeOp->getUsers()) {
      auto dealloc = dyn_cast<memref::DeallocOp>(user);
      if (dealloc)
        deallocOps.push_back(dealloc);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Alloca Op : ";
      allocLikeOp->dump();
      int numUses = std::distance(allocLikeOp.getResult().use_begin(),
                                  allocLikeOp.getResult().use_end());
      llvm::dbgs() << " num Uses : " << numUses;
    });
    std::optional<Value> replacement = hoistOneStaticallyBoundAllocation(
        funcOp, rewriter, allocLikeOp, vscaleRange);
    if (!replacement)
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "Replacement : ";
      replacement->dump();
    });
    Value replacementVal = replacement.value();
    rewriter.replaceOp(allocLikeOp, replacementVal);

    for (memref::DeallocOp deallocOp : deallocOps)
      rewriter.eraseOp(deallocOp);
  }
}

DiagnosedSilenceableFailure transform::AIRHoistStaticAllocOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Apply the hoisting pass to all memref.alloc ops in the target function.
  // If more alloc-like ops should be supported, template parameterization
  // allows calling this routine for those as well.
  hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(rewriter, target);
  return DiagnosedSilenceableFailure::success();
}

void transform::AIRHoistStaticAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRresolveTensorOpOperandConflictsWithNewTensors() {
  return std::make_unique<AIRresolveTensorOpOperandConflictsWithNewTensors>();
}

} // namespace air
} // namespace xilinx
