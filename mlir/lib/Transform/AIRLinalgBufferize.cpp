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
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
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

// Configure One-Shot Bufferization from bufferization dialect.
// Ref:
// https://github.com/iree-org/iree/blob/2a123d7b764dc8cc8d89b7591a00f646ab0ade38/compiler/src/iree/compiler/Codegen/Common/IREEComprehensiveBufferizePass.cpp#L145
static bufferization::OneShotBufferizationOptions getBufferizationOptions() {
  bufferization::OneShotBufferizationOptions options;
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToBufferOp>();
  options.unknownTypeConverterFn =
      [](TensorType tensorType, Attribute memorySpace,
         const bufferization::BufferizationOptions &options) {
        if (tensorType.hasStaticShape()) {
          return bufferization::getMemRefTypeWithStaticIdentityLayout(
              tensorType, memorySpace);
        }
        return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                                  memorySpace);
      };

  return options;
}

// Duplicate a `tensor.empty` if it has multiple uses so each use can be
// rewritten independently by downstream passes (e.g., dest-style, CSE).
// Returns success if all duplicates are created and wired to users.
LogicalResult duplicateTensorEmptyOps(OpBuilder &b, tensor::EmptyOp emptyOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(emptyOp);

  // Snapshot uses because we'll mutate operands below.
  SmallVector<OpOperand *> uses = llvm::map_to_vector(
      emptyOp->getUses(), [](OpOperand &use) { return &use; });

  // Keep the first use attached to the original op; clone for the rest.
  for (auto use : llvm::make_range(std::next(uses.begin()), uses.end())) {
    auto newOp = cast<tensor::EmptyOp>(b.clone(*emptyOp.getOperation()));
    Operation *user = use->getOwner();
    user->setOperand(use->getOperandNumber(), newOp);
  }
  return success();
}

// End-to-end helper that:
//   1) Duplicates multi-use `tensor.empty` ops,
//   2) Converts eligible Linalg ops to destination style,
//   3) Runs one-shot bufferization analysis,
//   4) Eliminates `tensor.empty` producers where possible.
//
// This is intended to be invoked on a target `Operation *op` that owns the IR
// region(s) to clean up.
LogicalResult AIRDuplicateAndEliminateEmptyTensors(RewriterBase &rewriter,
                                                   Operation *op) {
  MLIRContext *context = rewriter.getContext();

  // Step 1: duplicate multi-use `tensor.empty` so each user can get a distinct
  // producer (enables better folding and bufferization).
  SmallVector<tensor::EmptyOp> emptyOps;
  op->walk([&](tensor::EmptyOp emptyOp) { emptyOps.push_back(emptyOp); });
  if (llvm::any_of(emptyOps, [&](tensor::EmptyOp emptyOp) {
        return failed(duplicateTensorEmptyOps(rewriter, emptyOp));
      })) {
    return op->emitError("Failed to duplicate empty tensors");
  }

  // Step 2: run "convert to destination style" patterns. This rewrites ops to
  // take explicit output buffers, which makes later bufferization more precise.
  {
    RewritePatternSet patterns(context);
    linalg::populateConvertToDestinationStylePatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return op->emitOpError(
          "Failed in conversion to destination style patterns");
    }
  }

  // Step 3: configure and run one-shot bufferization analysis on `op`.
  auto bufferizationOptions = getBufferizationOptions();
  bufferization::OneShotAnalysisState state(op, bufferizationOptions);
  if (failed(analyzeOp(op, state)))
    return op->emitOpError("Failed in anaylzing op");

  // Step 4: eliminate `tensor.empty` by forwarding to in-place buffers or by
  // replacing with buffer-backed allocations where justified by the analysis.
  if (failed(bufferization::eliminateEmptyTensors(rewriter, op, state)))
    return op->emitOpError("Failed to eliminate empty tensors");

  return success();
}

// Create a linalg::GenericOp version of an n-D copy that can further tile,
// lower to loops or vectorize, unlike the current implementation of
// memref::CopyOp.
// Ref:
// https://github.com/iree-org/iree/blob/2a123d7b764dc8cc8d89b7591a00f646ab0ade38/compiler/src/iree/compiler/Codegen/Utils/Utils.cpp#L980
Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {}) {
  auto memrefTypeFrom = llvm::dyn_cast<MemRefType>(from.getType());
  auto memrefTypeTo = llvm::dyn_cast<MemRefType>(to.getType());
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank()) {
    mlir::emitError(
        loc, "unable to generate copy op within bufferization from type ")
        << memrefTypeFrom << " to " << memrefTypeTo;
    return nullptr;
  }
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      },
      attributes);
}

} // namespace air
} // namespace xilinx

//===----------------------------------------------------------------------===//
// AIRBufferizeOp
//===----------------------------------------------------------------------===//

// Default allocation callback: materialize a memref::AllocOp of the requested
// type/sizes. The `alignment` is ignored by this default (many backends do).
static FailureOr<Value> defaultAllocationFn(OpBuilder &builder, Location loc,
                                            MemRefType allocationType,
                                            ValueRange dynamicSizes,
                                            unsigned int alignment) {
  return builder.create<memref::AllocOp>(loc, allocationType, dynamicSizes)
      .getResult();
}

// Default memcpy callback: emit an n-D linalg.generic copy instead of
// memref.copy so that downstream tiling/vectorization passes have room to act.
static LogicalResult defaultMemCpyFn(OpBuilder &builder, Location loc,
                                     Value from, Value to) {
  Operation *copyOp = createLinalgCopyOp(builder, loc, from, to);
  return success(static_cast<bool>(copyOp));
}

void transform::AIRBufferizeOp::build(OpBuilder &builder,
                                      OperationState &result, Value target) {
  result.addOperands(target);
  MLIRContext *ctx = builder.getContext();
  result.addTypes(pdl::OperationType::get(ctx));
}

DiagnosedSilenceableFailure
transform::AIRBufferizeOp::apply(transform::TransformRewriter &rewriter,
                                 transform::TransformResults &results,
                                 transform::TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());
  Operation *target = *payload.begin();
  ErrorCheckingTrackingListener listener(state, *this);

  // Configure one-shot bufferization for AIR.
  bufferization::OneShotBufferizationOptions options =
      xilinx::air::getBufferizationOptions();
  options.allocationFn = defaultAllocationFn;
  options.memCpyFn = defaultMemCpyFn;
  options.checkParallelRegions = false;
  bufferization::BufferizationState bufferizationState;
  if (failed(bufferization::runOneShotBufferize(target, options,
                                                bufferizationState))) {
    return mlir::emitDefiniteFailure(target, "bufferization failed");
  }

  // Post-bufferization cleanup: Drop dead operands/results introduced by
  // dest-style rewrites.
  {
    RewritePatternSet patterns(getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    GreedyRewriteConfig config;
    config.setListener(&listener);
    LogicalResult result =
        applyOpPatternsGreedily(target, std::move(patterns), config);
    if (failed(result)) {
      return mlir::emitDefiniteFailure(target,
                                       "greedy pattern application failed");
    }
    if (listener.failed())
      return listener.checkAndResetError();
  }

  results.set(getOperation()->getOpResult(0), {target});
  return listener.checkAndResetError();
}

//===----------------------------------------------------------------------===//
// AIRDuplicateAndEliminateEmptyTensorsOp
//===----------------------------------------------------------------------===//

void transform::AIRDuplicateAndEliminateEmptyTensorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::AIRDuplicateAndEliminateEmptyTensorsOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  for (Operation *target : state.getPayloadOps(getTarget())) {
    if (failed(xilinx::air::AIRDuplicateAndEliminateEmptyTensors(rewriter,
                                                                 target)))
      return mlir::emitSilenceableFailure(target->getLoc())
             << "empty tensor elimination failed";
  }
  return DiagnosedSilenceableFailure::success();
}

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRresolveTensorOpOperandConflictsWithNewTensors() {
  return std::make_unique<AIRresolveTensorOpOperandConflictsWithNewTensors>();
}

} // namespace air
} // namespace xilinx
