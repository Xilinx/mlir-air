//===- AIRLinalgBufferize.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRLinalgBufferize.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRresolveTensorOpOperandConflictsWithNewTensors() {
  return std::make_unique<AIRresolveTensorOpOperandConflictsWithNewTensors>();
}

} // namespace air
} // namespace xilinx
