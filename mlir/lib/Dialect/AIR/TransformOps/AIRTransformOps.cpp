//===- AIRTransformOps.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Conversion/AIRToAIEPass.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

using namespace mlir;

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// GetPartitionForOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetPartitionForOp::apply(transform::TransformResults &results,
                                    transform::TransformState &state) {
  SetVector<Operation *> partitions;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    xilinx::air::PartitionOp partition =
        target->getParentOfType<xilinx::air::PartitionOp>();
    if (!partition) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "could not find an '"
          << xilinx::air::PartitionOp::getOperationName() << "' parent";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    partitions.insert(partition);
  }
  results.set(getResult().cast<OpResult>(), partitions.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PartitionToAIEOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PartitionToAIEOp::applyToOne(xilinx::air::PartitionOp target,
                                        SmallVectorImpl<Operation *> &results,
                                        transform::TransformState &state) {
  SimpleRewriter rewriter(target->getContext());
  FailureOr<ModuleOp> res = convertAIRToAIE(rewriter, target);
  if (failed(res))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.push_back(res->getOperation());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MapSubviewsOp
//===----------------------------------------------------------------------===//

// DiagnosedSilenceableFailure
// transform::MapSubviewsOp::apply(transform::TransformResults &results,
//                                     transform::TransformState &state) {
//   SetVector<Operation *> memrefs;
//   for (Operation *target : state.getPayloadOps(getTarget())) {
//     MLIRContext *ctx = target->getParentOfType<func::FuncOp>().getContext();
//     RewritePatternSet patterns(ctx);
//     patterns.insert<RemoveSubViewOpsPattern, FoldSubViewOpsPattern,
//                     RemoveViewOpsPattern, HoistReduceBufferPattern>(ctx);
//     (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
//   }
//   results.set(getResult().cast<OpResult>(), memrefs.getArrayRef());
//   return DiagnosedSilenceableFailure::success();
// }

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class AIRTransformDialectExtension
    : public transform::TransformDialectExtension<
          AIRTransformDialectExtension> {
public:
  AIRTransformDialectExtension() {
    declareDependentDialect<func::FuncDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "air/Dialect/AIR/AIRTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIRTransformOps.cpp.inc"

void xilinx::air::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AIRTransformDialectExtension>();
}
