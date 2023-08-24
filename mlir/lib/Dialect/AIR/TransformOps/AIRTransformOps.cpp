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

//===----------------------------------------------------------------------===//
// GetSegmentForOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetSegmentForOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  SetVector<Operation *> segments;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    xilinx::air::SegmentOp segment =
        target->getParentOfType<xilinx::air::SegmentOp>();
    if (!segment) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "could not find an '" << xilinx::air::SegmentOp::getOperationName()
          << "' parent";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    segments.insert(segment);
  }
  results.set(getResult().cast<OpResult>(), segments.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// SegmentToAIEOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SegmentToAIEOp::applyToOne(transform::TransformRewriter &rewriter,
                                      xilinx::air::SegmentOp target,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  FailureOr<ModuleOp> res = convertAIRToAIE(rewriter, target);
  if (failed(res))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.push_back(res->getOperation());
  return DiagnosedSilenceableFailure::success();
}

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
