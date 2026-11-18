//===- AIRTransformIntepreter.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRTransformInterpreter.h"

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "air-transform-interpreter"

using namespace mlir;

namespace {

class AIRTransformInterpreterPass
    : public xilinx::air::impl::AIRTransformInterpreterPassBase<
          AIRTransformInterpreterPass> {

public:
  AIRTransformInterpreterPass() = default;
  AIRTransformInterpreterPass(const AIRTransformInterpreterPass &pass){};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect, transform::TransformDialect>();
  }

  void runOnOperation() override {
    auto payload = getOperation();
    auto ctx = payload->getContext();

    if (clTransformFileName.empty()) {
      emitError(UnknownLoc::get(ctx), "no transform file name specified");
      return signalPassFailure();
    }
    OwningOpRef<ModuleOp> transformModule;
    if (failed(transform::detail::parseTransformModuleFromFile(
            ctx, clTransformFileName, transformModule))) {
      signalPassFailure();
      return;
    }
    if (failed(xilinx::air::runAIRTransform(transformModule.get(), payload))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

/// Find the __transform_main named sequence in a module, searching both
/// the top level and any nested modules with transform.with_named_sequence.
static transform::NamedSequenceOp findEntryPoint(ModuleOp transformModule) {
  // Check the top-level module directly.
  if (auto namedSeq = transformModule.lookupSymbol<transform::NamedSequenceOp>(
          transform::TransformDialect::kTransformEntryPointSymbolName))
    return namedSeq;

  // Check nested modules (the parsed file may wrap the transform module).
  for (auto nestedModule : transformModule.getBody()->getOps<ModuleOp>()) {
    if (auto namedSeq = nestedModule.lookupSymbol<transform::NamedSequenceOp>(
            transform::TransformDialect::kTransformEntryPointSymbolName))
      return namedSeq;
  }
  return nullptr;
}

LogicalResult xilinx::air::runAIRTransform(ModuleOp transformModule,
                                           ModuleOp payloadModule) {
  // Expensive checks are disabled because the handle invalidation tracker
  // segfaults with transform.include: it accesses erased payload operations
  // via dangling pointers when validating handles across region scopes.
  // See https://github.com/Xilinx/mlir-air/issues/1464.
  transform::TransformOptions options;
  options.enableExpensiveChecks(false);

  // Try named_sequence entry point first (modern transform dialect style).
  if (auto namedSeq = findEntryPoint(transformModule)) {
    return transform::applyTransforms(payloadModule, namedSeq, {}, options,
                                      /*enforceToplevelTransformOp=*/false);
  }

  // Fallback: iterate top-level TransformOpInterface ops (legacy style).
  for (auto op :
       transformModule.getBody()->getOps<transform::TransformOpInterface>()) {
    if (isa<transform::NamedSequenceOp>(op))
      continue;
    if (failed(transform::applyTransforms(payloadModule, op, {}, options)))
      return failure();
  }
  return success();
}

std::unique_ptr<Pass> xilinx::air::createAIRTransformInterpreterPass() {
  return std::make_unique<AIRTransformInterpreterPass>();
}