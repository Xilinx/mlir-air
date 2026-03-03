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
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "air-transform-interpreter"

using namespace mlir;

/// Utility to parse the content of a `transformFileName` mlir file containing
/// a transform dialect specification.
/// Modified from IREE
static LogicalResult
parseTransformModuleFromFile(MLIRContext *context,
                             llvm::StringRef transformFileName,
                             OwningOpRef<ModuleOp> &transformModule) {
  if (transformFileName.empty()) {
    emitError(UnknownLoc::get(context), "no transform file name specified");
    return failure();
  }
  // Parse transformFileName content into a ModuleOp.
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(transformFileName, &errorMessage);
  if (!memoryBuffer) {
    emitError(FileLineColLoc::get(context, transformFileName, 0, 0),
              "failed to open transform file: " + transformFileName + "\n");
    return failure();
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  transformModule =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, context));
  if (!transformModule)
    return failure();

  return success();
}

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

    OwningOpRef<ModuleOp> transformModule;
    if (failed(parseTransformModuleFromFile(ctx, clTransformFileName,
                                            transformModule))) {
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
  // Try named_sequence entry point first (modern transform dialect style).
  if (auto namedSeq = findEntryPoint(transformModule)) {
    return transform::applyTransforms(
        payloadModule, namedSeq, {},
        transform::TransformOptions().enableExpensiveChecks(true),
        /*enforceToplevelTransformOp=*/false);
  }

  // Fallback: iterate top-level TransformOpInterface ops (legacy style).
  for (auto op :
       transformModule.getBody()->getOps<transform::TransformOpInterface>()) {
    if (isa<transform::NamedSequenceOp>(op))
      continue;
    if (failed(transform::applyTransforms(
            payloadModule, op, {},
            transform::TransformOptions().enableExpensiveChecks(
                /*enableExpensiveChecks=*/true))))
      return failure();
  }
  return success();
}

std::unique_ptr<Pass> xilinx::air::createAIRTransformInterpreterPass() {
  return std::make_unique<AIRTransformInterpreterPass>();
}