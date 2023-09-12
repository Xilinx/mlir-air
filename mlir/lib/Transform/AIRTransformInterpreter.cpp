//===- AIRTransformIntepreter.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRTransformInterpreter.h"
#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "air-transform-interpreter"

using namespace mlir;
using namespace xilinx;

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
    : public xilinx::air::AIRTransformInterpreterPassBase<
          AIRTransformInterpreterPass> {

public:
  AIRTransformInterpreterPass() = default;
  AIRTransformInterpreterPass(const AIRTransformInterpreterPass &pass){};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect, transform::TransformDialect>();
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

LogicalResult xilinx::air::runAIRTransform(ModuleOp transformModule,
                                           ModuleOp payloadModule) {
  for (auto op :
       transformModule.getBody()->getOps<transform::TransformOpInterface>()) {
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