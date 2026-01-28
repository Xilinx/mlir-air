//===- Passes.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Conversion/Passes.h.inc"
} // namespace

void xilinx::air::registerConversionPasses() { ::registerPasses(); }

#ifndef AIR_ENABLE_AIE
// Stub implementations for AIE-dependent passes when AIE is not enabled.
// These passes will emit an error if invoked.

namespace xilinx {
namespace air {

namespace {
struct AIRToAIEStubPass
    : public PassWrapper<AIRToAIEStubPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRToAIEStubPass)
  StringRef getArgument() const override { return "air-to-aie"; }
  StringRef getDescription() const override {
    return "AIR to AIE lowering (stub - requires AIE support)";
  }
  void runOnOperation() override {
    getOperation().emitError("AIRToAIE pass requires AIE support. "
                             "Rebuild with -DAIR_ENABLE_AIE=ON");
    signalPassFailure();
  }
};

struct AIRLoweringStubPass
    : public PassWrapper<AIRLoweringStubPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRLoweringStubPass)
  StringRef getArgument() const override { return "air-lowering"; }
  StringRef getDescription() const override {
    return "AIR lowering (stub - requires AIE support)";
  }
  void runOnOperation() override {
    getOperation().emitError("AIRLowering pass requires AIE support. "
                             "Rebuild with -DAIR_ENABLE_AIE=ON");
    signalPassFailure();
  }
};

struct AIRLinalgToFuncStubPass
    : public PassWrapper<AIRLinalgToFuncStubPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRLinalgToFuncStubPass)
  StringRef getArgument() const override { return "air-linalg-to-func"; }
  StringRef getDescription() const override {
    return "AIR linalg to func (stub - requires AIE support)";
  }
  void runOnOperation() override {
    getOperation().emitError("AIRLinalgToFunc pass requires AIE support. "
                             "Rebuild with -DAIR_ENABLE_AIE=ON");
    signalPassFailure();
  }
};

struct AIRSplitDevicesStubPass
    : public PassWrapper<AIRSplitDevicesStubPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRSplitDevicesStubPass)
  StringRef getArgument() const override { return "air-split-devices"; }
  StringRef getDescription() const override {
    return "AIR split devices (stub - requires AIE support)";
  }
  void runOnOperation() override {
    getOperation().emitError("AIRSplitDevices pass requires AIE support. "
                             "Rebuild with -DAIR_ENABLE_AIE=ON");
    signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createAIRToAIEPass() {
  return std::make_unique<AIRToAIEStubPass>();
}

std::unique_ptr<Pass> createAIRLoweringPass() {
  return std::make_unique<AIRLoweringStubPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAIRLinalgToFuncPass() {
  return std::make_unique<AIRLinalgToFuncStubPass>();
}

std::unique_ptr<Pass> createAIRSplitDevicesPass() {
  return std::make_unique<AIRSplitDevicesStubPass>();
}

} // namespace air

namespace airrt {

namespace {
struct AIRRtToNpuStubPass
    : public PassWrapper<AIRRtToNpuStubPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRRtToNpuStubPass)
  StringRef getArgument() const override { return "airrt-to-npu"; }
  StringRef getDescription() const override {
    return "AIRRt to NPU (stub - requires AIE support)";
  }
  void runOnOperation() override {
    getOperation().emitError("AIRRtToNpu pass requires AIE support. "
                             "Rebuild with -DAIR_ENABLE_AIE=ON");
    signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createAIRRtToNpuPass() {
  return std::make_unique<AIRRtToNpuStubPass>();
}

} // namespace airrt
} // namespace xilinx
#endif // AIR_ENABLE_AIE
