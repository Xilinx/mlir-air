//===- GPUKernelOutlinePass.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------===//
#include "air/Conversion/GPUKernelOutlinePass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {
#define GEN_PASS_DEF_CONVERTGPUKERNELOUTLINE
#include "air/Conversion/Passes.h.inc"

struct ConvertGPUKernelOutlinePass
    : public xilinx::air::impl::ConvertGPUKernelOutlineBase<
          ConvertGPUKernelOutlinePass> {

  ConvertGPUKernelOutlinePass() = default;
  ConvertGPUKernelOutlinePass(const ConvertGPUKernelOutlinePass &pass) {}

  static void convertToLaunchFuncOp(gpu::LaunchOp launchOp,
                                    gpu::GPUFuncOp kernelFunc,
                                    ValueRange operands) {
    OpBuilder builder(launchOp);
    // The launch op has an optional dynamic shared memory size. If it doesn't
    // exist, we use zero.
    Value asyncToken = launchOp.getAsyncToken();
    std::optional<gpu::KernelDim3> clusterSize =
        launchOp.getClusterSizeOperandValues();
    auto launchFunc = gpu::LaunchFuncOp::create(
        builder, launchOp.getLoc(), kernelFunc,
        launchOp.getGridSizeOperandValues(),
        launchOp.getBlockSizeOperandValues(),
        launchOp.getDynamicSharedMemorySize(), operands,
        asyncToken ? asyncToken.getType() : nullptr,
        launchOp.getAsyncDependencies(), /*asyncObject=*/nullptr, clusterSize);
    launchOp.replaceAllUsesWith(launchFunc);
    launchOp.erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module.getContext());
    OpBuilder builder(module.getContext());

    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createSCFToControlFlowPass());

    if (failed(pm.run(module))) {
      module.emitError("Sub-pipeline failed in LowerRockOpsToGPUPass");
      signalPassFailure();
    }

    SymbolTable symbolTable(getOperation());
    module.walk([&](func::FuncOp func) {
      func.walk([&](gpu::LaunchOp launchOp) {
        ModuleOp op = getOperation();
        MLIRContext *ctx = op.getContext();
        OpBuilder b(ctx);
        Location loc = op.getLoc();

        // Annotate this module as a container module.
        op->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    UnitAttr::get(ctx));

        // Constants are rebuilt inside the kernel instead of being passed
        // as arguments.
        (void)sinkOperationsIntoLaunchOp(
            launchOp, [](Operation *op) { return isa<arith::ConstantOp>(op); });
        SmallVector<Value> operands;
        std::string gfname = func.getName().str();
        gfname += "_module";
        // create a GPUModuleOp in case the GPU module specified does not exist.
        auto gpuModule = gpu::GPUModuleOp::create(b, loc, gfname);

        // add the GPUModuleOp into the symbol table.
        SymbolTable symbolTable(op);
        symbolTable.insert(gpuModule);

        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFunc(launchOp, gfname, operands);
        SymbolTable gpuModuleSymbolTable(gpuModule);
        // insert the GPUFuncOp into GPUModuleOp.
        gpuModuleSymbolTable.insert(outlinedFunc);
        convertToLaunchFuncOp(launchOp, outlinedFunc, operands);
      });
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>(),
        registry.insert<mlir::scf::SCFDialect>(),
        registry.insert<mlir::arith::ArithDialect>(),
        registry.insert<mlir::cf::ControlFlowDialect>(),
        registry.insert<mlir::memref::MemRefDialect>(),
        registry.insert<mlir::func::FuncDialect>(),
        registry.insert<mlir::vector::VectorDialect>(), // If used anywhere
        registry.insert<mlir::ROCDL::ROCDLDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<scf::SCFDialect>();
  }
};
} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createGPUKernelOutlinePass() {
  return std::make_unique<ConvertGPUKernelOutlinePass>();
}

} // namespace air
} // namespace xilinx
