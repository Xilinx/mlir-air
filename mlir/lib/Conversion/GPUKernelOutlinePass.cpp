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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
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

  static DenseI32ArrayAttr maybeConstantDimsAttr(gpu::KernelDim3 dims) {
    SmallVector<int32_t, 3> constants;
    MLIRContext *ctx = dims.x.getContext();
    for (Value v : {dims.x, dims.y, dims.z}) {
      APInt constValue;
      if (!matchPattern(v, m_ConstantInt(&constValue)))
        return nullptr;
      // In the event someone called for a too-large block or grid dimension,
      // don't set bounds as it is likely to cause more confusing behavior.
      if (constValue.ugt(std::numeric_limits<uint32_t>::max()))
        return nullptr;
      constants.push_back(
          constValue.getLimitedValue(std::numeric_limits<uint32_t>::max()));
    }
    return DenseI32ArrayAttr::get(ctx, constants);
  }

  template <typename OpTy>
  static void createForAllDimensions(OpBuilder &builder, Location loc,
                                     SmallVectorImpl<Value> &values) {
    for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z})
      values.push_back(OpTy::create(builder, loc, builder.getIndexType(), dim));
  }

  /// Adds operations generating block/thread ids and grid/block dimensions at
  /// the beginning of the `launchFuncOpBody` region. Add mapping from argument
  /// in entry block of `launchOpBody`, to the corresponding result value of the
  /// added operations.
  static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody,
                                       Region &launchOpBody, IRMapping &map,
                                       bool hasCluster = false) {
    OpBuilder builder(loc->getContext());
    Block &firstBlock = launchOpBody.front();
    builder.setInsertionPointToStart(&launchFuncOpBody.front());
    SmallVector<Value> indexOps;
    // The order is important here, as it must match the order of the arguments
    createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
    createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
    createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
    createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
    if (hasCluster) {
      createForAllDimensions<gpu::ClusterIdOp>(builder, loc, indexOps);
      createForAllDimensions<gpu::ClusterDimOp>(builder, loc, indexOps);
    }
    // Replace the leading 12 function args with the respective thread/block
    // index operations. Iterate backwards since args are erased and indices
    // change.
    for (const auto &indexOp : enumerate(indexOps))
      map.map(firstBlock.getArgument(indexOp.index()), indexOp.value());
  }

  static gpu::GPUFuncOp
  outlineKernelFuncImpl(gpu::LaunchOp launchOp, StringRef kernelFnName,
                        SetVector<Value> &operands,
                        SetVector<Value> &filteredOperands) {
    Location loc = launchOp.getLoc();
    // Create a builder with no insertion point, insertion will happen
    // separately due to symbol table manipulation.
    OpBuilder builder(launchOp.getContext());
    Region &launchOpBody = launchOp.getBody();

    // Identify uses from values defined outside of the scope of the launch
    // operation.
    mlir::getUsedValuesDefinedAbove(launchOpBody, operands);

    DenseMap<Value, Attribute> constantValues;

    for (Value operand : operands) {
      if (auto defOp = operand.getDefiningOp()) {
        if (auto constOp = dyn_cast_if_present<arith::ConstantOp>(defOp)) {
          // Record the constant value for later inlining
          constantValues[operand] = constOp.getValue();
          continue; // Don't pass this constant as an argument
        }
      }
      filteredOperands.insert(operand); // Only non-constant operands are passed
    }

    // Create the gpu.func operation.
    SmallVector<Type, 4> kernelOperandTypes;
    kernelOperandTypes.reserve(filteredOperands.size());
    for (Value operand : filteredOperands) {
      kernelOperandTypes.push_back(operand.getType());
    }
    FunctionType type =
        FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
    auto outlinedFunc = gpu::GPUFuncOp::create(
        builder, loc, kernelFnName, type,
        TypeRange(ValueRange(launchOp.getWorkgroupAttributionBBArgs())),
        TypeRange(ValueRange(launchOp.getPrivateAttributions())));
    outlinedFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                          builder.getUnitAttr());

    // If we can infer bounds on the grid and/or block sizes from the arguments
    // to the launch op, propagate them to the generated kernel. This is safe
    // because multiple launches with the same body are not deduplicated.
    if (auto blockBounds =
            maybeConstantDimsAttr(launchOp.getBlockSizeOperandValues()))
      outlinedFunc.setKnownBlockSizeAttr(blockBounds);
    if (auto gridBounds =
            maybeConstantDimsAttr(launchOp.getGridSizeOperandValues()))
      outlinedFunc.setKnownGridSizeAttr(gridBounds);

    IRMapping map;

    // Map the arguments corresponding to the launch parameters like blockIdx,
    // threadIdx, etc. If cluster is present, then we also generate clusterIdx
    // and clusterDim.
    Region &outlinedFuncBody = outlinedFunc.getBody();
    injectGpuIndexOperations(loc, outlinedFuncBody, launchOpBody, map,
                             launchOp.hasClusterSize());

    // Map memory attributions from the LaunOp op to the GPUFuncOp attributions.
    for (const auto &[launchArg, funcArg] :
         llvm::zip(launchOp.getWorkgroupAttributionBBArgs(),
                   outlinedFunc.getWorkgroupAttributionBBArgs()))
      map.map(launchArg, funcArg);
    for (const auto &[launchArg, funcArg] :
         llvm::zip(launchOp.getPrivateAttributions(),
                   outlinedFunc.getPrivateAttributions()))
      map.map(launchArg, funcArg);

    // Map arguments from gpu.launch region to the arguments of the gpu.func
    // operation.
    Block &entryBlock = outlinedFuncBody.front();
    for (const auto &operand : enumerate(filteredOperands))
      map.map(operand.value(), entryBlock.getArgument(operand.index()));

    for (auto [originalValue, constAttr] : constantValues) {
      // Get the location of original constant for accurate IR tracing
      Location constLoc = originalValue.getLoc();
      OpBuilder constBuilder =
          OpBuilder::atBlockBegin(&outlinedFunc.getBody().front());

      Value newConst = arith::ConstantOp::create(
          constBuilder, constLoc, llvm::cast<TypedAttr>(constAttr));

      // Update the mapping so that cloned uses map to new constant
      map.map(originalValue, newConst);
    }

    launchOpBody.cloneInto(&outlinedFuncBody, map);

    // Replace the terminator op with returns.
    for (Block &block : launchOpBody) {
      Block *clonedBlock = map.lookup(&block);
      auto terminator =
          dyn_cast_if_present<gpu::TerminatorOp>(clonedBlock->getTerminator());
      if (!terminator)
        continue;
      OpBuilder replacer(terminator);
      gpu::ReturnOp::create(replacer, terminator->getLoc());
      terminator->erase();
    }

    // Splice now the entry block of the gpu.launch operation at the end of the
    // gpu.func entry block and erase the redundant block.
    Block *clonedLaunchOpEntry = map.lookup(&launchOpBody.front());
    entryBlock.getOperations().splice(entryBlock.getOperations().end(),
                                      clonedLaunchOpEntry->getOperations());
    clonedLaunchOpEntry->erase();

    return outlinedFunc;
  }

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

        SetVector<Value> operands;
        SetVector<Value> filteredOperands;
        std::string gfname = func.getName().str();
        gfname += "_module";
        // create a GPUModuleOp in case the GPU module specified does not exist.
        auto gpuModule = gpu::GPUModuleOp::create(b, loc, gfname);

        // add the GPUModuleOp into the symbol table.
        SymbolTable symbolTable(op);
        symbolTable.insert(gpuModule);

        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFuncImpl(launchOp, gfname, operands, filteredOperands);
        SymbolTable gpuModuleSymbolTable(gpuModule);
        // insert the GPUFuncOp into GPUModuleOp.
        gpuModuleSymbolTable.insert(outlinedFunc);
        convertToLaunchFuncOp(launchOp, outlinedFunc,
                              filteredOperands.getArrayRef());
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
