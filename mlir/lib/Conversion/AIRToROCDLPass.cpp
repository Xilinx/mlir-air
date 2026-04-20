//===- AIRToROCDLPass.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-------------------------------------------------------------------===//
#include "air/Conversion/AIRToROCDLPass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h" // Includes the ops like scf::ForOp
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
#define GEN_PASS_DEF_CONVERTAIRTOROCDL
#include "air/Conversion/Passes.h.inc"

SmallVector<mlir::BlockArgument, 4> gpuArgs;
class AffineApplyToSubPattern
    : public mlir::OpRewritePattern<mlir::affine::AffineApplyOp> {
  using OpRewritePattern<mlir::affine::AffineApplyOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::affine::AffineApplyOp affineOp,
                  mlir::PatternRewriter &rewriter) const override {

    for (unsigned j = 0; j < affineOp.getNumOperands(); ++j) {
      mlir::Value nestedOperand = affineOp.getOperand(j);

      if (auto blockArg =
              mlir::dyn_cast_if_present<mlir::BlockArgument>(nestedOperand)) {
        mlir::Block *parentBlock = blockArg.getOwner();

        mlir::Operation *parentOp = parentBlock->getParentOp();
        if (llvm::isa<air::SegmentOp>(parentOp)) {
          unsigned argNumber = blockArg.getArgNumber();
          affineOp.setOperand(j, gpuArgs[j + argNumber]);
        } else if (llvm::isa<air::HerdOp>(parentOp)) {
          unsigned argNumber = blockArg.getArgNumber();
          affineOp.setOperand(j, gpuArgs[j + 3 + argNumber]);
        }
      }
    }

    return mlir::success();
  }
};

class SCFForToSubPattern : public mlir::OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Region &region = forOp.getRegion();

    mlir::Block &entryBlock = region.front();
    for (mlir::Operation &nestedOp : entryBlock.getOperations()) {
      if (auto storeOp =
              llvm::dyn_cast_if_present<mlir::memref::StoreOp>(nestedOp)) {
        mlir::Value memref = storeOp.getMemRef();
        llvm::SmallVector<mlir::Value> indices(storeOp.getIndices().begin(),
                                               storeOp.getIndices().end());

        if (auto blockArg =
                mlir::dyn_cast_if_present<mlir::BlockArgument>(memref)) {
          mlir::Block *parentBlock = blockArg.getOwner();
          mlir::Operation *parentOp = parentBlock->getParentOp();
          Type operandType = memref.getType();
          if (auto memRefType =
                  mlir::dyn_cast_if_present<MemRefType>(operandType)) {
            unsigned argNumber = blockArg.getArgNumber();
            if (auto segmentOp =
                    mlir::dyn_cast_if_present<air::SegmentOp>(parentOp)) {
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                mlir::Value correspondingValue = segmentOp.getKernelOperand(
                    argNumber); // Map back to the original value
                storeOp->setOperand(1, correspondingValue);
              }
            } else if (auto herdOp =
                           mlir::dyn_cast_if_present<air::HerdOp>(parentOp)) {
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                mlir::Value correspondingValue =
                    herdOp.getKernelOperand(argNumber - 4);
                storeOp->setOperand(1, correspondingValue);
              }
            } else if (auto launchOp =
                           mlir::dyn_cast_if_present<air::LaunchOp>(parentOp)) {
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                mlir::Value correspondingValue =
                    launchOp.getKernelOperand(argNumber - 4);
                storeOp->setOperand(1, correspondingValue);
              }
            }
          }
        }
      }
      for (mlir::Value operand : nestedOp.getOperands()) {
        if (auto opResult = dyn_cast_if_present<mlir::OpResult>(operand)) {
          mlir::Operation *op = operand.getDefiningOp();
          for (unsigned j = 0; j < op->getNumOperands(); ++j) {
            mlir::Value nestedOperand = op->getOperand(j);

            if (auto blockArg = mlir::dyn_cast_if_present<mlir::BlockArgument>(
                    nestedOperand)) {
              mlir::Block *parentBlock = blockArg.getOwner();
              mlir::Operation *parentOp = parentBlock->getParentOp();
              Type operandType = nestedOperand.getType();
              if (auto memRefType =
                      mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                unsigned argNumber = blockArg.getArgNumber();
                if (auto segmentOp =
                        mlir::dyn_cast_if_present<air::SegmentOp>(parentOp)) {
                  if (auto memRefType =
                          mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                    mlir::Value correspondingValue = segmentOp.getKernelOperand(
                        argNumber); // Map back to the original value
                    op->setOperand(j, correspondingValue);
                  }
                } else if (auto herdOp = mlir::dyn_cast_if_present<air::HerdOp>(
                               parentOp)) {
                  if (auto memRefType =
                          mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                    mlir::Value correspondingValue =
                        herdOp.getKernelOperand(argNumber - 4);
                    op->setOperand(j, correspondingValue);
                  }
                } else if (auto launchOp =
                               mlir::dyn_cast_if_present<air::LaunchOp>(
                                   parentOp)) {
                  if (auto memRefType =
                          mlir::dyn_cast_if_present<MemRefType>(operandType)) {
                    mlir::Value correspondingValue =
                        launchOp.getKernelOperand(argNumber - 4);
                    op->setOperand(j, correspondingValue);
                  }
                }

              } else {
                if (llvm::isa<air::SegmentOp>(parentOp)) {
                  unsigned argNumber = blockArg.getArgNumber();
                  op->setOperand(j, gpuArgs[j + argNumber]);
                } else if (llvm::isa<air::HerdOp>(parentOp)) {
                  unsigned argNumber = blockArg.getArgNumber();
                  op->setOperand(j, gpuArgs[j + 3 + argNumber]);
                }
              }
            }
          }
        }
      }
    }

    return mlir::success();
  }
};
class DMAMemcpyToSubPattern
    : public mlir::OpRewritePattern<air::DmaMemcpyNdOp> {
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;

  void replaceDMAOperand(BlockArgument blockArg,
                         air::DmaMemcpyNdOp dmaOp) const {
    int numOp = -1;
    mlir::Block *parentBlock = blockArg.getOwner();
    Type operandType = blockArg.getType();
    unsigned argNumber = blockArg.getArgNumber(); // Get the argument number

    for (unsigned i = 0; i < dmaOp->getNumOperands(); ++i) {
      if (dmaOp->getOperand(i) == blockArg) {
        numOp = i; // Found the operand index
      }
    }
    // Get the parent operation of the block
    mlir::Operation *parentOp = parentBlock->getParentOp();
    if (auto segmentOp = mlir::dyn_cast_if_present<air::SegmentOp>(parentOp)) {
      if (auto memRefType =
              mlir::dyn_cast_if_present<MemRefType>(operandType)) {
        mlir::Value correspondingValue = segmentOp.getKernelOperand(
            argNumber); // Map back to the original value
        dmaOp.setOperand(numOp, correspondingValue);
      }
    } else if (auto herdOp = mlir::dyn_cast_if_present<air::HerdOp>(parentOp)) {
      if (auto memRefType =
              mlir::dyn_cast_if_present<MemRefType>(operandType)) {
        mlir::Value correspondingValue = herdOp.getKernelOperand(argNumber - 4);
        dmaOp.setOperand(numOp, correspondingValue);
      }
    } else if (auto launchOp =
                   mlir::dyn_cast_if_present<air::LaunchOp>(parentOp)) {
      if (auto memRefType =
              mlir::dyn_cast_if_present<MemRefType>(operandType)) {
        mlir::Value correspondingValue =
            launchOp.getKernelOperand(argNumber - 4);
        dmaOp.setOperand(numOp, correspondingValue);
      }
    }
  }

  mlir::LogicalResult
  matchAndRewrite(air::DmaMemcpyNdOp dmaOp,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Value sourceMemRef = dmaOp.getSrcMemref();
    mlir::Value dstMemRef = dmaOp.getDstMemref();
    if (auto blockArg =
            mlir::dyn_cast_if_present<BlockArgument>(sourceMemRef)) {
      replaceDMAOperand(blockArg, dmaOp);
    }
    if (auto blockArg = mlir::dyn_cast_if_present<BlockArgument>(dstMemRef)) {
      replaceDMAOperand(blockArg, dmaOp);
    }

    return mlir::success();
  }
};

struct ConvertAIRToROCDLPass
    : public xilinx::air::impl::ConvertAIRToROCDLBase<ConvertAIRToROCDLPass> {

  ConvertAIRToROCDLPass() = default;
  ConvertAIRToROCDLPass(const ConvertAIRToROCDLPass &pass) {}
  SmallVector<Value, 4> blkIdx;
  SmallVector<Value, 4> gridIdx;

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

  static gpu::GPUFuncOp outlineKernelFuncImpl(gpu::LaunchOp launchOp,
                                              StringRef kernelFnName,
                                              SetVector<Value> &operands) {
    Location loc = launchOp.getLoc();
    // Create a builder with no insertion point, insertion will happen
    // separately due to symbol table manipulation.
    OpBuilder builder(launchOp.getContext());
    Region &launchOpBody = launchOp.getBody();

    // Identify uses from values defined outside of the scope of the launch
    // operation.
    mlir::getUsedValuesDefinedAbove(launchOpBody, operands);

    // Create the gpu.func operation.
    SmallVector<Type, 4> kernelOperandTypes;
    kernelOperandTypes.reserve(operands.size());
    for (Value operand : operands) {
      kernelOperandTypes.push_back(operand.getType());
    }
    FunctionType type =
        FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
    auto outlinedFunc = gpu::GPUFuncOp::create(
        builder, loc, kernelFnName, type,
        TypeRange(ValueRange(launchOp.getWorkgroupAttributions())),
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
         llvm::zip(launchOp.getWorkgroupAttributions(),
                   outlinedFunc.getWorkgroupAttributions()))
      map.map(launchArg, funcArg);
    for (const auto &[launchArg, funcArg] :
         llvm::zip(launchOp.getPrivateAttributions(),
                   outlinedFunc.getPrivateAttributions()))
      map.map(launchArg, funcArg);

    // Map arguments from gpu.launch region to the arguments of the gpu.func
    // operation.
    Block &entryBlock = outlinedFuncBody.front();
    for (const auto &operand : enumerate(operands))
      map.map(operand.value(), entryBlock.getArgument(operand.index()));

    // Clone the region of the gpu.launch operation into the gpu.func operation.
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
        launchOp.getAsyncDependencies(), clusterSize);
    launchOp.replaceAllUsesWith(launchFunc);
    launchOp.erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module.getContext());
    OpBuilder builder(module.getContext());
    mlir::ModuleOp moduleOp = getOperation();
    Value gridXVal, gridYVal;

    // Create a pattern rewriter to apply transformations to each function
    PatternRewriter rewriter(moduleOp.getContext());
    // Create a set of patterns for transformation
    RewritePatternSet patterns(&getContext());
    patterns.add<AffineApplyToSubPattern, DMAMemcpyToSubPattern,
                 SCFForToSubPattern>(&getContext());
    // Traverse the module and look for air.launch and air.herd ops.
    module.walk([&](air::LaunchOp launchOp) {
      launchOp.walk([&](air::SegmentOp segmentOp) {
        segmentOp.walk([&](Operation *childOp) {
          if (auto herdOp = dyn_cast_if_present<xilinx::air::HerdOp>(childOp)) {

            gridXVal = launchOp.getSizeOperands()[0];
            gridYVal = launchOp.getSizeOperands()[1];
            blkIdx.push_back(herdOp.getSizeOperands()[0]);
            blkIdx.push_back(herdOp.getSizeOperands()[1]);
            gridIdx.push_back(launchOp.getSizeOperands()[0]);
            gridIdx.push_back(launchOp.getSizeOperands()[1]);
          }
        });
        gpu::LaunchOp gpuLaunchOp =
            convertLaunchToGPULaunch(launchOp, builder, gridXVal, gridYVal);
        Block &gpuLaunchBlock = gpuLaunchOp.getBody().front();
        auto blockArgs = gpuLaunchBlock.getArguments();

        gpuArgs.assign(blockArgs.begin(), blockArgs.end());
        (void)applyPatternsGreedily(launchOp, std::move(patterns));
        deleteAirHerd(segmentOp, builder, gpuLaunchOp);
        deleteAirSegment(launchOp, builder, gpuLaunchOp);
      });
    });

    module.walk([&](gpu::LaunchOp gpuLaunchOp) {
      module.walk([&](air::LaunchOp launchOp) {
        Block &launchBlock = launchOp.getRegion().front();
        mlir::Block &block = gpuLaunchOp.getBody().front();
        for (auto &operation :
             llvm::make_early_inc_range(launchBlock.without_terminator())) {
          // Iterate over each operand of the operation
          mlir::Operation &lastOp = block.back();
          operation.moveBefore(&lastOp);
        }
      });
      hoistAlloc(gpuLaunchOp, builder);
    });
    module.walk([&](air::DmaMemcpyNdOp dmaOp) {
      convertDMAToGPUMemcpy(dmaOp, builder);
    });
    module.walk([&](air::LaunchOp launchOp) {
      Block &launchBlock = launchOp.getBody().front();
      launchBlock.getTerminator()->erase(); // Erase the terminator
      launchOp.erase();                     // Erase the herd operation
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

  // Function to print detailed information about a Value (including block
  // arguments)
  void printValueDetails(Value val) {
    if (auto blockArg = mlir::dyn_cast_if_present<BlockArgument>(val)) {
      llvm::outs() << "Block argument: index=" << blockArg.getArgNumber()
                   << " type=" << blockArg.getType() << "\n";
    } else {
      llvm::outs() << "Operation result: " << val << "\n";
    }
  }

  void deleteAirSegment(air::LaunchOp launchOp, OpBuilder &builder,
                        gpu::LaunchOp gpuLaunchOp) {
    launchOp.walk([&](Operation *childOp) {
      if (auto segmentOp =
              dyn_cast_if_present<xilinx::air::SegmentOp>(childOp)) {
        if (!segmentOp.getRegion().empty()) {
          Block &segmentBlock = segmentOp.getRegion().front();

          // Remap segment block arguments to kernel operands.
          unsigned numKernelArgs = segmentOp.getNumKernelOperands();
          for (unsigned i = 0; i < numKernelArgs; ++i) {
            Value outerVal = segmentOp.getKernelOperand(i);
            segmentBlock.getArgument(i).replaceAllUsesWith(outerVal);
          }

          for (auto &operation :
               llvm::make_early_inc_range(segmentBlock.without_terminator())) {
            operation.moveBefore(segmentOp);
          }
          segmentBlock.getTerminator()->erase();
          segmentOp.erase();
        }
      }
    });
  }

  void deleteAirHerd(xilinx::air::SegmentOp segmentOp, OpBuilder &builder,
                     gpu::LaunchOp gpuLaunchOp) {
    segmentOp.walk([&](Operation *childOp) {
      if (auto herdOp = dyn_cast_if_present<xilinx::air::HerdOp>(childOp)) {
        if (!herdOp.getRegion().empty()) {
          Block &herdBlock = herdOp.getRegion().front();
          Location loc = herdOp.getLoc();

          // Remap herd block arguments before moving ops out.
          // Block args layout: [tile_x, tile_y, size_x, size_y, kernel_args...]
          builder.setInsertionPoint(herdOp);
          Value tidx = gpu::ThreadIdOp::create(
              builder, loc, builder.getIndexType(), gpu::Dimension::x);
          Value tidy = gpu::ThreadIdOp::create(
              builder, loc, builder.getIndexType(), gpu::Dimension::y);
          Value bdimx = gpu::BlockDimOp::create(
              builder, loc, builder.getIndexType(), gpu::Dimension::x);
          Value bdimy = gpu::BlockDimOp::create(
              builder, loc, builder.getIndexType(), gpu::Dimension::y);

          herdBlock.getArgument(0).replaceAllUsesWith(tidx);
          herdBlock.getArgument(1).replaceAllUsesWith(tidy);
          herdBlock.getArgument(2).replaceAllUsesWith(bdimx);
          herdBlock.getArgument(3).replaceAllUsesWith(bdimy);

          // Remap kernel operands to the values passed from enclosing scope.
          unsigned numKernelArgs = herdOp.getNumKernelOperands();
          for (unsigned i = 0; i < numKernelArgs; ++i) {
            Value outerVal = herdOp.getKernelOperand(i);
            herdBlock.getArgument(4 + i).replaceAllUsesWith(outerVal);
          }

          for (auto &operation :
               llvm::make_early_inc_range(herdBlock.without_terminator())) {
            operation.moveBefore(herdOp);
          }
          herdBlock.getTerminator()->erase();
          herdOp.erase();
        }
      }
    });
  }

  // Convert air.launch -> gpu.launch with thread block tuning
  gpu::LaunchOp convertLaunchToGPULaunch(xilinx::air::LaunchOp launchOp,
                                         OpBuilder &builder, Value gridXVal,
                                         Value gridYVal) {
    Location loc = launchOp.getLoc();
    // Define grid and block sizes (modify these values as needed for your use
    // case)
    int64_t gridSizeZ = 1;
    int64_t blockSizeZ = 1;

    builder.setInsertionPoint(launchOp);
    Value gridZVal = arith::ConstantOp::create(builder, loc,
                                               builder.getIndexAttr(gridSizeZ));
    Value blockZVal = arith::ConstantOp::create(
        builder, loc, builder.getIndexAttr(blockSizeZ));

    Operation *blockXValOp = blkIdx[0].getDefiningOp();
    blockXValOp->moveBefore(launchOp);
    Operation *blockYValOp = blkIdx[1].getDefiningOp();
    blockYValOp->moveBefore(launchOp);

    // Create the gpu.launch operation
    auto gpuLaunchOp =
        gpu::LaunchOp::create(builder, loc, gridXVal, gridYVal, gridZVal,
                              blkIdx[0], blkIdx[1], blockZVal);

    // Get thread indices for use within the gpu.launch body
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&gpuLaunchOp.getBody().front());

    gpu::TerminatorOp::create(builder, loc);
    return gpuLaunchOp;
  }

  void hoistAlloc(gpu::LaunchOp launchOp, OpBuilder &builder) {
    Location loc = launchOp.getLoc();
    launchOp.walk([&](memref::AllocOp allocOp) {
      // workgroup
      MemRefType memRefType = allocOp.getType();
      if (air::isL2(memRefType)) {
        mlir::Type elementType = memRefType.getElementType();
        llvm::ArrayRef<int64_t> shape = memRefType.getShape();

        // Create a new MemRefType with the same shape, element type, but with a
        // different memory space
        mlir::MemRefType newType =
            mlir::MemRefType::get(shape, elementType, /*affineMap=*/{}, 3);

        auto wg = launchOp.addWorkgroupAttribution(newType, loc);
        allocOp.replaceAllUsesWith(wg); // Replace row with globalRow
        allocOp.erase();
      } else if (air::isL1(memRefType)) {
        mlir::Type elementType = memRefType.getElementType();
        llvm::ArrayRef<int64_t> shape = memRefType.getShape();

        // Create a new MemRefType with the same shape, element type, but with a
        // different memory space
        mlir::MemRefType newType =
            mlir::MemRefType::get(shape, elementType, /*affineMap=*/{}, 5);
        auto wg = launchOp.addPrivateAttribution(newType, loc);
        allocOp.replaceAllUsesWith(wg); // Replace row with globalRow
        allocOp.erase();
      }
    });
    launchOp.walk([&](memref::DeallocOp deallocOp) { deallocOp.erase(); });
  }

  // Delinearize a flat index into multi-dimensional indices for a given shape.
  static SmallVector<Value> delinearizeIndex(OpBuilder &b, Location loc,
                                             Value linear,
                                             ArrayRef<int64_t> shape) {
    int rank = shape.size();
    SmallVector<Value> indices(rank);
    Value remaining = linear;
    for (int i = rank - 1; i >= 0; --i) {
      Value dimSize = arith::ConstantIndexOp::create(b, loc, shape[i]);
      indices[i] = arith::RemSIOp::create(b, loc, remaining, dimSize);
      remaining = arith::DivSIOp::create(b, loc, remaining, dimSize);
    }
    return indices;
  }

  // Linearize multi-dimensional indices into a flat index for a given shape.
  static Value linearizeIndices(OpBuilder &b, Location loc,
                                ArrayRef<Value> indices,
                                ArrayRef<int64_t> shape) {
    int rank = indices.size();
    assert(rank == (int)shape.size());
    Value flat = arith::ConstantIndexOp::create(b, loc, 0);
    for (int i = 0; i < rank; ++i) {
      int64_t stride = 1;
      for (int j = i + 1; j < rank; ++j)
        stride *= shape[j];
      Value strideVal = arith::ConstantIndexOp::create(b, loc, stride);
      Value term = arith::MulIOp::create(b, loc, indices[i], strideVal);
      flat = arith::AddIOp::create(b, loc, flat, term);
    }
    return flat;
  }

  // Compute memref indices from transfer indices, offsets, and strides.
  // Handles rank mismatches between transfer descriptor and memref.
  SmallVector<Value> computeMemrefIndices(OpBuilder &b, Location loc,
                                          ArrayRef<Value> transferIndices,
                                          ArrayRef<Value> offsets,
                                          ArrayRef<Value> strides,
                                          MemRefType memrefType,
                                          ArrayRef<Value> transferSizes) {
    int memrefRank = memrefType.getRank();
    int transferRank = transferIndices.size();

    if (offsets.empty()) {
      // Entire memref addressed.
      if (memrefRank == transferRank)
        return SmallVector<Value>(transferIndices);
      // Rank mismatch: linearize transfer indices, delinearize into memref.
      SmallVector<int64_t> transferShape;
      for (auto sz : transferSizes) {
        APInt val;
        if (matchPattern(sz, m_ConstantInt(&val)))
          transferShape.push_back(val.getSExtValue());
        else
          transferShape.push_back(1);
      }
      Value flat = linearizeIndices(b, loc, transferIndices, transferShape);
      return delinearizeIndex(b, loc, flat, memrefType.getShape());
    }

    if (memrefRank == transferRank) {
      // Same rank: idx[i] = offset[i] + transferIdx[i]
      SmallVector<Value> result(memrefRank);
      for (int i = 0; i < memrefRank; ++i)
        result[i] =
            arith::AddIOp::create(b, loc, offsets[i], transferIndices[i]);
      return result;
    }

    // Rank-reducing: offsets are in memref dimensions, transfer is lower rank.
    // Linearize: base_flat = linearize(offsets, memref_shape)
    //            flat = base_flat + iv[0] * strides[0] (+ iv[1]*strides[1]...)
    // Then delinearize flat back into memref shape.
    Value baseFlat = linearizeIndices(
        b, loc, SmallVector<Value>(offsets.begin(), offsets.end()),
        memrefType.getShape());
    Value transferFlat = arith::ConstantIndexOp::create(b, loc, 0);
    for (int i = 0; i < transferRank; ++i) {
      Value s = (i < (int)strides.size())
                    ? strides[i]
                    : arith::ConstantIndexOp::create(b, loc, 1);
      Value term = arith::MulIOp::create(b, loc, transferIndices[i], s);
      transferFlat = arith::AddIOp::create(b, loc, transferFlat, term);
    }
    Value flat = arith::AddIOp::create(b, loc, baseFlat, transferFlat);
    return delinearizeIndex(b, loc, flat, memrefType.getShape());
  }

  // Lower air.dma_memcpy_nd to SCF loops with memref.load/store.
  // L3→L2 transfers use thread-cooperative loading with gpu.barrier.
  // All other transfers use per-thread nested loops.
  // TODO: Handle async form (async_dependencies and async_token result).
  // Currently only synchronous DMAs are supported on the GPU path.
  void convertDMAToGPUMemcpy(xilinx::air::DmaMemcpyNdOp dmaOp,
                             OpBuilder &builder) {
    builder.setInsertionPointAfter(dmaOp);
    Location loc = dmaOp.getLoc();

    Value srcMemref = dmaOp.getSrcMemref();
    Value dstMemref = dmaOp.getDstMemref();
    auto srcType = cast<MemRefType>(srcMemref.getType());
    auto dstType = cast<MemRefType>(dstMemref.getType());
    SmallVector<Value> srcOffsets(dmaOp.getSrcOffsets());
    SmallVector<Value> dstOffsets(dmaOp.getDstOffsets());
    SmallVector<Value> srcSizes(dmaOp.getSrcSizes());
    SmallVector<Value> dstSizes(dmaOp.getDstSizes());
    SmallVector<Value> srcStrides(dmaOp.getSrcStrides());
    SmallVector<Value> dstStrides(dmaOp.getDstStrides());

    // Determine transfer sizes from whichever side has explicit sizes,
    // or fall back to the smaller memref's static shape.
    SmallVector<Value> transferSizes;
    if (!srcSizes.empty()) {
      transferSizes = srcSizes;
    } else if (!dstSizes.empty()) {
      transferSizes = dstSizes;
    } else {
      ArrayRef<int64_t> shape = (srcType.getRank() <= dstType.getRank())
                                    ? srcType.getShape()
                                    : dstType.getShape();
      for (int64_t s : shape)
        transferSizes.push_back(
            arith::ConstantIndexOp::create(builder, loc, s));
    }
    int transferRank = transferSizes.size();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

    bool isGlobalToShared = air::isL3(srcType) && air::isL2(dstType);

    if (isGlobalToShared) {
      // Thread-cooperative loading: distribute total elements across threads.
      Value total = transferSizes[0];
      for (int i = 1; i < transferRank; ++i)
        total = arith::MulIOp::create(builder, loc, total, transferSizes[i]);

      // Linearize 3D thread index to handle multi-dimensional blocks.
      Value tx = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                         gpu::Dimension::x);
      Value ty = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                         gpu::Dimension::y);
      Value tz = gpu::ThreadIdOp::create(builder, loc, builder.getIndexType(),
                                         gpu::Dimension::z);
      Value bx = gpu::BlockDimOp::create(builder, loc, builder.getIndexType(),
                                         gpu::Dimension::x);
      Value by = gpu::BlockDimOp::create(builder, loc, builder.getIndexType(),
                                         gpu::Dimension::y);
      Value bz = gpu::BlockDimOp::create(builder, loc, builder.getIndexType(),
                                         gpu::Dimension::z);
      // tidx = tx + ty * bx + tz * bx * by
      Value tyBx = arith::MulIOp::create(builder, loc, ty, bx);
      Value bxBy = arith::MulIOp::create(builder, loc, bx, by);
      Value tzBxBy = arith::MulIOp::create(builder, loc, tz, bxBy);
      Value tidx = arith::AddIOp::create(builder, loc, tx, tyBx);
      tidx = arith::AddIOp::create(builder, loc, tidx, tzBxBy);
      // bdim = bx * by * bz
      Value bdim = arith::MulIOp::create(builder, loc, bxBy, bz);

      auto loop = scf::ForOp::create(builder, loc, tidx, total, bdim);
      builder.setInsertionPointToStart(loop.getBody());
      Value linear = loop.getInductionVar();

      // Delinearize into transfer-dimension indices.
      SmallVector<int64_t> transferShape;
      for (auto sz : transferSizes) {
        APInt val;
        if (matchPattern(sz, m_ConstantInt(&val)))
          transferShape.push_back(val.getSExtValue());
        else
          transferShape.push_back(1);
      }
      SmallVector<Value> tIdx =
          delinearizeIndex(builder, loc, linear, transferShape);

      SmallVector<Value> sIdx = computeMemrefIndices(
          builder, loc, tIdx, srcOffsets, srcStrides, srcType, transferSizes);
      SmallVector<Value> dIdx = computeMemrefIndices(
          builder, loc, tIdx, dstOffsets, dstStrides, dstType, transferSizes);

      Value val = memref::LoadOp::create(builder, loc, srcMemref, sIdx);
      memref::StoreOp::create(builder, loc, val, dstMemref, dIdx);

      builder.setInsertionPointAfter(loop);
      gpu::BarrierOp::create(builder, loc);
    } else {
      // Per-thread path: nested loops over transfer dimensions.
      SmallVector<scf::ForOp> loops;
      for (int i = 0; i < transferRank; ++i) {
        auto loop = scf::ForOp::create(builder, loc, c0, transferSizes[i], c1);
        loops.push_back(loop);
        builder.setInsertionPointToStart(loop.getBody());
      }

      SmallVector<Value> tIdx;
      for (auto &loop : loops)
        tIdx.push_back(loop.getInductionVar());

      SmallVector<Value> sIdx = computeMemrefIndices(
          builder, loc, tIdx, srcOffsets, srcStrides, srcType, transferSizes);
      SmallVector<Value> dIdx = computeMemrefIndices(
          builder, loc, tIdx, dstOffsets, dstStrides, dstType, transferSizes);

      Value val = memref::LoadOp::create(builder, loc, srcMemref, sIdx);
      memref::StoreOp::create(builder, loc, val, dstMemref, dIdx);
    }

    dmaOp.erase();
  }
};
} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToROCDLPass() {
  return std::make_unique<ConvertAIRToROCDLPass>();
}

} // namespace air
} // namespace xilinx
