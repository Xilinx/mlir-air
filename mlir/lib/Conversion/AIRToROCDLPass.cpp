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
#include "mlir/Dialect/Arith/Utils/Utils.h"
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

/// Map a block argument of an air hierarchy op (launch/segment/herd) back to
/// the value passed in from the enclosing scope, or null if `blockArg` is not
/// a kernel argument at all.
///
/// Hierarchy block args are laid out [ids..., sizes..., kernel_args...], with
/// 2 * getNumDims() leading ids and sizes, so a kernel arg's index is *not*
/// its block-arg number. getTiedKernelOperand does that arithmetic and returns
/// null for an id or size arg, which has no counterpart outside the region.
/// Open-coding the offset here is what made a segment with an iteration space
/// index past the end of the operand list and abort.
static mlir::Value getTiedHierarchyOperand(mlir::BlockArgument blockArg) {
  auto hier = mlir::dyn_cast_if_present<air::HierarchyInterface>(
      blockArg.getOwner()->getParentOp());
  if (!hier)
    return {};
  return hier.getTiedKernelOperand(blockArg);
}
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
          // Operand 1 of a memref.store is the memref; if it is a hierarchy
          // kernel argument, name the value from the enclosing scope instead.
          if (mlir::Value outerVal = getTiedHierarchyOperand(blockArg))
            storeOp->setOperand(1, outerVal);
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
              if (mlir::Value outerVal = getTiedHierarchyOperand(blockArg)) {
                // A kernel argument: use the value from the enclosing scope.
                op->setOperand(j, outerVal);
              } else if (mlir::isa<MemRefType>(operandType)) {
                // A memref that is an id/size arg cannot happen; nothing to do.
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
  /// One run of the chiplet reporting protocol per region that needs it.
  /// air.chiplet_block_id and air.chiplet_dim_blocks are two questions with one
  /// answer; emitting the barrier and the scan once per op would make a program
  /// that asks both wait twice.
  DenseMap<Region *, std::pair<Value, Value>> chipletReporting;
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
        launchOp.getAsyncDependencies(), /*asyncObject=*/nullptr, clusterSize);
    launchOp.replaceAllUsesWith(launchFunc);
    launchOp.erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module.getContext());
    OpBuilder builder(module.getContext());
    mlir::ModuleOp moduleOp = getOperation();
    Value gridXVal, gridYVal;
    // Set when a nested walk hits IR this pass cannot lower. The walks below
    // cannot return out of runOnOperation, so they check this flag and stop
    // doing work; the pass result is signalled once they unwind.
    bool hadFailure = false;

    // Create a pattern rewriter to apply transformations to each function
    PatternRewriter rewriter(moduleOp.getContext());
    // Create a set of patterns for transformation. Freeze once so the same
    // pattern set can be reused across multiple launches in the module
    // (FrozenRewritePatternSet is shareable; the underlying mutable
    // RewritePatternSet would be consumed by std::move on first use).
    RewritePatternSet patterns(&getContext());
    patterns.add<AffineApplyToSubPattern, DMAMemcpyToSubPattern,
                 SCFForToSubPattern>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // Helper: get the i'th size operand of a launch/herd, or a freshly
    // materialized index constant 1 if the dim is absent. air.launch and
    // air.herd support N-D iteration spaces, but the gpu.launch op this
    // pass produces always wants X, Y, Z grid + block dims, so missing
    // higher dims default to 1 (single iteration in that axis).
    auto sizeOrOne = [&](OperandRange sizes, unsigned i,
                         Operation *insertBefore) -> Value {
      if (i < sizes.size())
        return sizes[i];
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(insertBefore);
      return arith::ConstantOp::create(builder, insertBefore->getLoc(),
                                       builder.getIndexAttr(1));
    };

    // Traverse the module and look for air.launch and air.herd ops.
    module.walk([&](air::LaunchOp launchOp) {
      if (hadFailure)
        return;
      launchOp.walk([&](air::SegmentOp segmentOp) {
        if (hadFailure)
          return;
        // Reset grid/block-dim collections for each (launch, segment)
        // pair. These are class members (not local) so without an
        // explicit clear they would carry stale operands from previous
        // launches in the same module — which then dangle into the new
        // gpu.launch op as use-after-frees.
        blkIdx.clear();
        gridIdx.clear();
        gridXVal = nullptr;
        gridYVal = nullptr;
        segmentOp.walk([&](Operation *childOp) {
          if (auto herdOp = dyn_cast_if_present<xilinx::air::HerdOp>(childOp)) {

            auto launchSizes = launchOp.getSizeOperands();
            auto herdSizes = herdOp.getSizeOperands();
            gridXVal = sizeOrOne(launchSizes, 0, launchOp);
            gridYVal = sizeOrOne(launchSizes, 1, launchOp);
            blkIdx.push_back(sizeOrOne(herdSizes, 0, herdOp));
            blkIdx.push_back(sizeOrOne(herdSizes, 1, herdOp));
            gridIdx.push_back(gridXVal);
            gridIdx.push_back(gridYVal);
          }
        });
        if (failed(checkCoResidency(launchOp, gridXVal, gridYVal))) {
          hadFailure = true;
          return;
        }
        gpu::LaunchOp gpuLaunchOp =
            convertLaunchToGPULaunch(launchOp, builder, gridXVal, gridYVal);
        Block &gpuLaunchBlock = gpuLaunchOp.getBody().front();
        auto blockArgs = gpuLaunchBlock.getArguments();

        gpuArgs.assign(blockArgs.begin(), blockArgs.end());
        (void)applyPatternsGreedily(launchOp, frozenPatterns);
        deleteAirHerd(segmentOp, builder, gpuLaunchOp);
        if (failed(deleteAirSegment(launchOp, builder, gpuLaunchOp))) {
          hadFailure = true;
          return;
        }

        // Move the (now-flattened) air.launch body into this specific
        // gpu.launch's body. Pairing must be 1:1 — a previous nested-walk
        // implementation pairwise-merged every gpu.launch with every
        // air.launch in the module, which folded multi-launch programs
        // into a single gpu.launch and dangled launch block args.
        //
        // Replace launch's block args with the launch's outer kernel
        // operands BEFORE moving the body. Without this, after the body
        // moves into gpu.launch and air.launch is erased, the moved ops
        // still reference air.launch's destroyed block args
        // (use-after-free during block destruction).
        Block &launchBlock = launchOp.getRegion().front();
        unsigned numLaunchKernelArgs = launchOp.getNumKernelOperands();
        // Block args layout: [tile_ids..., size_ids..., kernel_args...].
        // Tile ids and sizes are not used by the moved body in the
        // patterns we lower today (compute uses gpu.thread_id directly
        // after deleteAirHerd remap). Kernel-arg block args sit at the
        // tail of the block-arg list.
        unsigned numNonKernelArgs =
            launchBlock.getNumArguments() - numLaunchKernelArgs;
        for (unsigned i = 0; i < numLaunchKernelArgs; ++i) {
          Value outerVal = launchOp.getKernelOperand(i);
          launchBlock.getArgument(numNonKernelArgs + i)
              .replaceAllUsesWith(outerVal);
        }

        mlir::Block &gpuBlock = gpuLaunchOp.getBody().front();
        for (auto &operation :
             llvm::make_early_inc_range(launchBlock.without_terminator())) {
          mlir::Operation &lastOp = gpuBlock.back();
          operation.moveBefore(&lastOp);
        }
        hoistAlloc(gpuLaunchOp, builder);
      });
    });
    if (hadFailure)
      return signalPassFailure();

    // air.chiplet_block_id and air.chiplet_dim_blocks share one run of the
    // reporting protocol; lower them before air.chiplet_id so the register read
    // the protocol emits is itself lowered by the walk below.
    module.walk([&](Operation *op) {
      auto blockIdOp = dyn_cast<air::ChipletBlockIdOp>(op);
      auto dimOp = dyn_cast<air::ChipletDimBlocksOp>(op);
      if (!blockIdOp && !dimOp)
        return;
      auto [rank, count] = emitChipletReporting(op, builder);
      op->getResult(0).replaceAllUsesWith(blockIdOp ? rank : count);
      op->erase();
    });
    module.walk([&](air::ChipletIdOp chipletOp) {
      convertChipletIdToROCDL(chipletOp, builder);
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

  /// How many slots the reporting array has: one per workgroup that could be
  /// co-resident. Comes from the max-resident-workgroups option, which is also
  /// what bounds the launches this pass accepts, so a launch that fits is a
  /// launch whose workgroups all have a slot.
  int64_t reportingSlots() const {
    return static_cast<int64_t>(maxResidentWorkgroups);
  }

  /// Get, or create on first use, the module-scope globals the chiplet
  /// reporting protocol runs on:
  ///   @__air_chiplet_map        slot per workgroup: which chiplet it is on
  ///   @__air_chiplet_arrivals   how many workgroups have filled in their slot
  ///   @__air_chiplet_generation bumped once per completed barrier
  /// Zero-initialized. `arrivals` counts up to the grid size and is rearmed by
  /// the last workgroup through, so the barrier is reusable: a kernel launched
  /// a second time must not find the counter already satisfied and skip the
  /// wait entirely.
  std::pair<memref::GlobalOp, memref::GlobalOp>
  getOrCreateChipletGlobals(ModuleOp module, OpBuilder &builder) {
    SymbolTable symbolTable(module);
    auto map = symbolTable.lookup<memref::GlobalOp>("__air_chiplet_map");
    auto arrivals =
        symbolTable.lookup<memref::GlobalOp>("__air_chiplet_arrivals");
    if (map && arrivals)
      return {map, arrivals};

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    Location loc = module.getLoc();
    Type i32 = builder.getI32Type();
    auto makeGlobal = [&](StringRef name, int64_t n) {
      auto type = MemRefType::get({n}, i32);
      auto init = DenseElementsAttr::get(RankedTensorType::get({n}, i32),
                                         builder.getI32IntegerAttr(0));
      return memref::GlobalOp::create(
          builder, loc, builder.getStringAttr(name),
          builder.getStringAttr("private"), TypeAttr::get(type), init,
          /*constant=*/UnitAttr(), /*alignment=*/IntegerAttr());
    };
    if (!map)
      map = makeGlobal("__air_chiplet_map", reportingSlots());
    if (!arrivals)
      arrivals = makeGlobal("__air_chiplet_arrivals", 1);
    if (!symbolTable.lookup<memref::GlobalOp>("__air_chiplet_generation"))
      makeGlobal("__air_chiplet_generation", 1);
    return {map, arrivals};
  }

  /// The generation counter created alongside the other two above. Rearming
  /// `arrivals` is not enough on its own: the waiters need something that only
  /// moves forward to tell "the barrier I am in has opened" from "a later
  /// barrier has not started yet".
  memref::GlobalOp getChipletGenerationGlobal(ModuleOp module) {
    return SymbolTable(module).lookup<memref::GlobalOp>(
        "__air_chiplet_generation");
  }

  /// Emit the chiplet reporting protocol at `op` and return {rank, count}:
  /// this workgroup's rank among the workgroups sharing its chiplet, and how
  /// many there are. This is Fleet's algorithm
  /// (persistent_kernel.cuh:1083-1094)
  /// -- every workgroup records which die it landed on, everyone waits until
  /// all the records are in, then each counts the records matching its own die:
  ///
  ///   map[block_id] = chiplet_id                    // one store per workgroup
  ///   <all workgroups have stored>                  // device-wide barrier
  ///   rank  = |{w : map[w] == mine and w < me}|
  ///   count = |{w : map[w] == mine}|
  ///
  /// The scan is a pure function of (map, block_id, chiplet_id), so every
  /// thread of a workgroup computes the same answer from global memory and no
  /// broadcast through LDS is needed. Only the barrier is thread-0 work.
  ///
  /// The barrier terminates because air.launch guarantees its segments are
  /// co-resident on the device (AIROpBase.td, `launch`; AIRComputeModel.md
  /// section 2.2). This pass does not yet check that the grid it emits is
  /// small enough to keep that promise -- sizing the grid to the device is
  /// Phase 3 -- and a grid that outgrows the device turns the guarantee into a
  /// hang: a workgroup that was never scheduled cannot store its slot, so the
  /// ones already spinning never leave the loop.
  std::pair<Value, Value> emitChipletReporting(Operation *op,
                                               OpBuilder &builder) {
    Region *region = getChipletReportingRegion(op);
    auto cached = chipletReporting.find(region);
    if (cached != chipletReporting.end())
      return cached->second;

    OpBuilder::InsertionGuard guard(builder);
    // Emit at the top of the region rather than at the op, so the one result
    // dominates every op that asks for it.
    builder.setInsertionPointToStart(&region->front());
    Location loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto [mapGlobal, arrivalsGlobal] =
        getOrCreateChipletGlobals(module, builder);

    Type i32 = builder.getI32Type();
    Type idxTy = builder.getIndexType();
    Value zero =
        arith::ConstantOp::create(builder, loc, builder.getIndexAttr(0));
    Value one =
        arith::ConstantOp::create(builder, loc, builder.getIndexAttr(1));
    Value oneI32 = arith::ConstantOp::create(builder, loc, i32,
                                             builder.getI32IntegerAttr(1));

    // Linear workgroup id and grid size. gpu.launch is always 3-D here.
    auto linearize = [&](gpu::Dimension dx, gpu::Dimension dy,
                         gpu::Dimension dz, bool isId) -> Value {
      auto get = [&](gpu::Dimension d) -> Value {
        if (isId)
          return gpu::BlockIdOp::create(builder, loc, idxTy, d);
        return gpu::GridDimOp::create(builder, loc, idxTy, d);
      };
      Value x = get(dx), y = get(dy), z = get(dz);
      if (!isId) {
        Value xy = arith::MulIOp::create(builder, loc, x, y);
        return arith::MulIOp::create(builder, loc, xy, z);
      }
      Value ny = gpu::GridDimOp::create(builder, loc, idxTy, dy);
      Value nx = gpu::GridDimOp::create(builder, loc, idxTy, dx);
      Value zy = arith::MulIOp::create(builder, loc, z, ny);
      Value zyy = arith::AddIOp::create(builder, loc, zy, y);
      Value scaled = arith::MulIOp::create(builder, loc, zyy, nx);
      return arith::AddIOp::create(builder, loc, scaled, x);
    };
    Value blockId = linearize(gpu::Dimension::x, gpu::Dimension::y,
                              gpu::Dimension::z, /*isId=*/true);
    Value numBlocks = linearize(gpu::Dimension::x, gpu::Dimension::y,
                                gpu::Dimension::z, /*isId=*/false);

    Value chipletI32 = emitChipletIdRead(builder, loc);
    Value mapRef = memref::GetGlobalOp::create(
        builder, loc, mapGlobal.getType(), mapGlobal.getSymName());

    // Thread 0 of the workgroup publishes the slot and runs the barrier.
    Value tx = gpu::ThreadIdOp::create(builder, loc, idxTy, gpu::Dimension::x);
    Value ty = gpu::ThreadIdOp::create(builder, loc, idxTy, gpu::Dimension::y);
    Value tz = gpu::ThreadIdOp::create(builder, loc, idxTy, gpu::Dimension::z);
    Value txz = arith::OrIOp::create(builder, loc, tx, ty);
    Value tAll = arith::OrIOp::create(builder, loc, txz, tz);
    Value isLeader = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, tAll, zero);

    auto leaderIf =
        scf::IfOp::create(builder, loc, isLeader, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(leaderIf.thenBlock());
      memref::StoreOp::create(builder, loc, chipletI32, mapRef,
                              ValueRange{blockId});

      // A sense-reversing barrier, so it survives the kernel being launched
      // more than once. Counting arrivals up to the grid size and spinning on
      // "are we there yet" only works the first time: on a second launch the
      // counter is already past the mark, nobody waits, and the scan below runs
      // against a half-filled map.
      //
      // Read the generation first. A workgroup that has not arrived cannot be
      // the one that bumps it, so no arrival is missed between the two.
      Value arrivalsPtr = memrefToPointer(builder, loc, arrivalsGlobal);
      Value genPtr =
          memrefToPointer(builder, loc, getChipletGenerationGlobal(module));
      Value entryGen = LLVM::LoadOp::create(
          builder, loc, i32, genPtr, /*alignment=*/4, /*isVolatile=*/false,
          /*isNonTemporal=*/false, /*isInvariant=*/false,
          /*isInvariantGroup=*/false, LLVM::AtomicOrdering::acquire,
          builder.getStringAttr("agent"));

      // The slot store has to reach the other dies before the arrival that
      // announces it, hence release here and acquire on every poll below.
      Value previous = LLVM::AtomicRMWOp::create(
          builder, loc, LLVM::AtomicBinOp::add, arrivalsPtr, oneI32,
          LLVM::AtomicOrdering::release, builder.getStringAttr("agent"));
      Value numBlocksI32 =
          arith::IndexCastOp::create(builder, loc, i32, numBlocks);
      Value lastIndex =
          arith::SubIOp::create(builder, loc, numBlocksI32, oneI32);
      Value isLast = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, previous, lastIndex);

      auto lastIf = scf::IfOp::create(builder, loc, isLast, /*withElse=*/true);
      {
        OpBuilder::InsertionGuard g2(builder);
        // Last one in: rearm the counter, then release everybody by bumping the
        // generation. Rearming first is what makes the next barrier start from
        // zero.
        builder.setInsertionPointToStart(lastIf.thenBlock());
        Value zeroI32 = arith::ConstantOp::create(builder, loc, i32,
                                                  builder.getI32IntegerAttr(0));
        LLVM::AtomicRMWOp::create(
            builder, loc, LLVM::AtomicBinOp::xchg, arrivalsPtr, zeroI32,
            LLVM::AtomicOrdering::monotonic, builder.getStringAttr("agent"));
        LLVM::AtomicRMWOp::create(builder, loc, LLVM::AtomicBinOp::add, genPtr,
                                  oneI32, LLVM::AtomicOrdering::release,
                                  builder.getStringAttr("agent"));

        // Everyone else waits for that bump. Comparing against the generation
        // read on entry, rather than against a fixed value, is what keeps this
        // correct across repeats.
        builder.setInsertionPointToStart(lastIf.elseBlock());
        auto whileOp =
            scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
        Block *before = builder.createBlock(&whileOp.getBefore());
        builder.setInsertionPointToStart(before);
        Value gen = LLVM::LoadOp::create(
            builder, loc, i32, genPtr, /*alignment=*/4, /*isVolatile=*/false,
            /*isNonTemporal=*/false, /*isInvariant=*/false,
            /*isInvariantGroup=*/false, LLVM::AtomicOrdering::acquire,
            builder.getStringAttr("agent"));
        Value notDone = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, gen, entryGen);
        scf::ConditionOp::create(builder, loc, notDone, ValueRange{});
        Block *after = builder.createBlock(&whileOp.getAfter());
        builder.setInsertionPointToStart(after);
        scf::YieldOp::create(builder, loc, ValueRange{});
      }
    }
    // Hold the rest of the workgroup until its leader has left the barrier.
    gpu::BarrierOp::create(builder, loc);

    // Count the matching slots. Same inputs in every thread, same answer.
    auto scan = scf::ForOp::create(
        builder, loc, zero, numBlocks, one, ValueRange{zero, zero},
        [&](OpBuilder &b, Location l, Value w, ValueRange iter) {
          Value slot = memref::LoadOp::create(b, l, mapRef, ValueRange{w});
          Value same = arith::CmpIOp::create(b, l, arith::CmpIPredicate::eq,
                                             slot, chipletI32);
          Value earlier = arith::CmpIOp::create(b, l, arith::CmpIPredicate::ult,
                                                w, blockId);
          Value sameAndEarlier = arith::AndIOp::create(b, l, same, earlier);
          Value rankInc =
              arith::SelectOp::create(b, l, sameAndEarlier, one, zero);
          Value cntInc = arith::SelectOp::create(b, l, same, one, zero);
          Value rank = arith::AddIOp::create(b, l, iter[0], rankInc);
          Value cnt = arith::AddIOp::create(b, l, iter[1], cntInc);
          scf::YieldOp::create(b, l, ValueRange{rank, cnt});
        });
    std::pair<Value, Value> result{scan.getResult(0), scan.getResult(1)};
    chipletReporting[region] = result;
    return result;
  }

  /// Where one run of the protocol covers: the kernel body if there is one,
  /// otherwise the enclosing function. Workgroup identity is constant across
  /// it, so one answer serves every ask inside.
  static Region *getChipletReportingRegion(Operation *op) {
    if (auto launch = op->getParentOfType<gpu::LaunchOp>())
      return &launch.getBody();
    if (auto func = op->getParentOfType<func::FuncOp>())
      return &func.getBody();
    return op->getBlock()->getParent();
  }

  /// Take the address of a module-scope global as an !llvm.ptr, which is what
  /// llvm.atomicrmw and atomic llvm.load want. Going through
  /// memref.extract_aligned_pointer_as_index keeps this working before the
  /// memref lowering has run; air-translate-to-llvm spells peer pointers the
  /// same way.
  Value memrefToPointer(OpBuilder &builder, Location loc,
                        memref::GlobalOp global) {
    Value ref = memref::GetGlobalOp::create(builder, loc, global.getType(),
                                            global.getSymName());
    Value asIndex =
        memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, ref);
    Value asI64 =
        arith::IndexCastOp::create(builder, loc, builder.getI64Type(), asIndex);
    return LLVM::IntToPtrOp::create(
        builder, loc, LLVM::LLVMPointerType::get(builder.getContext()), asI64);
  }

  /// Reject a launch that asks for more co-resident workgroups than the target
  /// can hold.
  ///
  /// air.launch is not "run these, in some order": it promises the segments in
  /// its body are all resident when the body starts (AIROpBase.td, `launch`;
  /// AIRComputeModel.md section 2.2). Emitting a grid the device has to
  /// serialize would quietly downgrade that promise to a scheduling hint, and
  /// anything that waits on all workgroups reporting in --
  /// air.chiplet_dim_blocks, say -- would hang instead of returning a wrong
  /// answer. Say so at compile time.
  ///
  /// Only a compile-time-constant iteration space can be checked. A dynamic one
  /// passes; the promise is then the caller's to keep.
  LogicalResult checkCoResidency(air::LaunchOp launchOp, Value gridXVal,
                                 Value gridYVal) {
    APInt x, y;
    if (!matchPattern(gridXVal, m_ConstantInt(&x)) ||
        !matchPattern(gridYVal, m_ConstantInt(&y)))
      return success();
    uint64_t requested = x.getZExtValue() * y.getZExtValue();
    // The option is an llvm::cl wrapper; take its value explicitly so the
    // diagnostic stream formats the number rather than the wrapper.
    uint64_t capacity = maxResidentWorkgroups;
    if (requested <= capacity)
      return success();
    launchOp.emitError()
        << "air.launch asks for " << requested
        << " co-resident workgroups, more than the " << capacity
        << " this target holds; air.launch guarantees its segments are "
           "co-resident when the body begins, and a grid this size cannot "
           "keep that guarantee. Tile the launch, or raise "
           "-air-to-rocdl=max-resident-workgroups if the target really is "
           "this large";
    return failure();
  }

  /// Lower air.chiplet_id to a read of the AMDGPU XCC_ID hardware register.
  ///
  /// ROCDL has no s_getreg op (checked ROCDLOps.td: zero matches for getreg,
  /// hwreg or xcc), so this goes straight to the llvm.amdgcn.s.getreg
  /// intrinsic. Its operand is the packed hwreg descriptor
  ///   id | (offset << 6) | ((size - 1) << 11)
  /// (AMDGPUBaseInfo.h: HwregId = bits 5:0, HwregOffset = 10:6,
  /// HwregSize = 15:11 storing size-1), so HW_REG_XCC_ID = 20 read as
  /// bits [0, 16) is 20 | (15 << 11) = 30740. llc -mcpu=gfx942 prints that
  /// back as `hwreg(HW_REG_XCC_ID, 0, 16)`, the same operand Fleet writes by
  /// hand in persistent_kernel.cuh:188.
  void convertChipletIdToROCDL(air::ChipletIdOp op, OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value raw = emitChipletIdRead(builder, loc);
    Value asIndex =
        arith::IndexCastOp::create(builder, loc, builder.getIndexType(), raw);
    op.getResult().replaceAllUsesWith(asIndex);
    op.erase();
  }

  /// The XCC_ID read itself, as an i32, shared by air.chiplet_id and by the
  /// reporting protocol behind air.chiplet_block_id.
  Value emitChipletIdRead(OpBuilder &builder, Location loc) {
    constexpr int32_t hwRegXccIdBits0To16 = 30740;
    Type i32 = builder.getI32Type();
    // arith.constant, not llvm.mlir.constant: the descriptor is an ImmArg, so
    // it must still be a literal by the time LLVM sees the call. It is Pure and
    // operand-free, so LICM will hoist it out of the gpu.launch, and
    // GPUKernelOutlinePass only rematerializes arith::ConstantOp inside the
    // kernel -- an llvm.mlir.constant instead becomes a kernel argument and the
    // immediate is lost.
    Value desc = arith::ConstantOp::create(
        builder, loc, i32, builder.getI32IntegerAttr(hwRegXccIdBits0To16));
    auto call = LLVM::CallIntrinsicOp::create(
        builder, loc, /*resultType=*/i32,
        builder.getStringAttr("llvm.amdgcn.s.getreg"), ValueRange{desc});
    return call->getResult(0);
  }

  LogicalResult deleteAirSegment(air::LaunchOp launchOp, OpBuilder &builder,
                                 gpu::LaunchOp gpuLaunchOp) {
    WalkResult walkResult = launchOp.walk([&](Operation *childOp) {
      auto segmentOp = dyn_cast_if_present<xilinx::air::SegmentOp>(childOp);
      if (!segmentOp || segmentOp.getRegion().empty())
        return WalkResult::advance();
      Block &segmentBlock = segmentOp.getRegion().front();

      // Flattening runs the segment body exactly once. gpu.launch offers two
      // coordinate levels and this pass already spends both -- air.launch
      // supplies the grid, air.herd the block -- so a segment has nowhere to
      // put an iteration space of its own. Dropping it is only sound when the
      // space is a single instance; anything wider would silently compute
      // 1/N of the work.
      bool singleInstance =
          llvm::all_of(segmentOp.getSizeOperands(), [](Value v) {
            APInt c;
            return matchPattern(v, m_ConstantInt(&c)) && c.isOne();
          });
      if (!singleInstance) {
        segmentOp.emitError()
            << "air.segment iteration space is not supported by "
               "-air-to-rocdl: gpu.launch's grid is taken by air.launch and "
               "its block by air.herd, so the segment body can only be run "
               "once; only an all-ones iteration space can be flattened";
        return WalkResult::interrupt();
      }

      // Single instance: every id is 0 and every size is 1. Materialize those
      // so uses in the body survive the segment being erased.
      unsigned numDims = segmentOp.getNumDims();
      if (numDims) {
        auto notEmpty = [](BlockArgument a) { return !a.use_empty(); };
        bool idsUsed = llvm::any_of(segmentOp.getIds(), notEmpty);
        bool sizesUsed = llvm::any_of(segmentOp.getSize(), notEmpty);
        if (idsUsed || sizesUsed) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(segmentOp);
          Location loc = segmentOp.getLoc();
          if (idsUsed) {
            Value zero = arith::ConstantOp::create(builder, loc,
                                                   builder.getIndexAttr(0));
            for (BlockArgument id : segmentOp.getIds())
              id.replaceAllUsesWith(zero);
          }
          if (sizesUsed) {
            Value one = arith::ConstantOp::create(builder, loc,
                                                  builder.getIndexAttr(1));
            for (BlockArgument sz : segmentOp.getSize())
              sz.replaceAllUsesWith(one);
          }
        }
      }

      // Remap segment kernel arguments to the values passed in from the
      // enclosing scope. Index through getKernelArgument(), which skips the
      // leading ids/sizes: indexing segmentBlock directly remaps the ids and
      // leaves the kernel args dangling once the segment is erased.
      unsigned numKernelArgs = segmentOp.getNumKernelOperands();
      for (unsigned i = 0; i < numKernelArgs; ++i) {
        Value outerVal = segmentOp.getKernelOperand(i);
        segmentOp.getKernelArgument(i).replaceAllUsesWith(outerVal);
      }

      for (auto &operation :
           llvm::make_early_inc_range(segmentBlock.without_terminator())) {
        operation.moveBefore(segmentOp);
      }
      segmentBlock.getTerminator()->erase();
      segmentOp.erase();
      return WalkResult::advance();
    });
    return failure(walkResult.wasInterrupted());
  }

  void deleteAirHerd(xilinx::air::SegmentOp segmentOp, OpBuilder &builder,
                     gpu::LaunchOp gpuLaunchOp) {
    segmentOp.walk([&](Operation *childOp) {
      if (auto herdOp = dyn_cast_if_present<xilinx::air::HerdOp>(childOp)) {
        if (!herdOp.getRegion().empty()) {
          Block &herdBlock = herdOp.getRegion().front();
          Location loc = herdOp.getLoc();

          // Remap herd block arguments per the AIR compute model
          // (docs/AIRComputeModel.md §2.3 + §4): a PE is one wavefront, so
          // the herd's tile ids name warps, not threads:
          //   tile_x   = thread_id_x / wave_size  (= warp-id within block)
          //   tile_y   = thread_id_y               (block-y is unscaled)
          //   size_x   = block_dim_x / wave_size
          //   size_y   = block_dim_y
          // Lane within a PE is available via gpu.lane_id inside the herd
          // body; user code uses it directly for wave-cooperative ops.
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
          Value waveSizeC = arith::ConstantOp::create(
              builder, loc, builder.getIndexAttr(waveSize));
          Value warpIdX = arith::DivUIOp::create(builder, loc, tidx, waveSizeC);
          Value numWarpsX =
              arith::DivUIOp::create(builder, loc, bdimx, waveSizeC);

          herdBlock.getArgument(0).replaceAllUsesWith(warpIdX);
          herdBlock.getArgument(1).replaceAllUsesWith(tidy);
          herdBlock.getArgument(2).replaceAllUsesWith(numWarpsX);
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

    // blkIdx[i] may be a BlockArgument (herd size operand passed in as an
    // SSA value rather than a constant declared in scope); getDefiningOp()
    // is null in that case and the value already dominates launchOp from
    // the enclosing scope — no move needed.
    if (Operation *blockXValOp = blkIdx[0].getDefiningOp())
      blockXValOp->moveBefore(launchOp);
    if (Operation *blockYValOp = blkIdx[1].getDefiningOp())
      blockYValOp->moveBefore(launchOp);

    // Per AIR compute model §2.3: PE = wavefront. blockDim.x = herd.Nx *
    // wave_size so the herd's PE count becomes a warp count, not a thread
    // count. blockDim.y stays at herd.Ny (PEs are 1D in the wave dim).
    Value waveSizeC =
        arith::ConstantOp::create(builder, loc, builder.getIndexAttr(waveSize));
    Value blockXVal = arith::MulIOp::create(builder, loc, blkIdx[0], waveSizeC);

    // Create the gpu.launch operation
    auto gpuLaunchOp =
        gpu::LaunchOp::create(builder, loc, gridXVal, gridYVal, gridZVal,
                              blockXVal, blkIdx[1], blockZVal);

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
      if (strides.empty()) {
        // Unit strides: idx[i] = offset[i] + transferIdx[i]
        SmallVector<Value> result(memrefRank);
        for (int i = 0; i < memrefRank; ++i)
          result[i] =
              arith::AddIOp::create(b, loc, offsets[i], transferIndices[i]);
        return result;
      }
      // Non-unit strides: linearize via offsets + iv*strides, then delinearize.
      Value baseFlat = linearizeIndices(
          b, loc, SmallVector<Value>(offsets.begin(), offsets.end()),
          memrefType.getShape());
      Value transferFlat = arith::ConstantIndexOp::create(b, loc, 0);
      for (int i = 0; i < transferRank; ++i) {
        Value term =
            arith::MulIOp::create(b, loc, transferIndices[i], strides[i]);
        transferFlat = arith::AddIOp::create(b, loc, transferFlat, term);
      }
      Value flat = arith::AddIOp::create(b, loc, baseFlat, transferFlat);
      return delinearizeIndex(b, loc, flat, memrefType.getShape());
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
    // DmaMemcpyNdOp stores offsets/sizes/strides as mixed static/dynamic
    // values; materialize each into an index Value (constant for static
    // entries) so the loop-nest lowering below can consume them uniformly.
    SmallVector<Value> srcOffsets = getValueOrCreateConstantIndexOp(
        builder, loc, dmaOp.getMixedSrcOffsets());
    SmallVector<Value> dstOffsets = getValueOrCreateConstantIndexOp(
        builder, loc, dmaOp.getMixedDstOffsets());
    SmallVector<Value> srcSizes =
        getValueOrCreateConstantIndexOp(builder, loc, dmaOp.getMixedSrcSizes());
    SmallVector<Value> dstSizes =
        getValueOrCreateConstantIndexOp(builder, loc, dmaOp.getMixedDstSizes());
    SmallVector<Value> srcStrides = getValueOrCreateConstantIndexOp(
        builder, loc, dmaOp.getMixedSrcStrides());
    SmallVector<Value> dstStrides = getValueOrCreateConstantIndexOp(
        builder, loc, dmaOp.getMixedDstStrides());

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
      if (!isa_and_nonnull<gpu::BarrierOp>(dmaOp->getNextNode()))
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
