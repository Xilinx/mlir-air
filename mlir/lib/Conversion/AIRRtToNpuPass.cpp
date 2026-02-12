//===- AIRRtToNpuPass.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRRtToNpuPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Transform/AIRDependencyScheduleOpt.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>

#define DEBUG_TYPE "airrt-to-npu-pass"

using namespace mlir;

// Helper function to check if an aie.device contains core/memtile DMAs with
// repeat_count > 0. This indicates that the DMA engine state needs to be reset
// after each launch to avoid stale repeat counters affecting the next launch.
static bool deviceHasRepeatCountDMAs(xilinx::AIE::DeviceOp device) {
  bool hasRepeatCount = false;

  // Walk through all DMAStartOp operations in the device
  device.walk([&](xilinx::AIE::DMAStartOp dmaStart) {
    // Check if repeat_count attribute is set and > 0
    if (dmaStart.getRepeatCount() > 0)
      hasRepeatCount = true;
  });

  return hasRepeatCount;
}

namespace {

// Helper function to check if a value is a memref on host memory (space 0)
static bool isHostMemory(Value val) {
  if (auto memrefType = dyn_cast<BaseMemRefType>(val.getType()))
    return memrefType.getMemorySpaceAsInt() == 0;
  return false;
}

// Helper function to check if an op has memory effects on host memory
static bool hasMemoryEffectsOnHostMemory(Operation *op) {
  // Check if this op has memory effects interface
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;

  SmallVector<MemoryEffects::EffectInstance> memEffects;
  effects.getEffects(memEffects);

  for (auto &effect : memEffects) {
    // Check if the effect is on a host memory value
    Value val = effect.getValue();
    if (val && isHostMemory(val))
      return true;
  }
  return false;
}

// Helper function to check if an op is a "live root" that should be preserved
bool isLiveRoot(Operation *op) {
  // Ops in airrt dialect are always live roots
  if (op->getDialect()->getNamespace() == "airrt")
    return true;

  // Ops in aie/aiex dialects are live roots
  if (op->getDialect()->getNamespace() == "aie" ||
      op->getDialect()->getNamespace() == "aiex")
    return true;

  // Terminators are live roots
  if (op->hasTrait<OpTrait::IsTerminator>())
    return true;

  // func.func is a live root
  if (isa<func::FuncOp>(op))
    return true;

  // Ops that have memory effects on host memory (space 0) should be kept
  if (hasMemoryEffectsOnHostMemory(op))
    return true;

  return false;
}

// Recursively mark an op and its operand-defining ops as live
void markLive(Operation *op, DenseSet<Operation *> &liveOps) {
  if (!liveOps.insert(op).second)
    return; // Already marked

  // Mark all operand-defining ops as live
  for (Value operand : op->getOperands()) {
    if (auto *defOp = operand.getDefiningOp())
      markLive(defOp, liveOps);
  }

  // Also mark parent ops as live (for nested ops in regions)
  if (auto *parentOp = op->getParentOp()) {
    if (!isa<ModuleOp>(parentOp))
      markLive(parentOp, liveOps);
  }
}

// Check if a loop body only contains the yield terminator (effectively empty)
bool isLoopBodyEmpty(LoopLikeOpInterface loopOp) {
  auto regions = loopOp.getLoopRegions();
  if (regions.empty())
    return false;
  return llvm::hasSingleElement(regions.front()->front().getOperations());
}

// Remove dead device compute ops (L1/L2 memory ops, pure compute) that won't
// be converted to NPU ops. This is a performance optimization to avoid
// processing thousands of ops that will just be removed.
void removeDeadDeviceComputeOps(func::FuncOp funcOp) {
  DenseSet<Operation *> liveOps;

  // Step 1: Find all live roots and propagate liveness backwards
  funcOp.walk([&](Operation *op) {
    if (isLiveRoot(op))
      markLive(op, liveOps);
  });

  // Step 2: Collect dead ops (those not in liveOps)
  // We need to process in reverse order so that users are erased before defs
  SmallVector<Operation *> deadOps;

  // Walk the function and collect dead ops
  funcOp.walk([&](Operation *op) {
    // Skip the function itself
    if (op == funcOp.getOperation())
      return;

    if (!liveOps.contains(op))
      deadOps.push_back(op);
  });

  // Step 3: Erase dead ops in reverse order
  // Reverse the list so we erase inner-most ops first
  for (Operation *op : llvm::reverse(deadOps)) {
    // Double-check the op is still dead (use_empty)
    // Skip if it still has uses (defensive programming)
    if (!op->use_empty())
      continue;

    op->erase();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Removed " << deadOps.size()
                 << " dead device compute ops from function "
                 << funcOp.getSymName() << "\n";
  });

  // Step 4: Remove empty loops (loops that have empty bodies after dead code
  // removal). This needs to be done iteratively since removing inner loops may
  // make outer loops empty.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> emptyLoops;

    funcOp.walk([&](LoopLikeOpInterface loopOp) {
      // Check if loop has no results being used
      if (!loopOp->use_empty())
        return;

      // Check if loop body is empty (only contains yield)
      if (isLoopBodyEmpty(loopOp))
        emptyLoops.push_back(loopOp);
    });

    for (Operation *op : llvm::reverse(emptyLoops)) {
      if (op->use_empty()) {
        op->erase();
        changed = true;
      }
    }
  }
}

} // namespace

namespace xilinx {

#define GEN_PASS_DECL_AIRRTTONPU
#define GEN_PASS_DEF_AIRRTTONPU
#include "air/Conversion/Passes.h.inc"

//
//
// Converts IR like:
//
// %0 = some.op
// %1 = memref.assume_alignment %0
// %2 = unrealized_conversion_cast %0
//
// to IR like:
//
// %0 = some.op
// %1 = unrealized_conversion_cast %0
// %2 = memref.assume_alignment %1
//

struct RelocateAssumeAlignmentOp
    : public mlir::OpRewritePattern<memref::AssumeAlignmentOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(memref::AssumeAlignmentOp assumeOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto producerOp = assumeOp.getOperand().getDefiningOp();
    if (!producerOp)
      return rewriter.notifyMatchFailure(assumeOp,
                                         "No producer for AssumeAlignmentOp");

    auto castConsumerOp = [&]() -> mlir::Operation * {
      for (auto u : producerOp->getUsers()) {
        if (auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(u)) {
          return castOp;
        }
      }
      return {};
    }();

    if (!castConsumerOp)
      return rewriter.notifyMatchFailure(
          assumeOp, "No unrealized_conversion_cast consumer of producer.");

    // Create a new AssumeAlignmentOp that consumes the cast operation's result
    (void)memref::AssumeAlignmentOp::create(rewriter, assumeOp.getLoc(),
                                            castConsumerOp->getResult(0),
                                            assumeOp.getAlignment());

    // Erase the old AssumeAlignmentOp
    rewriter.eraseOp(assumeOp);

    return success();
  }
};

struct DmaToNpuPattern : public OpConversionPattern<airrt::DmaMemcpyNdOp> {
  using OpConversionPattern<airrt::DmaMemcpyNdOp>::OpConversionPattern;

  DmaToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<airrt::DmaMemcpyNdOp>(context, benefit) {}

  // Helper to check if a DMA uses an S2MM (output/device-to-host) channel
  bool isDeviceToHostChannel(airrt::DmaMemcpyNdOp op) const {
    if (!op->hasAttr("metadata"))
      return false;

    auto metadata = op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata");
    if (!metadata)
      return false;

    // Find the parent DeviceOp to look up the allocation
    auto device = op->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return false;

    // Look up the ShimDMAAllocationOp for this metadata symbol
    StringRef metadataStr = metadata.getValue();
    auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(device, metadataStr);
    if (!allocOp)
      return false;

    // S2MM = device to host = output = needs wait
    return allocOp.getChannelDir() == AIE::DMAChannelDir::S2MM;
  }

  LogicalResult
  matchAndRewrite(airrt::DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memref = adaptor.getMemref();
    BaseMemRefType memrefTy = cast<BaseMemRefType>(memref.getType());
    unsigned int bitwidth = memrefTy.getElementTypeBitWidth();
    if (bitwidth != 32 && bitwidth != 16 && bitwidth != 8)
      return failure();

    // Get metadata symbol - must exist
    SymbolRefAttr metadata;
    if (!op->hasAttr("metadata"))
      return failure();
    metadata = op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata");
    if (!metadata)
      return failure();

    // Verify the metadata symbol exists as a ShimDMAAllocationOp or
    // ObjectFifo. If device doesn't exist or symbol lookup fails, we fail
    // the pattern to avoid crashes.
    auto device = op->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    StringRef metadataStr = cast<FlatSymbolRefAttr>(metadata).getValue();
    auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(device, metadataStr);
    if (!allocOp) {
      // Check for objectfifo as alternative
      auto objFifo = device.lookupSymbol<AIE::ObjectFifoCreateOp>(metadataStr);
      if (!objFifo)
        return failure();
    }

    // Get static offsets
    SmallVector<int64_t> staticOffsets;
    if (auto const_int = getConstantIntValue(adaptor.getOffset3()))
      staticOffsets.push_back(*const_int);
    else
      staticOffsets.push_back(0);
    if (auto const_int = getConstantIntValue(adaptor.getOffset2()))
      staticOffsets.push_back(*const_int);
    else
      staticOffsets.push_back(0);
    if (auto const_int = getConstantIntValue(adaptor.getOffset1()))
      staticOffsets.push_back(*const_int);
    else
      staticOffsets.push_back(0);
    if (auto const_int = getConstantIntValue(adaptor.getOffset0()))
      staticOffsets.push_back(*const_int);
    else
      staticOffsets.push_back(0);

    // Get static sizes
    SmallVector<int64_t> staticSizes;
    if (auto const_int = getConstantIntValue(adaptor.getLength3()))
      staticSizes.push_back(*const_int);
    else
      staticSizes.push_back(1);
    if (auto const_int = getConstantIntValue(adaptor.getLength2()))
      staticSizes.push_back(*const_int);
    else
      staticSizes.push_back(1);
    if (auto const_int = getConstantIntValue(adaptor.getLength1()))
      staticSizes.push_back(*const_int);
    else
      staticSizes.push_back(1);
    if (auto const_int = getConstantIntValue(adaptor.getLength0()))
      staticSizes.push_back(std::max((int64_t)1, *const_int));
    else
      staticSizes.push_back(1);

    // Get static strides
    SmallVector<int64_t> staticStrides;
    if (auto const_int = getConstantIntValue(adaptor.getStride3()))
      staticStrides.push_back(*const_int);
    else
      staticStrides.push_back(0);
    if (auto const_int = getConstantIntValue(adaptor.getStride2()))
      staticStrides.push_back(*const_int);
    else
      staticStrides.push_back(0);
    if (auto const_int = getConstantIntValue(adaptor.getStride1()))
      staticStrides.push_back(*const_int);
    else
      staticStrides.push_back(0);
    staticStrides.push_back(1); // Last stride is always 1

    // Calculate total offset in elements
    // For npu.dma_memcpy_nd, the offset is computed as:
    //   offset = sum(offsets[i] * strides[i]) for each dimension
    int64_t totalOffset = 0;
    for (int i = 0; i < 4; i++) {
      totalOffset += staticOffsets[i] * staticStrides[i];
    }

    // Transfer length is ALWAYS the product of lowest 3 dimensions only
    int64_t transferLen = staticSizes[1] * staticSizes[2] * staticSizes[3];

    // repeat_count is ALWAYS size[0] - 1 (the highest dimension)
    // repeat_count = 0 means execute once, repeat_count = 3 means execute 4
    // times
    int64_t repeatCount = std::max((int64_t)0, staticSizes[0] - 1);

    // The 4th dimension is included in dma_bd dimensions if stride[0] != 0
    // (the iteration_stride tells the hardware how to advance offset each
    // repeat)
    bool use4thDimInBd = (staticStrides[0] != 0);

    // Build BDDimLayoutArrayAttr for the data layout transformation
    SmallVector<AIE::BDDimLayoutAttr> dimLayouts;
    auto ctx = rewriter.getContext();

    // Determine starting index for dims based on whether we use 4th dim
    int startDim = use4thDimInBd ? 0 : 1;

    // Build dimension layouts from sizes and strides
    for (int i = startDim; i < 4; i++) {
      int64_t size = staticSizes[i];
      int64_t stride = staticStrides[i];
      // Include dimension if size > 1, or if it's the innermost dimension
      if (size > 1 || i == 3) {
        auto dimLayout = AIE::BDDimLayoutAttr::get(ctx, size, stride);
        dimLayouts.push_back(dimLayout);
      }
    }

    AIE::BDDimLayoutArrayAttr dimsAttr =
        AIE::BDDimLayoutArrayAttr::get(ctx, dimLayouts);

    // Determine if this is an output (S2MM) channel
    // S2MM channels issue tokens by default, MM2S channels do not
    bool issueToken = isDeviceToHostChannel(op);

    // Create DMAConfigureTaskForOp with proper repeat_count from highest
    // dimension
    auto configTaskOp = AIEX::DMAConfigureTaskForOp::create(
        rewriter, op.getLoc(),
        rewriter.getIndexType(),          // result type
        metadata,                         // alloc symbol reference
        rewriter.getBoolAttr(issueToken), // issue_token = true for S2MM only
        rewriter.getI32IntegerAttr(repeatCount) // repeat_count from highest dim
    );

    // Create the body region of the configure task op
    Block *bodyBlock = rewriter.createBlock(&configTaskOp.getBody());
    rewriter.setInsertionPointToStart(bodyBlock);

    // Create aie.dma_bd inside the task body
    if (dimLayouts.empty()) {
      AIE::DMABDOp::create(rewriter, op.getLoc(), memref,
                           static_cast<int>(totalOffset),
                           static_cast<int>(transferLen));
    } else {
      AIE::DMABDOp::create(rewriter, op.getLoc(), memref,
                           static_cast<int>(totalOffset),
                           static_cast<int>(transferLen), dimsAttr);
    }

    // Create aie.end to terminate the block
    AIE::EndOp::create(rewriter, op.getLoc());

    // Move insertion point after the configure task op
    rewriter.setInsertionPointAfter(configTaskOp);

    // Create DMAStartTaskOp
    AIEX::DMAStartTaskOp::create(rewriter, op.getLoc(),
                                 configTaskOp.getResult());

    // NOTE: We do NOT generate DMAAwaitTaskOp here. Awaits are generated
    // by AIRRtWaitAllOpToAwaitPattern AFTER DMA conversion, at the location
    // of the WaitAllOp (clustered together), replicating the original behavior
    // where NpuDmaWaitOp was generated at WaitAllOp location.

    // Erase the original op
    rewriter.eraseOp(op);

    return success();
  }
};

// Helper method to get AIE device by segment name
AIE::DeviceOp getDeviceByName(ModuleOp module, StringAttr segmentName) {
  for (auto d : module.getOps<AIE::DeviceOp>()) {
    if (d.getSymName() == segmentName)
      return d;
  }
  return nullptr;
}

// Helper method to get AIE device by segment name.
// This overload accepts the segment name as a StringRef and returns the
// AIE::DeviceOp whose symbol name matches the given segment name, or nullptr
// if no matching device is found in the module.
AIE::DeviceOp getDeviceByName(ModuleOp module, StringRef segmentName) {
  for (auto d : module.getOps<AIE::DeviceOp>()) {
    if (d.getSymName() == segmentName)
      return d;
  }
  return nullptr;
}

struct HerdLoadToNpuPattern : public OpConversionPattern<airrt::HerdLoadOp> {
  using OpConversionPattern<airrt::HerdLoadOp>::OpConversionPattern;

  HerdLoadToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<airrt::HerdLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(airrt::HerdLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<ModuleOp>();

    // get the size metadata associated with this herd load
    int64_t size_x = -1;
    int64_t size_y = -1;
    int64_t loc_x = -1;
    int64_t loc_y = -1;
    module.walk([&](airrt::HerdMetadataOp metadata) {
      // return the first match by name
      if (metadata.getSymName() != op.getSymName())
        return WalkResult::advance();
      auto sxAttr = metadata->getAttrOfType<IntegerAttr>("size_x");
      auto syAttr = metadata->getAttrOfType<IntegerAttr>("size_y");
      auto lxAttr = metadata->getAttrOfType<IntegerAttr>("loc_x");
      auto lyAttr = metadata->getAttrOfType<IntegerAttr>("loc_y");
      if (sxAttr && syAttr && lxAttr && lyAttr) {
        size_x = sxAttr.getInt();
        size_y = syAttr.getInt();
        loc_x = lxAttr.getInt();
        loc_y = lyAttr.getInt();
      } else {
        metadata.emitWarning(
            "airrt.herd_metadata missing size_x, size_y, loc_x, or loc_y.");
      }
      return WalkResult::interrupt();
    });
    if (size_x < 0 || size_y < 0 || loc_x < 0 || loc_y < 0) {
      op.emitWarning("airrt.herd_metadata missing or incomplete.");
      return failure();
    }

    // Get the segment_name attribute and look up the device early
    auto segmentName = op->getAttrOfType<StringAttr>("segment_name");
    AIE::DeviceOp device = nullptr;
    if (segmentName) {
      device = getDeviceByName(module, segmentName);
      if (!device) {
        return rewriter.notifyMatchFailure(
            op, "segment_name attribute is set, but no matching AIE device "
                "was found in the module");
      }
    }

    // for each herd core, emit write_rtp ops for every herd operand
    // followed by a write32 to the herd lock, setting it to 1.
    for (int phys_x = loc_x; phys_x < size_x + loc_x; phys_x++) {
      for (int phys_y = loc_y; phys_y < size_y + loc_y; phys_y++) {

        std::string name = "__air_herd_rtp_" + std::to_string(phys_x) + "_" +
                           std::to_string(phys_y);

        // Only generate RTP writes if the RTP buffer was actually created.
        bool rtpBufferExists = false;
        if (device) {
          rtpBufferExists =
              static_cast<bool>(device.lookupSymbol<AIE::BufferOp>(name));
        } else {
          // Fallback for IR without segment_name: search all AIE::DeviceOp's.
          module.walk([&](AIE::DeviceOp d) {
            if (!rtpBufferExists && d.lookupSymbol<AIE::BufferOp>(name))
              rtpBufferExists = true;
          });
        }

        if (rtpBufferExists) {
          for (int i = 0, e = op.getNumOperands(); i < e; i++) {
            Value oper = adaptor.getOperands()[i];
            if (!llvm::isa<IntegerType, IndexType, FloatType>(oper.getType()))
              continue;

            auto constOp =
                dyn_cast_if_present<arith::ConstantOp>(oper.getDefiningOp());
            if (!constOp)
              continue;
            uint32_t v = cast<IntegerAttr>(constOp.getValue()).getInt();
            AIEX::NpuWriteRTPOp::create(rewriter, op.getLoc(), name, i, v);
          }
        }
        // FIXME: this should depend on the metadata to enable and to get the id
        if (!op.getNumOperands())
          continue;

        std::string lock_name = "__air_herd_lock_" + std::to_string(phys_x) +
                                "_" + std::to_string(phys_y);

        // Find the corresponding device using the segment_name attribute
        auto segmentName = op->getAttrOfType<StringAttr>("segment_name");
        if (!segmentName)
          continue;

        auto device = getDeviceByName(module, segmentName);
        if (!device)
          continue;

        auto lockOp = device.lookupSymbol<AIE::LockOp>(lock_name);
        if (!lockOp)
          continue;

        AIEX::SetLockOp::create(rewriter, op.getLoc(), lockOp.getResult(),
                                rewriter.getI32IntegerAttr(1));
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct SegmentLoadToNpuPattern
    : public OpConversionPattern<airrt::SegmentLoadOp> {
  using OpConversionPattern<airrt::SegmentLoadOp>::OpConversionPattern;

  SegmentLoadToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<airrt::SegmentLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(airrt::SegmentLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ModuleMetadataToNpuPattern
    : public OpConversionPattern<airrt::ModuleMetadataOp> {
  using OpConversionPattern<airrt::ModuleMetadataOp>::OpConversionPattern;

  ModuleMetadataToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<airrt::ModuleMetadataOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(airrt::ModuleMetadataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class L1AffineStoreOpConversion
    : public OpConversionPattern<affine::AffineStoreOp> {
public:
  using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<BaseMemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class L1MemRefStoreOpConversion : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<BaseMemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class AIRRtAllocOpConversion : public OpConversionPattern<airrt::AllocOp> {
public:
  using OpConversionPattern<airrt::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(airrt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

class AIRRtDeallocOpConversion : public OpConversionPattern<airrt::DeallocOp> {
public:
  using OpConversionPattern<airrt::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(airrt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

// Erase remaining WaitAllOps that weren't converted to NpuDmaWaitOp.
// These are pure synchronization ops that don't generate NPU ops.
// For WaitAllOps with "air.launch_end" attribute, we may need to insert
// aiex.npu.load_pdi to reset the DMA engine state if:
// 1. output-elf mode is enabled, AND
// 2. The device contains core/memtile DMAs with repeat_count > 0
class AIRRtWaitAllOpConversion : public OpConversionPattern<airrt::WaitAllOp> {
public:
  AIRRtWaitAllOpConversion(MLIRContext *context, bool outputElf,
                           PatternBenefit benefit = 1)
      : OpConversionPattern<airrt::WaitAllOp>(context, benefit),
        outputElf(outputElf) {}

  LogicalResult
  matchAndRewrite(airrt::WaitAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if this is a launch_end wait_all
    if (op->hasAttr("air.launch_end")) {
      // Find the parent device
      auto device = op->getParentOfType<AIE::DeviceOp>();
      if (device) {
        // Only apply for NPU2 family devices
        const AIE::AIETargetModel &tm = device.getTargetModel();
        if (llvm::isa<AIE::BaseNPU2TargetModel>(tm)) {
          // Insert aiex.npu.load_pdi to reset DMA engine state if:
          // 1. output-elf mode is enabled, AND
          // 2. The device has core/memtile DMAs with repeat_count > 0
          if (outputElf && deviceHasRepeatCountDMAs(device)) {
            rewriter.setInsertionPoint(op);
            auto deviceRef = FlatSymbolRefAttr::get(rewriter.getContext(),
                                                    device.getSymName());
            AIEX::NpuLoadPdiOp::create(rewriter, op.getLoc(), deviceRef);
          }
        }
      }
    }

    // Erase the op - synchronization is handled by NpuDmaWaitOp
    rewriter.eraseOp(op);
    return success();
  }

private:
  bool outputElf;
};

// Convert FuncOp control function into aiex.runtime_sequence op.
// Functions are converted if they are not external, are inside an aie.device
// and contain aiex.npu.* ops, aiex.dma_* ops, or airrt.dma_memcpy_nd ops
class ControlFuncConversion : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.isExternal())
      return failure();

    auto device = op->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    bool contains_relevant_ops = false;
    op.walk([&](Operation *o) {
      if (o->getName().getStringRef().starts_with("aiex.npu.") ||
          o->getName().getStringRef().starts_with("aiex.dma_") ||
          isa<airrt::DmaMemcpyNdOp>(o))
        contains_relevant_ops = true;
    });
    if (!contains_relevant_ops)
      return failure();

    auto seq = AIE::RuntimeSequenceOp::create(rewriter, op->getLoc(),
                                              op.getSymNameAttr());
    seq.getBody().push_back(new Block);

    // Add and remap the arguments
    IRMapping mapper;
    for (int i = 0, e = op.getNumArguments(); i < e; i++) {
      auto a = op.getBody().getArgument(i);
      seq.getBody().addArgument(a.getType(), a.getLoc());
      mapper.map(a, seq.getBody().getArgument(i));
    }

    // Clone the body of the function into the sequence, skipping the return op.
    rewriter.setInsertionPointToStart(&seq.getBody().front());
    for (auto &o : op.getBody().front().getOperations())
      if (!isa<func::ReturnOp>(o))
        rewriter.clone(o, mapper);

    rewriter.eraseOp(op);
    return success();
  }
};

// This is a hack due to the short-term limited support with lowering host code.
// This should be removed in the future.
class HostMemRefCopyOpConversion : public OpConversionPattern<memref::CopyOp> {
public:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallSet<Operation *, 1> erased;
    if (auto alloc = op.getSource().getDefiningOp()) {
      op.getSource().replaceAllUsesWith(op.getTarget());
      erased.insert(alloc);
    } else if (auto alloc = op.getTarget().getDefiningOp()) {
      op.getTarget().replaceAllUsesWith(op.getSource());
      erased.insert(alloc);
    }
    for (auto o : erased)
      rewriter.eraseOp(o);
    rewriter.eraseOp(op);
    return success();
  }
};

// Pattern to convert WaitAllOp to NpuDmaWaitOp(s).
// This runs BEFORE DMA conversion. NpuDmaWaitOp takes a symbol reference,
// so it can be created before DMAConfigureTaskForOp exists.
// Later, after DMA conversion, we convert:
//   - S2MM waits to DMAAwaitTaskOp (wait + free BD)
//   - MM2S waits to DMAFreeTaskOp (just free BD, no wait needed)
struct AIRRtWaitAllOpToNpuWaitPattern
    : public OpRewritePattern<airrt::WaitAllOp> {
public:
  AIRRtWaitAllOpToNpuWaitPattern(MLIRContext *context, bool outputElf,
                                 PatternBenefit benefit = 1)
      : OpRewritePattern<airrt::WaitAllOp>(context, benefit),
        outputElf(outputElf) {}

  LogicalResult matchAndRewrite(airrt::WaitAllOp op,
                                PatternRewriter &rewriter) const override {
    // Only match if at least one operand is a DmaMemcpyNdOp
    if (llvm::none_of(op->getOperands(), [](Value oper) {
          return (bool)oper.getDefiningOp<airrt::DmaMemcpyNdOp>();
        }))
      return failure();

    for (auto oper : op->getOperands()) {
      auto airrtDmaOp = oper.getDefiningOp<airrt::DmaMemcpyNdOp>();
      if (!airrtDmaOp)
        continue;
      auto metadataAttr =
          airrtDmaOp->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata");
      if (!metadataAttr)
        continue;

      // Generate NpuDmaWaitOp for ALL channels (both S2MM and MM2S)
      // The conversion to DMAAwaitTaskOp vs DMAFreeTaskOp happens later
      // based on channel direction
      StringRef metadata = metadataAttr.getValue();
      AIEX::NpuDmaWaitOp::create(rewriter, op.getLoc(), metadata);
    }

    // Check if this is a launch_end wait_all and needs load_pdi
    if (op->hasAttr("air.launch_end")) {
      auto device = op->getParentOfType<AIE::DeviceOp>();
      if (device) {
        // Only apply for NPU2 family devices
        const AIE::AIETargetModel &tm = device.getTargetModel();
        if (llvm::isa<AIE::BaseNPU2TargetModel>(tm)) {
          // Insert aiex.npu.load_pdi to reset DMA engine state if:
          // 1. output-elf mode is enabled, AND
          // 2. The device has core/memtile DMAs with repeat_count > 0
          if (outputElf && deviceHasRepeatCountDMAs(device)) {
            auto deviceRef = FlatSymbolRefAttr::get(rewriter.getContext(),
                                                    device.getSymName());
            AIEX::NpuLoadPdiOp::create(rewriter, op.getLoc(), deviceRef);
          }
        }
      }
    }

    // The WaitAllOp may have uses (other WaitAllOps depending on its result).
    // Replace with a new WaitAllOp with no operands to break the dependency
    // chain. This is safe because the synchronization is now handled by
    // NpuDmaWaitOp.
    if (op->getNumResults() > 0 && !op->use_empty()) {
      // Create a replacement WaitAllOp with no DMA operands (only non-DMA deps)
      SmallVector<Value> nonDmaOpers;
      for (auto oper : op->getOperands()) {
        if (!oper.getDefiningOp<airrt::DmaMemcpyNdOp>())
          nonDmaOpers.push_back(oper);
      }
      auto newWaitAll = airrt::WaitAllOp::create(
          rewriter, op.getLoc(), airrt::EventType::get(op->getContext()),
          nonDmaOpers);
      rewriter.replaceOp(op, newWaitAll->getResult(0));
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  bool outputElf;
};

AIE::DeviceOp getDeviceForSegmentLoad(Operation *s) {
  auto module = s->getParentOfType<ModuleOp>();

  // Use the airrt metadata to lookup the segment associated with each head
  // or segment load operation.
  if (auto segmentName =
          s->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
    return getDeviceByName(module, segmentName);
  }
  return nullptr;
}

// Represents a launch region identified by its affine.for boundary
struct LaunchRegion {
  affine::AffineForOp boundaryOp; // The affine.for %arg = 0 to 1 loop
  StringRef deviceName;           // Name of the target aie.device
  AIE::DeviceOp device;           // The target device op
};

// Check if an affine.for loop is a launch boundary.
// Launch boundaries are affine.for %arg = 0 to 1 loops with the
// "affine_opt_label" attribute.
bool isLaunchBoundaryLoop(affine::AffineForOp forOp) {
  // Check for affine_opt_label attribute (marks original air.launch boundary)
  if (!forOp->hasAttr("affine_opt_label"))
    return false;

  // Check bounds: 0 to 1
  if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound())
    return false;
  if (forOp.getConstantLowerBound() != 0 || forOp.getConstantUpperBound() != 1)
    return false;

  return true;
}

// Identify launch regions within a function.
// Launch regions are delimited by affine.for %arg = 0 to 1 loops
// with "affine_opt_label" attribute, and contain airrt.segment_load
// operations that link to device ops.
SmallVector<LaunchRegion> identifyLaunchRegions(func::FuncOp funcOp,
                                                ModuleOp module) {
  SmallVector<LaunchRegion> regions;

  funcOp.walk([&](affine::AffineForOp forOp) {
    // Check if this is a launch boundary
    if (!isLaunchBoundaryLoop(forOp))
      return;

    // Look for airrt.segment_load inside this loop
    forOp.walk([&](airrt::SegmentLoadOp segLoadOp) {
      StringRef deviceName = segLoadOp.getSymName();
      AIE::DeviceOp device = getDeviceByName(module, deviceName);
      if (device) {
        regions.push_back({forOp, deviceName, device});
      }
    });
  });

  return regions;
}

// Collect operations that should be part of the function "prologue" -
// operations that are used by multiple launch regions and should be
// cloned to each device's function.
SmallVector<Operation *>
collectPrologueOps(func::FuncOp funcOp, SmallVector<LaunchRegion> &regions) {
  SmallVector<Operation *> prologueOps;
  DenseSet<Operation *> launchOps;

  // Collect all operations that are inside launch regions
  for (auto &region : regions) {
    region.boundaryOp->walk([&](Operation *op) { launchOps.insert(op); });
  }

  // Prologue ops are those in the function body but not inside any launch
  // region
  for (auto &op : funcOp.getBody().front().getOperations()) {
    if (!launchOps.contains(&op) && !isa<func::ReturnOp>(&op)) {
      prologueOps.push_back(&op);
    }
  }

  return prologueOps;
}

// Structure representing a device and its sequence that needs a main wrapper.
struct DeviceSequenceInfo {
  // Name of the target device for which the sequence is generated.
  std::string deviceName;
  // Name of the sequence associated with this device.
  std::string sequenceName;
  // Types of the arguments passed to the sequence. This must be kept in
  // lockstep with `argLocs` such that `argTypes[i]` has source location
  // information stored in `argLocs[i]`.
  SmallVector<Type> argTypes;
  // Source locations corresponding to each entry in `argTypes`. This vector
  // must always be the same size as `argTypes`, and both arrays are indexed
  // in parallel.
  SmallVector<Location> argLocs;
};

// Structure to track pending main device creation
struct PendingMainDevice {
  LocationAttr loc;
  AIE::AIEDevice deviceType;
  std::string mainSeqName;
  // deviceNames and sequenceNames are parallel arrays:
  //   - they must have the same length
  //   - deviceNames[i] corresponds to sequenceNames[i]
  SmallVector<std::string> deviceNames;
  SmallVector<std::string> sequenceNames;
};

// Helper to create a main device with orchestration runtime_sequence.
// This is used both for multi-device func lowering and for wrapping
// existing aie.device ops with runtime_sequence when emit-main-device is set.
AIE::DeviceOp createMainDeviceWrapper(
    ModuleOp module, Location loc, AIE::AIEDevice deviceType,
    StringRef mainSeqName,
    const SmallVector<DeviceSequenceInfo> &deviceSequences) {

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());

  // Create main device with the specified device type
  auto mainDevice = AIE::DeviceOp::create(builder, loc, deviceType);
  mainDevice->setAttr(SymbolTable::getSymbolAttrName(),
                      StringAttr::get(builder.getContext(), "main"));

  // Add a body block to the main device
  Block *mainDeviceBody = new Block;
  mainDevice.getRegion().push_back(mainDeviceBody);

  // Create runtime_sequence inside main device
  builder.setInsertionPointToStart(mainDeviceBody);
  auto mainSeq = AIE::RuntimeSequenceOp::create(
      builder, loc, builder.getStringAttr(mainSeqName.str()));
  mainSeq.getBody().push_back(new Block);

  // Add arguments to runtime_sequence based on first device's signature
  // (all devices should have the same signature)
  if (!deviceSequences.empty()) {
    assert(deviceSequences[0].argTypes.size() ==
               deviceSequences[0].argLocs.size() &&
           "argTypes and argLocs must be parallel arrays");
    for (unsigned i = 0; i < deviceSequences[0].argTypes.size(); ++i) {
      mainSeq.getBody().addArgument(deviceSequences[0].argTypes[i],
                                    deviceSequences[0].argLocs[i]);
    }
  }

  builder.setInsertionPointToStart(&mainSeq.getBody().front());

  // Generate aiex.configure and aiex.run for each device in order
  for (const auto &devInfo : deviceSequences) {
    StringRef deviceName = devInfo.deviceName;

    // Create aiex.configure @device_name { ... }
    auto configureOp = AIEX::ConfigureOp::create(
        builder, loc, FlatSymbolRefAttr::get(builder.getContext(), deviceName));
    configureOp.getBody().push_back(new Block);
    builder.setInsertionPointToStart(&configureOp.getBody().front());

    // Create aiex.run @sequence_name (args)
    SmallVector<Value> args;
    for (unsigned i = 0; i < mainSeq.getBody().getNumArguments(); ++i) {
      args.push_back(mainSeq.getBody().getArgument(i));
    }
    AIEX::RunOp::create(
        builder, loc,
        FlatSymbolRefAttr::get(builder.getContext(), devInfo.sequenceName),
        args);

    // Move insertion point after configure op
    builder.setInsertionPointAfter(configureOp);
  }

  // Add aie.end terminator to the main device body
  builder.setInsertionPointToEnd(mainDeviceBody);
  AIE::EndOp::create(builder, loc);

  return mainDevice;
}

// AIE2 hardware constraints.
const std::vector<int> AIE2_WRAP_UPPER_BOUNDS = {64, 1024, 1024, 1024};
const int AIE2_STRIDE_UPPER_BOUND = 1048576;
const int AIE2_DIM_COUNT = 4;

bool violatesAIE2WrapLimit(airrt::DmaMemcpyNdOp dma) {
  SmallVector<Value> wrap_list;
  wrap_list.push_back(dma.getLength3());
  wrap_list.push_back(dma.getLength2());
  wrap_list.push_back(dma.getLength1());
  wrap_list.push_back(dma.getLength0());
  for (unsigned i = 0; i < wrap_list.size(); i++) {
    if (auto const_val = getConstantIntValue(wrap_list[i])) {
      // Detected wrap that goes beyond the AIE2 hardware limit.
      if (*const_val >= AIE2_WRAP_UPPER_BOUNDS[i])
        return true;
    }
  }
  return false;
}

// Find the largest factor of 'num' which is not larger than 'max'. Ref:
// https://github.com/nod-ai/iree-amd-aie/blob/main/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEUtils.cpp#L334
int findLargestFactor(int num, int max) {
  // No factors less than or equal to 0 exist
  if (max <= 0)
    return 0;

  // Do O(1) instead of O(sqrt(num)) computation for this common case.
  if (num <= max) {
    return num;
  }

  int largestLowFactor = 1;
  for (int lowFactor = 2; lowFactor <= max; ++lowFactor) {
    const int highFactor = num / lowFactor;

    // This early exit is what makes this O(sqrt(num)) instead of O(num).
    if (highFactor < lowFactor)
      return largestLowFactor;

    const bool areActuallyFactors = num % lowFactor == 0;
    if (areActuallyFactors) {
      // We're certain that here lowFactor <= highFactor, and highFactor is
      // descending in this loop. So we can return immediately if highFactor
      // is good.
      if (highFactor <= max)
        return highFactor;
      largestLowFactor = lowFactor;
    }
  }
  return largestLowFactor;
}

void tileIllegalWrapDim(airrt::DmaMemcpyNdOp memcpy_op) {
  auto loc = memcpy_op->getLoc();
  auto ctx = memcpy_op->getContext();
  auto oper_begin = memcpy_op.getOperands().begin();
  SmallVector<Value> offsets(oper_begin + 4, oper_begin + 8);
  SmallVector<Value> wraps(oper_begin + 8, oper_begin + 12);
  SmallVector<Value> strides(oper_begin + 12, oper_begin + 15);
  // Stride field implicit last element one
  OpBuilder builder(memcpy_op);
  strides.push_back(
      arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                IntegerAttr::get(builder.getI64Type(), 1)));

  for (int i = wraps.size() - 1; i >= 0; i--) {
    auto const_wrap = *getConstantIntValue(wraps[i]);
    auto const_stride = *getConstantIntValue(strides[i]);
    if (const_wrap >= AIE2_WRAP_UPPER_BOUNDS[i]) {
      // Found dimension with illegal wrap. Tiling. (Prefers smaller outer
      // wrap values, as long as stride fits)
      int a_wrap = findLargestFactor(const_wrap, AIE2_WRAP_UPPER_BOUNDS[i] - 1);
      int b_wrap = llvm::divideCeilSigned(const_wrap, a_wrap);
      int new_a_stride = const_stride * a_wrap;
      auto volume = air::getTensorVolume(
          llvm::cast<BaseMemRefType>(memcpy_op.getMemref().getType()));
      if (volume != 1)
        new_a_stride %=
            volume; // Avoids striding out of memory size, if memref is ranked
      int inner_wrap = (new_a_stride > AIE2_STRIDE_UPPER_BOUND && i != 0)
                           ? (b_wrap)
                           : (a_wrap);
      int outer_wrap = (new_a_stride > AIE2_STRIDE_UPPER_BOUND && i != 0)
                           ? (a_wrap)
                           : (b_wrap);
      wraps[i] = arith::ConstantOp::create(
          builder, loc, builder.getI64Type(),
          IntegerAttr::get(builder.getI64Type(), inner_wrap));
      wraps.insert(wraps.begin() + i,
                   arith::ConstantOp::create(
                       builder, loc, builder.getI64Type(),
                       IntegerAttr::get(builder.getI64Type(), outer_wrap)));
      auto new_const_stride = const_stride * inner_wrap;
      if (volume != 1)
        new_const_stride %=
            volume; // Avoids striding out of memory size, if memref is ranked
      strides.insert(
          strides.begin() + i,
          arith::ConstantOp::create(
              builder, loc, builder.getI64Type(),
              IntegerAttr::get(builder.getI64Type(), new_const_stride)));
      offsets.insert(
          offsets.begin() + i,
          arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                    IntegerAttr::get(builder.getI64Type(), 0)));
      // Attempt to find one dummy dimension in the wrap-and-stride list and
      // erase.
      auto offsetWrapZip = llvm::zip_equal(offsets, wraps);
      auto it =
          llvm::find_if(offsetWrapZip, [](std::tuple<Value, Value> entry) {
            auto off = getConstantIntValue(std::get<0>(entry));
            auto siz = getConstantIntValue(std::get<1>(entry));
            return off && siz && *off == 0 && *siz == 1;
          });
      if (it != offsetWrapZip.end()) {
        offsets.erase(offsets.begin() +
                      std::distance(offsetWrapZip.begin(), it));
        wraps.erase(wraps.begin() + std::distance(offsetWrapZip.begin(), it));
        strides.erase(strides.begin() +
                      std::distance(offsetWrapZip.begin(), it));
      }
      i++;
    }
  }

  // Unroll highest dimensions of wrap and stride, if the new dimension count
  // goes beyond 4.
  SmallVector<affine::AffineForOp> for_loop_nest;
  Value inner_affine_for_iv = nullptr;
  while (wraps.size() > AIE2_DIM_COUNT) {
    affine::AffineForOp inner_affine_for = nullptr;
    auto const_offset = *getConstantIntValue(offsets[0]);
    auto const_lowest_offset = *getConstantIntValue(offsets.back());
    auto const_wrap = *getConstantIntValue(wraps[0]);
    auto const_stride = *getConstantIntValue(strides[0]);

    // Convert the outer dimension into an affine.for loop.
    int const_lower_bound =
        const_stride ? (const_offset * const_stride + const_lowest_offset) : 0;
    auto const_upper_bound =
        const_stride ? (const_offset * const_stride +
                        const_wrap * const_stride + const_lowest_offset)
                     : const_wrap;
    int const_step = const_stride ? const_stride : 1;
    auto new_for_op =
        (inner_affine_for_iv)
            ? (affine::AffineForOp::create(
                  builder, loc,
                  SmallVector<Value>{arith::AddIOp::create(
                      builder, loc, inner_affine_for_iv,
                      arith::ConstantIndexOp::create(builder, loc,
                                                     const_lower_bound))},
                  AffineMap::get(ctx),
                  SmallVector<Value>{arith::AddIOp::create(
                      builder, loc, inner_affine_for_iv,
                      arith::ConstantIndexOp::create(builder, loc,
                                                     const_upper_bound))},
                  AffineMap::get(ctx), const_step))
            : (affine::AffineForOp::create(builder, loc, const_lower_bound,
                                           const_upper_bound, const_step));
    for_loop_nest.push_back(new_for_op);
    inner_affine_for = new_for_op;

    // Pop front.
    offsets.erase(offsets.begin());
    wraps.erase(wraps.begin());
    strides.erase(strides.begin());

    builder.setInsertionPointToStart(inner_affine_for.getBody());
    if (const_stride)
      inner_affine_for_iv = inner_affine_for.getInductionVar();
  }

  // Stride field implicit last element one, pop.
  strides.pop_back();

  // Create new airrt.dma_memcpy_nd op.
  SmallVector<Value> new_opers;
  SmallVector<Type> tys;
  auto old_opers = memcpy_op.getOperands();
  // Insert
  new_opers.insert(new_opers.end(), old_opers.begin(), old_opers.begin() + 4);
  if (inner_affine_for_iv) {
    // Innermost tiled affine.for loop induction variable as lowest offset, if
    // original rank exceeds hw limit.
    new_opers.insert(new_opers.end(), offsets.begin(), offsets.end() - 1);
    auto new_inner_offset = arith::IndexCastOp::create(
        builder, loc, IntegerType::get(ctx, 64), inner_affine_for_iv);
    new_opers.push_back(new_inner_offset);
  } else
    new_opers.insert(new_opers.end(), offsets.begin(), offsets.end());
  new_opers.insert(new_opers.end(), wraps.begin(), wraps.end());
  new_opers.insert(new_opers.end(), strides.begin(), strides.end());
  airrt::DmaMemcpyNdOp::create(builder, loc, tys, new_opers,
                               memcpy_op->getAttrs());

  // Unroll the affine loop nest.
  for (auto forOp : llvm::reverse(for_loop_nest)) {
    (void)loopUnrollFull(forOp);
  }

  memcpy_op.erase();
}

void enforceAIE2WrapLimit(ModuleOp module) {
  // Identify airrt.dma_memcpy_nd ops that violate the AIE2 wrap size
  // constraint.
  SmallVector<airrt::DmaMemcpyNdOp> target_airrt_dmas;
  SmallVector<func::FuncOp> funcOps;
  module.walk([&](func::FuncOp f) { funcOps.push_back(f); });
  for (auto f : funcOps) {
    f.walk([&](airrt::DmaMemcpyNdOp dma) {
      if (violatesAIE2WrapLimit(dma))
        target_airrt_dmas.push_back(dma);
    });
  }

  // Enforce the AIE2 wrap limit by tiling that dimension.
  for (auto memcpy_op : target_airrt_dmas)
    tileIllegalWrapDim(memcpy_op);
}

struct AIRRtToNpuPass : public impl::AIRRtToNpuBase<AIRRtToNpuPass> {
  // Track pending main device creation - stores info needed to create main
  // device AFTER all argument-modifying patterns have run
  std::optional<PendingMainDevice> pendingMainDevice;

  void runOnOperation() override {

    ModuleOp module = getOperation();

    // Move func op to the end of device op's body
    moveFuncOpToEndOfDeviceOp(module);

    // Early cleanup: remove dead device compute ops (L1/L2 memory ops, pure
    // compute) that won't be converted to NPU ops. This is a performance
    // optimization to avoid processing thousands of ops during loop unrolling
    // and pattern matching.
    SmallVector<func::FuncOp> funcOps;
    module.walk([&](func::FuncOp f) { funcOps.push_back(f); });
    for (auto f : funcOps)
      removeDeadDeviceComputeOps(f);

    // Purge all wait all ops
    purgeSCFParContainingOnlyWaitAllOps(module);

    // Purge airrt.dma x and y fields, as they are obsolete for AIE2.
    purgeAIRRtDmaXAndY(module);

    // Remove any duplicate shim dma allocations
    purgeDuplicateShimDmaAllocs(module);

    // Simplify affine apply ops
    auto ctx = &getContext();

    // Unroll for loops
    unrollAffineFors(module);
    unrollSCFFors(module);

    // Convert WaitAllOp → NpuDmaWaitOp and purge DMA async tokens.
    // This must happen BEFORE DMA conversion because:
    // 1. WaitAllOp has SSA operands to DmaMemcpyNdOp event tokens
    // 2. NpuDmaWaitOp uses symbol reference (can be created before DMA
    // conversion)
    // 3. After this, DMA tokens can be safely purged
    generateNpuWaitFromAIRRtWaitAll(module);

    // Enforce AIE2 hardware constraints.
    enforceAIE2WrapLimit(module);

    // Simplify arith ops (from airrt)
    RewritePatternSet canoPatterns_3(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns_3, ctx);
    (void)applyPatternsGreedily(module, std::move(canoPatterns_3));

    ConversionTarget target(getContext());
    target.addIllegalDialect<airrt::AIRRtDialect>();
    target.addLegalDialect<arith::ArithDialect, AIE::AIEDialect,
                           AIEX::AIEXDialect, memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<affine::AffineStoreOp>(
        [&](affine::AffineStoreOp op) {
          if (op->getParentOfType<AIE::CoreOp>())
            return true;
          return (llvm::cast<BaseMemRefType>(op.getMemref().getType())
                      .getMemorySpaceAsInt() !=
                  (int)xilinx::air::MemorySpace::L1);
        });
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      if (op->getParentOfType<AIE::CoreOp>())
        return true;
      return (llvm::cast<BaseMemRefType>(op.getMemref().getType())
                  .getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1);
    });
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
      auto f = op->getParentOfType<func::FuncOp>();
      if (f) {
        for (auto arg : f.getArguments()) {
          if (op.getTarget() == arg)
            return false;
          else if (op.getSource() == arg)
            return false;
        }
      }
      return true;
    });
    RewritePatternSet patterns(ctx);
    patterns.add<DmaToNpuPattern, HerdLoadToNpuPattern, SegmentLoadToNpuPattern,
                 ModuleMetadataToNpuPattern, L1MemRefStoreOpConversion,
                 L1AffineStoreOpConversion, HostMemRefCopyOpConversion,
                 AIRRtAllocOpConversion, AIRRtDeallocOpConversion>(ctx);
    patterns.add<AIRRtWaitAllOpConversion>(ctx, clOutputElf);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // Simplify arith ops (from airrt-to-npu)
    RewritePatternSet canoPatterns_2(ctx);
    canoPatterns_2.insert<RelocateAssumeAlignmentOp>(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns_2, ctx);
    (void)applyPatternsGreedily(module, std::move(canoPatterns_2));

    // Unroll any affine for loops
    unrollAffineFors(module);

    // Buffer npu.dma_memcpy_nd memref to function's argument list.
    RewritePatternSet castPattern(ctx);
    air::populateBufferMemrefToFuncArgsPattern(castPattern);
    (void)applyPatternsGreedily(module, std::move(castPattern));

    // Convert NpuDmaWaitOp → DMAAwaitTaskOp AFTER DMA conversion.
    // NpuDmaWaitOp was placed at WaitAllOp locations (clustered), and now we
    // replace each one with DMAAwaitTaskOp referencing the corresponding
    // DMAConfigureTaskForOp result.
    generateAwaitsFromWaitAllOps(module);

    // Renumber npu dma ops
    renumberNpuDmaOps(module.getBody());

    // Configure the tile trace units and the shimDMA
    if (clTraceSize > 0)
      if (failed(insertNpuWrite32ForTrace(module, clTraceSize, clTraceOffset)))
        signalPassFailure();

    RewritePatternSet funcToSeqPatterns(ctx);
    funcToSeqPatterns.add<ControlFuncConversion>(ctx);

    if (failed(applyPartialConversion(module, target,
                                      std::move(funcToSeqPatterns))))
      signalPassFailure();

    // Generate main device wrapper if needed. This handles two mutually
    // exclusive cases:
    // 1. Multi-device: pendingMainDevice was set by moveFuncOpToEndOfDeviceOp
    // 2. Single device fallback: XRTRunner path with emit-main-device flag
    // This MUST run at the very end after ALL patterns that modify
    // runtime_sequence arguments.
    generateMainDeviceIfNeeded(module);
  }

  void moveFuncOpToEndOfDeviceOp(ModuleOp module) {
    // Collect all func ops that need to be processed
    SmallVector<func::FuncOp> funcOps;
    module.walk([&](func::FuncOp f) {
      // Only process functions that contain segment/herd load ops
      bool hasSegmentOrHerd = false;
      f.walk([&](Operation *o) {
        if (isa<airrt::SegmentLoadOp, airrt::HerdLoadOp>(o))
          hasSegmentOrHerd = true;
      });
      if (hasSegmentOrHerd)
        funcOps.push_back(f);
    });

    for (auto funcOp : funcOps) {
      // Identify launch regions (affine.for with affine_opt_label containing
      // segment_load)
      SmallVector<LaunchRegion> regions = identifyLaunchRegions(funcOp, module);

      if (regions.empty()) {
        // Fallback: no launch boundaries found, use old behavior
        funcOp.walk([&](Operation *o) {
          if (isa<airrt::SegmentLoadOp, airrt::HerdLoadOp>(o)) {
            auto d = getDeviceForSegmentLoad(o);
            if (d)
              funcOp->moveBefore(d.getBody()->getTerminator());
          }
        });
        continue;
      }

      // Group regions by device
      llvm::MapVector<AIE::DeviceOp, SmallVector<LaunchRegion *>>
          deviceToRegions;
      for (auto &region : regions) {
        deviceToRegions[region.device].push_back(&region);
      }

      // If all regions target the same device and we're not forcing main
      // device generation, just move the entire func to that device
      if (deviceToRegions.size() == 1 && !clOutputElf) {
        AIE::DeviceOp device = deviceToRegions.begin()->first;
        funcOp->moveBefore(device.getBody()->getTerminator());
        continue;
      }

      // Multiple devices: verify all have the same device type
      AIE::AIEDevice deviceType = deviceToRegions.begin()->first.getDevice();
      for (auto &[device, _] : deviceToRegions) {
        if (device.getDevice() != deviceType) {
          funcOp.emitError("Multiple devices with different device types "
                           "are not supported");
          signalPassFailure();
          return;
        }
      }

      // Collect prologue ops (constants and other shared ops)
      SmallVector<Operation *> prologueOps =
          collectPrologueOps(funcOp, regions);

      OpBuilder builder(module.getContext());

      // For each device, create a new func with device-specific name
      for (auto &[device, deviceRegions] : deviceToRegions) {
        builder.setInsertionPoint(device.getBody()->getTerminator());

        // Create new function with device-specific name (e.g.,
        // add_two_sequence)
        std::string newFuncName = device.getSymName().str() + "_sequence";
        auto newFuncOp = func::FuncOp::create(
            builder, funcOp.getLoc(), newFuncName, funcOp.getFunctionType());
        newFuncOp.setVisibility(funcOp.getVisibility());

        // Create entry block with same arguments
        Block *entryBlock = newFuncOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        // Map from old values to new values
        IRMapping mapper;
        for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
          mapper.map(funcOp.getArgument(i), newFuncOp.getArgument(i));
        }

        // Clone prologue ops
        for (Operation *op : prologueOps) {
          builder.clone(*op, mapper);
        }

        // Clone each launch region for this device
        for (LaunchRegion *region : deviceRegions) {
          builder.clone(*region->boundaryOp.getOperation(), mapper);
        }

        // Add return
        func::ReturnOp::create(builder, funcOp.getLoc());
      }

      // Record pending main device creation - will be done at the end of the
      // pass after all argument-modifying patterns have run
      pendingMainDevice = PendingMainDevice{};
      pendingMainDevice->loc = funcOp.getLoc();
      pendingMainDevice->deviceType = deviceType;
      pendingMainDevice->mainSeqName = funcOp.getName().str();
      for (auto &region : regions) {
        pendingMainDevice->deviceNames.push_back(region.deviceName.str());
        pendingMainDevice->sequenceNames.push_back(region.deviceName.str() +
                                                   "_sequence");
      }

      // Erase the original function
      funcOp.erase();
    }
  }

  // Wrap existing aie.device ops with a main device when emit-main-device is
  // set but no func.func with segment_load was processed. This handles the
  // XRTRunner path where IR goes directly to AIE dialect with
  // runtime_sequence.
  void wrapExistingDevicesWithMainIfNeeded(ModuleOp module) {
    // Only proceed if output-elf mode is enabled
    if (!clOutputElf)
      return;

    // If pendingMainDevice is set, createDeferredMainDeviceWrapper will
    // handle main device creation instead
    if (pendingMainDevice)
      return;

    // Check if a "main" device already exists (created by
    // moveFuncOpToEndOfDeviceOp)
    bool mainDeviceExists = false;
    module.walk([&](AIE::DeviceOp d) {
      if (d.getSymName() == "main")
        mainDeviceExists = true;
    });

    if (mainDeviceExists)
      return;

    // Find existing devices that have runtime_sequence but no main wrapper
    SmallVector<AIE::DeviceOp> devices;
    module.walk([&](AIE::DeviceOp d) { devices.push_back(d); });

    // Only handle the single-device case for now
    if (devices.size() != 1)
      return;

    AIE::DeviceOp device = devices[0];
    AIE::RuntimeSequenceOp existingSeq = nullptr;
    device.walk([&](AIE::RuntimeSequenceOp seq) { existingSeq = seq; });

    if (!existingSeq)
      return;

    // Get the original sequence name and rename it to <device>_sequence
    StringRef deviceName = device.getSymName();
    std::string originalSeqName = existingSeq.getSymName().str();
    std::string newSeqName = deviceName.str() + "_sequence";

    // Rename the existing sequence
    OpBuilder builder(module.getContext());
    existingSeq->setAttr(SymbolTable::getSymbolAttrName(),
                         builder.getStringAttr(newSeqName));

    // Collect argument types and locations from existing sequence
    SmallVector<DeviceSequenceInfo> deviceSequences;
    DeviceSequenceInfo devInfo;
    devInfo.deviceName = device.getSymName();
    devInfo.sequenceName = newSeqName;
    for (auto arg : existingSeq.getBody().getArguments()) {
      devInfo.argTypes.push_back(arg.getType());
      devInfo.argLocs.push_back(arg.getLoc());
    }
    deviceSequences.push_back(devInfo);

    // Create main device wrapper using the helper function
    createMainDeviceWrapper(module, device.getLoc(), device.getDevice(),
                            originalSeqName, deviceSequences);
  }

  // Unified entry point for main device generation. Handles two mutually
  // exclusive cases:
  // 1. Multi-device: pendingMainDevice was set by moveFuncOpToEndOfDeviceOp
  // 2. Single device fallback: XRTRunner path with emit-main-device flag
  void generateMainDeviceIfNeeded(ModuleOp module) {
    // Early exit if no main device generation is needed
    if (!clOutputElf && !pendingMainDevice) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping main device generation: not requested\n");
      return;
    }

    // Two mutually exclusive paths based on how the IR was processed
    if (pendingMainDevice) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Creating main device for multi-device func.func\n");
      createDeferredMainDeviceWrapperImpl(module);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Creating main device wrapper for existing device\n");
      wrapExistingDevicesWithMainIfNeeded(module);
    }
  }

  // Create the deferred main device wrapper from func.func that was split
  // into multiple device-specific functions. This reads the FINAL argument
  // list from the runtime_sequences after all patterns (including
  // buffer-to-funcargs) have run.
  void createDeferredMainDeviceWrapperImpl(ModuleOp module) {
    if (!pendingMainDevice)
      return;

    // Build DeviceSequenceInfo by reading the FINAL argument list from each
    // device's runtime_sequence. This is done AFTER ControlFuncConversion
    // has converted func.func to runtime_sequence AND after
    // buffer-to-funcargs has added output memrefs to the argument list.
    SmallVector<DeviceSequenceInfo> deviceSequences;

    for (unsigned i = 0; i < pendingMainDevice->deviceNames.size(); ++i) {
      StringRef deviceName = pendingMainDevice->deviceNames[i];
      StringRef sequenceName = pendingMainDevice->sequenceNames[i];

      AIE::DeviceOp device = getDeviceByName(module, deviceName);
      if (!device)
        continue;

      // Find the runtime_sequence with the expected name
      AIE::RuntimeSequenceOp seq = nullptr;
      device.walk([&](AIE::RuntimeSequenceOp s) {
        if (s.getSymName() == sequenceName)
          seq = s;
      });

      if (!seq)
        continue;

      DeviceSequenceInfo devInfo;
      devInfo.deviceName = deviceName.str();
      devInfo.sequenceName = sequenceName.str();

      // Read the FINAL argument list from the runtime_sequence
      for (auto arg : seq.getBody().getArguments()) {
        devInfo.argTypes.push_back(arg.getType());
        devInfo.argLocs.push_back(arg.getLoc());
      }

      deviceSequences.push_back(devInfo);
    }

    if (deviceSequences.empty())
      return;

    // Create the main device wrapper with the correct (final) argument types
    createMainDeviceWrapper(module, pendingMainDevice->loc,
                            pendingMainDevice->deviceType,
                            pendingMainDevice->mainSeqName, deviceSequences);

    // Clear the pending request
    pendingMainDevice = std::nullopt;
  }

  // Convert WaitAllOp → NpuDmaWaitOp and purge DMA async tokens.
  // This must happen BEFORE DMA conversion.
  void generateNpuWaitFromAIRRtWaitAll(ModuleOp module) {
    auto ctx = module.getContext();

    // Apply the pattern to convert WaitAllOp → NpuDmaWaitOp
    RewritePatternSet patterns(ctx);
    patterns.insert<AIRRtWaitAllOpToNpuWaitPattern>(ctx, clOutputElf);
    (void)applyPatternsGreedily(module, std::move(patterns));

    // Now that WaitAllOps with DMA operands are erased, purge DMA async
    // tokens (they no longer have uses from WaitAllOps)
    purgeDmaAsyncTokens(module);
  }

  // Convert NpuDmaWaitOp → DMAAwaitTaskOp or DMAFreeTaskOp AFTER DMA
  // conversion. This processes NpuDmaWaitOp ops and DMAConfigureTaskForOp ops
  // in order, matching each wait to its corresponding configure task by
  // channel. The key insight: waits and configures for the same channel must
  // be matched in FIFO order - the Nth wait for channel X awaits the Nth
  // config for X.
  //
  // For S2MM (output) channels: generate DMAAwaitTaskOp (wait + free BD)
  // For MM2S (input) channels: generate DMAFreeTaskOp (just free BD, no wait)
  void generateAwaitsFromWaitAllOps(ModuleOp module) {
    SmallVector<func::FuncOp> funcOps;
    module.walk([&](func::FuncOp f) { funcOps.push_back(f); });

    for (auto f : funcOps) {
      auto device = f->getParentOfType<AIE::DeviceOp>();
      if (!device)
        continue;

      if (f.getBody().empty())
        continue;

      // First pass: collect all DMAConfigureTaskForOp per channel in order
      // Map from metadata symbol -> list of ConfigTasks in order
      llvm::MapVector<StringRef, SmallVector<AIEX::DMAConfigureTaskForOp>>
          channelToConfigTasks;

      // Also track per-channel indices for matching
      llvm::DenseMap<StringRef, unsigned> channelToNextConfigIdx;

      // Walk the function body in order
      f.walk([&](AIEX::DMAConfigureTaskForOp configTask) {
        auto allocSymbol = configTask.getAlloc();
        StringRef metadata = allocSymbol.getLeafReference().getValue();
        channelToConfigTasks[metadata].push_back(configTask);
      });

      // Initialize indices
      for (auto &kv : channelToConfigTasks) {
        channelToNextConfigIdx[kv.first] = 0;
      }

      // Second pass: process NpuDmaWaitOp ops in order
      // For each wait, find the next unconsumed ConfigTask for that channel
      SmallVector<AIEX::NpuDmaWaitOp> waitOps;
      f.walk([&](AIEX::NpuDmaWaitOp waitOp) { waitOps.push_back(waitOp); });

      for (auto waitOp : waitOps) {
        StringRef metadata = waitOp.getSymbol();

        // Determine channel direction
        // First try ShimDMAAllocationOp
        auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(device, metadata);
        bool isS2MM = false;
        if (allocOp) {
          isS2MM = allocOp.getChannelDir() == AIE::DMAChannelDir::S2MM;
        } else {
          // Check for objectfifo - if consumer is shim tile, it's S2MM
          auto objFifo = device.lookupSymbol<AIE::ObjectFifoCreateOp>(metadata);
          if (objFifo) {
            for (auto consumerTileOp : objFifo.getConsumerTiles()) {
              auto consTileOp = consumerTileOp.getDefiningOp<AIE::TileOp>();
              if (consTileOp && consTileOp.isShimTile()) {
                isS2MM = true;
                break;
              }
            }
          }
        }

        // Find the next ConfigTask for this channel
        AIEX::DMAConfigureTaskForOp matchingConfigTask = nullptr;
        auto it = channelToConfigTasks.find(metadata);
        if (it != channelToConfigTasks.end()) {
          auto &configTasks = it->second;
          unsigned &nextIdx = channelToNextConfigIdx[metadata];
          if (nextIdx < configTasks.size()) {
            matchingConfigTask = configTasks[nextIdx];
            nextIdx++;
          }
        }

        if (matchingConfigTask) {
          OpBuilder builder(waitOp);
          if (isS2MM) {
            // S2MM (output): await task - waits for completion AND frees BD
            AIEX::DMAAwaitTaskOp::create(builder, waitOp.getLoc(),
                                         matchingConfigTask.getResult());
          } else {
            // MM2S (input): free task - just frees BD for reuse, no wait
            AIEX::DMAFreeTaskOp::create(builder, waitOp.getLoc(),
                                        matchingConfigTask.getResult());
          }
        }
        // Erase the NpuDmaWaitOp regardless of whether we found a match
        waitOp->erase();
      }
    }
  }

  // Purge DMA async tokens - they are no longer needed after WaitAllOp
  // processing. Call this BEFORE DMA conversion.
  void purgeDmaAsyncTokens(ModuleOp module) {
    SmallVector<airrt::DmaMemcpyNdOp> dmas;
    module.walk([&](airrt::DmaMemcpyNdOp dma) { dmas.push_back(dma); });
    for (auto dma : dmas) {
      if (dma->getNumResults()) {
        OpBuilder builder(dma);
        SmallVector<Type, 1> tys;
        auto newOp = airrt::DmaMemcpyNdOp::create(builder, dma->getLoc(), tys,
                                                  dma->getOperands());
        newOp->setAttrs(dma->getDiscardableAttrDictionary());
        dma->erase();
      }
    }
  }

  // Set all X and Y values of airrt::dma_memcpy_nd ops to 0.
  void purgeAIRRtDmaXAndY(ModuleOp module) {
    auto i64Ty = IntegerType::get(module.getContext(), 64);
    SmallVector<airrt::DmaMemcpyNdOp> dmas;
    module.walk([&](airrt::DmaMemcpyNdOp dma) { dmas.push_back(dma); });
    for (auto dma : dmas) {
      OpBuilder builder(dma);
      bool resetX = !(getConstantIntValue(dma.getX()) &&
                      *getConstantIntValue(dma.getX()) == 0);
      bool resetY = !(getConstantIntValue(dma.getY()) &&
                      *getConstantIntValue(dma.getY()) == 0);
      if (resetX)
        dma.getXMutable().assign(arith::ConstantOp::create(
            builder, dma->getLoc(), i64Ty, IntegerAttr::get(i64Ty, 0)));
      if (resetY)
        dma.getYMutable().assign(arith::ConstantOp::create(
            builder, dma->getLoc(), i64Ty, IntegerAttr::get(i64Ty, 0)));
    }
  }

  void purgeDuplicateShimDmaAllocs(ModuleOp module) {
    // Process each device separately to avoid cross-device deduplication
    SmallVector<AIE::DeviceOp> devices;
    module.walk([&](AIE::DeviceOp d) { devices.push_back(d); });

    for (auto device : devices) {
      llvm::SetVector<AIE::ShimDMAAllocationOp> allocs;
      device.walk(
          [&](AIE::ShimDMAAllocationOp alloc) { allocs.insert(alloc); });
      llvm::SmallSet<AIE::ShimDMAAllocationOp, 1> uniqueAllocs;

      // Map each unique set of <dir, chan, col> to a shim dma alloc op
      // within THIS device only
      DenseMap<StringRef, StringRef> uniqueAllocMap;
      for (auto alloc : allocs) {
        AIE::TileOp shimtile = alloc.getTileOp();
        std::tuple<bool, int, int> allocInfo = {
            alloc.getChannelDir() == AIE::DMAChannelDir::MM2S,
            alloc.getChannelIndex(), shimtile.getCol()};

        auto it =
            llvm::find_if(uniqueAllocs, [&](AIE::ShimDMAAllocationOp ualloc) {
              AIE::TileOp shimtile = ualloc.getTileOp();
              std::tuple<bool, int, int> uallocInfo = {
                  ualloc.getChannelDir() == AIE::DMAChannelDir::MM2S,
                  ualloc.getChannelIndex(), shimtile.getCol()};
              return allocInfo == uallocInfo;
            });
        if (it != uniqueAllocs.end()) {
          AIE::ShimDMAAllocationOp uniqueAlloc = *it;
          uniqueAllocMap[alloc.getSymName()] = uniqueAlloc.getSymName();
        } else {
          uniqueAllocs.insert(alloc);
          uniqueAllocMap[alloc.getSymName()] = alloc.getSymName();
        }
      }

      // Replace all uses of metadata to unique within THIS device only
      device.walk([&](airrt::DmaMemcpyNdOp dma) {
        if (!dma->hasAttr("metadata"))
          return;
        StringRef metadata =
            dma->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata").getValue();
        if (!uniqueAllocMap.count(metadata))
          return;
        if (uniqueAllocMap[metadata] != metadata) {
          dma->setAttr("metadata",
                       FlatSymbolRefAttr::get(dma->getContext(),
                                              uniqueAllocMap[metadata]));
        }
      });
    }
  }

  void unrollAffineFors(ModuleOp module) {
    // Taking into account for loop nests
    module.walk([&](mlir::func::FuncOp f) {
      SmallVector<affine::AffineForOp> afos;
      for (auto op : f.getOps<affine::AffineForOp>()) {
        afos.push_back(op);
      }
      for (auto op : afos) {
        unrollAffineFor(op);
        // Renumber unrolled memcpy ops
        int unrolled_op_id = 0;
        f.walk([&](airrt::DmaMemcpyNdOp dma) {
          if (dma->hasAttr("unrolled")) {
            auto metadata =
                dma->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")
                    .getValue()
                    .str();
            // Start from unrolled_op_id 1
            if (unrolled_op_id)
              dma->setAttr("metadata", FlatSymbolRefAttr::get(
                                           dma->getContext(),
                                           metadata + "_" +
                                               std::to_string(unrolled_op_id)));
            unrolled_op_id++;
            dma->removeAttr("unrolled");
          }
        });
      }
    });
  }

  void unrollAffineFor(affine::AffineForOp affine_for_op) {
    SmallVector<affine::AffineForOp> afos;
    affine_for_op.walk([&](affine::AffineForOp afo) { afos.push_back(afo); });
    for (auto afo : afos)
      if (failed(loopUnrollFull(afo))) {
        afo->emitOpError("failed to fully unroll");
        signalPassFailure();
      }
  }

  void unrollSCFFors(ModuleOp module) {
    SmallVector<scf::ForOp> scf_fors;
    module.walk([&](mlir::func::FuncOp f) {
      f.walk([&](scf::ForOp for_op) { scf_fors.push_back(for_op); });
    });
    for (auto for_op : scf_fors) {
      std::optional<int64_t> lbCstOp =
          mlir::getConstantIntValue(for_op.getLowerBound());
      std::optional<int64_t> ubCstOp =
          mlir::getConstantIntValue(for_op.getUpperBound());
      std::optional<int64_t> stepCstOp =
          mlir::getConstantIntValue(for_op.getStep());
      if (lbCstOp && ubCstOp && stepCstOp) {
        if (failed(loopUnrollFull(for_op))) {
          for_op->emitOpError("failed to fully unroll");
          signalPassFailure();
        }
      }
    }
  }

  void purgeSCFParContainingOnlyWaitAllOps(ModuleOp module) {
    SmallVector<scf::ParallelOp> scf_pars;
    module.walk([&](mlir::func::FuncOp f) {
      f.walk([&](scf::ParallelOp par_op) { scf_pars.push_back(par_op); });
    });
    OpBuilder builder(module);
    for (auto par_op : scf_pars) {
      bool containsOnlyWaitAll = true;
      par_op.walk([&](Operation *o) {
        if (isa<airrt::WaitAllOp>(o))
          return;
        else if (isa<scf::ParallelOp>(o))
          return;
        else if (o->mightHaveTrait<OpTrait::IsTerminator>())
          return;
        else {
          containsOnlyWaitAll = false;
          return;
        }
      });
      builder.setInsertionPoint(par_op);
      auto newWaitAll = airrt::WaitAllOp::create(
          builder, par_op->getLoc(),
          airrt::EventType::get(par_op->getContext()), par_op.getInitVals());
      for (auto res : par_op->getResults())
        res.replaceAllUsesWith(newWaitAll->getResult(0));
      par_op->erase();
    }
  }

  std::optional<AIE::ShimDMAAllocationOp>
  getAllocOpForSymbol(SmallVector<AIE::ShimDMAAllocationOp> shimDmaAllocOps,
                      StringRef sym_name) {
    for (auto shimDmaAllocOp : shimDmaAllocOps)
      if (shimDmaAllocOp.getSymName() == sym_name)
        return shimDmaAllocOp;
    return std::nullopt;
  }

  std::optional<AIE::ObjectFifoCreateOp> getObjectFifoCreateOpForSymbol(
      SmallVector<AIE::ObjectFifoCreateOp> objectFifoCreateOps,
      StringRef sym_name) {
    for (auto objectFifoCreateOp : objectFifoCreateOps)
      if (objectFifoCreateOp.getSymName().str() == sym_name.str())
        return objectFifoCreateOp;
    return std::nullopt;
  }

  // Remove npu wait op on inbound dma data movements.
  // TODO: this is an aggressive optimization which might prove problematic
  // for some applications. To be revised.
  void removeNpuWaitOnInboundMemcpy(ModuleOp module) {
    SmallVector<mlir::func::FuncOp> funcOps;
    module.walk([&](mlir::func::FuncOp f) { funcOps.push_back(f); });
    for (auto f : funcOps) {
      SmallVector<AIEX::NpuDmaWaitOp> waits;
      f.walk([&](AIEX::NpuDmaWaitOp wait) { waits.push_back(wait); });
      auto d = f->getParentOfType<AIE::DeviceOp>();

      SmallVector<AIE::ShimDMAAllocationOp> shimDmaAllocOps;
      if (d)
        d.walk([&](AIE::ShimDMAAllocationOp shimDmaAllocOp) {
          shimDmaAllocOps.push_back(shimDmaAllocOp);
        });
      llvm::DenseMap<StringRef, std::optional<AIE::ShimDMAAllocationOp>>
          allocationCache;

      if (!d)
        continue;
      OpBuilder builder(f);
      for (auto wait : waits) {
        auto infoOp =
            AIE::ShimDMAAllocationOp::getForSymbol(d, wait.getSymbol());
        if (!infoOp)
          continue;
        if (infoOp.getChannelDir() != AIE::DMAChannelDir::MM2S)
          continue;
        // Found dma op copying results from host to device
        wait->erase();
      }
    }
  }

  // Each element of 'port' is a {Port_N_Master_Slave, Port_N_ID} pair. They
  // will be read sequentially to select up to 8 stream switch ports to
  // monitor, using the select register at address {col, row, offset}.
  void insertNpuWriteStreamSwitchEventSel(
      OpBuilder &builder, std::vector<std::pair<uint8_t, uint8_t>> &ports,
      uint32_t offset, IntegerAttr col, IntegerAttr row) {
    uint32_t v0 = 0;
    for (unsigned i = 0; i < std::min(ports.size(), 4UL); i++) {
      v0 |= (ports[i].second << (i * 8));
      v0 |= (ports[i].first << ((i * 8) + 5));
    }
    AIEX::NpuWrite32Op::create(builder, builder.getUnknownLoc(), offset, v0,
                               nullptr, col, row);
    uint32_t v1 = 0;
    if (ports.size() > 4)
      for (unsigned i = 4; i < std::min(ports.size(), 8UL); i++) {
        v1 |= (ports[i].second << ((i - 4) * 8));
        v1 |= (ports[i].first << (((i - 4) * 8) + 5));
      }
    AIEX::NpuWrite32Op::create(builder, builder.getUnknownLoc(), offset + 0x4,
                               v1, nullptr, col, row);
  }

  // up to 8 events (up to 64 bits) will be written to the 8 event slots
  // (bytes) at address {col, row, offset}
  void insertNpuWriteTraceEvents(OpBuilder &builder,
                                 SmallVectorImpl<uint32_t> &events,
                                 uint32_t offset, IntegerAttr col,
                                 IntegerAttr row) {
    uint32_t v0 = 0;
    for (unsigned i = 0; i < std::min(events.size(), 4UL); i++)
      v0 |= ((events[i] & 0xff) << (i * 8));
    uint32_t v1 = 0;
    if (events.size() > 4)
      for (unsigned i = 4; i < std::min(events.size(), 8UL); i++)
        v1 |= ((events[i] & 0xff) << ((i - 4) * 8));

    AIEX::NpuWrite32Op::create(builder, builder.getUnknownLoc(), offset, v0,
                               nullptr, col, row);
    AIEX::NpuWrite32Op::create(builder, builder.getUnknownLoc(), offset + 0x4,
                               v1, nullptr, col, row);
  }

  // configure events to monitor
  LogicalResult insertNpuWrite32ForTrace(ModuleOp module, int64_t trace_size,
                                         int64_t trace_offset) {
    SmallVector<mlir::func::FuncOp> funcOps;
    module.walk([&](mlir::func::FuncOp f) { funcOps.push_back(f); });

    for (auto f : funcOps) {
      OpBuilder builder(f);
      auto d = f->getParentOfType<AIE::DeviceOp>();
      if (!d)
        continue;

      auto &target_model = d.getTargetModel();
      std::map<int, int> chanToIdMap;
      if (f.getBody().empty())
        continue;
      builder.setInsertionPointToStart(&f.front());
      for (auto pktFlow : d.getOps<AIE::PacketFlowOp>()) {
        Region &r = pktFlow.getPorts();
        Block &b = r.front();
        int flowID = pktFlow.IDInt();
        AIE::Port sourcePort, destPort;
        AIE::TileOp srcTile, destTile;

        // find all packet flow with trace port as source
        for (Operation &Op : b.getOperations()) {
          if (auto pktSrc = dyn_cast<AIE::PacketSourceOp>(Op)) {
            srcTile = dyn_cast<AIE::TileOp>(pktSrc.getTile().getDefiningOp());
            sourcePort = pktSrc.port();
          } else if (auto pktDest = dyn_cast<AIE::PacketDestOp>(Op)) {
            destTile = dyn_cast<AIE::TileOp>(pktDest.getTile().getDefiningOp());
            destPort = pktDest.port();
          }
        }
        if (sourcePort.bundle != AIE::WireBundle::Trace)
          continue;

        int srcColIndex = srcTile.colIndex();
        int srcRowIndex = srcTile.rowIndex();
        int dstColIndex = destTile.colIndex();
        int dstRowIndex = destTile.rowIndex();
        if (!target_model.isCoreTile(srcColIndex, srcRowIndex) &&
            !target_model.isMemTile(srcColIndex, srcRowIndex)) {
          pktFlow->emitOpError("unsupported trace src.");
          return failure();
        }
        if (!target_model.isShimNOCTile(dstColIndex, dstRowIndex)) {
          pktFlow->emitOpError("unsupported trace dest.");
          return failure();
        }
        int pkt_type = 0;
        if (target_model.isMemTile(srcColIndex, srcRowIndex))
          pkt_type = 3;
        else if (sourcePort.channel == 1)
          pkt_type = 1;
        int buff_size = trace_size / target_model.columns();
        int buff_offset = trace_offset; // todo: get from func args?
        buff_offset += dstColIndex * buff_size;
        auto col = builder.getIntegerAttr(builder.getI32Type(), srcColIndex);
        auto row = builder.getIntegerAttr(builder.getI32Type(), srcRowIndex);

        // configure tile trace
        if (target_model.isCoreTile(srcColIndex, srcRowIndex)) {
          // event boardcast to sync timer
          uint32_t core_reg_timer_control = 0x34000;
          uint32_t core_reg_trace_control0 = 0x340D0;
          uint32_t core_reg_trace_control1 = 0x340D4;
          uint32_t core_event_broadcast_15 = 122;
          AIEX::NpuWrite32Op::create(
              builder, builder.getUnknownLoc(), core_reg_timer_control,
              core_event_broadcast_15 << 8, nullptr, col, row);
          AIEX::NpuWrite32Op::create(
              builder, builder.getUnknownLoc(), core_reg_trace_control0,
              core_event_broadcast_15 << 16, nullptr, col, row);
          AIEX::NpuWrite32Op::create(
              builder, builder.getUnknownLoc(), core_reg_trace_control1,
              pkt_type << 12 | flowID, nullptr, col, row);

          // configure events to monitor
          // todo: allow user to specify?
          // INSTR_VECTOR, INSTR_EVENT_1, INSTR_EVENT_0, true,
          // PORT_RUNNING_1 PORT_RUNNING_0, LOCK_RELEASE_REQ,LOCK_ACQUIRE_REQ
          SmallVector<uint32_t> trace_events = {37, 34, 33, 1, 79, 75, 45, 44};
          uint32_t core_reg_trace_event0 = 0x340E0;
          insertNpuWriteTraceEvents(builder, trace_events,
                                    core_reg_trace_event0, col, row);

          // configure ports to monitor
          // todo: allow user to specify?
          // {Port_N_Master_Slave, Port_N_ID}
          std::vector<std::pair<uint8_t, uint8_t>> ports{{1, 1}, {0, 1}};
          uint32_t core_reg_strm_sw_event_sel_0 = 0x3FF00;
          insertNpuWriteStreamSwitchEventSel(
              builder, ports, core_reg_strm_sw_event_sel_0, col, row);

        } else if (target_model.isMemTile(dstColIndex, srcRowIndex)) {
          // event boardcast to sync timer
          uint32_t mem_reg_timer_control = 0x94000;
          uint32_t mem_reg_trace_control0 = 0x940D0;
          uint32_t mem_reg_trace_control1 = 0x940D4;
          uint32_t mem_event_broadcast_15 = 157;
          AIEX::NpuWrite32Op::create(
              builder, builder.getUnknownLoc(), mem_reg_timer_control,
              mem_event_broadcast_15 << 8, nullptr, col, row);
          AIEX::NpuWrite32Op::create(
              builder, builder.getUnknownLoc(), mem_reg_trace_control0,
              mem_event_broadcast_15 << 16, nullptr, col, row);
          AIEX::NpuWrite32Op::create(
              builder, builder.getUnknownLoc(), mem_reg_trace_control1,
              pkt_type << 12 | flowID, nullptr, col, row);

          // configure events to monitor
          // todo: allow user to specify?
          // PORT_RUNNING_2, PORT_RUNNING_1, PORT_RUNNING_0, true,
          // PORT_RUNNING_6, PORT_RUNNING_5, PORT_RUNNING_4, PORT_RUNNING_3
          SmallVector<uint32_t> trace_events = {88,  84,  80, 1,
                                                104, 100, 96, 92};
          uint32_t mem_reg_trace_event0 = 0x940E0;
          insertNpuWriteTraceEvents(builder, trace_events, mem_reg_trace_event0,
                                    col, row);

          // {Port_N_Master_Slave, Port_N_ID}
          std::vector<std::pair<uint8_t, uint8_t>> ports{
              {1, 0}, {1, 1}, {1, 2}, {0, 0}, {0, 1}, {0, 2}, {0, 3}};
          uint32_t mem_reg_strm_sw_event_sel_0 = 0xB0F00;
          insertNpuWriteStreamSwitchEventSel(
              builder, ports, mem_reg_strm_sw_event_sel_0, col, row);
        }

        // configure shim tile
        if (chanToIdMap.count(dstColIndex) == 0)
          chanToIdMap[dstColIndex] = 15;
        int bdID = chanToIdMap[dstColIndex];
        if (bdID < 4) {
          pktFlow->emitOpError("runs out of bd_id.");
          return failure();
        }

        AIEX::NpuWriteBdOp::create(
            builder, builder.getUnknownLoc(), dstColIndex, bdID, buff_size,
            buff_offset,
            /*enable_packet*/ 1, /*out_of_order_id*/ 0,
            /*packet_id*/ flowID, pkt_type,
            /* d0_size */ 0, /* d0_stride */ 0, /* d1_size */ 0,
            /* d1_stride */ 0, /* d2_size */ 0, /* d2_stride */ 0,
            /* iteration_current */ 0, /* iteration_size */ 0,
            /* iteration_stride */ 0, /* next_bd */ 0, dstRowIndex,
            /* use_next_bd */ 0,
            /* valid_bd */ 1, /* lock_rel_val */ 0, /* lock_rel_id */ 0,
            /* lock_acq_enable */ 0, /* lock_acq_val */ 0,
            /* lock_acq_id */ 0,
            /* d0_zero_before */ 0, /* d1_zero_before */ 0,
            /* d2_zero_before */ 0,
            /* d0_zero_after */ 0, /* d1_zero_after */ 0,
            /* d2_zero_after */ 0);
        uint32_t addr = (dstColIndex << target_model.getColumnShift()) |
                        (0x1D004 + bdID * 0x20);
        AIEX::NpuAddressPatchOp::create(builder, builder.getUnknownLoc(), addr,
                                        /* ddr_id */ 2, buff_offset);

        int address;
        if (destPort.channel == 0)
          address = 0x1D204;
        else if (destPort.channel == 1)
          address = 0x1D20C;
        else {
          pktFlow->emitOpError("unknown trace dest.");
          return failure();
        }
        AIEX::NpuWrite32Op::create(
            builder, builder.getUnknownLoc(), address, bdID, nullptr,
            builder.getIntegerAttr(builder.getI32Type(), dstColIndex),
            builder.getIntegerAttr(builder.getI32Type(), dstRowIndex));
        chanToIdMap[dstColIndex]--;
      }

      // broadcast event to sync timer
      auto zero = builder.getIntegerAttr(builder.getI32Type(), 0);
      AIEX::NpuWrite32Op::create(builder, builder.getUnknownLoc(), 0x34000,
                                 127 << 8, nullptr, zero, zero);
      AIEX::NpuWrite32Op::create(builder, builder.getUnknownLoc(), 0x3404C, 127,
                                 nullptr, zero, zero);
      AIEX::NpuWrite32Op::create(builder, builder.getUnknownLoc(), 0x34008, 127,
                                 nullptr, zero, zero);
    }
    return success();
  }

  // Renumber aiex.npu.dma_memcpy_nd ops per column of AIEs.
  void renumberNpuDmaOps(Block *blk) {
    std::map<int, int> chanToIdMap;
    AIE::DeviceOp d = nullptr;
    blk->walk([&](AIE::DeviceOp op) { d = op; });
    SmallVector<AIE::ShimDMAAllocationOp> shimDmaAllocOps;
    if (d)
      d.walk([&](AIE::ShimDMAAllocationOp shimDmaAllocOp) {
        shimDmaAllocOps.push_back(shimDmaAllocOp);
      });
    llvm::DenseMap<StringRef, std::optional<AIE::ShimDMAAllocationOp>>
        allocationCache;
    SmallVector<AIE::ObjectFifoCreateOp> objectFifoCreateOps;
    if (d)
      d.walk([&](AIE::ObjectFifoCreateOp objectFifoCreateOp) {
        objectFifoCreateOps.push_back(objectFifoCreateOp);
      });
    OpBuilder builder(blk->getParentOp());
    blk->walk([&](Operation *op) {
      auto dma = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op);
      auto sync = dyn_cast<AIEX::NpuSyncOp>(op);
      auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(op);
      if (sync || wait) {
        chanToIdMap.clear();
        return;
      }
      if (!dma)
        return;
      builder.setInsertionPoint(dma);
      int col = -1;
      if (d) {
        if (auto infoOp = AIE::ShimDMAAllocationOp::getForSymbol(
                d, dma.getMetadata().getRootReference())) {
          AIE::TileOp shimtile = infoOp.getTileOp();
          col = shimtile.getCol();
        } else if (auto objFifoCreateOp = getObjectFifoCreateOpForSymbol(
                       objectFifoCreateOps,
                       dma.getMetadata().getLeafReference().getValue())) {
          auto prodTileOp =
              objFifoCreateOp->getProducerTile().getDefiningOp<AIE::TileOp>();
          if (prodTileOp.isShimTile())
            col = prodTileOp.colIndex();
          for (auto consumerTileOp : objFifoCreateOp->getConsumerTiles()) {
            auto consTileOp = consumerTileOp.getDefiningOp<AIE::TileOp>();
            if (consTileOp.isShimTile()) {
              col = consTileOp.colIndex();
            }
          }
        }
      }
      if (!chanToIdMap.count(col))
        chanToIdMap[col] = 0;
      dma->setAttr("id", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(dma->getContext(), 64),
                             chanToIdMap[col]++));
    });
  }
};

} // namespace xilinx

namespace xilinx {
namespace airrt {

std::unique_ptr<mlir::Pass> createAIRRtToNpuPass() {
  return std::make_unique<AIRRtToNpuPass>();
}

} // namespace airrt
} // namespace xilinx
