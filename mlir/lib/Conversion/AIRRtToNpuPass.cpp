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
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
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

// Path B: airrt-to-npu runs before aie-place-tiles (which now lives only in
// aiecc). Read the shim col from either a physical aie.tile or, if the
// shim hasn't been placed yet, the col hint on aie.logical_tile<...>(col,?).
// AIR sets that hint to the compute-side col so the placer's hint-respecting
// behavior gives the same physical col here as it will downstream.
// Returns -1 if neither is available.
static int getColFromTileValue(mlir::Value tile) {
  if (!tile)
    return -1;
  mlir::Operation *def = tile.getDefiningOp();
  if (auto t = llvm::dyn_cast_or_null<xilinx::AIE::TileOp>(def))
    return t.getCol();
  if (auto lto = llvm::dyn_cast_or_null<xilinx::AIE::LogicalTileOp>(def))
    if (auto col = lto.tryGetCol())
      return *col;
  return -1;
}

// True if `tile` is a shim tile defining op. Accepts either a physical
// aie.tile or an unplaced aie.logical_tile<ShimNOCTile|ShimPLTile>.
static bool isShimTileValue(mlir::Value tile) {
  if (!tile)
    return false;
  mlir::Operation *def = tile.getDefiningOp();
  if (auto t = llvm::dyn_cast_or_null<xilinx::AIE::TileOp>(def))
    return t.isShimTile();
  if (auto lto = llvm::dyn_cast_or_null<xilinx::AIE::LogicalTileOp>(def))
    return lto.getTileType() == xilinx::AIE::AIETileType::ShimNOCTile ||
           lto.getTileType() == xilinx::AIE::AIETileType::ShimPLTile;
  return false;
}

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

// Helper function to check if an aie.device uses the cascade bus (cascade flows
// or core-side get/put_cascade). Cascade core locks/credits, like repeat_count
// DMA state, must be re-armed after each launch via the load_pdi reset (which
// runs initLocks). A single-trip launch has repeat_count == 0, so
// deviceHasRepeatCountDMAs() misses cascade kernels -> they abort on the 2nd
// host dispatch ("qds_device::wait() unexpected command state"). Detecting
// cascade lets the existing reset path fire for single-trip cascade too.
static bool deviceHasCascade(xilinx::AIE::DeviceOp device) {
  bool hasCascade = false;
  device.walk([&](mlir::Operation *op) {
    if (llvm::isa<xilinx::AIE::CascadeFlowOp, xilinx::AIE::GetCascadeOp,
                  xilinx::AIE::PutCascadeOp>(op))
      hasCascade = true;
  });
  return hasCascade;
}

// Static trip count of the affine loops enclosing `op` (1 if none). Used to
// tell a SINGLE-TRIP launch (the launch boundary runs once per host dispatch)
// from a MULTI-ITERATION launch (the boundary sits inside an iteration loop,
// e.g. flash-attention's lq_iters loop). A non-constant affine loop or any scf
// loop is treated as multi-trip. Must be evaluated before unrollAffineFors
// strips these loops.
static int64_t enclosingLoopTrips(mlir::Operation *op) {
  int64_t trips = 1;
  for (mlir::Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto forOp = llvm::dyn_cast<mlir::affine::AffineForOp>(p)) {
      if (!forOp.hasConstantBounds())
        return std::numeric_limits<int64_t>::max();
      int64_t lb = forOp.getConstantLowerBound();
      int64_t ub = forOp.getConstantUpperBound();
      int64_t step = forOp.getStepAsInt();
      int64_t n = (step > 0 && ub > lb) ? (ub - lb + step - 1) / step : 1;
      trips *= n;
    } else if (llvm::isa<mlir::scf::ForOp, mlir::scf::WhileOp>(p)) {
      return std::numeric_limits<int64_t>::max();
    }
  }
  return trips;
}

// Count air.launch_end markers in a device and whether any are enclosed by a
// multi-trip loop. Returns {count, anyMultiTrip}. Both
// deviceHasSingleTripCascade and markDevicesNeedingLockReset need this
// information; centralizing it here avoids duplicating the walk.
static std::pair<int64_t, bool> countLaunchEnds(xilinx::AIE::DeviceOp device) {
  int64_t n = 0;
  bool anyMultiTrip = false;
  device.walk([&](xilinx::airrt::WaitAllOp wa) {
    if (!wa->hasAttr("air.launch_end"))
      return;
    ++n;
    if (enclosingLoopTrips(wa) != 1)
      anyMultiTrip = true;
  });
  return {n, anyMultiTrip};
}

// Cascade core locks/credits need the per-launch reset only for a SINGLE-TRIP
// launch: across a host re-dispatch they are stale and the kernel aborts on the
// 2nd dispatch without the reset. A MULTI-ITERATION launch re-arms them every
// iteration on its own, so inserting the reset between iterations is
// unnecessary and costs a load_pdi PDI reload per boundary (a silent perf
// regression, e.g. flash-attention prefill). A multi-iteration launch presents
// two ways depending on whether its iteration loop survived: either one
// air.launch_end inside a multi-trip loop, or -- once the loop is unrolled --
// several air.launch_end markers. Single-trip is therefore exactly one
// air.launch_end that is not enclosed by a multi-trip loop. MUST be evaluated
// before the launch_end wait_all ops are converted/erased and before
// unrollAffineFors -- see markDevicesNeedingLockReset.
static bool deviceHasSingleTripCascade(xilinx::AIE::DeviceOp device) {
  if (!deviceHasCascade(device))
    return false;
  auto [numLaunchEnds, anyMultiTrip] = countLaunchEnds(device);
  return numLaunchEnds == 1 && !anyMultiTrip;
}

// Internal marker carrying the per-launch reset decision from
// markDevicesNeedingLockReset() (computed while the launch_end markers and
// their loops still exist) to the load_pdi insertion and reset-clone sites.
// Stripped at the end of the pass.
static constexpr llvm::StringLiteral kNeedsLockResetAttr =
    "air.needs_lock_reset";

// A device needs the per-launch lock/DMA-state reset (load_pdi + initLocks) if
// it has repeat_count DMAs OR is a single-trip cascade launch. The decision is
// cached on the device because the launch_end markers it depends on are erased
// during conversion (and the loops unrolled afterwards), so it cannot be
// recomputed at the insertion / reset-clone sites.
static bool deviceNeedsLockReset(xilinx::AIE::DeviceOp device) {
  return device->hasAttr(kNeedsLockResetAttr);
}

// Internal marker: the device's launch boundary runs MORE THAN ONCE per host
// dispatch (a fused multi-iteration launch, e.g. an scf.for wrapping air.launch
// to stitch N iterations into one dispatch). Such a launch must fully DRAIN
// every shim channel at each iteration boundary before the next iteration's
// feeds issue -- otherwise consecutive iterations overlap and a finite-depth
// (e.g. 2-slot ping-pong) device lock accumulates an imbalance and deadlocks
// after a few iterations. The single-dispatch shim drain that enforces this is
// otherwise only emitted on the output-elf path (issue #1373); this marker
// extends it to the multi-iteration xclbin path too. Computed while the
// launch_end markers + their loops still exist (they are erased during
// conversion), same as kNeedsLockResetAttr.
static constexpr llvm::StringLiteral kMultiIterLaunchAttr =
    "air.multi_iter_launch";

// Number of launch iterations (stitched iterations) for a fused multi-iteration
// launch, carried from markDevicesNeedingLockReset (computed from the unrolled
// launch_end count) to synthesizeDoubleBufferedAwaits, which must segment its
// per-channel paced-MM2S backpressure PER ITERATION so the double-buffered
// pacing does not overlap an iteration boundary (which accumulates a 2-deep
// in-flight imbalance and deadlocks after a couple of iterations).
static constexpr llvm::StringLiteral kNumLaunchItersAttr =
    "air.num_launch_iters";

static bool deviceHasMultiIterLaunch(xilinx::AIE::DeviceOp device) {
  return device->hasAttr(kMultiIterLaunchAttr);
}

// Stamp a per-op launch-iteration ("wave") index on the source airrt ops of a
// fused multi-iteration launch (airrt.dma_memcpy_nd / airrt.herd_load). Must
// run after the fused launch loop is unrolled (so each iteration's ops are laid
// out contiguously and program order reflects wave membership) and before the
// launch_end airrt.wait_all markers are lowered away. The index rides through
// lowering onto the emitted DMAConfigureTaskForOp / NpuWriteRTPOp / SetLockOp,
// so the downstream per-wave hoist groups by index rather than inferring wave
// boundaries from op positions. Gated to fused devices; single-dispatch funcs
// are left untagged (wave 0 everywhere) for byte-identical output.
static void assignLaunchWaveIndices(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp f) {
    if (f.getBody().empty())
      return;
    auto device = f->getParentOfType<xilinx::AIE::DeviceOp>();
    if (!device || !deviceHasMultiIterLaunch(device))
      return;
    int64_t wave = 0;
    auto i64 = mlir::IntegerType::get(f.getContext(), 64);
    f.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      if (isa<xilinx::airrt::DmaMemcpyNdOp, xilinx::airrt::HerdLoadOp>(op))
        op->setAttr(xilinx::air::attrs::LaunchWave,
                    mlir::IntegerAttr::get(i64, wave));
      if (auto w = dyn_cast<xilinx::airrt::WaitAllOp>(op))
        if (w->hasAttr("air.launch_end"))
          wave++;
    });
  });
}

// Compute and stamp the reset decision on every device. MUST run after
// moveFuncOpToEndOfDeviceOp (so the control funcs, hence the launch_end
// markers, are inside their devices) and before generateNpuWaitFromAIRRtWaitAll
// / unrollAffineFors (which erase the markers and strip the loops).
static void markDevicesNeedingLockReset(mlir::ModuleOp module) {
  module.walk([&](xilinx::AIE::DeviceOp device) {
    if (deviceHasRepeatCountDMAs(device) || deviceHasSingleTripCascade(device))
      device->setAttr(kNeedsLockResetAttr,
                      mlir::UnitAttr::get(device.getContext()));
    // Multi-iteration launch = a launch_end enclosed by a multi-trip loop, or
    // several launch_end markers (an already-unrolled iteration loop). Single
    // dispatch is exactly one launch_end not in a multi-trip loop (mirrors
    // deviceHasSingleTripCascade's launch-count logic).
    auto [numLaunchEnds, anyMultiTrip] = countLaunchEnds(device);
    if (numLaunchEnds >= 1 && (numLaunchEnds > 1 || anyMultiTrip)) {
      device->setAttr(kMultiIterLaunchAttr,
                      mlir::UnitAttr::get(device.getContext()));
      // air-opt-shim-dma-bds stamps one marker per air.launch, so several
      // markers means several launches flattened into this func. A launch
      // iteration loop is NOT that shape: it keeps its single marker inside
      // the loop, so the count stays 1 and the pacing stays per-loop-body,
      // which is already correct in that form.
      if (numLaunchEnds > 1)
        device->setAttr(kNumLaunchItersAttr,
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get(device.getContext(), 64),
                            numLaunchEnds));
    }
  });
}

namespace {

// Helper function to check if a value is a memref on host memory (space 0)
static bool isHostMemory(Value val) {
  if (auto memrefType = dyn_cast_if_present<BaseMemRefType>(val.getType()))
    return xilinx::air::isL3(memrefType);
  return false;
}

// Helper function to check if an op has memory effects on host memory
static bool hasMemoryEffectsOnHostMemory(Operation *op) {
  // Check if this op has memory effects interface
  auto effects = dyn_cast_if_present<MemoryEffectOpInterface>(op);
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
        if (auto castOp =
                dyn_cast_if_present<mlir::UnrealizedConversionCastOp>(u)) {
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

// Fold a constant-index scf.index_switch on the host (runtime-sequence) side.
// After the fused per-wave launch loop is fully unrolled (unrollSCFFors), the
// wave induction variable becomes a constant in each copy, so a launch-scope
// scf.index_switch that selects per-wave host feeds (each wave may run a
// different per-wave mode) has a constant condition. Inline the selected branch
// so its feed ops land
// directly in the runtime-sequence body -- an scf.index_switch cannot be a
// parent of aiex.dma_configure_task_for. On-core (CoreOp) index_switches carry
// a runtime RTP arm and are lowered elsewhere, so they are left untouched.
struct FoldConstIndexSwitchPattern
    : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern<scf::IndexSwitchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<AIE::CoreOp>())
      return failure();
    auto argConst = mlir::getConstantIntValue(op.getArg());
    if (!argConst)
      return failure();
    Region *reg = nullptr;
    ArrayRef<int64_t> cases = op.getCases();
    for (size_t i = 0; i < cases.size(); ++i) {
      if (cases[i] == *argConst) {
        reg = &op.getCaseRegions()[i];
        break;
      }
    }
    if (!reg)
      reg = &op.getDefaultRegion();
    Block &bb = reg->front();
    auto yield = cast<scf::YieldOp>(bb.getTerminator());
    SmallVector<Value> results(yield.getOperands());
    rewriter.eraseOp(yield);
    // If the switch carried an async drain token (a fused per-wave mode switch
    // wrapping host feeds is made async upstream so shim-dma-bds can build the
    // launch_end drain), sever the drain's dependency on it. At this stage feed
    // synchronization is handled by NpuDmaWaitOp on channels; the launch_end
    // wait_all only needs its marker attribute (already counted by
    // markDevicesNeedingLockReset). Rebuild each consuming airrt.wait_all
    // without the switch-token operand, so the token becomes dead and the
    // switch (and its inlined branch wait_all) can be erased cleanly -- a
    // surviving airrt.wait_all -> launch_end wait_all token chain otherwise
    // breaks the dialect conversion's erase ordering.
    if (op.getNumResults() > 0 &&
        isa<airrt::EventType>(op.getResult(op.getNumResults() - 1).getType())) {
      Value switchTok = op.getResult(op.getNumResults() - 1);
      SmallVector<Operation *> users(switchTok.getUsers());
      for (Operation *u : users) {
        auto wa = dyn_cast<airrt::WaitAllOp>(u);
        if (!wa)
          continue;
        SmallVector<Value> newOperands;
        for (Value v : wa->getOperands())
          if (v != switchTok)
            newOperands.push_back(v);
        rewriter.setInsertionPoint(wa);
        auto nwa = airrt::WaitAllOp::create(rewriter, wa.getLoc(),
                                            wa->getResultTypes(), newOperands);
        // Copy the full attribute set (matching AIRRtDialect's FoldWaitAll) so
        // the air.launch_end marker -- load-bearing for multi-iteration launch
        // detection and wave tagging -- is preserved on the rebuilt op.
        nwa->setAttrs(wa->getAttrs());
        rewriter.replaceOp(wa, nwa->getResults());
      }
    }
    rewriter.inlineBlockBefore(&bb, op);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct DmaToNpuPattern : public OpConversionPattern<airrt::DmaMemcpyNdOp> {
  using OpConversionPattern<airrt::DmaMemcpyNdOp>::OpConversionPattern;

  DmaToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<airrt::DmaMemcpyNdOp>(context, benefit) {}

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

    // Reduce the mixed static/dynamic access pattern to plain constants. A
    // still-dynamic entry indicates an unresolved loop induction variable
    // (e.g. from an unhandled scf.forall); fall back to a neutral value, and
    // for offsets warn so that such bugs are caught early instead of silently
    // producing wrong results.
    auto constify = [](ArrayRef<OpFoldResult> mixed, int64_t defaultValue,
                       llvm::function_ref<void(int)> onDynamic) {
      SmallVector<int64_t> values;
      for (auto [dim, ofr] : llvm::enumerate(mixed)) {
        if (auto constInt = getConstantIntValue(ofr)) {
          values.push_back(*constInt);
          continue;
        }
        if (onDynamic)
          onDynamic(dim);
        values.push_back(defaultValue);
      }
      return values;
    };
    auto *ctx = rewriter.getContext();

    // Runtime-valued access pattern. mlir-aie's shim-NOC BD lowering
    // (AIEDMATasksToNPU rewriteSingleBDDynamic) takes a runtime size as a
    // descriptor operand, so a loop-invariant dynamic dim does not have to be
    // folded away. The descriptor keeps its dimensions and the runtime value
    // rides in the mixed sizes list -- the shape upstream's own static-vs-
    // dynamic equivalence test uses, and the one that lands register-equivalent
    // to the static baseline. (Collapsing a contiguous nest to a runtime `len`
    // instead looks simpler but programs d1_size differently.)
    //
    // Sizes and offsets may be runtime -- a size for a context-length-sized
    // transfer, an offset for a slot that moves per dispatch (a KV append
    // writes position L-1). At most one size, though: constify() below would
    // otherwise silently substitute a default and emit a wrong-sized transfer
    // with no diagnostic.
    SmallVector<OpFoldResult> mixedLengths =
        getMixedValues(adaptor.getStaticLengths(), adaptor.getLengths(), ctx);
    SmallVector<OpFoldResult> mixedStrides =
        getMixedValues(adaptor.getStaticStrides(), adaptor.getStrides(), ctx);
    SmallVector<OpFoldResult> mixedOffsets =
        getMixedValues(adaptor.getStaticOffsets(), adaptor.getOffsets(), ctx);
    // A runtime size becomes the BD's `len`, which is only meaningful if the
    // transfer is one linear run: contiguous row-major below the runtime
    // dimension and inert above it. Then no descriptor dimension is needed and
    // the length is that size scaled by the extent below it.
    Value dynLen;       // runtime transfer length, in elements (i64)
    Value dynOuterSize; // runtime repeat count (zero outer stride)
    Value dynOffset;    // runtime start offset, in elements (i64)
    {
      for (auto ofr : mixedStrides)
        if (!getConstantIntValue(ofr))
          return op->emitOpError("runtime-valued DMA stride is not supported");
      SmallVector<unsigned> dynDims;
      for (auto [dim, ofr] : llvm::enumerate(mixedLengths))
        if (!getConstantIntValue(ofr))
          dynDims.push_back(dim);
      if (dynDims.size() > 1)
        return op->emitOpError(
            "more than one runtime-valued DMA size is not supported");
      if (!dynDims.empty()) {
        unsigned dim = dynDims.front();
        Value dynVal = cast<Value>(mixedLengths[dim]);
        int64_t below = 1;
        bool contiguous = true;
        for (int i = mixedLengths.size() - 1; i > (int)dim; i--) {
          auto sz = *getConstantIntValue(mixedLengths[i]);
          auto st = *getConstantIntValue(mixedStrides[i]);
          if (st != below && sz != 1)
            contiguous = false;
          below *= sz;
        }
        // A dimension ABOVE the runtime one breaks the single linear run the
        // `len` operand encodes -- unless its stride is zero. A zero-stride
        // dimension contributes nothing to addressing: it re-runs the same
        // descriptor, and repeat_count already carries it (dim 0 below, dims
        // 1-2 in the dimLayouts loop, which folds them the same way). The
        // dynamic-length BD is emitted with no dimensions at all, so leaving
        // such a dimension out of the linear run is not an approximation.
        //
        // This is what a batched decode's KV readback looks like: lengths
        // [B, ceil(L/16), 16, 512] with strides [0, 8192, 512, 1], one runtime
        // block count and a zero outer stride because every token of the block
        // re-reads the same context. Requiring length 1 above the runtime
        // dimension rejected it, which is why DECODE_DYNSEQ built only at
        // batch 1.
        for (unsigned i = 0; i < dim; i++)
          if (*getConstantIntValue(mixedLengths[i]) != 1 &&
              *getConstantIntValue(mixedStrides[i]) != 0)
            contiguous = false;
        int64_t thisStride = *getConstantIntValue(mixedStrides[dim]);
        if (contiguous && thisStride == below) {
          dynLen = below == 1 ? dynVal
                              : arith::MulIOp::create(
                                    rewriter, op.getLoc(), dynVal,
                                    arith::ConstantOp::create(
                                        rewriter, op.getLoc(),
                                        rewriter.getI64IntegerAttr(below)))
                                    .getResult();
        } else if (dim == 0 && thisStride == 0) {
          // A pure repeat: the outermost dim re-runs the same descriptor, which
          // the task's repeat_count takes as an operand.
          dynOuterSize = dynVal;
        } else {
          return op->emitOpError("runtime-valued DMA size in dimension ")
                 << (3 - dim)
                 << " requires a contiguous transfer or a zero outer stride";
        }
      }
      // A runtime offset rides in the BD's `offset` operand. Only one dimension
      // may carry it, and the dimensions it is scaled against must be constant,
      // so fold the scaling here.
      for (auto [dim, ofr] : llvm::enumerate(mixedOffsets)) {
        if (getConstantIntValue(ofr))
          continue;
        if (dynOffset)
          return op->emitOpError(
              "more than one runtime-valued DMA offset is not supported");
        int64_t stride = *getConstantIntValue(mixedStrides[dim]);
        Value v = cast<Value>(ofr);
        dynOffset =
            stride == 1
                ? v
                : arith::MulIOp::create(rewriter, op.getLoc(), v,
                                        arith::ConstantOp::create(
                                            rewriter, op.getLoc(),
                                            rewriter.getI64IntegerAttr(stride)))
                      .getResult();
      }
    }

    // Entry i of each list is dimension 3-i, outermost first.
    SmallVector<int64_t> staticOffsets = constify(
        getMixedValues(adaptor.getStaticOffsets(), adaptor.getOffsets(), ctx),
        /*defaultValue=*/0, [&](int dim) {
          op->emitWarning("non-constant DMA offset (dim ")
              << (3 - dim) << ") defaulting to 0";
        });
    SmallVector<int64_t> staticSizes = constify(
        getMixedValues(adaptor.getStaticLengths(), adaptor.getLengths(), ctx),
        /*defaultValue=*/1, nullptr);
    // The innermost transfer length is at least one element.
    staticSizes[3] = std::max((int64_t)1, staticSizes[3]);
    SmallVector<int64_t> staticStrides = constify(
        getMixedValues(adaptor.getStaticStrides(), adaptor.getStrides(), ctx),
        /*defaultValue=*/0, nullptr);

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
    // A pure-repeat runtime outer size drives the task's repeat_count operand;
    // the descriptor itself stays constant. mlir-aie wants (n - 1).
    Value repeatCountVal;
    if (dynOuterSize) {
      auto i32 = rewriter.getI32Type();
      Value n =
          arith::TruncIOp::create(rewriter, op.getLoc(), i32, dynOuterSize);
      repeatCountVal = arith::SubIOp::create(
          rewriter, op.getLoc(), n,
          arith::ConstantOp::create(rewriter, op.getLoc(),
                                    rewriter.getI32IntegerAttr(1)));
    }

    // The 4th dimension is included in dma_bd dimensions if stride[0] != 0
    // (the iteration_stride tells the hardware how to advance offset each
    // repeat)
    bool use4thDimInBd = (staticStrides[0] != 0);

    // Build BDDimLayoutArrayAttr for the data layout transformation
    SmallVector<AIE::BDDimLayoutAttr> dimLayouts;

    // Determine starting index for dims based on whether we use 4th dim
    int startDim = use4thDimInBd ? 0 : 1;

    // Build dimension layouts from sizes and strides
    for (int i = startDim; i < 4; i++) {
      int64_t size = staticSizes[i];
      int64_t stride = staticStrides[i];
      // stride=0 with size>1 at dims 1-2 means "repeat at same address"
      // (broadcast pattern). Fold into repeat_count instead of passing to
      // BD dimensions, since aie.dma_bd rejects stride=0. Dim 3 (innermost)
      // is excluded because stride=0 there is the trivial/degenerate case.
      if (i > 0 && i < 3 && stride == 0 && size > 1) {
        repeatCount = (repeatCount + 1) * size - 1;
        // Adjust transfer length to exclude the folded dimension.
        transferLen /= size;
        continue;
      }
      // Include dimension if size > 1, if it's the innermost dimension,
      // or if the 4th dim is in use and this is a middle dim (retain size-1
      // dims to preserve the 4-entry layout needed for iteration_stride).
      if (size > 1 || i == 3 || (use4thDimInBd && i > 0)) {
        auto dimLayout = AIE::BDDimLayoutAttr::get(ctx, size, stride);
        dimLayouts.push_back(dimLayout);
      }
    }

    AIE::BDDimLayoutArrayAttr dimsAttr =
        AIE::BDDimLayoutArrayAttr::get(ctx, dimLayouts);

    // Determine if this is an output (S2MM) channel.
    // S2MM channels issue tokens by default, MM2S channels do not.
    // Feeds marked `air.preserve_shim_dma_order` (lockstep-coupled shim feeds
    // that opted out of per-channel BD folding) also issue a token so they can
    // be awaited for bounded double-buffered backpressure (see
    // synthesizeDoubleBufferedAwaits below).
    bool paced = op->hasAttr(air::attrs::PreserveShimDmaOrder);
    bool issueToken = air::isDeviceToHostShimDMA(op) || paced;

    // Narrow the runtime length/offset here, not inside the descriptor block:
    // that block admits only the BD ops themselves. Place each narrowing at its
    // operand's definition rather than at the rewriter's cursor -- a converted
    // operand can be materialized further down the block, and a use built at
    // the cursor would then not be dominated by it.
    auto narrowToI32 = [&](Value v) -> Value {
      if (!v)
        return Value();
      OpBuilder::InsertionGuard g(rewriter);
      if (Operation *def = v.getDefiningOp())
        rewriter.setInsertionPointAfter(def);
      else
        rewriter.setInsertionPointToStart(cast<BlockArgument>(v).getOwner());
      return arith::TruncIOp::create(rewriter, op.getLoc(),
                                     rewriter.getI32Type(), v);
    };
    Value dynLenI32 = narrowToI32(dynLen);
    Value dynOffsetI32 = narrowToI32(dynOffset);

    // Create DMAConfigureTaskForOp with proper repeat_count from highest
    // dimension
    auto configTaskOp = AIEX::DMAConfigureTaskForOp::create(
        rewriter, op.getLoc(),
        rewriter.getIndexType(),          // result type
        metadata,                         // alloc symbol reference
        rewriter.getBoolAttr(issueToken), // issue_token = true for S2MM / paced
        rewriter.getI32IntegerAttr(
            repeatCount), // repeat_count from highest dim
        repeatCountVal    // runtime repeat count, when the outer size is one
    );
    if (paced)
      configTaskOp->setAttr(air::attrs::PreserveShimDmaOrder,
                            rewriter.getUnitAttr());
    // Carry the runtime-sequence hoist marker onto the task so the
    // post-lowering reordering step can move this input feed ahead of the
    // weight stream.
    if (op->hasAttr(air::attrs::RuntimeHoist))
      configTaskOp->setAttr(air::attrs::RuntimeHoist, rewriter.getUnitAttr());
    if (op->hasAttr(air::attrs::AwaitAppends))
      configTaskOp->setAttr(air::attrs::AwaitAppends, rewriter.getUnitAttr());
    // Carry the append-barrier marker so the append->readback ordering step can
    // find this append's completion await and move it before the tagged
    // readback (see the air.await_appends barrier below).
    if (op->hasAttr(air::attrs::AppendBarrier))
      configTaskOp->setAttr(air::attrs::AppendBarrier, rewriter.getUnitAttr());
    // Carry the coalesced-feed marker so the double-buffered await synthesis
    // paces this merged channel at depth 1 (no cross-run overlap).
    if (op->hasAttr(air::attrs::CoalescedShimFeed))
      configTaskOp->setAttr(air::attrs::CoalescedShimFeed,
                            rewriter.getUnitAttr());
    // Carry the fused-launch wave index so the per-wave
    // RTP/set_lock/output-S2MM hoist can group this feed by its launch
    // iteration.
    if (auto wave = op->getAttr(air::attrs::LaunchWave))
      configTaskOp->setAttr(air::attrs::LaunchWave, wave);

    // Create the body region of the configure task op
    Block *bodyBlock = rewriter.createBlock(&configTaskOp.getBody());
    rewriter.setInsertionPointToStart(bodyBlock);

    // Check for packet attribute on the source DMA op. This is needed for
    // direct L3→L1 packet-switched flows where the shim DMA BD must include
    // the packet header for correct routing.
    auto pktAttr = op->getAttrOfType<AIE::PacketInfoAttr>("packet");

    // Runtime length or offset: the descriptor carries no dimensions -- it is
    // one linear run -- and takes `len` / `offset` as operands, the form
    // mlir-aie's shim-NOC lowering encodes into a runtime BD. Staying on the
    // DMA-task path matters: npu.dma_memcpy_nd's dynamic lowering hardwires BD
    // id 0, so a dynamic transfer sharing a shim tile with a task-path one
    // would silently overwrite its descriptor.
    if (dynLenI32) {
      // A runtime length only arises for a transfer this pass has already
      // checked is one linear run, so the descriptor needs no dimensions --
      // the length carries the whole extent.
      AIE::DMABDOp::create(
          rewriter, op.getLoc(), memref, dynOffsetI32, dynLenI32,
          /*static_offset=*/
          dynOffsetI32 ? nullptr : rewriter.getI32IntegerAttr(totalOffset),
          /*static_len=*/nullptr,
          /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
          /*static_sizes=*/nullptr, /*static_strides=*/nullptr,
          /*pad_dimensions=*/nullptr, /*bd_id=*/nullptr, pktAttr,
          /*out_of_order_id=*/nullptr,
          /*burst_length=*/nullptr, /*axcache=*/nullptr,
          /*iteration=*/nullptr,
          /*offset_parameter=*/nullptr,
          /*offset_state_table_idx=*/nullptr, /*next_bd_id=*/nullptr);
    } else if (dynOffsetI32) {
      // Only the address moves. Build the descriptor exactly as the static path
      // would -- a KV append writes NGRP chunks at a region stride, and
      // collapsing that to one linear run would scatter every group but the
      // first -- then swap the constant offset for the runtime one.
      AIE::DMABDOp bd =
          dimLayouts.empty()
              ? (pktAttr
                     ? AIE::DMABDOp::create(rewriter, op.getLoc(), memref, 0,
                                            static_cast<int>(transferLen),
                                            pktAttr)
                     : AIE::DMABDOp::create(rewriter, op.getLoc(), memref, 0,
                                            static_cast<int>(transferLen)))
              : (pktAttr
                     ? AIE::DMABDOp::create(rewriter, op.getLoc(), memref, 0,
                                            static_cast<int>(transferLen),
                                            dimsAttr, pktAttr)
                     : AIE::DMABDOp::create(rewriter, op.getLoc(), memref, 0,
                                            static_cast<int>(transferLen),
                                            dimsAttr));
      bd.getOffsetMutable().assign(dynOffsetI32);
      bd.removeStaticOffsetAttr();
    } else if (dimLayouts.empty() && !pktAttr) {
      AIE::DMABDOp::create(rewriter, op.getLoc(), memref,
                           static_cast<int>(totalOffset),
                           static_cast<int>(transferLen));
    } else if (dimLayouts.empty() && pktAttr) {
      AIE::DMABDOp::create(rewriter, op.getLoc(), memref,
                           static_cast<int>(totalOffset),
                           static_cast<int>(transferLen), pktAttr);
    } else if (!dimLayouts.empty() && !pktAttr) {
      AIE::DMABDOp::create(rewriter, op.getLoc(), memref,
                           static_cast<int>(totalOffset),
                           static_cast<int>(transferLen), dimsAttr);
    } else {
      AIE::DMABDOp::create(rewriter, op.getLoc(), memref,
                           static_cast<int>(totalOffset),
                           static_cast<int>(transferLen), dimsAttr, pktAttr);
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

    // Fused-launch wave index (if any): propagated onto the emitted RTP writes
    // and set_locks so the per-wave hoist groups this arm by its iteration.
    auto waveAttr = op->getAttr(air::attrs::LaunchWave);

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
          unsigned rtp_slot = 0;
          for (int i = 0, e = op.getNumOperands(); i < e; i++) {
            Value oper = adaptor.getOperands()[i];
            if (!llvm::isa<IntegerType, IndexType, FloatType>(oper.getType()))
              continue;

            // The core loads this slot unconditionally, so a slot left
            // unwritten is a read of stale memory -- not a value that merely
            // defaults. A runtime operand therefore has to be written too: it
            // is how a core's trip count can follow a dispatch-time context
            // length, and how that count stays equal to the shim's push count.
            Value vVal;
            if (isa<IntegerType, IndexType>(oper.getType())) {
              auto i32Ty = rewriter.getI32Type();
              if (auto constOp = dyn_cast_if_present<arith::ConstantOp>(
                      oper.getDefiningOp())) {
                vVal = arith::ConstantOp::create(
                    rewriter, op.getLoc(), i32Ty,
                    rewriter.getI32IntegerAttr(
                        cast<IntegerAttr>(constOp.getValue()).getInt()));
              } else if (isa<IndexType>(oper.getType())) {
                vVal = arith::IndexCastOp::create(rewriter, op.getLoc(), i32Ty,
                                                  oper);
              } else if (oper.getType() == i32Ty) {
                vVal = oper;
              } else if (oper.getType().getIntOrFloatBitWidth() > 32) {
                vVal =
                    arith::TruncIOp::create(rewriter, op.getLoc(), i32Ty, oper);
              } else {
                vVal =
                    arith::ExtUIOp::create(rewriter, op.getLoc(), i32Ty, oper);
              }
            }
            if (vVal) {
              auto rtpOp = AIEX::NpuWriteRTPOp::create(rewriter, op.getLoc(),
                                                       name, rtp_slot, vVal);
              if (waveAttr)
                rtpOp->setAttr(air::attrs::LaunchWave, waveAttr);
            }
            rtp_slot++;
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

        auto setLockOp =
            AIEX::SetLockOp::create(rewriter, op.getLoc(), lockOp.getResult(),
                                    rewriter.getI32IntegerAttr(1));
        if (waveAttr)
          setLockOp->setAttr(air::attrs::LaunchWave, waveAttr);
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
    if (!xilinx::air::isL1(memrefTy))
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
    if (!xilinx::air::isL1(memrefTy))
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
// aiex.npu.load_pdi to reset the DMA engine / cascade state if:
// 1. output-elf mode is enabled, AND
// 2. deviceNeedsLockReset(device) -- i.e. the device has core/memtile DMAs
//    with repeat_count > 0, OR is a single-trip cascade launch.
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
          // A fused multi-iteration launch (scf.for over air.launch stitching N
          // iterations into one dispatch) needs the between-iteration shim
          // drain on the xclbin (non-elf) path too: otherwise consecutive
          // iterations overlap and a finite-depth device lock deadlocks after a
          // few iterations.
          bool multiIter = deviceHasMultiIterLaunch(device);
          if (outputElf && deviceNeedsLockReset(device)) {
            // Insert aiex.npu.load_pdi to reset DMA engine / cascade state
            // (repeat_count DMAs, or a single-trip cascade launch).
            rewriter.setInsertionPoint(op);
            auto deviceRef = FlatSymbolRefAttr::get(rewriter.getContext(),
                                                    device.getSymName());
            AIEX::NpuLoadPdiOp::create(rewriter, op.getLoc(), deviceRef,
                                       IntegerAttr(), IntegerAttr(),
                                       IntegerAttr(), AIEX::ExpandModeAttr());
          } else if (outputElf || multiIter) {
            // No PDI reload needed (no repeat_count DMAs), but still need
            // between-iteration synchronization to prevent the next
            // iteration's shim DMA configuration from racing with the
            // current iteration's compute (issue #1373; extended to the
            // multi-iteration xclbin path).
            rewriter.setInsertionPoint(op);
            for (auto alloc : device.getOps<AIE::ShimDMAAllocationOp>())
              AIEX::NpuDmaWaitOp::create(rewriter, op.getLoc(),
                                         alloc.getSymName());
          }
        }
      }
    }

    // Erase the op - synchronization is handled by NpuDmaWaitOp/load_pdi
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

    auto seq = AIE::RuntimeSequenceOp::create(
        rewriter, op->getLoc(), op.getSymNameAttr(),
        /*emit_parameter_sync_preamble=*/nullptr);
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

    bool isLaunchEnd = op->hasAttr("air.launch_end");
    bool multiIter = false;
    if (isLaunchEnd) {
      if (auto device = op->getParentOfType<AIE::DeviceOp>())
        if (llvm::isa<AIE::BaseNPU2TargetModel>(device.getTargetModel()))
          multiIter = deviceHasMultiIterLaunch(device);
    }

    llvm::SmallDenseSet<StringRef> waitedChannels;
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
      waitedChannels.insert(metadata);
    }

    // Check if this is a launch_end wait_all and needs between-iteration sync
    if (op->hasAttr("air.launch_end")) {
      auto device = op->getParentOfType<AIE::DeviceOp>();
      if (device) {
        // Only apply for NPU2 family devices
        const AIE::AIETargetModel &tm = device.getTargetModel();
        if (llvm::isa<AIE::BaseNPU2TargetModel>(tm)) {
          // A fused multi-iteration launch (scf.for over air.launch, e.g. N
          // stitched iterations) must fence every iteration boundary even on
          // the xclbin (non-elf) path: without it consecutive iterations
          // overlap and a finite-depth device lock deadlocks after a few
          // iterations.
          if (outputElf && deviceNeedsLockReset(device)) {
            auto deviceRef = FlatSymbolRefAttr::get(rewriter.getContext(),
                                                    device.getSymName());
            AIEX::NpuLoadPdiOp::create(rewriter, op.getLoc(), deviceRef,
                                       IntegerAttr(), IntegerAttr(),
                                       IntegerAttr(), AIEX::ExpandModeAttr());
          } else if (outputElf || multiIter) {
            // No PDI reload needed, but emit NpuDmaWaitOp for any shim
            // channels not already waited on to synchronize before the
            // next iteration (issue #1373; extended to the multi-iteration
            // xclbin path so each launch iteration fully drains).
            for (auto alloc : device.getOps<AIE::ShimDMAAllocationOp>())
              if (!waitedChannels.contains(alloc.getSymName()))
                AIEX::NpuDmaWaitOp::create(rewriter, op.getLoc(),
                                           alloc.getSymName());
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
// or airrt.herd_load operations that link to device ops.
SmallVector<LaunchRegion> identifyLaunchRegions(func::FuncOp funcOp,
                                                ModuleOp module) {
  SmallVector<LaunchRegion> regions;

  funcOp.walk([&](affine::AffineForOp forOp) {
    // Check if this is a launch boundary
    if (!isLaunchBoundaryLoop(forOp))
      return;

    // Look for airrt.segment_load or airrt.herd_load inside this loop
    forOp.walk([&](Operation *op) {
      StringRef deviceName;
      if (auto segLoad = dyn_cast<airrt::SegmentLoadOp>(op))
        deviceName = segLoad.getSymName();
      else if (auto herdLoad = dyn_cast<airrt::HerdLoadOp>(op))
        deviceName = herdLoad.getSymName();
      else
        return;
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
      builder, loc, builder.getStringAttr(mainSeqName.str()),
      /*emit_parameter_sync_preamble=*/nullptr);
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
        builder, loc, FlatSymbolRefAttr::get(builder.getContext(), deviceName),
        AIEX::ExpandModeAttr());
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
  // Linear shim BDs (contiguous row-major + optional outer dummies/repeat)
  // use the wide buffer_length register and bypass the per-dim 10-bit limit.
  SmallVector<OpFoldResult> wrap_list = dma.getMixedLengths();
  SmallVector<OpFoldResult> stride_list = dma.getMixedStrides();
  if (air::isContiguousRowMajorOrRepeated(wrap_list, stride_list))
    return false;
  for (unsigned i = 0; i < wrap_list.size(); i++) {
    if (auto const_val = getConstantIntValue(wrap_list[i])) {
      // Detected wrap that goes beyond the AIE2 hardware limit.
      if (*const_val >= AIE2_WRAP_UPPER_BOUNDS[i])
        return true;
    }
  }
  return false;
}

LogicalResult tileIllegalWrapDim(airrt::DmaMemcpyNdOp memcpy_op) {
  auto loc = memcpy_op->getLoc();
  auto ctx = memcpy_op->getContext();
  SmallVector<OpFoldResult> offsets = memcpy_op.getMixedOffsets();
  SmallVector<OpFoldResult> wraps = memcpy_op.getMixedLengths();
  SmallVector<OpFoldResult> strides = memcpy_op.getMixedStrides();
  OpBuilder builder(memcpy_op);

  auto memrefTy =
      llvm::dyn_cast<BaseMemRefType>(memcpy_op.getMemref().getType());
  int innerAlignment =
      memrefTy ? air::getDmaInnerElementAlignment(memrefTy, memcpy_op) : 1;

  for (int i = wraps.size() - 1; i >= 0; i--) {
    auto const_wrap = *getConstantIntValue(wraps[i]);
    auto const_stride = *getConstantIntValue(strides[i]);
    if (const_wrap >= AIE2_WRAP_UPPER_BOUNDS[i]) {
      // Found dimension with illegal wrap. Prefers smaller outer wrap as
      // long as stride fits. For stride==1, force the inner wrap to a
      // multiple of innerAlignment elements so its byte size stays aligned
      // to the shim address granularity (otherwise aie.dma_bd rejects it).
      int alignment = (const_stride == 1) ? innerAlignment : 1;
      int a_wrap = air::findLargestAlignedFactor(
          const_wrap, AIE2_WRAP_UPPER_BOUNDS[i] - 1, alignment);
      if (a_wrap == 0) {
        return memcpy_op.emitOpError()
               << "cannot tile dim " << i << " of size " << const_wrap
               << " into shim-legal chunks: no factor in [" << alignment << ", "
               << (AIE2_WRAP_UPPER_BOUNDS[i] - 1) << "] is a multiple of "
               << alignment
               << " elements. Reshape the transfer or pad the inner dimension.";
      }
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
      wraps[i] = builder.getI64IntegerAttr(inner_wrap);
      wraps.insert(wraps.begin() + i, builder.getI64IntegerAttr(outer_wrap));
      auto new_const_stride = const_stride * inner_wrap;
      if (volume != 1)
        new_const_stride %=
            volume; // Avoids striding out of memory size, if memref is ranked
      strides.insert(strides.begin() + i,
                     builder.getI64IntegerAttr(new_const_stride));
      offsets.insert(offsets.begin() + i, builder.getI64IntegerAttr(0));
      // Attempt to find one dummy dimension in the wrap-and-stride list and
      // erase.
      auto offsetWrapZip = llvm::zip_equal(offsets, wraps);
      auto it = llvm::find_if(
          offsetWrapZip, [](std::tuple<OpFoldResult, OpFoldResult> entry) {
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

  // Keep all strides including the innermost (stride0).

  // Create new airrt.dma_memcpy_nd op.
  if (inner_affine_for_iv) {
    // Innermost tiled affine.for loop induction variable as lowest offset, if
    // original rank exceeds hw limit.
    offsets.back() =
        arith::IndexCastOp::create(builder, loc, IntegerType::get(ctx, 64),
                                   inner_affine_for_iv)
            .getResult();
  }
  auto newOp = airrt::DmaMemcpyNdOp::create(
      builder, loc, SmallVector<Type>{}, memcpy_op.getId(), memcpy_op.getX(),
      memcpy_op.getY(), memcpy_op.getMemref(), offsets, wraps, strides);
  // Only discardable attrs carry over; the static_* arrays are inherent and
  // already set by the builder above.
  newOp->setAttrs(memcpy_op->getDiscardableAttrDictionary());

  // Unroll the affine loop nest.
  for (auto forOp : llvm::reverse(for_loop_nest)) {
    (void)loopUnrollFull(forOp);
  }

  memcpy_op.erase();
  return success();
}

// Coalesce consecutive contiguous shim DMA transfers on the same channel
// (marked air.preserve_shim_dma_order) into one wide contiguous transfer.
//
// A large host feed can be lowered as many small per-block shim transfers, each
// of which the shim sequencer executes as a separate configure/start/await
// triplet. That per-triplet microcontroller overhead can dominate the dataflow
// floor. Merging a run of same-channel BDs whose source offsets are contiguous
// (offset[k+1] == offset[k] + len[k]) into a single BD collapses the triplet
// count without touching the device configuration: the receiving memtile ring
// drains the wider stream via backpressure exactly as it drained the fragments.
//
// Only pure contiguous 1D runs are merged, producing a linear row-major BD.
// Such a BD uses the wide buffer_length register and bypasses the per-dim wrap
// limit (see violatesAIE2WrapLimit / isContiguousRowMajorOrRepeated), so the
// coalesced transfer remains a single DMA task. Gated to
// air.preserve_shim_dma_order feeds -- the lockstep-coupled shim feeds that
// already opt out of the per-channel BD fold -- because the merge reorders the
// channel's transfers ahead of the sibling-channel round-major interleave; that
// reorder is numerically equivalent to the fragmented feed (verified
// output-identical on an exercising design).
// Peel an offset into (base, constant addend): a rolled body's offsets are
// `addi(loop-derived base, const)`, so two feeds are contiguous when they
// share a base and their addends are. A constant offset has a null base,
// which is the straight-line case.
static std::pair<Value, int64_t> peelOffset(Value v) {
  int64_t addend = 0;
  while (v) {
    if (auto c = getConstantIntValue(v))
      return {nullptr, addend + *c};
    Operation *def = v.getDefiningOp();
    if (auto cast = dyn_cast_if_present<arith::IndexCastOp>(def)) {
      v = cast.getIn();
      continue;
    }
    if (auto add = dyn_cast_if_present<arith::AddIOp>(def)) {
      if (auto c = getConstantIntValue(add.getRhs())) {
        addend += *c;
        v = add.getLhs();
        continue;
      }
      if (auto c = getConstantIntValue(add.getLhs())) {
        addend += *c;
        v = add.getRhs();
        continue;
      }
    }
    break;
  }
  return {v, addend};
}

static void coalesceShimDmaOrder(ModuleOp module) {
  SmallVector<func::FuncOp> funcOps;
  module.walk([&](func::FuncOp f) { funcOps.push_back(f); });

  // Return (totalOffset, len) for a pure contiguous 1D transfer, or nullopt if
  // the op is not a mergeable contiguous 1D transfer (non-constant descriptor,
  // multi-dim wrap, or non-unit inner stride).
  struct Desc {
    Value base;
    int64_t offset;
    int64_t len;
  };
  auto describe = [](airrt::DmaMemcpyNdOp d) -> std::optional<Desc> {
    // Never merge packet-switched transfers: consecutive BDs on one channel may
    // carry different packet ids (different routing destinations). Merging them
    // into one BD would send all data with the first id, starving the other
    // destinations and deadlocking. Only circuit-switched feeds (no packet) are
    // safe to coalesce.
    if (d->hasAttr("packet"))
      return std::nullopt;
    auto lengths = getConstantIntValues(d.getMixedLengths());
    auto strides = getConstantIntValues(d.getMixedStrides());
    if (!lengths || !strides)
      return std::nullopt;
    // Pure contiguous 1D: only the innermost dim carries data, unit inner
    // stride (the higher dims are size-1 dummies).
    if ((*lengths)[0] != 1 || (*lengths)[1] != 1 || (*lengths)[2] != 1 ||
        (*strides)[3] != 1)
      return std::nullopt;
    Value base = nullptr;
    int64_t totalOffset = 0;
    for (auto [off, str] : llvm::zip_equal(d.getMixedOffsets(), *strides)) {
      if (auto c = getConstantIntValue(off)) {
        totalOffset += *c * str;
        continue;
      }
      // One dynamic offset, on the unit-stride innermost dim.
      if (base || str != 1)
        return std::nullopt;
      auto [b, addend] = peelOffset(cast<Value>(off));
      if (!b)
        return std::nullopt;
      base = b;
      totalOffset += addend;
    }
    return Desc{base, totalOffset, (*lengths)[3]};
  };

  for (auto f : funcOps) {
    // Collect paced shim feeds (program order preserved by the walk).
    SmallVector<airrt::DmaMemcpyNdOp> paced;
    f.walk([&](airrt::DmaMemcpyNdOp dma) {
      if (dma->hasAttr(air::attrs::PreserveShimDmaOrder))
        paced.push_back(dma);
    });
    if (paced.size() < 2)
      continue;

    // Group by (launch wave, channel metadata symbol, source memref). The wave
    // key prevents merging a channel's feed across launch iterations: in a
    // fused multi-iteration launch each wave loads a distinct slice and has its
    // own per-wave arming that must sit between waves, so wave N+1's feed must
    // not be delivered inside wave N even if their source offsets happen to be
    // contiguous.
    struct Entry {
      airrt::DmaMemcpyNdOp op;
      int64_t offset;
      int64_t len;
    };
    // The base joins the key: feeds off different bases are never contiguous,
    // and for a rolled body the base is what distinguishes one wave's slice.
    llvm::MapVector<std::tuple<int64_t, StringRef, void *, void *>,
                    SmallVector<Entry>>
        groups;
    for (auto d : paced) {
      auto desc = describe(d);
      if (!desc)
        continue;
      auto md = d->getAttrOfType<FlatSymbolRefAttr>("metadata");
      if (!md)
        continue;
      int64_t wave = -1;
      if (auto w = d->getAttrOfType<IntegerAttr>(air::attrs::LaunchWave))
        wave = w.getInt();
      groups[{wave, md.getValue(), d.getMemref().getAsOpaquePointer(),
              desc->base.getAsOpaquePointer()}]
          .push_back({d, desc->offset, desc->len});
    }

    for (auto &kv : groups) {
      auto &entries = kv.second;
      if (entries.size() < 2)
        continue;
      // Entries are in program (walk) order. Merge only ops that are BOTH
      // consecutive in program order AND contiguous in source offset
      // (entries[k+1].offset == entries[k].offset + entries[k].len). This
      // preserves the channel's original delivery order: we never reorder a
      // later-program-order op ahead of an earlier one, so a channel whose
      // program order is not monotonically increasing by offset is coalesced
      // only where the program already delivers a contiguous run.
      size_t i = 0;
      while (i < entries.size()) {
        size_t j = i;
        int64_t end = entries[i].offset + entries[i].len;
        while (j + 1 < entries.size() && entries[j + 1].offset == end) {
          j++;
          end = entries[j].offset + entries[j].len;
        }
        if (j > i) {
          // Merge run [i..j]: enlarge the first op's innermost length to cover
          // the whole run, then erase the rest (redirecting their event tokens
          // to the merged op so any WaitAll dependencies stay valid).
          auto first = entries[i].op;
          // Only merge if the merged op can carry the redirected tokens.
          bool tokensOk = true;
          for (size_t k = i + 1; k <= j && tokensOk; k++)
            if (entries[k].op.getEvent() && !first.getEvent())
              tokensOk = false;
          if (tokensOk) {
            int64_t total = end - entries[i].offset;
            OpBuilder b(first);
            auto lengths = first.getMixedLengths();
            lengths.back() = b.getI64IntegerAttr(total);
            first.setMixedLengths(lengths);
            // Mark the merged feed so the double-buffered await synthesis paces
            // it with the cross-channel phase barrier (no cross-run overlap).
            first->setAttr(air::attrs::CoalescedShimFeed, b.getUnitAttr());
            for (size_t k = i + 1; k <= j; k++) {
              auto d = entries[k].op;
              if (d.getEvent() && first.getEvent())
                d.getEvent().replaceAllUsesWith(first.getEvent());
              d.erase();
            }
          }
        }
        i = j + 1;
      }
    }
  }
}

// AIE2 shim BDs encode each dimension's stride in a 20-bit field, so a stride
// above AIE2_STRIDE_UPPER_BOUND cannot be expressed at all. Unlike an illegal
// wrap, an illegal stride cannot be tiled away: splitting the dim leaves the
// stride unchanged.
//
// It CAN be turned into base offsets. A dim (wrap W, stride S) contributes the
// address set {(off + k) * S : k in [0, W)}, so the transfer is equivalent to W
// separate BDs that drop the dim and fold its contribution into the base:
//   offset_k = offset + (off + k) * S
// Same bytes, same addresses, same order, same channel -- only the descriptor
// count changes, by a factor of W.
//
// That is only worth doing when W is small, which is the case this exists for:
// a KV cache laid out region-major strides its per-group dim by
// ATTN_MAXL*REGION_W, which crosses the 20-bit field once the context is long
// enough, while the dim itself stays tiny (one entry per KV group). A large W
// is left alone rather than silently exploding the runtime sequence.
const int AIE2_STRIDE_UNROLL_LIMIT = 16;

// Index of the outermost dim whose stride exceeds the hardware field, or
// std::nullopt. Dims of wrap 1 are ignored: they contribute a single address
// and the stride is dead, so the BD builder drops them.
static std::optional<unsigned>
findIllegalStrideDim(SmallVector<OpFoldResult> wraps,
                     SmallVector<OpFoldResult> strides) {
  for (unsigned i = 0; i < strides.size(); i++) {
    auto s = getConstantIntValue(strides[i]);
    auto w = getConstantIntValue(wraps[i]);
    if (s && w && *s > AIE2_STRIDE_UPPER_BOUND && *w > 1)
      return i;
  }
  return std::nullopt;
}

bool violatesAIE2StrideLimit(airrt::DmaMemcpyNdOp dma) {
  return findIllegalStrideDim(dma.getMixedLengths(), dma.getMixedStrides())
      .has_value();
}

// Replace one over-strided dim with `wrap` copies of the transfer, each
// carrying that dim's address contribution in its own base offset.
//
// The rewrite is in place, not a rank reduction: DmaToNpuPattern indexes the
// wrap-and-stride lists as a fixed rank-4 layout. Instead the dim is set to
// wrap 1 with offset (off + k), which
//   - keeps the rank, and
//   - keeps the address, because that pattern computes the BD base as
//     sum(offsets[i] * strides[i]) -- so (off + k) * S lands in the base
//     exactly as the strided walk would have reached it, and
//   - drops the dim from the BD, because a middle dim of size 1 fails the
//     `size > 1 || i == 3 || (use4thDimInBd && i > 0)` test that decides which
//     dims become BDDimLayout entries. The over-limit stride is therefore never
//     encoded.
// The last point is what makes this legal, so it is checked rather than
// assumed: only a middle dim of a BD that does not use the 4th dim qualifies.
LogicalResult unrollIllegalStrideDim(airrt::DmaMemcpyNdOp memcpy_op,
                                     bool &folded) {
  auto loc = memcpy_op->getLoc();
  OpBuilder builder(memcpy_op);
  SmallVector<OpFoldResult> offsets = memcpy_op.getMixedOffsets();
  SmallVector<OpFoldResult> wraps = memcpy_op.getMixedLengths();
  SmallVector<OpFoldResult> strides = memcpy_op.getMixedStrides();

  folded = false;
  auto dim = findIllegalStrideDim(wraps, strides);
  if (!dim)
    return success();
  unsigned i = *dim;

  auto constOff = getConstantIntValue(offsets[i]);
  auto constWrap = getConstantIntValue(wraps[i]);
  auto constStride = getConstantIntValue(strides[i]);
  auto outerStride =
      strides.empty() ? std::nullopt : getConstantIntValue(strides[0]);
  // Everything below is a REASON NOT TO FOLD, never a reason to fail. A large
  // airrt-level stride does not by itself mean the emitted BD is illegal --
  // later steps still reshape the access pattern -- so an op this rewrite
  // cannot help is left exactly as it was, and downstream decides. Erroring
  // here instead would reject transfers that compiled fine before this existed.
  if (!constOff || !constWrap || !constStride || !outerStride)
    return success();
  // Only a middle dim of a BD whose 4th dim is unused is dropped once its wrap
  // becomes 1; anywhere else the over-limit stride would still be encoded.
  if (strides.size() != AIE2_DIM_COUNT || i == 0 || i + 1 == strides.size() ||
      *outerStride != 0)
    return success();
  // Folding costs one descriptor per entry of the folded dim.
  if (*constWrap > AIE2_STRIDE_UNROLL_LIMIT)
    return success();
  // The pieces are issued back to back, so an enclosing dim that still walks is
  // traversed inside each piece instead of around them: (d,k) order becomes
  // (k,d). Same addresses, different arrival order, which an S2MM consumer
  // synchronising on locks reads as corrupt data. A stride-0 repeat counts --
  // it revisits the same addresses, so only the order distinguishes it. Only
  // fold when every enclosing dim is degenerate, which is the shape this exists
  // for: the KV append is wrap 1 outside the folded dim.
  for (unsigned d = 0; d < i; d++)
    if (getConstantIntValue(wraps[d]).value_or(0) != 1)
      return success();
  // generateAwaitsFromWaitAllOps matches waits to configure tasks FIFO per
  // channel -- the Nth wait for a channel awaits the Nth config for it. This
  // rewrite turns one config into `wrap` of them, so the channel needs `wrap`
  // waits or the pairing shifts: the original wait would land on the first
  // piece, every later transfer on the channel would be awaited one slot
  // early, and the tail pieces would go unawaited -- on an S2MM channel that
  // is both a missed completion token and a BD that is never freed. Emit the
  // extra waits alongside the extra configs so the 1:1 invariant survives.
  //
  // Only on a channel that is already waited: adding waits to a fire-and-
  // forget channel would impose synchronisation the design never asked for.
  StringRef waitedSym;
  if (auto metadata = memcpy_op->getAttrOfType<FlatSymbolRefAttr>("metadata"))
    if (auto func = memcpy_op->getParentOfType<func::FuncOp>())
      func.walk([&](AIEX::NpuDmaWaitOp wait) {
        if (wait.getSymbol() == metadata.getValue())
          waitedSym = metadata.getValue();
      });

  // The op's !airrt.event is optional and, unlike tileIllegalWrapDim (which
  // rewrites in place), this replaces the op -- so the result has to be
  // carried across. The pieces are issued in program order on one channel, so
  // the last one completing implies all of them have; give it the event and
  // let it stand in for the original. This mirrors how coalesceShimDmaOrder
  // redirects the event tokens of the ops it merges away.
  SmallVector<Type> eventTy(memcpy_op->getResultTypes());
  airrt::DmaMemcpyNdOp lastOp;
  for (int64_t k = 0; k < *constWrap; k++) {
    SmallVector<OpFoldResult> newOffsets(offsets), newWraps(wraps),
        newStrides(strides);
    newOffsets[i] = builder.getI64IntegerAttr(*constOff + k);
    newWraps[i] = builder.getI64IntegerAttr(1);
    bool isLast = k + 1 == *constWrap;
    lastOp = airrt::DmaMemcpyNdOp::create(
        builder, loc, isLast ? eventTy : SmallVector<Type>{}, memcpy_op.getId(),
        memcpy_op.getX(), memcpy_op.getY(), memcpy_op.getMemref(), newOffsets,
        newWraps, newStrides);
    // Ordering/barrier markers must ride along on every piece, or the split
    // silently drops the append barrier and the shim order constraint.
    lastOp->setAttrs(memcpy_op->getDiscardableAttrDictionary());
  }
  // Where they go does not change the pairing -- FIFO orders waits among
  // themselves, and anywhere after the pieces and before the transfer's own
  // wait gives piece k the kth of these and the last piece the original wait.
  // It does change when the awaits execute, so prefer the transfer's own wait:
  // a design that deliberately waits late keeps the overlap it asked for
  // instead of being synchronised early. Only in the same block, though -- a
  // wait in a nested region (a per-iteration drain inside an scf.for, say)
  // would multiply these by the trip count and unbalance the very pairing they
  // exist to preserve. Falling back to right after the pieces is always valid:
  // every piece dominates them, which the pairing's dominance guard requires.
  if (!waitedSym.empty()) {
    for (Operation *o = memcpy_op->getNextNode(); o; o = o->getNextNode())
      if (auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(o))
        if (wait.getSymbol() == waitedSym) {
          builder.setInsertionPoint(wait);
          break;
        }
    for (int64_t k = 1; k < *constWrap; k++)
      AIEX::NpuDmaWaitOp::create(builder, loc, waitedSym);
  }
  memcpy_op->replaceAllUsesWith(lastOp);
  memcpy_op.erase();
  folded = true;
  return success();
}

LogicalResult enforceAIE2StrideLimit(ModuleOp module) {
  // One pass peels one dim; a transfer with several illegal strides needs
  // several. Iterate to a fixpoint, bounded by the dim count.
  for (unsigned round = 0; round < AIE2_DIM_COUNT; round++) {
    SmallVector<airrt::DmaMemcpyNdOp> targets;
    module.walk([&](func::FuncOp f) {
      f.walk([&](airrt::DmaMemcpyNdOp dma) {
        if (violatesAIE2StrideLimit(dma))
          targets.push_back(dma);
      });
    });
    if (targets.empty())
      return success();
    bool changed = false;
    for (auto op : targets) {
      bool folded = false;
      if (failed(unrollIllegalStrideDim(op, folded)))
        return failure();
      changed |= folded;
    }
    // Nothing left that this rewrite can help with; the rest is downstream's.
    if (!changed)
      return success();
  }
  return success();
}

LogicalResult enforceAIE2WrapLimit(ModuleOp module) {
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
    if (failed(tileIllegalWrapDim(memcpy_op)))
      return failure();
  return success();
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

    auto ctx = &getContext();

    // Convert any surviving scf.forall ops to scf.for before unrolling.
    // scf.forall is not handled by the loop unrolling passes below and would
    // leave dynamic induction variables that DmaToNpuPattern silently zeros.
    {
      SmallVector<scf::ForallOp> forallOps;
      module.walk([&](scf::ForallOp op) { forallOps.push_back(op); });
      if (!forallOps.empty()) {
        IRRewriter rewriter(ctx);
        for (auto forallOp : forallOps) {
          if (failed(scf::forallToForLoop(rewriter, forallOp))) {
            forallOp->emitOpError("failed to convert forall to for loop");
            signalPassFailure();
            return;
          }
        }
      }
    }

    // The runtime sequence's loops stay rolled; aiecc's
    // aie-unroll-runtime-sequence-loops is the one unroller in the stack. The
    // affine.for nests are the launch/segment scaffolding, which still has to
    // go for the airrt ops to sit directly in the sequence.
    unrollAffineFors(module);
    dropAirrtEventResults(module);
    dropAirrtEventIterArgs(module);

    // Fold affine.apply ops with constant operands after loop unrolling.
    // After unrolling, induction variables become constants, but
    // affine.apply(constant) is not automatically folded. Without this,
    // DmaToNpuPattern's getConstantIntValue() fails and defaults offsets to 0.
    {
      RewritePatternSet affinePatterns(ctx);
      affine::AffineApplyOp::getCanonicalizationPatterns(affinePatterns, ctx);
      if (failed(applyPatternsGreedily(module, std::move(affinePatterns)))) {
        module.emitError("failed to canonicalize affine.apply ops after loop "
                         "unrolling");
        signalPassFailure();
        return;
      }
    }

    // Fold constant-condition launch-scope scf.index_switch after unrolling.
    // A fused per-wave launch that selects per-wave host feeds via an
    // scf.index_switch(select(wave_iv < K, ...)) has a constant condition once
    // the wave loop is unrolled; the arith chain (cmpi/select/index_cast) folds
    // to a constant and the switch's chosen branch must be inlined so its feed
    // ops sit directly in the runtime sequence (an index_switch cannot parent
    // aiex.dma_configure_task_for). Runs the arith folds + branch inlining
    // greedily so the chain collapses in one sweep.
    {
      RewritePatternSet p(ctx);
      arith::CmpIOp::getCanonicalizationPatterns(p, ctx);
      arith::SelectOp::getCanonicalizationPatterns(p, ctx);
      arith::IndexCastOp::getCanonicalizationPatterns(p, ctx);
      arith::ExtUIOp::getCanonicalizationPatterns(p, ctx);
      arith::ExtSIOp::getCanonicalizationPatterns(p, ctx);
      arith::AddIOp::getCanonicalizationPatterns(p, ctx);
      arith::MulIOp::getCanonicalizationPatterns(p, ctx);
      arith::SubIOp::getCanonicalizationPatterns(p, ctx);
      p.insert<FoldConstIndexSwitchPattern>(ctx);
      (void)applyPatternsGreedily(module, std::move(p));
    }

    // Convert WaitAllOp → NpuDmaWaitOp and purge DMA async tokens.
    // This must happen BEFORE DMA conversion because:
    // 1. WaitAllOp has SSA operands to DmaMemcpyNdOp event tokens
    // 2. NpuDmaWaitOp uses symbol reference (can be created before DMA
    // conversion)
    // 3. After this, DMA tokens can be safely purged

    // Decide the per-launch lock/DMA reset for each device now, while the
    // launch_end markers (and the loops that mark a multi-iteration launch) are
    // still present: the conversion below erases the markers and
    // unrollAffineFors strips the loops, so the decision cannot be recomputed
    // afterwards.
    markDevicesNeedingLockReset(module);

    // Tag each fused-launch source airrt op with its wave index while the
    // launch_end markers are still present and program order reflects wave
    // membership (before the markers are lowered by the next step).
    assignLaunchWaveIndices(module);

    // Coalesce contiguous same-channel paced shim feeds into wider
    // transfers before the wait-generation and wrap-enforcement steps, so
    // fewer DMA tasks and awaits are emitted. Runs after wave tagging (the kept
    // op keeps its wave attribute) and before WaitAll->wait conversion (erased
    // ops' event tokens are redirected to the merged op).
    if (clCoalesceShimDma)
      coalesceShimDmaOrder(module);

    generateNpuWaitFromAIRRtWaitAll(module);

    // Enforce AIE2 hardware constraints.
    if (failed(enforceAIE2WrapLimit(module))) {
      signalPassFailure();
      return;
    }

    // After wrap tiling, since tiling can introduce a new outer dim whose
    // stride is the product of the tiled ones.
    if (failed(enforceAIE2StrideLimit(module))) {
      signalPassFailure();
      return;
    }

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
          return !xilinx::air::isL1(
              llvm::cast<BaseMemRefType>(op.getMemref().getType()));
        });
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      if (op->getParentOfType<AIE::CoreOp>())
        return true;
      return !xilinx::air::isL1(
          llvm::cast<BaseMemRefType>(op.getMemref().getType()));
    });
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
      // Either container: on the rolled path the control func has already
      // become its runtime_sequence by the time this runs, and a copy against
      // a sequence argument still has to be folded away.
      ValueRange args;
      if (auto f = op->getParentOfType<func::FuncOp>())
        args = f.getArguments();
      else if (auto s = op->getParentOfType<AIE::RuntimeSequenceOp>())
        args = s.getBody().getArguments();
      for (auto arg : args) {
        if (op.getTarget() == arg)
          return false;
        else if (op.getSource() == arg)
          return false;
      }
      return true;
    });
    // A DMA that stays inside an scf.for has to be created under the
    // runtime_sequence directly: the AIEX ops require it as an ancestor, and
    // nothing hoists them out of the loop later the way unrolling did. So for
    // the rolled path the func becomes the sequence first.
    {
      // Its own target: the airrt ops are still there and are legal until the
      // main conversion below runs.
      ConversionTarget seqTarget(getContext());
      seqTarget.addLegalDialect<arith::ArithDialect, AIE::AIEDialect,
                                AIEX::AIEXDialect, memref::MemRefDialect,
                                airrt::AIRRtDialect, scf::SCFDialect,
                                affine::AffineDialect>();
      // Illegal exactly where ControlFuncConversion matches: the kernel
      // declarations sitting in the device are not control funcs.
      seqTarget.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp f) {
        if (f.isExternal() || !f->getParentOfType<AIE::DeviceOp>())
          return true;
        bool relevant = false;
        f.walk([&](Operation *o) {
          if (o->getName().getStringRef().starts_with("aiex.npu.") ||
              o->getName().getStringRef().starts_with("aiex.dma_") ||
              isa<airrt::DmaMemcpyNdOp>(o))
            relevant = true;
        });
        return !relevant;
      });
      RewritePatternSet earlySeq(ctx);
      earlySeq.add<ControlFuncConversion>(ctx);
      if (failed(
              applyPartialConversion(module, seqTarget, std::move(earlySeq)))) {
        signalPassFailure();
        return;
      }
    }

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

    // Buffer npu.dma_memcpy_nd memref to function's argument list. Has to run
    // here, after the copy folding above has removed the L3 allocs it would
    // otherwise promote; on the rolled path the container is already a
    // runtime_sequence, which the func-shaped pattern cannot see.
    RewritePatternSet castPattern(ctx);
    air::populateBufferMemrefToFuncArgsPattern(castPattern);
    (void)applyPatternsGreedily(module, std::move(castPattern));
    bufferMemrefToSequenceArgs(module);

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

    // Create a lightweight copy of the segment device (without core
    // bodies/ELFs) and redirect between-iteration load_pdi to it. Only needed
    // in ELF output mode -- load_pdi is never emitted otherwise, so the clone
    // would be dead IR.
    if (clOutputElf)
      createLightweightResetDevice(module);

    // Generate main device wrapper if needed. This handles two mutually
    // exclusive cases:
    // 1. Multi-device: pendingMainDevice was set by moveFuncOpToEndOfDeviceOp
    // 2. Single device fallback: XRTRunner path with emit-main-device flag
    // This MUST run at the very end after ALL patterns that modify
    // runtime_sequence arguments.
    generateMainDeviceIfNeeded(module);

    // Strip the internal reset-decision marker so it does not leak into the
    // emitted IR (the reset clones copy it too).
    module.walk([](xilinx::AIE::DeviceOp device) {
      device->removeAttr(kNeedsLockResetAttr);
    });

    // Hoist each herd's RTP writes and herd-release set_locks to the front of
    // their OWN configuration region, so every RTP value is latched and every
    // core released before the data movement that region drives. RTP writes and
    // set_locks are emitted at the herd-load position, which can fall after the
    // DMAs that trigger the cores (the control program may even await a core's
    // output before reaching them) -- the core then latches a stale (zero) RTP
    // or is never released and produces no output. This ordering is not
    // expressible in the async dependence graph (RTP writes have no SSA
    // operands; set_locks are batched separately).
    //
    // For single-dispatch sequences the region is delimited by
    // aiex.npu.load_pdi resets (ELF path); each region's own RTP block hoists
    // to the front of its own feeds. Fused multi-iteration launches take the
    // wave-keyed path below instead (each arm is placed by its air.launch_wave
    // index).
    // Move an NpuWriteRTPOp before `anchor`, re-materializing any defining op
    // that would otherwise end up after it. The #1732 AIEX API materializes the
    // RTP value as an SSA operand (not an attribute), so its definition must
    // dominate the moved write. Constants are the usual case; a runtime RTP
    // value adds the narrowing cast that carries a sequence argument into the
    // slot's i32. Both are pure and rooted at block arguments, so cloning them
    // at the new position is sound; anything else is left where it is.
    auto moveRtpBefore = [](Operation *rtp, Operation *anchor) {
      rtp->moveBefore(anchor);
      for (OpOperand &operandUse : rtp->getOpOperands()) {
        Value operand = operandUse.get();
        auto *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != rtp->getBlock())
          continue;
        if (!isMemoryEffectFree(defOp) || defOp->getNumResults() != 1)
          continue;
        if (llvm::any_of(defOp->getOperands(),
                         [](Value v) { return v.getDefiningOp() != nullptr; }))
          continue;
        bool defIsAfter = false;
        for (Operation *cur = rtp->getNextNode(); cur != nullptr;
             cur = cur->getNextNode())
          if (cur == defOp) {
            defIsAfter = true;
            break;
          }
        if (defIsAfter) {
          OpBuilder b(rtp);
          Operation *cloned = b.clone(*defOp);
          operandUse.set(cloned->getResult(0));
          // The original is left for canonicalization: the op-order snapshot
          // this hoist walks still points at it.
        }
      }
    };
    module.walk([&](AIE::RuntimeSequenceOp seq) {
      if (seq.getBody().empty())
        return;
      // Rolled, the arms and the feeds they arm both live in the loop body, so
      // hoisting only within the sequence's entry block moves nothing and the
      // cores are released after the first wave's data is already in flight.
      // Hoist within every block instead.
      SmallVector<Block *> blocks;
      seq.getBody().walk([&](Block *b) { blocks.push_back(b); });
      for (Block *blkPtr : blocks) {
        Block &blk = *blkPtr;
        // Snapshot the op order; the delimiters never move, and RTP/set_lock
        // ops only move within their own region, so region membership computed
        // from this snapshot stays valid across the moves below.
        SmallVector<Operation *> ops;
        for (auto &o : blk)
          ops.push_back(&o);

        // One arm group per launch. Both things that can delimit a launch
        // boundary are cuts of this block's op list, emitted from the SAME
        // air.launch_end marker: AIRRtWaitAllOpConversion turns each marker
        // into an aiex.npu.load_pdi (ELF + lock reset) or into dma_waits
        // (plain xclbin), while assignLaunchWaveIndices turns markers into a
        // wave index on each op. So cut on either signal: a load_pdi, or a
        // change of wave. Single dispatch has neither and stays one group.
        //
        // Cutting on both matters because neither alone covers every mode. A
        // plain xclbin multi-iteration launch emits no load_pdi, so only the
        // wave index separates its launches -- collapsing them would let a
        // later launch's RTP value land before an earlier one has run. And an
        // arm must never be hoisted above a load_pdi, or the reload clobbers
        // the RTP write it just placed.
        auto getWave = [](Operation *o) -> std::optional<int64_t> {
          if (auto a = o->getAttrOfType<IntegerAttr>(air::attrs::LaunchWave))
            return a.getInt();
          return std::nullopt;
        };
        struct ArmGroup {
          Operation *anchor = nullptr;
          SmallVector<Operation *> rtps;
          SmallVector<Operation *> locks;
        };
        SmallVector<ArmGroup> groups(1);
        std::optional<int64_t> curWave;
        for (Operation *o : ops) {
          if (isa<AIEX::NpuLoadPdiOp>(o)) {
            groups.emplace_back();
            curWave.reset();
            continue;
          }
          if (auto w = getWave(o)) {
            if (curWave && *w != *curWave)
              groups.emplace_back();
            curWave = w;
          }
          ArmGroup &g = groups.back();
          if (isa<AIEX::NpuWriteRTPOp>(o))
            g.rtps.push_back(o);
          else if (isa<AIEX::SetLockOp>(o))
            g.locks.push_back(o);
          else if (!o->hasAttr(air::attrs::RuntimeHoist) && !g.anchor)
            g.anchor = o;
        }
        for (auto &g : groups) {
          if (!g.anchor)
            continue;
          for (auto *rtp : g.rtps)
            moveRtpBefore(rtp, g.anchor);
          for (auto *lk : g.locks)
            lk->moveBefore(g.anchor);
        }
      }
    });

    // Per-wave output-S2MM-first ordering for fused multi-iteration launches.
    // The runtime emits each wave's output S2MM (core->DDR) tasks AFTER that
    // wave's input MM2S feeds (they follow the
    // producing air.channel.get in program order). An output S2MM must be armed
    // BEFORE its producing core hands off data, else the core blocks releasing
    // its output buffer lock (the S2MM BD is not yet running to drain it) while
    // the control program is still issuing later input feeds -> deadlock. A
    // single-mode sequence already arms its outputs first; extend that
    // to each fused wave by hoisting the wave's output-S2MM configure+start
    // ahead of the wave's first input feed (after the arm/set_lock block the
    // RTP hoist just placed). The output drain (dma_await_task) stays at the
    // wave boundary (emitted by generateAwaitsFromWaitAllOps). Gated to fused
    // multi-iteration launches so single-dispatch sequences are byte-identical.
    module.walk([&](AIE::RuntimeSequenceOp seq) {
      if (seq.getBody().empty())
        return;
      auto device = seq->getParentOfType<xilinx::AIE::DeviceOp>();
      if (!device || !deviceHasMultiIterLaunch(device))
        return;
      auto isS2MMOutput = [&](AIEX::DMAConfigureTaskForOp cfg) {
        StringRef sym = cfg.getAlloc().getLeafReference().getValue();
        auto alloc = AIE::ShimDMAAllocationOp::getForSymbol(device, sym);
        return alloc && alloc.getChannelDir() == AIE::DMAChannelDir::S2MM;
      };
      // Rolled, the wave is the loop IV, so key by block instead: within each
      // block the first input feed anchors that block's output arming. Same
      // rule, one level down.
      SmallVector<Block *> blocks;
      seq.getBody().walk([&](Block *b) { blocks.push_back(b); });
      for (Block *blkPtr : blocks) {
        Block &blk = *blkPtr;
        SmallVector<Operation *> ops;
        for (auto &o : blk)
          ops.push_back(&o);
        // Wave-keyed: for each wave, its first input feed (an MM2S configure)
        // is the anchor; hoist that wave's output-S2MM configure+start ahead
        // of it. Rolled, the key is the block itself.
        // Block first, then wave within the block: a rolled body holds one
        // wave and keys to 0, while an already-flattened multi-wave block
        // still arms each wave's output ahead of that wave's own input.
        auto getWave = [](Operation *o) -> std::optional<int64_t> {
          if (auto a = o->getAttrOfType<IntegerAttr>(air::attrs::LaunchWave))
            return a.getInt();
          return std::nullopt;
        };
        llvm::SmallDenseSet<int64_t> wavesHere;
        for (Operation *o : ops)
          if (auto w = getWave(o))
            wavesHere.insert(*w);
        bool byWave = wavesHere.size() > 1;
        llvm::DenseMap<int64_t, Operation *> waveInputAnchor;
        auto key = [&](Operation *o) -> std::optional<int64_t> {
          if (!byWave)
            return 0;
          return getWave(o);
        };
        for (Operation *o : ops) {
          auto cfg = dyn_cast<AIEX::DMAConfigureTaskForOp>(o);
          if (!cfg || isS2MMOutput(cfg))
            continue;
          if (auto w = key(o))
            waveInputAnchor.try_emplace(*w, o);
        }
        for (Operation *o : ops) {
          auto cfg = dyn_cast<AIEX::DMAConfigureTaskForOp>(o);
          if (!cfg || !isS2MMOutput(cfg))
            continue;
          auto w = key(o);
          if (!w)
            continue;
          auto it = waveInputAnchor.find(*w);
          if (it == waveInputAnchor.end())
            continue;
          Operation *anchor = it->second;
          if (!cfg->isBeforeInBlock(anchor))
            cfg->moveBefore(anchor);
          // its single start task moves with it, staying right after the cfg.
          for (auto *u : cfg.getResult().getUsers())
            if (isa<AIEX::DMAStartTaskOp>(u) && !u->isBeforeInBlock(anchor))
              u->moveBefore(anchor);
        }
      }
    });

    // Hoist input DMA feeds (air.runtime_hoist) and enforce the
    // append->readback barrier (air.await_appends). These opt-in orderings
    // target a single configuration region, so anchor them at the global front
    // of the sequence.
    module.walk([&](AIE::RuntimeSequenceOp seq) {
      if (seq.getBody().empty())
        return;
      // Rolled, the hoisted feeds and the append/readback pair live in the
      // loop body or a select arm, so scoping to the entry block finds nothing
      // and the append barrier is silently not applied.
      SmallVector<Block *> blocks;
      seq.getBody().walk([&](Block *b) { blocks.push_back(b); });
      for (Block *blkPtr : blocks) {
        Block &blk = *blkPtr;
        Operation *anchor = nullptr;
        for (auto &o : blk) {
          if (isa<AIEX::NpuLoadPdiOp, AIEX::NpuWriteRTPOp, AIEX::SetLockOp>(&o))
            continue;
          if (o.hasAttr(air::attrs::RuntimeHoist))
            continue;
          anchor = &o;
          break;
        }
        if (!anchor)
          return; // nothing to order before.
        // Hoist input DMA feeds marked `air.runtime_hoist` ahead of the bulk
        // input-feed DMAs. Otherwise the control program can block on a later
        // input's dma_await -- whose consumer is stalled in a feedback loop
        // waiting on the compute that the hoisted feed drives -- BEFORE it ever
        // issues the hoisted feed, so that compute never receives its input,
        // produces no output, and the sequence deadlocks. Each configure task
        // and its single start task are moved together, preserving relative
        // order.
        SmallVector<AIEX::DMAConfigureTaskForOp> hoistCfgs;
        for (auto &o : blk)
          if (auto c = dyn_cast<AIEX::DMAConfigureTaskForOp>(&o))
            if (c->hasAttr(air::attrs::RuntimeHoist))
              hoistCfgs.push_back(c);
        for (auto c : hoistCfgs) {
          Operation *startOp = nullptr;
          for (auto *u : c.getResult().getUsers())
            if (isa<AIEX::DMAStartTaskOp>(u)) {
              startOp = u;
              break;
            }
          c->moveBefore(anchor);
          if (startOp)
            startOp->moveBefore(anchor);
          else
            // A present marker that cannot be honored is a silent-deadlock
            // hazard; surface it rather than dropping the requested ordering
            // quietly.
            c->emitWarning("air.runtime_hoist: no matching dma_start_task; the "
                           "feed was hoisted but its start was not, so the "
                           "requested ordering may not take effect");
        }

        // air.await_appends barrier: a same-L3 write-after-write / read-after-
        // write ordering that the async dependence graph cannot express. A
        // shared- DDR readback tagged `air.await_appends` must observe values
        // written by one or more device-side appends (S2MM drains into that
        // same DDR buffer), but an append's completion await is deferred to the
        // launch terminator, so the readback -- issued in program order after
        // the append START -- would race the append S2MM and read a stale slot.
        // Each participating append is tagged `air.append_barrier`; move those
        // appends' completion awaits to just BEFORE the tagged readback's
        // start, so the runtime blocks on append completion before reading
        // back.
        //
        // A runtime sequence may contain one or MORE independent readbacks
        // (e.g. an unrolled loop with N append/readback pairs). Each append's
        // completion await is moved before the FIRST tagged readback start that
        // follows the append in program order -- the readback that consumes it.
        // Collapsing every append onto the first readback would move a later
        // readback's append await ahead of an earlier readback, violating SSA
        // dominance and the append->readback ordering. With a single readback
        // this reduces to moving every append's await before that one readback.
        SmallVector<AIEX::DMAConfigureTaskForOp> awaitCfgs;
        for (auto &o : blk)
          if (auto c = dyn_cast<AIEX::DMAConfigureTaskForOp>(&o))
            if (c->hasAttr(air::attrs::AwaitAppends))
              awaitCfgs.push_back(c);
        if (!awaitCfgs.empty()) {
          // First dma_start_task among a configure task's users, if any.
          auto getStart = [](AIEX::DMAConfigureTaskForOp c) {
            for (auto *u : c.getResult().getUsers())
              if (auto s = dyn_cast<AIEX::DMAStartTaskOp>(u))
                return s;
            return AIEX::DMAStartTaskOp(nullptr);
          };
          // Program-order index for every op in the block. Only awaits are
          // relocated below (append/readback starts stay put), so the indices
          // used for the interval decisions remain valid throughout.
          DenseMap<Operation *, unsigned> order;
          unsigned idx = 0;
          for (auto &o : blk)
            order[&o] = idx++;
          // The L3 buffer a configure task's BD touches, or null when it cannot
          // be determined. The barrier is a SAME-BUFFER ordering (see the
          // comment above: "a shared-DDR readback ... must observe values
          // written by ... appends into that same DDR buffer"), so the append
          // and the readback have to be matched on it. Without the check, an
          // append's await is moved before the first tagged readback of ANY
          // buffer -- which is not just imprecise, it deadlocks: a design whose
          // drain into buffer A is only producible after a feed issued later
          // than an unrelated readback of buffer B has that drain awaited
          // before the feed goes out, and the dispatch hangs with nothing
          // written. (RMS_BAND_STREAM>=3 in programming_examples/fused_decode:
          // @layerOut appends into X, @inKV_K reads the KV cache.)
          auto cfgBuffer = [](AIEX::DMAConfigureTaskForOp c) -> Value {
            Value v = nullptr;
            c.walk([&](AIE::DMABDOp bd) {
              if (!v)
                v = bd.getBuffer();
            });
            return v;
          };
          // Tagged readback starts, in program order, each with the buffer it
          // reads back.
          SmallVector<std::pair<AIEX::DMAStartTaskOp, Value>> barrierStarts;
          for (auto c : awaitCfgs) {
            if (auto s = getStart(c))
              barrierStarts.push_back({s, cfgBuffer(c)});
            else
              c->emitWarning("air.await_appends: tagged readback has no "
                             "dma_start_task; the "
                             "append barrier cannot be applied");
          }
          bool anyAppendAwait = false;
          for (auto &o : blk) {
            auto cfg = dyn_cast<AIEX::DMAConfigureTaskForOp>(&o);
            if (!cfg || !cfg->hasAttr(air::attrs::AppendBarrier))
              continue;
            AIEX::DMAAwaitTaskOp aAwait = nullptr;
            for (auto *u : cfg.getResult().getUsers())
              if (auto aw = dyn_cast<AIEX::DMAAwaitTaskOp>(u))
                aAwait = aw;
            if (!aAwait)
              continue;
            anyAppendAwait = true;
            AIEX::DMAStartTaskOp aStart = getStart(cfg);
            unsigned apos = order[aStart ? aStart.getOperation() : &o];
            Value aBuf = cfgBuffer(cfg);
            AIEX::DMAStartTaskOp target = nullptr;
            unsigned best = std::numeric_limits<unsigned>::max();
            for (auto [s, rBuf] : barrierStarts) {
              // Same buffer only. Unknown on either side falls back to the
              // old, buffer-blind behaviour: the barrier stays conservative
              // when it cannot prove the two are unrelated.
              if (aBuf && rBuf && aBuf != rBuf)
                continue;
              unsigned sp = order[s.getOperation()];
              if (sp > apos && sp < best) {
                best = sp;
                target = s;
              }
            }
            if (target)
              aAwait->moveBefore(target);
          }
          if (!anyAppendAwait && !barrierStarts.empty())
            barrierStarts.front().first->emitWarning(
                "air.await_appends: readback tagged but no air.append_barrier "
                "appends found to await; no ordering was enforced");
        }
      }
    });

    // Bound how far one shim feed channel may run ahead of its siblings.
    boundShimFeedBursts(module);

    // Repair dominance after the reordering above. Every hoist here moves a
    // configure task or an RTP write past other ops, and a runtime access
    // pattern brings along the small arith chain that narrows a sequence
    // argument into the BD's i32 length or offset. That chain is pure and
    // rooted at block arguments, so re-materializing it at the moved use is
    // sound -- and is the only option, since the hoists are ordering decisions
    // that cannot be undone. A chain reaching anything impure is left alone and
    // will be caught by the verifier rather than silently miscompiled.
    module.walk([&](AIE::RuntimeSequenceOp seq) {
      if (seq.getBody().empty())
        return;
      // Same reason as the hoist above: rolled, the ops that moved are in the
      // loop body, not the sequence's entry block.
      SmallVector<Block *> blocks;
      seq.getBody().walk([&](Block *b) { blocks.push_back(b); });
      for (Block *blkPtr : blocks) {
        Block &blk = *blkPtr;
        llvm::DenseSet<Operation *> seen;
        // Clone `v`'s defining chain immediately before `user`, returning the
        // re-materialized value, or null if any step is not safely clonable.
        std::function<Value(Value, Operation *)> rematerialize =
            [&](Value v, Operation *user) -> Value {
          Operation *def = v.getDefiningOp();
          if (!def || def->getBlock() != &blk || seen.contains(def))
            return v;
          if (!isMemoryEffectFree(def) || def->getNumResults() != 1)
            return Value();
          SmallVector<Value> operands;
          for (Value o : def->getOperands()) {
            Value r = rematerialize(o, user);
            if (!r)
              return Value();
            operands.push_back(r);
          }
          OpBuilder b(user);
          Operation *cloned = b.clone(*def);
          cloned->setOperands(operands);
          seen.insert(cloned);
          return cloned->getResult(0);
        };
        for (Operation &o : blk) {
          SmallVector<OpOperand *> uses;
          for (OpOperand &u : o.getOpOperands())
            uses.push_back(&u);
          o.walk([&](Operation *nested) {
            for (OpOperand &u : nested->getOpOperands())
              uses.push_back(&u);
          });
          for (OpOperand *u : uses) {
            Operation *def = u->get().getDefiningOp();
            if (!def || def->getBlock() != &blk || seen.contains(def))
              continue;
            if (Value r = rematerialize(u->get(), &o))
              u->set(r);
          }
          seen.insert(&o);
        }
      }
    });

    // Strip the internal per-op wave index now that all per-wave ordering
    // (paced segmentation, RTP/set_lock hoist, output-S2MM hoist) has consumed
    // it, so it does not leak into the emitted IR.
    module.walk([](Operation *o) {
      if (o->hasAttr(air::attrs::LaunchWave))
        o->removeAttr(air::attrs::LaunchWave);
    });
    // Strip the internal multi-iteration launch markers -- all their consumers
    // (the launch_end drain, per-iteration paced-MM2S segmentation) have run --
    // so they do not leak into the emitted device IR.
    module.walk([](xilinx::AIE::DeviceOp device) {
      device->removeAttr(kMultiIterLaunchAttr);
      device->removeAttr(kNumLaunchItersAttr);
    });
  }

  // Bound how far one shim MM2S feed channel may run ahead of its siblings, and
  // how many tasks it may have in flight at once.
  //
  // Feeds come out of the conversion channel-major -- every task for A, then
  // every task for B -- because that is the order the channels' puts appear in,
  // and nothing bounds how many one channel may have outstanding. A shim
  // channel absorbs only a few: its DMA task queue (4 entries on AIE2) plus
  // whatever the L2 consumer is double-buffering. Measured on a 2x2 herd GEMM,
  // <= 6 tasks in flight on one channel completes and >= 7 hangs, independent
  // of the output drain's structure, the launch count, and the dtype
  // (Xilinx/mlir-air#1822).
  //
  // Two things are wrong, and both need fixing:
  //
  //   - Channel-major order lets A consume the whole budget before B is fed at
  //     all, so the cores wait on a B chunk that was never sent. Interleaving
  //     the burst round-robin fixes that, and costs nothing.
  //
  //   - The overflow itself is per channel and absolute: a push past what the
  //     channel holds is dropped, not deferred. A perfect A/B/A/B weave at 16
  //     deep still hangs, so interleaving alone is not a fix. Capping each
  //     channel's in-flight set -- await task i-limit before starting task i --
  //     is what actually removes the deadlock.
  //
  // Only bursts that exceed the limit are touched. A design whose channels were
  // already short-run keeps its emission order and its await structure byte for
  // byte, which is what keeps this off the hot path of the tuned LLM decoders.
  void boundShimFeedBursts(ModuleOp module) {
    // Per-channel tasks in flight that a shim channel absorbs without the
    // control program blocking on the push: the AIE2 shim DMA task queue depth.
    // The measured limit is 6 (queue + an L2 ping-pong); staying at the queue
    // depth alone keeps the bound independent of what the consumer buffers.
    constexpr unsigned burstLimit = 4;

    module.walk([&](AIE::RuntimeSequenceOp seq) {
      if (seq.getBody().empty())
        return;
      auto device = seq->getParentOfType<AIE::DeviceOp>();
      if (!device)
        return;

      // A feed whose position is already load-bearing is left alone, and fences
      // the burst it sits in: `air.runtime_hoist` was deliberately moved to the
      // front, the paced/coalesced feeds are bounded by
      // synthesizeDoubleBufferedAwaits, and the append-barrier feeds carry a
      // write-after-write ordering the dependence graph cannot express.
      auto isReorderable = [&](AIEX::DMAConfigureTaskForOp cfg) {
        for (StringRef a :
             {air::attrs::RuntimeHoist, air::attrs::PreserveShimDmaOrder,
              air::attrs::CoalescedShimFeed, air::attrs::AppendBarrier,
              air::attrs::AwaitAppends})
          if (cfg->hasAttr(a))
            return false;
        auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
            device, cfg.getAlloc().getLeafReference().getValue());
        return allocOp && allocOp.getChannelDir() == AIE::DMAChannelDir::MM2S;
      };

      struct Unit {
        AIEX::DMAConfigureTaskForOp cfg;
        Operation *start;
        StringRef chan;
      };

      SmallVector<Block *> blocks;
      seq.getBody().walk([&](Block *b) { blocks.push_back(b); });
      for (Block *blk : blocks) {
        SmallVector<Unit> run;
        SmallVector<Operation *> staleFrees;

        // Lay the burst back down just before `fence`, round-robin over the
        // channels it feeds, then cap each channel's in-flight set. Each
        // channel keeps its own relative order -- the consumer ring depends on
        // it -- so the weave only changes how the channels are interleaved with
        // each other. A single-channel burst has nothing to weave and just gets
        // the cap.
        auto flush = [&](Operation *fence) {
          if (run.size() < 2 || !fence)
            return;
          llvm::MapVector<StringRef, SmallVector<Unit>> byChan;
          for (const Unit &u : run)
            byChan[u.chan].push_back(u);
          unsigned deepest = 0;
          for (auto &kv : byChan)
            deepest = std::max<unsigned>(deepest, kv.second.size());
          if (deepest <= burstLimit)
            return; // already within what a channel absorbs; leave as emitted.
          for (unsigned i = 0; i < deepest; i++)
            for (auto &kv : byChan) {
              if (i >= kv.second.size())
                continue;
              const Unit &u = kv.second[i];
              u.cfg->moveBefore(fence);
              u.start->moveBefore(fence);
            }
          // Interleaving alone is not enough. It does bound how far one channel
          // runs ahead, but the overflow is per channel and absolute: a push
          // past what the channel can hold is dropped, not deferred, so the
          // chunk is simply never sent. Measured with a perfect A/B/A/B weave
          // at 16 deep, the design still hangs. So cap each channel's in-flight
          // set as well: before starting task i, await the token from task
          // i-burstLimit, which cannot have retired any later than that.
          for (auto &kv : byChan) {
            SmallVector<Unit> &chan = kv.second;
            for (unsigned i = burstLimit; i < chan.size(); i++) {
              AIEX::DMAConfigureTaskForOp older = chan[i - burstLimit].cfg;
              // An MM2S task issues no completion token by default, so there
              // would be nothing to wait on.
              older.setIssueToken(true);
              OpBuilder b(chan[i].start);
              AIEX::DMAAwaitTaskOp::create(b, chan[i].start->getLoc(),
                                           older.getResult());
              // An await also frees the BD, so the fire-and-free this task was
              // given at conversion would now be a second release. Erase those
              // only once the block walk is done -- a free sits after the
              // burst, so erasing it here would pull the ground out from under
              // the iterator.
              for (auto *u : older.getResult().getUsers())
                if (isa<AIEX::DMAFreeTaskOp>(u))
                  staleFrees.push_back(u);
            }
          }
        };

        // Walk the block, growing a run of reorderable feeds. Anything that is
        // not such a feed and is not pure ends it: an await, a free, an RTP
        // write, a lock set, a PDI load -- each is an ordering point, and a
        // burst only exists between them. Pure ops (the arith chain narrowing a
        // runtime length or offset) are transparent; moving a feed above its
        // operands is repaired by the rematerialization pass below, which is
        // there for exactly this.
        for (Operation &o : llvm::make_early_inc_range(*blk)) {
          if (auto cfg = dyn_cast<AIEX::DMAConfigureTaskForOp>(&o)) {
            Operation *start = nullptr;
            for (auto *u : cfg.getResult().getUsers()) {
              if (!isa<AIEX::DMAStartTaskOp>(u) || u->getBlock() != blk)
                continue;
              if (start) { // more than one start: not a plain single-shot feed
                start = nullptr;
                break;
              }
              start = u;
            }
            if (start && isReorderable(cfg)) {
              run.push_back(
                  {cfg, start, cfg.getAlloc().getLeafReference().getValue()});
              continue;
            }
            flush(&o);
            run.clear();
            continue;
          }
          if (isa<AIEX::DMAStartTaskOp>(&o)) {
            // The start of a feed already in the run travels with its configure
            // and does not break the burst; any other start does.
            bool owned =
                llvm::any_of(run, [&](const Unit &u) { return u.start == &o; });
            if (owned)
              continue;
            flush(&o);
            run.clear();
            continue;
          }
          if (isMemoryEffectFree(&o))
            continue;
          flush(&o);
          run.clear();
        }
        if (blk->mightHaveTerminator())
          flush(blk->getTerminator());
        run.clear();
        for (Operation *f : staleFrees)
          f->erase();
      }
    });
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
      // segment_load/herd_load)
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

  // Create a lightweight device clone for between-iteration load_pdi.
  // The clone has the same DMA BDs, locks, and switches but empty core
  // bodies (no ELFs). The between-iteration load_pdi references this
  // clone, so aie-expand-load-pdi generates a PDI without ELF data.
  void createLightweightResetDevice(ModuleOp module) {
    SmallVector<std::pair<AIE::DeviceOp, std::string>> devicesToClone;
    module.walk([&](AIE::DeviceOp device) {
      if (!deviceNeedsLockReset(device))
        return;
      if (device.getSymName().empty())
        return;
      devicesToClone.push_back({device, device.getSymName().str()});
    });

    for (auto &[device, origName] : devicesToClone) {
      std::string resetName = origName + "_reset";
      OpBuilder builder(device);
      auto clone = cast<AIE::DeviceOp>(builder.clone(*device));
      clone.setSymName(resetName);

      // Strip core bodies and attributes from the clone, and remove
      // runtime_sequence. CoreOps are kept (empty, no elf_file,
      // no link_with) so that initLocks does core reset/unreset and
      // addCoreEnable re-enables cores. aiecc.py skips compilation
      // for cores without link_with/elf_file.
      SmallVector<AIE::RuntimeSequenceOp> seqsToErase;
      clone.walk([&](AIE::RuntimeSequenceOp op) { seqsToErase.push_back(op); });
      for (auto op : seqsToErase)
        op->erase();

      SmallVector<xilinx::AIE::CoreOp> coresToReplace;
      clone.walk([&](xilinx::AIE::CoreOp coreOp) {
        coresToReplace.push_back(coreOp);
      });
      for (auto coreOp : coresToReplace) {
        OpBuilder b(coreOp);
        Value tile = coreOp.getTile();
        auto newCore = xilinx::AIE::CoreOp::create(b, coreOp.getLoc(), tile);
        Block *body = b.createBlock(&newCore.getBody());
        b.setInsertionPointToEnd(body);
        xilinx::AIE::EndOp::create(b, coreOp.getLoc());
        LLVM_DEBUG(llvm::dbgs()
                   << "Created empty CoreOp in reset device for tile: " << tile
                   << "\n");
        coreOp->erase();
      }

      // Redirect between-iteration load_pdi to the clone
      AIE::RuntimeSequenceOp runtimeSeq = nullptr;
      device.walk([&](AIE::RuntimeSequenceOp seq) { runtimeSeq = seq; });
      if (!runtimeSeq)
        continue;
      auto resetRef = FlatSymbolRefAttr::get(module.getContext(), resetName);
      auto &origNameRef = origName;
      runtimeSeq.walk([&origNameRef, &resetRef](AIEX::NpuLoadPdiOp op) {
        if (auto ref = op.getDeviceRefAttr()) {
          if (ref.getValue() == origNameRef)
            op.setDeviceRefAttr(resetRef);
        }
      });
    }
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
    // Either container is possible here: the rolled path has already turned
    // the control func into its runtime_sequence.
    SmallVector<Operation *> funcOps;
    module.walk([&](func::FuncOp f) { funcOps.push_back(f); });
    module.walk([&](AIE::RuntimeSequenceOp s) { funcOps.push_back(s); });

    for (auto f : funcOps) {
      auto device = f->getParentOfType<AIE::DeviceOp>();
      if (!device)
        continue;

      if (f->getRegion(0).empty())
        continue;

      mlir::DominanceInfo domInfo(f);

      // First pass: collect all DMAConfigureTaskForOp per channel in order
      // Map from metadata symbol -> list of ConfigTasks in order
      llvm::MapVector<StringRef, SmallVector<AIEX::DMAConfigureTaskForOp>>
          channelToConfigTasks;

      // Also track per-channel indices for matching
      llvm::DenseMap<StringRef, unsigned> channelToNextConfigIdx;

      // Walk the function body in order
      f->walk([&](AIEX::DMAConfigureTaskForOp configTask) {
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
      f->walk([&](AIEX::NpuDmaWaitOp waitOp) { waitOps.push_back(waitOp); });

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
              if (isShimTileValue(consumerTileOp)) {
                isS2MM = true;
                break;
              }
            }
          }
        }

        // Find the next ConfigTask for this channel.
        //
        // The Nth wait for a channel awaits its Nth config in FIFO order, BUT
        // only if that config dominates the wait. A fused multi-iteration
        // launch with HETEROGENEOUS waves (some waves use a channel and
        // others do not) emits a per-iteration all-shim
        // boundary drain that waits on EVERY channel at each boundary. For a
        // channel with no config in the current wave, FIFO would otherwise pair
        // that boundary wait with a config from a LATER wave -- producing a
        // dma_await_task whose config operand does not dominate it
        // (use-before-def). That invalid IR is not caught until
        // ControlFuncConversion clones the func (the clone leaves the await
        // referencing the un-mapped original config, crossing into the new
        // runtime_sequence -> erase-with-uses assert). Guard the match on
        // dominance: if the next unconsumed config does not dominate this wait,
        // leave it for a later (dominated) wait and emit no await here.
        AIEX::DMAConfigureTaskForOp matchingConfigTask = nullptr;
        auto it = channelToConfigTasks.find(metadata);
        if (it != channelToConfigTasks.end()) {
          auto &configTasks = it->second;
          unsigned &nextIdx = channelToNextConfigIdx[metadata];
          if (nextIdx < configTasks.size() &&
              domInfo.properlyDominates(configTasks[nextIdx].getOperation(),
                                        waitOp.getOperation())) {
            matchingConfigTask = configTasks[nextIdx];
            nextIdx++;
          }
        }

        if (matchingConfigTask) {
          OpBuilder builder(waitOp);
          // Block dominance is satisfied by an op in a sibling region -- a
          // configure in one arm of a rolled feed select -- but its result is
          // not visible outside that arm. Await at the end of the arm instead.
          if (!matchingConfigTask->getParentRegion()->isAncestor(
                  waitOp->getParentRegion()))
            builder.setInsertionPoint(
                matchingConfigTask->getBlock()->getTerminator());
          if (isS2MM) {
            // S2MM (output): await task - waits for completion AND frees BD
            AIEX::DMAAwaitTaskOp::create(builder, waitOp.getLoc(),
                                         matchingConfigTask.getResult());
          } else if (matchingConfigTask->hasAttr(
                         air::attrs::PreserveShimDmaOrder)) {
            // MM2S, paced: do not emit a fire-and-free here. Bounded
            // double-buffered awaits are synthesized in
            // synthesizeDoubleBufferedAwaits() below.
          } else {
            // MM2S (input): free task - just frees BD for reuse, no wait
            AIEX::DMAFreeTaskOp::create(builder, waitOp.getLoc(),
                                        matchingConfigTask.getResult());
          }
        }
        // Erase the NpuDmaWaitOp regardless of whether we found a match
        waitOp->erase();
      }

      // Emit bounded double-buffered awaits for paced (lockstep-coupled) MM2S
      // shim feeds. Must run after the default await/free emission above so it
      // can rely on the per-task config ops being in final program order.
      synthesizeDoubleBufferedAwaits(f, device, /*depth=*/2);
    }
  }

  // For shim feeds marked `air.preserve_shim_dma_order`, replace fire-and-free
  // with bounded double-buffered completion-token awaits. Such feeds are
  // coupled by a downstream broadcast/multicast consumer that advances all its
  // destinations in lockstep; with no backpressure the runtime over-commits one
  // channel's in-flight BDs while sibling channels starve, and deadlocks.
  // Keeping at most `depth` tasks in flight per channel (await task i-depth
  // before reusing its BD at start i, then drain the final `depth`) bounds the
  // in-flight set and preserves the drainable round-major schedule.
  void synthesizeDoubleBufferedAwaits(Operation *f, AIE::DeviceOp device,
                                      unsigned depth) {
    // Group marked MM2S config tasks per channel, in program order.
    llvm::MapVector<StringRef, SmallVector<AIEX::DMAConfigureTaskForOp>> groups;
    f->walk([&](AIEX::DMAConfigureTaskForOp ct) {
      if (!ct->hasAttr(air::attrs::PreserveShimDmaOrder))
        return;
      StringRef md = ct.getAlloc().getLeafReference().getValue();
      auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(device, md);
      if (!allocOp || allocOp.getChannelDir() != AIE::DMAChannelDir::MM2S)
        return;
      groups[md].push_back(ct);
    });
    if (groups.empty())
      return;
    // Map each config task to its start op.
    llvm::DenseMap<Operation *, AIEX::DMAStartTaskOp> startOf;
    f->walk([&](AIEX::DMAStartTaskOp st) {
      if (auto *def = st.getTask().getDefiningOp())
        startOf[def] = st;
    });

    // A segment is one (block, launch wave): a rolled body is one block and one
    // wave, a flattened multi-wave body is one block and several waves.
    auto waveOf = [](AIEX::DMAConfigureTaskForOp ct) -> int64_t {
      if (auto a = ct->getAttrOfType<IntegerAttr>(air::attrs::LaunchWave))
        return a.getInt();
      return -1;
    };
    // The last start among ALL paced channels in each segment. The tail drain
    // anchors here rather than after the individual channel's own last start:
    // both honour the fence (everything drained before the next segment
    // begins), but the per-channel anchor also serializes channels that have to
    // be in flight TOGETHER, and a one-task channel then awaits itself
    // immediately after its own start -- a synchronous feed, which deadlocks
    // when the transfer can only complete if its consumer runs and the consumer
    // is waiting on a sibling channel that has not been issued yet. See the
    // drain site below for the measured case.
    //
    // COALESCED FEEDS COUNT TOWARDS THE ANCHOR, even though they are paced by
    // the cross-channel phase barrier rather than by paceSegment. The anchor is
    // only asking "where does this segment stop issuing", and a coalesced
    // weight feed issued after the last paced start is still part of the
    // segment -- excluding it drains the paced channels while the segment is
    // still going, which is the same synchronous-feed deadlock one step out.
    // Measured (fused_decode RMS_BAND_STREAM>=3): the last @rmsX task is
    // residual2's band feed, and the core takes those bands only after the DOWN
    // projection has produced its output. Anchored at that feed's own start the
    // drain awaits it BEFORE the down weight feed is issued, so the projection
    // never runs and the dispatch hangs with the layer output unwritten. The
    // anchor only ever moves LATER within the same segment, so the fence it
    // provides -- everything drained before the next segment's first start --
    // is unchanged.
    llvm::DenseMap<std::pair<Block *, int64_t>, AIEX::DMAStartTaskOp>
        segLastStart;
    for (auto &kv : groups)
      for (auto ct : kv.second) {
        auto st = startOf.lookup(ct.getOperation());
        if (!st)
          continue;
        auto key = std::make_pair(st->getBlock(), waveOf(ct));
        auto it = segLastStart.find(key);
        if (it == segLastStart.end() ||
            it->second->isBeforeInBlock(st.getOperation()))
          segLastStart[key] = st;
      }
    // `fenceEnd` forces the segment's in-flight tail to be fully drained after
    // its last start (a per-iteration fence) even when the segment fits in
    // flight; used for the per-iteration segments of a fused multi-iteration
    // launch. The whole-list single-dispatch call passes fenceEnd=false to keep
    // the original behavior (no drain when the whole channel fits in flight).
    auto paceSegment = [&](ArrayRef<AIEX::DMAConfigureTaskForOp> tasks,
                           bool fenceEnd) {
      unsigned n = tasks.size();
      if (n == 0)
        return;
      // Before reusing task i's BD (start i), await task i-depth.
      for (unsigned i = depth; i < n; i++) {
        AIEX::DMAConfigureTaskForOp ti = tasks[i];
        AIEX::DMAConfigureTaskForOp tprev = tasks[i - depth];
        auto startOp = startOf.lookup(ti.getOperation());
        if (!startOp) {
          ti->emitWarning(
              "air.preserve_shim_dma_order: paced task has no dma_start_task; "
              "backpressure await was not inserted for it");
          continue;
        }
        OpBuilder b(startOp);
        AIEX::DMAAwaitTaskOp::create(b, startOp.getLoc(), tprev.getResult());
      }
      if (!fenceEnd && n <= depth)
        return; // whole-list, fits in flight: no pacing/drain needed
      // Drain the in-flight tail after the last start. For a per-iteration
      // segment this fences the iteration: its feeds fully drain before the
      // next iteration's first start. In-flight = the last `depth` tasks once
      // pacing ran, or all `n` when the segment itself fits in flight (n <=
      // depth).
      AIEX::DMAConfigureTaskForOp tlast = tasks[n - 1];
      auto lastStart = startOf.lookup(tlast.getOperation());
      if (!lastStart) {
        tlast->emitWarning(
            "air.preserve_shim_dma_order: last paced task has no "
            "dma_start_task; drain awaits were not inserted");
        return;
      }
      // Drain after the last start in the SEGMENT, not after this channel's own
      // last start. Both honour the fence -- everything is drained before the
      // next segment's first start, which for a rolled body is the back edge --
      // but the per-channel anchor additionally serializes channels that have
      // to be in flight TOGETHER.
      //
      // A one-task segment is the sharp case: drainStart == 0, so the task is
      // awaited immediately after its own start, turning a fire-and-forget feed
      // into a synchronous one. That deadlocks whenever the transfer can only
      // complete if its consumer runs and the consumer needs a SIBLING channel
      // that has not been issued yet.
      //
      // Measured: the LFM2 hybrid's KV read-back is four one-task channels
      // (K/V for two CU pairs) consumed by four CU pairs in parallel. The
      // per-channel anchor emitted
      //     cfg K_0; start K_0; await K_0; cfg V_0; await V_0; cfg K_1; ...
      // so CU0 could not drain K_0 until V_0 arrived, and V_0 was not issued
      // until K_0 completed. It survived while a whole region still fit in the
      // memtile ring plus the CUs' own buffering -- up to 5 blocks, ATTN_MAXL
      // 80 -- and cold-deadlocked from 6 blocks (96) upward at every ring
      // depth. Anchoring at the end of the segment puts all four in flight
      // together, which is what this same design already did whenever the feeds
      // happened to coalesce instead of being paced.
      //
      // The SEGMENT, not the block: a flattened multi-wave body is several
      // segments in ONE block, and draining at the block end there would let an
      // earlier wave's in-flight set straddle the next wave.
      AIEX::DMAStartTaskOp anchor = lastStart;
      auto segIt = segLastStart.find(
          std::make_pair(lastStart->getBlock(), waveOf(tasks[0])));
      if (segIt != segLastStart.end() &&
          anchor->isBeforeInBlock(segIt->second.getOperation()))
        anchor = segIt->second;
      OpBuilder b(anchor);
      b.setInsertionPointAfter(anchor);
      unsigned drainStart = (n > depth) ? n - depth : 0;
      for (unsigned j = drainStart; j < n; j++) {
        AIEX::DMAConfigureTaskForOp tj = tasks[j];
        AIEX::DMAAwaitTaskOp::create(b, lastStart.getLoc(), tj.getResult());
      }
    };

    // Cross-channel phase barrier for coalesced feeds. After coalescing, each
    // channel contributes one task per phase, emitted phase-major across
    // channels (all channels' phase p, then all channels' phase p+1, ...). Each
    // phase feeds a distinct downstream consumer, so overlapping two phases
    // deadlocks (a per-channel double-buffer would keep phase p of one channel
    // in flight while a sibling channel is still on phase p-1). Instead drain
    // every channel's phase-p transfer before any phase-(p+1) transfer starts:
    // fire all of a phase in parallel, then await the whole phase. A new phase
    // group begins whenever a channel repeats in program order (robust to
    // varying per-phase channel counts and to fused multi-iteration launches,
    // whose waves just append more phase groups).
    {
      SmallVector<AIEX::DMAConfigureTaskForOp> coalTasks;
      f->walk([&](AIEX::DMAConfigureTaskForOp ct) {
        if (!ct->hasAttr(air::attrs::CoalescedShimFeed))
          return;
        StringRef md = ct.getAlloc().getLeafReference().getValue();
        auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(device, md);
        if (allocOp && allocOp.getChannelDir() == AIE::DMAChannelDir::MM2S)
          coalTasks.push_back(ct);
      });
      if (!coalTasks.empty()) {
        SmallVector<SmallVector<AIEX::DMAConfigureTaskForOp>> phaseGroups;
        SmallVector<AIEX::DMAConfigureTaskForOp> cur;
        llvm::DenseSet<StringRef> seen;
        // A phase group is a run within one block: the awaits go after the
        // group's last start, which only has a meaning among siblings.
        Block *curBlock = nullptr;
        for (auto ct : coalTasks) {
          StringRef c = ct.getAlloc().getLeafReference().getValue();
          if (seen.contains(c) || (curBlock && ct->getBlock() != curBlock)) {
            phaseGroups.push_back(cur);
            cur.clear();
            seen.clear();
          }
          curBlock = ct->getBlock();
          cur.push_back(ct);
          seen.insert(c);
        }
        if (!cur.empty())
          phaseGroups.push_back(cur);
        for (auto &grp : phaseGroups) {
          // Insert the phase's awaits after its last start (program order), so
          // the whole phase drains before the next phase's first start.
          AIEX::DMAStartTaskOp lastStart = nullptr;
          for (auto ct : grp) {
            auto st = startOf.lookup(ct.getOperation());
            if (!st)
              continue;
            if (!lastStart || lastStart->isBeforeInBlock(st))
              lastStart = st;
          }
          if (!lastStart)
            continue;
          OpBuilder b(lastStart);
          b.setInsertionPointAfter(lastStart);
          for (auto ct : grp)
            AIEX::DMAAwaitTaskOp::create(b, lastStart.getLoc(), ct.getResult());
        }
      }
    }

    for (auto &kv : groups) {
      // Coalesced feeds are paced by the cross-channel phase barrier below
      // (each coalesced task is a whole contiguous run feeding a distinct
      // consumer; per-channel pacing cannot prevent phase p of channel A
      // overlapping phase p-1 of channels B/C/D). Pace only the NON-coalesced
      // tasks here: a channel may hold a mix (a coalesced contiguous run plus
      // isolated non-contiguous fragments), and those fragments still need
      // per-channel backpressure -- filtering instead of skipping the whole
      // channel avoids leaving them with no synthesized awaits.
      SmallVector<AIEX::DMAConfigureTaskForOp> tasks;
      for (auto ct : kv.second)
        if (!ct->hasAttr(air::attrs::CoalescedShimFeed))
          tasks.push_back(ct);
      unsigned n = tasks.size();
      if (n == 0)
        continue;
      // Segment per launch iteration when the iteration count divides the task
      // count evenly (each iteration emits the same per-channel feeds).
      // Otherwise fall back to whole-list pacing.
      // Rolled: the wave is the loop IV, so every task in the body carries the
      // same wave index and wave segmentation collapses. Segment by block
      // instead -- a loop body and each arm of a feed select are separate
      // segments, pacing never has to pair across them, and the back edge is
      // the iteration fence the per-wave segmentation was providing.
      {
        SmallVector<AIEX::DMAConfigureTaskForOp> seg;
        Block *segBlock = nullptr;
        auto flush = [&]() {
          if (!seg.empty()) {
            paceSegment(seg, /*fenceEnd=*/true);
            seg.clear();
          }
        };
        // Segment on block, and on wave within a block: a rolled body is one
        // block and one wave, so only the block boundary fires; an already
        // flattened multi-wave block is still paced and fenced per wave, so no
        // channel's in-flight set straddles an iteration boundary.
        // waveOf is shared with the segment-anchor map built above, so the
        // segmentation here and the drain anchor there cannot drift apart.
        int64_t segWave = -1;
        for (auto ct : tasks) {
          if (ct->getBlock() != segBlock || waveOf(ct) != segWave) {
            flush();
            segBlock = ct->getBlock();
            segWave = waveOf(ct);
          }
          seg.push_back(ct);
        }
        flush();
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
        auto newOp = airrt::DmaMemcpyNdOp::create(
            builder, dma->getLoc(), tys, dma.getId(), dma.getX(), dma.getY(),
            dma.getMemref(), dma.getMixedOffsets(), dma.getMixedLengths(),
            dma.getMixedStrides());
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
        std::tuple<bool, int, int> allocInfo = {
            alloc.getChannelDir() == AIE::DMAChannelDir::MM2S,
            alloc.getChannelIndex(), getColFromTileValue(alloc.getTile())};

        auto it =
            llvm::find_if(uniqueAllocs, [&](AIE::ShimDMAAllocationOp ualloc) {
              std::tuple<bool, int, int> uallocInfo = {
                  ualloc.getChannelDir() == AIE::DMAChannelDir::MM2S,
                  ualloc.getChannelIndex(),
                  getColFromTileValue(ualloc.getTile())};
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
      // The launch/segment affine scaffolding nested inside the kept scf.for
      // has nothing else to unroll it.
      SmallVector<affine::AffineForOp> afos;
      f.walk([&](affine::AffineForOp op) {
        if (!op->getParentOfType<affine::AffineForOp>())
          afos.push_back(op);
      });
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

  // Runtime-sequence twin of BufferMemrefToFuncArgsPattern: promote L3 allocs
  // sitting directly in the sequence body to sequence arguments.
  void bufferMemrefToSequenceArgs(ModuleOp module) {
    module.walk([&](AIE::RuntimeSequenceOp seq) {
      if (seq.getBody().empty())
        return;
      llvm::SetVector<Value> memrefs;
      for (auto &op : seq.getBody().front()) {
        if (!isa<memref::AllocOp, memref::AssumeAlignmentOp>(op))
          continue;
        for (auto res : op.getResults())
          if (auto t = dyn_cast_if_present<BaseMemRefType>(res.getType()))
            if (air::isL3(t))
              memrefs.insert(res);
      }
      for (Value v : memrefs)
        v.replaceAllUsesWith(
            seq.getBody().addArgument(v.getType(), seq.getLoc()));
    });
  }

  // The same for an scf.if / scf.index_switch that yields a token -- a
  // per-wave feed select does, once the loop around it stays rolled and its
  // condition no longer folds. Results and yield operands line up one to one
  // here, so dropping is symmetric.
  void dropAirrtEventResults(ModuleOp module) {
    SmallVector<Operation *> ops;
    module.walk([&](Operation *o) {
      if (!isa<scf::IfOp, scf::IndexSwitchOp>(o))
        return;
      if (llvm::any_of(o->getResultTypes(),
                       [](Type t) { return isa<airrt::EventType>(t); }))
        ops.push_back(o);
    });
    for (auto *op : llvm::reverse(ops)) {
      OpBuilder builder(op);
      auto eventTy = airrt::EventType::get(op->getContext());
      SmallVector<Type> keptTypes;
      SmallVector<unsigned> keptIdx;
      for (auto [i, t] : llvm::enumerate(op->getResultTypes()))
        if (!isa<airrt::EventType>(t)) {
          keptTypes.push_back(t);
          keptIdx.push_back(i);
        }
      OperationState state(op->getLoc(), op->getName());
      state.addOperands(op->getOperands());
      state.addTypes(keptTypes);
      state.addAttributes(op->getAttrs());
      for (unsigned i = 0, e = op->getNumRegions(); i < e; i++)
        state.addRegion();
      Operation *newOp = builder.create(state);
      for (auto [oldRegion, newRegion] :
           llvm::zip(op->getRegions(), newOp->getRegions()))
        newRegion.takeBody(oldRegion);
      for (Region &r : newOp->getRegions())
        for (Block &blk : r) {
          auto yield = dyn_cast<scf::YieldOp>(blk.getTerminator());
          if (!yield)
            continue;
          SmallVector<Value> keptYields;
          for (unsigned i : keptIdx)
            keptYields.push_back(yield.getOperand(i));
          yield->setOperands(keptYields);
        }
      builder.setInsertionPointAfter(newOp);
      unsigned kept = 0;
      for (auto res : op->getResults())
        res.replaceAllUsesWith(
            isa<airrt::EventType>(res.getType())
                ? airrt::WaitAllOp::create(builder, op->getLoc(), eventTy,
                                           SmallVector<Value>{})
                      .getResult(0)
                : newOp->getResult(kept++));
      op->erase();
    }
  }

  // Drop !airrt.event iter_args from scf.for loops. Unrolling used to dissolve
  // them; a loop that stays rolled carries the token across the back edge, and
  // the conversion has nothing to replace a loop-carried event with. Every
  // reader becomes a fresh empty airrt.wait_all, which the conversion erases.
  void dropAirrtEventIterArgs(ModuleOp module) {
    SmallVector<scf::ForOp> forOps;
    module.walk([&](mlir::func::FuncOp f) {
      f.walk([&](scf::ForOp forOp) {
        if (llvm::any_of(forOp.getResultTypes(),
                         [](Type t) { return isa<airrt::EventType>(t); }))
          forOps.push_back(forOp);
      });
    });
    for (auto forOp : llvm::reverse(forOps)) {
      OpBuilder builder(forOp);
      auto eventTy = airrt::EventType::get(forOp->getContext());
      SmallVector<Value> keptInits;
      SmallVector<unsigned> keptIdx;
      for (auto [i, init] : llvm::enumerate(forOp.getInitArgs()))
        if (!isa<airrt::EventType>(init.getType())) {
          keptInits.push_back(init);
          keptIdx.push_back(i);
        }
      auto newFor =
          scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                             forOp.getUpperBound(), forOp.getStep(), keptInits);
      newFor->setAttrs(forOp->getAttrs());
      Block *newBody = newFor.getBody();
      // scf.for materializes an scf.yield in an empty body; the moved body
      // brings its own, with the event operands dropped below.
      newBody->getTerminator()->erase();
      SmallVector<Value> argRepl;
      argRepl.push_back(newFor.getInductionVar());
      builder.setInsertionPointToStart(newBody);
      unsigned kept = 0;
      for (auto regionArg : forOp.getRegionIterArgs())
        argRepl.push_back(
            isa<airrt::EventType>(regionArg.getType())
                ? airrt::WaitAllOp::create(builder, forOp.getLoc(), eventTy,
                                           SmallVector<Value>{})
                      .getResult(0)
                : newFor.getRegionIterArgs()[kept++]);
      newBody->getOperations().splice(newBody->end(),
                                      forOp.getBody()->getOperations());
      for (auto [oldArg, newArg] :
           llvm::zip(forOp.getBody()->getArguments(), argRepl))
        oldArg.replaceAllUsesWith(newArg);
      auto yield = cast<scf::YieldOp>(newBody->getTerminator());
      SmallVector<Value> keptYields;
      for (unsigned i : keptIdx)
        keptYields.push_back(yield.getOperand(i));
      yield->setOperands(keptYields);
      builder.setInsertionPointAfter(newFor);
      kept = 0;
      for (auto res : forOp.getResults())
        res.replaceAllUsesWith(
            isa<airrt::EventType>(res.getType())
                ? airrt::WaitAllOp::create(builder, forOp.getLoc(), eventTy,
                                           SmallVector<Value>{})
                      .getResult(0)
                : newFor.getResult(kept++));
      forOp.erase();
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
    for (unsigned i = 0; i < std::min(ports.size(), (size_t)4); i++) {
      v0 |= (ports[i].second << (i * 8));
      v0 |= (ports[i].first << ((i * 8) + 5));
    }
    auto loc0 = builder.getUnknownLoc();
    auto i32Ty = builder.getI32Type();
    Value addrVal0 = arith::ConstantOp::create(
        builder, loc0, i32Ty, builder.getI32IntegerAttr(offset));
    Value dataVal0 = arith::ConstantOp::create(builder, loc0, i32Ty,
                                               builder.getI32IntegerAttr(v0));
    AIEX::NpuWrite32Op::create(builder, loc0, addrVal0, dataVal0,
                               FlatSymbolRefAttr{}, col, row);
    uint32_t v1 = 0;
    if (ports.size() > 4)
      for (unsigned i = 4; i < std::min(ports.size(), (size_t)8); i++) {
        v1 |= (ports[i].second << ((i - 4) * 8));
        v1 |= (ports[i].first << (((i - 4) * 8) + 5));
      }
    Value addrVal1 = arith::ConstantOp::create(
        builder, loc0, i32Ty, builder.getI32IntegerAttr(offset + 0x4));
    Value dataVal1 = arith::ConstantOp::create(builder, loc0, i32Ty,
                                               builder.getI32IntegerAttr(v1));
    AIEX::NpuWrite32Op::create(builder, loc0, addrVal1, dataVal1,
                               FlatSymbolRefAttr{}, col, row);
  }

  // up to 8 events (up to 64 bits) will be written to the 8 event slots
  // (bytes) at address {col, row, offset}
  void insertNpuWriteTraceEvents(OpBuilder &builder,
                                 SmallVectorImpl<uint32_t> &events,
                                 uint32_t offset, IntegerAttr col,
                                 IntegerAttr row) {
    uint32_t v0 = 0;
    for (unsigned i = 0; i < std::min(events.size(), (size_t)4); i++)
      v0 |= ((events[i] & 0xff) << (i * 8));
    uint32_t v1 = 0;
    if (events.size() > 4)
      for (unsigned i = 4; i < std::min(events.size(), (size_t)8); i++)
        v1 |= ((events[i] & 0xff) << ((i - 4) * 8));

    auto loc1 = builder.getUnknownLoc();
    auto i32Ty1 = builder.getI32Type();
    Value addrV0 = arith::ConstantOp::create(builder, loc1, i32Ty1,
                                             builder.getI32IntegerAttr(offset));
    Value dataV0 = arith::ConstantOp::create(builder, loc1, i32Ty1,
                                             builder.getI32IntegerAttr(v0));
    AIEX::NpuWrite32Op::create(builder, loc1, addrV0, dataV0,
                               FlatSymbolRefAttr{}, col, row);
    Value addrV1 = arith::ConstantOp::create(
        builder, loc1, i32Ty1, builder.getI32IntegerAttr(offset + 0x4));
    Value dataV1 = arith::ConstantOp::create(builder, loc1, i32Ty1,
                                             builder.getI32IntegerAttr(v1));
    AIEX::NpuWrite32Op::create(builder, loc1, addrV1, dataV1,
                               FlatSymbolRefAttr{}, col, row);
  }

  // configure events to monitor
  LogicalResult insertNpuWrite32ForTrace(ModuleOp module, int64_t trace_size,
                                         int64_t trace_offset) {
    // Either container: the rolled path has already turned the control func
    // into its runtime_sequence by the time this runs.
    SmallVector<Operation *> funcOps;
    module.walk([&](mlir::func::FuncOp f) { funcOps.push_back(f); });
    module.walk([&](AIE::RuntimeSequenceOp s) { funcOps.push_back(s); });

    for (auto f : funcOps) {
      OpBuilder builder(f);
      auto d = f->getParentOfType<AIE::DeviceOp>();
      if (!d)
        continue;

      auto &target_model = d.getTargetModel();
      std::map<int, int> chanToIdMap;
      if (f->getRegion(0).empty())
        continue;
      builder.setInsertionPointToStart(&f->getRegion(0).front());
      // Helper: emit NpuWrite32Op from integer address/value literals.
      auto makeNpuWrite32Fn = [&](uint32_t addr, uint32_t val,
                                  IntegerAttr colAttr, IntegerAttr rowAttr) {
        auto loc = builder.getUnknownLoc();
        auto ty = builder.getI32Type();
        Value addrV = arith::ConstantOp::create(
            builder, loc, ty, builder.getI32IntegerAttr(addr));
        Value dataV = arith::ConstantOp::create(builder, loc, ty,
                                                builder.getI32IntegerAttr(val));
        AIEX::NpuWrite32Op::create(builder, loc, addrV, dataV,
                                   FlatSymbolRefAttr{}, colAttr, rowAttr);
      };
      for (auto pktFlow : d.getOps<AIE::PacketFlowOp>()) {
        Region &r = pktFlow.getPorts();
        Block &b = r.front();
        int flowID = pktFlow.IDInt();
        AIE::Port sourcePort, destPort;
        AIE::TileOp srcTile, destTile;

        // find all packet flow with trace port as source
        for (Operation &Op : b.getOperations()) {
          if (auto pktSrc = dyn_cast_if_present<AIE::PacketSourceOp>(Op)) {
            srcTile = dyn_cast_if_present<AIE::TileOp>(
                pktSrc.getTile().getDefiningOp());
            sourcePort = pktSrc.port();
          } else if (auto pktDest =
                         dyn_cast_if_present<AIE::PacketDestOp>(Op)) {
            destTile = dyn_cast_if_present<AIE::TileOp>(
                pktDest.getTile().getDefiningOp());
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
          makeNpuWrite32Fn(core_reg_timer_control, core_event_broadcast_15 << 8,
                           col, row);
          makeNpuWrite32Fn(core_reg_trace_control0,
                           core_event_broadcast_15 << 16, col, row);
          makeNpuWrite32Fn(core_reg_trace_control1, pkt_type << 12 | flowID,
                           col, row);

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
          makeNpuWrite32Fn(mem_reg_timer_control, mem_event_broadcast_15 << 8,
                           col, row);
          makeNpuWrite32Fn(mem_reg_trace_control0, mem_event_broadcast_15 << 16,
                           col, row);
          makeNpuWrite32Fn(mem_reg_trace_control1, pkt_type << 12 | flowID, col,
                           row);

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
            /* d2_zero_after */ 0, /* burst_length */ 0,
            /* axcache */ nullptr);
        uint32_t addr = (dstColIndex << target_model.getColumnShift()) |
                        (0x1D004 + bdID * 0x20);
        {
          auto patchLoc = builder.getUnknownLoc();
          Value buffOffsetVal =
              arith::ConstantOp::create(builder, patchLoc, builder.getI32Type(),
                                        builder.getI32IntegerAttr(buff_offset));
          AIEX::NpuAddressPatchOp::create(builder, patchLoc, addr,
                                          /* addr_val */ Value(),
                                          /* ddr_id */ 2, buffOffsetVal);
        }

        int address;
        if (destPort.channel == 0)
          address = 0x1D204;
        else if (destPort.channel == 1)
          address = 0x1D20C;
        else {
          pktFlow->emitOpError("unknown trace dest.");
          return failure();
        }
        makeNpuWrite32Fn(
            address, bdID,
            builder.getIntegerAttr(builder.getI32Type(), dstColIndex),
            builder.getIntegerAttr(builder.getI32Type(), dstRowIndex));
        chanToIdMap[dstColIndex]--;
      }

      // broadcast event to sync timer
      auto zero = builder.getIntegerAttr(builder.getI32Type(), 0);
      makeNpuWrite32Fn(0x34000, 127 << 8, zero, zero);
      makeNpuWrite32Fn(0x3404C, 127, zero, zero);
      makeNpuWrite32Fn(0x34008, 127, zero, zero);
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
      auto dma = dyn_cast_if_present<AIEX::NpuDmaMemcpyNdOp>(op);
      auto sync = dyn_cast_if_present<AIEX::NpuSyncOp>(op);
      auto wait = dyn_cast_if_present<AIEX::NpuDmaWaitOp>(op);
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
          col = getColFromTileValue(infoOp.getTile());
        } else if (auto objFifoCreateOp = getObjectFifoCreateOpForSymbol(
                       objectFifoCreateOps,
                       dma.getMetadata().getLeafReference().getValue())) {
          if (isShimTileValue(objFifoCreateOp->getProducerTile()))
            col = getColFromTileValue(objFifoCreateOp->getProducerTile());
          for (auto consumerTileOp : objFifoCreateOp->getConsumerTiles()) {
            if (isShimTileValue(consumerTileOp))
              col = getColFromTileValue(consumerTileOp);
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
