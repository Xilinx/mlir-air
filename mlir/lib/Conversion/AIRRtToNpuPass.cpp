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

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
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
using namespace xilinx;
using namespace xilinx::airrt;

namespace {
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
    (void)rewriter.create<memref::AssumeAlignmentOp>(
        assumeOp.getLoc(), castConsumerOp->getResult(0),
        assumeOp.getAlignment());

    // Erase the old AssumeAlignmentOp
    rewriter.eraseOp(assumeOp);

    return success();
  }
};

struct DmaToNpuPattern : public OpConversionPattern<DmaMemcpyNdOp> {
  using OpConversionPattern<DmaMemcpyNdOp>::OpConversionPattern;

  DmaToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<DmaMemcpyNdOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto idOp = adaptor.getOperands().front();
    uint64_t idInt = 0;
    if (auto const_int = getConstantIntValue(idOp))
      idInt = *const_int;
    else
      return failure();
    uint64_t xInt = 0;
    if (auto const_int = getConstantIntValue(adaptor.getX()))
      xInt = *const_int;
    else
      return failure();
    uint64_t yInt = 0;
    if (auto const_int = getConstantIntValue(adaptor.getY()))
      yInt = *const_int;
    else
      return failure();

    Value memref = adaptor.getMemref();
    BaseMemRefType memrefTy = cast<BaseMemRefType>(memref.getType());
    unsigned int bitwidth = memrefTy.getElementTypeBitWidth();
    if (bitwidth != 32 && bitwidth != 16 && bitwidth != 8)
      return failure();

    SmallVector<Value> offsets;
    SmallVector<int64_t> staticOffsets;
    if (auto const_int = getConstantIntValue(adaptor.getOffset3()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset3());
    if (auto const_int = getConstantIntValue(adaptor.getOffset2()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset2());
    if (auto const_int = getConstantIntValue(adaptor.getOffset1()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset1());
    if (auto const_int = getConstantIntValue(adaptor.getOffset0()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset0());
    SmallVector<Value> sizes;
    SmallVector<int64_t> staticSizes;
    if (auto const_int = getConstantIntValue(adaptor.getLength3()))
      staticSizes.push_back(*const_int);
    else
      sizes.push_back(adaptor.getLength3());
    if (auto const_int = getConstantIntValue(adaptor.getLength2()))
      staticSizes.push_back(*const_int);
    else
      sizes.push_back(adaptor.getLength2());
    if (auto const_int = getConstantIntValue(adaptor.getLength1()))
      staticSizes.push_back(*const_int);
    else
      sizes.push_back(adaptor.getLength1());
    if (auto const_int = getConstantIntValue(adaptor.getLength0()))
      staticSizes.push_back(std::max((int64_t)1, *const_int));
    else
      sizes.push_back(adaptor.getLength0());
    SmallVector<Value> strides;
    SmallVector<int64_t> staticStrides;
    if (auto const_int = getConstantIntValue(adaptor.getStride3()))
      staticStrides.push_back(*const_int);
    else
      strides.push_back(adaptor.getStride3());
    if (auto const_int = getConstantIntValue(adaptor.getStride2()))
      staticStrides.push_back(*const_int);
    else
      strides.push_back(adaptor.getStride2());
    if (auto const_int = getConstantIntValue(adaptor.getStride1()))
      staticStrides.push_back(*const_int);
    else
      strides.push_back(adaptor.getStride1());
    staticStrides.push_back(1);

    StringRef metadata;
    if (op->hasAttr("metadata"))
      metadata =
          op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata").getValue();
    else
      metadata =
          FlatSymbolRefAttr::get(op->getContext(),
                                 rewriter.getStringAttr("MetadataNotFound"))
              .getValue();

    AIE::PacketInfoAttr packet =
        op->getAttrOfType<AIE::PacketInfoAttr>("packet");
    rewriter.replaceOpWithNewOp<AIEX::NpuDmaMemcpyNdOp>(
        op, xInt, yInt, memref, offsets, sizes, strides, staticOffsets,
        staticSizes, staticStrides, packet, metadata, idInt);

    return success();
  }
};

struct HerdLoadToNpuPattern : public OpConversionPattern<HerdLoadOp> {
  using OpConversionPattern<HerdLoadOp>::OpConversionPattern;

  HerdLoadToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<HerdLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(HerdLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct SegmentLoadToNpuPattern : public OpConversionPattern<SegmentLoadOp> {
  using OpConversionPattern<SegmentLoadOp>::OpConversionPattern;

  SegmentLoadToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<SegmentLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(SegmentLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ModuleMetadataToNpuPattern
    : public OpConversionPattern<ModuleMetadataOp> {
  using OpConversionPattern<ModuleMetadataOp>::OpConversionPattern;

  ModuleMetadataToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<ModuleMetadataOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(ModuleMetadataOp op, OpAdaptor adaptor,
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

// Convert FuncOp control function into aiex.runtime_sequence op.
// Functions are converted if they are not external, are inside an aie.device
// and contain aiex.npu.* ops
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

    bool contains_npu_ops = false;
    op.walk([&](Operation *o) {
      if (o->getName().getStringRef().starts_with("aiex.npu."))
        contains_npu_ops = true;
    });
    if (!contains_npu_ops)
      return failure();

    auto seq = rewriter.create<AIEX::RuntimeSequenceOp>(op->getLoc(),
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

AIE::DeviceOp getDeviceForSegmentLoad(Operation *s) {
  auto module = s->getParentOfType<ModuleOp>();

  // Use the airrt metadata to lookup the segment associated with each head
  // or segment load operation.
  for (auto d : module.getOps<AIE::DeviceOp>()) {
    if (s->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()) ==
        d->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      return d;
  }
  return nullptr;
}

// Splits an Affine for loop into two for loops, by hoisting target operations
// in for loop to a new for loop located at the same scope.
void hoistTargetOpsToNewAffineFor(OpBuilder builder, affine::AffineForOp for_op,
                                  llvm::SetVector<Operation *> target_ops) {
  // Get loop nest
  SmallVector<affine::AffineForOp> for_loops;
  affine::AffineForOp parent_for =
      target_ops[0]->getParentOfType<affine::AffineForOp>();
  while (parent_for != for_op) {
    for_loops.push_back(parent_for);
    parent_for = parent_for->getParentOfType<affine::AffineForOp>();
  }
  for_loops.push_back(for_op);

  // Clone loop nest
  builder.setInsertionPoint(for_op);
  IRMapping remap;
  for (int i = for_loops.size() - 1; i >= 0; i--) {
    auto new_for_op = builder.create<affine::AffineForOp>(
        for_loops[i].getLoc(), for_loops[i].getConstantLowerBound(),
        for_loops[i].getConstantUpperBound());
    remap.map(for_loops[i].getInductionVar(), new_for_op.getInductionVar());
    builder.setInsertionPointToStart(new_for_op.getBody());
    // Bottom of rabbit hole
    if (i == 0) {
      for (auto op : target_ops) {
        builder.clone(*op, remap);
      }
    }
  }
}

// Build up a queue of aiex::DmaMemcpyNdOps per shim alloc metadata (i.e. per
// shim dma channel).
void identifyTargetAffineForAndOps(
    func::FuncOp f,
    llvm::MapVector<affine::AffineForOp,
                    llvm::MapVector<StringRef, llvm::SetVector<Operation *>>>
        &opQueuePerMeta) {
  // Ensure memcpy ops operating on the same metadata (i.e. the same shim
  // dma channel) are hoisted together, to maintain data dependency.
  for (auto for_op : f.getBody().getOps<affine::AffineForOp>()) {
    // Get for_op's immediate child op
    for_op.walk([&](airrt::DmaMemcpyNdOp memcpyOp) {
      StringRef metadata =
          memcpyOp->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")
              .getValue();
      // Check if any operand's defining ops needs to be hoisted together.
      SmallVector<Operation *> oper_def_ops;
      xilinx::air::getDefiningOpsToOperands(memcpyOp.getOperation(),
                                            oper_def_ops);
      for (auto o : oper_def_ops)
        if (o->getParentRegion() == memcpyOp->getParentRegion())
          opQueuePerMeta[for_op][metadata].insert(o);
      opQueuePerMeta[for_op][metadata].insert(memcpyOp.getOperation());
    });
  }
}

// Split affine.for loops to ensure that each loop nest only contains memcpy ops
// onto a single shim dma channel.
void isolateAIRRtDmaLoopNests(ModuleOp module) {
  // Identify affine.for ops and target child ops for hoisting.
  llvm::MapVector<affine::AffineForOp,
                  llvm::MapVector<StringRef, llvm::SetVector<Operation *>>>
      opQueuePerMeta;
  SmallVector<func::FuncOp> funcOps;
  module.walk([&](func::FuncOp f) { funcOps.push_back(f); });
  for (auto f : funcOps) {
    f.walk(
        [&](affine::AffineForOp afo) { (void)promoteIfSingleIteration(afo); });
    identifyTargetAffineForAndOps(f, opQueuePerMeta);
  }

  // Hoist ops out of each scf.for.
  llvm::SmallSet<Operation *, 1> erased;
  for (auto &[affineForBand, queue] : opQueuePerMeta) {
    for (auto &[metadata, vec] : queue) {
      OpBuilder builder(affineForBand);
      hoistTargetOpsToNewAffineFor(builder, affineForBand, vec);
      erased.insert(affineForBand.getOperation());
    }
  }
  for (auto o : erased)
    o->erase();
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
  assert(max > 0 && "No factors less than or equal to 0 exist");

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
      // descending in this loop. So we can return immediately if highFactor is
      // good.
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
  strides.push_back(builder.create<arith::ConstantOp>(
      loc, builder.getI64Type(), IntegerAttr::get(builder.getI64Type(), 1)));

  for (int i = wraps.size() - 1; i >= 0; i--) {
    auto const_wrap = *getConstantIntValue(wraps[i]);
    auto const_stride = *getConstantIntValue(strides[i]);
    if (const_wrap >= AIE2_WRAP_UPPER_BOUNDS[i]) {
      // Found dimension with illegal wrap. Tiling. (Prefers smaller outer wrap
      // values, as long as stride fits)
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
      wraps[i] = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          IntegerAttr::get(builder.getI64Type(), inner_wrap));
      wraps.insert(wraps.begin() + i,
                   builder.create<arith::ConstantOp>(
                       loc, builder.getI64Type(),
                       IntegerAttr::get(builder.getI64Type(), outer_wrap)));
      auto new_const_stride = const_stride * inner_wrap;
      if (volume != 1)
        new_const_stride %=
            volume; // Avoids striding out of memory size, if memref is ranked
      strides.insert(
          strides.begin() + i,
          builder.create<arith::ConstantOp>(
              loc, builder.getI64Type(),
              IntegerAttr::get(builder.getI64Type(), new_const_stride)));
      offsets.insert(offsets.begin() + i,
                     builder.create<arith::ConstantOp>(
                         loc, builder.getI64Type(),
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
            ? (builder.create<affine::AffineForOp>(
                  loc,
                  SmallVector<Value>{builder.create<arith::AddIOp>(
                      loc, inner_affine_for_iv,
                      builder.create<arith::ConstantIndexOp>(
                          loc, const_lower_bound))},
                  AffineMap::get(ctx),
                  SmallVector<Value>{builder.create<arith::AddIOp>(
                      loc, inner_affine_for_iv,
                      builder.create<arith::ConstantIndexOp>(
                          loc, const_upper_bound))},
                  AffineMap::get(ctx), const_step))
            : (builder.create<affine::AffineForOp>(
                  loc, const_lower_bound, const_upper_bound, const_step));
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
    auto new_inner_offset = builder.create<arith::IndexCastOp>(
        loc, IntegerType::get(ctx, 64), inner_affine_for_iv);
    new_opers.push_back(new_inner_offset);
  } else
    new_opers.insert(new_opers.end(), offsets.begin(), offsets.end());
  new_opers.insert(new_opers.end(), wraps.begin(), wraps.end());
  new_opers.insert(new_opers.end(), strides.begin(), strides.end());
  builder.create<airrt::DmaMemcpyNdOp>(loc, tys, new_opers,
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

struct AIRSpecializeAIRRtDmaWrapAndStrideInAffineFor
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp for_op,
                                PatternRewriter &rewriter) const override {
    auto loc = for_op->getLoc();
    auto ctx = for_op->getContext();

    // Declaration of constants
    auto i64Ty = rewriter.getI64Type();
    auto i64_zero = rewriter.create<arith::ConstantOp>(
        loc, i64Ty, IntegerAttr::get(i64Ty, 0));
    auto i64_one = rewriter.create<arith::ConstantOp>(
        loc, i64Ty, IntegerAttr::get(i64Ty, 1));

    if (!air::hasNImpureOps(for_op.getBody(), 1))
      return failure();

    // Check if the loop contains exactly one memcpy op
    if (llvm::range_size(for_op.getBody()->getOps<airrt::DmaMemcpyNdOp>()) != 1)
      return failure();
    airrt::DmaMemcpyNdOp memcpy_op =
        *(for_op.getBody()->getOps<airrt::DmaMemcpyNdOp>().begin());

    // Fold for loops into memcpy op's wrap and stride fields
    auto memref = memcpy_op->getOperand(3);
    auto memref_shape = xilinx::air::getTensorShape(memref.getType());
    auto oper_begin = memcpy_op.getOperands().begin();
    SmallVector<Value> offsets(oper_begin + 4, oper_begin + 8);
    SmallVector<Value> wraps(oper_begin + 8, oper_begin + 12);
    SmallVector<Value> strides(oper_begin + 12, oper_begin + 15);
    // Stride field implicit last element one
    strides.push_back(i64_one);

    (void)air::canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(memref.getType()));

    // If empty offsets/sizes/strides, then populate the lists with default
    // values.
    if (offsets.empty() && wraps.empty() && strides.empty())
      air::populateDefaultWrapsAndStrides(rewriter, memref, offsets, wraps,
                                          strides);

    auto res = air::foldForLoopNestAsExtendedSizesAndStrides(
        rewriter, for_op.getOperation(), memcpy_op.getOperation(), offsets,
        wraps, strides, memref);
    if (res.failed())
      return failure();

    if (offsets.size() > 4 || wraps.size() > 4 || strides.size() > 4)
      return failure();

    (void)air::canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(memref.getType()));

    // Stride field implicit last element one
    strides.pop_back();
    while (offsets.size() < 4) {
      offsets.insert(offsets.begin(), i64_zero);
    }
    while (wraps.size() < 4) {
      wraps.insert(wraps.begin(), i64_one);
    }
    while (strides.size() < 3) {
      strides.insert(strides.begin(), i64_zero);
    }

    // Stride = 0 means repeat that dimension. If highest dimension (dim 0) is
    // not used, then move the repeat dimension to dim 0, which is the only dim
    // with repeat capability. Else, fall back to unrolling BDs.
    unsigned activeDimsInBetween = 0;
    for (unsigned i = 1; i < strides.size(); i++) {
      auto constWrap = mlir::getConstantIntValue(wraps[i]);
      auto constStride = mlir::getConstantIntValue(strides[i]);
      if (!constWrap)
        continue;
      if (!constStride)
        continue;
      if (*constWrap <= 1)
        continue; // Inactive dimension. Continue.
      if (*constStride) {
        // Found active dimension after dim 0. Any subsequent repeat dimension
        // shall not bump to dim 0 anymore.
        activeDimsInBetween++;
        continue;
      }
      // This is a repeat dimension.
      if (mlir::getConstantIntValue(wraps[0]) &&
          *mlir::getConstantIntValue(wraps[0]) == 1 && !activeDimsInBetween) {
        // Dimension 0 is available. Move the repeat dimension i to dimension 0.
        auto tmp = wraps[0];
        wraps[0] = wraps[i];
        wraps[i] = tmp;
        tmp = strides[0];
        strides[0] = strides[i];
        strides[i] = tmp;
        tmp = offsets[0];
        offsets[0] = offsets[i];
        offsets[i] = tmp;
      } else {
        (void)loopUnrollFull(for_op);
        return success();
      }
    }

    // Create new airrt.dma_memcpy_nd
    SmallVector<Type, 1> tys;
    if (memcpy_op->getNumResults())
      tys.push_back(airrt::EventType::get(ctx));

    SmallVector<Value, 16> opers;
    auto old_opers = memcpy_op->getOperands();
    opers.insert(opers.end(), old_opers.begin(), old_opers.begin() + 4);
    opers[1] = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                  IntegerAttr::get(i64Ty, 0));
    opers[2] = rewriter.create<arith::ConstantOp>(loc, i64Ty,
                                                  IntegerAttr::get(i64Ty, 0));
    opers.insert(opers.end(), offsets.begin(), offsets.end());
    opers.insert(opers.end(), wraps.begin(), wraps.end());
    opers.insert(opers.end(), strides.begin(), strides.end());

    // Hoist const ops; create index_cast ops.
    IRMapping remap;
    for (unsigned i = 0; i < opers.size(); i++) {
      auto defOp = opers[i].getDefiningOp<arith::ConstantOp>();
      if (!defOp)
        continue;
      if (opers[i].getType() == memcpy_op->getOperandTypes()[i])
        continue;
      opers[i] = rewriter.clone(*defOp, remap)->getResult(0);
      opers[i] = getValueOrCreateCastToIndexLike(
          rewriter, loc, memcpy_op->getOperandTypes()[i], opers[i]);
    }

    // Hoist any pure ops that the new channel op depends on.
    llvm::SetVector<Operation *> backwardSlices;
    air::getBackwardSliceInRegion(rewriter, &for_op.getRegion(), opers,
                                  backwardSlices);
    for (auto o : backwardSlices)
      rewriter.clone(*o, remap);

    auto new_dma = rewriter.create<airrt::DmaMemcpyNdOp>(
        loc, tys, air::lookupOrDefaultRange(opers, remap));
    new_dma->setAttrs(memcpy_op->getDiscardableAttrDictionary());

    rewriter.replaceAllUsesWith(for_op.getInductionVar(),
                                rewriter.create<arith::ConstantIndexOp>(
                                    loc, for_op.getConstantLowerBound()));
    rewriter.eraseOp(for_op.getOperation());

    return success();
  }

private:
};

struct AIRRtToNpuPass : public impl::AIRRtToNpuBase<AIRRtToNpuPass> {
  void runOnOperation() override {

    ModuleOp module = getOperation();

    // Move func op to the end of device op's body
    moveFuncOpToEndOfDeviceOp(module);

    // Purge all wait all ops
    purgeSCFParContainingOnlyWaitAllOps(module);
    purgeWaitAlls(module);

    // Purge airrt.dma x and y fields, as they are obsolete for AIE2.
    purgeAIRRtDmaXAndY(module);

    // Remove any duplicate shim dma allocations
    purgeDuplicateShimDmaAllocs(module);

    // Separate affine for loop nest into loop nests each containing one dma
    // memcpy op
    unrollSCFFors(module);
    isolateAIRRtDmaLoopNests(module);

    // Simplify affine apply ops
    auto ctx = &getContext();
    RewritePatternSet canoPatterns_0(ctx);
    xilinx::air::populateAIRLoopIndexCanonicalizationPatterns(canoPatterns_0);
    (void)applyPatternsGreedily(module, std::move(canoPatterns_0));

    // Specialize affine for loop nest into wraps and strides
    RewritePatternSet loopFoldPattern(ctx);
    loopFoldPattern.add<AIRSpecializeAIRRtDmaWrapAndStrideInAffineFor>(ctx);
    air::populateAIRLoopIndexCanonicalizationPatterns(loopFoldPattern);
    (void)applyPatternsGreedily(module, std::move(loopFoldPattern));
    unrollAffineFors(module);

    // Simplify arith ops (from airrt)
    RewritePatternSet canoPatterns_1(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns_1, ctx);
    (void)applyPatternsGreedily(module, std::move(canoPatterns_1));

    // Purge all wait ops again after unroll, in case there were loop carried
    // events which couldn't be purged before
    purgeWaitAlls(module);

    // Purge dma ops' async tokens
    purgeDmaAsyncTokens(module);

    // Enforce AIE2 hardware constraint: wrap size limit within [0, 1023].
    enforceAIE2WrapLimit(module);

    // Simplify arith ops (from airrt)
    RewritePatternSet canoPatterns_3(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns_3, ctx);
    (void)applyPatternsGreedily(module, std::move(canoPatterns_3));

    ConversionTarget target(getContext());
    target.addIllegalDialect<AIRRtDialect>();
    target.addLegalDialect<arith::ArithDialect, AIEX::AIEXDialect,
                           memref::MemRefDialect>();
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

    // Insert sync op after copying data out to host
    insertNpuSyncOpForResults(module);

    // Renumber npu dma ops
    renumberNpuDmaOps(module.getBody());

    // Configure the tile trace units and the shimDMA
    if (clTraceSize > 0)
      insertNpuWrite32ForTrace(module, clTraceSize, clTraceOffset);

    RewritePatternSet funcToSeqPatterns(ctx);
    funcToSeqPatterns.add<ControlFuncConversion>(ctx);

    if (failed(applyPartialConversion(module, target,
                                      std::move(funcToSeqPatterns))))
      signalPassFailure();
  }

  void moveFuncOpToEndOfDeviceOp(ModuleOp module) {
    // Move func op to the end of device op's body
    SmallVector<Operation *> segs;
    module.walk([&](Operation *o) {
      if (isa<SegmentLoadOp, HerdLoadOp>(o)) {
        segs.push_back(o);
      }
    });
    for (auto s : segs) {
      auto f = s->getParentOfType<func::FuncOp>();
      auto d = getDeviceForSegmentLoad(s);
      if (!f || !d)
        continue;
      f->moveBefore(d.getBody()->getTerminator());
    }
  }

  void purgeDmaAsyncTokens(ModuleOp module) {
    SmallVector<DmaMemcpyNdOp> dmas;
    module.walk([&](DmaMemcpyNdOp dma) { dmas.push_back(dma); });
    for (auto dma : dmas) {
      if (dma->getNumResults()) {
        OpBuilder builder(dma);
        SmallVector<Type, 1> tys = {};
        auto newOp = builder.create<DmaMemcpyNdOp>(dma->getLoc(), tys,
                                                   dma->getOperands());
        newOp->setAttrs(dma->getDiscardableAttrDictionary());
        dma->erase();
      }
    }
  }

  void purgeWaitAlls(ModuleOp module) {
    int size = 0;
    int last_size = 1;
    while (size < last_size) {
      SmallVector<WaitAllOp> waits;
      module.walk([&](WaitAllOp w) { waits.push_back(w); });
      size = waits.size();
      last_size = size;
      for (auto &w : waits) {
        if (!w->use_empty())
          continue;
        w->eraseOperands(0, w->getNumOperands());
        w.erase();
        size--;
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
        dma.getXMutable().assign(builder.create<arith::ConstantOp>(
            dma->getLoc(), i64Ty, IntegerAttr::get(i64Ty, 0)));
      if (resetY)
        dma.getYMutable().assign(builder.create<arith::ConstantOp>(
            dma->getLoc(), i64Ty, IntegerAttr::get(i64Ty, 0)));
    }
  }

  void purgeDuplicateShimDmaAllocs(ModuleOp module) {
    llvm::SetVector<AIE::ShimDMAAllocationOp> allocs;
    module.walk([&](AIE::ShimDMAAllocationOp alloc) { allocs.insert(alloc); });
    llvm::SmallSet<AIE::ShimDMAAllocationOp, 1> uniqueAllocs;

    // Map each unique set of <dir, chan, col> to a shim dma alloc op
    DenseMap<StringRef, StringRef> uniqueAllocMap;
    for (auto alloc : allocs) {
      std::tuple<bool, int, int> allocInfo = {
          alloc.getChannelDir() == AIE::DMAChannelDir::MM2S,
          alloc.getChannelIndex(), alloc.getCol()};

      auto it =
          llvm::find_if(uniqueAllocs, [&](AIE::ShimDMAAllocationOp ualloc) {
            std::tuple<bool, int, int> uallocInfo = {
                ualloc.getChannelDir() == AIE::DMAChannelDir::MM2S,
                ualloc.getChannelIndex(), ualloc.getCol()};
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

    // Replace all uses of metadata to unique
    module.walk([&](DmaMemcpyNdOp dma) {
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
      (void)loopUnrollFull(afo);
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
        int64_t tripCount = llvm::divideCeilSigned(
            ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
        (void)loopUnrollByFactor(for_op, tripCount);
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
      auto newWaitAll = builder.create<airrt::WaitAllOp>(
          par_op->getLoc(), airrt::EventType::get(par_op->getContext()),
          par_op.getInitVals());
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

  void insertNpuSyncOpForResults(ModuleOp module) {
    SmallVector<mlir::func::FuncOp> funcOps;
    module.walk([&](mlir::func::FuncOp f) { funcOps.push_back(f); });
    for (auto f : funcOps) {
      SmallVector<AIEX::NpuDmaMemcpyNdOp> dmas;
      f.walk([&](AIEX::NpuDmaMemcpyNdOp dma) { dmas.push_back(dma); });
      auto d = f->getParentOfType<AIE::DeviceOp>();

      SmallVector<AIE::ShimDMAAllocationOp> shimDmaAllocOps;
      if (d)
        d.walk([&](AIE::ShimDMAAllocationOp shimDmaAllocOp) {
          shimDmaAllocOps.push_back(shimDmaAllocOp);
        });
      // Performance optimization: instead of repeating calls to
      // getAllocOpForSymbol with the same symbol name, cache the result of the
      // first call and use the cache for subsequent calls. This dramatically
      // improves compile time for some designs.
      llvm::DenseMap<StringRef, std::optional<AIE::ShimDMAAllocationOp>>
          allocationCache;
      auto getAllocOpForSymbolWithCaching = [&](StringRef sym_name) {
        auto iter = allocationCache.find(sym_name);
        if (iter != allocationCache.end()) {
          return iter->second;
        }
        auto infaOp = getAllocOpForSymbol(shimDmaAllocOps, sym_name);
        allocationCache.insert({sym_name, infaOp});
        return infaOp;
      };

      if (!d)
        continue;
      OpBuilder builder(f);
      for (auto dma : dmas) {
        auto infoOp = getAllocOpForSymbolWithCaching(dma.getMetadata());
        if (!infoOp)
          continue;
        if (infoOp->getChannelDir() != AIE::DMAChannelDir::S2MM)
          continue;
        // Found dma op copying results to host
        auto col = builder.getI32IntegerAttr(infoOp->getCol());
        auto row = builder.getI32IntegerAttr(0);
        auto dir = builder.getI32IntegerAttr(0);
        auto chan = builder.getI32IntegerAttr(infoOp->getChannelIndex());
        auto col_num = builder.getI32IntegerAttr(1);
        auto row_num = builder.getI32IntegerAttr(1);
        builder.setInsertionPointAfter(dma);
        builder.create<AIEX::NpuSyncOp>(dma->getLoc(), col, row, dir, chan,
                                        col_num, row_num);
      }

      // Attempt to make npu.sync ops contiguous if they are not operating on
      // the same channel.
      SmallVector<AIEX::NpuSyncOp> previsouSyncs;
      f.walk([&](Operation *op) {
        if (auto sync = dyn_cast<AIEX::NpuSyncOp>(op))
          previsouSyncs.push_back(sync);
        else if (auto dma = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
          auto infoOp = getAllocOpForSymbolWithCaching(dma.getMetadata());
          if (!infoOp)
            return;
          if (previsouSyncs.empty())
            return;
          if (infoOp->getChannelDir() == AIE::DMAChannelDir::S2MM) {
            for (auto prevSync : previsouSyncs)
              prevSync->moveAfter(op);
          } else if (infoOp->getChannelDir() == AIE::DMAChannelDir::MM2S) {
            previsouSyncs.clear();
          }
        }
      });
    }
  }

  // Each element of 'port' is a {Port_N_Master_Slave, Port_N_ID} pair. They
  // will be read sequentially to select up to 8 stream switch ports to monitor,
  // using the select register at address {col, row, offset}.
  void insertNpuWriteStreamSwitchEventSel(
      OpBuilder &builder, std::vector<std::pair<uint8_t, uint8_t>> &ports,
      uint32_t offset, IntegerAttr col, IntegerAttr row) {
    uint32_t v0 = 0;
    for (unsigned i = 0; i < std::min(ports.size(), 4UL); i++) {
      v0 |= (ports[i].second << (i * 8));
      v0 |= (ports[i].first << ((i * 8) + 5));
    }
    builder.create<AIEX::NpuWrite32Op>(builder.getUnknownLoc(), offset, v0,
                                       nullptr, col, row);
    uint32_t v1 = 0;
    if (ports.size() > 4)
      for (unsigned i = 4; i < std::min(ports.size(), 8UL); i++) {
        v1 |= (ports[i].second << ((i - 4) * 8));
        v1 |= (ports[i].first << (((i - 4) * 8) + 5));
      }
    builder.create<AIEX::NpuWrite32Op>(builder.getUnknownLoc(), offset + 0x4,
                                       v1, nullptr, col, row);
  }

  // up to 8 events (up to 64 bits) will be written to the 8 event slots (bytes)
  // at address {col, row, offset}
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

    builder.create<AIEX::NpuWrite32Op>(builder.getUnknownLoc(), offset, v0,
                                       nullptr, col, row);
    builder.create<AIEX::NpuWrite32Op>(builder.getUnknownLoc(), offset + 0x4,
                                       v1, nullptr, col, row);
  }

  // configure events to monitor
  void insertNpuWrite32ForTrace(ModuleOp module, int64_t trace_size,
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
          return;
        }
        if (!target_model.isShimNOCTile(dstColIndex, dstRowIndex)) {
          pktFlow->emitOpError("unsupported trace dest.");
          return;
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
          builder.create<AIEX::NpuWrite32Op>(
              builder.getUnknownLoc(), core_reg_timer_control,
              core_event_broadcast_15 << 8, nullptr, col, row);
          builder.create<AIEX::NpuWrite32Op>(
              builder.getUnknownLoc(), core_reg_trace_control0,
              core_event_broadcast_15 << 16, nullptr, col, row);
          builder.create<AIEX::NpuWrite32Op>(
              builder.getUnknownLoc(), core_reg_trace_control1,
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
          builder.create<AIEX::NpuWrite32Op>(
              builder.getUnknownLoc(), mem_reg_timer_control,
              mem_event_broadcast_15 << 8, nullptr, col, row);
          builder.create<AIEX::NpuWrite32Op>(
              builder.getUnknownLoc(), mem_reg_trace_control0,
              mem_event_broadcast_15 << 16, nullptr, col, row);
          builder.create<AIEX::NpuWrite32Op>(
              builder.getUnknownLoc(), mem_reg_trace_control1,
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
          return;
        }

        builder.create<AIEX::NpuWriteBdOp>(
            builder.getUnknownLoc(), dstColIndex, bdID, buff_size, buff_offset,
            /*enable_packet*/ 1, /*out_of_order_id*/ 0,
            /*packet_id*/ flowID, pkt_type,
            /* d0_size */ 0, /* d0_stride */ 0, /* d1_size */ 0,
            /* d1_stride */ 0, /* d2_size */ 0, /* d2_stride */ 0,
            /* iteration_current */ 0, /* iteration_size */ 0,
            /* iteration_stride */ 0, /* next_bd */ 0, dstRowIndex,
            /* use_next_bd */ 0,
            /* valid_bd */ 1, /* lock_rel_val */ 0, /* lock_rel_id */ 0,
            /* lock_acq_enable */ 0, /* lock_acq_val */ 0, /* lock_acq_id */ 0,
            /* d0_zero_before */ 0, /* d1_zero_before */ 0,
            /* d2_zero_before */ 0,
            /* d0_zero_after */ 0, /* d1_zero_after */ 0,
            /* d2_zero_after */ 0);
        uint32_t addr = (dstColIndex << target_model.getColumnShift()) |
                        (0x1D004 + bdID * 0x20);
        builder.create<AIEX::NpuAddressPatchOp>(builder.getUnknownLoc(), addr,
                                                /* ddr_id */ 2, buff_offset);

        int address;
        if (destPort.channel == 0)
          address = 0x1D204;
        else if (destPort.channel == 1)
          address = 0x1D20C;
        else {
          pktFlow->emitOpError("unknown trace dest.");
          return;
        }
        builder.create<AIEX::NpuWrite32Op>(
            builder.getUnknownLoc(), address, bdID, nullptr,
            builder.getIntegerAttr(builder.getI32Type(), dstColIndex),
            builder.getIntegerAttr(builder.getI32Type(), dstRowIndex));
        chanToIdMap[dstColIndex]--;
      }

      // broadcast event to sync timer
      auto zero = builder.getIntegerAttr(builder.getI32Type(), 0);
      builder.create<AIEX::NpuWrite32Op>(builder.getUnknownLoc(), 0x34000,
                                         127 << 8, nullptr, zero, zero);
      builder.create<AIEX::NpuWrite32Op>(builder.getUnknownLoc(), 0x3404C, 127,
                                         nullptr, zero, zero);
      builder.create<AIEX::NpuWrite32Op>(builder.getUnknownLoc(), 0x34008, 127,
                                         nullptr, zero, zero);
    }
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
    // Performance optimization: instead of repeating calls to
    // getAllocOpForSymbol with the same symbol name, cache the result of the
    // first call and use the cache for subsequent calls. This dramatically
    // improves compile time for some designs.
    llvm::DenseMap<StringRef, std::optional<AIE::ShimDMAAllocationOp>>
        allocationCache;
    auto getAllocOpForSymbolWithCaching = [&](StringRef sym_name) {
      auto iter = allocationCache.find(sym_name);
      if (iter != allocationCache.end()) {
        return iter->second;
      }
      auto infaOp = getAllocOpForSymbol(shimDmaAllocOps, sym_name);
      allocationCache.insert({sym_name, infaOp});
      return infaOp;
    };
    SmallVector<AIE::ObjectFifoCreateOp> objectFifoCreateOps;
    if (d)
      d.walk([&](AIE::ObjectFifoCreateOp objectFifoCreateOp) {
        objectFifoCreateOps.push_back(objectFifoCreateOp);
      });
    OpBuilder builder(blk->getParentOp());
    blk->walk([&](Operation *op) {
      auto dma = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op);
      auto sync = dyn_cast<AIEX::NpuSyncOp>(op);
      if (sync) {
        chanToIdMap.clear();
        return;
      }
      if (!dma)
        return;
      builder.setInsertionPoint(dma);
      int col = -1;
      if (d) {
        if (auto infoOp = getAllocOpForSymbolWithCaching(dma.getMetadata())) {
          col = infoOp->getCol();
        } else if (auto objFifoCreateOp = getObjectFifoCreateOpForSymbol(
                       objectFifoCreateOps, dma.getMetadata())) {
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

} // namespace

namespace xilinx {
namespace airrt {

std::unique_ptr<mlir::Pass> createAIRRtToNpuPass() {
  return std::make_unique<AIRRtToNpuPass>();
}

} // namespace airrt
} // namespace xilinx
