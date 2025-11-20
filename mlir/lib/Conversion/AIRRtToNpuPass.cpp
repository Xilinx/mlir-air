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
    (void)rewriter.create<memref::AssumeAlignmentOp>(
        assumeOp.getLoc(), castConsumerOp->getResult(0),
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

  LogicalResult
  matchAndRewrite(airrt::DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto idOp = adaptor.getOperands().front();
    uint64_t idInt = 0;
    if (auto const_int = getConstantIntValue(idOp))
      idInt = *const_int;
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

    SymbolRefAttr metadata;
    if (op->hasAttr("metadata"))
      metadata = op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata");
    else
      metadata = SymbolRefAttr::get(op->getContext(),
                                    rewriter.getStringAttr("MetadataNotFound"));

    AIE::PacketInfoAttr packet =
        op->getAttrOfType<AIE::PacketInfoAttr>("packet");
    rewriter.replaceOpWithNewOp<AIEX::NpuDmaMemcpyNdOp>(
        op, memref, offsets, sizes, strides, staticOffsets, staticSizes,
        staticStrides, packet, metadata, idInt);

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

    // for each herd core, emit write_rtp ops for every herd operand
    // followed by a write32 to the herd lock, setting it to 1.
    for (int phys_x = loc_x; phys_x < size_x + loc_x; phys_x++) {
      for (int phys_y = loc_y; phys_y < size_y + loc_y; phys_y++) {

        for (int i = 0, e = op.getNumOperands(); i < e; i++) {
          Value oper = adaptor.getOperands()[i];
          if (!llvm::isa<IntegerType, IndexType, FloatType>(oper.getType()))
            continue;

          std::string name = "__air_herd_rtp_" + std::to_string(phys_x) + "_" +
                             std::to_string(phys_y);
          auto constOp =
              dyn_cast_if_present<arith::ConstantOp>(oper.getDefiningOp());
          if (!constOp)
            continue;
          uint32_t v = cast<IntegerAttr>(constOp.getValue()).getInt();
          rewriter.create<AIEX::NpuWriteRTPOp>(op.getLoc(), name, i, v);
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

        rewriter.create<AIEX::SetLockOp>(op.getLoc(), lockOp.getResult(),
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

    auto seq = rewriter.create<AIE::RuntimeSequenceOp>(op->getLoc(),
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

struct AIRRtWaitAllOpToNpuWaitPattern
    : public OpRewritePattern<airrt::WaitAllOp> {
public:
  using OpRewritePattern<airrt::WaitAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(airrt::WaitAllOp op,
                                PatternRewriter &rewriter) const override {
    if (llvm::none_of(op->getOperands(), [](Value oper) {
          return (bool)oper.getDefiningOp<airrt::DmaMemcpyNdOp>();
        }))
      return failure();
    for (auto oper : op->getOperands()) {
      auto airrtDmaOp = oper.getDefiningOp<airrt::DmaMemcpyNdOp>();
      if (!airrtDmaOp)
        continue;
      StringRef metadata =
          airrtDmaOp->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")
              .getValue();
      rewriter.create<AIEX::NpuDmaWaitOp>(op.getLoc(), metadata);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
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

struct AIRRtToNpuPass : public impl::AIRRtToNpuBase<AIRRtToNpuPass> {
  void runOnOperation() override {

    ModuleOp module = getOperation();

    // Move func op to the end of device op's body
    moveFuncOpToEndOfDeviceOp(module);

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

    // Purge dma ops' async tokens
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

    // Optimization: purge npu wait on device inbound shim data movements.
    removeNpuWaitOnInboundMemcpy(module);

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
  }

  void moveFuncOpToEndOfDeviceOp(ModuleOp module) {
    // Move func op to the end of device op's body
    SmallVector<Operation *> segs;
    module.walk([&](Operation *o) {
      if (isa<airrt::SegmentLoadOp, airrt::HerdLoadOp>(o)) {
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

  // Generate npu wait ops from blocking airrt.wait_all ops.
  void generateNpuWaitFromAIRRtWaitAll(ModuleOp module) {

    // Canonicalize airrt.wait_all, to remove redundant ops.
    auto ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    airrt::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
    patterns.insert<AIRRtWaitAllOpToNpuWaitPattern>(ctx);
    (void)applyPatternsGreedily(module, std::move(patterns));

    // Dma event tokens are no longer needed. Purge them.
    SmallVector<airrt::DmaMemcpyNdOp> dmas;
    module.walk([&](airrt::DmaMemcpyNdOp dma) { dmas.push_back(dma); });
    for (auto dma : dmas) {
      if (dma->getNumResults()) {
        OpBuilder builder(dma);
        SmallVector<Type, 1> tys;
        auto newOp = builder.create<airrt::DmaMemcpyNdOp>(dma->getLoc(), tys,
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
    module.walk([&](airrt::DmaMemcpyNdOp dma) {
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

  // Remove npu wait op on inbound dma data movements.
  // TODO: this is an aggressive optimization which might prove problematic for
  // some applications. To be revised.
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
          return failure();
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
          return failure();
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
          col = infoOp.getCol();
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
