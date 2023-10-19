//===- AIRRtToIpuPass.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRRtToIpuPass.h"
#include "PassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "airrt-to-ipu-pass"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::airrt;

namespace {

struct DmaToIpuPattern : public OpConversionPattern<DmaMemcpyNdOp> {
  using OpConversionPattern<DmaMemcpyNdOp>::OpConversionPattern;

  DmaToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<DmaMemcpyNdOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newOperands;
    for (auto o : adaptor.getOperands().drop_front()) {
      auto t = o.getType();
      if (isa<IntegerType>(t))
        newOperands.push_back(rewriter.create<arith::TruncIOp>(
            op->getLoc(), rewriter.getI32Type(), o));
      else
        newOperands.push_back(o);
    }
    auto idOp = adaptor.getOperands().front();
    auto constOp = idOp.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return failure();
    auto idAttr = constOp.getValue();
    auto newOp = rewriter.replaceOpWithNewOp<AIEX::IpuDmaMemcpyNdOp>(
        op, SmallVector<Type, 1>(), newOperands);
    newOp->setAttr("id", idAttr);
    newOp->setAttr("metadata",
                   op->getAttrOfType<mlir::SymbolRefAttr>("metadata"));
    return success();
  }
};

struct HerdLoadToIpuPattern : public OpConversionPattern<HerdLoadOp> {
  using OpConversionPattern<HerdLoadOp>::OpConversionPattern;

  HerdLoadToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<HerdLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(HerdLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct SegmentLoadToIpuPattern : public OpConversionPattern<SegmentLoadOp> {
  using OpConversionPattern<SegmentLoadOp>::OpConversionPattern;

  SegmentLoadToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<SegmentLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(SegmentLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ModuleMetadataToIpuPattern
    : public OpConversionPattern<ModuleMetadataOp> {
  using OpConversionPattern<ModuleMetadataOp>::OpConversionPattern;

  ModuleMetadataToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<ModuleMetadataOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(ModuleMetadataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class L1AffineStoreOpConversion : public OpConversionPattern<AffineStoreOp> {
public:
  using OpConversionPattern<AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.getMemref().getType().cast<MemRefType>();
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

    auto memrefTy = op.getMemref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

AIE::DeviceOp getDeviceForSegmentLoad(SegmentLoadOp s) {
  auto module = s->getParentOfType<ModuleOp>();

  // Use the airrt metadata to lookup the segment associated with each head
  // load operation.
  for (auto d : module.getOps<AIE::DeviceOp>()) {
    if (s.getSymName() ==
        d->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      return d;
  }
  return nullptr;
}

struct AIRRtToIpuPass : public air::AIRRtToIpuBase<AIRRtToIpuPass> {
  void runOnOperation() override {

    ModuleOp module = getOperation();

    SmallVector<SegmentLoadOp> segs;
    module.walk([&](SegmentLoadOp s) { segs.push_back(s); });
    for (auto s : segs) {
      auto f = s->getParentOfType<func::FuncOp>();
      auto d = getDeviceForSegmentLoad(s);
      if (!f || !d)
        continue;
      f->moveAfter(&d.getBody()->back());
    }

    ConversionTarget target(getContext());
    target.addIllegalDialect<AIRRtDialect>();
    target.addLegalDialect<arith::ArithDialect, AIEX::AIEXDialect>();

    target.addDynamicallyLegalOp<AffineStoreOp>([&](AffineStoreOp op) {
      if (op->getParentOfType<AIE::CoreOp>())
        return true;
      return (
          op.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
          (int)xilinx::air::MemorySpace::L1);
    });
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      if (op->getParentOfType<AIE::CoreOp>())
        return true;
      return (
          op.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
          (int)xilinx::air::MemorySpace::L1);
    });
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<DmaToIpuPattern, HerdLoadToIpuPattern,
                    SegmentLoadToIpuPattern, ModuleMetadataToIpuPattern,
                    L1MemRefStoreOpConversion, L1AffineStoreOpConversion>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace xilinx {
namespace airrt {

std::unique_ptr<mlir::Pass> createAIRRtToIpuPass() {
  return std::make_unique<AIRRtToIpuPass>();
}

} // namespace airrt
} // namespace xilinx
