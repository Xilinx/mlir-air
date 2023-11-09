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
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

class L1AffineStoreOpConversion
    : public OpConversionPattern<affine::AffineStoreOp> {
public:
  using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineStoreOp op, OpAdaptor adaptor,
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

    // Move func op to the end of device op's body
    moveFuncOpToEndOfDeviceOp(module);

    // Purge dma ops' async tokens
    purgeDmaAsyncTokens(module);

    // Purge all wait all ops
    purgeWaitAlls(module);

    ConversionTarget target(getContext());
    target.addIllegalDialect<AIRRtDialect>();
    target.addLegalDialect<arith::ArithDialect, AIEX::AIEXDialect>();

    target.addDynamicallyLegalOp<affine::AffineStoreOp>(
        [&](affine::AffineStoreOp op) {
          if (op->getParentOfType<AIE::CoreOp>())
            return true;
          return (op.getMemref()
                      .getType()
                      .cast<MemRefType>()
                      .getMemorySpaceAsInt() !=
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

    // Simplify arith ops
    RewritePatternSet canoPatterns(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns, ctx);
    (void)applyPatternsAndFoldGreedily(module, std::move(canoPatterns));

    // Unroll any affine for loops
    unrollAffineFors(module);

    // Insert sync op after copying data out to host
    insertIpuSyncOpForResults(module);
  }

  void moveFuncOpToEndOfDeviceOp(ModuleOp module) {
    // Move func op to the end of device op's body
    SmallVector<SegmentLoadOp> segs;
    module.walk([&](SegmentLoadOp s) { segs.push_back(s); });
    for (auto s : segs) {
      auto f = s->getParentOfType<func::FuncOp>();
      auto d = getDeviceForSegmentLoad(s);
      if (!f || !d)
        continue;
      f->moveAfter(&d.getBody()->back());
    }
  }

  void purgeDmaAsyncTokens(ModuleOp module) {
    SmallVector<DmaMemcpyNdOp> dmas;
    module.walk([&](DmaMemcpyNdOp dma) { dmas.push_back(dma); });
    for (auto dma : dmas) {
      if (dma->getNumResults()) {
        OpBuilder buider(dma);
        SmallVector<Type, 1> tys = {};
        auto newOp = buider.create<DmaMemcpyNdOp>(dma->getLoc(), tys,
                                                  dma->getOperands());
        if (dma->hasAttr("metadata"))
          newOp->setAttr("metadata",
                         dma->getAttrOfType<mlir::SymbolRefAttr>("metadata"));
        dma->erase();
      }
    }
  }

  void purgeWaitAlls(ModuleOp module) {
    SmallVector<WaitAllOp> waits;
    module.walk([&](WaitAllOp w) { waits.push_back(w); });
    for (auto w : waits) {
      w->eraseOperands(0, w->getNumOperands());
    }
    for (auto w : waits) {
      w.erase();
    }
  }

  void unrollAffineFors(ModuleOp module) {
    SmallVector<affine::AffineForOp> afos;
    module.walk([&](mlir::func::FuncOp f) {
      f.walk([&](affine::AffineForOp afo) { afos.push_back(afo); });
    });
    for (auto afo : afos) {
      (void)loopUnrollFull(afo);
    }
  }

  std::optional<AIE::ShimDMAAllocationOp>
  getAllocOpForSymbol(AIE::DeviceOp dev, StringRef sym_name) {
    auto sym = dev.lookupSymbol(sym_name);
    if (!sym)
      return std::nullopt;

    auto uses = SymbolTable::getSymbolUses(sym, dev);
    for (auto use : *uses)
      if (auto infoOp = dyn_cast<AIE::ShimDMAAllocationOp>(use.getUser()))
        return infoOp;

    return std::nullopt;
  }

  void insertIpuSyncOpForResults(ModuleOp module) {
    module.walk([&](mlir::func::FuncOp f) {
      SmallVector<AIEX::IpuDmaMemcpyNdOp> dmas;
      f.walk([&](AIEX::IpuDmaMemcpyNdOp dma) { dmas.push_back(dma); });
      auto d = f->getParentOfType<AIE::DeviceOp>();
      for (auto dma : dmas) {
        if (auto infoOp = getAllocOpForSymbol(d, dma.getMetadata())) {
          if (infoOp->getChannelDir() == AIE::DMAChannelDir::S2MM) {
            // Found dma op copying results to host
            OpBuilder builder(dma);
            auto col = builder.getI32IntegerAttr(infoOp->getCol());
            auto row = builder.getI32IntegerAttr(0);
            auto dir = builder.getI32IntegerAttr(0);
            auto chan = builder.getI32IntegerAttr(infoOp->getChannelIndex());
            auto col_num = builder.getI32IntegerAttr(1);
            auto row_num = builder.getI32IntegerAttr(1);
            builder.setInsertionPointAfter(dma);
            builder.create<AIEX::IpuSyncOp>(dma->getLoc(), col, row, dir, chan,
                                            col_num, row_num);
          }
        }
      }
    });
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
