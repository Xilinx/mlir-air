//===- AIRRtToIpuPass.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRRtToIpuPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Support/MathExtras.h"
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
    if (op->hasAttr("metadata"))
      newOp->setAttr("metadata",
                     op->getAttrOfType<mlir::SymbolRefAttr>("metadata"));
    else
      newOp->setAttr(
          "metadata",
          FlatSymbolRefAttr::get(newOp->getContext(),
                                 rewriter.getStringAttr("MetadataNotFound")));
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

// This is a hack due to the short-term limited support with lowering host code.
// This should be removed in the future.
class HostMemRefCopyOpConversion : public OpConversionPattern<memref::CopyOp> {
public:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Operation *> erased;
    if (auto alloc = op.getSource().getDefiningOp()) {
      op.getSource().replaceAllUsesWith(op.getTarget());
      erased.push_back(alloc);
    } else if (auto alloc = op.getTarget().getDefiningOp()) {
      op.getTarget().replaceAllUsesWith(op.getSource());
      erased.push_back(alloc);
    }
    for (auto o : erased)
      rewriter.eraseOp(o);
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

struct AIRRtToIpuPass : public airrt::impl::AIRRtToIpuBase<AIRRtToIpuPass> {
  void runOnOperation() override {

    ModuleOp module = getOperation();

    // Move func op to the end of device op's body
    moveFuncOpToEndOfDeviceOp(module);

    // Unroll affine for loops lowered from air.launch iterations
    unrollAffineFors(module);
    // Simplify arith ops (from airrt)
    auto ctx = &getContext();
    RewritePatternSet canoPatterns(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns, ctx);
    (void)applyPatternsAndFoldGreedily(module, std::move(canoPatterns));
    module.walk(
        [&](airrt::DmaMemcpyNdOp dma) { updateMetadataForUnrolledDma(dma); });
    unrollSCFFors(module);

    // Purge all wait all ops
    purgeWaitAlls(module);

    // Purge dma ops' async tokens
    purgeDmaAsyncTokens(module);

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
    patterns
        .insert<DmaToIpuPattern, HerdLoadToIpuPattern, SegmentLoadToIpuPattern,
                ModuleMetadataToIpuPattern, L1MemRefStoreOpConversion,
                L1AffineStoreOpConversion, HostMemRefCopyOpConversion>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // Simplify arith ops (from airrt-to-ipu)
    RewritePatternSet canoPatterns_1(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns_1, ctx);
    (void)applyPatternsAndFoldGreedily(module, std::move(canoPatterns_1));

    // Unroll any affine for loops
    unrollAffineFors(module);

    // Buffer ipu.dma_memcpy_nd memref to function's argument list.
    BufferMemrefToFuncArgs(module);

    // Insert sync op after copying data out to host
    insertIpuSyncOpForResults(module);

    // Renumber ipu dma ops
    renumberIpuDmaOps(module.getBody());
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
    // Taking into account for loop nests
    SmallVector<affine::AffineForOp> afos;
    module.walk([&](mlir::func::FuncOp f) {
      for (auto op : f.getOps<affine::AffineForOp>()) {
        afos.push_back(op);
      }
      for (auto op : afos) {
        unrollAffineFors(op);
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

  void unrollAffineFors(affine::AffineForOp affine_for_op) {
    SmallVector<affine::AffineForOp> afos;
    affine_for_op.walk([&](affine::AffineForOp afo) { afos.push_back(afo); });
    for (auto afo : afos) {
      int64_t tripCount = mlir::ceilDiv(afo.getConstantUpperBound() -
                                            afo.getConstantLowerBound(),
                                        afo.getStepAsInt());
      auto annotateFn = [&](unsigned i, Operation *op, OpBuilder b) {
        if (op->hasAttr("metadata") && isa<airrt::DmaMemcpyNdOp>(op)) {
          auto metadata = op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")
                              .getValue()
                              .str();
          std::string prefix = "air_channel_";
          std::size_t pos = metadata.find(prefix);
          if (pos != std::string::npos) {
            std::string base_chan_idx =
                metadata.substr(pos + prefix.length(), 1);
            int base_chan_idx_val = std::stoi(base_chan_idx);
            // Unrolled dma ops are labelled for later update of metadata air
            // channel id
            op->removeAttr("metadata");
            op->setAttr("metadata_base",
                        b.getI32IntegerAttr(base_chan_idx_val));
          }

          // default metadata format
          prefix = "airMemcpyId";
          pos = metadata.find(prefix);
          if (pos != std::string::npos) {
            op->setAttr("unrolled", b.getI32IntegerAttr(1));
          }
        }
      };
      (void)loopUnrollByFactor(afo, tripCount, annotateFn);
    }
  }

  void getOperandsFromAIRRtDma(airrt::DmaMemcpyNdOp op,
                               SmallVector<uint32_t, 4> &offsets,
                               SmallVector<uint32_t, 4> &lengths,
                               SmallVector<uint32_t, 3> &strides) {
    if (auto c = op.getOffset0().getDefiningOp<arith::ConstantIntOp>())
      offsets[0] = static_cast<uint32_t>(c.value());
    if (auto c = op.getOffset1().getDefiningOp<arith::ConstantIntOp>())
      offsets[1] = static_cast<uint32_t>(c.value());
    if (auto c = op.getOffset2().getDefiningOp<arith::ConstantIntOp>())
      offsets[2] = static_cast<uint32_t>(c.value());
    if (auto c = op.getOffset3().getDefiningOp<arith::ConstantIntOp>())
      offsets[3] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength0().getDefiningOp<arith::ConstantIntOp>())
      lengths[0] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength1().getDefiningOp<arith::ConstantIntOp>())
      lengths[1] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength2().getDefiningOp<arith::ConstantIntOp>())
      lengths[2] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength3().getDefiningOp<arith::ConstantIntOp>())
      lengths[3] = static_cast<uint32_t>(c.value());
    if (auto c = op.getStride1().getDefiningOp<arith::ConstantIntOp>())
      strides[0] = static_cast<uint32_t>(c.value());
    if (auto c = op.getStride2().getDefiningOp<arith::ConstantIntOp>())
      strides[1] = static_cast<uint32_t>(c.value());
    if (auto c = op.getStride3().getDefiningOp<arith::ConstantIntOp>())
      strides[2] = static_cast<uint32_t>(c.value());
  }

  // Get base channel index from "metadata_base" attr, increment with
  // op-specific iter, update metadata
  void updateMetadataForUnrolledDma(airrt::DmaMemcpyNdOp airrt_dma) {
    if (!airrt_dma->hasAttr("metadata_base"))
      return;

    SmallVector<uint32_t, 4> offsets(4, 0);
    SmallVector<uint32_t, 4> lengths(4, 1);
    SmallVector<uint32_t, 3> strides(3, 0);
    getOperandsFromAIRRtDma(airrt_dma, offsets, lengths, strides);

    int metadata_base =
        airrt_dma->getAttrOfType<IntegerAttr>("metadata_base").getInt();
    std::vector<unsigned> dims;
    std::vector<unsigned> position;
    SmallVector<int> memref_shape =
        air::getTensorShape(airrt_dma.getMemref().getType());

    for (int i = 0; i < memref_shape.size(); i++) {
      int cstSizeOp = lengths[i];
      dims.push_back((unsigned)mlir::ceilDiv(memref_shape[i], cstSizeOp));
      int cstOffsetOp = offsets[i];
      position.push_back((unsigned)mlir::ceilDiv(cstOffsetOp, cstSizeOp));
    }
    int iter = air::getIteratorFromMDVector(dims, position);
    auto new_metadata_attr =
        StringAttr::get(airrt_dma->getContext(),
                        "air_channel_" + std::to_string(iter + metadata_base));
    airrt_dma->setAttr(
        "metadata",
        FlatSymbolRefAttr::get(airrt_dma->getContext(), new_metadata_attr));
    airrt_dma->removeAttr("metadata_base");
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
        int64_t tripCount =
            mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
        (void)loopUnrollByFactor(for_op, tripCount);
      }
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

  void renumberIpuDmaOps(Block *blk) {
    unsigned id = 0;
    blk->walk([&](AIEX::IpuDmaMemcpyNdOp dma) {
      dma->setAttr("id",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(dma->getContext(), 32), ++id));
    });
  }

  // Buffers ipu.dma_memcpy_op memref as function argument
  void BufferMemrefToFuncArgs(ModuleOp module) {
    module.walk([&](mlir::func::FuncOp f) { BufferMemrefToFuncArgs(f); });
  }
  void BufferMemrefToFuncArgs(func::FuncOp funcOp) {
    if (!funcOp)
      return;

    // Collect illegal dma ops whose memrefs are not in function's arguments.
    SmallVector<Type, 6> memrefTypes;
    SmallVector<Value, 6> memrefs;
    funcOp.walk([&](AIEX::IpuDmaMemcpyNdOp dma) {
      if (std::find(funcOp.getArguments().begin(), funcOp.getArguments().end(),
                    dma.getMemref()) == funcOp.getArguments().end()) {
        memrefTypes.push_back(dma.getMemref().getType());
        memrefs.push_back(dma.getMemref());
      }
    });

    // Append memref to function's arguments.
    auto functionType = funcOp.getFunctionType();
    auto newArgTypes = llvm::to_vector<6>(
        llvm::concat<const Type>(functionType.getInputs(), memrefTypes));
    auto newFunctionType = FunctionType::get(funcOp.getContext(), newArgTypes,
                                             functionType.getResults());
    funcOp.setType(newFunctionType);

    // Add the new arguments to the entry block if the function is not external.
    if (!funcOp.isExternal()) {
      Location loc = funcOp.getLoc();
      for (Value v : memrefs) {
        auto newArg = funcOp.front().addArgument(v.getType(), loc);
        v.replaceAllUsesWith(newArg);
      }
    }
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
