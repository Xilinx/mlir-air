// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "air/Conversion/AffineToAIRPass.h"
#include "PassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

#define DEBUG_TYPE "affine-to-air"

namespace {

static uint64_t DmaMemcpyOpID;

class MemrefCopyToAIRDmaConversion : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.source();
    auto dst = op.target();

    // It must already be a memref
    auto src_type = src.getType().dyn_cast<MemRefType>();
    auto dst_type = dst.getType().dyn_cast<MemRefType>();
    if (!src_type)
      return failure();

    if ((src_type.getMemorySpaceAsInt() == (int)MemorySpace::L3) &&
        (dst_type.getMemorySpaceAsInt() == (int)MemorySpace::L3))
      return failure();

    if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
      return failure();

    auto rank = src_type.getShape().size();

    SmallVector<Value, 4> src_offsets, dst_offsets;
    SmallVector<Value, 4> src_strides, dst_strides;
    SmallVector<Value, 4> src_sizes, dst_sizes;
    auto extractOperandsFromSubview = [&](memref::SubViewOp subview,
                                          auto &offsets, auto &sizes,
                                          auto &strides) {
      auto subview_offsets = subview.offsets().begin();
      auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
      auto static_sizes = extractFromI64ArrayAttr(subview.static_sizes());
      auto static_strides = extractFromI64ArrayAttr(subview.static_strides());
      auto loc = subview.getLoc();

      // get the strides and offsets from the memref type
      auto inferredType = memref::SubViewOp::inferResultType(
                              subview.getSourceType(), static_offsets,
                              static_sizes, static_strides)
                              .cast<MemRefType>();
      int64_t offset;
      SmallVector<int64_t, 4> layout_strides;
      auto successStrides =
          getStridesAndOffset(inferredType, layout_strides, offset);
      if (failed(successStrides)) {
        llvm::outs() << "Failed to get strides\n";
        return; // failure();
      }

      for (auto o : static_offsets) {
        if (o >= 0)
          offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, o));
        else
          offsets.push_back(*subview_offsets++);
      }
      for (auto s : static_sizes)
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
      for (auto s : layout_strides)
        strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
    };

    if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
      extractOperandsFromSubview(subview, src_offsets, src_sizes, src_strides);

      if (src_sizes.size() != rank)
        return failure();
      if (src_strides.size() != rank)
        return failure();

      src = subview.source();
    }

    if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
      extractOperandsFromSubview(subview, dst_offsets, dst_sizes, dst_strides);

      if (dst_sizes.size() != rank)
        return failure();
      if (dst_strides.size() != rank)
        return failure();

      dst = subview.source();
    }

    SmallVector<Value, 4> deps;
    SmallVector<Type, 4> tys;
    auto dma = rewriter.create<air::DmaMemcpyNdOp>(
        loc, tys, deps, dst, dst_offsets, dst_sizes, dst_strides, src,
        src_offsets, src_sizes, src_strides);
    dma->setAttr("id", mlir::IntegerAttr::get(
                           mlir::IntegerType::get(op->getContext(), 32),
                           ++DmaMemcpyOpID));

    rewriter.eraseOp(op);
    return success();
  }
};

class LinalgCopyToAIRDmaConversion : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.inputs()[0];
    auto dst = op.outputs()[0];

    // It must already be a memref
    auto src_type = src.getType().dyn_cast<MemRefType>();
    auto dst_type = dst.getType().dyn_cast<MemRefType>();
    if (!src_type)
      return failure();

    if ((src_type.getMemorySpaceAsInt() == (int)MemorySpace::L3) &&
        (dst_type.getMemorySpaceAsInt() == (int)MemorySpace::L3))
      return failure();

    if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
      return failure();

    auto rank = src_type.getShape().size();

#if DONT_USE_ND_COPY
    if (rank == 2) {
      SmallVector<Value, 4> src_indices;
      SmallVector<Value, 4> dst_indices;
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value stride = zero;
      Value elem_per_stride = zero;

      if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
        src_indices.push_back(zero);
        src_indices.push_back(zero);
        elem_per_stride = rewriter.create<arith::ConstantIndexOp>(
            loc, alloc.getType().getShape()[1]);
      } else if (auto cast = src.getDefiningOp<bufferization::ToMemrefOp>()) {
        src_indices.push_back(zero);
        src_indices.push_back(zero);
        elem_per_stride = rewriter.create<arith::ConstantIndexOp>(
            loc, cast.getType().cast<MemRefType>().getShape()[1]);
      } else if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
        auto offsets = subview.offsets().begin();
        auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
        for (auto o : static_offsets) {
          if (o >= 0)
            src_indices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, o));
          else
            src_indices.push_back(*offsets++);
        }
        src = subview.source();
        stride = rewriter.create<arith::ConstantIndexOp>(
            loc, src.getType().cast<MemRefType>().getShape()[1]);
      } else
        return failure();

      if (auto alloc = dst.getDefiningOp<memref::AllocOp>()) {
        dst_indices.push_back(zero);
        dst_indices.push_back(zero);
        elem_per_stride = rewriter.create<arith::ConstantIndexOp>(
            loc, alloc.getType().getShape()[1]);
      } else if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
        auto offsets = subview.offsets().begin();
        auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
        for (auto o : static_offsets) {
          if (o >= 0)
            dst_indices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, o));
          else
            dst_indices.push_back(*offsets++);
        }
        dst = subview.source();
        stride = rewriter.create<arith::ConstantIndexOp>(
            loc, dst.getType().cast<MemRefType>().getShape()[1]);
      }

      SmallVector<Value, 1> deps;
      SmallVector<Type, 1> tys;

      auto num_elements = 0;
      if (src_type.hasStaticShape())
        num_elements = src_type.getNumElements();
      else
        num_elements = dst_type.getNumElements();

      auto dma = rewriter.create<air::DmaMemcpy2dOp>(
          loc, tys, deps, dst, src, dst_indices[0], dst_indices[1],
          src_indices[0], src_indices[1],
          rewriter.create<arith::ConstantIndexOp>(loc, num_elements), stride,
          elem_per_stride);
      dma->setAttr("id", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(op->getContext(), 32),
                             ++DmaMemcpyOpID));
    } else if (rank == 4) {
      SmallVector<Value, 4> src_indices;
      SmallVector<Value, 4> dst_indices;
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value stride = zero;
      Value elem_per_stride = zero;

      if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
        src_indices.push_back(zero);
        src_indices.push_back(zero);
        src_indices.push_back(zero);
        src_indices.push_back(zero);
        elem_per_stride = rewriter.create<arith::ConstantIndexOp>(
            loc, alloc.getType().getShape()[1]);
      } else if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
        auto offsets = subview.offsets().begin();
        auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
        for (auto o : static_offsets) {
          if (o >= 0)
            src_indices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, o));
          else
            src_indices.push_back(*offsets++);
        }
        src = subview.source();
        stride = rewriter.create<arith::ConstantIndexOp>(
            loc, src.getType().cast<MemRefType>().getShape()[1]);
      } else
        return failure();

      if (auto alloc = dst.getDefiningOp<memref::AllocOp>()) {
        dst_indices.push_back(zero);
        dst_indices.push_back(zero);
        dst_indices.push_back(zero);
        dst_indices.push_back(zero);
        elem_per_stride = rewriter.create<arith::ConstantIndexOp>(
            loc, alloc.getType().getShape()[1]);
      } else if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
        auto offsets = subview.offsets().begin();
        auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
        for (auto o : static_offsets) {
          if (o >= 0)
            dst_indices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, o));
          else
            dst_indices.push_back(*offsets++);
        }
        dst = subview.source();
        stride = rewriter.create<arith::ConstantIndexOp>(
            loc, dst.getType().cast<MemRefType>().getShape()[1]);
      }

      SmallVector<Value, 1> deps;
      SmallVector<Type, 1> tys;

      auto num_elements = 0;
      if (src_type.hasStaticShape())
        num_elements = src_type.getNumElements();
      else
        num_elements = dst_type.getNumElements();

      auto dma = rewriter.create<air::DmaMemcpy4dOp>(
          loc, tys, deps, dst, src, dst_indices[0], dst_indices[1],
          dst_indices[2], dst_indices[3], src_indices[0], src_indices[1],
          src_indices[2], src_indices[3],
          rewriter.create<arith::ConstantIndexOp>(loc, num_elements), stride,
          elem_per_stride);
      dma->setAttr("id", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(op->getContext(), 32),
                             ++DmaMemcpyOpID));
    } else {
#else
    {
#endif
      SmallVector<Value, 4> src_offsets, dst_offsets;
      SmallVector<Value, 4> src_strides, dst_strides;
      SmallVector<Value, 4> src_sizes, dst_sizes;
      auto extractOperandsFromSubview = [&](memref::SubViewOp subview,
                                            auto &offsets, auto &sizes,
                                            auto &strides) {
        auto subview_offsets = subview.offsets().begin();
        auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
        auto static_sizes = extractFromI64ArrayAttr(subview.static_sizes());
        auto static_strides = extractFromI64ArrayAttr(subview.static_strides());
        auto loc = subview.getLoc();

        // get the strides and offsets from the memref type
        auto inferredType = memref::SubViewOp::inferResultType(
                                subview.getSourceType(), static_offsets,
                                static_sizes, static_strides)
                                .cast<MemRefType>();
        int64_t offset;
        SmallVector<int64_t, 4> layout_strides;
        auto successStrides =
            getStridesAndOffset(inferredType, layout_strides, offset);
        if (failed(successStrides)) {
          llvm::outs() << "Failed to get strides\n";
          return; // failure();
        }

        for (auto o : static_offsets) {
          if (o >= 0)
            offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, o));
          else
            offsets.push_back(*subview_offsets++);
        }
        for (auto s : static_sizes)
          sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
        for (auto s : layout_strides)
          strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
      };

      if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
        extractOperandsFromSubview(subview, src_offsets, src_sizes,
                                   src_strides);

        if (src_sizes.size() != rank)
          return failure();
        if (src_strides.size() != rank)
          return failure();

        src = subview.source();
      }

      if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
        extractOperandsFromSubview(subview, dst_offsets, dst_sizes,
                                   dst_strides);

        if (dst_sizes.size() != rank)
          return failure();
        if (dst_strides.size() != rank)
          return failure();

        dst = subview.source();
      }

      SmallVector<Value, 4> deps;
      SmallVector<Type, 4> tys;
      auto dma = rewriter.create<air::DmaMemcpyNdOp>(
          loc, tys, deps, dst, dst_offsets, dst_sizes, dst_strides, src,
          src_offsets, src_sizes, src_strides);
      dma->setAttr("id", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(op->getContext(), 32),
                             ++DmaMemcpyOpID));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class AffineCopyToAIRDMAConversion : public ConversionPattern {
public:
  explicit AffineCopyToAIRDMAConversion(MLIRContext *context)
      : ConversionPattern(AffineDmaStartOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto affine_dma_start = cast<AffineDmaStartOp>(op);

    auto src = affine_dma_start.getSrcMemRef();
    auto src_indices = affine_dma_start.getSrcIndices();

    auto dst = affine_dma_start.getDstMemRef();
    auto dst_indices = affine_dma_start.getDstIndices();

    SmallVector<AffineApplyOp, 4> src_applies;
    SmallVector<AffineApplyOp, 4> dst_applies;
    unsigned dims = affine_dma_start.getDstMap().getNumResults();
    for (unsigned i = 0; i < dims; i++) {
      auto src_submap = affine_dma_start.getSrcMap().getSubMap({i});
      auto dst_submap = affine_dma_start.getDstMap().getSubMap({i});
      src_applies.push_back(rewriter.create<AffineApplyOp>(
          op->getLoc(), src_submap, src_indices));
      dst_applies.push_back(rewriter.create<AffineApplyOp>(
          op->getLoc(), dst_submap, dst_indices));
    }

    SmallVector<Type, 1> tys;
    SmallVector<Value, 1> deps;
    Operation *dma = nullptr;
    Value stride;
    Value elem_per_stride;
    if (affine_dma_start.isStrided()) {
      stride = affine_dma_start.getStride();
      elem_per_stride = affine_dma_start.getNumElementsPerStride();
    } else {
      stride = elem_per_stride = affine_dma_start.getNumElements();
    }
    if (dims == 1) {
      dma = rewriter.create<air::DmaMemcpyOp>(
          op->getLoc(), tys, deps, dst, src, dst_applies[0], src_applies[0],
          affine_dma_start.getNumElements());
    } else if (dims == 2) {
      dma = rewriter.create<air::DmaMemcpy2dOp>(
          op->getLoc(), tys, deps, dst, src, dst_applies[0], dst_applies[1],
          src_applies[0], src_applies[1], affine_dma_start.getNumElements(),
          stride, elem_per_stride);
    } else if (dims == 4) {
      dma = rewriter.create<air::DmaMemcpy4dOp>(
          op->getLoc(), tys, deps, dst, src, dst_applies[0], dst_applies[1],
          dst_applies[2], dst_applies[3], src_applies[0], src_applies[1],
          src_applies[2], src_applies[3], affine_dma_start.getNumElements(),
          stride, elem_per_stride);
    } else {
      llvm::outs() << "unsupported memcpy in affine-to-air";
      op->print(llvm::outs());
      return failure();
    }
    dma->setAttr("id", mlir::IntegerAttr::get(
                           mlir::IntegerType::get(op->getContext(), 32),
                           ++DmaMemcpyOpID));

    rewriter.eraseOp(op);
    return success();
  }
};

class AffineParToHerdLaunchConversion
    : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDims() == 2) {
      auto loc = op.getLoc();
      auto ub0 = op.upperBoundsMap().getResult(0).cast<AffineConstantExpr>();
      auto ub1 = op.upperBoundsMap().getResult(1).cast<AffineConstantExpr>();
      SmallVector<Value, 4> args;
      SmallVector<Value, 4> constants;
      llvm::SetVector<Value> region_args;
      getUsedValuesDefinedAbove(op.getRegion(), region_args);
      for (Value v : region_args) {
        if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
          constants.push_back(v);
        else
          args.push_back(v);
      }
      air::HerdDim2 dims{
          rewriter.create<arith::ConstantIndexOp>(loc, ub0.getValue()),
          rewriter.create<arith::ConstantIndexOp>(loc, ub1.getValue())};
      auto launch = rewriter.create<air::HerdLaunchOp>(op.getLoc(), dims, args);

      if (auto attr =
              op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        launch->setAttr(SymbolTable::getSymbolAttrName(), attr);

      auto &bb = launch.body().front();
      auto ivs = op.getIVs();
      ivs[0].replaceAllUsesWith(launch.getTileIds().x);
      ivs[1].replaceAllUsesWith(launch.getTileIds().y);
      auto &body = op.getBody()->getOperations();
      bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
      rewriter.setInsertionPointToStart(&launch.getRegion().front());
      for (auto c : constants) {
        replaceAllUsesInRegionWith(
            c, rewriter.clone(*c.getDefiningOp())->getResult(0),
            launch.getRegion());
      }
      auto builder = OpBuilder::atBlockEnd(&bb);
      builder.create<air::HerdTerminatorOp>(loc);

      int i = 0;
      auto kernel_args = launch.getKernelArguments();
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());

      rewriter.eraseOp(op);

      return success();
    }
    return failure();
  }
};

class ScfParToHerdLaunchConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToHerdLaunchConversion(MLIRContext *ctx,
                               llvm::SmallSet<scf::ParallelOp, 2> &filteredOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps){};

  LogicalResult normalizeScfParallel(scf::ParallelOp parOp,
                                     PatternRewriter &rewriter) const {
    auto loc = parOp.getLoc();

    // everything must be a constant
    for (auto step : parOp.getStep()) {
      if (!step.getDefiningOp<arith::ConstantIndexOp>())
        return failure();
    }
    for (auto lowerBound : parOp.getLowerBound()) {
      if (!lowerBound.getDefiningOp<arith::ConstantIndexOp>())
        return failure();
    }
    for (auto upperBound : parOp.getUpperBound()) {
      if (!upperBound.getDefiningOp<arith::ConstantIndexOp>())
        return failure();
    }

    auto ivs = parOp.getInductionVars().begin();
    auto step = parOp.getStep().begin();
    auto lowerBound = parOp.getLowerBound().begin();
    auto upperBound = parOp.getUpperBound().begin();

    SmallVector<Value, 4> new_step;
    SmallVector<Value, 4> new_ub;
    SmallVector<Value, 4> new_lb;

    auto builder = OpBuilder::atBlockBegin(parOp.getBody());
    while (step != parOp.getStep().end()) {
      Value sv = *step++;
      Value lbv = *lowerBound++;
      float s = sv.getDefiningOp<arith::ConstantIndexOp>().value();
      float lb = lbv.getDefiningOp<arith::ConstantIndexOp>().value();
      float ub =
          (*upperBound++).getDefiningOp<arith::ConstantIndexOp>().value();
      new_ub.push_back(rewriter.create<arith::ConstantIndexOp>(
          loc, (uint64_t)ceil((ub - lb) / s)));
      new_lb.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      new_step.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      auto iv = *ivs++;
      AffineExpr d0 = builder.getAffineDimExpr(0);
      AffineExpr mul = d0 * sv.getDefiningOp<arith::ConstantIndexOp>().value();
      AffineExpr add =
          mul + lbv.getDefiningOp<arith::ConstantIndexOp>().value();
      auto map = AffineMap::get(1, 0, add);
      auto new_iv = builder.create<AffineApplyOp>(loc, map, iv);
      SmallPtrSet<Operation *, 1> keep{new_iv};
      iv.replaceAllUsesExcept(new_iv.getResult(), keep);
    }

    parOp.getLowerBoundMutable().assign(new_lb);
    parOp.getUpperBoundMutable().assign(new_ub);
    parOp.getStepMutable().assign(new_step);

    return success();
  }

  LogicalResult matchAndRewrite(scf::ParallelOp parOp,
                                PatternRewriter &rewriter) const override {

    scf::ParallelOp op = parOp;

    if (!filteredOps.contains(op))
      return failure();

    if (failed(normalizeScfParallel(op, rewriter)))
      return failure();

    auto loc = op.getLoc();

    if (op.getNumLoops() > 2) {
      unsigned split_idx = op.getNumLoops() - 2;
      SmallVector<Value, 2> outerLowerBounds, outerUpperBounds, outerSteps;
      SmallVector<Value, 2> innerLowerBounds, innerUpperBounds, innerSteps;

      for (unsigned i = 0, e = split_idx; i < e; ++i) {
        outerLowerBounds.push_back(op.getLowerBound()[i]);
        outerUpperBounds.push_back(op.getUpperBound()[i]);
        outerSteps.push_back(op.getStep()[i]);
      }
      auto outerLoop = rewriter.create<scf::ParallelOp>(
          loc, outerLowerBounds, outerUpperBounds, outerSteps);
      for (unsigned i = 0, e = split_idx; i < e; ++i)
        op.getInductionVars()[i].replaceAllUsesWith(
            outerLoop.getInductionVars()[i]);

      rewriter.setInsertionPointToStart(outerLoop.getBody());

      for (unsigned i = split_idx, e = op.getNumLoops(); i < e; ++i) {
        innerLowerBounds.push_back(op.getLowerBound()[i]);
        innerUpperBounds.push_back(op.getUpperBound()[i]);
        innerSteps.push_back(op.getStep()[i]);
      }
      auto innerLoop = rewriter.create<scf::ParallelOp>(
          loc, innerLowerBounds, innerUpperBounds, innerSteps);
      for (unsigned i = split_idx, e = op.getNumLoops(); i < e; ++i)
        op.getInductionVars()[i].replaceAllUsesWith(
            innerLoop.getInductionVars()[i - split_idx]);

      auto &body = op.getBody()->getOperations();
      innerLoop.getBody()->getOperations().splice(
          innerLoop.getBody()->begin(), body, body.begin(), --body.end());
      op = innerLoop;
    }

    SmallVector<int, 2> bounds{1, 1};
    for (unsigned int i = 0; i < op.getNumLoops(); i++) {
      auto lb = dyn_cast<arith::ConstantIndexOp>(
          op.getLowerBound()[i].getDefiningOp());
      auto ub = dyn_cast<arith::ConstantIndexOp>(
          op.getUpperBound()[i].getDefiningOp());
      auto step =
          dyn_cast<arith::ConstantIndexOp>(op.getStep()[i].getDefiningOp());

      // lowerBound, upperBound and step must be arith::ConstantIndexOps
      if (!(lb && step && ub))
        return failure();

      auto ub_int = ub.value();
      auto lb_int = lb.value();
      auto step_int = step.value();

      // must start at 0
      if (lb_int)
        return failure();

      // step must divide upper bound evenly
      if (ub_int % step_int)
        return failure();

      ub_int = ub_int / step_int;
      bounds[i] = ub_int;
    }
    SmallVector<Value, 4> args;
    SmallVector<Value, 4> constants;
    llvm::SetVector<Value> region_args;
    getUsedValuesDefinedAbove(op.getRegion(), region_args);
    for (Value v : region_args) {
      if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
        constants.push_back(v);
      else
        args.push_back(v);
    }
    air::HerdDim2 dims{rewriter.create<arith::ConstantIndexOp>(loc, bounds[0]),
                       rewriter.create<arith::ConstantIndexOp>(loc, bounds[1])};
    auto launch = rewriter.create<air::HerdLaunchOp>(op.getLoc(), dims, args);
    auto &bb = launch.body().front();
    auto ivs = op.getInductionVars();

    ivs[0].replaceAllUsesWith(launch.getTileIds().x);
    if (op.getNumLoops() == 2)
      ivs[1].replaceAllUsesWith(launch.getTileIds().y);

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&launch.getRegion().front());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, rewriter.clone(*c.getDefiningOp())->getResult(0),
          launch.getRegion());
    }
    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<air::HerdTerminatorOp>(loc);

    int i = 0;
    auto kernel_args = launch.getKernelArguments();
    for (Value v : args)
      replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());

    if (op != parOp)
      op.erase();
    rewriter.eraseOp(parOp);

    return success();
  }

private:
  llvm::SmallSet<scf::ParallelOp, 2> &filteredOps;
};

struct AffineToAIRPass : public AffineToAIRBase<AffineToAIRPass> {

  AffineToAIRPass() = default;
  AffineToAIRPass(const AffineToAIRPass &pass) {}

  Option<int> clHerdAssignDepth{
      *this, "herd-assign-depth",
      llvm::cl::desc("Given a nest of parallel for loops, which depth to map "
                     "to herd launch"),
      llvm::cl::init(-1)};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    RewritePatternSet patterns(context);
    patterns.add<AffineParToHerdLaunchConversion>(context);

    llvm::SmallVector<xilinx::air::HerdLaunchOp, 2> herdOps;
    module.walk([&](xilinx::air::HerdLaunchOp op) {
      herdOps.push_back(op);
    });

    llvm::SmallSet<scf::ParallelOp, 2> filteredOps;
    module.walk([&](scf::ParallelOp op) {
      if (op->getParentOfType<xilinx::air::HerdLaunchOp>())
        return;
      for (auto &h : herdOps)
        if (op->isProperAncestor(h))
          return;
      if (clHerdAssignDepth < 0) {
        filteredOps.insert(op);
        return;
      }
      // the number of nested scf.parallel above this one
      int parallel_depth = 0;
      Operation *par = op.getOperation();
      while ((par = par->getParentOp()))
        if (isa<scf::ParallelOp>(par))
          parallel_depth++;
      if (parallel_depth != clHerdAssignDepth)
        return;
      filteredOps.insert(op);
    });
    patterns.add<ScfParToHerdLaunchConversion>(context, filteredOps);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           arith::ArithmeticDialect>();

    target.addLegalOp<xilinx::air::DmaMemcpyOp>();
    target.addLegalOp<xilinx::air::DmaMemcpy2dOp>();
    target.addLegalOp<xilinx::air::DmaMemcpy4dOp>();
    target.addLegalOp<xilinx::air::DmaMemcpyNdOp>();
    target.addLegalOp<xilinx::air::HerdLaunchOp>();

    target.addLegalOp<AffineApplyOp, AffineForOp, AffineLoadOp, AffineStoreOp,
                      AffineYieldOp, scf::YieldOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });

    DmaMemcpyOpID = 0;

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    // Simplify all the subviews so we can rewrite them easily.
    // Mostly this is propagating constant sizes into dimensioned memref types.
    RewritePatternSet stage2Patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    memref::AllocOp::getCanonicalizationPatterns(stage2Patterns, context);
    (void)applyPatternsAndFoldGreedily(module, std::move(stage2Patterns));

    RewritePatternSet stage3Patterns(context);
    stage3Patterns
        .insert<AffineCopyToAIRDMAConversion, LinalgCopyToAIRDmaConversion,
                MemrefCopyToAIRDmaConversion>(context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(stage3Patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    std::vector<Operation *> waits;
    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto wo = dyn_cast<AffineDmaWaitOp>(op)) {
          auto memref = wo.getTagMemRef();
          for (auto u : memref.getUsers()) {
            waits.push_back(u);
          }
        }
      });
    }
    for (auto o : waits)
      o->erase();

    std::vector<std::string> herd_syms;
    for (auto f : module.getOps<func::FuncOp>()) {
      // record existing symbol names
      f.walk([&](xilinx::air::HerdLaunchOp op) {
        if (auto attr = op->getAttrOfType<StringAttr>(
                SymbolTable::getSymbolAttrName())) {
          std::string name = attr.getValue().str();
          assert((std::find(herd_syms.begin(), herd_syms.end(), name) ==
                  herd_syms.end()) &&
                 "unexpected duplicate symbol");
          herd_syms.push_back(name);
        }
      });
      // generate missing symbol names
      f.walk([&](xilinx::air::HerdLaunchOp op) {
        if (!op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          unsigned id = 0;
          std::string name;
          do {
            std::stringstream ss;
            ss << "herd_" << id++;
            name = ss.str();
          } while (std::find(herd_syms.begin(), herd_syms.end(), name) !=
                   herd_syms.end());
          herd_syms.push_back(name);
          op->setAttr(SymbolTable::getSymbolAttrName(),
                      StringAttr::get(op->getContext(), name));
        }
      });
    }
    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineToAIRPass() {
  return std::make_unique<AffineToAIRPass>();
}

} // namespace air
} // namespace xilinx