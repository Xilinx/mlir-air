//===- ConvertToAIRPass.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/ConvertToAIRPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "convert-to-air"

static std::atomic<uint64_t> DmaMemcpyOpID;

static FailureOr<air::DmaMemcpyNdOp>
matchAndRewriteCopyOp(memref::CopyOp op, RewriterBase &rewriter) {
  auto loc = op.getLoc();
  Value src = op.getSource();
  Value dst = op.getTarget();

  rewriter.setInsertionPoint(op);

  // It must already be a memref
  auto src_type = llvm::dyn_cast<MemRefType>(src.getType());
  auto dst_type = llvm::dyn_cast<MemRefType>(dst.getType());
  if (!src_type)
    return failure();

  if ((src_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) &&
      (dst_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3))
    return failure();

  SmallVector<Value, 4> src_offsets, dst_offsets;
  SmallVector<Value, 4> src_strides, dst_strides;
  SmallVector<Value, 4> src_sizes, dst_sizes;
  auto extractOperandsFromSubview = [&](memref::SubViewOp subview,
                                        auto &offsets, auto &sizes,
                                        auto &strides) {
    auto subview_offsets = subview.getOffsets().begin();
    auto static_offsets = subview.getStaticOffsets();
    auto subview_sizes = subview.getSizes().begin();
    auto static_sizes = subview.getStaticSizes();
    auto subview_strides = subview.getStrides().begin();
    auto static_strides = subview.getStaticStrides();
    auto loc = subview.getLoc();

    // get the strides and offsets from the memref type
    auto inferredType =
        llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
            subview.getSourceType(), static_offsets, static_sizes,
            static_strides));
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
      if (s >= 0)
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
      else
        sizes.push_back(*subview_sizes++);
    for (auto s : layout_strides)
      if (s >= 0)
        strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
      else
        strides.push_back(*subview_strides++);
  };

  if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
    extractOperandsFromSubview(subview, src_offsets, src_sizes, src_strides);
    src = subview.getSource();
  }

  if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
    extractOperandsFromSubview(subview, dst_offsets, dst_sizes, dst_strides);
    dst = subview.getSource();
  }

  SmallVector<Value, 4> deps;
  SmallVector<Type, 4> tys;
  auto dma = rewriter.create<air::DmaMemcpyNdOp>(
      loc, tys, deps, dst, dst_offsets, dst_sizes, dst_strides, src,
      src_offsets, src_sizes, src_strides);
  dma->setAttr(
      "id", mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                                   ++DmaMemcpyOpID));

  rewriter.eraseOp(op);
  return dma;
}

static void extractOperandsFromSubview(memref::SubViewOp subview,
                                       OpBuilder &builder,
                                       SmallVector<Value, 4> &offsets,
                                       SmallVector<Value, 4> &sizes,
                                       SmallVector<Value, 4> &strides) {
  auto subview_offsets = subview.getOffsets().begin();
  auto static_offsets = subview.getStaticOffsets();
  auto static_sizes = subview.getStaticSizes();
  auto static_strides = subview.getStaticStrides();
  auto loc = subview.getLoc();

  // get the strides and offsets from the memref type
  auto inferredType = llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
      subview.getSourceType(), static_offsets, static_sizes, static_strides));
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
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, o));
    else
      offsets.push_back(*subview_offsets++);
  }
  for (auto s : static_sizes)
    sizes.push_back(builder.create<arith::ConstantIndexOp>(loc, s));
  for (auto s : layout_strides)
    strides.push_back(builder.create<arith::ConstantIndexOp>(loc, s));
}

static void
replaceAllUsesOfConstsInRegionWithNew(SmallVector<Value, 4> constants,
                                      OpBuilder builder, Region &region) {
  for (auto c : constants) {
    replaceAllUsesInRegionWith(
        c, builder.clone(*c.getDefiningOp())->getResult(0), region);
  }
}

void getUsedConstsAndArgsDefinedAbove(MutableArrayRef<Region> region,
                                      SmallVector<Value, 4> &constants,
                                      SmallVector<Value, 4> &args) {
  llvm::SetVector<Value> region_args;
  getUsedValuesDefinedAbove(region, region_args);
  for (Value v : region_args) {
    if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
      constants.push_back(v);
    else
      args.push_back(v);
  }
}

namespace {

class MemrefCopyToAIRDmaConversion : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(matchAndRewriteCopyOp(op, rewriter)))
      return failure();
    return success();
  }
};

class LinalgCopyToAIRDmaConversion : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getInputs()[0];
    auto dst = op.getOutputs()[0];

    // It must already be a memref
    auto src_type = llvm::dyn_cast<MemRefType>(src.getType());
    auto dst_type = llvm::dyn_cast<MemRefType>(dst.getType());
    if (!src_type)
      return failure();

    if ((src_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) &&
        (dst_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3))
      return failure();

    if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
      return failure();

    SmallVector<Value, 4> src_offsets, dst_offsets;
    SmallVector<Value, 4> src_strides, dst_strides;
    SmallVector<Value, 4> src_sizes, dst_sizes;

    if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
      extractOperandsFromSubview(subview, rewriter, src_offsets, src_sizes,
                                 src_strides);
      src = subview.getSource();
    }

    if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
      extractOperandsFromSubview(subview, rewriter, dst_offsets, dst_sizes,
                                 dst_strides);
      dst = subview.getSource();
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

class AffineParToHerdConversion
    : public OpRewritePattern<affine::AffineParallelOp> {
public:
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  AffineParToHerdConversion(MLIRContext *ctx,
                            SmallPtrSet<Operation *, 8> &filteredOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps){};

  LogicalResult matchAndRewrite(affine::AffineParallelOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getNumDims() != 2) {
      return op->emitOpError(
          "failed conversion to 'air.herd': only 2d loops are supported");
    }

    normalizeAffineParallel(op);

    auto loc = op.getLoc();
    auto ub0 =
        dyn_cast<AffineConstantExpr>(op.getUpperBoundsMap().getResult(0));
    auto ub1 =
        dyn_cast<AffineConstantExpr>(op.getUpperBoundsMap().getResult(1));

    if (!ub0 || !ub1) {
      return op->emitOpError("failed conversion to 'air.herd': only constant "
                             "loop bounds are supported");
    }

    SmallVector<Value, 4> args;
    SmallVector<Value, 4> constants;
    getUsedConstsAndArgsDefinedAbove(op.getRegion(), constants, args);
    SmallVector<Value, 2> dims{
        rewriter.create<arith::ConstantIndexOp>(loc, ub0.getValue()),
        rewriter.create<arith::ConstantIndexOp>(loc, ub1.getValue())};
    auto launch = rewriter.create<air::HerdOp>(op.getLoc(), dims, args);

    if (auto attr =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      launch->setAttr(SymbolTable::getSymbolAttrName(), attr);

    auto &bb = launch.getBody().front();
    auto ivs = op.getIVs();
    ivs[0].replaceAllUsesWith(launch.getIds()[0]);
    ivs[1].replaceAllUsesWith(launch.getIds()[1]);
    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&launch.getRegion().front());
    replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                          launch.getRegion());

    int i = 0;
    auto kernel_args = launch.getKernelArguments();
    for (Value v : args)
      replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());

    rewriter.eraseOp(op);

    return success();
  }

private:
  llvm::SmallPtrSet<Operation *, 8> filteredOps;
};

LogicalResult normalizeScfParallel(scf::ParallelOp parOp,
                                   PatternRewriter &rewriter) {

  // everything must be a constant
  for (auto step : parOp.getStep())
    if (!step.getDefiningOp<arith::ConstantIndexOp>())
      return parOp->emitOpError("failed to normalize: step is not a constant");
  for (auto lowerBound : parOp.getLowerBound())
    if (!lowerBound.getDefiningOp<arith::ConstantIndexOp>())
      return parOp->emitOpError(
          "failed to normalize: lower bound is not a constant");
  for (auto upperBound : parOp.getUpperBound())
    if (!upperBound.getDefiningOp<arith::ConstantIndexOp>())
      return parOp->emitOpError(
          "failed to normalize: upper bound is not a constant");

  SmallVector<Value> new_step;
  SmallVector<Value> new_ub;
  SmallVector<Value> new_lb;

  for (unsigned i = 0; i < parOp.getNumLoops(); i++) {
    Value iv = parOp.getInductionVars()[i];
    Value sv = parOp.getStep()[i];
    Value lbv = parOp.getLowerBound()[i];
    Value ubv = parOp.getUpperBound()[i];
    auto s = sv.getDefiningOp<arith::ConstantIndexOp>().value();
    auto lb = lbv.getDefiningOp<arith::ConstantIndexOp>().value();
    auto ub = ubv.getDefiningOp<arith::ConstantIndexOp>().value();

    auto new_ub_int = (ub - lb) / s;
    if ((new_ub_int * s) != (ub - lb))
      return parOp->emitOpError()
             << "failed to normalize: step '" << s
             << "' does not evenly divide range '" << (ub - lb) << "'";

    auto loc = parOp.getLoc();
    new_ub.push_back(rewriter.create<arith::ConstantIndexOp>(loc, new_ub_int));
    new_lb.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    new_step.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr mul = d0 * s;
    AffineExpr add = mul + lb;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(parOp.getBody());
      auto map = AffineMap::get(1, 0, add);
      auto new_iv = rewriter.create<affine::AffineApplyOp>(loc, map, iv);
      SmallPtrSet<Operation *, 1> keep{new_iv};
      iv.replaceAllUsesExcept(new_iv.getResult(), keep);
    }
  }

  parOp.getLowerBoundMutable().assign(new_lb);
  parOp.getUpperBoundMutable().assign(new_ub);
  parOp.getStepMutable().assign(new_step);

  return success();
}

void InsertEmptyLaunchOverHerd(air::HerdOp op) {
  OpBuilder builder(op);
  if (op->getParentOfType<air::SegmentOp>()) {
    return;
  }
  if (op->getParentOfType<air::LaunchOp>()) {
    return;
  }

  auto loc = op.getLoc();

  SmallVector<Value, 4> args;
  for (unsigned i = 0; i < op.getNumKernelOperands(); i++) {
    args.push_back(op.getKernelOperand(i));
  }
  SmallVector<Value, 4> sizes;
  // Generate a surrounding launch op with size of 1.
  for (unsigned i = 0; i < op.getNumDims(); i++)
    sizes.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));

  // The outermost launch op inherits herd's async interface
  air::LaunchOp launch = nullptr;
  if (op.getAsyncToken())
    launch = builder.create<air::LaunchOp>(
        op.getLoc(), op.getAsyncDependencies(), sizes, args, true);
  else
    launch = builder.create<air::LaunchOp>(op.getLoc(), sizes, args);
  builder.setInsertionPointToStart(&launch.getRegion().front());
  SmallVector<Value, 1> segmentSizes = {};
  SmallVector<Value, 4> segmentOpers;
  for (Value v : launch.getIds()) {
    segmentOpers.push_back(v);
  }
  for (Value v : launch.getSize()) {
    segmentOpers.push_back(v);
  }
  for (Value v : launch.getKernelArguments()) {
    segmentOpers.push_back(v);
  }

  air::SegmentOp segment = nullptr;
  if (op.getAsyncToken())
    segment = builder.create<air::SegmentOp>(op.getLoc(), SmallVector<Value>{},
                                             segmentSizes, segmentOpers, true);
  else
    segment =
        builder.create<air::SegmentOp>(op.getLoc(), segmentSizes, segmentOpers);

  builder.setInsertionPointToStart(&segment.getRegion().front());

  SmallVector<Value, 2> herdSizes = {};
  SmallVector<Value, 4> herdOpers;
  for (Value v : segment.getIds()) {
    herdOpers.push_back(v);
  }
  for (Value v : segment.getSize()) {
    herdOpers.push_back(v);
  }
  for (Value v : segment.getKernelArguments()) {
    herdOpers.push_back(v);
  }
  for (unsigned i = 0; i < op.getNumDims(); i++) {
    herdSizes.push_back(
        builder.clone(*op.getSizeOperands()[i].getDefiningOp())->getResult(0));
  }

  air::HerdOp herdOp = nullptr;
  if (op.getAsyncToken())
    herdOp = builder.create<air::HerdOp>(op.getLoc(), SmallVector<Value>{},
                                         herdSizes,
                                         segment.getKernelArguments(), true);
  else
    herdOp = builder.create<air::HerdOp>(op.getLoc(), herdSizes,
                                         segment.getKernelArguments());

  IRMapping remap;
  for (unsigned i = 0; i < op.getNumDims(); i++) {
    remap.map(op.getIds()[i], herdOp.getIds()[i]);
    remap.map(op.getSize()[i], herdOp.getSize()[i]);
  }
  for (unsigned i = 0; i < op.getNumKernelOperands(); i++) {
    remap.map(op.getKernelArgument(i),
              herdOp.getKernelArgument(launch.getNumDims() * 2 + i));
  }

  builder.setInsertionPointToStart(&herdOp.getRegion().front());
  for (auto &o : op.getBody().front().getOperations())
    if (!isa<air::HerdTerminatorOp>(o))
      builder.clone(o, remap);

  // Copy over herd name
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
    std::string name = attr.getValue().str();
    herdOp->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(op->getContext(), name));
  }

  if (auto token = op.getAsyncToken()) {
    replaceAllUsesInRegionWith(token, launch.getAsyncToken(),
                               *op->getParentRegion());
  }
  op->erase();
  return;
}

// func.call itself has a `link_with` which we can absorb into air.herd.
// Walk through all the func.call operations (immediate/nested children)
// within parallel loop. Currently we only assume and enforce that we relay
// `link_with` information from just one func.call op.
static void propagateLinkWith(Operation *op, air::HerdOp herdOp) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  op->walk([&](func::CallOp callOp) {
    // Fetch name.
    StringRef fnName = callOp.getCallee();
    auto fnDecl = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupSymbolIn(moduleOp, fnName));
    assert(fnDecl && "expected function declaration");
    assert(fnDecl->hasAttr("link_with") &&
           "expected 'link_with' construct for the function declaration");
    herdOp->setAttr("link_with", fnDecl->getAttr("link_with"));
    return WalkResult::interrupt();
  });
}

class ScfParToHerdConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToHerdConversion(MLIRContext *ctx,
                         SmallPtrSet<Operation *, 8> &filteredOps,
                         llvm::SmallSet<air::HerdOp, 2> &replacementOps,
                         int firstDim)
      : OpRewritePattern(ctx), filteredOps(filteredOps),
        replacementOps(replacementOps), firstDim(firstDim){};

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
      // these are arith.constant after normalizeScfParallel
      auto to_int = [](Value v) {
        return cast<arith::ConstantIndexOp>(v.getDefiningOp()).value();
      };
      auto ub_int = to_int(op.getUpperBound()[i]);
      auto step_int = to_int(op.getStep()[i]);
      bounds[i] = ub_int / step_int;
    }
    SmallVector<Value, 4> args;
    SmallVector<Value, 4> constants;
    getUsedConstsAndArgsDefinedAbove(op.getRegion(), constants, args);

    int idx0 = firstDim;
    int idx1 = (firstDim + 1) % 2;
    SmallVector<Value, 2> dims{
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[idx0]),
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[idx1])};
    auto herdOp = rewriter.create<air::HerdOp>(op.getLoc(), dims, args);
    auto &body = op.getBody()->getOperations();

    propagateLinkWith(op, herdOp);

    auto &bb = herdOp.getBody().front();
    auto ivs = op.getInductionVars();

    ivs[0].replaceAllUsesWith(herdOp.getIds()[idx0]);
    if (op.getNumLoops() == 2)
      ivs[1].replaceAllUsesWith(herdOp.getIds()[idx1]);

    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&herdOp.getRegion().front());
    replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                          herdOp.getRegion());

    int i = 0;
    auto kernel_args = herdOp.getKernelArguments();
    for (Value v : args)
      replaceAllUsesInRegionWith(v, kernel_args[i++], herdOp.getRegion());

    if (op != parOp)
      op.erase();
    rewriter.eraseOp(parOp);
    replacementOps.insert(herdOp);

    return success();
  }

private:
  llvm::SmallPtrSet<Operation *, 8> &filteredOps;
  llvm::SmallSet<air::HerdOp, 2> &replacementOps;
  int firstDim;
};

class ScfForallToHerdConversion : public OpRewritePattern<scf::ForallOp> {
public:
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  ScfForallToHerdConversion(MLIRContext *ctx,
                            SmallPtrSet<Operation *, 8> &filteredOps,
                            llvm::SmallSet<air::HerdOp, 2> &replacementOps,
                            int firstDim)
      : OpRewritePattern(ctx), filteredOps(filteredOps),
        replacementOps(replacementOps), firstDim(firstDim){};

  LogicalResult matchAndRewrite(scf::ForallOp parOp,
                                PatternRewriter &rewriter) const override {

    scf::ForallOp op = parOp;

    if (!filteredOps.contains(op))
      return failure();

    auto loc = op.getLoc();

    if (op.getRank() > 2) {
      unsigned split_idx = op.getRank() - 2;
      SmallVector<OpFoldResult> outerLowerBounds, outerUpperBounds, outerSteps;
      SmallVector<OpFoldResult> innerLowerBounds, innerUpperBounds, innerSteps;

      for (unsigned i = 0, e = split_idx; i < e; ++i) {
        outerLowerBounds.push_back(op.getMixedLowerBound()[i]);
        outerUpperBounds.push_back(op.getMixedUpperBound()[i]);
        outerSteps.push_back(op.getMixedStep()[i]);
      }
      auto outerLoop = rewriter.create<scf::ParallelOp>(
          loc, getValueOrCreateConstantIndexOp(rewriter, loc, outerLowerBounds),
          getValueOrCreateConstantIndexOp(rewriter, loc, outerUpperBounds),
          getValueOrCreateConstantIndexOp(rewriter, loc, outerSteps));
      for (unsigned i = 0, e = split_idx; i < e; ++i)
        op.getInductionVars()[i].replaceAllUsesWith(
            outerLoop.getInductionVars()[i]);

      rewriter.setInsertionPointToStart(outerLoop.getBody());

      for (unsigned i = split_idx, e = op.getRank(); i < e; ++i) {
        innerLowerBounds.push_back(op.getMixedLowerBound()[i]);
        innerUpperBounds.push_back(op.getMixedUpperBound()[i]);
        innerSteps.push_back(op.getMixedStep()[i]);
      }
      auto innerLoop = rewriter.create<scf::ForallOp>(
          loc, innerLowerBounds, innerUpperBounds, innerSteps, ValueRange(),
          std::nullopt);
      for (unsigned i = split_idx, e = op.getRank(); i < e; ++i)
        op.getInductionVars()[i].replaceAllUsesWith(
            innerLoop.getInductionVars()[i - split_idx]);

      auto &body = op.getBody()->getOperations();
      innerLoop.getBody()->getOperations().splice(
          innerLoop.getBody()->begin(), body, body.begin(), --body.end());
      op = innerLoop;
    }

    SmallVector<int, 2> bounds{1, 1};
    for (unsigned int i = 0; i < op.getRank(); i++) {
      int64_t ub_int = op.getStaticUpperBound()[i];
      int64_t step_int = op.getStaticStep()[i];
      bounds[i] = ub_int / step_int;
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

    int idx0 = firstDim;
    int idx1 = (firstDim + 1) % 2;
    SmallVector<Value, 2> dims{
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[idx0]),
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[idx1])};
    auto herdOp = rewriter.create<air::HerdOp>(op.getLoc(), dims, args);
    auto &bb = herdOp.getBody().front();
    auto ivs = op.getInductionVars();

    propagateLinkWith(op, herdOp);

    ivs[0].replaceAllUsesWith(herdOp.getIds()[idx0]);
    if (op.getRank() == 2)
      ivs[1].replaceAllUsesWith(herdOp.getIds()[idx1]);

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&herdOp.getRegion().front());
    replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                          herdOp.getRegion());

    int i = 0;
    auto kernel_args = herdOp.getKernelArguments();
    for (Value v : args)
      replaceAllUsesInRegionWith(v, kernel_args[i++], herdOp.getRegion());

    if (op != parOp)
      rewriter.eraseOp(op);
    rewriter.eraseOp(parOp);
    replacementOps.insert(herdOp);

    return success();
  }

private:
  llvm::SmallPtrSet<Operation *, 8> &filteredOps;
  llvm::SmallSet<air::HerdOp, 2> &replacementOps;
  int firstDim;
};

LogicalResult
getMemrefBackwardSlices(Value &memref, Operation *&memrefAlloc,
                        SmallVector<Operation *> &backwardSlices) {
  if (!memrefAlloc)
    return failure();
  while (!isa<memref::AllocOp>(memrefAlloc)) {
    backwardSlices.push_back(memrefAlloc);
    memrefAlloc = memrefAlloc->getOperand(0).getDefiningOp();
    if (!memrefAlloc)
      return failure();
  }
  memref = dyn_cast<memref::AllocOp>(memrefAlloc).getMemref();
  return success();
}

// Attempt to tile an L2-L1 air.dma_memcpy_nd op using an scf.parallel op.
LogicalResult TileL1L2AIRMemcpyUsingScfParallel(air::DmaMemcpyNdOp op,
                                                bool SrcIsL1) {
  OpBuilder builder(op);
  auto loc = op->getLoc();
  auto L1Memref = SrcIsL1 ? op.getSrcMemref() : op.getDstMemref();
  auto L2Memref = SrcIsL1 ? op.getDstMemref() : op.getSrcMemref();
  auto L1MemrefShape = air::getTensorShape(L1Memref.getType());
  auto L2MemrefShape = air::getTensorShape(L2Memref.getType());

  Operation *L1MemrefAlloc = L1Memref.getDefiningOp();
  SmallVector<Operation *> L1MemrefOpLog;
  if (getMemrefBackwardSlices(L1Memref, L1MemrefAlloc, L1MemrefOpLog).failed())
    return failure();

  Operation *L2MemrefAlloc = L2Memref.getDefiningOp();
  SmallVector<Operation *> L2MemrefOpLog;
  if (getMemrefBackwardSlices(L2Memref, L2MemrefAlloc, L2MemrefOpLog).failed())
    return failure();

  memref::SubViewOp tilingHintSubview = nullptr;
  scf::ParallelOp previousTilingScfPar = nullptr;
  for (auto user : L1Memref.getUsers()) {
    if (auto subViewUser = dyn_cast<memref::SubViewOp>(user))
      tilingHintSubview = subViewUser;
    else
      continue;
    if (auto subViewParentPar =
            tilingHintSubview->getParentOfType<scf::ParallelOp>())
      previousTilingScfPar = subViewParentPar;
    else
      continue;
  }
  if (!tilingHintSubview || !previousTilingScfPar)
    return failure();
  if (L1MemrefShape.size() < previousTilingScfPar.getStep().size())
    return failure();
  if (L2MemrefShape.size() < previousTilingScfPar.getStep().size())
    return failure();
  builder.setInsertionPointAfter(op);
  auto newTilingPar = builder.create<scf::ParallelOp>(
      loc, previousTilingScfPar.getLowerBound(),
      previousTilingScfPar.getUpperBound(), previousTilingScfPar.getStep());
  IRMapping remap;
  for (unsigned i = 0; i < previousTilingScfPar.getInductionVars().size(); i++)
    remap.map(previousTilingScfPar.getInductionVars()[i],
              newTilingPar.getInductionVars()[i]);
  // Generate memref subview op leading the tiling of the L1 memref
  builder.setInsertionPointToStart(newTilingPar.getBody());
  auto newL1Subview =
      dyn_cast<memref::SubViewOp>(builder.clone(*tilingHintSubview, remap));
  remap.map(L1Memref, newL1Subview.getResult());
  for (auto o : L1MemrefOpLog) {
    if (auto tr = dyn_cast<memref::TransposeOp>(o)) {
      memref::TransposeOp transposeOp = builder.create<memref::TransposeOp>(
          loc, newL1Subview.getResult(),
          AffineMapAttr::get(tr.getPermutation()));
      remap.map(tr.getResult(), transposeOp.getResult());
    } else
      assert(false && "NYI memref operation type on L1 memref");
  }
  // Generate memref subview op leading the tiling of the L2 memref
  SmallVector<int64_t> tilingFactors;
  for (unsigned i = 0; i < newTilingPar.getStep().size(); i++) {
    auto factor = llvm::divideCeilSigned(
        *getConstantIntValue(newTilingPar.getUpperBound()[i]) -
            *getConstantIntValue(newTilingPar.getLowerBound()[i]),
        *getConstantIntValue(newTilingPar.getStep()[i]));
    tilingFactors.push_back(factor);
  }
  Attribute zeroIdxAttr = builder.getIndexAttr(0);
  Attribute oneIdxAttr = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> L2Offsets(L2MemrefShape.size(), zeroIdxAttr);
  SmallVector<int> L2TiledShape = L2MemrefShape;
  // Tiling the L2 memref with the first two tilable dimensions. TODO:
  // generalize/replace this logic.
  int dimIndex = 0;
  for (unsigned i = 0; i < L2MemrefShape.size(); i++) {
    int stepSizeInInt = *getConstantIntValue(newTilingPar.getStep()[dimIndex]);
    if (L2MemrefShape[i] >= tilingFactors[dimIndex] * stepSizeInInt) {
      int applyFactor = llvm::divideCeilSigned(
          L2MemrefShape[i], tilingFactors[dimIndex] * stepSizeInInt);
      AffineExpr d0 = builder.getAffineDimExpr(0);
      AffineExpr mul = d0 * applyFactor;
      auto map = AffineMap::get(1, 0, mul);
      Value new_iv = builder.create<affine::AffineApplyOp>(
          loc, map, newTilingPar.getInductionVars()[dimIndex]);
      L2Offsets[i] = new_iv;
      L2TiledShape[i] =
          llvm::divideCeilSigned(L2MemrefShape[i], tilingFactors[dimIndex]);
      dimIndex++;
    }
    if (dimIndex >= 2)
      break;
  }
  SmallVector<OpFoldResult> L2Strides(L2MemrefShape.size(), oneIdxAttr);
  SmallVector<OpFoldResult> L2Sizes;
  for (unsigned i = 0; i < L2MemrefShape.size(); i++)
    L2Sizes.push_back(builder.getIndexAttr(L2TiledShape[i]));
  auto subviewOutputType =
      llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
          llvm::cast<MemRefType>(L2Memref.getType()), L2Offsets, L2Sizes,
          L2Strides));
  auto newL2Subview = builder.create<memref::SubViewOp>(
      loc, subviewOutputType, L2Memref, L2Offsets, L2Sizes, L2Strides);
  remap.map(L2Memref, newL2Subview.getResult());
  for (auto o : L2MemrefOpLog) {
    if (auto tr = dyn_cast<memref::TransposeOp>(o)) {
      memref::TransposeOp transposeOp = builder.create<memref::TransposeOp>(
          loc, newL2Subview.getResult(),
          AffineMapAttr::get(tr.getPermutation()));
      remap.map(tr.getResult(), transposeOp.getResult());
    } else
      assert(false && "NYI memref operation type on L1 memref");
  }
  builder.clone(*op, remap);
  op->erase();
  return success();
}

template <class T>
air::SegmentOp generateEmptySegmentOp(OpBuilder &rewriter, T op,
                                      air::LaunchOp launch) {
  SmallVector<Value, 1> segmentSizes = {};
  SmallVector<Value, 4> segmentOpers;
  for (Value v : launch.getIds()) {
    segmentOpers.push_back(v);
  }
  for (Value v : launch.getSize()) {
    segmentOpers.push_back(v);
  }
  for (Value v : launch.getKernelArguments()) {
    segmentOpers.push_back(v);
  }
  auto segment =
      rewriter.create<air::SegmentOp>(op->getLoc(), segmentSizes, segmentOpers);
  auto &bb = segment.getBody().front();
  auto ivs = op.getInductionVars();

  for (int i = 0, e = ivs.size(); i < e; i++) {
    ivs[i].replaceAllUsesWith(segment.getKernelArgument(i));
  }

  auto &body = op.getBody()->getOperations();
  bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
  rewriter.setInsertionPointToStart(&segment.getRegion().front());

  return segment;
}

class ScfParToLaunchConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToLaunchConversion(MLIRContext *ctx,
                           llvm::SmallSet<Operation *, 8> &filteredOps,
                           llvm::SmallSet<air::LaunchOp, 2> &replacementOps,
                           bool generateSegment)
      : OpRewritePattern(ctx), filteredOps(filteredOps),
        replacementOps(replacementOps), generateSegment(generateSegment){};

  LogicalResult matchAndRewrite(scf::ParallelOp parOp,
                                PatternRewriter &rewriter) const override {

    scf::ParallelOp op = parOp;

    if (!filteredOps.contains(op))
      return failure();

    if (failed(normalizeScfParallel(op, rewriter)))
      return failure();

    auto loc = op.getLoc();

    SmallVector<int, 4> bounds(op.getNumLoops(), 1);
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
    getUsedConstsAndArgsDefinedAbove(op.getRegion(), constants, args);

    SmallVector<Value, 4> sizes;
    for (auto b : bounds)
      sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, b));
    auto launch = rewriter.create<air::LaunchOp>(op.getLoc(), sizes, args);
    rewriter.setInsertionPointToStart(&launch.getRegion().front());

    if (generateSegment) {
      auto segment = generateEmptySegmentOp(rewriter, op, launch);
      replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                            segment.getRegion());
      int i = 0;
      auto kernel_args = segment.getKernelArguments();
      kernel_args = kernel_args.drop_front(
          launch.getIds().size() +
          launch.getSize().size()); // Launch's induction vars
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], segment.getRegion());
    } else {
      auto &bb = launch.getBody().front();
      auto ivs = op.getInductionVars();

      for (int i = 0, e = ivs.size(); i < e; i++) {
        ivs[i].replaceAllUsesWith(launch.getIds()[i]);
      }

      auto &body = op.getBody()->getOperations();
      bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
      replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                            launch.getRegion());
      int i = 0;
      auto kernel_args = launch.getKernelArguments();
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());
    }

    if (op != parOp)
      op.erase();
    rewriter.eraseOp(parOp);
    replacementOps.insert(launch);

    return success();
  }

private:
  llvm::SmallSet<Operation *, 8> &filteredOps;
  llvm::SmallSet<air::LaunchOp, 2> &replacementOps;
  bool generateSegment;
};

class ScfForallToLaunchConversion : public OpRewritePattern<scf::ForallOp> {
public:
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  ScfForallToLaunchConversion(MLIRContext *ctx,
                              llvm::SmallSet<Operation *, 8> &filteredOps,
                              llvm::SmallSet<air::LaunchOp, 2> &replacementOps,
                              bool generateSegment)
      : OpRewritePattern(ctx), filteredOps(filteredOps),
        replacementOps(replacementOps), generateSegment(generateSegment){};

  LogicalResult matchAndRewrite(scf::ForallOp forOp,
                                PatternRewriter &rewriter) const override {

    scf::ForallOp op = forOp;

    if (!filteredOps.contains(op))
      return failure();

    // if (failed(normalizeScfParallel(op, rewriter)))
    //   return failure();

    auto loc = op.getLoc();

    SmallVector<int, 4> bounds(op.getRank(), 1);
    for (unsigned int i = 0; i < op.getRank(); i++) {
      int64_t lb_int = op.getStaticLowerBound()[i];
      int64_t ub_int = op.getStaticUpperBound()[i];
      int64_t step_int = op.getStaticStep()[i];

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

    SmallVector<Value, 4> sizes;
    for (auto b : bounds)
      sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, b));
    auto launch = rewriter.create<air::LaunchOp>(op.getLoc(), sizes, args);

    rewriter.setInsertionPointToStart(&launch.getRegion().front());

    if (generateSegment) {
      auto segment = generateEmptySegmentOp(rewriter, op, launch);
      replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                            segment.getRegion());
      int i = 0;
      auto kernel_args = segment.getKernelArguments();
      kernel_args = kernel_args.drop_front(
          launch.getIds().size() +
          launch.getSize().size()); // Launch's induction vars
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], segment.getRegion());
    } else {
      auto &bb = launch.getBody().front();
      auto ivs = op.getInductionVars();

      for (int i = 0, e = ivs.size(); i < e; i++) {
        ivs[i].replaceAllUsesWith(launch.getIds()[i]);
      }

      auto &body = op.getBody()->getOperations();
      bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
      replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                            launch.getRegion());
      int i = 0;
      auto kernel_args = launch.getKernelArguments();
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());
    }

    if (op != forOp)
      op.erase();
    rewriter.eraseOp(forOp);
    replacementOps.insert(launch);

    return success();
  }

private:
  llvm::SmallSet<Operation *, 8> &filteredOps;
  llvm::SmallSet<air::LaunchOp, 2> &replacementOps;
  bool generateSegment;
};

/// Build a strided memref type by applying `permutationMap` tp `memRefType`.
static MemRefType inferTransposeResultType(MemRefType memRefType,
                                           AffineMap permutationMap) {
  auto rank = memRefType.getRank();
  auto originalSizes = memRefType.getShape();
  auto [originalStrides, offset] = getStridesAndOffset(memRefType);
  assert(originalStrides.size() == static_cast<unsigned>(rank));

  // Compute permuted sizes and strides.
  SmallVector<int64_t> sizes(rank, 0);
  SmallVector<int64_t> strides(rank, 1);
  for (const auto &en : llvm::enumerate(permutationMap.getResults())) {
    unsigned position = cast<AffineDimExpr>(en.value()).getPosition();
    sizes[en.index()] = originalSizes[position];
    strides[en.index()] = originalStrides[position];
  }

  return MemRefType::Builder(memRefType)
      .setShape(sizes)
      .setLayout(
          StridedLayoutAttr::get(memRefType.getContext(), offset, strides));
}

static SmallVector<Value, 4> extractStridesFromMemrefType(MemRefType memrefTy,
                                                          OpBuilder &builder) {
  // get the strides and offsets from the memref type
  SmallVector<Value, 4> strides;
  int64_t offset;
  SmallVector<int64_t, 4> layout_strides;
  auto successStrides = getStridesAndOffset(memrefTy, layout_strides, offset);
  if (failed(successStrides)) {
    llvm::outs() << "Failed to get strides\n";
    return strides;
  }

  for (auto s : layout_strides)
    strides.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));

  return strides;
}

static SmallVector<Value, 4> extractSizesFromMemrefType(MemRefType memrefTy,
                                                        OpBuilder &builder) {
  SmallVector<Value, 4> sizes;
  for (auto s : memrefTy.getShape())
    sizes.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));
  return sizes;
}

static void extractOffsetsFromSubview(memref::SubViewOp subview,
                                      OpBuilder &builder,
                                      SmallVector<Value, 4> &offsets) {
  auto subview_offsets = subview.getOffsets().begin();
  auto static_offsets = subview.getStaticOffsets();
  auto loc = subview.getLoc();

  for (auto o : static_offsets) {
    if (o >= 0)
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, o));
    else
      offsets.push_back(*subview_offsets++);
  }
}

static LogicalResult canonicalizeAIRDmaOperands(OpBuilder builder,
                                                SmallVector<Value, 4> &offsets,
                                                SmallVector<Value, 4> &sizes,
                                                SmallVector<Value, 4> &strides,
                                                MemRefType memref) {
  // Increase vector sizes up to memref size. When offsets, sizes and strides
  // are all empty, then it implies that the whole memref is accessed in the
  // default order.
  auto max_dim_size =
      std::max(std::max(offsets.size(), sizes.size()), strides.size());
  auto target_dim_size = std::max(max_dim_size, (size_t)memref.getRank());
  if (max_dim_size && offsets.size() < target_dim_size) {
    for (unsigned i = offsets.size(); i < target_dim_size; i++) {
      offsets.insert(offsets.begin(), builder.create<arith::ConstantIndexOp>(
                                          builder.getUnknownLoc(), 0));
    }
  }
  if (max_dim_size && sizes.size() < target_dim_size) {
    for (unsigned i = sizes.size(); i < target_dim_size; i++) {
      sizes.insert(sizes.begin(), builder.create<arith::ConstantIndexOp>(
                                      builder.getUnknownLoc(), 1));
    }
  }
  int memref_size = 1;
  for (auto size : memref.getShape())
    memref_size *= size;
  if (max_dim_size && strides.size() < target_dim_size) {
    for (unsigned i = strides.size(); i < target_dim_size; i++) {
      strides.insert(strides.begin(),
                     builder.create<arith::ConstantIndexOp>(
                         builder.getUnknownLoc(), memref_size));
    }
  }

  // Reduce highest dimensions if more than memref size
  while (strides.size() > target_dim_size && getConstantIntValue(strides[0]) &&
         *getConstantIntValue(strides[0]) == memref_size) {
    strides.erase(strides.begin());
  }
  while (sizes.size() > target_dim_size && getConstantIntValue(sizes[0]) &&
         *getConstantIntValue(sizes[0]) == 1) {
    sizes.erase(sizes.begin());
  }
  while (offsets.size() > std::min(sizes.size(), strides.size()) &&
         getConstantIntValue(offsets[0]) &&
         *getConstantIntValue(offsets[0]) == 0) {
    offsets.erase(offsets.begin());
  }

  if (offsets.size() != sizes.size() || sizes.size() != strides.size())
    return failure();

  return success();
}

static LogicalResult condenseMemrefDataReorderingToAIRDma(
    air::DmaMemcpyNdOp dmaOp, std::vector<Operation *> src_ancestor_memref_ops,
    std::vector<Operation *> dst_ancestor_memref_ops) {
  OpBuilder rewriter(dmaOp);
  auto src = dmaOp.getSrcMemref();
  auto dst = dmaOp.getDstMemref();
  auto loc = dmaOp->getLoc();

  // It must already be a memref
  auto src_type = llvm::dyn_cast<MemRefType>(src.getType());
  auto dst_type = llvm::dyn_cast<MemRefType>(dst.getType());
  if (!src_type)
    return failure();
  if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
    return failure();

  // Revert the vector of memref ops, as it was built with push_back.
  std::reverse(src_ancestor_memref_ops.begin(), src_ancestor_memref_ops.end());
  std::reverse(dst_ancestor_memref_ops.begin(), dst_ancestor_memref_ops.end());

  SmallVector<Value, 4> src_offsets, dst_offsets;
  SmallVector<Value, 4> src_strides, dst_strides;
  SmallVector<Value, 4> src_sizes, dst_sizes;
  SmallVector<Value, 4> empty;
  auto constZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  MemRefType src_memref_ty;
  if (!src_ancestor_memref_ops.empty()) {
    if (auto subviewOp =
            dyn_cast<memref::SubViewOp>(src_ancestor_memref_ops[0])) {
      // Init. offsets
      extractOffsetsFromSubview(subviewOp, rewriter, src_offsets);
      // Init. memref type
      src_memref_ty = subviewOp.getSourceType();
      src = subviewOp.getSource();
    } else if (auto transposeOp =
                   dyn_cast<memref::TransposeOp>(src_ancestor_memref_ops[0])) {
      // Init. memref type
      src_memref_ty = llvm::cast<MemRefType>(transposeOp.getIn().getType());
      src = transposeOp.getIn();
      // Init. offsets
      src_offsets.clear();
      for (unsigned i = 0; i < transposeOp.getPermutation().getNumInputs(); i++)
        src_offsets.push_back(constZero);
    }
  }
  MemRefType dst_memref_ty;
  if (!dst_ancestor_memref_ops.empty()) {
    if (auto subviewOp =
            dyn_cast<memref::SubViewOp>(dst_ancestor_memref_ops[0])) {
      // Init. offsets
      extractOffsetsFromSubview(subviewOp, rewriter, dst_offsets);
      // Init. memref type
      dst_memref_ty = subviewOp.getSourceType();
      dst = subviewOp.getSource();
    } else if (auto transposeOp =
                   dyn_cast<memref::TransposeOp>(dst_ancestor_memref_ops[0])) {
      // Init. memref type
      dst_memref_ty = llvm::cast<MemRefType>(transposeOp.getIn().getType());
      dst = transposeOp.getIn();
      // Init. offsets
      dst_offsets.clear();
      for (unsigned i = 0; i < transposeOp.getPermutation().getNumInputs(); i++)
        dst_offsets.push_back(constZero);
    }
  }

  for (auto memrefOp : src_ancestor_memref_ops) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      // Init. memref type
      src_memref_ty =
          inferTransposeResultType(src_memref_ty, transposeOp.getPermutation());
      // Init. offsets
      if (transposeOp.getPermutation().getNumInputs() != src_offsets.size())
        continue;
      src_offsets =
          applyPermutationMap<Value>(transposeOp.getPermutation(), src_offsets);
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      // Init. offsets
      for (int i = (int)expandShapeOp.getReassociationIndices().size() - 1;
           i >= 0; i--) {
        if (expandShapeOp.getReassociationIndices()[i].size() <= 1)
          continue;
        for (unsigned j = 1;
             j < expandShapeOp.getReassociationIndices()[i].size(); j++)
          src_offsets.insert(src_offsets.begin() + i,
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
      // Init. memref type
      FailureOr<MemRefType> compute_expand =
          memref::ExpandShapeOp::computeExpandedType(
              src_memref_ty, expandShapeOp.getResultType().getShape(),
              expandShapeOp.getReassociationIndices());
      if (failed(compute_expand)) {
        assert(false);
      } else {
        src_memref_ty = *compute_expand;
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      // Check if subview is rank reduced
      if (subviewOp.getSourceType().getRank() > subviewOp.getType().getRank())
        src_memref_ty = llvm::cast<MemRefType>(
            memref::SubViewOp::inferRankReducedResultType(
                subviewOp.getType().getShape(), src_memref_ty,
                subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
                subviewOp.getStaticStrides()));
      else
        src_memref_ty =
            llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
                src_memref_ty, subviewOp.getStaticOffsets(),
                subviewOp.getStaticSizes(), subviewOp.getStaticStrides()));
    }
  }

  for (auto memrefOp : dst_ancestor_memref_ops) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      // Init. memref type
      dst_memref_ty =
          inferTransposeResultType(dst_memref_ty, transposeOp.getPermutation());
      // Init. offsets
      if (transposeOp.getPermutation().getNumInputs() != dst_offsets.size())
        continue;
      dst_offsets =
          applyPermutationMap<Value>(transposeOp.getPermutation(), dst_offsets);
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      // Init. offsets
      for (int i = (int)expandShapeOp.getReassociationIndices().size() - 1;
           i >= 0; i--) {
        if (expandShapeOp.getReassociationIndices()[i].size() <= 1)
          continue;
        for (unsigned j = 1;
             j < expandShapeOp.getReassociationIndices()[i].size(); j++)
          dst_offsets.insert(dst_offsets.begin() + i,
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
      // Init. memref type
      FailureOr<MemRefType> compute_expand =
          memref::ExpandShapeOp::computeExpandedType(
              dst_memref_ty, expandShapeOp.getResultType().getShape(),
              expandShapeOp.getReassociationIndices());
      if (failed(compute_expand)) {
        assert(false);
      } else {
        dst_memref_ty = *compute_expand;
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      if (subviewOp.getSourceType().getRank() > subviewOp.getType().getRank())
        dst_memref_ty = llvm::cast<MemRefType>(
            memref::SubViewOp::inferRankReducedResultType(
                subviewOp.getType().getShape(), dst_memref_ty,
                subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
                subviewOp.getStaticStrides()));
      else
        dst_memref_ty =
            llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
                dst_memref_ty, subviewOp.getStaticOffsets(),
                subviewOp.getStaticSizes(), subviewOp.getStaticStrides()));
    }
  }

  if (src_ancestor_memref_ops.size()) {
    src_strides = extractStridesFromMemrefType(src_memref_ty, rewriter);
    src_sizes = extractSizesFromMemrefType(src_memref_ty, rewriter);
  }
  if (dst_ancestor_memref_ops.size()) {
    dst_strides = extractStridesFromMemrefType(dst_memref_ty, rewriter);
    dst_sizes = extractSizesFromMemrefType(dst_memref_ty, rewriter);
  }

  SmallVector<Value, 4> deps;
  SmallVector<Type, 4> tys;

  if (failed(canonicalizeAIRDmaOperands(
          rewriter, src_offsets, src_sizes, src_strides,
          llvm::cast<MemRefType>(src.getType()))) ||
      failed(canonicalizeAIRDmaOperands(
          rewriter, dst_offsets, dst_sizes, dst_strides,
          llvm::cast<MemRefType>(dst.getType())))) {
    assert(false);
  }
  auto new_dma = rewriter.create<xilinx::air::DmaMemcpyNdOp>(
      loc, tys, deps, dst, dst_offsets, dst_sizes, dst_strides, src,
      src_offsets, src_sizes, src_strides);

  assert(!new_dma.getSrcMemref().getDefiningOp<memref::TransposeOp>());
  assert(!new_dma.getDstMemref().getDefiningOp<memref::TransposeOp>());

  dmaOp->erase();

  return success();
}

struct CopyToDmaPass : public air::impl::CopyToDmaBase<CopyToDmaPass> {

  CopyToDmaPass() = default;
  CopyToDmaPass(const CopyToDmaPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           scf::SCFDialect, air::airDialect,
                           arith::ArithDialect, memref::MemRefDialect>();

    target.addLegalOp<affine::AffineApplyOp, affine::AffineForOp,
                      affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineYieldOp>();

    target.addDynamicallyLegalOp<memref::CopyOp>([](memref::CopyOp co) {
      auto src_type = llvm::dyn_cast<MemRefType>(co.getSource().getType());
      auto dst_type = llvm::dyn_cast<MemRefType>(co.getTarget().getType());
      return src_type.getMemorySpaceAsInt() == dst_type.getMemorySpaceAsInt();
    });

    DmaMemcpyOpID = 0;

    // Simplify all the subviews so we can rewrite them easily.
    // Mostly this is propagating constant sizes into dimensioned memref types.
    RewritePatternSet stage1Patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    memref::AllocOp::getCanonicalizationPatterns(stage1Patterns, context);
    memref::populateComposeSubViewPatterns(stage1Patterns, context);
    (void)applyPatternsAndFoldGreedily(module, std::move(stage1Patterns));

    RewritePatternSet stage2Patterns(context);
    stage2Patterns
        .insert<LinalgCopyToAIRDmaConversion, MemrefCopyToAIRDmaConversion>(
            context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(stage2Patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      module.dump();
      assert(0);
    }

    std::vector<Operation *> waits;
    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto wo = dyn_cast<affine::AffineDmaWaitOp>(op)) {
          auto memref = wo.getTagMemRef();
          for (auto u : memref.getUsers()) {
            waits.push_back(u);
          }
        }
      });
    }
    for (auto o : waits)
      o->erase();

    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    // Condense memref data pattern reordering ops, including memref.subview,
    // memref.tranpose and memref.expand_shape into air.dma_memcpy_nd op's
    // offsets, sizes and strides fields.
    auto scope = getOperation();
    std::vector<std::tuple<xilinx::air::DmaMemcpyNdOp, std::vector<Operation *>,
                           std::vector<Operation *>>>
        dma_ops;

    scope->walk([&](xilinx::air::DmaMemcpyNdOp dmaOp) {
      bool src_condense = false;
      if (auto src_defop = dmaOp.getSrcMemref().getDefiningOp()) {
        src_condense |= isa<memref::TransposeOp>(src_defop);
        src_condense |= isa<memref::ExpandShapeOp>(src_defop);
        src_condense |= isa<memref::SubViewOp>(src_defop);
      }
      bool dst_condense = false;
      if (auto dst_defop = dmaOp.getDstMemref().getDefiningOp()) {
        dst_condense |= isa<memref::TransposeOp>(dst_defop);
        dst_condense |= isa<memref::ExpandShapeOp>(dst_defop);
        dst_condense |= isa<memref::SubViewOp>(dst_defop);
      }
      if (src_condense || dst_condense) {
        // Fields in the tuple: (1) dma op, (2) list of memref ops producing the
        // src memref, and (3) list of memref ops producing the dst memref.
        std::tuple<air::DmaMemcpyNdOp, std::vector<Operation *>,
                   std::vector<Operation *>>
            log_entry;
        std::get<0>(log_entry) = dmaOp;
        if (src_condense) {
          Operation *ancestor = dmaOp.getSrcMemref().getDefiningOp();
          bool exit = false;
          while (ancestor && !exit) {
            if (auto transpose_anc = dyn_cast<memref::TransposeOp>(ancestor)) {
              std::get<1>(log_entry).push_back(ancestor);
              ancestor = transpose_anc.getIn().getDefiningOp();
            } else if (auto expand_anc =
                           dyn_cast<memref::ExpandShapeOp>(ancestor)) {
              std::get<1>(log_entry).push_back(ancestor);
              ancestor = expand_anc.getSrc().getDefiningOp();
            } else if (auto subview_anc =
                           dyn_cast<memref::SubViewOp>(ancestor)) {
              std::get<1>(log_entry).push_back(ancestor);
              ancestor = subview_anc.getSource().getDefiningOp();
            } else
              exit = true;
          }
        }
        if (dst_condense) {
          Operation *ancestor = dmaOp.getDstMemref().getDefiningOp();
          bool exit = false;
          while (ancestor && !exit) {
            if (auto transpose_anc = dyn_cast<memref::TransposeOp>(ancestor)) {
              std::get<2>(log_entry).push_back(ancestor);
              ancestor = transpose_anc.getIn().getDefiningOp();
            } else if (auto expand_anc =
                           dyn_cast<memref::ExpandShapeOp>(ancestor)) {
              std::get<2>(log_entry).push_back(ancestor);
              ancestor = expand_anc.getSrc().getDefiningOp();
            } else if (auto subview_anc =
                           dyn_cast<memref::SubViewOp>(ancestor)) {
              std::get<2>(log_entry).push_back(ancestor);
              ancestor = subview_anc.getSource().getDefiningOp();
            } else
              exit = true;
          }
        }
        dma_ops.push_back(log_entry);
      }
    });
    for (auto dmaOp : dma_ops) {
      if (failed(condenseMemrefDataReorderingToAIRDma(
              std::get<0>(dmaOp), std::get<1>(dmaOp), std::get<2>(dmaOp)))) {
        return signalPassFailure();
      }
    }
  }
};

static void getHerdNames(ModuleOp module) {
  std::vector<std::string> herd_syms;
  for (auto f : module.getOps<func::FuncOp>()) {
    // record existing symbol names
    SmallVector<air::HerdOp> herds;
    f.walk([&](air::HerdOp op) {
      if (auto attr =
              op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
        std::string name = attr.getValue().str();
        if (std::find(herd_syms.begin(), herd_syms.end(), name) ==
            herd_syms.end())
          herd_syms.push_back(name);
      }
      herds.push_back(op);
    });
    // generate shared symbol name across herds using shared L1 memref as
    // argument; herds sharing the same symbolic name represent different time
    // phases of the same physical herd.
    if (herds.size() > 1)
      for (unsigned i = 0; i < herds.size() - 1; i++) {
        for (unsigned j = i + 1; j < herds.size(); j++) {
          auto herdI = herds[i];
          auto herdJ = herds[j];
          for (auto operI : herdI->getOperands()) {
            for (auto operJ : herdJ->getOperands()) {
              if (!isa<MemRefType>(operI.getType()))
                continue;
              if (!isa<MemRefType>(operJ.getType()))
                continue;
              if (llvm::cast<MemRefType>(operI.getType())
                      .getMemorySpaceAsInt() != (int)air::MemorySpace::L1)
                continue;
              if (llvm::cast<MemRefType>(operJ.getType())
                      .getMemorySpaceAsInt() != (int)air::MemorySpace::L1)
                continue;
              if (operI != operJ)
                continue;

              std::string name;
              if (auto attr = herdI->getAttrOfType<StringAttr>(
                      SymbolTable::getSymbolAttrName()))
                name = attr.getValue().str();
              else if (auto attr = herdJ->getAttrOfType<StringAttr>(
                           SymbolTable::getSymbolAttrName()))
                name = attr.getValue().str();
              else {
                unsigned id = 0;
                do {
                  std::stringstream ss;
                  ss << "herd_" << id++;
                  name = ss.str();
                } while (std::find(herd_syms.begin(), herd_syms.end(), name) !=
                         herd_syms.end());
              }
              herdI->setAttr(SymbolTable::getSymbolAttrName(),
                             StringAttr::get(herdI->getContext(), name));
              herdJ->setAttr(SymbolTable::getSymbolAttrName(),
                             StringAttr::get(herdJ->getContext(), name));
              herd_syms.push_back(name);
            }
          }
        }
      }
    // generate missing symbol names
    f.walk([&](air::HerdOp op) {
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
}

static void getSegmentNames(ModuleOp module) {
  std::vector<std::string> seg_syms;
  for (auto f : module.getOps<func::FuncOp>()) {
    // record existing symbol names
    f.walk([&](air::SegmentOp op) {
      if (auto attr =
              op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
        std::string name = attr.getValue().str();
        assert((std::find(seg_syms.begin(), seg_syms.end(), name) ==
                seg_syms.end()) &&
               "unexpected duplicate symbol");
        seg_syms.push_back(name);
      }
    });
    // generate missing symbol names
    f.walk([&](air::SegmentOp op) {
      if (!op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
        unsigned id = 0;
        std::string name;
        do {
          std::stringstream ss;
          assert(
              f->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()) &&
              "enclosing function of air.sgement op expected to have a symbol "
              "name");
          ss << f->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .str()
             << "_" << id++;
          name = ss.str();
        } while (std::find(seg_syms.begin(), seg_syms.end(), name) !=
                 seg_syms.end());
        seg_syms.push_back(name);
        op->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(op->getContext(), name));
      }
    });
  }
}

struct ParallelToHerdPass
    : public air::impl::ParallelToHerdBase<ParallelToHerdPass> {

  ParallelToHerdPass() = default;
  ParallelToHerdPass(const ParallelToHerdPass &pass) {}
  ParallelToHerdPass(const ::xilinx::air::ParallelToHerdOptions &options)
      : ParallelToHerdBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    // Ensure that air.dma_memcpy_nd ops between L1 and L2 are within at least
    // two parent scf.parallel loops.
    module.walk([&](air::DmaMemcpyNdOp op) {
      auto srcMemrefTy = llvm::cast<MemRefType>(op.getSrcMemref().getType());
      auto dstMemrefTy = llvm::cast<MemRefType>(op.getDstMemref().getType());
      Value L1Memref = nullptr;
      Value L2Memref = nullptr;
      bool SrcIsL1 = false;
      if ((srcMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L1) &&
          (dstMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L2)) {
        L1Memref = op.getSrcMemref();
        L2Memref = op.getDstMemref();
        SrcIsL1 = true;
      } else if ((srcMemrefTy.getMemorySpaceAsInt() ==
                  (int)air::MemorySpace::L2) &&
                 (dstMemrefTy.getMemorySpaceAsInt() ==
                  (int)air::MemorySpace::L1)) {
        L1Memref = op.getDstMemref();
        L2Memref = op.getSrcMemref();
        SrcIsL1 = false;
      } else
        return;
      // L2-side dma data access pattern needs to be the default. Otherwise,
      // NYI.
      if (SrcIsL1 && (!op.getDstOffsets().empty()))
        return;
      if ((!SrcIsL1) && (!op.getSrcOffsets().empty()))
        return;
      // Check if the memcpy op has at least two parent scf.parallel loops.
      int parentParOpCount = 0;
      Operation *parentParOp = op;
      while (parentParOp->getParentOfType<scf::ParallelOp>()) {
        parentParOp = parentParOp->getParentOfType<scf::ParallelOp>();
        parentParOpCount++;
      }
      parentParOp = op;
      while (parentParOp->getParentOfType<scf::ForallOp>()) {
        parentParOp = parentParOp->getParentOfType<scf::ForallOp>();
        parentParOpCount++;
      }
      if (parentParOpCount > 1)
        return;
      if (!parentParOpCount)
        return;
      if (TileL1L2AIRMemcpyUsingScfParallel(op, SrcIsL1).failed())
        return;
    });

    llvm::SmallVector<Operation *> hierOps;
    module.walk([&](air::HierarchyInterface op) { hierOps.push_back(op); });

    SmallPtrSet<Operation *, 8> filteredOps;
    llvm::SmallSet<air::HerdOp, 2> replacementOps;
    module.walk([&](Operation *op) {
      if (!isa<scf::ForallOp, scf::ParallelOp, affine::AffineParallelOp>(op))
        return;
      // skip parallel op already inside herd
      if (op->getParentOfType<air::HerdOp>())
        return;
      // skip parallel ops already containing herd/segment/launch
      for (auto &h : hierOps)
        if (op->isProperAncestor(h))
          return;
      if (clAssignDepth < 0) {
        filteredOps.insert(op);
        return;
      }
      // the number of nested scf.parallel above this one
      int parallel_depth = 0;
      Operation *par = op;
      while ((par = par->getParentOp()))
        if (isa<scf::ForallOp, scf::ParallelOp, affine::AffineParallelOp>(par))
          parallel_depth++;
      if (parallel_depth != clAssignDepth)
        return;
      filteredOps.insert(op);
    });

    RewritePatternSet patterns(context);
    patterns.add<AffineParToHerdConversion>(context);
    patterns.add<ScfParToHerdConversion>(context, filteredOps, replacementOps,
                                         clFirstDim);
    patterns.add<ScfForallToHerdConversion>(context, filteredOps,
                                            replacementOps, clFirstDim);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           air::airDialect, arith::ArithDialect>();

    target.addLegalOp<affine::AffineApplyOp, affine::AffineForOp,
                      affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineYieldOp, scf::YieldOp, scf::ReduceOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });
    target.addDynamicallyLegalOp<scf::ForallOp>(
        [&](scf::ForallOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }

    getHerdNames(module);
    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }
};

struct ParallelToLaunchPass
    : public air::impl::ParallelToLaunchBase<ParallelToLaunchPass> {

  ParallelToLaunchPass() = default;
  ParallelToLaunchPass(const ParallelToLaunchPass &pass) {}
  ParallelToLaunchPass(const xilinx::air::ParallelToLaunchOptions &options)
      : ParallelToLaunchBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    llvm::SmallVector<air::LaunchOp> launchOps;
    module.walk([&](air::LaunchOp op) { launchOps.push_back(op); });

    llvm::SmallSet<Operation *, 8> filteredOps;
    llvm::SmallSet<air::LaunchOp, 2> replacementOps;
    module.walk([&](Operation *op) {
      if (!isa<scf::ForallOp, scf::ParallelOp>(op))
        return;
      if (op->getParentOfType<air::HerdOp>())
        return;
      if (op->getParentOfType<air::LaunchOp>())
        return;
      for (auto &l : launchOps)
        if (op->isProperAncestor(l))
          return;
      if (clAssignDepth < 0) {
        filteredOps.insert(op);
        return;
      }
      // the number of nested scf.parallel above this one
      int parallel_depth = 0;
      Operation *par = op;
      while ((par = par->getParentOp()))
        if (isa<scf::ForallOp, scf::ParallelOp>(par))
          parallel_depth++;
      if (parallel_depth != clAssignDepth)
        return;
      filteredOps.insert(op);
    });

    // if any L1 or L2 memref has lifetime only within the scf.par/forall
    // region, but allocated outside, then attempt to move them into the region.
    std::map<memref::AllocOp, SmallVector<Block *>> allocToParBodyMap;
    SmallVector<memref::AllocOp> globalAllocs;
    for (auto op : filteredOps) {
      Block *opBlk = nullptr;
      if (auto par = dyn_cast<scf::ParallelOp>(op))
        opBlk = par.getBody();
      else if (auto forall = dyn_cast<scf::ForallOp>(op))
        opBlk = forall.getBody();
      if (!opBlk)
        continue;
      llvm::SetVector<Value> regionArgs;
      getUsedValuesDefinedAbove(*opBlk->getParent(), regionArgs);
      for (auto arg : regionArgs) {
        if (auto allocOp = arg.getDefiningOp<memref::AllocOp>()) {
          allocToParBodyMap[allocOp].push_back(opBlk);
          if (std::find(globalAllocs.begin(), globalAllocs.end(), allocOp) ==
              globalAllocs.end())
            globalAllocs.push_back(allocOp);
        }
      }
    }
    // check for memref lifetime
    for (auto allocOp : globalAllocs) {
      if (allocToParBodyMap[allocOp].size() != 1)
        continue;
      auto blk = allocToParBodyMap[allocOp][0];
      memref::DeallocOp dealloc = nullptr;
      bool hasExtraUsers = false;
      for (auto user : allocOp.getMemref().getUsers()) {
        if (!blk->getParentOp()->isProperAncestor(user)) {
          if (auto da = dyn_cast<memref::DeallocOp>(user))
            dealloc = da;
          else
            hasExtraUsers = true;
        }
      }
      if (hasExtraUsers)
        continue;
      OpBuilder builder(allocOp);
      builder.setInsertionPointToStart(blk);
      auto newAlloc = dyn_cast<memref::AllocOp>(builder.clone(*allocOp));
      allocOp.getMemref().replaceAllUsesWith(newAlloc.getMemref());
      allocOp->erase();
      if (dealloc) {
        if (auto term = blk->getTerminator()) {
          builder.setInsertionPoint(term);
          builder.clone(*dealloc);
        } else {
          builder.setInsertionPointToEnd(blk);
          builder.clone(*dealloc);
        }
        dealloc->erase();
      }
    }

    RewritePatternSet patterns(context);
    patterns.add<ScfParToLaunchConversion>(context, filteredOps, replacementOps,
                                           clHasSegment);
    patterns.add<ScfForallToLaunchConversion>(context, filteredOps,
                                              replacementOps, clHasSegment);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           air::airDialect, arith::ArithDialect>();

    target.addLegalOp<affine::AffineApplyOp, affine::AffineForOp,
                      affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineYieldOp, scf::YieldOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });
    target.addDynamicallyLegalOp<scf::ForallOp>(
        [&](scf::ForallOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    getSegmentNames(module);
    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }
};

struct InsertEmptyLaunchOverHerdPass
    : public air::impl::InsertEmptyLaunchOverHerdBase<
          InsertEmptyLaunchOverHerdPass> {

  InsertEmptyLaunchOverHerdPass() = default;
  InsertEmptyLaunchOverHerdPass(const InsertEmptyLaunchOverHerdPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    module.getContext();

    module.walk([&](air::HerdOp op) {
      if (!op->getParentOfType<air::LaunchOp>())
        InsertEmptyLaunchOverHerd(op);
      else if (!op->getParentOfType<air::SegmentOp>())
        InsertEmptyLaunchOverHerd(op);
    });
    getSegmentNames(module);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ParToHerdOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ParToHerdOp::applyToOne(transform::TransformRewriter &rewriter,
                                   scf::ParallelOp target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  auto ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  llvm::SmallSet<air::HerdOp, 2> herdOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<ScfParToHerdConversion>(ctx, filteredOps, herdOps,
                                       getFirstDim());
  patterns.add<ScfForallToHerdConversion>(ctx, filteredOps, herdOps,
                                          getFirstDim());
  (void)applyPatternsAndFoldGreedily(
      target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));
  for (auto h : herdOps) {
    getHerdNames(h->getParentOfType<ModuleOp>());
    results.push_back(h);
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ParToLaunchOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ParToLaunchOp::applyToOne(transform::TransformRewriter &rewriter,
                                     scf::ParallelOp target,
                                     transform::ApplyToEachResultList &results,
                                     transform::TransformState &state) {
  auto ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  llvm::SmallSet<air::LaunchOp, 2> launchOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<ScfParToLaunchConversion>(ctx, filteredOps, launchOps,
                                         getHasAirSegment());
  patterns.add<ScfForallToLaunchConversion>(ctx, filteredOps, launchOps,
                                            getHasAirSegment());
  (void)applyPatternsAndFoldGreedily(
      target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));
  for (auto l : launchOps)
    results.push_back(l);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// CopyToDmaOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CopyToDmaOp::applyToOne(transform::TransformRewriter &rewriter,
                                   memref::CopyOp op,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  auto res = matchAndRewriteCopyOp(op, rewriter);
  if (failed(res))
    return emitDefaultDefiniteFailure(op);
  results.push_back(*res);
  return DiagnosedSilenceableFailure::success();
}

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createParallelToHerdPass() {
  return std::make_unique<ParallelToHerdPass>();
}
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createParallelToHerdPass(const ParallelToHerdOptions &options) {
  return std::make_unique<ParallelToHerdPass>(options);
}

std::unique_ptr<mlir::Pass> createParallelToLaunchPass() {
  return std::make_unique<ParallelToLaunchPass>();
}
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createParallelToLaunchPass(const ParallelToLaunchOptions &options) {
  return std::make_unique<ParallelToLaunchPass>(options);
}

std::unique_ptr<mlir::Pass> createCopyToDmaPass() {
  return std::make_unique<CopyToDmaPass>();
}

std::unique_ptr<mlir::Pass> createInsertEmptyLaunchOverHerdPass() {
  return std::make_unique<InsertEmptyLaunchOverHerdPass>();
}

} // namespace air
} // namespace xilinx
