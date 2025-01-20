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
#include "air/Transform/AIRDependencyScheduleOpt.h"
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
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
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

  if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
    return failure();

  SmallVector<Value, 4> src_offsets, dst_offsets;
  SmallVector<Value, 4> src_strides, dst_strides;
  SmallVector<Value, 4> src_sizes, dst_sizes;
  auto extractOperandsFromSubview = [&](memref::SubViewOp subview,
                                        auto &offsets, auto &sizes,
                                        auto &strides) {
    auto subview_offsets = subview.getOffsets().begin();
    auto static_offsets = subview.getStaticOffsets();
    auto static_sizes = subview.getStaticSizes();
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
      sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
    for (auto s : layout_strides)
      strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, s));
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
    if (isa_and_present<arith::ConstantOp>(v.getDefiningOp()))
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

// Given an scf.parallel loop with N dimensions, return a pair of outer- and
// inner-scf.parallel loops, such that the inner loop has `innerNumLoops` loops,
// and the outer loop has `N - innerNumLoops` loops.
FailureOr<std::pair<scf::ParallelOp, scf::ParallelOp>>
separateScfParallel(scf::ParallelOp op, unsigned innerNumLoops,
                    OpBuilder &builder) {
  if (innerNumLoops >= op.getNumLoops())
    return failure();
  auto loc = op->getLoc();

  unsigned outerNumLoops = op.getNumLoops() - innerNumLoops;
  SmallVector<Value, 2> outerLowerBounds, outerUpperBounds, outerSteps;
  SmallVector<Value, 2> innerLowerBounds, innerUpperBounds, innerSteps;

  for (unsigned i = 0, e = outerNumLoops; i < e; ++i) {
    outerLowerBounds.push_back(op.getLowerBound()[i]);
    outerUpperBounds.push_back(op.getUpperBound()[i]);
    outerSteps.push_back(op.getStep()[i]);
  }
  scf::ParallelOp outerLoop = builder.create<scf::ParallelOp>(
      loc, outerLowerBounds, outerUpperBounds, outerSteps);
  for (unsigned i = 0, e = outerNumLoops; i < e; ++i)
    op.getInductionVars()[i].replaceAllUsesWith(
        outerLoop.getInductionVars()[i]);
  for (unsigned i = outerNumLoops, e = op.getNumLoops(); i < e; ++i) {
    innerLowerBounds.push_back(op.getLowerBound()[i]);
    innerUpperBounds.push_back(op.getUpperBound()[i]);
    innerSteps.push_back(op.getStep()[i]);
  }
  builder.setInsertionPointToStart(outerLoop.getBody());
  scf::ParallelOp innerLoop = builder.create<scf::ParallelOp>(
      loc, innerLowerBounds, innerUpperBounds, innerSteps);
  for (unsigned i = outerNumLoops, e = op.getNumLoops(); i < e; ++i)
    op.getInductionVars()[i].replaceAllUsesWith(
        innerLoop.getInductionVars()[i - outerNumLoops]);
  auto &body = op.getBody()->getOperations();
  innerLoop.getBody()->getOperations().splice(innerLoop.getBody()->begin(),
                                              body, body.begin(), --body.end());
  return std::make_pair(outerLoop, innerLoop);
}

template <typename hierTy>
FailureOr<hierTy> ScfParToAIRHierarchyConversionImpl(
    scf::ParallelOp parOp, SmallPtrSet<Operation *, 8> &filteredOps,
    int firstDim, int fixedNumLoops, PatternRewriter &rewriter) {
  scf::ParallelOp op = parOp;

  if (fixedNumLoops > 0 && firstDim > fixedNumLoops) {
    parOp->emitOpError("firstDim exceeds fixedNumLoops.");
    return failure();
  }

  if (!filteredOps.contains(op))
    return failure();

  if (failed(normalizeScfParallel(op, rewriter)))
    return failure();

  auto loc = op.getLoc();

  // If given a positive `fixedNumLoops`, separate the scf.parallel by
  // `fixedNumLoops`.
  if (fixedNumLoops > 0 && op.getNumLoops() > (unsigned)fixedNumLoops) {
    auto parPair = separateScfParallel(op, (unsigned)fixedNumLoops, rewriter);
    if (failed(parPair))
      return failure();
    op = parPair->second; // Assign op to be the inner scf.parallel.
  }

  int newHierNumLoops = fixedNumLoops > 0 ? fixedNumLoops : op.getNumLoops();
  SmallVector<int, 2> bounds(newHierNumLoops, 1);
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
  SmallVector<int> ids;
  for (int i = 0; i < newHierNumLoops; i++) {
    ids.push_back((firstDim + i) % newHierNumLoops);
  }
  SmallVector<Value, 2> dims;
  for (auto id : ids)
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, bounds[id]));
  auto hierOp = rewriter.create<hierTy>(op.getLoc(), dims, args);
  auto &body = op.getBody()->getOperations();
  if (auto herdOp = dyn_cast<air::HerdOp>(hierOp.getOperation()))
    propagateLinkWith(op, herdOp);
  auto &bb = hierOp.getBody().front();
  auto ivs = op.getInductionVars();
  for (unsigned i = 0; i < op.getNumLoops(); i++) {
    ivs[i].replaceAllUsesWith(hierOp.getIds()[ids[i]]);
  }
  bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
  rewriter.setInsertionPointToStart(&hierOp.getRegion().front());
  replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                        hierOp.getRegion());
  int arg_idx = 0;
  auto kernel_args = hierOp.getKernelArguments();
  for (Value v : args)
    replaceAllUsesInRegionWith(v, kernel_args[arg_idx++], hierOp.getRegion());
  if (op != parOp)
    rewriter.eraseOp(op);
  return hierOp;
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
    auto res = ScfParToAIRHierarchyConversionImpl<air::HerdOp>(
        parOp, filteredOps, firstDim, /*fixedNumLoops*/ 2, rewriter);
    if (failed(res)) {
      return failure();
    }
    air::HerdOp herdOp = *res;
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

FailureOr<air::SegmentOp> insertAIRSegmentOpAroundRegion(OpBuilder &rewriter,
                                                         Region *region) {
  if (!region->hasOneBlock())
    return failure();
  SmallVector<Value, 1> segmentSizes = {};
  SmallVector<Value, 4> segmentOpers;
  for (Value v : region->getArguments())
    segmentOpers.push_back(v);
  rewriter.setInsertionPointToStart(&region->front());
  auto segment = rewriter.create<air::SegmentOp>(rewriter.getUnknownLoc(),
                                                 segmentSizes, segmentOpers);
  auto &bb = segment.getBody().front();
  auto &body = region->front().getOperations();
  bb.getOperations().splice(bb.begin(), body, ++body.begin(), --body.end());
  for (int i = 0, e = segmentOpers.size(); i < e; i++)
    replaceAllUsesInRegionWith(segmentOpers[i], segment.getKernelArgument(i),
                               segment.getBody());
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

    auto res = ScfParToAIRHierarchyConversionImpl<air::LaunchOp>(
        parOp, filteredOps, /*firstDim*/ 0, /*fixedNumLoops*/ -1, rewriter);
    if (failed(res)) {
      return failure();
    }
    air::LaunchOp launchOp = *res;
    if (generateSegment) {
      auto segment =
          insertAIRSegmentOpAroundRegion(rewriter, &launchOp.getBody());
      if (failed(segment))
        return failure();
    }
    rewriter.eraseOp(parOp);
    replacementOps.insert(launchOp);
    return success();
  }

private:
  llvm::SmallSet<Operation *, 8> &filteredOps;
  llvm::SmallSet<air::LaunchOp, 2> &replacementOps;
  bool generateSegment;
};

class ScfParToSegmentConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToSegmentConversion(MLIRContext *ctx,
                            llvm::SmallSet<Operation *, 8> &filteredOps,
                            llvm::SmallSet<air::SegmentOp, 2> &replacementOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps),
        replacementOps(replacementOps){};

  LogicalResult matchAndRewrite(scf::ParallelOp parOp,
                                PatternRewriter &rewriter) const override {

    auto res = ScfParToAIRHierarchyConversionImpl<air::SegmentOp>(
        parOp, filteredOps, /*firstDim*/ 0, /*fixedNumLoops*/ -1, rewriter);
    if (failed(res)) {
      return failure();
    }
    air::SegmentOp segmentOp = *res;
    rewriter.eraseOp(parOp);
    replacementOps.insert(segmentOp);
    return success();
  }

private:
  llvm::SmallSet<Operation *, 8> &filteredOps;
  llvm::SmallSet<air::SegmentOp, 2> &replacementOps;
};

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
    (void)applyPatternsGreedily(module, std::move(stage1Patterns));

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

    RewritePatternSet pattern(context);
    air::DmaMemcpyNdOp::getCanonicalizationPatterns(pattern, context);
    (void)applyPatternsGreedily(module, std::move(pattern));
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

// Convert forall to parallel in filtered ops
LogicalResult
ConvertForallToParallelInFilteredOps(SmallPtrSet<Operation *, 8> &filteredOps,
                                     mlir::MLIRContext *context) {
  IRRewriter rewriter(context);
  SmallVector<Operation *> fErased, fAdded;
  for (auto op : filteredOps) {
    auto forall = dyn_cast<scf::ForallOp>(op);
    if (!forall)
      continue;
    scf::ParallelOp newPar;
    fErased.push_back(op);
    if (failed(forallToParallelLoop(rewriter, forall, &newPar)))
      return failure();
    fAdded.push_back(newPar);
  }
  for (auto e : fErased)
    assert(filteredOps.erase(e));
  filteredOps.insert(fAdded.begin(), fAdded.end());
  return success();
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
      if (llvm::any_of(hierOps,
                       [op](Operation *h) { return op->isProperAncestor(h); }))
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

    // Convert forall to parallel in filtered ops
    if (failed(ConvertForallToParallelInFilteredOps(filteredOps, context)))
      signalPassFailure();

    RewritePatternSet patterns(context);
    patterns.add<AffineParToHerdConversion>(context);
    patterns.add<ScfParToHerdConversion>(context, filteredOps, replacementOps,
                                         clFirstDim);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           air::airDialect, arith::ArithDialect>();

    target.addLegalOp<affine::AffineApplyOp, affine::AffineForOp,
                      affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineYieldOp, scf::YieldOp, scf::ReduceOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
    getHerdNames(module);

    // Postprocessing: fuse allocs and deallocs into air.hierarchy, if their
    // memref is never used outside.
    RewritePatternSet postProcPatterns(context);
    air::populateAIRFuseAllocDeallocToAIRHierPatterns(postProcPatterns);
    (void)applyPatternsGreedily(module, std::move(postProcPatterns));

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
      if (llvm::any_of(launchOps, [op](air::LaunchOp l) {
            return op->isProperAncestor(l);
          }))
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

    // Convert forall to parallel in filtered ops
    if (failed(ConvertForallToParallelInFilteredOps(filteredOps, context)))
      signalPassFailure();

    RewritePatternSet patterns(context);
    patterns.add<ScfParToLaunchConversion>(context, filteredOps, replacementOps,
                                           clHasSegment);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           air::airDialect, arith::ArithDialect>();

    target.addLegalOp<affine::AffineApplyOp, affine::AffineForOp,
                      affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineYieldOp, scf::YieldOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }
    getSegmentNames(module);

    // Postprocessing: fuse allocs and deallocs into air.hierarchy, if their
    // memref is never used outside.
    RewritePatternSet postProcPatterns(context);
    air::populateAIRFuseAllocDeallocToAIRHierPatterns(postProcPatterns);
    (void)applyPatternsGreedily(module, std::move(postProcPatterns));

    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }
};

struct ParallelToSegmentPass
    : public air::impl::ParallelToSegmentBase<ParallelToSegmentPass> {

  ParallelToSegmentPass() = default;
  ParallelToSegmentPass(const ParallelToSegmentPass &pass) {}
  ParallelToSegmentPass(const xilinx::air::ParallelToSegmentOptions &options)
      : ParallelToSegmentBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    llvm::SmallVector<air::SegmentOp> segmentOps;
    module.walk([&](air::SegmentOp op) { segmentOps.push_back(op); });

    llvm::SmallSet<Operation *, 8> filteredOps;
    llvm::SmallSet<air::SegmentOp, 2> replacementOps;
    module.walk([&](Operation *op) {
      if (!isa<scf::ForallOp, scf::ParallelOp>(op))
        return;
      if (op->getParentOfType<air::HerdOp>())
        return;
      if (op->getParentOfType<air::SegmentOp>())
        return;
      if (llvm::any_of(segmentOps, [op](air::SegmentOp s) {
            return op->isProperAncestor(s);
          }))
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

    // Convert forall to parallel in filtered ops
    if (failed(ConvertForallToParallelInFilteredOps(filteredOps, context)))
      signalPassFailure();

    RewritePatternSet patterns(context);
    patterns.add<ScfParToSegmentConversion>(context, filteredOps,
                                            replacementOps);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           air::airDialect, arith::ArithDialect>();

    target.addLegalOp<affine::AffineApplyOp, affine::AffineForOp,
                      affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineYieldOp, scf::YieldOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }
    getSegmentNames(module);

    // Postprocessing: fuse allocs and deallocs into air.hierarchy, if their
    // memref is never used outside.
    RewritePatternSet postProcPatterns(context);
    air::populateAIRFuseAllocDeallocToAIRHierPatterns(postProcPatterns);
    (void)applyPatternsGreedily(module, std::move(postProcPatterns));

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
  (void)applyPatternsGreedily(
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
  (void)applyPatternsGreedily(
      target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));
  for (auto l : launchOps)
    results.push_back(l);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ParToSegmentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ParToSegmentOp::applyToOne(transform::TransformRewriter &rewriter,
                                      scf::ParallelOp target,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  auto ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  llvm::SmallSet<air::SegmentOp, 2> segmentOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<ScfParToSegmentConversion>(ctx, filteredOps, segmentOps);
  (void)applyPatternsGreedily(
      target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));
  for (auto s : segmentOps)
    results.push_back(s);
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

std::unique_ptr<mlir::Pass> createParallelToSegmentPass() {
  return std::make_unique<ParallelToSegmentPass>();
}
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createParallelToSegmentPass(const ParallelToSegmentOptions &options) {
  return std::make_unique<ParallelToSegmentPass>(options);
}

std::unique_ptr<mlir::Pass> createCopyToDmaPass() {
  return std::make_unique<CopyToDmaPass>();
}

std::unique_ptr<mlir::Pass> createInsertEmptyLaunchOverHerdPass() {
  return std::make_unique<InsertEmptyLaunchOverHerdPass>();
}

} // namespace air
} // namespace xilinx
