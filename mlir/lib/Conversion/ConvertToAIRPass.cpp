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
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
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

#define DEBUG_TYPE "convert-to-air"

namespace xilinx {
namespace air {

static std::atomic<uint64_t> DmaMemcpyOpID;

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
      inferredType.getStridesAndOffset(layout_strides, offset);
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

static void extractOperandsFromReinterpretCast(
    memref::ReinterpretCastOp reinterpretCast, OpBuilder &builder,
    SmallVector<Value, 4> &offsets, SmallVector<Value, 4> &sizes,
    SmallVector<Value, 4> &strides) {
  auto reinterpretCast_offsets = reinterpretCast.getOffsets().begin();
  auto static_offsets = reinterpretCast.getStaticOffsets();
  auto static_sizes = reinterpretCast.getStaticSizes();
  auto loc = reinterpretCast.getLoc();

  // Fixup an issue in the reinterpretCast output memref's strided layout giving
  // false dynamic strides
  auto constifyStridesInStridedLayout =
      [](MemRefType rankedMemRefType,
         memref::ReinterpretCastOp reinterpretCast) {
        StridedLayoutAttr stridedLayout =
            dyn_cast<StridedLayoutAttr>(rankedMemRefType.getLayout());
        SmallVector<int64_t> correctedStaticStrides(
            stridedLayout.getStrides().size(), 0);
        for (auto [index, stride] :
             llvm::enumerate(reinterpretCast.getMixedStrides())) {
          if (auto constStride = getConstantIntValue(stride))
            correctedStaticStrides[index] = *constStride;
          else
            correctedStaticStrides[index] = stridedLayout.getStrides()[index];
        }
        auto correctedStridedLayout = StridedLayoutAttr::get(
            reinterpretCast->getContext(), stridedLayout.getOffset(),
            ArrayRef(correctedStaticStrides));
        return MemRefType::Builder(rankedMemRefType)
            .setShape(rankedMemRefType.getShape())
            .setLayout(correctedStridedLayout);
      };

  MemRefType reinterpretCastType = constifyStridesInStridedLayout(
      reinterpretCast.getType(), reinterpretCast);

  // get the strides and offsets from the memref type
  int64_t offset;
  SmallVector<int64_t, 4> layout_strides;
  auto successStrides =
      reinterpretCastType.getStridesAndOffset(layout_strides, offset);
  if (failed(successStrides)) {
    llvm::outs() << "Failed to get strides\n";
    return; // failure();
  }

  for (auto o : static_offsets) {
    if (o >= 0)
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, o));
    else
      offsets.push_back(*reinterpretCast_offsets++);
  }
  for (auto s : static_sizes)
    sizes.push_back(builder.create<arith::ConstantIndexOp>(loc, s));
  for (auto s : layout_strides)
    strides.push_back(builder.create<arith::ConstantIndexOp>(loc, s));
  while (offsets.size() < sizes.size())
    offsets.insert(offsets.begin(),
                   builder.create<arith::ConstantIndexOp>(loc, 0));
}

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

  if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
    extractOperandsFromSubview(subview, rewriter, src_offsets, src_sizes,
                               src_strides);
    src = subview.getSource();
  } else if (auto reinterpretCast =
                 src.getDefiningOp<memref::ReinterpretCastOp>()) {
    extractOperandsFromReinterpretCast(reinterpretCast, rewriter, src_offsets,
                                       src_sizes, src_strides);
    src = reinterpretCast.getSource();
  }

  if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
    extractOperandsFromSubview(subview, rewriter, dst_offsets, dst_sizes,
                               dst_strides);
    dst = subview.getSource();
  } else if (auto reinterpretCast =
                 dst.getDefiningOp<memref::ReinterpretCastOp>()) {
    extractOperandsFromReinterpretCast(reinterpretCast, rewriter, dst_offsets,
                                       dst_sizes, dst_strides);
    dst = reinterpretCast.getSource();
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

static void
replaceAllUsesOfConstsInRegionWithNew(SmallVector<Value, 4> constants,
                                      OpBuilder builder, Region &region) {
  for (auto c : constants) {
    replaceAllUsesInRegionWith(
        c, builder.clone(*c.getDefiningOp())->getResult(0), region);
  }
}

void getUsedConstsDefinedAbove(MutableArrayRef<Region> region,
                               SmallVector<Value, 4> &constants) {
  llvm::SetVector<Value> region_args;
  getUsedValuesDefinedAbove(region, region_args);
  for (Value v : region_args) {
    if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
      constants.push_back(v);
  }
}

void getUsedArgsDefinedAbove(MutableArrayRef<Region> region,
                             SmallVector<Value, 4> &args) {
  llvm::SetVector<Value> region_args;
  getUsedValuesDefinedAbove(region, region_args);
  for (Value v : region_args) {
    if (!isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
      args.push_back(v);
  }
}

void getUsedConstsAndArgsDefinedAbove(MutableArrayRef<Region> region,
                                      SmallVector<Value, 4> &constants,
                                      SmallVector<Value, 4> &args) {
  getUsedConstsDefinedAbove(region, constants);
  getUsedConstsDefinedAbove(region, args);
}

class MemrefCopyToAIRDmaConversion : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(matchAndRewriteCopyOp(op, rewriter)))
      return failure();
    return success();
  }
};

// Pattern to rewrite `linalg.copy` to `memref.copy`.
class LinalgCopyToMemRefCopy : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.hasIndexSemantics()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::CopyOp>(
        copyOp, copyOp.getInputs().front(), copyOp.getDpsInits().front());
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

void InsertEmptyLaunchOverHerd(air::HerdOp op, bool insertSegment = true) {
  OpBuilder builder(op);
  if (op->getParentOfType<air::SegmentOp>() ||
      op->getParentOfType<air::LaunchOp>())
    return;

  auto loc = op.getLoc();

  // Collect kernel operands
  SmallVector<Value, 4> args(op.getKernelOperands());

  // Launch of size 1 in each dimension
  SmallVector<Value, 4> launchSizes;
  for (unsigned i = 0; i < op.getNumDims(); ++i)
    launchSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));

  // Create LaunchOp
  air::LaunchOp launch;
  if (op.getAsyncToken())
    launch = builder.create<air::LaunchOp>(loc, op.getAsyncDependencies(),
                                           launchSizes, args, true);
  else
    launch = builder.create<air::LaunchOp>(loc, launchSizes, args);

  builder.setInsertionPointToStart(&launch.getRegion().front());

  // Create optional SegmentOp
  air::SegmentOp segment;

  if (insertSegment) {
    SmallVector<Value, 4> segmentOpers(launch.getIds());
    llvm::append_range(segmentOpers, launch.getSize());
    llvm::append_range(segmentOpers, launch.getKernelArguments());
    SmallVector<Value> segmentSizes; // TODO: Currently we generate
                                     // single-iteration segments only.
    if (op.getAsyncToken())
      segment = builder.create<air::SegmentOp>(loc, ValueRange{}, segmentSizes,
                                               segmentOpers, true);
    else
      segment = builder.create<air::SegmentOp>(loc, segmentSizes, segmentOpers);
    builder.setInsertionPointToStart(&segment.getRegion().front());
  }

  // Construct new HerdOp in the correct region
  SmallVector<Value, 2> herdSizes;
  for (auto v : op.getSizeOperands())
    herdSizes.push_back(builder.clone(*v.getDefiningOp())->getResult(0));

  air::HerdOp newHerd;
  SmallVector<Value, 4> herdOpers;
  if (insertSegment) {
    llvm::append_range(herdOpers, segment.getIds());
    llvm::append_range(herdOpers, segment.getSize());
    llvm::append_range(herdOpers, segment.getKernelArguments());
  } else {
    llvm::append_range(herdOpers, launch.getIds());
    llvm::append_range(herdOpers, launch.getSize());
    llvm::append_range(herdOpers, launch.getKernelArguments());
  }
  if (op.getAsyncToken())
    newHerd = builder.create<air::HerdOp>(loc, ValueRange{}, herdSizes,
                                          herdOpers, true);
  else
    newHerd = builder.create<air::HerdOp>(loc, herdSizes, herdOpers);

  // Map values from old Herd to new Herd
  IRMapping remap;
  for (unsigned i = 0; i < op.getNumDims(); ++i) {
    remap.map(op.getIds()[i], newHerd.getIds()[i]);
    remap.map(op.getSize()[i], newHerd.getSize()[i]);
  }
  for (unsigned i = 0; i < op.getNumKernelOperands(); i++) {
    int blockArgOffset = launch.getNumDims() *
                         2; // Each dim has an induction variable and a size.
    if (segment)
      blockArgOffset += segment.getNumDims() * 2;
    remap.map(op.getKernelArgument(i),
              newHerd.getKernelArgument(blockArgOffset + i));
  }

  builder.setInsertionPointToStart(&newHerd.getRegion().front());
  for (Operation &o : op.getBody().front().without_terminator())
    builder.clone(o, remap);

  // Copy symbol and attributes
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    newHerd->setAttr(SymbolTable::getSymbolAttrName(), attr);
  if (op.getLinkWith())
    newHerd.setLinkWith(op.getLinkWith());

  if (auto token = op.getAsyncToken())
    replaceAllUsesInRegionWith(token, launch.getAsyncToken(),
                               *op->getParentRegion());

  op->erase();
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
    if (!fnDecl) {
      callOp->emitOpError("expected function declaration");
      return WalkResult::interrupt();
    }
    if (!fnDecl->hasAttr("link_with")) {
      callOp->emitOpError(
          "expected 'link_with' construct for the function declaration.");
      return WalkResult::interrupt();
    }
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

// Create a new air.channel symbol in the module for the cascade pipeline.
// The symbol name is unique in the module, and the channel is tagged with
// the "cascade" attribute.
air::ChannelOp
createCascadeChannelOp(OpBuilder &builder, ModuleOp module, Location loc,
                       SmallVector<int64_t> channel_bundle_sizes) {

  // Generate a unique channel symbol name within the module.
  std::string cname = air::createChannelName(module);

  // Insert the channel op at the top of the module body, but *after* any
  // existing channel ops (so channels are grouped together).
  OpBuilder::InsertionGuard guard(builder);
  Operation *o = &module.getBody()->front();
  while (dyn_cast_or_null<air::ChannelOp>(o))
    o = o->getNextNode();
  builder.setInsertionPoint(o);

  // Create the channel op with the given bundle sizes and "cascade" tag.
  auto channel_op = builder.create<air::ChannelOp>(
      loc, cname, builder.getI64ArrayAttr(channel_bundle_sizes),
      builder.getStringAttr("cascade"));

  return channel_op;
}

// Transform an scf.reduce inside an scf.parallel into an affine.if pipeline
// split into prologue, steady-state (pipeline body), and epilogue, connected
// by cascade channels. Clones necessary producers/consumers into each stage.
LogicalResult ScfReduceToAffineIf(scf::ReduceOp reduceOp,
                                  air::HierarchyInterface hierOp,
                                  PatternRewriter &rewriter) {
  Location loc = hierOp.getLoc();

  // Ensure the reduce is under an scf.parallel and collect initial values.
  auto parallelOp = reduceOp->getParentOfType<scf::ParallelOp>();
  auto parInitValues = parallelOp.getInitVals();
  if (!parallelOp)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "reduce not under scf.parallel");
  if (parallelOp.getNumLoops() != static_cast<int64_t>(parInitValues.size()))
    return rewriter.notifyMatchFailure(reduceOp, "init vals size mismatch");

  // Get constant upper bounds for each parallel dimension (must be > 0).
  // Subtract one so that bounds represent the last iteration index.
  SmallVector<int64_t> ubCsts;
  for (Value ub : parallelOp.getUpperBound()) {
    auto c = getConstantIntValue(ub);
    if (!c)
      return rewriter.notifyMatchFailure(parallelOp,
                                         "non-constant upper bound");
    if (*c <= 0)
      return rewriter.notifyMatchFailure(parallelOp, "upper bound <= 0");
    ubCsts.push_back(*c - 1);
  }

  // Create a cascade channel op sized according to the upper bounds.
  air::ChannelOp newCascadeChannel =
      createCascadeChannelOp(rewriter, hierOp->getParentOfType<ModuleOp>(), loc,
                             /*channel_bundle_sizes*/ ubCsts);

  // === Identify ops to clone into each pipeline stage ===

  SmallVector<SetVector<Operation *>> VecOfProducer, VecOfConsumers;

  // For each init value, record its defining op (if any) and any other
  // user ops except the parallel op itself -- these are "producers" to clone.
  for (auto val : parInitValues) {
    SetVector<Operation *> producers;
    if (auto defOp = val.getDefiningOp())
      producers.insert(defOp);
    for (auto user : val.getUsers()) {
      if (user == parallelOp)
        continue;
      producers.insert(user);
    }
    VecOfProducer.push_back(producers);
  }

  // For each parallel result, record all of its users -- these are "consumers".
  for (auto res : parallelOp->getResults()) {
    auto resUsers = res.getUsers();
    SetVector<Operation *> consumers(resUsers.begin(), resUsers.end());
    VecOfConsumers.push_back(consumers);
  }

  // === Small helper lambdas for code reuse ===

  // Compute "decremented" index values (iv - 1) for channel index operands.
  auto decIndices = [&](ValueRange ifOpers) {
    SmallVector<Value> idx;
    idx.reserve(ifOpers.size());
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (Value v : ifOpers)
      idx.push_back(rewriter.create<arith::SubIOp>(loc, v, c1));
    return idx;
  };

  // Emit a ChannelPutOp with given channel name, indices, and value.
  auto emitChannelPut = [&](StringRef ch, ValueRange idx, Value val) {
    rewriter.create<air::ChannelPutOp>(
        loc, /*types*/ TypeRange{}, /*async_deps*/ ValueRange{}, ch, idx, val,
        /*offsets*/ ValueRange{},
        /*sizes*/ ValueRange{},
        /*strides*/ ValueRange{});
  };

  // Emit a ChannelGetOp with given channel name, indices, and value.
  auto emitChannelGet = [&](StringRef ch, ValueRange idx, Value val) {
    rewriter.create<air::ChannelGetOp>(
        loc, /*types*/ TypeRange{}, /*async_deps*/ ValueRange{}, ch, idx, val,
        /*offsets*/ ValueRange{},
        /*sizes*/ ValueRange{},
        /*strides*/ ValueRange{});
  };

  // Clone a list of ops into the current insertion point using a remap.
  auto cloneOps = [&](const auto &ops, IRMapping &map) {
    for (Operation *o : ops)
      rewriter.clone(*o, map);
  };

  // Clone the body of the reduction for the given dim, remapping lhs/rhs.
  auto cloneReductionBody = [&](int64_t dim, IRMapping &map) {
    Block &blk = reduceOp.getReductions()[dim].front();
    BlockArgument lhs = blk.getArgument(0);
    BlockArgument rhs = blk.getArgument(1);
    map.map(lhs, reduceOp.getOperands()[dim]);
    map.map(rhs, map.lookupOrDefault(parInitValues[dim]));
    for (Operation &o : blk.without_terminator())
      rewriter.clone(o, map);
  };

  // === Prologue stage ===
  // Builds an IntegerSet that selects the last iteration of the parallel loop.
  auto makePrologueSet = [](scf::ParallelOp parallelOp) {
    auto ctx = parallelOp->getContext();
    SmallVector<AffineExpr> constraints;
    SmallVector<bool> eqFlags;
    for (auto dim : llvm::seq<int64_t>(0, parallelOp.getNumLoops())) {
      auto ubCst = getConstantIntValue(parallelOp.getUpperBound()[dim]);
      constraints.push_back(getAffineSymbolExpr(dim, ctx) -
                            getAffineConstantExpr(*ubCst - 1, ctx));
      eqFlags.push_back(true);
    }
    return IntegerSet::get(0, parallelOp.getNumLoops(), constraints, eqFlags);
  };
  IntegerSet prologIS = makePrologueSet(parallelOp);

  // Condition operands for the if-ops: hierarchy IDs for each parallel IV.
  SmallVector<Value> ifOpers(hierOp.getIds().begin(),
                             hierOp.getIds().begin() +
                                 parallelOp.getNumLoops());

  // Outer if: prologue vs everything else.
  auto ifTop = rewriter.create<affine::AffineIfOp>(loc, prologIS, ifOpers,
                                                   /*has_else*/ true);
  rewriter.setInsertionPointToStart(ifTop.getThenBlock());

  // Prologue execution: clone producers and first reduction step, then put into
  // channel.
  auto runProlog = [&] {
    IRMapping prologRemap;
    for (auto dim : llvm::seq<int64_t>(0, parallelOp.getNumLoops())) {
      prologRemap.map(parallelOp.getInductionVars()[dim], hierOp.getIds()[dim]);
    }
    for (int64_t dim = 0, N = VecOfProducer.size(); dim < N; ++dim) {
      cloneOps(VecOfProducer[dim], prologRemap);
      cloneReductionBody(dim, prologRemap);
      emitChannelPut(newCascadeChannel.getSymName(), decIndices(ifOpers),
                     reduceOp.getOperands()[dim]);
    }
  };
  runProlog();

  // === Pipeline body (steady state) ===
  // IntegerSet selects iterations that are neither first nor last.
  auto makePplBodySet = [](scf::ParallelOp parallelOp) {
    auto ctx = parallelOp->getContext();
    SmallVector<AffineExpr> constraints;
    SmallVector<bool> eqFlags;
    for (auto dim : llvm::seq<int64_t>(0, parallelOp.getNumLoops())) {
      auto ubCst = getConstantIntValue(parallelOp.getUpperBound()[dim]);
      auto symbolExpr = getAffineSymbolExpr(dim, ctx);
      constraints.push_back(symbolExpr - getAffineConstantExpr(1, ctx));
      eqFlags.push_back(false);
      constraints.push_back(getAffineConstantExpr(*ubCst - 2, ctx) -
                            symbolExpr);
      eqFlags.push_back(false);
    }

    return IntegerSet::get(0, parallelOp.getNumLoops(), constraints, eqFlags);
  };
  IntegerSet pplBodyIS = makePplBodySet(parallelOp);

  // Else branch of prologue: steady state vs epilogue.
  rewriter.setInsertionPointToStart(ifTop.getElseBlock());
  auto elIfTop = rewriter.create<affine::AffineIfOp>(loc, pplBodyIS, ifOpers,
                                                     /*has_else*/ true);
  rewriter.setInsertionPointToStart(elIfTop.getThenBlock());

  // Steady-state: clone producers, get from channel, do reduction, put back.
  auto runPipelineBody = [&] {
    IRMapping pplBodyRemap;
    for (auto dim : llvm::seq<int64_t>(0, parallelOp.getNumLoops())) {
      pplBodyRemap.map(parallelOp.getInductionVars()[dim],
                       hierOp.getIds()[dim]);
    }
    for (int64_t dim = 0, N = VecOfProducer.size(); dim < N; ++dim) {
      cloneOps(VecOfProducer[dim], pplBodyRemap);
      emitChannelGet(newCascadeChannel.getSymName(), ifOpers,
                     reduceOp.getOperands()[dim]);
      cloneReductionBody(dim, pplBodyRemap);
      emitChannelPut(newCascadeChannel.getSymName(), decIndices(ifOpers),
                     reduceOp.getOperands()[dim]);
    }
  };
  runPipelineBody();

  // === Epilogue ===
  // Else branch of steady-state: final iteration of pipeline.
  rewriter.setInsertionPointToStart(elIfTop.getElseBlock());

  auto runEpilog = [&] {
    IRMapping epilogRemap;
    for (auto dim : llvm::seq<int64_t>(0, parallelOp.getNumLoops())) {
      epilogRemap.map(parallelOp.getInductionVars()[dim], hierOp.getIds()[dim]);
    }
    for (int64_t dim = 0, N = VecOfProducer.size(); dim < N; ++dim) {
      cloneOps(VecOfProducer[dim], epilogRemap);
      emitChannelGet(newCascadeChannel.getSymName(), ifOpers,
                     reduceOp.getOperands()[dim]);
      cloneReductionBody(dim, epilogRemap);

      // Map final reduction results to parallel loop results for consumer
      // cloning.
      epilogRemap.map(parallelOp->getResult(dim), reduceOp.getOperands()[dim]);
      for (auto *o : VecOfConsumers[dim]) {
        air::cloneOpAndOperands(rewriter, epilogRemap, o);
      }
    }
  };
  runEpilog();

  // === Cleanup ===
  // Remove original producer and consumer ops that have been cloned into
  // stages.
  llvm::SetVector<Operation *> toErase;
  for (auto &sv : VecOfProducer)
    toErase.insert(sv.begin(), sv.end());
  for (auto &sv : VecOfConsumers)
    toErase.insert(sv.begin(), sv.end());
  for (Operation *op : toErase)
    rewriter.eraseOp(op);

  return success();
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
  getUsedArgsDefinedAbove(op.getRegion(), args);
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

  // If scf.parallel has scf.reduce with non-empty region, then convert to
  // affine.if.
  auto reduceOp = dyn_cast<scf::ReduceOp>(op.getBody()->getTerminator());
  if (!reduceOp.getReductions().empty()) {
    rewriter.setInsertionPoint(bb.getTerminator());
    if (failed(ScfReduceToAffineIf(reduceOp, hierOp, rewriter)))
      return failure();
  }

  rewriter.setInsertionPointToStart(&hierOp.getRegion().front());
  getUsedConstsDefinedAbove(hierOp.getRegion(), constants);
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
    } else {
      o->emitOpError("memref operation type unsupported on L1 memref.");
      return failure();
    }
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
    } else {
      o->emitOpError("memref operation type unsupported on L1 memref.");
      return failure();
    }
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
    stage2Patterns.insert<LinalgCopyToMemRefCopy, MemrefCopyToAIRDmaConversion>(
        context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(stage2Patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      module.dump();
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
        if (std::find(seg_syms.begin(), seg_syms.end(), name) !=
            seg_syms.end()) {
          op->emitOpError("unexpected duplicate symbol.");
          return;
        }
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
          if (!f->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
            op->emitOpError("enclosing function of air.sgement op expected to "
                            "have a symbol name.");
            return;
          }
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
    if (!filteredOps.erase(e))
      return failure();
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
      // Depth = -1 means converting the innermost parallel ops
      if (clAssignDepth == -1) {
        SmallVector<Operation *> parOpsInOp;
        op->walk([&parOpsInOp](Operation *o) {
          if (isa<scf::ForallOp, scf::ParallelOp, affine::AffineParallelOp>(o))
            parOpsInOp.push_back(o);
        });
        if (parOpsInOp.size() > 1)
          return;
        filteredOps.insert(op);
        return;
      }
      // Assigning depth to other negative values means converting all
      // parallel ops
      if (clAssignDepth < 0) {
        filteredOps.insert(op);
        return;
      }
      // the number of nested parallel above this one
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
                           air::airDialect, arith::ArithDialect, ub::UBDialect,
                           affine::AffineDialect, memref::MemRefDialect,
                           scf::SCFDialect, linalg::LinalgDialect>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });
    target.addDynamicallyLegalOp<affine::AffineParallelOp>(
        [&](affine::AffineParallelOp p) { return !filteredOps.contains(p); });

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
      // Depth = -1 means converting the innermost parallel ops
      if (clAssignDepth == -1) {
        SmallVector<Operation *> parOpsInOp;
        op->walk([&parOpsInOp](Operation *o) {
          if (isa<scf::ParallelOp>(o))
            parOpsInOp.push_back(o);
        });
        if (parOpsInOp.size() > 1)
          return;
        filteredOps.insert(op);
        return;
      }
      // Assigning depth to other negative values means converting all
      // parallel ops
      if (clAssignDepth < 0) {
        filteredOps.insert(op);
        return;
      }
      // the number of nested parallel above this one
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
      // Depth = -1 means converting the innermost parallel ops
      if (clAssignDepth == -1) {
        SmallVector<Operation *> parOpsInOp;
        op->walk([&parOpsInOp](Operation *o) {
          if (isa<scf::ParallelOp>(o))
            parOpsInOp.push_back(o);
        });
        if (parOpsInOp.size() > 1)
          return;
        filteredOps.insert(op);
        return;
      }
      // Assigning depth to other negative values means converting all
      // parallel ops
      if (clAssignDepth < 0) {
        filteredOps.insert(op);
        return;
      }
      // the number of nested parallel above this one
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

// Inserts an empty launch over air.herd, where an air.launch is always
// inserted, and an air.segment is inserted only if insertSegment is set to
// true.
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
        InsertEmptyLaunchOverHerd(op, /*insertSegment=*/clInsertSegment);
      else if (!op->getParentOfType<air::SegmentOp>())
        InsertEmptyLaunchOverHerd(op, /*insertSegment=*/clInsertSegment);
    });
    getSegmentNames(module);
  }
};

// Identifies arith operations where all operands are either constants, or
// produced by IndexCastOp casting from IndexType. If detected, canonicalize
// IndexCast ops by changing the arith op's input/output types to IndexType.
template <typename T>
LogicalResult canonicalizeArithBinaryOpToIndexType(T arithOp,
                                                   RewriterBase &rewriter) {
  Value lhs = arithOp.getLhs();
  Value rhs = arithOp.getRhs();

  SmallVector<Value> inVals = {lhs, rhs};
  if (llvm::all_of(inVals, [](Value v) { return isa<IndexType>(v.getType()); }))
    return failure();
  if (llvm::any_of(inVals, [](Value v) {
        if (!v.getDefiningOp())
          return true;
        if (getConstantIntValue(v))
          return false;
        else if (auto castOp = dyn_cast_if_present<arith::IndexCastOp>(
                     v.getDefiningOp())) {
          if (llvm::all_of(castOp->getOperands(), [](Value oper) {
                return isa<IndexType>(oper.getType());
              }))
            return false;
          else
            return true;
        }
        return true;
      }))
    return failure();

  auto loc = arithOp.getLoc();
  if (!isa<IndexType>(lhs.getType())) {
    lhs =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), lhs);
  }
  if (!isa<IndexType>(rhs.getType())) {
    rhs =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), rhs);
  }
  auto newArithOp = rewriter.create<T>(loc, rewriter.getIndexType(), lhs, rhs);
  rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
      arithOp, arithOp.getResult().getType(), newArithOp);

  return success();
}

struct CanonicalizeArithAddIOpToIndexTypePattern
    : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp arithOp,
                                PatternRewriter &rewriter) const override {
    return canonicalizeArithBinaryOpToIndexType<arith::AddIOp>(arithOp,
                                                               rewriter);
  }
};
struct CanonicalizeArithMulIOpToIndexTypePattern
    : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp arithOp,
                                PatternRewriter &rewriter) const override {
    return canonicalizeArithBinaryOpToIndexType<arith::MulIOp>(arithOp,
                                                               rewriter);
  }
};

// Wraps the body of a given func.func operation inside an scf.parallel loop.
// The pass assumes that:
// (1) The function arguments consist of: M memref arguments, N loop upper
// bounds, N loop induction variable indices. (2) The scf.parallel loop is
// constructed using the N upper bounds and induction variable indices. (3) The
// scf.parallel loop is inserted at the beginning of the function, wrapping all
// existing operations.

struct WrapFuncWithParallelPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  WrapFuncWithParallelPattern(MLIRContext *context,
                              SmallVector<int64_t> &bounds)
      : OpRewritePattern(context), loopBounds(bounds) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    if (funcOp.isExternal())
      return failure(); // Ignore external functions

    if (loopBounds.empty()) {
      funcOp.emitError("Pass option 'loop-bounds' must be specified.");
      return failure();
    }

    unsigned N = loopBounds.size(); // Number of loop dimensions

    // Get function arguments
    auto args = funcOp.getArguments();
    unsigned numArgs = args.size();

    if (numArgs < 2) {
      funcOp.emitError(
          "Expected at least 2 arguments: memrefs and loop bounds.");
      return failure();
    }

    // Determine M (memrefs count)
    unsigned M = 0;
    for (Type argType : funcOp.getFunctionType().getInputs()) {
      if (isa<UnrankedMemRefType, MemRefType>(argType))
        M++;
      else
        break;
    }
    if (M + N * 2 > numArgs) {
      funcOp.emitError("Expected func op arguments contain at least M memrefs "
                       "and N x 2 loop bounds.");
      return failure();
    }

    // Extract indices
    ValueRange inductionVars = args.slice(M + N, N);

    if (llvm::all_of(inductionVars, [](Value iv) { return iv.use_empty(); }))
      return failure();

    // Store original function body operations
    SmallVector<Operation *> originalFuncBodyOps;
    for (auto &op : funcOp.getBody().front().without_terminator())
      originalFuncBodyOps.push_back(&op);

    // Create scf.parallel loop
    Location loc = funcOp.getLoc();
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    SmallVector<Value, 4> lowerBounds(
        N, rewriter.create<arith::ConstantIndexOp>(loc, 0));
    SmallVector<Value, 4> steps(
        N, rewriter.create<arith::ConstantIndexOp>(loc, 1));
    SmallVector<Value, 4> upperBoundsVals;
    for (int64_t bound : loopBounds) {
      upperBoundsVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, bound));
    }

    auto parallelOp = rewriter.create<scf::ParallelOp>(loc, lowerBounds,
                                                       upperBoundsVals, steps);

    // Redirect arguments properly inside the loop
    Block &loopBlock = parallelOp.getRegion().front();
    rewriter.setInsertionPointToStart(&loopBlock);
    IRMapping remap;
    for (unsigned i = 0; i < N; i++) {
      Value loopBlockArg = loopBlock.getArgument(i);
      if (inductionVars[i].getType() != loopBlockArg.getType())
        loopBlockArg = rewriter.create<arith::IndexCastOp>(
            loc, inductionVars[i].getType(), loopBlockArg);
      remap.map(inductionVars[i], loopBlockArg);
    }

    // Move function body into the loop
    for (auto op : originalFuncBodyOps) {
      rewriter.clone(*op, remap);
    }

    // Erase original function body ops
    for (auto o : llvm::reverse(originalFuncBodyOps))
      rewriter.eraseOp(o);

    return success();
  }

private:
  SmallVector<int64_t> &loopBounds; // External loop bounds
};

class AIRWrapFuncWithParallelPass
    : public air::impl::AIRWrapFuncWithParallelPassBase<
          AIRWrapFuncWithParallelPass> {

public:
  AIRWrapFuncWithParallelPass() = default;
  AIRWrapFuncWithParallelPass(const AIRWrapFuncWithParallelPass &pass){};
  AIRWrapFuncWithParallelPass(
      const ::xilinx::air::AIRWrapFuncWithParallelPassOptions &options)
      : AIRWrapFuncWithParallelPassBase(options) {}

  void runOnOperation() override;

private:
};

void AIRWrapFuncWithParallelPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet wrapParPatterns(context);
  SmallVector<int64_t> loopBoundsVec;
  for (auto i : clLoopBounds)
    loopBoundsVec.push_back(i);
  wrapParPatterns.add<WrapFuncWithParallelPattern>(context, loopBoundsVec);
  (void)applyOpPatternsGreedily(SmallVector<Operation *>{funcOp.getOperation()},
                                std::move(wrapParPatterns));

  RewritePatternSet patterns(context);
  patterns.add<CanonicalizeArithAddIOpToIndexTypePattern,
               CanonicalizeArithMulIOpToIndexTypePattern>(context);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace air
} // namespace xilinx

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
  llvm::SmallSet<xilinx::air::HerdOp, 2> herdOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<xilinx::air::ScfParToHerdConversion>(ctx, filteredOps, herdOps,
                                                    getFirstDim());
  (void)applyPatternsGreedily(
      target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));
  for (auto h : herdOps) {
    xilinx::air::getHerdNames(h->getParentOfType<ModuleOp>());
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
  llvm::SmallSet<xilinx::air::LaunchOp, 2> launchOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<xilinx::air::ScfParToLaunchConversion>(
      ctx, filteredOps, launchOps, getHasAirSegment());
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
  llvm::SmallSet<xilinx::air::SegmentOp, 2> segmentOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<xilinx::air::ScfParToSegmentConversion>(ctx, filteredOps,
                                                       segmentOps);
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
  auto res = xilinx::air::matchAndRewriteCopyOp(op, rewriter);
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

std::unique_ptr<Pass> createAIRWrapFuncWithParallelPass() {
  return std::make_unique<AIRWrapFuncWithParallelPass>();
}
std::unique_ptr<Pass>
createAIRWrapFuncWithParallelPass(AIRWrapFuncWithParallelPassOptions options) {
  return std::make_unique<AIRWrapFuncWithParallelPass>(options);
}

} // namespace air
} // namespace xilinx
