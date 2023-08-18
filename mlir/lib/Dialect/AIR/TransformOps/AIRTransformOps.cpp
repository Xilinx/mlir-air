//===- AIRTransformOps.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Conversion/AIRToAIEPass.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SmallSet.h"

#include <numeric>

#define DEBUG_TYPE "air-transform-ops"

using namespace mlir;
using namespace mlir::affine;
using namespace xilinx::air;

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GetSegmentForOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetSegmentForOp::apply(mlir::transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  SetVector<Operation *> segments;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto segment = target->getParentOfType<SegmentOp>();
    if (!segment) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "could not find an '"
                                 << SegmentOp::getOperationName() << "' parent";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    segments.insert(segment);
  }
  results.set(getResult().cast<OpResult>(), segments.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// SegmentToAIEOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SegmentToAIEOp::applyToOne(transform::TransformRewriter &rewriter,
                                      Operation *op,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  SegmentOp target = llvm::dyn_cast<SegmentOp>(op);
  FailureOr<ModuleOp> res = convertAIRToAIE(rewriter, target);
  if (failed(res))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.push_back(res->getOperation());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ParToHerdOp
//===----------------------------------------------------------------------===//

void xilinx::air::replaceAllUsesOfConstsInRegionWithNew(
    SmallVector<Value, 4> constants, OpBuilder builder, Region &region) {
  for (auto c : constants) {
    replaceAllUsesInRegionWith(
        c, builder.clone(*c.getDefiningOp())->getResult(0), region);
  }
}

LogicalResult xilinx::air::normalizeScfParallel(scf::ParallelOp parOp,
                                                PatternRewriter &rewriter) {
  auto loc = parOp.getLoc();

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

  auto ivs = parOp.getInductionVars().begin();
  auto step = parOp.getStep().begin();
  auto lowerBound = parOp.getLowerBound().begin();
  auto upperBound = parOp.getUpperBound().begin();

  SmallVector<Value, 4> new_step;
  SmallVector<Value, 4> new_ub;
  SmallVector<Value, 4> new_lb;

  auto builder = OpBuilder::atBlockBegin(parOp.getBody());
  while (step != parOp.getStep().end()) {
    auto iv = *ivs++;
    Value sv = *step++;
    Value lbv = *lowerBound++;
    Value ubv = *upperBound++;
    auto s = sv.getDefiningOp<arith::ConstantIndexOp>().value();
    auto lb = lbv.getDefiningOp<arith::ConstantIndexOp>().value();
    auto ub = ubv.getDefiningOp<arith::ConstantIndexOp>().value();

    auto new_ub_int = (ub - lb) / s;
    if ((new_ub_int * s) != (ub - lb))
      return parOp->emitOpError()
             << "failed to normalize: step '" << s
             << "' does not evenly divide range '" << (ub - lb) << "'";

    new_ub.push_back(rewriter.create<arith::ConstantIndexOp>(loc, new_ub_int));
    new_lb.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    new_step.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    AffineExpr d0 = builder.getAffineDimExpr(0);
    AffineExpr mul = d0 * sv.getDefiningOp<arith::ConstantIndexOp>().value();
    AffineExpr add = mul + lbv.getDefiningOp<arith::ConstantIndexOp>().value();
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

class ScfParToHerdConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToHerdConversion(MLIRContext *ctx,
                         SmallPtrSet<Operation *, 8> &filteredOps,
                         llvm::SmallSet<HerdOp, 2> &replacementOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps),
        replacementOps(replacementOps){};

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
    llvm::SetVector<Value> region_args;
    getUsedValuesDefinedAbove(op.getRegion(), region_args);
    for (Value v : region_args) {
      if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
        constants.push_back(v);
      else
        args.push_back(v);
    }
    SmallVector<Value, 2> dims{
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[1])};
    auto herdOp = rewriter.create<HerdOp>(op.getLoc(), dims, args);
    auto &bb = herdOp.getBody().front();
    auto ivs = op.getInductionVars();

    ivs[0].replaceAllUsesWith(herdOp.getIds()[0]);
    if (op.getNumLoops() == 2)
      ivs[1].replaceAllUsesWith(herdOp.getIds()[1]);

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&herdOp.getRegion().front());
    replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                          herdOp.getRegion());
    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<HerdTerminatorOp>(loc);

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
  llvm::SmallPtrSet<Operation *, 8> filteredOps;
  llvm::SmallSet<HerdOp, 2> &replacementOps;
};

void xilinx::air::populateScfParToHerdConversionPattern(
    RewritePatternSet &patterns, SmallPtrSet<Operation *, 8> &filteredOps,
    llvm::SmallSet<HerdOp, 2> &replacementOps) {
  patterns.add<ScfParToHerdConversion>(patterns.getContext(), filteredOps,
                                       replacementOps);
}

void xilinx::air::getHerdNames(ModuleOp module) {
  std::vector<std::string> herd_syms;
  for (auto f : module.getOps<func::FuncOp>()) {
    // record existing symbol names
    f.walk([&](HerdOp op) {
      if (auto attr =
              op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
        std::string name = attr.getValue().str();
        assert((std::find(herd_syms.begin(), herd_syms.end(), name) ==
                herd_syms.end()) &&
               "unexpected duplicate symbol");
        herd_syms.push_back(name);
      }
    });
    // generate missing symbol names
    f.walk([&](HerdOp op) {
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

DiagnosedSilenceableFailure
transform::ParToHerdOp::applyToOne(transform::TransformRewriter &rewriter,
                                   ::mlir::Operation *target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  auto ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  llvm::SmallSet<HerdOp, 2> herdOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<ScfParToHerdConversion>(ctx, filteredOps, herdOps);
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

class ScfParToLaunchConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToLaunchConversion(MLIRContext *ctx,
                           llvm::SmallSet<Operation *, 8> &filteredOps,
                           llvm::SmallSet<LaunchOp, 2> &replacementOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps),
        replacementOps(replacementOps){};

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
    auto launch = rewriter.create<LaunchOp>(op.getLoc(), sizes, args);
    auto &bb = launch.getBody().front();
    auto ivs = op.getInductionVars();

    for (int i = 0, e = ivs.size(); i < e; i++) {
      ivs[i].replaceAllUsesWith(launch.getIds()[i]);
    }

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&launch.getRegion().front());
    replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                          launch.getRegion());

    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<LaunchTerminatorOp>(loc);

    int i = 0;
    auto kernel_args = launch.getKernelArguments();
    for (Value v : args)
      replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());

    if (op != parOp)
      op.erase();
    rewriter.eraseOp(parOp);
    replacementOps.insert(launch);

    return success();
  }

private:
  llvm::SmallSet<Operation *, 8> &filteredOps;
  llvm::SmallSet<LaunchOp, 2> &replacementOps;
};

void xilinx::air::populateScfParToLaunchConversionPattern(
    RewritePatternSet &patterns, llvm::SmallSet<Operation *, 8> &filteredOps,
    llvm::SmallSet<LaunchOp, 2> &replacementOps) {
  patterns.add<ScfParToLaunchConversion>(patterns.getContext(), filteredOps,
                                         replacementOps);
}

DiagnosedSilenceableFailure
transform::ParToLaunchOp::applyToOne(transform::TransformRewriter &rewriter,
                                     ::mlir::Operation *target,
                                     transform::ApplyToEachResultList &results,
                                     transform::TransformState &state) {
  auto ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  llvm::SmallSet<Operation *, 8> filteredOps;
  llvm::SmallSet<LaunchOp, 2> launchOps;
  filteredOps.insert(target);
  patterns.add<ScfParToLaunchConversion>(ctx, filteredOps, launchOps);
  (void)applyPatternsAndFoldGreedily(
      target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));
  for (auto l : launchOps)
    results.push_back(l);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PipelineReduceOp
//===----------------------------------------------------------------------===//

std::optional<Value>
xilinx::air::allocBufferCallBack(OpBuilder &b, memref::SubViewOp subView,
                                 ArrayRef<Value> boundingSubViewSize,
                                 DataLayout &layout) {
  MemRefType viewType = subView.getType();
  MemRefType allocType =
      MemRefType::get(viewType.getShape(), viewType.getElementType(), {},
                      (unsigned)MemorySpace::L1);
  Value buffer = b.createOrFold<memref::AllocOp>(subView.getLoc(), allocType);
  return buffer;
}

LogicalResult xilinx::air::deallocBufferCallBack(OpBuilder &b, Value buffer) {
  // b.create<memref::DeallocOp>(buffer.getLoc(), buffer);
  return success();
}

// Create channel name as string
static std::string createChannelName(ModuleOp module) {
  std::string new_cname = "channel_0";
  std::string cname = "channel";
  int which_try = 0;
  while (module.lookupSymbol(new_cname))
    new_cname = cname + "_" + std::to_string(++which_try);
  cname = new_cname;
  return cname;
}

// Split a linalg reduction into 'pipeline_depth' consecutive
// stages, each one feeding partial reductions to the next stage.
// Stages are mapped to Nx1 or Nx1 herd.
FailureOr<linalg::TiledLinalgOp> xilinx::air::pipelineReduceLinalgOp(
    RewriterBase &b, linalg::LinalgOp op, ArrayRef<int64_t> static_tile_sizes,
    unsigned int pipeline_depth, std::string pipeline_direction, bool promote) {

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto loc = op.getLoc();
  auto ctx = op.getContext();

  if (!(pipeline_direction == "vert" || pipeline_direction == "horiz"))
    return failure();

  auto iteratorTypes = op.getIteratorTypesArray();
  if (linalg::isParallelIterator(iteratorTypes.back()))
    return failure();

  bool isHoriz = pipeline_direction == "horiz";
  int new_herd_x = isHoriz ? pipeline_depth : 1;
  int new_herd_y = !isHoriz ? pipeline_depth : 1;

  SmallVector<Value, 2> dims{b.create<arith::ConstantIndexOp>(loc, new_herd_x),
                             b.create<arith::ConstantIndexOp>(loc, new_herd_y)};

  SmallVector<Value, 4> args;
  for (auto o : op->getOperands())
    args.push_back(o);

  auto herd = b.create<HerdOp>(loc, dims, args);
  b.setInsertionPointToStart(&herd.getBody().front());

  Value x = herd.getIds()[0];
  Value y = herd.getIds()[1];

  auto nLoops = op.getNumLoops();
  auto tileSizes = static_tile_sizes.take_front(nLoops);

  SmallVector<OpFoldResult, 4> tileSizeVector;
  for (auto s : tileSizes)
    tileSizeVector.push_back(
        b.create<arith::ConstantIndexOp>(loc, s).getResult());
  if (tileSizeVector.size() < nLoops) {
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero.getResult());
  }

  auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();
  SmallVector<OpFoldResult> sizeBounds =
      makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                               allShapeSizes);

  SmallVector<OpFoldResult> tileIds;
  for (auto s : tileSizes) {
    if (s == 0)
      continue;
    AffineExpr d0 = b.getAffineDimExpr(0);
    auto map = AffineMap::get(1, 0, d0 * s);
    tileIds.push_back(
        b.create<AffineApplyOp>(loc, map,
                                isHoriz ? herd.getIds()[0] : herd.getIds()[1])
            .getResult());
  }
  SmallVector<Value, 4> tiledOperands = linalg::makeTiledShapes(
      b, loc, op, args, tileIds, tileSizeVector, sizeBounds, true);

  unsigned int resultIdx = 0;
  for (OpOperand *opOperand : op.getDpsInitOperands()) {
    resultIdx = opOperand->getOperandNumber();
    break;
  }

  Value firstOutputOperand = tiledOperands[resultIdx];
  SmallVector<ChannelOp> channels(pipeline_depth, nullptr);
  for (unsigned int i = 0; i < pipeline_depth; i++) {
    OpBuilder::InsertionGuard pipeline_guard(b);
    bool last_stage = i == pipeline_depth - 1;
    bool first_stage = i == 0;

    SmallVector<AffineExpr, 2> constraints{
        getAffineDimExpr(isHoriz ? 0 : 1, ctx) - getAffineConstantExpr(i, ctx),
        getAffineDimExpr(isHoriz ? 1 : 0, ctx)};
    SmallVector<bool, 2> eqflags{true, false};
    auto int_set = IntegerSet::get(2, 0, constraints, eqflags);
    SmallVector<Value, 2> int_set_args{x, y};
    AffineIfOp aif =
        b.create<AffineIfOp>(op->getLoc(), int_set, int_set_args, false);

    Block *stageBlock = aif.getBody();
    b.setInsertionPointToStart(stageBlock);

    if (i) {
      auto ty = tiledOperands[resultIdx].getType().cast<MemRefType>();
      auto alloc = b.create<memref::AllocOp>(
          loc, MemRefType::get(ty.getShape(), ty.getElementType(), AffineMap(),
                               (int)MemorySpace::L1));
      tiledOperands[resultIdx] = alloc.getResult();
      SmallVector<Value> src_offsets;
      SmallVector<Value> src_sizes;
      SmallVector<Value> src_strides;
      SmallVector<Value> channel_idx;
      SmallVector<Value> deps;
      SmallVector<Type> tys;
      b.create<ChannelGetOp>(loc, tys, deps, channels[i - 1].getSymName(),
                             channel_idx, tiledOperands[resultIdx], src_offsets,
                             src_sizes, src_strides);
    }

    linalg::LinalgOp linalgOp = clone(b, op, {}, tiledOperands);

    auto defaultCopyCallBack = [loc](OpBuilder &bldr, Value src,
                                     Value dst) -> LogicalResult {
      bldr.create<memref::CopyOp>(loc, src, dst);
      return success();
    };

    if (promote) {
      SmallVector<int64_t, 3> opers_to_promote(linalgOp->getNumOperands() - 1);
      std::iota(opers_to_promote.begin(), opers_to_promote.end(), 0);
      if (first_stage /* || last_stage*/)
        opers_to_promote.push_back(linalgOp->getNumOperands() - 1);

      auto emptyCopyCallBack = [](OpBuilder &bldr, Value src,
                                  Value dst) -> LogicalResult {
        return success();
      };
      b.setInsertionPoint(linalgOp);
      auto options = linalg::LinalgPromotionOptions()
                         .setOperandsToPromote(opers_to_promote)
                         .setAllocationDeallocationFns(allocBufferCallBack,
                                                       deallocBufferCallBack);
      if (first_stage)
        options.setCopyInOutFns(defaultCopyCallBack, emptyCopyCallBack);
      auto res = linalg::promoteSubViews(b, linalgOp, options);
      if (failed(res))
        return failure();
    }

    if (last_stage) {
      b.setInsertionPointAfter(linalgOp);
      (void)defaultCopyCallBack(b, tiledOperands[resultIdx],
                                firstOutputOperand);
      b.setInsertionPoint(stageBlock->getTerminator());
    } else {
      auto mref = tiledOperands[resultIdx];
      if (promote && first_stage) {
        memref::SubViewOp sv = dyn_cast<memref::SubViewOp>(
            linalgOp.getDpsInitOperand(0)->get().getDefiningOp());
        mref = sv.getSource();
        sv.replaceAllUsesWith(mref);
      }

      auto module = op->getParentOfType<ModuleOp>();
      auto cname = createChannelName(module);
      b.setInsertionPointToStart(module.getBody());
      auto channel_op = b.create<ChannelOp>(loc, cname, b.getI64ArrayAttr({1}));
      b.setInsertionPoint(stageBlock->getTerminator());
      SmallVector<Value> src_offsets;
      SmallVector<Value> src_sizes;
      SmallVector<Value> src_strides;
      SmallVector<Value> channel_idx;
      SmallVector<Value> deps;
      SmallVector<Type> tys;
      b.create<ChannelPutOp>(loc, tys, deps, FlatSymbolRefAttr::get(ctx, cname),
                             channel_idx, mref, src_offsets, src_sizes,
                             src_strides);
      channels[i] = channel_op;
    }
    // if (erased) erased.erase();
  }

  b.setInsertionPointToEnd(&herd.getBody().front());
  b.create<HerdTerminatorOp>(loc);
  int i = 0;
  for (auto a : args) {
    replaceAllUsesInRegionWith(a, herd.getKernelArgument(i++), herd.getBody());
  }
  return linalg::TiledLinalgOp{op, {herd}, {}};
}

DiagnosedSilenceableFailure transform::PipelineReduceOp::applyToOne(
    transform::TransformRewriter &rewriter, ::mlir::Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  linalg::LinalgOp linalgOp = llvm::dyn_cast<linalg::LinalgOp>(target);
  auto result = pipelineReduceLinalgOp(
      rewriter, linalgOp, extractFromI64ArrayAttr(getTileSize()),
      getPipelineDepth(), getDirection().str(), getPromote());
  if (failed(result))
    return emitDefiniteFailure() << "Failed";
  results.push_back(result->op);
  rewriter.eraseOp(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LinalgTileOp
//===----------------------------------------------------------------------===//

void transform::LinalgTileOp::build(OpBuilder &builder, OperationState &result,
                                    Value target,
                                    ArrayRef<int64_t> staticTileSizes,
                                    ArrayRef<int64_t> interchange) {
  return build(builder, result,
               /*target=*/target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               interchange);
}

void transform::LinalgTileOp::build(OpBuilder &builder, OperationState &result,
                                    Value target,
                                    ArrayRef<OpFoldResult> mixedTileSizes,
                                    ArrayRef<int64_t> interchange) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = pdl::OperationType::get(ctx);
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*dynamic_sizes=*/dynamicTileSizes,
        /*static_sizes=*/staticTileSizesAttr,
        /*interchange=*/builder.getDenseI64ArrayAttr(interchange));
}

DiagnosedSilenceableFailure
transform::LinalgTileOp::apply(transform::TransformRewriter &rewriter,
                               TransformResults &transformResults,
                               TransformState &state) {
  ArrayRef<int64_t> tileSizes = getStaticSizes();
  SmallVector<Operation *> targets;
  llvm::append_range(targets, state.getPayloadOps(getTarget()));
  SmallVector<SmallVector<Operation *>> dynamicSizeProducers;
  dynamicSizeProducers.reserve(getDynamicSizes().size());
  for (Value dynamicSizeProducerHandle : getDynamicSizes()) {
    llvm::append_range(dynamicSizeProducers.back(),
                       state.getPayloadOps(dynamicSizeProducerHandle));

    if (dynamicSizeProducers.back().size() != targets.size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected as many dynamic size-producing operations ("
          << dynamicSizeProducers.back().size() << ") as target ops ("
          << targets.size() << ")";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }

    for (Operation *op : dynamicSizeProducers.back()) {
      if (op->getNumResults() == 1 &&
          op->getResult(0).getType().isa<IndexType>())
        continue;
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "expected sizes to be produced by ops "
                                    "with a single index-type result";
      diag.attachNote(op->getLoc()) << "size producer op";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }
  }

  SmallVector<Operation *> tiled;
  SmallVector<SmallVector<Operation *, 4>, 4> loops;
  loops.resize(getLoops().size());
  for (auto en : llvm::enumerate(targets)) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(en.value());
    if (!linalgOp) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "only linalg ops are supported";
      diag.attachNote(en.value()->getLoc()) << "target op";
      return diag;
    }

    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);
    unsigned index = en.index();
    if (!tileSizes.empty()) {
      tilingOptions.setTileSizeComputationFunction(
          [&, index](OpBuilder &b, Operation *) {
            SmallVector<Value, 4> sizes;
            sizes.reserve(tileSizes.size());
            unsigned dynamicIdx = 0;
            for (OpFoldResult ofr : getMixedSizes()) {
              if (auto attr = ofr.dyn_cast<Attribute>()) {
                sizes.push_back(b.create<arith::ConstantIndexOp>(
                    getLoc(), attr.cast<IntegerAttr>().getInt()));
              } else {
                sizes.push_back(
                    dynamicSizeProducers[dynamicIdx++][index]->getResult(0));
              }
            }
            return sizes;
          });
    }

    SmallVector<unsigned int> inter(getInterchange());
    tilingOptions.setInterchange(inter);
    FailureOr<linalg::TiledLinalgOp> maybeTilingResult =
        linalg::tileLinalgOp(rewriter, linalgOp, tilingOptions);
    if (failed(maybeTilingResult))
      return DiagnosedSilenceableFailure::definiteFailure();

    if (linalgOp.hasBufferSemantics())
      rewriter.eraseOp(linalgOp);
    else
      rewriter.replaceOp(linalgOp,
                         maybeTilingResult->loops.front()->getResults());

    tiled.push_back(maybeTilingResult->op);
    for (const auto &en2 : llvm::enumerate(maybeTilingResult->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(getTiledLinalgOp().cast<OpResult>(), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(getLoops()[en.index()].cast<OpResult>(), en.value());

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::LinalgTileOp::getMixedSizes() {
  ValueRange dynamic = getDynamicSizes();
  ArrayRef<int64_t> tileSizes = getStaticSizes();
  SmallVector<OpFoldResult> results;
  results.reserve(tileSizes.size());
  unsigned dynamicPos = 0;
  Builder builder(getContext());
  for (int64_t size : tileSizes) {
    if (size == ShapedType::kDynamic) {
      results.push_back(dynamic[dynamicPos++]);
    } else {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

// We want to parse `DenseI64ArrayAttr` using the short form without the
// `array` prefix to be consistent in the IR with `parseDynamicIndexList`.
static ParseResult parseInterchange(OpAsmParser &parser,
                                    OperationState &result) {
  if (succeeded(parser.parseOptionalLBrace())) {
    if (failed(parser.parseKeyword("interchange")))
      return parser.emitError(parser.getNameLoc()) << "expect `interchange`";
    if (failed(parser.parseEqual()))
      return parser.emitError(parser.getNameLoc()) << "expect `=`";
    result.addAttribute("interchange",
                        DenseI64ArrayAttr::parse(parser, Type{}));
    if (failed(parser.parseRBrace()))
      return parser.emitError(parser.getNameLoc()) << "expect `}`";
  }
  return success();
}

static void printInterchange(OpAsmPrinter &p,
                             ArrayRef<int64_t> interchangeVals) {
  if (!interchangeVals.empty()) {
    p << " {interchange = [";
    llvm::interleaveComma(interchangeVals, p,
                          [&](int64_t integer) { p << integer; });
    p << "]}";
  }
}

ParseResult transform::LinalgTileOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  DenseI64ArrayAttr staticSizes;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes) ||
      parser.resolveOperands(dynamicSizes, pdlOperationType, result.operands))
    return ParseResult::failure();

  // Parse optional interchange.
  if (failed(parseInterchange(parser, result)))
    return ParseResult::failure();

  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  size_t numExpectedLoops =
      staticSizes.size() - llvm::count(staticSizes.asArrayRef(), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOperationType));
  return success();
}

void transform::LinalgTileOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes());
  printInterchange(p, getInterchange());
}

void transform::LinalgTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  onlyReadsHandle(getDynamicSizes(), effects);
  producesHandle(getTiledLinalgOp(), effects);
  producesHandle(getLoops(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LinalgPromoteOp
//===----------------------------------------------------------------------===//

// Replace a pattern like this:
// %7 = memref.alloc() : memref<20736xi8>
// %8 = memref.view %7[%c0][] : memref<20736xi8> to
// memref<1x16x18x18xf32> With this %7 = memref.alloc() : memref<
// 1x16x18x18xf32, 2>
struct RemoveSubViewOpsPattern : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  RemoveSubViewOpsPattern(MLIRContext *ctx, unsigned int fast_memory_space = 1)
      : OpRewritePattern(ctx), fast_space(fast_memory_space) {}

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto view = op.getSource().getDefiningOp<memref::ViewOp>();
    if (!view)
      return failure();
    auto alloc = view.getSource().getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    /* Force memory space */
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op,
        MemRefType::get(op.getType().getShape(), op.getType().getElementType(),
                        {}, fast_space),
        op.sizes());
    alloc.replaceAllUsesWith(newOp);
    return success();
  }

private:
  unsigned int fast_space;
};

void xilinx::air::populateRemoveSubViewOpsPattern(
    RewritePatternSet &patterns, unsigned int fast_memory_space) {
  patterns.add<RemoveSubViewOpsPattern>(patterns.getContext(),
                                        fast_memory_space);
}

struct FoldSubViewOpsPattern : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    if (!dyn_cast_or_null<memref::SubViewOp>(op.getSource().getDefiningOp()))
      return failure();

    auto source_subview =
        cast<memref::SubViewOp>(op.getSource().getDefiningOp());

    // FIXME: do we still need this?
    // for (auto m : llvm::zip(source_subview.getType().getLayout(),
    //                         op.getType().getLayout()))
    //   if (std::get<0>(m) != std::get<1>(m))
    //     return failure();

    auto offsets = op.offsets().begin();
    auto source_offsets = source_subview.offsets().begin();
    SmallVector<Value, 4> result_offsets;

    auto static_offsets = op.getStaticOffsets();
    auto source_static_offsets = source_subview.getStaticOffsets();
    SmallVector<int64_t, 4> result_static_offsets;

    for (auto p : llvm::zip(static_offsets, source_static_offsets)) {
      auto op_offset = std::get<0>(p);
      auto source_offset = std::get<1>(p);
      if (op_offset >= 0 && source_offset >= 0) {
        result_static_offsets.push_back(op_offset + source_offset);
      } else if (op_offset < 0 && source_offset >= 0) {
        result_static_offsets.push_back(op_offset);
        if (source_offset == 0) {
          result_offsets.push_back(*offsets++);
        } else {
          Value a = *offsets++;
          Value b = rewriter.create<arith::ConstantIndexOp>(op.getLoc(),
                                                            source_offset);
          result_offsets.push_back(
              rewriter.create<arith::AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset >= 0 && source_offset < 0) {
        result_static_offsets.push_back(source_offset);
        if (op_offset == 0) {
          result_offsets.push_back(*source_offsets++);
        } else {
          Value a = *source_offsets++;
          Value b =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), op_offset);
          result_offsets.push_back(
              rewriter.create<arith::AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset < 0 && source_offset < 0) {
        Value a = *source_offsets++;
        Value b = *offsets++;
        result_offsets.push_back(
            rewriter.create<arith::AddIOp>(op.getLoc(), a.getType(), a, b));
        result_static_offsets.push_back(source_offset);
      }
    }

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        op.getOperation(), op.getType(), source_subview.getSource(),
        result_offsets, op.sizes(), op.strides(),
        rewriter.getDenseI64ArrayAttr(result_static_offsets),
        op.getStaticSizes(), op.getStaticStrides());

    return success();
  }
};

void xilinx::air::populateFoldSubViewOpsPattern(RewritePatternSet &patterns) {
  patterns.add<FoldSubViewOpsPattern>(patterns.getContext());
}

struct RemoveViewOpsPattern : public OpRewritePattern<memref::ViewOp> {
  using OpRewritePattern<memref::ViewOp>::OpRewritePattern;

  RemoveViewOpsPattern(MLIRContext *ctx, unsigned int fast_memory_space = 1)
      : OpRewritePattern(ctx), fast_space(fast_memory_space) {}

  LogicalResult matchAndRewrite(memref::ViewOp op,
                                PatternRewriter &rewriter) const override {
    auto alloc = op.getSource().getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    /* Force memory space */
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op,
        MemRefType::get(op.getType().getShape(), op.getType().getElementType(),
                        {}, fast_space),
        op.getSizes());
    alloc.replaceAllUsesWith(newOp);
    return success();
  }

private:
  unsigned int fast_space;
};

void xilinx::air::populateRemoveViewOpsPattern(RewritePatternSet &patterns,
                                               unsigned int fast_memory_space) {
  patterns.add<RemoveViewOpsPattern>(patterns.getContext(), fast_memory_space);
}

// Replace a pattern like this:
//  %alloc_1 = memref.alloc() : memref<?x?xi32, 2>
//  ...
//  memref.copy %alloc_1, %m : memref<?x?xi32, 2> to memref<?x?xi32, 2>
//  memref.dealloc %alloc_1 : memref<?x?xi32, 2>
//  %alloc_2 = memref.alloc() : memref<?x?xi32, 2>
//  memref.copy %m, %alloc_2 : memref<?x?xi32, 2> to memref<?x?xi32, 2>
// with this:
//  %alloc_1 = memref.alloc() : memref<?x?xi32, 2>
//  ...
//  memref.copy %alloc_1, %m : memref<?x?xi32, 2> to memref<?x?xi32, 2>
//  [ and replace uses of %alloc_2 with %alloc_1 ]
struct RemoveExtraAllocPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {

    auto existingAlloc =
        dyn_cast<memref::AllocOp>(op.getOperand(0).getDefiningOp());
    if (!existingAlloc)
      return failure();

    auto iter = op->getIterator();
    auto deallocOp = dyn_cast<memref::DeallocOp>(++iter);
    if (!deallocOp)
      return failure();

    auto allocOp = dyn_cast<memref::AllocOp>(++iter);
    if (!allocOp)
      return failure();
    if (allocOp.getType() != existingAlloc.getType())
      return failure();

    auto copyOp = dyn_cast<memref::CopyOp>(++iter);
    if (!copyOp)
      return failure();

    if (op.getOperand(0) != deallocOp.getOperand())
      return failure();

    if (op.getOperand(1) != copyOp.getOperand(0))
      return failure();

    if (allocOp.getResult() != copyOp.getOperand(1))
      return failure();

    rewriter.replaceAllUsesWith(allocOp.getResult(), {op.getOperand(0)});
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(allocOp);
    rewriter.eraseOp(deallocOp);

    return success();
  }
};

void xilinx::air::populateRemoveExtraAllocPattern(RewritePatternSet &patterns) {
  patterns.add<RemoveExtraAllocPattern>(patterns.getContext());
}

// Replace a pattern like this:
//  memref.copy %1, %2 : memref<?xi32> to memref<?xi32>
//  linalg op with write to %1, no use of %2
//  memref.copy %1, %2 : memref<?xi32> to memref<?xi32>
// with this:
//  linalg op with write to %1, no use of %2
//  memref.copy %1, %2 : memref<?xi32> to memref<?xi32>
struct RemoveDeadCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {

    auto iter = op->getIterator();
    auto linalgOp = dyn_cast<linalg::LinalgOp>(++iter);
    if (!linalgOp)
      return failure();
    auto copyOp = dyn_cast<memref::CopyOp>(++iter);
    if (!copyOp)
      return failure();

    auto oper0 = copyOp->getOperand(0);
    auto oper1 = copyOp->getOperand(1);
    if (op.getOperand(0) != oper0)
      return failure();
    if (op.getOperand(1) != oper1)
      return failure();

    // no use of %2
    auto lopers = linalgOp->getOperands();
    if (std::find(lopers.begin(), lopers.end(), oper1) != lopers.end())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

void xilinx::air::populateRemoveDeadCopyPattern(RewritePatternSet &patterns) {
  patterns.add<RemoveDeadCopyPattern>(patterns.getContext());
}

// Replace a pattern like this:
//  %sv = memref.subview ...
//  %alloc = memref.alloc() : memref<...>
//  memref.copy %sv, %alloc
//  linalg.generic with outs(%alloc : memref<...>), does not read %alloc
// with this:
//  %sv = memref.subview ...
//  %alloc = memref.alloc() : memref<...>
//  linalg.generic with outs(%alloc : memref<...>), does not read %alloc
// that is, remove the no-op copy.
struct RemoveAllocCopyLinalgOpCopyPattern
    : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {

    // the target of the copy is an alloc
    Operation *allocOp =
        dyn_cast_if_present<memref::AllocOp>(op.getTarget().getDefiningOp());
    if (!allocOp)
      return failure();

    // find the next linalg use in this block
    linalg::LinalgOp linalgOp = nullptr;
    for (auto &u : allocOp->getResult(0).getUses()) {
      if (auto l = dyn_cast<linalg::LinalgOp>(u.getOwner())) {
        // bail without trying to resolve the ordering
        // if there's a linalg use in a different block
        if (l->getBlock() != op->getBlock())
          return failure();
        if (l.payloadUsesValueFromOperand(&u))
          continue;
        // take the earliest use
        if (linalgOp && linalgOp->isBeforeInBlock(l))
          continue;
        linalgOp = l;
      }
    }
    if (!linalgOp)
      return failure();

    for (auto &u : allocOp->getResult(0).getUses()) {
      auto use = u.getOwner();
      if (use == op)
        continue;
      // if there's a use between the copy and the linalg op
      if (!isa<linalg::LinalgOp>(use)) {
        if (use->getBlock() != op->getBlock())
          continue;
        if (use->isBeforeInBlock(linalgOp))
          return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

void xilinx::air::populateRemoveAllocCopyLinalgOpCopyPattern(
    RewritePatternSet &patterns) {
  patterns.add<RemoveAllocCopyLinalgOpCopyPattern>(patterns.getContext());
}

DiagnosedSilenceableFailure
transform::LinalgPromoteOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {

  SmallVector<Operation *> payloadOps;
  llvm::append_range(payloadOps, state.getPayloadOps(getTarget()));
  if (!payloadOps.size())
    DiagnosedSilenceableFailure::success();

  linalg::LinalgPromotionOptions promotionOptions;
  auto operandsToPromote = extractFromI64ArrayAttr(getOperandsToPromote());

  if (getUseFullTilesByDefault())
    promotionOptions = promotionOptions.setUseFullTileBuffersByDefault(
        getUseFullTilesByDefault());
  if (getUseAlloca())
    promotionOptions = promotionOptions.setUseAlloca(getUseAlloca());
  if (!getUseFullTileBuffers().empty())
    promotionOptions = promotionOptions.setUseFullTileBuffers(
        llvm::to_vector(getUseFullTileBuffers().getAsValueRange<BoolAttr>()));
  if (getAlignment().has_value())
    promotionOptions = promotionOptions.setAlignment(*getAlignment());

  auto memorySpace = MemorySpace::L1;
  if (getMemorySpace() == "L1")
    memorySpace = MemorySpace::L1;
  else if (getMemorySpace() == "L2")
    memorySpace = MemorySpace::L2;
  else if (getMemorySpace() == "L3")
    memorySpace = MemorySpace::L3;

  SetVector<Operation *> transformed;
  int64_t operandOffset = 0;

  uint32_t group_size = getGroupSize();
  uint32_t group = 0;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(target);
    if (!linalgOp)
      continue;

    int64_t numOperands = linalgOp->getNumOperands();
    SmallVector<int64_t, 4> opersToPromote;
    if (!operandsToPromote.size()) {
      opersToPromote.resize_for_overwrite(numOperands);
      std::iota(opersToPromote.begin(), opersToPromote.end(), 0);
    } else {
      for (auto &o : operandsToPromote) {
        int64_t operand = o - operandOffset;
        if (operand < 0)
          continue;
        if (operand >= numOperands)
          continue;
        opersToPromote.push_back(operand);
      }
    }
    operandOffset += numOperands;
    if (++group == group_size) {
      group = 0;
      operandOffset = 0;
    }
    if (opersToPromote.empty())
      continue;

    promotionOptions.setOperandsToPromote(opersToPromote);

    if (failed(promoteSubviewsPrecondition(target, promotionOptions)))
      return emitDefaultDefiniteFailure(target);

    auto ctx = target->getContext();
    rewriter.setInsertionPoint(target);
    FailureOr<linalg::LinalgOp> res = promoteSubViews(
        rewriter, llvm::dyn_cast<linalg::LinalgOp>(target), promotionOptions);
    if (failed(res))
      return emitDefaultDefiniteFailure(target);

    transformed.insert(linalgOp);
  }

  auto ctx = payloadOps[0]->getContext();
  RewritePatternSet patterns(ctx);
  // promoteSubViews generates extra copies and subviews, these patterns try to
  // simplify them.
  patterns.insert<RemoveSubViewOpsPattern>(ctx, (int)memorySpace);
  patterns.insert<FoldSubViewOpsPattern, RemoveViewOpsPattern>(ctx);
  patterns.insert<RemoveExtraAllocPattern, RemoveDeadCopyPattern,
                  RemoveAllocCopyLinalgOpCopyPattern>(ctx);
  // canonicalize allocs like:
  //  memref.alloc(%c32, %c32) : memref<?x?xi32, 2>
  // to:
  //  memref.alloc() : memref<32x32xi32, 2>
  memref::AllocOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsAndFoldGreedily(
      payloadOps[0]->getParentOfType<func::FuncOp>(), std::move(patterns));

  if (!transformed.size())
    return emitDefaultDefiniteFailure(payloadOps[0]);

  results.set(getResult().cast<OpResult>(), transformed.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

void transform::LinalgPromoteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  producesHandle(getResult(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// FuseIntoContainingMemrefOp
//===----------------------------------------------------------------------===//

void transform::FuseIntoContainingMemrefOp::build(OpBuilder &builder,
                                                  OperationState &result,
                                                  Value producerOp,
                                                  Value containingOp) {
  result.addOperands({producerOp, containingOp});
  result.addTypes(pdl::OperationType::get(builder.getContext()));
}

void transform::FuseIntoContainingMemrefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerOp(), effects);
  onlyReadsHandle(getContainingOp(), effects);
  producesHandle(getFusedOp(), effects);
  modifiesPayload(effects);
}

static FailureOr<linalg::LinalgOp>
generateResultTileValue(Operation *op, Operation *forOp, OpBuilder &b,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  auto loc = op->getLoc();
  SmallVector<Value> args;
  for (auto o : op->getOperands())
    args.push_back(o);
  auto allShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();
  SmallVector<OpFoldResult> sizeBounds =
      makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                               allShapeSizes);
  SmallVector<OpFoldResult, 2> ivs =
      cast<scf::ParallelOp>(forOp).getInductionVars();
  SmallVector<Value> tiledOperands = linalg::makeTiledShapes(
      b, op->getLoc(), linalgOp, args, ivs, sizes, sizeBounds, true);

  SmallVector<Value> operands;
  auto ti = tiledOperands.begin();
  for (auto o : op->getOperands()) {
    if (isa<MemRefType>(o.getType()))
      operands.push_back(*ti);
    else
      operands.push_back(o);
    ti++;
  }

  linalg::LinalgOp newLinalgOp =
      llvm::dyn_cast<linalg::LinalgOp>(clone(b, op, {}, operands));
  return newLinalgOp;
}

/// Find the first subview user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static Operation *tileAndFuseFirstExtractUse(RewriterBase &rewriter,
                                             Diagnostic &diag,
                                             Operation *producerOp,
                                             Operation *containingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return nullptr;
  }

  linalg::LinalgOp producerLinalgOp = cast<linalg::LinalgOp>(producerOp);
  auto users = producerLinalgOp.getDpsInitOperands()[0]->get().getUsers();
  auto it = llvm::find_if(users, [&](Operation *user) {
    auto sliceOp = dyn_cast<memref::SubViewOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (it == users.end()) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find fusion opportunity for: " << *tileableProducer;
    return nullptr;
  }
  auto sliceOpToTile = cast<memref::SubViewOp>(*it);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  FailureOr<linalg::LinalgOp> tiledProducer = generateResultTileValue(
      producerOp, containingOp, rewriter, sliceOpToTile.getMixedOffsets(),
      sliceOpToTile.getMixedSizes());
  if (failed(tiledProducer)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "tiledProducer: " << *tiledProducer << "\n");

  // Replace the extract op.
  rewriter.replaceOp(sliceOpToTile,
                     tiledProducer.value().getDpsInitOperand(0)->get());
  return *tiledProducer;
}

DiagnosedSilenceableFailure transform::FuseIntoContainingMemrefOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Operation *> fusedOps;
  SmallVector<Operation *> producerOps;
  llvm::append_range(producerOps, state.getPayloadOps(getProducerOp()));
  // If nothing to fuse, propagate success.
  if (producerOps.empty()) {
    results.set(getFusedOp().cast<OpResult>(),
                SmallVector<mlir::Operation *>{});
    return DiagnosedSilenceableFailure::success();
  }
  if (producerOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one producer_op handle (got "
           << producerOps.size() << ")";
  }
  Operation *producerOp = producerOps.front();

  SmallVector<Operation *> containingOps;
  llvm::append_range(containingOps, state.getPayloadOps(getContainingOp()));
  if (containingOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one containing_op handle (got "
           << containingOps.size() << ")";
  }
  Operation *containingOp = containingOps.front();

  linalg::LinalgOp producerLinalgOp = dyn_cast<linalg::LinalgOp>(producerOp);
  if (!producerLinalgOp) {
    return emitDefiniteFailure() << "requires producer_op to be LinalgOp";
  }
  if (producerLinalgOp.getNumDpsInits() != 1) {
    return emitDefiniteFailure()
           << "requires producer_op to have exactly one init operand (got "
           << producerLinalgOp.getNumDpsInits() << ")";
  }

  auto initOperand = producerLinalgOp.getDpsInitOperands()[0]->get();
  // The containing op may be a user of producerOp: use isAncestor.
  int64_t numUsesInContainingOp =
      llvm::count_if(initOperand.getUsers(), [&](Operation *op) {
        return containingOp->isAncestor(op);
      });
  if (numUsesInContainingOp == 0) {
    results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation *>());
    Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "producer_op does not have uses in the container";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  // Default diagnostic, to be complemented with more failure information.
  Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
  diag << "could not fuse " << *producerOp << " into " << *containingOp;

  Operation *tiled =
      tileAndFuseFirstExtractUse(rewriter, diag, producerOp, containingOp);
  if (tiled) {
    LLVM_DEBUG(llvm::dbgs() << "\nFused a direct extract use\n"
                            << *containingOp);
    fusedOps.push_back(tiled);
    rewriter.eraseOp(producerOp);

    results.set(getFusedOp().cast<OpResult>(), fusedOps);
    return DiagnosedSilenceableFailure::success();
  }

  results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation *>());
  return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
}

namespace {
class AIRTransformDialectExtension
    : public transform::TransformDialectExtension<
          AIRTransformDialectExtension> {
public:
  AIRTransformDialectExtension() {
    declareDependentDialect<func::FuncDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "air/Dialect/AIR/AIRTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIRTransformOps.cpp.inc"

void xilinx::air::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AIRTransformDialectExtension>();
}
