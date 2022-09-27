//===- ConvertToAIRPass.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/ConvertToAIRPass.h"
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

#define DEBUG_TYPE "convert-to-air"

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

class AffineParToHerdConversion
    : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDims() == 2) {
      auto loc = op.getLoc();
      auto ub0 = op.getUpperBoundsMap().getResult(0).cast<AffineConstantExpr>();
      auto ub1 = op.getUpperBoundsMap().getResult(1).cast<AffineConstantExpr>();
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
          rewriter.create<arith::ConstantIndexOp>(loc, ub0.getValue()),
          rewriter.create<arith::ConstantIndexOp>(loc, ub1.getValue())};
      auto launch = rewriter.create<air::HerdOp>(op.getLoc(), dims, args);

      if (auto attr =
              op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        launch->setAttr(SymbolTable::getSymbolAttrName(), attr);

      auto &bb = launch.body().front();
      auto ivs = op.getIVs();
      ivs[0].replaceAllUsesWith(launch.getIds()[0]);
      ivs[1].replaceAllUsesWith(launch.getIds()[1]);
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

LogicalResult normalizeScfParallel(scf::ParallelOp parOp,
                                   PatternRewriter &rewriter) {
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
    float ub = (*upperBound++).getDefiningOp<arith::ConstantIndexOp>().value();
    new_ub.push_back(rewriter.create<arith::ConstantIndexOp>(
        loc, (uint64_t)ceil((ub - lb) / s)));
    new_lb.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    new_step.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    auto iv = *ivs++;
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
                               llvm::SmallSet<scf::ParallelOp, 2> &filteredOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps){};

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
    SmallVector<Value, 2> dims{rewriter.create<arith::ConstantIndexOp>(loc, bounds[0]),
                       rewriter.create<arith::ConstantIndexOp>(loc, bounds[1])};
    auto launch = rewriter.create<air::HerdOp>(op.getLoc(), dims, args);
    auto &bb = launch.body().front();
    auto ivs = op.getInductionVars();

    ivs[0].replaceAllUsesWith(launch.getIds()[0]);
    if (op.getNumLoops() == 2)
      ivs[1].replaceAllUsesWith(launch.getIds()[1]);

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

class ScfParToLaunchConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToLaunchConversion(MLIRContext *ctx,
                           llvm::SmallSet<scf::ParallelOp, 2> &filteredOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps){};

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
    auto launch = rewriter.create<air::LaunchOp>(op.getLoc(), sizes, args);
    auto &bb = launch.body().front();
    auto ivs = op.getInductionVars();

    for (int i = 0, e = ivs.size(); i < e; i++) {
      ivs[i].replaceAllUsesWith(launch.getIds()[i]);
    }

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&launch.getRegion().front());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, rewriter.clone(*c.getDefiningOp())->getResult(0),
          launch.getRegion());
    }

    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<air::LaunchTerminatorOp>(loc);

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

class ScfParToLaunchAndPartitionConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToLaunchAndPartitionConversion(MLIRContext *ctx,
                           llvm::SmallSet<scf::ParallelOp, 2> &filteredOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps){};

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
    auto launch = rewriter.create<air::LaunchOp>(op.getLoc(), sizes, args);
    rewriter.setInsertionPointToStart(&launch.getRegion().front());
    SmallVector<Value, 1> partitionSizes = {};
    SmallVector<Value, 4> partitionOpers;
    for (Value v : launch.getIds()) {
      partitionOpers.push_back(v);
    }
    for (Value v : launch.getSize()) {
      partitionOpers.push_back(v);
    }
    for (Value v : launch.getKernelArguments()) {
      partitionOpers.push_back(v);
    }
    auto partition = rewriter.create<air::PartitionOp>(op.getLoc(), partitionSizes, partitionOpers);
    auto &bb = partition.body().front();
    auto ivs = op.getInductionVars();

    for (int i = 0, e = ivs.size(); i < e; i++) {
      ivs[i].replaceAllUsesWith(partition.getKernelArgument(i));
    }

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&partition.getRegion().front());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, rewriter.clone(*c.getDefiningOp())->getResult(0),
          partition.getRegion());
    }

    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<air::PartitionTerminatorOp>(builder.getUnknownLoc());
    builder = OpBuilder::atBlockEnd(&launch.body().front());
    builder.create<air::LaunchTerminatorOp>(builder.getUnknownLoc());

    int i = 0;
    auto kernel_args = partition.getKernelArguments();
    kernel_args = kernel_args.drop_front(ivs.size() + launch.getSize().size()); // Launch's induction vars
    for (Value v : args)
      replaceAllUsesInRegionWith(v, kernel_args[i++], partition.getRegion());

    if (op != parOp)
      op.erase();
    rewriter.eraseOp(parOp);

    return success();
  }

private:
  llvm::SmallSet<scf::ParallelOp, 2> &filteredOps;
};

struct CopyToDmaPass : public CopyToDmaBase<CopyToDmaPass> {

  CopyToDmaPass() = default;
  CopyToDmaPass(const CopyToDmaPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           xilinx::air::airDialect, arith::ArithmeticDialect>();

    target.addLegalOp<AffineApplyOp, AffineForOp, AffineLoadOp, AffineStoreOp,
                      AffineYieldOp, scf::YieldOp>();

    DmaMemcpyOpID = 0;

    // Simplify all the subviews so we can rewrite them easily.
    // Mostly this is propagating constant sizes into dimensioned memref types.
    RewritePatternSet stage1Patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    memref::AllocOp::getCanonicalizationPatterns(stage1Patterns, context);
    (void)applyPatternsAndFoldGreedily(module, std::move(stage1Patterns));

    RewritePatternSet stage2Patterns(context);
    stage2Patterns
        .insert<LinalgCopyToAIRDmaConversion, MemrefCopyToAIRDmaConversion>(
            context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(stage2Patterns)))) {
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
      f.walk([&](xilinx::air::HerdOp op) {
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
      f.walk([&](xilinx::air::HerdOp op) {
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

struct ParallelToHerdPass : public ParallelToHerdBase<ParallelToHerdPass> {

  ParallelToHerdPass() = default;
  ParallelToHerdPass(const ParallelToHerdPass &pass) {}

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
    patterns.add<AffineParToHerdConversion>(context);

    llvm::SmallVector<xilinx::air::HerdOp, 2> herdOps;
    module.walk([&](xilinx::air::HerdOp op) { herdOps.push_back(op); });
    llvm::SmallVector<xilinx::air::LaunchOp, 2> launchOps;
    module.walk([&](xilinx::air::LaunchOp op) { launchOps.push_back(op); });

    llvm::SmallSet<scf::ParallelOp, 2> filteredOps;
    module.walk([&](scf::ParallelOp op) {
      if (op->getParentOfType<xilinx::air::HerdOp>())
        return;
      for (auto &h : herdOps)
        if (op->isProperAncestor(h))
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
      Operation *par = op.getOperation();
      while ((par = par->getParentOp()))
        if (isa<scf::ParallelOp>(par))
          parallel_depth++;
      if (parallel_depth != clAssignDepth)
        return;
      filteredOps.insert(op);
    });
    patterns.add<ScfParToHerdConversion>(context, filteredOps);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           xilinx::air::airDialect, arith::ArithmeticDialect>();

    target.addLegalOp<AffineApplyOp, AffineForOp, AffineLoadOp, AffineStoreOp,
                      AffineYieldOp, scf::YieldOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    std::vector<std::string> herd_syms;
    for (auto f : module.getOps<func::FuncOp>()) {
      // record existing symbol names
      f.walk([&](xilinx::air::HerdOp op) {
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
      f.walk([&](xilinx::air::HerdOp op) {
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

struct ParallelToLaunchPass
    : public ParallelToLaunchBase<ParallelToLaunchPass> {

  ParallelToLaunchPass() = default;
  ParallelToLaunchPass(const ParallelToLaunchPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    llvm::SmallVector<xilinx::air::LaunchOp, 2> launchOps;
    module.walk([&](xilinx::air::LaunchOp op) { launchOps.push_back(op); });

    llvm::SmallSet<scf::ParallelOp, 2> filteredOps;
    module.walk([&](scf::ParallelOp op) {
      if (op->getParentOfType<xilinx::air::HerdOp>())
        return;
      if (op->getParentOfType<xilinx::air::LaunchOp>())
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
      Operation *par = op.getOperation();
      while ((par = par->getParentOp()))
        if (isa<scf::ParallelOp>(par))
          parallel_depth++;
      if (parallel_depth != clAssignDepth)
        return;
      filteredOps.insert(op);
    });

    RewritePatternSet patterns(context);
    if (clHasPartition){
      patterns.add<ScfParToLaunchAndPartitionConversion>(context, filteredOps);
    }
    else {
      patterns.add<ScfParToLaunchConversion>(context, filteredOps);
  
    }

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           xilinx::air::airDialect, arith::ArithmeticDialect>();

    target.addLegalOp<AffineApplyOp, AffineForOp, AffineLoadOp, AffineStoreOp,
                      AffineYieldOp, scf::YieldOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createParallelToHerdPass() {
  return std::make_unique<ParallelToHerdPass>();
}

std::unique_ptr<mlir::Pass> createParallelToLaunchPass() {
  return std::make_unique<ParallelToLaunchPass>();
}

std::unique_ptr<mlir::Pass> createCopyToDmaPass() {
  return std::make_unique<CopyToDmaPass>();
}

} // namespace air
} // namespace xilinx