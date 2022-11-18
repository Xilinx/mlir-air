//===- ConvertToAIRPass.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/ConvertToAIRPass.h"
#include "PassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Dependency.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
    auto src = op.getSource();
    auto dst = op.getTarget();

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

      src = subview.getSource();
    }

    if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
      extractOperandsFromSubview(subview, dst_offsets, dst_sizes, dst_strides);

      if (dst_sizes.size() != rank)
        return failure();
      if (dst_strides.size() != rank)
        return failure();

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

static void extractOperandsFromSubview(memref::SubViewOp subview,
                                       OpBuilder &builder,
                                       SmallVector<Value, 4> &offsets,
                                       SmallVector<Value, 4> &sizes,
                                       SmallVector<Value, 4> &strides) {
  auto subview_offsets = subview.offsets().begin();
  auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
  auto static_sizes = extractFromI64ArrayAttr(subview.static_sizes());
  auto static_strides = extractFromI64ArrayAttr(subview.static_strides());
  auto loc = subview.getLoc();

  // get the strides and offsets from the memref type
  auto inferredType =
      memref::SubViewOp::inferResultType(
          subview.getSourceType(), static_offsets, static_sizes, static_strides)
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
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, o));
    else
      offsets.push_back(*subview_offsets++);
  }
  for (auto s : static_sizes)
    sizes.push_back(builder.create<arith::ConstantIndexOp>(loc, s));
  for (auto s : layout_strides)
    strides.push_back(builder.create<arith::ConstantIndexOp>(loc, s));
}

static void addReduceOpToAsyncParallel(OpBuilder builder,
                                       scf::ParallelOp scf_par,
                                       MLIRContext *ctx) {

  builder.setInsertionPointToEnd(scf_par.getBody());
  auto wait_all_op_yielded = builder.create<air::WaitAllOp>(
      scf_par.getLoc(), air::AsyncTokenType::get(ctx), SmallVector<Value, 1>{});
  auto reduce_op = builder.create<scf::ReduceOp>(
      scf_par.getLoc(), wait_all_op_yielded.getResult(0));
  builder.setInsertionPointToStart(&reduce_op.getRegion().front());
  SmallVector<Value, 4> reduce_tokens;
  reduce_tokens.push_back(reduce_op.getRegion().front().getArgument(0));
  reduce_tokens.push_back(reduce_op.getRegion().front().getArgument(1));
  auto reduce_res = builder.create<xilinx::air::WaitAllOp>(
      builder.getUnknownLoc(), air::AsyncTokenType::get(ctx), reduce_tokens);
  builder.create<scf::ReduceReturnOp>(builder.getUnknownLoc(),
                                      reduce_res.getResult(0));
  builder.setInsertionPointToEnd(scf_par.getBody());
  builder.create<scf::YieldOp>(scf_par.getLoc());

  wait_all_op_yielded->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
  reduce_op->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
  reduce_res->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
}

static scf::ParallelOp hoistHerdToAsyncParallel(OpBuilder builder, Location loc,
                                                MLIRContext *ctx,
                                                air::HerdOp herd) {
  auto herd_size = herd.getSizeOperands();

  auto lb = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 2> lbs{lb, lb};
  SmallVector<Value, 2> ubs{herd_size[0], herd_size[1]};
  SmallVector<Value, 2> steps{step, step};

  auto wa_op = builder.create<xilinx::air::WaitAllOp>(
      loc, air::AsyncTokenType::get(ctx), SmallVector<Value, 1>{});
  SmallVector<Value, 1> deps_in{wa_op.getAsyncToken()};
  auto scf_par = builder.create<scf::ParallelOp>(loc, lbs, ubs, steps, deps_in);

  addReduceOpToAsyncParallel(builder, scf_par, ctx);

  scf_par->setAttr("hoist-channel", StringAttr::get(ctx, "hoistedLoop"));
  scf_par->setAttr("loop-carried-dep", StringAttr::get(ctx, "hoistedLoop"));

  return scf_par;
}

static void updateUsesInScfFor(OpBuilder builder, scf::ForOp new_loop_op,
                               scf::ForOp loop_op) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  // Splice the operations inside loop op
  SmallVector<Value, 4> incoming_tokens;
  SmallVector<Value, 4> constants;
  llvm::SetVector<Value> region_args;
  getUsedValuesDefinedAbove(loop_op.getRegion(), region_args);
  for (Value v : region_args) {
    if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
      constants.push_back(v);
    else if (v.getDefiningOp()) {
      if (auto v_op =
              mlir::dyn_cast<air::AsyncOpInterface>(v.getDefiningOp())) {
        if (v_op.getAsyncToken() == v)
          incoming_tokens.push_back(v);
      } else if (auto v_op = dyn_cast<scf::ForOp>(v.getDefiningOp())) {
        if (v_op.getResult(0) == v)
          incoming_tokens.push_back(v);
      } else if (auto v_op = dyn_cast<scf::ParallelOp>(v.getDefiningOp())) {
        if (v_op.getResult(0) == v)
          incoming_tokens.push_back(v);
      }
    }
  }

  auto iv = loop_op.getInductionVar();
  replaceAllUsesInRegionWith(iv, new_loop_op.getInductionVar(),
                             new_loop_op.getRegion());
  if (loop_op.getRegionIterArgs().size()) {
    for (unsigned i = 0; i < loop_op.getRegionIterArgs().size(); i++) {
      auto ia = loop_op.getRegionIterArgs()[i];
      replaceAllUsesInRegionWith(ia, new_loop_op.getRegionIterArgs()[i],
                                 new_loop_op.getRegion());
    }
  }
  builder.setInsertionPointToStart(new_loop_op.getBody());
  for (auto c : constants) {
    replaceAllUsesInRegionWith(c,
                               builder.clone(*c.getDefiningOp())->getResult(0),
                               new_loop_op.getRegion());
  }

  for (Value v : incoming_tokens) {
    replaceAllUsesInRegionWith(v, new_loop_op.getRegionIterArgs()[0],
                               new_loop_op.getRegion());
  }

  builder.restoreInsertionPoint(insertionCheckpoint);
}

scf::YieldOp generateYieldOpFromChannelOp(OpBuilder builder, MLIRContext *ctx,
                                          scf::ForOp scf_for) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToEnd(scf_for.getBody());
  SmallVector<air::ChannelInterface, 1> channel_ops;
  for (auto channel_op : scf_for.getOps<air::ChannelInterface>()) {
    if (channel_op->hasAttr("hoist-channel")) {
      channel_ops.push_back(channel_op);
    }
  }
  assert(channel_ops.size() <= 1 &&
         "found multiple channel ops in one hoisted for loop");
  SmallVector<Value, 1> yield_token;
  if (channel_ops.size()) {
    assert(channel_ops[0]->getResult(0) &&
           "found sync channel op in async for loop");
    auto wa_op = builder.create<air::WaitAllOp>(
        builder.getUnknownLoc(), air::AsyncTokenType::get(ctx),
        SmallVector<Value, 1>{channel_ops[0]->getResult(0)});
    yield_token.push_back(wa_op.getAsyncToken());
    wa_op->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
  } else {
    auto wa_op = builder.create<air::WaitAllOp>(builder.getUnknownLoc(),
                                                air::AsyncTokenType::get(ctx),
                                                SmallVector<Value, 1>{});
    yield_token.push_back(wa_op.getAsyncToken());
    wa_op->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
  }
  scf::YieldOp output =
      builder.create<scf::YieldOp>(builder.getUnknownLoc(), yield_token);
  builder.restoreInsertionPoint(insertionCheckpoint);
  return output;
}

scf::ForOp cloneForUsingRemap(OpBuilder builder, BlockAndValueMapping remap,
                              scf::ForOp loop_op) {
  SmallVector<Value, 1> remap_iter_operands;
  for (auto iter_operand : loop_op.getIterOperands()) {
    remap_iter_operands.push_back(remap.lookupOrDefault(iter_operand));
  }
  scf::ForOp new_loop_op = builder.create<scf::ForOp>(
      builder.getUnknownLoc(), remap.lookupOrDefault(loop_op.getLowerBound()),
      remap.lookupOrDefault(loop_op.getUpperBound()),
      remap.lookupOrDefault(loop_op.getStep()), remap_iter_operands);
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(new_loop_op.getBody());
  for (Operation &for_child_op : loop_op.getBody()->getOperations()) {
    if (for_child_op.hasAttr("hoist-channel")) {
      builder.clone(for_child_op, remap);
    }
  }
  // Re-establish uses after hoisting
  updateUsesInScfFor(builder, new_loop_op, loop_op);

  new_loop_op->setAttr("hoist-channel",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));
  new_loop_op->setAttr("loop-carried-dep",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));

  // Generate yield op if async for
  if (remap_iter_operands.size()) {
    generateYieldOpFromChannelOp(builder, loop_op->getContext(), new_loop_op);
  }

  builder.restoreInsertionPoint(insertionCheckpoint);

  return new_loop_op;
}

// Clone with remap, but replaces channel op with wait_all op
void replaceChannelOpWithWaitAllAndClone(OpBuilder builder, BlockAndValueMapping &remap, air::ChannelInterface op){
  auto async_op = dyn_cast<air::AsyncOpInterface>(op.getOperation());
  SmallVector<Value, 1> dep_list_remap;
  for (auto dep : async_op.getAsyncDependencies()){
    dep_list_remap.push_back(remap.lookup(dep));
  }
  auto wa_op = builder.create<air::WaitAllOp>(builder.getUnknownLoc(),
                                              air::AsyncTokenType::get(op->getContext()),
                                              dep_list_remap);
  wa_op->setAttr("hoist-channel", StringAttr::get(op->getContext(), "dep"));
  remap.map(async_op.getAsyncToken(), wa_op.getAsyncToken());
}

class LinalgCopyToAIRDmaConversion : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getInputs()[0];
    auto dst = op.getOutputs()[0];

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

    if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
      extractOperandsFromSubview(subview, rewriter, src_offsets, src_sizes,
                                 src_strides);

      if (src_sizes.size() != rank)
        return failure();
      if (src_strides.size() != rank)
        return failure();

      src = subview.getSource();
    }

    if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
      extractOperandsFromSubview(subview, rewriter, dst_offsets, dst_sizes,
                                 dst_strides);

      if (dst_sizes.size() != rank)
        return failure();
      if (dst_strides.size() != rank)
        return failure();

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

class AIRDmaToAIRChannelConversion
    : public OpRewritePattern<air::DmaMemcpyNdOp> {
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto src = op.getSrcMemref();
    auto dst = op.getDstMemref();
    auto ctx = op->getContext();

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

    air::HierarchyInterface hier_op = nullptr;
    unsigned int innerMemorySpace = 0;
    auto herd = op->getParentOfType<air::HerdOp>();
    auto partition = op->getParentOfType<air::PartitionOp>();
    if (herd) {
      hier_op = dyn_cast<air::HierarchyInterface>(herd.getOperation());
      innerMemorySpace = (int)MemorySpace::L1;
    } else if (partition) {
      hier_op = dyn_cast<air::HierarchyInterface>(partition.getOperation());
      innerMemorySpace = (int)MemorySpace::L2;
    } else
      return failure();

    auto rank = src_type.getShape().size();

    SmallVector<Value, 4> src_offsets = op.getSrcOffsets();
    SmallVector<Value, 4> dst_offsets = op.getDstOffsets();
    SmallVector<Value, 4> src_sizes = op.getSrcSizes();
    SmallVector<Value, 4> dst_sizes = op.getDstSizes();
    SmallVector<Value, 4> src_strides = op.getSrcStrides();
    SmallVector<Value, 4> dst_strides = op.getDstStrides();

    if (src_offsets.size()) {
      if (src_sizes.size() != rank)
        return failure();
      if (src_strides.size() != rank)
        return failure();
    }

    if (dst_offsets.size()) {
      if (dst_sizes.size() != rank)
        return failure();
      if (dst_strides.size() != rank)
        return failure();
    }

    // The internal channel op shall inherit the dma op's dep list
    SmallVector<Value, 4> internalDeps = op.getAsyncDependencies();
    // The external channel op shall inherit the loop-carried token only
    SmallVector<Value, 4> externalDeps;
    if (op->getParentOp() && dyn_cast<scf::ForOp>(op->getParentOp())) {
      auto parent_for = dyn_cast<scf::ForOp>(op->getParentOp());
      if (parent_for.getRegionIterArgs().size()) {
        externalDeps.push_back(parent_for.getRegionIterArgs()[0]);
      }
    }

    SmallVector<Value, 4> emptyDeps;
    SmallVector<Type, 4> tys;
    if (auto op_token = op.getAsyncToken()) {
      tys.push_back(air::AsyncTokenType::get(ctx));
    }
    std::string new_cname = "channel_0";
    std::string cname = "channel";
    int which_try = 0;

    // Create channel symbol
    auto module = op->getParentOfType<ModuleOp>();
    while (module.lookupSymbol(new_cname))
      new_cname = cname + "_" + std::to_string(++which_try);
    cname = new_cname;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<air::ChannelOp>(
          loc, cname,
          mlir::ArrayAttr::get(ctx, {mlir::IntegerAttr::get(
                                        mlir::IntegerType::get(ctx, 64), 1)}));
    }

    std::set<Operation *> erased;
    Operation *externalGetPut = nullptr;
    Operation *internalGetPut = nullptr;
    SmallVector<Value, 1> channel_idx{
        rewriter.create<arith::ConstantIndexOp>(loc, 1)};

    // Create channel put-get pair
    if (dst_type.getMemorySpaceAsInt() == innerMemorySpace) {
      internalGetPut = rewriter.create<air::ChannelGetOp>(
          loc, tys, internalDeps, FlatSymbolRefAttr::get(ctx, cname),
          channel_idx, dst, dst_offsets, dst_sizes, dst_strides);
    } else {
      externalGetPut = rewriter.create<air::ChannelGetOp>(
          loc, tys, externalDeps, FlatSymbolRefAttr::get(ctx, cname),
          channel_idx, dst, dst_offsets, dst_sizes, dst_strides);
    }

    if (src_type.getMemorySpaceAsInt() == innerMemorySpace) {
      internalGetPut = rewriter.create<air::ChannelPutOp>(
          loc, tys, internalDeps, FlatSymbolRefAttr::get(ctx, cname),
          channel_idx, src, src_offsets, src_sizes, src_strides);
    } else {
      externalGetPut = rewriter.create<air::ChannelPutOp>(
          loc, tys, externalDeps, FlatSymbolRefAttr::get(ctx, cname),
          channel_idx, src, src_offsets, src_sizes, src_strides);
    }

    // Replace all uses to dma token with internal put/get token
    if (auto op_token = op.getAsyncToken()) {
      auto asyncInternalGetPut =
          dyn_cast<air::AsyncOpInterface>(internalGetPut);
      op_token.replaceAllUsesWith(asyncInternalGetPut.getAsyncToken());
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);

      externalGetPut->setAttr("hoist-channel",
                              StringAttr::get(op->getContext(), "dep"));
      internalGetPut->setAttr(
          "loop-carried-dep",
          StringAttr::get(op->getContext(), "internalGetPut"));
      externalGetPut->setAttr(
          "loop-carried-dep",
          StringAttr::get(op->getContext(), "externalGetPut"));

      SetVector<Operation *> backwardSlice;
      getBackwardSlice(externalGetPut, &backwardSlice,
                       [&](Operation *o) { return o != hier_op; });

      for (auto parent = op->getParentOp();
           !isa<air::HierarchyInterface>(parent);
           parent = parent->getParentOp()) {
        getBackwardSlice(parent, &backwardSlice,
                         [&](Operation *o) { return o != hier_op; });
        backwardSlice.insert(parent);
      }

      // Hoist hierarchy op into scf op
      Operation *scf_loop = nullptr;
      mlir::OpBuilder::InsertPoint
          insertionPointAtHierOp; // To keep a record of the insertion point as
                                  // destination for hoisting
      rewriter.setInsertionPoint(hier_op);
      if (herd) {
        scf::ParallelOp scf_par =
            hoistHerdToAsyncParallel(rewriter, loc, ctx, herd);
        scf_loop = scf_par.getOperation();
      } else if (partition) {
        // Since partition doesn't have iteration space, it doesn't hoist a loop
        insertionPointAtHierOp = rewriter.saveInsertionPoint();
      }

      for (auto b : backwardSlice) {
        b->setAttr("hoist-channel", StringAttr::get(op->getContext(), "dep"));
        if (dyn_cast<air::ExecuteOp>(b)) {
          auto child_op = &(*b->getRegions().front().op_begin());
          child_op->setAttr("hoist-channel",
                            StringAttr::get(op->getContext(), "dep"));
        }
      }

      if (herd) {
        auto scf_par = dyn_cast<scf::ParallelOp>(scf_loop);
        // Get mapping for remapped ssa values entering the hoisted scf.parallel
        BlockAndValueMapping remap;
        auto herd_size = herd.getSizeOperands();
        remap.map(herd.getSize()[0], herd_size[0]);
        remap.map(herd.getSize()[1], herd_size[1]);
        remap.map(herd.getIds()[0], scf_par.getInductionVars()[0]);
        remap.map(herd.getIds()[1], scf_par.getInductionVars()[1]);
        int arg_idx = 0;
        for (auto arg : herd.getKernelArguments())
          remap.map(arg, herd.getKernelOperand(arg_idx++));

        // Clone ops into hoisted scf.parallel
        rewriter.setInsertionPointToStart(scf_par.getBody());
        for (Operation &o :
             herd->getRegions().front().getBlocks().front().getOperations()) {
          if (isa<air::HerdTerminatorOp>(o))
            continue;
          if (o.hasAttr("hoist-channel")){
            if (dyn_cast<scf::ForOp>(o)){
              cloneForUsingRemap(rewriter, remap, dyn_cast<scf::ForOp>(o));
            }
            else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)){
              if (o.hasAttr("loop-carried-dep") && o.getAttrOfType<StringAttr>("loop-carried-dep").getValue().str() == "internalGetPut") {
                // Found channel op labelled as "internalGetPut", which shouldn't be hoisted
                replaceChannelOpWithWaitAllAndClone(rewriter, remap, channel_op);
              }
              else {
                rewriter.clone(o, remap);
              }
            }
            else {
              rewriter.clone(o, remap);
            }
          }
        }
      } else if (partition) {
        // Get mapping for remapped ssa values entering the hoisted scf.for
        BlockAndValueMapping remap;
        int arg_idx = 0;
        for (auto arg : partition.getKernelArguments())
          remap.map(arg, partition.getKernelOperand(arg_idx++));

        // Hoist ops
        rewriter.restoreInsertionPoint(insertionPointAtHierOp);
        for (Operation &o : partition->getRegions()
                                .front()
                                .getBlocks()
                                .front()
                                .getOperations()) {
          if (isa<air::PartitionTerminatorOp>(o))
            continue;
          if (o.hasAttr("hoist-channel")){
            if (dyn_cast<scf::ForOp>(o)){
              cloneForUsingRemap(rewriter, remap, dyn_cast<scf::ForOp>(o));
            }
            else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)){
              if (o.hasAttr("loop-carried-dep") && o.getAttrOfType<StringAttr>("loop-carried-dep").getValue().str() == "internalGetPut") {
                // Found channel op labelled as "internalGetPut", which shouldn't be hoisted
                replaceChannelOpWithWaitAllAndClone(rewriter, remap, channel_op);
                // rewriter.clone(o, remap);
              }
              else {
                rewriter.clone(o, remap);
              }
            }
            else {
              rewriter.clone(o, remap);
            }
          }
        }
      }

      if (scf_loop) {
        scf_loop->walk([&](mlir::Operation *o) {
          if (o == o->getBlock()->getTerminator())
            return;
          if (!o->hasAttr("hoist-channel"))
            erased.insert(o);
          else
            o->removeAttr("hoist-channel");
        });
      }
      hier_op.walk([&](mlir::Operation *o) {
        if (o->hasAttr("hoist-channel"))
          o->removeAttr("hoist-channel");
      });
      erased.insert(externalGetPut);
    }
    erased.insert(op);
    for (auto e : erased) {
      rewriter.eraseOp(e);
    }

    return success();
  }
};

class AffineParToHerdConversion : public OpRewritePattern<AffineParallelOp> {
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

      auto &bb = launch.getBody().front();
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
    SmallVector<Value, 2> dims{
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, bounds[1])};
    auto launch = rewriter.create<air::HerdOp>(op.getLoc(), dims, args);
    auto &bb = launch.getBody().front();
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
    auto &bb = launch.getBody().front();
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

class ScfParToLaunchAndPartitionConversion
    : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToLaunchAndPartitionConversion(
      MLIRContext *ctx, llvm::SmallSet<scf::ParallelOp, 2> &filteredOps)
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
    auto partition = rewriter.create<air::PartitionOp>(
        op.getLoc(), partitionSizes, partitionOpers);
    auto &bb = partition.getBody().front();
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
    builder = OpBuilder::atBlockEnd(&launch.getBody().front());
    builder.create<air::LaunchTerminatorOp>(builder.getUnknownLoc());

    int i = 0;
    auto kernel_args = partition.getKernelArguments();
    kernel_args = kernel_args.drop_front(
        ivs.size() + launch.getSize().size()); // Launch's induction vars
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
                           scf::SCFDialect, xilinx::air::airDialect,
                           arith::ArithDialect, memref::MemRefDialect>();

    target.addLegalOp<AffineApplyOp, AffineForOp, AffineLoadOp, AffineStoreOp,
                      AffineYieldOp>();

    target.addDynamicallyLegalOp<memref::CopyOp>([](memref::CopyOp co) {
      auto src_type = co.getSource().getType().dyn_cast<MemRefType>();
      auto dst_type = co.getTarget().getType().dyn_cast<MemRefType>();
      return src_type.getMemorySpaceAsInt() == dst_type.getMemorySpaceAsInt();
    });

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

    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }
};

struct DmaToChannelPass : public DmaToChannelBase<DmaToChannelPass> {

  DmaToChannelPass() = default;
  DmaToChannelPass(const DmaToChannelPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    ConversionTarget target(*context);

    target.addLegalDialect<
        LLVM::LLVMDialect, func::FuncDialect, scf::SCFDialect, AffineDialect,
        xilinx::air::airDialect, arith::ArithDialect, memref::MemRefDialect>();

    target.addIllegalOp<air::DmaMemcpyNdOp>();

    RewritePatternSet air_dma_patterns(context);
    air_dma_patterns.add<AIRDmaToAIRChannelConversion>(context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_dma_patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
    }

    // Dep tracing
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps) {
      updateDependencyOnFunction(f);
    }

    // Clear channel attributes
    for (auto f : funcOps) {
      f.walk([&](Operation *op) {
        op->removeAttr("loop-carried-dep");
        op->removeAttr("hoist-channel");
      });
    }
  }

  void updateDependencyOnFunction(func::FuncOp f) {
    dependencyTracer depTracer;
    f.walk([&](Operation *op) {
      if (auto channel_op = mlir::dyn_cast<xilinx::air::ChannelInterface>(op)) {
        if (channel_op->getAttrOfType<StringAttr>("loop-carried-dep") &&
            channel_op->getAttrOfType<StringAttr>("loop-carried-dep")
                    .getValue()
                    .str() == "externalGetPut") {

          // Start tracing dependency only if this put/get op is async
          auto async_op = dyn_cast<air::AsyncOpInterface>(op);
          if (!async_op.getAsyncToken())
            return;

          // Connect async dependency of external put/get scf parallel
          SmallVector<partialMemref, 1> sink_op_memref_reads;
          SmallVector<partialMemref, 1> sink_op_memref_writes;
          SmallVector<Value, 1> sink_op_scalar_ins;
          SmallVector<Value, 1> sink_op_scalar_outs;

          air::WaitAllOp sink_wait_all_op = nullptr;
          for (auto parent = op->getParentOp(); !isa<func::FuncOp>(parent);
               parent = parent->getParentOp()) {
            if (parent->getAttrOfType<StringAttr>("loop-carried-dep") &&
                parent->getAttrOfType<StringAttr>("loop-carried-dep")
                        .getValue()
                        .str() == "hoistedLoop") {
              if (auto scf_par = dyn_cast<scf::ParallelOp>(parent)) {
                sink_wait_all_op = dyn_cast<air::WaitAllOp>(
                    scf_par.getInitVals()[0].getDefiningOp());
              } else if (auto scf_for = dyn_cast<scf::ForOp>(parent)) {
                sink_wait_all_op = dyn_cast<air::WaitAllOp>(
                    scf_for.getIterOperands()[0].getDefiningOp());
              }
            }
          }

          depTracer.getPartialMemrefFromOp(
              channel_op.getOperation(), sink_op_memref_reads,
              sink_op_memref_writes, sink_op_scalar_ins, sink_op_scalar_outs);

          assert(sink_op_memref_reads.size() ||
                 sink_op_memref_writes.size() &&
                     "cannot read memref from channel op");

          if (sink_wait_all_op){
            // Detect RAW deps
            depTracer.template traceDependencyFromOp<air::WaitAllOp>(
                sink_op_memref_reads, sink_wait_all_op, "RAW");
            // Detect WAW and WAR deps
            depTracer.template traceDependencyFromOp<air::WaitAllOp>(
                sink_op_memref_writes, sink_wait_all_op, "WAW/WAR");

            // Rebuild loop-carried dependency in scf loop nest
            depTracer.reconnectLoopCarriedDependencyFromOp(op);
          }

          // Trace dependency of external put/get within scf loop
          depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
              sink_op_memref_reads,
              dyn_cast<air::AsyncOpInterface>(channel_op.getOperation()),
              "RAW");
          depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
              sink_op_memref_writes,
              dyn_cast<air::AsyncOpInterface>(channel_op.getOperation()),
              "WAW/WAR");
          // Detect tile index deps
          depTracer.traceTileIndices(
              sink_op_memref_reads, sink_op_memref_writes, sink_op_scalar_ins,
              sink_op_scalar_outs,
              dyn_cast<air::AsyncOpInterface>(channel_op.getOperation()));
        }
      }
    });
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
                           xilinx::air::airDialect, arith::ArithDialect>();

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
    if (clHasPartition) {
      patterns.add<ScfParToLaunchAndPartitionConversion>(context, filteredOps);
    } else {
      patterns.add<ScfParToLaunchConversion>(context, filteredOps);
    }

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           xilinx::air::airDialect, arith::ArithDialect>();

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

std::unique_ptr<mlir::Pass> createDmaToChannelPass() {
  return std::make_unique<DmaToChannelPass>();
}

} // namespace air
} // namespace xilinx