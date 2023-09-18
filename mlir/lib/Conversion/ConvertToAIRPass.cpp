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
#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "mlir/IR/IntegerSet.h"
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

#define DEBUG_TYPE "convert-to-air"

static uint64_t DmaMemcpyOpID;

static FailureOr<air::DmaMemcpyNdOp>
matchAndRewriteCopyOp(memref::CopyOp op, PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto src = op.getSource();
  auto dst = op.getTarget();

  rewriter.setInsertionPoint(op);

  // It must already be a memref
  auto src_type = src.getType().dyn_cast<MemRefType>();
  auto dst_type = dst.getType().dyn_cast<MemRefType>();
  if (!src_type)
    return failure();

  if ((src_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) &&
      (dst_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3))
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
    auto static_offsets = subview.getStaticOffsets();
    auto static_sizes = subview.getStaticSizes();
    auto static_strides = subview.getStaticStrides();
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
  auto subview_offsets = subview.offsets().begin();
  auto static_offsets = subview.static_offsets();
  auto static_sizes = subview.getStaticSizes();
  auto static_strides = subview.getStaticStrides();
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

static void generateYieldAndOrReduceToScfLoop(OpBuilder builder,
                                              MLIRContext *ctx,
                                              scf::ParallelOp scf_par) {

  // Check if scf::YieldOp already exists in scf parallel
  SmallVector<scf::YieldOp, 2> y_ops(scf_par.getOps<scf::YieldOp>());
  if (y_ops.size()) {
    assert(y_ops.size() == 1);
    builder.setInsertionPoint(y_ops[0]);
  } else {
    builder.setInsertionPointToEnd(scf_par.getBody());
  }

  auto wait_all_op_yielded = builder.create<air::WaitAllOp>(
      scf_par.getLoc(), air::AsyncTokenType::get(ctx), SmallVector<Value, 1>{});
  auto reduce_op = air::createSCFReduceForAsyncSCFParallel(
      builder, scf_par.getLoc(), wait_all_op_yielded.getAsyncToken(), ctx);
  builder.setInsertionPointToEnd(scf_par.getBody());

  wait_all_op_yielded->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
  reduce_op->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
  reduce_op.walk([&](mlir::Operation *o) {
    if (!isa<scf::YieldOp>(o)) {
      o->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
    }
  });
}

static scf::ParallelOp hoistHerdToAsyncParallel(OpBuilder builder, Location loc,
                                                MLIRContext *ctx,
                                                air::HerdOp herd,
                                                SmallVector<int, 2> lbs_int,
                                                SmallVector<int, 2> ubs_int) {

  auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 2> steps{step, step};
  SmallVector<Value, 2> lbs;
  SmallVector<Value, 2> ubs;

  if (lbs_int.size()) {
    for (auto v : lbs_int) {
      auto lb = builder.create<arith::ConstantIndexOp>(loc, v);
      lbs.push_back(lb);
    }
    for (auto v : ubs_int) {
      auto ub = builder.create<arith::ConstantIndexOp>(loc, v);
      ubs.push_back(ub);
    }
  } else {
    auto herd_size = herd.getSizeOperands();
    auto lb = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (unsigned i = 0; i < herd_size.size(); i++) {
      lbs.push_back(lb);
      ubs.push_back(herd_size[i]);
    }
  }

  auto wa_op = builder.create<air::WaitAllOp>(
      loc, air::AsyncTokenType::get(ctx), SmallVector<Value, 1>{});
  SmallVector<Value, 1> deps_in{wa_op.getAsyncToken()};
  auto scf_par = builder.create<scf::ParallelOp>(loc, lbs, ubs, steps, deps_in);

  generateYieldAndOrReduceToScfLoop(builder, ctx, scf_par);

  scf_par->setAttr("hoist-channel", StringAttr::get(ctx, "hoistedLoop"));
  scf_par->setAttr("loop-carried-dep", StringAttr::get(ctx, "hoistedLoop"));

  return scf_par;
}

SmallVector<Value, 1> getLoopTokens(scf::ForOp loop) {
  SmallVector<Value, 1> output;
  for (auto v : loop.getIterOperands()) {
    output.push_back(v);
  }
  return output;
}

SmallVector<Value, 1> getLoopTokens(scf::ParallelOp loop) {
  SmallVector<Value, 1> output;
  for (auto v : loop.getInitVals()) {
    output.push_back(v);
  }
  return output;
}

void replaceAllUsesOfInductionVarsWith(scf::ForOp old_loop,
                                       scf::ForOp new_loop) {
  replaceAllUsesInRegionWith(old_loop.getInductionVar(),
                             new_loop.getInductionVar(), new_loop.getRegion());
}

void replaceAllUsesOfInductionVarsWith(scf::ParallelOp old_loop,
                                       scf::ParallelOp new_loop) {
  for (unsigned i = 0; i < old_loop.getInductionVars().size(); i++) {
    replaceAllUsesInRegionWith(old_loop.getInductionVars()[i],
                               new_loop.getInductionVars()[i],
                               new_loop.getRegion());
  }
}

void replaceAllUsesOfLoopTokensWith(scf::ForOp old_loop, scf::ForOp new_loop) {
  if (old_loop.getRegionIterArgs().size()) {
    for (unsigned i = 0; i < old_loop.getRegionIterArgs().size(); i++) {
      replaceAllUsesInRegionWith(old_loop.getRegionIterArgs()[i],
                                 new_loop.getRegionIterArgs()[i],
                                 new_loop.getRegion());
    }
  }
}

void replaceAllUsesOfLoopTokensWith(scf::ParallelOp old_loop,
                                    scf::ParallelOp new_loop) {
  if (old_loop.getInitVals().size()) {
    for (unsigned i = 0; i < old_loop.getInitVals().size(); i++) {
      replaceAllUsesInRegionWith(old_loop.getInitVals()[i],
                                 new_loop.getInitVals()[i],
                                 new_loop.getRegion());
    }
  }
}

void replaceAllUsesOfIncomingTokensWith(SmallVector<Value, 4> incoming_tokens,
                                        scf::ForOp new_loop) {
  for (Value v : incoming_tokens) {
    replaceAllUsesInRegionWith(v, new_loop.getRegionIterArgs()[0],
                               new_loop.getRegion());
  }
}

void replaceAllUsesOfIncomingTokensWith(SmallVector<Value, 4> incoming_tokens,
                                        scf::ParallelOp new_loop) {
  for (Value v : incoming_tokens) {
    replaceAllUsesInRegionWith(v, new_loop.getInitVals()[0],
                               new_loop.getRegion());
  }
}

void replaceAllUsesOfConstsInRegionWithNew(SmallVector<Value, 4> constants,
                                           OpBuilder builder, Region &region) {
  for (auto c : constants) {
    replaceAllUsesInRegionWith(
        c, builder.clone(*c.getDefiningOp())->getResult(0), region);
  }
}

template <typename T>
static void updateUsesInScfLoop(OpBuilder builder, T new_loop_op, T loop_op) {
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

  replaceAllUsesOfInductionVarsWith(loop_op, new_loop_op);
  replaceAllUsesOfLoopTokensWith(loop_op, new_loop_op);
  builder.setInsertionPointToStart(new_loop_op.getBody());
  replaceAllUsesOfConstsInRegionWithNew(constants, builder,
                                        new_loop_op.getRegion());
  replaceAllUsesOfIncomingTokensWith(incoming_tokens, new_loop_op);

  builder.restoreInsertionPoint(insertionCheckpoint);
}

scf::YieldOp generateYieldAndOrReduceToScfLoop(OpBuilder builder,
                                               MLIRContext *ctx,
                                               scf::ForOp scf_loop) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToEnd(scf_loop.getBody());
  SmallVector<air::ChannelInterface, 1> channel_ops;
  for (auto channel_op : scf_loop.getOps<air::ChannelInterface>()) {
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

// Clone with remap, but replace async op with wait_all op
void replaceAsyncOpWithWaitAllAndClone(OpBuilder builder, IRMapping &remap,
                                       Operation *op,
                                       bool cloneDepList = true) {
  auto async_op = dyn_cast<air::AsyncOpInterface>(op);
  assert(async_op);
  SmallVector<Value, 1> dep_list_remap;
  if (cloneDepList) {
    for (auto dep : async_op.getAsyncDependencies()) {
      dep_list_remap.push_back(remap.lookupOrDefault(dep));
    }
  }
  auto wa_op = builder.create<air::WaitAllOp>(
      builder.getUnknownLoc(), air::AsyncTokenType::get(op->getContext()),
      dep_list_remap);
  wa_op->setAttr("hoist-channel", StringAttr::get(op->getContext(), "dep"));
  remap.map(async_op.getAsyncToken(), wa_op.getAsyncToken());
}

// Clone affine if's block with remap
void replaceAffineIfOpWithChannelOpAndClone(
    OpBuilder builder, IRMapping &remap, air::ChannelInterface externalGetPut) {
  for (Operation &child_op : externalGetPut->getBlock()->getOperations()) {
    if (child_op.hasAttr("hoist-channel")) {
      if (child_op.hasAttr("loop-carried-dep") &&
          child_op.getAttrOfType<StringAttr>("loop-carried-dep")
                  .getValue()
                  .str() == "internalGetPut") {
      } else {
        builder.clone(child_op, remap);
      }
    }
  }
}

Value lookupOrDefaultRange(Value v, IRMapping &remap) {
  return remap.lookupOrDefault(v);
}

SmallVector<Value, 1> lookupOrDefaultRange(SmallVector<Value, 1> vec,
                                           IRMapping &remap) {
  SmallVector<Value, 1> output;
  for (auto v : vec) {
    output.push_back(remap.lookupOrDefault(v));
  }
  return output;
}

template <typename T>
T cloneScfLoopUsingRemap(OpBuilder builder, IRMapping &remap, T loop_op,
                         air::ChannelInterface externalGetPut = nullptr) {
  T new_loop_op =
      builder.create<T>(builder.getUnknownLoc(),
                        lookupOrDefaultRange(loop_op.getLowerBound(), remap),
                        lookupOrDefaultRange(loop_op.getUpperBound(), remap),
                        lookupOrDefaultRange(loop_op.getStep(), remap),
                        lookupOrDefaultRange(getLoopTokens(loop_op), remap));
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(new_loop_op.getBody());
  for (Operation &child_op : loop_op.getBody()->getOperations()) {
    if (child_op.hasAttr("hoist-channel")) {
      if (auto for_op = dyn_cast<scf::ForOp>(child_op)) {
        cloneScfLoopUsingRemap<scf::ForOp>(builder, remap, for_op,
                                           externalGetPut);
      } else if (auto parallel_op = dyn_cast<scf::ParallelOp>(child_op)) {
        cloneScfLoopUsingRemap<scf::ParallelOp>(builder, remap, parallel_op,
                                                externalGetPut);
      } else if (auto channel_op = dyn_cast<air::ChannelInterface>(child_op)) {
        if (child_op.hasAttr("loop-carried-dep") &&
            child_op.getAttrOfType<StringAttr>("loop-carried-dep")
                    .getValue()
                    .str() == "internalGetPut") {
          // Found channel op labelled as "internalGetPut", which shouldn't be
          // hoisted
          replaceAsyncOpWithWaitAllAndClone(builder, remap, &child_op, false);
        } else {
          builder.clone(child_op, remap);
        }
      } else if (externalGetPut && dyn_cast<mlir::AffineIfOp>(child_op)) {
        // If externalGetPut is not nullptr, then broadcast lowering mode is on
        replaceAffineIfOpWithChannelOpAndClone(builder, remap, externalGetPut);
      } else {
        builder.clone(child_op, remap);
      }
    }
  }
  // Re-establish uses after hoisting
  updateUsesInScfLoop<T>(builder, new_loop_op, loop_op);

  new_loop_op->setAttr("hoist-channel",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));
  new_loop_op->setAttr("loop-carried-dep",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));

  // Generate yield op and/or reduce op if async
  if (getLoopTokens(loop_op).size()) {
    generateYieldAndOrReduceToScfLoop(builder, loop_op->getContext(),
                                      new_loop_op);
  }

  builder.restoreInsertionPoint(insertionCheckpoint);

  return new_loop_op;
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
    auto src_type = src.getType().dyn_cast<MemRefType>();
    auto dst_type = dst.getType().dyn_cast<MemRefType>();
    if (!src_type)
      return failure();

    if ((src_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) &&
        (dst_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3))
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

unsigned getScfParDimIdFromBCastDma(air::DmaMemcpyNdOp memcpyOp) {
  // Get all ops on the dependency connection between dma and herd launch
  SmallVector<Value, 1> loop_dep_history;
  std::vector<Operation *> op_history;
  traceDependentInductionVar(memcpyOp, loop_dep_history, op_history);

  // Walk constraints in broadcast pattern, and get shape of the broadcast
  // pattern

  // Check which dimension op operates on; initialize current_shape_expr
  for (auto v : loop_dep_history) {
    if (auto hl_op = air::getHerdArgOwner(v)) {
      for (unsigned j = 0; j < hl_op.getNumDims(); j++) {
        if (v == hl_op.getIds()[j]) {
          return j;
        }
      }
    }
  }
  assert(false && "cannot trace dependency to parent herd");
  return 0;
}

// Create channel name as string
std::string createChannelName(ModuleOp module) {
  std::string new_cname = "channel_0";
  std::string cname = "channel";
  int which_try = 0;
  while (module.lookupSymbol(new_cname))
    new_cname = cname + "_" + std::to_string(++which_try);
  cname = new_cname;
  return cname;
}

// Create channel symbol
air::ChannelOp createChannelOpWithBCast(OpBuilder builder, ModuleOp module,
                                        std::string cname, Location loc,
                                        SmallVector<int64_t, 2> bcast_sizes) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(module.getBody());

  auto channel_op = builder.create<air::ChannelOp>(
      loc, cname, builder.getI64ArrayAttr(bcast_sizes));

  builder.restoreInsertionPoint(insertionCheckpoint);

  return channel_op;
}

// Annotate post-broadcast shape
void annotateChannelOpWithBCastShape(OpBuilder builder,
                                     air::ChannelOp channel_op,
                                     air::HerdOp herd) {
  auto herd_size = herd.getSizeOperands();
  SmallVector<int64_t, 1> output_shape;
  for (auto operand : herd_size) {
    output_shape.push_back(
        operand.getDefiningOp<arith::ConstantIndexOp>().value());
  }
  channel_op->setAttr("broadcast_shape", builder.getI64ArrayAttr(output_shape));
}

void replaceAIRDmaWithAIRChannelPairs(
    OpBuilder &builder, unsigned innerMemorySpace, air::DmaMemcpyNdOp op,
    SmallVector<air::ChannelInterface, 1> &internalGetPutVector,
    SmallVector<air::ChannelInterface, 1> &externalGetPutVector) {
  auto loc = op->getLoc();
  auto src = op.getSrcMemref();
  auto dst = op.getDstMemref();
  auto ctx = op->getContext();

  auto src_type = src.getType().dyn_cast<MemRefType>();
  auto dst_type = dst.getType().dyn_cast<MemRefType>();
  SmallVector<Value, 4> src_offsets = op.getSrcOffsets();
  SmallVector<Value, 4> dst_offsets = op.getDstOffsets();
  SmallVector<Value, 4> src_sizes = op.getSrcSizes();
  SmallVector<Value, 4> dst_sizes = op.getDstSizes();
  SmallVector<Value, 4> src_strides = op.getSrcStrides();
  SmallVector<Value, 4> dst_strides = op.getDstStrides();

  // The internal channel op shall inherit the dma op's dep list
  SmallVector<Value, 4> internalDeps = op.getAsyncDependencies();
  // The external channel op shall inherit the loop-carried token only
  SmallVector<Value, 4> externalDeps;
  for (auto token : internalDeps) {
    if (air::getForRegionIterArgsOwner(token)) {
      externalDeps.push_back(token);
    }
  }

  air::ChannelInterface externalGetPut = nullptr;
  air::ChannelInterface internalGetPut = nullptr;

  // Create channel symbol
  auto module = op->getParentOfType<ModuleOp>();
  auto cname = createChannelName(module);

  // Infer broadcast shape from integer set, if broadcast_set attribute is set
  if (op->hasAttr("broadcast_set")) {
    auto int_set =
        op->getAttrOfType<mlir::IntegerSetAttr>("broadcast_set").getValue();
    SmallVector<int, 2> lbs_int = {-1, -1};
    SmallVector<int, 2> ubs_int = {-1, -1};
    SmallVector<int64_t, 2> channel_sizes = {1, 1};
    air::getSizesFromIntegerSet(ctx, int_set, lbs_int, ubs_int);
    SmallVector<int64_t, 2> bcast_sizes = {ubs_int[0] - lbs_int[0] + 1,
                                           ubs_int[1] - lbs_int[1] + 1};
    auto channel_op =
        createChannelOpWithBCast(builder, module, cname, loc, channel_sizes);
    channel_op->setAttr("broadcast_shape",
                        builder.getI64ArrayAttr(bcast_sizes));
  } else if (op->hasAttr("broadcast_pattern")) {
    // Else if broadcast_pattern attribute is set, then infer channel's input
    // and output shapes from the broadcast_pattern affine set
    SmallVector<int, 2> lbs_int = {-1};
    SmallVector<int, 2> ubs_int = {-1};
    mlir::IntegerSet int_set =
        op->getAttrOfType<mlir::IntegerSetAttr>("broadcast_pattern").getValue();
    air::getSizesFromIntegerSet(ctx, int_set, lbs_int, ubs_int);
    SmallVector<int64_t, 2> channel_sizes = {1, 1};
    channel_sizes[getScfParDimIdFromBCastDma(dyn_cast<air::DmaMemcpyNdOp>(
        op.getOperation()))] = ubs_int[0] - lbs_int[0] + 1;
    auto channel_op =
        createChannelOpWithBCast(builder, module, cname, loc, channel_sizes);
    annotateChannelOpWithBCastShape(builder, channel_op,
                                    op->getParentOfType<air::HerdOp>());
  } else {
    // Else, infer channel's input shape from parent herd, if within a herd
    SmallVector<int64_t, 2> channel_sizes = {1, 1};
    if (auto parent_herd_op = op->getParentOfType<air::HerdOp>()) {
      auto herd_size = parent_herd_op.getSizeOperands();
      for (unsigned i = 0; i < herd_size.size(); i++) {
        channel_sizes[i] =
            herd_size[i].getDefiningOp<arith::ConstantIndexOp>().value();
      }
    }
    createChannelOpWithBCast(builder, module, cname, loc, channel_sizes);
  }

  SmallVector<Value, 1> channel_idx_internal{};
  SmallVector<Value, 1> channel_idx_external{};
  if (op->hasAttr("broadcast_set")) {
    // If broadcasting, let internal channel inherit affine.if's operands
    auto parent_affine_if_op = op->getParentOfType<mlir::AffineIfOp>();
    for (auto operand : parent_affine_if_op->getOperands()) {
      channel_idx_internal.push_back(operand);
    }
  } else if (auto parent_herd_op = op->getParentOfType<air::HerdOp>()) {
    // Else, let both channel ops inherit herd's induction variables
    for (auto iv : parent_herd_op.getIds()) {
      channel_idx_internal.push_back(iv);
      channel_idx_external.push_back(iv);
    }
  }

  // Create channel put-get pair
  SmallVector<Type, 4> tys;
  if (auto op_token = op.getAsyncToken()) {
    tys.push_back(air::AsyncTokenType::get(ctx));
  }
  if (dst_type.getMemorySpaceAsInt() == innerMemorySpace) {
    auto internal = builder.create<air::ChannelGetOp>(
        loc, tys, internalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_internal, dst, dst_offsets, dst_sizes, dst_strides);
    internalGetPut = dyn_cast<air::ChannelInterface>(internal.getOperation());
  } else {
    auto external = builder.create<air::ChannelGetOp>(
        loc, tys, externalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_external, dst, dst_offsets, dst_sizes, dst_strides);
    externalGetPut = dyn_cast<air::ChannelInterface>(external.getOperation());
  }

  if (src_type.getMemorySpaceAsInt() == innerMemorySpace) {
    auto internal = builder.create<air::ChannelPutOp>(
        loc, tys, internalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_internal, src, src_offsets, src_sizes, src_strides);
    internalGetPut = dyn_cast<air::ChannelInterface>(internal.getOperation());
  } else {
    auto external = builder.create<air::ChannelPutOp>(
        loc, tys, externalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_external, src, src_offsets, src_sizes, src_strides);
    externalGetPut = dyn_cast<air::ChannelInterface>(external.getOperation());
  }

  // Replace all uses to dma token with internal put/get token
  if (auto op_token = op.getAsyncToken()) {
    auto asyncInternalGetPut =
        dyn_cast<air::AsyncOpInterface>(internalGetPut.getOperation());
    op_token.replaceAllUsesWith(asyncInternalGetPut.getAsyncToken());
  }

  // Add attributes to label internal/external channel ops
  externalGetPut->setAttr("hoist-channel",
                          StringAttr::get(op->getContext(), "dep"));
  internalGetPut->setAttr("loop-carried-dep",
                          StringAttr::get(op->getContext(), "internalGetPut"));
  externalGetPut->setAttr("loop-carried-dep",
                          StringAttr::get(op->getContext(), "externalGetPut"));

  externalGetPutVector.push_back(externalGetPut);
  internalGetPutVector.push_back(internalGetPut);
}

void HoistingAffineIf(mlir::AffineIfOp op) {
  auto ctx = op->getContext();

  air::HierarchyInterface hier_op = nullptr;
  unsigned int innerMemorySpace = 0;
  auto herd = op->getParentOfType<air::HerdOp>();
  assert(herd && "affine if op has no air.herdOp as parent");
  auto segment = op->getParentOfType<air::SegmentOp>();
  if (herd) {
    hier_op = dyn_cast<air::HierarchyInterface>(herd.getOperation());
    innerMemorySpace = (int)air::MemorySpace::L1;
  } else if (segment) {
    assert(false &&
           "broadcast lowering with air.segmentOp currently not supported");
  } else
    assert(false && "affine if op has no air.hierarchy as parent");

  SmallVector<air::ChannelInterface, 1> externalGetPut;
  SmallVector<air::ChannelInterface, 1> internalGetPut;
  SmallVector<air::DmaMemcpyNdOp, 2> dmas;

  // Recursively search for and replace air.dma ops
  auto module = op->getParentOfType<ModuleOp>();
  OpBuilder module_builder(module);
  // The first then block
  auto then_block_dma = air::getAIRDmaInBlock(op.getThenBlock());
  dmas.push_back(then_block_dma);
  module_builder.setInsertionPoint(then_block_dma);
  replaceAIRDmaWithAIRChannelPairs(module_builder, innerMemorySpace,
                                   then_block_dma, internalGetPut,
                                   externalGetPut);
  // Recursion
  mlir::AffineIfOp current_if = op;
  while (air::getAffineIfInBlock(current_if.getElseBlock())) {
    auto child_if_op = air::getAffineIfInBlock(current_if.getElseBlock());

    auto child_then_block_dma =
        air::getAIRDmaInBlock(child_if_op.getThenBlock());
    dmas.push_back(child_then_block_dma);
    module_builder.setInsertionPoint(child_then_block_dma);
    replaceAIRDmaWithAIRChannelPairs(module_builder, innerMemorySpace,
                                     child_then_block_dma, internalGetPut,
                                     externalGetPut);

    current_if = child_if_op;
  }
  // The last else block
  auto else_block_dma = air::getAIRDmaInBlock(current_if.getElseBlock());
  dmas.push_back(else_block_dma);
  module_builder.setInsertionPoint(else_block_dma);
  replaceAIRDmaWithAIRChannelPairs(module_builder, innerMemorySpace,
                                   else_block_dma, internalGetPut,
                                   externalGetPut);

  // Get dependent ops to hoist together with external get/put
  SetVector<Operation *> backwardSlice;
  for (auto ext_channel_op : externalGetPut) {
    getBackwardSlice(ext_channel_op.getOperation(), &backwardSlice,
                     [&](Operation *o) { return o != hier_op; });

    for (auto parent = ext_channel_op->getParentOp();
         !isa<air::HierarchyInterface>(parent);
         parent = parent->getParentOp()) {
      getBackwardSlice(parent, &backwardSlice,
                       [&](Operation *o) { return o != hier_op; });
      backwardSlice.insert(parent);
    }
  }

  // Label dependent ops to hoist
  for (auto b : backwardSlice) {
    b->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
    if (dyn_cast<air::ExecuteOp>(b)) {
      auto child_op = &(*b->getRegions().front().op_begin());
      child_op->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
    }
  }

  // Hoist hierarchy op into scf op
  module_builder.setInsertionPoint(hier_op);
  MemRefType externalMemrefTy =
      externalGetPut[0].getMemref().getType().cast<MemRefType>();
  if (externalMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3 &&
      segment) {
    module_builder.setInsertionPoint(segment);
  }

  // Exclude herd's iteration space since broadcasted copies are already
  // specialized
  mlir::OpBuilder::InsertPoint
      insertionPointAtHierOp; // To keep a record of the insertion point as
                              // destination for hoisting
  insertionPointAtHierOp = module_builder.saveInsertionPoint();

  // Herd's constantOp operands
  auto zero_const_op = module_builder.create<arith::ConstantIndexOp>(
      module_builder.getUnknownLoc(), 0);

  // Check for op buffer sizes
  assert(internalGetPut.size() == externalGetPut.size());
  assert(externalGetPut.size() == dmas.size());

  // Fill up hoisted scf op region with cloned ops
  unsigned dma_index = 0;
  for (size_t i = 0; i < dmas.size(); i++) {
    // Get mapping for remapped ssa values entering the hoisted scf.parallel
    IRMapping remap;
    remap.map(herd.getIds()[0], zero_const_op);
    remap.map(herd.getIds()[1], zero_const_op);
    int arg_idx = 0;
    if (externalMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3 &&
        segment) {
      // If hoisiting directly from herd to launch
      for (auto arg : herd.getKernelArguments())
        remap.map(arg, segment.getKernelOperand(arg_idx++));
    } else {
      for (auto arg : herd.getKernelArguments())
        remap.map(arg, herd.getKernelOperand(arg_idx++));
    }

    // Clone ops into hoisted scf.parallel
    module_builder.restoreInsertionPoint(insertionPointAtHierOp);
    for (Operation &o :
         herd->getRegions().front().getBlocks().front().getOperations()) {
      if (isa<air::HerdTerminatorOp>(o))
        continue;
      if (o.hasAttr("hoist-channel")) {
        if (auto child_for_op = dyn_cast<scf::ForOp>(o)) {
          cloneScfLoopUsingRemap<scf::ForOp>(
              module_builder, remap, child_for_op, externalGetPut[dma_index]);
        } else if (auto child_parallel_op = dyn_cast<scf::ParallelOp>(o)) {
          cloneScfLoopUsingRemap<scf::ParallelOp>(module_builder, remap,
                                                  child_parallel_op,
                                                  externalGetPut[dma_index]);
        } else if (dyn_cast<mlir::AffineIfOp>(o)) {
          replaceAffineIfOpWithChannelOpAndClone(module_builder, remap,
                                                 externalGetPut[dma_index]);
        } else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)) {
          if (o.hasAttr("loop-carried-dep") &&
              o.getAttrOfType<StringAttr>("loop-carried-dep")
                      .getValue()
                      .str() == "internalGetPut") {
            // Found channel op labelled as "internalGetPut", which shouldn't be
            // hoisted
            replaceAsyncOpWithWaitAllAndClone(module_builder, remap, &o, false);
          } else {
            module_builder.clone(o, remap);
          }
        } else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(o)) {
          replaceAsyncOpWithWaitAllAndClone(module_builder, remap, &o, false);
        } else {
          module_builder.clone(o, remap);
        }
      }
    }
    dma_index++;
  }

  module.walk([&](mlir::Operation *o) {
    if (o->hasAttr("hoist-channel")) {
      o->removeAttr("hoist-channel");
    }
  });
  hier_op.walk([&](mlir::Operation *o) {
    if (o->hasAttr("loop-carried-dep") &&
        o->getAttrOfType<StringAttr>("loop-carried-dep").getValue().str() ==
            "externalGetPut") {
      o->erase();
    }
  });
  for (auto &dma : dmas) {
    dma->erase();
  }
}

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

    if ((src_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3) &&
        (dst_type.getMemorySpaceAsInt() == (int)air::MemorySpace::L3))
      return failure();

    if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
      return failure();

    air::HierarchyInterface hier_op = nullptr;
    unsigned int innerMemorySpace = 0;
    auto herd = op->getParentOfType<air::HerdOp>();
    auto segment = op->getParentOfType<air::SegmentOp>();
    if (herd) {
      hier_op = dyn_cast<air::HierarchyInterface>(herd.getOperation());
      innerMemorySpace = (int)air::MemorySpace::L1;
    } else if (segment) {
      hier_op = dyn_cast<air::HierarchyInterface>(segment.getOperation());
      innerMemorySpace = (int)air::MemorySpace::L2;
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

    std::set<Operation *> erased;
    SmallVector<air::ChannelInterface, 1> externalGetPut;
    SmallVector<air::ChannelInterface, 1> internalGetPut;

    replaceAIRDmaWithAIRChannelPairs(rewriter, innerMemorySpace, op,
                                     internalGetPut, externalGetPut);

    {
      OpBuilder::InsertionGuard guard(rewriter);

      SetVector<Operation *> backwardSlice;
      for (auto ext_channel_op : externalGetPut) {
        getBackwardSlice(ext_channel_op.getOperation(), &backwardSlice,
                         [&](Operation *o) { return o != hier_op; });
      }

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
      MemRefType externalMemrefTy =
          externalGetPut[0].getMemref().getType().cast<MemRefType>();
      if (herd) {
        if (externalMemrefTy.getMemorySpaceAsInt() ==
                (int)air::MemorySpace::L3 &&
            segment) {
          rewriter.setInsertionPoint(segment);
        }
        // Scf parallel shape is either herd shape, or channel set shape if
        // broadcasting
        SmallVector<int, 2> lbs;
        SmallVector<int, 2> ubs;
        auto module = op->getParentOfType<ModuleOp>();
        auto channel_op = dyn_cast<air::ChannelOp>(
            module.lookupSymbol(externalGetPut[0].getChanName()));
        auto size = extractFromI64ArrayAttr(channel_op.getSize());
        for (auto s : size) {
          lbs.push_back(0);
          ubs.push_back(s);
        }
        scf::ParallelOp scf_par =
            hoistHerdToAsyncParallel(rewriter, loc, ctx, herd, lbs, ubs);
        scf_loop = scf_par.getOperation();
      } else if (segment) {
        // Since segment doesn't have iteration space, it doesn't hoist a loop
        insertionPointAtHierOp = rewriter.saveInsertionPoint();
      }

      for (auto b : backwardSlice) {
        if (dyn_cast<air::ExecuteOp>(b)) {
          for (auto &exec_child_op : b->getRegions().front().getOps()) {
            getBackwardSlice(&exec_child_op, &backwardSlice,
                             [&](Operation *o) { return o != hier_op; });
            backwardSlice.insert(&exec_child_op);
          }
        }
      }

      for (auto b : backwardSlice) {
        b->setAttr("hoist-channel", StringAttr::get(ctx, "dep"));
      }

      if (herd) {
        auto scf_par = dyn_cast<scf::ParallelOp>(scf_loop);
        // Get mapping for remapped ssa values entering the hoisted scf.parallel
        IRMapping remap;
        auto herd_size = herd.getSizeOperands();
        remap.map(herd.getSize()[0], herd_size[0]);
        remap.map(herd.getSize()[1], herd_size[1]);
        remap.map(herd.getIds()[0], scf_par.getInductionVars()[0]);
        remap.map(herd.getIds()[1], scf_par.getInductionVars()[1]);
        int arg_idx = 0;
        if (externalMemrefTy.getMemorySpaceAsInt() ==
                (int)air::MemorySpace::L3 &&
            segment) {
          // If hoisting directly to launch region
          for (auto arg : herd.getKernelArguments())
            remap.map(arg, segment.getKernelOperand(arg_idx++));
        } else {
          for (auto arg : herd.getKernelArguments())
            remap.map(arg, herd.getKernelOperand(arg_idx++));
        }

        // Clone ops into hoisted scf.parallel
        rewriter.setInsertionPointToStart(scf_par.getBody());
        for (Operation &o :
             herd->getRegions().front().getBlocks().front().getOperations()) {
          if (isa<air::HerdTerminatorOp>(o))
            continue;
          if (o.hasAttr("hoist-channel")) {
            if (auto child_for_op = dyn_cast<scf::ForOp>(o)) {
              cloneScfLoopUsingRemap<scf::ForOp>(rewriter, remap, child_for_op);
            } else if (auto child_parallel_op = dyn_cast<scf::ParallelOp>(o)) {
              cloneScfLoopUsingRemap<scf::ParallelOp>(rewriter, remap,
                                                      child_parallel_op);
            } else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)) {
              if (o.hasAttr("loop-carried-dep") &&
                  o.getAttrOfType<StringAttr>("loop-carried-dep")
                          .getValue()
                          .str() == "internalGetPut") {
                // Found channel op labelled as "internalGetPut", which
                // shouldn't be hoisted
                replaceAsyncOpWithWaitAllAndClone(rewriter, remap, &o, false);
              } else {
                rewriter.clone(o, remap);
              }
            } else {
              rewriter.clone(o, remap);
            }
          }
        }
      } else if (segment) {
        // Get mapping for remapped ssa values entering the hoisted scf.for
        IRMapping remap;
        int arg_idx = 0;
        for (auto arg : segment.getKernelArguments())
          remap.map(arg, segment.getKernelOperand(arg_idx++));

        // Hoist ops
        rewriter.restoreInsertionPoint(insertionPointAtHierOp);
        for (Operation &o : segment->getRegions()
                                .front()
                                .getBlocks()
                                .front()
                                .getOperations()) {
          if (isa<air::SegmentTerminatorOp>(o))
            continue;
          if (o.hasAttr("hoist-channel")) {
            if (auto child_for_op = dyn_cast<scf::ForOp>(o)) {
              cloneScfLoopUsingRemap<scf::ForOp>(rewriter, remap, child_for_op);
            } else if (auto child_parallel_op = dyn_cast<scf::ParallelOp>(o)) {
              cloneScfLoopUsingRemap<scf::ParallelOp>(rewriter, remap,
                                                      child_parallel_op);
            } else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)) {
              if (o.hasAttr("loop-carried-dep") &&
                  o.getAttrOfType<StringAttr>("loop-carried-dep")
                          .getValue()
                          .str() == "internalGetPut") {
                // Found channel op labelled as "internalGetPut", which
                // shouldn't be hoisted
                replaceAsyncOpWithWaitAllAndClone(rewriter, remap, &o, false);
              } else {
                rewriter.clone(o, remap);
              }
            } else {
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
      for (auto ext_channel_op : externalGetPut) {
        erased.insert(ext_channel_op.getOperation());
      }
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

  AffineParToHerdConversion(MLIRContext *ctx,
                            SmallPtrSet<Operation *, 8> &filteredOps)
      : OpRewritePattern(ctx), filteredOps(filteredOps){};

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getNumDims() != 2) {
      return op->emitOpError(
          "failed conversion to 'air.herd': only 2d loops are supported");
    }

    normalizeAffineParallel(op);

    auto loc = op.getLoc();
    auto ub0 =
        op.getUpperBoundsMap().getResult(0).dyn_cast<AffineConstantExpr>();
    auto ub1 =
        op.getUpperBoundsMap().getResult(1).dyn_cast<AffineConstantExpr>();

    if (!ub0 || !ub1) {
      return op->emitOpError("failed conversion to 'air.herd': only constant "
                             "loop bounds are supported");
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
    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<air::HerdTerminatorOp>(loc);

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
  for (auto &o : op.getBody().front().getOperations()) {
    builder.clone(o, remap);
  }

  // Terminators
  builder.setInsertionPointToEnd(&segment.getRegion().front());
  builder.create<air::SegmentTerminatorOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(&launch.getRegion().front());
  builder.create<air::LaunchTerminatorOp>(builder.getUnknownLoc());

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

    ivs[0].replaceAllUsesWith(herdOp.getIds()[idx0]);
    if (op.getNumLoops() == 2)
      ivs[1].replaceAllUsesWith(herdOp.getIds()[idx1]);

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&herdOp.getRegion().front());
    replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                          herdOp.getRegion());
    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<air::HerdTerminatorOp>(loc);

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
  llvm::SmallSet<air::HerdOp, 2> &replacementOps;
  int firstDim;
};

class ScfParToLaunchConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToLaunchConversion(MLIRContext *ctx,
                           llvm::SmallSet<scf::ParallelOp, 8> &filteredOps,
                           llvm::SmallSet<air::LaunchOp, 2> &replacementOps)
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
    auto launch = rewriter.create<air::LaunchOp>(op.getLoc(), sizes, args);
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
    builder.create<air::LaunchTerminatorOp>(loc);

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
  llvm::SmallSet<scf::ParallelOp, 8> &filteredOps;
  llvm::SmallSet<air::LaunchOp, 2> &replacementOps;
};

class ScfParToLaunchAndSegmentConversion
    : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ScfParToLaunchAndSegmentConversion(
      MLIRContext *ctx, llvm::SmallSet<scf::ParallelOp, 8> &filteredOps)
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
    auto segment = rewriter.create<air::SegmentOp>(op.getLoc(), segmentSizes,
                                                   segmentOpers);
    auto &bb = segment.getBody().front();
    auto ivs = op.getInductionVars();

    for (int i = 0, e = ivs.size(); i < e; i++) {
      ivs[i].replaceAllUsesWith(segment.getKernelArgument(i));
    }

    auto &body = op.getBody()->getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&segment.getRegion().front());
    replaceAllUsesOfConstsInRegionWithNew(constants, rewriter,
                                          segment.getRegion());

    auto builder = OpBuilder::atBlockEnd(&bb);
    builder.create<air::SegmentTerminatorOp>(builder.getUnknownLoc());
    builder = OpBuilder::atBlockEnd(&launch.getBody().front());
    builder.create<air::LaunchTerminatorOp>(builder.getUnknownLoc());

    int i = 0;
    auto kernel_args = segment.getKernelArguments();
    kernel_args = kernel_args.drop_front(
        ivs.size() + launch.getSize().size()); // Launch's induction vars
    for (Value v : args)
      replaceAllUsesInRegionWith(v, kernel_args[i++], segment.getRegion());

    if (op != parOp)
      op.erase();
    rewriter.eraseOp(parOp);

    return success();
  }

private:
  llvm::SmallSet<scf::ParallelOp, 8> &filteredOps;
};

struct CopyToDmaPass : public air::CopyToDmaBase<CopyToDmaPass> {

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

struct DmaToChannelPass : public air::DmaToChannelBase<DmaToChannelPass> {

  DmaToChannelPass() = default;
  DmaToChannelPass(const DmaToChannelPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });

    // Hoist broadcast pattern
    for (auto f : funcOps) {
      f.walk([&](mlir::AffineIfOp op) {
        if (!op->getParentOfType<mlir::AffineIfOp>()) {
          // Only hoist top-level affine if op with a nest of if ops
          HoistingAffineIf(op);
        }
      });
    }

    ConversionTarget target(*context);

    target
        .addLegalDialect<LLVM::LLVMDialect, func::FuncDialect, scf::SCFDialect,
                         AffineDialect, air::airDialect, arith::ArithDialect,
                         memref::MemRefDialect, linalg::LinalgDialect>();

    target.addIllegalOp<air::DmaMemcpyNdOp>();

    RewritePatternSet air_dma_patterns(context);
    air_dma_patterns.add<AIRDmaToAIRChannelConversion>(context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_dma_patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
    }

    // Dep tracing
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
    air::dependencyTracer depTracer;
    f.walk([&](Operation *op) {
      if (auto channel_op = mlir::dyn_cast<air::ChannelInterface>(op)) {
        if (channel_op->getAttrOfType<StringAttr>("loop-carried-dep") &&
            channel_op->getAttrOfType<StringAttr>("loop-carried-dep")
                    .getValue()
                    .str() == "externalGetPut") {

          // Start tracing dependency only if this put/get op is async
          auto async_op = dyn_cast<air::AsyncOpInterface>(op);
          if (!async_op.getAsyncToken())
            return;

          // Connect async dependency of external put/get scf parallel
          SmallVector<air::partialMemref, 1> sink_op_memref_reads;
          SmallVector<air::partialMemref, 1> sink_op_memref_writes;
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
                if (scf_par.getInitVals().size() &&
                    scf_par.getInitVals()[0].getDefiningOp()) {
                  sink_wait_all_op = dyn_cast<air::WaitAllOp>(
                      scf_par.getInitVals()[0].getDefiningOp());
                }
              } else if (auto scf_for = dyn_cast<scf::ForOp>(parent)) {
                if (scf_for.getIterOperands().size() &&
                    scf_for.getIterOperands()[0].getDefiningOp()) {
                  sink_wait_all_op = dyn_cast<air::WaitAllOp>(
                      scf_for.getIterOperands()[0].getDefiningOp());
                }
              }
            }
          }

          depTracer.getPartialMemrefFromOp(
              channel_op.getOperation(), sink_op_memref_reads,
              sink_op_memref_writes, sink_op_scalar_ins, sink_op_scalar_outs);

          assert(sink_op_memref_reads.size() ||
                 sink_op_memref_writes.size() &&
                     "cannot read memref from channel op");

          if (sink_wait_all_op) {
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

static void getHerdNames(ModuleOp module) {
  std::vector<std::string> herd_syms;
  for (auto f : module.getOps<func::FuncOp>()) {
    // record existing symbol names
    f.walk([&](air::HerdOp op) {
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

struct ParallelToHerdPass : public air::ParallelToHerdBase<ParallelToHerdPass> {

  ParallelToHerdPass() = default;
  ParallelToHerdPass(const ParallelToHerdPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    llvm::SmallVector<Operation *> hierOps;
    module.walk([&](air::HierarchyInterface op) { hierOps.push_back(op); });

    SmallPtrSet<Operation *, 8> filteredOps;
    llvm::SmallSet<air::HerdOp, 2> replacementOps;
    module.walk([&](Operation *op) {
      if (!isa<scf::ParallelOp, AffineParallelOp>(op))
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
        if (isa<scf::ParallelOp, AffineParallelOp>(par))
          parallel_depth++;
      if (parallel_depth != clAssignDepth)
        return;
      filteredOps.insert(op);
    });

    RewritePatternSet patterns(context);
    patterns.add<AffineParToHerdConversion>(context);
    patterns.add<ScfParToHerdConversion>(context, filteredOps, replacementOps,
                                         clFirstDim);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           air::airDialect, arith::ArithDialect>();

    target.addLegalOp<AffineApplyOp, AffineForOp, AffineLoadOp, AffineStoreOp,
                      AffineYieldOp, scf::YieldOp>();

    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp p) { return !filteredOps.contains(p); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }

    getHerdNames(module);
    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }
};

struct ParallelToLaunchPass
    : public air::ParallelToLaunchBase<ParallelToLaunchPass> {

  ParallelToLaunchPass() = default;
  ParallelToLaunchPass(const ParallelToLaunchPass &pass) {}

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

    llvm::SmallSet<scf::ParallelOp, 8> filteredOps;
    llvm::SmallSet<air::LaunchOp, 2> replacementOps;
    module.walk([&](scf::ParallelOp op) {
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
      Operation *par = op.getOperation();
      while ((par = par->getParentOp()))
        if (isa<scf::ParallelOp>(par))
          parallel_depth++;
      if (parallel_depth != clAssignDepth)
        return;
      filteredOps.insert(op);
    });

    RewritePatternSet patterns(context);
    if (clHasSegment) {
      patterns.add<ScfParToLaunchAndSegmentConversion>(context, filteredOps);
    } else {
      patterns.add<ScfParToLaunchConversion>(context, filteredOps,
                                             replacementOps);
    }

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           air::airDialect, arith::ArithDialect>();

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

struct InsertEmptyLaunchOverHerdPass
    : public air::InsertEmptyLaunchOverHerdBase<InsertEmptyLaunchOverHerdPass> {

  InsertEmptyLaunchOverHerdPass() = default;
  InsertEmptyLaunchOverHerdPass(const InsertEmptyLaunchOverHerdPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    module.walk([&](air::HerdOp op) {
      if (!op->getParentOfType<air::LaunchOp>())
        InsertEmptyLaunchOverHerd(op);
      else if (!op->getParentOfType<air::SegmentOp>())
        InsertEmptyLaunchOverHerd(op);
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ParToHerdOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ParToHerdOp::applyToOne(scf::ParallelOp target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  auto ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  llvm::SmallSet<air::HerdOp, 2> herdOps;
  llvm::SmallSet<Operation *, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<ScfParToHerdConversion>(ctx, filteredOps, herdOps,
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
transform::ParToLaunchOp::applyToOne(scf::ParallelOp target,
                                     transform::ApplyToEachResultList &results,
                                     transform::TransformState &state) {
  auto ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  llvm::SmallSet<air::LaunchOp, 2> launchOps;
  llvm::SmallSet<scf::ParallelOp, 8> filteredOps;
  filteredOps.insert(target);
  patterns.add<ScfParToLaunchConversion>(ctx, filteredOps, launchOps);
  (void)applyPatternsAndFoldGreedily(
      target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      std::move(patterns));
  for (auto l : launchOps)
    results.push_back(l);
  return DiagnosedSilenceableFailure::success();
}

class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

//===----------------------------------------------------------------------===//
// CopyToDmaOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CopyToDmaOp::applyToOne(memref::CopyOp op,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  auto ctx = op->getContext();
  // RewritePatternSet stage1Patterns =
  //   linalg::getLinalgTilingCanonicalizationPatterns(ctx);
  // memref::AllocOp::getCanonicalizationPatterns(stage1Patterns, ctx);
  // (void)applyPatternsAndFoldGreedily(op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
  //                                    std::move(stage1Patterns));
  SimpleRewriter rewriter(ctx);
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

std::unique_ptr<mlir::Pass> createParallelToLaunchPass() {
  return std::make_unique<ParallelToLaunchPass>();
}

std::unique_ptr<mlir::Pass> createCopyToDmaPass() {
  return std::make_unique<CopyToDmaPass>();
}

std::unique_ptr<mlir::Pass> createDmaToChannelPass() {
  return std::make_unique<DmaToChannelPass>();
}

std::unique_ptr<mlir::Pass> createInsertEmptyLaunchOverHerdPass() {
  return std::make_unique<InsertEmptyLaunchOverHerdPass>();
}

} // namespace air
} // namespace xilinx