//===- AIRDmaToChannel.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRDmaToChannel.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Dependency.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

#define DEBUG_TYPE "dma-to-channel"

namespace xilinx {

static void generateYieldAndOrReduceToScfLoop(OpBuilder builder,
                                              MLIRContext *ctx,
                                              scf::ParallelOp scf_par) {

  // Check if scf::YieldOp already exists in scf parallel
  SmallVector<scf::YieldOp, 2> y_ops(scf_par.getOps<scf::YieldOp>());
  if (y_ops.size()) {
    if (y_ops.size() != 1) {
      scf_par->emitOpError("number of yield op isn't one.");
      return;
    }
    builder.setInsertionPoint(y_ops[0]);
  } else {
    builder.setInsertionPointToEnd(scf_par.getBody());
  }

  auto wait_all_op_yielded = builder.create<air::WaitAllOp>(
      scf_par.getLoc(), air::AsyncTokenType::get(ctx), SmallVector<Value, 1>{});
  auto reduce_op = air::createSCFReduceForAsyncSCFParallel(
      builder, scf_par.getLoc(), wait_all_op_yielded.getAsyncToken(), ctx);
  builder.setInsertionPointToEnd(scf_par.getBody());

  wait_all_op_yielded->setAttr("hoist", StringAttr::get(ctx, "dep"));
  reduce_op->setAttr("hoist", StringAttr::get(ctx, "dep"));
  reduce_op.walk([&](mlir::Operation *o) {
    if (!isa<scf::YieldOp>(o)) {
      o->setAttr("hoist", StringAttr::get(ctx, "dep"));
    }
  });
}

static void getLeavesInDepGraph(Operation *op, SetVector<Value> &leaves_list) {
  SmallVector<Value> tokens;
  for (auto res : op->getResults())
    if (isa<air::AsyncTokenType>(res.getType()))
      tokens.push_back(res);
  for (auto token : tokens) {
    if (token.use_empty()) {
      leaves_list.insert(token);
    } else {
      for (auto u : token.getUsers())
        getLeavesInDepGraph(u, leaves_list);
    }
  }
}

static void getLeavesInDepGraph(Value v, SetVector<Value> &leaves_list) {
  for (auto u : v.getUsers())
    getLeavesInDepGraph(u, leaves_list);
}

static air::WaitAllOp
generateWaitAllToDanglingTokens(OpBuilder &builder, MLIRContext *ctx,
                                SmallVector<Value> inputTokens) {
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 1> yield_token;
  // Collect dangling leaves into yield
  SetVector<Value> dep_list;
  for (auto token : inputTokens) {
    getLeavesInDepGraph(token, dep_list);
  }
  return builder.create<air::WaitAllOp>(builder.getUnknownLoc(),
                                        air::AsyncTokenType::get(ctx),
                                        dep_list.takeVector());
}

static scf::YieldOp generateYieldAndOrReduceToScfLoop(OpBuilder &builder,
                                                      MLIRContext *ctx,
                                                      scf::ForOp scf_loop) {
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 1> yield_token;
  // Collect dangling leaves into yield
  SmallVector<Value> inputTokens;
  inputTokens.push_back(
      air::getLoopCarriedTokenFromScfOp(scf_loop, "argument"));
  auto wa_op = generateWaitAllToDanglingTokens(builder, ctx, inputTokens);
  yield_token.push_back(wa_op.getAsyncToken());
  wa_op->setAttr("hoist", StringAttr::get(ctx, "dep"));
  scf::YieldOp output =
      builder.create<scf::YieldOp>(builder.getUnknownLoc(), yield_token);
  return output;
}

// Clone ops in a block.
SmallVector<Operation *> air::cloneOpsInBlock(Block *blk, OpBuilder &builder,
                                              IRMapping &remap) {
  SmallVector<Operation *> clonedOps;
  for (Operation &o : blk->without_terminator()) {
    if (!o.hasAttr("hoist")) {
      if (air::isAsyncOp(&o)) {
        auto wa_op = air::replaceAsyncOpWithWaitAll(builder, remap, &o, false);
        wa_op->setAttr("hoist", StringAttr::get(o.getContext(), "dep"));
        clonedOps.push_back(wa_op);
      }
      continue;
    }
    if (auto child_for_op = dyn_cast<LoopLikeOpInterface>(o)) {
      auto clonedScfLoopOps =
          air::cloneScfLoopUsingRemap(builder, remap, child_for_op);
      clonedOps.insert(clonedOps.end(), clonedScfLoopOps.begin(),
                       clonedScfLoopOps.end());
    } else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)) {
      if (o.hasAttr("loop-carried-dep") &&
          o.getAttrOfType<StringAttr>("loop-carried-dep").getValue().str() ==
              "internalGetPut") {
        // Found channel op labelled as "internalGetPut", which
        // shouldn't be hoisted
        if (air::isAsyncOp(&o)) {
          auto wa_op =
              air::replaceAsyncOpWithWaitAll(builder, remap, &o, false);
          wa_op->setAttr("hoist", StringAttr::get(o.getContext(), "dep"));
          clonedOps.push_back(wa_op);
        }
      } else {
        clonedOps.push_back(builder.clone(o, remap));
      }
    } else if (auto aif_op = dyn_cast<affine::AffineIfOp>(o)) {
      auto clonedAifOps = air::cloneAffineIfUsingRemap(builder, remap, aif_op);
      clonedOps.insert(clonedOps.end(), clonedAifOps.begin(),
                       clonedAifOps.end());
    } else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(o)) {
      if (o.hasAttr("loop-carried-dep"))
        clonedOps.push_back(builder.clone(o, remap));
      else {
        auto wa_op = air::replaceAsyncOpWithWaitAll(builder, remap, &o, false);
        wa_op->setAttr("hoist", StringAttr::get(o.getContext(), "dep"));
        clonedOps.push_back(wa_op);
      }
    } else if (!air::isPure(&o) && !isa<air::WaitAllOp>(o)) {
      if (air::isAsyncOp(&o)) {
        auto wa_op = air::replaceAsyncOpWithWaitAll(builder, remap, &o, false);
        wa_op->setAttr("hoist", StringAttr::get(o.getContext(), "dep"));
        clonedOps.push_back(wa_op);
      }
    } else {
      clonedOps.push_back(builder.clone(o, remap));
    }
  }
  return clonedOps;
}

SmallVector<Operation *>
air::cloneAffineIfUsingRemap(OpBuilder builder, IRMapping &remap,
                             affine::AffineIfOp aif_op) {
  // Clone the affine if op body instead of the if op.
  SmallVector<Operation *> clonedOps;
  auto clonedThenOps = cloneOpsInBlock(aif_op.getThenBlock(), builder, remap);
  clonedOps.insert(clonedOps.end(), clonedThenOps.begin(), clonedThenOps.end());
  if (aif_op.hasElse()) {
    auto clonedElseOps = cloneOpsInBlock(aif_op.getElseBlock(), builder, remap);
    clonedOps.insert(clonedOps.end(), clonedElseOps.begin(),
                     clonedElseOps.end());
  }
  return clonedOps;
}

template <typename T>
SmallVector<Operation *>
air::cloneScfLoopUsingRemap(OpBuilder builder, IRMapping &remap, T loop_op,
                            air::ChannelInterface externalGetPut) {
  SmallVector<Value> loop_init_args = air::getAsyncDependenciesFromOp(loop_op);
  T new_loop_op = builder.create<T>(
      builder.getUnknownLoc(),
      air::lookupOrDefaultRange(loop_op.getLowerBound(), remap),
      air::lookupOrDefaultRange(loop_op.getUpperBound(), remap),
      air::lookupOrDefaultRange(loop_op.getStep(), remap),
      air::lookupOrDefaultRange(loop_init_args, remap));

  OpBuilder::InsertionGuard guard(builder);

  // Remap newly created loop op
  for (unsigned i = 0; i < loop_op->getNumResults(); i++)
    remap.map(loop_op->getResult(i), new_loop_op->getResult(i));

  auto remapVals = [&](std::optional<SmallVector<OpFoldResult>> oldValues,
                       std::optional<SmallVector<OpFoldResult>> newValues) {
    if (!oldValues || !newValues)
      return;
    SmallVector<OpFoldResult> o = *oldValues;
    SmallVector<OpFoldResult> n = *newValues;
    for (auto p : llvm::zip(o, n))
      remap.map(cast<Value>(std::get<0>(p)), cast<Value>(std::get<1>(p)));
  };
  remapVals(loop_op.getLoopLowerBounds(), new_loop_op.getLoopLowerBounds());
  remapVals(loop_op.getLoopUpperBounds(), new_loop_op.getLoopUpperBounds());
  remapVals(loop_op.getLoopSteps(), new_loop_op.getLoopSteps());

  for (auto p :
       llvm::zip(loop_op.getRegionIterArgs(), new_loop_op.getRegionIterArgs()))
    remap.map(std::get<0>(p), std::get<1>(p));

  for (auto p : llvm::zip(*loop_op.getLoopInductionVars(),
                          *new_loop_op.getLoopInductionVars()))
    remap.map(std::get<0>(p), std::get<1>(p));

  builder.setInsertionPointToStart(new_loop_op.getBody());
  auto clonedOps = cloneOpsInBlock(loop_op.getBody(), builder, remap);

  new_loop_op->setAttr("hoist",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));
  new_loop_op->setAttr("loop-carried-dep",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));

  // Generate yield op and/or reduce op if async
  if (air::getAsyncDependenciesFromOp(loop_op).size()) {
    generateYieldAndOrReduceToScfLoop(builder, loop_op->getContext(),
                                      new_loop_op);
  }

  clonedOps.push_back(new_loop_op);

  return clonedOps;
}

template <>
SmallVector<Operation *> air::cloneScfLoopUsingRemap<LoopLikeOpInterface>(
    OpBuilder builder, IRMapping &remap, LoopLikeOpInterface loop_op,
    air::ChannelInterface externalGetPut) {
  Operation *op = loop_op.getOperation();
  if (scf::ForOp fop = dyn_cast<scf::ForOp>(op)) {
    return cloneScfLoopUsingRemap(builder, remap, fop, externalGetPut);
  } else if (scf::ParallelOp pop = dyn_cast<scf::ParallelOp>(op)) {
    return cloneScfLoopUsingRemap(builder, remap, pop, externalGetPut);
  }
  loop_op.emitOpError("unsupported loop type");
  return SmallVector<Operation *>();
}

static scf::ParallelOp
hoistAIRHierToScfParallel(OpBuilder builder, Location loc, MLIRContext *ctx,
                          air::HierarchyInterface hierOp,
                          SmallVector<Operation *> targetOpsToHoist) {

  auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 2> steps;
  SmallVector<Value, 2> lbs;
  SmallVector<Value, 2> ubs;

  // Infer the scf.parallel shape through any affine.if nest around the target
  // ops, until it reaches a parent spatial loop (e.g. scf.parallel or
  // air.hierarchy).
  std::vector<Operation *> affine_if_nest;
  Operation *spatial_loop = nullptr;
  (void)air::getAffineIfNestAndSpatialLoopFromOp(targetOpsToHoist.front(),
                                                 affine_if_nest, spatial_loop);
  SmallVector<std::pair<int, int>> conditionBounds =
      air::getRectangularConditionBoundsThroughAffineIfs(
          targetOpsToHoist.front(), spatial_loop, affine_if_nest);
  for (auto [lbs_int, ubs_int] : conditionBounds) {
    lbs.push_back(builder.create<arith::ConstantIndexOp>(loc, lbs_int));
    ubs.push_back(builder.create<arith::ConstantIndexOp>(loc, ubs_int + 1));
    steps.push_back(step);
  }

  auto hierAsyncIfOp = dyn_cast<air::AsyncOpInterface>(hierOp.getOperation());

  auto wa_op = builder.create<air::WaitAllOp>(
      loc, air::AsyncTokenType::get(ctx), hierAsyncIfOp.getAsyncDependencies());
  SmallVector<Value, 1> deps_in{wa_op.getAsyncToken()};
  scf::ParallelOp scf_par = nullptr;
  if (isAsyncOp(hierAsyncIfOp)) {
    scf_par = builder.create<scf::ParallelOp>(loc, lbs, ubs, steps, deps_in);
    generateYieldAndOrReduceToScfLoop(builder, ctx, scf_par);
    hierAsyncIfOp.getAsyncToken().replaceAllUsesWith(
        air::getAsyncTokenFromOp(scf_par));
  } else
    scf_par = builder.create<scf::ParallelOp>(loc, lbs, ubs, steps);

  scf_par->setAttr("hoist", StringAttr::get(ctx, "hoistedLoop"));
  scf_par->setAttr("loop-carried-dep", StringAttr::get(ctx, "hoistedLoop"));

  return scf_par;
}

// Create channel symbol
static air::ChannelOp
createChannelOp(OpBuilder builder, ModuleOp module, std::string cname,
                Location loc, SmallVector<int64_t, 2> channel_bundle_sizes) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  Operation *o = &module.getBody()->front();
  while (dyn_cast_or_null<air::ChannelOp>(o))
    o = o->getNextNode();
  builder.setInsertionPoint(o);

  auto channel_op = builder.create<air::ChannelOp>(
      loc, cname, builder.getI64ArrayAttr(channel_bundle_sizes),
      builder.getStringAttr("dma_stream"));

  builder.restoreInsertionPoint(insertionCheckpoint);

  return channel_op;
}

static void replaceAIRDmaWithAIRChannelPairs(
    OpBuilder &builder, unsigned innerMemorySpace, air::DmaMemcpyNdOp op,
    SmallVector<air::ChannelInterface, 1> &internalGetPutVector,
    SmallVector<air::ChannelInterface, 1> &externalGetPutVector) {
  auto loc = op->getLoc();
  auto src = op.getSrcMemref();
  auto dst = op.getDstMemref();
  auto ctx = op->getContext();

  auto src_type = llvm::dyn_cast<BaseMemRefType>(src.getType());
  auto dst_type = llvm::dyn_cast<BaseMemRefType>(dst.getType());
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
  std::string cname = air::createChannelName(module);

  if (op->hasAttr("broadcast_set")) {
    // If the data movement is subject to a broadcasting pattern, then
    // specialize each broadcast source in a bundle into a separate channel.
    // Infer broadcast shape from integer set, if broadcast_set attribute is
    // set.
    auto int_set =
        op->getAttrOfType<mlir::IntegerSetAttr>("broadcast_set").getValue();
    SmallVector<int, 2> lbs_int = {-1, -1};
    SmallVector<int, 2> ubs_int = {-1, -1};
    SmallVector<int64_t, 2> channel_sizes = {1, 1};
    air::getSizesFromIntegerSet(ctx, int_set, lbs_int, ubs_int);
    SmallVector<int64_t, 2> bcast_sizes = {ubs_int[0] - lbs_int[0] + 1,
                                           ubs_int[1] - lbs_int[1] + 1};
    auto channel_op =
        createChannelOp(builder, module, cname, loc, channel_sizes);
    channel_op->setAttr("broadcast_shape",
                        builder.getI64ArrayAttr(bcast_sizes));
  } else {
    // Else, infer channel's input shape from parent spatial loop, i.e. herd if
    // within a herd, or scf.parallel if within an scf.parallel.
    SmallVector<int64_t, 2> channel_sizes;
    if (auto parent_herd_op = op->getParentOfType<air::HerdOp>()) {
      auto herd_size = parent_herd_op.getSizeOperands();
      for (unsigned i = 0; i < herd_size.size(); i++) {
        channel_sizes.push_back(
            herd_size[i].getDefiningOp<arith::ConstantIndexOp>().value());
      }
    } else if (auto parent_par_op = op->getParentOfType<scf::ParallelOp>()) {
      SmallVector<int, 2> lbs_spatial, ubs_spatial;
      air::getSizesFromSpatialLoop(parent_par_op, lbs_spatial, ubs_spatial);
      for (unsigned i = 0; i < ubs_spatial.size(); i++)
        channel_sizes.push_back(ubs_spatial[i] - lbs_spatial[i] + 1);
    }
    createChannelOp(builder, module, cname, loc, channel_sizes);

    // Issue warnings.
    if (op->hasAttr("broadcast_pattern"))
      op->emitWarning("Attribute broadcast_pattern is set, but data movement "
                      "isn't specialized via affine if guards. Therefore, the "
                      "broadcast pattern is ignored.");
  }

  SmallVector<Value, 1> channel_idx_internal{};
  SmallVector<Value, 1> channel_idx_external{};
  if (op->hasAttr("broadcast_set")) {
    // If broadcasting, let internal channel inherit affine.if's operands
    auto parent_affine_if_op = op->getParentOfType<affine::AffineIfOp>();
    for (auto operand : parent_affine_if_op->getOperands()) {
      channel_idx_internal.push_back(operand);
    }
  } else if (auto parent_herd_op = op->getParentOfType<air::HerdOp>()) {
    // Let both channel ops inherit herd's induction variables
    for (auto iv : parent_herd_op.getIds()) {
      channel_idx_internal.push_back(iv);
      channel_idx_external.push_back(iv);
    }
  } else if (auto parent_par_op = op->getParentOfType<scf::ParallelOp>()) {
    // Likewise, inherit scf.paralel op's induction variables
    for (auto iv : parent_par_op.getInductionVars()) {
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

  if (!internalGetPut) {
    op->emitOpError("has unexpected memref memory space at internal-side");
    return;
  }
  if (!externalGetPut) {
    op->emitOpError("has unexpected memref memory space at external-side");
    return;
  }

  // Replace all uses to dma token with internal put/get token
  if (auto op_token = op.getAsyncToken()) {
    auto asyncInternalGetPut =
        dyn_cast<air::AsyncOpInterface>(internalGetPut.getOperation());
    op_token.replaceAllUsesWith(asyncInternalGetPut.getAsyncToken());
  }

  // Add attributes to label internal/external channel ops
  externalGetPut->setAttr("hoist", StringAttr::get(op->getContext(), "dep"));
  internalGetPut->setAttr("loop-carried-dep",
                          StringAttr::get(op->getContext(), "internalGetPut"));
  externalGetPut->setAttr("loop-carried-dep",
                          StringAttr::get(op->getContext(), "external"));
  if (op->hasAttr("broadcast_set"))
    externalGetPut->setAttr("broadcast_set", op->getAttr("broadcast_set"));

  externalGetPutVector.push_back(externalGetPut);
  internalGetPutVector.push_back(internalGetPut);
}

// Check whether an channel op is within a matching air hierarchy (launch for
// any of [L1, L2, L3] memref, segment for [L1, L2] memref, and herd for L1
// memref).
bool isInMatchingHierarchy(air::ChannelInterface getput) {
  auto memref = getput.getMemref();
  auto memrefType = llvm::dyn_cast<BaseMemRefType>(memref.getType());
  if (!memrefType)
    return false;
  // Skip if channel op is already at its correct memory hierarchy.
  if (!getput->getParentOfType<air::HierarchyInterface>())
    return true;
  if (isa<air::HerdOp>(getput->getParentOfType<air::HierarchyInterface>()) &&
      (memrefType.getMemorySpaceAsInt() == (int)air::MemorySpace::L1))
    return true;
  else if (isa<air::SegmentOp>(
               getput->getParentOfType<air::HierarchyInterface>()) &&
           (memrefType.getMemorySpaceAsInt() == (int)air::MemorySpace::L2 ||
            memrefType.getMemorySpaceAsInt() == (int)air::MemorySpace::L1))
    return true;
  else if (isa<air::LaunchOp>(
               getput->getParentOfType<air::HierarchyInterface>())) {
    // Already at the outermost hierarchy level. No where to hoist.
    return true;
  }
  return false;
}

// Check whether a channel op is an "external" side channel op.
bool isValidExternalChannelOp(air::ChannelInterface getput) {
  // It must be the "external" half of the data movement.
  StringAttr dmaToChanAttr =
      getput->getAttrOfType<StringAttr>("loop-carried-dep");
  if (!dmaToChanAttr)
    return false;
  if (dmaToChanAttr.str() != "external")
    return false;

  // It must operate on a memref with static shape.
  auto memref = getput.getMemref();
  auto memrefType = llvm::dyn_cast<BaseMemRefType>(memref.getType());
  if (!memrefType)
    return false;

  // Skip if channel op is already at its correct memory hierarchy.
  if (isInMatchingHierarchy(getput))
    return false;
  return true;
}

} // namespace xilinx

namespace xilinx {
namespace air {

class AIRDmaToAIRChannelConversion
    : public OpRewritePattern<air::DmaMemcpyNdOp> {
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {

    auto src = op.getSrcMemref();
    auto dst = op.getDstMemref();

    // It must already be a memref
    auto src_type = llvm::dyn_cast<BaseMemRefType>(src.getType());
    auto dst_type = llvm::dyn_cast<BaseMemRefType>(dst.getType());
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

    SmallVector<air::ChannelInterface, 1> externalGetPut;
    SmallVector<air::ChannelInterface, 1> internalGetPut;

    replaceAIRDmaWithAIRChannelPairs(rewriter, innerMemorySpace, op,
                                     internalGetPut, externalGetPut);

    rewriter.eraseOp(op);

    return success();
  }
};

// Hoist the "external" half of the data movement out by one level of air
// hierarchy, based on the memory space that it is operating on.
template <typename AIRHierOpTy>
class AIRHoistExternalAIRChannelPattern : public OpRewritePattern<AIRHierOpTy> {
  using OpRewritePattern<AIRHierOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(AIRHierOpTy hier_op,
                                PatternRewriter &rewriter) const override {

    auto loc = hier_op->getLoc();
    auto ctx = hier_op->getContext();

    // Collect the "external" side channel operations, as targets for hoisting.
    // Do not dive into any child air hierarchy ops.
    SmallVector<air::ChannelInterface> externalGetPuts;
    hier_op.template walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [&externalGetPuts, hier_op](Operation *o) {
          if (isa<air::HierarchyInterface>(o) && o != hier_op)
            return WalkResult::skip();
          auto getput = dyn_cast<air::ChannelInterface>(o);
          if (!getput)
            return WalkResult::advance();
          // It must be the "external" half of the data movement.
          if (!isValidExternalChannelOp(getput))
            return WalkResult::advance();
          externalGetPuts.push_back(getput);
          return WalkResult::advance();
        });
    if (externalGetPuts.empty())
      return failure();

    // Get backward slices to the target "external" side channel ops, to be
    // hoisted together.
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions bsOptions{[&](Operation *o) { return o != hier_op; }};
    for (auto op : externalGetPuts) {
      (void)getBackwardSlice(op.getOperation(), &backwardSlice, bsOptions);

      for (auto parent = op->getParentOp();
           !isa<air::HierarchyInterface>(parent);
           parent = parent->getParentOp()) {
        (void)getBackwardSlice(parent, &backwardSlice, bsOptions);
        backwardSlice.insert(parent);
      }
    }
    // Get constant values used by backward slices, and add to backward
    // slices.
    for (auto o : backwardSlice) {
      for (auto &region : o->getRegions()) {
        visitUsedValuesDefinedAbove(region, [&backwardSlice](OpOperand *use) {
          if (getConstantIntValue(use->get())) {
            backwardSlice.insert(use->get().getDefiningOp());
          }
        });
      }
    }

    // Don't miss out the backward slices of air.execute op's child ops.
    auto backwardSliceCopy = backwardSlice;
    for (auto b : backwardSliceCopy) {
      if (auto execOp = dyn_cast<air::ExecuteOp>(b)) {
        for (auto &exec_child_op : execOp.getChildOps()) {
          (void)getBackwardSlice(&exec_child_op, &backwardSlice, bsOptions);
          backwardSlice.insert(&exec_child_op);
        }
      }
    }

    // Label backward slices with attribute; ops not labelled with "hoist" flag
    // shall either not get hoisted, if IR is not async, or become air.wait_all
    // (null op) after being hoisted.
    backwardSlice.insert(externalGetPuts.begin(), externalGetPuts.end());
    for (auto b : backwardSlice) {
      b->setAttr("hoist", StringAttr::get(ctx, "dep"));
    }

    // Hoist hierarchy op into scf op
    Operation *scf_loop = nullptr;
    mlir::OpBuilder::InsertPoint
        insertionPointAtHierOp; // To keep a record of the insertion point as
                                // destination for hoisting

    rewriter.setInsertionPoint(hier_op);
    insertionPointAtHierOp = rewriter.saveInsertionPoint();

    // Check if broadcasting happens for any "external" side channel ops. If so,
    // the hoisted scf parallel should respect the broadcast shape instead. If
    // broadcasting is detected, then hoist and specialize each data movement
    // (i.e. do not hoist the air.hierarchy iteration space.)
    if (llvm::any_of(externalGetPuts, [](air::ChannelInterface getput) {
          return air::getChannelDeclarationThroughSymbol(getput)->hasAttr(
              "broadcast_shape");
        }))
      insertionPointAtHierOp = rewriter.saveInsertionPoint();
    else {
      if (hier_op.getNumDims()) {
        // Hoist air.hierarchy as scf.parallel; scf.parallel shape is equal to
        // air.hierarchy shape.
        SmallVector<Operation *> targetOpsToHoist(externalGetPuts.begin(),
                                                  externalGetPuts.end());
        scf::ParallelOp scf_par = hoistAIRHierToScfParallel(
            rewriter, loc, ctx, hier_op, targetOpsToHoist);
        scf_loop = scf_par.getOperation();
      } else {
        // Air.hierarchy op has no dimensions. No need to hoist into any
        // scf.parallel loop.
        insertionPointAtHierOp = rewriter.saveInsertionPoint();
      }
    }

    // Hoist ops to "external" side code region, by cloning with remap.
    SmallVector<Operation *> clonedOps;
    IRMapping remap;
    if (auto scf_par = dyn_cast_or_null<scf::ParallelOp>(scf_loop)) {
      // If air.hierarchy is hoisted into an scf.parallel loop.

      // Remap the air.hierarchy to the hoisted scf.parallel.
      auto hier_size = hier_op.getSizeOperands();
      for (unsigned i = 0; i < hier_op.getNumDims(); i++) {
        remap.map(hier_op.getSize()[i], hier_size[i]);
        remap.map(hier_op.getIds()[i], scf_par.getInductionVars()[i]);
      }
      int arg_idx = 0;
      for (auto arg : hier_op.getKernelArguments())
        remap.map(arg, hier_op.getKernelOperand(arg_idx++));
      // Clone ops into hoisted scf.parallel
      rewriter.setInsertionPointToStart(scf_par.getBody());
      clonedOps =
          air::cloneOpsInBlock(&hier_op.getBody().front(), rewriter, remap);
      if (clonedOps.empty())
        return failure();
    } else {
      rewriter.restoreInsertionPoint(insertionPointAtHierOp);
      // Remap isolated-from-above air hierarchy op arguments.
      int arg_idx = 0;
      for (auto arg : hier_op.getKernelArguments())
        remap.map(arg, hier_op.getKernelOperand(arg_idx++));

      // Remap ssa values used by the hoisted ops
      for (auto externalGetPut : externalGetPuts) {
        if (!externalGetPut->hasAttr("broadcast_set"))
          continue;
        // If the "external" side channel op is subject to a broadcasting
        // pattern, then specailze the original induction variables by applying
        // the affine.if's integer set.
        auto is = externalGetPut->getAttrOfType<IntegerSetAttr>("broadcast_set")
                      .getValue();
        for (size_t hierDim = 0; hierDim < hier_op.getNumDims(); hierDim++) {
          remap.map(hier_op.getIds()[hierDim],
                    rewriter.create<arith::ConstantIndexOp>(
                        rewriter.getUnknownLoc(), 0));
          for (unsigned i = 0; i < is.getNumConstraints(); i++) {
            if (!is.isEq(i))
              continue;
            auto c = is.getConstraint(i);
            if (!c.isFunctionOfSymbol(hierDim))
              continue;
            auto constIV = rewriter.create<arith::ConstantIndexOp>(
                rewriter.getUnknownLoc(),
                air::evaluateSymbolEqualityInSet(c, ctx));
            remap.map(hier_op.getIds()[hierDim], constIV);
          }
        }
      }

      // Hoist ops
      clonedOps =
          air::cloneOpsInBlock(&hier_op.getBody().front(), rewriter, remap);
      if (clonedOps.empty())
        return failure();
    }

    // Check if hoisted channel ops are now under a matching air.hierarchy.
    // Update compiler flags accordingly.
    for (auto cloned : clonedOps) {
      auto clonedExternalGetPut = dyn_cast<air::ChannelInterface>(cloned);
      if (!clonedExternalGetPut)
        continue;
      if (!clonedExternalGetPut->hasAttr("loop-carried-dep"))
        continue;
      auto compilerFlagAttr =
          clonedExternalGetPut->getAttrOfType<StringAttr>("loop-carried-dep");
      if (compilerFlagAttr.str() == "external" &&
          isInMatchingHierarchy(clonedExternalGetPut)) {
        clonedExternalGetPut->setAttr("loop-carried-dep",
                                      rewriter.getStringAttr("internalGetPut"));
      }
    }

    std::set<Operation *> erased;
    // Remove "hoist" flags to avoid conflict with the next greedily applied
    // pattern rewrite.
    if (scf_loop) {
      scf_loop->walk([&](mlir::Operation *o) {
        if (o == o->getBlock()->getTerminator())
          return;
        if (!o->hasAttr("hoist"))
          erased.insert(o);
        else
          o->removeAttr("hoist");
      });
    }
    hier_op.walk([&](mlir::Operation *o) {
      if (o->hasAttr("hoist"))
        o->removeAttr("hoist");
    });
    for (auto cloned : clonedOps) {
      if (cloned->hasAttr("hoist"))
        cloned->removeAttr("hoist");
    }

    // Remove the original "external" side puts and gets.
    for (auto getput : externalGetPuts) {
      if (air::isAsyncOp(getput)) {
        IRMapping remap;
        rewriter.setInsertionPoint(getput);
        auto waOp =
            air::replaceAsyncOpWithWaitAll(rewriter, remap, getput, false);
        rewriter.replaceOp(getput, waOp);
      } else
        rewriter.eraseOp(getput);
    }
    for (auto e : erased) {
      rewriter.eraseOp(e);
    }

    return success();
  }
};

template <class T>
static Value insertArgToHierOpImpl(OpBuilder &builder, T op,
                                   SmallVector<Value> vec) {
  // make a list of new hierarchy operands
  SmallVector<Value> newOperands;
  SmallVector<int> newOperandsIdx;
  for (int i = 0, e = op.getNumKernelOperands(); i < e; i++) {
    newOperands.push_back(op.getKernelOperand(i));
    newOperandsIdx.push_back(i);
  }
  newOperands.insert(newOperands.end(), vec.begin(), vec.end());

  // make a list of new async token operands
  SmallVector<Value> newAsyncDeps = op.getAsyncDependencies();

  // replace hier op
  builder.setInsertionPoint(op);
  IRMapping remap;
  auto newOp =
      builder.create<T>(op.getLoc(), newAsyncDeps, op.getSizeOperands(),
                        newOperands, op->getNumResults() > 0, op->getAttrs());

  builder.setInsertionPointToStart(&newOp.getBody().front());
  for (auto p : llvm::zip(op.getSize(), newOp.getSize()))
    remap.map(std::get<0>(p), std::get<1>(p));
  for (auto p : llvm::zip(op.getIds(), newOp.getIds()))
    remap.map(std::get<0>(p), std::get<1>(p));

  int newIdx = 0;
  for (int i : newOperandsIdx)
    remap.map(op.getKernelArgument(i), newOp.getKernelArgument(newIdx++));
  for (uint64_t i = 0; i < vec.size(); i++)
    remap.map(vec[i], newOp.getKernelArgument(op.getNumKernelOperands() + i));

  for (Operation &o : op.getRegion().front().getOperations())
    if (!isa<air::HerdTerminatorOp>(o))
      builder.clone(o, remap);

  int res_idx = 0;
  for (auto r : op.getResults())
    r.replaceAllUsesWith(newOp->getResult(res_idx++));
  op->erase();

  return newOp.getKernelOperand(newOp.getNumKernelOperands() - 1);
}

static Value insertArgToHierOp(OpBuilder &builder, Operation *op,
                               SmallVector<Value> vec) {
  if (!isa<air::HierarchyInterface>(op))
    return nullptr;
  else if (auto herd = dyn_cast<air::HerdOp>(op))
    return insertArgToHierOpImpl<air::HerdOp>(builder, herd, vec);
  else if (auto segment = dyn_cast<air::SegmentOp>(op))
    return insertArgToHierOpImpl<air::SegmentOp>(builder, segment, vec);
  else if (auto launch = dyn_cast<air::LaunchOp>(op))
    return insertArgToHierOpImpl<air::LaunchOp>(builder, launch, vec);
  else
    return nullptr;
}

static LogicalResult AIRDemoteMemrefToAIRHierarchy(
    std::pair<air::HierarchyInterface, std::vector<Operation *>> pair,
    OpBuilder &builder) {

  air::HierarchyInterface hier_op = pair.first;
  unsigned int hierMemorySpace = 0;
  if (isa<air::HerdOp>(hier_op.getOperation())) {
    hierMemorySpace = (int)air::MemorySpace::L1;
  } else if (isa<air::SegmentOp>(hier_op.getOperation())) {
    hierMemorySpace = (int)air::MemorySpace::L2;
  } else
    return failure();

  {
    OpBuilder::InsertionGuard guard(builder);

    SmallVector<Value> new_memrefs;
    for (auto op : pair.second) {
      auto loc = op->getLoc();
      auto memref =
          isa<air::ExecuteOp>(op) ? op->getResult(1) : op->getResult(0);
      auto token = isa<air::ExecuteOp>(op) ? op->getResult(0) : nullptr;
      auto memref_type = llvm::dyn_cast<BaseMemRefType>(memref.getType());

      if (memref_type.getMemorySpaceAsInt() == hierMemorySpace)
        continue; // Alloc op is already under correct hierarchy
      else if (memref_type.getMemorySpaceAsInt() > hierMemorySpace)
        continue; // This pass is currently not able to promote in memory tier

      // Get dealloc
      Operation *dealloc = nullptr;
      for (auto u : memref.getUsers()) {
        if (isa<memref::DeallocOp>(u)) {
          // If async
          if (auto exec = u->getParentOfType<air::ExecuteOp>()) {
            dealloc = exec.getOperation();
          } else
            dealloc = u;
        }
      }

      // Hierarchy ops are isolated from above. Inserting arguments.
      builder.setInsertionPoint(hier_op);
      auto new_op = builder.clone(*op);
      if (auto new_alloc = dyn_cast<memref::AllocOp>(new_op)) {
        memref.replaceAllUsesWith(new_alloc.getMemref());
        new_memrefs.push_back(new_alloc.getMemref());
      } else if (auto new_exec = dyn_cast<air::ExecuteOp>(new_op)) {
        memref.replaceAllUsesWith(new_exec->getResult(1));
        new_memrefs.push_back(new_exec->getResult(1));
        // token.replaceAllUsesWith(new_exec->getResult(0));
        builder.setInsertionPoint(op);
        token.replaceAllUsesWith(
            builder
                .create<air::WaitAllOp>(
                    loc, air::AsyncTokenType::get(op->getContext()),
                    new_exec.getAsyncDependencies())
                .getAsyncToken());
        // Update async deps
        clearAsyncDependenciesOfAsyncOp(new_exec);
        auto async_hier_op =
            dyn_cast<air::AsyncOpInterface>(hier_op.getOperation());
        for (auto dep : async_hier_op.getAsyncDependencies()) {
          new_exec.addAsyncDependency(dep);
        }
        async_hier_op.addAsyncDependency(new_exec.getAsyncToken());
      } else
        return failure();
      op->erase();

      if (dealloc) {
        builder.setInsertionPointAfter(hier_op);
        auto new_dealloc = builder.clone(*dealloc);
        if (auto new_exec = dyn_cast<air::ExecuteOp>(new_dealloc)) {
          builder.setInsertionPoint(dealloc);
          dealloc->getResult(0).replaceAllUsesWith(
              builder
                  .create<air::WaitAllOp>(
                      loc, air::AsyncTokenType::get(op->getContext()),
                      new_exec.getAsyncDependencies())
                  .getAsyncToken());
          clearAsyncDependenciesOfAsyncOp(new_exec);
          new_exec.addAsyncDependency(hier_op->getResult(0));
        }
        dealloc->erase();
      }
    }

    insertArgToHierOp(builder, hier_op.getOperation(), new_memrefs);
  }

  return success();
}

class AIRDemoteDmaToAIRHierarchyConversion
    : public OpRewritePattern<air::DmaMemcpyNdOp> {
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto src = op.getSrcMemref();
    auto dst = op.getDstMemref();
    auto ctx = op->getContext();

    // It must already be a memref
    auto src_type = llvm::dyn_cast<BaseMemRefType>(src.getType());
    auto dst_type = llvm::dyn_cast<BaseMemRefType>(dst.getType());
    if (!src_type)
      return failure();

    if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
      return failure();

    auto herd = op->getParentOfType<air::HerdOp>();
    auto segment = op->getParentOfType<air::SegmentOp>();

    if (src_type.getMemorySpaceAsInt() == dst_type.getMemorySpaceAsInt())
      return failure(); // Src and dst under same memory space

    air::HierarchyInterface hier_op = nullptr;
    unsigned int innerMemorySpace = 0;
    if (herd) {
      hier_op = dyn_cast<air::HierarchyInterface>(herd.getOperation());
      innerMemorySpace = (int)air::MemorySpace::L1;
    } else if (segment) {
      hier_op = dyn_cast<air::HierarchyInterface>(segment.getOperation());
      innerMemorySpace = (int)air::MemorySpace::L2;
    } else
      return failure();

    auto memcpyInnerMemorySpace = std::max(src_type.getMemorySpaceAsInt(),
                                           dst_type.getMemorySpaceAsInt());
    if (memcpyInnerMemorySpace == innerMemorySpace)
      return failure(); // Dma op is already under correct hierarchy
    else if (memcpyInnerMemorySpace > innerMemorySpace)
      return failure(); // This pass is currently not able to promote in memory
                        // tier

    SmallVector<Value, 4> src_offsets = op.getSrcOffsets();
    SmallVector<Value, 4> dst_offsets = op.getDstOffsets();
    SmallVector<Value, 4> src_sizes = op.getSrcSizes();
    SmallVector<Value, 4> dst_sizes = op.getDstSizes();
    SmallVector<Value, 4> src_strides = op.getSrcStrides();
    SmallVector<Value, 4> dst_strides = op.getDstStrides();

    std::set<Operation *> erased;

    {
      OpBuilder::InsertionGuard guard(rewriter);

      bool hoist_herd = false;
      for (auto &elem : traceDependentHerdId(op)) {
        for (auto v : std::get<1>(elem)) {
          if (air::getHerdArgOwner(v)) {
            hoist_herd = true;
          }
        }
      }

      SetVector<Operation *> backwardSlice;
      // Transitive defs up to scf.for.
      BackwardSliceOptions bsOptions{
          [&](Operation *o) { return o != hier_op && !isa<scf::ForOp>(o); }};
      (void)getBackwardSlice(op.getOperation(), &backwardSlice, bsOptions);

      if (hoist_herd) {
        // Transitive defs up to air.herd.
        BackwardSliceOptions bsOptionsHoistHerd{
            [&](Operation *o) { return o != hier_op; }};
        for (auto parent = op->getParentOp();
             !isa<air::HierarchyInterface>(parent);
             parent = parent->getParentOp()) {
          (void)getBackwardSlice(parent, &backwardSlice, bsOptionsHoistHerd);
          backwardSlice.insert(parent);
        }
      } else {
        // Add scf.for op, and any associate constant operands, to transitive
        // defs.
        if (auto parent_for = dyn_cast<scf::ForOp>(op->getParentOp())) {
          backwardSlice.insert(parent_for);
          for (auto oper : parent_for->getOperands())
            if (getConstantIntValue(oper))
              backwardSlice.insert(oper.getDefiningOp());
        }
      }

      for (auto b : backwardSlice) {
        auto execOp = dyn_cast<air::ExecuteOp>(b);
        if (!execOp)
          continue;
        for (auto &childOp : execOp.getChildOps()) {
          (void)getBackwardSlice(&childOp, &backwardSlice, bsOptions);
          backwardSlice.insert(&childOp);
        }
      }
      // Get constant values used by backward slices, and add to backward
      // slices.
      for (auto o : backwardSlice) {
        for (auto &region : o->getRegions()) {
          visitUsedValuesDefinedAbove(region, [&backwardSlice](OpOperand *use) {
            if (getConstantIntValue(use->get())) {
              backwardSlice.insert(use->get().getDefiningOp());
            }
          });
        }
      }

      for (auto b : backwardSlice) {
        b->setAttr("hoist", StringAttr::get(ctx, "dep"));
      }
      op->setAttr("hoist", StringAttr::get(op->getContext(), "dep"));
      op->setAttr("loop-carried-dep",
                  StringAttr::get(op->getContext(), "external"));

      // Hoist hierarchy op into scf op
      scf::ParallelOp scf_par = nullptr;
      rewriter.setInsertionPoint(hier_op);
      if (herd && hoist_herd) {
        scf_par = hoistAIRHierToScfParallel(
            rewriter, loc, ctx, herd,
            SmallVector<Operation *>{op.getOperation()});
      } else if (segment) {
        // Since segment doesn't have iteration space, it doesn't hoist a loop
      }

      if (herd) {
        // Get mapping for remapped ssa values entering the hoisted scf.parallel
        IRMapping remap;
        auto herd_size = herd.getSizeOperands();
        remap.map(herd.getSize()[0], herd_size[0]);
        remap.map(herd.getSize()[1], herd_size[1]);
        if (scf_par) {
          remap.map(herd.getIds()[0], scf_par.getInductionVars()[0]);
          remap.map(herd.getIds()[1], scf_par.getInductionVars()[1]);
        }
        if (isa<scf::ForOp>(op->getParentOp()) && !hoist_herd) {
          // Dangling incoming dependency edge to hoisted scf.for.
          auto for_op = dyn_cast<scf::ForOp>(op->getParentOp());
          for (auto init_arg : for_op.getInitArgs())
            remap.map(init_arg,
                      rewriter
                          .create<air::WaitAllOp>(
                              loc, air::AsyncTokenType::get(op->getContext()),
                              SmallVector<Value>{})
                          .getAsyncToken());
        }
        int arg_idx = 0;
        for (auto arg : herd.getKernelArguments())
          remap.map(arg, herd.getKernelOperand(arg_idx++));

        // Clone ops into hoisted scf.parallel
        if (scf_par)
          rewriter.setInsertionPointToStart(scf_par.getBody());
        (void)air::cloneOpsInBlock(&herd.getBody().front(), rewriter, remap);
      } else if (segment) {
        // This shouldn't ever need to happen, because there's no where to
        // demote dma to
      } else
        return failure();

      if (scf_par) {
        scf_par->walk([&](mlir::Operation *o) {
          if (o == o->getBlock()->getTerminator()) {
            return;
          }
          if (!o->hasAttr("hoist"))
            erased.insert(o);
          else
            o->removeAttr("hoist");
        });
      }
      hier_op.walk([&](mlir::Operation *o) {
        if (o->hasAttr("hoist"))
          o->removeAttr("hoist");
      });
    }
    erased.insert(op);
    if (isAsyncOp(op)) {
      rewriter.setInsertionPoint(op);
      op.getAsyncToken().replaceAllUsesWith(
          rewriter
              .create<air::WaitAllOp>(
                  loc, air::AsyncTokenType::get(op->getContext()),
                  op.getAsyncDependencies())
              .getAsyncToken());
    }

    for (auto e : erased) {
      rewriter.eraseOp(e);
    }

    return success();
  }
};
struct DmaToChannelPass : public air::impl::DmaToChannelBase<DmaToChannelPass> {

  DmaToChannelPass() = default;
  DmaToChannelPass(const DmaToChannelPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });

    // Demote memref alloc pattern
    std::map<air::HierarchyInterface, std::vector<Operation *>> hier_to_allocs;
    for (auto f : funcOps) {
      f.walk([&](memref::AllocOp alloc) {
        auto memref_type =
            dyn_cast<BaseMemRefType>(alloc.getMemref().getType());
        int hierMemorySpace = (int)air::MemorySpace::L3;
        air::HierarchyInterface hier_op =
            alloc->getParentOfType<air::HierarchyInterface>();
        if (hier_op && isa<air::HerdOp>(hier_op.getOperation()))
          hierMemorySpace = (int)air::MemorySpace::L1;
        else if (hier_op && isa<air::SegmentOp>(hier_op.getOperation()))
          hierMemorySpace = (int)air::MemorySpace::L2;
        else
          return;
        // If async, then log the execute op around alloc
        Operation *alloc_op =
            alloc->getParentOfType<air::ExecuteOp>()
                ? alloc->getParentOfType<air::ExecuteOp>().getOperation()
                : alloc.getOperation();
        if (memref_type.getMemorySpaceAsInt() < (unsigned)hierMemorySpace) {
          hier_to_allocs[hier_op].push_back(alloc_op);
        }
      });
    }
    for (auto pair : hier_to_allocs) {
      OpBuilder builder(pair.first);
      (void)AIRDemoteMemrefToAIRHierarchy(pair, builder);
    }

    // First pattern to demote dma ops to corresponding air hierarchy
    ConversionTarget target_0(*context);

    target_0.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                             scf::SCFDialect, affine::AffineDialect,
                             air::airDialect, arith::ArithDialect,
                             memref::MemRefDialect, linalg::LinalgDialect>();

    target_0.addDynamicallyLegalOp<air::DmaMemcpyNdOp>(
        [&](air::DmaMemcpyNdOp dma) {
          auto src_type =
              llvm::dyn_cast<BaseMemRefType>(dma.getSrcMemref().getType());
          auto dst_type =
              llvm::dyn_cast<BaseMemRefType>(dma.getDstMemref().getType());
          if (dma->getParentOfType<air::HerdOp>()) {
            if (src_type.getMemorySpaceAsInt() < (int)air::MemorySpace::L1 &&
                dst_type.getMemorySpaceAsInt() < (int)air::MemorySpace::L1)
              return false;
          }
          return true;
        });

    RewritePatternSet air_dma_demotion(context);
    air_dma_demotion.add<AIRDemoteDmaToAIRHierarchyConversion>(context);
    if (failed(applyPartialConversion(module, target_0,
                                      std::move(air_dma_demotion)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
    }

    // Dep tracing
    for (auto f : funcOps) {
      updateDependencyOnFunction(f);
    }

    // Clear dma attributes
    for (auto f : funcOps) {
      f.walk([&](Operation *op) {
        op->removeAttr("loop-carried-dep");
        op->removeAttr("hoist");
      });
    }

    // Second pattern to convert dma into channels
    ConversionTarget target_1(*context);

    target_1.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                             scf::SCFDialect, affine::AffineDialect,
                             air::airDialect, arith::ArithDialect,
                             memref::MemRefDialect, linalg::LinalgDialect>();

    target_1.addIllegalOp<air::DmaMemcpyNdOp>();

    RewritePatternSet air_dma_conversion(context);
    air_dma_conversion.add<AIRDmaToAIRChannelConversion>(context);
    if (failed(applyPartialConversion(module, target_1,
                                      std::move(air_dma_conversion)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
    }

    // Hoist every "external" side channel to their respective air hierarchy.
    // For each channel op, hoist greedily until it reaches its corresponding
    // memory hierarchy.
    SetVector<air::ChannelInterface> externalChannelOps;
    for (auto f : funcOps) {
      f.walk([&externalChannelOps](air::HerdOp herd) {
        herd.walk([&externalChannelOps](air::ChannelInterface getput) {
          if (isValidExternalChannelOp(getput))
            externalChannelOps.insert(getput);
        });
      });
      f.walk([&externalChannelOps](air::SegmentOp segment) {
        segment.walk([&externalChannelOps](air::ChannelInterface getput) {
          if (isValidExternalChannelOp(getput))
            externalChannelOps.insert(getput);
        });
      });
    }

    for (auto getput : externalChannelOps) {
      getput->setAttr("loop-carried-dep",
                      StringAttr::get(context, "internalGetPut"));
    }

    for (auto getput : externalChannelOps) {
      getput->setAttr("loop-carried-dep", StringAttr::get(context, "external"));
      RewritePatternSet hoistChannelPatterns(context);
      hoistChannelPatterns
          .add<AIRHoistExternalAIRChannelPattern<air::HerdOp>,
               AIRHoistExternalAIRChannelPattern<air::SegmentOp>>(context);
      (void)applyPatternsGreedily(module, std::move(hoistChannelPatterns));
    }

    // Dep tracing
    for (auto f : funcOps) {
      updateDependencyOnFunction(f);
    }

    // Clear channel attributes
    for (auto f : funcOps) {
      f.walk([&](Operation *op) {
        op->removeAttr("loop-carried-dep");
        op->removeAttr("hoist");
      });
    }
  }

  void updateDependencyOnFunction(func::FuncOp f) {
    air::dependencyTracer depTracer;
    f.walk<WalkOrder::PreOrder,
           ForwardDominanceIterator<>>([&](air::MemcpyInterface memcpy_op) {
      if (!memcpy_op->hasAttr("loop-carried-dep"))
        return WalkResult::advance();
      auto LoopCarriedDepAttr =
          memcpy_op->getAttrOfType<StringAttr>("loop-carried-dep");
      if (LoopCarriedDepAttr.str() != "external" &&
          LoopCarriedDepAttr.str() != "internalGetPut")
        return WalkResult::advance();

      // Start tracing dependency only if this put/get op is async
      auto async_op = dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation());
      if (!async_op)
        return WalkResult::advance();

      // Connect async dependency of external put/get scf parallel
      SmallVector<air::partialMemref, 1> sink_op_memref_reads;
      SmallVector<air::partialMemref, 1> sink_op_memref_writes;
      SmallVector<Value, 1> sink_op_scalar_ins;
      SmallVector<Value, 1> sink_op_scalar_outs;

      air::WaitAllOp sink_wait_all_op = nullptr;
      for (auto parent = memcpy_op->getParentOp(); !isa<func::FuncOp>(parent);
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
            if (scf_for.getInitArgs().size() &&
                scf_for.getInitArgs()[0].getDefiningOp()) {
              sink_wait_all_op = dyn_cast<air::WaitAllOp>(
                  scf_for.getInitArgs()[0].getDefiningOp());
            }
          }
        }
      }

      depTracer.getPartialMemrefFromOp(
          memcpy_op.getOperation(), sink_op_memref_reads, sink_op_memref_writes,
          sink_op_scalar_ins, sink_op_scalar_outs);

      if (sink_op_memref_reads.empty() && sink_op_memref_writes.empty()) {
        memcpy_op->emitOpError("cannot read memref from channel op.");
        return WalkResult::skip();
      }

      if (sink_wait_all_op) {
        // Detect RAW deps
        if (failed(depTracer.template traceDependencyFromOp<air::WaitAllOp>(
                sink_op_memref_reads, sink_wait_all_op, "RAW")))
          signalPassFailure();
        // Detect WAW and WAR deps
        if (failed(depTracer.template traceDependencyFromOp<air::WaitAllOp>(
                sink_op_memref_writes, sink_wait_all_op, "WAW/WAR")))
          signalPassFailure();

        // Rebuild loop-carried dependency in scf loop nest
        air::clearAsyncDependenciesOfAsyncOp(memcpy_op);
        depTracer.reconnectLoopCarriedDependencyFromOp(
            memcpy_op.getOperation());
      }

      // Trace dependency of external put/get within scf loop
      if (failed(
              depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
                  sink_op_memref_reads,
                  dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation()),
                  "RAW")))
        signalPassFailure();
      if (failed(
              depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
                  sink_op_memref_writes,
                  dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation()),
                  "WAW/WAR")))
        signalPassFailure();
      // Detect tile index deps
      depTracer.traceTileIndices(
          sink_op_memref_reads, sink_op_memref_writes, sink_op_scalar_ins,
          sink_op_scalar_outs,
          dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation()));
      return WalkResult::advance();
    });
  }
};

} // namespace air
} // namespace xilinx

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createDmaToChannelPass() {
  return std::make_unique<DmaToChannelPass>();
}

} // namespace air
} // namespace xilinx
