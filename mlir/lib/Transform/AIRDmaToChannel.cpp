//===- AIRDmaToChannel.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRDmaToChannel.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
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

  auto wait_all_op_yielded = air::WaitAllOp::create(
      builder, scf_par.getLoc(), air::AsyncTokenType::get(ctx),
      SmallVector<Value, 1>{});
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
  return air::WaitAllOp::create(builder, builder.getUnknownLoc(),
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
      scf::YieldOp::create(builder, builder.getUnknownLoc(), yield_token);
  return output;
}

// Materialize the PURE defs a labelled op depends on but which were not
// themselves labelled.
//
// cloneOpsInBlock skips an unlabelled non-async op outright: nothing clones it
// and nothing maps it. A later clone of a labelled CONSUMER then resolves that
// operand through lookupOrDefault and keeps the ORIGINAL value -- which still
// lives in the block being left behind. The result is an op at the outer scope
// referring to a def at the inner one, and the failure surfaces far away as
// "operand #0 does not dominate this use".
//
// The shape that hits this in practice is a rebuilt arm guard:
// `scf.index_switch` is labelled (it carries the async token the hoisted
// transfer depends on) while the `arith.index_cast` feeding its condition is
// not (it produces no token, so no dependence edge reaches it).
//
// A pure op is free to duplicate, so clone it on demand rather than drop the
// dependence. Ops with regions are excluded: cloning one would duplicate a
// whole nest, and a region-carrying op on this path is always labelled anyway.
static void materializePureDefs(Operation *o, Block *blk, OpBuilder &builder,
                                IRMapping &remap,
                                SmallVector<Operation *> &clonedOps) {
  SmallVector<Value> worklist;
  auto collect = [&](Operation *op) {
    for (auto v : op->getOperands())
      worklist.push_back(v);
    for (auto &r : op->getRegions()) {
      SetVector<Value> above;
      getUsedValuesDefinedAbove(r, above);
      for (auto v : above)
        worklist.push_back(v);
    }
  };
  collect(o);
  SmallVector<Operation *> toClone;
  llvm::SmallDenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (remap.contains(v))
      continue;
    auto *d = v.getDefiningOp();
    if (!d || d->getBlock() != blk)
      continue;
    // Labelled ops are the main loop's business.
    if (d->hasAttr("hoist"))
      continue;
    if (d->getNumRegions() || !air::isPure(d))
      continue;
    if (!seen.insert(d).second)
      continue;
    toClone.push_back(d);
    collect(d);
  }
  // Clone defs before uses.
  llvm::sort(toClone,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (auto *d : toClone) {
    auto *c = builder.clone(*d, remap);
    // Label it: the caller's cleanup erases every unlabelled op in the newly
    // built outer loop, and this clone has uses.
    c->setAttr("hoist", StringAttr::get(c->getContext(), "dep"));
    clonedOps.push_back(c);
  }
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
    materializePureDefs(&o, blk, builder, remap, clonedOps);
    if (auto child_for_op = dyn_cast_if_present<LoopLikeOpInterface>(o)) {
      auto clonedScfLoopOps =
          air::cloneScfLoopUsingRemap(builder, remap, child_for_op);
      clonedOps.insert(clonedOps.end(), clonedScfLoopOps.begin(),
                       clonedScfLoopOps.end());
    } else if (auto channel_op =
                   dyn_cast_if_present<air::ChannelInterface>(o)) {
      auto depAttr = o.getAttrOfType<StringAttr>("loop-carried-dep");
      bool isInternalGetPut =
          depAttr && depAttr.getValue().str() == "internalGetPut";
      // A user-written channel op pulled into the backward slice as a
      // transitive dependency of a DMA being hoisted. It has "hoist"
      // (guaranteed true here — line 119 filters non-hoist ops) but no
      // "loop-carried-dep" (only DMA-derived channel ops receive that
      // attribute; see labelBackwardSlice below). Must not be cloned to
      // segment level — replace with wait_all to preserve async token chain.
      bool isUserWrittenChannel = !depAttr;
      if (isInternalGetPut || isUserWrittenChannel) {
        if (air::isAsyncOp(&o)) {
          auto wa_op =
              air::replaceAsyncOpWithWaitAll(builder, remap, &o, false);
          wa_op->setAttr("hoist", StringAttr::get(o.getContext(), "dep"));
          clonedOps.push_back(wa_op);
        }
      } else {
        clonedOps.push_back(builder.clone(o, remap));
      }
    } else if (auto aif_op = dyn_cast_if_present<affine::AffineIfOp>(o)) {
      auto clonedAifOps = air::cloneAffineIfUsingRemap(builder, remap, aif_op);
      clonedOps.insert(clonedOps.end(), clonedAifOps.begin(),
                       clonedAifOps.end());
    } else if (auto scf_if_op = dyn_cast_if_present<scf::IfOp>(o)) {
      auto clonedScfIfOps =
          air::cloneScfIfUsingRemap(builder, remap, scf_if_op);
      clonedOps.insert(clonedOps.end(), clonedScfIfOps.begin(),
                       clonedScfIfOps.end());
    } else if (auto switch_op = dyn_cast_if_present<scf::IndexSwitchOp>(o)) {
      auto clonedSwitchOps =
          air::cloneIndexSwitchUsingRemap(builder, remap, switch_op);
      clonedOps.insert(clonedOps.end(), clonedSwitchOps.begin(),
                       clonedSwitchOps.end());
    } else if (auto dma_op = dyn_cast_if_present<air::DmaMemcpyNdOp>(o)) {
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
  // Clone the affine if op body instead of the if op, flattening both
  // then/else blocks into the parent scope.
  SmallVector<Operation *> clonedOps;
  auto clonedThenOps = cloneOpsInBlock(aif_op.getThenBlock(), builder, remap);
  clonedOps.insert(clonedOps.end(), clonedThenOps.begin(), clonedThenOps.end());
  if (aif_op.hasElse()) {
    auto clonedElseOps = cloneOpsInBlock(aif_op.getElseBlock(), builder, remap);
    clonedOps.insert(clonedOps.end(), clonedElseOps.begin(),
                     clonedElseOps.end());
  }

  // When the affine.if has results (e.g., async tokens from
  // air-specialize-dma-broadcast), map them to replacement values so
  // downstream uses don't become orphaned (SSA dominance fix for #1484).
  if (aif_op.getNumResults() > 0) {
    // Check whether any results are async tokens.
    bool hasAsyncTokenResult = false;
    for (Value res : aif_op.getResults()) {
      if (isa<air::AsyncTokenType>(res.getType())) {
        hasAsyncTokenResult = true;
        break;
      }
    }

    // Fallback mapping source for all non-token results (and for token
    // results if we fail to build a wait_all): the then-block's yielded
    // values, remapped through the IRMapping.
    auto thenYield = aif_op.getThenBlock()->getTerminator();

    air::WaitAllOp waitAllOp;
    if (hasAsyncTokenResult) {
      // Collect async tokens produced by cloned ops to create a wait_all.
      SmallVector<Value> asyncDeps;
      // Only ops landing directly in the destination block can be waited on
      // here. cloneOpsInBlock returns what it cloned at EVERY depth -- that is
      // how the caller finds channel ops buried in a hoisted loop body -- so if
      // a branch held a loop, its interior tokens are in `clonedOps` too, and
      // waiting on one produces "operand #N does not dominate this use".
      Block *destBlk = builder.getInsertionBlock();
      for (auto *clonedOp : clonedOps) {
        if (clonedOp->getBlock() != destBlk)
          continue;
        if (auto asyncOp =
                dyn_cast_if_present<air::AsyncOpInterface>(clonedOp)) {
          if (auto token = asyncOp.getAsyncToken())
            asyncDeps.push_back(token);
        }
      }
      // Create a wait_all that merges all cloned ops' async tokens, which
      // can be used to replace async-token results of the affine.if.
      if (!asyncDeps.empty()) {
        waitAllOp = air::WaitAllOp::create(
            builder, aif_op.getLoc(),
            air::AsyncTokenType::get(aif_op->getContext()), asyncDeps);
        waitAllOp->setAttr("hoist",
                           StringAttr::get(aif_op->getContext(), "dep"));
        clonedOps.push_back(waitAllOp);
      }
    }

    // Map each affine.if result:
    //  - async-token results: to the wait_all token if it exists, otherwise
    //    to the corresponding remapped yielded value;
    //  - non-token results: always to the corresponding remapped yielded
    //    value.
    for (unsigned i = 0; i < aif_op.getNumResults(); i++) {
      Value ifResult = aif_op.getResult(i);
      Value mappedValue;
      if (isa<air::AsyncTokenType>(ifResult.getType()) && waitAllOp) {
        mappedValue = waitAllOp.getAsyncToken();
      } else {
        Value yieldedVal = thenYield->getOperand(i);
        mappedValue = remap.lookupOrDefault(yieldedVal);
      }
      remap.map(ifResult, mappedValue);
    }
  }

  return clonedOps;
}

SmallVector<Operation *> air::cloneScfIfUsingRemap(OpBuilder builder,
                                                   IRMapping &remap,
                                                   scf::IfOp scf_if_op) {
  // Clone scf.if preserving the if structure with remapped condition.
  // Only supports scf.if with no results (the expected pattern for hoisting
  // external channel ops). Fall back to flattening if results are present.
  SmallVector<Operation *> clonedOps;
  if (scf_if_op.getNumResults() != 0) {
    // Flatten: clone body ops without the scf.if wrapper.
    auto clonedThenOps = cloneOpsInBlock(scf_if_op.thenBlock(), builder, remap);
    clonedOps.insert(clonedOps.end(), clonedThenOps.begin(),
                     clonedThenOps.end());
    if (scf_if_op.elseBlock()) {
      auto clonedElseOps =
          cloneOpsInBlock(scf_if_op.elseBlock(), builder, remap);
      clonedOps.insert(clonedOps.end(), clonedElseOps.begin(),
                       clonedElseOps.end());
    }

    // Map scf.if results to replacement values so downstream uses don't
    // become orphaned (same pattern as cloneAffineIfUsingRemap).
    bool hasAsyncTokenResult = false;
    for (Value res : scf_if_op.getResults()) {
      if (isa<air::AsyncTokenType>(res.getType())) {
        hasAsyncTokenResult = true;
        break;
      }
    }

    auto thenYield = scf_if_op.thenBlock()->getTerminator();

    air::WaitAllOp waitAllOp;
    if (hasAsyncTokenResult) {
      SmallVector<Value> asyncDeps;
      // Only ops landing directly in the destination block can be waited on
      // here. cloneOpsInBlock returns what it cloned at EVERY depth -- that is
      // how the caller finds channel ops buried in a hoisted loop body -- so if
      // a branch held a loop, its interior tokens are in `clonedOps` too, and
      // waiting on one produces "operand #N does not dominate this use".
      Block *destBlk = builder.getInsertionBlock();
      for (auto *clonedOp : clonedOps) {
        if (clonedOp->getBlock() != destBlk)
          continue;
        if (auto asyncOp =
                dyn_cast_if_present<air::AsyncOpInterface>(clonedOp)) {
          if (auto token = asyncOp.getAsyncToken())
            asyncDeps.push_back(token);
        }
      }
      if (!asyncDeps.empty()) {
        waitAllOp = air::WaitAllOp::create(
            builder, scf_if_op.getLoc(),
            air::AsyncTokenType::get(scf_if_op->getContext()), asyncDeps);
        waitAllOp->setAttr("hoist",
                           StringAttr::get(scf_if_op->getContext(), "dep"));
        clonedOps.push_back(waitAllOp);
      }
    }

    for (unsigned i = 0; i < scf_if_op.getNumResults(); i++) {
      Value ifResult = scf_if_op.getResult(i);
      Value mappedValue;
      if (isa<air::AsyncTokenType>(ifResult.getType()) && waitAllOp) {
        mappedValue = waitAllOp.getAsyncToken();
      } else {
        Value yieldedVal = thenYield->getOperand(i);
        mappedValue = remap.lookupOrDefault(yieldedVal);
      }
      remap.map(ifResult, mappedValue);
    }

    return clonedOps;
  }

  // Remap the condition value.
  Value cond = remap.lookupOrDefault(scf_if_op.getCondition());

  // Create a new scf.if with no results (external channel ops are async and
  // don't return values through the scf.if).
  bool hasElse = (scf_if_op.elseBlock() != nullptr);
  auto newIfOp =
      scf::IfOp::create(builder, scf_if_op.getLoc(), /*resultTypes=*/{}, cond,
                        /*withElseRegion=*/hasElse);
  // Mark the new scf.if with "hoist" to prevent it from being erased during
  // the cleanup step that removes non-hoisted ops from the hoisted
  // scf.parallel.
  newIfOp->setAttr("hoist", StringAttr::get(builder.getContext(), "dep"));

  // Clone ops in the then block. Insert before the existing yield terminator.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(newIfOp.thenBlock()->getTerminator());
    auto clonedThenOps = cloneOpsInBlock(scf_if_op.thenBlock(), builder, remap);
    // Collect channel ops from the then block.
    for (auto *op : clonedThenOps) {
      if (isa<air::ChannelInterface>(op))
        clonedOps.push_back(op);
    }
  }

  // Clone ops in the else block if present.
  if (hasElse) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(newIfOp.elseBlock()->getTerminator());
    auto clonedElseOps = cloneOpsInBlock(scf_if_op.elseBlock(), builder, remap);
    for (auto *op : clonedElseOps) {
      if (isa<air::ChannelInterface>(op))
        clonedOps.push_back(op);
    }
  }

  return clonedOps;
}

// Clone an scf.index_switch, preserving the switch so a hoisted external
// channel op stays on the arm it was written for.
//
// Without this the op is simply DROPPED: cloneOpsInBlock has no case for
// scf.index_switch, so the external half of a DMA written inside a switch arm
// never reaches the parent scope and its partner get is left unpaired. That
// does not fail here -- it fails 30-odd passes later in
// air-verify-hierarchy-locality with "found channel op not in pairs", pointing
// at the surviving half rather than at the arm the other one was lost from.
//
// Only the no-results form is rebuilt, which is the shape external channel ops
// take (they are async and do not yield through the switch). A switch WITH
// results falls back to flattening, matching cloneScfIfUsingRemap: the arms
// stop being mutually exclusive, which is wrong in general, so it is reported
// rather than done silently.
SmallVector<Operation *>
air::cloneIndexSwitchUsingRemap(OpBuilder builder, IRMapping &remap,
                                scf::IndexSwitchOp switch_op) {
  SmallVector<Operation *> clonedOps;
  auto loc = switch_op.getLoc();
  auto *ctx = switch_op->getContext();

  // Async token results are the norm here, not an edge case: air-dependency
  // runs before air-dma-to-channel in aircc, so by the time a switch reaches
  // this pass every arm yields a token. Flattening it -- the fallback
  // cloneScfIfUsingRemap takes for scf.if -- would make a copy written on one
  // arm issue on EVERY arm, so rebuild the switch instead and give each arm a
  // token of its own.
  bool hasToken = false;
  for (Value res : switch_op.getResults())
    if (isa<air::AsyncTokenType>(res.getType()))
      hasToken = true;
  if (switch_op.getNumResults() > (hasToken ? 1u : 0u)) {
    switch_op->emitWarning(
        "hoisting an external channel op out of an scf.index_switch yielding "
        "non-token results; the arms are flattened, so a copy written on one "
        "arm will issue on every arm");
    for (Region &region : switch_op->getRegions()) {
      if (region.empty())
        continue;
      auto cloned = cloneOpsInBlock(&region.front(), builder, remap);
      for (auto *op : cloned)
        if (isa<air::ChannelInterface>(op))
          clonedOps.push_back(op);
    }
    return clonedOps;
  }

  SmallVector<Type> resTys;
  if (hasToken)
    resTys.push_back(air::AsyncTokenType::get(ctx));
  Value arg = remap.lookupOrDefault(switch_op.getArg());
  auto newSwitch = scf::IndexSwitchOp::create(
      builder, loc, resTys, arg, switch_op.getCases(), switch_op.getNumCases());
  // Keep it through the cleanup that erases non-hoisted ops from the hoisted
  // scf.parallel, exactly as cloneScfIfUsingRemap does for its scf.if.
  newSwitch->setAttr("hoist", StringAttr::get(ctx, "dep"));

  auto cloneRegionInto = [&](Region &src, Region &dst) {
    if (dst.empty())
      dst.emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&dst.front());
    SmallVector<Operation *> cloned;
    if (!src.empty())
      cloned = cloneOpsInBlock(&src.front(), builder, remap);
    for (auto *op : cloned)
      if (isa<air::ChannelInterface>(op))
        clonedOps.push_back(op);
    if (!hasToken) {
      scf::YieldOp::create(builder, loc);
      return;
    }
    // Every arm must yield a token, including one that got nothing: an empty
    // air.wait_all is the identity the other arms' tokens are typed against.
    //
    // Collect the deps from the destination BLOCK, not from `cloned`.
    // cloneOpsInBlock returns the ops it cloned at EVERY depth -- that is how
    // the caller finds channel ops buried in a hoisted loop body (see the
    // clonedOps scan in AIRHoistExternalAIRChannelPattern) -- so an arm holding
    // a loop would otherwise yield tokens defined inside that loop's region and
    // fail verification with "operand #N does not dominate this use". One
    // switch is enough to hit this; it needs no second level of nesting.
    //
    // Taking only the tokens still unused in this block yields exactly what the
    // arm has left outstanding: a nested loop contributes its own result, and
    // the ops inside it stay where they belong.
    SmallVector<Value> deps;
    for (Operation &op : dst.front())
      for (Value res : op.getResults())
        if (isa<air::AsyncTokenType>(res.getType()) && res.use_empty())
          deps.push_back(res);
    auto wa = air::WaitAllOp::create(builder, loc,
                                     air::AsyncTokenType::get(ctx), deps);
    wa->setAttr("hoist", StringAttr::get(ctx, "dep"));
    scf::YieldOp::create(builder, loc, SmallVector<Value>{wa.getAsyncToken()});
  };

  cloneRegionInto(switch_op.getDefaultRegion(), newSwitch.getDefaultRegion());
  for (unsigned i = 0; i < switch_op.getNumCases(); i++)
    cloneRegionInto(switch_op.getCaseRegions()[i],
                    newSwitch.getCaseRegions()[i]);

  if (hasToken)
    for (Value res : switch_op.getResults())
      if (isa<air::AsyncTokenType>(res.getType()))
        remap.map(res, newSwitch.getResult(0));

  return clonedOps;
}

template <typename T>
SmallVector<Operation *>
air::cloneScfLoopUsingRemap(OpBuilder builder, IRMapping &remap, T loop_op,
                            air::ChannelInterface externalGetPut) {
  SmallVector<Value> loop_init_args = air::getAsyncDependenciesFromOp(loop_op);
  T new_loop_op =
      T::create(builder, builder.getUnknownLoc(),
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
  if (scf::ForOp fop = dyn_cast_if_present<scf::ForOp>(op)) {
    return cloneScfLoopUsingRemap(builder, remap, fop, externalGetPut);
  } else if (scf::ParallelOp pop = dyn_cast_if_present<scf::ParallelOp>(op)) {
    return cloneScfLoopUsingRemap(builder, remap, pop, externalGetPut);
  }
  loop_op.emitOpError("unsupported loop type");
  return SmallVector<Operation *>();
}

static scf::ParallelOp
hoistAIRHierToScfParallel(OpBuilder builder, Location loc, MLIRContext *ctx,
                          air::HierarchyInterface hierOp,
                          SmallVector<Operation *> targetOpsToHoist) {

  auto step = arith::ConstantIndexOp::create(builder, loc, 1);
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
    lbs.push_back(arith::ConstantIndexOp::create(builder, loc, lbs_int));
    ubs.push_back(arith::ConstantIndexOp::create(builder, loc, ubs_int + 1));
    steps.push_back(step);
  }

  auto hierAsyncIfOp =
      dyn_cast_if_present<air::AsyncOpInterface>(hierOp.getOperation());

  auto wa_op =
      air::WaitAllOp::create(builder, loc, air::AsyncTokenType::get(ctx),
                             hierAsyncIfOp.getAsyncDependencies());
  SmallVector<Value, 1> deps_in{wa_op.getAsyncToken()};
  scf::ParallelOp scf_par = nullptr;
  if (isAsyncOp(hierAsyncIfOp)) {
    scf_par = scf::ParallelOp::create(builder, loc, lbs, ubs, steps, deps_in);
    generateYieldAndOrReduceToScfLoop(builder, ctx, scf_par);
    hierAsyncIfOp.getAsyncToken().replaceAllUsesWith(
        air::getAsyncTokenFromOp(scf_par));
  } else
    scf_par = scf::ParallelOp::create(builder, loc, lbs, ubs, steps);

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

  auto channel_op = air::ChannelOp::create(
      builder, loc, cname, builder.getI64ArrayAttr(channel_bundle_sizes),
      builder.getStringAttr("npu_dma_stream"));

  builder.restoreInsertionPoint(insertionCheckpoint);

  return channel_op;
}

// Volume of an access pattern in elements, or nullopt if it is not static.
static std::optional<int64_t> staticVolume(ArrayRef<OpFoldResult> sizes,
                                           Value memref) {
  if (sizes.empty()) {
    auto ty = dyn_cast<MemRefType>(memref.getType());
    if (!ty || !ty.hasStaticShape())
      return std::nullopt;
    return (int64_t)air::getTensorVolume(ty);
  }
  int64_t v = 1;
  for (OpFoldResult s : sizes) {
    std::optional<int64_t> c = getConstantIntValue(s);
    if (!c)
      return std::nullopt;
    v *= *c;
  }
  return v;
}

// When the far buffer holds SEVERAL of the near window, the far half is that
// buffer tiled by it.
//
// Nothing states this and nothing needs to. A channel is a FIFO, and the near
// side's sequence is fixed by its own access pattern and its enclosing nest;
// the far buffer is named on the same op. If the far buffer is N near-windows
// wide then the near side takes it N pieces at a time, in order, and the far
// half is what produces exactly that: N windows of the near size, ascending.
//
// N SEPARATE transfers, not one N-wide descriptor. Matching the byte sequence
// is necessary and NOT sufficient: a channel is a FIFO of TRANSFERS, and each
// one is a lock event, so N gets expect N puts. A single 2-D descriptor
// [base,0] [N,near] [near,1] sends the same bytes in the same order as N puts
// of [base+i*near] [near] [1] and raises the semaphore once instead of N times.
// The consumer's Nth acquire then never arrives and the design hangs -- and it
// hangs at whatever dispatch the count first drifts far enough, not at the
// first, so it reads like an unrelated intermittent fault.
//
// This rewrites `offsets`/`sizes`/`strides` to piece 0 and returns N; the
// caller emits the remaining N-1. Returns 0 when the tiling is declined.
//
// Deriving it is what lets the front end stop hand-writing the far half's
// offsets -- which it often CANNOT write, because they step with a loop that
// does not exist where the transfer is spelled.
//
// Only a 1-D far window is folded, and only an exact multiple. Anything else is
// left alone: a partial tiling has no unique reading, and guessing one is how a
// silent misroute gets built.
static int64_t deriveTiledFarWindow(OpBuilder &b, Value farMemref,
                                    SmallVector<OpFoldResult> &offsets,
                                    SmallVector<OpFoldResult> &sizes,
                                    SmallVector<OpFoldResult> &strides,
                                    int64_t nearVolume) {
  if (nearVolume <= 0)
    return 0;
  if (sizes.size() > 1)
    return 0;
  std::optional<int64_t> farVol = staticVolume(sizes, farMemref);
  if (!farVol || *farVol <= nearVolume)
    return 0;
  if (*farVol % nearVolume)
    return 0;
  int64_t n = *farVol / nearVolume;
  // A unit stride is what makes "the buffer holds N windows back to back" true;
  // anything else is already a shaped access and is left as written.
  if (!strides.empty()) {
    std::optional<int64_t> st = getConstantIntValue(strides[0]);
    if (!st || *st != 1)
      return 0;
  }
  OpFoldResult base =
      offsets.empty() ? OpFoldResult(b.getIndexAttr(0)) : offsets[0];
  if (!getConstantIntValue(base))
    return 0;
  offsets.assign({base});
  sizes.assign({b.getIndexAttr(nearVolume)});
  strides.assign({b.getIndexAttr(1)});
  return n;
}

// Where the far buffer is FILLED, outside the hierarchy being hoisted out of.
//
// A far half carrying a whole buffer's worth belongs once per fill, not once
// per near execution, so its position is not a free choice and does not need to
// be named: it is wherever something writes the buffer it reads. Finding that
// site by the BUFFER rather than by a channel symbol is what lets it work when
// the loop it must land in holds no channel endpoint of its own -- which is the
// ordinary case for a feed whose inner loop exists only to step the window.
//
// Returns the last writer PER REGION (the fill completes before the forward),
// not one writer overall.
//
// One buffer is commonly filled in several mutually exclusive places -- a vocab
// arm and a decode arm of the same scf.index_switch, each with its own feed
// loop. Those are not competing candidates to choose between; the derived half
// belongs in every one of them, exactly as an anchored hoist replicates across
// arms. Returning a single site silently starves the arms that did not win:
// their consumers wait on a producer emitted somewhere they never execute, and
// the design hangs with no missing-endpoint diagnostic anywhere.
//
// Grouping by REGION rather than by switch arm keeps it simple and is the same
// rule in the cases that matter: two arms are two regions, and a second fill in
// the same region really is a later fill of the same buffer, where last wins.
static void findFillSites(Value farMemref, Operation *hier_op,
                          SmallVectorImpl<Operation *> &out) {
  Value root = air::resolveBufferRoot(farMemref);
  llvm::MapVector<Region *, Operation *> lastPerRegion;
  hier_op->getParentRegion()->walk([&](Operation *o) {
    if (hier_op->isAncestor(o))
      return WalkResult::advance();
    if (o == hier_op)
      return WalkResult::advance();
    // A get lands bytes in it; that is a fill. A put reads it, and is not.
    if (auto g = dyn_cast<air::ChannelGetOp>(o))
      if (air::resolveBufferRoot(g.getMemref()) == root)
        lastPerRegion[o->getParentRegion()] = o;
    return WalkResult::advance();
  });
  for (auto &kv : lastPerRegion)
    out.push_back(kv.second);
}

// Whether anything outside the hierarchy refills the buffer at all.
static bool hasFillSite(Value farMemref, Operation *hier_op) {
  SmallVector<Operation *> sites;
  findFillSites(farMemref, hier_op, sites);
  return !sites.empty();
}

static LogicalResult replaceAIRDmaWithAIRChannelPairs(
    OpBuilder &builder, air::MemorySpace innerMemorySpace,
    air::DmaMemcpyNdOp op,
    SmallVector<air::ChannelInterface, 1> &internalGetPutVector,
    SmallVector<air::ChannelInterface, 1> &externalGetPutVector) {
  auto loc = op->getLoc();
  auto src = op.getSrcMemref();
  auto dst = op.getDstMemref();
  auto ctx = op->getContext();

  auto src_type = llvm::dyn_cast_if_present<BaseMemRefType>(src.getType());
  auto dst_type = llvm::dyn_cast_if_present<BaseMemRefType>(dst.getType());
  SmallVector<OpFoldResult> src_offsets = op.getMixedSrcOffsets();
  SmallVector<OpFoldResult> dst_offsets = op.getMixedDstOffsets();
  SmallVector<OpFoldResult> src_sizes = op.getMixedSrcSizes();
  SmallVector<OpFoldResult> dst_sizes = op.getMixedDstSizes();
  SmallVector<OpFoldResult> src_strides = op.getMixedSrcStrides();
  SmallVector<OpFoldResult> dst_strides = op.getMixedDstStrides();

  // Derive the far half's window from the near one, when the far buffer holds
  // several of it. See deriveTiledFarWindow: this is the descriptor that makes
  // the two byte sequences match, and the front end frequently cannot write it.
  // 0 = not derived. Otherwise the number of pieces the far buffer is tiled
  // into; the descriptor built below is piece 0 and the rest are emitted after.
  int64_t farPieces = 0;
  int64_t farPieceVolume = 0;
  auto enclosingHier = op->getParentOfType<air::HierarchyInterface>();
  if (dst_type && src_type && enclosingHier) {
    bool dstIsInner = air::getMemorySpace(dst_type) == innerMemorySpace;
    Value nearMemref = dstIsInner ? dst : src;
    Value farMemref = dstIsInner ? src : dst;
    std::optional<int64_t> nearVol =
        staticVolume(dstIsInner ? dst_sizes : src_sizes, nearMemref);
    // The tiling reads "the near side takes this buffer N pieces at a time".
    // Two independent ways to know that is a reading and not a guess, and one
    // of them has to hold.
    //
    // 1. COUNT. If the near side's trip count is static it must be exactly N.
    //    That is the invariant the tiling asserts, checked directly. It is also
    //    what catches a far buffer nothing writes, read once by a near side
    //    with no loop: N is 2 against a trip count of 1, and tiling it would
    //    silently send twice what was asked for.
    //
    // 2. REUSE. With a non-static trip count there is nothing to compare, so
    //    fall back to evidence that the buffer cycles -- a fill, then N takes,
    //    then the next fill. That same fill is where the derived half is then
    //    placed, so window and position rest on one fact rather than two
    //    guesses.
    //
    // Deliberately NOT "either alone suffices": a static trip count that
    // disagrees is a refutation, and no amount of reuse evidence overrides it.
    std::optional<int64_t> nearTrip = air::getStaticTripCountInRange(
        op.getOperation(), enclosingHier.getOperation());
    std::optional<int64_t> farVol =
        staticVolume(dstIsInner ? src_sizes : dst_sizes, farMemref);
    bool countAgrees = false, countRefutes = false;
    if (nearTrip && nearVol && farVol && *nearVol > 0 &&
        *farVol % *nearVol == 0) {
      int64_t n = *farVol / *nearVol;
      countAgrees = *nearTrip == n;
      countRefutes = *nearTrip != n;
    }
    bool refilled = hasFillSite(farMemref, enclosingHier.getOperation());
    if (nearVol && !countRefutes && (countAgrees || refilled)) {
      if (dstIsInner)
        farPieces = deriveTiledFarWindow(builder, src, src_offsets, src_sizes,
                                         src_strides, *nearVol);
      else
        farPieces = deriveTiledFarWindow(builder, dst, dst_offsets, dst_sizes,
                                         dst_strides, *nearVol);
      if (farPieces)
        farPieceVolume = *nearVol;
    }
  }

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

  // A DMA naming a channel lowers onto THAT declaration instead of a fresh one.
  // This is what lets several DMAs share one channel (a convergent
  // multi-producer feed), and what keeps author-written channel properties --
  // channel_type, broadcast_shape, air.shared_resident_ring,
  // air.tile_dma_channel -- attached to a symbol the front end controls.
  //
  // The declaration is never created here. A name with nothing behind it is a
  // typo, and minting an empty channel for it would convert that typo into a
  // silent point-to-point circuit flow that deadlocks much further downstream,
  // in air-to-aie, with no trace of where the name came from.
  air::ChannelOp namedChanOp = nullptr;
  if (op.hasNamedChannel()) {
    namedChanOp = air::getChannelDeclarationThroughSymbol(op.getOperation(),
                                                          op.getChannelAttr());
    if (!namedChanOp)
      return op.emitOpError()
             << "names channel " << op.getChannelAttr()
             << ", which is not declared in any enclosing symbol table";
  }
  std::string cname = namedChanOp ? namedChanOp.getSymName().str()
                                  : air::createChannelName(module);

  if (namedChanOp) {
    // The declaration already carries its bundle shape and every property the
    // front end wrote on it; re-deriving either would overwrite the reason for
    // naming it in the first place.
    //
    // A broadcast still works when the channel is named: the set is forwarded
    // to the put below, and the internal indices come from the affine.if that
    // guards it, exactly as for an unnamed channel. Only the SHAPE comes from
    // the declaration rather than being derived from the set. So say nothing
    // while the two agree, and report it when they do not -- a declaration
    // whose fan-out disagrees with the guard that implements it is a real bug,
    // and silently preferring either one hides it.
    if (auto setAttr =
            op->getAttrOfType<mlir::IntegerSetAttr>("broadcast_set")) {
      SmallVector<int, 2> lbs_int = {-1, -1};
      SmallVector<int, 2> ubs_int = {-1, -1};
      air::getSizesFromIntegerSet(ctx, setAttr.getValue(), lbs_int, ubs_int);
      SmallVector<int64_t, 2> fromSet = {ubs_int[0] - lbs_int[0] + 1,
                                         ubs_int[1] - lbs_int[1] + 1};
      SmallVector<int64_t, 2> declared;
      if (auto bs = namedChanOp.getBroadcastShape())
        for (auto d : bs)
          if (auto i = llvm::dyn_cast<IntegerAttr>(d))
            declared.push_back(i.getInt());
      if (declared != fromSet) {
        auto fmt = [](ArrayRef<int64_t> v) {
          std::string s;
          llvm::raw_string_ostream os(s);
          os << "[";
          llvm::interleaveComma(v, os);
          os << "]";
          return s;
        };
        std::string declStr = fmt(declared), setStr = fmt(fromSet);
        op->emitWarning()
            << "broadcast_set on this DMA implies a fan-out of " << setStr
            << ", but the channel it names, @" << namedChanOp.getSymName()
            << ", declares broadcast_shape " << declStr
            << ". The declaration wins; the guard is unchanged, so the two "
               "disagree on device.";
      }
    }
  } else if (op->hasAttr("broadcast_set")) {
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
  if (!op.getDynamicChannelIndices().empty()) {
    // Sub-channel selectors known only at run time -- a tile indexing its own
    // column, say. They win over the static form and over the spatial
    // inference: the front end said exactly which sub-channel this is, and it
    // is not a constant. Both halves index the same one.
    for (Value v : op.getDynamicChannelIndices()) {
      channel_idx_internal.push_back(v);
      channel_idx_external.push_back(v);
    }
  } else if (auto staticIndices = op.getChannelIndices()) {
    // An explicit index overrides the spatial inference below. It is what a
    // sub-channel of a bundle is selected with when the index is NOT the
    // enclosing spatial index -- e.g. a per-column weight feed inside a loop
    // whose IV is not the column. Both halves index the same sub-channel.
    for (int64_t idx : *staticIndices) {
      auto c = arith::ConstantIndexOp::create(builder, loc, idx);
      channel_idx_internal.push_back(c);
      channel_idx_external.push_back(c);
    }
  } else if (op->hasAttr("broadcast_set")) {
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

  // For a named channel the DECLARATION fixes the bundle rank, and the spatial
  // inference above knows nothing about it -- it counts enclosing herd /
  // scf.parallel dimensions. Those agree only when the bundle *is* the spatial
  // iteration. When they disagree, indexing a rank-N bundle with M != N indices
  // is malformed IR that survives all the way to air-to-aie, so resolve it
  // here.
  if (namedChanOp && !op.getChannelIndices()) {
    size_t declRank = namedChanOp.getSize().size();
    if (declRank == 0) {
      // Unbundled: a single flow, addressed with no index.
      channel_idx_internal.clear();
      channel_idx_external.clear();
    } else if (declRank != channel_idx_internal.size()) {
      return op.emitOpError()
             << "names channel @" << namedChanOp.getSymName() << ", declared "
             << "with " << declRank << " bundle dimension(s), but the "
             << "enclosing spatial iteration supplies "
             << channel_idx_internal.size()
             << ". Give the sub-channel explicitly with channel_indices.";
    }
  }

  // On a BROADCAST channel the index identifies the RECEIVER, so it belongs to
  // the consuming half only. There is one source and it addresses no
  // sub-channel: that is what makes it a broadcast rather than a bundle.
  //
  // The same rule is already applied on the broadcast_set path, where only the
  // internal half inherits the affine.if's operands. It has to hold here too,
  // for a channel whose fan-out comes from its DECLARATION instead of a guard.
  // Without it a DMA written on the core side hands the producer the consumer's
  // tile indices -- herd induction variables, which do not exist where the
  // producer lands, so the hoist emits IR that does not verify:
  //
  //   error: operand #0 does not dominate this use
  //
  // Whether the external half is the source is not assumed: a broadcast's
  // external half is the put by construction, and clearing indices off a get
  // would silently redirect a consumer to sub-channel 0.
  if (namedChanOp && namedChanOp.getBroadcastShape() &&
      air::getMemorySpace(src_type) != innerMemorySpace)
    channel_idx_external.clear();

  // Extract padding attributes from the DMA op (applies to source/put side).
  DenseI32ArrayAttr padBefore = op.getPadBeforeAttr();
  DenseI32ArrayAttr padAfter = op.getPadAfterAttr();

  // Create channel put-get pair
  SmallVector<Type, 4> tys;
  if (auto op_token = op.getAsyncToken()) {
    tys.push_back(air::AsyncTokenType::get(ctx));
  }
  if (air::getMemorySpace(dst_type) == innerMemorySpace) {
    auto internal = air::ChannelGetOp::create(
        builder, loc, tys, internalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_internal, dst, dst_offsets, dst_sizes, dst_strides,
        /*pad_before=*/nullptr, /*pad_after=*/nullptr);
    internalGetPut =
        dyn_cast_if_present<air::ChannelInterface>(internal.getOperation());
  } else {
    auto external = air::ChannelGetOp::create(
        builder, loc, tys, externalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_external, dst, dst_offsets, dst_sizes, dst_strides,
        /*pad_before=*/nullptr, /*pad_after=*/nullptr);
    externalGetPut =
        dyn_cast_if_present<air::ChannelInterface>(external.getOperation());
  }

  // A runtime packet-demux destination rides on the INTERNAL put -- the one
  // that stays on the core, which is where the demux index is computed and what
  // air-annotate-packet-ids reads. The external half is the memtile/shim side
  // and has no destination to select.
  Value demuxDest = op.getDest();

  if (air::getMemorySpace(src_type) == innerMemorySpace) {
    auto internal = air::ChannelPutOp::create(
        builder, loc, tys, internalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_internal, src, src_offsets, src_sizes, src_strides,
        demuxDest, padBefore, padAfter);
    internalGetPut =
        dyn_cast_if_present<air::ChannelInterface>(internal.getOperation());
  } else {
    auto external = air::ChannelPutOp::create(
        builder, loc, tys, externalDeps, FlatSymbolRefAttr::get(ctx, cname),
        channel_idx_external, src, src_offsets, src_sizes, src_strides,
        demuxDest, padBefore, padAfter);
    externalGetPut =
        dyn_cast_if_present<air::ChannelInterface>(external.getOperation());
  }

  if (!internalGetPut) {
    return op->emitOpError(
        "has unexpected memref memory space at internal-side");
  }
  if (!externalGetPut) {
    return op->emitOpError(
        "has unexpected memref memory space at external-side");
  }

  // Replace all uses to dma token with internal put/get token
  if (auto op_token = op.getAsyncToken()) {
    auto asyncInternalGetPut = dyn_cast_if_present<air::AsyncOpInterface>(
        internalGetPut.getOperation());
    op_token.replaceAllUsesWith(asyncInternalGetPut.getAsyncToken());
  }

  // Add attributes to label internal/external channel ops
  externalGetPut->setAttr("hoist", StringAttr::get(op->getContext(), "dep"));
  internalGetPut->setAttr("loop-carried-dep",
                          StringAttr::get(op->getContext(), "internalGetPut"));
  externalGetPut->setAttr("loop-carried-dep",
                          StringAttr::get(op->getContext(), "external"));
  // A derived far window is a whole buffer's worth per execution, so it belongs
  // where the buffer is FILLED -- once per fill, not once per near execution.
  // The hoist finds that site from the buffer itself.
  if (farPieces)
    externalGetPut->setAttr("air.derived_far_window",
                            UnitAttr::get(op->getContext()));
  if (op->hasAttr("broadcast_set"))
    externalGetPut->setAttr("broadcast_set", op->getAttr("broadcast_set"));
  // Carry the issue-order anchor onto the EXTERNAL half only. It is the
  // producer/consumer that leaves the hierarchy and lands in the destination
  // block, so it is the only one whose position is a free choice.
  if (auto anchor = op.getHoistAfterAttr())
    externalGetPut->setAttr("air.hoist_after", anchor);
  if (auto anchor = op.getHoistBeforeAttr())
    externalGetPut->setAttr("air.hoist_before", anchor);
  // Same reasoning for "do not carry my guards": it is a statement about where
  // the external half lands, so it belongs on the external half only.
  if (op->hasAttr("hoist_unguarded"))
    externalGetPut->setAttr("air.hoist_unguarded",
                            UnitAttr::get(op->getContext()));
  if (op->hasAttr("hoist_outside_loops"))
    externalGetPut->setAttr("air.hoist_outside_loops",
                            UnitAttr::get(op->getContext()));

  externalGetPutVector.push_back(externalGetPut);
  internalGetPutVector.push_back(internalGetPut);

  // The remaining N-1 pieces of a derived tiling. One per near execution, so
  // that the number of lock events on the far side equals the number on the
  // near side -- see deriveTiledFarWindow. Bytes alone would be satisfied by
  // the single wrapped descriptor these replace.
  //
  // Everything but the base offset is identical, and the base is constant by
  // construction (deriveTiledFarWindow declines otherwise), so the piece is a
  // clone with one static offset rewritten. Emitting them adjacent and
  // ascending keeps the FIFO order the near side reads them in.
  if (farPieces > 1) {
    bool farIsSrc = isa<air::ChannelPutOp>(externalGetPut.getOperation());
    StringRef offsAttrName =
        farIsSrc ? "static_src_offsets" : "static_dst_offsets";
    auto base = externalGetPut->getAttrOfType<DenseI64ArrayAttr>(offsAttrName);
    if (base && base.size() == 1) {
      OpBuilder::InsertionGuard guard(builder);
      Operation *prev = externalGetPut.getOperation();
      for (int64_t i = 1; i < farPieces; i++) {
        builder.setInsertionPointAfter(prev);
        Operation *piece = builder.clone(*externalGetPut.getOperation());
        piece->setAttr(offsAttrName, builder.getDenseI64ArrayAttr(
                                         {base[0] + i * farPieceVolume}));
        externalGetPutVector.push_back(dyn_cast<air::ChannelInterface>(piece));
        prev = piece;
      }
    }
  }
  return success();
}

// Check whether an channel op is within a matching air hierarchy (launch for
// any of [L1, L2, L3] memref, segment for [L1, L2] memref, and herd for L1
// memref).
bool isInMatchingHierarchy(air::ChannelInterface getput) {
  auto memref = getput.getMemref();
  auto memrefType = llvm::dyn_cast_if_present<BaseMemRefType>(memref.getType());
  if (!memrefType)
    return false;
  // Skip if channel op is already at its correct memory hierarchy.
  auto parentHier = getput->getParentOfType<air::HierarchyInterface>();
  if (!parentHier)
    return true;
  if (isa<air::HerdOp>(parentHier) && air::isL1(memrefType))
    return true;
  else if (isa<air::SegmentOp>(parentHier) &&
           (air::isL2(memrefType) || air::isL1(memrefType)))
    return true;
  else if (isa<air::LaunchOp>(parentHier)) {
    // Already at the launch level. Nowhere to hoist.
    return true;
  } else if (isa<air::RankOp>(parentHier)) {
    // RankOp is outermost only if it has no parent hierarchy op.
    if (!parentHier->getParentOfType<air::HierarchyInterface>())
      return true;
  }
  return false;
}

// Whether hoisting `getput` out of `hier_op` lands it in its matching memory
// hierarchy, i.e. whether this is the LAST hop of the walk outwards.
//
// An issue-order anchor may only be honoured on the last hop. A channel that
// stages through memory -- an L3 -> L2 -> L1 weight feed, say -- has endpoints
// at more than one level, so naming it as an anchor from inside a herd matches
// its SEGMENT-level endpoint and pins the transfer two levels short of where it
// belongs. The anchor is then consumed, and the remaining hop is unanchored, so
// the transfer silently lands at the hierarchy's position after all.
static bool hoistReachesMatchingHierarchy(air::ChannelInterface getput,
                                          Operation *hier_op) {
  auto memrefType =
      llvm::dyn_cast_if_present<BaseMemRefType>(getput.getMemref().getType());
  if (!memrefType)
    return false;
  auto destHier = hier_op->getParentOfType<air::HierarchyInterface>();
  if (!destHier || isa<air::LaunchOp>(destHier.getOperation()))
    return true;
  if (isa<air::SegmentOp>(destHier.getOperation()))
    return air::isL2(memrefType) || air::isL1(memrefType);
  if (isa<air::HerdOp>(destHier.getOperation()))
    return air::isL1(memrefType);
  return true;
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
  auto memrefType = llvm::dyn_cast_if_present<BaseMemRefType>(memref.getType());
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
    auto src_type = llvm::dyn_cast_if_present<BaseMemRefType>(src.getType());
    auto dst_type = llvm::dyn_cast_if_present<BaseMemRefType>(dst.getType());
    if (!src_type)
      return failure();

    if (air::isL3(src_type) && air::isL3(dst_type))
      return failure();

    if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
      return failure();

    air::HierarchyInterface hier_op = nullptr;
    air::MemorySpace innerMemorySpace = air::MemorySpace::L3;
    auto herd = op->getParentOfType<air::HerdOp>();
    auto segment = op->getParentOfType<air::SegmentOp>();
    if (herd) {
      hier_op =
          dyn_cast_if_present<air::HierarchyInterface>(herd.getOperation());
      innerMemorySpace = air::MemorySpace::L1;
    } else if (segment) {
      hier_op =
          dyn_cast_if_present<air::HierarchyInterface>(segment.getOperation());
      innerMemorySpace = air::MemorySpace::L2;
    } else
      return failure();

    SmallVector<air::ChannelInterface, 1> externalGetPut;
    SmallVector<air::ChannelInterface, 1> internalGetPut;

    if (failed(replaceAIRDmaWithAIRChannelPairs(
            rewriter, innerMemorySpace, op, internalGetPut, externalGetPut)))
      return failure();

    rewriter.eraseOp(op);

    return success();
  }
};

// The anchor symbol a batch agrees on, or null. Same rule as
// findIssueOrderAnchor: one insertion point serves the whole batch, so a batch
// that disagrees has no anchor.
static FlatSymbolRefAttr
anchorAttrOf(ArrayRef<air::ChannelInterface> externalGetPuts) {
  FlatSymbolRefAttr anchor;
  for (auto getput : externalGetPuts) {
    auto a = getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_after");
    if (!a)
      continue;
    if (anchor && anchor != a)
      return nullptr;
    anchor = a;
  }
  return anchor;
}

// Find the op named by an "air.hoist_after" anchor on any of the external
// channel ops: the LAST endpoint of that channel in the region the hierarchy op
// lives in, skipping anything inside the hierarchy op itself. Returns null when
// nothing is anchored or the anchor names a channel with no endpoint out here.
static SmallVector<Operation *>
findIssueOrderAnchors(ArrayRef<air::ChannelInterface> externalGetPuts,
                      Operation *hier_op, bool &placeBefore) {
  FlatSymbolRefAttr anchor;
  placeBefore = false;
  for (auto getput : externalGetPuts) {
    if (!hoistReachesMatchingHierarchy(getput, hier_op))
      continue;
    auto a = getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_after");
    if (!a) {
      a = getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_before");
      if (a)
        placeBefore = true;
    }
    if (!a)
      continue;
    // One shared insertion point is used for the whole batch, so only honour
    // the anchor when the batch agrees on it.
    if (anchor && anchor != a)
      return {};
    anchor = a;
  }
  if (!anchor)
    return {};

  // Only endpoints at the SAME hierarchy level count. Without this the herd
  // -level hoist would match an endpoint of the anchor channel sitting inside a
  // SIBLING herd, and drop the transfer in there.
  auto *hierParent =
      hier_op->getParentOfType<air::HierarchyInterface>()
          ? hier_op->getParentOfType<air::HierarchyInterface>().getOperation()
          : nullptr;
  // Which arm of an enclosing scf.index_switch an op sits in, or -1 if none.
  // scf.index_switch numbers its DEFAULT region 0 and its cases 1..n, but
  // prints the cases first -- so "last in walk order" is NOT "last in program
  // order", and picking by walk order lands a decode-only feed in the vocab
  // arm. Match the arm instead, which is what the front end means.
  // The whole PATH of arms, outermost first, not just the innermost one.
  // Matching a single index treats two different switches' region 1 as the same
  // place: a transfer guarded by the outer switch's case 0 then resolves onto
  // an endpoint sitting in a NESTED switch's case 0 and is emitted there, so
  // the arm it belonged to loses its feed entirely and another arm gets it
  // twice. The consumer in the starved arm waits forever. That is what withdrew
  // RMSW_DMA: on qwen3_8b the outer vocab arm's @rmsW put vanished and a
  // duplicate appeared two levels down.
  auto armPathOf = [](Operation *o) -> SmallVector<int, 4> {
    SmallVector<int, 4> path;
    for (Operation *p = o->getParentOp(); p; p = p->getParentOp()) {
      if (isa<air::HierarchyInterface>(p))
        break;
      if (isa<scf::IndexSwitchOp>(p))
        for (unsigned r = 0; r < p->getNumRegions(); r++)
          if (p->getRegion(r).isAncestor(o->getParentRegion()) ||
              &p->getRegion(r) == o->getParentRegion()) {
            path.push_back((int)r);
            break;
          }
    }
    std::reverse(path.begin(), path.end());
    return path;
  };
  SmallVector<int, 4> wantArm;
  bool haveWantArm = false;
  for (auto getput : externalGetPuts)
    if (getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_after") ||
        getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_before")) {
      wantArm = armPathOf(getput.getOperation());
      haveWantArm = true;
    }

  // "After the LAST endpoint" and "before the FIRST endpoint" are the mirror
  // pair. Taking the last one in both directions would drop a hoist_before
  // transfer in between a multi-endpoint anchor's own transfers instead of
  // ahead of the group -- e.g. anchoring ahead of a weight feed that is spelled
  // as two contiguous halves.
  // Three rankings, best first. EXACT is the endpoint standing in precisely the
  // arm the transfer belongs to. INSIDE is one standing in a switch NESTED in
  // that arm -- still the right arm, just further in; a weight feed split per
  // column group is the usual source. ANY is a last resort and is where a
  // wrong answer comes from, so it is only taken when the other two are empty.
  Operation *pickExact = nullptr, *pickAny = nullptr;
  // Inside matches kept ONE PER ARM. A transfer shallower than its anchor
  // belongs in every arm the anchor has an endpoint in, not in one of them, so
  // which arms those are is the answer -- collapsing them to a single winner
  // throws the question away.
  SmallVector<std::pair<SmallVector<int, 4>, Operation *>, 4> insideByArm;
  hier_op->getParentRegion()->walk([&](air::ChannelInterface o) {
    if (hier_op->isAncestor(o.getOperation()))
      return WalkResult::advance();
    if (o.getChanName() != anchor.getAttr())
      return WalkResult::advance();
    auto oParent = o->getParentOfType<air::HierarchyInterface>();
    if ((oParent ? oParent.getOperation() : nullptr) != hierParent)
      return WalkResult::advance();
    auto path = armPathOf(o.getOperation());
    bool exact = haveWantArm && path == wantArm;
    bool inside = haveWantArm && path.size() > wantArm.size() &&
                  std::equal(wantArm.begin(), wantArm.end(), path.begin());
    auto keep = [&](Operation *&slot) {
      // "After the LAST endpoint" and "before the FIRST" are the mirror pair.
      if (!placeBefore || !slot)
        slot = o.getOperation();
    };
    if (exact)
      keep(pickExact);
    else if (inside) {
      auto *hit = llvm::find_if(
          insideByArm, [&](const auto &kv) { return kv.first == path; });
      if (hit == insideByArm.end())
        insideByArm.push_back({path, o.getOperation()});
      else
        keep(hit->second);
    }
    keep(pickAny);
    return WalkResult::advance();
  });

  if (pickExact)
    return {pickExact};

  // Climbing out of an ARM is free; climbing out of a LOOP is not.
  //
  // That is the same distinction the anchored placement rests on elsewhere: an
  // arm changes only WHETHER the transfer is issued, which is context and is
  // what an anchor is for, while a loop changes how MANY times, which is a
  // property of the transfer itself and must never be inherited or discarded.
  // So ask of each inside match whether reaching the transfer's own arm depth
  // would cross a loop.
  auto climbCrossesLoop = [&](Operation *o) {
    for (Operation *p = o->getParentOp();
         p && !isa<air::HierarchyInterface>(p) &&
         armPathOf(p).size() >= wantArm.size();
         p = p->getParentOp()) {
      if (isa<LoopLikeOpInterface>(p))
        return true;
      if (armPathOf(p).size() == wantArm.size())
        break;
    }
    return false;
  };

  if (insideByArm.size() > 1 && llvm::any_of(insideByArm, [&](const auto &kv) {
        return climbCrossesLoop(kv.second);
      })) {
    // The anchor has endpoints in several arms, each inside a loop, and the
    // transfer is in none of them -- so it belongs in all of them: one copy per
    // arm, at that arm's own endpoint, inside that arm's loop.
    //
    // Climbing out issues the transfer the right number of TIMES, which is why
    // it looks equivalent, but it lands outside every one of those loops. In
    // fused_decode's egress the arms are the two phases and each runs its own
    // round loop, so what should be four memtile descriptors per round becomes
    // four times the round count, chained, at segment scope -- and
    // air-dependency-canonicalize rejects the result.
    //
    // The starvation this replaces -- a transfer landing in ONE nested arm
    // while the others wait forever, which withdrew RMSW_DMA on qwen3_8b -- is
    // the failure of picking one arm. Taking all of them answers it rather than
    // returning to it.
    SmallVector<Operation *> sites;
    for (auto &kv : insideByArm)
      sites.push_back(kv.second);
    return sites;
  }

  Operation *pickInside = nullptr;
  for (auto &kv : insideByArm)
    if (!placeBefore || !pickInside)
      pickInside = kv.second;
  Operation *pick = pickInside ? pickInside : pickAny;
  // An INSIDE match is deeper than the transfer, and no loop stands in the way.
  // Climb back out to the structure that holds every arm: before it covers each
  // arm's first endpoint, after it covers each arm's last.
  if (pick && pickInside) {
    while (armPathOf(pick).size() > wantArm.size()) {
      auto *p = pick->getParentOp();
      if (!p || isa<air::HierarchyInterface>(p))
        break;
      pick = p;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[dma-to-channel] anchor " << anchor
                          << " hoisting out of " << hier_op->getName()
                          << ": armDepth=" << wantArm.size() << " resolved="
                          << (pickExact
                                  ? "same-arm"
                                  : (!insideByArm.empty()
                                         ? "inside-arm"
                                         : (pickAny ? "any-arm" : "NONE")))
                          << " sites=" << (pick ? 1 : 0) << "\n");
  if (!pick)
    return {};
  return {pick};
}

// Do two access-pattern lists describe the same window?
//
// By VALUE, not by SSA identity. Each hoisted transfer materializes its own
// copy of the pure defs it needs -- that is what lets a clone stand alone in
// the block it lands in -- so two spellings of one constant offset are
// different Values by construction. Comparing the Values answers "not the same
// window" for every transfer this is ever asked about.
static bool sameWindow(ArrayRef<OpFoldResult> a, ArrayRef<OpFoldResult> b) {
  if (a.size() != b.size())
    return false;
  for (auto [x, y] : llvm::zip_equal(a, b)) {
    std::optional<int64_t> cx = getConstantIntValue(x);
    std::optional<int64_t> cy = getConstantIntValue(y);
    if (cx || cy) {
      // One side constant and the other not is a difference this cannot see
      // through, so say so rather than guess.
      if (cx != cy)
        return false;
      continue;
    }
    if (x != y)
      return false;
  }
  return true;
}

// Is `a` the same transfer as `b`, ignoring async plumbing?
//
// Same channel, same sub-channel, same buffer, same window. The async
// dependency lists are deliberately NOT compared: they are how the transfer is
// SCHEDULED, not what it moves, and two clones of one transfer reaching the
// same point by different paths carry different tokens by construction.
static bool sameTransfer(air::ChannelInterface a, air::ChannelInterface b) {
  if (a->getName() != b->getName())
    return false;
  if (a.getChanName() != b.getChanName())
    return false;
  if (a.getMemref() != b.getMemref())
    return false;
  // Note OperandRange's own == compares the RANGE rather than its elements,
  // and two distinct ops never share one.
  SmallVector<OpFoldResult> ai, bi;
  for (Value v : a.getIndices())
    ai.push_back(v);
  for (Value v : b.getIndices())
    bi.push_back(v);
  return sameWindow(ai, bi) &&
         sameWindow(a.getMixedOffsets(), b.getMixedOffsets()) &&
         sameWindow(a.getMixedSizes(), b.getMixedSizes()) &&
         sameWindow(a.getMixedStrides(), b.getMixedStrides());
}

// A transfer already emitted at this point, or null.
//
// A producer stream SHARED by several consumer sites is one transfer, and the
// hoist sees it once per site: each site names the same channel, the same
// memtile buffer and the same window, and each is anchored to the same feed. A
// second descriptor for it is not a second transfer -- it re-sends bytes the
// first already sent, on the same sub-channel, acquiring the fill lock a second
// time for a fill that happens once. The memtile MM2S ring then carries two
// acquires per release against an S2MM that still has one, and the design reads
// stale weights.
//
// The front end could not have said this differently: with the transfer spelled
// as an air.channel.put it writes ONE put in the producer's own loop and lets
// several gets share it -- which is legal, and is the shape this restores.
static Operation *findEmittedTransfer(air::ChannelInterface getput,
                                      Block *blk) {
  for (Operation &o : *blk) {
    if (&o == getput.getOperation())
      continue;
    auto other = dyn_cast<air::ChannelInterface>(&o);
    if (!other)
      continue;
    if (sameTransfer(getput, other))
      return &o;
  }
  return nullptr;
}

// Drop async dependencies that do not reach where the op landed.
//
// The external half inherits the loop-carried token of the loop it was written
// in. When it is placed by position -- at an anchor, or at the fill its derived
// window belongs to -- it leaves that loop behind, and the token becomes a
// block argument of a region it is no longer inside. Rebuilding the loop is not
// an option here: that is the whole point of placing by position rather than by
// structure. Drop the stale edges; air-dependency-canonicalize re-derives the
// real ones from the buffers.
static void pruneNonDominatingDeps(Operation *op) {
  auto async = dyn_cast<air::AsyncOpInterface>(op);
  if (!async)
    return;
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return;
  DominanceInfo dom(func);
  // Erase from the back so the earlier indices stay valid.
  SmallVector<unsigned> drop;
  for (auto [i, d] : llvm::enumerate(async.getAsyncDependencies()))
    if (!dom.properlyDominates(d, op))
      drop.push_back(i);
  for (unsigned i : llvm::reverse(drop))
    async.eraseAsyncDependency(i);
}

// How many tiles of `hier_op` actually execute `getput`?
//
// A DMA carries one multiplicity, the CONSUMER's: it sits inside the hierarchy,
// under whatever scf.if chain selects a tile. The PRODUCER's multiplicity has
// to be derived, and it is derivable -- it is the number of tiles satisfying
// the guard. Wrapping a one-tile transfer in an scf.parallel over the whole
// iteration space issues it once per tile instead of once.
//
// Do NOT read this off `broadcast_set`. That attribute is the herd's BOUNDING
// BOX -- for a 2x4 herd it is literally `0 <= s0 <= 1, 0 <= s1 <= 3`, with no
// equalities -- so it says "all tiles" no matter what the guards say. The guard
// lives in the scf.if chain, still intact around the external half here because
// that half is created in the DMA's own position.
//
// Herd extents are compile-time constants, so enumerate rather than reach for
// affine machinery: it is exact, and it handles the else-branch case
// (`ty < 2` then `not (ty == 0)` gives `ty == 1`) that constraint solving needs
// integer reasoning for. Returns nullopt for anything that is not a constant
// comparison against a hierarchy induction variable, which keeps every existing
// design on the old path.
static std::optional<int64_t>
countExecutingTiles(air::ChannelInterface getput,
                    air::HierarchyInterface hier_op) {
  SmallVector<int64_t> extents;
  for (auto sz : hier_op.getSizeOperands()) {
    auto c = getConstantIntValue(sz);
    if (!c || *c <= 0 || *c > 64)
      return std::nullopt;
    extents.push_back(*c);
  }
  if (extents.empty())
    return std::nullopt;
  auto ids = hier_op.getIds();
  auto dimOf = [&](Value v) -> std::optional<unsigned> {
    for (unsigned i = 0; i < ids.size(); i++)
      if (ids[i] == v)
        return i;
    return std::nullopt;
  };

  struct Guard {
    unsigned dim;
    arith::CmpIPredicate pred;
    int64_t rhs;
    bool inThen;
  };
  SmallVector<Guard> guards;
  Operation *child = getput.getOperation();
  for (Operation *p = child->getParentOp(); p && p != hier_op.getOperation();
       child = p, p = p->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(p);
    if (!ifOp)
      continue;
    auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return std::nullopt;
    auto d = dimOf(cmp.getLhs());
    auto rhs = getConstantIntValue(cmp.getRhs());
    if (!d || !rhs)
      return std::nullopt;
    guards.push_back(
        {*d, cmp.getPredicate(), *rhs,
         ifOp.getThenRegion().isAncestor(child->getParentRegion())});
  }
  if (guards.empty())
    return std::nullopt;

  int64_t total = 1;
  for (auto e : extents)
    total *= e;
  int64_t hits = 0;
  SmallVector<int64_t> iv(extents.size(), 0);
  for (int64_t flat = 0; flat < total; flat++) {
    int64_t r = flat;
    for (int i = (int)extents.size() - 1; i >= 0; i--) {
      iv[i] = r % extents[i];
      r /= extents[i];
    }
    bool ok = true;
    for (auto &g : guards) {
      int64_t v = iv[g.dim];
      bool t;
      switch (g.pred) {
      case arith::CmpIPredicate::eq:
        t = v == g.rhs;
        break;
      case arith::CmpIPredicate::ne:
        t = v != g.rhs;
        break;
      case arith::CmpIPredicate::slt:
        t = v < g.rhs;
        break;
      case arith::CmpIPredicate::sle:
        t = v <= g.rhs;
        break;
      case arith::CmpIPredicate::sgt:
        t = v > g.rhs;
        break;
      case arith::CmpIPredicate::sge:
        t = v >= g.rhs;
        break;
      default:
        return std::nullopt;
      }
      if (t != g.inThen) {
        ok = false;
        break;
      }
    }
    if (ok)
      hits++;
  }
  return hits;
}

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
          auto getput = dyn_cast_if_present<air::ChannelInterface>(o);
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

    // "air.hoist_unguarded": place by default, but do NOT rebuild my guards.
    //
    // An unanchored hoist clones the guards the transfer sat under so the
    // external half stays conditional; an anchored one skips them and inherits
    // the anchor's context instead. A transfer whose hand-written counterpart
    // was UNGUARDED at the outer scope wants neither.
    //
    // It is not merely cosmetic. A guard on the hierarchy's own induction
    // variable is fine -- the tile-count machinery collapses it. A guard on an
    // i32 RUNTIME PARAMETER is not: rebuilding it at segment scope emits an
    // `arith.index_cast` on the segment's own i32 block argument, and that
    // cannot survive the segment becoming an aie.device. air-to-aie reports
    // "'arith.index_cast' op using value defined outside the region".
    //
    // fused_decode's hybrid is exactly that shape -- its arm is a per-layer
    // IS_ATTN RTP that must survive to runtime -- and anchoring, the other way
    // to skip the rebuild, has no endpoint at the right depth to name there.
    bool unguarded =
        llvm::any_of(externalGetPuts, [](air::ChannelInterface getput) {
          return getput->hasAttr("air.hoist_unguarded");
        });

    // Resolve the issue-order anchor up front: it decides whether the enclosing
    // control ops are pulled in and rebuilt at all.
    // Partition the batch by the anchor each op names.
    //
    // One shared insertion point used to serve the whole batch, so a batch that
    // disagreed simply lost its anchors. That is not a disagreement to resolve:
    // a weight fan written from the core side puts every column's transfers on
    // ONE channel while each names the feed IT is fed from, so sixteen
    // transfers name eight anchors and every one of them is right. Resolve and
    // place per group; a batch that agrees is the one-group case and is
    // unchanged.
    llvm::MapVector<Attribute, SmallVector<air::ChannelInterface>> byAnchor;
    bool everyOpAnchored = true;
    for (auto getput : externalGetPuts) {
      auto a = getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_after");
      if (!a)
        a = getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_before");
      if (!a) {
        everyOpAnchored = false;
        break;
      }
      byAnchor[a].push_back(getput);
    }
    // Mixed anchored/unanchored keeps the old single-point behaviour: the
    // unanchored ops have no site of their own to go to.
    if (!everyOpAnchored || byAnchor.size() < 2) {
      byAnchor.clear();
      byAnchor[FlatSymbolRefAttr()] = SmallVector<air::ChannelInterface>(
          externalGetPuts.begin(), externalGetPuts.end());
    }

    bool placeBefore = false;
    // A derived far window has a position that follows from the buffer, so it
    // needs no anchor: place it after the fill, in the fill's loop.
    SmallVector<Operation *> derivedSites;
    for (auto getput : externalGetPuts)
      if (getput->hasAttr("air.derived_far_window")) {
        SmallVector<Operation *> fills;
        findFillSites(getput.getMemref(), hier_op.getOperation(), fills);
        for (Operation *fill : fills)
          if (!llvm::is_contained(derivedSites, fill))
            derivedSites.push_back(fill);
      }

    SmallVector<Operation *> anchorOps = findIssueOrderAnchors(
        byAnchor.front().second, hier_op.getOperation(), placeBefore);
    // More than one site means the anchor lives in several switch arms and the
    // transfer belongs in each; the clone below runs once per site.
    if (anchorOps.empty() && !derivedSites.empty())
      anchorOps = derivedSites;
    Operation *anchorOp = anchorOps.empty() ? nullptr : anchorOps.front();

    // Get backward slices to the target "external" side channel ops, to be
    // hoisted together.
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions bsOptions{[&](Operation *o) { return o != hier_op; }};
    for (auto op : externalGetPuts) {
      (void)getBackwardSlice(op.getOperation(), &backwardSlice, bsOptions);

      // Anchored: the op is placed inside the anchor's control context, so the
      // guards it sits under on the hierarchy side must NOT come along -- their
      // conditions are defined in the region being left behind, which is what
      // "using value defined outside the region" reports.
      if (!anchorOp && !unguarded)
        for (auto parent = op->getParentOp();
             !isa<air::HierarchyInterface>(parent);
             parent = parent->getParentOp()) {
          (void)getBackwardSlice(parent, &backwardSlice, bsOptions);
          backwardSlice.insert(parent);
        }
    }
    // Get constant values used by backward slices, and add to backward
    // slices. Collect into a temporary first: inserting into backwardSlice
    // from within this loop can reallocate the SetVector's backing storage and
    // invalidate the range-for iterator (a SIGSEGV reproducible when the
    // hoisted op nests under enough affine.if/scf.if regions, e.g. a cascade
    // GEMV). Newly added ops are always constants (no regions), so deferring
    // their insertion does not change the result.
    SmallVector<Operation *> constantsToAdd;
    for (auto o : backwardSlice) {
      for (auto &region : o->getRegions()) {
        visitUsedValuesDefinedAbove(region, [&constantsToAdd](OpOperand *use) {
          if (getConstantIntValue(use->get())) {
            constantsToAdd.push_back(use->get().getDefiningOp());
          }
        });
      }
    }
    backwardSlice.insert(constantsToAdd.begin(), constantsToAdd.end());

    // Don't miss out the backward slices of air.execute op's child ops.
    auto backwardSliceCopy = backwardSlice;
    for (auto b : backwardSliceCopy) {
      if (auto execOp = dyn_cast_if_present<air::ExecuteOp>(b)) {
        for (auto &exec_child_op : execOp.getChildOps()) {
          (void)getBackwardSlice(&exec_child_op, &backwardSlice, bsOptions);
          backwardSlice.insert(&exec_child_op);
        }
      }
    }

    // Label backward slices with attribute; ops not labelled with "hoist" flag
    // shall either not get hoisted, if IR is not async, or become air.wait_all
    // (null op) after being hoisted.
    //
    // Note: user-written channel ops may end up in backwardSlice as transitive
    // dependencies (e.g., a broadcast ChannelGet whose token feeds a wait_all
    // gating the DMA's loop). These ops receive "hoist" here but intentionally
    // do NOT receive "loop-carried-dep" — cloneOpsInBlock uses the absence of
    // "loop-carried-dep" to distinguish them from DMA-derived channel ops and
    // replaces them with wait_all instead of cloning to segment level.
    backwardSlice.insert(externalGetPuts.begin(), externalGetPuts.end());
    for (auto b : backwardSlice) {
      b->setAttr("hoist", StringAttr::get(ctx, "dep"));
    }

    // The backward slice is operand-producers only, so it never contains the
    // CONTROL ops a target is nested in. cloneOpsInBlock turns every unlabelled
    // op into a wait_all, so an unlabelled enclosing region op takes the target
    // inside it down with it -- the external half is silently dropped and its
    // partner is left unpaired, surfacing 30-odd passes later in
    // air-verify-hierarchy-locality. Label the ancestors up to the hierarchy op
    // being hoisted out of, so the structure is rebuilt around the target.
    //
    // Skipped when anchored: the op is being placed INSIDE the anchor's control
    // context, so rebuilding the hierarchy-side guards would both duplicate
    // that context and reference conditions defined in the region being left
    // behind.
    if (!anchorOp && !unguarded) {
      for (auto getput : externalGetPuts) {
        for (Operation *p = getput->getParentOp();
             p && p != hier_op.getOperation(); p = p->getParentOp()) {
          if (isa<air::HierarchyInterface>(p))
            break;
          if (isa<scf::IfOp, affine::AffineIfOp, scf::IndexSwitchOp,
                  LoopLikeOpInterface>(p))
            p->setAttr("hoist", StringAttr::get(ctx, "dep"));
        }
      }
    }

    // Hoist hierarchy op into scf op
    bool anchored = false;
    Operation *scf_loop = nullptr;
    mlir::OpBuilder::InsertPoint
        insertionPointAtHierOp; // To keep a record of the insertion point as
                                // destination for hoisting

    rewriter.setInsertionPoint(hier_op);
    insertionPointAtHierOp = rewriter.saveInsertionPoint();

    // Issue-order anchor. By default the external half lands immediately before
    // the hierarchy op, i.e. after every producer the front end wrote by hand.
    // On a launch carrying air.preserve_shim_dma_order that reorders the shim
    // BD queue, which is load-bearing in real designs -- moving one feed from
    // slot 6 to slot 18 behind a weight stream is enough to deadlock a core
    // waiting on it. When the front end names an anchor, place the op straight
    // after that channel's LAST endpoint instead, so it inherits both the
    // anchor's position and its control context (no new guard is synthesised;
    // if the anchor sits in a switch arm, so does this).
    // Turn one resolved anchor into an insertion point. Factored so that every
    // anchor GROUP gets the same treatment -- the outside-loops climb and the
    // group-ordering walk are properties of the anchor, not of being first.
    auto siteFor = [&](Operation *at, ArrayRef<air::ChannelInterface> ops,
                       bool before) -> mlir::OpBuilder::InsertPoint {
      if (llvm::any_of(ops, [](air::ChannelInterface getput) {
            return getput->hasAttr("air.hoist_outside_loops");
          })) {
        while (auto *parent = at->getParentOp()) {
          if (!isa<LoopLikeOpInterface>(parent))
            break;
          at = parent;
        }
        before = true;
      }
      if (before) {
        rewriter.setInsertionPoint(at);
      } else {
        Operation *tail = at;
        for (Operation *n = at->getNextNode(); n; n = n->getNextNode()) {
          auto a = n->getAttrOfType<FlatSymbolRefAttr>("air.hoist_after");
          if (!a || a != anchorAttrOf(ops))
            break;
          tail = n;
        }
        rewriter.setInsertionPointAfter(tail);
      }
      return rewriter.saveInsertionPoint();
    };

    if (anchorOp) {
      // "air.hoist_outside_loops": resolve the anchor, then step OUT of any
      // loops enclosing it, stopping at the first non-loop parent.
      //
      // An anchor names a channel, so it resolves to an op, so the transfer
      // becomes that op's SIBLING -- it inherits the anchor's depth exactly.
      // When the transfer belongs one level shallower there is nothing to say,
      // and the failure is quiet: land a transfer inside a loop its consumers
      // sit outside of and the consumers stop dominating it.
      //
      // Stepping out of LOOPS specifically, and not out of arms, is the same
      // distinction this whole hoist rests on. A loop changes how MANY times
      // the transfer is issued, which is a property of the transfer and must
      // not be inherited from a neighbour. An arm changes only WHETHER it is
      // issued, which is a property of the surrounding context and is exactly
      // what an anchor is for. So: inherit the anchor's predicate, not its
      // trip count.
      if (llvm::any_of(externalGetPuts, [](air::ChannelInterface getput) {
            return getput->hasAttr("air.hoist_outside_loops");
          })) {
        while (auto *parent = anchorOp->getParentOp()) {
          if (!isa<LoopLikeOpInterface>(parent))
            break;
          anchorOp = parent;
        }
        // Stepping out inverts the direction: "after the last endpoint" inside
        // a loop means "after the loop", but "before the first" likewise means
        // "before the loop", and the group-ordering walk below keys off
        // siblings that no longer exist at this level. Place before the loop in
        // both directions, which is where the hand-written endpoint sat.
        placeBefore = true;
      }
      if (placeBefore) {
        // Inserting each op directly before the anchor already preserves the
        // group's order: [A, T], then [A, B, T].
        rewriter.setInsertionPoint(anchorOp);
      } else {
        // "After the anchor" does NOT preserve order -- inserting each op
        // directly after T gives [T, A], then [T, B, A], so a group of N
        // siblings sharing one anchor comes out reversed. A per-CU fan is
        // exactly that shape, and reversal is silent: the transfers are all
        // legal, just issued back to front. Step past everything already
        // anchored to the same target so the group keeps its order.
        Operation *tail = anchorOp;
        for (Operation *n = anchorOp->getNextNode(); n; n = n->getNextNode()) {
          auto a = n->getAttrOfType<FlatSymbolRefAttr>("air.hoist_after");
          if (!a || a != anchorAttrOf(externalGetPuts))
            break;
          tail = n;
        }
        rewriter.setInsertionPointAfter(tail);
      }
      insertionPointAtHierOp = rewriter.saveInsertionPoint();
      anchored = true;
    }

    // Check if broadcasting happens for any "external" side channel ops. If so,
    // the hoisted scf parallel should respect the broadcast shape instead. If
    // broadcasting is detected, then hoist and specialize each data movement
    // (i.e. do not hoist the air.hierarchy iteration space.)
    if (anchored || unguarded) {
      // The anchor fixes position and control context. Wrapping the iteration
      // space in an scf.parallel here would move it back to the hierarchy op.
      // An unguarded hoist keeps the default position but has likewise asked
      // not to have control structure synthesised around it.
    } else if (llvm::any_of(externalGetPuts,
                            [](air::ChannelInterface getput) {
                              return air::getChannelDeclarationThroughSymbol(
                                         getput)
                                  ->hasAttr("broadcast_shape");
                            }) ||
               llvm::all_of(externalGetPuts, [&](air::ChannelInterface getput) {
                 return countExecutingTiles(getput, hier_op) == 1;
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
    if (anchored || unguarded) {
      // Anchored, or unguarded: clone ONLY the external ops and what feeds
      // them, straight in at the insertion point. cloneOpsInBlock is not usable
      // here -- it walks the hierarchy's top-level block and turns everything
      // unlabelled into a wait_all, so an op nested in a guard whose ancestors
      // are deliberately not labelled (they belong to the context being left
      // behind) would be dropped and its partner left unpaired.
      //
      // The two cases differ only in WHERE: an anchored op goes to the anchor,
      // an unguarded one to the default point just before the hierarchy op.
      // They agree on what to clone, which is why they share this path.
      SetVector<Operation *> toClone;
      for (auto *b : backwardSlice) {
        if (isa<air::ChannelInterface>(b) ||
            b->hasTrait<OpTrait::IsTerminator>() || b->getNumRegions())
          continue;
        // An op nested inside a region op cannot stand alone where this lands:
        // its wrapper is what defines the values it uses, and cloning it out of
        // that wrapper orphans it. The slice reaches such ops only by following
        // ASYNC TOKENS -- the wrapper's token, not any value the transfer reads
        // -- and those edges are dropped anyway when the transfer is placed by
        // position, so nothing here is lost.
        if (auto *parent = b->getParentOp())
          if (isa<air::ExecuteOp>(parent))
            continue;
        toClone.insert(b);
      }

      // One emission per anchor site. Unanchored, and in the ordinary anchored
      // case, that is a single site and this loop runs once; several sites
      // means the anchor's endpoints are spread over sibling switch arms and
      // the transfer belongs in each of them.
      //
      // Each site gets its OWN mapping. Sharing one would make the second site
      // reuse the first site's clones, which sit in a sibling region and
      // dominate nothing here.
      // Set when a clone was dropped as a duplicate of a transfer already at
      // the site. Emitting nothing is then the correct outcome, not a failure.
      bool dedupedAny = false;

      // Several GROUPS means the ops name different anchors; several SITES
      // within a group means that anchor's endpoints are spread over sibling
      // switch arms.
      for (auto &group : byAnchor) {
        bool firstGroup = &group == &byAnchor.front();
        SmallVector<mlir::OpBuilder::InsertPoint> sites;
        if (anchored) {
          bool groupBefore = placeBefore;
          SmallVector<Operation *> groupAnchors =
              firstGroup
                  ? anchorOps
                  : findIssueOrderAnchors(group.second, hier_op.getOperation(),
                                          groupBefore);
          // The first group with a single site already has its insertion point
          // computed above, outside-loops climb and ordering walk included.
          // Recomputing it here would drop both.
          if (firstGroup && groupAnchors.size() < 2)
            sites.push_back(insertionPointAtHierOp);
          else
            for (Operation *site : groupAnchors)
              sites.push_back(siteFor(site, group.second, groupBefore));
        }
        if (sites.empty())
          sites.push_back(insertionPointAtHierOp);

        for (auto &site : sites) {
          IRMapping siteRemap;
          int arg_idx = 0;
          for (auto arg : hier_op.getKernelArguments())
            siteRemap.map(arg, hier_op.getKernelOperand(arg_idx++));
          rewriter.restoreInsertionPoint(site);
          // getBackwardSlice fills the SetVector defs-before-uses, so its own
          // order is already a valid clone order.
          for (auto *b : toClone)
            pruneNonDominatingDeps(rewriter.clone(*b, siteRemap));
          for (auto getput : group.second) {
            Operation *c = rewriter.clone(*getput.getOperation(), siteRemap);
            pruneNonDominatingDeps(c);
            // Already emitted here by another consumer site sharing this
            // producer stream: keep the one descriptor and let this site's
            // internal half share it, exactly as the hand-written form does.
            if (auto ci = dyn_cast<air::ChannelInterface>(c)) {
              if (Operation *prior = findEmittedTransfer(ci, c->getBlock())) {
                c->replaceAllUsesWith(prior);
                rewriter.eraseOp(c);
                dedupedAny = true;
                continue;
              }
            }
            clonedOps.push_back(c);
          }
        }
      }
      // Every clone being a duplicate is success: the transfer is already at
      // the site, and the internal halves left behind will share it. Failing
      // here rolls the whole conversion back and strands this DMA's external
      // half inside the hierarchy.
      if (clonedOps.empty() && !dedupedAny)
        return failure();
    } else if (auto scf_par = dyn_cast_or_null<scf::ParallelOp>(scf_loop)) {
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
                    arith::ConstantIndexOp::create(
                        rewriter, rewriter.getUnknownLoc(), 0));
          for (unsigned i = 0; i < is.getNumConstraints(); i++) {
            if (!is.isEq(i))
              continue;
            auto c = is.getConstraint(i);
            if (!c.isFunctionOfSymbol(hierDim))
              continue;
            auto constIV = arith::ConstantIndexOp::create(
                rewriter, rewriter.getUnknownLoc(),
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
      auto clonedExternalGetPut =
          dyn_cast_if_present<air::ChannelInterface>(cloned);
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
      // The anchor is a front-end directive, not output metadata. Drop it once
      // it has been honoured, but only then -- an unanchored hoist has to carry
      // it further out (herd -> segment -> launch) to be honoured later.
      // NOTE: deliberately NOT removing air.hoist_after here. A later sibling
      // anchored to the same target has to see it, to step past this op rather
      // than land in front of it. Cleared for everything in the final sweep.
      if (anchored)
        cloned->removeAttr("air.hoist_before");
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
      T::create(builder, op.getLoc(), newAsyncDeps, op.getSizeOperands(),
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

// Specialized version for RankOp that threads the optional universe operand.
static Value insertArgToRankOp(OpBuilder &builder, air::RankOp op,
                               SmallVector<Value> vec) {
  SmallVector<Value> newOperands;
  SmallVector<int> newOperandsIdx;
  for (int i = 0, e = op.getNumKernelOperands(); i < e; i++) {
    newOperands.push_back(op.getKernelOperand(i));
    newOperandsIdx.push_back(i);
  }
  newOperands.insert(newOperands.end(), vec.begin(), vec.end());

  SmallVector<Value> newAsyncDeps = op.getAsyncDependencies();

  builder.setInsertionPoint(op);
  IRMapping remap;
  auto newOp =
      air::RankOp::create(builder, op.getLoc(), newAsyncDeps, op.getUniverse(),
                          op.getSizeOperands(), newOperands,
                          op->getNumResults() > 0, op->getAttrs());

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
    if (!isa<air::RankTerminatorOp>(o))
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
  else if (auto herd = dyn_cast_if_present<air::HerdOp>(op))
    return insertArgToHierOpImpl<air::HerdOp>(builder, herd, vec);
  else if (auto segment = dyn_cast_if_present<air::SegmentOp>(op))
    return insertArgToHierOpImpl<air::SegmentOp>(builder, segment, vec);
  else if (auto launch = dyn_cast_if_present<air::LaunchOp>(op))
    return insertArgToHierOpImpl<air::LaunchOp>(builder, launch, vec);
  else if (auto rank = dyn_cast_if_present<air::RankOp>(op))
    return insertArgToRankOp(builder, rank, vec);
  else
    return nullptr;
}

static LogicalResult AIRDemoteMemrefToAIRHierarchy(
    std::pair<air::HierarchyInterface, std::vector<Operation *>> pair,
    OpBuilder &builder) {

  air::HierarchyInterface hier_op = pair.first;
  air::MemorySpace hierMemorySpace = air::MemorySpace::L3;
  if (isa<air::HerdOp>(hier_op.getOperation())) {
    hierMemorySpace = air::MemorySpace::L1;
  } else if (isa<air::SegmentOp>(hier_op.getOperation())) {
    hierMemorySpace = air::MemorySpace::L2;
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
      auto memref_type =
          llvm::dyn_cast_if_present<BaseMemRefType>(memref.getType());

      auto allocMemSpace = air::getMemorySpace(memref_type);
      if (!allocMemSpace)
        continue; // Unrecognized memory space, skip
      if (*allocMemSpace == hierMemorySpace)
        continue; // Alloc op is already under correct hierarchy
      else if (air::isMoreLocal(*allocMemSpace, hierMemorySpace))
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
      if (auto new_alloc = dyn_cast_if_present<memref::AllocOp>(new_op)) {
        memref.replaceAllUsesWith(new_alloc.getMemref());
        new_memrefs.push_back(new_alloc.getMemref());
      } else if (auto new_exec = dyn_cast_if_present<air::ExecuteOp>(new_op)) {
        memref.replaceAllUsesWith(new_exec->getResult(1));
        new_memrefs.push_back(new_exec->getResult(1));
        // token.replaceAllUsesWith(new_exec->getResult(0));
        builder.setInsertionPoint(op);
        token.replaceAllUsesWith(
            air::WaitAllOp::create(builder, loc,
                                   air::AsyncTokenType::get(op->getContext()),
                                   new_exec.getAsyncDependencies())
                .getAsyncToken());
        // Update async deps
        clearAsyncDependenciesOfAsyncOp(new_exec);
        auto async_hier_op =
            dyn_cast_if_present<air::AsyncOpInterface>(hier_op.getOperation());
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
        if (auto new_exec = dyn_cast_if_present<air::ExecuteOp>(new_dealloc)) {
          builder.setInsertionPoint(dealloc);
          dealloc->getResult(0).replaceAllUsesWith(
              air::WaitAllOp::create(builder, loc,
                                     air::AsyncTokenType::get(op->getContext()),
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
    auto src_type = llvm::dyn_cast_if_present<BaseMemRefType>(src.getType());
    auto dst_type = llvm::dyn_cast_if_present<BaseMemRefType>(dst.getType());
    if (!src_type)
      return failure();

    if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
      return failure();

    auto herd = op->getParentOfType<air::HerdOp>();
    auto segment = op->getParentOfType<air::SegmentOp>();

    if (air::getMemorySpace(src_type) == air::getMemorySpace(dst_type))
      return failure(); // Src and dst under same memory space

    air::HierarchyInterface hier_op = nullptr;
    air::MemorySpace innerMemorySpace = air::MemorySpace::L3;
    if (herd) {
      hier_op =
          dyn_cast_if_present<air::HierarchyInterface>(herd.getOperation());
      innerMemorySpace = air::MemorySpace::L1;
    } else if (segment) {
      hier_op =
          dyn_cast_if_present<air::HierarchyInterface>(segment.getOperation());
      innerMemorySpace = air::MemorySpace::L2;
    } else
      return failure();

    auto srcMS = air::getMemorySpace(src_type);
    auto dstMS = air::getMemorySpace(dst_type);
    if (!srcMS || !dstMS)
      return failure();
    auto memcpyInnerMemorySpace = air::moreLocal(*srcMS, *dstMS);
    if (memcpyInnerMemorySpace == innerMemorySpace)
      return failure(); // Dma op is already under correct hierarchy
    else if (air::isMoreLocal(memcpyInnerMemorySpace, innerMemorySpace))
      return failure(); // This pass is currently not able to promote in memory
                        // tier

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
        if (auto parent_for =
                dyn_cast_if_present<scf::ForOp>(op->getParentOp())) {
          backwardSlice.insert(parent_for);
          for (auto oper : parent_for->getOperands())
            if (getConstantIntValue(oper))
              backwardSlice.insert(oper.getDefiningOp());
        }
      }

      for (auto b : backwardSlice) {
        auto execOp = dyn_cast_if_present<air::ExecuteOp>(b);
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
          auto for_op = dyn_cast_if_present<scf::ForOp>(op->getParentOp());
          for (auto init_arg : for_op.getInitArgs())
            remap.map(init_arg, air::WaitAllOp::create(
                                    rewriter, loc,
                                    air::AsyncTokenType::get(op->getContext()),
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
          air::WaitAllOp::create(rewriter, loc,
                                 air::AsyncTokenType::get(op->getContext()),
                                 op.getAsyncDependencies())
              .getAsyncToken());
    }

    for (auto e : erased) {
      rewriter.eraseOp(e);
    }

    return success();
  }
};
// Hoisting clones the ops a transfer depends on, so the guard condition around
// the hoisted copy is usually a duplicate SSA value rather than the original --
// `%48 = arith.index_cast %13` beside the `%14 = arith.index_cast %13` the
// producer switches on. CSE merges the two, but not until two passes later, so
// equality here has to be structural. Pure ops only: two calls to something
// with side effects are two events, not one value.
static bool sameValue(Value a, Value b, unsigned depth = 4) {
  if (a == b)
    return true;
  if (!depth)
    return false;
  Operation *da = a.getDefiningOp(), *db = b.getDefiningOp();
  if (!da || !db || da->getName() != db->getName() ||
      da->getAttrDictionary() != db->getAttrDictionary() ||
      da->getNumOperands() != db->getNumOperands() || !isMemoryEffectFree(da) ||
      !isMemoryEffectFree(db))
    return false;
  if (cast<OpResult>(a).getResultNumber() !=
      cast<OpResult>(b).getResultNumber())
    return false;
  for (auto [x, y] : llvm::zip(da->getOperands(), db->getOperands()))
    if (!sameValue(x, y, depth - 1))
      return false;
  return true;
}

// Two guards select the same arm for the same reason: same op kind, the same
// operands up to the equivalence above, and -- for a switch -- the same cases.
static bool sameGuard(Operation *a, Operation *b) {
  if (a->getName() != b->getName() ||
      a->getNumOperands() != b->getNumOperands() ||
      a->getNumRegions() != b->getNumRegions())
    return false;
  for (auto [x, y] : llvm::zip(a->getOperands(), b->getOperands()))
    if (!sameValue(x, y))
      return false;
  auto sw = dyn_cast<scf::IndexSwitchOp>(a);
  auto esw = dyn_cast<scf::IndexSwitchOp>(b);
  if (sw && esw && sw.getCases() != esw.getCases())
    return false;
  return true;
}

// Every memref any channel endpoint under `op` touches.
static llvm::SmallSetVector<Value, 8> channelMemrefsUnder(Operation *op) {
  llvm::SmallSetVector<Value, 8> memrefs;
  op->walk([&](air::ChannelInterface ci) { memrefs.insert(ci.getMemref()); });
  return memrefs;
}

static bool touchesAny(Operation *op,
                       const llvm::SmallSetVector<Value, 8> &memrefs) {
  bool touches = false;
  auto scan = [&](Operation *o) {
    for (Value operand : o->getOperands())
      if (memrefs.contains(operand))
        touches = true;
  };
  scan(op);
  op->walk(scan);
  return touches;
}

// The two loops must agree on trip count for fusion to preserve meaning. The
// bounds are compared as SSA values first, then as constants, because the
// hoisted loop is a clone: it usually reuses the producer's own bound values,
// but a bound that was materialised inside the hierarchy comes out as a fresh
// arith.constant.
static bool sameTripCount(scf::ForOp a, scf::ForOp b) {
  auto same = [](Value x, Value y) {
    if (x == y)
      return true;
    APInt xc, yc;
    return matchPattern(x, m_ConstantInt(&xc)) &&
           matchPattern(y, m_ConstantInt(&yc)) && xc == yc;
  };
  return same(a.getLowerBound(), b.getLowerBound()) &&
         same(a.getUpperBound(), b.getUpperBound()) &&
         same(a.getStep(), b.getStep());
}

// The memrefs a loop's channel endpoints touch.
static llvm::SmallSetVector<Value, 4> channelMemrefsIn(scf::ForOp loop) {
  llvm::SmallSetVector<Value, 4> memrefs;
  loop.walk([&](air::ChannelInterface ci) { memrefs.insert(ci.getMemref()); });
  return memrefs;
}

// Merge a hoisted transfer's rebuilt guard into the identical guard already
// standing in front of it.
//
// cloneIndexSwitchUsingRemap rebuilds the guard around a hoisted transfer
// rather than flattening it, which is right -- flattening would make a copy
// written on one arm issue on every arm. But it rebuilds unconditionally, so a
// design whose producer is already guarded ends up with two switches on the
// same condition, the fill in one arm and the derived drain in the other's.
// They cannot then be fused as loops: a value defined in one scf.index_switch
// arm is invisible from a sibling switch's arm, so the producer loop's result
// can never stand in for the hoisted loop's. The guards have to become one
// first; after that the two loops are ordinary siblings and the fusion below
// handles them.
//
// Moving the later guard's work earlier is only sound when nothing it reads was
// produced in between, and nothing in between touches the buffers it moves --
// both are checked. Everything else is left where it is.
static void mergeHoistedGuards(func::FuncOp f, DominanceInfo &dom) {
  auto hasHoistedLoop = [](Operation *op) {
    bool found = false;
    op->walk([&](scf::ForOp forOp) {
      auto a = forOp->getAttrOfType<StringAttr>("loop-carried-dep");
      if (a && a.getValue() == "hoistedLoop")
        found = true;
    });
    return found;
  };
  auto singleBlockArms = [](Operation *op) {
    for (Region &r : op->getRegions())
      if (!r.hasOneBlock())
        return false;
    return op->getNumRegions() > 0;
  };

  SmallVector<Operation *> guards;
  f.walk([&](Operation *op) {
    if (isa<scf::IndexSwitchOp, scf::IfOp>(op) && hasHoistedLoop(op) &&
        singleBlockArms(op))
      guards.push_back(op);
  });

  for (Operation *late : guards) {
    // One token in, one token out, or the yield rewiring below has no meaning.
    if (late->getNumResults() > 1 ||
        (late->getNumResults() == 1 &&
         !isa<air::AsyncTokenType>(late->getResult(0).getType())))
      continue;
    auto memrefs = channelMemrefsUnder(late);

    Operation *early = nullptr;
    for (Operation *e = late->getPrevNode(); e; e = e->getPrevNode()) {
      // Same condition is not enough: a design guards several independent
      // feeds on one predicate, so the first switch scanned backwards is
      // usually somebody else's. The one we want is the one holding an endpoint
      // on a buffer this transfer touches.
      if (sameGuard(late, e) && singleBlockArms(e) &&
          e->getNumResults() == late->getNumResults()) {
        // The endpoint that qualifies `e` must be one the front end wrote, not
        // another hoisted copy waiting its turn -- otherwise the first two
        // guards pair off with each other and the producer is never reached.
        // A guard that has ALREADY absorbed a hoisted transfer still qualifies,
        // which is what lets a fan of several transfers collect in one place.
        bool sharesBuffer = false;
        e->walk([&](air::ChannelInterface ci) {
          if (!memrefs.contains(ci.getMemref()))
            return;
          for (Operation *a = ci->getParentOp(); a && a != e;
               a = a->getParentOp()) {
            auto attr = a->getAttrOfType<StringAttr>("loop-carried-dep");
            if (attr && attr.getValue() == "hoistedLoop")
              return;
          }
          sharesBuffer = true;
        });
        if (sharesBuffer) {
          early = e;
          break;
        }
      }
      if (touchesAny(e, memrefs))
        break;
    }
    if (!early)
      continue;

    // Nothing the late guard's ARMS read may have been defined after the early
    // one. The guard's own condition is deliberately not part of this: it is a
    // clone that CSE would fold into the early guard's condition anyway, and it
    // dies with the op being erased.
    bool readsSomethingInBetween = false;
    for (Region &r : late->getRegions())
      r.walk([&](Operation *inner) {
        for (Value operand : inner->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (!def || late->isProperAncestor(def))
            continue;
          if (!dom.properlyDominates(def, early))
            readsSomethingInBetween = true;
        }
      });
    if (readsSomethingInBetween)
      continue;

    OpBuilder builder(late);
    for (unsigned i = 0, e = late->getNumRegions(); i < e; ++i) {
      Block *src = &late->getRegion(i).front();
      Block *dst = &early->getRegion(i).front();
      Operation *dstTerm = dst->getTerminator();
      Operation *srcTerm = src->getTerminator();

      SmallVector<Operation *> body;
      for (Operation &op : src->without_terminator())
        body.push_back(&op);
      for (Operation *op : body)
        op->moveBefore(dstTerm);

      // The arm now does both halves, so it must wait on both. Dep tracing
      // reruns over this and will prune whatever is redundant.
      if (dstTerm->getNumOperands() == 1 && srcTerm->getNumOperands() == 1) {
        builder.setInsertionPoint(dstTerm);
        SmallVector<Value> deps{dstTerm->getOperand(0), srcTerm->getOperand(0)};
        auto joined = air::WaitAllOp::create(
            builder, late->getLoc(),
            air::AsyncTokenType::get(late->getContext()), deps);
        dstTerm->setOperand(0, joined.getAsyncToken());
      }
    }
    for (auto [oldRes, newRes] :
         llvm::zip(late->getResults(), early->getResults()))
      oldRes.replaceAllUsesWith(newRes);
    late->erase();
  }
}

// Fuse a hoisted transfer's freshly built loop into the loop that fills the
// buffer it reads.
//
// Hoisting a transfer out of a hierarchy clones its whole enclosing loop nest,
// so a design whose producer is one loop --
//
//   scf.for { %b = alloc; air.channel.get @fill (%b); air.channel.put @drain
//   (%b) }
//
// -- comes back as two sibling loops over the same buffer with the same trip
// count when the drain half is spelled as an air.dma_memcpy_nd instead: the
// clone has no way to know its transfer belongs in the loop already standing
// next to it. Two things then go wrong, and only the first is visible:
//
//   - The derived put's only incoming dependency is the buffer's alloc token,
//     not the get that writes the buffer. The RAW edge is simply absent.
//   - air-fuse-alloc-dealloc can no longer sink the alloc into a loop, because
//     the uses are split across two. air-label-scf-for-to-ping-pong keys on a
//     loop owning its buffer, so it skips both, and a producer that should be a
//     ring of N independently locked slots is emitted as N/2 double-buffered
//     pairs -- same buffers, same bytes, coarser synchronisation, and 6 fewer
//     locks per memtile.
//
// Fusing restores both. The guards below are what make it safe: identical trip
// count, a shared memref, and nothing in between that touches that memref (an
// intervening reader or writer would be reordered across the appended ops).
static void fuseHoistedLoopsIntoProducer(func::FuncOp f, DominanceInfo &dom) {
  // "hoist" is stripped by AIRHoistExternalAIRChannelPattern once the transfer
  // reaches its destination; "loop-carried-dep" is the marker that survives to
  // here, and is what the rest of the pipeline keys on too.
  auto isHoisted = [](Operation *op) {
    auto a = op->getAttrOfType<StringAttr>("loop-carried-dep");
    return a && a.getValue() == "hoistedLoop";
  };
  SmallVector<scf::ForOp> hoisted;
  f.walk([&](scf::ForOp forOp) {
    if (isHoisted(forOp))
      hoisted.push_back(forOp);
  });

  for (scf::ForOp newFor : hoisted) {
    // Async plumbing below rewires exactly one token through the fused body.
    if (newFor.getNumResults() != 1 ||
        !isa<air::AsyncTokenType>(newFor.getResult(0).getType()))
      continue;
    auto memrefs = channelMemrefsIn(newFor);
    if (memrefs.empty())
      continue;

    // An op may be stepped over on the way back only if it leaves our buffers
    // alone; anything that reads or writes one would be reordered across the
    // ops we are about to append.
    auto touchesOurMemrefs = [&](Operation *p) {
      bool touches = false;
      auto scan = [&](Operation *o) {
        for (Value operand : o->getOperands())
          if (memrefs.contains(operand))
            touches = true;
      };
      scan(p);
      p->walk(scan);
      return touches;
    };
    auto isCandidate = [&](Operation *p, Value &shared) {
      auto candidate = dyn_cast<scf::ForOp>(p);
      if (!candidate || isHoisted(candidate) ||
          candidate.getNumResults() != 1 ||
          !isa<air::AsyncTokenType>(candidate.getResult(0).getType()) ||
          !sameTripCount(candidate, newFor))
        return scf::ForOp();
      for (Value m : channelMemrefsIn(candidate))
        if (memrefs.contains(m)) {
          shared = m;
          return candidate;
        }
      return scf::ForOp();
    };
    // Scan one block backwards from `from` (exclusive), or from its end when
    // `from` is null.
    auto scanBack = [&](Block *block, Operation *from, Value &shared) {
      Operation *start = from ? from->getPrevNode()
                              : (block->empty() ? nullptr : &block->back());
      for (Operation *p = start; p; p = p->getPrevNode()) {
        if (scf::ForOp hit = isCandidate(p, shared))
          return hit;
        if (touchesOurMemrefs(p))
          break;
      }
      return scf::ForOp();
    };

    scf::ForOp producer;
    Value shared;
    producer = scanBack(newFor->getBlock(), newFor, shared);

    if (!producer)
      continue;

    // The body is about to move backwards, into the producer. Anything it reads
    // from outside itself has to already be available there. When it is not --
    // a hoisted transfer whose operands were materialised alongside it, past
    // the producer -- leave the pair alone rather than build invalid IR.
    bool dominates = true;
    newFor.getBody()->walk([&](Operation *inner) {
      for (Value operand : inner->getOperands()) {
        if (Operation *def = operand.getDefiningOp()) {
          if (newFor->isProperAncestor(def))
            continue;
          if (!dom.properlyDominates(def, producer))
            dominates = false;
        } else if (auto arg = dyn_cast<BlockArgument>(operand)) {
          if (arg.getOwner() != newFor.getBody() &&
              !dom.dominates(arg.getOwner(), producer->getBlock()))
            dominates = false;
        }
      }
    });
    if (!dominates)
      continue;

    // Land the reads before the producer tears the buffer down, not at the end
    // of its body: a loop that frees what it filled -- which is the norm, the
    // buffer is per-iteration -- would otherwise get its dealloc ordered ahead
    // of the transfers now reading it.
    Operation *producerYield = producer.getBody()->getTerminator();
    Operation *insertPt = producerYield;
    for (Operation &op : producer.getBody()->without_terminator()) {
      bool frees = false;
      op.walk([&](memref::DeallocOp d) {
        if (memrefs.contains(d.getMemref()))
          frees = true;
      });
      if (frees) {
        insertPt = &op;
        break;
      }
    }

    // The hoisted loop's iter_arg becomes the token of the last endpoint in the
    // producer that writes the buffer -- the RAW edge the clone lost. Falling
    // back to the loop's own iter_arg keeps the ordering no weaker than before.
    Value carriedIn = producer.getRegionIterArgs()[0];
    for (Operation &op : producer.getBody()->without_terminator()) {
      if (&op == insertPt)
        break;
      auto ci = dyn_cast<air::ChannelInterface>(&op);
      if (ci && memrefs.contains(ci.getMemref()) && op.getNumResults() &&
          isa<air::AsyncTokenType>(op.getResult(0).getType()))
        carriedIn = op.getResult(0);
    }
    Value newYield = newFor.getBody()->getTerminator()->getOperand(0);

    newFor.getInductionVar().replaceAllUsesWith(producer.getInductionVar());
    newFor.getRegionIterArgs()[0].replaceAllUsesWith(carriedIn);

    SmallVector<Operation *> body;
    for (Operation &op : newFor.getBody()->without_terminator())
      body.push_back(&op);
    for (Operation *op : body)
      op->moveBefore(insertPt);

    // The iteration now carries both halves of the feed.
    OpBuilder yb(producerYield);
    SmallVector<Value> joined{producerYield->getOperand(0), newYield};
    producerYield->setOperand(
        0, air::WaitAllOp::create(
               yb, producer.getLoc(),
               air::AsyncTokenType::get(producer.getContext()), joined)
               .getAsyncToken());
    newFor.getResult(0).replaceAllUsesWith(producer.getResult(0));
    newFor.erase();
  }
}

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
            dyn_cast_if_present<BaseMemRefType>(alloc.getMemref().getType());
        air::MemorySpace hierMemorySpace = air::MemorySpace::L3;
        air::HierarchyInterface hier_op =
            alloc->getParentOfType<air::HierarchyInterface>();
        if (hier_op && isa<air::HerdOp>(hier_op.getOperation()))
          hierMemorySpace = air::MemorySpace::L1;
        else if (hier_op && isa<air::SegmentOp>(hier_op.getOperation()))
          hierMemorySpace = air::MemorySpace::L2;
        else
          return;
        // If async, then log the execute op around alloc
        Operation *alloc_op =
            alloc->getParentOfType<air::ExecuteOp>()
                ? alloc->getParentOfType<air::ExecuteOp>().getOperation()
                : alloc.getOperation();
        auto allocMemSpace = air::getMemorySpace(memref_type);
        if (allocMemSpace &&
            air::isMoreLocal(hierMemorySpace, *allocMemSpace)) {
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
          auto src_type = llvm::dyn_cast_if_present<BaseMemRefType>(
              dma.getSrcMemref().getType());
          auto dst_type = llvm::dyn_cast_if_present<BaseMemRefType>(
              dma.getDstMemref().getType());
          if (dma->getParentOfType<air::HerdOp>()) {
            if (!air::isL1(src_type) && !air::isL1(dst_type))
              return false;
          }
          return true;
        });

    RewritePatternSet air_dma_demotion(context);
    air_dma_demotion.add<AIRDemoteDmaToAIRHierarchyConversion>(context);
    if (failed(applyPartialConversion(module, target_0,
                                      std::move(air_dma_demotion)))) {
      // No diagnostic here: applyPartialConversion already reported the
      // op it could not legalize, and the failing pattern reported why.
      // A contentless error on top of those only obscures them.
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
      // No diagnostic here: applyPartialConversion already reported the
      // op it could not legalize, and the failing pattern reported why.
      // A contentless error on top of those only obscures them.
      signalPassFailure();
    }

    // Hoist every "external" side channel to their respective air hierarchy.
    // For each channel op, hoist greedily until it reaches its corresponding
    // memory hierarchy.
    SetVector<air::ChannelInterface> externalChannelOps;
    for (auto f : funcOps) {
      f.walk([&externalChannelOps](air::ChannelInterface getput) {
        if (!isInMatchingHierarchy(getput))
          externalChannelOps.insert(getput);
      });
    }

    for (auto getput : externalChannelOps) {
      getput->setAttr("loop-carried-dep",
                      StringAttr::get(context, "internalGetPut"));
    }

    // Anchors CHAIN: a transfer can be anchored to a channel whose own external
    // half is derived by this pass too. In a real design the head of an arm's
    // feed block ends up entirely derived, each feed pinned behind the previous
    // one. Anchor resolution searches the live IR, so a target that has not
    // been hoisted yet is simply not found and the anchored transfer silently
    // falls back to the hierarchy's position. Hoist a channel before whatever
    // is anchored to it.
    //
    // Rank = length of the anchor chain back to a channel that is hand-written
    // or unanchored. A channel in a CYCLE keeps rank 0, i.e. its original
    // relative order: a cyclic chain has no correct order, and reordering it on
    // a guess would be worse than leaving it alone.
    llvm::StringMap<FlatSymbolRefAttr> anchorOfChan;
    llvm::DenseSet<StringRef> derivedChans;
    for (auto getput : externalChannelOps) {
      derivedChans.insert(getput.getChanName());
      auto a = getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_after");
      if (!a)
        a = getput->getAttrOfType<FlatSymbolRefAttr>("air.hoist_before");
      if (a)
        anchorOfChan[getput.getChanName()] = a;
    }
    llvm::StringMap<unsigned> rank;
    std::function<unsigned(StringRef, unsigned)> rankOf =
        [&](StringRef c, unsigned depth) -> unsigned {
      if (auto it = rank.find(c); it != rank.end())
        return it->second;
      // Deeper than the number of channels means a cycle was walked.
      if (depth > derivedChans.size())
        return 0;
      auto it = anchorOfChan.find(c);
      if (it == anchorOfChan.end() ||
          !derivedChans.count(it->second.getValue()))
        return 0;
      unsigned r = 1 + rankOf(it->second.getValue(), depth + 1);
      rank[c] = r;
      return r;
    };
    SmallVector<air::ChannelInterface> hoistOrder(externalChannelOps.begin(),
                                                  externalChannelOps.end());
    llvm::stable_sort(
        hoistOrder, [&](air::ChannelInterface a, air::ChannelInterface b) {
          return rankOf(a.getChanName(), 0) < rankOf(b.getChanName(), 0);
        });

    // Hoist one transfer per round, EXCEPT that transfers sharing a channel and
    // an enclosing loop go in the same round.
    //
    // One round marks one op "external" and runs the patterns, so the hoisting
    // pattern sees a batch of one and rebuilds the enclosing loop for that op
    // alone. Two rounds therefore produce two loops -- and two loops in
    // sequence do not mean what one loop containing two transfers meant. The
    // loop INTERLEAVES them (a, b, a, b, ...) and the pair of loops SERIALISES
    // them (a*N, then b*N). A channel is a FIFO, so its consumer's Nth transfer
    // takes the Nth arrival: serialising the producer sends arrival 1 where
    // arrival 2 belonged. Measured as wrong output on device, not as a
    // different schedule.
    //
    // Marking the whole group before the patterns run makes the batch complete,
    // and one clone of the hierarchy body reproduces the interleave exactly.
    // Grouping is by nearest enclosing loop as well as by channel: two
    // transfers on one channel in DIFFERENT loops are already ordered by those
    // loops and have nothing to preserve.
    auto nearestLoop = [](air::ChannelInterface gp) -> Operation * {
      for (Operation *p = gp->getParentOp(); p; p = p->getParentOp()) {
        if (isa<air::HierarchyInterface>(p))
          return nullptr;
        if (isa<LoopLikeOpInterface>(p))
          return p;
      }
      return nullptr;
    };
    // Membership only -- these handles are never dereferenced. An op hoisted in
    // an earlier round has been erased, and asking a dangling ChannelInterface
    // for its name is a segfault.
    llvm::SmallDenseSet<Operation *> queued;
    for (auto gp : hoistOrder)
      queued.insert(gp.getOperation());
    llvm::SmallDenseSet<Operation *> hoisted;
    for (auto getput : hoistOrder) {
      if (!hoisted.insert(getput.getOperation()).second)
        continue;
      getput->setAttr("loop-carried-dep", StringAttr::get(context, "external"));
      // Siblings come from a walk of the LIVE IR, not from hoistOrder, for the
      // same reason.
      if (auto *loop = nearestLoop(getput)) {
        auto name = getput.getChanName();
        loop->walk([&](air::ChannelInterface sibling) {
          if (sibling == getput)
            return;
          if (!queued.count(sibling.getOperation()))
            return;
          if (sibling.getChanName() != name)
            return;
          if (nearestLoop(sibling) != loop)
            return;
          if (!hoisted.insert(sibling.getOperation()).second)
            return;
          sibling->setAttr("loop-carried-dep",
                           StringAttr::get(context, "external"));
        });
      }
      RewritePatternSet hoistChannelPatterns(context);
      hoistChannelPatterns
          .add<AIRHoistExternalAIRChannelPattern<air::HerdOp>,
               AIRHoistExternalAIRChannelPattern<air::SegmentOp>>(context);
      (void)applyPatternsGreedily(module, std::move(hoistChannelPatterns));
    }

    // Put each hoisted transfer back in the loop that feeds it, before dep
    // tracing re-derives tokens over the result.
    for (auto f : funcOps) {
      DominanceInfo dom(f);
      mergeHoistedGuards(f, dom);
      fuseHoistedLoopsIntoProducer(f, dom);
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
        op->removeAttr("air.hoist_after");
        op->removeAttr("air.hoist_before");
        op->removeAttr("air.hoist_unguarded");
        op->removeAttr("air.hoist_outside_loops");
        op->removeAttr("air.derived_far_window");
      });
    }

    // Auto-detect channels that need packet switching. For each segment,
    // estimate per-column shim DMA pressure from L3-bound channels with
    // herd-side endpoints.
    //
    // The downstream shim DMA allocator deduplicates by channel declaration,
    // so each distinct channel consumes exactly ONE shim DMA slot. The
    // question is how many channels compete for the same shim column.
    //
    // Non-broadcast channels: all initially target the same column (via the
    // same_column allocation constraint). Per-column pressure equals their
    // count.
    //
    // Broadcast channels (those with broadcast_shape): each consumes one
    // shim DMA slot, but the allocator distributes them across available
    // columns within their broadcast column span (broadcast_shape[0]).
    // For K broadcast channels sharing column span C, worst-case per-column
    // pressure is ceil(K / C). Channels with different column spans are
    // grouped separately since they can only distribute within their own
    // span.
    //
    // Total per-column pressure:
    //   numNonBroadcast + sum_over_spans(ceil(count_i / span_i))
    //
    // Channels with only segment-level endpoints (L3<->L2) are globally
    // allocated across columns and do NOT create per-column pressure.
    //
    // Pre-existing dma_packet channels count toward pressure but are
    // not upgraded (already packet flow).
    module.walk([&](air::SegmentOp seg) {
      SmallVector<air::ChannelOp> inputChannels, outputChannels;
      int64_t preExistingInputPackets = 0, preExistingOutputPackets = 0;
      for (auto &op : module.getBody()->getOperations()) {
        auto chanOp = dyn_cast<air::ChannelOp>(op);
        if (!chanOp)
          continue;

        // mmio channels are runtime-sequence MMIO writes, not shim DMA, so
        // they neither contribute to per-column shim pressure nor are
        // eligible for dma_packet upgrade.
        if (chanOp.getChannelType() == "npu_mmio")
          continue;

        bool isAlreadyPacket = (chanOp.getChannelType() == "npu_dma_packet");
        auto channelName = chanOp.getSymName();

        // Check if this channel has a herd-side endpoint in this segment.
        bool hasHerdSideGet = false;
        bool hasHerdSidePut = false;
        seg.walk([&](air::ChannelInterface ci) -> WalkResult {
          if (hasHerdSideGet || hasHerdSidePut)
            return WalkResult::interrupt();
          if (ci.getChanName() != channelName)
            return WalkResult::advance();
          if (ci->getParentOfType<air::HerdOp>()) {
            if (isa<air::ChannelGetOp>(ci.getOperation()))
              hasHerdSideGet = true;
            else
              hasHerdSidePut = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        if (!hasHerdSideGet && !hasHerdSidePut)
          continue;

        // Verify the launch-side endpoint operates on an L3 memref.
        bool hasLaunchSideL3Put = false;
        bool hasLaunchSideL3Get = false;
        module.walk([&](air::ChannelInterface ci) -> WalkResult {
          if (hasLaunchSideL3Put || hasLaunchSideL3Get)
            return WalkResult::interrupt();
          if (ci.getChanName() != channelName)
            return WalkResult::advance();
          // Must be at launch level, not inside herd or segment.
          if (ci->getParentOfType<air::HerdOp>() ||
              ci->getParentOfType<air::SegmentOp>())
            return WalkResult::advance();
          auto memrefTy =
              dyn_cast_if_present<BaseMemRefType>(ci.getMemref().getType());
          if (!memrefTy || !air::isL3(memrefTy))
            return WalkResult::advance();
          if (isa<air::ChannelPutOp>(ci.getOperation()))
            hasLaunchSideL3Put = true;
          else
            hasLaunchSideL3Get = true;
          return WalkResult::interrupt();
        });

        // Input (L3->L1): herd-side get + launch-side L3 put.
        // Output (L1->L3): herd-side put + launch-side L3 get.
        if (hasHerdSideGet && hasLaunchSideL3Put) {
          if (isAlreadyPacket)
            preExistingInputPackets++;
          else
            inputChannels.push_back(chanOp);
        } else if (hasHerdSidePut && hasLaunchSideL3Get) {
          if (isAlreadyPacket)
            preExistingOutputPackets++;
          else
            outputChannels.push_back(chanOp);
        }
      }

      // Estimate per-column shim DMA pressure for a set of channels.
      // Broadcast channels can spread across their column span (first
      // dimension of broadcast_shape); non-broadcast channels all compete
      // for the same column. Broadcast channels are grouped by column span
      // since channels with different spans distribute independently.
      auto computePerColumnPressure =
          [](const SmallVector<air::ChannelOp> &channels,
             int64_t preExistingPackets) -> int64_t {
        int64_t numNonBroadcast = 0;

        // Group broadcast channels by their column span.
        llvm::SmallDenseMap<int64_t, int64_t> broadcastCountBySpan;

        for (auto chanOp : channels) {
          if (chanOp.isBroadcast()) {
            int64_t colSpan = 1;
            auto bcastShape = chanOp.getBroadcastShape();
            if (bcastShape && bcastShape.size() > 0) {
              if (auto colSpanAttr =
                      llvm::dyn_cast_if_present<IntegerAttr>(bcastShape[0]))
                colSpan = std::max((int64_t)1, colSpanAttr.getInt());
            }
            broadcastCountBySpan[colSpan]++;
          } else {
            numNonBroadcast++;
          }
        }

        // Per group: K channels spanning C columns have worst-case
        // per-column pressure ceil(K / C).
        int64_t broadcastPressure = 0;
        for (auto &[span, count] : broadcastCountBySpan)
          broadcastPressure += (count + span - 1) / span;

        return numNonBroadcast + broadcastPressure + preExistingPackets;
      };

      int64_t shimChannelsPerCol = clShimDmaChannelsPerCol;
      int64_t inputPressure =
          computePerColumnPressure(inputChannels, preExistingInputPackets);
      int64_t outputPressure =
          computePerColumnPressure(outputChannels, preExistingOutputPackets);

      auto upgradeToPacket = [&](SmallVector<air::ChannelOp> &channels,
                                 StringRef direction, int64_t pressure) {
        seg->emitWarning() << "auto-upgrading " << channels.size() << " "
                           << direction
                           << " channels to dma_packet (per-column pressure "
                           << pressure << " exceeds shim DMA limit of "
                           << shimChannelsPerCol << ")";
        for (auto chanOp : channels) {
          chanOp.setChannelType(StringAttr::get(context, "npu_dma_packet"));
        }
      };

      // Force mode: upgrade all shim-bound channels unconditionally.
      if (clForceShimPacketFlow) {
        if (!inputChannels.empty() || !outputChannels.empty())
          seg->emitRemark() << "force-upgrading "
                            << inputChannels.size() + outputChannels.size()
                            << " shim-bound channels to dma_packet";
        for (auto chanOp : inputChannels)
          chanOp.setChannelType(StringAttr::get(context, "npu_dma_packet"));
        for (auto chanOp : outputChannels)
          chanOp.setChannelType(StringAttr::get(context, "npu_dma_packet"));
        return;
      }

      if (inputPressure > shimChannelsPerCol)
        upgradeToPacket(inputChannels, "input", inputPressure);
      if (outputPressure > shimChannelsPerCol)
        upgradeToPacket(outputChannels, "output", outputPressure);
    });
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
      auto async_op =
          dyn_cast_if_present<air::AsyncOpInterface>(memcpy_op.getOperation());
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
          if (auto scf_par = dyn_cast_if_present<scf::ParallelOp>(parent)) {
            if (scf_par.getInitVals().size() &&
                scf_par.getInitVals()[0].getDefiningOp()) {
              sink_wait_all_op = dyn_cast_if_present<air::WaitAllOp>(
                  scf_par.getInitVals()[0].getDefiningOp());
            }
          } else if (auto scf_for = dyn_cast_if_present<scf::ForOp>(parent)) {
            if (scf_for.getInitArgs().size() &&
                scf_for.getInitArgs()[0].getDefiningOp()) {
              sink_wait_all_op = dyn_cast_if_present<air::WaitAllOp>(
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
                  dyn_cast_if_present<air::AsyncOpInterface>(
                      memcpy_op.getOperation()),
                  "RAW")))
        signalPassFailure();
      if (failed(
              depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
                  sink_op_memref_writes,
                  dyn_cast_if_present<air::AsyncOpInterface>(
                      memcpy_op.getOperation()),
                  "WAW/WAR")))
        signalPassFailure();
      // Detect tile index deps
      depTracer.traceTileIndices(
          sink_op_memref_reads, sink_op_memref_writes, sink_op_scalar_ins,
          sink_op_scalar_outs,
          dyn_cast_if_present<air::AsyncOpInterface>(memcpy_op.getOperation()));
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
