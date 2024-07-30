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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "dma-to-channel"

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

  wait_all_op_yielded->setAttr("hoist", StringAttr::get(ctx, "dep"));
  reduce_op->setAttr("hoist", StringAttr::get(ctx, "dep"));
  reduce_op.walk([&](mlir::Operation *o) {
    if (!isa<scf::YieldOp>(o)) {
      o->setAttr("hoist", StringAttr::get(ctx, "dep"));
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
  scf::ParallelOp scf_par = nullptr;
  if (isAsyncOp(herd)) {
    scf_par = builder.create<scf::ParallelOp>(loc, lbs, ubs, steps, deps_in);
    generateYieldAndOrReduceToScfLoop(builder, ctx, scf_par);
  } else
    scf_par = builder.create<scf::ParallelOp>(loc, lbs, ubs, steps);

  scf_par->setAttr("hoist", StringAttr::get(ctx, "hoistedLoop"));
  scf_par->setAttr("loop-carried-dep", StringAttr::get(ctx, "hoistedLoop"));

  return scf_par;
}

static SmallVector<Value, 1> getLoopTokens(scf::ForOp loop) {
  SmallVector<Value, 1> output;
  for (auto v : loop.getInitArgs()) {
    output.push_back(v);
  }
  return output;
}

static SmallVector<Value, 1> getLoopTokens(scf::ParallelOp loop) {
  SmallVector<Value, 1> output;
  for (auto v : loop.getInitVals()) {
    output.push_back(v);
  }
  return output;
}

static void getLeavesInDepGraph(Operation *op,
                                SmallVector<Value> &leaves_list) {
  Value token = nullptr;
  for (auto res : op->getResults())
    if (isa<air::AsyncTokenType>(res.getType()))
      token = res;
  if (token) {
    if (token.getUsers().empty()) {
      // Push back if unique
      if (std::find(leaves_list.begin(), leaves_list.end(), token) ==
          leaves_list.end()) {
        leaves_list.push_back(token);
      }
    } else {
      for (auto u : token.getUsers())
        getLeavesInDepGraph(u, leaves_list);
    }
  }
}

static void getLeavesInDepGraph(Value v, SmallVector<Value> &leaves_list) {
  for (auto u : v.getUsers())
    getLeavesInDepGraph(u, leaves_list);
}

static scf::YieldOp generateYieldAndOrReduceToScfLoop(OpBuilder builder,
                                                      MLIRContext *ctx,
                                                      scf::ForOp scf_loop) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToEnd(scf_loop.getBody());
  SmallVector<air::MemcpyInterface, 1> memcpy_ops;
  for (auto memcpy_op : scf_loop.getOps<air::MemcpyInterface>()) {
    if (memcpy_op->hasAttr("hoist")) {
      memcpy_ops.push_back(memcpy_op);
    }
  }
  assert(memcpy_ops.size() <= 1 &&
         "found multiple memcpy ops in one hoisted for loop");
  SmallVector<Value, 1> yield_token;
  if (memcpy_ops.size()) {
    assert(memcpy_ops[0]->getResult(0) &&
           "found sync memcpy op in async for loop");
    auto wa_op = builder.create<air::WaitAllOp>(
        builder.getUnknownLoc(), air::AsyncTokenType::get(ctx),
        SmallVector<Value, 1>{memcpy_ops[0]->getResult(0)});
    yield_token.push_back(wa_op.getAsyncToken());
    wa_op->setAttr("hoist", StringAttr::get(ctx, "dep"));
  } else {
    // Collect dangling leaves into yield
    SmallVector<Value> dep_list;
    getLeavesInDepGraph(scf_loop.getRegionIterArgs()[0], dep_list);
    auto wa_op = builder.create<air::WaitAllOp>(
        builder.getUnknownLoc(), air::AsyncTokenType::get(ctx), dep_list);
    yield_token.push_back(wa_op.getAsyncToken());
    wa_op->setAttr("hoist", StringAttr::get(ctx, "dep"));
  }
  scf::YieldOp output =
      builder.create<scf::YieldOp>(builder.getUnknownLoc(), yield_token);
  builder.restoreInsertionPoint(insertionCheckpoint);
  return output;
}

// Replace async op with wait_all op
static void replaceAsyncOpWithWaitAll(OpBuilder builder, IRMapping &remap,
                                      Operation *op, bool cloneDepList = true) {
  auto async_op = dyn_cast<air::AsyncOpInterface>(op);
  assert(async_op);
  SmallVector<Value> dep_list_remap;
  if (cloneDepList) {
    for (auto dep : async_op.getAsyncDependencies()) {
      dep_list_remap.push_back(remap.lookupOrDefault(dep));
    }
  }
  auto wa_op = builder.create<air::WaitAllOp>(
      builder.getUnknownLoc(), air::AsyncTokenType::get(op->getContext()),
      dep_list_remap);
  wa_op->setAttr("hoist", StringAttr::get(op->getContext(), "dep"));
  remap.map(async_op.getAsyncToken(), wa_op.getAsyncToken());
}

// Clone affine if's block with remap
static void
replaceAffineIfOpWithChannelOpAndClone(OpBuilder builder, IRMapping &remap,
                                       air::ChannelInterface externalGetPut) {
  for (Operation &child_op : externalGetPut->getBlock()->getOperations()) {
    if (!child_op.hasAttr("hoist"))
      continue;
    if (child_op.hasAttr("loop-carried-dep") &&
        child_op.getAttrOfType<StringAttr>("loop-carried-dep")
                .getValue()
                .str() == "internalGetPut")
      continue;
    builder.clone(child_op, remap);
  }
}

static Value lookupOrDefaultRange(Value v, IRMapping &remap) {
  return remap.lookupOrDefault(v);
}

static Operation *getCoreComputeOpFromExecuteOp(Operation *op) {
  // We assume all linalg ops (except for linalg.copy) and func.call ops do
  // computations only and do not participate in data movement.
  if (auto exec = dyn_cast<air::ExecuteOp>(op)) {
    if (isa<linalg::LinalgOp, func::CallOp>(exec.getChildOp()))
      return exec.getChildOp();
  }
  return nullptr;
}

static SmallVector<Value> lookupOrDefaultRange(SmallVector<Value> vec,
                                               IRMapping &remap) {
  SmallVector<Value> output;
  for (auto v : vec) {
    output.push_back(remap.lookupOrDefault(v));
  }
  return output;
}

template <typename T>
static LogicalResult
cloneScfLoopUsingRemap(OpBuilder builder, IRMapping &remap, T loop_op,
                       air::ChannelInterface externalGetPut = nullptr) {

  T new_loop_op =
      builder.create<T>(builder.getUnknownLoc(),
                        lookupOrDefaultRange(loop_op.getLowerBound(), remap),
                        lookupOrDefaultRange(loop_op.getUpperBound(), remap),
                        lookupOrDefaultRange(loop_op.getStep(), remap),
                        lookupOrDefaultRange(getLoopTokens(loop_op), remap));

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
      remap.map(std::get<0>(p).get<Value>(), std::get<1>(p).get<Value>());
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
  for (Operation &child_op : loop_op.getBody()->getOperations()) {
    if (!child_op.hasAttr("hoist"))
      continue;

    if (auto for_op = dyn_cast<LoopLikeOpInterface>(child_op)) {
      auto res = cloneScfLoopUsingRemap(builder, remap, for_op, externalGetPut);
      if (failed(res))
        return res;
    } else if (auto channel_op = dyn_cast<air::ChannelInterface>(child_op)) {
      if (child_op.hasAttr("loop-carried-dep") &&
          child_op.getAttrOfType<StringAttr>("loop-carried-dep")
                  .getValue()
                  .str() == "internalGetPut") {
        // Found channel op labelled as "internalGetPut", which shouldn't be
        // hoisted
        replaceAsyncOpWithWaitAll(builder, remap, &child_op, false);
      } else {
        builder.clone(child_op, remap);
      }
    } else if (externalGetPut && dyn_cast<affine::AffineIfOp>(child_op)) {
      // If externalGetPut is not nullptr, then broadcast lowering mode is on
      replaceAffineIfOpWithChannelOpAndClone(builder, remap, externalGetPut);
    } else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(child_op)) {
      if (child_op.hasAttr("loop-carried-dep"))
        builder.clone(child_op, remap);
      else
        replaceAsyncOpWithWaitAll(builder, remap, &child_op, false);
    } else if (getCoreComputeOpFromExecuteOp(&child_op)) {
      replaceAsyncOpWithWaitAll(builder, remap, &child_op, false);
    } else {
      builder.clone(child_op, remap);
    }
  }

  new_loop_op->setAttr("hoist",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));
  new_loop_op->setAttr("loop-carried-dep",
                       StringAttr::get(loop_op->getContext(), "hoistedLoop"));

  // Generate yield op and/or reduce op if async
  if (getLoopTokens(loop_op).size()) {
    generateYieldAndOrReduceToScfLoop(builder, loop_op->getContext(),
                                      new_loop_op);
  }

  return success();
}

template <>
LogicalResult cloneScfLoopUsingRemap<LoopLikeOpInterface>(
    OpBuilder builder, IRMapping &remap, LoopLikeOpInterface loop_op,
    air::ChannelInterface externalGetPut) {
  Operation *op = loop_op.getOperation();
  if (scf::ForOp fop = dyn_cast<scf::ForOp>(op)) {
    return cloneScfLoopUsingRemap(builder, remap, fop, externalGetPut);
  } else if (scf::ParallelOp pop = dyn_cast<scf::ParallelOp>(op)) {
    return cloneScfLoopUsingRemap(builder, remap, pop, externalGetPut);
  }
  return loop_op.emitOpError("unsupported loop type");
}

static unsigned getScfParDimIdFromBCastDma(air::MemcpyInterface memcpyOp) {
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
static std::string createChannelName(ModuleOp module) {
  std::string new_cname = "channel_0";
  std::string cname = "channel";
  int which_try = 0;
  while (module.lookupSymbol(new_cname))
    new_cname = cname + "_" + std::to_string(++which_try);
  cname = new_cname;
  return cname;
}

// Create channel symbol
static air::ChannelOp
createChannelOpWithBCast(OpBuilder builder, ModuleOp module, std::string cname,
                         Location loc, SmallVector<int64_t, 2> bcast_sizes) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  Operation *o = &module.getBody()->front();
  while (dyn_cast_or_null<air::ChannelOp>(o))
    o = o->getNextNode();
  builder.setInsertionPoint(o);

  auto channel_op = builder.create<air::ChannelOp>(
      loc, cname, builder.getI64ArrayAttr(bcast_sizes));

  builder.restoreInsertionPoint(insertionCheckpoint);

  return channel_op;
}

// Annotate post-broadcast shape
static void annotateChannelOpWithBCastShape(OpBuilder builder,
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

static void replaceAIRDmaWithAIRChannelPairs(
    OpBuilder &builder, unsigned innerMemorySpace, air::DmaMemcpyNdOp op,
    SmallVector<air::ChannelInterface, 1> &internalGetPutVector,
    SmallVector<air::ChannelInterface, 1> &externalGetPutVector) {
  auto loc = op->getLoc();
  auto src = op.getSrcMemref();
  auto dst = op.getDstMemref();
  auto ctx = op->getContext();

  auto src_type = llvm::dyn_cast<MemRefType>(src.getType());
  auto dst_type = llvm::dyn_cast<MemRefType>(dst.getType());
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
    channel_sizes[getScfParDimIdFromBCastDma(dyn_cast<air::MemcpyInterface>(
        op.getOperation()))] = ubs_int[0] - lbs_int[0] + 1;
    auto channel_op =
        createChannelOpWithBCast(builder, module, cname, loc, channel_sizes);
    annotateChannelOpWithBCastShape(builder, channel_op,
                                    op->getParentOfType<air::HerdOp>());
  } else {
    // Else, infer channel's input shape from parent spatial loop, i.e. herd if
    // within a herd, or scf.parallel if within an scf.parallel.
    SmallVector<int64_t, 2> channel_sizes = {1, 1};
    if (auto parent_herd_op = op->getParentOfType<air::HerdOp>()) {
      auto herd_size = parent_herd_op.getSizeOperands();
      for (unsigned i = 0; i < herd_size.size(); i++) {
        channel_sizes[i] =
            herd_size[i].getDefiningOp<arith::ConstantIndexOp>().value();
      }
    } else if (auto parent_par_op = op->getParentOfType<scf::ParallelOp>()) {
      SmallVector<int, 2> lbs_spatial, ubs_spatial;
      air::getSizesFromSpatialLoop(parent_par_op, lbs_spatial, ubs_spatial);
      for (unsigned i = 0; i < ubs_spatial.size(); i++)
        channel_sizes[i] = ubs_spatial[i] - lbs_spatial[i] + 1;
    }
    createChannelOpWithBCast(builder, module, cname, loc, channel_sizes);
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

  externalGetPutVector.push_back(externalGetPut);
  internalGetPutVector.push_back(internalGetPut);
}

static void HoistingAffineIf(affine::AffineIfOp op) {
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
  affine::AffineIfOp current_if = op;
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
  if (else_block_dma) {
    dmas.push_back(else_block_dma);
    module_builder.setInsertionPoint(else_block_dma);
    replaceAIRDmaWithAIRChannelPairs(module_builder, innerMemorySpace,
                                     else_block_dma, internalGetPut,
                                     externalGetPut);
  }

  // Get dependent ops to hoist together with external get/put
  SetVector<Operation *> backwardSlice;
  BackwardSliceOptions bsOptions{[&](Operation *o) { return o != hier_op; }};
  for (auto ext_channel_op : externalGetPut) {
    getBackwardSlice(ext_channel_op.getOperation(), &backwardSlice, bsOptions);

    for (auto parent = ext_channel_op->getParentOp();
         !isa<air::HierarchyInterface>(parent);
         parent = parent->getParentOp()) {
      getBackwardSlice(parent, &backwardSlice, bsOptions);
      backwardSlice.insert(parent);
    }
  }

  // Label dependent ops to hoist
  for (auto b : backwardSlice) {
    b->setAttr("hoist", StringAttr::get(ctx, "dep"));
    if (dyn_cast<air::ExecuteOp>(b)) {
      auto child_op = &(*b->getRegions().front().op_begin());
      child_op->setAttr("hoist", StringAttr::get(ctx, "dep"));
    }
  }

  // Hoist hierarchy op into scf op
  module_builder.setInsertionPoint(hier_op);
  MemRefType externalMemrefTy =
      llvm::cast<MemRefType>(externalGetPut[0].getMemref().getType());
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
    for (Operation &o : herd.getBody().getOps()) {
      if (isa<air::HerdTerminatorOp>(o))
        continue;
      if (!o.hasAttr("hoist"))
        continue;

      if (auto child_for_op = dyn_cast<LoopLikeOpInterface>(o)) {
        (void)cloneScfLoopUsingRemap(module_builder, remap, child_for_op,
                                     externalGetPut[dma_index]);
      } else if (dyn_cast<affine::AffineIfOp>(o)) {
        replaceAffineIfOpWithChannelOpAndClone(module_builder, remap,
                                               externalGetPut[dma_index]);
      } else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)) {
        if (o.hasAttr("loop-carried-dep") &&
            o.getAttrOfType<StringAttr>("loop-carried-dep").getValue().str() ==
                "internalGetPut") {
          // Found channel op labelled as "internalGetPut", which shouldn't be
          // hoisted
          replaceAsyncOpWithWaitAll(module_builder, remap, &o, false);
        } else {
          module_builder.clone(o, remap);
        }
      } else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(o)) {
        replaceAsyncOpWithWaitAll(module_builder, remap, &o, false);
      } else if (getCoreComputeOpFromExecuteOp(&o)) {
        replaceAsyncOpWithWaitAll(module_builder, remap, &o, false);
      } else {
        module_builder.clone(o, remap);
      }
    }
    dma_index++;
  }

  module.walk([&](mlir::Operation *o) {
    if (o->hasAttr("hoist")) {
      o->removeAttr("hoist");
    }
  });
  hier_op.walk([&](mlir::Operation *o) {
    if (o->hasAttr("loop-carried-dep") &&
        o->getAttrOfType<StringAttr>("loop-carried-dep").getValue().str() ==
            "external") {
      o->erase();
    }
  });
  for (auto &dma : dmas) {
    dma->erase();
  }
}

namespace {
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
    auto src_type = llvm::dyn_cast<MemRefType>(src.getType());
    auto dst_type = llvm::dyn_cast<MemRefType>(dst.getType());
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

    SmallVector<Value, 4> src_offsets = op.getSrcOffsets();
    SmallVector<Value, 4> dst_offsets = op.getDstOffsets();
    SmallVector<Value, 4> src_sizes = op.getSrcSizes();
    SmallVector<Value, 4> dst_sizes = op.getDstSizes();
    SmallVector<Value, 4> src_strides = op.getSrcStrides();
    SmallVector<Value, 4> dst_strides = op.getDstStrides();

    std::set<Operation *> erased;
    SmallVector<air::ChannelInterface, 1> externalGetPut;
    SmallVector<air::ChannelInterface, 1> internalGetPut;

    replaceAIRDmaWithAIRChannelPairs(rewriter, innerMemorySpace, op,
                                     internalGetPut, externalGetPut);

    {
      OpBuilder::InsertionGuard guard(rewriter);

      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions bsOptions{
          [&](Operation *o) { return o != hier_op; }};
      for (auto ext_channel_op : externalGetPut) {
        getBackwardSlice(ext_channel_op.getOperation(), &backwardSlice,
                         bsOptions);
      }

      for (auto parent = op->getParentOp();
           !isa<air::HierarchyInterface>(parent);
           parent = parent->getParentOp()) {
        getBackwardSlice(parent, &backwardSlice, bsOptions);
        backwardSlice.insert(parent);
      }

      // Hoist hierarchy op into scf op
      Operation *scf_loop = nullptr;
      mlir::OpBuilder::InsertPoint
          insertionPointAtHierOp; // To keep a record of the insertion point as
                                  // destination for hoisting
      rewriter.setInsertionPoint(hier_op);
      if (herd) {
        // Scf parallel shape is either herd shape, or channel set shape if
        // broadcasting
        SmallVector<int, 2> lbs;
        SmallVector<int, 2> ubs;
        auto module = op->getParentOfType<ModuleOp>();
        auto channel_op = dyn_cast<air::ChannelOp>(
            module.lookupSymbol(externalGetPut[0].getChanName()));
        auto size = extractFromIntegerArrayAttr<int64_t>(channel_op.getSize());
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

      auto backwardSliceCopy = backwardSlice;
      for (auto b : backwardSliceCopy) {
        if (dyn_cast<air::ExecuteOp>(b)) {
          for (auto &exec_child_op : b->getRegions().front().getOps()) {
            getBackwardSlice(&exec_child_op, &backwardSlice, bsOptions);
            backwardSlice.insert(&exec_child_op);
          }
        }
      }

      for (auto b : backwardSlice) {
        b->setAttr("hoist", StringAttr::get(ctx, "dep"));
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
        for (auto arg : herd.getKernelArguments())
          remap.map(arg, herd.getKernelOperand(arg_idx++));

        // Clone ops into hoisted scf.parallel
        rewriter.setInsertionPointToStart(scf_par.getBody());
        for (Operation &o :
             herd->getRegions().front().getBlocks().front().getOperations()) {
          if (isa<air::HerdTerminatorOp>(o))
            continue;
          if (!o.hasAttr("hoist"))
            continue;
          if (auto child_for_op = dyn_cast<LoopLikeOpInterface>(o)) {
            auto res = cloneScfLoopUsingRemap(rewriter, remap, child_for_op);
            if (failed(res))
              return res;
          } else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)) {
            if (o.hasAttr("loop-carried-dep") &&
                o.getAttrOfType<StringAttr>("loop-carried-dep")
                        .getValue()
                        .str() == "internalGetPut") {
              // Found channel op labelled as "internalGetPut", which
              // shouldn't be hoisted
              replaceAsyncOpWithWaitAll(rewriter, remap, &o, false);
            } else {
              rewriter.clone(o, remap);
            }
          } else if (getCoreComputeOpFromExecuteOp(&o)) {
            replaceAsyncOpWithWaitAll(rewriter, remap, &o, false);
          } else {
            rewriter.clone(o, remap);
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
          // When hoisting air.channel puts/gets from air.segment to air.launch,
          // any dependence to air.herd should drop. TODO: generalize this to
          // cover more event types.
          if (air::isAsyncOp(&o)) {
            for (auto operand : o.getOperands()) {
              auto depHerdOp =
                  dyn_cast_if_present<air::HerdOp>(operand.getDefiningOp());
              if (!depHerdOp)
                continue;
              auto checkpoint = rewriter.saveInsertionPoint();
              remap.map(depHerdOp.getAsyncToken(),
                        rewriter
                            .create<air::WaitAllOp>(
                                loc, air::AsyncTokenType::get(o.getContext()),
                                SmallVector<Value>{})
                            .getAsyncToken());
              rewriter.restoreInsertionPoint(checkpoint);
            }
          }

          if (!o.hasAttr("hoist"))
            continue;

          if (auto child_for_op = dyn_cast<LoopLikeOpInterface>(o)) {
            auto res = cloneScfLoopUsingRemap(rewriter, remap, child_for_op);
            if (failed(res))
              return res;
            continue;
          } else if (auto channel_op = dyn_cast<air::ChannelInterface>(o)) {
            if (o.hasAttr("loop-carried-dep") &&
                o.getAttrOfType<StringAttr>("loop-carried-dep")
                        .getValue()
                        .str() == "internalGetPut") {
              // Found channel op labelled as "internalGetPut", which
              // shouldn't be hoisted
              replaceAsyncOpWithWaitAll(rewriter, remap, &o, false);
              continue;
            }
          }
          rewriter.clone(o, remap);
        }
      }

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

} // namespace

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
      auto memref_type = llvm::dyn_cast<MemRefType>(memref.getType());

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

namespace {
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
    auto src_type = llvm::dyn_cast<MemRefType>(src.getType());
    auto dst_type = llvm::dyn_cast<MemRefType>(dst.getType());
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
      getBackwardSlice(op.getOperation(), &backwardSlice, bsOptions);

      if (hoist_herd) {
        // Transitive defs up to air.herd.
        BackwardSliceOptions bsOptionsHoistHerd{
            [&](Operation *o) { return o != hier_op; }};
        for (auto parent = op->getParentOp();
             !isa<air::HierarchyInterface>(parent);
             parent = parent->getParentOp()) {
          getBackwardSlice(parent, &backwardSlice, bsOptionsHoistHerd);
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
        if (auto execOp = dyn_cast<air::ExecuteOp>(b)) {
          getBackwardSlice(execOp.getChildOp(), &backwardSlice, bsOptions);
          backwardSlice.insert(execOp.getChildOp());
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
        SmallVector<int, 2> lbs;
        SmallVector<int, 2> ubs;
        auto size = herd.getSizeOperands();
        for (auto s : size) {
          lbs.push_back(0);
          ubs.push_back(*mlir::getConstantIntValue(s));
        }
        scf_par = hoistHerdToAsyncParallel(rewriter, loc, ctx, herd, lbs, ubs);
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
        for (Operation &o :
             herd->getRegions().front().getBlocks().front().getOperations()) {
          if (isa<air::HerdTerminatorOp>(o))
            continue;
          if (!o.hasAttr("hoist"))
            continue;

          if (auto child_for_op = dyn_cast<LoopLikeOpInterface>(o)) {
            auto res = cloneScfLoopUsingRemap(rewriter, remap, child_for_op);
            if (failed(res))
              return res;
          } else if (getCoreComputeOpFromExecuteOp(&o)) {
            replaceAsyncOpWithWaitAll(rewriter, remap, &o, false);
          } else {
            rewriter.clone(o, remap);
          }
        }

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
        auto memref_type = dyn_cast<MemRefType>(alloc.getMemref().getType());
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

    // Hoist broadcast pattern
    for (auto f : funcOps) {
      f.walk([&](affine::AffineIfOp op) {
        if (!op->getParentOfType<affine::AffineIfOp>()) {
          // Only hoist top-level affine if op with a nest of if ops
          HoistingAffineIf(op);
        }
      });
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
              llvm::dyn_cast<MemRefType>(dma.getSrcMemref().getType());
          auto dst_type =
              llvm::dyn_cast<MemRefType>(dma.getDstMemref().getType());
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
    f.walk([&](air::MemcpyInterface memcpy_op) {
      if (memcpy_op->getAttrOfType<StringAttr>("loop-carried-dep") &&
          memcpy_op->getAttrOfType<StringAttr>("loop-carried-dep")
                  .getValue()
                  .str() == "external") {

        // Start tracing dependency only if this put/get op is async
        auto async_op =
            dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation());
        if (!async_op)
          return;

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
            memcpy_op.getOperation(), sink_op_memref_reads,
            sink_op_memref_writes, sink_op_scalar_ins, sink_op_scalar_outs);

        assert((sink_op_memref_reads.size() || sink_op_memref_writes.size()) &&
               "cannot read memref from channel op");

        if (sink_wait_all_op) {
          // Detect RAW deps
          depTracer.template traceDependencyFromOp<air::WaitAllOp>(
              sink_op_memref_reads, sink_wait_all_op, "RAW");
          // Detect WAW and WAR deps
          depTracer.template traceDependencyFromOp<air::WaitAllOp>(
              sink_op_memref_writes, sink_wait_all_op, "WAW/WAR");

          // Rebuild loop-carried dependency in scf loop nest
          air::clearAsyncDependenciesOfAsyncOp(memcpy_op);
          depTracer.reconnectLoopCarriedDependencyFromOp(
              memcpy_op.getOperation());
        }

        // Trace dependency of external put/get within scf loop
        depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
            sink_op_memref_reads,
            dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation()), "RAW");
        depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
            sink_op_memref_writes,
            dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation()),
            "WAW/WAR");
        // Detect tile index deps
        depTracer.traceTileIndices(
            sink_op_memref_reads, sink_op_memref_writes, sink_op_scalar_ins,
            sink_op_scalar_outs,
            dyn_cast<air::AsyncOpInterface>(memcpy_op.getOperation()));
      }
    });
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createDmaToChannelPass() {
  return std::make_unique<DmaToChannelPass>();
}

} // namespace air
} // namespace xilinx
