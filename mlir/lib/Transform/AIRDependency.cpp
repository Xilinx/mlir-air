//===- AIRDependency.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRDependency.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Dependency.h"
#include "air/Util/DirectedAdjacencyMap.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;
#define DEBUG_TYPE "air-dependency"

namespace xilinx {
namespace air {

// Remove an op if it has no users, else return failure.
// This is a temporary measure, while the issue
// https://github.com/Xilinx/mlir-air/issues/372
// is open. Once the root cause is found, there should be no ops erased here
// whose results have users.
LogicalResult eraseOpWithCheck(RewriterBase &rewriter, Operation *op,
                               std::string_view context = "") {
  for (auto opResult : op->getResults()) {
    for (auto &&user : opResult.getUsers()) {
      auto result =
          op->emitOpError("is being erased, but it has at least one user.");
      result.attachNote(user->getLoc()) << "erased op has user:\n" << *user;
      result.attachNote(op->getLoc())
          << "additional context:'" << context << "'\n";
      return result;
    }
  }

  rewriter.eraseOp(op);
  return success();
}

// Construction of a dependency graph

struct executeNode {
  std::string asyncEventName;
  std::string asyncEventType;
  std::string color;
  std::string shape;
  std::string style;
  unsigned operationId;
};

using ExecuteGraph = air::TypedDirectedAdjacencyMap<air::executeNode>;

typedef std::map<ExecuteGraph::VertexId, ExecuteGraph::VertexId> vertex_map;
typedef std::map<unsigned, ExecuteGraph::VertexId> operation_id_to_vertex_map;

// A channel put and get that move data to each other share a transfer. When
// one reads a buffer the other writes, a token edge between them deadlocks
// once both are async: the get waits for the put to finish, and the put for
// the get to take its data. Whether they need one is decided by the stream
// steps at which the buffer's elements are read and written: an element read
// at step i and written at step j is read before it is written iff i <= j
// (D36(a)), so the transfer orders the two itself when that is the program's
// order. If a put must read an element before a get overwrites it but the
// stream writes it first, no async version is correct: the slot's gets stay
// synchronous, which buffers its transfers (D30).
//
// This holds for a put and get on one slot whether or not they move data to
// each other: through the slot's FIFO order, an edge between any two of them
// can close a cycle.
//
// The steps are known only when every endpoint of the slot sits in one block,
// runs once, in program order, with constant patterns on whole buffers. Other
// slots are left to the memory tracing as before.
struct ChannelTransfers {
  // Put and get pairs that need no memory edge.
  llvm::DenseSet<std::pair<Operation *, Operation *>> unordered;
  // Gets of slots whose transfers need a buffer.
  llvm::SmallPtrSet<Operation *, 4> keepSync;
};

// The buffer elements a channel op moves, in stream order, if its pattern is
// constant (an empty pattern is the whole static buffer).
static std::optional<SmallVector<int64_t>>
streamElements(air::ChannelInterface chan) {
  auto type = dyn_cast<MemRefType>(chan.getMemref().getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  constexpr int64_t limit = 1 << 16;
  auto offsets = chan.getMixedOffsets();
  auto sizes = chan.getMixedSizes();
  auto strides = chan.getMixedStrides();
  SmallVector<int64_t> elems;
  if (sizes.empty()) {
    if (type.getNumElements() > limit)
      return std::nullopt;
    return llvm::to_vector(llvm::seq<int64_t>(0, type.getNumElements()));
  }
  if (offsets.size() != sizes.size() || strides.size() != sizes.size())
    return std::nullopt;
  SmallVector<int64_t> o, n, st;
  int64_t count = 1;
  for (auto [off, sz, str] : llvm::zip(offsets, sizes, strides)) {
    auto oc = getConstantIntValue(off), nc = getConstantIntValue(sz),
         sc = getConstantIntValue(str);
    if (!oc || !nc || !sc || *nc <= 0)
      return std::nullopt;
    o.push_back(*oc);
    n.push_back(*nc);
    st.push_back(*sc);
    count *= *nc;
    if (count > limit)
      return std::nullopt;
  }
  SmallVector<int64_t> idx(n.size(), 0);
  for (int64_t k = 0; k < count; k++) {
    int64_t pos = 0;
    for (unsigned d = 0; d < n.size(); d++)
      pos += (o[d] + idx[d]) * st[d];
    elems.push_back(pos);
    for (int d = n.size() - 1; d >= 0; d--) {
      if (++idx[d] < n[d])
        break;
      idx[d] = 0;
    }
  }
  return elems;
}

static ChannelTransfers analyzeChannelTransfers(ModuleOp module) {
  ChannelTransfers result;
  // Endpoints per function and channel, in program order. Each function is a
  // program of its own, with streams of its own.
  llvm::MapVector<std::pair<Operation *, StringRef>,
                  SmallVector<air::ChannelInterface>>
      byChannel;
  module.walk([&](air::ChannelInterface chan) {
    byChannel[{chan->getParentOfType<func::FuncOp>(), chan.getChanName()}]
        .push_back(chan);
  });
  // Runs once, in its block's program order: no loop, branch or execute
  // around it, and every hierarchy op around it of one instance.
  auto runsOnce = [](Operation *op) {
    for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
      if (isa<func::FuncOp>(p))
        return true;
      if (auto hier = dyn_cast<air::HierarchyInterface>(p)) {
        if (!llvm::all_of(hier.getSizeOperands(),
                          [](Value v) { return getConstantIntValue(v) == 1; }))
          return false;
        continue;
      }
      return false;
    }
    return false;
  };
  for (auto &[key, endpoints] : byChannel) {
    auto decl = air::getChannelDeclarationThroughSymbol(endpoints.front());
    if (!decl || decl->hasAttr("broadcast_shape"))
      continue;
    // Slots by constant indices; a non-constant index or a routed put leaves
    // the whole channel alone.
    std::map<SmallVector<int64_t>, SmallVector<air::ChannelInterface>> slots;
    bool known = true;
    for (air::ChannelInterface chan : endpoints) {
      SmallVector<int64_t> key;
      for (Value i : chan.getIndices()) {
        auto c = getConstantIntValue(i);
        if (!c) {
          known = false;
          break;
        }
        key.push_back(*c);
      }
      if (auto put = dyn_cast<air::ChannelPutOp>(chan.getOperation()))
        known &= !put.getDest();
      if (!known)
        break;
      slots[key].push_back(chan);
    }
    if (!known)
      continue;
    for (auto &[indices, slot] : slots) {
      Block *block = slot.front()->getBlock();
      if (!llvm::all_of(slot, [&](air::ChannelInterface c) {
            return c->getBlock() == block && runsOnce(c) &&
                   air::resolveBufferRoot(c.getMemref()) == c.getMemref();
          }))
        continue;
      // Stream step of each endpoint's first element, per direction.
      struct Side {
        air::ChannelInterface op;
        int64_t first;
        SmallVector<int64_t> elems;
      };
      SmallVector<Side> puts, gets;
      int64_t putStep = 0, getStep = 0;
      bool ok = true;
      for (air::ChannelInterface c : slot) {
        auto elems = streamElements(c);
        if (!elems) {
          ok = false;
          break;
        }
        bool isPut = isa<air::ChannelPutOp>(c.getOperation());
        int64_t &step = isPut ? putStep : getStep;
        (isPut ? puts : gets).push_back({c, step, *elems});
        step += elems->size();
      }
      if (!ok)
        continue;
      // Each conflict: the put reads an element at step i that the get
      // writes at step j. The stream orders the two as the program does when
      // the put comes first and i <= j, or the get comes first and j < i.
      // A put first with i > j reads, in the program, what is there before a
      // write the stream has already made.
      bool needsBuffer = false;
      SmallVector<std::pair<Operation *, Operation *>> ordered;
      for (Side &g : gets) {
        if (needsBuffer)
          break;
        llvm::DenseMap<int64_t, int64_t> writeStep;
        for (auto [k, e] : llvm::enumerate(g.elems))
          writeStep.try_emplace(e, g.first + k);
        for (Side &pt : puts) {
          if (pt.op.getMemref() != g.op.getMemref())
            continue;
          bool putFirst = pt.op->isBeforeInBlock(g.op);
          bool conflict = false, consistent = true;
          for (auto [k, e] : llvm::enumerate(pt.elems)) {
            auto it = writeStep.find(e);
            if (it == writeStep.end())
              continue;
            conflict = true;
            int64_t i = pt.first + k, j = it->second;
            if (putFirst && i > j)
              needsBuffer = true;
            if (putFirst ? i > j : j >= i)
              consistent = false;
          }
          if (needsBuffer)
            break;
          if (conflict && consistent)
            ordered.push_back({pt.op, g.op});
        }
      }
      // A buffered get still waits for the puts before it that read its
      // buffer, and the puts it waits for are buffered only if every get on
      // the slot is synchronous.
      if (needsBuffer) {
        for (Side &g : gets)
          result.keepSync.insert(g.op);
        continue;
      }
      for (auto pair : ordered)
        result.unordered.insert(pair);
    }
  }
  return result;
}

class AIRDependency : public air::impl::AIRDependencyBase<AIRDependency> {

public:
  AIRDependency() = default;
  AIRDependency(const AIRDependency &pass) = default;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();

    // Preprocessing: renumber the air dma op ids
    for (auto f : module.getOps<func::FuncOp>()) {
      air::renumberMemcpyIfOps(&f.getRegion());
    }

    // 1st traversal: create async ops with empty dep list.

    IRRewriter rewriter(module.getContext());
    rewriter.setInsertionPoint(module);

    ExecuteOpID = 0;
    HierarchyOpID = 0;
    WaitAllOpID = 0;
    ChannelOpID = 0;
    DmaOpID = 0;

    transfers = analyzeChannelTransfers(module);

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (air::isAsyncOp(op) || isa<air::WaitAllOp>(op)) {
          assignOpId(op);
          updateAsyncExecuteGraphWithNewNode(op, asyncExecuteGraph);
          return; // Skip if is already async.
        }
        if (op->getParentOfType<linalg::LinalgOp>())
          return; // Skip if is inside a linalg.generic.
        if (op->getParentOfType<air::ExecuteOp>()) {
          // Already inside an async region. A transfer there would finish
          // after the execute that holds it.
          if (isa<air::DmaMemcpyNdOp, air::ChannelInterface>(op)) {
            op->emitOpError("inside an air.execute is not supported");
            signalPassFailure();
          }
          return;
        }

        if (isa<air::DmaMemcpyNdOp>(op))
          createAsyncDMA(rewriter, op);
        else if (isa<air::ChannelInterface>(op)) {
          if (!transfers.keepSync.contains(op))
            createAsyncChannel(rewriter, op);
        } else if (isa<linalg::LinalgOp, func::CallOp, memref::DeallocOp>(op)) {
          if (op->getNumResults())
            createAsyncExecute(rewriter, op,
                               op->getResults().front().getType());
          else
            createAsyncExecute(rewriter, op);
        } else if (isa<memref::CopyOp>(op)) {
          // Inside a branch op too: ops after the branch wait for the branch
          // op's token (addBranchDeps), so the copy is ordered like any
          // other access.
          createAsyncExecute(rewriter, op);
        } else if (auto hierarchy_op =
                       dyn_cast_if_present<air::HierarchyInterface>(op))
          createAsyncHierarchyImpls(rewriter, hierarchy_op);
        // Create async execute region for memref.alloc
        else if (auto memalloc_op = dyn_cast_if_present<memref::AllocOp>(op)) {
          // Alloc can be used to specify shapes for operations such
          // as reshape ops. If this alloc is used to specify shape of
          // a reshap op, ignore this operation.
          if (!alloc_for_reshape(memalloc_op->getOpResult(0)))
            createAsyncExecute(rewriter, op, memalloc_op.getMemref().getType());
        }

        // Create async execute region for an unknown op which has memref or
        // index-type operands
        else {
          bool isCandidateExecute = false;
          for (auto operand : op->getOperands()) {
            if (llvm::isa<BaseMemRefType>(operand.getType())) {
              isCandidateExecute = true;
            }
          }
          // If a memref store is storing to an alloca for shape,
          // it must not be executed.
          if (auto storeOp = dyn_cast_if_present<memref::StoreOp>(op))
            if (alloc_for_reshape(storeOp.getMemRef()))
              isCandidateExecute = false;
          // No air execute for expand, collapse and reshape ops
          if (isa_and_present<memref::ReshapeOp, memref::ExpandShapeOp,
                              memref::CollapseShapeOp>(op)) {
            isCandidateExecute = false;
          }
          // No air execute for loop ops
          if (isa<mlir::LoopLikeOpInterface>(op))
            isCandidateExecute = false;
          // No air execute for subview ops
          if (isa<mlir::OffsetSizeAndStrideOpInterface>(op))
            isCandidateExecute = false;
          // No air execute for memref.extract_strided_metadata and
          // memref.AssumeAlignmentOp ops.
          if (isa<memref::ExtractStridedMetadataOp, memref::AssumeAlignmentOp>(
                  op))
            isCandidateExecute = false;
          // No air execute for terminators
          if (op->mightHaveTrait<OpTrait::IsTerminator>()) {
            isCandidateExecute = false;
          }
          if (isCandidateExecute) {
            if (op->getNumResults())
              createAsyncExecute(rewriter, op,
                                 op->getResults().front().getType());
            else
              createAsyncExecute(rewriter, op);
          }
        }
      });
    }

    // The async endpoints are new ops: pair them again.
    transfers.unordered = analyzeChannelTransfers(module).unordered;

    // 2nd traversal: trace deps among async execute regions; build a dep graph

    llvm::DenseMap<std::pair<StringRef, int>, Operation *> opIdToOpMap;
    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        // A get kept synchronous is a blocking event, waited for below.
        if (transfers.keepSync.contains(op))
          return;
        Operation *sink_op = nullptr;
        if (auto async_execute_op = dyn_cast_if_present<air::ExecuteOp>(op)) {
          for (auto &child_op : async_execute_op.getChildOps())
            if (!dyn_cast_if_present<air::ExecuteTerminatorOp>(child_op))
              sink_op = &child_op;
        } else if (isa<air::AsyncOpInterface>(op))
          sink_op = op;
        else
          return;

        // Preserve any existing async dependency edges in the input IR, before
        // creating new ones.
        if (auto execute_op = dyn_cast_if_present<air::ExecuteOp>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::ExecuteOp>(dep, execute_op);
          }
        } else if (auto dma_op = dyn_cast_if_present<air::DmaMemcpyNdOp>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::DmaMemcpyNdOp>(dep, dma_op);
          }
        } else if (auto channel_op =
                       dyn_cast_if_present<air::ChannelInterface>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::ChannelInterface>(dep, channel_op);
          }
        } else if (auto hier_op =
                       dyn_cast_if_present<air::HierarchyInterface>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::HierarchyInterface>(dep, hier_op);
          }
        }

        SmallVector<partialMemref, 1> sink_op_memref_reads;
        SmallVector<partialMemref, 1> sink_op_memref_writes;
        SmallVector<Value, 1> sink_op_scalar_ins;
        SmallVector<Value, 1> sink_op_scalar_outs;

        dependencyTracer DTObject;
        // An execute accesses what every op in its body does; one this pass
        // made holds a single op, one in the input may hold several.
        SmallVector<Operation *> accessors{sink_op};
        if (auto async_execute_op = dyn_cast_if_present<air::ExecuteOp>(op)) {
          accessors.clear();
          for (auto &child_op : async_execute_op.getChildOps())
            if (!isa<air::ExecuteTerminatorOp>(child_op))
              accessors.push_back(&child_op);
        }
        for (Operation *accessor : accessors)
          DTObject.getPartialMemrefFromOp(
              accessor, sink_op_memref_reads, sink_op_memref_writes,
              sink_op_scalar_ins, sink_op_scalar_outs);

        // Detect dependencies
        if (auto async_execute_op = dyn_cast_if_present<air::ExecuteOp>(op)) {
          // Detect RAW deps
          traceDeps<air::ExecuteOp>(sink_op_memref_reads, async_execute_op,
                                    "RAW");
          // Detect WAW and WAR deps
          traceDeps<air::ExecuteOp>(sink_op_memref_writes, async_execute_op,
                                    "WAW/WAR");
          // Detect tile index deps
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs,
                           async_execute_op);
          // Keep track of processed async execute region ops. Deps should point
          // to the past, not future.
          opIdToOpMap[std::make_pair("execute", async_execute_op.getId())] =
              async_execute_op;
        } else if (auto dma_op =
                       mlir::dyn_cast_if_present<air::DmaMemcpyNdOp>(op)) {
          traceDeps<air::DmaMemcpyNdOp>(sink_op_memref_reads, dma_op, "RAW");
          traceDeps<air::DmaMemcpyNdOp>(sink_op_memref_writes, dma_op,
                                        "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs, dma_op);
          opIdToOpMap[std::make_pair("dma", dma_op.getId())] = dma_op;
        } else if (auto channel_op =
                       mlir::dyn_cast_if_present<air::ChannelInterface>(op)) {
          traceDeps<air::ChannelInterface>(sink_op_memref_reads, channel_op,
                                           "RAW");
          traceDeps<air::ChannelInterface>(sink_op_memref_writes, channel_op,
                                           "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs, channel_op);
          traceChannelProducerHierarchyDep(channel_op);
          opIdToOpMap[std::make_pair("channel", channel_op.getId())] =
              channel_op;
        } else if (auto hier_op =
                       dyn_cast_if_present<air::HierarchyInterface>(op)) {
          traceDeps<air::HierarchyInterface>(sink_op_memref_reads, hier_op,
                                             "RAW");
          traceDeps<air::HierarchyInterface>(sink_op_memref_writes, hier_op,
                                             "WAW/WAR");
          opIdToOpMap[std::make_pair("hierarchy", hier_op.getId())] = hier_op;
        }
      });
    }

    // 3rd traversal: perform transitive reduction on dependency graph.

    std::vector<size_t> id_map(asyncExecuteGraph.numVertices());
    std::iota(id_map.begin(), id_map.end(), 0u);

    asyncExecuteGraph.applyTransitiveReduction();

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        // Fill dep list of air execute ops
        if (auto async_execute_op = dyn_cast_if_present<air::ExecuteOp>(op)) {
          fillAIRDepListUsingGraphTR<air::ExecuteOp>(async_execute_op,
                                                     opIdToOpMap);
        }
        // Fill dep list of air dmamemcpy2d ops
        else if (auto dma_op = dyn_cast_if_present<air::DmaMemcpyNdOp>(op)) {
          fillAIRDepListUsingGraphTR<air::DmaMemcpyNdOp>(dma_op, opIdToOpMap);
        }
        // Fill dep list of air channel ops
        else if (auto channel_op =
                     dyn_cast_if_present<air::ChannelInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::ChannelInterface>(channel_op,
                                                            opIdToOpMap);
        }
        // Fill dep list of air hierarchy ops
        else if (auto hier_op =
                     dyn_cast_if_present<air::HierarchyInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::HierarchyInterface>(hier_op,
                                                              opIdToOpMap);
        }
        // Fill dep list of air wait_all ops
        else if (auto wa_op = dyn_cast_if_present<air::WaitAllOp>(op)) {
          fillAIRDepListUsingGraphTR<air::WaitAllOp>(wa_op, opIdToOpMap);
        }
      });
    }

    // 4th traversal: loop-carried deps.
    // Add wait_all events to collect sinks in loop bodies. Add iter_args to scp
    // for loops representing loop-carried deps.

    auto getYieldedTokens = [&](Region &region) {
      SmallVector<Value, 1> yielded_tokens;
      if (region.empty())
        return yielded_tokens;
      for (auto async_op : region.getOps<air::AsyncOpInterface>()) {
        auto token = async_op.getAsyncToken();
        if (!token)
          continue;
        if (!isNotLoopCarriedOp(async_op) &&
            isOnlyUsedByNoLoopCarryOpsInBlock(token, &region.front())) {
          yielded_tokens.push_back(token);
        }
      }
      for (auto child_for_op : region.front().getOps<scf::ForOp>()) {
        if (child_for_op.getNumResults() == 0)
          continue;
        // Get the async token from the loop using the generic helper
        auto token = air::getAsyncTokenFromOp(child_for_op);
        if (!token)
          continue; // Loop is not async, skip it
        if (isOnlyUsedByNoLoopCarryOpsInBlock(token, &region.front()))
          yielded_tokens.push_back(token);
      }
      for (auto child_parallel_op : region.front().getOps<scf::ParallelOp>()) {
        if (child_parallel_op.getNumResults() == 0)
          continue;
        // Get the async token from the loop using the generic helper
        auto token = air::getAsyncTokenFromOp(child_parallel_op);
        if (!token)
          continue; // Loop is not async, skip it
        if (isOnlyUsedByNoLoopCarryOpsInBlock(token, &region.front()))
          yielded_tokens.push_back(token);
      }
      // Nested branch ops, made async before this region's op (post-order
      // walk): their tokens must reach this op's.
      for (auto &child : region.front()) {
        if (!isa<scf::IfOp, affine::AffineIfOp, scf::IndexSwitchOp>(child))
          continue;
        auto token = air::getAsyncTokenFromOp(&child);
        if (token && isOnlyUsedByNoLoopCarryOpsInBlock(token, &region.front()))
          yielded_tokens.push_back(token);
      }
      return yielded_tokens;
    };

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (LoopLikeOpInterface loop_op =
                dyn_cast_if_present<LoopLikeOpInterface>(op)) {
          Region &region = *loop_op.getLoopRegions().front();
          if (air::getAsyncTokenFromOp(op)) {
            if (failed(adoptAsyncOp(op, "already carries an async token")))
              signalPassFailure();
            return;
          }
          SmallVector<Value, 1> yielded_tokens_in_for_op =
              getYieldedTokens(region);
          if (yielded_tokens_in_for_op.empty())
            return; // No async op in loop successors, continue.
          auto new_loop_op = createAsyncLoopLikeOp(rewriter, loop_op);
          if (failed(new_loop_op)) {
            signalPassFailure();
            return;
          }
          Region &new_region = *new_loop_op->getLoopRegions().front();
          insertLoopCarriedDepsInRegion(rewriter, new_region,
                                        yielded_tokens_in_for_op);
        } else if (RegionBranchOpInterface branch_op =
                       dyn_cast_if_present<RegionBranchOpInterface>(op)) {
          // air hierarchy ops also implement RegionBranchOpInterface (so
          // upstream alias/dataflow analyses can walk their kernel-arg
          // mapping), but they have their own async-token machinery and
          // must not be processed by the generic region-branch handling
          // below.
          if (isa<air::HierarchyInterface>(op))
            return;
          if (isa<scf::IfOp, affine::AffineIfOp, scf::IndexSwitchOp>(op) &&
              air::getAsyncTokenFromOp(op)) {
            if (failed(adoptAsyncOp(op, "already yields an async token")))
              signalPassFailure();
            return;
          }
          auto regions = op->getRegions();
          SmallVector<SmallVector<Value, 1>> yielded_tokens_per_region(
              regions.size());
          for (unsigned i = 0; i < regions.size(); i++) {
            yielded_tokens_per_region[i] = getYieldedTokens(regions[i]);
          }
          if (llvm::all_of(
                  yielded_tokens_per_region,
                  [](SmallVector<Value, 1> toks) { return toks.empty(); }))
            return; // No async op in branch successors, continue.
          // The branch's token goes after its results, where dependency
          // users would not look for it past one already there.
          if (llvm::any_of(op->getResultTypes(),
                           llvm::IsaPred<air::AsyncTokenType>)) {
            op->emitOpError("already yields an async token");
            signalPassFailure();
            return;
          }
          auto new_branch_op = createAsyncRegionBranchOp(rewriter, branch_op);
          // Every region also yields the token collecting the branch op's
          // incoming deps: the transitive reduction may have dropped an
          // edge to an op after the branch because a path went through one
          // region, which other regions do not have.
          Value incoming;
          if (auto wa = dyn_cast_if_present<air::WaitAllOp>(
                  (*new_branch_op)->getPrevNode()))
            incoming = wa.getAsyncToken();
          auto new_regions = (*new_branch_op)->getRegions();
          for (unsigned i = 0; i < new_regions.size(); i++) {
            if (new_regions[i].empty())
              continue;
            SmallVector<Value> incomingDeps;
            if (incoming)
              llvm::append_range(incomingDeps,
                                 incoming.getDefiningOp<air::WaitAllOp>()
                                     .getAsyncDependencies());
            if (incoming &&
                !tokensFollowAll(yielded_tokens_per_region[i], incomingDeps))
              yielded_tokens_per_region[i].push_back(incoming);
            insertLoopCarriedDepsInRegion(rewriter, new_regions[i],
                                          yielded_tokens_per_region[i]);
          }
        }
      });
    }

    // Dependencies on ops inside branch ops, now that branch ops have
    // tokens.
    addBranchDeps();

    // 5th traversal: channel endpoint ordering.
    //
    // Endpoints on one channel slot share a FIFO and must be ordered even when
    // they write unrelated buffers, which the memref tracing above cannot see.
    //
    // Runs after the transitive reduction, which has no notion of the shared
    // channel and would drop these edges, and after loop-carried deps, which
    // supply the iter_args that in-loop scopes hang off.
    air::enforceChannelFifoOrder(module);

    // 6th traversal: an op other than an air async op (index arithmetic, a
    // view, a loop's bound, a branch's condition, a yield) reading an
    // air.execute's result runs when the host reaches it, so it has to wait
    // for the execute: a blocking wait_all before the first such op in each
    // block, per execute.
    for (auto f : module.getOps<func::FuncOp>())
      waitForExecuteResultsInBareOps(f);

    // A get kept synchronous runs when the host reaches it, after the ops
    // before it in its block that touch its buffer, and the gets before it on
    // its channel, have finished.
    for (Operation *get : transfers.keepSync)
      waitBeforeSyncGet(cast<air::ChannelGetOp>(get));
  }

private:
  void waitBeforeSyncGet(air::ChannelGetOp get) {
    llvm::SmallDenseSet<Value> aliases;
    air::collectBufferAliases(get.getMemref(), aliases);
    SmallVector<Value> tokens;
    for (Operation &o : *get->getBlock()) {
      if (&o == get.getOperation())
        break;
      Value token = air::getAsyncTokenFromOp(&o);
      if (!token)
        continue;
      bool touches = llvm::any_of(aliases, [&](Value a) {
        return llvm::any_of(a.getUsers(),
                            [&](Operation *u) { return o.isAncestor(u); });
      });
      auto other = dyn_cast<air::ChannelGetOp>(o);
      if (touches || (other && other.getChanName() == get.getChanName()))
        tokens.push_back(token);
    }
    if (tokens.empty())
      return;
    IRRewriter rewriter(get.getContext());
    rewriter.setInsertionPoint(get);
    air::WaitAllOp::create(rewriter, get.getLoc(), /*asyncToken=*/Type(),
                           tokens);
  }

  void waitForExecuteResultsInBareOps(func::FuncOp f) {
    llvm::DenseMap<Block *, llvm::SmallPtrSet<Operation *, 4>> waited;
    SmallVector<std::pair<Operation *, air::ExecuteOp>> toWait;
    f.walk([&](Operation *op) {
      // An scf.for or scf.if with a token of its own still reads its bounds
      // or condition right away, and a yield its operands; only the air ops
      // wait on their dependencies first.
      if (isa<air::AsyncOpInterface>(op) ||
          op->getParentOfType<air::ExecuteOp>())
        return;
      for (Value v : op->getOperands()) {
        auto exec = v.getDefiningOp<air::ExecuteOp>();
        if (!exec || v == exec.getAsyncToken())
          continue;
        if (waited[op->getBlock()].insert(exec).second)
          toWait.push_back({op, exec});
      }
    });
    IRRewriter rewriter(f.getContext());
    for (auto [op, exec] : toWait) {
      rewriter.setInsertionPoint(op);
      air::WaitAllOp::create(rewriter, op->getLoc(), /*asyncToken=*/Type(),
                             ValueRange{exec.getAsyncToken()});
    }
  }

  ChannelTransfers transfers;
  uint64_t ExecuteOpID;
  uint64_t HierarchyOpID;
  uint64_t WaitAllOpID;
  uint64_t ChannelOpID;
  uint64_t DmaOpID;

  //===----------------------------------------------------------------------===//
  // Handling lingering reshape-related ops
  //===----------------------------------------------------------------------===//

  bool alloc_for_reshape(mlir::Value val) {
    for (auto user : val.getUsers()) {
      // If one of the user is a herd, segment or a launch op,
      // explore the hierarchy further.
      if (auto hier_op = dyn_cast_if_present<air::HierarchyInterface>(user)) {
        for (int i = 0, e = hier_op.getNumKernelOperands(); i < e; i++) {
          if (hier_op.getKernelOperand(i) == val) {
            if (alloc_for_reshape(hier_op.getKernelArgument(i)))
              return true;
          }
        }
      } else if (auto reshape = dyn_cast_if_present<memref::ReshapeOp>(user)) {
        if (reshape.getShape() == val)
          return true;
      }
    }
    return false;
  }

  //===----------------------------------------------------------------------===//
  // Creating async events
  //===----------------------------------------------------------------------===//

  // Create air execute op with async interface (no ssa result returned); update
  // graph
  air::ExecuteOp createAsyncExecute(RewriterBase &rewriter, Operation *op) {
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::ExecuteOp async_region;
    async_region = air::ExecuteOp::create(
        rewriter, loc, air::AsyncTokenType::get(op->getContext()), deps);
    assignOpId(async_region);

    // Insert op to the new async execute region's body.
    Block *async_region_bb = rewriter.createBlock(&async_region.getRegion());
    rewriter.setInsertionPointToStart(async_region_bb);

    // Handle cases when the operand(s) of the given op that is
    // reshape/expand/collapse ops.
    for (unsigned idx = 0; idx < op->getNumOperands(); idx++) {
      auto operand = op->getOperand(idx);
      auto alt_shape_op = operand.getDefiningOp();
      if (isa_and_present<memref::ReshapeOp, memref::ExpandShapeOp,
                          memref::CollapseShapeOp>(alt_shape_op)) {
        auto *cloned_op = rewriter.insert(alt_shape_op->clone());
        op->setOperand(idx, cloned_op->getResult(0));
        auto new_shape_val = alt_shape_op->getResult(0);
        if (new_shape_val.use_empty()) {
          // Erase this shape altering op
          alt_shape_op->erase();
        }
      }
    }

    rewriter.clone(*op);
    air::ExecuteTerminatorOp::create(rewriter, rewriter.getUnknownLoc());

    // Update op-to-graph map
    updateAsyncExecuteGraphWithNewNode(async_region, asyncExecuteGraph);

    // Erase op
    if (eraseOpWithCheck(rewriter, op, "createAsyncExecute (no SSA return)")
            .failed()) {
      signalPassFailure();
    }

    return async_region;
  }

  // Create air execute op with async interface (with one ssa result returned);
  // update graph
  air::ExecuteOp createAsyncExecute(RewriterBase &rewriter, Operation *op,
                                    mlir::Type valueType) {
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::ExecuteOp async_region;
    async_region = air::ExecuteOp::create(
        rewriter, loc, air::AsyncTokenType::get(op->getContext()),
        op->getResults().getType(), deps);
    assignOpId(async_region);

    // Insert op to the new async execute region's body.
    Block *async_region_bb = rewriter.createBlock(&async_region.getRegion());
    rewriter.setInsertionPointToStart(async_region_bb);
    auto op_cloned = rewriter.clone(*op);
    air::ExecuteTerminatorOp::create(rewriter, rewriter.getUnknownLoc(),
                                     op_cloned->getResults());
    SmallVector<Value, 1> returnVals;
    for (auto val : async_region.getResults()) {
      returnVals.push_back(val);
    }
    op->replaceAllUsesWith(returnVals);

    // Update op-to-graph map
    updateAsyncExecuteGraphWithNewNode(async_region, asyncExecuteGraph);

    // Erase op
    if (eraseOpWithCheck(rewriter, op, "createAsyncExecute (one SSA return)")
            .failed()) {
      signalPassFailure();
    }
    return async_region;
  }

  // Re-instantiate the dmamemcpy2d op with async interface; update graph
  void createAsyncDMA(RewriterBase &rewriter, Operation *op) {
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    auto dma_op = mlir::dyn_cast_if_present<air::DmaMemcpyNdOp>(op);
    air::DmaMemcpyNdOp new_dmaNd_op = air::DmaMemcpyNdOp::create(
        rewriter, loc, air::AsyncTokenType::get(dma_op->getContext()), deps,
        dma_op.getDstMemref(), dma_op.getMixedDstOffsets(),
        dma_op.getMixedDstSizes(), dma_op.getMixedDstStrides(),
        dma_op.getSrcMemref(), dma_op.getMixedSrcOffsets(),
        dma_op.getMixedSrcSizes(), dma_op.getMixedSrcStrides(),
        dma_op.getPadBeforeAttr(), dma_op.getPadAfterAttr());
    assignOpId(new_dmaNd_op);

    // Update op-to-graph map
    updateAsyncExecuteGraphWithNewNode(new_dmaNd_op, asyncExecuteGraph);

    // Erase op
    if (eraseOpWithCheck(rewriter, op, "createAsyncDMA").failed()) {
      signalPassFailure();
    }
  }

  // Re-instantiate the channel op with async interface; update graph
  void createAsyncChannel(RewriterBase &rewriter, Operation *op) {
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    std::string event_name = "";
    if (auto channel_put_op = dyn_cast_if_present<air::ChannelPutOp>(op)) {
      air::ChannelPutOp new_channel_put_op = air::ChannelPutOp::create(
          rewriter, loc, air::AsyncTokenType::get(channel_put_op->getContext()),
          deps, channel_put_op.getChanName(), channel_put_op.getIndices(),
          channel_put_op.getSrc(), channel_put_op.getMixedSrcOffsets(),
          channel_put_op.getMixedSrcSizes(),
          channel_put_op.getMixedSrcStrides(),
          /*pad_before=*/nullptr, /*pad_after=*/nullptr);
      air::copyPaddingAttributes(channel_put_op, new_channel_put_op);
      // The optional `dest` operand is not part of the builder above, so it
      // has to be carried over by hand -- dropping it silently discards the
      // put's routing destination, and the packet header never gets written.
      if (Value dest = channel_put_op.getDest())
        new_channel_put_op.getDestMutable().assign(dest);
      assignOpId(new_channel_put_op);
      event_name = "Put";
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_channel_put_op, asyncExecuteGraph);
    } else if (auto channel_get_op =
                   dyn_cast_if_present<air::ChannelGetOp>(op)) {
      air::ChannelGetOp new_channel_get_op = air::ChannelGetOp::create(
          rewriter, loc, air::AsyncTokenType::get(channel_get_op->getContext()),
          deps, channel_get_op.getChanName(), channel_get_op.getIndices(),
          channel_get_op.getDst(), channel_get_op.getMixedDstOffsets(),
          channel_get_op.getMixedDstSizes(),
          channel_get_op.getMixedDstStrides(),
          /*pad_before=*/nullptr, /*pad_after=*/nullptr);
      air::copyPaddingAttributes(channel_get_op, new_channel_get_op);
      assignOpId(new_channel_get_op);
      event_name = "Get";
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_channel_get_op, asyncExecuteGraph);
    } else
      op->emitOpError("unknown air channel op");

    // Erase op
    if (eraseOpWithCheck(rewriter, op, "createAsyncChannel").failed()) {
      signalPassFailure();
    }
  }

  // Re-instantiate the hierarchy op with async interface; update graph
  air::HierarchyInterface
  createAsyncHierarchyImpls(RewriterBase &rewriter,
                            air::HierarchyInterface op) {
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 1> deps;
    SmallVector<Value, 4> args;
    SmallVector<Value, 4> constants;
    for (unsigned i = 0; i < op.getNumKernelOperands(); i++) {
      auto v = op.getKernelOperand(i);
      if (isa_and_present<arith::ConstantOp>(v.getDefiningOp())) {
        constants.push_back(v);
        args.push_back(v);
      } else
        args.push_back(v);
    }
    Operation *new_op = nullptr;
    if (auto rank = dyn_cast_if_present<air::RankOp>(op.getOperation())) {
      auto new_rank =
          createAsyncHierarchyRank(rewriter, rank, deps, args, constants);
      new_op = new_rank.getOperation();
      updateAsyncExecuteGraphWithNewNode(new_rank, asyncExecuteGraph);
    } else if (auto launch =
                   dyn_cast_if_present<air::LaunchOp>(op.getOperation())) {
      auto new_launch = createAsyncHierarchy<air::LaunchOp>(
          rewriter, launch, deps, args, constants);
      new_op = new_launch.getOperation();
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_launch, asyncExecuteGraph);
    } else if (auto segment =
                   dyn_cast_if_present<air::SegmentOp>(op.getOperation())) {
      auto new_segment = createAsyncHierarchy<air::SegmentOp>(
          rewriter, segment, deps, args, constants);
      new_op = new_segment.getOperation();
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_segment, asyncExecuteGraph);
    } else if (auto herd =
                   dyn_cast_if_present<air::HerdOp>(op.getOperation())) {
      auto new_herd = createAsyncHierarchy<air::HerdOp>(rewriter, herd, deps,
                                                        args, constants);
      new_op = new_herd.getOperation();
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_herd, asyncExecuteGraph);
    } else {
      op->emitOpError("unknown hierarchy operation");
    }
    auto new_hier = dyn_cast_if_present<air::HierarchyInterface>(new_op);

    // Erase op
    if (eraseOpWithCheck(rewriter, op, "createAsyncHierarchyImpls").failed()) {
      signalPassFailure();
    }
    return new_hier;
  }

  template <typename T>
  T createAsyncHierarchy(RewriterBase &rewriter, T op,
                         SmallVector<Value, 1> deps, SmallVector<Value, 4> args,
                         SmallVector<Value, 4> constants) {
    auto loc = op->getLoc();
    T new_op = T::create(rewriter, loc, deps, op.getSizeOperands(), args, true,
                         op->getAttrs());
    assignOpId(new_op);

    auto &bb = new_op.getBody().front();
    for (unsigned i = 0; i < op.getIds().size(); i++) {
      auto ivs = op.getIds()[i];
      ivs.replaceAllUsesWith(new_op.getIds()[i]);
    }
    for (unsigned i = 0; i < op.getSize().size(); i++) {
      auto s = op.getSize()[i];
      s.replaceAllUsesWith(new_op.getSize()[i]);
    }
    auto &body = op.getBody().front().getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&new_op.getRegion().front());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, rewriter.clone(*c.getDefiningOp())->getResult(0),
          new_op.getRegion());
    }

    int i = 0;
    auto old_kernel_args = op.getKernelArguments();
    auto new_kernel_args = new_op.getKernelArguments();
    for (Value v : old_kernel_args)
      replaceAllUsesInRegionWith(v, new_kernel_args[i++], new_op.getRegion());

    return new_op;
  }

  // Specialized version for RankOp that threads the universe operand.
  air::RankOp createAsyncHierarchyRank(RewriterBase &rewriter, air::RankOp op,
                                       SmallVector<Value, 1> deps,
                                       SmallVector<Value, 4> args,
                                       SmallVector<Value, 4> constants) {
    auto loc = op->getLoc();
    air::RankOp new_op;
    if (op.getUniverse()) {
      new_op =
          air::RankOp::create(rewriter, loc, deps, op.getUniverse(),
                              op.getSizeOperands(), args, true, op->getAttrs());
    } else {
      new_op = air::RankOp::create(rewriter, loc, deps, op.getSizeOperands(),
                                   args, true, op->getAttrs());
    }
    assignOpId(new_op);

    auto &bb = new_op.getBody().front();
    for (unsigned i = 0; i < op.getIds().size(); i++) {
      auto ivs = op.getIds()[i];
      ivs.replaceAllUsesWith(new_op.getIds()[i]);
    }
    for (unsigned i = 0; i < op.getSize().size(); i++) {
      auto s = op.getSize()[i];
      s.replaceAllUsesWith(new_op.getSize()[i]);
    }
    auto &body = op.getBody().front().getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.setInsertionPointToStart(&new_op.getRegion().front());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, rewriter.clone(*c.getDefiningOp())->getResult(0),
          new_op.getRegion());
    }

    int i = 0;
    auto old_kernel_args = op.getKernelArguments();
    auto new_kernel_args = new_op.getKernelArguments();
    for (Value v : old_kernel_args)
      replaceAllUsesInRegionWith(v, new_kernel_args[i++], new_op.getRegion());

    return new_op;
  }

  void updateAsyncExecuteGraphWithNewNode(Operation *op, ExecuteGraph &graph) {
    auto v = graph.addVertex();
    auto &node = graph[v];
    // Create a vertex out of the current async execute region
    node.asyncEventName = air::to_string(op);
    auto setNodeAttrsBasedOnOp = [&node, &v, this](Operation *op) {
      if (auto execOp = dyn_cast_if_present<air::ExecuteOp>(op)) {
        node.asyncEventType = "execute";
        node.color = "chartreuse";
        node.shape = "oval";
        node.operationId = execOp.getId();
        region_to_g[node.operationId] = v;
      } else if (auto dmaOp = dyn_cast_if_present<air::DmaMemcpyNdOp>(op)) {
        node.asyncEventType = "dma";
        node.color = "cyan";
        node.shape = "oval";
        node.operationId = dmaOp.getId();
        dma_to_g[node.operationId] = v;
      } else if (auto ciOp = dyn_cast_if_present<air::ChannelInterface>(op)) {
        node.asyncEventName = air::to_string(op);
        node.asyncEventType = "channel";
        node.color = "cyan";
        node.shape = "oval";
        node.operationId = ciOp.getId();
        channel_to_g[node.operationId] = v;
      } else if (auto hiOp = dyn_cast_if_present<air::HierarchyInterface>(op)) {
        node.asyncEventName = air::to_string(op);
        node.asyncEventType = "hierarchy";
        node.color = "yellow";
        node.shape = "box";
        node.operationId = hiOp.getId();
        hier_to_g[node.operationId] = v;
      }
    };
    setNodeAttrsBasedOnOp(op);
  }

  void assignOpId(Operation *op) {
    if (isa<air::ExecuteOp>(op))
      op->setAttr("id", mlir::IntegerAttr::get(
                            mlir::IntegerType::get(op->getContext(), 32),
                            ++ExecuteOpID));
    else if (isa<air::ChannelInterface>(op))
      op->setAttr("id", mlir::IntegerAttr::get(
                            mlir::IntegerType::get(op->getContext(), 32),
                            ++ChannelOpID));
    else if (isa<air::HierarchyInterface>(op))
      op->setAttr("id", mlir::IntegerAttr::get(
                            mlir::IntegerType::get(op->getContext(), 32),
                            ++HierarchyOpID));
    else if (isa<air::WaitAllOp>(op))
      op->setAttr("id", mlir::IntegerAttr::get(
                            mlir::IntegerType::get(op->getContext(), 32),
                            ++WaitAllOpID));
    else if (isa<air::DmaMemcpyNdOp>(op))
      op->setAttr("id",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(op->getContext(), 32), ++DmaOpID));
  }

  //===----------------------------------------------------------------------===//
  // Data dependency tracing
  //===----------------------------------------------------------------------===//

  // Check if operand is returned from ExecuteOp (memref.alloc)
  template <typename T>
  void pushDefiningOpAsDep(Value operand, T op) {
    // Check memref deps
    if (auto defop = operand.getDefiningOp<air::ExecuteOp>()) {
      DominanceInfo domInfo(defop);
      if (domInfo.properlyDominates(defop, op)) {
        // if (foundAsyncOpUsesAboveCurrentLine(&defop)) {
        addAsyncDepToGraphIfNew<T>(defop.getResult(0), op);
      }
    }
  }

  // Trace tile index deps
  template <typename T>
  void pushTileIndexAsDep(mlir::Value tile_index, T op) {
    if (tile_index != nullptr) {
      // If tile_index is not a nullptr
      // If created by async_region
      if (auto defop = tile_index.getDefiningOp<air::ExecuteOp>()) {
        DominanceInfo domInfo(defop);
        if (domInfo.properlyDominates(defop, op)) {
          // if (foundAsyncOpUsesAboveCurrentLine(&defop)) {
          addAsyncDepToGraphIfNew<T>(defop.getResult(0), op);
        }
      }
      // If created by hierarchy (as loop iter)
      else if (auto hier = dyn_cast_if_present<air::HierarchyInterface>(
                   tile_index.getParentRegion()->getParentOp())) {
        for (auto id : hier.getIds()) {
          if (id == tile_index) {
            addAsyncDepToGraphIfNew<T>(tile_index, op);
          }
        }
      }
    }
  }

  char checkOperandReadOrWrite(mlir::Value operand) {
    if (!llvm::isa<BaseMemRefType>(operand.getType())) {
      operand.getDefiningOp()->emitOpError(
          "operand being traced is not a memref");
    }
    bool foundWriteAccess = false;
    bool foundReadAccess = false;
    for (auto &u : operand.getUses()) {
      auto writeAccesses =
          air::getAllWriteAccessedMemrefOperandsFromOp(u.getOwner());
      auto readAccesses =
          air::getAllReadAccessedMemrefOperandsFromOp(u.getOwner());
      if (failed(writeAccesses) || failed(readAccesses)) {
        operand.getDefiningOp()->emitOpError(
            "has user which failed to get memref accesses.");
        return 'x';
      }
      foundWriteAccess =
          foundWriteAccess ||
          llvm::any_of(*writeAccesses,
                       [&u](const air::MemrefAccessPattern &entry) {
                         return entry.first == u.get();
                       });
      foundReadAccess =
          foundReadAccess ||
          llvm::any_of(*readAccesses,
                       [&u](const air::MemrefAccessPattern &entry) {
                         return entry.first == u.get();
                       });
    }
    if (foundWriteAccess)
      return 'w';
    else if (foundReadAccess)
      return 'r';
    else
      return 'w';
  }

  // Trace operand's uses at current scope
  template <typename T>
  void pushDepsAtCurrentScope(mlir::Value operand, T op, char rw = 'n',
                              partialMemref *tile = nullptr) {
    if (!llvm::isa<BaseMemRefType>(operand.getType())) {
      operand.getDefiningOp()->emitOpError(
          "operand being traced is not a memref");
    }
    for (auto &u : operand.getUses()) {
      if (!air::opOrAncestorIsDominantOver(u.getOwner(), op))
        continue;
      // A get kept synchronous is ordered by the host; a put and get sharing
      // a transfer are ordered by it (ChannelTransfers).
      if (transfers.keepSync.contains(u.getOwner()) ||
          transfers.unordered.contains({u.getOwner(), op.getOperation()}) ||
          transfers.unordered.contains({op.getOperation(), u.getOwner()}))
        continue;
      // If used in MemcpyInterface Op
      if (auto memcpy =
              dyn_cast_if_present<air::MemcpyInterface>(u.getOwner())) {
        partialMemref memcpy_src, memcpy_dst;
        if (memcpy.getSrcMemref()) {
          memcpy_src = partialMemref(
              memcpy.getSrcMemref(), memcpy.getMixedSrcOffsets(),
              memcpy.getMixedSrcSizes(), memcpy.getMixedSrcStrides());
        }
        if (memcpy.getDstMemref()) {
          memcpy_dst = partialMemref(
              memcpy.getDstMemref(), memcpy.getMixedDstOffsets(),
              memcpy.getMixedDstSizes(), memcpy.getMixedDstStrides());
        }

        if (rw == 'r') {
          if (u.is(memcpy.getSrcMemref())) {
            if (tile == nullptr) {
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
            } else if (areEqualIndexPartialMemrefs(tile, &memcpy_src))
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
          }
        } else if (rw == 'w') {
          if (u.is(memcpy.getDstMemref())) {
            if (tile == nullptr) {
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
            } else if (areEqualIndexPartialMemrefs(tile, &memcpy_dst) ||
                       footprintsIntersectInHostMemory(tile, &memcpy_dst))
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
          }
        } else {
          // A write after a write or a read: as for a read after a write,
          // footprints that start apart may still share elements.
          if (tile == nullptr) {
            addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0), op);
          } else if (u.is(memcpy.getDstMemref())) {
            if (areEqualIndexPartialMemrefs(tile, &memcpy_dst) ||
                footprintsIntersectInHostMemory(tile, &memcpy_dst))
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
          } else if (u.is(memcpy.getSrcMemref())) {
            if (areEqualIndexPartialMemrefs(tile, &memcpy_src) ||
                footprintsIntersectInHostMemory(tile, &memcpy_src))
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
          }
        }
      }

      // If used in a linalg op
      else if (auto linalgop =
                   mlir::dyn_cast_if_present<linalg::LinalgOp>(u.getOwner())) {
        if (auto ar =
                dyn_cast_if_present<air::ExecuteOp>(linalgop->getParentOp())) {
          if (rw == 'r') {
            if (u.getOperandNumber() <
                linalgop.getNumDpsInputs() + linalgop.getNumDpsInits())
              addAsyncDepToGraphIfNew<T>(ar.getResult(0), op);
          } else if (rw == 'w') {
            if (u.getOperandNumber() >= linalgop.getNumDpsInputs() &&
                u.getOperandNumber() - linalgop.getNumDpsInputs() <
                    linalgop.getNumDpsInits())
              addAsyncDepToGraphIfNew<T>(ar.getResult(0), op);
          } else {
            addAsyncDepToGraphIfNew<T>(ar.getResult(0), op);
          }
        }
      }

      // If used in hierarchy op
      else if (auto hier =
                   dyn_cast_if_present<air::HierarchyInterface>(u.getOwner())) {
        // check if the use inside hierarchy op matches with the tracing mode (r
        // or w)
        for (unsigned hier_argument_id = 0;
             hier_argument_id < hier.getNumKernelOperands();
             hier_argument_id++) {
          if (u.is(hier.getKernelOperand(hier_argument_id))) {
            auto child_op = hier.getKernelArgument(hier_argument_id);
            char rw_check = checkOperandReadOrWrite(child_op);
            if (rw == 'n' || rw_check == rw) {
              addAsyncDepToGraphIfNew<T>(hier->getResult(0), op);
            }
          }
        }
      }

      // If used in an unknown op
      else {
        if (auto ar = dyn_cast_if_present<air::ExecuteOp>(
                u.getOwner()->getParentOp()))
          addAsyncDepToGraphIfNew<T>(ar.getResult(0), op);
      }
    }
  }

  template <typename T>
  void traceDeps(SmallVector<partialMemref, 1> operands, T sink_air_op,
                 std::string dep_type) {

    char dep_tracing_mode = 'n';
    if (dep_type == "RAW")
      dep_tracing_mode = 'w';
    else if (dep_type == "WAW/WAR")
      dep_tracing_mode = 'n';
    else
      sink_air_op->emitOpError("unknown dependency type");

    // Detect deps
    for (auto operand : operands) {
      // Trace the defining op of sink op, RAW
      pushDefiningOpAsDep<T>(operand.memrefValue, sink_air_op);

      // If sink op and operand's use are under the same scope
      pushDepsAtCurrentScope<T>(operand.memrefValue, sink_air_op,
                                dep_tracing_mode, &operand);

      // If sink op is in hierarchy op
      if (auto hier =
              sink_air_op
                  ->template getParentOfType<air::HierarchyInterface>()) {
        // Search for deps outside (before) hierarchy op
        for (unsigned hier_operand_id = 0;
             hier_operand_id < hier.getNumKernelOperands(); hier_operand_id++) {
          if (hier.getKernelArguments()[hier_operand_id] ==
              operand.memrefValue) {
            auto ancestor_op = hier.getKernelOperand(hier_operand_id);
            partialMemref ancestor_operand(ancestor_op);
            SmallVector<partialMemref, 1> ancestor_operands = {
                ancestor_operand};
            traceDeps<air::HierarchyInterface>(ancestor_operands, hier,
                                               dep_type);
          }
        }
      }

      // Check if operand is returned from memref.subview
      if (auto subview =
              operand.memrefValue.getDefiningOp<memref::SubViewOp>()) {
        partialMemref subview_tile(
            subview.getSource(), subview.getMixedOffsets(),
            subview.getMixedSizes(), subview.getMixedStrides());
        SmallVector<partialMemref, 1> subview_operands = {subview_tile};
        traceDeps<T>(subview_operands, sink_air_op, dep_type);
      }
    }
  }

  // Channel put->get backpressure across a hierarchy boundary. A
  // launch/segment-scope air.channel.get (consumer) whose matching
  // air.channel.put (producer) lives inside a sibling hierarchy op
  // (air.segment/herd/launch) has an implicit data dependency on that
  // hierarchy: the get drains what the hierarchy produces. The memref-based
  // tracer cannot see it -- the two ops touch disjoint memrefs across the
  // scope boundary (e.g. an L1 producer buffer vs an L3 host buffer) -- so add
  // the edge explicitly. Without it a launch-scope drain to host DDR is
  // scheduled ahead of the producing segment and its wait is emitted before the
  // data can exist.
  void traceChannelProducerHierarchyDep(air::ChannelInterface channel_op) {
    auto get = dyn_cast<air::ChannelGetOp>(channel_op.getOperation());
    if (!get)
      return;
    Block *getBlock = get->getBlock();
    DominanceInfo domInfo(getBlock->getParentOp());
    for (air::ChannelPutOp put : getTheOtherChannelOpThroughSymbol(get)) {
      // Walk up from the producer to the ancestor that is a sibling of the get
      // (a direct child of the get's block). A same-scope partner (anc == put)
      // or one under an unrelated block (anc == null) needs no explicit edge.
      Operation *anc = put.getOperation();
      while (anc && anc->getBlock() != getBlock)
        anc = anc->getParentOp();
      if (!anc || anc == get.getOperation())
        continue;
      auto hier = dyn_cast<air::HierarchyInterface>(anc);
      if (!hier)
        continue;
      if (!domInfo.properlyDominates(hier.getOperation(), get.getOperation()))
        continue;
      addAsyncDepToGraphIfNew<air::ChannelInterface>(hier->getResult(0),
                                                     channel_op);
    }
  }

  template <typename T>
  void traceTileIndices(SmallVector<partialMemref, 1> read_operands,
                        SmallVector<partialMemref, 1> write_operands,
                        SmallVector<Value, 1> in_scalars,
                        SmallVector<Value, 1> out_scalars, T sink_air_op) {
    for (auto operand : read_operands) {
      for (auto v : operand.getDynamicEntries())
        pushTileIndexAsDep<T>(v, sink_air_op);
    }
    for (auto operand : write_operands) {
      for (auto v : operand.getDynamicEntries())
        pushTileIndexAsDep<T>(v, sink_air_op);
    }
    for (auto scalar : in_scalars) {
      pushTileIndexAsDep<T>(scalar, sink_air_op);
    }
    for (auto scalar : out_scalars) {
      pushTileIndexAsDep<T>(scalar, sink_air_op);
    }
    // A value bare ops compute from an air.execute's results (index
    // arithmetic, views, metadata) is ready only once that execute is done:
    // the op using it waits for the execute, through the bare ops.
    llvm::SetVector<Value> used;
    if (auto exec = dyn_cast<air::ExecuteOp>(sink_air_op.getOperation()))
      getUsedValuesDefinedAbove(exec.getRegion(), used);
    else
      used.insert(sink_air_op->operand_begin(), sink_air_op->operand_end());
    llvm::SmallPtrSet<Value, 8> seen;
    SmallVector<Value> work(used.begin(), used.end());
    while (!work.empty()) {
      Value v = work.pop_back_val();
      if (isa<air::AsyncTokenType>(v.getType()) || !seen.insert(v).second)
        continue;
      Operation *def = v.getDefiningOp();
      if (!def)
        continue;
      if (isa<air::ExecuteOp>(def)) {
        pushTileIndexAsDep<T>(v, sink_air_op);
        continue;
      }
      if (air::isAsyncOp(def) || def->getNumRegions())
        continue;
      llvm::append_range(work, def->getOperands());
    }
  }

  //===----------------------------------------------------------------------===//
  // SCF for loop-carried dependency
  //===----------------------------------------------------------------------===//

  void insertVertexBetweenTwoOps(Operation *a, Operation *b,
                                 ExecuteGraph::VertexId v) {
    unsigned v_a = 0;
    unsigned v_b = 0;
    if (auto op = dyn_cast_if_present<air::ExecuteOp>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast_if_present<air::DmaMemcpyNdOp>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast_if_present<air::ChannelInterface>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast_if_present<air::HierarchyInterface>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast_if_present<scf::ForOp>(a)) {
      v_a = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast_if_present<scf::ParallelOp>(a)) {
      v_a = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast_if_present<air::WaitAllOp>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    }
    if (auto op = dyn_cast_if_present<air::ExecuteOp>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast_if_present<air::DmaMemcpyNdOp>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast_if_present<air::ChannelInterface>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast_if_present<air::HierarchyInterface>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast_if_present<scf::ForOp>(b)) {
      v_b = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast_if_present<scf::ParallelOp>(b)) {
      v_b = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast_if_present<air::WaitAllOp>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    }
    if (asyncExecuteGraph.hasEdge(v_a, v_b)) { // if an edge exists
      asyncExecuteGraph.removeEdge(v_a, v_b);
      if (!asyncExecuteGraph.hasEdge(v_a, v))
        asyncExecuteGraph.addEdge(v_a, v);
      if (!asyncExecuteGraph.hasEdge(v, v_b))
        asyncExecuteGraph.addEdge(v, v_b);
    }
  }

  air::WaitAllOp insertWaitAllOpBeforeLoopYield(
      OpBuilder &builder, Region &region,
      SmallVector<Value, 1> yielded_tokens_in_loop_op) {
    // Create one wait_all event at the end of current loop body, BEFORE the
    // terminator Output token of wait_all shall be yielded
    if (region.front().mightHaveTerminator()) {
      builder.setInsertionPoint(region.front().getTerminator());
    } else {
      builder.setInsertionPointToEnd(&region.front());
    }
    air::WaitAllOp wait_all_op_yielded =
        air::WaitAllOp::create(builder, builder.getUnknownLoc(),
                               air::AsyncTokenType::get(builder.getContext()),
                               yielded_tokens_in_loop_op);
    wait_all_op_yielded->setAttr(
        "id",
        mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 32),
                               ++WaitAllOpID));
    return wait_all_op_yielded;
  }

  ExecuteGraph::VertexId addVertexWaitAllOpBeforeLoopYield(
      SmallVector<Value, 1> yielded_tokens_in_loop_op, std::string loop_type) {
    // Create vertex
    ExecuteGraph::VertexId wait_all_op_yielded_v =
        asyncExecuteGraph.addVertex();
    asyncExecuteGraph[wait_all_op_yielded_v].asyncEventName = "scf::";
    asyncExecuteGraph[wait_all_op_yielded_v].asyncEventName += loop_type;
    asyncExecuteGraph[wait_all_op_yielded_v].asyncEventName += "_loop_end";
    asyncExecuteGraph[wait_all_op_yielded_v].asyncEventType = "wait_all";
    asyncExecuteGraph[wait_all_op_yielded_v].color = "crimson";
    asyncExecuteGraph[wait_all_op_yielded_v].shape = "oval";
    asyncExecuteGraph[wait_all_op_yielded_v].operationId = 0;
    // Update graph connectivity
    for (auto token : yielded_tokens_in_loop_op) {
      unsigned src_id = 0;
      if (auto async_execute_op =
              dyn_cast_if_present<air::ExecuteOp>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(async_execute_op);
      } else if (auto dma_op = dyn_cast_if_present<air::DmaMemcpyNdOp>(
                     token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(dma_op);
      } else if (auto channel_op = dyn_cast_if_present<air::ChannelInterface>(
                     token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(channel_op);
      } else if (auto hier_op = dyn_cast_if_present<air::HierarchyInterface>(
                     token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(hier_op);
      } else if (auto scf_for_op =
                     dyn_cast_if_present<scf::ForOp>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(
            scf_for_op); // g_to_tr not needed since wait_all created after TR
      } else if (auto scf_parallel_op = dyn_cast_if_present<scf::ParallelOp>(
                     token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(
            scf_parallel_op); // g_to_tr not needed since wait_all created after
                              // TR
      } else if (auto wa_op = dyn_cast_if_present<air::WaitAllOp>(
                     token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(wa_op);
      } else {
        // A branch op's token: no vertex of its own.
        continue;
      }
      asyncExecuteGraph.addEdge(src_id, wait_all_op_yielded_v);
    }
    return wait_all_op_yielded_v;
  }

  template <typename T>
  air::WaitAllOp insertWaitAllOpAtLoopBegin(
      OpBuilder &builder, T loop_op, std::string loop_type,
      SmallVector<Value, 4> incoming_tokens, SmallVector<Value, 4> constants) {
    // Create a new wait_all event before the for op which collects the incoming
    // deps. Output token of wait_all shall be the iter_arg of for op.
    builder.setInsertionPoint(loop_op);
    air::WaitAllOp wait_all_op_before_loop = air::WaitAllOp::create(
        builder, builder.getUnknownLoc(),
        air::AsyncTokenType::get(loop_op->getContext()), incoming_tokens);
    wait_all_op_before_loop->setAttr(
        "id",
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(loop_op->getContext(), 32), ++WaitAllOpID));

    // Create vertex
    uint64_t wait_all_op_before_loop_v = asyncExecuteGraph.addVertex();

    auto &node = asyncExecuteGraph[wait_all_op_before_loop_v];
    node.asyncEventName = "scf::";
    node.asyncEventName += loop_type;
    node.asyncEventName += "_loop_begin";
    node.asyncEventType = "wait_all";
    node.color = "crimson";
    node.shape = "oval";
    node.operationId = 0;
    // Update op-to-graph map
    wa_to_g[wait_all_op_before_loop.getId()] = wait_all_op_before_loop_v;

    // Update graph connectivity
    for (Value v : incoming_tokens) {
      for (auto user : v.getUsers()) {
        insertVertexBetweenTwoOps(v.getDefiningOp(), user,
                                  wait_all_op_before_loop_v);
      }
    }

    return wait_all_op_before_loop;
  }

  scf::ForOp convertScfForToAsync(RewriterBase &rewriter, scf::ForOp loop_op,
                                  air::WaitAllOp wait_all_op_before_loop,
                                  SmallVector<Value, 4> incoming_tokens,
                                  SmallVector<Value, 4> constants) {
    // Preserve existing iter_args and add async token as additional iter_arg
    SmallVector<Value> all_init_args;
    all_init_args.append(loop_op.getInitArgs().begin(),
                         loop_op.getInitArgs().end());
    all_init_args.push_back(wait_all_op_before_loop.getResult(0));

    scf::ForOp new_loop_op = scf::ForOp::create(
        rewriter, loop_op.getLoc(), loop_op.getLowerBound(),
        loop_op.getUpperBound(), loop_op.getStep(), all_init_args);

    if (auto attr = loop_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_loop_op->setAttr(SymbolTable::getSymbolAttrName(), attr);
    // Preserve user-attached loop_annotation (e.g., unroll-disable for
    // Peano). This is dropped by default since the new loop is a fresh
    // op with only a few hand-copied attributes.
    if (auto attr = loop_op->getAttr("loop_annotation"))
      new_loop_op->setAttr("loop_annotation", attr);
    // Preserve the user-facing ping-pong opt-out so labeling later still
    // sees it.
    if (auto attr = loop_op->getAttr("air.disable_ping_pong"))
      new_loop_op->setAttr("air.disable_ping_pong", attr);

    // Splice the operations inside loop op INCLUDING the terminator
    auto &bb = new_loop_op.getBody()->getOperations();
    auto &body = loop_op.getBody()->getOperations();
    bb.splice(bb.begin(), body, body.begin(), body.end());

    // Replace old induction variable and existing iter_args
    auto iv = loop_op.getInductionVar();
    iv.replaceAllUsesWith(new_loop_op.getInductionVar());

    for (unsigned i = 0; i < loop_op.getRegionIterArgs().size(); i++) {
      loop_op.getRegionIterArgs()[i].replaceAllUsesWith(
          new_loop_op.getRegionIterArgs()[i]);
    }

    rewriter.setInsertionPointToStart(new_loop_op.getBody());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, rewriter.clone(*c.getDefiningOp())->getResult(0),
          new_loop_op.getRegion());
    }

    // Get the async token iter_arg (now the last one since we appended it)
    Value asyncTokenIterArg =
        air::getLoopCarriedTokenFromScfOp(new_loop_op, "argument");

    for (Value v : incoming_tokens) {
      replaceAllUsesInRegionWith(v, asyncTokenIterArg, new_loop_op.getRegion());
    }

    // Connect sources in loop body with iter_args
    for (auto async_op : new_loop_op.getOps<air::AsyncOpInterface>()) {
      if (!isNotLoopCarriedOp(async_op)) {
        addAsyncDependencyIfNew(async_op, asyncTokenIterArg);
      }
    }

    return new_loop_op;
  }

  scf::ParallelOp convertScfParToAsync(RewriterBase &rewriter,
                                       scf::ParallelOp loop_op,
                                       air::WaitAllOp wait_all_op_before_loop,
                                       SmallVector<Value, 4> incoming_tokens,
                                       SmallVector<Value, 4> constants) {
    if (loop_op.getNumReductions() != 0)
      loop_op->emitOpError(
          "currently only supporting input scf::ParallelOp with no reductions");

    // Create new parallel op with init_val.
    SmallVector<Value, 4> merged_incoming_token;
    merged_incoming_token.push_back(wait_all_op_before_loop.getResult(0));
    scf::ParallelOp new_loop_op = scf::ParallelOp::create(
        rewriter, loop_op.getLoc(), loop_op.getLowerBound(),
        loop_op.getUpperBound(), loop_op.getStep(), merged_incoming_token);

    if (auto attr = loop_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_loop_op->setAttr(SymbolTable::getSymbolAttrName(), attr);
    if (auto attr = loop_op->getAttr("loop_annotation"))
      new_loop_op->setAttr("loop_annotation", attr);
    if (auto attr = loop_op->getAttr("air.disable_ping_pong"))
      new_loop_op->setAttr("air.disable_ping_pong", attr);

    // Splice the operations inside loop op
    auto &bb = new_loop_op.getBody()->getOperations();
    auto &body = loop_op.getBody()->getOperations();
    bb.splice(bb.begin(), body, body.begin(), --body.end());

    for (unsigned i = 0; i < loop_op.getInductionVars().size(); i++) {
      auto iv_old = loop_op.getInductionVars()[i];
      auto iv_new = new_loop_op.getInductionVars()[i];
      iv_old.replaceAllUsesWith(iv_new);
    }

    rewriter.setInsertionPointToStart(new_loop_op.getBody());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, rewriter.clone(*c.getDefiningOp())->getResult(0),
          new_loop_op.getRegion());
    }

    for (Value v : incoming_tokens) {
      replaceAllUsesInRegionWith(v, new_loop_op.getInitVals()[0],
                                 new_loop_op.getRegion());
    }

    // Connect sources in loop body with init_val
    for (auto async_execute_op : new_loop_op.getOps<air::ExecuteOp>()) {
      if (async_execute_op.getAsyncDependencies().size() == 0) {
        async_execute_op.addAsyncDependency(new_loop_op.getInitVals()[0]);
      }
    }
    for (auto dma_op : new_loop_op.getOps<air::DmaMemcpyNdOp>()) {
      auto async_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
          dma_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getInitVals()[0]);
      }
    }
    for (auto channel_op : new_loop_op.getOps<air::ChannelInterface>()) {
      auto async_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
          channel_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getInitVals()[0]);
      }
    }
    for (auto hier_op : new_loop_op.getOps<air::HierarchyInterface>()) {
      auto async_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
          hier_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getInitVals()[0]);
      }
    }

    return new_loop_op;
  }

  scf::IfOp convertScfIfToAsync(RewriterBase &rewriter, scf::IfOp branch_op) {
    // Create new if op with its results and a yielded async token.
    SmallVector<Type> yielded_tys(branch_op.getResultTypes());
    yielded_tys.push_back(air::AsyncTokenType::get(rewriter.getContext()));
    scf::IfOp new_branch_op = scf::IfOp::create(
        rewriter, branch_op.getLoc(), yielded_tys, branch_op.getCondition(),
        /*withElseRegion*/ true);

    if (auto attr = branch_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_branch_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

    spliceBranchRegions<scf::YieldOp>(rewriter, branch_op, new_branch_op);
    return new_branch_op;
  }

  // Splice old_op's regions (without their terminators) into new_op's, yield
  // what each yielded, and replace old_op's results with new_op's leading ones;
  // insertLoopCarriedDepsInRegion appends the async token to the yields.
  template <typename YieldT>
  void spliceBranchRegions(RewriterBase &rewriter, Operation *old_op,
                           Operation *new_op) {
    OpBuilder::InsertionGuard guard(rewriter);
    for (auto [o_r, n_r] :
         llvm::zip_equal(old_op->getRegions(), new_op->getRegions())) {
      if (o_r.empty() || n_r.empty())
        continue;
      auto &bb = n_r.front().getOperations();
      auto &body = o_r.front().getOperations();
      bb.splice(bb.begin(), body, body.begin(), --body.end());
      if (old_op->getNumResults()) {
        rewriter.setInsertionPointToEnd(&n_r.front());
        YieldT::create(rewriter, old_op->getLoc(),
                       o_r.front().getTerminator()->getOperands());
      }
    }
    for (auto [o, n] : llvm::zip(old_op->getResults(), new_op->getResults()))
      rewriter.replaceAllUsesWith(o, n);
  }

  affine::AffineIfOp convertAffineIfToAsync(RewriterBase &rewriter,
                                            affine::AffineIfOp branch_op) {
    // Create new if op with its results and a yielded async token.
    SmallVector<Type> yielded_tys(branch_op.getResultTypes());
    yielded_tys.push_back(air::AsyncTokenType::get(rewriter.getContext()));
    affine::AffineIfOp new_branch_op = affine::AffineIfOp::create(
        rewriter, branch_op.getLoc(), yielded_tys, branch_op.getIntegerSet(),
        branch_op.getOperands(),
        /*withElseRegion*/ true);

    if (auto attr = branch_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_branch_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

    spliceBranchRegions<affine::AffineYieldOp>(rewriter, branch_op,
                                               new_branch_op);
    return new_branch_op;
  }

  scf::IndexSwitchOp
  convertScfIndexSwitchToAsync(RewriterBase &rewriter,
                               scf::IndexSwitchOp branch_op) {
    // Append the async token result, splice regions (excluding terminators) and
    // RAUW; insertLoopCarriedDepsInRegion appends the async yields afterwards
    // and the caller erases branch_op. The rebuild leaves each region's old
    // yield behind: keep what it yielded.
    SmallVector<SmallVector<Value>> old_yields;
    for (Region &r : branch_op->getRegions())
      old_yields.emplace_back(
          r.empty()
              ? SmallVector<Value>{}
              : SmallVector<Value>(r.front().getTerminator()->getOperands()));
    auto new_branch_op =
        air::rebuildIndexSwitchWithTrailingAsyncToken(rewriter, branch_op);
    OpBuilder::InsertionGuard guard(rewriter);
    for (auto [n_r, vals] :
         llvm::zip_equal(new_branch_op->getRegions(), old_yields)) {
      if (n_r.empty() || vals.empty())
        continue;
      rewriter.setInsertionPointToEnd(&n_r.front());
      scf::YieldOp::create(rewriter, branch_op.getLoc(), vals);
    }
    return new_branch_op;
  }

  // Elevating tokens from inside loop body to the yielded token, to maintain
  // legal domination T: loop type (scf::ForOp or scf::ParallelOp) U: source op
  // type
  template <typename U>
  void elevateAsyncTokens(Region &region, ExecuteGraph::VertexId wait_all_op) {
    for (auto source : region.getOps<U>()) {
      SmallPtrSet<Operation *, 1> keep;
      if (source->getNumResults() == 0)
        continue;

      // Get the async token from the source operation
      Value tokenResult = air::getAsyncTokenFromOp(source.getOperation());
      if (!tokenResult)
        continue; // No token result, skip elevation

      if (tokenResult) {
        for (auto sink : tokenResult.getUsers()) {
          // Keep token if source already dominates sink
          if (source->getParentOp()->isAncestor(sink)) {
            keep.insert(sink);
          } else {
            // Update graph connectivity
            insertVertexBetweenTwoOps(source.getOperation(), sink, wait_all_op);
          }
        }
        // Only replace the token result, not other results
        tokenResult.replaceAllUsesExcept(
            region.getParentOp()->getResults().back(), keep);
      }
    }
  }

  FailureOr<LoopLikeOpInterface>
  createAsyncLoopLikeOp(RewriterBase &rewriter, LoopLikeOpInterface loop_op) {
    if (auto scfForOp =
            dyn_cast_if_present<scf::ForOp>(loop_op.getOperation())) {
      return createAsyncLoopLikeOp(rewriter, scfForOp);
    } else if (auto scfParOp = dyn_cast_if_present<scf::ParallelOp>(
                   loop_op.getOperation())) {
      return createAsyncLoopLikeOp(rewriter, scfParOp);
    } else {
      loop_op->emitOpError("unknown LoopLikeOpInterface op type. Currently "
                           "supports scf.for and scf.parallel");
      return failure();
    }
  }
  FailureOr<LoopLikeOpInterface> createAsyncLoopLikeOp(RewriterBase &rewriter,
                                                       scf::ForOp &loop_op) {

    // (2) Create a new wait_all event before the for op which collects the
    // incoming deps.
    SmallVector<Value, 4> incoming_tokens;
    SmallVector<Value, 4> constants;
    llvm::SetVector<Value> region_args;
    getUsedValuesDefinedAbove(loop_op.getRegion(), region_args);
    for (Value v : region_args) {
      if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
        constants.push_back(v);
      else if (v.getDefiningOp()) {
        if (auto v_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
                v.getDefiningOp())) {
          if (v_op.getAsyncToken() == v)
            incoming_tokens.push_back(v);
        } else if (isa_and_present<LoopLikeOpInterface,
                                   RegionBranchOpInterface>(
                       v.getDefiningOp())) {
          auto v_op = v.getDefiningOp();
          // Check if v is an async token from this op
          auto token = air::getAsyncTokenFromOp(v_op);
          if (token && token == v)
            incoming_tokens.push_back(v);
        }
      }
    }
    air::WaitAllOp wait_all_op_before_loop =
        insertWaitAllOpAtLoopBegin<scf::ForOp>(rewriter, loop_op, "for",
                                               incoming_tokens, constants);

    // (3) Create new for op with iter_args.
    scf::ForOp new_loop_op = convertScfForToAsync(
        rewriter, loop_op, wait_all_op_before_loop, incoming_tokens, constants);

    // Replace old loop's results with new loop's corresponding results
    // The new loop has: [existing results..., async token result]
    for (unsigned i = 0; i < loop_op.getNumResults(); i++) {
      loop_op.getResult(i).replaceAllUsesWith(new_loop_op.getResult(i));
    }

    if (eraseOpWithCheck(rewriter, loop_op, "insertLoopCarriedDeps").failed()) {
      signalPassFailure();
    }
    return dyn_cast_if_present<LoopLikeOpInterface>(new_loop_op.getOperation());
  }

  FailureOr<LoopLikeOpInterface>
  createAsyncLoopLikeOp(RewriterBase &rewriter, scf::ParallelOp &loop_op) {

    // (2) Create a new wait_all event before the parallel op which collects
    // the incoming deps.
    SmallVector<Value, 4> incoming_tokens;
    SmallVector<Value, 4> constants;
    llvm::SetVector<Value> region_args;
    getUsedValuesDefinedAbove(loop_op.getRegion(), region_args);
    for (Value v : region_args) {
      if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
        constants.push_back(v);
      else if (v.getDefiningOp()) {
        if (auto v_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
                v.getDefiningOp())) {
          if (v_op.getAsyncToken() == v)
            incoming_tokens.push_back(v);
        } else if (isa_and_present<LoopLikeOpInterface,
                                   RegionBranchOpInterface>(
                       v.getDefiningOp())) {
          auto v_op = v.getDefiningOp();
          // Check if v is an async token from this op
          auto token = air::getAsyncTokenFromOp(v_op);
          if (token && token == v)
            incoming_tokens.push_back(v);
        }
      }
    }
    air::WaitAllOp wait_all_op_before_loop =
        insertWaitAllOpAtLoopBegin<scf::ParallelOp>(
            rewriter, loop_op, "parallel", incoming_tokens, constants);

    // (3) Create new parallel op with init_val.
    scf::ParallelOp new_loop_op = convertScfParToAsync(
        rewriter, loop_op, wait_all_op_before_loop, incoming_tokens, constants);

    if (eraseOpWithCheck(rewriter, loop_op, "insertLoopCarriedDeps 2").failed())
      signalPassFailure();
    return dyn_cast_if_present<LoopLikeOpInterface>(new_loop_op.getOperation());
  }

  FailureOr<RegionBranchOpInterface>
  createAsyncRegionBranchOp(RewriterBase &rewriter,
                            RegionBranchOpInterface branch_op) {
    if (auto scfIfOp =
            dyn_cast_if_present<scf::IfOp>(branch_op.getOperation())) {
      return createAsyncRegionBranchOp(rewriter, scfIfOp);
    } else if (auto affineIfOp = dyn_cast_if_present<affine::AffineIfOp>(
                   branch_op.getOperation())) {
      return createAsyncRegionBranchOp(rewriter, affineIfOp);
    } else if (auto indexSwitchOp = dyn_cast_if_present<scf::IndexSwitchOp>(
                   branch_op.getOperation())) {
      return createAsyncRegionBranchOp(rewriter, indexSwitchOp);
    } else {
      branch_op->emitOpError("unknown RegionBranchOpInterface op type. "
                             "Currently supports scf.if / affine.if / "
                             "scf.index_switch.");
      return failure();
    }
  }
  FailureOr<RegionBranchOpInterface>
  createAsyncRegionBranchOp(RewriterBase &rewriter, scf::IfOp &branch_op) {

    // Create a new wait_all event before the branch op which collects the
    // incoming deps.
    SmallVector<Value, 4> incoming_tokens;
    SmallVector<Value, 4> constants;
    llvm::SetVector<Value> region_args;
    auto regions = branch_op->getRegions();
    for (auto &region : regions) {
      if (region.empty())
        continue;
      getUsedValuesDefinedAbove(region, region_args);
      for (Value v : region_args) {
        if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
          constants.push_back(v);
        else if (v.getDefiningOp()) {
          if (auto v_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
                  v.getDefiningOp())) {
            if (v_op.getAsyncToken() == v)
              incoming_tokens.push_back(v);
          } else if (isa_and_present<LoopLikeOpInterface,
                                     RegionBranchOpInterface>(
                         v.getDefiningOp())) {
            auto v_op = v.getDefiningOp();
            // Check if v is an async token from this op
            auto token = air::getAsyncTokenFromOp(v_op);
            if (token && token == v)
              incoming_tokens.push_back(v);
          }
        }
      }
    }
    insertWaitAllOpAtLoopBegin<scf::IfOp>(rewriter, branch_op, "if",
                                          incoming_tokens, constants);
    // Create new for op with iter_args.
    scf::IfOp new_branch_op = convertScfIfToAsync(rewriter, branch_op);

    if (eraseOpWithCheck(rewriter, branch_op, "insertLoopCarriedDeps 3")
            .failed()) {
      signalPassFailure();
    }
    return dyn_cast_if_present<RegionBranchOpInterface>(
        new_branch_op.getOperation());
  }

  FailureOr<RegionBranchOpInterface>
  createAsyncRegionBranchOp(RewriterBase &rewriter,
                            affine::AffineIfOp &branch_op) {

    // Create a new wait_all event before the branch op which collects the
    // incoming deps.
    SmallVector<Value, 4> incoming_tokens;
    SmallVector<Value, 4> constants;
    llvm::SetVector<Value> region_args;
    auto regions = branch_op->getRegions();
    for (auto &region : regions) {
      getUsedValuesDefinedAbove(region, region_args);
      for (Value v : region_args) {
        if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
          constants.push_back(v);
        else if (v.getDefiningOp()) {
          if (auto v_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
                  v.getDefiningOp())) {
            if (v_op.getAsyncToken() == v)
              incoming_tokens.push_back(v);
          } else if (isa_and_present<LoopLikeOpInterface,
                                     RegionBranchOpInterface>(
                         v.getDefiningOp())) {
            auto v_op = v.getDefiningOp();
            // Check if v is an async token from this op
            auto token = air::getAsyncTokenFromOp(v_op);
            if (token && token == v)
              incoming_tokens.push_back(v);
          }
        }
      }
    }
    air::WaitAllOp waAtLoopBegin =
        insertWaitAllOpAtLoopBegin<affine::AffineIfOp>(
            rewriter, branch_op, "if", incoming_tokens, constants);
    // Create new for op with iter_args.
    affine::AffineIfOp new_branch_op =
        convertAffineIfToAsync(rewriter, branch_op);

    if (eraseOpWithCheck(rewriter, branch_op, "insertLoopCarriedDeps 4")
            .failed()) {
      signalPassFailure();
    }

    // Connect branch op regions to the overall dependency graph.
    for (Region &branchReg : new_branch_op->getRegions()) {
      for (Value tok : incoming_tokens) {
        replaceAllUsesInRegionWith(tok, waAtLoopBegin.getAsyncToken(),
                                   branchReg);
      }
    }

    return dyn_cast_if_present<RegionBranchOpInterface>(
        new_branch_op.getOperation());
  }

  FailureOr<RegionBranchOpInterface>
  createAsyncRegionBranchOp(RewriterBase &rewriter,
                            scf::IndexSwitchOp &branch_op) {
    // Collect incoming async deps used across all (default + case) regions.
    SmallVector<Value, 4> incoming_tokens;
    SmallVector<Value, 4> constants;
    llvm::SetVector<Value> region_args;
    for (Region &region : branch_op->getRegions()) {
      if (region.empty())
        continue;
      getUsedValuesDefinedAbove(region, region_args);
      for (Value v : region_args) {
        if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
          constants.push_back(v);
        else if (v.getDefiningOp()) {
          if (auto v_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
                  v.getDefiningOp())) {
            if (v_op.getAsyncToken() == v)
              incoming_tokens.push_back(v);
          } else if (isa_and_present<LoopLikeOpInterface,
                                     RegionBranchOpInterface>(
                         v.getDefiningOp())) {
            auto v_op = v.getDefiningOp();
            auto token = air::getAsyncTokenFromOp(v_op);
            if (token && token == v)
              incoming_tokens.push_back(v);
          }
        }
      }
    }
    air::WaitAllOp waAtLoopBegin =
        insertWaitAllOpAtLoopBegin<scf::IndexSwitchOp>(
            rewriter, branch_op, "index_switch", incoming_tokens, constants);
    scf::IndexSwitchOp new_branch_op =
        convertScfIndexSwitchToAsync(rewriter, branch_op);

    if (eraseOpWithCheck(rewriter, branch_op, "insertLoopCarriedDeps 5")
            .failed()) {
      signalPassFailure();
    }

    // Connect branch op regions to the overall dependency graph.
    for (Region &branchReg : new_branch_op->getRegions()) {
      for (Value tok : incoming_tokens) {
        replaceAllUsesInRegionWith(tok, waAtLoopBegin.getAsyncToken(),
                                   branchReg);
      }
    }

    return dyn_cast_if_present<RegionBranchOpInterface>(
        new_branch_op.getOperation());
  }

  void insertLoopCarriedDepsInRegion(
      RewriterBase &rewriter, Region &region,
      SmallVector<Value, 1> yielded_tokens_in_loop_op) {

    // Create one wait_all event at the end of current for loop body.
    air::WaitAllOp wait_all_op_yielded = insertWaitAllOpBeforeLoopYield(
        rewriter, region, yielded_tokens_in_loop_op);

    // Update graph
    ExecuteGraph::VertexId wait_all_op_yielded_v =
        addVertexWaitAllOpBeforeLoopYield(yielded_tokens_in_loop_op,
                                          air::to_string(region.getParentOp()));
    // Update op-to-graph map for wait_all ops
    wa_to_g[wait_all_op_yielded.getId()] = wait_all_op_yielded_v;

    if (isa<scf::ForOp, scf::IfOp, scf::IndexSwitchOp>(region.getParentOp())) {
      // For scf::ForOp / scf::IfOp / scf::IndexSwitchOp, preserve existing
      // yield values and append the async token (all use an scf.yield
      // terminator).
      SmallVector<Value, 4> yield_values;

      // Get existing yield if it exists and extract its operands
      if (region.front().mightHaveTerminator()) {
        if (auto old_yield = dyn_cast_if_present<scf::YieldOp>(
                region.front().getTerminator())) {
          yield_values.append(old_yield.getOperands().begin(),
                              old_yield.getOperands().end());
          rewriter.eraseOp(old_yield);
        }
      }

      // Append the async token
      yield_values.push_back(wait_all_op_yielded.getResult(0));
      rewriter.setInsertionPointToEnd(&region.front());
      scf::YieldOp::create(rewriter, rewriter.getUnknownLoc(), yield_values);
    }

    else if (isa<scf::ParallelOp>(region.getParentOp())) {
      // Remove the old scf::YieldOp
      SmallVector<scf::YieldOp, 2> y_ops(region.getOps<scf::YieldOp>());
      for (auto y_op : y_ops)
        if (eraseOpWithCheck(rewriter, y_op, "insertLoopCarriedDeps").failed())
          signalPassFailure();

      // Create scf::ReduceOp
      rewriter.setInsertionPointToEnd(&region.front());
      air::createSCFReduceForAsyncSCFParallel(
          rewriter, rewriter.getUnknownLoc(),
          wait_all_op_yielded.getAsyncToken(), rewriter.getContext());
    }

    else if (isa<affine::AffineIfOp>(region.getParentOp())) {
      // Preserve existing yield values and append the async token.
      SmallVector<Value, 4> yield_values;
      if (region.front().mightHaveTerminator()) {
        if (auto old_yield = dyn_cast_if_present<affine::AffineYieldOp>(
                region.front().getTerminator())) {
          yield_values.append(old_yield.getOperands().begin(),
                              old_yield.getOperands().end());
          rewriter.eraseOp(old_yield);
        }
      }
      yield_values.push_back(wait_all_op_yielded.getResult(0));
      rewriter.setInsertionPointToEnd(&region.front());
      affine::AffineYieldOp::create(rewriter, rewriter.getUnknownLoc(),
                                    yield_values);
    }

    // Elevating tokens from inside forOp body to the yielded token, to maintain
    // dominance
    elevateAsyncTokens<air::AsyncOpInterface>(region, wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ForOp>(region, wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ParallelOp>(region, wait_all_op_yielded_v);
    elevateAsyncTokens<affine::AffineIfOp>(region, wait_all_op_yielded_v);
  }

  // Check if current for op is the single child in a parent for op
  bool isSingleChildInParentForOp(scf::ForOp child_for_op) {
    if (!child_for_op->getParentOp())
      return false;
    if (!dyn_cast_if_present<scf::ForOp>(child_for_op->getParentOp()))
      return false;
    auto parent_op =
        dyn_cast_if_present<scf::ForOp>(child_for_op->getParentOp());
    if (parent_op.getBody()->getOperations().size() == 2)
      return true; // child for op plus terminator
    else
      return false;
  }

  // Count for loop nest dimensions
  unsigned getNumberOfNestedForOps(scf::ForOp for_op) {
    unsigned number_of_nested_for_ops = 1;
    scf::ForOp parent_for_op;
    scf::ForOp currnet_for_op = for_op;
    while (isSingleChildInParentForOp(currnet_for_op)) {
      number_of_nested_for_ops++;
      auto parent_op = currnet_for_op->getParentOp();
      if (dyn_cast_if_present<scf::ForOp>(parent_op))
        currnet_for_op = dyn_cast_if_present<scf::ForOp>(parent_op);
    }
    return number_of_nested_for_ops;
  }

  // Dependency graph
  ExecuteGraph asyncExecuteGraph;

  operation_id_to_vertex_map
      region_to_g; // Map between air executes and vertices in graph
  operation_id_to_vertex_map
      dma_to_g; // Map between air dmamemcpy2d and vertices in graph
  operation_id_to_vertex_map
      channel_to_g; // Map between air channel put/get and vertices in graph
  operation_id_to_vertex_map
      hier_to_g; // Map between air hierarchy and vertices in graph
  operation_id_to_vertex_map
      wa_to_g; // Map between air wait_all and vertices in graph

  // (sink, source) dependencies whose source is inside a branch op not
  // containing the sink; resolved by addBranchDeps.
  SmallVector<std::pair<Operation *, Operation *>> branchDeps;

  // g vertex to air op mapping
  air::ExecuteOp getExecuteOpFromVertex(
      ExecuteGraph::VertexId v, ExecuteGraph g,
      llvm::DenseMap<std::pair<StringRef, int>, Operation *> &opIdToOpMap) {
    auto op = opIdToOpMap[std::make_pair("execute", g[v].operationId)];
    if (g[v].asyncEventType != "execute") {
      op->emitOpError("vertex is not an ExecuteOp.");
      return air::ExecuteOp();
    }
    return dyn_cast_if_present<air::ExecuteOp>(op);
  }
  air::DmaMemcpyNdOp getDmaOpFromVertex(
      ExecuteGraph::VertexId v, ExecuteGraph g,
      llvm::DenseMap<std::pair<StringRef, int>, Operation *> &opIdToOpMap) {
    auto op = opIdToOpMap[std::make_pair("dma", g[v].operationId)];
    if (g[v].asyncEventType != "dma") {
      op->emitOpError("vertex is not a DmaMemcpy op.");
      return air::DmaMemcpyNdOp();
    }
    return dyn_cast_if_present<air::DmaMemcpyNdOp>(op);
  }
  air::ChannelInterface getChannelOpFromVertex(
      ExecuteGraph::VertexId v, ExecuteGraph g,
      llvm::DenseMap<std::pair<StringRef, int>, Operation *> &opIdToOpMap) {
    auto op = opIdToOpMap[std::make_pair("channel", g[v].operationId)];
    if (g[v].asyncEventType != "channel") {
      op->emitOpError("vertex is not a Channel op.");
      return air::ChannelInterface();
    }
    return dyn_cast_if_present<air::ChannelInterface>(op);
  }
  air::HierarchyInterface getHierOpFromVertex(
      ExecuteGraph::VertexId v, ExecuteGraph g,
      llvm::DenseMap<std::pair<StringRef, int>, Operation *> &opIdToOpMap) {
    auto op = opIdToOpMap[std::make_pair("hierarchy", g[v].operationId)];
    if (g[v].asyncEventType != "hierarchy") {
      op->emitOpError("vertex is not a Hierarchy op.");
      return air::HierarchyInterface();
    }
    return dyn_cast_if_present<air::HierarchyInterface>(op);
  }

  // air execute op to g vertex mapping
  ExecuteGraph::VertexId getGraphGVertexFromAIROp(air::ExecuteOp op) {
    return region_to_g[op.getId()];
  }

  ExecuteGraph::VertexId getGraphGVertexFromAIROp(air::DmaMemcpyNdOp op) {
    return dma_to_g[op.getId()];
  }

  ExecuteGraph::VertexId getGraphGVertexFromAIROp(air::ChannelInterface op) {
    return channel_to_g[op.getId()];
  }

  ExecuteGraph::VertexId getGraphGVertexFromAIROp(air::HierarchyInterface op) {
    return hier_to_g[op.getId()];
  }

  ExecuteGraph::VertexId getGraphGVertexFromAIROp(air::WaitAllOp op) {
    return wa_to_g[op.getId()];
  }

  ExecuteGraph::VertexId getGraphGVertexFromAIROp(scf::ForOp op) {
    // Note: using forOp's last wait_all's id as the forOp's id
    air::WaitAllOp last_wa_op;
    // MLIR iterators cannot get the last element directly?
    for (air::WaitAllOp wa_op : op.getOps<air::WaitAllOp>()) {
      last_wa_op = wa_op;
    }
    return wa_to_g[last_wa_op.getId()];
  }

  ExecuteGraph::VertexId getGraphGVertexFromAIROp(scf::ParallelOp op) {
    // Note: using parallelOp's last wait_all's id as the parallelOp's id
    air::WaitAllOp last_wa_op;
    // MLIR iterators cannot get the last element directly?
    for (air::WaitAllOp wa_op : op.getOps<air::WaitAllOp>()) {
      last_wa_op = wa_op;
    }
    return wa_to_g[last_wa_op.getId()];
  }

  // Fill in dep list of air async ops using graph tr's connectivity
  template <typename T>
  void fillAIRDepListUsingGraphTR(
      T op,
      llvm::DenseMap<std::pair<StringRef, int>, Operation *> &opIdToOpMap) {
    if (auto async_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
            op.getOperation())) {
      uint64_t dstTRVertex = getGraphGVertexFromAIROp(op);
      for (auto TRVertex :
           asyncExecuteGraph.inverseAdjacentVertices(dstTRVertex)) {
        Operation *srcOp = nullptr;
        Value depToken;
        if (asyncExecuteGraph[TRVertex].asyncEventType == "execute") {
          auto execOp =
              getExecuteOpFromVertex(TRVertex, asyncExecuteGraph, opIdToOpMap);
          srcOp = execOp.getOperation();
          depToken = execOp.getResult(0);
        } else if (asyncExecuteGraph[TRVertex].asyncEventType == "dma") {
          srcOp = getDmaOpFromVertex(TRVertex, asyncExecuteGraph, opIdToOpMap)
                      .getOperation();
          depToken = srcOp->getResult(0);
        } else if (asyncExecuteGraph[TRVertex].asyncEventType == "channel") {
          srcOp =
              getChannelOpFromVertex(TRVertex, asyncExecuteGraph, opIdToOpMap)
                  .getOperation();
          depToken = srcOp->getResult(0);
        } else if (asyncExecuteGraph[TRVertex].asyncEventType == "hierarchy") {
          srcOp = getHierOpFromVertex(TRVertex, asyncExecuteGraph, opIdToOpMap)
                      .getOperation();
          depToken = srcOp->getResult(0);
        } else {
          op->emitOpError("unknown async event type");
          continue;
        }
        // A source op inside a branch op (scf.if, affine.if,
        // scf.index_switch) that does not contain the sink: its token cannot
        // dominate the sink. The branch op only gets a token in the 4th
        // traversal, so record the dependency and make the sink wait for the
        // branch op's token then (addBranchDeps).
        if (srcOp && getEnclosingBranchNotContaining(srcOp, op)) {
          branchDeps.push_back({op.getOperation(), srcOp});
          continue;
        }
        async_op.addAsyncDependency(depToken);
      }
    } else
      op->emitOpError("operation has no async interface");
  }

  // Whether every token of `targets` is reached from `tokens` through async
  // dependencies (and scf.for/scf.parallel init tokens). Conservative: false
  // when unsure.
  bool tokensFollowAll(ArrayRef<Value> tokens, ArrayRef<Value> targets) {
    llvm::SmallPtrSet<void *, 16> seen;
    llvm::SmallVector<Value> work(tokens.begin(), tokens.end());
    llvm::SmallPtrSet<void *, 16> reached;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      if (!seen.insert(v.getAsOpaquePointer()).second)
        continue;
      reached.insert(v.getAsOpaquePointer());
      Operation *d = v.getDefiningOp();
      if (!d)
        continue;
      if (auto a = dyn_cast<air::AsyncOpInterface>(d))
        for (Value dep : a.getAsyncDependencies())
          work.push_back(dep);
      else if (auto f = dyn_cast<scf::ForOp>(d))
        for (Value init : f.getInitArgs())
          work.push_back(init);
      else if (auto par = dyn_cast<scf::ParallelOp>(d))
        for (Value init : par.getInitVals())
          work.push_back(init);
      else if (isa<scf::IfOp, affine::AffineIfOp, scf::IndexSwitchOp>(d)) {
        // A branch op's token follows what every region yields.
        for (Region &r : d->getRegions()) {
          if (r.empty() || !r.front().mightHaveTerminator())
            return false;
          for (Value y : r.front().getTerminator()->getOperands())
            if (isa<air::AsyncTokenType>(y.getType()))
              work.push_back(y);
        }
      }
    }
    return llvm::all_of(targets, [&](Value t) {
      return reached.contains(t.getAsOpaquePointer());
    });
  }

  // The outermost branch op (scf.if, affine.if, scf.index_switch) enclosing
  // `src` but not `sink`, if any.
  Operation *getEnclosingBranchNotContaining(Operation *src, Operation *sink) {
    Operation *found = nullptr;
    for (Operation *a = src->getParentOp(); a; a = a->getParentOp()) {
      if (a->isAncestor(sink))
        break;
      if (isa<scf::IfOp, affine::AffineIfOp, scf::IndexSwitchOp>(a))
        found = a;
    }
    return found;
  }

  // After the 4th traversal: each sink recorded in `branchDeps` waits for the
  // token of the op that encloses its source op in the block holding the sink
  // (or one of its ancestors): the branch op, now async, or a loop around it.
  void addBranchDeps() {
    for (auto [sink, src] : branchDeps) {
      auto async_sink = dyn_cast<air::AsyncOpInterface>(sink);
      if (!async_sink)
        continue;
      Operation *a = src;
      Operation *sinkAncestor = nullptr;
      for (; a; a = a->getParentOp()) {
        if (!a->getBlock())
          continue;
        if ((sinkAncestor = a->getBlock()->findAncestorOpInBlock(*sink)))
          break;
      }
      if (!a || !sinkAncestor || a == sinkAncestor ||
          !a->isBeforeInBlock(sinkAncestor))
        continue;
      // A token cannot cross into an op isolated from above (a hierarchy
      // op); the hierarchy op's own dependencies order it instead.
      bool isolated = false;
      for (Operation *p = sink->getParentOp();
           p && p != sinkAncestor->getParentOp(); p = p->getParentOp())
        if (p->hasTrait<OpTrait::IsIsolatedFromAbove>())
          isolated = true;
      if (isolated)
        continue;
      Value token = air::getAsyncTokenFromOp(a);
      if (!token ||
          llvm::is_contained(async_sink.getAsyncDependencies(), token))
        continue;
      async_sink.addAsyncDependency(token);
      // The branch ops between the sink and the token's block now follow the
      // token too, whichever region runs: an op after one of them may rely
      // on that order through the sink (the transitive reduction dropped its
      // direct edge), and a region without the sink would not provide it.
      for (Operation *p = sink->getParentOp();
           p && p != sinkAncestor->getParentOp(); p = p->getParentOp())
        if (isa<scf::IfOp, affine::AffineIfOp, scf::IndexSwitchOp>(p))
          orderBranchAfter(p, token);
    }
    branchDeps.clear();
  }

  // Make every region of branch op `branch` yield a token that follows
  // `token`, so the branch op's token does.
  void orderBranchAfter(Operation *branch, Value token) {
    for (Region &r : branch->getRegions()) {
      if (r.empty() || !r.front().mightHaveTerminator())
        continue;
      Operation *term = r.front().getTerminator();
      for (OpOperand &y : term->getOpOperands()) {
        if (!isa<air::AsyncTokenType>(y.get().getType()) ||
            tokensFollowAll({y.get()}, {token}))
          continue;
        if (auto wa = y.get().getDefiningOp<air::WaitAllOp>();
            wa && wa->getBlock() == &r.front()) {
          wa.addAsyncDependency(token);
          continue;
        }
        OpBuilder builder(term);
        auto wa =
            air::WaitAllOp::create(builder, term->getLoc(),
                                   air::AsyncTokenType::get(term->getContext()),
                                   SmallVector<Value>{y.get(), token});
        wa->setAttr("id", builder.getI32IntegerAttr(++WaitAllOpID));
        y.set(wa.getAsyncToken());
      }
    }
  }

  template <typename T>
  void addAsyncDepToGraphIfNew(Value dep, T op) {
    if (auto async_op = mlir::dyn_cast_if_present<air::AsyncOpInterface>(
            op.getOperation())) {
      for (auto old_dep : async_op.getAsyncDependencies())
        if (old_dep == dep)
          return;

      // Add edge to graph, iff dep is async execute region (i.e. not a
      // loop iterator)
      if (auto srcOp = dep.getDefiningOp()) {
        uint64_t srcNode = 0;
        if (auto execute_op = dyn_cast_if_present<air::ExecuteOp>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(execute_op);
        } else if (auto dma_op =
                       dyn_cast_if_present<air::DmaMemcpyNdOp>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(dma_op);
        } else if (auto channel_op =
                       dyn_cast_if_present<air::ChannelInterface>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(channel_op);
        } else if (auto hier_op =
                       dyn_cast_if_present<air::HierarchyInterface>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(hier_op);
        } else if (auto wa_op = dyn_cast_if_present<air::WaitAllOp>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(wa_op);
        } else
          srcOp->emitOpError(
              "dependency token should be generated by an async op");
        uint64_t dstNode = getGraphGVertexFromAIROp(op);
        asyncExecuteGraph.addEdge(srcNode, dstNode);
      }
    } else
      op->emitOpError("operation has no async interface");
  }

  //===----------------------------------------------------------------------===//
  // Other utilities
  //===----------------------------------------------------------------------===//

  // Whether a read of `tile_0` reads what a write of `tile_1` wrote, for a
  // host buffer. A launch can read back from host memory what its own drains
  // wrote, at an offset inside the drained region or from a loop over it, and
  // the equal-start test below misses both. An access whose footprint is not
  // known keeps the equal-start test alone.
  //
  // Not only host buffers: an L2 buffer that several transfers fill at
  // different offsets and one transfer then reads whole (a gather, as a
  // memtile assembling the blocks of several cores) has the same shape, and
  // without these edges the read waited for the first fill only.
  bool footprintsIntersectInHostMemory(partialMemref *tile_0,
                                       partialMemref *tile_1) {
    auto ty = dyn_cast<BaseMemRefType>(tile_0->memrefValue.getType());
    if (!ty || !(air::isL3(ty) || air::isL2(ty)))
      return false;
    return air::accessesIntersect(tile_0->offsets, tile_0->sizes,
                                  tile_0->strides, tile_1->offsets,
                                  tile_1->sizes, tile_1->strides)
        .value_or(false);
  }

  // Check if two partial memref tiles have potentially conflicting access
  // patterns. Returns true if the accesses overlap or cannot be proven
  // disjoint. A full-buffer access (empty offsets) is treated as conflicting
  // with any other access.
  bool areEqualIndexPartialMemrefs(partialMemref *tile_0,
                                   partialMemref *tile_1) {
    // A full-buffer access (empty offsets) overlaps with any other access.
    if (tile_0->offsets.empty() || tile_1->offsets.empty())
      return true;

    // Check if all static offsets of each partialMemref lead to equal overall
    // offset.
    auto getOffsetFromOffsetsAndStrides = [&](partialMemref *tile) {
      unsigned offset = 0;
      for (unsigned i = 0; i < tile->offsets.size(); i++) {
        auto constOffset = getConstantIntValue(tile->offsets[i]);
        auto constStride = getConstantIntValue(tile->strides[i]);
        if (!constOffset || !constStride)
          continue;
        offset += (*constOffset) * (*constStride);
      }
      return offset;
    };
    if (getOffsetFromOffsetsAndStrides(tile_0) !=
        getOffsetFromOffsetsAndStrides(tile_1))
      return false;

    // Check if all static offsets, hash mapped by strides, are equal across the
    // two partialMemrefs.
    auto buildStrideToVarOffsetsMap =
        [](partialMemref *tile,
           DenseMap<unsigned, llvm::SetVector<Value>> &map) {
          for (unsigned i = 0; i < tile->offsets.size(); i++) {
            auto constOffset = getConstantIntValue(tile->offsets[i]);
            auto constStride = getConstantIntValue(tile->strides[i]);
            if (!constStride)
              continue;
            if (constOffset)
              continue;
            if (auto v = dyn_cast_if_present<Value>(tile->offsets[i]))
              map[*constStride].insert(v);
          }
        };
    DenseMap<unsigned, llvm::SetVector<Value>> strideToVarOffsetsMap;
    buildStrideToVarOffsetsMap(tile_0, strideToVarOffsetsMap);
    buildStrideToVarOffsetsMap(tile_1, strideToVarOffsetsMap);
    // More than 1 unique variadic offsets per stride across the two
    // partialMemrefs.
    for (auto &[_, set] : strideToVarOffsetsMap)
      if (set.size() > 1)
        return false;

    return true;
  }

  // Check if a value is only used outside of a given block
  bool isOnlyUsedOutsideOfBlock(Value v, Block *block) {
    for (auto u : v.getUsers())
      if (u->getBlock() == block)
        return false;
    return true;
  }

  // Check if a value is not used inside a given block
  bool isNotUsedInsideOfBlock(Value v, Block *block) {
    if (v.use_empty() || isOnlyUsedOutsideOfBlock(v, block))
      return true;
    else
      return false;
  }

  // Check if op is not considered for loop-carried dependency
  bool isNotLoopCarriedOp(Operation *op) {
    if (dyn_cast_if_present<memref::DeallocOp>(op)) {
      return true;
    } else if (dyn_cast_if_present<memref::AllocOp>(op)) {
      return true;
    } else
      return false;
  }
  bool isNotLoopCarriedOp(air::AsyncOpInterface op) {
    if (auto exec_op = dyn_cast_if_present<air::ExecuteOp>(op.getOperation())) {
      auto &bb = exec_op.getRegion().front();
      Operation &child_op = bb.getOperations().front();
      return isNotLoopCarriedOp(&child_op);
    } else
      return false;
  }

  // An op that was already async: uses outside it of tokens defined inside it
  // (dependencies the 3rd traversal found) wait for its own token instead,
  // which its yields cover. elevateAsyncTokens does this for the ops this pass
  // makes async.
  void elevateTokensToOp(Operation *op) {
    Value opToken = air::getAsyncTokenFromOp(op);
    if (!opToken)
      return;
    llvm::SetVector<Operation *> users;
    op->walk([&](Operation *inner) {
      if (inner == op)
        return;
      for (Value r : inner->getResults())
        if (isa<air::AsyncTokenType>(r.getType()))
          r.replaceUsesWithIf(opToken, [&](OpOperand &use) {
            if (op->isAncestor(use.getOwner()))
              return false;
            users.insert(use.getOwner());
            return true;
          });
    });
    // Two elevated tokens leave the same dependency twice.
    for (Operation *user : users) {
      auto async = dyn_cast<air::AsyncOpInterface>(user);
      if (!async)
        continue;
      llvm::SmallPtrSet<Value, 4> seen;
      for (int i = async.getAsyncDependencies().size() - 1; i >= 0; --i)
        if (!seen.insert(async.getAsyncDependencies()[i]).second)
          async.eraseAsyncDependency(i);
    }
  }

  // The tokens certain to have completed once `v` has: `v`, what it depends
  // on, and for a token a loop or branch returns, what every way through it
  // yields in that position (a loop may also run zero times).
  llvm::SmallPtrSet<Value, 8>
  completedWith(Value v,
                llvm::DenseMap<Value, llvm::SmallPtrSet<Value, 8>> &memo) {
    if (auto it = memo.find(v); it != memo.end())
      return it->second;
    llvm::SmallPtrSet<Value, 8> out;
    out.insert(v);
    Operation *def = v.getDefiningOp();
    if (auto async = dyn_cast_if_present<air::AsyncOpInterface>(def)) {
      for (Value dep : async.getAsyncDependencies())
        for (Value c : completedWith(dep, memo))
          out.insert(c);
    } else if (isa_and_present<scf::IfOp, affine::AffineIfOp,
                               scf::IndexSwitchOp, scf::ForOp, scf::ParallelOp>(
                   def)) {
      unsigned k = cast<OpResult>(v).getResultNumber();
      SmallVector<Value> ways;
      bool known = true;
      for (Region &r : def->getRegions()) {
        if (r.empty() || !r.front().mightHaveTerminator() ||
            r.front().getTerminator()->getNumOperands() <= k) {
          known = false;
          break;
        }
        ways.push_back(r.front().getTerminator()->getOperand(k));
      }
      if (auto forOp = dyn_cast<scf::ForOp>(def))
        ways.push_back(forOp.getInitArgs()[k]);
      if (auto parOp = dyn_cast<scf::ParallelOp>(def))
        ways.push_back(parOp.getInitVals()[k]);
      if (known && !ways.empty()) {
        llvm::SmallPtrSet<Value, 8> all = completedWith(ways.front(), memo);
        for (Value w : llvm::drop_begin(ways)) {
          llvm::SmallPtrSet<Value, 8> these = completedWith(w, memo);
          llvm::SmallVector<Value> gone;
          for (Value c : all)
            if (!these.contains(c))
              gone.push_back(c);
          for (Value c : gone)
            all.erase(c);
        }
        for (Value c : all)
          out.insert(c);
      }
    }
    memo[v] = out;
    return out;
  }

  // A loop or branch in the input that already returns a token is left as it
  // is when that token covers the work inside it: every token made directly
  // in its regions has completed once what the region yields in the token's
  // position has. Otherwise it is rejected with `error`.
  LogicalResult adoptAsyncOp(Operation *op, StringRef error) {
    Value token = air::getAsyncTokenFromOp(op);
    unsigned k = cast<OpResult>(token).getResultNumber();
    llvm::DenseMap<Value, llvm::SmallPtrSet<Value, 8>> memo;
    for (Region &r : op->getRegions()) {
      if (r.empty())
        continue;
      Block &b = r.front();
      llvm::SmallPtrSet<Value, 8> done;
      if (b.mightHaveTerminator() && b.getTerminator()->getNumOperands() > k)
        done = completedWith(b.getTerminator()->getOperand(k), memo);
      for (Operation &inner : b)
        for (Value res : inner.getResults())
          if (isa<air::AsyncTokenType>(res.getType()) && !done.contains(res))
            return op->emitOpError(error);
    }
    elevateTokensToOp(op);
    return success();
  }

  // Check if a value is not used inside a given block, or is only used by ops
  // not considered for loop-carried dependency
  bool isOnlyUsedByNoLoopCarryOpsInBlock(Value token, Block *block) {
    if (token.use_empty())
      return true;
    bool isOnlyUsedByNoCarryOps = true;
    for (auto user : token.getUsers()) {
      if (user->getBlock() == block) {
        if (auto async_user = dyn_cast_if_present<air::ExecuteOp>(user)) {
          auto &bb = async_user.getRegion().front();
          Operation &child_op = bb.getOperations().front();
          if (!isNotLoopCarriedOp(&child_op))
            isOnlyUsedByNoCarryOps = false;
        } else
          isOnlyUsedByNoCarryOps = false;
      }
    }
    if (isOnlyUsedByNoCarryOps)
      return true;
    else
      return false;
  }
};

} // namespace air
} // namespace xilinx

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyPass() {
  return std::make_unique<AIRDependency>();
}

} // namespace air
} // namespace xilinx
