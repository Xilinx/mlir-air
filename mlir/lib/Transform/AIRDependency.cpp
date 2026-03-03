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

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (air::isAsyncOp(op) || isa<air::WaitAllOp>(op)) {
          assignOpId(op);
          updateAsyncExecuteGraphWithNewNode(op, asyncExecuteGraph);
          return; // Skip if is already async.
        }
        if (op->getParentOfType<linalg::LinalgOp>())
          return; // Skip if is inside a linalg.generic.

        if (isa<air::DmaMemcpyNdOp>(op))
          createAsyncDMA(rewriter, op);
        else if (isa<air::ChannelInterface>(op))
          createAsyncChannel(rewriter, op);
        else if (isa<linalg::LinalgOp, func::CallOp, memref::DeallocOp>(op))
          createAsyncExecute(rewriter, op);
        else if (isa<memref::CopyOp>(op)) {
          // Skip wrapping memref.copy in air.execute when inside scf.if,
          // as the resulting async token would not dominate uses outside
          // the enclosing loop. L1-to-L1 copies are synchronous and don't
          // need async tracking.
          if (!op->getParentOfType<scf::IfOp>())
            createAsyncExecute(rewriter, op);
        } else if (auto hierarchy_op = dyn_cast<air::HierarchyInterface>(op))
          createAsyncHierarchyImpls(rewriter, hierarchy_op);
        // Create async execute region for memref.alloc
        else if (auto memalloc_op = dyn_cast<memref::AllocOp>(op)) {
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
          if (auto storeOp = dyn_cast<memref::StoreOp>(op))
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

    // 2nd traversal: trace deps among async execute regions; build a dep graph

    llvm::DenseMap<std::pair<StringRef, int>, Operation *> opIdToOpMap;
    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        Operation *sink_op = nullptr;
        if (auto async_execute_op = dyn_cast<air::ExecuteOp>(op)) {
          for (auto &child_op : async_execute_op.getChildOps())
            if (!dyn_cast<air::ExecuteTerminatorOp>(child_op))
              sink_op = &child_op;
        } else if (isa<air::AsyncOpInterface>(op))
          sink_op = op;
        else
          return;

        // Preserve any existing async dependency edges in the input IR, before
        // creating new ones.
        if (auto execute_op = dyn_cast<air::ExecuteOp>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::ExecuteOp>(dep, execute_op);
          }
        } else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::DmaMemcpyNdOp>(dep, dma_op);
          }
        } else if (auto channel_op = dyn_cast<air::ChannelInterface>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::ChannelInterface>(dep, channel_op);
          }
        } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
          for (auto dep : air::getAsyncDependenciesFromOp(op)) {
            addAsyncDepToGraphIfNew<air::HierarchyInterface>(dep, hier_op);
          }
        }

        SmallVector<partialMemref, 1> sink_op_memref_reads;
        SmallVector<partialMemref, 1> sink_op_memref_writes;
        SmallVector<Value, 1> sink_op_scalar_ins;
        SmallVector<Value, 1> sink_op_scalar_outs;

        dependencyTracer DTObject;
        DTObject.getPartialMemrefFromOp(
            sink_op, sink_op_memref_reads, sink_op_memref_writes,
            sink_op_scalar_ins, sink_op_scalar_outs);

        // Detect dependencies
        if (auto async_execute_op = dyn_cast<air::ExecuteOp>(op)) {
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
        } else if (auto dma_op = mlir::dyn_cast<air::DmaMemcpyNdOp>(op)) {
          traceDeps<air::DmaMemcpyNdOp>(sink_op_memref_reads, dma_op, "RAW");
          traceDeps<air::DmaMemcpyNdOp>(sink_op_memref_writes, dma_op,
                                        "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs, dma_op);
          opIdToOpMap[std::make_pair("dma", dma_op.getId())] = dma_op;
        } else if (auto channel_op =
                       mlir::dyn_cast<air::ChannelInterface>(op)) {
          traceDeps<air::ChannelInterface>(sink_op_memref_reads, channel_op,
                                           "RAW");
          traceDeps<air::ChannelInterface>(sink_op_memref_writes, channel_op,
                                           "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs, channel_op);
          opIdToOpMap[std::make_pair("channel", channel_op.getId())] =
              channel_op;
        } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
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
        if (auto async_execute_op = dyn_cast<air::ExecuteOp>(op)) {
          fillAIRDepListUsingGraphTR<air::ExecuteOp>(async_execute_op,
                                                     opIdToOpMap);
        }
        // Fill dep list of air dmamemcpy2d ops
        else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(op)) {
          fillAIRDepListUsingGraphTR<air::DmaMemcpyNdOp>(dma_op, opIdToOpMap);
        }
        // Fill dep list of air channel ops
        else if (auto channel_op = dyn_cast<air::ChannelInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::ChannelInterface>(channel_op,
                                                            opIdToOpMap);
        }
        // Fill dep list of air hierarchy ops
        else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::HierarchyInterface>(hier_op,
                                                              opIdToOpMap);
        }
        // Fill dep list of air wait_all ops
        else if (auto wa_op = dyn_cast<air::WaitAllOp>(op)) {
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
      return yielded_tokens;
    };

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (LoopLikeOpInterface loop_op = dyn_cast<LoopLikeOpInterface>(op)) {
          Region &region = *loop_op.getLoopRegions().front();
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
                       dyn_cast<RegionBranchOpInterface>(op)) {
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
          auto new_branch_op = createAsyncRegionBranchOp(rewriter, branch_op);
          auto new_regions = (*new_branch_op)->getRegions();
          for (unsigned i = 0; i < new_regions.size(); i++) {
            if (new_regions[i].empty())
              continue;
            insertLoopCarriedDepsInRegion(rewriter, new_regions[i],
                                          yielded_tokens_per_region[i]);
          }
        }
      });
    }
  }

private:
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
      if (auto hier_op = dyn_cast<air::HierarchyInterface>(user)) {
        for (int i = 0, e = hier_op.getNumKernelOperands(); i < e; i++) {
          if (hier_op.getKernelOperand(i) == val) {
            if (alloc_for_reshape(hier_op.getKernelArgument(i)))
              return true;
          }
        }
      } else if (auto reshape = dyn_cast<memref::ReshapeOp>(user)) {
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
    auto dma_op = mlir::dyn_cast<air::DmaMemcpyNdOp>(op);
    air::DmaMemcpyNdOp new_dmaNd_op = air::DmaMemcpyNdOp::create(
        rewriter, loc, air::AsyncTokenType::get(dma_op->getContext()), deps,
        dma_op.getDstMemref(), dma_op.getDstOffsets(), dma_op.getDstSizes(),
        dma_op.getDstStrides(), dma_op.getSrcMemref(), dma_op.getSrcOffsets(),
        dma_op.getSrcSizes(), dma_op.getSrcStrides());
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
    if (auto channel_put_op = dyn_cast<air::ChannelPutOp>(op)) {
      air::ChannelPutOp new_channel_put_op = air::ChannelPutOp::create(
          rewriter, loc, air::AsyncTokenType::get(channel_put_op->getContext()),
          deps, channel_put_op.getChanName(), channel_put_op.getIndices(),
          channel_put_op.getSrc(), channel_put_op.getSrcOffsets(),
          channel_put_op.getSrcSizes(), channel_put_op.getSrcStrides());
      assignOpId(new_channel_put_op);
      event_name = "Put";
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_channel_put_op, asyncExecuteGraph);
    } else if (auto channel_get_op = dyn_cast<air::ChannelGetOp>(op)) {
      air::ChannelGetOp new_channel_get_op = air::ChannelGetOp::create(
          rewriter, loc, air::AsyncTokenType::get(channel_get_op->getContext()),
          deps, channel_get_op.getChanName(), channel_get_op.getIndices(),
          channel_get_op.getDst(), channel_get_op.getDstOffsets(),
          channel_get_op.getDstSizes(), channel_get_op.getDstStrides());
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
    if (auto launch = dyn_cast<air::LaunchOp>(op.getOperation())) {
      auto new_launch = createAsyncHierarchy<air::LaunchOp>(
          rewriter, launch, deps, args, constants);
      new_op = new_launch.getOperation();
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_launch, asyncExecuteGraph);
    } else if (auto segment = dyn_cast<air::SegmentOp>(op.getOperation())) {
      auto new_segment = createAsyncHierarchy<air::SegmentOp>(
          rewriter, segment, deps, args, constants);
      new_op = new_segment.getOperation();
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_segment, asyncExecuteGraph);
    } else if (auto herd = dyn_cast<air::HerdOp>(op.getOperation())) {
      auto new_herd = createAsyncHierarchy<air::HerdOp>(rewriter, herd, deps,
                                                        args, constants);
      new_op = new_herd.getOperation();
      // Update op-to-graph map
      updateAsyncExecuteGraphWithNewNode(new_herd, asyncExecuteGraph);
    } else {
      op->emitOpError("unknown hierarchy operation");
    }
    auto new_hier = dyn_cast<air::HierarchyInterface>(new_op);

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

  void updateAsyncExecuteGraphWithNewNode(Operation *op, ExecuteGraph &graph) {
    auto v = graph.addVertex();
    auto &node = graph[v];
    // Create a vertex out of the current async execute region
    node.asyncEventName = air::to_string(op);
    auto setNodeAttrsBasedOnOp = [&node, &v, this](Operation *op) {
      if (auto execOp = dyn_cast<air::ExecuteOp>(op)) {
        node.asyncEventType = "execute";
        node.color = "chartreuse";
        node.shape = "oval";
        node.operationId = execOp.getId();
        region_to_g[node.operationId] = v;
      } else if (auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(op)) {
        node.asyncEventType = "dma";
        node.color = "cyan";
        node.shape = "oval";
        node.operationId = dmaOp.getId();
        dma_to_g[node.operationId] = v;
      } else if (auto ciOp = dyn_cast<air::ChannelInterface>(op)) {
        node.asyncEventName = air::to_string(op);
        node.asyncEventType = "channel";
        node.color = "cyan";
        node.shape = "oval";
        node.operationId = ciOp.getId();
        channel_to_g[node.operationId] = v;
      } else if (auto hiOp = dyn_cast<air::HierarchyInterface>(op)) {
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
      else if (auto hier = dyn_cast<air::HierarchyInterface>(
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
          llvm::any_of(
              *writeAccesses,
              [&u](std::pair<Value,
                             std::tuple<SmallVector<Value>, SmallVector<Value>,
                                        SmallVector<Value>>>
                       entry) { return entry.first == u.get(); });
      foundReadAccess =
          foundReadAccess ||
          llvm::any_of(
              *readAccesses,
              [&u](std::pair<Value,
                             std::tuple<SmallVector<Value>, SmallVector<Value>,
                                        SmallVector<Value>>>
                       entry) { return entry.first == u.get(); });
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
      // If used in MemcpyInterface Op
      if (auto memcpy = dyn_cast<air::MemcpyInterface>(u.getOwner())) {
        partialMemref memcpy_src, memcpy_dst;
        if (memcpy.getSrcMemref()) {
          memcpy_src =
              partialMemref(memcpy.getSrcMemref(), memcpy.getSrcOffsets(),
                            memcpy.getSrcSizes(), memcpy.getSrcStrides());
        }
        if (memcpy.getDstMemref()) {
          memcpy_dst =
              partialMemref(memcpy.getDstMemref(), memcpy.getDstOffsets(),
                            memcpy.getDstSizes(), memcpy.getDstStrides());
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
            } else if (areEqualIndexPartialMemrefs(tile, &memcpy_dst))
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
          }
        } else {
          if (tile == nullptr) {
            addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0), op);
          } else if (u.is(memcpy.getDstMemref())) {
            if (areEqualIndexPartialMemrefs(tile, &memcpy_dst))
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
          } else if (u.is(memcpy.getSrcMemref())) {
            if (areEqualIndexPartialMemrefs(tile, &memcpy_src))
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
          }
        }
      }

      // If used in a linalg op
      else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(u.getOwner())) {
        if (auto ar = dyn_cast<air::ExecuteOp>(linalgop->getParentOp())) {
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
      else if (auto hier = dyn_cast<air::HierarchyInterface>(u.getOwner())) {
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
        if (auto ar = dyn_cast<air::ExecuteOp>(u.getOwner()->getParentOp()))
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
        OpBuilder svBuilder(subview);
        auto loc = subview->getLoc();
        SmallVector<Value> subviewOffsetVals, subviewSizeVals,
            subviewStrideVals;
        for (auto fr : subview.getMixedOffsets())
          subviewOffsetVals.push_back(
              getValueOrCreateConstantIndexOp(svBuilder, loc, fr));
        for (auto fr : subview.getMixedSizes())
          subviewSizeVals.push_back(
              getValueOrCreateConstantIndexOp(svBuilder, loc, fr));
        for (auto fr : subview.getMixedStrides())
          subviewStrideVals.push_back(
              getValueOrCreateConstantIndexOp(svBuilder, loc, fr));
        partialMemref subview_tile(subview.getSource(), subviewOffsetVals,
                                   subviewSizeVals, subviewStrideVals);
        SmallVector<partialMemref, 1> subview_operands = {subview_tile};
        traceDeps<T>(subview_operands, sink_air_op, dep_type);
      }
    }
  }

  template <typename T>
  void traceTileIndices(SmallVector<partialMemref, 1> read_operands,
                        SmallVector<partialMemref, 1> write_operands,
                        SmallVector<Value, 1> in_scalars,
                        SmallVector<Value, 1> out_scalars, T sink_air_op) {
    for (auto operand : read_operands) {
      for (auto v :
           llvm::concat<Value>(operand.offsets, operand.sizes, operand.strides))
        pushTileIndexAsDep<T>(v, sink_air_op);
    }
    for (auto operand : write_operands) {
      for (auto v :
           llvm::concat<Value>(operand.offsets, operand.sizes, operand.strides))
        pushTileIndexAsDep<T>(v, sink_air_op);
    }
    for (auto scalar : in_scalars) {
      pushTileIndexAsDep<T>(scalar, sink_air_op);
    }
    for (auto scalar : out_scalars) {
      pushTileIndexAsDep<T>(scalar, sink_air_op);
    }
  }

  //===----------------------------------------------------------------------===//
  // SCF for loop-carried dependency
  //===----------------------------------------------------------------------===//

  void insertVertexBetweenTwoOps(Operation *a, Operation *b,
                                 ExecuteGraph::VertexId v) {
    unsigned v_a = 0;
    unsigned v_b = 0;
    if (auto op = dyn_cast<air::ExecuteOp>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast<air::DmaMemcpyNdOp>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast<air::ChannelInterface>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast<air::HierarchyInterface>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast<scf::ForOp>(a)) {
      v_a = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast<scf::ParallelOp>(a)) {
      v_a = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast<air::WaitAllOp>(a)) {
      v_a = getGraphGVertexFromAIROp(op);
    }
    if (auto op = dyn_cast<air::ExecuteOp>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast<air::DmaMemcpyNdOp>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = mlir::dyn_cast<air::ChannelInterface>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast<air::HierarchyInterface>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    } else if (auto op = dyn_cast<scf::ForOp>(b)) {
      v_b = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast<scf::ParallelOp>(b)) {
      v_b = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast<air::WaitAllOp>(b)) {
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
              dyn_cast<air::ExecuteOp>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(async_execute_op);
      } else if (auto dma_op =
                     dyn_cast<air::DmaMemcpyNdOp>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(dma_op);
      } else if (auto channel_op =
                     dyn_cast<air::ChannelInterface>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(channel_op);
      } else if (auto hier_op =
                     dyn_cast<air::HierarchyInterface>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(hier_op);
      } else if (auto scf_for_op =
                     dyn_cast<scf::ForOp>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(
            scf_for_op); // g_to_tr not needed since wait_all created after TR
      } else if (auto scf_parallel_op =
                     dyn_cast<scf::ParallelOp>(token.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(
            scf_parallel_op); // g_to_tr not needed since wait_all created after
                              // TR
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
      auto async_op =
          mlir::dyn_cast<air::AsyncOpInterface>(dma_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getInitVals()[0]);
      }
    }
    for (auto channel_op : new_loop_op.getOps<air::ChannelInterface>()) {
      auto async_op =
          mlir::dyn_cast<air::AsyncOpInterface>(channel_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getInitVals()[0]);
      }
    }
    for (auto hier_op : new_loop_op.getOps<air::HierarchyInterface>()) {
      auto async_op =
          mlir::dyn_cast<air::AsyncOpInterface>(hier_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getInitVals()[0]);
      }
    }

    return new_loop_op;
  }

  scf::IfOp convertScfIfToAsync(RewriterBase &rewriter, scf::IfOp branch_op) {
    // Create new if op with a yielded async token.
    SmallVector<Type> yielded_tys = {
        air::AsyncTokenType::get(rewriter.getContext())};
    scf::IfOp new_branch_op = scf::IfOp::create(
        rewriter, branch_op.getLoc(), yielded_tys, branch_op.getCondition(),
        /*withElseRegion*/ (bool)branch_op.elseBlock());

    if (auto attr = branch_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_branch_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

    // Splice the operations inside then and else regions
    auto old_regions = branch_op->getRegions();
    auto new_regions = new_branch_op->getRegions();
    for (auto [o_r, n_r] : llvm::zip_equal(old_regions, new_regions)) {
      if (o_r.empty() || n_r.empty())
        continue;
      auto &bb = n_r.front().getOperations();
      auto &body = o_r.front().getOperations();
      bb.splice(bb.begin(), body, body.begin(), --body.end());
    }

    return new_branch_op;
  }

  affine::AffineIfOp convertAffineIfToAsync(RewriterBase &rewriter,
                                            affine::AffineIfOp branch_op) {
    // Create new if op with a yielded async token.
    SmallVector<Type> yielded_tys = {
        air::AsyncTokenType::get(rewriter.getContext())};
    affine::AffineIfOp new_branch_op = affine::AffineIfOp::create(
        rewriter, branch_op.getLoc(), yielded_tys, branch_op.getIntegerSet(),
        branch_op.getOperands(),
        /*withElseRegion*/ (bool)branch_op.hasElse());

    if (auto attr = branch_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_branch_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

    // Splice the operations inside then and else regions
    auto old_regions = branch_op->getRegions();
    auto new_regions = new_branch_op->getRegions();
    for (auto [o_r, n_r] : llvm::zip_equal(old_regions, new_regions)) {
      auto &bb = n_r.front().getOperations();
      auto &body = o_r.front().getOperations();
      bb.splice(bb.begin(), body, body.begin(), --body.end());
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
        if (auto v_op =
                mlir::dyn_cast<air::AsyncOpInterface>(v.getDefiningOp())) {
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
    return dyn_cast<LoopLikeOpInterface>(new_loop_op.getOperation());
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
        if (auto v_op =
                mlir::dyn_cast<air::AsyncOpInterface>(v.getDefiningOp())) {
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
    return dyn_cast<LoopLikeOpInterface>(new_loop_op.getOperation());
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
    } else {
      branch_op->emitOpError("unknown RegionBranchOpInterface op type. "
                             "Currently supports scf.if.");
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
          if (auto v_op =
                  mlir::dyn_cast<air::AsyncOpInterface>(v.getDefiningOp())) {
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
    return dyn_cast<RegionBranchOpInterface>(new_branch_op.getOperation());
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
          if (auto v_op =
                  mlir::dyn_cast<air::AsyncOpInterface>(v.getDefiningOp())) {
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

    return dyn_cast<RegionBranchOpInterface>(new_branch_op.getOperation());
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

    if (isa<scf::ForOp, scf::IfOp>(region.getParentOp())) {
      // For scf::ForOp and scf::IfOp, preserve existing yield values and append
      // async token
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
      // Yield an async token
      SmallVector<Value, 4> yield_token;
      yield_token.push_back(wait_all_op_yielded.getResult(0));
      rewriter.setInsertionPointToEnd(&region.front());
      affine::AffineYieldOp::create(rewriter, rewriter.getUnknownLoc(),
                                    yield_token);
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
    if (!dyn_cast<scf::ForOp>(child_for_op->getParentOp()))
      return false;
    auto parent_op = dyn_cast<scf::ForOp>(child_for_op->getParentOp());
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
      if (dyn_cast<scf::ForOp>(parent_op))
        currnet_for_op = dyn_cast<scf::ForOp>(parent_op);
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
    if (auto async_op =
            mlir::dyn_cast<air::AsyncOpInterface>(op.getOperation())) {
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
        // Skip dependency if the source op is inside an scf.if that does
        // not contain the sink op. The token defined inside scf.if cannot
        // dominate ops outside it; the 4th traversal (loop-carried deps)
        // will handle this by threading the token through iter_args.
        if (srcOp) {
          if (auto ifOp = srcOp->getParentOfType<scf::IfOp>()) {
            if (!ifOp->isAncestor(op))
              continue;
          }
        }
        async_op.addAsyncDependency(depToken);
      }
    } else
      op->emitOpError("operation has no async interface");
  }

  template <typename T>
  void addAsyncDepToGraphIfNew(Value dep, T op) {
    if (auto async_op =
            mlir::dyn_cast<air::AsyncOpInterface>(op.getOperation())) {
      for (auto old_dep : async_op.getAsyncDependencies())
        if (old_dep == dep)
          return;

      // Add edge to graph, iff dep is async execute region (i.e. not a
      // loop iterator)
      if (auto srcOp = dep.getDefiningOp()) {
        uint64_t srcNode = 0;
        if (auto execute_op = dyn_cast<air::ExecuteOp>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(execute_op);
        } else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(dma_op);
        } else if (auto channel_op = dyn_cast<air::ChannelInterface>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(channel_op);
        } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(hier_op);
        } else if (auto wa_op = dyn_cast<air::WaitAllOp>(srcOp)) {
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

  // Check if two partial memref tiles have identical access patterns
  bool areEqualIndexPartialMemrefs(partialMemref *tile_0,
                                   partialMemref *tile_1) {
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
            map[*constStride].insert(tile->offsets[i]);
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
    if (dyn_cast<memref::DeallocOp>(op)) {
      return true;
    } else if (dyn_cast<memref::AllocOp>(op)) {
      return true;
    } else
      return false;
  }
  bool isNotLoopCarriedOp(air::AsyncOpInterface op) {
    if (auto exec_op = dyn_cast<air::ExecuteOp>(op.getOperation())) {
      auto &bb = exec_op.getRegion().front();
      Operation &child_op = bb.getOperations().front();
      return isNotLoopCarriedOp(&child_op);
    } else
      return false;
  }

  // Check if a value is not used inside a given block, or is only used by ops
  // not considered for loop-carried dependency
  bool isOnlyUsedByNoLoopCarryOpsInBlock(Value token, Block *block) {
    if (token.use_empty())
      return true;
    bool isOnlyUsedByNoCarryOps = true;
    for (auto user : token.getUsers()) {
      if (user->getBlock() == block) {
        if (auto async_user = dyn_cast<air::ExecuteOp>(user)) {
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
