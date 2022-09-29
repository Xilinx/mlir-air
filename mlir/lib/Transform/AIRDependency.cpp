//===- AIRDependency.cpp ----------------------------------------*- C++ -*-===//
//
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

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependency.h"
#include "air/Util/Dependency.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
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

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const &e) {
  llvm_unreachable("boost exception");
}

// boost graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/transitive_reduction.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;
using namespace boost;

#define DEBUG_TYPE "air-dependency"

namespace {

// Construction of a dependency graph as a Boost graph

struct executeNode {
  std::string asyncEventName;
  std::string asyncEventType;
  std::string color;
  std::string shape;
  std::string style;
  unsigned operationId;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                              executeNode>
    Graph;
typedef boost::graph_traits<Graph>::in_edge_iterator in_edge_iterator;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;

typedef std::map<Graph::vertex_descriptor, Graph::vertex_descriptor> vertex_map;
typedef std::map<unsigned, Graph::vertex_descriptor> operation_id_to_vertex_map;

static uint64_t ExecuteOpID;
static uint64_t HierarchyOpID;
static uint64_t WaitAllOpID;

class AIRDependency : public AIRDependencyBase<AIRDependency> {

public:
  AIRDependency() = default;
  AIRDependency(const AIRDependency &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Preprocessing: renumber the air dma op ids
    for (auto f : module.getOps<func::FuncOp>()) {
      xilinx::air::renumberDmaOps(f, "global");
    }

    // 1st traversal: create async ops with empty dep list.

    OpBuilder module_builder(module);

    ExecuteOpID = 0;
    HierarchyOpID = 0;

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        // Create async interface for air.dmamemcpy ops
        if (mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op))
          createAsyncDMA(module_builder, op);

        // Create async execute region for linalg.matmul
        else if (dyn_cast<linalg::MatmulOp>(op))
          createAsyncExecute(module_builder, op, "linalg::matmul", ExecuteOpID);

        // Create async execute region for linalg.fill
        else if (dyn_cast<linalg::FillOp>(op))
          createAsyncExecute(module_builder, op, "linalg::fill", ExecuteOpID);

        // Create async execute region for linalg.copy
        else if (dyn_cast<linalg::CopyOp>(op))
          createAsyncExecute(module_builder, op, "linalg::copy", ExecuteOpID);

        // Create async execute region for linalg op
        else if (mlir::dyn_cast<linalg::LinalgOp>(op))
          createAsyncExecute(module_builder, op, "linalg::unknown",
                             ExecuteOpID);

        // Create async execute region for memref.alloc
        else if (auto memalloc_op = dyn_cast<memref::AllocOp>(op))
          createAsyncExecute(module_builder, op, "memref::alloc", ExecuteOpID,
                             memalloc_op.memref().getType());

        // Create async execute region for memref.alloc
        else if (auto memcast_op = dyn_cast<memref::CastOp>(op))
          createAsyncExecute(module_builder, op, "memref::cast", ExecuteOpID,
                             memcast_op.dest().getType());

        // Create async execute region for memref.dealloc
        else if (dyn_cast<memref::DeallocOp>(op))
          createAsyncExecute(module_builder, op, "memref::dealloc",
                             ExecuteOpID);

        // Create async execute region for memref.copy
        else if (dyn_cast<memref::CopyOp>(op))
          createAsyncExecute(module_builder, op, "memref::copy", ExecuteOpID);

        // Create async execute region for arith.muli
        else if (auto arith_op = dyn_cast<arith::MulIOp>(op)) {
          if (arith_op.getResult().getType().isa<IndexType>()) {
            createAsyncExecute(module_builder, op, "arith::muli", ExecuteOpID,
                               arith_op.getResult().getType());
          }
        }

        // Create async execute region for arith.addi
        else if (auto arith_op = dyn_cast<arith::AddIOp>(op)) {
          if (arith_op.getResult().getType().isa<IndexType>()) {
            createAsyncExecute(module_builder, op, "arith::addi", ExecuteOpID,
                               arith_op.getResult().getType());
          }
        }

        // Create async execute region for affine.apply
        else if (auto apply_op = dyn_cast<mlir::AffineApplyOp>(op))
          createAsyncExecute(module_builder, op, "affine::apply", ExecuteOpID,
                             apply_op.getResult().getType());

        // Create async execute region for air hierarchy ops (air.launch and
        // air.partition, TODO: air.herd).
        else if (auto hierarchy_op = dyn_cast<air::HierarchyInterface>(op)) {
          createAsyncHierarchyImpls(module_builder, hierarchy_op,
                                    HierarchyOpID);
        }

        // Create async execute region for an unknown op which has memref or
        // index-type operands
        else {
          bool isCandidateExecute = false;
          for (auto operand : op->getOperands()) {
            if (operand.getType().isa<MemRefType>() ||
                operand.getType().isa<IndexType>()) {
              isCandidateExecute = true;
            }
          }
          // No air execute for loop ops
          if (mlir::dyn_cast<mlir::LoopLikeOpInterface>(op))
            isCandidateExecute = false;
          // No air execute for subview ops
          if (mlir::dyn_cast<mlir::OffsetSizeAndStrideOpInterface>(op))
            isCandidateExecute = false;
          // No air execute for terminators
          if (op->mightHaveTrait<OpTrait::IsTerminator>()) {
            isCandidateExecute = false;
          }
          // No air execute in linalg.generic
          if (op->getParentOfType<mlir::linalg::GenericOp>()) {
            isCandidateExecute = false;
          }
          if (isCandidateExecute) {
            if (op->getNumResults())
              createAsyncExecute(module_builder, op, "unknown", ExecuteOpID,
                                 op->getResults().front().getType());
            else
              createAsyncExecute(module_builder, op, "unknown", ExecuteOpID);
          }
        }
      });
    }

    // 2nd traversal: trace deps among async execute regions; build a boost dep
    // graph.

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        Operation *sink_op = nullptr;
        if (auto async_execute_op = dyn_cast<air::ExecuteOp>(op)) {
          for (auto &bb : async_execute_op.body()) {
            for (auto &child_op : bb.getOperations()) {
              if (!dyn_cast<air::ExecuteTerminatorOp>(child_op))
                sink_op = &child_op;
            }
          }
        } else if (mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)) {
          sink_op = op;
        } else if (dyn_cast<air::HierarchyInterface>(op)) {
          sink_op = op;
        } else
          return;

        SmallVector<partialMemref, 1> sink_op_memref_reads;
        SmallVector<partialMemref, 1> sink_op_memref_writes;
        SmallVector<Value, 1> sink_op_scalar_ins;
        SmallVector<Value, 1> sink_op_scalar_outs;

        // If the sink op is linalg op
        if (auto sink_op_linalgop = dyn_cast<linalg::LinalgOp>(sink_op)) {
          for (auto linalg_ins : sink_op_linalgop.inputs()) {
            if (linalg_ins.getType().isa<MemRefType>()) {
              unsigned memRefRank =
                  linalg_ins.getType().cast<MemRefType>().getRank();
              partialMemref tile = createPartialMemref(linalg_ins, memRefRank);
              sink_op_memref_reads.push_back(tile);
            } else if (linalg_ins.getType().isa<IndexType>()) {
              sink_op_scalar_ins.push_back(linalg_ins);
            }
          }
          for (auto linalg_outs : sink_op_linalgop.outputs()) {
            if (linalg_outs.getType().isa<MemRefType>()) {
              unsigned memRefRank =
                  linalg_outs.getType().cast<MemRefType>().getRank();
              partialMemref tile = createPartialMemref(linalg_outs, memRefRank);
              sink_op_memref_reads.push_back(
                  tile); // linalg op both reads and writes the output memref
              sink_op_memref_writes.push_back(tile);
            } else if (linalg_outs.getType().isa<IndexType>()) {
              sink_op_scalar_ins.push_back(
                  linalg_outs); // linalg op both reads and writes the output
                                // memref
              sink_op_scalar_outs.push_back(linalg_outs);
            }
          }
          if (sink_op_linalgop->getNumResults()) {
            for (auto linalg_results : sink_op_linalgop->getResults()) {
              if (linalg_results.getType().isa<MemRefType>()) {
                unsigned memRefRank =
                    linalg_results.getType().cast<MemRefType>().getRank();
                partialMemref tile =
                    createPartialMemref(linalg_results, memRefRank);
                sink_op_memref_writes.push_back(tile);
              } else if (linalg_results.getType().isa<IndexType>()) {
                sink_op_scalar_outs.push_back(linalg_results);
              }
            }
          }
        }

        // If the sink op is memref::dealloc
        else if (auto sink_op_memdealloc =
                     dyn_cast<memref::DeallocOp>(sink_op)) {
          unsigned memRefRank = sink_op_memdealloc.memref()
                                    .getType()
                                    .cast<MemRefType>()
                                    .getRank();
          partialMemref tile =
              createPartialMemref(sink_op_memdealloc.memref(), memRefRank);
          sink_op_memref_reads.push_back(tile);
          sink_op_memref_writes.push_back(
              tile); // dealloc erases (i.e. writes to) output memref
        }

        // If the sink op is memref::copy
        else if (auto sink_op_memref_copy = dyn_cast<memref::CopyOp>(sink_op)) {
          unsigned memRefRankSrc = sink_op_memref_copy.source()
                                       .getType()
                                       .cast<MemRefType>()
                                       .getRank();
          partialMemref tileSrc =
              createPartialMemref(sink_op_memref_copy.source(), memRefRankSrc);
          sink_op_memref_reads.push_back(tileSrc);
          unsigned memRefRankDst = sink_op_memref_copy.target()
                                       .getType()
                                       .cast<MemRefType>()
                                       .getRank();
          partialMemref tileDst =
              createPartialMemref(sink_op_memref_copy.target(), memRefRankDst);
          sink_op_memref_reads.push_back(tileDst);
          sink_op_memref_writes.push_back(tileDst);
        }

        // If the sink op is an air::DmaMemcpy op
        else if (auto sink_op_dma =
                     mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(sink_op)) {
          SmallVector<Value, 2> src_indices;
          SmallVector<Value, 2> dst_indices;
          unsigned numDimsSrc = sink_op_dma.getNumDims();
          unsigned numDimsDst = sink_op_dma.getNumDims();
          // air.dmamemcpynd op has unknown # of dims (thus numdims defaults to
          // 0)
          if (numDimsSrc == 0) {
            numDimsSrc = sink_op_dma.getSrcMemref()
                             .getType()
                             .cast<MemRefType>()
                             .getRank();
            numDimsDst = sink_op_dma.getDstMemref()
                             .getType()
                             .cast<MemRefType>()
                             .getRank();
          }
          // Special case with ND DMA op
          if (auto sink_op_nddma = dyn_cast<air::DmaMemcpyNdOp>(sink_op)) {
            // air.dmamemcpynd op has extra scalar operands
            for (unsigned i = 0; i < sink_op_nddma.getDstOffsets().size(); i++)
              sink_op_scalar_outs.push_back(sink_op_nddma.getDstOffsets()[i]);
            for (unsigned i = 0; i < sink_op_nddma.getDstSizes().size(); i++)
              sink_op_scalar_outs.push_back(sink_op_nddma.getDstSizes()[i]);
            for (unsigned i = 0; i < sink_op_nddma.getDstStrides().size(); i++)
              sink_op_scalar_outs.push_back(sink_op_nddma.getDstStrides()[i]);
            for (unsigned i = 0; i < sink_op_nddma.getSrcOffsets().size(); i++)
              sink_op_scalar_ins.push_back(sink_op_nddma.getSrcOffsets()[i]);
            for (unsigned i = 0; i < sink_op_nddma.getSrcSizes().size(); i++)
              sink_op_scalar_ins.push_back(sink_op_nddma.getSrcSizes()[i]);
            for (unsigned i = 0; i < sink_op_nddma.getSrcStrides().size(); i++)
              sink_op_scalar_ins.push_back(sink_op_nddma.getSrcStrides()[i]);
            if (sink_op_nddma.getSrcOffsets().size()) {
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(sink_op_nddma.getSrcOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(nullptr);
              }
            }
            if (sink_op_nddma.getDstOffsets().size()) {
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(sink_op_nddma.getDstOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(nullptr);
              }
            }
          } else {
            for (unsigned i = 0; i < numDimsSrc; i++) {
              sink_op_scalar_ins.push_back(sink_op_dma.getSrcMemrefDim(i));
              src_indices.push_back(sink_op_dma.getSrcMemrefDim(i));
            }
            for (unsigned i = 0; i < numDimsDst; i++) {
              sink_op_scalar_outs.push_back(sink_op_dma.getDstMemrefDim(i));
              dst_indices.push_back(sink_op_dma.getDstMemrefDim(i));
            }
          }
          partialMemref tile_in = createPartialMemref(
              sink_op_dma.getSrcMemref(), numDimsSrc, src_indices);
          sink_op_memref_reads.push_back(tile_in);
          partialMemref tile_out = createPartialMemref(
              sink_op_dma.getDstMemref(), numDimsDst, dst_indices);
          sink_op_memref_writes.push_back(tile_out);
        }

        // If the sink op is arith::MulIOp
        else if (auto sink_op_arith = dyn_cast<arith::MulIOp>(sink_op)) {
          sink_op_scalar_ins.push_back(sink_op_arith.getLhs());
          sink_op_scalar_ins.push_back(sink_op_arith.getRhs());
          sink_op_scalar_outs.push_back(sink_op_arith.getResult());
        }

        // If the sink op is arith::AddIOp
        else if (auto sink_op_arith = dyn_cast<arith::AddIOp>(sink_op)) {
          sink_op_scalar_ins.push_back(sink_op_arith.getLhs());
          sink_op_scalar_ins.push_back(sink_op_arith.getRhs());
          sink_op_scalar_outs.push_back(sink_op_arith.getResult());
        }

        // If the sink op is mlir::AffineApplyOp
        else if (auto sink_op_apply = dyn_cast<mlir::AffineApplyOp>(sink_op)) {
          for (auto applyop_operand : sink_op_apply.getMapOperands()) {
            sink_op_scalar_ins.push_back(applyop_operand);
          }
          sink_op_scalar_outs.push_back(sink_op_apply.getResult());
        }

        // If the sink op is an unknown op
        else {
          for (auto sink_op_op : sink_op->getOperands()) {
            if (sink_op_op.getType().isa<MemRefType>()) {
              unsigned memRefRank =
                  sink_op_op.getType().cast<MemRefType>().getRank();
              partialMemref tile = createPartialMemref(sink_op_op, memRefRank);
              sink_op_memref_reads.push_back(
                  tile); // Assuming all operands are both read and written to
              sink_op_memref_writes.push_back(tile);
            } else if (sink_op_op.getType().isa<IndexType>()) {
              sink_op_scalar_ins.push_back(
                  sink_op_op); // Assuming all operands are both read and
                               // written to
              sink_op_scalar_outs.push_back(sink_op_op);
            }
          }
          if (sink_op->getNumResults()) {
            for (auto sink_op_results : sink_op->getResults()) {
              if (sink_op_results.getType().isa<MemRefType>()) {
                unsigned memRefRank =
                    sink_op_results.getType().cast<MemRefType>().getRank();
                partialMemref tile =
                    createPartialMemref(sink_op_results, memRefRank);
                sink_op_memref_writes.push_back(tile);
              } else if (sink_op_results.getType().isa<IndexType>()) {
                sink_op_scalar_outs.push_back(sink_op_results);
              }
            }
          }
        }

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
          async_execute_op_history.push_back(async_execute_op);
        } else if (auto dma_op =
                       mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)) {
          traceDeps<air::DmaMemcpyInterface>(sink_op_memref_reads, dma_op,
                                             "RAW");
          traceDeps<air::DmaMemcpyInterface>(sink_op_memref_writes, dma_op,
                                             "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs, dma_op);
          dma_op_history.push_back(dma_op);
        } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
          hier_op_history.push_back(hier_op);
        }
      });
    }

    // 3rd traversal: perform transitive reduction on dependency graph.

    std::vector<size_t> id_map(num_vertices(asyncExecuteGraph));
    std::iota(id_map.begin(), id_map.end(), 0u);

    transitive_reduction(asyncExecuteGraph, asyncExecuteGraphTR,
                         make_assoc_property_map(g_to_tr), id_map.data());

    for (vertex_map::iterator i = g_to_tr.begin(); i != g_to_tr.end(); ++i) {
      // Copy over graph properties
      asyncExecuteGraphTR[i->second].asyncEventName =
          asyncExecuteGraph[i->first].asyncEventName;
      asyncExecuteGraphTR[i->second].asyncEventType =
          asyncExecuteGraph[i->first].asyncEventType;
      asyncExecuteGraphTR[i->second].color = asyncExecuteGraph[i->first].color;
      asyncExecuteGraphTR[i->second].shape = asyncExecuteGraph[i->first].shape;
      asyncExecuteGraphTR[i->second].operationId =
          asyncExecuteGraph[i->first].operationId;
      // Build reverse map tr_to_g, for convenient vertex mapping
      tr_to_g[i->second] = i->first;
    }

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        // Fill dep list of air execute ops
        if (auto async_execute_op = dyn_cast<air::ExecuteOp>(op)) {
          fillAIRDepListUsingGraphTR<air::ExecuteOp>(async_execute_op);
        }
        // Fill dep list of air dmamemcpy2d ops
        else if (auto dma_op = dyn_cast<air::DmaMemcpyInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::DmaMemcpyInterface>(dma_op);
        } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::HierarchyInterface>(hier_op);
        }
      });
    }

    // 4th traversal: loop-carried deps.
    // Add wait_all events to collect sinks in loop bodies. Add iter_args to scp
    // for loops representing loop-carried deps.

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (scf::ForOp for_op = dyn_cast<scf::ForOp>(op)) {

          // Get async execute region in loop body
          bool hasAsyncTokensInBody = false;
          SmallVector<Value, 1> sinks_in_for_op;
          for (auto async_execute_op : for_op.getOps<air::ExecuteOp>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (isNotUsedInsideOfBlock(async_execute_op.getResult(0),
                                       for_op.getBody()))
              sinks_in_for_op.push_back(async_execute_op.getResult(0));
          }
          // Get async dma in loop body
          for (auto dma_op : for_op.getOps<air::DmaMemcpyInterface>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (isNotUsedInsideOfBlock(dma_op.getOperation()->getResult(0),
                                       for_op.getBody()))
              sinks_in_for_op.push_back(dma_op.getOperation()->getResult(0));
          }
          // Get async hierarchy op in loop body
          for (auto hier_op : for_op.getOps<air::HierarchyInterface>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (isNotUsedInsideOfBlock(hier_op.getOperation()->getResult(0),
                                       for_op.getBody()))
              sinks_in_for_op.push_back(hier_op.getOperation()->getResult(0));
          }
          // Get async for_op in loop body
          for (auto child_for_op : for_op.getOps<scf::ForOp>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (auto v = child_for_op.getResult(0)) {
              if (isNotUsedInsideOfBlock(v, for_op.getBody()))
                sinks_in_for_op.push_back(v);
            }
          }
          // Get async parallel_op in loop body
          for (auto child_parallel_op : for_op.getOps<scf::ParallelOp>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (auto v = child_parallel_op.getResult(0)) {
              if (isNotUsedInsideOfBlock(v, for_op.getBody()))
                sinks_in_for_op.push_back(v);
            }
          }

          if (hasAsyncTokensInBody) {
            insertLoopCarriedDeps(module_builder, for_op, sinks_in_for_op);
          }
        }

        else if (scf::ParallelOp for_op = dyn_cast<scf::ParallelOp>(op)) {

          // Get async execute region in loop body
          bool hasAsyncTokensInBody = false;
          SmallVector<Value, 1> sinks_in_parallel_op;
          for (auto async_execute_op : for_op.getOps<air::ExecuteOp>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (isNotUsedInsideOfBlock(async_execute_op.getResult(0),
                                       for_op.getBody()))
              sinks_in_parallel_op.push_back(async_execute_op.getResult(0));
          }
          // Get async dma in loop body
          for (auto dma_op : for_op.getOps<air::DmaMemcpyInterface>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (isNotUsedInsideOfBlock(dma_op.getOperation()->getResult(0),
                                       for_op.getBody()))
              sinks_in_parallel_op.push_back(
                  dma_op.getOperation()->getResult(0));
          }
          // Get async hierarchy op in loop body
          for (auto hier_op : for_op.getOps<air::HierarchyInterface>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (isNotUsedInsideOfBlock(hier_op.getOperation()->getResult(0),
                                       for_op.getBody()))
              sinks_in_parallel_op.push_back(
                  hier_op.getOperation()->getResult(0));
          }
          // Get async for_op in loop body
          for (auto child_for_op : for_op.getOps<scf::ForOp>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (auto v = child_for_op.getResult(0)) {
              if (isNotUsedInsideOfBlock(v, for_op.getBody()))
                sinks_in_parallel_op.push_back(v);
            }
          }
          // Get async parallel_op in loop body
          for (auto child_parallel_op : for_op.getOps<scf::ParallelOp>()) {
            hasAsyncTokensInBody = true;
            // Detect dep graph's leaves in loop body
            if (auto v = child_parallel_op.getResult(0)) {
              if (isNotUsedInsideOfBlock(v, for_op.getBody()))
                sinks_in_parallel_op.push_back(v);
            }
          }

          if (hasAsyncTokensInBody) {
            insertLoopCarriedDeps(module_builder, for_op, sinks_in_parallel_op);
          }
        }
      });
    }

    // Final traversal: Clean up
    // Remove repetition in dependency list.
    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto async_op = dyn_cast<air::AsyncOpInterface>(op)) {
          if (async_op.getAsyncDependencies().size() >= 1) {
            auto dependency_list = async_op.getAsyncDependencies();
            // Initialize repetition mask
            std::vector<bool> hasRepeat;
            for (auto i = dependency_list.begin(); i != dependency_list.end();
                 ++i) {
              hasRepeat.push_back(false);
            }
            // Iterate the dependency list
            for (unsigned i = 0; i < dependency_list.size(); i++) {
              for (unsigned j = i + 1; j < dependency_list.size(); j++) {
                if (dependency_list[i] == dependency_list[j]) {
                  hasRepeat[j] = true;
                }
              }
            }
            for (int i = dependency_list.size() - 1; i >= 0; i--) {
              if (hasRepeat[i]) {
                async_op.eraseAsyncDependency(i);
              }
            }
          }
        }
      });
    }

    // Remove wait_all with single operand.
    // Remove wait_all's id attribute.
    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (air::WaitAllOp wa_op = dyn_cast<air::WaitAllOp>(op)) {
          if (wa_op.getAsyncDependencies().size() == 1) {
            wa_op.getAsyncToken().replaceAllUsesWith(
                wa_op.getAsyncDependencies()[0]);
            wa_op.erase();
          } else {
            wa_op->removeAttr("id");
          }
        }
      });
    }

    // Dump graph
    dump_graph("out.dot");
  }

private:
  //===----------------------------------------------------------------------===//
  // Creating async events
  //===----------------------------------------------------------------------===//

  // Air async op history
  std::vector<air::ExecuteOp> async_execute_op_history;
  std::vector<air::DmaMemcpyInterface> dma_op_history;
  std::vector<air::HierarchyInterface> hier_op_history;

  // Create air execute op with async interface (no ssa result returned); update
  // graph
  air::ExecuteOp createAsyncExecute(OpBuilder &builder, Operation *op,
                                    std::string asyncEventName,
                                    uint64_t &ExecuteOpID) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::ExecuteOp async_region;
    async_region = builder.create<xilinx::air::ExecuteOp>(
        loc, air::AsyncTokenType::get(op->getContext()), deps);
    async_region->setAttr(
        "id", mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32), ++ExecuteOpID));

    // Insert op to the new async execute region's body.
    Block *async_region_bb = builder.createBlock(&async_region.body());
    builder.setInsertionPointToStart(async_region_bb);

    builder.clone(*op);
    builder.create<xilinx::air::ExecuteTerminatorOp>(builder.getUnknownLoc());

    // Create a vertex out of the current async execute region
    auto v = add_vertex(asyncExecuteGraph);
    asyncExecuteGraph[v].asyncEventName = asyncEventName;
    asyncExecuteGraph[v].asyncEventType = "execute";
    asyncExecuteGraph[v].color = "chartreuse";
    asyncExecuteGraph[v].shape = "oval";
    asyncExecuteGraph[v].operationId = ExecuteOpID;

    // Update op-to-graph map
    region_to_g[async_region.getId()] = v;

    // Erase op
    op->erase();
    return async_region;
  }

  // Create air execute op with async interface (with one ssa result returned);
  // update graph
  air::ExecuteOp createAsyncExecute(OpBuilder &builder, Operation *op,
                                    std::string asyncEventName,
                                    uint64_t &ExecuteOpID,
                                    mlir::Type valueType) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::ExecuteOp async_region;
    async_region = builder.create<xilinx::air::ExecuteOp>(
        loc, air::AsyncTokenType::get(op->getContext()), valueType, deps);
    async_region->setAttr(
        "id", mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32), ++ExecuteOpID));

    // Insert op to the new async execute region's body.
    Block *async_region_bb = builder.createBlock(&async_region.body());
    builder.setInsertionPointToStart(async_region_bb);
    auto op_cloned = builder.clone(*op);
    builder.create<xilinx::air::ExecuteTerminatorOp>(
        builder.getUnknownLoc(), op_cloned->getResults().front());
    SmallVector<Value, 1> returnVals;
    returnVals.push_back(async_region.getResult(1));
    op->replaceAllUsesWith(returnVals);

    // Create a vertex out of the current async execute region
    auto v = add_vertex(asyncExecuteGraph);
    asyncExecuteGraph[v].asyncEventName = asyncEventName;
    asyncExecuteGraph[v].asyncEventType = "execute";
    asyncExecuteGraph[v].color = "chartreuse";
    asyncExecuteGraph[v].shape = "oval";
    asyncExecuteGraph[v].operationId = ExecuteOpID;

    // Update op-to-graph map
    region_to_g[async_region.getId()] = v;

    // Erase op
    op->erase();
    return async_region;
  }

  // Re-instantiate the dmamemcpy2d op with async interface; update graph
  void createAsyncDMA(OpBuilder &builder, Operation *op) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op);
    unsigned id = dma_op.getId();
    std::string event_name = "";
    if (auto dmaNd_op = dyn_cast<air::DmaMemcpyNdOp>(op)) {
      air::DmaMemcpyNdOp new_dmaNd_op = builder.create<air::DmaMemcpyNdOp>(
          loc, air::AsyncTokenType::get(dmaNd_op->getContext()), deps,
          dmaNd_op.getDstMemref(), dmaNd_op.getDstOffsets(),
          dmaNd_op.getDstSizes(), dmaNd_op.getDstStrides(),
          dmaNd_op.getSrcMemref(), dmaNd_op.getSrcOffsets(),
          dmaNd_op.getSrcSizes(), dmaNd_op.getSrcStrides());
      new_dmaNd_op->setAttr(
          "id", mlir::IntegerAttr::get(
                    mlir::IntegerType::get(op->getContext(), 32), id));
      event_name = "Nd";
    }

    // Create a vertex out of the current dmamemcpy2d op
    auto v = add_vertex(asyncExecuteGraph);
    asyncExecuteGraph[v].asyncEventName = "air::dma" + event_name;
    asyncExecuteGraph[v].asyncEventType = "dma";
    asyncExecuteGraph[v].color = "cyan";
    asyncExecuteGraph[v].shape = "oval";
    asyncExecuteGraph[v].operationId = id;

    // Update op-to-graph map
    dma_to_g[id] = v;

    // Erase op
    op->erase();
  }

  // Re-instantiate the hierarchy op with async interface; update graph
  air::HierarchyInterface createAsyncHierarchyImpls(OpBuilder &builder,
                                                    air::HierarchyInterface op,
                                                    uint64_t &HierarchyOpID) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    SmallVector<Value, 4> args;
    SmallVector<Value, 4> constants;
    for (unsigned i = 0; i < op.getNumKernelOperands(); i++) {
      auto v = op.getKernelOperand(i);
      if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp())) {
        constants.push_back(v);
        args.push_back(v);
      } else
        args.push_back(v);
    }
    Operation *new_op = nullptr;
    if (auto launch = dyn_cast<air::LaunchOp>(op.getOperation())) {
      auto new_launch = createAsyncHierarchy<air::LaunchOp>(
          builder, launch, HierarchyOpID, deps, args, constants);
      new_op = new_launch.getOperation();
      auto &bb = new_launch.body().front();
      builder.setInsertionPointToEnd(&bb);
      builder.create<air::LaunchTerminatorOp>(loc);
      // Create a vertex out of the current hierarchy op
      auto v = add_vertex(asyncExecuteGraph);
      asyncExecuteGraph[v].asyncEventName = "air::launch";
      asyncExecuteGraph[v].asyncEventType = "hierarchy";
      asyncExecuteGraph[v].color = "yellow";
      asyncExecuteGraph[v].shape = "box";
      asyncExecuteGraph[v].operationId = HierarchyOpID;
      // Update op-to-graph map
      hier_to_g[HierarchyOpID] = v;
    } else if (auto partition = dyn_cast<air::PartitionOp>(op.getOperation())) {
      auto new_partition = createAsyncHierarchy<air::PartitionOp>(
          builder, partition, HierarchyOpID, deps, args, constants);
      new_op = new_partition.getOperation();
      auto &bb = new_partition.body().front();
      builder.setInsertionPointToEnd(&bb);
      builder.create<air::PartitionTerminatorOp>(loc);
      // Create a vertex out of the current hierarchy op
      auto v = add_vertex(asyncExecuteGraph);
      asyncExecuteGraph[v].asyncEventName = "air::partition";
      asyncExecuteGraph[v].asyncEventType = "hierarchy";
      asyncExecuteGraph[v].color = "yellow";
      asyncExecuteGraph[v].shape = "box";
      asyncExecuteGraph[v].operationId = HierarchyOpID;
      // Update op-to-graph map
      hier_to_g[HierarchyOpID] = v;
    } else if (auto herd = dyn_cast<air::HerdOp>(op.getOperation())) {
      auto new_herd = createAsyncHierarchy<air::HerdOp>(
          builder, herd, HierarchyOpID, deps, args, constants);
      new_op = new_herd.getOperation();
      auto &bb = new_herd.body().front();
      builder.setInsertionPointToEnd(&bb);
      builder.create<air::HerdTerminatorOp>(loc);
      // Create a vertex out of the current hierarchy op
      auto v = add_vertex(asyncExecuteGraph);
      asyncExecuteGraph[v].asyncEventName = "air::herd";
      asyncExecuteGraph[v].asyncEventType = "hierarchy";
      asyncExecuteGraph[v].color = "yellow";
      asyncExecuteGraph[v].shape = "box";
      asyncExecuteGraph[v].operationId = HierarchyOpID;
      // Update op-to-graph map
      hier_to_g[HierarchyOpID] = v;
    } else {
      assert(false && "Unknown hierarchy operation");
    }
    auto new_hier = dyn_cast<air::HierarchyInterface>(new_op);

    // Erase op
    op->erase();
    return new_hier;
  }

  template <typename T>
  T createAsyncHierarchy(OpBuilder &builder, T op, uint64_t &OpID,
                         SmallVector<Value, 1> deps, SmallVector<Value, 4> args,
                         SmallVector<Value, 4> constants) {
    auto loc = op->getLoc();
    T new_op = builder.create<T>(loc, deps, op.getSizeOperands(), args, true);
    new_op->setAttr("id",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 32), ++OpID));

    if (auto attr = op->template getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

    auto &bb = new_op.body().front();
    for (unsigned i = 0; i < op.getIds().size(); i++) {
      auto ivs = op.getIds()[i];
      ivs.replaceAllUsesWith(new_op.getIds()[i]);
    }
    for (unsigned i = 0; i < op.getSize().size(); i++) {
      auto s = op.getSize()[i];
      s.replaceAllUsesWith(new_op.getSize()[i]);
    }
    auto &body = op.body().front().getOperations();
    bb.getOperations().splice(bb.begin(), body, body.begin(), --body.end());
    builder.setInsertionPointToStart(&new_op.getRegion().front());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, builder.clone(*c.getDefiningOp())->getResult(0),
          new_op.getRegion());
    }

    int i = 0;
    auto old_kernel_args = op.getKernelArguments();
    auto new_kernel_args = new_op.getKernelArguments();
    for (Value v : old_kernel_args)
      replaceAllUsesInRegionWith(v, new_kernel_args[i++], new_op.getRegion());

    return new_op;
  }

  //===----------------------------------------------------------------------===//
  // Data dependency tracing
  //===----------------------------------------------------------------------===//

  struct partialMemref {
    Value memrefValue;
    unsigned numDims;
    SmallVector<Value, 2> memrefIndices;
  };

  partialMemref createPartialMemref(mlir::Value memrefValue, unsigned numDims) {
    partialMemref tile;
    tile.memrefValue = memrefValue;
    tile.numDims = numDims;
    for (unsigned i = 0; i < numDims; i++) {
      tile.memrefIndices.push_back(nullptr);
    }
    return tile;
  }

  partialMemref createPartialMemref(mlir::Value memrefValue, unsigned numDims,
                                    SmallVector<Value, 2> memrefIndices) {
    partialMemref tile;
    tile.memrefValue = memrefValue;
    tile.numDims = numDims;
    for (unsigned i = 0; i < numDims; i++) {
      tile.memrefIndices.push_back(memrefIndices[i]);
    }
    return tile;
  }

  // Check if operand is returned from ExecuteOp (memref.alloc)
  template <typename T> void pushDefiningOpAsDep(Value operand, T op) {
    // Check memref deps
    if (auto defop = operand.getDefiningOp<air::ExecuteOp>()) {
      if (foundAsyncOpUsesAboveCurrentLine(&defop)) {
        addNewAsyncDepToGraph<T>(defop.getResult(0), op);
      }
    }
  }

  // Trace tile index deps
  template <typename T> void pushTileIndexAsDep(mlir::Value tile_index, T op) {
    if (tile_index != nullptr) {
      // If tile_index is not a nullptr
      // If created by async_region
      if (auto defop = tile_index.getDefiningOp<air::ExecuteOp>()) {
        if (foundAsyncOpUsesAboveCurrentLine(&defop)) {
          addNewAsyncDepToGraph<T>(defop.getResult(0), op);
        }
      }
      // If created by hierarchy (as loop iter)
      else if (auto hier = dyn_cast<air::HierarchyInterface>(
                   tile_index.getParentRegion()->getParentOp())) {
        for (auto id : hier.getIds()) {
          if (id == tile_index) {
            addNewAsyncDepToGraph<T>(tile_index, op);
          }
        }
      }
    }
  }

  char checkOperandReadOrWrite(mlir::Value operand) {
    assert(operand.getType().isa<MemRefType>() &&
           "operand being traced is not a memref");
    bool foundWriteAccess = false;
    bool foundReadAccess = false;
    for (auto &u : operand.getUses()) {
      // If used in DmaMemcpy Op
      if (auto dma = dyn_cast<xilinx::air::DmaMemcpyInterface>(u.getOwner())) {
        if (u.is(dma.getSrcMemref())) {
          foundReadAccess = true;
        } else if (u.is(dma.getDstMemref())) {
          foundWriteAccess = true;
        } else {
          assert(false && "Unknown operand in air.dma");
        }
      }
      // If used in a linalg op
      else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(u.getOwner())) {
        if (u.getOperandNumber() <
            linalgop.getNumInputs() + linalgop.getNumOutputs()) {
          foundReadAccess = true;
        } else if (u.getOperandNumber() >= linalgop.getNumInputs() &&
                   u.getOperandNumber() - linalgop.getNumInputs() <
                       linalgop.getNumOutputs()) {
          foundWriteAccess = true;
        } else {
          assert(false && "Unknown operand in linalg op");
        }
      }
      // If unknown op, then assume write access for safety
      else
        foundWriteAccess = true;
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
    assert(operand.getType().isa<MemRefType>() &&
           "operand being traced is not a memref");
    for (auto &u : operand.getUses()) {
      // If used in DmaMemcpy Op
      if (auto dma = dyn_cast<xilinx::air::DmaMemcpyInterface>(u.getOwner())) {
        if (foundAsyncOpUsesAboveCurrentLine(
                &dma)) { // If this use is above current line
          // DMA2D: Need to check for overlapping partial memrefs in use
          unsigned numDimsSrc = dma.getNumDims();
          unsigned numDimsDst = dma.getNumDims();
          if (numDimsSrc == 0)
            numDimsSrc =
                dma.getSrcMemref().getType().cast<MemRefType>().getRank();
          if (numDimsDst == 0)
            numDimsDst =
                dma.getDstMemref().getType().cast<MemRefType>().getRank();
          SmallVector<Value, 2> src_indices;
          SmallVector<Value, 2> dst_indices;
          if (auto nddma =
                  dyn_cast<xilinx::air::DmaMemcpyNdOp>(dma.getOperation())) {
            if (nddma.getSrcOffsets().size()) {
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(nddma.getSrcOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(nullptr);
              }
            }
            if (nddma.getDstOffsets().size()) {
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(nddma.getDstOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(nullptr);
              }
            }
          } else {
            for (unsigned i = 0; i < numDimsSrc; i++) {
              src_indices.push_back(dma.getSrcMemrefDim(i));
            }
            for (unsigned i = 0; i < numDimsDst; i++) {
              dst_indices.push_back(dma.getDstMemrefDim(i));
            }
          }
          partialMemref dma_src =
              createPartialMemref(dma.getSrcMemref(), numDimsSrc, src_indices);
          partialMemref dma_dst =
              createPartialMemref(dma.getDstMemref(), numDimsDst, dst_indices);

          if (rw == 'r') {
            if (u.is(dma.getSrcMemref())) {
              if (tile == nullptr) {
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
              } else if (areEqualIndexPartialMemrefs(tile, &dma_src))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
          } else if (rw == 'w') {
            if (u.is(dma.getDstMemref())) {
              if (tile == nullptr) {
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
              } else if (areEqualIndexPartialMemrefs(tile, &dma_dst))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
          } else {
            if (tile == nullptr) {
              addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            } else if (u.is(dma.getDstMemref())) {
              if (areEqualIndexPartialMemrefs(tile, &dma_dst))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            } else if (u.is(dma.getSrcMemref())) {
              if (areEqualIndexPartialMemrefs(tile, &dma_src))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
          }
        }
      }

      // If used in a linalg op
      else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(u.getOwner())) {
        if (auto ar =
                dyn_cast<xilinx::air::ExecuteOp>(linalgop->getParentOp())) {
          if (foundAsyncOpUsesAboveCurrentLine(&ar)) {
            if (rw == 'r') {
              if (u.getOperandNumber() <
                  linalgop.getNumInputs() + linalgop.getNumOutputs())
                addNewAsyncDepToGraph<T>(ar.getResult(0), op);
            } else if (rw == 'w') {
              if (u.getOperandNumber() >= linalgop.getNumInputs() &&
                  u.getOperandNumber() - linalgop.getNumInputs() <
                      linalgop.getNumOutputs())
                addNewAsyncDepToGraph<T>(ar.getResult(0), op);
            } else {
              addNewAsyncDepToGraph<T>(ar.getResult(0), op);
            }
          }
        }
      }

      // If used in hierarchy op
      else if (auto hier =
                   dyn_cast<xilinx::air::HierarchyInterface>(u.getOwner())) {
        if (foundAsyncOpUsesAboveCurrentLine(&hier)) {
          // check if the use inside hierarchy op matches with the tracing mode
          // (r or w)
          for (unsigned hier_argument_id = 0;
               hier_argument_id < hier.getNumKernelOperands();
               hier_argument_id++) {
            if (u.is(hier.getKernelOperand(hier_argument_id))) {
              auto child_op = hier.getKernelArgument(hier_argument_id);
              char rw_check = checkOperandReadOrWrite(child_op);
              if (rw == 'n' || rw_check == rw) {
                addNewAsyncDepToGraph<T>(hier->getResult(0), op);
              }
            }
          }
        }
      }

      // If used in an unknown op
      else {
        auto unknownop = u.getOwner();
        if (auto ar =
                dyn_cast<xilinx::air::ExecuteOp>(unknownop->getParentOp())) {
          if (foundAsyncOpUsesAboveCurrentLine(&ar)) {
            addNewAsyncDepToGraph<T>(ar.getResult(0), op);
          }
        }
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
      assert(false && "Unknown dependency type");

    // Detect deps
    for (auto operand : operands) {
      // Trace the defining op of sink op, RAW
      pushDefiningOpAsDep<T>(operand.memrefValue, sink_air_op);

      // If sink op and operand's use are under the same scope
      pushDepsAtCurrentScope<T>(operand.memrefValue, sink_air_op,
                                dep_tracing_mode, &operand);

      // If sink op is in hierarchy op
      if (auto hier = sink_air_op->template getParentOfType<
                      xilinx::air::HierarchyInterface>()) {
        // Search for deps outside (before) hierarchy op
        for (unsigned hier_operand_id = 0;
             hier_operand_id < hier.getNumKernelOperands(); hier_operand_id++) {
          if (hier.getKernelArguments()[hier_operand_id] ==
              operand.memrefValue) {
            auto ancestor_op = hier.getKernelOperand(hier_operand_id);
            partialMemref ancestor_operand =
                createPartialMemref(ancestor_op, operand.numDims);
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
        partialMemref subview_tile = createPartialMemref(
            subview.source(), subview.sizes().size(), subview.offsets());
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
      for (unsigned i = 0; i < operand.numDims; i++) {
        pushTileIndexAsDep<T>(operand.memrefIndices[i], sink_air_op);
      }
    }
    for (auto operand : write_operands) {
      for (unsigned i = 0; i < operand.numDims; i++) {
        pushTileIndexAsDep<T>(operand.memrefIndices[i], sink_air_op);
      }
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
                                 Graph::vertex_descriptor v) {
    unsigned v_a = 0;
    unsigned v_b = 0;
    if (auto op = dyn_cast<air::ExecuteOp>(a)) {
      v_a = g_to_tr[getGraphGVertexFromAIROp(op)];
    } else if (auto op = mlir::dyn_cast<air::DmaMemcpyInterface>(a)) {
      v_a = g_to_tr[getGraphGVertexFromAIROp(op)];
    } else if (auto op = dyn_cast<air::HierarchyInterface>(a)) {
      v_a = g_to_tr[getGraphGVertexFromAIROp(op)];
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
      v_b = g_to_tr[getGraphGVertexFromAIROp(op)];
    } else if (auto op = mlir::dyn_cast<air::DmaMemcpyInterface>(b)) {
      v_b = g_to_tr[getGraphGVertexFromAIROp(op)];
    } else if (auto op = dyn_cast<air::HierarchyInterface>(b)) {
      v_b = g_to_tr[getGraphGVertexFromAIROp(op)];
    } else if (auto op = dyn_cast<scf::ForOp>(b)) {
      v_b = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast<scf::ParallelOp>(b)) {
      v_b = getGraphGVertexFromAIROp(
          op); // g_to_tr not needed since wait_all is created after TR
    } else if (auto op = dyn_cast<air::WaitAllOp>(b)) {
      v_b = getGraphGVertexFromAIROp(op);
    }
    if (edge(v_a, v_b, asyncExecuteGraphTR).second) { // if an edge exists
      remove_edge(v_a, v_b, asyncExecuteGraphTR);
      if (!edge(v_a, v, asyncExecuteGraphTR).second)
        add_edge(v_a, v, asyncExecuteGraphTR);
      if (!edge(v, v_b, asyncExecuteGraphTR).second)
        add_edge(v, v_b, asyncExecuteGraphTR);
    }
  }

  template <typename T>
  air::WaitAllOp
  insertWaitAllOpBeforeLoopYield(OpBuilder &builder, T loop_op,
                                 SmallVector<Value, 1> sinks_in_loop_op) {
    // Create one wait_all event at the end of current loop body.
    // Output token of wait_all shall be yielded
    auto loop_op_terminator = loop_op.getBody()->getTerminator();
    builder.setInsertionPoint(loop_op_terminator);
    air::WaitAllOp wait_all_op_yielded = builder.create<xilinx::air::WaitAllOp>(
        builder.getUnknownLoc(),
        air::AsyncTokenType::get(loop_op->getContext()), sinks_in_loop_op);
    wait_all_op_yielded->setAttr(
        "id",
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(loop_op->getContext(), 32), ++WaitAllOpID));
    return wait_all_op_yielded;
  }

  Graph::vertex_descriptor
  addVertexWaitAllOpBeforeLoopYield(SmallVector<Value, 1> sinks_in_loop_op,
                                    std::string loop_type) {
    // Create vertex
    Graph::vertex_descriptor wait_all_op_yielded_v =
        add_vertex(asyncExecuteGraphTR);
    asyncExecuteGraphTR[wait_all_op_yielded_v].asyncEventName = "scf::";
    asyncExecuteGraphTR[wait_all_op_yielded_v].asyncEventName += loop_type;
    asyncExecuteGraphTR[wait_all_op_yielded_v].asyncEventName += "_loop_end";
    asyncExecuteGraphTR[wait_all_op_yielded_v].asyncEventType = "wait_all";
    asyncExecuteGraphTR[wait_all_op_yielded_v].color = "crimson";
    asyncExecuteGraphTR[wait_all_op_yielded_v].shape = "oval";
    asyncExecuteGraphTR[wait_all_op_yielded_v].operationId = 0;
    // Update graph connectivity
    for (auto sink : sinks_in_loop_op) {
      unsigned src_id = 0;
      if (auto async_execute_op =
              dyn_cast<air::ExecuteOp>(sink.getDefiningOp())) {
        src_id = g_to_tr[getGraphGVertexFromAIROp(async_execute_op)];
      } else if (auto dma_op =
                     dyn_cast<air::DmaMemcpyInterface>(sink.getDefiningOp())) {
        src_id = g_to_tr[getGraphGVertexFromAIROp(dma_op)];
      } else if (auto hier_op =
                     dyn_cast<air::HierarchyInterface>(sink.getDefiningOp())) {
        src_id = g_to_tr[getGraphGVertexFromAIROp(hier_op)];
      } else if (auto scf_for_op = dyn_cast<scf::ForOp>(sink.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(
            scf_for_op); // g_to_tr not needed since wait_all created after TR
      } else if (auto scf_parallel_op =
                     dyn_cast<scf::ParallelOp>(sink.getDefiningOp())) {
        src_id = getGraphGVertexFromAIROp(
            scf_parallel_op); // g_to_tr not needed since wait_all created after
                              // TR
      }
      add_edge(src_id, wait_all_op_yielded_v, asyncExecuteGraphTR);
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
    air::WaitAllOp wait_all_op_before_loop =
        builder.create<xilinx::air::WaitAllOp>(
            builder.getUnknownLoc(),
            air::AsyncTokenType::get(loop_op->getContext()), incoming_tokens);
    wait_all_op_before_loop->setAttr(
        "id",
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(loop_op->getContext(), 32), ++WaitAllOpID));

    // Create vertex
    Graph::vertex_descriptor wait_all_op_before_loop_v =
        add_vertex(asyncExecuteGraphTR);
    asyncExecuteGraphTR[wait_all_op_before_loop_v].asyncEventName = "scf::";
    asyncExecuteGraphTR[wait_all_op_before_loop_v].asyncEventName += loop_type;
    asyncExecuteGraphTR[wait_all_op_before_loop_v].asyncEventName +=
        "_loop_begin";
    asyncExecuteGraphTR[wait_all_op_before_loop_v].asyncEventType = "wait_all";
    asyncExecuteGraphTR[wait_all_op_before_loop_v].color = "crimson";
    asyncExecuteGraphTR[wait_all_op_before_loop_v].shape = "oval";
    asyncExecuteGraphTR[wait_all_op_before_loop_v].operationId = 0;
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

  scf::ForOp
  replaceLoopOpWithNewTerminator(OpBuilder &builder, scf::ForOp loop_op,
                                 air::WaitAllOp wait_all_op_before_loop,
                                 SmallVector<Value, 4> incoming_tokens,
                                 SmallVector<Value, 4> constants) {
    // Create new for op with iter_args.
    SmallVector<Value, 4> merged_incoming_token;
    merged_incoming_token.push_back(wait_all_op_before_loop.getResult(0));
    scf::ForOp new_loop_op = builder.create<scf::ForOp>(
        loop_op.getLoc(), loop_op.getLowerBound(), loop_op.getUpperBound(),
        loop_op.getStep(), merged_incoming_token);

    if (auto attr = loop_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_loop_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

    // Splice the operations inside loop op
    auto &bb = new_loop_op.getBody()->getOperations();
    auto &body = loop_op.getBody()->getOperations();
    bb.splice(bb.begin(), body, body.begin(), --body.end());

    auto iv = loop_op.getInductionVar();
    iv.replaceAllUsesWith(new_loop_op.getInductionVar());
    builder.setInsertionPointToStart(new_loop_op.getBody());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, builder.clone(*c.getDefiningOp())->getResult(0),
          new_loop_op.getRegion());
    }

    for (Value v : incoming_tokens) {
      replaceAllUsesInRegionWith(v, new_loop_op.getRegionIterArgs()[0],
                                 new_loop_op.getRegion());
    }

    // Connect sources in loop body with iter_args
    for (auto async_execute_op : new_loop_op.getOps<air::ExecuteOp>()) {
      if (async_execute_op.getAsyncDependencies().size() == 0) {
        async_execute_op.addAsyncDependency(new_loop_op.getRegionIterArgs()[0]);
      }
    }
    for (auto dma_op : new_loop_op.getOps<air::DmaMemcpyInterface>()) {
      auto async_op =
          mlir::dyn_cast<air::AsyncOpInterface>(dma_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getRegionIterArgs()[0]);
      }
    }
    for (auto hier_op : new_loop_op.getOps<air::HierarchyInterface>()) {
      auto async_op =
          mlir::dyn_cast<air::AsyncOpInterface>(hier_op.getOperation());
      if (async_op.getAsyncDependencies().size() == 0) {
        async_op.addAsyncDependency(new_loop_op.getRegionIterArgs()[0]);
      }
    }

    return new_loop_op;
  }

  scf::ParallelOp
  replaceLoopOpWithNewTerminator(OpBuilder &builder, scf::ParallelOp loop_op,
                                 air::WaitAllOp wait_all_op_before_loop,
                                 SmallVector<Value, 4> incoming_tokens,
                                 SmallVector<Value, 4> constants) {

    assert(
        loop_op.getNumReductions() == 0 &&
        "Currently only supporting input scf::ParallelOp with no reductions");

    // Create new parallel op with init_val.
    SmallVector<Value, 4> merged_incoming_token;
    merged_incoming_token.push_back(wait_all_op_before_loop.getResult(0));
    scf::ParallelOp new_loop_op = builder.create<scf::ParallelOp>(
        loop_op.getLoc(), loop_op.getLowerBound(), loop_op.getUpperBound(),
        loop_op.getStep(), merged_incoming_token);

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

    builder.setInsertionPointToStart(new_loop_op.getBody());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(
          c, builder.clone(*c.getDefiningOp())->getResult(0),
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
    for (auto dma_op : new_loop_op.getOps<air::DmaMemcpyInterface>()) {
      auto async_op =
          mlir::dyn_cast<air::AsyncOpInterface>(dma_op.getOperation());
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

  // Elevating tokens from inside loop body to the yielded token, to maintain
  // legal domination T: loop type (scf::ForOp or scf::ParallelOp) U: source op
  // type
  template <typename T, typename U>
  void elevateAsyncTokens(T new_loop_op, Graph::vertex_descriptor wait_all_op) {
    for (auto source : new_loop_op.template getOps<U>()) {
      SmallPtrSet<Operation *, 1> keep;
      if (source->getResult(0)) {
        for (auto sink : source->getResult(0).getUsers()) {
          // Keep token if source already dominates sink
          if (source->getParentRegion() == sink->getParentRegion()) {
            keep.insert(sink);
          } else {
            // Update graph connectivity
            insertVertexBetweenTwoOps(source.getOperation(), sink, wait_all_op);
          }
        }
      }
      source->getResult(0).replaceAllUsesExcept(new_loop_op.getResult(0), keep);
    }
  }

  void insertLoopCarriedDeps(OpBuilder &builder, scf::ForOp &loop_op,
                             SmallVector<Value, 1> sinks_in_loop_op) {
    // (1) Create one wait_all event at the end of current for loop body.
    air::WaitAllOp wait_all_op_yielded =
        insertWaitAllOpBeforeLoopYield<scf::ForOp>(builder, loop_op,
                                                   sinks_in_loop_op);

    // Update boost graph
    Graph::vertex_descriptor wait_all_op_yielded_v =
        addVertexWaitAllOpBeforeLoopYield(sinks_in_loop_op, "for");
    // Update op-to-graph map for wait_all ops
    wa_to_g[wait_all_op_yielded.getId()] = wait_all_op_yielded_v;

    // (2) Create a new wait_all event before the for op which collects the
    // incoming deps.
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
    air::WaitAllOp wait_all_op_before_loop =
        insertWaitAllOpAtLoopBegin<scf::ForOp>(builder, loop_op, "for",
                                               incoming_tokens, constants);

    // (3) Create new for op with iter_args.
    scf::ForOp new_loop_op = replaceLoopOpWithNewTerminator(
        builder, loop_op, wait_all_op_before_loop, incoming_tokens, constants);

    // Yield an async token
    SmallVector<Value, 4> yield_token;
    yield_token.push_back(wait_all_op_yielded.getResult(0));
    builder.setInsertionPointToEnd(new_loop_op.getBody());
    builder.create<scf::YieldOp>(new_loop_op.getLoc(), yield_token);

    // Elevating tokens from inside forOp body to the yielded token, to maintain
    // dominance
    elevateAsyncTokens<scf::ForOp, air::AsyncOpInterface>(
        new_loop_op, wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ForOp, scf::ForOp>(new_loop_op,
                                               wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ForOp, scf::ParallelOp>(new_loop_op,
                                                    wait_all_op_yielded_v);

    loop_op.erase();

    loop_op = new_loop_op;
  }

  void insertLoopCarriedDeps(OpBuilder &builder, scf::ParallelOp &loop_op,
                             SmallVector<Value, 1> sinks_in_loop_op) {
    // (1) Create one wait_all event at the end of current parallel loop body.
    air::WaitAllOp wait_all_op_yielded =
        insertWaitAllOpBeforeLoopYield<scf::ParallelOp>(builder, loop_op,
                                                        sinks_in_loop_op);

    // Update boost graph
    Graph::vertex_descriptor wait_all_op_yielded_v =
        addVertexWaitAllOpBeforeLoopYield(sinks_in_loop_op, "parallel");
    // Update op-to-graph map for wait_all ops
    wa_to_g[wait_all_op_yielded.getId()] = wait_all_op_yielded_v;

    // (2) Create a new wait_all event before the parallel op which collects the
    // incoming deps.
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
    air::WaitAllOp wait_all_op_before_loop =
        insertWaitAllOpAtLoopBegin<scf::ParallelOp>(
            builder, loop_op, "parallel", incoming_tokens, constants);

    // (3) Create new parallel op with init_val.
    scf::ParallelOp new_loop_op = replaceLoopOpWithNewTerminator(
        builder, loop_op, wait_all_op_before_loop, incoming_tokens, constants);

    // Remove the old scf::YieldOp
    SmallVector<scf::YieldOp, 2> y_ops(new_loop_op.getOps<scf::YieldOp>());
    for (auto y_op : y_ops)
      y_op.erase();

    // Create scf::ReduceOp
    builder.setInsertionPointToEnd(new_loop_op.getBody());
    auto reduce_op = builder.create<scf::ReduceOp>(
        new_loop_op.getLoc(), wait_all_op_yielded.getResult(0));
    builder.setInsertionPointToStart(&reduce_op.getRegion().front());
    SmallVector<Value, 4> reduce_tokens;
    reduce_tokens.push_back(reduce_op.getRegion().front().getArgument(0));
    reduce_tokens.push_back(reduce_op.getRegion().front().getArgument(1));
    auto reduce_res = builder.create<xilinx::air::WaitAllOp>(
        builder.getUnknownLoc(),
        air::AsyncTokenType::get(loop_op->getContext()), reduce_tokens);
    builder.create<scf::ReduceReturnOp>(builder.getUnknownLoc(),
                                        reduce_res.getResult(0));
    builder.setInsertionPointToEnd(new_loop_op.getBody());
    builder.create<scf::YieldOp>(new_loop_op.getLoc());

    // Elevating tokens from inside forOp body to the yielded token, to maintain
    // dominance
    elevateAsyncTokens<scf::ParallelOp, air::AsyncOpInterface>(
        new_loop_op, wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ParallelOp, scf::ForOp>(new_loop_op,
                                                    wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ParallelOp, scf::ParallelOp>(new_loop_op,
                                                         wait_all_op_yielded_v);

    loop_op.erase();

    loop_op = new_loop_op;
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

  //===----------------------------------------------------------------------===//
  // Async event to Boost graph mapping
  //===----------------------------------------------------------------------===//

  // Dependency graph constructed as Boost graph
  Graph asyncExecuteGraph;
  Graph asyncExecuteGraphTR;
  vertex_map g_to_tr,
      tr_to_g; // Map between graph g and graph tr (post-tr graph)
  operation_id_to_vertex_map
      region_to_g; // Map between air executes and vertices in graph
  operation_id_to_vertex_map
      dma_to_g; // Map between air dmamemcpy2d and vertices in graph
  operation_id_to_vertex_map
      hier_to_g; // Map between air hierarchy and vertices in graph
  operation_id_to_vertex_map
      wa_to_g; // Map between air wait_all and vertices in graph

  // g vertex to air op mapping
  air::ExecuteOp getExecuteOpFromVertex(Graph::vertex_descriptor v, Graph g) {
    assert(g[v].asyncEventType == "execute" &&
           "This vertex is not a ExecuteOp");
    return async_execute_op_history[g[v].operationId - 1];
  }
  air::DmaMemcpyInterface getDmaOpFromVertex(Graph::vertex_descriptor v,
                                             Graph g) {
    assert(g[v].asyncEventType == "dma" && "This vertex is not a DmaMemcpy op");
    return dma_op_history[g[v].operationId - 1];
  }
  air::HierarchyInterface getHierOpFromVertex(Graph::vertex_descriptor v,
                                              Graph g) {
    assert(g[v].asyncEventType == "hierarchy" &&
           "This vertex is not a Hierarchy op");
    return hier_op_history[g[v].operationId - 1];
  }

  // air execute op to g vertex mapping
  Graph::vertex_descriptor getGraphGVertexFromAIROp(air::ExecuteOp op) {
    return region_to_g[op.getId()];
  }

  Graph::vertex_descriptor
  getGraphGVertexFromAIROp(air::DmaMemcpyInterface op) {
    return dma_to_g[op.getId()];
  }

  Graph::vertex_descriptor
  getGraphGVertexFromAIROp(air::HierarchyInterface op) {
    return hier_to_g[op.getId()];
  }

  Graph::vertex_descriptor getGraphGVertexFromAIROp(air::WaitAllOp op) {
    return wa_to_g[op.getId()];
  }

  Graph::vertex_descriptor getGraphGVertexFromAIROp(scf::ForOp op) {
    // Note: using forOp's last wait_all's id as the forOp's id
    air::WaitAllOp last_wa_op;
    // MLIR iterators cannot get the last element directly?
    for (air::WaitAllOp wa_op : op.getOps<air::WaitAllOp>()) {
      last_wa_op = wa_op;
    }
    return wa_to_g[last_wa_op.getId()];
  }

  Graph::vertex_descriptor getGraphGVertexFromAIROp(scf::ParallelOp op) {
    // Note: using parallelOp's last wait_all's id as the parallelOp's id
    air::WaitAllOp last_wa_op;
    // MLIR iterators cannot get the last element directly?
    for (air::WaitAllOp wa_op : op.getOps<air::WaitAllOp>()) {
      last_wa_op = wa_op;
    }
    return wa_to_g[last_wa_op.getId()];
  }

  // Fill in dep list of air async ops using graph tr's connectivity
  template <typename T> void fillAIRDepListUsingGraphTR(T op) {
    if (auto async_op =
            mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op.getOperation())) {
      uint64_t dstTRVertex = g_to_tr[getGraphGVertexFromAIROp(op)];
      auto incoming_deps = in_edges(dstTRVertex, asyncExecuteGraphTR);
      for (in_edge_iterator it = incoming_deps.first;
           it != incoming_deps.second; it++) {
        auto TRVertex = source(*it, asyncExecuteGraphTR);
        if (asyncExecuteGraphTR[TRVertex].asyncEventType == "execute")
          async_op.addAsyncDependency(
              getExecuteOpFromVertex(TRVertex, asyncExecuteGraphTR)
                  .getResult(0));
        else if (asyncExecuteGraphTR[TRVertex].asyncEventType == "dma")
          async_op.addAsyncDependency(
              getDmaOpFromVertex(TRVertex, asyncExecuteGraphTR)
                  .getOperation()
                  ->getResult(0));
        else if (asyncExecuteGraphTR[TRVertex].asyncEventType == "hierarchy")
          async_op.addAsyncDependency(
              getHierOpFromVertex(TRVertex, asyncExecuteGraphTR)
                  .getOperation()
                  ->getResult(0));
        else
          assert(false && "Unknown async event type");
      }
    } else
      assert(false && "Operation has no async interface");
  }

  template <typename T> void addNewAsyncDepToGraph(Value dep, T op) {
    if (auto async_op =
            mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op.getOperation())) {
      for (auto old_dep : async_op.getAsyncDependencies())
        if (old_dep == dep)
          return;

      // Add edge to boost graph, iff dep is async execute region (i.e. not a
      // loop iterator)
      if (auto srcOp = dep.getDefiningOp()) {
        uint64_t srcNode;
        if (auto execute_op = dyn_cast<air::ExecuteOp>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(execute_op);
        } else if (auto dma_op = dyn_cast<air::DmaMemcpyInterface>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(dma_op);
        } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(srcOp)) {
          srcNode = getGraphGVertexFromAIROp(hier_op);
        } else
          assert(false &&
                 "dependency token should be generated by an async op");
        uint64_t dstNode = getGraphGVertexFromAIROp(op);
        add_edge(srcNode, dstNode, asyncExecuteGraph);
      }
    } else
      assert(false && "Operation has no async interface");
  }

  // Dump graphviz
  void dump_graph(std::string filename) {
    std::ofstream ofs(filename, std::ofstream::out);
    boost::dynamic_properties dp;
    dp.property("label",
                boost::get(&executeNode::asyncEventName, asyncExecuteGraphTR));
    dp.property("color", boost::get(&executeNode::color, asyncExecuteGraphTR));
    dp.property("shape", boost::get(&executeNode::shape, asyncExecuteGraphTR));
    dp.property("node_id",
                boost::get(boost::vertex_index, asyncExecuteGraphTR));
    dp.property(
        "style",
        boost::make_constant_property<Graph::vertex_descriptor>(+"filled"));
    write_graphviz_dp(ofs, asyncExecuteGraphTR, dp);
  }

  //===----------------------------------------------------------------------===//
  // Other utilities
  //===----------------------------------------------------------------------===//

  bool foundAsyncOpUsesAboveCurrentLine(air::ExecuteOp *op) {
    if (!async_execute_op_history.empty())
      for (auto &iter : async_execute_op_history)
        if (iter.getResult(0) == op->getResult(0))
          return true;
    return false;
  }

  bool foundAsyncOpUsesAboveCurrentLine(air::DmaMemcpyInterface *op) {
    if (!dma_op_history.empty())
      for (auto &iter : dma_op_history)
        if (iter->getResult(0) == op->getOperation()->getResult(0))
          return true;
    return false;
  }

  bool foundAsyncOpUsesAboveCurrentLine(air::HierarchyInterface *op) {
    if (!hier_op_history.empty())
      for (auto &iter : hier_op_history)
        if (iter->getResult(0) == op->getOperation()->getResult(0))
          return true;
    return false;
  }

  // Check if two partial memref tiles have identical indices
  bool areEqualIndexPartialMemrefs(partialMemref *tile_0,
                                   partialMemref *tile_1) {
    if (tile_0->numDims != tile_1->numDims) {
      // Unequal # dimensions
      return false;
    } else {
      for (unsigned i = 0; i < tile_0->numDims; i++) {
        if (!areEqualIndices(tile_0->memrefIndices[i],
                             tile_1->memrefIndices[i]))
          return false;
      }
    }
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
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyPass() {
  return std::make_unique<AIRDependency>();
}

} // namespace air
} // namespace xilinx