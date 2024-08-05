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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
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
using namespace xilinx;

#define DEBUG_TYPE "air-dependency"

namespace {

// Remove an op if it has no users, else return failure.
// This is a temporary measure, while the issue
// https://github.com/Xilinx/mlir-air/issues/372
// is open. Once the root cause is found, there should be no ops erased here
// whose results have users.
LogicalResult eraseOpWithCheck(Operation *op, std::string_view context = "") {
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

  op->erase();
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

using ExecuteGraph = xilinx::air::TypedDirectedAdjacencyMap<::executeNode>;

typedef std::map<ExecuteGraph::VertexId, ExecuteGraph::VertexId> vertex_map;
typedef std::map<unsigned, ExecuteGraph::VertexId> operation_id_to_vertex_map;

class AIRDependency
    : public xilinx::air::impl::AIRDependencyBase<AIRDependency> {

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
      xilinx::air::renumberDmaOps(f, "global");
    }

    // 1st traversal: create async ops with empty dep list.

    OpBuilder module_builder(module);

    ExecuteOpID = 0;
    HierarchyOpID = 0;
    WaitAllOpID = 0;
    ChannelOpID = 0;

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        // Create async interface for air.dmamemcpy ops
        if (isa<xilinx::air::DmaMemcpyNdOp>(op))
          createAsyncDMA(module_builder, op);

        // Create async interface for air.channel ops
        else if (isa<xilinx::air::ChannelInterface>(op))
          createAsyncChannel(module_builder, op, ChannelOpID);

        // Create async execute region for linalg.matmul
        else if (isa<linalg::MatmulOp>(op))
          createAsyncExecute(module_builder, op, "linalg::matmul", ExecuteOpID);

        // Create async execute region for linalg.fill
        else if (isa<linalg::FillOp>(op))
          createAsyncExecute(module_builder, op, "linalg::fill", ExecuteOpID);

        // Create async execute region for linalg.copy
        else if (isa<linalg::CopyOp>(op))
          createAsyncExecute(module_builder, op, "linalg::copy", ExecuteOpID);

        // Create async execute region for linalg op
        else if (isa<linalg::LinalgOp>(op))
          createAsyncExecute(module_builder, op, "linalg::unknown",
                             ExecuteOpID);

        // Create async interface for func.call ops.
        else if (isa<func::CallOp>(op))
          createAsyncExecute(module_builder, op, "func::call", ExecuteOpID);

        // Create async execute region for memref.alloc
        else if (auto memcast_op = dyn_cast<memref::CastOp>(op))
          createAsyncExecute(module_builder, op, "memref::cast", ExecuteOpID,
                             memcast_op.getDest().getType());

        // Create async execute region for memref.dealloc
        else if (isa<memref::DeallocOp>(op))
          createAsyncExecute(module_builder, op, "memref::dealloc",
                             ExecuteOpID);

        // Create async execute region for memref.copy
        else if (isa<memref::CopyOp>(op))
          createAsyncExecute(module_builder, op, "memref::copy", ExecuteOpID);

        // Create async execute region for arith.muli
        else if (auto arith_op = dyn_cast<arith::MulIOp>(op)) {
          if (llvm::isa<IndexType>(arith_op.getResult().getType())) {
            createAsyncExecute(module_builder, op, "arith::muli", ExecuteOpID,
                               arith_op.getResult().getType());
          }
        }

        // Create async execute region for arith.addi
        else if (auto arith_op = dyn_cast<arith::AddIOp>(op)) {
          if (llvm::isa<IndexType>(arith_op.getResult().getType())) {
            createAsyncExecute(module_builder, op, "arith::addi", ExecuteOpID,
                               arith_op.getResult().getType());
          }
        }

        // Create async execute region for affine.apply
        else if (auto apply_op = dyn_cast<mlir::affine::AffineApplyOp>(op))
          createAsyncExecute(module_builder, op, "affine::apply", ExecuteOpID,
                             apply_op.getResult().getType());

        // Create async execute region for air hierarchy ops (air.launch and
        // air.segment, TODO: air.herd).
        else if (auto hierarchy_op = dyn_cast<air::HierarchyInterface>(op)) {
          createAsyncHierarchyImpls(module_builder, hierarchy_op,
                                    HierarchyOpID);
        }

        // Create async execute region for memref.alloc
        else if (auto memalloc_op = dyn_cast<memref::AllocOp>(op)) {
          // Alloc can be used to specify shapes for operations such
          // as reshape ops. If this alloc is used to specify shape of
          // a reshap op, ignore this operation.
          assert(memalloc_op->getNumResults() == 1 &&
                 "Number of results of alloc op == 1");
          if (!alloc_for_reshape(memalloc_op->getOpResult(0)))
            createAsyncExecute(module_builder, op, "memref::alloc", ExecuteOpID,
                               memalloc_op.getMemref().getType());
        }

        // Create async execute region for an unknown op which has memref or
        // index-type operands
        else {
          bool isCandidateExecute = false;
          for (auto operand : op->getOperands()) {
            if (llvm::isa<MemRefType>(operand.getType()) ||
                llvm::isa<IndexType>(operand.getType())) {
              isCandidateExecute = true;
            }
          }
          // If a memref store is storing to an alloca for shape,
          // it must not be executed.
          if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
            if (auto memalloc_op = dyn_cast<memref::AllocOp>(
                    storeOp.getMemRef().getDefiningOp())) {
              assert(memalloc_op->getNumResults() == 1 &&
                     "Number of results of alloc op == 1");
              if (alloc_for_reshape(memalloc_op->getOpResult(0)))
                isCandidateExecute = false;
            }
          }
          // No air execute for expand, collapse and reshape ops
          if (dyn_cast<memref::ReshapeOp>(op) ||
              dyn_cast<memref::ExpandShapeOp >(op) ||
              dyn_cast<memref::CollapseShapeOp >(op)) {
            isCandidateExecute = false;
          }
          // No air execute for loop ops
          if (isa<mlir::LoopLikeOpInterface>(op))
            isCandidateExecute = false;
          // No air execute for subview ops
          if (isa<mlir::OffsetSizeAndStrideOpInterface>(op))
            isCandidateExecute = false;
          // No air execute for memref.extract_strided_metadata ops.
          if (isa<memref::ExtractStridedMetadataOp>(op))
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

    llvm::errs() << "\n\nBEFORE SECOND TRAVERSAL:\n";
    module.dump();

    // 2nd traversal: trace deps among async execute regions; build a dep graph

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        Operation *sink_op = nullptr;
        if (auto async_execute_op = dyn_cast<air::ExecuteOp>(op)) {
          for (auto &bb : async_execute_op.getBody()) {
            for (auto &child_op : bb.getOperations()) {
              if (!dyn_cast<air::ExecuteTerminatorOp>(child_op))
                sink_op = &child_op;
            }
          }
        } else if (isa<xilinx::air::DmaMemcpyNdOp>(op)) {
          sink_op = op;
        } else if (isa<xilinx::air::ChannelInterface>(op)) {
          sink_op = op;
        } else if (dyn_cast<air::HierarchyInterface>(op)) {
          sink_op = op;
        } else
          return;

        SmallVector<partialMemref, 1> sink_op_memref_reads;
        SmallVector<partialMemref, 1> sink_op_memref_writes;
        SmallVector<Value, 1> sink_op_scalar_ins;
        SmallVector<Value, 1> sink_op_scalar_outs;
        llvm::errs() << "SINK OP:\n";
        sink_op->dump();
        // If the sink op is linalg op
        if (auto sink_op_linalgop = dyn_cast<linalg::LinalgOp>(sink_op)) {
          for (auto ins_value : sink_op_linalgop.getDpsInputs()) {
            if (llvm::isa<MemRefType>(ins_value.getType())) {
              unsigned memRefRank =
                  llvm::cast<MemRefType>(ins_value.getType()).getRank();
              partialMemref tile = createPartialMemref(ins_value, memRefRank);
              sink_op_memref_reads.push_back(tile);
            } else if (llvm::isa<IndexType>(ins_value.getType())) {
              sink_op_scalar_ins.push_back(ins_value);
            }
          }
          for (auto outs_value : sink_op_linalgop.getDpsInits()) {
            if (llvm::isa<MemRefType>(outs_value.getType())) {
              unsigned memRefRank =
                  llvm::cast<MemRefType>(outs_value.getType()).getRank();
              partialMemref tile = createPartialMemref(outs_value, memRefRank);
              sink_op_memref_reads.push_back(
                  tile); // linalg op both reads and writes the output memref
              sink_op_memref_writes.push_back(tile);
            } else if (llvm::isa<IndexType>(outs_value.getType())) {
              sink_op_scalar_ins.push_back(
                  outs_value); // linalg op both reads and writes the output
                               // memref
              sink_op_scalar_outs.push_back(outs_value);
            }
          }
          if (sink_op_linalgop->getNumResults()) {
            for (auto linalg_results : sink_op_linalgop->getResults()) {
              if (llvm::isa<MemRefType>(linalg_results.getType())) {
                unsigned memRefRank =
                    llvm::cast<MemRefType>(linalg_results.getType()).getRank();
                partialMemref tile =
                    createPartialMemref(linalg_results, memRefRank);
                sink_op_memref_writes.push_back(tile);
              } else if (llvm::isa<IndexType>(linalg_results.getType())) {
                sink_op_scalar_outs.push_back(linalg_results);
              }
            }
          }
        }

        // If the sink op is memref::dealloc
        else if (auto sink_op_memdealloc =
                     dyn_cast<memref::DeallocOp>(sink_op)) {
          mlir::Value sink_op_memref = sink_op_memdealloc.getMemref();
          if (auto operation = sink_op_memref.getDefiningOp()) {
            if (dyn_cast<memref::ReshapeOp>(operation) || 
                dyn_cast<memref::ExpandShapeOp>(operation) ||
                dyn_cast<memref::CollapseShapeOp>(operation)) {
                sink_op_memref = operation->getOperand(0);
            }
          }
          unsigned memRefRank =
              llvm::cast<MemRefType>(sink_op_memref.getType()).getRank();
          partialMemref tile =
              createPartialMemref(sink_op_memref, memRefRank);
          sink_op_memref_reads.push_back(tile);
          sink_op_memref_writes.push_back(
              tile); // dealloc erases (i.e. writes to) output memref
        }

        // If the sink op is memref::copy
        else if (auto sink_op_memref_copy = dyn_cast<memref::CopyOp>(sink_op)) {
          mlir::Value op_src_memref = sink_op_memref_copy.getSource();
          if (auto operation = op_src_memref.getDefiningOp()) {
            if (dyn_cast<memref::ReshapeOp>(operation) || 
                dyn_cast<memref::ExpandShapeOp>(operation) ||
                dyn_cast<memref::CollapseShapeOp>(operation)) {
                op_src_memref = operation->getOperand(0);
            }
          }
          unsigned memRefRankSrc =
              llvm::cast<MemRefType>(op_src_memref.getType()).getRank();
          partialMemref tileSrc = createPartialMemref(
              op_src_memref, memRefRankSrc);
          sink_op_memref_reads.push_back(tileSrc);
          unsigned memRefRankDst =
              llvm::cast<MemRefType>(sink_op_memref_copy.getTarget().getType())
                  .getRank();
          partialMemref tileDst = createPartialMemref(
              sink_op_memref_copy.getTarget(), memRefRankDst);
          sink_op_memref_reads.push_back(tileDst);
          sink_op_memref_writes.push_back(tileDst);
        }

        // If the sink op is an air::MemcpyInterface op
        else if (auto sink_op_memcpy =
                     mlir::dyn_cast<xilinx::air::MemcpyInterface>(sink_op)) {
          auto op_src_memref = sink_op_memcpy.getSrcMemref();
          if (op_src_memref) {
            if (auto operation = op_src_memref.getDefiningOp()) {
                if (dyn_cast<memref::ReshapeOp>(operation) || 
                    dyn_cast<memref::ExpandShapeOp>(operation) ||
                    dyn_cast<memref::CollapseShapeOp>(operation)) {
                    op_src_memref = operation->getOperand(0);
                }
            }
          }
          if (llvm::isa<MemRefType>(op_src_memref.getType())) {
            SmallVector<Value, 2> src_indices;
            unsigned numDimsSrc =
                llvm::cast<MemRefType>(op_src_memref.getType()).getRank();
            for (unsigned i = 0; i < sink_op_memcpy.getSrcOffsets().size(); i++)
              sink_op_scalar_ins.push_back(sink_op_memcpy.getSrcOffsets()[i]);
            for (unsigned i = 0; i < sink_op_memcpy.getSrcSizes().size(); i++)
              sink_op_scalar_ins.push_back(sink_op_memcpy.getSrcSizes()[i]);
            for (unsigned i = 0; i < sink_op_memcpy.getSrcStrides().size(); i++)
              sink_op_scalar_ins.push_back(sink_op_memcpy.getSrcStrides()[i]);
            if (sink_op_memcpy.getSrcOffsets().size()) {
              numDimsSrc = sink_op_memcpy.getSrcOffsets().size();
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(sink_op_memcpy.getSrcOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(nullptr);
              }
            }
            partialMemref tile_in = createPartialMemref(
                op_src_memref, numDimsSrc, src_indices);
            sink_op_memref_reads.push_back(tile_in);
          }
          if (sink_op_memcpy.getDstMemref()) {
            SmallVector<Value, 2> dst_indices;
            unsigned numDimsDst =
                llvm::cast<MemRefType>(sink_op_memcpy.getDstMemref().getType())
                    .getRank();
            for (unsigned i = 0; i < sink_op_memcpy.getDstOffsets().size(); i++)
              sink_op_scalar_outs.push_back(sink_op_memcpy.getDstOffsets()[i]);
            for (unsigned i = 0; i < sink_op_memcpy.getDstSizes().size(); i++)
              sink_op_scalar_outs.push_back(sink_op_memcpy.getDstSizes()[i]);
            for (unsigned i = 0; i < sink_op_memcpy.getDstStrides().size(); i++)
              sink_op_scalar_outs.push_back(sink_op_memcpy.getDstStrides()[i]);
            if (sink_op_memcpy.getDstOffsets().size()) {
              numDimsDst = sink_op_memcpy.getDstOffsets().size();
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(sink_op_memcpy.getDstOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(nullptr);
              }
            }
            partialMemref tile_out = createPartialMemref(
                sink_op_memcpy.getDstMemref(), numDimsDst, dst_indices);
            sink_op_memref_writes.push_back(tile_out);
          }
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

        // If the sink op is mlir::affine::AffineApplyOp
        else if (auto sink_op_apply =
                     dyn_cast<mlir::affine::AffineApplyOp>(sink_op)) {
          for (auto applyop_operand : sink_op_apply.getMapOperands()) {
            sink_op_scalar_ins.push_back(applyop_operand);
          }
          sink_op_scalar_outs.push_back(sink_op_apply.getResult());
        }

        // If the sink op is an unknown op
        else {
          for (auto sink_op_op : sink_op->getOperands()) {
            if (auto operation = sink_op_op.getDefiningOp()) {
                if (dyn_cast<memref::ReshapeOp>(operation) || 
                    dyn_cast<memref::ExpandShapeOp>(operation) ||
                    dyn_cast<memref::CollapseShapeOp>(operation)) {
                    sink_op_op = operation->getOperand(0);
                }
            }
            if (llvm::isa<MemRefType>(sink_op_op.getType())) {
              unsigned memRefRank =
                  llvm::cast<MemRefType>(sink_op_op.getType()).getRank();
              partialMemref tile = createPartialMemref(sink_op_op, memRefRank);
              sink_op_memref_reads.push_back(
                  tile); // Assuming all operands are both read and written to
              sink_op_memref_writes.push_back(tile);
            } else if (llvm::isa<IndexType>(sink_op_op.getType())) {
              sink_op_scalar_ins.push_back(
                  sink_op_op); // Assuming all operands are both read and
                               // written to
              sink_op_scalar_outs.push_back(sink_op_op);
            }
          }
          if (sink_op->getNumResults()) {
            for (auto sink_op_results : sink_op->getResults()) {
              if (llvm::isa<MemRefType>(sink_op_results.getType())) {
                unsigned memRefRank =
                    llvm::cast<MemRefType>(sink_op_results.getType()).getRank();
                partialMemref tile =
                    createPartialMemref(sink_op_results, memRefRank);
                sink_op_memref_writes.push_back(tile);
              } else if (llvm::isa<IndexType>(sink_op_results.getType())) {
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
                       mlir::dyn_cast<xilinx::air::DmaMemcpyNdOp>(op)) {
          traceDeps<air::DmaMemcpyNdOp>(sink_op_memref_reads, dma_op, "RAW");
          traceDeps<air::DmaMemcpyNdOp>(sink_op_memref_writes, dma_op,
                                        "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs, dma_op);
          dma_op_history.push_back(dma_op);
        } else if (auto channel_op =
                       mlir::dyn_cast<xilinx::air::ChannelInterface>(op)) {
          traceDeps<air::ChannelInterface>(sink_op_memref_reads, channel_op,
                                           "RAW");
          traceDeps<air::ChannelInterface>(sink_op_memref_writes, channel_op,
                                           "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                           sink_op_scalar_ins, sink_op_scalar_outs, channel_op);
          channel_op_history.push_back(channel_op);
        } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
          hier_op_history.push_back(hier_op);
        }
      });
    }

    llvm::errs() << "\n\nBEFORE THIRD TRAVERSAL:\n";
    module.dump();

    // 3rd traversal: perform transitive reduction on dependency graph.

    std::vector<size_t> id_map(asyncExecuteGraph.numVertices());
    std::iota(id_map.begin(), id_map.end(), 0u);

    asyncExecuteGraph.applyTransitiveReduction();

    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        // Fill dep list of air execute ops
        if (auto async_execute_op = dyn_cast<air::ExecuteOp>(op)) {
          fillAIRDepListUsingGraphTR<air::ExecuteOp>(async_execute_op);
        }
        // Fill dep list of air dmamemcpy2d ops
        else if (auto dma_op = dyn_cast<air::DmaMemcpyNdOp>(op)) {
          fillAIRDepListUsingGraphTR<air::DmaMemcpyNdOp>(dma_op);
        }
        // Fill dep list of air channel ops
        else if (auto channel_op = dyn_cast<air::ChannelInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::ChannelInterface>(channel_op);
        }
        // Fill dep list of air hierarchy ops
        else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
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

          bool hasAsyncTokensInBody = false;
          SmallVector<Value, 1> yielded_tokens_in_for_op;

          // Conservative loop-carried dependency: no pipelining
          // TODO: loop pipelining support
          for (auto async_op : for_op.getOps<air::AsyncOpInterface>()) {
            hasAsyncTokensInBody = true;
            auto token = async_op.getOperation()->getResult(0);
            if (!isNotLoopCarriedOp(async_op) &&
                isOnlyUsedByNoLoopCarryOpsInBlock(token, for_op.getBody())) {
              yielded_tokens_in_for_op.push_back(token);
            }
          }
          for (auto child_for_op : for_op.getOps<scf::ForOp>()) {
            hasAsyncTokensInBody = true;
            if (auto token = child_for_op.getResult(0)) {
              if (isOnlyUsedByNoLoopCarryOpsInBlock(token, for_op.getBody()))
                yielded_tokens_in_for_op.push_back(token);
            }
          }
          for (auto child_parallel_op : for_op.getOps<scf::ParallelOp>()) {
            hasAsyncTokensInBody = true;
            if (auto token = child_parallel_op.getResult(0)) {
              if (isOnlyUsedByNoLoopCarryOpsInBlock(token, for_op.getBody()))
                yielded_tokens_in_for_op.push_back(token);
            }
          }

          if (hasAsyncTokensInBody) {
            insertLoopCarriedDeps(module_builder, for_op,
                                  yielded_tokens_in_for_op);
          }
        }

        else if (scf::ParallelOp for_op = dyn_cast<scf::ParallelOp>(op)) {

          bool hasAsyncTokensInBody = false;
          SmallVector<Value, 1> yielded_tokens_in_parallel_op;

          for (auto async_op : for_op.getOps<air::AsyncOpInterface>()) {
            hasAsyncTokensInBody = true;
            auto token = async_op.getOperation()->getResult(0);
            if (!isNotLoopCarriedOp(async_op) &&
                isOnlyUsedByNoLoopCarryOpsInBlock(token, for_op.getBody())) {
              yielded_tokens_in_parallel_op.push_back(token);
            }
          }
          for (auto child_for_op : for_op.getOps<scf::ForOp>()) {
            hasAsyncTokensInBody = true;
            if (auto token = child_for_op.getResult(0)) {
              if (isOnlyUsedByNoLoopCarryOpsInBlock(token, for_op.getBody()))
                yielded_tokens_in_parallel_op.push_back(token);
            }
          }
          for (auto child_parallel_op : for_op.getOps<scf::ParallelOp>()) {
            hasAsyncTokensInBody = true;
            if (auto token = child_parallel_op.getResult(0)) {
              if (isOnlyUsedByNoLoopCarryOpsInBlock(token, for_op.getBody()))
                yielded_tokens_in_parallel_op.push_back(token);
            }
          }

          if (hasAsyncTokensInBody) {
            insertLoopCarriedDeps(module_builder, for_op,
                                  yielded_tokens_in_parallel_op);
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

  //===----------------------------------------------------------------------===//
  // Handling lingering reshape-related ops
  //===----------------------------------------------------------------------===//

  bool alloc_for_reshape(mlir::Value val) {
    for (auto user : val.getUsers()) {
      // If one of the user is a herd, segment or a launch op,
      // explore the hierarchy further.
      if (auto hier_op = dyn_cast<xilinx::air::HierarchyInterface>(user)) {
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

  // Air async op history
  std::vector<air::ExecuteOp> async_execute_op_history;
  std::vector<air::DmaMemcpyNdOp> dma_op_history;
  std::vector<air::ChannelInterface> channel_op_history;
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
    Block *async_region_bb = builder.createBlock(&async_region.getBody());
    builder.setInsertionPointToStart(async_region_bb);

    // Handle cases when the operand(s) of the given op that is 
    // reshape/expand/collapse ops.
    for (unsigned idx = 0; idx < op->getNumOperands(); idx++) {
        auto operand = op->getOperand(idx);
        auto alt_shape_op = operand.getDefiningOp();
        llvm::errs() << "alt_shape_op:\n";
        alt_shape_op->dump();
        if (dyn_cast<memref::ReshapeOp>(alt_shape_op) || 
            dyn_cast<memref::ExpandShapeOp>(alt_shape_op) ||
            dyn_cast<memref::CollapseShapeOp>(alt_shape_op)) {
            assert(alt_shape_op->getNumResults() == 1 &&
                    "Number of results of shape changing op == 1");
            auto *cloned_op = builder.insert(alt_shape_op->clone());
            op->setOperand(idx, cloned_op->getResult(0));
            auto new_shape_val = alt_shape_op->getResult(0);
            if (new_shape_val.use_empty()) {
                // Erase this shape altering op
                alt_shape_op->erase();
            }
        }
    }
    
    builder.clone(*op);
    builder.create<xilinx::air::ExecuteTerminatorOp>(builder.getUnknownLoc());

    auto v = asyncExecuteGraph.addVertex();
    auto &node = asyncExecuteGraph[v];
    // Create a vertex out of the current async execute region
    node.asyncEventName = asyncEventName;
    node.asyncEventType = "execute";
    node.color = "chartreuse";
    node.shape = "oval";
    node.operationId = ExecuteOpID;

    // Update op-to-graph map
    region_to_g[async_region.getId()] = v;

    // Erase op
    if (eraseOpWithCheck(op, "createAsyncExecute (no SSA return)").failed()) {
      signalPassFailure();
    }

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
        loc, air::AsyncTokenType::get(op->getContext()),
        op->getResults().getType(), deps);
    async_region->setAttr(
        "id", mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32), ++ExecuteOpID));

    // Insert op to the new async execute region's body.
    Block *async_region_bb = builder.createBlock(&async_region.getBody());
    builder.setInsertionPointToStart(async_region_bb);
    auto op_cloned = builder.clone(*op);
    builder.create<xilinx::air::ExecuteTerminatorOp>(builder.getUnknownLoc(),
                                                     op_cloned->getResults());
    SmallVector<Value, 1> returnVals;
    for (auto val : async_region.getResults()) {
      returnVals.push_back(val);
    }
    op->replaceAllUsesWith(returnVals);

    // Create a vertex out of the current async execute region
    auto v = asyncExecuteGraph.addVertex();
    asyncExecuteGraph[v].asyncEventName = asyncEventName;
    asyncExecuteGraph[v].asyncEventType = "execute";
    asyncExecuteGraph[v].color = "chartreuse";
    asyncExecuteGraph[v].shape = "oval";
    asyncExecuteGraph[v].operationId = ExecuteOpID;

    // Update op-to-graph map
    region_to_g[async_region.getId()] = v;

    // Erase op
    if (eraseOpWithCheck(op, "createAsyncExecute (one SSA return)").failed()) {
      signalPassFailure();
    }
    return async_region;
  }

  // Re-instantiate the dmamemcpy2d op with async interface; update graph
  void createAsyncDMA(OpBuilder &builder, Operation *op) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyNdOp>(op);
    unsigned id = dma_op.getId();
    air::DmaMemcpyNdOp new_dmaNd_op = builder.create<air::DmaMemcpyNdOp>(
        loc, air::AsyncTokenType::get(dma_op->getContext()), deps,
        dma_op.getDstMemref(), dma_op.getDstOffsets(), dma_op.getDstSizes(),
        dma_op.getDstStrides(), dma_op.getSrcMemref(), dma_op.getSrcOffsets(),
        dma_op.getSrcSizes(), dma_op.getSrcStrides());
    new_dmaNd_op->setAttr(
        "id", mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32), id));

    // Create a vertex out of the current dmamemcpy2d op
    auto v = asyncExecuteGraph.addVertex();
    asyncExecuteGraph[v].asyncEventName = "air::dmaNd";
    asyncExecuteGraph[v].asyncEventType = "dma";
    asyncExecuteGraph[v].color = "cyan";
    asyncExecuteGraph[v].shape = "oval";
    asyncExecuteGraph[v].operationId = id;

    // Update op-to-graph map
    dma_to_g[id] = v;

    // Erase op
    if (eraseOpWithCheck(op, "createAsyncDMA").failed()) {
      signalPassFailure();
    }
  }

  // Re-instantiate the channel op with async interface; update graph
  void createAsyncChannel(OpBuilder &builder, Operation *op,
                          uint64_t &ChannelOpID) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    std::string event_name = "";
    if (auto channel_put_op = dyn_cast<air::ChannelPutOp>(op)) {
      air::ChannelPutOp new_channel_put_op = builder.create<air::ChannelPutOp>(
          loc, air::AsyncTokenType::get(channel_put_op->getContext()), deps,
          channel_put_op.getChanName(), channel_put_op.getIndices(),
          channel_put_op.getSrc(), channel_put_op.getSrcOffsets(),
          channel_put_op.getSrcSizes(), channel_put_op.getSrcStrides());
      new_channel_put_op->setAttr(
          "id",
          mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                                 ++ChannelOpID));
      event_name = "Put";
    } else if (auto channel_get_op = dyn_cast<air::ChannelGetOp>(op)) {
      air::ChannelGetOp new_channel_get_op = builder.create<air::ChannelGetOp>(
          loc, air::AsyncTokenType::get(channel_get_op->getContext()), deps,
          channel_get_op.getChanName(), channel_get_op.getIndices(),
          channel_get_op.getDst(), channel_get_op.getDstOffsets(),
          channel_get_op.getDstSizes(), channel_get_op.getDstStrides());
      new_channel_get_op->setAttr(
          "id",
          mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                                 ++ChannelOpID));
      event_name = "Get";
    } else
      op->emitOpError("unknown air channel op");

    // Create a vertex out of the current channel op
    auto v = asyncExecuteGraph.addVertex();
    asyncExecuteGraph[v].asyncEventName = "air::Channel" + event_name;
    asyncExecuteGraph[v].asyncEventType = "channel";
    asyncExecuteGraph[v].color = "cyan";
    asyncExecuteGraph[v].shape = "oval";
    asyncExecuteGraph[v].operationId = ChannelOpID;

    // Update op-to-graph map
    channel_to_g[ChannelOpID] = v;

    // Erase op
    if (eraseOpWithCheck(op, "createAsyncChannel").failed()) {
      signalPassFailure();
    }
  }

  // Re-instantiate the hierarchy op with async interface; update graph
  air::HierarchyInterface createAsyncHierarchyImpls(OpBuilder &builder,
                                                    air::HierarchyInterface op,
                                                    uint64_t &HierarchyOpID) {
    builder.setInsertionPoint(op);
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
      // Create a vertex out of the current hierarchy op
      auto v = asyncExecuteGraph.addVertex();
      asyncExecuteGraph[v].asyncEventName = "air::launch";
      asyncExecuteGraph[v].asyncEventType = "hierarchy";
      asyncExecuteGraph[v].color = "yellow";
      asyncExecuteGraph[v].shape = "box";
      asyncExecuteGraph[v].operationId = HierarchyOpID;
      // Update op-to-graph map
      hier_to_g[HierarchyOpID] = v;
    } else if (auto segment = dyn_cast<air::SegmentOp>(op.getOperation())) {
      auto new_segment = createAsyncHierarchy<air::SegmentOp>(
          builder, segment, HierarchyOpID, deps, args, constants);
      new_op = new_segment.getOperation();
      // Create a vertex out of the current hierarchy op
      auto v = asyncExecuteGraph.addVertex();
      asyncExecuteGraph[v].asyncEventName = "air::segment";
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
      // Create a vertex out of the current hierarchy op
      auto v = asyncExecuteGraph.addVertex();
      asyncExecuteGraph[v].asyncEventName = "air::herd";
      asyncExecuteGraph[v].asyncEventType = "hierarchy";
      asyncExecuteGraph[v].color = "yellow";
      asyncExecuteGraph[v].shape = "box";
      asyncExecuteGraph[v].operationId = HierarchyOpID;
      // Update op-to-graph map
      hier_to_g[HierarchyOpID] = v;
    } else {
      op->emitOpError("unknown hierarchy operation");
    }
    auto new_hier = dyn_cast<air::HierarchyInterface>(new_op);

    // Erase op
    if (eraseOpWithCheck(op, "createAsyncHierarchyImpls").failed()) {
      signalPassFailure();
    }
    return new_hier;
  }

  template <typename T>
  T createAsyncHierarchy(OpBuilder &builder, T op, uint64_t &OpID,
                         SmallVector<Value, 1> deps, SmallVector<Value, 4> args,
                         SmallVector<Value, 4> constants) {
    auto loc = op->getLoc();
    T new_op = builder.create<T>(loc, deps, op.getSizeOperands(), args, true,
                                 op->getAttrs());
    new_op->setAttr("id",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 32), ++OpID));

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
        addAsyncDepToGraphIfNew<T>(defop.getResult(0), op);
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
    if (!llvm::isa<MemRefType>(operand.getType())) {
      operand.getDefiningOp()->emitOpError(
          "operand being traced is not a memref");
    }
    bool foundWriteAccess = false;
    bool foundReadAccess = false;
    for (auto &u : operand.getUses()) {
      // If used in MemcpyInterface Op
      if (auto memcpy = dyn_cast<xilinx::air::MemcpyInterface>(u.getOwner())) {
        if (u.is(memcpy.getSrcMemref())) {
          foundReadAccess = true;
        } else if (u.is(memcpy.getDstMemref())) {
          foundWriteAccess = true;
        } else {
          memcpy->emitOpError("unknown operand in air::MemcpyInterface");
        }
      }
      // If used in a linalg op
      else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(u.getOwner())) {
        if (u.getOperandNumber() <
            linalgop.getNumDpsInputs() + linalgop.getNumDpsInits()) {
          foundReadAccess = true;
        } else if (u.getOperandNumber() >= linalgop.getNumDpsInputs() &&
                   u.getOperandNumber() - linalgop.getNumDpsInputs() <
                       linalgop.getNumDpsInits()) {
          foundWriteAccess = true;
        } else {
          linalgop->emitOpError("unknown operand in linalg op");
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
    if (!llvm::isa<MemRefType>(operand.getType())) {
      operand.getDefiningOp()->emitOpError(
          "operand being traced is not a memref");
    }
    for (auto &u : operand.getUses()) {
      // If used in MemcpyInterface Op
      if (auto memcpy = dyn_cast<xilinx::air::MemcpyInterface>(u.getOwner())) {
        bool isUsedAboveThisLine = false;
        if (auto dma = dyn_cast<xilinx::air::DmaMemcpyNdOp>(u.getOwner())) {
          if (foundAsyncOpUsesAboveCurrentLine(
                  &dma)) { // If this use is above current line
            isUsedAboveThisLine = true;
          }
        } else if (auto channel =
                       dyn_cast<xilinx::air::ChannelInterface>(u.getOwner())) {
          if (foundAsyncOpUsesAboveCurrentLine(
                  &channel)) { // If this use is above current line
            isUsedAboveThisLine = true;
          }
        }

        if (isUsedAboveThisLine) {
          partialMemref memcpy_src, memcpy_dst;
          if (memcpy.getSrcMemref()) {
            SmallVector<Value, 2> src_indices;
            unsigned numDimsSrc =
                llvm::cast<MemRefType>(memcpy.getSrcMemref().getType())
                    .getRank();
            if (memcpy.getSrcOffsets().size()) {
              numDimsSrc = memcpy.getSrcOffsets().size();
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(memcpy.getSrcOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsSrc; i++) {
                src_indices.push_back(nullptr);
              }
            }
            memcpy_src = createPartialMemref(memcpy.getSrcMemref(), numDimsSrc,
                                             src_indices);
          }
          if (memcpy.getDstMemref()) {
            unsigned numDimsDst =
                llvm::cast<MemRefType>(memcpy.getDstMemref().getType())
                    .getRank();
            SmallVector<Value, 2> dst_indices;
            if (memcpy.getDstOffsets().size()) {
              numDimsDst = memcpy.getDstOffsets().size();
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(memcpy.getDstOffsets()[i]);
              }
            } else {
              for (unsigned i = 0; i < numDimsDst; i++) {
                dst_indices.push_back(nullptr);
              }
            }
            memcpy_dst = createPartialMemref(memcpy.getDstMemref(), numDimsDst,
                                             dst_indices);
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
              addAsyncDepToGraphIfNew<T>(memcpy.getOperation()->getResult(0),
                                         op);
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
      }

      // If used in a linalg op
      else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(u.getOwner())) {
        if (auto ar =
                dyn_cast<xilinx::air::ExecuteOp>(linalgop->getParentOp())) {
          if (foundAsyncOpUsesAboveCurrentLine(&ar)) {
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
                addAsyncDepToGraphIfNew<T>(hier->getResult(0), op);
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
            addAsyncDepToGraphIfNew<T>(ar.getResult(0), op);
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
      sink_air_op->emitOpError("unknown dependency type");

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
        partialMemref subview_tile =
            createPartialMemref(subview.getSource(), subview.getSizes().size(),
                                subview.getOffsets());
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

  template <typename T>
  air::WaitAllOp insertWaitAllOpBeforeLoopYield(
      OpBuilder &builder, T loop_op,
      SmallVector<Value, 1> yielded_tokens_in_loop_op) {
    // Create one wait_all event at the end of current loop body.
    // Output token of wait_all shall be yielded
    auto loop_op_terminator = loop_op.getBody()->getTerminator();
    builder.setInsertionPoint(loop_op_terminator);
    air::WaitAllOp wait_all_op_yielded = builder.create<xilinx::air::WaitAllOp>(
        builder.getUnknownLoc(),
        air::AsyncTokenType::get(loop_op->getContext()),
        yielded_tokens_in_loop_op);
    wait_all_op_yielded->setAttr(
        "id",
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(loop_op->getContext(), 32), ++WaitAllOpID));
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
    air::WaitAllOp wait_all_op_before_loop =
        builder.create<xilinx::air::WaitAllOp>(
            builder.getUnknownLoc(),
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
    for (auto async_op : new_loop_op.getOps<air::AsyncOpInterface>()) {
      if (!isNotLoopCarriedOp(async_op)) {
        addAsyncDependencyIfNew(async_op, new_loop_op.getRegionIterArgs()[0]);
      }
    }

    return new_loop_op;
  }

  scf::ParallelOp
  replaceLoopOpWithNewTerminator(OpBuilder &builder, scf::ParallelOp loop_op,
                                 air::WaitAllOp wait_all_op_before_loop,
                                 SmallVector<Value, 4> incoming_tokens,
                                 SmallVector<Value, 4> constants) {
    if (loop_op.getNumReductions() != 0)
      loop_op->emitOpError(
          "currently only supporting input scf::ParallelOp with no reductions");

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

  // Elevating tokens from inside loop body to the yielded token, to maintain
  // legal domination T: loop type (scf::ForOp or scf::ParallelOp) U: source op
  // type
  template <typename T, typename U>
  void elevateAsyncTokens(T new_loop_op, ExecuteGraph::VertexId wait_all_op) {
    for (auto source : new_loop_op.template getOps<U>()) {
      SmallPtrSet<Operation *, 1> keep;
      if (source->getResult(0)) {
        for (auto sink : source->getResult(0).getUsers()) {
          // Keep token if source already dominates sink
          if (source->getParentOp()->isAncestor(sink)) {
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
                             SmallVector<Value, 1> yielded_tokens_in_loop_op) {
    // (1) Create one wait_all event at the end of current for loop body.
    air::WaitAllOp wait_all_op_yielded =
        insertWaitAllOpBeforeLoopYield<scf::ForOp>(builder, loop_op,
                                                   yielded_tokens_in_loop_op);

    // Update graph
    ExecuteGraph::VertexId wait_all_op_yielded_v =
        addVertexWaitAllOpBeforeLoopYield(yielded_tokens_in_loop_op, "for");
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

    if (eraseOpWithCheck(loop_op, "insertLoopCarriedDeps").failed()) {
      signalPassFailure();
    }
    loop_op = new_loop_op;
  }

  void insertLoopCarriedDeps(OpBuilder &builder, scf::ParallelOp &loop_op,
                             SmallVector<Value, 1> yielded_tokens_in_loop_op) {
    // (1) Create one wait_all event at the end of current parallel loop body.
    air::WaitAllOp wait_all_op_yielded =
        insertWaitAllOpBeforeLoopYield<scf::ParallelOp>(
            builder, loop_op, yielded_tokens_in_loop_op);

    // Update graph
    ExecuteGraph::VertexId wait_all_op_yielded_v =
        addVertexWaitAllOpBeforeLoopYield(yielded_tokens_in_loop_op,
                                          "parallel");
    // Update op-to-graph map for wait_all ops
    wa_to_g[wait_all_op_yielded.getId()] = wait_all_op_yielded_v;

    // (2) Create a new wait_all event before the parallel op which collects
    // the incoming deps.
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
      if (eraseOpWithCheck(y_op, "insertLoopCarriedDeps").failed())
        signalPassFailure();

    // Create scf::ReduceOp
    builder.setInsertionPointToEnd(new_loop_op.getBody());
    air::createSCFReduceForAsyncSCFParallel(builder, new_loop_op.getLoc(),
                                            wait_all_op_yielded.getAsyncToken(),
                                            loop_op->getContext());

    // Elevating tokens from inside forOp body to the yielded token, to maintain
    // dominance
    elevateAsyncTokens<scf::ParallelOp, air::AsyncOpInterface>(
        new_loop_op, wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ParallelOp, scf::ForOp>(new_loop_op,
                                                    wait_all_op_yielded_v);
    elevateAsyncTokens<scf::ParallelOp, scf::ParallelOp>(new_loop_op,
                                                         wait_all_op_yielded_v);

    if (eraseOpWithCheck(loop_op, "insertLoopCarriedDeps 2").failed())
      signalPassFailure();
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
  air::ExecuteOp getExecuteOpFromVertex(ExecuteGraph::VertexId v,
                                        ExecuteGraph g) {
    assert(g[v].asyncEventType == "execute" &&
           "This vertex is not a ExecuteOp");
    return async_execute_op_history[g[v].operationId - 1];
  }
  air::DmaMemcpyNdOp getDmaOpFromVertex(ExecuteGraph::VertexId v,
                                        ExecuteGraph g) {
    assert(g[v].asyncEventType == "dma" && "This vertex is not a DmaMemcpy op");
    return dma_op_history[g[v].operationId - 1];
  }
  air::ChannelInterface getChannelOpFromVertex(ExecuteGraph::VertexId v,
                                               ExecuteGraph g) {
    assert(g[v].asyncEventType == "channel" &&
           "This vertex is not a Channel op");
    return channel_op_history[g[v].operationId - 1];
  }
  air::HierarchyInterface getHierOpFromVertex(ExecuteGraph::VertexId v,
                                              ExecuteGraph g) {
    assert(g[v].asyncEventType == "hierarchy" &&
           "This vertex is not a Hierarchy op");
    return hier_op_history[g[v].operationId - 1];
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
  template <typename T> void fillAIRDepListUsingGraphTR(T op) {
    if (auto async_op =
            mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op.getOperation())) {
      uint64_t dstTRVertex = getGraphGVertexFromAIROp(op);
      for (auto TRVertex :
           asyncExecuteGraph.inverseAdjacentVertices(dstTRVertex)) {
        if (asyncExecuteGraph[TRVertex].asyncEventType == "execute")
          async_op.addAsyncDependency(
              getExecuteOpFromVertex(TRVertex, asyncExecuteGraph).getResult(0));
        else if (asyncExecuteGraph[TRVertex].asyncEventType == "dma")
          async_op.addAsyncDependency(
              getDmaOpFromVertex(TRVertex, asyncExecuteGraph)
                  .getOperation()
                  ->getResult(0));
        else if (asyncExecuteGraph[TRVertex].asyncEventType == "channel")
          async_op.addAsyncDependency(
              getChannelOpFromVertex(TRVertex, asyncExecuteGraph)
                  .getOperation()
                  ->getResult(0));
        else if (asyncExecuteGraph[TRVertex].asyncEventType == "hierarchy")
          async_op.addAsyncDependency(
              getHierOpFromVertex(TRVertex, asyncExecuteGraph)
                  .getOperation()
                  ->getResult(0));
        else
          op->emitOpError("unknown async event type");
      }
    } else
      op->emitOpError("operation has no async interface");
  }

  template <typename T> void addAsyncDepToGraphIfNew(Value dep, T op) {
    if (auto async_op =
            mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op.getOperation())) {
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

  bool foundAsyncOpUsesAboveCurrentLine(air::ExecuteOp *op) {
    if (!async_execute_op_history.empty())
      for (auto &iter : async_execute_op_history)
        if (iter.getResult(0) == op->getResult(0))
          return true;
    return false;
  }

  bool foundAsyncOpUsesAboveCurrentLine(air::DmaMemcpyNdOp *op) {
    if (!dma_op_history.empty())
      for (auto &iter : dma_op_history)
        if (iter->getResult(0) == op->getOperation()->getResult(0))
          return true;
    return false;
  }

  bool foundAsyncOpUsesAboveCurrentLine(air::ChannelInterface *op) {
    if (!channel_op_history.empty())
      for (auto &iter : channel_op_history)
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
        if (!xilinx::air::areEqualIndices(tile_0->memrefIndices[i],
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
      auto &bb = exec_op.getBody().front();
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
          auto &bb = async_user.getBody().front();
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

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyPass() {
  return std::make_unique<AIRDependency>();
}

} // namespace air
} // namespace xilinx
