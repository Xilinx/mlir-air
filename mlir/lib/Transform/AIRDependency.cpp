// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependency.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const & e) {
  llvm_unreachable("boost exception");
}

// boost graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/transitive_reduction.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>

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

struct regionNode {
    std::string asyncEventName;
    std::string asyncEventType;
    unsigned operationId;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, regionNode> Graph;
typedef boost::graph_traits<Graph>::in_edge_iterator in_edge_iterator;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;

typedef std::map<Graph::vertex_descriptor, Graph::vertex_descriptor> vertex_map;
typedef std::map<unsigned, Graph::vertex_descriptor> operation_id_to_vertex_map;

static uint64_t RegionOpID;
static uint64_t HerdLaunchOpID;

class AIRDependency : public AIRDependencyBase<AIRDependency> {

public:
  AIRDependency() = default;
  AIRDependency(const AIRDependency &pass) {}

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder module_builder(module);

    RegionOpID = 0;
    HerdLaunchOpID = 0;

    // 1st traversal: create async ops with empty dep list.

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        // Create async interface for air.dmamemcpy ops
        if (mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op))
          createAsyncDMA(module_builder, op);

        // Create async region for linalg.matmul
        else if (dyn_cast<linalg::MatmulOp>(op))
          createAsyncRegion(module_builder, op, "linalg::matmul", RegionOpID);

        // Create async region for linalg.fill
        else if (dyn_cast<linalg::FillOp>(op))
          createAsyncRegion(module_builder, op, "linalg::fill", RegionOpID);

        // Create async region for linalg.copy
        else if (dyn_cast<linalg::CopyOp>(op))
          createAsyncRegion(module_builder, op, "linalg::copy", RegionOpID);

        // Create async region for linalg op
        else if (mlir::dyn_cast<linalg::LinalgOp>(op))
          createAsyncRegion(module_builder, op, "linalg::unknown", RegionOpID);

        // Create async region for memref.alloc
        else if (auto memalloc_op = dyn_cast<memref::AllocOp>(op))
          createAsyncRegion(module_builder, op, "memref::alloc", RegionOpID, memalloc_op.memref().getType());

        // Create async region for memref.alloc
        else if (auto memcast_op = dyn_cast<memref::CastOp>(op))
          createAsyncRegion(module_builder, op, "memref::cast", RegionOpID, memcast_op.dest().getType());

        // Create async region for memref.dealloc
        else if (dyn_cast<memref::DeallocOp>(op))
          createAsyncRegion(module_builder, op, "memref::dealloc", RegionOpID);

        // Create async region for arith.muli
        else if (auto arith_op = dyn_cast<arith::MulIOp>(op))
          createAsyncRegion(module_builder, op, "arith::muli", RegionOpID, arith_op.getResult().getType());

        // Create async region for arith.addi
        else if (auto arith_op = dyn_cast<arith::AddIOp>(op))
          createAsyncRegion(module_builder, op, "arith::addi", RegionOpID, arith_op.getResult().getType());

        // Create async region for affine.apply
        else if (auto apply_op = dyn_cast<mlir::AffineApplyOp>(op))
          createAsyncRegion(module_builder, op, "affine::apply", RegionOpID, apply_op.getResult().getType());

        // Create async region for air.herdlaunch.
        else if (auto hl_op = dyn_cast<air::HerdLaunchOp>(op))
          createAsyncHerdLaunch(module_builder, hl_op, HerdLaunchOpID);

        // Create async region for an unknown op which has memref or index-type operands
        else {
          bool isCandidateRegion = false;
          for (auto operand :  op->getOperands()){
            if (operand.getType().isa<MemRefType>() || operand.getType().isa<IndexType>()){
              isCandidateRegion = true;
            }
          }
          // No air region for loop ops
          if (mlir::dyn_cast<mlir::LoopLikeOpInterface>(op))
            isCandidateRegion = false;
          // No air region for subview ops
          if (mlir::dyn_cast<mlir::OffsetSizeAndStrideOpInterface>(op))
            isCandidateRegion = false;
          if (isCandidateRegion){
            if (op->getNumResults())
              createAsyncRegion(module_builder, op, "unknown", RegionOpID, op->getResults().front().getType());
            else
              createAsyncRegion(module_builder, op, "unknown", RegionOpID);
          }
        }
      });
    }

    // 2nd traversal: trace deps among async regions; build a boost dep graph.

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        Operation* sink_op = nullptr;
        if (auto async_region_op = dyn_cast<air::RegionOp>(op)) {
          for (auto &bb : async_region_op.body()){
            for (auto &child_op : bb.getOperations()){
              if (!dyn_cast<air::RegionTerminatorOp>(child_op)) sink_op = &child_op;
            }
          }
        }
        else if (mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)){
          sink_op = op;
        }
        else if (dyn_cast<air::HerdLaunchOp>(op)){
          sink_op = op;
        }
        else return;

        SmallVector<partialMemref, 1> sink_op_memref_reads;
        SmallVector<partialMemref, 1> sink_op_memref_writes;
        SmallVector<Value, 1> sink_op_scalar_ins;
        SmallVector<Value, 1> sink_op_scalar_outs;

        // If the sink op is linalg op
        if (auto sink_op_linalgop = dyn_cast<linalg::LinalgOp>(sink_op)){
          for (auto linalg_ins : sink_op_linalgop.inputs()){
            if (linalg_ins.getType().isa<MemRefType>()){
              unsigned memRefRank = linalg_ins.getType().cast<MemRefType>().getRank();
              partialMemref tile = createPartialMemref(linalg_ins, memRefRank);
              sink_op_memref_reads.push_back(tile);
            }
            else if (linalg_ins.getType().isa<IndexType>()){
              sink_op_scalar_ins.push_back(linalg_ins);
            }
          }
          for (auto linalg_outs : sink_op_linalgop.outputs()){
            if (linalg_outs.getType().isa<MemRefType>()){
              unsigned memRefRank = linalg_outs.getType().cast<MemRefType>().getRank();
              partialMemref tile = createPartialMemref(linalg_outs, memRefRank);
              sink_op_memref_reads.push_back(tile); // linalg op both reads and writes the output memref
              sink_op_memref_writes.push_back(tile);
            }
            else if (linalg_outs.getType().isa<IndexType>()){
              sink_op_scalar_ins.push_back(linalg_outs); // linalg op both reads and writes the output memref
              sink_op_scalar_outs.push_back(linalg_outs);
            }
          }
          if (sink_op_linalgop->getNumResults()){
            for (auto linalg_results : sink_op_linalgop->getResults()){
              if (linalg_results.getType().isa<MemRefType>()){
                unsigned memRefRank = linalg_results.getType().cast<MemRefType>().getRank();
                partialMemref tile = createPartialMemref(linalg_results, memRefRank);
                sink_op_memref_writes.push_back(tile);
              }
              else if (linalg_results.getType().isa<IndexType>()){
                sink_op_scalar_outs.push_back(linalg_results);
              }
            }
          }
        }
        
        // If the sink op is memref::dealloc
        if (auto sink_op_memdealloc = dyn_cast<memref::DeallocOp>(sink_op)){
          unsigned memRefRank = sink_op_memdealloc.memref().getType().cast<MemRefType>().getRank();
          partialMemref tile = createPartialMemref(sink_op_memdealloc.memref(), memRefRank);
          sink_op_memref_reads.push_back(tile);
          sink_op_memref_writes.push_back(tile); // dealloc erases (i.e. writes to) output memref
        }
        
        // If the sink op is an air::DmaMemcpy op
        else if (auto sink_op_dma = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(sink_op)){
          SmallVector<Value, 2> src_indices;
          SmallVector<Value, 2> dst_indices;
          unsigned numDimsSrc = sink_op_dma.getNumDims();
          unsigned numDimsDst = sink_op_dma.getNumDims();
          // air.dmamemcpynd op has unknown # of dims (thus numdims defaults to 0)
          if (numDimsSrc == 0)
            numDimsSrc = sink_op_dma.getSrcMemref().getType().cast<MemRefType>().getRank();
          if (numDimsDst == 0)
            numDimsDst = sink_op_dma.getDstMemref().getType().cast<MemRefType>().getRank();
          for (unsigned i = 0; i < numDimsSrc; i++){
            src_indices.push_back(sink_op_dma.getSrcMemrefDim(i));
            sink_op_scalar_ins.push_back(sink_op_dma.getSrcMemrefDim(i));
          }
          for (unsigned i = 0; i < numDimsDst; i++){
            dst_indices.push_back(sink_op_dma.getDstMemrefDim(i));
            sink_op_scalar_outs.push_back(sink_op_dma.getDstMemrefDim(i));
          }
          partialMemref tile_in = createPartialMemref(sink_op_dma.getSrcMemref(), numDimsSrc, src_indices);
          sink_op_memref_reads.push_back(tile_in);
          partialMemref tile_out = createPartialMemref(sink_op_dma.getDstMemref(), numDimsDst, dst_indices);
          sink_op_memref_writes.push_back(tile_out);
          // Special case with ND DMA op
          if (auto sink_op_nddma = dyn_cast<air::DmaMemcpyNdOp>(sink_op)){
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
          }
        }
        
        // If the sink op is arith::MulIOp
        else if (auto sink_op_arith = dyn_cast<arith::MulIOp>(sink_op)){
          sink_op_scalar_ins.push_back(sink_op_arith.getLhs());
          sink_op_scalar_ins.push_back(sink_op_arith.getRhs());
          sink_op_scalar_outs.push_back(sink_op_arith.getResult());
        }
        
        // If the sink op is arith::AddIOp
        else if (auto sink_op_arith = dyn_cast<arith::AddIOp>(sink_op)){
          sink_op_scalar_ins.push_back(sink_op_arith.getLhs());
          sink_op_scalar_ins.push_back(sink_op_arith.getRhs());
          sink_op_scalar_outs.push_back(sink_op_arith.getResult());
        }
        
        // If the sink op is mlir::AffineApplyOp
        else if (auto sink_op_apply = dyn_cast<mlir::AffineApplyOp>(sink_op)){
          for (auto applyop_operand : sink_op_apply.getMapOperands()){
            sink_op_scalar_ins.push_back(applyop_operand);
          }
          sink_op_scalar_outs.push_back(sink_op_apply.getResult());
        }

        // If the sink op is an unknown op
        else {
          for (auto sink_op_op : sink_op->getOperands()){
            if (sink_op_op.getType().isa<MemRefType>()){
              unsigned memRefRank = sink_op_op.getType().cast<MemRefType>().getRank();
              partialMemref tile = createPartialMemref(sink_op_op, memRefRank);
              sink_op_memref_reads.push_back(tile); // Assuming all operands are both read and written to
              sink_op_memref_writes.push_back(tile);
            }
            else if (sink_op_op.getType().isa<IndexType>()){
              sink_op_scalar_ins.push_back(sink_op_op); // Assuming all operands are both read and written to
              sink_op_scalar_outs.push_back(sink_op_op);
            }
          }
          if (sink_op->getNumResults()){
            for (auto sink_op_results : sink_op->getResults()){
              if (sink_op_results.getType().isa<MemRefType>()){
                unsigned memRefRank = sink_op_results.getType().cast<MemRefType>().getRank();
                partialMemref tile = createPartialMemref(sink_op_results, memRefRank);
                sink_op_memref_writes.push_back(tile);
              }
              else if (sink_op_results.getType().isa<IndexType>()){
                sink_op_scalar_outs.push_back(sink_op_results);
              }
            }
          }

        }


        // Detect dependencies
        if (auto async_region_op = dyn_cast<air::RegionOp>(op)){
          // Detect RAW deps
          traceDeps<air::RegionOp>(sink_op_memref_reads, async_region_op, "RAW");
          // Detect WAW and WAR deps
          traceDeps<air::RegionOp>(sink_op_memref_writes, async_region_op, "WAW/WAR");
          // Detect tile index deps
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes, sink_op_scalar_ins, sink_op_scalar_outs, async_region_op);
          // Keep track of processed async region ops. Deps should point to the past, not future.
          async_region_op_history.push_back(async_region_op);
        }
        else if (auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)){
          traceDeps<air::DmaMemcpyInterface>(sink_op_memref_reads, dma_op, "RAW");
          traceDeps<air::DmaMemcpyInterface>(sink_op_memref_writes, dma_op, "WAW/WAR");
          traceTileIndices(sink_op_memref_reads, sink_op_memref_writes, sink_op_scalar_ins, sink_op_scalar_outs, dma_op);
          dma_op_history.push_back(dma_op);
        }
        else if (auto hl_op = dyn_cast<air::HerdLaunchOp>(op)){
          hl_op_history.push_back(hl_op);
        }
      });
    }

    // 3rd traversal: perform transitive reduction on dependency graph.

    std::vector<size_t> id_map(num_vertices(asyncRegionGraph));
    std::iota(id_map.begin(), id_map.end(), 0u);

    transitive_reduction(asyncRegionGraph, asyncRegionGraphTR, make_assoc_property_map(g_to_tr), id_map.data());

    for (vertex_map::iterator i = g_to_tr.begin(); i != g_to_tr.end(); ++i){
      // Copy over graph properties
      asyncRegionGraphTR[i->second].asyncEventName = asyncRegionGraph[i->first].asyncEventName;
      asyncRegionGraphTR[i->second].asyncEventType = asyncRegionGraph[i->first].asyncEventType;
      asyncRegionGraphTR[i->second].operationId = asyncRegionGraph[i->first].operationId;
      // Build reverse map tr_to_g, for convenient vertex mapping
      tr_to_g[i->second] = i->first;
    }

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        // Fill dep list of air region ops
        if (auto async_region_op = dyn_cast<air::RegionOp>(op)) {
          fillAIRDepListUsingGraphTR<air::RegionOp>(async_region_op);
        }
        // Fill dep list of air dmamemcpy2d ops
        else if (auto dma_op = dyn_cast<air::DmaMemcpyInterface>(op)) {
          fillAIRDepListUsingGraphTR<air::DmaMemcpyInterface>(dma_op);
        }
        else if (auto hl_op = dyn_cast<air::HerdLaunchOp>(op)) {
          fillAIRDepListUsingGraphTR<air::HerdLaunchOp>(hl_op);
        }
      });
    }

    // 4th traversal: loop-carried deps.
    // Add wait_all events to collect sinks in loop bodies. Add iter_args to scp for loops representing loop-carried deps.

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto for_op = dyn_cast<scf::ForOp>(op)) {
          // Check for nested for loops
          unsigned number_of_nested_for_ops = getNumberOfNestedForOps(for_op);

          // Get async region in loop body
          bool hasAsyncRegionsInBody = false;
          SmallVector<Value, 1> sinks_in_for_op;
          for (auto async_region_op : for_op.getOps<air::RegionOp>()){
            hasAsyncRegionsInBody = true;
            // Get sinks of dep graph
            if (async_region_op.getResult(0).use_empty())
              sinks_in_for_op.push_back(async_region_op.getResult(0));
          }
          // Get async dma in loop body
          for (auto dma_op : for_op.getOps<air::DmaMemcpyInterface>()){
            hasAsyncRegionsInBody = true;
            // Get sinks of dep graph
            if (dma_op.getOperation()->getResult(0).use_empty())
              sinks_in_for_op.push_back(dma_op.getOperation()->getResult(0));
          }
          // Get async herd_launch in loop body
          for (auto hl_op : for_op.getOps<air::HerdLaunchOp>()){
            hasAsyncRegionsInBody = true;
            // Get sinks of dep graph
            if (hl_op.getResult(0).use_empty())
              sinks_in_for_op.push_back(hl_op.getResult(0));
          }
          if (hasAsyncRegionsInBody){
            // (1) Create one wait_all event at the end of current for loop body.
            auto for_op_terminator = for_op.getBody()->getTerminator();
            module_builder.setInsertionPoint(for_op_terminator);
            auto wait_all_op_yielded = module_builder.create<xilinx::air::WaitAllOp>(module_builder.getUnknownLoc(), air::AsyncTokenType::get(for_op->getContext()), sinks_in_for_op);

            // Update boost graph
            auto wait_all_op_yielded_v = add_vertex(asyncRegionGraphTR);
            asyncRegionGraphTR[wait_all_op_yielded_v].asyncEventName = to_string(number_of_nested_for_ops);
            asyncRegionGraphTR[wait_all_op_yielded_v].asyncEventName += "d_for_loop_end";
            asyncRegionGraphTR[wait_all_op_yielded_v].asyncEventType = "wait_all";
            asyncRegionGraphTR[wait_all_op_yielded_v].operationId = 0;
            for (auto sink : sinks_in_for_op){
              unsigned src_id = 0;
              if (auto async_region_op = dyn_cast<air::RegionOp>(sink.getDefiningOp())){
                src_id = getGraphGVertexFromAIROp(async_region_op);
              }
              else if (auto dma_op = dyn_cast<air::DmaMemcpyInterface>(sink.getDefiningOp())){
                src_id = getGraphGVertexFromAIROp(dma_op);
              }
              else if (auto hl_op = dyn_cast<air::HerdLaunchOp>(sink.getDefiningOp())){
                src_id = getGraphGVertexFromAIROp(hl_op);
              }
              auto src = g_to_tr[src_id];
              add_edge(src, wait_all_op_yielded_v, asyncRegionGraphTR);
            }

            // (2) Create a new wait_all event before the for op which collects the incoming deps.
            SmallVector<Value, 4> incoming_tokens;
            SmallVector<Value, 4> constants;
            llvm::SetVector<Value> region_args;
            getUsedValuesDefinedAbove(for_op.getRegion(), region_args);
            for (Value v : region_args) {
              if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
                constants.push_back(v);
              else if (v.getDefiningOp()){
                if (auto v_op = dyn_cast<air::RegionOp>(v.getDefiningOp())){
                  if (v_op.getAsyncToken() == v)
                    incoming_tokens.push_back(v);
                }
                else if (auto v_op = dyn_cast<air::DmaMemcpy2dOp>(v.getDefiningOp())){
                  if (v_op.getAsyncToken() == v)
                    incoming_tokens.push_back(v);
                }
                else if (auto v_op = dyn_cast<air::HerdLaunchOp>(v.getDefiningOp())){
                  if (v_op.getAsyncToken() == v)
                    incoming_tokens.push_back(v);
                }
              }
            }
            module_builder.setInsertionPoint(for_op);
            auto wait_all_op_before_loop = module_builder.create<xilinx::air::WaitAllOp>(module_builder.getUnknownLoc(), air::AsyncTokenType::get(for_op->getContext()), incoming_tokens);
            
            // Update boost graph
            auto wait_all_op_before_loop_v = add_vertex(asyncRegionGraphTR);
            asyncRegionGraphTR[wait_all_op_before_loop_v].asyncEventName = to_string(number_of_nested_for_ops);
            asyncRegionGraphTR[wait_all_op_before_loop_v].asyncEventName += "d_for_loop_begin";
            asyncRegionGraphTR[wait_all_op_before_loop_v].asyncEventType = "wait_all";
            asyncRegionGraphTR[wait_all_op_before_loop_v].operationId = 0;

            // (3) Create new for op with iter_args.
            SmallVector<Value, 4> merged_incoming_token;
            merged_incoming_token.push_back(wait_all_op_before_loop.getResult(0));
            auto new_for_op = module_builder.create<scf::ForOp>(for_op.getLoc(), for_op.getLowerBound(),
                                         for_op.getUpperBound(), for_op.getStep(), merged_incoming_token);

            if (auto attr = for_op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
              new_for_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

            // Splice the operations inside for op
            auto &bb = new_for_op.getBody()->getOperations();
            auto &body = for_op.getBody()->getOperations();
            bb.splice(bb.begin(), body,
                                      body.begin(), --body.end());

            auto iv = for_op.getInductionVar();
            iv.replaceAllUsesWith(new_for_op.getInductionVar());
            module_builder.setInsertionPointToStart(new_for_op.getBody());
            for (auto c : constants) {
              replaceAllUsesInRegionWith(c,
                                        module_builder.clone(*c.getDefiningOp())->getResult(0),
                                        new_for_op.getRegion());
            }
            for (Value v : incoming_tokens) {

              // Update boost graph
              for (auto user : v.getUsers()){
                unsigned src = 0;
                unsigned dst = 0;
                if (auto dst_op = dyn_cast<air::RegionOp>(user)){
                  dst = g_to_tr[getGraphGVertexFromAIROp(dst_op)];
                }
                else if (auto dst_op = mlir::dyn_cast<air::DmaMemcpyInterface>(user)){
                  dst = g_to_tr[getGraphGVertexFromAIROp(dst_op)];
                }
                else if (auto dst_op = dyn_cast<air::HerdLaunchOp>(user)){
                  dst = g_to_tr[getGraphGVertexFromAIROp(dst_op)];
                }
                if (auto src_op = dyn_cast<air::RegionOp>(v.getDefiningOp())){
                  src = g_to_tr[getGraphGVertexFromAIROp(src_op)];
                }
                else if (auto src_op = mlir::dyn_cast<air::DmaMemcpyInterface>(v.getDefiningOp())){
                  src = g_to_tr[getGraphGVertexFromAIROp(src_op)];
                }
                else if (auto src_op = dyn_cast<air::HerdLaunchOp>(v.getDefiningOp())){
                  src = g_to_tr[getGraphGVertexFromAIROp(src_op)];
                }
                if (edge(src, dst, asyncRegionGraphTR).second){ // if an edge exists
                  remove_edge(src, dst, asyncRegionGraphTR);
                  if (!edge(src, wait_all_op_before_loop_v, asyncRegionGraphTR).second)
                    add_edge(src, wait_all_op_before_loop_v, asyncRegionGraphTR);
                  if (!edge(wait_all_op_before_loop_v, dst, asyncRegionGraphTR).second)
                    add_edge(wait_all_op_before_loop_v, dst, asyncRegionGraphTR);
                }
              }

              replaceAllUsesInRegionWith(v,
                                        new_for_op.getRegionIterArgs()[0],
                                        new_for_op.getRegion());
            }

            // Connect sources in loop body with iter_args
            for (auto async_region_op : new_for_op.getOps<air::RegionOp>()){
              if (async_region_op.getAsyncDependencies().size() == 0){
                async_region_op.addAsyncDependency(new_for_op.getRegionIterArgs()[0]);
              }
            }
            for (auto dma_op : new_for_op.getOps<air::DmaMemcpyInterface>()){
              auto async_op = mlir::dyn_cast<air::AsyncOpInterface>(dma_op.getOperation());
              if (async_op.getAsyncDependencies().size() == 0){
                async_op.addAsyncDependency(new_for_op.getRegionIterArgs()[0]);
              }
            }
            for (auto hl_op : new_for_op.getOps<air::HerdLaunchOp>()){
              if (hl_op.getAsyncDependencies().size() == 0){
                hl_op.addAsyncDependency(new_for_op.getRegionIterArgs()[0]);
              }
            }

            // Yield an async token
            SmallVector<Value, 4> yield_token;
            yield_token.push_back(wait_all_op_yielded.getResult(0));
            module_builder.setInsertionPointToEnd(new_for_op.getBody());
            module_builder.create<scf::YieldOp>(new_for_op.getLoc(), yield_token);

            for_op.erase();
          }
        }
      });
    }

    // Dump graph
    dump_graph("out.dot");
  }

private:

  // Air async op history
  std::vector<air::RegionOp> async_region_op_history;
  std::vector<air::DmaMemcpyInterface> dma_op_history;
  std::vector<air::HerdLaunchOp> hl_op_history;

  struct partialMemref {
    Value memrefValue;
    unsigned numDims;
    SmallVector<Value, 2> memrefIndices;
  };

  partialMemref createPartialMemref(mlir::Value memrefValue, unsigned numDims){
    partialMemref tile;
    tile.memrefValue = memrefValue;
    tile.numDims = numDims;
    for (unsigned i = 0; i < numDims; i++){
      tile.memrefIndices.push_back(nullptr);
    }
    return tile;
  }

  partialMemref createPartialMemref(mlir::Value memrefValue, unsigned numDims, SmallVector<Value, 2> memrefIndices){
    partialMemref tile;
    tile.memrefValue = memrefValue;
    tile.numDims = numDims;
    for (unsigned i = 0; i < numDims; i++){
      tile.memrefIndices.push_back(memrefIndices[i]);
    }
    return tile;
  }

  
  // Create air region op with async interface (no ssa result returned); update graph
  air::RegionOp createAsyncRegion(OpBuilder &builder, Operation *op, std::string asyncEventName, uint64_t &RegionOpID){
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::RegionOp async_region;
    async_region = builder.create<xilinx::air::RegionOp>(loc, air::AsyncTokenType::get(op->getContext()), deps);
    async_region->setAttr("id",
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                    ++RegionOpID));

    // Insert op to the new async region's body.
    Block *async_region_bb = builder.createBlock(&async_region.body());
    builder.setInsertionPointToStart(async_region_bb);

    builder.clone(*op);
    builder.create<xilinx::air::RegionTerminatorOp>(builder.getUnknownLoc());

    // Create a vertex out of the current async region
    auto v = add_vertex(asyncRegionGraph);
    asyncRegionGraph[v].asyncEventName = asyncEventName;
    asyncRegionGraph[v].asyncEventType = "region";
    asyncRegionGraph[v].operationId = RegionOpID;

    // Update op-to-graph map
    region_to_g[async_region.getId()] = v;

    // Erase op
    op->erase();
    return async_region;
  }

  
  // Create air region op with async interface (with one ssa result returned); update graph
  air::RegionOp createAsyncRegion(OpBuilder &builder, Operation *op, std::string asyncEventName, uint64_t &RegionOpID, mlir::Type valueType){
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::RegionOp async_region;
      async_region = builder.create<xilinx::air::RegionOp>(loc, air::AsyncTokenType::get(op->getContext()), valueType, deps);
    async_region->setAttr("id",
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                    ++RegionOpID));

    // Insert op to the new async region's body.
    Block *async_region_bb = builder.createBlock(&async_region.body());
    builder.setInsertionPointToStart(async_region_bb);
    auto op_cloned = builder.clone(*op);
    builder.create<xilinx::air::RegionTerminatorOp>(builder.getUnknownLoc(), op_cloned->getResults().front());
    SmallVector<Value, 1> returnVals;
    returnVals.push_back(async_region.getResult(1));
    op->replaceAllUsesWith(returnVals);
    
    // Create a vertex out of the current async region
    auto v = add_vertex(asyncRegionGraph);
    asyncRegionGraph[v].asyncEventName = asyncEventName;
    asyncRegionGraph[v].asyncEventType = "region";
    asyncRegionGraph[v].operationId = RegionOpID;

    // Update op-to-graph map
    region_to_g[async_region.getId()] = v;

    // Erase op
    op->erase();
    return async_region;
  }

  // Re-instantiate the dmamemcpy2d op with async interface; update graph
  void createAsyncDMA(OpBuilder &builder, Operation *op){
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op);
    unsigned id = dma_op.getId();
    std::string event_name = "";
    if (auto dma_op = dyn_cast<air::DmaMemcpyOp>(op)){
      air::DmaMemcpyOp new_dma_op = builder.create<air::DmaMemcpyOp>(loc, air::AsyncTokenType::get(dma_op->getContext()), 
              deps, dma_op.getDstMemref(), dma_op.getSrcMemref(), dma_op.getDstMemrefDim(0), dma_op.getSrcMemrefDim(0), dma_op.getLength()); 
      new_dma_op->setAttr("id",
              mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
              id));
    }
    else if (auto dma2d_op = dyn_cast<air::DmaMemcpy2dOp>(op)){
      air::DmaMemcpy2dOp new_dma2d_op = builder.create<air::DmaMemcpy2dOp>(loc, air::AsyncTokenType::get(dma2d_op->getContext()), 
              deps, dma2d_op.getDstMemref(), dma2d_op.getSrcMemref(), dma2d_op.getDstMemrefDim(0), dma2d_op.getDstMemrefDim(1), 
              dma2d_op.getSrcMemrefDim(0), dma2d_op.getSrcMemrefDim(1), dma2d_op.getLength(), dma2d_op.getStride(), dma2d_op.getElemPerStride()); 
      new_dma2d_op->setAttr("id",
              mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
              id));
      event_name = "2d";
    }
    else if (auto dma4d_op = dyn_cast<air::DmaMemcpy4dOp>(op)){
      air::DmaMemcpy4dOp new_dma4d_op = builder.create<air::DmaMemcpy4dOp>(loc, air::AsyncTokenType::get(dma4d_op->getContext()), 
              deps, dma4d_op.getDstMemref(), dma4d_op.getSrcMemref(), dma4d_op.getDstMemrefDim(0), dma4d_op.getDstMemrefDim(1), dma4d_op.getDstMemrefDim(2), dma4d_op.getDstMemrefDim(3),
              dma4d_op.getSrcMemrefDim(0), dma4d_op.getSrcMemrefDim(1), dma4d_op.getSrcMemrefDim(2), dma4d_op.getSrcMemrefDim(3), dma4d_op.getLength(), dma4d_op.getStride(), dma4d_op.getElemPerStride()); 
      new_dma4d_op->setAttr("id",
              mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
              id));
      event_name = "4d";      
    }
    else if (auto dmaNd_op = dyn_cast<air::DmaMemcpyNdOp>(op)){
      air::DmaMemcpyNdOp new_dmaNd_op = builder.create<air::DmaMemcpyNdOp>(loc, air::AsyncTokenType::get(dmaNd_op->getContext()), 
              deps, dmaNd_op.getDstMemref(), dmaNd_op.getDstOffsets(), dmaNd_op.getDstSizes(), dmaNd_op.getDstStrides(), dmaNd_op.getSrcMemref(), dmaNd_op.getSrcOffsets(), dmaNd_op.getSrcSizes(), dmaNd_op.getSrcStrides()); 
      new_dmaNd_op->setAttr("id",
              mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
              id));
      event_name = "Nd";      
    }

    // Create a vertex out of the current dmamemcpy2d op
    auto v = add_vertex(asyncRegionGraph);
    asyncRegionGraph[v].asyncEventName = "air::dma" + event_name;
    asyncRegionGraph[v].asyncEventType = "dma";
    asyncRegionGraph[v].operationId = id;

    // Update op-to-graph map
    dma_to_g[id] = v;

    // Erase op
    op->erase();
  }

  // Re-instantiate the herd_launch op with async interface; update graph
  air::HerdLaunchOp createAsyncHerdLaunch(OpBuilder &builder, air::HerdLaunchOp op, uint64_t &HerdLaunchOpID){
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::HerdDim2 dims{op.getHerdSizeOperands().x, op.getHerdSizeOperands().y};
    SmallVector<Value, 4> args;
    SmallVector<Value, 4> constants;
    for (unsigned i = 0; i < op.getNumKernelOperands(); i++){
      auto v = op.getKernelOperand(i);
      if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
        constants.push_back(v);
      else
        args.push_back(v);
    }
    air::HerdLaunchOp new_launch = builder.create<air::HerdLaunchOp>(loc, deps, dims, args);
    new_launch->setAttr("id",
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
            ++HerdLaunchOpID));

    if (auto attr = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      new_launch->setAttr(SymbolTable::getSymbolAttrName(), attr);

    auto &bb = new_launch.body().front();
    auto ivs = op.getTileIds();
    ivs.x.replaceAllUsesWith(new_launch.getTileIds().x);
    ivs.y.replaceAllUsesWith(new_launch.getTileIds().y);
    auto &body = op.body().front().getOperations();
    bb.getOperations().splice(bb.begin(), body,
                              body.begin(), --body.end());
    builder.setInsertionPointToStart(&new_launch.getRegion().front());
    for (auto c : constants) {
      replaceAllUsesInRegionWith(c,
                                  builder.clone(*c.getDefiningOp())->getResult(0),
                                  new_launch.getRegion());
    }
    builder.setInsertionPointToEnd(&bb);
    builder.create<air::HerdTerminatorOp>(loc);

    int i = 0;
    auto old_kernel_args = op.getKernelArguments();
    auto new_kernel_args = new_launch.getKernelArguments();
    for (Value v : old_kernel_args)
      replaceAllUsesInRegionWith(v, new_kernel_args[i++], new_launch.getRegion());

    // Create a vertex out of the current herd_launch op
    auto v = add_vertex(asyncRegionGraph);
    asyncRegionGraph[v].asyncEventName = "air::herd_launch";
    asyncRegionGraph[v].asyncEventType = "herd_launch";
    asyncRegionGraph[v].operationId = HerdLaunchOpID;

    // Update op-to-graph map
    hl_to_g[HerdLaunchOpID] = v;

    // Erase op
    op->erase();
    return new_launch;
  }

  bool foundAsyncOpUsesAboveCurrentLine(air::RegionOp *op){
    if (!async_region_op_history.empty())
      for (auto &iter : async_region_op_history)
        if (iter.getResult(0) == op->getResult(0)) return true;
    return false;
  }

  bool foundAsyncOpUsesAboveCurrentLine(air::DmaMemcpyInterface *op){
    if (!dma_op_history.empty())
      for (auto &iter : dma_op_history)
        if (iter->getResult(0) == op->getOperation()->getResult(0)) return true;
    return false;
  }

  bool foundAsyncOpUsesAboveCurrentLine(air::HerdLaunchOp *op){
    if (!hl_op_history.empty())
      for (auto &iter : hl_op_history)
        if (iter.getResult(0) == op->getResult(0)) return true;
    return false;
  }
  
  // Check if operand is returned from RegionOp (memref.alloc)
  template <typename T>
  void pushDefiningOpAsDep(Value operand, T op){
    // Check memref deps
    if (auto defop = operand.getDefiningOp<air::RegionOp>()){
      if (foundAsyncOpUsesAboveCurrentLine(&defop)){
        addNewAsyncDepToGraph<T>(defop.getResult(0), op);
      }
    }
  }

  // Trace tile index deps
  template <typename T>
  void pushTileIndexAsDep(mlir::Value tile_index, T op){
    if (tile_index != nullptr){
    // If tile_index is not a nullptr
      // If created by async_region
      if (auto defop = tile_index.getDefiningOp<air::RegionOp>()){
        if (foundAsyncOpUsesAboveCurrentLine(&defop)){
          addNewAsyncDepToGraph<T>(defop.getResult(0), op);
        }
      }
      // If created by launch_herd (as loop iter)
      else if (auto lh = dyn_cast<air::HerdLaunchOp>(tile_index.getParentRegion()->getParentOp())){
        if (lh.getTileIds().x == tile_index || lh.getTileIds().y == tile_index){
          addNewAsyncDepToGraph<T>(tile_index, op);
        }
      }
      // If created by scf.for (as loop iter)
      else if (auto forloop = dyn_cast<scf::ForOp>(tile_index.getParentRegion()->getParentOp())){
        if (forloop.getInductionVar() == tile_index){
          addNewAsyncDepToGraph<T>(tile_index, op);
        }
      }
    }
  }

  // Check if two partial memref tiles have identical indices
  bool areEqualIndexPartialMemrefs(partialMemref *tile_0, partialMemref *tile_1){
    if (tile_0->numDims != tile_1->numDims){
      // Unequal # dimensions
      return false;
    }
    else{
      for (unsigned i = 0; i < tile_0->numDims; i++){
        if (!areEqualIndices(tile_0->memrefIndices[i], tile_1->memrefIndices[i]))
          return false;
      }
    }
    return true;
  }

  // Trace operand's uses at current scope
  template <typename T>
  void pushDepsAtCurrentScope(mlir::Value operand, T op, char rw = 'n', partialMemref *tile = nullptr){
    assert(operand.getType().isa<MemRefType>() && "operand being traced is not a memref");
    for (auto &u : operand.getUses()){
      // If used in DmaMemcpy Op
      if (auto dma = dyn_cast<xilinx::air::DmaMemcpyInterface>(u.getOwner())){
        if (foundAsyncOpUsesAboveCurrentLine(&dma)){ // If this use is above current line
          // DMA2D: Need to check for overlapping partial memrefs in use
          unsigned numDimsSrc = dma.getNumDims();
          unsigned numDimsDst = dma.getNumDims();
          if (numDimsSrc == 0)
            numDimsSrc = dma.getSrcMemref().getType().cast<MemRefType>().getRank();
          if (numDimsDst == 0)
            numDimsDst = dma.getDstMemref().getType().cast<MemRefType>().getRank();
          SmallVector<Value, 2> src_indices;
          SmallVector<Value, 2> dst_indices;
          for (unsigned i = 0; i < numDimsSrc; i++){
            src_indices.push_back(dma.getSrcMemrefDim(i));
          }
          for (unsigned i = 0; i < numDimsDst; i++){
            dst_indices.push_back(dma.getDstMemrefDim(i));
          }
          partialMemref dma_src = createPartialMemref(dma.getSrcMemref(), numDimsSrc, src_indices);
          partialMemref dma_dst = createPartialMemref(dma.getDstMemref(), numDimsDst, dst_indices);
          
          if (rw == 'r'){
            if (u.is(dma.getSrcMemref())){
              if (tile == nullptr){
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
              }
              else if (areEqualIndexPartialMemrefs(tile, &dma_src))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
          }
          else if (rw == 'w'){
            if (u.is(dma.getDstMemref())){
              if (tile == nullptr){
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
              }
              else if (areEqualIndexPartialMemrefs(tile, &dma_dst))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
          }
          else{
            if (tile == nullptr) {
              addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
            else if (u.is(dma.getDstMemref())){
              if (areEqualIndexPartialMemrefs(tile, &dma_dst))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
            else if (u.is(dma.getSrcMemref())){
              if (areEqualIndexPartialMemrefs(tile, &dma_src))
                addNewAsyncDepToGraph<T>(dma.getOperation()->getResult(0), op);
            }
          }
        }
      }

      // If used in a linalg op
      else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(u.getOwner())){
        if (auto ar = dyn_cast<xilinx::air::RegionOp>(linalgop->getParentOp())){
          if (foundAsyncOpUsesAboveCurrentLine(&ar)){
            if (rw == 'r'){
              if (u.getOperandNumber() < linalgop.getNumInputs() + linalgop.getNumOutputs())
                addNewAsyncDepToGraph<T>(ar.getResult(0), op);
            }
            else if (rw == 'w'){
              if (u.getOperandNumber() >= linalgop.getNumInputs() && u.getOperandNumber() - linalgop.getNumInputs() < linalgop.getNumOutputs())
                addNewAsyncDepToGraph<T>(ar.getResult(0), op);
            }
            else{
              addNewAsyncDepToGraph<T>(ar.getResult(0), op);
            }
          }
        }
      }

      // If used in an unknown op
      else{
        auto unknownop = u.getOwner();
        if (auto ar = dyn_cast<xilinx::air::RegionOp>(unknownop->getParentOp())){
          if (foundAsyncOpUsesAboveCurrentLine(&ar)){
            addNewAsyncDepToGraph<T>(ar.getResult(0), op);
          }
        }

      }
    }
  }


  template <typename T>
  void traceDeps(SmallVector<partialMemref, 1> operands, T sink_air_op, std::string dep_type){

    char dep_tracing_mode = 'n';
    if (dep_type == "RAW")
      dep_tracing_mode = 'w';
    else if (dep_type == "WAW/WAR")
      dep_tracing_mode = 'n';
    else assert(false && "Unknown dependency type");

    // Detect RAW deps
    for (auto operand : operands) {
      // Trace the defining op of sink op, RAW
      pushDefiningOpAsDep<T>(operand.memrefValue, sink_air_op);

      // If sink op and operand's use are under the same scope
      pushDepsAtCurrentScope<T>(operand.memrefValue, sink_air_op, dep_tracing_mode, &operand);

      // If sink op is in HerdLaunchOp
      if (auto lh = sink_air_op->template getParentOfType<xilinx::air::HerdLaunchOp>()){
        // Search for dma deps outside (before) HerdLaunchOp
        for (unsigned lh_operand_id = 0; lh_operand_id < lh.getNumKernelOperands(); lh_operand_id++){
          if (lh.getKernelArguments()[lh_operand_id] == operand.memrefValue){
            auto ancestor_op = lh.getKernelOperand(lh_operand_id);
            pushDepsAtCurrentScope<air::HerdLaunchOp>(ancestor_op, lh, dep_tracing_mode);
          }
        }
      }

      // If sink op is outside HerdLaunchOp
      else {
        // If the input memref was used inside HerdLaunchOp in the past
        for (auto &u : operand.memrefValue.getUses()){
          if (auto lh = dyn_cast<xilinx::air::HerdLaunchOp>(u.getOwner())){
            if (foundAsyncOpUsesAboveCurrentLine(&lh)){
              addNewAsyncDepToGraph<T>(lh.getResult(0), sink_air_op);
            }
          }
        }
      }
    }
  }

  template <typename T>
  void traceTileIndices(SmallVector<partialMemref, 1> read_operands, SmallVector<partialMemref, 1> write_operands, SmallVector<Value, 1> in_scalars, SmallVector<Value, 1> out_scalars, T sink_air_op){
    for (auto operand : read_operands) {
      for (unsigned i = 0; i < operand.numDims; i++){
        pushTileIndexAsDep<T>(operand.memrefIndices[i], sink_air_op);
      }
    }
    for (auto operand : write_operands) {
      for (unsigned i = 0; i < operand.numDims; i++){
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

  bool areEqualIndices (mlir::Value index_0, mlir::Value index_1){
    if (index_0 == nullptr || index_1 == nullptr) {
      // Note: memref with index is subset to memref without index (i.e. the entire memref)
      return true;
    }
    else {
      if (index_0 == index_1) return true;
      else if (!index_0.getDefiningOp()) return false;
      else if (!index_1.getDefiningOp()) return false;
      else {
        auto index_0_const_op = dyn_cast<arith::ConstantOp>(index_0.getDefiningOp());
        auto index_1_const_op = dyn_cast<arith::ConstantOp>(index_1.getDefiningOp());
        if (index_0_const_op.getValue() == index_1_const_op.getValue()) return true;
        else return false;
      }
    }
  }

  template <typename T>
  void addNewAsyncDepToGraph(Value dep, T op){
    if (auto async_op = mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op.getOperation())){
      for (auto old_dep : async_op.getAsyncDependencies())
        if (old_dep == dep) return;

      // Add edge to boost graph, iff dep is async region (i.e. not a loop iterator)
      if (auto srcOp = dep.getDefiningOp()) {
        uint64_t srcNode;
        if (auto region_op = dyn_cast<air::RegionOp>(srcOp)){
          srcNode = getGraphGVertexFromAIROp(region_op);
        }
        else if (auto dma_op = dyn_cast<air::DmaMemcpyInterface>(srcOp)){
          srcNode = getGraphGVertexFromAIROp(dma_op);
        }
        else if (auto hl_op = dyn_cast<air::HerdLaunchOp>(srcOp)){
          srcNode = getGraphGVertexFromAIROp(hl_op);
        }
        else assert(false && "dependency token should be generated by an async op");
        uint64_t dstNode = getGraphGVertexFromAIROp(op);
        add_edge(srcNode, dstNode, asyncRegionGraph);
      }
    }
    else assert(false && "Operation has no async interface");
  }

  // Check if current for op is the single child in a parent for op
  bool isSingleChildInParentForOp(scf::ForOp child_for_op){
    if (!child_for_op->getParentOp())
      return false;
    if (!dyn_cast<scf::ForOp>(child_for_op->getParentOp()))
      return false;
    auto parent_op = dyn_cast<scf::ForOp>(child_for_op->getParentOp());
    if (parent_op.getBody()->getOperations().size() == 2) return true; // child for op plus terminator
    else return false;
  }

  // Count for loop nest dimensions
  unsigned getNumberOfNestedForOps(scf::ForOp for_op){
    unsigned number_of_nested_for_ops = 1;
    scf::ForOp parent_for_op;
    scf::ForOp currnet_for_op = for_op;
    while (isSingleChildInParentForOp(currnet_for_op)){
      number_of_nested_for_ops ++;
      auto parent_op = currnet_for_op->getParentOp();
      if (dyn_cast<scf::ForOp>(parent_op))
        currnet_for_op = dyn_cast<scf::ForOp>(parent_op);
    }
    return number_of_nested_for_ops;
  }

  // Dependency graph constructed as Boost graph
  Graph asyncRegionGraph;
  Graph asyncRegionGraphTR;
  vertex_map g_to_tr, tr_to_g; // Map between graph g and graph tr (post-tr graph)
  operation_id_to_vertex_map region_to_g; // Map between air regions and vertices in graph
  operation_id_to_vertex_map dma_to_g; // Map between air dmamemcpy2d and vertices in graph
  operation_id_to_vertex_map hl_to_g; // Map between air herd_launch and vertices in graph

  // g vertex to air op mapping
  air::RegionOp getRegionOpFromVertex (Graph::vertex_descriptor v, Graph g){
    assert(g[v].asyncEventType == "region" && "This vertex is not a RegionOp");
    return async_region_op_history[g[v].operationId - 1];
  }
  air::DmaMemcpyInterface getDmaOpFromVertex (Graph::vertex_descriptor v, Graph g){
    assert(g[v].asyncEventType == "dma" && "This vertex is not a DmaMemcpy op");
    return dma_op_history[g[v].operationId - 1];
  }
  air::HerdLaunchOp getHLOpFromVertex (Graph::vertex_descriptor v, Graph g){
    assert(g[v].asyncEventType == "herd_launch" && "This vertex is not a HerdLaunchOp");
    return hl_op_history[g[v].operationId - 1];
  }

  // air region op to g vertex mapping
  Graph::vertex_descriptor getGraphGVertexFromAIROp (air::RegionOp op){
    return region_to_g[op.getId()];
  }

  Graph::vertex_descriptor getGraphGVertexFromAIROp (air::DmaMemcpyInterface op){
    return dma_to_g[op.getId()];
  }

  Graph::vertex_descriptor getGraphGVertexFromAIROp (air::HerdLaunchOp op){
    return hl_to_g[op.getId()];
  }

  // Fill in dep list of air async ops using graph tr's connectivity
  template <typename T>
  void fillAIRDepListUsingGraphTR(T op){
    if (auto async_op = mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op.getOperation())){
      uint64_t dstTRVertex = g_to_tr[getGraphGVertexFromAIROp(op)];
      auto incoming_deps = in_edges(dstTRVertex, asyncRegionGraphTR);
      for (in_edge_iterator it = incoming_deps.first; it != incoming_deps.second; it++) {
        auto TRVertex = source(*it, asyncRegionGraphTR);
        if (asyncRegionGraphTR[TRVertex].asyncEventType == "region")
          async_op.addAsyncDependency(getRegionOpFromVertex(TRVertex, asyncRegionGraphTR).getResult(0));
        else if (asyncRegionGraphTR[TRVertex].asyncEventType == "dma")
          async_op.addAsyncDependency(getDmaOpFromVertex(TRVertex, asyncRegionGraphTR).getOperation()->getResult(0));
        else if (asyncRegionGraphTR[TRVertex].asyncEventType == "herd_launch")
          async_op.addAsyncDependency(getHLOpFromVertex(TRVertex, asyncRegionGraphTR).getResult(0));
        else assert(false && "Unknown async event type");
      }
    }
    else assert(false && "Operation has no async interface");
  }

  // Dump graphviz
  void dump_graph(char *filename)
  {
    std::ofstream ofs (filename, std::ofstream::out); 
    write_graphviz(ofs, asyncRegionGraphTR, boost::make_label_writer(boost::get(&regionNode::asyncEventName, asyncRegionGraphTR)));
  };

};

}// namespace


namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyPass() {
  return std::make_unique<AIRDependency>();
}

} // namespace air
} // namespace xilinx
