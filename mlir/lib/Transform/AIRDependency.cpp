// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependency.h"
#include "air/Util/CostModel.h"
#include "air/Util/Outliner.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>
#include <numeric> 

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

#define DEBUG_TYPE "air-dependency"

namespace {

static uint64_t RegionOpID;

class AIRDependency : public AIRDependencyBase<AIRDependency> {

public:
  AIRDependency() = default;
  AIRDependency(const AIRDependency &pass) {}

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder module_builder(module);

    RegionOpID = 0;

    // 1st traversal: create async regions with empty dep list.

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto dma2d_op = dyn_cast<air::DmaMemcpy2dOp>(op)) {
          
          module_builder.setInsertionPoint(op);

          // Create async region operation.
          auto async_region = createAsyncRegion(module_builder, op, "air::dma2d", RegionOpID);
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          module_builder.clone(*op);
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc());

          op->erase();

        }

        else if (auto matmul_op = dyn_cast<linalg::MatmulOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for linalg.matmul
          auto async_region = createAsyncRegion(module_builder, op, "linalg::matmul", RegionOpID);
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          module_builder.clone(*op);
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc());

          op->erase();
        }

        else if (auto linalg_fill_op = dyn_cast<linalg::FillOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for linalg.fill
          auto async_region = createAsyncRegion(module_builder, op, "linalg::fill", RegionOpID);
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          module_builder.clone(*op);
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc());

          op->erase();
        }

        else if (auto linalg_copy_op = dyn_cast<linalg::CopyOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for linalg.copy
          auto async_region = createAsyncRegion(module_builder, op, "linalg::copy", RegionOpID);
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          module_builder.clone(*op);
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc());

          op->erase();
        }

        else if (auto memalloc_op = dyn_cast<memref::AllocOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for memref.alloc
          auto async_region = createAsyncRegion(module_builder, op, "memref::alloc", RegionOpID, memalloc_op.memref().getType());
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          auto new_memref_alloc_op = dyn_cast<memref::AllocOp>(module_builder.clone(*op));
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc(), new_memref_alloc_op.memref());

          SmallVector<Value, 1> returnVals;
          returnVals.push_back(async_region.getResult(1));
          op->replaceAllUsesWith(returnVals);

          op->erase();

        }

        else if (auto memcast_op = dyn_cast<memref::CastOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for memref.alloc
          auto async_region = createAsyncRegion(module_builder, op, "memref::cast", RegionOpID, memcast_op.dest().getType());
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          auto new_memref_cast_op = dyn_cast<memref::CastOp>(module_builder.clone(*op));
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc(), new_memref_cast_op.dest());

          SmallVector<Value, 1> returnVals;
          returnVals.push_back(async_region.getResult(1));
          op->replaceAllUsesWith(returnVals);

          op->erase();

        }

        else if (auto memdealloc_op = dyn_cast<memref::DeallocOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for memref.dealloc
          auto async_region = createAsyncRegion(module_builder, op, "memref::dealloc", RegionOpID);
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          module_builder.clone(*op);
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc());

          op->erase();
        }

        else if (auto arith_op = dyn_cast<arith::MulIOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for arith.muli
          auto async_region = createAsyncRegion(module_builder, op, "arith::muli", RegionOpID, arith_op.getResult().getType());
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          auto new_arith_muli_op = dyn_cast<arith::MulIOp>(module_builder.clone(*op));
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc(), new_arith_muli_op.getResult());

          SmallVector<Value, 1> returnVals;
          returnVals.push_back(async_region.getResult(1));
          op->replaceAllUsesWith(returnVals);

          op->erase();

        }

        else if (auto apply_op = dyn_cast<mlir::AffineApplyOp>(op)) {

          module_builder.setInsertionPoint(op);

          // Create async region for affine.apply
          auto async_region = createAsyncRegion(module_builder, op, "affine::apply", RegionOpID, apply_op.getResult().getType());
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          auto new_apply_op = dyn_cast<mlir::AffineApplyOp>(module_builder.clone(*op));
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc(), new_apply_op.getResult());

          SmallVector<Value, 1> returnVals;
          returnVals.push_back(async_region.getResult(1));
          op->replaceAllUsesWith(returnVals);

          op->erase();

        }

        else if (auto hl_op = dyn_cast<air::HerdLaunchOp>(op)) {
          
          module_builder.setInsertionPoint(op);

          // Create async region operation.
          auto async_region = createAsyncRegion(module_builder, op, "air::herd_launch", RegionOpID);
          Block *async_region_bb = module_builder.createBlock(&async_region.body());

          // Insert op to the new async region's body.
          module_builder.setInsertionPointToStart(async_region_bb);
          module_builder.clone(*op);
          module_builder.create<xilinx::air::RegionTerminatorOp>(module_builder.getUnknownLoc());

          op->erase();

        }
      });
    }

    // 2nd traversal: trace deps among async regions; build a boost dep graph.

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto async_region_op = dyn_cast<air::RegionOp>(op)) {

          for (auto &bb : async_region_op.body()){
            
            for (auto &child_op : bb.getOperations()){

              SmallVector<partialMemref, 1> child_op_memref_reads;
              SmallVector<partialMemref, 1> child_op_memref_writes;
              SmallVector<Value, 1> child_op_scalar_ins;
              SmallVector<Value, 1> child_op_scalar_outs;

              // If the child op is linalg::matmul
              if (auto child_mmult = dyn_cast<linalg::MatmulOp>(child_op)){
                for (auto mmult_ins : child_mmult.inputs()){
                  partialMemref tile = createPartialMemref(mmult_ins);
                  child_op_memref_reads.push_back(tile);
                }
                // TODO: assert if # of output memref is not one
                for (auto mmult_outs : child_mmult.outputs()){
                  partialMemref tile = createPartialMemref(mmult_outs);
                  child_op_memref_reads.push_back(tile); // linalg.matmul accumulates on output
                  child_op_memref_writes.push_back(tile);
                }
              }
              
              // If the child op is linalg::fill
              if (auto child_fill = dyn_cast<linalg::FillOp>(child_op)){
                partialMemref tile = createPartialMemref(child_fill.output());
                child_op_memref_reads.push_back(tile);
                child_op_memref_writes.push_back(tile);
              }
              
              // If the child op is linalg::copy
              if (auto child_copy = dyn_cast<linalg::CopyOp>(child_op)){
                partialMemref tile_in = createPartialMemref(child_copy.input());
                partialMemref tile_out = createPartialMemref(child_copy.output());
                child_op_memref_reads.push_back(tile_in);
                child_op_memref_reads.push_back(tile_out); // linalg.copy both reads and writes output
                child_op_memref_writes.push_back(tile_out);
              }
              
              // If the child op is memref::dealloc
              if (auto child_memdealloc = dyn_cast<memref::DeallocOp>(child_op)){
                partialMemref tile = createPartialMemref(child_memdealloc.memref());
                child_op_memref_reads.push_back(tile);
                child_op_memref_writes.push_back(tile); // dealloc erases (i.e. writes to) output memref
              }
              
              // If the child op is air::DmaMemcpy2dOp
              if (auto child_dma2d = dyn_cast<air::DmaMemcpy2dOp>(child_op)){
                partialMemref tile_in = createPartialMemref(child_dma2d.getSrcMemref(), child_dma2d.getSrcMemrefD1(), child_dma2d.getSrcMemrefD0());
                child_op_memref_reads.push_back(tile_in);
                partialMemref tile_out = createPartialMemref(child_dma2d.getDstMemref(), child_dma2d.getDstMemrefD1(), child_dma2d.getDstMemrefD0());
                child_op_memref_writes.push_back(tile_out);
              }
              
              // If the child op is linalg::MulIOp
              if (auto child_arith = dyn_cast<arith::MulIOp>(child_op)){
                child_op_scalar_ins.push_back(child_arith.getLhs());
                child_op_scalar_ins.push_back(child_arith.getRhs());
                child_op_scalar_outs.push_back(child_arith.getResult());
              }
              
              // If the child op is mlir::AffineApplyOp
              if (auto child_apply = dyn_cast<mlir::AffineApplyOp>(child_op)){
                for (auto applyop_operand : child_apply.getMapOperands()){
                  child_op_scalar_ins.push_back(applyop_operand);
                }
                child_op_scalar_outs.push_back(child_apply.getResult());
              }


              // Detect RAW deps
              for (auto operand : child_op_memref_reads) {
                // Trace the defining op of target op, RAW
                pushDefiningOpAsDep(operand.memrefValue, async_region_op);

                // If target op and operand's use are under the same scope
                pushDepsAtCurrentScope(operand.memrefValue, async_region_op, 'w', operand.memrefIdx, operand.memrefIdy);

                // If target op is in HerdLaunchOp
                if (auto lh = op->getParentOfType<xilinx::air::HerdLaunchOp>()){
                  // Search for dma deps outside (before) HerdLaunchOp
                  for (unsigned lh_operand_id = 0; lh_operand_id < lh.getNumKernelOperands(); lh_operand_id++){
                    if (lh.getKernelArguments()[lh_operand_id] == operand.memrefValue){
                      auto ancestor_op = lh.getKernelOperand(lh_operand_id);
                      pushDepsAtCurrentScope(ancestor_op, lh->getParentOfType<xilinx::air::RegionOp>(), 'w');
                    }
                  }
                }

                // If target op is outside HerdLaunchOp
                else {
                  // If the input memref was used inside HerdLaunchOp in the past
                  for (auto &u : operand.memrefValue.getUses()){
                    if (auto lh = dyn_cast<xilinx::air::HerdLaunchOp>(u.getOwner())){
                      auto ar = lh->getParentOfType<xilinx::air::RegionOp>();
                      if (foundARUsesAboveCurrentLine(&ar)){
                        addNewAsyncDependency(ar.getResult(0), async_region_op);
                      }
                    }
                  }
                }
              }

              // Detect WAW and WAR deps
              for (auto operand : child_op_memref_writes) {
                // Trace the defining op of target op, WAW
                pushDefiningOpAsDep(operand.memrefValue, async_region_op);

                // If target op and operand's use are under the same scope
                pushDepsAtCurrentScope(operand.memrefValue, async_region_op, 'n', operand.memrefIdx, operand.memrefIdy);

                // If target op is in HerdLaunchOp
                if (auto lh = op->getParentOfType<xilinx::air::HerdLaunchOp>()){
                  // Search for dma deps outside (before) HerdLaunchOp
                  for (unsigned lh_operand_id = 0; lh_operand_id < lh.getNumKernelOperands(); lh_operand_id++){
                    if (lh.getKernelArguments()[lh_operand_id] == operand.memrefValue){
                      auto ancestor_op = lh.getKernelOperand(lh_operand_id);
                      pushDepsAtCurrentScope(ancestor_op, lh->getParentOfType<xilinx::air::RegionOp>());
                    }
                  }
                }

                // If target op is outside HerdLaunchOp
                else {

                  // If the input memref was used inside HerdLaunchOp in the past
                  for (auto &u : operand.memrefValue.getUses()){
                    if (auto lh = dyn_cast<xilinx::air::HerdLaunchOp>(u.getOwner())){
                      auto ar = lh->getParentOfType<xilinx::air::RegionOp>();
                      if (foundARUsesAboveCurrentLine(&ar)){
                        addNewAsyncDependency(ar.getResult(0), async_region_op);
                      }
                    }
                  }
                }
              }
              // Tile index deps
              for (auto operand : child_op_memref_reads) {
                if (operand.memrefIdx){
                  pushTileIndexAsDep(operand.memrefIdx, async_region_op);
                }
                if (operand.memrefIdy){
                  pushTileIndexAsDep(operand.memrefIdy, async_region_op);                  
                }
              }
              for (auto operand : child_op_memref_writes) {
                if (operand.memrefIdx){
                  pushTileIndexAsDep(operand.memrefIdx, async_region_op);
                }
                if (operand.memrefIdy){
                  pushTileIndexAsDep(operand.memrefIdy, async_region_op);                  
                }
              }
              for (auto scalar : child_op_scalar_ins) {
                pushTileIndexAsDep(scalar, async_region_op);
              }
              for (auto scalar : child_op_scalar_outs) {
                pushTileIndexAsDep(scalar, async_region_op);
              }
            }
          }
          // Keep track of processed async region ops. Deps should point to the past, not future.
          async_region_op_history.push_back(async_region_op);
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
        if (auto async_region_op = dyn_cast<air::RegionOp>(op)) {
          uint64_t dstTRVertex = g_to_tr[getGraphGVertexFromRegionOp(async_region_op)];
          auto incoming_deps = in_edges(dstTRVertex, asyncRegionGraphTR);
          for (in_edge_iterator it = incoming_deps.first; it != incoming_deps.second; it++) {
            auto TRVertex = source(*it, asyncRegionGraphTR);
            // async_region_op.addAsyncDependency(getRegionOpFromGraphGVertex(tr_to_g[TRVertex]).getResult(0));
            async_region_op.addAsyncDependency(getRegionOpFromVertex(TRVertex, asyncRegionGraphTR).getResult(0));
          }
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
              unsigned src_id = dyn_cast<air::RegionOp>(sink.getDefiningOp()).getId() - 1;
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
              else if (v.getDefiningOp())
                if (auto v_op = dyn_cast<air::RegionOp>(v.getDefiningOp()))
                  if (v_op.getAsyncToken() == v)
                    incoming_tokens.push_back(v);
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
            merged_incoming_token.push_back(wait_all_op_before_loop);
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
                  dst = g_to_tr[getGraphGVertexFromRegionOp(dst_op)];
                }
                if (auto src_op = dyn_cast<air::RegionOp>(v.getDefiningOp())){
                  src = g_to_tr[getGraphGVertexFromRegionOp(src_op)];
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

            // Yield an async token
            SmallVector<Value, 4> yield_token;
            yield_token.push_back(wait_all_op_yielded.getResult());
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

  std::vector<air::RegionOp> async_region_op_history;

  struct partialMemref {
    Value memrefValue;
    Value memrefIdx;
    Value memrefIdy;
  };

  partialMemref createPartialMemref(mlir::Value memrefValue, mlir::Value memrefIdx = 0, mlir::Value memrefIdy = 0){
    partialMemref tile;
    tile.memrefValue = memrefValue;
    tile.memrefIdx = memrefIdx;
    tile.memrefIdy = memrefIdy;
    return tile;
  }

  

  air::RegionOp createAsyncRegion(OpBuilder &builder, Operation *op, std::string asyncEventName, uint64_t &RegionOpID, mlir::Type valueType = NULL){
    auto loc = op->getLoc();
    SmallVector<Value, 1> deps;
    air::RegionOp async_region;
    if (valueType)
      async_region = builder.create<xilinx::air::RegionOp>(loc, air::AsyncTokenType::get(op->getContext()), valueType, deps);
    else
      async_region = builder.create<xilinx::air::RegionOp>(loc, air::AsyncTokenType::get(op->getContext()), deps);
    async_region->setAttr("id",
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                    ++RegionOpID));
    
    // Create a vertex out of the current async region
    auto v = add_vertex(asyncRegionGraph);
    asyncRegionGraph[v].asyncEventName = asyncEventName;
    asyncRegionGraph[v].asyncEventType = "region";
    asyncRegionGraph[v].operationId = RegionOpID;
    // Update op-to-graph map
    region_to_g[async_region.getId()] = v;
    return async_region;
  }

  bool foundARUsesAboveCurrentLine(air::RegionOp *op){
    if (!async_region_op_history.empty())
      for (auto &iter : async_region_op_history)
        if (iter.getResult(0) == op->getResult(0)) return true;
    return false;
  }
  
  // Check if operand is returned from RegionOp (memref.alloc)
  void pushDefiningOpAsDep(Value operand, air::RegionOp op){
    // Check memref deps
    if (auto defop = operand.getDefiningOp<air::RegionOp>()){
      if (foundARUsesAboveCurrentLine(&defop)){
        addNewAsyncDependency(defop.getResult(0), op);
      }
    }
  }

  // Trace tile index deps
  void pushTileIndexAsDep(mlir::Value tile_index, air::RegionOp op){
    // If created by async_region
    if (auto defop = tile_index.getDefiningOp<air::RegionOp>()){
      if (foundARUsesAboveCurrentLine(&defop)){
        addNewAsyncDependency(defop.getResult(0), op);
      }
    }
    // If created by launch_herd (as loop iter)
    else if (auto lh = dyn_cast<air::HerdLaunchOp>(tile_index.getParentRegion()->getParentOp())){
      if (lh.getTileIds().x == tile_index || lh.getTileIds().y == tile_index){
        addNewAsyncDependency(tile_index, op);
      }
    }
    // If created by scf.for (as loop iter)
    else if (auto forloop = dyn_cast<scf::ForOp>(tile_index.getParentRegion()->getParentOp())){
      if (forloop.getInductionVar() == tile_index){
        addNewAsyncDependency(tile_index, op);
      }
    }
  }

  // Trace operand's uses at current scope
  void pushDepsAtCurrentScope(mlir::Value operand, air::RegionOp op, char rw = 'n', mlir::Value idx = 0, mlir::Value idy = 0){
    for (auto &u : operand.getUses()){
      // If used in DmaMemcpy2dOp
      if (auto dma2d = dyn_cast<xilinx::air::DmaMemcpy2dOp>(u.getOwner())){
        if (auto ar = dyn_cast<xilinx::air::RegionOp>(dma2d->getParentOp())){
          if (foundARUsesAboveCurrentLine(&ar)){ // If this use is above current line
            // DMA2D: Need to check for overlapping partial memrefs in use
            if (rw == 'r'){
              if (u.getOperandNumber() == 1){
                if (idx == 0 && idy == 0){
                  addNewAsyncDependency(ar.getResult(0), op);
                }
                else if (areEqualIndices(idx, dma2d.getSrcMemrefD1()) && areEqualIndices(idy, dma2d.getSrcMemrefD0())){
                  addNewAsyncDependency(ar.getResult(0), op);
                }
              }
            }
            else if (rw == 'w'){
              if (u.getOperandNumber() == 0){
                if (idx == 0 && idy == 0){
                  addNewAsyncDependency(ar.getResult(0), op);
                }
                else if (areEqualIndices(idx, dma2d.getDstMemrefD1()) && areEqualIndices(idy, dma2d.getDstMemrefD0())){
                  addNewAsyncDependency(ar.getResult(0), op);
                }
              }
            }
            else{
              if (idx == 0 && idy == 0) {
                addNewAsyncDependency(ar.getResult(0), op);
              }
              else if (u.getOperandNumber() == 0){
                if (areEqualIndices(idx, dma2d.getDstMemrefD1()) && areEqualIndices(idy, dma2d.getDstMemrefD0()))
                  addNewAsyncDependency(ar.getResult(0), op);
              }
              else if (u.getOperandNumber() == 1){
                if (areEqualIndices(idx, dma2d.getSrcMemrefD1()) && areEqualIndices(idy, dma2d.getSrcMemrefD0()))
                  addNewAsyncDependency(ar.getResult(0), op);
              }
            }
          }
        }
      }

      // If used in MatmulOp
      if (auto matmul = dyn_cast<linalg::MatmulOp>(u.getOwner())){
        if (auto ar = dyn_cast<xilinx::air::RegionOp>(matmul->getParentOp())){
          if (foundARUsesAboveCurrentLine(&ar)){
            if (rw == 'r'){
              if (u.getOperandNumber() <= 2){
                addNewAsyncDependency(ar.getResult(0), op);
              }
            }
            else if (rw == 'w'){
              if (u.getOperandNumber() == 2){
                addNewAsyncDependency(ar.getResult(0), op);
              }
            }
            else{
              addNewAsyncDependency(ar.getResult(0), op);
            }
          }
        }
      }

      // If used in linalg.fill
      if (auto fill_op = dyn_cast<linalg::FillOp>(u.getOwner())){
        if (auto ar = dyn_cast<xilinx::air::RegionOp>(fill_op->getParentOp())){
          if (foundARUsesAboveCurrentLine(&ar)){
            if (rw == 'r'){
              if (u.getOperandNumber() == 1){
                addNewAsyncDependency(ar.getResult(0), op);
              }
            }
            else if (rw == 'w'){
              if (u.getOperandNumber() == 1){
                addNewAsyncDependency(ar.getResult(0), op);
              }
            }
            else{
              addNewAsyncDependency(ar.getResult(0), op);
            }
          }
        }
      }

      // If used in linalg.copy
      if (auto copy_op = dyn_cast<linalg::CopyOp>(u.getOwner())){
        if (auto ar = dyn_cast<xilinx::air::RegionOp>(copy_op->getParentOp())){
          if (foundARUsesAboveCurrentLine(&ar)){
            if (rw == 'r'){
              if (u.getOperandNumber() <= 1){
                addNewAsyncDependency(ar.getResult(0), op);
              }
            }
            else if (rw == 'w'){
              if (u.getOperandNumber() == 1){
                addNewAsyncDependency(ar.getResult(0), op);
              }
            }
            else{
              addNewAsyncDependency(ar.getResult(0), op);
            }
          }
        }
      }

    }
  }

  bool areEqualIndices (mlir::Value index_0, mlir::Value index_1){
    if (index_0 == 0 || index_1 == 0) return false;
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

  void addNewAsyncDependency(Value dep, air::RegionOp op){
    for (auto old_dep : op.getAsyncDependencies())
      if (old_dep == dep) return;

    // Add edge to boost graph, iff dep is async region (i.e. not a loop iterator)
    if (auto srcOp = dep.getDefiningOp()) {
      assert(dyn_cast<air::RegionOp>(srcOp) && "dependency token should be generated by async region");
      uint64_t srcNode = getGraphGVertexFromRegionOp(dyn_cast<air::RegionOp>(srcOp));
      uint64_t dstNode = getGraphGVertexFromRegionOp(op);
      add_edge(srcNode, dstNode, asyncRegionGraph);
    }
  }

  bool isSingleChildInParentForOp(scf::ForOp child_for_op){
    if (!child_for_op->getParentOp())
      return false;
    if (!dyn_cast<scf::ForOp>(child_for_op->getParentOp()))
      return false;
    auto parent_op = dyn_cast<scf::ForOp>(child_for_op->getParentOp());
    unsigned number_of_nested_for_ops = 0;
    for (auto &op : parent_op.getBody()->getOperations()){
      number_of_nested_for_ops ++;
    }
    if (number_of_nested_for_ops == 2) return true; // child for op plus terminator
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
  region_id_to_vertex_map region_to_g; // Map between air ops and vertices in graph

  // air region op to g vertex mapping
  air::RegionOp getRegionOpFromVertex (Graph::vertex_descriptor v, Graph g){
    assert(g[v].asyncEventType == "region" && "This vertex is not a RegionOp");
    return async_region_op_history[g[v].operationId - 1];
  }

  // g vertex to air region op mapping
  Graph::vertex_descriptor getGraphGVertexFromRegionOp (air::RegionOp op){
    return region_to_g[op.getId()];
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
