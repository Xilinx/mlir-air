//===- AIRToAIEPass.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToAIESchedulingUtils.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Transform/AIRDependencyScheduleOpt.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"
#include <numeric>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "air-to-aie"

using namespace mlir;

// namespace {
namespace xilinx {
namespace air {

struct AIRToAIEConversionOptions {
  int64_t col_offset;
  int64_t row_offset;
  bool emit_while;
  bool emit_herd_lock;
  bool generate_shim_dma;
  bool insert_trace_packet_flow;
  bool use_lock_race_condition_fix;
  // v2: daisy-chained lock emission for shared L2 buffers with
  // fan-in (N writers + 1 reader) or fan-out (1 writer + N readers) sub-
  // region access patterns. Emits 1 cap lock + N init=0 signal locks
  // forming a strict producer-ordering chain, eliminating concurrent-
  // access races on the memtile DMA. Mutually exclusive with v1.
  bool use_lock_race_condition_fix_v2;
  AIE::AIEDevice device;
  unsigned stack_size;
};

// Breakpoint stages for debugging with --test-patterns
// Each stage represents a point in the pipeline where execution can stop
// for debugging purposes.
enum class PipelineStage {
  AfterCreateAIEModules,  // After createAIEModulesAndOutlineCores
  AfterCloneMemcpys,      // After cloneL2AndL3MemcpysToDeviceOp
  AfterLowerExecute,      // After lowerAirExecute + lowerScfAirTokens
  AfterSpecializeChannel, // After specializeChannelBundle
  AfterAllocBuffers,      // After allocL1/L2Buffers
  AfterRenumberMemcpy,    // After renumberMemcpyIfOps
  AfterLowerAIRMemcpy,    // After lowerAIRMemcpyOp
  AfterTracePacketFlow,   // After createTracePacketFlow
  AfterLowerMemRefCopy,   // After lowerMemRefCopyToLoops
  Complete                // Full pipeline including metadata/cleanup
};

// get memcpy operation volumn (elements) as int
int getMemcpySizesAsInt(Value memref, SmallVector<Value> sizes) {
  BaseMemRefType memTy = llvm::cast<BaseMemRefType>(memref.getType());
  if (sizes.empty())
    return air::getTensorVolume(memTy);
  else {
    int output = 1;
    for (auto s : sizes) {
      auto c = dyn_cast_if_present<arith::ConstantIndexOp>(s.getDefiningOp());
      if (!c) {
        output = -1;
        break;
      }
      output *= c.value();
    }
    return output;
  }
}

// Allocator for shim NOC tiles. AIR no longer makes shim placement decisions;
// each call to getShimTile() emits an unplaced aie.logical_tile<ShimNOCTile>.
// resolveLogicalShimTiles() runs the placer after channel lowering completes
// to convert them into physical aie.tile ops. The placer's
// findTileWithCapacity / hasAvailableChannels logic in mlir-aie absorbs the
// channel-aware merging that used to live here. See RFC #1567.
struct ShimTileAllocator {

  const AIE::AIETargetModel &aie_target;
  // Channel symbol names that went through getShimTile(); consumed downstream
  // for objectfifo metadata labeling.
  std::vector<std::string> chan_names;
  // Logical shim tiles emitted during channel lowering, one per call. Each
  // is a fresh op created via the active PatternRewriter; entries are
  // cleared and replaced with physical aie.tile by resolveLogicalShimTiles.
  std::vector<AIE::LogicalTileOp> logical_shim_tiles;

  ShimTileAllocator(const AIE::AIETargetModel &target) : aie_target(target) {}

  // Emit a fresh aie.logical_tile<ShimNOCTile>(?, ?) and return its result.
  // No round-robin column selection, no per-direction capacity tracking; the
  // placer handles all of that. Must be called from inside a pattern with an
  // active rewriter, since the result is immediately consumed by ops the
  // rewriter is building.
  Value getShimTile(PatternRewriter &rewriter, AIE::DeviceOp aie_device,
                    std::string chan_name) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(aie_device.getBody());
    auto logical = AIE::LogicalTileOp::create(
        rewriter, aie_device.getLoc(), AIE::AIETileType::ShimNOCTile,
        /*col=*/IntegerAttr(),
        /*row=*/IntegerAttr(),
        /*allocation_scheme=*/StringAttr());
    logical_shim_tiles.push_back(logical);
    chan_names.push_back(chan_name);
    return logical.getResult();
  }

  // Resolve all aie.logical_tile<ShimNOCTile> into physical aie.tile via the
  // SequentialPlacer. Must be called after channel lowering completes — once
  // the greedy rewriter has settled — so no Operation* is held across rewrite
  // iterations.
  LogicalResult resolveLogicalShimTiles(AIE::DeviceOp aie_device) {
    if (logical_shim_tiles.empty())
      return success();

    AIE::SequentialPlacer placer;
    placer.initialize(aie_target);
    if (failed(placer.place(aie_device)))
      return aie_device.emitError("failed to place logical shim tiles");

    for (auto logical : logical_shim_tiles) {
      auto placement = placer.getPlacement(logical.getOperation());
      if (!placement)
        return logical.emitError("no placement for logical shim tile");
      auto physTile =
          air::getPhysTileOp(aie_device, placement->col, placement->row);
      logical.getResult().replaceAllUsesWith(physTile.getResult());
      logical.erase();
    }
    logical_shim_tiles.clear();
    return success();
  }
};

bool isMM2S(AIE::DMAChannel channel) {
  return (channel.direction == AIE::DMAChannelDir::MM2S);
}

std::string createSymbolName(Operation *symbol_table, std::string dma_name) {
  std::string new_cname = dma_name;
  std::string cname = "";
  int which_try = 0;
  while (SymbolTable::lookupSymbolIn(symbol_table, new_cname))
    new_cname = dma_name + "_" + std::to_string(++which_try);
  cname = new_cname;
  return cname;
}

// Accepts either a physical AIE::TileOp or an unplaced AIE::LogicalTileOp via
// the AIE::TileLike interface.
AIE::BufferOp allocateBufferOp(uint64_t &BufferId, MemRefType memrefTy,
                               AIE::TileLike tileLike,
                               mlir::StringAttr attr = nullptr, int x = -1,
                               int y = -1) {
  Operation *tileOp = tileLike.getOperation();
  OpBuilder builder(tileOp);
  Operation *t = tileOp;
  while (isa_and_present<AIE::TileLike>(t->getNextNode()))
    t = t->getNextNode();
  builder.setInsertionPointAfter(t);
  AIE::BufferOp bufferOp = AIE::BufferOp::create(
      builder, tileOp->getLoc(), memrefTy, tileOp->getResult(0),
      /*sym_name*/ nullptr,
      /*address*/ nullptr, /*initial_value*/ nullptr,
      /*mem_bank*/ nullptr);

  std::stringstream ss =
      air::generateBufferNameInStringStream("buf", BufferId, attr, x, y);
  bufferOp->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(tileOp->getContext(), ss.str()));

  return bufferOp;
}

// Set data layout attribute on AIE device to specify index type has 32 bits
// width
void setAIEDeviceDataLayout(OpBuilder &builder, AIE::DeviceOp aie_dev) {
  auto indexType = builder.getIndexType();
  auto dlEntry =
      DataLayoutEntryAttr::get(indexType, builder.getI64IntegerAttr(32));
  auto dlSpec = DataLayoutSpecAttr::get(builder.getContext(), {dlEntry});
  aie_dev->setAttr(DLTIDialect::kDataLayoutAttrName, dlSpec);
}

LogicalResult outlineAIECores(OpBuilder &builder, AIE::DeviceOp aie_device,
                              air::HerdOp h,
                              AIRToAIEConversionOptions &options) {
  builder.setInsertionPointToStart(aie_device.getBody());

  int64_t herd_size_x = h.getNumCols();
  int64_t herd_size_y = h.getNumRows();

  h.walk([&](air::ChannelInterface op) {
    if (!aie_device.lookupSymbol(op.getChanName())) {
      auto ch = air::getChannelDeclarationThroughSymbol(op);
      builder.clone(*ch.getOperation());
    }
  });

  // Use the offsets from options. For segment unroll, col_offset is already
  // set to 0 in iter_options by createAIEModulesAndOutlineCores.
  int64_t col_offset = options.col_offset;
  int64_t row_offset = options.row_offset;

  auto col_name = air::HerdOp::getColOffsetAttrName();
  auto row_name = air::HerdOp::getRowOffsetAttrName();
  auto ctx = h->getContext();
  if (auto co = h.getColOffset())
    col_offset = *co;
  else
    h->setAttr(col_name,
               IntegerAttr::get(IntegerType::get(ctx, 32), col_offset));
  if (auto ro = h.getRowOffset())
    row_offset = *ro;
  else
    h->setAttr(row_name,
               IntegerAttr::get(IntegerType::get(ctx, 32), row_offset));

  for (auto y = 0; y < herd_size_y; y++) {
    for (auto x = 0; x < herd_size_x; x++) {
      auto hloc = h.getLoc();
      IRMapping remap;
      auto phys_x = x + col_offset;
      auto phys_y = y + row_offset;

      // Emit aie.logical_tile<CoreTile>(phys_x, phys_y) and resolve via
      // mlir-aie's SequentialPlacer (RFC #1567 Stage A milestone 4). For
      // this milestone we keep both coordinates fully constrained, so the
      // Compute tiles here are fully constrained to (phys_x, phys_y) by the
      // AIR herd; we can resolve directly to a physical aie.tile without any
      // placer involvement. (Memtiles and shim tiles take the LTO route — see
      // outlineAIEMemtiles and ShimDMAAllocator::allocNewDmaChannel — and let
      // the downstream `aie-place-tiles` pass pick rows/columns.)
      auto tile = air::getPhysTileOp(aie_device, phys_x, phys_y);

      Operation *t = tile.getOperation();
      while (isa_and_present<AIE::TileLike>(t->getNextNode()))
        t = t->getNextNode();
      builder.setInsertionPointAfter(t);

      // make the aie.core for the tile core
      auto herd_name =
          aie_device
              ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
              .getValue()
              .str();
      auto core = tile.getCoreOp();
      if (!core) {
        core = AIE::CoreOp::create(builder, hloc, tile);
        if (options.stack_size != 1024)
          core.setStackSize(options.stack_size);
        // Persist herd metadata as aie.core attributes so downstream code
        // doesn't need a tileToHerdMap to recover (RFC #1567 Stage C #3).
        // air.herd_local_id stores the per-cell (x, y) within the herd;
        // air.herd_size stores the herd's (x_size, y_size); air.herd_name
        // stores the herd's symbol name when available.
        core->setAttr("air.herd_local_id",
                      builder.getDenseI64ArrayAttr(
                          {static_cast<int64_t>(x), static_cast<int64_t>(y)}));
        core->setAttr("air.herd_size",
                      builder.getDenseI64ArrayAttr({herd_size_x, herd_size_y}));
        if (auto sym =
                h->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
          core->setAttr("air.herd_name", sym);
        if (auto a = h->getAttrOfType<StringAttr>("link_with"))
          core->setAttr("link_with", a);
      }

      // Collect IDs from all parent hierarchy ops (Launch, Segment, etc.).
      // These values become compile-time constants and don't need RTP slots.
      llvm::SmallDenseSet<Value> hierarchyIdSet;
      for (auto *parentOp = h->getParentOp(); parentOp;
           parentOp = parentOp->getParentOp()) {
        if (auto hierarchy =
                dyn_cast_if_present<air::HierarchyInterface>(parentOp)) {
          for (auto id : hierarchy.getIds())
            hierarchyIdSet.insert(id);
        }
      }

      int64_t rtp_buffer_size = 0; // size in i32s
      for (unsigned ki = 0, ke = h.getNumKernelOperands(); ki < ke; ki++) {
        BlockArgument karg = h.getKernelArgument(ki);
        Value koperand = h.getKernelOperand(ki);
        // Skip operands that come from parent hierarchy IDs (they become
        // compile-time constants, not runtime parameters)
        if (hierarchyIdSet.contains(koperand))
          continue;
        // each one gets 32-bits in the rtp buffer
        if (llvm::isa<IntegerType, IndexType, FloatType>(karg.getType()))
          rtp_buffer_size++;
      }
      AIE::BufferOp rtp_buffer = nullptr;
      if (rtp_buffer_size) {
        uint64_t buffer_id = 0;
        rtp_buffer = allocateBufferOp(
            buffer_id, MemRefType::get({rtp_buffer_size}, builder.getI32Type()),
            tile, builder.getStringAttr("__air_herd_rtp"), phys_x, phys_y);
        if (!options.emit_herd_lock) {
          h.emitWarning("Herd RTP buffer allocated but herd lock disabled");
          h.emitWarning("Enabling herd lock for RTP buffer synchronization");
          options.emit_herd_lock = true;
        }
      }

      Value herd_lock = nullptr;
      if (options.emit_herd_lock) {
        StringAttr name =
            builder.getStringAttr("__air_herd_lock_" + std::to_string(phys_x) +
                                  "_" + std::to_string(phys_y));
        // herd lock is always lock zero
        herd_lock =
            air::allocateLockOp(aie_device, tile, /*init=*/0, /*id=*/0, name);
      }

      assert((h.getBody().getBlocks().size() == 1) &&
             "Launch body can only contain one Block");

      // set insertion point for anything below created on behalf of the core
      builder.setInsertionPoint(core);

      // generate the aie.core body
      //
      OpBuilder core_builder(core);
      Block *core_bb = nullptr;
      Block *entry_bb = nullptr;
      // check if entry block already exists
      if (core.getBody().empty()) {
        core_bb = core_builder.createBlock(&core.getBody());
        entry_bb = core_builder.createBlock(core_bb);
        core_builder.setInsertionPointToEnd(entry_bb);
        cf::BranchOp::create(core_builder, hloc, core_bb);
        core_builder.setInsertionPointToEnd(core_bb);
      } else {
        // extending upon the existing bb chain
        for (auto &b : core.getBody().getBlocks())
          if (b.isEntryBlock())
            entry_bb = &b;
        Block *prev_bb_back = &core.getBody().back();
        auto prev_bb_branch =
            dyn_cast_if_present<cf::BranchOp>(prev_bb_back->getTerminator());
        auto prev_bb_end =
            dyn_cast_if_present<AIE::EndOp>(prev_bb_back->getTerminator());
        core_bb = core_builder.createBlock(&core.getBody());
        if (prev_bb_branch)
          prev_bb_branch.setDest(core_bb);
        else if (prev_bb_end) {
          core_builder.setInsertionPoint(prev_bb_end);
          cf::BranchOp::create(core_builder, hloc, core_bb);
          prev_bb_end->erase();
        }
        core_builder.setInsertionPointToEnd(core_bb);
      }

      // map the tile ids and herd size to constants
      remap.map(h.getIds()[0],
                arith::ConstantIndexOp::create(core_builder, hloc, x));
      remap.map(h.getIds()[1],
                arith::ConstantIndexOp::create(core_builder, hloc, y));
      remap.map(h.getSize()[0], arith::ConstantIndexOp::create(
                                    core_builder, hloc, herd_size_x));
      remap.map(h.getSize()[1], arith::ConstantIndexOp::create(
                                    core_builder, hloc, herd_size_y));

      // Map segment unroll IDs to constants if the herd is in an unrolled
      // segment. Handle dimension-by-dimension to support 1-D segments.
      if (auto seg = h->getParentOfType<air::SegmentOp>()) {
        auto segIds = seg.getIds();
        // Get unroll indices from device attributes
        int64_t unrollX = 0, unrollY = 0;
        if (auto attr =
                aie_device->getAttrOfType<IntegerAttr>("segment_unroll_x"))
          unrollX = attr.getInt();
        if (auto attr =
                aie_device->getAttrOfType<IntegerAttr>("segment_unroll_y"))
          unrollY = attr.getInt();

        // Map segment unroll IDs to constants, dimension by dimension
        if (segIds.size() >= 1) {
          remap.map(segIds[0], arith::ConstantIndexOp::create(core_builder,
                                                              hloc, unrollX));
        }
        if (segIds.size() >= 2) {
          remap.map(segIds[1], arith::ConstantIndexOp::create(core_builder,
                                                              hloc, unrollY));
        }
      }

      if (options.emit_herd_lock) {
        AIE::LockAction lockAction =
            aie_device.getTargetModel().hasProperty(
                AIE::AIETargetModel::UsesSemaphoreLocks)
                ? AIE::LockAction::AcquireGreaterEqual
                : AIE::LockAction::Acquire;
        int lockValue = aie_device.getTargetModel().hasProperty(
                            AIE::AIETargetModel::UsesSemaphoreLocks)
                            ? 1
                            : 0;
        AIE::UseLockOp::create(core_builder, core_builder.getUnknownLoc(),
                               herd_lock, lockAction, lockValue);
      }

      unsigned rtp_slot = 0;
      for (unsigned ki = 0, ke = h.getNumKernelOperands(); ki < ke; ki++) {
        BlockArgument karg = h.getKernelArgument(ki);

        // If the kernel operand is already mapped (e.g., segment unroll ID),
        // use the mapped constant value instead of RTP load. This ensures that
        // segment unroll IDs passed as herd kernel arguments become
        // compile-time constants in the outlined cores.
        Value koperand = h.getKernelOperand(ki);
        if (remap.contains(koperand)) {
          remap.map(karg, remap.lookup(koperand));
          continue;
        }

        // Remap the kernel operands to the rtp buffer.
        // For each kernel operand of a supported type, load the data from the
        // rtp buffer and remap uses of the kernel operand to the loaded value.
        if (llvm::isa<IntegerType, IndexType, FloatType>(karg.getType())) {

          // load from rtp buffer
          SmallVector<Value> offsets{
              arith::ConstantIndexOp::create(core_builder, hloc, rtp_slot)};
          auto load = memref::LoadOp::create(core_builder, hloc,
                                             IntegerType::get(ctx, 32),
                                             rtp_buffer, offsets);

          // truncate, extend or bitcast the value to the correct type
          Value rtp = nullptr;
          llvm::TypeSwitch<Type>(karg.getType())
              .Case<IntegerType>([&](IntegerType ity) {
                unsigned int width = ity.getWidth();
                if (width < 32)
                  rtp = arith::TruncIOp::create(core_builder, hloc, ity, load);
                else if (width > 32)
                  rtp = arith::ExtUIOp::create(core_builder, hloc, ity, load);
                else
                  rtp = load;
              })
              .Case<IndexType>([&](IndexType ity) {
                rtp = arith::IndexCastOp::create(core_builder, hloc, ity, load);
              })
              .Case<FloatType>([&](FloatType fty) {
                if (fty.getWidth() == 32) {
                  rtp = arith::BitcastOp::create(core_builder, hloc, fty, load);
                } else if (fty.getWidth() == 16) {
                  auto ity = IntegerType::get(ctx, 16);
                  auto tr =
                      arith::TruncIOp::create(core_builder, hloc, ity, load);
                  rtp = arith::BitcastOp::create(core_builder, hloc, fty, tr);
                }
              });

          // remap the kernel operand
          if (rtp)
            remap.map(karg, rtp);
          else
            h.emitWarning("Unsupported runtime parameter int or float type");
          rtp_slot++;
        }

        auto memrefTy = llvm::dyn_cast_if_present<MemRefType>(karg.getType());
        if (!memrefTy)
          continue;

        if (air::isL1(memrefTy)) {
          // fused herds sometimes have L1 memref allocation outside of herds.
          // mapping them back
          remap.map(karg, h.getKernelOperand(ki));
          continue;
        }

        std::string sym_name = createSymbolName(aie_device, "__air_herd_arg");
        memref::GlobalOp::create(builder, builder.getUnknownLoc(), sym_name,
                                 builder.getStringAttr("public"), memrefTy,
                                 nullptr, false, nullptr);

        auto m = memref::GetGlobalOp::create(
            core_builder, hloc, SmallVector<Type, 1>{karg.getType()}, sym_name);
        remap.map(karg, m);
      }

      Region &r = h.getRegion();
      r.cloneInto(&core.getBody(), remap);

      Block *launch_bb = remap.lookup(&r.front());
      cf::BranchOp::create(core_builder, hloc, launch_bb);
      core_builder.setInsertionPoint(launch_bb->getTerminator());
      if (options.emit_herd_lock) {
        if (aie_device.getTargetModel().hasProperty(
                AIE::AIETargetModel::UsesSemaphoreLocks)) {
          // we could release something, but we don't have to a way to observe
          // it yet in NPU
        } else {
          AIE::UseLockOp::create(core_builder, core_builder.getUnknownLoc(),
                                 herd_lock, AIE::LockAction::Release, 0);
        }
      }

      if (options.emit_while) {
        auto entry_bb_br =
            dyn_cast_if_present<cf::BranchOp>(entry_bb->getTerminator());
        cf::BranchOp::create(core_builder, hloc, entry_bb_br.getDest());
      } else
        AIE::EndOp::create(core_builder, hloc);

      core.walk([&](Operation *op) {
        if (auto call = dyn_cast_if_present<func::CallOp>(op)) {
          auto fn = aie_device.lookupSymbol<func::FuncOp>(call.getCallee());
          if (!fn) {
            // Normalize memref types: strip strided layout so that
            // convert-func-to-llvm with bare-ptr calling convention can
            // handle the declaration. External kernels use C ABI (raw
            // pointers), so MLIR layout metadata is irrelevant.
            SmallVector<Type> normalizedInputs;
            for (Type t : call.getCalleeType().getInputs()) {
              if (auto memrefTy = dyn_cast<MemRefType>(t)) {
                normalizedInputs.push_back(MemRefType::get(
                    memrefTy.getShape(), memrefTy.getElementType(),
                    MemRefLayoutAttrInterface{}, memrefTy.getMemorySpace()));
              } else {
                normalizedInputs.push_back(t);
              }
            }
            auto normalizedType =
                FunctionType::get(aie_device.getContext(), normalizedInputs,
                                  call.getCalleeType().getResults());
            fn = func::FuncOp::create(aie_device.getLoc(), call.getCallee(),
                                      normalizedType);
            fn.setPrivate();
            // Copy attributes from the original declaration in the parent
            // module (e.g. link_with, llvm.emit_c_interface).
            if (auto parentModule = aie_device->getParentOfType<ModuleOp>()) {
              if (auto origFn = parentModule.lookupSymbol<func::FuncOp>(
                      call.getCallee())) {
                for (auto attr : origFn->getDiscardableAttrs())
                  fn->setAttr(attr.getName(), attr.getValue());
              }
            }
            // Fallback: if link_with was not found from parent module,
            // use the attribute from the aie.core op.
            if (!fn->hasAttr("link_with")) {
              if (auto attr = core->getAttrOfType<StringAttr>("link_with"))
                fn->setAttr("link_with", attr);
            }
            if (!fn->hasAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
              fn->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                          UnitAttr::get(aie_device.getContext()));
            }
            aie_device.insert(aie_device.getBody()->getTerminator(), fn);
          }
          // Insert memref.cast at call sites where operand types differ
          // from the (possibly normalized) declaration types.
          auto fnType = fn.getFunctionType();
          OpBuilder castBuilder(call);
          SmallVector<Value> newOperands;
          bool needsUpdate = false;
          for (auto [operand, inputType] :
               llvm::zip(call.getOperands(), fnType.getInputs())) {
            if (operand.getType() != inputType) {
              if (!memref::CastOp::areCastCompatible(operand.getType(),
                                                     inputType)) {
                call.emitError("cannot cast operand type ")
                    << operand.getType() << " to normalized function type "
                    << inputType;
                newOperands.push_back(operand);
                continue;
              }
              auto cast = memref::CastOp::create(castBuilder, call.getLoc(),
                                                 inputType, operand);
              newOperands.push_back(cast);
              needsUpdate = true;
            } else {
              newOperands.push_back(operand);
            }
          }
          if (needsUpdate)
            call->setOperands(newOperands);
        }
      });

      // erase air.herd_termintor ops
      launch_bb->walk([&](air::HerdTerminatorOp op) { op->erase(); });
    }
  }
  return success();
}

// Get all tile ops representing memtiles from device op.
// Return all memtile-typed tile-defining ops in the device, as TileLike.
// Picks up both physical AIE::TileOp (post-aie-place-tiles) and unplaced
// AIE::LogicalTileOp emitted by outlineAIEMemtiles. Callers that need a
// physical TileOp must check the underlying op type before casting.
std::vector<AIE::TileLike> getMemtilesFromDeviceOp(AIE::DeviceOp d) {
  std::vector<AIE::TileLike> memtiles;
  for (auto t : d.getBody()->getOps<AIE::TileLike>())
    if (t.isMemTile())
      memtiles.push_back(t);
  return memtiles;
}

// Get segment unroll factors from air.segment's iteration space
// Returns {unrollX, unrollY} where X is dim 0 and Y is dim 1
// Note: AIE only supports column-wise device slicing, so unrollY should be 1.
// Returns failure if any dimension has a non-static size, since segment
// unrolling requires static sizes to determine the number of AIE devices
// at compile time.
FailureOr<std::pair<int64_t, int64_t>>
getSegmentUnrollFactors(air::SegmentOp seg) {
  int64_t unrollX = 1, unrollY = 1;
  unsigned numDims = seg.getNumDims();
  if (numDims > 0) {
    auto sizeOperands = seg.getSizeOperands();
    if (auto constOp =
            sizeOperands[0].getDefiningOp<arith::ConstantIndexOp>()) {
      unrollX = constOp.value();
    } else if (auto constIntOp =
                   sizeOperands[0].getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr =
              dyn_cast_if_present<IntegerAttr>(constIntOp.getValue()))
        unrollX = intAttr.getInt();
      else {
        seg.emitOpError("segment unroll X-dimension size must be a static "
                        "constant, but got non-integer constant");
        return failure();
      }
    } else {
      seg.emitOpError(
          "segment unroll X-dimension size must be a static constant");
      return failure();
    }
    if (numDims > 1) {
      if (auto constOp =
              sizeOperands[1].getDefiningOp<arith::ConstantIndexOp>()) {
        unrollY = constOp.value();
      } else if (auto constIntOp =
                     sizeOperands[1].getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr =
                dyn_cast_if_present<IntegerAttr>(constIntOp.getValue()))
          unrollY = intAttr.getInt();
        else {
          seg.emitOpError("segment unroll Y-dimension size must be a static "
                          "constant, but got non-integer constant");
          return failure();
        }
      } else {
        seg.emitOpError(
            "segment unroll Y-dimension size must be a static constant");
        return failure();
      }
    }
  }
  return std::make_pair(unrollX, unrollY);
}

// Validate segment unroll factors for AIE lowering.
// AIE only supports column-wise device slicing, so Y-dimension unrolling
// (unrollY > 1) is not allowed.
LogicalResult validateSegmentUnrollFactors(air::SegmentOp seg) {
  auto unrollFactors = getSegmentUnrollFactors(seg);
  if (failed(unrollFactors))
    return failure();
  auto [unrollX, unrollY] = *unrollFactors;
  if (unrollY > 1) {
    return seg.emitOpError(
        "segment unroll Y-dimension (row-wise) is not supported. "
        "AIE only supports column-wise device slicing. "
        "Please set Y-dimension unroll factor to 1.");
  }
  return success();
}

// Map (original device, unroll factor) -> sub-device type for NPU1 and NPU2
// Returns the appropriate sub-device type based on column count division
AIE::AIEDevice computeSubDeviceType(AIE::AIEDevice origDevice,
                                    int unrollFactor) {
  switch (origDevice) {
  // NPU1 family (4 columns)
  case AIE::AIEDevice::npu1:
    // npu1 has 4 columns
    switch (unrollFactor) {
    case 1:
      return AIE::AIEDevice::npu1;
    case 2:
      return AIE::AIEDevice::npu1_2col;
    case 4:
      return AIE::AIEDevice::npu1_1col;
    default:
      return origDevice;
    }
  case AIE::AIEDevice::npu1_3col:
    // npu1_3col has 3 columns
    switch (unrollFactor) {
    case 1:
      return AIE::AIEDevice::npu1_3col;
    case 3:
      return AIE::AIEDevice::npu1_1col;
    default:
      return origDevice;
    }
  case AIE::AIEDevice::npu1_2col:
    // npu1_2col has 2 columns
    switch (unrollFactor) {
    case 1:
      return AIE::AIEDevice::npu1_2col;
    case 2:
      return AIE::AIEDevice::npu1_1col;
    default:
      return origDevice;
    }
  // NPU2 family (8 columns)
  case AIE::AIEDevice::npu2:
    // npu2 has 8 columns
    switch (unrollFactor) {
    case 1:
      return AIE::AIEDevice::npu2;
    case 2:
      return AIE::AIEDevice::npu2_4col;
    case 4:
      return AIE::AIEDevice::npu2_2col;
    case 8:
      return AIE::AIEDevice::npu2_1col;
    default:
      return origDevice;
    }
  case AIE::AIEDevice::npu2_4col:
    // npu2_4col has 4 columns
    switch (unrollFactor) {
    case 1:
      return AIE::AIEDevice::npu2_4col;
    case 2:
      return AIE::AIEDevice::npu2_2col;
    case 4:
      return AIE::AIEDevice::npu2_1col;
    default:
      return origDevice;
    }
  case AIE::AIEDevice::npu2_2col:
    // npu2_2col has 2 columns
    switch (unrollFactor) {
    case 1:
      return AIE::AIEDevice::npu2_2col;
    case 2:
      return AIE::AIEDevice::npu2_1col;
    default:
      return origDevice;
    }
  default:
    return origDevice;
  }
}

LogicalResult outlineAIEMemtiles(OpBuilder &builder, AIE::DeviceOp aie_device,
                                 air::SegmentOp seg,
                                 AIRToAIEConversionOptions &options) {
  builder.setInsertionPointToStart(aie_device.getBody());

  int64_t seg_size_x = 1;
  if (auto num_cols = seg.getNumCols()) {
    seg_size_x = *num_cols;
  }

  // For segment unroll, divide by unroll factor since each sub-device gets
  // a portion of the segment's columns
  auto unrollFactors = getSegmentUnrollFactors(seg);
  if (succeeded(unrollFactors)) {
    auto [unrollX, unrollY] = *unrollFactors;
    if (unrollX > 1) {
      seg_size_x = (seg_size_x + unrollX - 1) / unrollX; // ceiling division
    }
  }

  // Extend to the max herd column footprint inside the segment. When the
  // segment carries no explicit x_size (Python sugar like
  // `@segment(name="seg")` defaults to 1x1), it doesn't reflect the actual
  // physical column extent the herds will occupy. Memtile is column-bound
  // so we need one logical memtile per column the herds will land on, or
  // L2 buffers for distinct columns wrap-collide onto the same memtile
  // LTO and the placer fails (matvec_2tile_add 8-col reproduces this).
  // Falling back to herd footprint matches what an explicitly-sized
  // segment (e.g. matmul_i8 with x_size=8) already gets.
  int64_t herd_max_x = 0;
  seg.walk([&](air::HerdOp h) {
    int64_t x_loc = 0;
    if (auto loc = h->getAttrOfType<IntegerAttr>("x_loc"))
      x_loc = loc.getInt();
    int64_t x_size = 1;
    auto sizeOperands = h.getSizeOperands();
    if (!sizeOperands.empty()) {
      if (auto cst = sizeOperands[0].getDefiningOp<arith::ConstantIndexOp>())
        x_size = cst.value();
      else if (auto cst = sizeOperands[0].getDefiningOp<arith::ConstantOp>())
        if (auto intAttr = dyn_cast_if_present<IntegerAttr>(cst.getValue()))
          x_size = intAttr.getInt();
    }
    herd_max_x = std::max(herd_max_x, x_loc + x_size);
  });
  seg_size_x = std::max(seg_size_x, herd_max_x);

  seg.walk([&](air::ChannelInterface op) {
    if (!aie_device.lookupSymbol(op.getChanName())) {
      auto ch = air::getChannelDeclarationThroughSymbol(op);
      builder.clone(*ch.getOperation());
    }
  });

  // use the command line offsets unless the attribute is present
  int64_t col_offset = options.col_offset;

  // Emit each memtile as an unplaced aie.logical_tile<MemTile>(col, ?) and
  // leave it logical. The downstream `aie-place-tiles` pass picks the row
  // (and may merge multiple LTOs onto one physical memtile when DMA capacity
  // permits). The column is constrained because the segment owns that column.
  //
  // Skip columns that have no memtile in this device (e.g., out-of-range
  // columns due to a too-large segment x_size + col_offset). The placer is
  // strict on out-of-range hints, so we filter here.
  const auto &targetModel = aie_device.getTargetModel();
  auto colHasMemTile = [&](int col) {
    if (col < 0 || col >= targetModel.columns())
      return false;
    for (int row = 0; row < targetModel.rows(); row++)
      if (targetModel.isMemTile(col, row))
        return true;
    return false;
  };

  // Emit one unhinted memtile LTO per logical memtile slot the segment
  // needs; aie-place-tiles assigns the col. The merge-ltos=false pass
  // option (set by aircc) keeps each LTO on its own physical memtile.
  SmallVector<AIE::LogicalTileOp> logicalMemTiles;
  for (auto x = 0; x < seg_size_x; x++) {
    auto phys_x = x + col_offset;
    if (!colHasMemTile(phys_x))
      continue;
    logicalMemTiles.push_back(AIE::LogicalTileOp::create(
        builder, aie_device.getLoc(), AIE::AIETileType::MemTile,
        /*col=*/IntegerAttr(),
        /*row=*/IntegerAttr(),
        /*allocation_scheme=*/StringAttr()));
  }

  // Anchor each emitted memtile with a tiny L2 buffer so it isn't folded
  // away before L2 allocation runs.
  auto memrefTy = MemRefType::get(SmallVector<int64_t>{1}, builder.getI8Type());
  static uint64_t BufferId = 0;
  for (auto tile : logicalMemTiles) {
    allocateBufferOp(BufferId, memrefTy, tile,
                     builder.getStringAttr("__L2_tmp"));
  }
  return success();
}

template <typename T>
void push_back_if_unique(std::vector<T> &vec, T entry) {
  if (std::find(vec.begin(), vec.end(), entry) == vec.end())
    vec.push_back(entry);
}

/// Collect all values that alias the given buffer through view-like operations.
/// This includes memref.subview, memref.collapse_shape, memref.expand_shape,
/// memref.reinterpret_cast, etc.
static void collectBufferAliases(Value buffer,
                                 llvm::SmallDenseSet<Value> &aliases) {
  aliases.insert(buffer);
  SmallVector<Value> worklist;
  worklist.push_back(buffer);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *user : current.getUsers()) {
      // Check if this is a view-like op that creates an alias
      Value result = nullptr;
      if (auto viewOp = dyn_cast_if_present<ViewLikeOpInterface>(user)) {
        result = viewOp->getResult(0);
      }
      if (result && !aliases.contains(result)) {
        aliases.insert(result);
        worklist.push_back(result);
      }
    }
  }
}

/// Allocate producer/consumer locks for shared L1 buffer synchronization.
/// This enables safe inter-core communication via shared L1 memory.
///
/// The function:
/// 1. Pre-analyzes buffer access patterns across ALL participating cores
/// 2. Determines if producer/consumer protocol is needed or if mutex suffices
/// 3. Allocates appropriate locks and inserts lock pairs around operations
///
/// Access pattern handling:
/// - Both readers and writers exist: Use producer/consumer lock protocol
/// - Only writers exist (write-only): Use mutex-style lock (same lock for
/// acq/rel)
/// - Only readers exist (read-only): Use mutex-style lock (same lock for
/// acq/rel)
/// - Neither exists: Skip lock insertion entirely
///
/// This prevents deadlocks that occur when the producer/consumer protocol is
/// used on buffers that only have reads OR only have writes (locks never
/// get replenished without the complementary operation).
///
/// \param aie_device The surrounding AIE device operation
/// \param sharedBuffer The shared L1 buffer allocated on the owner tile
/// \param ownerTile The tile that owns the shared buffer
/// \param coreOps Set of cores that use the shared buffer
static void
allocateSharedL1BufferLocks(AIE::DeviceOp aie_device,
                            AIE::BufferOp sharedBuffer, AIE::TileOp ownerTile,
                            const llvm::SmallSet<AIE::CoreOp, 2> &coreOps) {

  const auto &targetModel = aie_device.getTargetModel();
  bool usesSemaphoreLocks =
      targetModel.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks);

  std::string bufName = sharedBuffer.getSymName().value_or("shared_l1").str();

  // Helper lambda to find the last memref operand index in a func.call
  auto findLastMemrefOperandIndex = [](func::CallOp callOp) -> int {
    int lastMemrefIdx = -1;
    for (int i = callOp.getNumOperands() - 1; i >= 0; --i) {
      if (isa<MemRefType>(callOp.getOperand(i).getType())) {
        lastMemrefIdx = i;
        break;
      }
    }
    return lastMemrefIdx;
  };

  // ========================================================================
  // PRE-ANALYSIS: Determine buffer access patterns across ALL participating
  // cores to detect write-only or read-only scenarios that would deadlock
  // with the standard producer/consumer protocol.
  // ========================================================================
  bool hasAnyProducer = false; // Any write to buffer across all cores
  bool hasAnyConsumer = false; // Any read from buffer across all cores

  // Collect all aliases of the shared buffer
  llvm::SmallDenseSet<Value> bufferAliases;
  collectBufferAliases(sharedBuffer->getResult(0), bufferAliases);

  for (auto coreOp : coreOps) {
    coreOp.walk([&](Operation *op) {
      // Check memory effects
      for (Value alias : bufferAliases) {
        if (hasEffect<MemoryEffects::Write>(op, alias))
          hasAnyProducer = true;
        if (hasEffect<MemoryEffects::Read>(op, alias))
          hasAnyConsumer = true;
      }

      // Special handling for func.call
      if (auto callOp = dyn_cast_if_present<func::CallOp>(op)) {
        int lastMemrefIdx = findLastMemrefOperandIndex(callOp);
        for (int i = 0; i < (int)callOp.getNumOperands(); ++i) {
          Value operand = callOp.getOperand(i);
          if (bufferAliases.contains(operand)) {
            if (i == lastMemrefIdx)
              hasAnyProducer = true;
            else
              hasAnyConsumer = true;
          }
        }
      }
    });
  }

  // Count how many cores are producers (write to the shared buffer)
  int numProducerCores = 0;
  for (auto coreOp : coreOps) {
    bool coreIsProducer = false;
    coreOp.walk([&](Operation *op) {
      for (Value alias : bufferAliases) {
        if (hasEffect<MemoryEffects::Write>(op, alias))
          coreIsProducer = true;
      }
      if (auto callOp = dyn_cast_if_present<func::CallOp>(op)) {
        int lastMemrefIdx = findLastMemrefOperandIndex(callOp);
        for (int i = 0; i < (int)callOp.getNumOperands(); ++i) {
          if (bufferAliases.contains(callOp.getOperand(i))) {
            if (i == lastMemrefIdx)
              coreIsProducer = true;
          }
        }
      }
    });
    if (coreIsProducer)
      numProducerCores++;
  }

  // ========================================================================
  // 3-WAY DETECTION: collect the DMA accessors of the shared buffer. An
  // air.channel.put whose source aliases the buffer becomes an MM2S BD that
  // READS the buffer out (DMA reader); an air.channel.get whose dest aliases
  // the buffer becomes an S2MM BD that WRITES it (DMA writer). These are the
  // "third participant" the core-only analysis above misses: with N writer
  // cores + 1 DMA reader on one buffer, the DMA is the
  // effective consumer. (At outline time the put/get still lives in the core
  // body; it is hoisted to mem_X_Y later by lowerAIRMemcpyOp.) We keep the ops
  // themselves so the shared prod/cons identity can be stamped onto the op that
  // becomes the DMA BD (see the 3-way carrier stash below).
  SmallVector<Operation *> dmaAccessOps;
  bool hasDmaReader = false;
  bool hasDmaWriter = false;
  for (auto coreOp : coreOps) {
    coreOp.walk([&](Operation *op) {
      bool isPut = isa<air::ChannelPutOp>(op);
      bool isGet = isa<air::ChannelGetOp>(op);
      if (!isPut && !isGet)
        return;
      for (Value operand : op->getOperands())
        if (bufferAliases.contains(operand)) {
          dmaAccessOps.push_back(op);
          hasDmaReader |= isPut;
          hasDmaWriter |= isGet;
          break;
        }
    });
  }

  // 3-way share: the cores access the buffer ONE-SIDED (write-only or
  // read-only) AND a DMA participant supplies the missing side. This is the
  // shared-L1 pattern (N writer cores + 1 DMA reader). It UPGRADES the
  // would-be Mutex case to producer/consumer; it must NOT change the Skip
  // case (a buffer touched only by a DMA and never by core compute keeps its
  // locks from getLockForDMA as before). Requires AIE2 semaphore locks: the
  // DMA-side reuse in getLockForDMA / generateDmaBd is UsesSemaphoreLocks-
  // gated, so on AIE1 the DMA cannot share the prod/cons pair.
  bool coreOneSidedProducer = hasAnyProducer && !hasAnyConsumer;
  bool coreOneSidedConsumer = hasAnyConsumer && !hasAnyProducer;
  bool isThreeWay =
      usesSemaphoreLocks && ((coreOneSidedProducer && hasDmaReader) ||
                             (coreOneSidedConsumer && hasDmaWriter));

  // ========================================================================
  // DECISION: Determine lock strategy based on access patterns
  // ========================================================================
  enum class LockStrategy {
    Skip,  // No synchronization needed (no accesses)
    Mutex, // Single lock for mutual exclusion (write-only or read-only)
    ProducerConsumer // Standard producer/consumer protocol (mixed read/write)
  };

  LockStrategy strategy;
  if (!hasAnyProducer && !hasAnyConsumer) {
    // No CORE operations access the buffer - skip lock insertion entirely.
    // (A buffer touched only by DMA gets its locks from getLockForDMA.)
    strategy = LockStrategy::Skip;
    LLVM_DEBUG(llvm::dbgs()
               << "AIRToAIE: Skipping locks for shared L1 buffer " << bufName
               << " - no read/write operations detected\n");
  } else if ((hasAnyProducer && hasAnyConsumer) || isThreeWay) {
    // Both readers and writers exist - use producer/consumer protocol. This
    // covers the classic core-reader/core-writer case AND the 3-way case
    // (one-sided cores + DMA other side), where the DMA participant is the
    // counterpart that releases the lock and prevents the write-only deadlock
    // the mutex path guards against.
    strategy = LockStrategy::ProducerConsumer;
    LLVM_DEBUG(
        llvm::dbgs()
        << "AIRToAIE: Using producer/consumer locks for shared L1 buffer "
        << bufName << " - both readers and writers detected"
        << (isThreeWay ? " (3-way: includes DMA participant)" : "") << "\n");
  } else {
    // Only writers OR only readers AND no DMA counterpart - use mutex to
    // prevent deadlock
    strategy = LockStrategy::Mutex;
    // Emit warning about potential deadlock pattern that was avoided
    sharedBuffer->emitWarning()
        << "shared L1 buffer has "
        << (hasAnyProducer ? "write-only" : "read-only")
        << " access pattern; using mutex lock to avoid producer/consumer "
           "deadlock";
    LLVM_DEBUG(llvm::dbgs()
               << "AIRToAIE: Using mutex lock for shared L1 buffer " << bufName
               << " - " << (hasAnyProducer ? "write-only" : "read-only")
               << " access pattern detected (avoids deadlock)\n");
  }

  if (strategy == LockStrategy::Skip)
    return;

  // ========================================================================
  // LOCK ALLOCATION: Allocate locks based on chosen strategy
  // ========================================================================
  AIE::LockOp prodLock = nullptr;
  AIE::LockOp consLock = nullptr;
  AIE::LockOp mutexLock = nullptr;

  OpBuilder lockBuilder(ownerTile);

  if (strategy == LockStrategy::ProducerConsumer) {
    // Producer/consumer: init prod lock to number of producers so all can
    // write concurrently before consumer reads.
    int prodLockInit = std::max(numProducerCores, 1);
    int consLockInit = 0;

    prodLock =
        air::allocateLockOp(aie_device, ownerTile, prodLockInit, -1,
                            lockBuilder.getStringAttr(bufName + "_prod_lock"));
    consLock =
        air::allocateLockOp(aie_device, ownerTile, consLockInit, -1,
                            lockBuilder.getStringAttr(bufName + "_cons_lock"));

    LLVM_DEBUG(llvm::dbgs()
               << "AIRToAIE: Allocated producer/consumer locks for " << bufName
               << " - prod_lock(init=" << prodLockInit
               << "), cons_lock(init=" << consLockInit << ")\n");

    // 3-way carrier stash: when a DMA participant accesses this shared
    // buffer, the DMA BD's locks are allocated much later (getLockForDMA,
    // after lowerAIRMemcpyOp), in a phase that has no view of these
    // prod/cons locks. Stamp the lock symbol-refs onto the channel put/get
    // op that becomes the DMA BD (the op both phases touch), so getLockForDMA
    // reads them off the op it is already processing and reuses the SAME pair
    // (instead of allocating a private channel-put pair, which would break the
    // 3-way share with no error). The per-side token count N is the prod
    // lock's init; the DMA side reads it back via LockOp::getInit(), so no
    // separate count attribute is carried.
    if (isThreeWay) {
      auto *ctx = aie_device.getContext();
      auto prodRef = FlatSymbolRefAttr::get(ctx, prodLock.getSymName().value());
      auto consRef = FlatSymbolRefAttr::get(ctx, consLock.getSymName().value());
      for (auto *op : dmaAccessOps) {
        op->setAttr("air.shared_prod_lock", prodRef);
        op->setAttr("air.shared_cons_lock", consRef);
      }
    }
  } else {
    // Mutex: allocate single lock initialized to 1 (available)
    mutexLock =
        air::allocateLockOp(aie_device, ownerTile, /*init=*/1, -1,
                            lockBuilder.getStringAttr(bufName + "_mutex_lock"));

    LLVM_DEBUG(llvm::dbgs() << "AIRToAIE: Allocated mutex lock for " << bufName
                            << " - mutex_lock(init=1)\n");
  }

  AIE::LockAction acqAction = usesSemaphoreLocks
                                  ? AIE::LockAction::AcquireGreaterEqual
                                  : AIE::LockAction::Acquire;

  // ========================================================================
  // LOCK INSERTION: Insert lock pairs around buffer-accessing operations.
  // For inline MLIR kernels (memref.load/store in loops), locks are placed
  // at the outermost loop scope that contains ALL accesses, not per-op.
  //
  // Cross-core scope normalization: pre-fix, per-core lock scope was
  // chosen independently. When producer and consumer cores accessed the
  // buffer at different nesting depths (e.g. one inside an inner scf.for,
  // the other at core-body level), the cadence of acquire/release
  // mismatched (N-per-iter on one side, 1-per-iter on the other) →
  // permanent stall after the first iteration. Two passes now:
  //   (1) compute per-core (accessingOps, lockScope, roles)
  //   (2) detect asymmetric scope (any NULL while any non-NULL); if so,
  //       HOIST ALL cores to core-body level by wrapping the closest
  //       ancestor of each accessing op that's a direct child of the
  //       core op (== the core-body-level "statement"). This produces
  //       1 acquire/release per AIE-core iteration regardless of inner
  //       scf.for nesting and is the minimum shared-L1 invariant.
  //   (3) emit per-core locks using the chosen scope.
  // ========================================================================
  struct PerCoreLockInfo {
    AIE::CoreOp coreOp;
    SmallVector<Operation *> accessingOps;
    bool coreIsProducer;
    bool coreIsConsumer;
    Operation *lockScope;
  };
  SmallVector<PerCoreLockInfo> coreInfos;

  for (auto coreOp : coreOps) {
    PerCoreLockInfo info;
    info.coreOp = coreOp;
    info.coreIsProducer = false;
    info.coreIsConsumer = false;
    info.lockScope = nullptr;

    coreOp.walk([&](Operation *op) {
      bool accessesBuffer = false;
      bool isProducer = false;
      bool isConsumer = false;

      for (Value alias : bufferAliases) {
        if (hasEffect<MemoryEffects::Write>(op, alias)) {
          accessesBuffer = true;
          isProducer = true;
        }
        if (hasEffect<MemoryEffects::Read>(op, alias)) {
          accessesBuffer = true;
          isConsumer = true;
        }
      }

      if (auto callOp = dyn_cast_if_present<func::CallOp>(op)) {
        int lastMemrefIdx = findLastMemrefOperandIndex(callOp);
        for (int i = 0; i < (int)callOp.getNumOperands(); ++i) {
          Value operand = callOp.getOperand(i);
          if (bufferAliases.contains(operand)) {
            accessesBuffer = true;
            if (i == lastMemrefIdx)
              isProducer = true;
            else
              isConsumer = true;
          }
        }
      }

      if (accessesBuffer) {
        info.accessingOps.push_back(op);
        if (isProducer)
          info.coreIsProducer = true;
        if (isConsumer)
          info.coreIsConsumer = true;
      }
    });

    if (info.accessingOps.empty()) {
      coreInfos.push_back(info);
      continue;
    }

    // Find the OUTERMOST scf.for that contains ALL accessing ops
    Operation *candidate = info.accessingOps[0]->getParentOp();
    while (candidate && candidate != coreOp.getOperation()) {
      if (isa<scf::ForOp>(candidate)) {
        bool containsAll = true;
        for (auto *other : info.accessingOps) {
          if (!candidate->isProperAncestor(other)) {
            containsAll = false;
            break;
          }
        }
        if (containsAll)
          info.lockScope = candidate;
      }
      candidate = candidate->getParentOp();
    }
    coreInfos.push_back(info);
  }

  // Detect cross-core scope asymmetry that needs hoisting to core-body level.
  //
  // The genuine hazard is a core that accesses the buffer ONLY at core-body
  // level (no enclosing scf.for at all) paired with a core that accesses it
  // INSIDE a loop: the core-body-level participant cycles the shared lock once
  // per AIE-core iter while the loop-nested participant cycles it once per loop
  // iteration -> cadence mismatch / deadlock. Hoisting the loop-nested locks
  // out to core-body level makes both cycle once per AIE-core iter. This is
  // only correct when the core-body-level participant produces/consumes the
  // buffer ONCE (e.g. a whole-buffer write read N times); hoisting is what the
  // shared_l1_asymmetric_scope regression needs.
  //
  // It must NOT trigger when EVERY participant is loop-nested but merely lacks
  // a single common enclosing loop (e.g. a producer that writes the buffer in
  // a prologue AND inside its loop -> lockScope==null despite being
  // loop-nested). Such buffers are reused per loop iteration and require
  // per-iteration lock handoff; hoisting out of the loop would let the producer
  // overwrite the buffer N times before the consumer reads (correctness bug /
  // hang). So key the decision on core-body-only vs loop-nested, NOT on the
  // presence of a single common loop (lockScope null-ness).
  auto hasLoopNestedAccess = [&](PerCoreLockInfo &info) {
    for (auto *op : info.accessingOps) {
      for (Operation *p = op->getParentOp();
           p && p != info.coreOp.getOperation(); p = p->getParentOp())
        if (isa<scf::ForOp>(p))
          return true;
    }
    return false;
  };
  bool hoistAllToCoreBody = false;
  {
    bool anyCoreBodyOnly = false, anyLoopNested = false;
    for (auto &info : coreInfos) {
      if (info.accessingOps.empty())
        continue;
      if (hasLoopNestedAccess(info))
        anyLoopNested = true;
      else
        anyCoreBodyOnly = true;
    }
    if (anyCoreBodyOnly && anyLoopNested) {
      hoistAllToCoreBody = true;
      sharedBuffer->emitWarning()
          << "shared L1 buffer " << bufName
          << ": asymmetric per-core lock scopes detected (some cores "
             "access buffer at core-body level, others inside scf.for); "
             "hoisting all locks to core-body level to avoid cadence "
             "mismatch deadlock";
    }
  }

  for (auto &info : coreInfos) {
    auto coreOp = info.coreOp;
    auto &accessingOps = info.accessingOps;
    bool coreIsProducer = info.coreIsProducer;
    bool coreIsConsumer = info.coreIsConsumer;
    Operation *lockScope = hoistAllToCoreBody ? nullptr : info.lockScope;

    if (accessingOps.empty())
      continue;

    // 3-way shared L1: wrap each accessing op INDIVIDUALLY (per-buffer
    // bracketing) rather than hoisting acquire/release to the loop-body
    // boundaries. Required when multiple shared buffers are ping-ponged in
    // one loop body: allocateSharedL1BufferLocks runs once per shared buffer,
    // and the for-body-boundary placement would batch every buffer's prod
    // acquire at loop top (serializing ping vs pong). Per-op placement keeps
    // each buffer's lock around only its own access; cadence stays 1-per-iter
    // because the accessing op is inside the loop. Symmetric across cores
    // (all writers access at the same nesting), so no cadence-mismatch risk.
    if (isThreeWay) {
      for (auto *op : accessingOps) {
        OpBuilder builder(op);
        auto loc = op->getLoc();
        if (coreIsProducer && !coreIsConsumer) {
          AIE::UseLockOp::create(builder, loc, prodLock, acqAction, 1);
          builder.setInsertionPointAfter(op);
          AIE::UseLockOp::create(builder, loc, consLock,
                                 AIE::LockAction::Release, 1);
        } else if (coreIsConsumer && !coreIsProducer) {
          AIE::UseLockOp::create(builder, loc, consLock, acqAction,
                                 numProducerCores);
          builder.setInsertionPointAfter(op);
          AIE::UseLockOp::create(builder, loc, prodLock,
                                 AIE::LockAction::Release, numProducerCores);
        } else {
          AIE::UseLockOp::create(builder, loc, prodLock, acqAction, 1);
          AIE::UseLockOp::create(builder, loc, consLock, acqAction,
                                 numProducerCores);
          builder.setInsertionPointAfter(op);
          AIE::UseLockOp::create(builder, loc, consLock,
                                 AIE::LockAction::Release, 1);
          AIE::UseLockOp::create(builder, loc, prodLock,
                                 AIE::LockAction::Release, numProducerCores);
        }
      }
      continue;
    }

    // Step 3: Insert locks at the determined scope
    if (lockScope) {
      // Place at start/end of the outermost enclosing scf.for body
      auto forOp = cast<scf::ForOp>(lockScope);
      OpBuilder builder(forOp);
      builder.setInsertionPointToStart(forOp.getBody());
      auto loc = forOp.getLoc();

      if (strategy == LockStrategy::Mutex) {
        AIE::UseLockOp::create(builder, loc, mutexLock, acqAction, 1);
        builder.setInsertionPoint(forOp.getBody()->getTerminator());
        AIE::UseLockOp::create(builder, loc, mutexLock,
                               AIE::LockAction::Release, 1);
      } else if (coreIsProducer && !coreIsConsumer) {
        AIE::UseLockOp::create(builder, loc, prodLock, acqAction, 1);
        builder.setInsertionPoint(forOp.getBody()->getTerminator());
        AIE::UseLockOp::create(builder, loc, consLock, AIE::LockAction::Release,
                               1);
      } else if (coreIsConsumer && !coreIsProducer) {
        AIE::UseLockOp::create(builder, loc, consLock, acqAction,
                               numProducerCores);
        builder.setInsertionPoint(forOp.getBody()->getTerminator());
        AIE::UseLockOp::create(builder, loc, prodLock, AIE::LockAction::Release,
                               numProducerCores);
      } else {
        // Both producer and consumer
        AIE::UseLockOp::create(builder, loc, prodLock, acqAction, 1);
        AIE::UseLockOp::create(builder, loc, consLock, acqAction,
                               numProducerCores);
        builder.setInsertionPoint(forOp.getBody()->getTerminator());
        AIE::UseLockOp::create(builder, loc, consLock, AIE::LockAction::Release,
                               1);
        AIE::UseLockOp::create(builder, loc, prodLock, AIE::LockAction::Release,
                               numProducerCores);
      }
    } else {
      // Fallback: no single enclosing scf.for contains all accesses.
      //
      // When hoistAllToCoreBody is set (asymmetric cross-core scopes), wrap
      // the closest core-body-level ancestor of each accessing op (the direct
      // child of the AIE::CoreOp): for a core-body-level access that is the op
      // itself; for an op inside an scf.for it is the scf.for. This hoists the
      // acquire/release out of inner loops so every core cycles the shared
      // lock once per AIE-core iter, avoiding the cadence-mismatch deadlock.
      //
      // Otherwise preserve the original per-op wrapping (acquire/release
      // around each accessing op), which is the pre-existing behavior for
      // symmetric cores with no common enclosing loop.
      SetVector<Operation *> wrappingOps;
      if (hoistAllToCoreBody) {
        for (auto *op : accessingOps) {
          Operation *stmt = op;
          while (stmt && stmt->getParentOp() != coreOp.getOperation())
            stmt = stmt->getParentOp();
          if (stmt)
            wrappingOps.insert(stmt);
        }
      } else {
        for (auto *op : accessingOps)
          wrappingOps.insert(op);
      }

      for (auto *op : wrappingOps) {
        OpBuilder builder(op);

        if (strategy == LockStrategy::Mutex) {
          AIE::UseLockOp::create(builder, op->getLoc(), mutexLock, acqAction,
                                 1);
          builder.setInsertionPointAfter(op);
          AIE::UseLockOp::create(builder, op->getLoc(), mutexLock,
                                 AIE::LockAction::Release, 1);
        } else if (coreIsProducer && !coreIsConsumer) {
          AIE::UseLockOp::create(builder, op->getLoc(), prodLock, acqAction, 1);
          builder.setInsertionPointAfter(op);
          AIE::UseLockOp::create(builder, op->getLoc(), consLock,
                                 AIE::LockAction::Release, 1);
        } else if (coreIsConsumer && !coreIsProducer) {
          AIE::UseLockOp::create(builder, op->getLoc(), consLock, acqAction,
                                 numProducerCores);
          builder.setInsertionPointAfter(op);
          AIE::UseLockOp::create(builder, op->getLoc(), prodLock,
                                 AIE::LockAction::Release, numProducerCores);
        } else {
          // Both producer and consumer
          AIE::UseLockOp::create(builder, op->getLoc(), prodLock, acqAction, 1);
          AIE::UseLockOp::create(builder, op->getLoc(), consLock, acqAction,
                                 numProducerCores);
          builder.setInsertionPointAfter(op);
          AIE::UseLockOp::create(builder, op->getLoc(), consLock,
                                 AIE::LockAction::Release, 1);
          AIE::UseLockOp::create(builder, op->getLoc(), prodLock,
                                 AIE::LockAction::Release, numProducerCores);
        }
      }
    }
  }
}

LogicalResult createAIEModulesAndOutlineCores(
    ModuleOp module,
    std::vector<std::tuple<AIE::DeviceOp, air::HerdOp,
                           AIRToAIEConversionOptions>> &aie_modules,
    AIRToAIEConversionOptions &options) {

  SmallVector<air::SegmentOp> segments;
  SmallVector<air::HerdOp> herds;
  module.walk([&](air::SegmentOp s) { segments.push_back(s); });
  module.walk([&](air::HerdOp h) {
    if (h->getParentOfType<air::SegmentOp>())
      return;
    herds.push_back(h);
  });

  for (auto seg : segments) {
    // Validate segment unroll factors - Y-dimension unrolling is not supported
    if (failed(validateSegmentUnrollFactors(seg)))
      return failure();

    // Get segment unroll factors
    auto unrollFactors = getSegmentUnrollFactors(seg);
    if (failed(unrollFactors))
      return failure();
    auto [unrollX, unrollY] = *unrollFactors;
    int64_t totalUnroll = unrollX * unrollY;

    // Get the base device target model to compute sub-device type
    AIE::AIEDevice baseDevice = options.device;
    AIE::AIEDevice subDevice = computeSubDeviceType(baseDevice, totalUnroll);

    // Track the last created device op to maintain correct ordering
    AIE::DeviceOp lastDeviceOp = nullptr;

    // For each unroll iteration, create a separate aie.device
    for (int64_t uy = 0; uy < unrollY; uy++) {
      for (int64_t ux = 0; ux < unrollX; ux++) {
        std::string segment_name;
        if (auto attr = seg->getAttrOfType<StringAttr>(
                SymbolTable::getSymbolAttrName()))
          segment_name = attr.getValue().str();
        else
          segment_name = "segment_" + std::to_string(aie_modules.size());

        // Append unroll indices to segment name if unrolling
        if (totalUnroll > 1)
          segment_name += "_" + std::to_string(ux) + "_" + std::to_string(uy);

        std::string aie_module_name = "aie." + segment_name;
        OpBuilder builder(module.getContext());
        // Insert after the last device to maintain iteration order (0_0, 1_0,
        // etc.)
        if (lastDeviceOp)
          builder.setInsertionPointAfter(lastDeviceOp);
        else
          builder.setInsertionPointToStart(module.getBody());
        auto aie_dev = AIE::DeviceOp::create(
            builder, module.getLoc(),
            AIE::AIEDeviceAttr::get(builder.getContext(), subDevice));
        lastDeviceOp = aie_dev;
        aie_dev->setAttr(SymbolTable::getSymbolAttrName(),
                         StringAttr::get(builder.getContext(), segment_name));
        setAIEDeviceDataLayout(builder, aie_dev);
        AIE::DeviceOp::ensureTerminator(aie_dev.getRegion(), builder,
                                        aie_dev.getLoc());

        // Create a modified options for this iteration
        AIRToAIEConversionOptions iter_options = options;
        iter_options.device = subDevice;
        // For segment unroll, reset col_offset to 0 since each sub-device
        // has its own column space starting from 0
        if (totalUnroll > 1) {
          iter_options.col_offset = 0;
        }

        // Store unroll indices as attributes on the device for later use
        if (totalUnroll > 1) {
          aie_dev->setAttr("segment_unroll_x", builder.getI64IntegerAttr(ux));
          aie_dev->setAttr("segment_unroll_y", builder.getI64IntegerAttr(uy));
        }

        seg.walk([&](air::HerdOp h) {
          aie_modules.push_back({aie_dev, h, iter_options});
        });
        // If the device has memtiles, then outline memtiles
        if (aie_dev.getTargetModel().getNumMemTileRows()) {
          if (failed(outlineAIEMemtiles(builder, aie_dev, seg, iter_options)))
            return failure();
        }
      }
    }
  };

  for (auto herd : herds) {
    std::string segment_name;
    if (auto attr =
            herd->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      segment_name = attr.getValue().str();
    else
      segment_name = "herd_" + std::to_string(aie_modules.size());
    std::string aie_module_name = "aie." + segment_name;
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    auto aie_dev = AIE::DeviceOp::create(
        builder, module.getLoc(),
        AIE::AIEDeviceAttr::get(builder.getContext(), options.device));
    aie_dev->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(builder.getContext(), segment_name));
    setAIEDeviceDataLayout(builder, aie_dev);
    AIE::DeviceOp::ensureTerminator(aie_dev.getRegion(), builder,
                                    aie_dev.getLoc());
    aie_modules.push_back({aie_dev, herd, options});
  };
  for (auto &p : aie_modules) {
    auto aie_dev = std::get<0>(p);
    auto h = std::get<1>(p);
    auto device_options = std::get<2>(p);
    OpBuilder builder(aie_dev);
    if (failed(outlineAIECores(builder, aie_dev, h, device_options)))
      return failure();
  }
  // Outline any L1 memref allocs used by herds but located outside of any herd
  // This includes both local L1 buffers (used by single herd) and shared L1
  // buffers (used by multiple herds for inter-herd communication).
  std::vector<Value> sharedL1Memrefs;
  for (auto &p : aie_modules) {
    auto h = std::get<1>(p);
    for (unsigned i = 0; i < h.getNumKernelOperands(); i++) {
      auto oper = h.getKernelOperand(i);
      if (!oper.getDefiningOp())
        continue;
      auto memrefTy = llvm::dyn_cast_if_present<MemRefType>(oper.getType());
      if (!memrefTy)
        continue;
      if (!air::isL1(memrefTy))
        continue;
      push_back_if_unique<Value>(sharedL1Memrefs, oper);
    }
  }

  // Map each L1 memref alloc to the set of cores that use it
  std::map<Operation *, llvm::SmallSet<AIE::CoreOp, 2>> sharedL1AllocsToCoreMap;
  for (auto memref : sharedL1Memrefs) {
    for (auto user : memref.getUsers()) {
      auto coreOp = user->getParentOfType<AIE::CoreOp>();
      if (!coreOp)
        continue;
      sharedL1AllocsToCoreMap[memref.getDefiningOp()].insert(coreOp);
    }
  }

  // Map each L1 memref alloc to the set of herds that use it.
  // This distinguishes truly shared L1 buffers (used by multiple herds for
  // inter-herd communication) from local L1 buffers that appear in multiple
  // cores only because a single herd was unrolled into multiple cores.
  std::map<Operation *, llvm::SmallSet<air::HerdOp, 2>> sharedL1AllocsToHerdMap;
  for (auto &p : aie_modules) {
    auto h = std::get<1>(p);
    for (unsigned i = 0; i < h.getNumKernelOperands(); i++) {
      auto oper = h.getKernelOperand(i);
      if (!oper.getDefiningOp())
        continue;
      auto memrefTy = llvm::dyn_cast_if_present<MemRefType>(oper.getType());
      if (!memrefTy)
        continue;
      if (!air::isL1(memrefTy))
        continue;
      sharedL1AllocsToHerdMap[oper.getDefiningOp()].insert(h);
    }
  }

  // Process L1 memrefs: distinguish between local and shared cases.
  // - Local L1: Used by a single herd (even if multi-core). Clone the alloc
  //             into each core.
  // - Shared L1: Used by multiple herds. Allocate a single aie.buffer on one
  //              tile, and have all cores reference the same buffer. This
  //              enables shared L1 memory communication between neighboring
  //              AIE tiles.
  uint64_t sharedBufferId = 0;
  for (auto memref : sharedL1Memrefs) {
    auto &coreOps = sharedL1AllocsToCoreMap[memref.getDefiningOp()];
    auto &herds = sharedL1AllocsToHerdMap[memref.getDefiningOp()];

    if (herds.size() > 1) {
      // SHARED L1 BUFFER: Multiple herds use the same memref
      // Allocate a single aie.buffer on one "owner" tile and have all cores
      // reference it. The downstream mlir-aie compiler will validate that
      // the tiles are adjacent and can share L1 memory.

      // Find a tile that has legal memory affinity with all other cores.
      // Use AIE target model's isLegalMemAffinity() to validate placement.
      AIE::TileOp ownerTile = nullptr;
      for (auto coreOp : coreOps) {
        AIE::TileOp candidateTile = coreOp.getTileOp();
        auto aie_device = candidateTile->getParentOfType<AIE::DeviceOp>();
        const auto &targetModel = aie_device.getTargetModel();
        int memCol = candidateTile.getCol();
        int memRow = candidateTile.getRow();

        // Check if this tile has legal memory affinity with all other cores
        bool validForAll = true;
        for (auto otherCore : coreOps) {
          AIE::TileOp otherTile = otherCore.getTileOp();
          int coreCol = otherTile.getCol();
          int coreRow = otherTile.getRow();
          if (!targetModel.isLegalMemAffinity(coreCol, coreRow, memCol,
                                              memRow)) {
            validForAll = false;
            break;
          }
        }
        if (validForAll) {
          ownerTile = candidateTile;
          break;
        }
      }

      // If no tile has legal memory affinity with all cores, the cores are
      // not adjacent and cannot share L1 memory. Fall back to the local path
      // where each core gets its own copy of the buffer.
      if (!ownerTile) {
        for (auto coreOp : coreOps) {
          OpBuilder builder(coreOp);
          builder.setInsertionPointToStart(&coreOp.getBody().front());
          auto outlinedL1Alloc = builder.clone(*memref.getDefiningOp());
          for (unsigned i = 0; i < memref.getDefiningOp()->getNumResults(); i++)
            replaceAllUsesInRegionWith(memref.getDefiningOp()->getResult(i),
                                       outlinedL1Alloc->getResult(i),
                                       coreOp.getBody());
        }
        continue;
      }

      // Get the memref type for buffer allocation
      auto memrefTy = llvm::cast<MemRefType>(memref.getType());

      // Create a single shared buffer on the owner tile
      OpBuilder bufBuilder(ownerTile);
      Operation *t = ownerTile.getOperation();
      while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
        t = t->getNextNode();
      bufBuilder.setInsertionPointAfter(t);

      AIE::BufferOp sharedBuffer =
          allocateBufferOp(sharedBufferId, memrefTy, ownerTile,
                           bufBuilder.getStringAttr("shared_l1"));

      LLVM_DEBUG(llvm::dbgs()
                 << "AIRToAIE: Created shared L1 buffer "
                 << sharedBuffer.getSymName().value_or("unnamed") << " on tile("
                 << ownerTile.getCol() << ", " << ownerTile.getRow() << ") for "
                 << coreOps.size() << " cores\n");

      // Replace uses in ALL cores with the SAME shared buffer
      for (auto coreOp : coreOps) {
        for (unsigned i = 0; i < memref.getDefiningOp()->getNumResults(); i++) {
          replaceAllUsesInRegionWith(memref.getDefiningOp()->getResult(i),
                                     sharedBuffer->getResult(0),
                                     coreOp.getBody());
        }
      }

      // Allocate producer/consumer locks for shared L1 buffer synchronization
      auto aie_device = ownerTile->getParentOfType<AIE::DeviceOp>();
      allocateSharedL1BufferLocks(aie_device, sharedBuffer, ownerTile, coreOps);
    } else {
      // LOCAL L1 BUFFER: Single core uses the memref
      // Clone the alloc into the core
      for (auto coreOp : coreOps) {
        OpBuilder builder(coreOp);
        builder.setInsertionPointToStart(&coreOp.getBody().front());
        auto outlinedL1Alloc = builder.clone(*memref.getDefiningOp());
        for (unsigned i = 0; i < memref.getDefiningOp()->getNumResults(); i++)
          replaceAllUsesInRegionWith(memref.getDefiningOp()->getResult(i),
                                     outlinedL1Alloc->getResult(i),
                                     coreOp.getBody());
      }
    }
  }
  return success();
}

bool isInSet(IntegerSet is) {
  auto constraints = is.getConstraints();
  auto eqFlags = is.getEqFlags();

  int i = 0;
  for (auto c : constraints) {
    auto expr =
        dyn_cast_if_present<AffineConstantExpr>(simplifyAffineExpr(c, 0, 1));
    if (!expr)
      return false;
    if (eqFlags[i++]) {
      if (expr.getValue() != 0)
        return false;
    } else {
      if (expr.getValue() < 0)
        return false;
    }
  }

  return true;
}

bool isInSet(int64_t x, int64_t y, affine::AffineIfOp aif) {
  auto is = aif.getIntegerSet();

  SmallVector<AffineExpr, 2> dims{
      getAffineConstantExpr(x, aif->getContext()),
      getAffineConstantExpr(y, aif->getContext()),
  };

  auto newIs = is.replaceDimsAndSymbols({}, dims, 0, 2);
  return isInSet(newIs);
}

struct SpecializeAffineIfPattern : public OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;

  SpecializeAffineIfPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(affine::AffineIfOp op,
                                PatternRewriter &rewriter) const override {

    // Allow specialization inside both AIE cores and at the segment level
    // within a device (for segment-unrolled affine.if ops on unroll indices).
    if (!op->getParentOfType<AIE::CoreOp>() &&
        !op->getParentOfType<AIE::DeviceOp>())
      return failure();

    bool in_set = false;
    if (op.getNumOperands() == 2) {
      SmallVector<int64_t, 2> operands;
      for (auto o : op.getOperands()) {
        if (auto v =
                dyn_cast_if_present<arith::ConstantIndexOp>(o.getDefiningOp()))
          operands.push_back(v.value());
        else if (auto v =
                     dyn_cast_if_present<arith::RemSIOp>(o.getDefiningOp())) {
          if (mlir::getConstantIntValue(v.getLhs()) &&
              mlir::getConstantIntValue(v.getRhs())) {
            int lhs = *mlir::getConstantIntValue(v.getLhs());
            int rhs = *mlir::getConstantIntValue(v.getRhs());
            operands.push_back(llvm::mod(lhs, rhs));
          } else
            return failure();
        } else if (auto v =
                       dyn_cast_if_present<arith::DivSIOp>(o.getDefiningOp())) {
          if (mlir::getConstantIntValue(v.getLhs()) &&
              mlir::getConstantIntValue(v.getRhs())) {
            int lhs = *mlir::getConstantIntValue(v.getLhs());
            int rhs = *mlir::getConstantIntValue(v.getRhs());
            operands.push_back(llvm::divideFloorSigned(lhs, rhs));
          } else
            return failure();
        } else
          return failure();
      }
      auto x = operands[0];
      auto y = operands[1];
      in_set = isInSet(x, y, op);
    } else {
      in_set = isInSet(op.getIntegerSet());
    }

    Block *bb = nullptr;
    if (in_set) {
      bb = op.getThenBlock();
    } else if (op.hasElse()) {
      bb = op.getElseBlock();
    }
    if (bb) {
      auto t = bb->getTerminator();
      auto &ops = bb->getOperations();
      op->getBlock()->getOperations().splice(Block::iterator(op), ops,
                                             ops.begin(), --ops.end());
      for (int i = 0, e = op.getNumResults(); i < e; i++)
        op.getResult(i).replaceAllUsesWith(t->getOperand(i));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct SpecializeScfIfPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  SpecializeScfIfPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {

    // Allow specialization inside both AIE cores and at the segment level
    // within a device (for segment-unrolled scf.if ops on unroll indices).
    if (!op->getParentOfType<AIE::CoreOp>() &&
        !op->getParentOfType<AIE::DeviceOp>())
      return failure();

    // Try to resolve the condition to a constant boolean.
    Value cond = op.getCondition();
    std::optional<bool> condValue;

    // Case 1: condition is a constant i1.
    if (auto constOp = cond.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast_if_present<IntegerAttr>(constOp.getValue()))
        condValue = intAttr.getValue().getBoolValue();
    }
    // Case 2: condition is arith.cmpi with constant operands.
    else if (auto cmpOp = cond.getDefiningOp<arith::CmpIOp>()) {
      auto lhsConst = mlir::getConstantIntValue(cmpOp.getLhs());
      auto rhsConst = mlir::getConstantIntValue(cmpOp.getRhs());
      if (lhsConst && rhsConst) {
        // Use 64-bit APInt for index types (which have no fixed bit width),
        // and the actual bit width for integer types.
        unsigned bitWidth =
            isa<IndexType>(cmpOp.getLhs().getType())
                ? 64
                : cmpOp.getLhs().getType().getIntOrFloatBitWidth();
        APInt lhs(bitWidth, *lhsConst, /*isSigned=*/true);
        APInt rhs(bitWidth, *rhsConst, /*isSigned=*/true);
        condValue = arith::applyCmpPredicate(cmpOp.getPredicate(), lhs, rhs);
      }
    }
    // Case 3: condition is arith.index_cast of a constant to i1.
    else if (auto castOp = cond.getDefiningOp<arith::IndexCastOp>()) {
      if (auto constVal = mlir::getConstantIntValue(castOp.getIn()))
        condValue = (*constVal != 0);
    }

    if (!condValue)
      return failure();

    Block *bb = nullptr;
    if (*condValue) {
      bb = op.thenBlock();
    } else if (op.elseBlock()) {
      bb = op.elseBlock();
    }
    if (bb) {
      auto t = bb->getTerminator();
      auto &ops = bb->getOperations();
      op->getBlock()->getOperations().splice(Block::iterator(op), ops,
                                             ops.begin(), --ops.end());
      for (int i = 0, e = op.getNumResults(); i < e; i++)
        op.getResult(i).replaceAllUsesWith(t->getOperand(i));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void specializeHerdAffineIf(AIE::DeviceOp m) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<SpecializeAffineIfPattern>(ctx);
  patterns.insert<SpecializeScfIfPattern>(ctx);
  (void)applyPatternsGreedily(m, std::move(patterns));
}

struct LowerAIRExecutePattern : public OpRewritePattern<air::ExecuteOp> {
  using OpRewritePattern<air::ExecuteOp>::OpRewritePattern;

  LowerAIRExecutePattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::ExecuteOp op,
                                PatternRewriter &rewriter) const override {
    auto &bb = op.getRegion().front();
    unsigned idx = 0;
    for (auto &arg : bb.getArguments()) {
      arg.replaceAllUsesWith(op.getOperand(idx));
      idx++;
    }
    if (op.getAsyncDependencies().size()) {
      air::WaitAllOp::create(rewriter, op->getLoc(), Type{},
                             op.getAsyncDependencies());
    }
    if (op.getNumResults() > 0) {
      rewriter.setInsertionPointAfter(op);
      auto w = air::WaitAllOp::create(
          rewriter, op->getLoc(), air::AsyncTokenType::get(op->getContext()),
          SmallVector<Value, 1>{});
      op.getResult(0).replaceAllUsesWith(w.getResult(0));
    }
    op.walk([&](air::ExecuteTerminatorOp t) {
      int resultIdx = 1;
      for (auto r : t->getOperands())
        op.getResult(resultIdx++).replaceAllUsesWith(r);
    });
    auto &ops = bb.getOperations();
    op->getBlock()->getOperations().splice(Block::iterator(op), ops,
                                           ops.begin(), --ops.end());

    rewriter.eraseOp(op);
    return success();
  }
};

void lowerAirExecute(AIE::DeviceOp d) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  int maxSize = isa<AIE::AIE1TargetModel>(AIE::getTargetModel(d)) ? -1 : 1023;
  int maxNumDims = isa<AIE::AIE1TargetModel>(AIE::getTargetModel(d)) ? 1 : 4;
  patterns.insert<LowerAIRExecutePattern>(ctx);
  bool enableRepeatAtHighestDim = false;
  air::populateAIRCanonicalizeChannelWrapAndStridePatterns(
      patterns, maxSize, maxNumDims, enableRepeatAtHighestDim);
  (void)applyPatternsGreedily(d, std::move(patterns));
}

struct LowerScfTokenPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LowerScfTokenPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp fop,
                                PatternRewriter &rewriter) const override {

    if (!fop.getInitArgs().size())
      return failure();

    SmallVector<Value, 4> iter_args;
    BitVector iter_args_idx(fop.getNumOperands());

    // erase air.event from the iter args
    for (auto block_arg : fop.getRegionIterArgs()) {
      Value v =
          fop.getOperand(block_arg.getArgNumber() - fop.getNumInductionVars() +
                         fop.getNumControlOperands());
      if (llvm::isa<air::AsyncTokenType>(v.getType())) {
        block_arg.replaceAllUsesWith(v);
        iter_args_idx.set(block_arg.getArgNumber());
      } else {
        iter_args.push_back(v);
      }
    }

    // if none of the iter args were air.async.token, return
    if (iter_args.size() == fop.getInitArgs().size())
      return failure();

    // make a new scf.for without air.async.token
    IRMapping remap;
    auto new_fop =
        scf::ForOp::create(rewriter, fop->getLoc(), fop.getLowerBound(),
                           fop.getUpperBound(), fop.getStep(), iter_args);
    auto &new_region = new_fop.getRegion();
    fop.getRegion().cloneInto(&new_region, new_region.begin(), remap);
    new_region.back().erase();
    new_region.front().eraseArguments(iter_args_idx);

    // copy ping-pong pattern flags over to the new scf.for
    if (fop->hasAttr("isolated")) {
      new_fop->setAttr("isolated", fop->getAttr("isolated"));
    }
    if (fop->hasAttr("unroll")) {
      new_fop->setAttr("unroll", fop->getAttr("unroll"));
    }
    // Preserve user-attached loop_annotation (e.g., unroll-disable for
    // Peano). Without this, Peano -O2 aggressively unrolls small-trip
    // loops and blows program memory on AIE2P.
    if (fop->hasAttr("loop_annotation")) {
      new_fop->setAttr("loop_annotation", fop->getAttr("loop_annotation"));
    }
    // Preserve the user-facing ping-pong opt-out so labeling later in
    // the pipeline still sees it.
    if (fop->hasAttr("air.disable_ping_pong")) {
      new_fop->setAttr("air.disable_ping_pong",
                       fop->getAttr("air.disable_ping_pong"));
    }

    // use the new for op's results
    int idx = 0;
    for (auto r : fop.getResults()) {
      if (llvm::isa<air::AsyncTokenType>(r.getType()))
        r.replaceAllUsesWith(
            air::WaitAllOp::create(rewriter, fop->getLoc(),
                                   air::AsyncTokenType::get(fop->getContext()),
                                   SmallVector<Value, 1>{})
                .getResult(0));
      else
        r.replaceAllUsesWith(new_fop.getResult(idx++));
    }

    // remove air.async.token from the yield op
    auto yield = new_region.back().getTerminator();
    rewriter.setInsertionPoint(yield);
    SmallVector<Value, 4> yield_operands;
    SmallVector<Value, 4> token_operands;
    for (auto o : yield->getOperands()) {
      if (llvm::isa<air::AsyncTokenType>(o.getType()))
        token_operands.push_back(o);
      else
        yield_operands.push_back(o);
    }
    air::WaitAllOp::create(rewriter, fop->getLoc(), SmallVector<Type, 1>{},
                           token_operands);
    scf::YieldOp::create(rewriter, yield->getLoc(), yield_operands);
    rewriter.eraseOp(yield);

    rewriter.eraseOp(fop);
    return success();
  }
};

struct AttachMustProgressPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  AttachMustProgressPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp fop,
                                PatternRewriter &rewriter) const override {

    // Check if the attribute is already present
    // if (fop->hasAttr("llvm.loop"))
    if (fop->hasAttr("loop_annotation"))
      return failure();

    // Create the loop annotation attribute with mustProgress = true
    auto ctx = fop->getContext();
    auto mustProgressAttr = LLVM::LoopAnnotationAttr::get(
        ctx,
        /*disableNonforced=*/nullptr,
        /*vectorize=*/nullptr,
        /*interleave=*/nullptr,
        /*unroll=*/nullptr,
        /*unrollAndJam=*/nullptr,
        /*licm=*/nullptr,
        /*distribute=*/nullptr,
        /*pipeline=*/nullptr,
        /*peeled=*/nullptr,
        /*unswitch=*/nullptr,
        /*mustProgress=*/rewriter.getBoolAttr(true),
        /*isVectorized=*/nullptr,
        /*startLoc=*/nullptr,
        /*endLoc=*/nullptr,
        /*parallelAccesses=*/{});

    // Set the attribute on the for loop
    fop->setAttr("loop_annotation", mustProgressAttr);

    return success();
  }
};

void lowerScfAirTokens(AIE::DeviceOp m) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<LowerScfTokenPattern>(ctx);
  patterns.insert<AttachMustProgressPattern>(ctx);
  (void)applyPatternsGreedily(m, std::move(patterns));
}

struct AllocL1BuffersPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  AllocL1BuffersPattern(MLIRContext *ctx, uint64_t &bufferId)
      : OpRewritePattern(ctx), BufferId(bufferId) {}

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {

    AIE::CoreOp core = alloc->getParentOfType<AIE::CoreOp>();
    if (!core)
      return failure();

    AIE::TileOp tile = core.getTileOp();
    if (!tile)
      return failure();

    MemRefType memrefTy = nullptr;
    memrefTy = alloc.getType();

    if (!air::isL1(memrefTy))
      return failure();

    // Read herd-local (x, y) from the aie.core attribute set at outline
    // time (RFC #1567 Stage C #3). Fall back to physical coords when the
    // attribute isn't present (e.g. cores not produced by outlineAIECores).
    int64_t herd_local_x = tile.getCol();
    int64_t herd_local_y = tile.getRow();
    if (auto idAttr =
            core->getAttrOfType<DenseI64ArrayAttr>("air.herd_local_id")) {
      auto ids = idAttr.asArrayRef();
      if (ids.size() == 2) {
        herd_local_x = ids[0];
        herd_local_y = ids[1];
      }
    }

    auto buffer = allocateBufferOp(
        BufferId, memrefTy, tile,
        alloc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        herd_local_x, herd_local_y);

    rewriter.replaceOp(alloc, buffer->getResults());
    return success();
  }

private:
  uint64_t &BufferId;
};

struct AllocL2BuffersPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  AllocL2BuffersPattern(
      MLIRContext *ctx,
      std::map<memref::AllocOp, AIE::TileLike> &memrefToTileMap,
      std::map<AIE::BufferOp, AIE::TileLike> &bufferToMemtileMap,
      uint64_t &bufferId)
      : OpRewritePattern(ctx), memrefToTileMap(memrefToTileMap),
        BufferId(bufferId), bufferToMemtileMap(bufferToMemtileMap) {}

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {

    // L2 memref allocs should exist inside of device op but outside of core op
    AIE::DeviceOp device = alloc->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();
    AIE::CoreOp core = alloc->getParentOfType<AIE::CoreOp>();
    if (core)
      return failure();

    MemRefType memrefTy = nullptr;
    memrefTy = alloc.getType();

    if (!air::isL2(memrefTy))
      return failure();

    // Allocation of L2 memrefs in segment to buffer ops
    if (!memrefToTileMap.count(alloc)) {
      alloc->emitOpError("alloc not found in memrefToTileMap.");
      return failure();
    }
    AIE::TileLike tile = memrefToTileMap[alloc];
    if (!tile)
      return failure();

    auto seg = alloc->getParentOfType<air::SegmentOp>();
    int64_t col_offset = 0;
    int64_t row_offset = 0;
    if (seg) {
      auto c = seg.getColOffset();
      auto r = seg.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    }
    // For unplaced memtiles (LogicalTileOp before aie-place-tiles runs)
    // tryGetCol/Row return nullopt; the buffer name suffix falls back to -1.
    int64_t tileCol = tile.tryGetCol().value_or(0);
    int64_t tileRow = tile.tryGetRow().value_or(0);
    AIE::BufferOp buffer = allocateBufferOp(
        BufferId, memrefTy, tile,
        alloc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        tileCol - col_offset, tileRow - row_offset);

    // Propagate the memtile L2 re-broadcast directive (mechanism 2: N re-reads
    // of this resident buffer per fill) onto the buffer op, the shared
    // fill/drain rendezvous that generateDmaBd / getLockForDMA read on the fill
    // (S2MM) side. This count is carried on the ALLOC and is distinct from any
    // channel-level re-feed (mechanism 1: the core-producer count on the
    // channel, applied in allocateCoreLocksPerMemcpyOp). air-to-aie skips the
    // channel re-feed for memtile producers, so the two must NOT be unified --
    // read only the alloc directive here.
    int64_t refeedN = air::getRefeedCount(alloc.getOperation());
    if (refeedN > 1)
      buffer->setAttr(air::attrs::RefeedCount,
                      rewriter.getI32IntegerAttr(refeedN));

    rewriter.replaceOp(alloc, buffer->getResults());
    bufferToMemtileMap[buffer] = tile;
    return success();
  }

private:
  std::map<memref::AllocOp, AIE::TileLike> &memrefToTileMap;
  uint64_t &BufferId;
  std::map<AIE::BufferOp, AIE::TileLike> &bufferToMemtileMap;
};

void allocL1Buffers(AIE::DeviceOp m, uint64_t &BufferId) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<AllocL1BuffersPattern>(ctx, BufferId);
  // AllocL1TensorsPattern
  (void)applyPatternsGreedily(m, std::move(patterns));
}

bool areReferencedByTheSameAIRChannel(Value memref_a, Value memref_b) {
  for (auto user_a : memref_a.getUsers()) {
    for (auto user_b : memref_b.getUsers()) {
      auto chan_user_a = dyn_cast_if_present<air::ChannelInterface>(user_a);
      auto chan_user_b = dyn_cast_if_present<air::ChannelInterface>(user_b);
      if (!chan_user_a || !chan_user_b)
        continue;
      if (chan_user_a.getChanName().str() != chan_user_b.getChanName().str())
        continue;
      if (chan_user_a.getIndices().size() != chan_user_b.getIndices().size())
        continue;

      bool hasIdenticalIndices = true;
      for (unsigned i = 0; i < chan_user_a.getIndices().size(); i++) {
        if (*getConstantIntValue(chan_user_a.getIndices()[i]) !=
            *getConstantIntValue(chan_user_b.getIndices()[i]))
          hasIdenticalIndices = false;
      }
      if (hasIdenticalIndices)
        return true;
    }
  }
  return false;
}

void L2MemrefToMemTileMap(
    AIE::DeviceOp m,
    std::map<memref::AllocOp, AIE::TileLike> &memrefToMemTileMap) {
  std::vector<memref::AllocOp> allocs;
  m.walk([&](memref::AllocOp alloc) {
    if (air::isL2(llvm::cast<MemRefType>(alloc.getMemref().getType()))) {
      allocs.push_back(alloc);
    }
  });
  std::vector<AIE::TileLike> memtiles = getMemtilesFromDeviceOp(m);
  if (memtiles.empty()) {
    if (!allocs.empty())
      m.emitWarning("L2 memrefs present but no memtiles available; skipping "
                    "memtile placement");
    return;
  }

  // First stage in memref placement: grouping memrefs referenced by the same
  // air.channel.
  SmallVector<SmallVector<memref::AllocOp>> memref_buckets;
  auto placeMemrefInSharedBucket =
      [&](SmallVector<SmallVector<memref::AllocOp>> &memref_buckets,
          memref::AllocOp alloc) {
        for (auto &bucket : memref_buckets) {
          for (auto bucket_elem : bucket) {
            if (areReferencedByTheSameAIRChannel(alloc.getMemref(),
                                                 bucket_elem.getMemref())) {
              bucket.push_back(alloc);
              return true;
            }
          }
        }
        return false;
      };
  for (auto alloc : allocs) {
    if (!placeMemrefInSharedBucket(memref_buckets, alloc)) {
      memref_buckets.push_back(SmallVector<memref::AllocOp>{alloc});
    }
  }
  // Second stage in memref placement: try column-affinity assignment, but
  // fall back to round-robin when any one column would be saturated.
  //
  // Col-affinity (preferred): each bucket targets the column of its
  // consumer core. Memtile LTO at col X holds buckets whose consumer
  // herds are at col X. The placer with merge-logical-tiles=false lands
  // each col-hinted LTO at its physical memtile, so round-trip
  // core->memtile flow stays intra-column. Avoids the matmul_i8
  // Triton-XDNA#50 pathfinder latency regression from cross-col flows.
  //
  // Saturation fallback: when all consumer herds share one column (e.g.
  // flash_attention with 6 herds all defaulting to col 0), concentrating
  // every bucket onto a single memtile exhausts its 16-lock budget. If
  // any column would receive more buckets than one memtile can hold,
  // revert to round-robin across the full LTO pool.
  const auto &targetModel = m.getTargetModel();
  auto colHasMemTile = [&](int col) {
    if (col < 0 || col >= targetModel.columns())
      return false;
    for (int row = 0; row < targetModel.rows(); row++)
      if (targetModel.isMemTile(col, row))
        return true;
    return false;
  };
  auto otherSideCoreCol = [&](StringRef channelName, bool weAreThePut) -> int {
    int col = -1;
    m.walk([&](air::ChannelInterface chIf) {
      if (chIf.getChanName() != channelName)
        return WalkResult::advance();
      bool isOurSide = (weAreThePut == isa<air::ChannelPutOp>(*chIf));
      if (isOurSide)
        return WalkResult::advance();
      auto core = chIf->getParentOfType<AIE::CoreOp>();
      if (!core)
        return WalkResult::advance();
      auto tile =
          dyn_cast_if_present<AIE::TileOp>(core.getTile().getDefiningOp());
      if (!tile)
        return WalkResult::advance();
      int c = tile.getCol();
      if (col == -1)
        col = c;
      else if (col != c)
        col = -2;
      return WalkResult::advance();
    });
    return col < 0 ? -1 : col;
  };
  auto deriveBucketCol = [&](ArrayRef<memref::AllocOp> bucket) -> int {
    int consensus = -1;
    for (auto alloc : bucket) {
      for (auto user : alloc.getMemref().getUsers()) {
        auto ch = dyn_cast_if_present<air::ChannelInterface>(user);
        if (!ch)
          continue;
        bool weArePut = isa<air::ChannelPutOp>(*ch);
        int cand = otherSideCoreCol(ch.getChanName(), weArePut);
        if (cand < 0)
          continue;
        if (consensus == -1)
          consensus = cand;
        else if (consensus != cand)
          return -1;
      }
    }
    return colHasMemTile(consensus) ? consensus : -1;
  };

  SmallVector<int> bucketCols;
  bucketCols.reserve(memref_buckets.size());
  llvm::DenseMap<int, int> colDemand;
  for (auto &bucket : memref_buckets) {
    int col = deriveBucketCol(bucket);
    bucketCols.push_back(col);
    if (col >= 0)
      colDemand[col]++;
  }

  // Saturation check: col-affinity is only safe when each column would
  // receive at most one bucket. With more, downstream allocation tends
  // to ping-pong L1 buffers to keep up with the concentrated L2->L1 flow
  // (flash_attention/dataflow_based: 9 buckets across 3 cols → col 0
  // gets 5-6 buckets → L1 doubled to 6 buffers on tile (1,2) → 65536B,
  // over the 64KB budget). Fall back to round-robin across the full LTO
  // pool to match the pre-Path-B distribution that flash_attention
  // relied on, while still keeping the matvec case (1 bucket per col)
  // on the col-affinity path.
  constexpr int kBucketsPerMemtileBudget = 1;
  bool saturated = false;
  for (auto &kv : colDemand)
    if (kv.second > kBucketsPerMemtileBudget) {
      saturated = true;
      break;
    }

  if (saturated) {
    // Multi-bucket shapes (operand classes like A/B/C) get a per-shape
    // counter so each class restarts at memtile 0 and bucket-i maps to
    // memtile-i; singleton shapes keep using a global counter so unrelated
    // one-off buffers still spread across the pool.
    auto bucketShape = [](SmallVectorImpl<memref::AllocOp> &bucket) -> Type {
      return bucket.empty() ? Type() : bucket.front().getMemref().getType();
    };
    llvm::DenseMap<Type, int> shapeCount;
    for (auto &bucket : memref_buckets)
      shapeCount[bucketShape(bucket)]++;
    llvm::DenseMap<Type, int> perShapeCounter;
    int globalCounter = 0;
    for (auto &bucket : memref_buckets) {
      Type shape = bucketShape(bucket);
      int slot =
          (shapeCount[shape] > 1) ? perShapeCounter[shape]++ : globalCounter++;
      auto memtile = memtiles[slot % memtiles.size()];
      for (auto bucket_elem : bucket)
        memrefToMemTileMap[bucket_elem] = memtile;
    }
    return;
  }

  // Col-affinity assignment: hint pre-emitted LTOs to derived cols and
  // share one LTO per col. Spill over to round-robin for unhinted buckets.
  OpBuilder builder(m);
  builder.setInsertionPointToStart(m.getBody());
  for (auto &o : m.getBody()->getOperations()) {
    if (isa<AIE::TileOp, AIE::LogicalTileOp>(o))
      builder.setInsertionPointAfter(&o);
    else
      break;
  }
  llvm::DenseMap<int, AIE::TileLike> colToMemtile;
  llvm::SmallPtrSet<Operation *, 8> claimedLtos;
  auto findOrCreateLtoForCol = [&](int col) -> AIE::TileLike {
    auto it = colToMemtile.find(col);
    if (it != colToMemtile.end())
      return it->second;
    for (auto memtile : memtiles) {
      if (claimedLtos.contains(memtile.getOperation()))
        continue;
      auto preLto = dyn_cast<AIE::LogicalTileOp>(memtile.getOperation());
      if (!preLto || preLto.getCol().has_value())
        continue;
      preLto.setColAttr(builder.getI32IntegerAttr(col));
      claimedLtos.insert(memtile.getOperation());
      colToMemtile[col] = memtile;
      return memtile;
    }
    auto newLto = AIE::LogicalTileOp::create(
        builder, m.getLoc(), AIE::AIETileType::MemTile,
        /*col=*/builder.getI32IntegerAttr(col),
        /*row=*/IntegerAttr(),
        /*allocation_scheme=*/StringAttr());
    AIE::TileLike tl = newLto;
    memtiles.push_back(tl);
    claimedLtos.insert(newLto.getOperation());
    colToMemtile[col] = tl;
    return tl;
  };

  SmallVector<size_t> unhintedBuckets;
  for (size_t i = 0; i < memref_buckets.size(); i++) {
    if (bucketCols[i] < 0) {
      unhintedBuckets.push_back(i);
      continue;
    }
    AIE::TileLike memtile = findOrCreateLtoForCol(bucketCols[i]);
    for (auto bucket_elem : memref_buckets[i])
      memrefToMemTileMap[bucket_elem] = memtile;
  }
  int rr = 0;
  for (size_t bi : unhintedBuckets) {
    AIE::TileLike memtile = nullptr;
    for (auto m2 : memtiles) {
      if (!claimedLtos.contains(m2.getOperation())) {
        memtile = m2;
        claimedLtos.insert(m2.getOperation());
        break;
      }
    }
    if (!memtile) {
      memtile = memtiles[rr % memtiles.size()];
      rr++;
    }
    for (auto bucket_elem : memref_buckets[bi])
      memrefToMemTileMap[bucket_elem] = memtile;
  }
}

void allocL2Buffers(AIE::DeviceOp m,
                    std::map<AIE::BufferOp, AIE::TileLike> &bufferToMemtileMap,
                    uint64_t &BufferId) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  if (m.getTargetModel().getNumMemTileRows()) {
    std::map<memref::AllocOp, AIE::TileLike> memrefToTileMap;
    L2MemrefToMemTileMap(m, memrefToTileMap);
    patterns.insert<AllocL2BuffersPattern>(ctx, memrefToTileMap,
                                           bufferToMemtileMap, BufferId);
    (void)applyPatternsGreedily(m, std::move(patterns));
  }

  // Remove L2 temporary buffer allocs now that
  // allocation is complete.
  SmallVector<AIE::BufferOp> buffers;
  m.walk([&](AIE::BufferOp buffer) {
    auto name = buffer.getSymName();
    if (!name)
      return;
    if (name->starts_with("__L2_tmp"))
      buffers.push_back(buffer);
  });
  for (auto b : buffers)
    b.erase();
}

struct LowerAIRChannelsPattern : public OpRewritePattern<air::ChannelOp> {
  using OpRewritePattern<air::ChannelOp>::OpRewritePattern;

  LowerAIRChannelsPattern(
      MLIRContext *ctx, ShimTileAllocator &shimTileAlloc,
      std::map<AIE::BufferOp, AIE::TileLike> &bufferToMemtileMap,
      std::map<Operation *, AIE::ObjectFifoCreateOp> &linksToComplete)
      : OpRewritePattern(ctx), shimTileAlloc(shimTileAlloc),
        bufferToMemtileMap(bufferToMemtileMap),
        linksToComplete(linksToComplete) {}

  LogicalResult matchAndRewrite(air::ChannelOp channel,
                                PatternRewriter &rewriter) const override {
    auto device = channel->getParentOfType<AIE::DeviceOp>();
    auto ctx = device->getContext();
    if (!device)
      return failure();

    // SpecializeChannelBundlePattern should have removed them
    if (channel.getBundleSize() > 1)
      return failure();

    AIE::AIEObjectFifoType datatype;
    std::vector<air::ChannelPutOp> channelPuts =
        getChannelPutOpThroughSymbol(channel, device);
    std::vector<air::ChannelGetOp> channelGets =
        getChannelGetOpThroughSymbol(channel, device);

    // keep track of potential LinkOp
    bool linkToComplete =
        false; // track if objFifo has to be added to linksToComplete
    bool linkFound = false; // all ends of a link have been found
    Operation *endOfLink;   // one end of a link

    // put/get come in pairs, if one is missing then it's L3
    Value producerTile;
    if (channelPuts.size() > 0) {
      if (channelPuts.size() > 1)
        return channel.emitOpError(
            "channel lowering currently does not support many-to-one/many");
      auto res = findChannelPutGetTile<air::ChannelPutOp>(
          channelPuts[0], &producerTile, &datatype);
      if (res.failed())
        return res;

      // check if this put is linked to a get from another channel
      BaseMemRefType memref =
          llvm::cast<BaseMemRefType>(channelPuts[0].getMemref().getType());
      if (air::isL2(memref)) {
        if (linksToComplete.find(channelPuts[0].getOperation()) !=
            linksToComplete.end()) {
          endOfLink = channelPuts[0].getOperation();
          linkFound = true;
        } else {
          AIE::BufferOp buff = dyn_cast_if_present<AIE::BufferOp>(
              channelPuts[0].getMemref().getDefiningOp());
          for (auto user : buff->getUsers()) {
            if (auto pairedGet = dyn_cast_if_present<air::ChannelGetOp>(user)) {
              endOfLink = pairedGet.getOperation();
              linkToComplete = true;
            }
          }
        }
      }
    } else {
      // put from L3
      producerTile =
          shimTileAlloc.getShimTile(rewriter, device, channel.getName().str());
    }

    // put/get come in pairs, if one is missing then it's L3
    std::vector<Value> consumers;
    Value consumerTile;
    if (channelGets.size() > 1 && !channel.isBroadcast())
      return channel.emitOpError("has multiple gets but no broadcast shape");

    int expectedGets = channel.isBroadcast() ? channel.getBroadcastNum() : 1;
    for (auto get : channelGets) {
      auto res = findChannelPutGetTile<air::ChannelGetOp>(get, &consumerTile,
                                                          &datatype);
      if (res.failed())
        return res;
      consumers.push_back(consumerTile);

      // check if this get is linked to a put from another channel
      BaseMemRefType memref =
          llvm::cast<BaseMemRefType>(get.getMemref().getType());
      if (air::isL2(memref)) {
        if (linksToComplete.find(get.getOperation()) != linksToComplete.end()) {
          endOfLink = get.getOperation();
          linkFound = true;
        } else {
          AIE::BufferOp buff = dyn_cast_if_present<AIE::BufferOp>(
              get.getMemref().getDefiningOp());
          for (auto user : buff->getUsers()) {
            if (auto pairedPut = dyn_cast_if_present<air::ChannelPutOp>(user)) {
              endOfLink = pairedPut.getOperation();
              linkToComplete = true;
            }
          }
        }
      }
    }
    for (int i = 0; i < expectedGets - (int)channelGets.size(); i++) {
      // get from L3
      consumerTile =
          shimTileAlloc.getShimTile(rewriter, device, channel.getName().str());
      consumers.push_back(consumerTile);
    }

    if (!datatype)
      return failure();

    // create objFifo. Path B emits MemTile (and ShimNOC) as
    // aie.logical_tile, and those LTOs can sit anywhere in the device body
    // (e.g. after the cores) once the __L2_tmp anchor buffers are erased
    // and the greedy rewriter has reordered things. Hoist any out-of-order
    // tile-likes to the front of the body so the producer/consumer tile
    // operands always dominate the objfifo, then insert the objfifo right
    // after the last tile-like op.
    Block *body = device.getBody();
    Operation *firstNonTile = nullptr;
    SmallVector<Operation *> tilesToHoist;
    for (auto &op : *body) {
      if (!isa<AIE::TileOp, AIE::LogicalTileOp>(op)) {
        if (!firstNonTile)
          firstNonTile = &op;
      } else if (firstNonTile) {
        tilesToHoist.push_back(&op);
      }
    }
    for (auto *t : tilesToHoist)
      t->moveBefore(firstNonTile);

    rewriter.setInsertionPointToStart(body);
    for (auto &op : body->getOperations()) {
      if (isa<AIE::TileOp, AIE::LogicalTileOp>(op))
        rewriter.setInsertionPointAfter(&op);
      else
        break;
    }
    AIE::ObjectFifoCreateOp objFifo = createObjectFifo(
        rewriter, datatype, producerTile, consumers,
        channel.getBufferResources(), "air_" + channel.getName().str());

    // if this channel's get is linked with another put, register it
    if (linkToComplete)
      linksToComplete[endOfLink] = objFifo;
    // once the corresponding objFifo has been made, complete the link
    if (linkFound) {
      AIE::ObjectFifoCreateOp producerFifo = linksToComplete[endOfLink];
      if (isa<air::ChannelGetOp>(endOfLink))
        AIE::ObjectFifoLinkOp::create(
            rewriter, rewriter.getUnknownLoc(),
            rewriter.getArrayAttr({SymbolRefAttr::get(ctx, objFifo.name())}),
            rewriter.getArrayAttr(
                {SymbolRefAttr::get(ctx, producerFifo.name())}),
            rewriter.getArrayAttr({}), rewriter.getArrayAttr({}));
      else
        AIE::ObjectFifoLinkOp::create(
            rewriter, rewriter.getUnknownLoc(),
            rewriter.getArrayAttr(
                {SymbolRefAttr::get(ctx, producerFifo.name())}),
            rewriter.getArrayAttr({SymbolRefAttr::get(ctx, objFifo.name())}),
            rewriter.getArrayAttr({}), rewriter.getArrayAttr({}));
    }

    // replace put/get and any associated memref alloc/dealloc
    llvm::SmallSet<Operation *, 2> erased_deallocs;
    llvm::SmallSet<Operation *, 2> erased_allocs;
    for (auto put : channelPuts) {
      rewriteChannelAllocs<air::ChannelPutOp>(
          rewriter, put, objFifo, AIE::ObjectFifoPort::Produce, erased_allocs);
      rewriteChannelDeallocs<air::ChannelPutOp>(rewriter, put, objFifo,
                                                AIE::ObjectFifoPort::Produce,
                                                erased_deallocs);
      // clear any dependence to put
      if (put.getAsyncToken()) {
        for (auto u : put.getAsyncToken().getUsers()) {
          if (auto async_u = dyn_cast_if_present<air::AsyncOpInterface>(u))
            air::eraseAsyncDependencyFromAsyncOp(async_u, put.getAsyncToken());
          // TODO: complete else: account for scf.for and scf.parallel users
        }
      }
    }
    for (auto get : channelGets) {
      rewriteChannelAllocs<air::ChannelGetOp>(
          rewriter, get, objFifo, AIE::ObjectFifoPort::Consume, erased_allocs);
      rewriteChannelDeallocs<air::ChannelGetOp>(rewriter, get, objFifo,
                                                AIE::ObjectFifoPort::Consume,
                                                erased_deallocs);
      // clear any dependence to get
      if (get.getAsyncToken()) {
        for (auto u : get.getAsyncToken().getUsers()) {
          if (auto async_u = dyn_cast_if_present<air::AsyncOpInterface>(u))
            air::eraseAsyncDependencyFromAsyncOp(async_u, get.getAsyncToken());
          // TODO: complete else: account for scf.for and scf.parallel users
        }
      }
    }
    // erase dangling deallocs
    for (auto o : erased_deallocs)
      rewriter.eraseOp(o);
    // erase channel puts and gets
    for (auto get : channelGets)
      rewriter.eraseOp(get);
    for (auto put : channelPuts)
      rewriter.eraseOp(put);
    // erase the channel
    rewriter.eraseOp(channel);
    // erase dangling allocs
    for (auto o : erased_allocs)
      if (o->use_empty())
        rewriter.eraseOp(o);
    return success();
  }

private:
  // find AIE cores and their tiles based on memory hierarchy levels
  template <typename MyOp>
  LogicalResult findChannelPutGetTile(MyOp op, Value *tile,
                                      AIE::AIEObjectFifoType *datatype) const {
    MemRefType memref = llvm::cast<MemRefType>(op.getMemref().getType());
    auto mem_space = air::getMemorySpace(memref);
    *datatype = AIE::AIEObjectFifoType::get(
        MemRefType::get(memref.getShape(), memref.getElementType()));
    if (mem_space == air::MemorySpace::L1) {
      AIE::CoreOp core = op->template getParentOfType<AIE::CoreOp>();
      if (!core)
        return op.emitOpError("could not retrieve core for channel put/get op");
      *tile = core.getTileOp();
      return success();
    } else if (mem_space == air::MemorySpace::L2) {
      if (bufferToMemtileMap.find(dyn_cast_if_present<AIE::BufferOp>(
              op.getMemref().getDefiningOp())) != bufferToMemtileMap.end()) {
        AIE::TileLike memtile =
            bufferToMemtileMap[dyn_cast_if_present<AIE::BufferOp>(
                op.getMemref().getDefiningOp())];
        *tile = memtile->getResult(0);
      } else {
        return op.emitOpError("missing L2 alloc");
      }
      return success();
    } else {
      op->dump();
      op.getMemref().dump();
      return op.emitOpError("unsupported memory space");
    }
  }

  AIE::ObjectFifoCreateOp createObjectFifo(OpBuilder &builder,
                                           AIE::AIEObjectFifoType datatype,
                                           Value prodTile,
                                           const std::vector<Value> &consTile,
                                           int depth, StringRef name) const {
    // Create empty dimension arrays for each consumer to satisfy the
    // dimensionsFromStreamPerConsumer attribute size requirement.
    SmallVector<AIE::BDDimLayoutArrayAttr> dimsFromStreamPerConsumer;
    for (size_t i = 0; i < consTile.size(); i++) {
      dimsFromStreamPerConsumer.push_back(
          AIE::BDDimLayoutArrayAttr::get(builder.getContext(), {}));
    }
    AIE::ObjectFifoCreateOp fifo = AIE::ObjectFifoCreateOp::create(
        builder, builder.getUnknownLoc(), builder.getStringAttr(name), prodTile,
        consTile, builder.getIntegerAttr(builder.getI32Type(), depth), datatype,
        /*dimensionsToStream=*/{},
        /*dimensionsFromStreamPerConsumer=*/dimsFromStreamPerConsumer);
    return fifo;
  }

  template <typename MyOp>
  void
  rewriteChannelAllocs(PatternRewriter &rewriter, MyOp op,
                       AIE::ObjectFifoCreateOp objFifo,
                       AIE::ObjectFifoPort port,
                       llvm::SmallSet<Operation *, 2> &erased_allocs) const {
    BaseMemRefType memref = cast<BaseMemRefType>(op.getMemref().getType());
    if (air::isL2(memref)) {
      // add alloc to list of ops to erase
      erased_allocs.insert(op.getMemref().getDefiningOp());
      return;
    }

    AIE::AIEObjectFifoType ofTy =
        cast<AIE::AIEObjectFifoType>(objFifo.getElemType());
    auto elementType = ofTy.getElementType();
    auto acqType = AIE::AIEObjectFifoSubviewType::get(elementType);

    rewriter.setInsertionPoint(&op->getBlock()->front());
    AIE::ObjectFifoAcquireOp producerAcq =
        AIE::ObjectFifoAcquireOp::create(rewriter, rewriter.getUnknownLoc(),
                                         acqType, port, objFifo.getName(), 1);
    rewriter.setInsertionPointAfter(producerAcq);
    AIE::ObjectFifoSubviewAccessOp producerAccess =
        AIE::ObjectFifoSubviewAccessOp::create(
            rewriter, rewriter.getUnknownLoc(), elementType,
            producerAcq.getSubview(),
            rewriter.getIntegerAttr(rewriter.getI32Type(), 0));

    // replace uses of alloc with result of acquire
    if (auto a = dyn_cast_if_present<memref::AllocOp>(
            op.getMemref().getDefiningOp()))
      rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
          a.getOperation(), a.getType(), producerAccess.getOutput());
  }

  template <typename MyOp>
  void rewriteChannelDeallocs(
      PatternRewriter &rewriter, MyOp op, AIE::ObjectFifoCreateOp objFifo,
      AIE::ObjectFifoPort port,
      llvm::SmallSet<Operation *, 2> &erased_deallocs) const {
    BaseMemRefType memref =
        llvm::cast<BaseMemRefType>(op.getMemref().getType());
    if (air::isL2(memref)) {
      return;
    }
    for (auto u : op.getMemref().getDefiningOp()->getUsers()) {
      if (auto dealloc = dyn_cast_if_present<memref::DeallocOp>(u)) {
        rewriter.setInsertionPoint(&op->getBlock()->back());
        AIE::ObjectFifoReleaseOp::create(rewriter, dealloc->getLoc(), port,
                                         objFifo.getName(), 1);
        // Delete ops at the end of the rewrite pattern to avoid repeatedly
        // deleting the same op
        erased_deallocs.insert(dealloc.getOperation());
      }
    }
  }

  ShimTileAllocator &shimTileAlloc;
  std::map<AIE::BufferOp, AIE::TileLike> &bufferToMemtileMap;
  std::map<Operation *, AIE::ObjectFifoCreateOp> &linksToComplete;
};

// This function replaces ChannelPutOp/ChannelGetOp with AIE_CreateObjectFifoOps
// and with ObjectFifoAcquireOp<Producer/Consumer>. It also erases memref allocs
// as the objFifo lowering allocates its own memory. It replaces the associated
// memref deallocs with ObjectFifoReleaseOps.
LogicalResult
lowerAIRChannels(AIE::DeviceOp &d, ShimTileAllocator &s,
                 std::map<AIE::BufferOp, AIE::TileLike> &bufferToMemtileMap) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  std::map<Operation *, AIE::ObjectFifoCreateOp> linksToComplete;
  patterns.insert<LowerAIRChannelsPattern>(ctx, s, bufferToMemtileMap,
                                           linksToComplete);
  (void)applyPatternsGreedily(d, std::move(patterns));
  // Leave shim LTOs unresolved here. Downstream `aie-place-tiles` (invoked
  // from aircc after air-merge-unrolled-devices) sees the full set of
  // aie.objectfifo connections and resolves shim/memtile LTOs together via
  // the same Adjacency-driven placer that mlir-aie's native ObjectFifo
  // flow uses. Doing it in-AIR with SequentialPlacer would lose that
  // objfifo-aware placement context.
  return success();
}

struct SpecializeChannelBundlePattern
    : public OpRewritePattern<air::ChannelOp> {
  using OpRewritePattern<air::ChannelOp>::OpRewritePattern;

  SpecializeChannelBundlePattern(
      MLIRContext *ctx, std::map<std::string, std::string> &chan_to_chan_map,
      int &maxSize)
      : OpRewritePattern(ctx), chan_to_chan_map(chan_to_chan_map),
        maxSize(maxSize) {}

  LogicalResult matchAndRewrite(air::ChannelOp channel,
                                PatternRewriter &rewriter) const override {

    auto device = channel->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    if (channel.getBundleSize() <= 1)
      return failure();

    // mmio channels are handled directly by lowerAIRMMIOChannelOps, which
    // matches host-side puts to per-core gets by constant index across the
    // full bundle. Splitting the bundle here would orphan the original
    // host-side puts (they sit outside the device, where this pattern's
    // rewrites don't reach), leaving them to fail later as
    // "no matching device-side air.channel.get".
    if (channel.getChannelType() == "npu_mmio")
      return failure();

    std::vector<air::ChannelPutOp> channelPuts =
        getChannelPutOpThroughSymbol(channel, device);
    std::vector<air::ChannelGetOp> channelGets =
        getChannelGetOpThroughSymbol(channel, device);

    // Walk through each element in a channel bundle
    auto bundle_size = extractFromIntegerArrayAttr<int64_t>(channel.getSize());
    auto bundle_size_stdvec = convertToStdVec(bundle_size);
    for (unsigned iter = 0; iter < (unsigned)channel.getBundleSize(); iter++) {
      rewriter.setInsertionPoint(channel);
      auto cname = air::createChannelName(device.getOperation());
      // Add chan name to chan name map
      chan_to_chan_map[cname] = channel.getName().str();
      SmallVector<int64_t, 2> channel_sizes = {1, 1};
      auto new_chan = air::ChannelOp::create(
          rewriter, channel->getLoc(), cname,
          rewriter.getI64ArrayAttr(channel_sizes), channel.getChannelType());
      if (channel->hasAttr("broadcast_shape")) {
        auto broadcast_shape = specializeBroadcastShape(rewriter, channel);
        new_chan->setAttr("broadcast_shape",
                          rewriter.getArrayAttr(ArrayRef(broadcast_shape)));
      }
      // Propagate DMA-steering markers (incl. user-pinned packet_ids) onto each
      // split channel: flow creation only sees the split channels, so without
      // this, indexed/convergent packet channels fall back to auto-assigned
      // consecutive ids, which alias under the switchbox arbiter's binary-mask
      // matching and deadlock. Routed through the single-source-of-truth helper
      // so this copy site stays in sync with the marker set.
      air::copyChannelSteeringAttrs(channel, new_chan);
      std::vector<unsigned> position =
          air::getMDVectorFromIterator(bundle_size_stdvec, iter);
      for (auto put : channelPuts) {
        auto indices_uint =
            air::convertVecOfConstIndexToVecOfUInt(put.getIndices());
        if (indices_uint.empty() && !put.getIndices().empty() && iter == 0)
          put->emitWarning(
              "channel bundle indices cannot be resolved to compile-time "
              "constants; this channel put will be replaced with "
              "air.wait_all, which may cause data loss");
        if (areIdenticalVectors(indices_uint, position)) {
          // Found channel put for this channel
          rewriter.setInsertionPoint(put);
          auto new_put = createChannelPutGetWithoutBundle(rewriter, new_chan,
                                                          put, maxSize);
          auto async_new_put = dyn_cast_if_present<air::AsyncOpInterface>(
              new_put.getOperation());
          if (put.getAsyncToken()) {
            replaceAllUsesInRegionWith(put.getAsyncToken(),
                                       async_new_put.getAsyncToken(),
                                       device.getRegion());
            clearAsyncDependenciesOfAsyncOp(new_put);
          }
        }
      }
      for (auto get : channelGets) {
        auto indices_uint =
            air::convertVecOfConstIndexToVecOfUInt(get.getIndices());
        if (indices_uint.empty() && !get.getIndices().empty() && iter == 0)
          get->emitWarning(
              "channel bundle indices cannot be resolved to compile-time "
              "constants; this channel get will be replaced with "
              "air.wait_all, which may cause data loss");
        if (areIdenticalVectors(indices_uint, position)) {
          // Found channel get for this channel
          rewriter.setInsertionPoint(get);
          auto new_get = createChannelPutGetWithoutBundle(rewriter, new_chan,
                                                          get, maxSize);
          auto async_new_get = dyn_cast_if_present<air::AsyncOpInterface>(
              new_get.getOperation());
          if (get.getAsyncToken()) {
            replaceAllUsesInRegionWith(get.getAsyncToken(),
                                       async_new_get.getAsyncToken(),
                                       device.getRegion());
            clearAsyncDependenciesOfAsyncOp(new_get);
          }
        }
      }
    }

    // Erase bundled channel ops and their corresponding put/get ops
    for (auto put : channelPuts) {
      if (!put->getNumResults()) {
        rewriter.eraseOp(put);
        continue;
      }
      rewriter.setInsertionPoint(put);
      rewriter.replaceOpWithNewOp<air::WaitAllOp>(
          put, air::AsyncTokenType::get(put->getContext()),
          put.getAsyncDependencies());
    }
    for (auto get : channelGets) {
      if (!get->getNumResults()) {
        rewriter.eraseOp(get);
        continue;
      }
      rewriter.setInsertionPoint(get);
      rewriter.replaceOpWithNewOp<air::WaitAllOp>(
          get, air::AsyncTokenType::get(get->getContext()),
          get.getAsyncDependencies());
    }
    rewriter.eraseOp(channel);

    return success();
  }

private:
  std::map<std::string, std::string> &chan_to_chan_map;
  int &maxSize;
  bool areIdenticalVectors(std::vector<unsigned> a,
                           std::vector<unsigned> b) const {
    if (a.empty())
      return false;
    if (b.empty())
      return false;
    if (a.size() != b.size())
      return false;
    for (unsigned i = 0; i < a.size(); i++) {
      if (a[i] != b[i])
        return false;
    }
    return true;
  }

  std::vector<unsigned> convertToStdVec(SmallVector<int64_t, 6> vec) const {
    std::vector<unsigned> output;
    for (auto v : vec) {
      output.push_back((unsigned)v);
    }
    return output;
  }

  air::ChannelInterface
  createChannelPutGetWithoutBundle(OpBuilder &builder, air::ChannelOp chan,
                                   air::ChannelInterface ci,
                                   int maxSize) const {
    SmallVector<Type, 4> tys = {};
    SmallVector<Value, 4> deps = {};
    auto asyncOp =
        dyn_cast_if_present<air::AsyncOpInterface>(ci.getOperation());
    if (asyncOp.getAsyncToken()) {
      tys.push_back(air::AsyncTokenType::get(builder.getContext()));
      deps = asyncOp.getAsyncDependencies();
    }
    SmallVector<Value, 4> indices = {};
    // Canonicalize wrap and stride lists after specialization
    SmallVector<Value> offsets = ci.getOffsets();
    SmallVector<Value> wraps = ci.getSizes();
    SmallVector<Value> strides = ci.getStrides();
    auto memrefTy = llvm::dyn_cast<BaseMemRefType>(ci.getMemref().getType());
    int innerAlignment =
        memrefTy ? air::getDmaInnerElementAlignment(memrefTy, ci) : 1;
    (void)air::canonicalizeWrapAndStrideList(
        builder, offsets, wraps, strides,
        air::getTensorVolume(ci.getMemref().getType()), maxSize,
        innerAlignment);
    air::ChannelInterface new_ci = nullptr;
    if (isa<air::ChannelPutOp>(ci))
      new_ci = air::ChannelPutOp::create(
          builder, ci->getLoc(), tys, deps, chan.getSymName(), indices,
          ci.getMemref(), offsets, wraps, strides,
          /*pad_before=*/nullptr, /*pad_after=*/nullptr);
    else if (isa<air::ChannelGetOp>(ci))
      new_ci = air::ChannelGetOp::create(
          builder, ci->getLoc(), tys, deps, chan.getSymName(), indices,
          ci.getMemref(), offsets, wraps, strides,
          /*pad_before=*/nullptr, /*pad_after=*/nullptr);
    new_ci->setAttrs(ci->getDiscardableAttrDictionary());
    air::copyPaddingAttributes(ci, new_ci);
    return new_ci;
  }

  std::vector<Attribute> specializeBroadcastShape(OpBuilder builder,
                                                  air::ChannelOp chan) const {
    auto broadcast_shape = chan.getBroadcastShape();
    int diffDimension = chan.getBroadcastDimension();
    std::vector<Attribute> new_shape;
    for (int i = 0; i < (int)broadcast_shape.size(); i++) {
      if (i == diffDimension) {
        auto broadcast_dim =
            dyn_cast_if_present<IntegerAttr>(broadcast_shape[i]).getInt();
        new_shape.push_back(builder.getI64IntegerAttr(broadcast_dim));
      } else
        new_shape.push_back(builder.getI64IntegerAttr(1));
    }
    return new_shape;
  }
};

// By specializing each air.channel op in a channel bundle, this function
// removes air.channel bundled representation in a aie.device op.
void specializeChannelBundle(
    AIE::DeviceOp &d, std::map<std::string, std::string> &chan_to_chan_map) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  // Enforce max size constraint
  int maxSize = isa<AIE::AIE1TargetModel>(AIE::getTargetModel(d)) ? -1 : 1023;
  patterns.insert<SpecializeChannelBundlePattern>(ctx, chan_to_chan_map,
                                                  maxSize);
  (void)applyPatternsGreedily(d, std::move(patterns));
}

// Remove orphaned specialized channels after specializeChannelBundle.
// An orphaned channel is one that has puts but no gets, or gets but no puts.
// This happens when cloning L3 ops to all devices, but each device only
// using a subset of them.
static void removeOrphanedChannels(AIE::DeviceOp &d) {
  SmallVector<air::ChannelOp> channelsToRemove;
  SmallVector<Operation *> opsToRemove;

  for (auto channel : d.getOps<air::ChannelOp>()) {
    auto puts = getChannelPutOpThroughSymbol(channel, d);
    auto gets = getChannelGetOpThroughSymbol(channel, d);

    // Orphaned: has puts but no gets, or has gets but no puts
    if ((puts.empty() && !gets.empty()) || (!puts.empty() && gets.empty())) {
      channelsToRemove.push_back(channel);
      for (auto put : puts)
        opsToRemove.push_back(put);
      for (auto get : gets)
        opsToRemove.push_back(get);
    }
  }

  // Replace uses of async tokens before erasing orphaned put/get ops
  OpBuilder builder(d.getContext());
  IRMapping remap;
  for (auto op : opsToRemove) {
    if (air::isAsyncOp(op)) {
      auto asyncToken = air::getAsyncTokenFromOp(op);
      // Only materialize a wait_all if the async token has uses to preserve.
      if (asyncToken && !asyncToken.use_empty()) {
        builder.setInsertionPoint(op);
        auto waitAll = air::replaceAsyncOpWithWaitAll(builder, remap, op,
                                                      /*cloneDepList=*/true);
        asyncToken.replaceAllUsesWith(waitAll.getAsyncToken());
      }
    }
    op->erase();
  }
  // Then erase orphaned channel declarations
  for (auto channel : channelsToRemove)
    channel->erase();
}

struct LowerAIRPingPongPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Check if the loop is already isolated for ping-pong transformation, so
    // that there are only data producers and consumers.
    if (!for_op->hasAttr("isolated"))
      return failure();

    // Check for ping-pong factor
    uint64_t unroll_factor = 0;
    if (!for_op->hasAttr("unroll"))
      return failure();
    unroll_factor = for_op->getAttrOfType<IntegerAttr>("unroll").getInt();

    // Get device op
    auto device = for_op->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    // Annotate channels with buffer_resource, i.e. object count
    for_op.walk([&](Operation *op) {
      if (auto get = dyn_cast_if_present<air::ChannelGetOp>(op)) {
        auto chan_op = air::getChannelDeclarationThroughSymbol(get);
        chan_op->setAttr(
            "buffer_resources",
            IntegerAttr::get(IntegerType::get(chan_op->getContext(), 32),
                             unroll_factor));
      } else if (auto put = dyn_cast_if_present<air::ChannelPutOp>(op)) {
        auto chan_op = air::getChannelDeclarationThroughSymbol(put);
        chan_op->setAttr(
            "buffer_resources",
            IntegerAttr::get(IntegerType::get(chan_op->getContext(), 32),
                             unroll_factor));
      }
    });

    for_op->removeAttr("isolated");
    for_op->removeAttr("unroll");

    return success();
  }

private:
};

// By specializing each air.channel op in a channel bundle, this function
// removes air.channel bundled representation in a aie.device op.
void LowerAIRPingPong(AIE::DeviceOp &d) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<LowerAIRPingPongPattern>(ctx);
  (void)applyPatternsGreedily(d, std::move(patterns));
}

template <typename OpT>
struct OpRemovalPattern : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpT::Adaptor;

  OpRemovalPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<OpT>(context, benefit) {}

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    for (auto res : op->getResults()) {
      if (res.use_empty())
        continue;
      if (isa<air::AsyncTokenType>(res.getType())) {
        res.replaceAllUsesWith(
            air::WaitAllOp::create(rewriter, op->getLoc(),
                                   air::AsyncTokenType::get(op->getContext()),
                                   air::getAsyncDependenciesFromOp(op))
                .getAsyncToken());
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Helper function to check if a copy operation is an L1-to-L1 copy inside an
// AIE core.
static bool isL1ToL1CopyInCore(MemRefType srcType, MemRefType dstType,
                               Operation *op) {
  // Only handle L1-to-L1 copies (memory space 2)
  if (!air::isL1(srcType) || !air::isL1(dstType))
    return false;

  // Only handle copies inside AIE cores
  return op->getParentOfType<AIE::CoreOp>() != nullptr;
}

// Pattern to convert memref.copy to linalg.copy for L1-to-L1 copies.
// The actual lowering to loops is handled by LinalgCopyToLoopsPattern.
struct MemRefCopyToLinalgCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto srcType = llvm::cast<MemRefType>(copyOp.getSource().getType());
    auto dstType = llvm::cast<MemRefType>(copyOp.getTarget().getType());

    if (!isL1ToL1CopyInCore(srcType, dstType, copyOp))
      return failure();

    // Convert memref.copy to linalg.copy; LinalgCopyToLoopsPattern will
    // lower it to loops.
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(copyOp, copyOp.getSource(),
                                                copyOp.getTarget());
    return success();
  }
};

// Pattern to lower L1-to-L1 linalg.copy to SCF loops.
// AIE cores don't have a native memcpy instruction, so we convert
// linalg.copy to explicit load/store loops using linalg::linalgOpToLoops.
struct LinalgCopyToLoopsPattern : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // Get input (source) and output (destination) memrefs
    auto inputs = copyOp.getDpsInputOperands();
    auto outputs = copyOp.getDpsInitsMutable();
    if (inputs.size() != 1 || outputs.size() != 1)
      return failure();

    auto srcType =
        llvm::dyn_cast_if_present<MemRefType>(inputs[0]->get().getType());
    auto dstType =
        llvm::dyn_cast_if_present<MemRefType>(outputs[0].get().getType());
    if (!srcType || !dstType)
      return failure();

    if (!isL1ToL1CopyInCore(srcType, dstType, copyOp))
      return failure();

    // Convert linalg.copy to SCF loops using linalgOpToLoops
    FailureOr<linalg::LinalgLoops> loops =
        linalg::linalgOpToLoops(rewriter, copyOp);
    if (failed(loops))
      return failure();

    rewriter.eraseOp(copyOp);
    return success();
  }
};

// Apply the memref.copy and linalg.copy to loops lowering patterns on L1-to-L1
// copies
void lowerMemRefCopyToLoops(AIE::DeviceOp d) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<MemRefCopyToLinalgCopyPattern>(ctx);
  patterns.insert<LinalgCopyToLoopsPattern>(ctx);
  (void)applyPatternsGreedily(d, std::move(patterns));
}

/// Remove dead memref.get_global ops and their corresponding memref.global
/// declarations. outlineAIECores creates these for ALL L2/L3 herd memref
/// args, but DMA/channel lowering erases the ops that referenced them,
/// leaving orphaned globals that cause linker errors (issue #1404).
static void removeDeadGlobalOps(AIE::DeviceOp device) {
  // Erase dead memref.get_global ops (no users).
  SmallVector<memref::GetGlobalOp> deadGetGlobals;
  device.walk([&](memref::GetGlobalOp op) {
    if (op.use_empty())
      deadGetGlobals.push_back(op);
  });
  for (auto op : deadGetGlobals)
    op->erase();

  // Collect symbols still referenced by remaining get_global ops.
  llvm::DenseSet<StringRef> referencedGlobals;
  device.walk(
      [&](memref::GetGlobalOp op) { referencedGlobals.insert(op.getName()); });

  // Erase unreferenced memref.global declarations.
  SmallVector<memref::GlobalOp> deadGlobals;
  for (auto globalOp : device.getOps<memref::GlobalOp>()) {
    if (!referencedGlobals.contains(globalOp.getSymName()))
      deadGlobals.push_back(globalOp);
  }
  for (auto op : deadGlobals)
    op->erase();
}

class AIRToAIEPass : public air::impl::AIRToAIEBase<AIRToAIEPass> {

  uint64_t BufferId = 0;

public:
  AIRToAIEPass() = default;
  AIRToAIEPass(const AIRToAIEPass &pass) {}
  AIRToAIEPass(const air::AIRToAIEOptions &options) : AIRToAIEBase(options) {}

  // Shared pipeline logic for a single AIE device.
  // This method runs the transformation pipeline for a device and stops at
  // the specified PipelineStage for debugging purposes.
  // Returns failure() if any transformation stage fails.
  LogicalResult
  runDevicePipeline(AIE::DeviceOp device, ModuleOp module, air::HerdOp herd,
                    std::map<AIE::BufferOp, AIE::TileLike> &bufferToMemtileMap,
                    AIRToAIEConversionOptions &options, bool useObjFifo,
                    PipelineStage stopAfter = PipelineStage::Complete) {

    auto ctx = device->getContext();
    OpBuilder builder(device);

    // Check hasDma/hasChan conflict
    bool hasDma = false;
    bool hasChan = false;
    device.walk([&](Operation *o) {
      hasDma |= isa<air::DmaMemcpyNdOp>(o);
      hasChan |= isa<air::ChannelInterface>(o);
    });
    if (hasDma && hasChan) {
      device.emitOpError(
          ": lowering of segments containing both dma copies and "
          "channels is not supported");
      return failure();
    }

    // Allocators
    air::ShimDMAAllocator shimDmaAlloc(device);
    ShimTileAllocator shimTileAlloc(device.getTargetModel());
    std::map<std::string, std::string> chan_to_chan_map;
    std::map<int, int> chan_renumber_reverse_map;

    if (stopAfter == PipelineStage::AfterCreateAIEModules)
      return success();

    // Get the parent launch for this herd to filter memcpy ops
    air::LaunchOp targetLaunch = herd->getParentOfType<air::LaunchOp>();

    // Stage: Clone memcpys to device
    if (useObjFifo) {
      cloneL2AndL3MemcpysToDeviceOp(
          builder, device, module, /*clone_l2*/ true, /*clone_l3*/ false,
          /*use_lock_race_cond_fix*/ options.use_lock_race_condition_fix,
          targetLaunch);
    } else {
      cloneL2AndL3MemcpysToDeviceOp(
          builder, device, module, /*clone_l2*/ true, /*clone_l3*/ true,
          /*use_lock_race_cond_fix*/ options.use_lock_race_condition_fix,
          targetLaunch);
    }
    if (stopAfter == PipelineStage::AfterCloneMemcpys)
      return success();

    // Stage: Lower execute and tokens
    specializeHerdAffineIf(device);
    lowerAirExecute(device);
    lowerScfAirTokens(device);
    if (stopAfter == PipelineStage::AfterLowerExecute)
      return success();

    // Stage: Specialize channel bundle
    specializeChannelBundle(device, chan_to_chan_map);
    // Remove orphaned channels that have puts but no gets (or vice versa).
    // This cleans up channels cloned from L3 that don't match any channel
    // in this device's segment unroll iteration.
    // Only run this when segment unroll is active.
    if (device->hasAttr("segment_unroll_x") ||
        device->hasAttr("segment_unroll_y"))
      removeOrphanedChannels(device);
    if (stopAfter == PipelineStage::AfterSpecializeChannel)
      return success();

    // Stage: Allocate buffers (ObjFifo vs non-ObjFifo paths diverge)
    if (useObjFifo) {
      air::renumberMemcpyIfOps(&device.getRegion());
      LowerAIRPingPong(device);
      allocL2Buffers(device, bufferToMemtileMap, BufferId);
      if (failed(lowerAIRChannels(device, shimTileAlloc, bufferToMemtileMap)))
        return failure();
      allocL1Buffers(device, BufferId);
    } else {
      specializeL2MemrefsIntoMemtiles(device);
      allocL1Buffers(device, BufferId);
      allocL2Buffers(device, bufferToMemtileMap, BufferId);
    }
    if (stopAfter == PipelineStage::AfterAllocBuffers)
      return success();

    // Stage: Renumber memcpy ops
    if (!useObjFifo)
      air::renumberMemcpyIfOps(&device.getRegion(), chan_renumber_reverse_map);
    if (stopAfter == PipelineStage::AfterRenumberMemcpy)
      return success();

    // Stage: Lower AIR memcpy ops (WHERE CRASH OCCURS for the debugging case)
    if (!useObjFifo) {
      if (failed(lowerAIRMemcpyOp<air::ChannelInterface>(device, shimDmaAlloc,
                                                         options)))
        return failure();
    }
    if (failed(lowerAIRMemcpyOp<air::DmaMemcpyNdOp>(device, shimDmaAlloc,
                                                    options)))
      return failure();
    if (stopAfter == PipelineStage::AfterLowerAIRMemcpy)
      return success();

    // Stage: Trace packet flow
    if (options.insert_trace_packet_flow)
      createTracePacketFlow(device);
    if (stopAfter == PipelineStage::AfterTracePacketFlow)
      return success();

    // Stage: Lower L1-to-L1 memref.copy to loops
    lowerMemRefCopyToLoops(device);
    if (stopAfter == PipelineStage::AfterLowerMemRefCopy)
      return success();

    // Complete stage: Canonicalization and op removal
    RewritePatternSet patterns(ctx);
    air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsGreedily(device, std::move(patterns));

    RewritePatternSet removepatterns(ctx);
    removepatterns
        .add<OpRemovalPattern<memref::DeallocOp>,
             OpRemovalPattern<air::WaitAllOp>, OpRemovalPattern<memref::CopyOp>,
             OpRemovalPattern<memref::AssumeAlignmentOp>>(ctx);
    ConversionTarget target(*ctx);
    target.addIllegalOp<memref::DeallocOp, air::WaitAllOp, memref::CopyOp,
                        memref::AssumeAlignmentOp>();
    if (failed(
            applyPartialConversion(device, target, std::move(removepatterns))))
      return failure();

    // Clean up dead memref.get_global/memref.global left by outlineAIECores
    // after DMA/channel lowering consumed their users.
    removeDeadGlobalOps(device);

    return success();
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<airrt::AIRRtDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<xilinx::AIEX::AIEXDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<DLTIDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
  }

  // Circuit-switched flow.
  AIE::FlowOp getFlowOp(AIE::DeviceOp aie_device, mlir::Value source,
                        xilinx::AIE::WireBundle sourceBundle,
                        uint32_t sourceChannel, mlir::Value dest,
                        xilinx::AIE::WireBundle destBundle,
                        uint32_t destChannel) {
    AIE::FlowOp flowOp = nullptr;
    aie_device.walk([&](Operation *op) {
      if (auto fop = dyn_cast_if_present<AIE::FlowOp>(op))
        if (source == fop.getSource() && dest == fop.getDest() &&
            sourceBundle == fop.getSourceBundle() &&
            destBundle == fop.getDestBundle() &&
            static_cast<int64_t>(sourceChannel) == fop.getSourceChannel() &&
            static_cast<int64_t>(destChannel) == fop.getDestChannel())
          flowOp = fop;
    });
    if (flowOp)
      return flowOp;

    OpBuilder builder(aie_device);
    builder.setInsertionPoint(aie_device.getBody()->getTerminator());
    return AIE::FlowOp::create(builder, builder.getUnknownLoc(), source,
                               sourceBundle, sourceChannel, dest, destBundle,
                               destChannel);
  }

  // Packet-switched flow.
  AIE::PacketSourceOp
  getPacketSourceOpInPacketFlowOp(AIE::PacketFlowOp packetFlowOp,
                                  Value source) {
    AIE::PacketSourceOp res = nullptr;
    packetFlowOp.walk([&](AIE::PacketSourceOp pktSoruceOp) {
      if (pktSoruceOp.getTile() == source)
        res = pktSoruceOp;
    });
    return res;
  }

  AIE::PacketDestOp
  getPacketDestOpInPacketFlowOp(AIE::PacketFlowOp packetFlowOp, Value dest) {
    AIE::PacketDestOp res = nullptr;
    packetFlowOp.walk([&](AIE::PacketDestOp pktDestOp) {
      if (pktDestOp.getTile() == dest)
        res = pktDestOp;
    });
    return res;
  }

  AIE::PacketFlowOp
  createPacketFlowOp(OpBuilder &builder, int flowID, Value source,
                     xilinx::AIE::WireBundle sourceBundle,
                     uint32_t sourceChannel, Value dest,
                     xilinx::AIE::WireBundle destBundle, uint32_t destChannel,
                     mlir::BoolAttr keep_pkt_header = nullptr) {
    AIE::PacketFlowOp pktFlow = AIE::PacketFlowOp::create(
        builder, builder.getUnknownLoc(), flowID, keep_pkt_header, nullptr);
    Region &r_pktFlow = pktFlow.getPorts();
    Block *b_pktFlow = builder.createBlock(&r_pktFlow);
    builder.setInsertionPointToStart(b_pktFlow);
    AIE::PacketSourceOp::create(builder, builder.getUnknownLoc(), source,
                                sourceBundle, sourceChannel);
    AIE::PacketDestOp::create(builder, builder.getUnknownLoc(), dest,
                              destBundle, destChannel);
    AIE::EndOp::create(builder, builder.getUnknownLoc());
    return pktFlow;
  }

  // Assign a pkt_id for a shim (or trace) packet flow: monotonic global
  // counter, skipping any id already claimed by an intra-device flow in this
  // device. Records the assignment in claimedPacketIDs.
  int assignShimPacketID() {
    while (claimedPacketIDs.count(nextGlobalShimPacketID))
      ++nextGlobalShimPacketID;
    int id = nextGlobalShimPacketID++;
    claimedPacketIDs.insert(id);
    return id;
  }

  // Assign a pkt_id for an intra-device packet flow: lowest pkt_id not yet
  // claimed in this device. Records the assignment in claimedPacketIDs.
  int assignIntraDevicePacketID() {
    int id = 0;
    while (claimedPacketIDs.count(id))
      ++id;
    claimedPacketIDs.insert(id);
    return id;
  }

  // This method generates broadcast packet flow if found multiple flows with
  // the same source. TODO: packet flows sharing source do not always mean
  // broadcast.
  AIE::PacketFlowOp getPacketFlowOp(AIE::DeviceOp aie_device,
                                    mlir::Value source,
                                    xilinx::AIE::WireBundle sourceBundle,
                                    uint32_t sourceChannel, mlir::Value dest,
                                    xilinx::AIE::WireBundle destBundle,
                                    uint32_t destChannel, int flowID) {
    AIE::PacketFlowOp packetFlowOp = nullptr;
    aie_device.walk([&](AIE::PacketFlowOp pktFlowOp) {
      auto pktSrcOp = getPacketSourceOpInPacketFlowOp(pktFlowOp, source);
      if (!pktSrcOp)
        return;
      auto pktSrcBundle = pktSrcOp.getBundle();
      if (pktSrcBundle != sourceBundle)
        return;
      auto pktSrcChannel = pktSrcOp.getChannel();
      if (pktSrcChannel != (int)sourceChannel)
        return;
      if (pktFlowOp.getID() != flowID)
        return;
      packetFlowOp = pktFlowOp;
    });

    OpBuilder builder(aie_device);
    if (packetFlowOp) {
      auto pktDestOp = getPacketDestOpInPacketFlowOp(packetFlowOp, dest);
      if (pktDestOp) {
        auto pktDestBundle = pktDestOp.getBundle();
        auto pktDestChannel = pktDestOp.getChannel();
        if (pktDestBundle == destBundle && pktDestChannel == (int)destChannel) {
          return packetFlowOp;
        }
      }
      builder.setInsertionPoint(packetFlowOp.getBody()->getTerminator());
      AIE::PacketDestOp::create(builder, builder.getUnknownLoc(), dest,
                                destBundle, destChannel);
      return packetFlowOp;
    }

    builder.setInsertionPoint(aie_device.getBody()->getTerminator());
    return createPacketFlowOp(builder, flowID, source, sourceBundle,
                              sourceChannel, dest, destBundle, destChannel);
  }

  /// Query an existing packet flow operation from within the AIE device.
  ///
  /// This method is used when looking up packet flows from operations that are
  /// already lowered within the aie.device operation. Since air.channel
  /// declarations are duplicated between the aie.device and its parent module,
  /// this method performs symbol name resolution to match the correct flow ID.
  ///
  /// \note Only air::ChannelInterface operations support packet-flow routing.
  ///       For non-channel memcpy operations, this returns null.
  AIE::PacketFlowOp getExistingPacketFlowOpFromDevice(
      mlir::Value source, xilinx::AIE::WireBundle sourceBundle,
      uint32_t sourceChannel, air::MemcpyInterface memcpyOp) {
    auto chanIfOp =
        dyn_cast_if_present<air::ChannelInterface>(memcpyOp.getOperation());
    if (!chanIfOp)
      return AIE::PacketFlowOp(); // Only air.channel_interface ops support
                                  // packet-flow routing.

    // packetIDForChannelName stores the pkt_id assigned at flow-creation time,
    // keyed by air.channel symbol name. Symbol names are stable across the
    // aie.device / parent module duplication of air.channel decls.
    auto it = packetIDForChannelName.find(chanIfOp.getChanName().str());
    if (it == packetIDForChannelName.end())
      return AIE::PacketFlowOp();
    return findPacketFlowOp(source, sourceBundle, sourceChannel,
                            /*checkFlowID=*/true, it->second);
  }

  /// Query an existing packet flow operation from the runtime function.
  ///
  /// This method is used when looking up packet flows from operations in the
  /// runtime func.func representation, where air.channel symbol linking works
  /// differently than within aie.device. No flowID matching is implemented
  /// because of that(TODO).
  ///
  /// \note This is a simpler lookup than getExistingPacketFlowOpFromDevice
  ///       because it doesn't need to resolve flowID through channel symbols.
  AIE::PacketFlowOp
  getExistingPacketFlowOpFromRuntime(mlir::Value source,
                                     xilinx::AIE::WireBundle sourceBundle,
                                     uint32_t sourceChannel) {
    // Search for the packet flow without flowID checking
    return findPacketFlowOp(source, sourceBundle, sourceChannel,
                            /*checkFlowID=*/false, /*flowID=*/-1);
  }

private:
  /// Core packet flow search logic shared by both device and runtime queries.
  ///
  /// This helper method encapsulates the common packet flow lookup logic,
  /// allowing both getExistingPacketFlowOpFromDevice and
  /// getExistingPacketFlowOpFromRuntime to reuse the same search implementation
  /// while differing only in whether they check the flowID.
  AIE::PacketFlowOp findPacketFlowOp(mlir::Value source,
                                     xilinx::AIE::WireBundle sourceBundle,
                                     uint32_t sourceChannel, bool checkFlowID,
                                     int flowID) {
    AIE::DeviceOp aieDeviceOp =
        source.getParentRegion()->getParentOfType<AIE::DeviceOp>();
    AIE::PacketFlowOp packetFlowOp = nullptr;

    aieDeviceOp.walk([&](AIE::PacketFlowOp pktFlowOp) {
      auto pktSrcOp = getPacketSourceOpInPacketFlowOp(pktFlowOp, source);
      if (!pktSrcOp)
        return;

      auto pktSrcBundle = pktSrcOp.getBundle();
      if (pktSrcBundle != sourceBundle)
        return;

      auto pktSrcChannel = pktSrcOp.getChannel();
      if (pktSrcChannel != (int)sourceChannel)
        return;

      // Only check flowID if requested (device context requires it)
      if (checkFlowID && pktFlowOp.getID() != flowID)
        return;

      packetFlowOp = pktFlowOp;
    });

    return packetFlowOp;
  }

public:
  // Cascade flow.

  // This helper function looks up or creates an AIE::CascadeFlowOp that
  // connects a given source tile to a destination tile.
  AIE::CascadeFlowOp getCascadeFlowOp(AIE::DeviceOp aie_device,
                                      mlir::Value source,
                                      xilinx::AIE::WireBundle sourceBundle,
                                      uint32_t sourceChannel, mlir::Value dest,
                                      xilinx::AIE::WireBundle destBundle,
                                      uint32_t destChannel) {
    AIE::CascadeFlowOp flowOp = nullptr;

    // Search the device for an existing CascadeFlowOp that matches the
    // source/dest tiles.
    aie_device.walk([&](Operation *op) {
      if (auto fop = dyn_cast_if_present<AIE::CascadeFlowOp>(op))
        if (source == fop.getSourceTile() && dest == fop.getDestTile())
          flowOp = fop;
    });

    // If found, return the existing flow.
    if (flowOp)
      return flowOp;

    // Otherwise, create a new CascadeFlowOp at the end of the device body.
    OpBuilder builder(aie_device);
    builder.setInsertionPoint(aie_device.getBody()->getTerminator());
    return AIE::CascadeFlowOp::create(builder, builder.getUnknownLoc(), source,
                                      dest);
  }

  template <typename T>
  void getAIRMemcpyOpInBlock(Block &b, std::vector<Operation *> &output) {
    for (Operation &o : b.getOperations()) {
      if (isa<T>(&o))
        output.push_back(&o);
      for (Region &r : o.getRegions())
        getAIRMemcpyOpInRegion<T>(r, output);
    }
  }

  template <typename T>
  void getAIRMemcpyOpInRegion(Region &r, std::vector<Operation *> &output) {
    for (Block &b : r.getBlocks())
      getAIRMemcpyOpInBlock<T>(b, output);
  }

  /// Insert dummy air.channel.put or air.channel.get operations for L2 memrefs
  /// to ensure that the number of puts and gets match for each buffer.
  /// This helps prevent the risk if a race condition due to imbalanced lock
  /// allocated on both sides.
  ///
  /// For memrefs in L2 memory space:
  ///   - If there are more gets than puts: clone and insert dummy puts before
  ///   the first put
  ///   - If there are more puts than gets: clone and insert dummy gets after
  ///   the last get
  ///
  /// \param aieDevice The surrounding aie.device operation to walk.
  /// \param builder An OpBuilder to insert new operations.
  ///
  void insertDummyChannelOpsForL2Memrefs(AIE::DeviceOp aieDevice,
                                         OpBuilder &builder) {
    // Map from L2 memref -> (list of puts, list of gets).
    // Use DenseMap for fast lookup, but track walk-order in a separate vector
    // to ensure deterministic iteration (DenseMap iterates in pointer-hash
    // order which varies with binary layout).
    llvm::DenseMap<Value, std::pair<llvm::SmallVector<air::ChannelPutOp>,
                                    llvm::SmallVector<air::ChannelGetOp>>>
        l2MemrefPutsGets;
    llvm::SmallVector<Value> l2MemrefOrder;

    // Walk all ChannelInterface ops under the device and categorize puts/gets
    // on L2 memrefs
    aieDevice.walk<mlir::WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [&](air::ChannelInterface chanI) {
          auto memrefTy =
              dyn_cast_if_present<BaseMemRefType>(chanI.getMemref().getType());
          if (!memrefTy || !air::isL2(memrefTy))
            return mlir::WalkResult::advance();

          Value memref = chanI.getMemref();
          if (!l2MemrefPutsGets.count(memref))
            l2MemrefOrder.push_back(memref);

          if (auto chanPut =
                  dyn_cast_if_present<air::ChannelPutOp>(chanI.getOperation()))
            l2MemrefPutsGets[memref].first.push_back(chanPut);
          else if (auto chanGet = dyn_cast_if_present<air::ChannelGetOp>(
                       chanI.getOperation()))
            l2MemrefPutsGets[memref].second.push_back(chanGet);

          return mlir::WalkResult::advance();
        });

    // Balance puts and gets by inserting dummy ops (iterate in walk order
    // for deterministic output regardless of binary layout)
    for (Value memref : l2MemrefOrder) {
      auto &putsAndGets = l2MemrefPutsGets[memref];
      auto &[puts, gets] = putsAndGets;
      if (puts.empty() || gets.empty())
        continue; // Skip buffers that only appear in one direction

      unsigned numOpsToClone = 0;
      Operation *templateOp = nullptr;

      // Determine imbalance pattern and insertion point
      if (puts.size() < gets.size()) {
        // "Join" pattern — add dummy puts
        builder.setInsertionPoint(puts.front());
        templateOp = puts.front();
        numOpsToClone = gets.size() - puts.size();
      } else if (gets.size() < puts.size()) {
        // "Distribute" pattern — add dummy gets
        builder.setInsertionPointAfter(gets.back());
        templateOp = gets.back();
        numOpsToClone = puts.size() - gets.size();
      } else {
        continue; // Already balanced
      }

      // Constants for dummy sizes: zero offset, one element
      Value zeroIdx =
          arith::ConstantIndexOp::create(builder, builder.getUnknownLoc(), 0);
      Value oneIdx =
          arith::ConstantIndexOp::create(builder, builder.getUnknownLoc(), 1);

      // Use the original op as a template to emit new dummy ops
      auto templateAsyncIf =
          dyn_cast_if_present<air::AsyncOpInterface>(templateOp);
      auto templateChanIf =
          dyn_cast_if_present<air::ChannelInterface>(templateOp);
      assert(templateAsyncIf && templateChanIf &&
             "Expected valid async/channel op");

      for (unsigned i = 0; i < numOpsToClone; ++i) {
        if (isa<air::ChannelPutOp>(templateOp)) {
          air::ChannelPutOp::create(
              builder, templateOp->getLoc(), templateOp->getResultTypes(),
              templateAsyncIf.getAsyncDependencies(),
              templateChanIf.getChanName(), templateChanIf.getIndices(),
              templateChanIf.getMemref(),
              /*sizes*/ SmallVector<Value>{zeroIdx},
              /*offsets*/ SmallVector<Value>{zeroIdx},
              /*steps*/ SmallVector<Value>{oneIdx},
              /*pad_before=*/nullptr, /*pad_after=*/nullptr);
        } else if (isa<air::ChannelGetOp>(templateOp)) {
          air::ChannelGetOp::create(
              builder, templateOp->getLoc(), templateOp->getResultTypes(),
              templateAsyncIf.getAsyncDependencies(),
              templateChanIf.getChanName(), templateChanIf.getIndices(),
              templateChanIf.getMemref(),
              /*sizes*/ SmallVector<Value>{zeroIdx},
              /*offsets*/ SmallVector<Value>{zeroIdx},
              /*steps*/ SmallVector<Value>{oneIdx},
              /*pad_before=*/nullptr, /*pad_after=*/nullptr);
        }
      }
    }
  }

  // Clone data movement ops to and from memtile and shim tile DMAs
  // If targetLaunch is provided, only clone ops from that specific launch.
  void cloneL2AndL3MemcpysToDeviceOp(OpBuilder &builder,
                                     AIE::DeviceOp aie_device, ModuleOp module,
                                     bool clone_l2, bool clone_l3,
                                     bool lock_race_condition_fix = true,
                                     air::LaunchOp targetLaunch = nullptr) {

    if (!clone_l2 && !clone_l3)
      return;

    auto ctx = builder.getContext();

    Operation *t = nullptr;
    for (auto tile_op : aie_device.getBody()->getOps<AIE::TileOp>()) {
      t = tile_op.getOperation();
    }
    builder.setInsertionPointAfter(t);
    IRMapping remap;

    // Set up segment operand -> constant remapping.
    // For unrolled segments (totalUnroll > 1), use the stored unroll indices.
    // For non-unrolled segments (totalUnroll == 1), also remap segment IDs to
    // constant 0 so that channel ops using segment indices as channel bundle
    // positions get properly specialized (e.g., ChannelPut with indices=[seg_x]
    // becomes indices=[0]).
    {
      int64_t unrollX = 0;
      int64_t unrollY = 0;
      if (auto unrollXAttr =
              aie_device->getAttrOfType<IntegerAttr>("segment_unroll_x"))
        unrollX = unrollXAttr.getInt();
      if (auto unrollYAttr =
              aie_device->getAttrOfType<IntegerAttr>("segment_unroll_y"))
        unrollY = unrollYAttr.getInt();
      for (auto func : module.getOps<func::FuncOp>()) {
        func.walk([&](air::SegmentOp segOp) {
          auto segIds = segOp.getIds();
          if (segIds.size() >= 1) {
            remap.map(segIds[0],
                      arith::ConstantIndexOp::create(
                          builder, builder.getUnknownLoc(), unrollX));
          }
          if (segIds.size() >= 2) {
            remap.map(segIds[1],
                      arith::ConstantIndexOp::create(
                          builder, builder.getUnknownLoc(), unrollY));
          }
        });
      }
    }

    // Map index-typed segment kernel arguments to constant 0.  When a
    // segment receives launch iteration indices as kernel arguments (e.g.,
    // for computing L3 subview offsets), those SSA values live outside the
    // aie.device's isolated-from-above region.  At the device level the
    // actual offsets are handled by the shimDMA / NPU instruction sequence,
    // so zero is the correct placeholder.
    for (auto func : module.getOps<func::FuncOp>()) {
      func.walk([&](air::SegmentOp segOp) {
        for (unsigned i = 0, e = segOp.getNumKernelOperands(); i < e; i++) {
          Value karg = segOp.getKernelArgument(i);
          if (!isa<IndexType>(karg.getType()))
            continue;
          if (remap.contains(karg))
            continue;
          remap.map(karg, arith::ConstantIndexOp::create(
                              builder, builder.getUnknownLoc(), 0));
        }
      });
    }

    // Pre-create AIE::ExternalBufferOp for any L3 memrefs that will be used
    // by cloned ops. This is necessary because aie.device is an isolated-from-
    // above region and cannot reference values defined outside it.
    llvm::DenseSet<Value> l3MemrefsHandled;
    for (auto func : module.getOps<func::FuncOp>()) {
      func.walk([&](Operation *op) {
        // Skip ops that won't be cloned
        if (isa<air::LaunchOp, func::FuncOp, air::HerdOp>(op))
          return WalkResult::advance();
        if (isa<air::LaunchTerminatorOp, air::SegmentTerminatorOp,
                func::ReturnOp, air::WaitAllOp>(op))
          return WalkResult::advance();
        // Filter by target launch
        if (targetLaunch) {
          auto parentLaunch = op->getParentOfType<air::LaunchOp>();
          if (!parentLaunch || parentLaunch != targetLaunch)
            return WalkResult::advance();
        }
        // Check for L3 memref operands
        for (auto operand : op->getOperands()) {
          auto memrefTy = dyn_cast_if_present<MemRefType>(operand.getType());
          if (!memrefTy)
            continue;
          if (!air::isL3(memrefTy))
            continue;
          // Skip if already handled
          if (l3MemrefsHandled.contains(operand))
            continue;
          l3MemrefsHandled.insert(operand);

          // Create AIE::ExternalBufferOp for this L3 memref
          std::string sym_name = createSymbolName(aie_device.getOperation(),
                                                  "__air_external_buffer");
          auto extBuf = AIE::ExternalBufferOp::create(
              builder, builder.getUnknownLoc(), memrefTy,
              builder.getStringAttr(sym_name), /*address=*/nullptr);
          remap.map(operand, extBuf.getResult());
        }
        return WalkResult::advance();
      });
    }

    SmallVector<func::FuncOp> funcs;
    module.walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [&](func::FuncOp f) {
          funcs.push_back(f);
          return WalkResult::advance();
        });
    for (auto f : funcs) {
      f.walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
          [&](Operation *op) {
            if (isa<air::LaunchOp, func::FuncOp>(op))
              return WalkResult::advance();
            if (isa<air::SegmentOp>(op) && clone_l2)
              return WalkResult::advance();
            if (isa<air::HerdOp>(op))
              return WalkResult::skip();
            if (isa<air::LaunchTerminatorOp, air::SegmentTerminatorOp,
                    func::ReturnOp, air::WaitAllOp>(op))
              return WalkResult::advance();
            bool hasParentSegmentOp = op->getParentOfType<air::SegmentOp>();
            if (!clone_l3 && !hasParentSegmentOp)
              return WalkResult::advance();
            // Filter by target launch: if a targetLaunch is specified, only
            // clone ops that belong to that launch
            if (targetLaunch) {
              auto parentLaunch = op->getParentOfType<air::LaunchOp>();
              // If the op is not inside any launch or is inside a different
              // launch, do not clone it.
              if (!parentLaunch || parentLaunch != targetLaunch)
                return WalkResult::advance();
            }
            builder.clone(*op, remap);
            return WalkResult::skip();
          });
    }

    // Remove ops which are irrelevant to L2 and L3 data movements.
    aie_device.walk([ctx](air::HierarchyInterface hierOp) {
      OpBuilder b(hierOp);
      for (auto r : hierOp->getResults()) {
        if (isa<air::AsyncTokenType>(r.getType())) {
          r.replaceAllUsesWith(
              air::WaitAllOp::create(b, hierOp->getLoc(),
                                     air::AsyncTokenType::get(ctx),
                                     air::getAsyncDependenciesFromOp(hierOp))
                  .getAsyncToken());
        }
      }
      hierOp->erase();
    });

    // Unroll scf.parallel
    RewritePatternSet patterns(ctx);
    air::populateAIRunrollAIRChannelPutGetInScfParallelPatterns(patterns);
    (void)applyPatternsGreedily(aie_device, std::move(patterns));

    // Substituting index operands, such as strides and offsets, to constant
    // zero for convenience. TODO: generalize this
    aie_device.walk([](air::ChannelInterface chanI) {
      OpBuilder b(chanI);
      for (auto oper : llvm::concat<Value>(chanI.getOffsets(), chanI.getSizes(),
                                           chanI.getStrides())) {
        if (!getConstantIntValue(oper)) {
          chanI->replaceUsesOfWith(
              oper, arith::ConstantIndexOp::create(b, b.getUnknownLoc(), 0));
        }
      }
    });

    // Generate dummy air.channel ops to balance the number of BDs at either
    // side of an L2 buffer, to protect against risks of race conditions.
    if (lock_race_condition_fix) {
      insertDummyChannelOpsForL2Memrefs(aie_device, builder);
    }
  }

  bool everyAIRChannelAccessIsContiguousRowMajor(
      std::vector<air::ChannelInterface> ops) {
    for (auto op : ops) {
      auto memref = op.getMemref();
      auto memrefShape = air::getTensorShape(memref.getType());
      // The default data access pattern is contiguous and row major.
      if (air::isDefaultDataAccessPattern(op.getSizes(), op.getStrides()))
        continue;
      if (op.getStrides().size() != memrefShape.size())
        return false;
      int current_stride = 1;
      for (int i = op.getStrides().size() - 1; i >= 0; i--) {
        if (*getConstantIntValue(op.getStrides()[i]) != current_stride)
          return false;
        current_stride *= memrefShape[i];
      }
    }
    return true;
  }
  // Check whether every channel op in the vector has non-overlapping access
  // pattern, by exhaustively scan through pairs of air channel ops in the
  // vector.
  bool everyAIRChannelAccessIsNonOverlapping(
      std::vector<air::ChannelInterface> ops) {
    if (!everyAIRChannelAccessIsContiguousRowMajor(ops))
      return false; // Incontiguous or not row-major, NYI.
    for (unsigned i = 0; i < ops.size() - 1; i++) {
      for (unsigned j = i + 1; j < ops.size(); j++) {
        air::ChannelInterface op1 = ops[i];
        air::ChannelInterface op2 = ops[j];
        if (op1.getOffsets().size() != op2.getOffsets().size())
          return false;
        if (op1.getSizes().size() != op2.getSizes().size())
          return false;
        bool isOverlappingPair =
            true; // True if every dimension is overlapping.
        for (unsigned k = 0; k < op1.getOffsets().size(); k++) {
          int op1Offset = *getConstantIntValue(op1.getOffsets()[k]);
          int op2Offset = *getConstantIntValue(op2.getOffsets()[k]);
          int op1LowerRange = op1Offset;
          int op1UpperRange =
              op1Offset + *getConstantIntValue(op1.getSizes()[k]);
          int op2LowerRange = op2Offset;
          int op2UpperRange =
              op2Offset + *getConstantIntValue(op2.getSizes()[k]);
          bool isOverlappingDim = false;
          if (op1Offset >= op2LowerRange && op1Offset < op2UpperRange)
            isOverlappingDim = true;
          else if (op2Offset >= op1LowerRange && op2Offset < op1UpperRange)
            isOverlappingDim = true;
          if (!isOverlappingDim)
            isOverlappingPair = false;
        }
        if (isOverlappingPair)
          return false;
      }
    }
    return true;
  }
  bool
  everyAIRChannelAccessIsNonOverlapping(std::vector<air::ChannelPutOp> &ops) {
    std::vector<air::ChannelInterface> chanOps;
    for (auto op : ops)
      chanOps.push_back(op);
    return everyAIRChannelAccessIsNonOverlapping(chanOps);
  }
  bool
  everyAIRChannelAccessIsNonOverlapping(std::vector<air::ChannelGetOp> &ops) {
    std::vector<air::ChannelInterface> chanOps;
    for (auto op : ops)
      chanOps.push_back(op);
    return everyAIRChannelAccessIsNonOverlapping(chanOps);
  }
  bool hasSinglePutAndGet(air::ChannelOp chan) {
    auto puts = getChannelPutOpThroughSymbol(
        chan, chan->getParentOfType<AIE::DeviceOp>());
    auto gets = getChannelGetOpThroughSymbol(
        chan, chan->getParentOfType<AIE::DeviceOp>());
    return puts.size() == 1 && gets.size() == 1;
  }

  void partitionMemref(std::vector<air::ChannelPutOp> &puts,
                       std::vector<air::ChannelGetOp> &gets) {
    std::map<int, SmallVector<air::ChannelInterface>> chanOpPartitions;
    std::vector<int> keys;
    for (auto op : puts) {
      if (op.getOffsets().empty())
        return; // Default access pattern (full memref), cannot partition.
      int firstOffset = *getConstantIntValue(op.getOffsets().front());
      push_back_if_unique<int>(keys, firstOffset);
      if (!chanOpPartitions.count(firstOffset))
        chanOpPartitions[firstOffset] = SmallVector<air::ChannelInterface>{op};
      else
        chanOpPartitions[firstOffset].push_back(op);
    }
    for (auto op : gets) {
      if (op.getOffsets().empty())
        return; // Default access pattern (full memref), cannot partition.
      int firstOffset = *getConstantIntValue(op.getOffsets().front());
      push_back_if_unique<int>(keys, firstOffset);
      if (!chanOpPartitions.count(firstOffset))
        chanOpPartitions[firstOffset] = SmallVector<air::ChannelInterface>{op};
      else
        chanOpPartitions[firstOffset].push_back(op);
    }
    for (auto key : keys) {
      auto memref = chanOpPartitions[key][0].getMemref();
      auto allocOp = memref.getDefiningOp();
      MemRefType ty = llvm::cast<MemRefType>(memref.getType());
      SmallVector<int64_t> newMemrefShape;
      for (unsigned i = 0; i < air::getTensorShape(ty).size(); i++) {
        newMemrefShape.push_back(air::getTensorShape(ty)[i]);
      }
      for (auto op : chanOpPartitions[key])
        if (op.getSizes().size() == newMemrefShape.size()) {
          newMemrefShape.front() = *getConstantIntValue(op.getSizes().front());
          break;
        }

      OpBuilder builder(allocOp);
      auto loc = allocOp->getLoc();
      Value newMemref = memref::AllocOp::create(
          builder, loc,
          MemRefType::get(newMemrefShape, ty.getElementType(),
                          ty.getLayout().getAffineMap(), ty.getMemorySpace()));
      for (auto op : chanOpPartitions[key]) {
        int memrefOperandOffset =
            dyn_cast_if_present<air::AsyncOpInterface>(op.getOperation())
                .getAsyncDependencies()
                .size();
        auto &memrefOpOper = op->getOpOperand(memrefOperandOffset);
        memrefOpOper.assign(newMemref);
        int firstOffsetOperandOffset = memrefOperandOffset + 1;
        auto &firstOffsetOpOper = op->getOpOperand(firstOffsetOperandOffset);
        firstOffsetOpOper.assign(
            arith::ConstantIndexOp::create(builder, loc, 0));
        // Update strides (contiguous, row-major) after memref tiling.
        SmallVector<Value> offsets;
        SmallVector<Value> wraps;
        SmallVector<Value> strides;
        // One dimensional default stride value.
        if (op.getSizes().size() == 1)
          strides.push_back(arith::ConstantIndexOp::create(builder, loc, 1));
        else
          air::populateDefaultWrapsAndStrides(builder, newMemref, offsets,
                                              wraps, strides);
        int firstStrideOperandOffset =
            memrefOperandOffset + op.getOffsets().size() * 2 + 1;
        for (unsigned i = 0; i < op.getStrides().size(); i++) {
          auto &strideOpOper = op->getOpOperand(firstStrideOperandOffset + i);
          strideOpOper.assign(strides[i]);
        }
      }
    }
  }

  // Optimize L2 (memtile) buffer allocation by attempting to partition
  // non-overlapping L2 memref accesses into (upto M, where M is the number of
  // memtiles being allocated to) separate memrefs.
  void specializeL2MemrefsIntoMemtiles(AIE::DeviceOp d) {
    // Get all memtiles to place L2 memrefs onto.
    std::vector<AIE::TileLike> memtiles = getMemtilesFromDeviceOp(d);
    if (memtiles.empty())
      return;
    int maxMemtileSrcConnections =
        memtiles[0].getNumSourceConnections(AIE::WireBundle::DMA);
    int maxMemtileDstConnections =
        memtiles[0].getNumDestConnections(AIE::WireBundle::DMA);

    // Get L2 memrefs which require partitioning, due to having more channel
    // puts/gets than memtile hardware limit.
    std::vector<Value> memrefs;
    d.walk([&](memref::AllocOp allocOp) {
      auto memref = allocOp.getMemref();
      auto memrefTy = llvm::cast<MemRefType>(memref.getType());
      if (air::isL2(memrefTy)) {
        // Count the number of unique incoming and outgoing channels.
        std::vector<std::string> uniqueS2MMChannels;
        std::vector<std::string> uniqueMM2SChannels;
        for (auto user : memref.getUsers()) {
          if (auto get = dyn_cast_if_present<air::ChannelGetOp>(user))
            push_back_if_unique<std::string>(uniqueS2MMChannels,
                                             get.getChanName().str());
          else if (auto put = dyn_cast_if_present<air::ChannelPutOp>(user))
            push_back_if_unique<std::string>(uniqueMM2SChannels,
                                             put.getChanName().str());
        }
        bool tooManyChannelConnections =
            (int)uniqueS2MMChannels.size() > maxMemtileDstConnections ||
            (int)uniqueMM2SChannels.size() > maxMemtileSrcConnections;
        if (tooManyChannelConnections) {
          if (auto exec =
                  dyn_cast_if_present<air::ExecuteOp>(allocOp->getParentOp()))
            memrefs.push_back(exec->getResult(1));
          else
            memrefs.push_back(memref);
        }
      }
    });
    if (memrefs.empty())
      return;

    // Tile the memrefs based on air.channel put/get access pattern.
    for (auto memref : memrefs) {
      std::vector<air::ChannelPutOp> puts;
      std::vector<air::ChannelGetOp> gets;
      for (auto user : memref.getUsers()) {
        if (auto put = dyn_cast_if_present<air::ChannelPutOp>(user))
          puts.push_back(put);
        else if (auto get = dyn_cast_if_present<air::ChannelGetOp>(user))
          gets.push_back(get);
      }
      if (everyAIRChannelAccessIsNonOverlapping(gets) &&
          everyAIRChannelAccessIsNonOverlapping(puts)) {
        partitionMemref(puts, gets);
      } else
        continue; // Multiple of puts and multiple of gets, NYI.
    }
  }

  template <typename T>
  LogicalResult
  placeDMAChannelsAndRouteFlows(AIE::DeviceOp aie_device,
                                air::ShimDMAAllocator &shim_dma_alloc,
                                air::MemTileDMAAllocator &memtile_dma_alloc,
                                air::TileDMAAllocator &tile_dma_alloc,
                                air::CascadeAllocator &core_cascade_alloc,
                                AIRToAIEConversionOptions options) {

    std::vector<Operation *> dma_memcpy_ops;

    aie_device.walk(
        [&](T memcpyOp) { dma_memcpy_ops.push_back(memcpyOp.getOperation()); });

    // Step 1: Pair up memcpy ops into flow ops. Each entry in memcpy_flows is a
    // bundle of memcpy ops which share the same aie.flow.
    std::vector<air::MemcpyBundleAsFlow> memcpy_flows;
    for (auto o : dma_memcpy_ops) {
      if (auto dma = dyn_cast_if_present<air::DmaMemcpyNdOp>(o)) {
        // DMA memcpy always creates a new flow bundle.
        air::MemcpyBundleAsFlow flow = air::MemcpyBundleAsFlow(dma);
        if (failed(flow.pushBackMemcpyOpToBundle(dma)))
          return failure();
        memcpy_flows.push_back(flow);
      } else if (auto putget = dyn_cast_if_present<air::ChannelInterface>(o)) {
        // Lookup channel declaration.
        auto chan = air::getChannelDeclarationThroughSymbol(putget);
        if (!chan) {
          putget->emitOpError("failed to get air.channel declaration.");
          return failure();
        }
        std::string chan_name = putget.getChanName().str();
        // Check if new pair
        bool found_in_flows = false;
        for (auto &f : memcpy_flows) {
          auto air_flow_op_chan =
              dyn_cast_if_present<air::ChannelOp>(f.air_flow_op);
          if (!air_flow_op_chan)
            continue;
          if (chan_name != air_flow_op_chan.getSymName().str())
            continue;
          if (failed(f.pushBackMemcpyOpToBundle(putget)))
            return failure();
          found_in_flows = true;
        }
        if (!found_in_flows) {
          // Create new entry in memcpy_flows
          air::MemcpyBundleAsFlow flow = air::MemcpyBundleAsFlow(chan);
          if (failed(flow.pushBackMemcpyOpToBundle(putget)))
            return failure();
          memcpy_flows.push_back(flow);
        }
      } else {
        return o->emitOpError(
            "unknown memcpy op type. Expected air::MemcpyInterface.");
      }
    }

    // Align shared-MM2S packet pkt_ids and launch-side L3 put IR order with
    // the receiver mem chain (herd-source order). See helper docs.
    air::sortPacketShimFlowsByReceiverOrder(memcpy_flows, aie_device);
    air::reorderL3PacketPutsByFlowOrder(aie_device, memcpy_flows);

    // Step 2: Allocate tile DMA channels, shim DMA channels and shim tiles
    auto r = simpleDMAChannelAllocation(memcpy_flows, shim_dma_alloc,
                                        memtile_dma_alloc, tile_dma_alloc,
                                        core_cascade_alloc);
    if (failed(r))
      return r;

    // Step 3: Sort all ops being allocated to each DMA channel, to avoid
    // ping-pong deadlock.
    tile_dma_alloc.sortMemcpyOps(dma_memcpy_ops);

    // Step 4: Connect flows.
    //
    // Packet flows are assigned pkt_ids in two passes so that within one
    // device, no two packet flows share a pkt_id. (Pre-fix: shim and intra-
    // device flows ran independent 0-based counters, the first of each got
    // pkt_id=0, and switchbox arbiters silently mis-routed where their
    // physical routes crossed.) Each air.channel symbol maps to exactly one
    // pkt_id, recorded in packetIDForChannelName for broadcast reuse and for
    // lookup by getExistingPacketFlowOpFromDevice.
    //
    //   Pass 1: shim packet flows (assignShimPacketID -- monotonic global).
    //   Pass 2: intra-device packet flows (assignIntraDevicePacketID --
    //           lowest gap in claimedPacketIDs).
    //   Pass 3: stream and cascade flows (no pkt_id involved).
    auto isInvalidAlloc = [&](air::MemcpyBundleAsFlow &f, int j, int i) {
      if (!f.MM2S_alloc[j].getDmaTile() || !f.S2MM_alloc[i].getDmaTile()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "AIRToAIE: skipping memcpy flow due to invalid DMA "
                      "tile allocation (MM2S or S2MM tile is null)\n");
        return true;
      }
      return false;
    };
    auto isShimFlowAt = [&](air::MemcpyBundleAsFlow &f, int j, int i) {
      return f.MM2S_alloc[j].getDmaTile().isShimNOCorPLTile() ||
             f.S2MM_alloc[i].getDmaTile().isShimNOCorPLTile();
    };
    // The flow-op key for packetIDForChannelName is the air.channel symbol
    // name. air.channel decls are duplicated under aie.device and its parent
    // module, so symbol name is the stable identifier across both contexts.
    auto getChannelName = [](air::MemcpyBundleAsFlow &f) -> std::string {
      if (auto chanOp = dyn_cast_if_present<air::ChannelOp>(f.air_flow_op))
        return chanOp.getSymName().str();
      return {};
    };
    // Assign or reuse a pkt_id for a packet flow. air.channel-rooted flows
    // dedupe by symbol name (broadcast: one source -> multiple receivers
    // share a pkt_id). Non-channel packet flows get a fresh id per call.
    auto assignOrLookupPacketID = [&](air::MemcpyBundleAsFlow &f,
                                      bool isShim) -> int {
      std::string chanName = getChannelName(f);
      if (chanName.empty())
        return isShim ? assignShimPacketID() : assignIntraDevicePacketID();
      auto [it, inserted] = packetIDForChannelName.try_emplace(chanName, -1);
      if (inserted)
        it->second =
            isShim ? assignShimPacketID() : assignIntraDevicePacketID();
      return it->second;
    };

    // Explicit per-destination packet ids pinned via the channel's `packet_ids`
    // array attribute. Read on-demand from the channel op (kept off the
    // MemcpyBundleAsFlow struct to avoid bloating it past the SmallVector
    // inline-size budget).
    auto getPinnedIDs = [](air::MemcpyBundleAsFlow &f) -> SmallVector<int> {
      SmallVector<int> ids;
      if (auto chanOp = dyn_cast_if_present<air::ChannelOp>(f.air_flow_op))
        if (auto attr = chanOp.getPacketIDs())
          for (auto idAttr : attr)
            // Verifier guarantees IntegerAttr elements in [0,31]; dyn_cast
            // keeps the pass robust on unverified IR (skip non-integer
            // entries).
            if (auto idInt = dyn_cast<IntegerAttr>(idAttr))
              ids.push_back((int)idInt.getInt());
      return ids;
    };

    // Pre-claim every explicitly-pinned packet id so the auto-assigner for
    // other flows skips them.
    for (auto &f : memcpy_flows)
      if (f.memcpyResourceType == "npu_dma_packet")
        for (int id : getPinnedIDs(f))
          claimedPacketIDs.insert(id);

    // Packet ids to emit for destination i. Normally one id (the shared
    // auto-assigned id memoized by channel name). When the channel pins
    // `packet_ids`:
    //   - N pinned ids with a SINGLE destination emits N flows (one per id) to
    //     that one dest -- every id routes to the same buffer and the
    //     kernel-written header carries the phase through to a later demux hop.
    //   - N pinned ids with multiple dests demuxes per-destination (dest i uses
    //     pinned[i]).
    //   - a SINGLE pinned id (incl. same-id N-producer convergence) is recorded
    //     in packetIDForChannelName so generateDmaBd STAMPS every producer
    //     (MM2S) and receiver (S2MM) BD with that id. Without this the pinned
    //     path left the map unset, so producers emitted UNTAGGED packets and
    //     the switchbox could not route N same-id sources onto one slave port.
    //   - MULTIPLE pinned ids are DELIBERATELY not recorded in
    //     packetIDForChannelName: under the pinned-multi-id contract the
    //     compute core writes the routing id into the payload header, so the
    //     DMA must NOT stamp/filter (generateDmaBd +
    //     labelMemcpyOpsWithPacketFlow both skip a channel with no map entry /
    //     >1 pinned id). The packet_flow ops alone install the switchbox
    //     routes.
    auto pktIDsForDest = [&](air::MemcpyBundleAsFlow &f, int i,
                             bool isShim) -> SmallVector<int> {
      auto pinned = getPinnedIDs(f);
      if (pinned.empty())
        return {assignOrLookupPacketID(f, isShim)};
      if (pinned.size() == 1) {
        std::string chanName = getChannelName(f);
        if (!chanName.empty())
          packetIDForChannelName[chanName] = pinned[0];
      }
      if (f.numS2MMAllocs == 1)
        return pinned; // N ids -> the single destination
      if (i < (int)pinned.size())
        return {pinned[i]}; // per-destination demux
      // Fewer pinned ids than destinations: the surplus dests fall back to
      // auto-assigned ids, silently mixing pinned + auto. Warn so this is not a
      // surprise -- a well-formed demux pins exactly one id per destination.
      if (auto chanOp = dyn_cast_if_present<air::ChannelOp>(f.air_flow_op))
        chanOp->emitWarning()
            << "packet_ids pins " << pinned.size() << " id(s) but channel has "
            << f.numS2MMAllocs << " destinations; destination " << i
            << " falls back to an auto-assigned id";
      return {assignOrLookupPacketID(f, isShim)};
    };

    // Pass 1: shim packet flows. Nested over producers (j) x dests (i): one
    // packet_flow per (producer, dest). For multi-producer convergence the
    // shared dest pkt id is memoized by channel name in assignOrLookupPacketID,
    // so all producers of one channel land on the dest S2MM with the SAME id.
    for (auto &f : memcpy_flows) {
      if (f.memcpyResourceType != "npu_dma_packet")
        continue;
      for (int i = 0; i < f.numS2MMAllocs; i++) {
        for (int j = 0; j < f.numMM2SAllocs; j++) {
          if (isInvalidAlloc(f, j, i))
            continue;
          if (!isShimFlowAt(f, j, i))
            continue;
          auto flowIDs = pktIDsForDest(f, i, /*isShim=*/true);
          // A shim source feeding >1 pinned id follows the kernel-header
          // contract (the core stamps the header), so no single id backlinks to
          // the shim alloc -- recording one would falsely tag the MM2S op with
          // a partial id (the last, pre-fix). Only the single-id path
          // backlinks.
          bool backlink = flowIDs.size() == 1;
          for (int flowID : flowIDs) {
            getPacketFlowOp(
                aie_device, f.MM2S_alloc[j].getDmaTile()->getResult(0),
                AIE::WireBundle::DMA,
                (uint32_t)f.MM2S_alloc[j].dma_channel.channel,
                f.S2MM_alloc[i].getDmaTile()->getResult(0),
                AIE::WireBundle::DMA,
                (uint32_t)f.S2MM_alloc[i].dma_channel.channel, flowID);
            if (!backlink)
              continue;
            // Backlink: the host runtime keys packet identification on
            // (shim tile, pkt_id), so the matching MM2S shim alloc carries the
            // id.
            for (auto &sa : shim_dma_alloc.mm2s_allocs) {
              if (sa.getDmaTile().getOperation() ==
                      f.MM2S_alloc[j].getDmaTile().getOperation() &&
                  sa.dma_channel == f.MM2S_alloc[j].dma_channel &&
                  sa.col == f.MM2S_alloc[j].col &&
                  sa.row == f.MM2S_alloc[j].row &&
                  sa.dma_id == f.MM2S_alloc[j].dma_id) {
                sa.packet_flow_id = flowID;
                break;
              }
            }
          }
        }
      }
    }

    // Pass 2: intra-device packet flows.
    for (auto &f : memcpy_flows) {
      if (f.memcpyResourceType != "npu_dma_packet")
        continue;
      for (int i = 0; i < f.numS2MMAllocs; i++) {
        for (int j = 0; j < f.numMM2SAllocs; j++) {
          if (isInvalidAlloc(f, j, i))
            continue;
          if (isShimFlowAt(f, j, i))
            continue;
          for (int flowID : pktIDsForDest(f, i, /*isShim=*/false)) {
            getPacketFlowOp(
                aie_device, f.MM2S_alloc[j].getDmaTile()->getResult(0),
                AIE::WireBundle::DMA,
                (uint32_t)f.MM2S_alloc[j].dma_channel.channel,
                f.S2MM_alloc[i].getDmaTile()->getResult(0),
                AIE::WireBundle::DMA,
                (uint32_t)f.S2MM_alloc[i].dma_channel.channel, flowID);
          }
        }
      }
    }

    // Pass 3: stream and cascade flows.
    for (auto &f : memcpy_flows) {
      if (f.memcpyResourceType != "npu_dma_stream" &&
          f.memcpyResourceType != "npu_cascade")
        continue;
      for (int i = 0; i < f.numS2MMAllocs; i++) {
        for (int j = 0; j < f.numMM2SAllocs; j++) {
          if (isInvalidAlloc(f, j, i))
            continue;
          if (f.memcpyResourceType == "npu_dma_stream")
            getFlowOp(aie_device, f.MM2S_alloc[j].getDmaTile()->getResult(0),
                      AIE::WireBundle::DMA,
                      (uint32_t)f.MM2S_alloc[j].dma_channel.channel,
                      f.S2MM_alloc[i].getDmaTile()->getResult(0),
                      AIE::WireBundle::DMA,
                      (uint32_t)f.S2MM_alloc[i].dma_channel.channel);
          else
            getCascadeFlowOp(aie_device,
                             f.MM2S_alloc[j].getDmaTile()->getResult(0),
                             AIE::WireBundle::DMA,
                             (uint32_t)f.MM2S_alloc[j].dma_channel.channel,
                             f.S2MM_alloc[i].getDmaTile()->getResult(0),
                             AIE::WireBundle::DMA,
                             (uint32_t)f.S2MM_alloc[i].dma_channel.channel);
        }
      }
    }
    return success();
  }

  void getDmaAllocationMetadata(OpBuilder builder, MLIRContext *ctx,
                                air::HierarchyInterface op,
                                std::vector<air::allocation_info_t> allocs,
                                AIE::DMAChannelDir dir,
                                std::map<int, int> chan_renumber_reverse_map,
                                std::vector<Attribute> &dma_allocations) {

    std::set<int64_t> dma_ids;
    op.walk([&](air::MemcpyInterface o) {
      if (isa<air::HerdOp>(op))
        dma_ids.insert(o.getId());
      else if (!o->getParentOfType<air::HerdOp>())
        dma_ids.insert(o.getId());
    });

    int64_t col_offset = 0;
    int64_t row_offset = 0;
    if (auto herd = dyn_cast_if_present<air::HerdOp>(op.getOperation())) {
      auto c = herd.getColOffset();
      auto r = herd.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    } else if (auto seg =
                   dyn_cast_if_present<air::SegmentOp>(op.getOperation())) {
      auto c = seg.getColOffset();
      auto r = seg.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    } else {
      return; // failure();
    }

    for (auto &t : allocs) {
      // Shim DMA tiles are emitted as logical tiles by ShimDMAAllocator and
      // resolved to physical TileOps by mlir-aie's `aie-place-tiles` pass,
      // which runs (in aircc) BEFORE this metadata is consumed. At AIR-to-AIE
      // time the col is therefore not yet known; write tryGetCol() and
      // accept -1 when unplaced. The downstream metadata-fixup pass (run
      // after aie-place-tiles) patches the "location" field for entries
      // whose shim tile got a physical column from the placer.
      AIE::TileLike tileLike = t.getDmaTile();
      int64_t shimCol = tileLike ? tileLike.tryGetCol().value_or(-1) : -1;
      int64_t col = t.col - col_offset;
      int64_t row = t.row - row_offset;
      int64_t chan = dir == AIE::DMAChannelDir::MM2S ? t.dma_channel.channel + 2
                                                     : t.dma_channel.channel;

      for (int64_t id : t.dma_id) {
        int original_id = chan_renumber_reverse_map.size()
                              ? chan_renumber_reverse_map[id]
                              : id;
        if (dma_ids.count(original_id) == 0)
          continue;
        SmallVector<NamedAttribute, 5> attrs;
        attrs.push_back(NamedAttribute(StringAttr::get(ctx, "id"),
                                       builder.getI64IntegerAttr(original_id)));
        attrs.push_back(NamedAttribute(StringAttr::get(ctx, "row"),
                                       builder.getI64IntegerAttr(row)));
        attrs.push_back(NamedAttribute(StringAttr::get(ctx, "col"),
                                       builder.getI64IntegerAttr(col)));
        attrs.push_back(NamedAttribute(StringAttr::get(ctx, "channel"),
                                       builder.getI64IntegerAttr(chan)));
        attrs.push_back(NamedAttribute(StringAttr::get(ctx, "location"),
                                       builder.getI64IntegerAttr(shimCol)));
        push_back_if_unique<Attribute>(dma_allocations,
                                       DictionaryAttr::get(ctx, attrs));
      }
    }
  }

  bool annotateMetadataPerShimAIRChannel(air::ChannelInterface chan_o,
                                         MemRefType memref_ty,
                                         StringAttr dma_name_attr) {
    auto ctx = chan_o->getContext();
    auto internalIndices = chan_o.getIndices();
    bool shim_chans_annotated = false;
    for (auto the_other_chan_o : getTheOtherChannelOpThroughSymbol(chan_o)) {
      // Many on shim, one on air.hierarchy.
      if (!internalIndices.empty() &&
          internalIndices.size() == the_other_chan_o.getIndices().size()) {
        // Check if the two end points of the connection match
        bool matchingSubChannel = true;
        bool allConstantIndices = false;
        // Walk the channel bundle.
        for (unsigned i = 0; i < internalIndices.size(); i++) {
          Value internalIdx = internalIndices[i];
          Value externalIdx = the_other_chan_o.getIndices()[i];
          // Find matching sub-channels by walking the (spatial) iteration space
          // around the channel bundle indices.
          auto getAllStaticStepsInLoopLike =
              [](LoopLikeOpInterface loopLikeOwner, BlockArgument iterArg) {
                SmallVector<int> steps;
                if (!llvm::is_contained(*loopLikeOwner.getLoopInductionVars(),
                                        iterArg))
                  return steps;
                int idx = iterArg.getArgNumber();
                int intLb = *getConstantIntValue(
                    (*loopLikeOwner.getLoopLowerBounds())[idx]);
                int intUb = *getConstantIntValue(
                    (*loopLikeOwner.getLoopUpperBounds())[idx]);
                int intStep =
                    *getConstantIntValue((*loopLikeOwner.getLoopSteps())[idx]);
                for (int intIter = intLb; intIter < intUb; intIter += intStep)
                  steps.push_back(intIter);
                return steps;
              };
          auto getAllStaticStepsInAIRHerd = [](air::HerdOp herdOwner,
                                               BlockArgument iterArg) {
            SmallVector<int> steps;
            if (!llvm::is_contained(herdOwner.getIds(), iterArg))
              return steps;
            int idx = iterArg.getArgNumber();
            int intUb =
                *getConstantIntValue((herdOwner.getSizeOperands())[idx]);
            for (int intIter = 0; intIter < intUb; intIter++)
              steps.push_back(intIter);
            return steps;
          };
          if (isEqualConstantIntOrValue(internalIdx, externalIdx))
            allConstantIndices = true;
          else {
            auto constInternalIdx = getConstantIntValue(internalIdx);
            auto constExternalIdx = getConstantIntValue(externalIdx);
            auto internalBlockArg =
                dyn_cast_if_present<BlockArgument>(internalIdx);
            auto externalBlockArg =
                dyn_cast_if_present<BlockArgument>(externalIdx);
            SmallVector<int> internalSteps, externalSteps;
            if (internalBlockArg) {
              if (LoopLikeOpInterface loopLikeOwner =
                      dyn_cast_if_present<LoopLikeOpInterface>(
                          internalBlockArg.getOwner()->getParentOp()))
                internalSteps = getAllStaticStepsInLoopLike(loopLikeOwner,
                                                            internalBlockArg);
              else if (air::HerdOp herdOwner = dyn_cast_if_present<air::HerdOp>(
                           internalBlockArg.getOwner()->getParentOp()))
                internalSteps =
                    getAllStaticStepsInAIRHerd(herdOwner, internalBlockArg);
            } else if (constInternalIdx)
              internalSteps.push_back(*constInternalIdx);
            if (externalBlockArg) {
              if (LoopLikeOpInterface loopLikeOwner =
                      dyn_cast_if_present<LoopLikeOpInterface>(
                          externalBlockArg.getOwner()->getParentOp()))
                externalSteps = getAllStaticStepsInLoopLike(loopLikeOwner,
                                                            externalBlockArg);
              else if (air::HerdOp herdOwner = dyn_cast_if_present<air::HerdOp>(
                           externalBlockArg.getOwner()->getParentOp()))
                externalSteps =
                    getAllStaticStepsInAIRHerd(herdOwner, externalBlockArg);
            } else if (constExternalIdx)
              externalSteps.push_back(*constExternalIdx);

            // Check if externalSteps and internalSteps include one another
            if (!std::includes(internalSteps.begin(), internalSteps.end(),
                               externalSteps.begin(), externalSteps.end()) &&
                !std::includes(externalSteps.begin(), externalSteps.end(),
                               internalSteps.begin(), internalSteps.end()))
              matchingSubChannel = false;
          }
        }
        if (matchingSubChannel) {
          shim_chans_annotated = true;
          if (!the_other_chan_o->hasAttr("metadata")) {
            the_other_chan_o->setAttr(
                "metadata", FlatSymbolRefAttr::get(ctx, dma_name_attr));
            break;
          } else {
            if (allConstantIndices)
              shim_chans_annotated = false;
          }
        }
      } else { // Channel isn't a bundle.
        the_other_chan_o->setAttr("metadata",
                                  FlatSymbolRefAttr::get(ctx, dma_name_attr));
        shim_chans_annotated = true;
      }
    }
    return shim_chans_annotated;
  }

  // AIE2 metadata format is symbolic linked to shim dma ops
  bool labelAIRDmaOpsWithMetadataObjFifo(
      std::vector<air::ChannelInterface> channel_ops,
      std::string specializedChanName,
      std::map<std::string, std::string> chan_to_chan_map) {
    bool dmaops_labeled = false;
    for (auto o : channel_ops) {
      if (o.getChanName().str() == specializedChanName) {
        auto dma_name_attr =
            StringAttr::get(o->getContext(), "air_" + specializedChanName);
        o->setAttr("metadata",
                   FlatSymbolRefAttr::get(o->getContext(), dma_name_attr));
        dmaops_labeled = true;
      } else if (o.getChanName().str() ==
                 chan_to_chan_map[specializedChanName]) {
        auto dma_name_attr =
            StringAttr::get(o->getContext(), "air_" + specializedChanName);
        o->setAttr("metadata",
                   FlatSymbolRefAttr::get(o->getContext(), dma_name_attr));
        dmaops_labeled = true;
      }
    }
    return dmaops_labeled;
  }

  // Annotate AIR DMA ops that correspond to a SHIM DMA allocation with packet
  // information, specifically for MM2S (host-to-AIE) directions. The tile
  // operand is passed as a Value so it works for both physical aie.tile and
  // unplaced aie.logical_tile.
  LogicalResult labelMemcpyOpsWithPacketFlow(air::MemcpyInterface memcpyOpIf,
                                             StringAttr dmaNameAttr,
                                             mlir::Value tileVal, int channel,
                                             int packetFlowId = -1) {
    // Multi-id pinned channels follow the kernel-header contract: the core
    // writes the routing id into the payload, so the shim DMA must not stamp a
    // packet header. Skip before the runtime fallback below could tag it with
    // an arbitrary flow id.
    if (auto ci = dyn_cast_if_present<air::ChannelInterface>(
            memcpyOpIf.getOperation()))
      if (auto chanOp = air::getChannelDeclarationThroughSymbol(ci))
        if (auto pids = chanOp.getPacketIDs(); pids && pids.size() > 1)
          return success();

    // When a packet flow ID is available (from flow creation phase), use
    // exact flow ID matching to disambiguate multiple flows sharing the
    // same shim DMA channel. Otherwise fall back to source-only lookup.
    AIE::PacketFlowOp pktFlowOp;
    if (packetFlowId >= 0)
      pktFlowOp = findPacketFlowOp(tileVal, AIE::WireBundle::DMA, channel,
                                   /*checkFlowID=*/true, packetFlowId);
    if (!pktFlowOp)
      pktFlowOp = getExistingPacketFlowOpFromRuntime(
          tileVal, AIE::WireBundle::DMA, channel);
    if (!pktFlowOp)
      return success();

    // If memcpy op is air.channel: filter out channel bundles based on
    // metadata; get metadata from metadataArray based on channel indices.
    if (auto ci = dyn_cast_if_present<air::ChannelInterface>(
            memcpyOpIf.getOperation())) {
      // Get index to metadataArray based on channel indices.
      auto iter = air::getIndexToMetadataArrayFromChannelIndices(ci);
      if (!iter) {
        ci->emitOpError(
            "channel indices failed to convert to metadataArray index.");
        return failure();
      }
      // Get metadata from metadataArray.
      auto metadataArray = ci->getAttrOfType<ArrayAttr>("metadataArray");
      if (!metadataArray || (size_t)*iter >= metadataArray.size())
        return success();
      auto dictAttr = dyn_cast_if_present<DictionaryAttr>(metadataArray[*iter]);
      if (!dictAttr)
        return success();
      auto shimNameAttr = dictAttr.getAs<StringAttr>("base");
      if (!shimNameAttr)
        return success();
      // Check if metadata's shim allocation name matches the target name.
      if (shimNameAttr.getValue() != dmaNameAttr.getValue())
        return success();
    }

    auto pktInfoAttr = AIE::PacketInfoAttr::get(
        memcpyOpIf->getContext(), /*pkt_type=*/0, pktFlowOp.getID());
    memcpyOpIf->setAttr("packet", pktInfoAttr);
    return success();
  }

  // Determine the tile-side memref type of the DMA op.
  MemRefType getTileSideMemrefTypeForMemcpy(air::MemcpyInterface dmaOp,
                                            AIE::DMAChannelDir dir) {
    if (auto dma =
            dyn_cast_if_present<air::DmaMemcpyNdOp>(dmaOp.getOperation()))
      return cast<MemRefType>((dir == AIE::DMAChannelDir::MM2S)
                                  ? dma.getDstMemref().getType()
                                  : dma.getSrcMemref().getType());
    if (auto chan =
            dyn_cast_if_present<air::ChannelInterface>(dmaOp.getOperation())) {
      air::ChannelInterface tileSideChannelOp =
          air::getTheOtherChannelOpThroughSymbol(chan).front();
      return cast<MemRefType>(tileSideChannelOp.getMemref().getType());
    }
    return nullptr;
  }

  // Create shim DMA allocation ops and annotate the corresponding memcpy
  // operations with symbolic metadata.
  // When skipUnlinked is true, shim-side memcpy ops that don't match any
  // allocation are silently skipped instead of producing an error. This is
  // used for segment-unrolled designs where each device processes its own
  // allocations independently, so ops belonging to other devices are expected
  // to be unlinked.
  LogicalResult createShimDMAAllocationOps(
      OpBuilder builder, MLIRContext *ctx,
      std::vector<air::MemcpyInterface> shimSideMemcpyIfOps,
      air::ShimDMAAllocator &shimDmaAllocs,
      std::map<int, int> chanRenumberReverseMap, bool skipUnlinked = false) {
    std::vector<air::MemcpyInterface> shimMemcpyS2MMOps, shimMemcpyMM2SOps;

    // Separate memcpy ops into S2MM and MM2S based on direction.
    for (auto memcpyIf : shimSideMemcpyIfOps) {
      if (auto put =
              dyn_cast_if_present<air::ChannelPutOp>(memcpyIf.getOperation()))
        shimMemcpyMM2SOps.push_back(memcpyIf);
      if (auto get =
              dyn_cast_if_present<air::ChannelGetOp>(memcpyIf.getOperation()))
        shimMemcpyS2MMOps.push_back(memcpyIf);
      if (auto dmaOp = dyn_cast_if_present<air::DmaMemcpyNdOp>(
              memcpyIf.getOperation())) {
        auto srcMemrefTy =
            dyn_cast_if_present<BaseMemRefType>(dmaOp.getSrcMemref().getType());
        auto dstMemrefTy =
            dyn_cast_if_present<BaseMemRefType>(dmaOp.getDstMemref().getType());
        if (air::isL3(srcMemrefTy))
          shimMemcpyMM2SOps.push_back(memcpyIf);
        if (air::isL3(dstMemrefTy))
          shimMemcpyS2MMOps.push_back(memcpyIf);
      }
    }

    // Create shim-side S2MM DMA allocs and annotate corresponding ops.
    if (failed(createShimDMAAllocationOpsImpl(
            builder, ctx, shimMemcpyS2MMOps, shimDmaAllocs.s2mm_allocs,
            AIE::DMAChannelDir::S2MM, chanRenumberReverseMap, skipUnlinked))) {
      return failure();
    }

    // Create shim-side MM2S DMA allocs and annotate corresponding ops.
    if (failed(createShimDMAAllocationOpsImpl(
            builder, ctx, shimMemcpyMM2SOps, shimDmaAllocs.mm2s_allocs,
            AIE::DMAChannelDir::MM2S, chanRenumberReverseMap, skipUnlinked))) {
      return failure();
    }
    return success();
  }

  // Get the original DMA IDs used on the tile side before any renumbering.
  // This function reverses any renumbering applied via chanRenumberReverseMap.
  SmallVector<int32_t>
  getOriginalTileSideDmaIds(air::allocation_info_t &t,
                            std::map<int, int> chanRenumberReverseMap) {
    SmallVector<int32_t> outputs;
    for (int32_t id : t.dma_id) {
      int original_id =
          chanRenumberReverseMap.size() ? chanRenumberReverseMap[id] : id;
      outputs.push_back(original_id);
    }
    return outputs;
  }

  // New metadata struct format
  static DictionaryAttr buildMetadataHint(OpBuilder &builder, StringRef base,
                                          int index) {
    return DictionaryAttr::get(
        builder.getContext(),
        {builder.getNamedAttr("base", builder.getStringAttr(base)),
         builder.getNamedAttr("index", builder.getI32IntegerAttr(index))});
  }

  LogicalResult createShimDMAAllocationOpsImpl(
      OpBuilder builder, MLIRContext *ctx,
      std::vector<air::MemcpyInterface> shimSideMemcpyIfOps,
      std::vector<air::allocation_info_t> allocs, AIE::DMAChannelDir dir,
      std::map<int, int> chanRenumberReverseMap, bool skipUnlinked = false) {

    // Helper function getting dma_name from the air::MemcpyInterface op.
    auto getDmaNameFromMemcpyIfOp = [](air::MemcpyInterface memcpyIfOp) {
      std::string dma_name = "";
      if (auto ci = dyn_cast_if_present<air::ChannelInterface>(
              memcpyIfOp.getOperation()))
        return "air_" + ci.getChanName().str();
      else if (auto dmaOp = dyn_cast_if_present<air::DmaMemcpyNdOp>(
                   memcpyIfOp.getOperation()))
        return "airMemcpyId" + std::to_string(dmaOp.getId());
      return dma_name;
    };

    // Get mapping between each shim channel op and shim allocations (one
    // channel op could map to many, as shim allocations are unrolled, but
    // channel op isn't).
    std::map<std::string, std::vector<air::allocation_info_t>>
        shimChanSymbolToAlloc;
    for (air::allocation_info_t &t : allocs) {
      // Currently, air::allocation_info_t only links the shim allocations to
      // the "internal" (tile-side) data movement op ids.
      auto memcpyIfOpIt = llvm::find_if(
          shimSideMemcpyIfOps, [&](air::MemcpyInterface memcpyIfOp) {
            if (auto ci = dyn_cast_if_present<air::ChannelInterface>(
                    memcpyIfOp.getOperation())) {
              for (auto tileSideChannelOp :
                   air::getTheOtherChannelOpThroughSymbol(ci)) {
                auto linkedTileSideDmaIds =
                    getOriginalTileSideDmaIds(t, chanRenumberReverseMap);
                if (llvm::is_contained(linkedTileSideDmaIds,
                                       tileSideChannelOp.getId()))
                  return true;
              }
            } else if (auto dmaOp = dyn_cast_if_present<air::DmaMemcpyNdOp>(
                           memcpyIfOp.getOperation())) {
              auto linkedTileSideDmaIds =
                  getOriginalTileSideDmaIds(t, chanRenumberReverseMap);
              return llvm::is_contained(linkedTileSideDmaIds, dmaOp.getId());
            }
            return false;
          });
      if (memcpyIfOpIt != shimSideMemcpyIfOps.end()) {
        std::string dma_name = getDmaNameFromMemcpyIfOp(*memcpyIfOpIt);
        shimChanSymbolToAlloc[dma_name].push_back(t);
      }
    }

    // Sort each bucket lexicographically by its far-end tile's (col, row).
    // The bucket order determines per-device shim allocation naming
    // (outD_0, outD_1, ...) at line ~4711, which is what
    // getIndexToMetadataArrayFromChannelIndices indexes into via the
    // row-major formula (`idx[0]*dims[1] + idx[1]`). Without this sort the
    // bucket arrives in whatever order shimSideMemcpyIfOps was traversed,
    // which is col-fast/row-slow in physical tile order — opposite of the
    // formula's row-fast convention. The two coincide only when any dim
    // == 1, so 1D-style herds work but 2D channels with both dims > 1
    // misroute off-diagonal channel instances.
    // Sorting by (col, row) lex gives col-slow/row-fast iteration, which
    // matches `idx[0]=cx` (slow) and `idx[1]=cy` (fast) when channel index
    // [i,j] corresponds to herd tile (i,j) at (x_loc+i, y_loc+j).
    for (auto &kv : shimChanSymbolToAlloc) {
      auto &allocVec = kv.second;
      if (allocVec.size() <= 1)
        continue;
      auto getFarEndCoords = [](air::allocation_info_t &t) {
        std::pair<int64_t, int64_t> coords{-1, -1};
        if (!t.otherSideLTO)
          return coords;
        if (auto tileLike = dyn_cast<AIE::TileLike>(t.otherSideLTO)) {
          coords.first = tileLike.tryGetCol().value_or(-1);
          coords.second = tileLike.tryGetRow().value_or(-1);
        }
        return coords;
      };
      std::stable_sort(allocVec.begin(), allocVec.end(),
                       [&](air::allocation_info_t a, air::allocation_info_t b) {
                         auto ca = getFarEndCoords(a);
                         auto cb = getFarEndCoords(b);
                         return ca < cb;
                       });
    }

    // Capture errors when any shim memcpy op fails to link to shim allocation.
    // When skipUnlinked is true (segment-unrolled per-device processing),
    // unlinked ops belong to other devices and are silently skipped.
    if (!skipUnlinked) {
      auto unlinkedMemcpyIfOp = llvm::find_if(
          shimSideMemcpyIfOps, [&](air::MemcpyInterface memcpyIfOp) {
            std::string dma_name = getDmaNameFromMemcpyIfOp(memcpyIfOp);
            return !shimChanSymbolToAlloc.count(dma_name);
          });
      if (unlinkedMemcpyIfOp != shimSideMemcpyIfOps.end()) {
        unlinkedMemcpyIfOp->emitOpError(
            "failed to link to any shim dma allocation.");
        return failure();
      }
    }

    // Create shim dma allocation ops.
    for (auto memcpyIfOp : shimSideMemcpyIfOps) {
      std::string dma_name = getDmaNameFromMemcpyIfOp(memcpyIfOp);
      // Start index from the existing metadataArray size so that per-device
      // calls produce globally sequential indices across devices.
      ArrayAttr existingMeta =
          memcpyIfOp->getAttrOfType<ArrayAttr>("metadataArray");
      int t_idx = existingMeta ? existingMeta.size() : 0;
      // Track per-device allocation index so the shim name encodes a
      // within-device tile index (not the globally-sequential t_idx).
      // The metadataArray sorting code parses this trailing index as
      // tileIdx and feeds it to getIteratorFromMDVector; using the global
      // t_idx causes out-of-bounds linearized indices for device 1+.
      Operation *prevDevice = nullptr;
      int perDeviceIdx = 0;
      for (air::allocation_info_t &t : shimChanSymbolToAlloc[dma_name]) {
        auto deviceOp = t.getDmaTile()->getParentOfType<AIE::DeviceOp>();
        if (deviceOp.getOperation() != prevDevice) {
          perDeviceIdx = 0;
          prevDevice = deviceOp.getOperation();
        }
        // Create shim allocation symbol name shim_name_attr.
        // When segment unroll is active, append unroll indices to ensure
        // unique symbol names across devices.
        std::string shim_name = dma_name;
        if (auto unrollXAttr =
                deviceOp->getAttrOfType<IntegerAttr>("segment_unroll_x")) {
          shim_name += "_" + std::to_string(unrollXAttr.getInt());
        }
        if (auto unrollYAttr =
                deviceOp->getAttrOfType<IntegerAttr>("segment_unroll_y")) {
          shim_name += "_" + std::to_string(unrollYAttr.getInt());
        }
        if (shimChanSymbolToAlloc[dma_name].size() > 1 || t_idx > 0)
          shim_name += "_" + std::to_string(perDeviceIdx);
        perDeviceIdx++;
        StringAttr shim_name_attr = builder.getStringAttr(shim_name);

        // Create shim allocation op in the allocation's own DeviceOp.
        builder.setInsertionPoint(deviceOp.getBody()->getTerminator());
        if (!SymbolTable::lookupSymbolIn(deviceOp, shim_name)) {
          auto shimAllocationOp = AIE::ShimDMAAllocationOp::create(
              builder, builder.getUnknownLoc(), shim_name_attr,
              t.getDmaTile()->getResult(0),
              AIE::DMAChannelDirAttr::get(ctx, dir),
              builder.getI64IntegerAttr(t.dma_channel.channel),
              /*plio*/ builder.getBoolAttr(false),
              /*packet*/ nullptr);

          // Get the tile-side channel op's MemRefType, needed when creating the
          // memref.global.
          MemRefType rankedTileSideMemRefTy =
              getTileSideMemrefTypeForMemcpy(memcpyIfOp, dir);

          if (!rankedTileSideMemRefTy)
            return shimAllocationOp->emitOpError(
                "failed to get MemRefType for memref.global op");
        }

        // Add metadata to each shim-side channel op, linking to shim
        // allocation. Get existing array if present
        ArrayAttr metadataArray =
            memcpyIfOp->getAttrOfType<ArrayAttr>("metadataArray");
        SmallVector<Attribute, 4> updatedMetadata;

        if (metadataArray)
          updatedMetadata.append(metadataArray.begin(), metadataArray.end());

        // Metadata hint format: {base: shim_name_attr, index: t_idx}
        DictionaryAttr hint =
            buildMetadataHint(builder, shim_name_attr, t_idx++);
        updatedMetadata.push_back(hint);
        memcpyIfOp->setAttr("metadataArray",
                            builder.getArrayAttr(updatedMetadata));

        // Annotate shim DMA packed-flow ops with packet information,
        // specifically for MM2S (host-to-AIE) directions.
        if (dir == AIE::DMAChannelDir::MM2S)
          if (failed(labelMemcpyOpsWithPacketFlow(
                  memcpyIfOp, shim_name_attr, t.getDmaTile()->getResult(0),
                  t.dma_channel.channel, t.packet_flow_id)))
            return failure();
      }

      // When segment unroll is active, the metadataArray may be in
      // device-iteration order (device 0 allocations first, then device 1),
      // but getIteratorFromMDVector expects a different linearization. Sort
      // the metadataArray to match by computing each entry's linearized
      // position from its channel coordinates.
      if (auto ci = dyn_cast_if_present<air::ChannelInterface>(
              memcpyIfOp.getOperation())) {
        ArrayAttr metadataArray =
            memcpyIfOp->getAttrOfType<ArrayAttr>("metadataArray");
        if (metadataArray && metadataArray.size() > 1) {
          auto chanDecl = air::getChannelDeclarationThroughSymbol(ci);
          if (chanDecl && chanDecl.getSize() &&
              chanDecl.getSize().size() >= 2) {
            // Check if any entry comes from a segment-unrolled device.
            bool hasUnroll = false;
            for (auto attr : metadataArray) {
              if (auto dict = dyn_cast_if_present<DictionaryAttr>(attr)) {
                auto base = dict.getAs<StringAttr>("base");
                if (base) {
                  // Segment unroll appends _X_ to the name. Check if the
                  // name has more than the usual number of underscores.
                  StringRef name = base.getValue();
                  // Count underscore-separated segments: "air_channel_0_X_Y_Z"
                  // Without unroll: "air_channel_0_Z" (4 segments)
                  // With unroll: "air_channel_0_X_Y_Z" (6+ segments)
                  size_t count = 0;
                  for (char c : name)
                    if (c == '_')
                      count++;
                  if (count >= 5) {
                    hasUnroll = true;
                    break;
                  }
                }
              }
            }

            if (hasUnroll) {
              // Build the linearized-index-to-entry map. Parse each entry's
              // base name to extract unrollCopy and tileIdx, then compute
              // the linearized position.
              std::vector<unsigned> channelDims;
              for (auto a : chanDecl.getSize()) {
                if (auto intAttr = dyn_cast_if_present<IntegerAttr>(a))
                  channelDims.push_back(
                      static_cast<unsigned>(intAttr.getInt()));
              }

              unsigned totalExpected = 1;
              for (unsigned d : channelDims)
                totalExpected *= d;

              if (channelDims.size() == 2 &&
                  totalExpected == metadataArray.size()) {
                // Determine which channel dimension corresponds to the
                // segment unroll index. Count distinct unrollCopy values
                // to get numUnrollCopies.
                llvm::SmallSet<int, 4> unrollValues;
                for (auto attr : metadataArray) {
                  auto dict = dyn_cast_if_present<DictionaryAttr>(attr);
                  if (!dict)
                    continue;
                  auto base = dict.getAs<StringAttr>("base");
                  if (!base)
                    continue;
                  SmallVector<StringRef, 8> parts;
                  base.getValue().split(parts, '_');
                  if (parts.size() >= 6) {
                    int unrollCopy = 0;
                    parts[3].getAsInteger(10, unrollCopy);
                    unrollValues.insert(unrollCopy);
                  }
                }
                unsigned numUnrollCopies = unrollValues.size();

                // Determine the unroll dimension index. The unroll
                // dimension is the one whose size equals
                // numUnrollCopies, while the other dimension is the
                // per-device tile count. When ambiguous (both dims
                // equal numUnrollCopies), skip sorting — device-
                // iteration order is already correct for this case.
                int unrollDimIdx = -1;
                int numMatches = 0;
                for (unsigned i = 0; i < channelDims.size(); i++) {
                  if (channelDims[i] == numUnrollCopies) {
                    unrollDimIdx = i;
                    numMatches++;
                  }
                }
                // Only sort when exactly one dimension matches the
                // unroll count, giving an unambiguous mapping.
                if (numMatches == 1) {
                  SmallVector<Attribute, 8> sorted(totalExpected);

                  for (auto attr : metadataArray) {
                    auto dict = dyn_cast_if_present<DictionaryAttr>(attr);
                    if (!dict)
                      continue;
                    auto base = dict.getAs<StringAttr>("base");
                    if (!base)
                      continue;

                    StringRef name = base.getValue();
                    SmallVector<StringRef, 8> parts;
                    name.split(parts, '_');
                    if (parts.size() >= 6) {
                      int unrollCopy = 0, tileIdx = 0;
                      parts[3].getAsInteger(10, unrollCopy);
                      parts.back().getAsInteger(10, tileIdx);

                      // Build position vector matching channelDims order:
                      // the unroll dimension gets unrollCopy, the other
                      // dimension gets tileIdx.
                      std::vector<unsigned> position(2);
                      position[unrollDimIdx] =
                          static_cast<unsigned>(unrollCopy);
                      position[1 - unrollDimIdx] =
                          static_cast<unsigned>(tileIdx);
                      unsigned linIdx =
                          air::getIteratorFromMDVector(channelDims, position);
                      if (linIdx < totalExpected)
                        sorted[linIdx] = attr;
                    }
                  }

                  // Verify all positions were filled.
                  bool allFilled = true;
                  for (auto &a : sorted) {
                    if (!a) {
                      allFilled = false;
                      break;
                    }
                  }
                  if (allFilled) {
                    memcpyIfOp->setAttr("metadataArray",
                                        builder.getArrayAttr(sorted));
                  }
                }
              }
            }
          }
        }
      }
    }
    return success();
  }

  airrt::SegmentMetadataOp
  getOrCreateSegmentMetadata(airrt::ModuleMetadataOp module_meta,
                             StringRef name) {

    for (auto pm :
         module_meta.getSegments().front().getOps<airrt::SegmentMetadataOp>())
      if (name == pm.getSymName().str())
        return pm;

    auto builder = OpBuilder::atBlockTerminator(module_meta.getBody());
    auto loc = builder.getUnknownLoc();
    auto segment_meta = airrt::SegmentMetadataOp::create(builder, loc, name);
    builder.createBlock(&segment_meta.getHerds());
    airrt::SegmentMetadataTerminatorOp::create(builder, loc);

    return segment_meta;
  }

  airrt::HerdMetadataOp
  createHerdMetadata(airrt::SegmentMetadataOp segment_meta, air::HerdOp herd) {
    auto builder = OpBuilder::atBlockTerminator(segment_meta.getBody());
    auto loc = builder.getUnknownLoc();

    std::string name = "herd";
    if (auto attr =
            herd->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      name = attr.getValue().str();

    auto herd_meta = airrt::HerdMetadataOp::create(builder, loc, name);
    herd_meta->setAttr("size_x", builder.getI64IntegerAttr(herd.getNumCols()));
    herd_meta->setAttr("size_y", builder.getI64IntegerAttr(herd.getNumRows()));
    if (auto co = herd.getColOffset())
      herd_meta->setAttr("loc_x", builder.getI64IntegerAttr(*co));
    if (auto ro = herd.getRowOffset())
      herd_meta->setAttr("loc_y", builder.getI64IntegerAttr(*ro));
    return herd_meta;
  }

  // Checks if the given operation writes to, or deallocates, the specified
  // buffer and all its views.
  bool isDmaWriteToBuffer(Operation *op, SetVector<Value> &bufferViews) {
    if (!isa_and_present<air::MemcpyInterface>(op))
      return false;
    if (llvm::any_of(*air::getAllWriteAccessedMemrefOperandsFromOp(op),
                     [&bufferViews](auto &entry) {
                       return llvm::is_contained(bufferViews, entry.first);
                     }))
      return true;
    return false;
  }

  // Recursively searches for operations that write to, or deallocate, the
  // target buffer inside a nested region. Returns `true` if a writer is found,
  // setting `ancestorOp` to the outermost op under the same region as
  // `memcpyOp`.
  bool findNextDmaWriteOpInRegion(Region &region, SetVector<Value> &bufferViews,
                                  Operation *&ancestorOp) {
    bool writeOpFound = false;
    region.walk([&](Operation *op) {
      // Save buffer views.
      if (auto view = dyn_cast_if_present<ViewLikeOpInterface>(op)) {
        if (llvm::is_contained(bufferViews, view.getViewSource()))
          bufferViews.insert(view->getResult(0));
      }
      if (isDmaWriteToBuffer(op, bufferViews)) {
        ancestorOp = op; // Found a writer, set it as the closest ancestor
        writeOpFound = true;
        return;
      }
    });
    return writeOpFound;
  }

  /// Walks ops in the block and finds the end of lifetime for this memcpy op.
  /// Returns the first operation in the same block that writes to, or
  /// deallocates, `destBuffer`.
  Operation *findNextDmaWriteOp(Operation *memcpyOp, Value destBuffer) {
    // Ensure the given operation is an air memcpy operation
    auto memcpyOpIf = dyn_cast_if_present<air::MemcpyInterface>(memcpyOp);
    if (!memcpyOpIf)
      return nullptr;

    // Ensure that destBuffer is used in mempcy op.
    if (destBuffer != memcpyOpIf.getSrcMemref() &&
        destBuffer != memcpyOpIf.getDstMemref())
      return nullptr;

    // Iterate over operations after the memcpyOp
    SetVector<Value> bufferViews;
    bufferViews.insert(destBuffer);
    for (Operation *op = memcpyOpIf->getNextNode(); op != nullptr;
         op = op->getNextNode()) {
      // Save buffer views.
      if (auto view = dyn_cast_if_present<ViewLikeOpInterface>(op)) {
        if (llvm::is_contained(bufferViews, view.getViewSource()))
          bufferViews.insert(view->getResult(0));
      }
      if (isDmaWriteToBuffer(op, bufferViews)) {
        return op; // Found the next writer
      }
      // Check within any regions of this operation
      Operation *ancestorOp = nullptr;
      for (Region &region : op->getRegions()) {
        if (findNextDmaWriteOpInRegion(region, bufferViews, ancestorOp)) {
          return op; // Return the ancestor operation at the same level as
                     // memcpyOp
        }
      }
    }
    return nullptr; // memcpyOp end-of-lifetime not found
  }

  /// Checks if an operation reads from or writes to the given buffer.
  bool isReadOrWriteToBuffer(Operation *op, SetVector<Value> bufferViews) {
    if (!op)
      return false;
    if (air::isPure(op))
      return false;

    // Check if the op reads from or writes to the buffer
    for (OpOperand &operand : op->getOpOperands())
      if (llvm::is_contained(bufferViews, operand.get()))
        return true;
    return false;
  }

  /// Walks ops in the block and finds the last reader/writer. Returns the last
  /// operation in the same block that accesses `destBuffer`.
  Operation *findLastReadOrWriteOp(Operation *memcpyOp, Value destBuffer) {
    // Ensure the given operation is an air memcpy operation
    auto memcpyOpIf = dyn_cast_if_present<air::MemcpyInterface>(memcpyOp);
    if (!memcpyOpIf)
      return nullptr;

    // Ensure that destBuffer is used in mempcy op.
    if (destBuffer != memcpyOpIf.getSrcMemref() &&
        destBuffer != memcpyOpIf.getDstMemref())
      return nullptr;

    // Iterate over operations after the memcpyOp
    Operation *lastAccessOp = nullptr;
    SetVector<Value> bufferViews;
    bufferViews.insert(destBuffer);
    for (Operation *op = memcpyOpIf->getNextNode(); op != nullptr;
         op = op->getNextNode()) {
      // Save buffer views.
      if (auto view = dyn_cast_if_present<ViewLikeOpInterface>(op)) {
        if (llvm::is_contained(bufferViews, view.getViewSource()))
          bufferViews.insert(view->getResult(0));
      }

      if (isReadOrWriteToBuffer(op, bufferViews))
        lastAccessOp = op;

      // Check within any regions of this operation
      for (Region &region : op->getRegions()) {
        region.walk([&](Operation *o) {
          // Save buffer views.
          if (auto view = dyn_cast_if_present<ViewLikeOpInterface>(o)) {
            if (llvm::is_contained(bufferViews, view.getViewSource()))
              bufferViews.insert(view->getResult(0));
          }
          if (isReadOrWriteToBuffer(o, bufferViews))
            lastAccessOp = op;
        });
      }
    }
    return lastAccessOp;
  }

  LogicalResult allocateCoreLocksPerMemcpyOp(
      OpBuilder builder, air::MemcpyInterface memcpyOpIf,
      llvm::SetVector<Operation *> &allocs_to_remap,
      const AIE::AIETargetModel &targetModel,
      air::TileDMAAllocator &tileDmaAlloc, AIE::TileOp tile) {
    bool UsesSemaphoreLocks =
        targetModel.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks);
    auto dma_alloc = tileDmaAlloc.lookupDMAAllocation(tile, memcpyOpIf);
    if (failed(dma_alloc)) {
      return memcpyOpIf->emitOpError("failed to look up dma allocation.");
    }
    auto tile_channel = dma_alloc.value().dma_channel;
    auto bufferOp = tileDmaAlloc.getBuffer(BufferId, tile, memcpyOpIf);
    if (failed(bufferOp)) {
      return memcpyOpIf->emitOpError("failed to get buffer.");
    }
    // 3-way shared L1: the core-side acquire/release for a channel put/get on
    // this buffer was already emitted by allocateSharedL1BufferLocks (the
    // shared prod/cons wrapping around the core's kernel write). Emitting
    // another core-side lock here would double-acquire the prod lock and
    // deadlock. The op carries the shared prod/cons symbol-refs; skip the
    // core-side lock. The DMA-side BD still picks up the shared pair via
    // getLockForDMA in generateDmaBdProgram.
    if (memcpyOpIf->hasAttr("air.shared_prod_lock"))
      return success();
    auto locks = tileDmaAlloc.getLockForDMA(memcpyOpIf, tile,
                                            bufferOp.value().getOperation(),
                                            /*lockRaceConditionFix*/ false);
    if (failed(locks))
      return memcpyOpIf->emitOpError("failed to get lock for dma.");
    auto acqLockOp =
        isMM2S(tile_channel) ? locks.value().second : locks.value().first;
    auto relLockOp =
        isMM2S(tile_channel) ? locks.value().first : locks.value().second;
    int64_t lockAqValue = -1;
    int64_t lockRelValue = -1;
    Value alloc = nullptr;
    auto tileInbound = isTileInbound(memcpyOpIf, air::MemorySpace::L1);
    if (failed(tileInbound))
      return failure();
    if (tileInbound.value()) {
      lockAqValue = UsesSemaphoreLocks ? 1 : 1;
      lockRelValue = UsesSemaphoreLocks ? 1 : 0;
      alloc = memcpyOpIf.getDstMemref();
    } else {
      lockAqValue = UsesSemaphoreLocks ? 1 : 0;
      lockRelValue = UsesSemaphoreLocks ? 1 : 1;
      alloc = memcpyOpIf.getSrcMemref();
    }

    // Producer-side re-feed. An outbound channel put that re-sends ONE resident
    // buffer N times (e.g. an activation output streamed once per GEMV
    // row-block iteration) is expressed by air.refeed_count=N. The producing
    // core writes the buffer once and the DMA's count-free self-loop BD
    // re-reads it; the single core-side release must free N read-tokens so the
    // BD fires N times. (The N>1 loop the front-end emits is canonicalized away
    // before air-to-aie, so the count is carried on the channel/put, not
    // inferred from a loop.)
    if (UsesSemaphoreLocks && !tileInbound.value()) {
      int64_t n = 1;
      if (auto chanIf =
              dyn_cast<air::ChannelInterface>(memcpyOpIf.getOperation()))
        n = air::getRefeedCount(chanIf);
      if (n > 1) {
        lockRelValue *= n;
        // Re-dispatch correctness: the DMA self-loop releases the buf-free
        // (acquire) lock once per re-send, i.e. N times per dispatch, while the
        // producing core overwrites the resident buffer only once. The core
        // must re-acquire ALL N freed tokens before the next overwrite, and the
        // buf-free lock must INIT to N (else the first dispatch's acquire>=N
        // can never fire). Without this the buf-free lock leaks N-1 every
        // dispatch, accumulating into a re-dispatch parity stall.
        lockAqValue *= n;
        // getRefeedCount guarantees 1 < n <= INT32_MAX, so the narrowing is
        // safe.
        if (auto initOpt = acqLockOp.getInit(); !initOpt || *initOpt < n)
          acqLockOp.setInit(static_cast<int32_t>(n));
      }
    }

    // Detect if multiple outbound puts in this DMA allocation share the same
    // source buffer. When true, use per-put interleaved lock placement to
    // prevent the second put from overwriting the buffer before the DMA
    // finishes reading the first put's data.
    bool sharedStagingBuffer = false;
    if (!tileInbound.value() && isa<AIE::BufferOp>(alloc.getDefiningOp()) &&
        dma_alloc.value().memcpyOps.size() > 1) {
      int sameBufCount = 0;
      for (auto *op : dma_alloc.value().memcpyOps) {
        if (auto other = dyn_cast_if_present<air::MemcpyInterface>(op)) {
          auto otherInbound = isTileInbound(other, air::MemorySpace::L1);
          if (succeeded(otherInbound) && !otherInbound.value() &&
              other.getSrcMemref() == alloc)
            sameBufCount++;
        }
      }
      sharedStagingBuffer = sameBufCount > 1;
    }

    if (auto bco = dyn_cast_if_present<bufferization::ToBufferOp>(
            alloc.getDefiningOp()))
      builder.setInsertionPoint(bco.getOperand().getDefiningOp());
    else if (isa<memref::AllocaOp>(alloc.getDefiningOp()))
      builder.setInsertionPoint(alloc.getDefiningOp());
    else if (!tileInbound.value() &&
             isa<AIE::BufferOp>(alloc.getDefiningOp())) {
      if (sharedStagingBuffer) {
        // Interleaved mode: acquire immediately before this specific put, so
        // the core waits for the DMA to finish reading the previous put's
        // data before overwriting the buffer.
        builder.setInsertionPoint(memcpyOpIf);
      } else {
        auto br = dyn_cast_if_present<cf::BranchOp>(
            memcpyOpIf->getBlock()->getTerminator());
        if (br)
          builder.setInsertionPointToStart(br.getDest());
        else
          builder.setInsertionPointToStart(memcpyOpIf->getBlock());
      }
    } else
      builder.setInsertionPoint(memcpyOpIf);

    AIE::UseLockOp::create(builder, memcpyOpIf->getLoc(), acqLockOp,
                           UsesSemaphoreLocks
                               ? AIE::LockAction::AcquireGreaterEqual
                               : AIE::LockAction::Acquire,
                           lockAqValue);

    // Try to find the end of lifetime for the data copied by memcpyOpIf, and
    // put the unlock.
    if (sharedStagingBuffer) {
      // Interleaved mode: release rlock immediately after the put so the DMA
      // can read the buffer before the next put overwrites it. The next put's
      // acquire(wlock) will block until the DMA completes reading.
      builder.setInsertionPointAfter(memcpyOpIf);
      AIE::UseLockOp::create(builder, memcpyOpIf->getLoc(), relLockOp,
                             AIE::LockAction::Release, lockRelValue);
    } else if (auto nextWriter = findNextDmaWriteOp(memcpyOpIf, alloc)) {
      // Lifetime ends if dma writes into the same buffer.
      builder.setInsertionPoint(nextWriter);
      AIE::UseLockOp::create(builder, nextWriter->getLoc(), relLockOp,
                             AIE::LockAction::Release, lockRelValue);
    } else if (auto lastAccessOp = findLastReadOrWriteOp(memcpyOpIf, alloc)) {
      // Lifetime ends after the last read/write access to buffer.
      builder.setInsertionPointAfter(lastAccessOp);
      AIE::UseLockOp::create(builder, lastAccessOp->getLoc(), relLockOp,
                             AIE::LockAction::Release, lockRelValue);
    } else {
      // Lifetime ends at end of block.
      auto t = memcpyOpIf->getBlock()->getTerminator();
      builder.setInsertionPoint(t);
      AIE::UseLockOp::create(builder, t->getLoc(), relLockOp,
                             AIE::LockAction::Release, lockRelValue);
    }
    allocs_to_remap.insert(alloc.getDefiningOp());
    return success();
  }

  template <typename dmaAllocatorTy, typename bufferOpTy, typename memOpTy>
  LogicalResult
  generateDmaBdProgram(OpBuilder builder,
                       const AIE::AIETargetModel &targetModel,
                       llvm::MapVector<std::pair<AIE::DMAChannelDir, int>,
                                       std::vector<Operation *>>
                           dma_memcpys,
                       dmaAllocatorTy dmaAlloc, mlir::Location loc, memOpTy mem,
                       AIE::TileLike tile, bool lockRaceConditionFix = false,
                       bool lockRaceConditionFixV2 = false) {

    llvm::MapVector<std::pair<AIE::DMAChannelDir, int>,
                    std::vector<Operation *>>
        dma_memcpys_sorted;
    if (lockRaceConditionFix) {
      // Sort MapVector dma_memcpys by moving all entries containing dummy BDs
      // to the end.
      auto isVectorOfZeros = [](SmallVector<Value> vector) {
        if (vector.empty())
          return false; // Return false if the vector is empty.
        return llvm::all_of(vector, [](Value v) {
          auto constV = getConstantIntValue(v);
          if (!constV)
            return false;
          return *constV == 0;
        });
      };
      // Find dummy BDs.
      auto containsDummyBDs = [isVectorOfZeros](std::vector<Operation *> ops) {
        return llvm::any_of(ops, [&](Operation *op) {
          auto chanIf = dyn_cast_if_present<air::ChannelInterface>(op);
          return chanIf && isVectorOfZeros(chanIf.getSizes());
        });
      };
      // Sort MapVector.
      for (auto &entry : dma_memcpys) {
        if (!containsDummyBDs(entry.second))
          dma_memcpys_sorted.insert(std::move(entry));
      }
      for (auto &entry : dma_memcpys) {
        if (containsDummyBDs(entry.second))
          dma_memcpys_sorted.insert(std::move(entry));
      }
    } else
      dma_memcpys_sorted = dma_memcpys;

    // The first block
    Block *channel_head = nullptr;
    Block *end_bb = nullptr;

    for (auto &[dma_chan, memcpy_ops] : dma_memcpys_sorted) {
      AIE::DMAChannelDir dir = dma_chan.first;
      int chan = dma_chan.second;

      // Map key: repeat counts. Map value: vector of memcpy operations sharing
      // the same repeat count.
      llvm::MapVector<int, llvm::SetVector<Operation *>> repeat_counts =
          air::getRepeatCounts(memcpy_ops);

      // Note: we designate each unique repeat value in repeat_counts map with a
      // new BD task. If there is only one repeat value for all memcpy ops
      // associated to the channel, then there is no need to do repeat count; we
      // generate BDs in infinite loop mode instead.
      bool infiniteBDLoopMode = repeat_counts.size() == 1;

      unsigned taskId = 0;
      // For every BD task
      for (auto &[rep, task_ops] : repeat_counts) {
        // The block containing aie.dma_start
        Block *start_bb = new Block();
        mem.getBody().push_back(start_bb);

        // The last block containing aie.end
        end_bb = new Block();
        mem.getBody().push_back(end_bb);
        auto end_bb_builder = OpBuilder::atBlockBegin(end_bb);
        end_bb_builder.setInsertionPointToEnd(end_bb);
        AIE::EndOp::create(end_bb_builder, loc);

        // First bd in task
        Block *first_bd = new Block();
        first_bd->insertBefore(end_bb);
        Block *next_bd = nullptr;
        for (size_t i = 0; i < task_ops.size(); i++) {
          auto memcpyOp = cast<air::MemcpyInterface>(task_ops[i]);
          Block *bd;
          if (i == 0)
            bd = first_bd;
          else
            bd = next_bd;
          auto b = OpBuilder::atBlockEnd(bd);
          if (i == task_ops.size() - 1) {
            if (infiniteBDLoopMode)
              AIE::NextBDOp::create(b, loc, first_bd);
            else
              AIE::NextBDOp::create(b, loc, end_bb);
          } else {
            next_bd = new Block();
            next_bd->insertBefore(end_bb);
            AIE::NextBDOp::create(b, loc, next_bd);
          }
          // ShimDMA/MemTileDMA/TileDMA getBuffer subclass APIs still take
          // AIE::TileOp; the tile parameter is unused by Shim/MemTile (which
          // derive the buffer from the memcpy op) and used only as the owner
          // tile by TileDMAAllocator. For TileDMA, `tile` here is always
          // physical (compute tiles use getPhysTileOp), so cast<TileOp> is
          // safe. Shim/MemTile may pass an LTO; the cast is unsafe in that
          // case but the body never dereferences the tile value, so the
          // cast<>'s null cast (to nullptr_t) does not blow up.
          auto bufferOp =
              dmaAlloc.getBuffer(BufferId,
                                 dyn_cast<AIE::TileOp>(tile.getOperation())
                                     ? cast<AIE::TileOp>(tile.getOperation())
                                     : nullptr,
                                 memcpyOp);
          if (failed(bufferOp)) {
            memcpyOp->emitOpError("failed to get buffer.");
            return failure();
          }
          auto locks = dmaAlloc.getLockForDMA(
              memcpyOp, tile, bufferOp.value().getOperation(),
              lockRaceConditionFix, lockRaceConditionFixV2);
          if (failed(locks))
            return memcpyOp->emitOpError("failed to get lock for dma.");
          auto newBD = generateDmaBd<bufferOpTy>(
              loc, dir, locks.value(), tile, targetModel, bd, memcpyOp,
              bufferOp.value(), chan, lockRaceConditionFixV2);
          // Attribute task_id is necessary to ensure that BDs do not get shared
          // across tasks, otherwise MLIR may fold BDs and cause BD sharing
          // across tasks.
          if (failed(newBD))
            return bufferOp.value()->emitOpError("failed to generate dma bd.");
          newBD.value()->setAttr(
              "task_id",
              IntegerAttr::get(IntegerType::get(b.getContext(), 32), taskId));

          // v2 chain-lock 2-slot ping-pong: splice a twin BD (on a second
          // buffer instance, sharing the chain locks) between the primary BD
          // and its next_bd target, giving producer-consumer overlap on the
          // shared L2 buffer.
          if constexpr (std::is_same_v<bufferOpTy, AIE::BufferOp>) {
            if (lockRaceConditionFixV2 && task_ops.size() == 1) {
              AIE::BufferOp primaryBuf = bufferOp.value();
              if (air::isChainLockCandidate(primaryBuf)) {
                auto clsOrFail =
                    dmaAlloc.getOrCreateChainLockSet(primaryBuf, tile);
                if (failed(clsOrFail))
                  return primaryBuf->emitOpError(
                      "v2 chain-lock: failed to look up chain lock set "
                      "for ping-pong twin");
                air::ChainLockSet *cls = clsOrFail.value();
                if (cls->pp_slots == 1) {
                  AIE::BufferOp twin = allocateBufferOp(
                      this->BufferId, primaryBuf.getType(), tile,
                      /*attr=*/nullptr, /*x=*/-1, /*y=*/-1);
                  dmaAlloc.activateChainPingPong(*cls, twin);
                }
                if (cls->twin_buf) {
                  // Splice bd_pong between bd and its current next_bd target.
                  auto primaryNextBd = cast<AIE::NextBDOp>(bd->getTerminator());
                  Block *origTarget = primaryNextBd.getDest();
                  Block *bd_pong = new Block();
                  bd_pong->insertBefore(end_bb);
                  primaryNextBd->setSuccessor(bd_pong, 0);
                  // Emit the twin's acq/dma_bd/rel into bd_pong using the
                  // SAME lock pair (chain locks are shared across ping/pong).
                  auto pongBD = generateDmaBd<bufferOpTy>(
                      loc, dir, locks.value(), tile, targetModel, bd_pong,
                      memcpyOp, cls->twin_buf, chan, lockRaceConditionFixV2);
                  if (failed(pongBD))
                    return cls->twin_buf->emitOpError(
                        "v2 chain-lock: failed to generate ping-pong twin BD");
                  pongBD.value()->setAttr(
                      "task_id",
                      IntegerAttr::get(IntegerType::get(b.getContext(), 32),
                                       taskId));
                  auto bd_pong_builder = OpBuilder::atBlockEnd(bd_pong);
                  AIE::NextBDOp::create(bd_pong_builder, loc, origTarget);
                }
              }
            }
          }
        }

        AIE::DMAStartOp startOp = nullptr;
        if (infiniteBDLoopMode)
          rep = 0;
        if (!channel_head) {
          channel_head = start_bb;
          auto b = OpBuilder::atBlockBegin(channel_head);
          startOp =
              AIE::DMAStartOp::create(b, loc, dir, chan, rep, first_bd, end_bb);
        } else {
          auto b = OpBuilder::atBlockBegin(start_bb);
          startOp = AIE::DMAStartOp::create(
              b, loc, dir, chan, rep, first_bd,
              channel_head->getTerminator()->getSuccessor(1));
          channel_head->getTerminator()->setSuccessor(start_bb, 1);
          channel_head = start_bb;
        }
        taskId++;
      }
    }
    return success();
  }

  template <typename bufferOpTy>
  FailureOr<AIE::DMABDOp>
  generateDmaBd(mlir::Location loc, AIE::DMAChannelDir dir,
                std::pair<AIE::LockOp, AIE::LockOp> locks, AIE::TileLike tile,
                const AIE::AIETargetModel &targetModel, Block *bd,
                air::MemcpyInterface memcpyOp, bufferOpTy bufferOp, int chan,
                bool lockRaceConditionFixV2 = false) {
    bool UsesSemaphoreLocks =
        targetModel.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks);
    bool isMM2S = (dir == AIE::DMAChannelDir::MM2S);

    auto b = OpBuilder::atBlockEnd(bd);
    auto acqLockOp = isMM2S ? locks.first : locks.second;
    auto relLockOp = isMM2S ? locks.second : locks.first;
    b.setInsertionPointToStart(bd);
    int64_t lockAqValue = -1;
    int64_t lockRelValue = -1;
    // 3-way shared L1: the channel put/get op carries the shared prod/cons
    // identity. The DMA participant must acquire/release N tokens against the
    // shared prod/cons pair (init=N / 0) so it drains all N writer releases
    // per cycle. N is the prod lock's init; getLockForDMA returns (cons, prod)
    // for the shared case, so N = locks.second.getInit().
    std::optional<int> sharedLockCount;
    if (memcpyOp->hasAttr("air.shared_prod_lock"))
      sharedLockCount = locks.second.getInit();
    bool useSharedL1LockCounts = sharedLockCount.has_value();
    // v2: when the chain-lock template applies for this buffer (fan-in/
    // fan-out shared L2 with per-stage signal locks), force per-BD lock
    // acq/rel counts to 1. The chain semantics rely on each writer/
    // reader holding/releasing exactly one token at a time; using the
    // legacy `getLockValuePair`-derived counts (N for the multi-side)
    // would break the chain because the multi-side BD would acquire/
    // release N tokens against the cap lock (init=#slots), reverting
    // to the legacy parallel-acquire behaviour.
    // The 3-way (compute-tile) and chain-lock (memtile) cases are mutually
    // exclusive, so a chained selection is unambiguous.
    bool useChainLockCounts =
        lockRaceConditionFixV2 &&
        isa_and_nonnull<AIE::BufferOp>(bufferOp.getOperation()) &&
        air::isChainLockCandidate(cast<AIE::BufferOp>(bufferOp.getOperation()));
    auto aie2LockVal =
        useSharedL1LockCounts
            ? std::pair<int64_t, int64_t>(*sharedLockCount, *sharedLockCount)
        : useChainLockCounts
            ? std::make_pair<int64_t, int64_t>(1, 1)
            : air::getLockValuePair(targetModel, bufferOp->getResult(0));
    if (!isMM2S) {
      lockAqValue = UsesSemaphoreLocks ? aie2LockVal.first : 0;
      lockRelValue = UsesSemaphoreLocks ? aie2LockVal.first : 1;
    } else {
      lockAqValue = UsesSemaphoreLocks ? aie2LockVal.second : 1;
      lockRelValue = UsesSemaphoreLocks ? aie2LockVal.second : 0;
    }
    // Count-free re-fed broadcast/relay: the resident buffer is re-broadcast N
    // times per fill, once per consumer row-block. On AIE2 memtile/core BDs
    // this cannot be a stride-0 repeat dim (HW unsupported) nor a repeat_count
    // BD (lock-once -> stale rebroadcast). It is realized as a count-free ring
    // (the MM2S self-loops via infiniteBDLoopMode) driven by the FILL releasing
    // the read-lock xN: the fill (S2MM) acquires the refill-lock xN (waits for
    // all N re-reads done before overwriting) and releases the read-lock xN
    // (enables N count-free MM2S re-reads). The MM2S side is unchanged (x1 per
    // fire). N = air.refeed_count, propagated onto the memtile buffer op (the
    // shared fill/drain rendezvous) by AllocL2BuffersPattern; read via the same
    // helper the lock allocator uses so init and acq/rel cannot diverge.
    if (isa_and_nonnull<AIE::BufferOp>(bufferOp.getOperation()) &&
        UsesSemaphoreLocks && !isMM2S) {
      int64_t refeedN = air::getRefeedCount(bufferOp.getOperation());
      if (refeedN > 1) {
        lockAqValue = refeedN;
        lockRelValue = refeedN;
      }
    }
    auto ndcpy = cast<air::MemcpyInterface>(memcpyOp);

    if (failed(isTileInbound(ndcpy, air::MemorySpace::L1)))
      return failure();

    Value memref = isTileInbound(ndcpy, air::MemorySpace::L1).value()
                       ? ndcpy.getDstMemref()
                       : ndcpy.getSrcMemref();
    SmallVector<Value> sizes =
        isTileInbound(ndcpy, air::MemorySpace::L1).value()
            ? ndcpy.getDstSizes()
            : ndcpy.getSrcSizes();
    SmallVector<Value> offsets =
        isTileInbound(ndcpy, air::MemorySpace::L1).value()
            ? ndcpy.getDstOffsets()
            : ndcpy.getSrcOffsets();
    SmallVector<Value> strides =
        isTileInbound(ndcpy, air::MemorySpace::L1).value()
            ? ndcpy.getDstStrides()
            : ndcpy.getSrcStrides();

    // Skip over repeat pattern at highest dimension; repeat pattern handled at
    // AIE::DMAStartOp.
    if (!strides.empty() && !sizes.empty() && !offsets.empty())
      if (auto const_highest_stride = getConstantIntValue(strides[0]))
        if (*const_highest_stride == 0) {
          strides.erase(strides.begin());
          sizes.erase(sizes.begin());
          offsets.erase(offsets.begin());
        }

    int64_t len = getMemcpySizesAsInt(memref, sizes);
    int64_t offset = air::get1DOffset(offsets, strides);

    Value length =
        arith::ConstantIndexOp::create(b, memcpyOp.getLoc(), len)->getResult(0);
    AIE::UseLockOp::create(b, loc, acqLockOp,
                           UsesSemaphoreLocks
                               ? AIE::LockAction::AcquireGreaterEqual
                               : AIE::LockAction::Acquire,
                           lockAqValue);

    // Packet flow routing: tag every BD (sender AND receiver) for a packet
    // channel with its pkt_id so the AIE switchbox and the receiving DMA's
    // per-BD packet-filter can demux. Pre-fix the lookup walked
    // PacketFlowOps by source, so receiver-side BDs (whose source value is
    // a different tile from the BD's owner) found nothing; the
    // `isMM2S && pktFlowOp` guard then dropped the filter entirely on the
    // receiver side. Result: a receiver channel multiplexed by two packet
    // flows (e.g. tile_0_2 S2MM 0 fed by gamma + res1ToCons in the
    // la_lgu_ld_cascade_fused design) accepted any arriving packet into
    // whichever BD was active, corrupting data and deadlocking on the
    // rate-imbalanced flow whose ping-pong BD slot got the wrong source's
    // packets. Each receiver memcpy op already emits its own BD in the
    // channel chain, so per-pkt_id filtering happens naturally once each
    // BD carries the right filter.
    //
    // Look up the pkt_id from packetIDForChannelName directly using the
    // air.channel symbol name. Non-packet channels (no entry) get no
    // filter -- same behaviour as today for circuit flows.
    AIE::PacketInfoAttr pktInfoAttr = nullptr;
    if (auto chanIfOp = dyn_cast_if_present<air::ChannelInterface>(
            memcpyOp.getOperation())) {
      auto it = packetIDForChannelName.find(chanIfOp.getChanName().str());
      if (it != packetIDForChannelName.end())
        pktInfoAttr =
            AIE::PacketInfoAttr::get(ndcpy->getContext(), 0, it->second);
    }

    std::vector<AIE::BDDimLayoutAttr> dims =
        air::getWrapsAndStrides(sizes, strides, ndcpy->getContext());
    auto wraps_and_strides =
        AIE::BDDimLayoutArrayAttr::get(ndcpy->getContext(), ArrayRef(dims));
    bool useDefaultDataAccessPattern =
        UsesSemaphoreLocks ? air::isDefaultDataAccessPattern(sizes, strides)
                           : true;

    // Extract padding attributes from the channel op, if present.
    AIE::BDPadLayoutArrayAttr padDims = nullptr;
    auto padBeforeAttr = ndcpy->getAttrOfType<DenseI32ArrayAttr>("pad_before");
    auto padAfterAttr = ndcpy->getAttrOfType<DenseI32ArrayAttr>("pad_after");
    if (padBeforeAttr && padAfterAttr) {
      auto padBefore = padBeforeAttr.asArrayRef();
      auto padAfter = padAfterAttr.asArrayRef();

      // If sizes/strides were canonicalized away (empty), reconstruct them
      // from the memref shape to match padding dimensionality.
      if (sizes.empty() && !padBefore.empty()) {
        auto memrefType = dyn_cast<MemRefType>(memref.getType());
        if (memrefType) {
          auto shape = memrefType.getShape();
          int64_t totalElements = std::accumulate(
              shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
          // For 1D memref with N-D padding, interpret as N-D with matching
          // dimensions. The innermost dimension is totalElements /
          // product-of-outer-pad-dims.
          if (shape.size() == 1 && padBefore.size() > 1) {
            // Reconstruct from the original sizes that were canonicalized:
            // We know padding has N dimensions. The total elements = product of
            // data sizes. Recover sizes from padding info:
            // size_i = (padded_total_i - pad_before_i - pad_after_i)
            // But we don't have padded_total_i. Use the flat memref size and
            // padding dims to infer: just use the flat length as innermost
            // with outer dim = 1 as a fallback, or look at buffer's actual
            // allocation for shape info.
            // For now, signal that wraps/strides must accompany padding.
            // Emit a warning and use simple 1D fallback.
            sizes.push_back(arith::ConstantIndexOp::create(b, memcpyOp.getLoc(),
                                                           totalElements)
                                ->getResult(0));
            strides.push_back(
                arith::ConstantIndexOp::create(b, memcpyOp.getLoc(), 1)
                    ->getResult(0));
            // Reduce padding to 1D to match
            padBefore = padBefore.take_back(1);
            padAfter = padAfter.take_back(1);
          } else if (shape.size() == padBefore.size()) {
            // N-D memref matches padding dims. Reconstruct default
            // sizes/strides
            int64_t stride = 1;
            for (int i = shape.size() - 1; i >= 0; i--) {
              sizes.insert(sizes.begin(), arith::ConstantIndexOp::create(
                                              b, memcpyOp.getLoc(), shape[i])
                                              ->getResult(0));
              strides.insert(strides.begin(), arith::ConstantIndexOp::create(
                                                  b, memcpyOp.getLoc(), stride)
                                                  ->getResult(0));
              stride *= shape[i];
            }
          } else {
            // 1D memref with 1D padding — simple case
            sizes.push_back(arith::ConstantIndexOp::create(b, memcpyOp.getLoc(),
                                                           totalElements)
                                ->getResult(0));
            strides.push_back(
                arith::ConstantIndexOp::create(b, memcpyOp.getLoc(), 1)
                    ->getResult(0));
          }
          // Recompute wraps and strides
          dims = air::getWrapsAndStrides(sizes, strides, ndcpy->getContext());
          wraps_and_strides = AIE::BDDimLayoutArrayAttr::get(
              ndcpy->getContext(), ArrayRef(dims));
        }
      }

      // Adjust padding rank to match sizes rank.
      // The sizes may have been recomputed by getWrapsAndStrides which can
      // change dimensionality (e.g., collapsing broadcast dimensions with
      // stride=0). Truncate leading zero-pad dimensions or extend with zeros.
      // Use SmallVectors for persistent storage of adjusted padding.
      SmallVector<int32_t> adjPadBefore(padBefore.begin(), padBefore.end());
      SmallVector<int32_t> adjPadAfter(padAfter.begin(), padAfter.end());
      if (!sizes.empty() && adjPadBefore.size() != sizes.size()) {
        if (adjPadBefore.size() > sizes.size()) {
          size_t excess = adjPadBefore.size() - sizes.size();
          bool canTruncate = true;
          for (size_t i = 0; i < excess; i++) {
            if (adjPadBefore[i] != 0 || adjPadAfter[i] != 0) {
              canTruncate = false;
              break;
            }
          }
          if (canTruncate) {
            adjPadBefore.erase(adjPadBefore.begin(),
                               adjPadBefore.begin() + excess);
            adjPadAfter.erase(adjPadAfter.begin(),
                              adjPadAfter.begin() + excess);
          } else {
            // Leading padding dimensions are non-zero — this happens when
            // getWrapsAndStrides collapsed a broadcast (stride=0) dimension
            // that had padding. The hardware handles broadcast via BD repeat
            // count, and the padded broadcast count is (size + pad_after).
            // Simply drop the leading padding dimensions — the BD repeat
            // count already accounts for the total iteration count including
            // padding via the air-split-launch-for-padding pass adjusting
            // the sizes in the boundary launch variants.
            adjPadBefore.erase(adjPadBefore.begin(),
                               adjPadBefore.begin() + excess);
            adjPadAfter.erase(adjPadAfter.begin(),
                              adjPadAfter.begin() + excess);
          }
        }
        if (adjPadBefore.size() < sizes.size()) {
          size_t deficit = sizes.size() - adjPadBefore.size();
          adjPadBefore.insert(adjPadBefore.begin(), deficit, 0);
          adjPadAfter.insert(adjPadAfter.begin(), deficit, 0);
        }
      }

      SmallVector<AIE::BDPadLayoutAttr> padLayouts;
      for (size_t i = 0; i < adjPadBefore.size(); i++) {
        padLayouts.push_back(AIE::BDPadLayoutAttr::get(
            ndcpy->getContext(), static_cast<uint16_t>(adjPadBefore[i]),
            static_cast<uint16_t>(adjPadAfter[i])));
      }
      padDims = AIE::BDPadLayoutArrayAttr::get(ndcpy->getContext(),
                                               ArrayRef(padLayouts));

      // Adjust len to include padding: product of (pad_before[i] + size[i] +
      // pad_after[i]) for all dimensions.
      int64_t paddedLen = 1;
      for (size_t i = 0; i < sizes.size(); i++) {
        auto sizeVal = getConstantIntValue(sizes[i]);
        if (!sizeVal) {
          return memcpyOp->emitOpError(
              "padding requires constant sizes for DMA BD length computation");
        }
        int32_t pb = i < adjPadBefore.size() ? adjPadBefore[i] : 0;
        int32_t pa = i < adjPadAfter.size() ? adjPadAfter[i] : 0;
        paddedLen *= (*sizeVal + pb + pa);
      }
      length = arith::ConstantIndexOp::create(b, memcpyOp.getLoc(), paddedLen)
                   ->getResult(0);
    }

    bool hasPad = padDims != nullptr;
    // When padding is present, wraps/strides must be emitted (the hardware
    // uses them together with padding).
    if (hasPad)
      useDefaultDataAccessPattern = false;
    bool hasWraps =
        !wraps_and_strides.getValue().empty() && !useDefaultDataAccessPattern;

    AIE::DMABDOp aieDmaBdOp = nullptr;
    if (hasWraps && hasPad)
      aieDmaBdOp = AIE::DMABDOp::create(
          b, loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value(),
          wraps_and_strides, padDims);
    else if (hasPad)
      aieDmaBdOp = AIE::DMABDOp::create(
          b, loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value(),
          padDims);
    else if (hasWraps)
      aieDmaBdOp = AIE::DMABDOp::create(
          b, loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value(),
          wraps_and_strides);
    else
      aieDmaBdOp = AIE::DMABDOp::create(
          b, loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value());
    if (pktInfoAttr)
      aieDmaBdOp->setAttr("packet", pktInfoAttr);
    AIE::UseLockOp::create(b, loc, relLockOp, AIE::LockAction::Release,
                           lockRelValue);
    return aieDmaBdOp;
  }

  // Converts an air.channel.put/get operation with channel_type = "npu_cascade"
  // into aie.get/put_cascade + vector.transfer_read/write sequence.
  // The conversion flattens the entire memref into a 1-D vector to match
  // the cascade data format expected by the AIE put/get_cascade ops.
  LogicalResult ConvertCascadeChannelIfToAIE(RewriterBase &rewriter,
                                             air::ChannelInterface op) {
    // Match only if the associated channel has channel_type = "npu_cascade".
    auto chan = air::getChannelDeclarationThroughSymbol(op);
    if (!chan)
      return op->emitOpError("cannot resolve channel symbol");

    if (chan.getChannelType().str() != "npu_cascade")
      return op->emitOpError("channel_type is not npu_cascade");

    Location loc = op.getLoc();
    Value memref = op.getMemref();
    MemRefType memrefTy = cast<MemRefType>(memref.getType());

    // Create the aie.get_cascade operation
    rewriter.setInsertionPoint(op);
    if (isa<air::ChannelGetOp>(op.getOperation())) {
      // For channel.get: Read data from the cascade interface into the memref.

      // Collapse the multi-dimensional shape into a 1D vector type to represent
      // the contiguous cascade payload.
      VectorType collapsedVecTy = VectorType::get(
          {std::accumulate(memrefTy.getShape().begin(),
                           memrefTy.getShape().end(), 1, std::multiplies<>())},
          memrefTy.getElementType());

      // Create the AIE get_cascade op to fetch cascade data as a single vector.
      Value cascadeData =
          AIE::GetCascadeOp::create(rewriter, loc, collapsedVecTy);

      // Collapse the destination memref into a 1D memref to match the data
      // layout.
      memref =
          collapseInnermostNDims(rewriter, loc, memrefTy.getRank(), memref);

      // Create a constant 0 index for writing at the beginning of the memref
      Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

      // Write the cascade vector into the memref.
      vector::TransferWriteOp::create(rewriter, loc, cascadeData, memref,
                                      ValueRange(c0),
                                      /*inBounds*/ SmallVector<bool>{true});
    } else if (isa<air::ChannelPutOp>(op.getOperation())) {
      // For channel.put: Read data from the memref and send it over the
      // cascade.

      // Collapse the source memref into a 1D memref to read out the full data.
      memref =
          collapseInnermostNDims(rewriter, loc, memrefTy.getRank(), memref);
      VectorType collapsedVecTy = VectorType::get(
          {std::accumulate(memrefTy.getShape().begin(),
                           memrefTy.getShape().end(), 1, std::multiplies<>())},
          memrefTy.getElementType());

      // Create a constant 0 index for writing at the beginning of the memref
      Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

      // Read the entire data payload as a 1D vector.
      Value cascadeData = vector::TransferReadOp::create(
          rewriter, loc, collapsedVecTy, memref, ValueRange(c0), /*padding*/
          arith::getZeroConstant(rewriter, loc, memrefTy.getElementType()),
          /*inBounds*/ SmallVector<bool>{true});

      // Send the vector data via AIE put_cascade.
      AIE::PutCascadeOp::create(rewriter, loc, cascadeData);
    }

    // Remove the original air.channel.get op
    if (air::isAsyncOp(op)) {
      IRMapping remap;
      auto newWaitAll = air::replaceAsyncOpWithWaitAll(rewriter, remap, op);
      rewriter.replaceOp(op, newWaitAll.getAsyncToken());
    } else
      rewriter.eraseOp(op);

    return success();
  }

  FailureOr<air::ChannelInterface> TileCascadeChannelIfUsingScfFor(
      RewriterBase &rewriter, air::ChannelInterface op, unsigned cascadeWidth) {
    // Match only if the associated channel has channel_type = "npu_cascade".
    auto chan = air::getChannelDeclarationThroughSymbol(op);
    if (!chan)
      return op->emitOpError("cannot resolve channel symbol");

    if (chan.getChannelType().str() != "npu_cascade")
      return op->emitOpError("channel_type is not npu_cascade");

    Location loc = op.getLoc();
    Value memref = op.getMemref();
    MemRefType memrefTy = cast<MemRefType>(memref.getType());

    // Calculate cascade tile size based on cascade width and element size
    Type elemType = memrefTy.getElementType();
    if (!isa<IntegerType, FloatType>(elemType))
      return op->emitOpError("cascade channel requires integer or float "
                             "element type, got ")
             << elemType;
    unsigned elementWidth = elemType.getIntOrFloatBitWidth();
    if (elementWidth == 0)
      return op->emitOpError("cascade channel element type has zero bit width");
    int64_t cascadeTileSize = llvm::divideCeil(cascadeWidth, elementWidth);

    // Calculate total number of elements
    int64_t totalElements = 1;
    for (int64_t dim : memrefTy.getShape()) {
      if (ShapedType::isDynamic(dim))
        return op->emitOpError("cascade channel requires static memref shape");
      totalElements *= dim;
    }

    // If total elements is less than or equal to cascade tile size, no tiling
    // needed
    if (totalElements <= cascadeTileSize)
      return op;

    // ========== FLATTEN THE MEMREF TO 1D ==========
    rewriter.setInsertionPoint(op);

    // Check for non-trivial layouts that cannot be safely flattened.
    // Only identity (default contiguous row-major) layouts are supported.
    if (!memrefTy.getLayout().isIdentity())
      return op->emitOpError("cascade channel requires contiguous row-major "
                             "memref layout, got ")
             << memrefTy;

    // Create reassociation indices to collapse all dims into one
    // e.g., for rank 3: [[0, 1, 2]]
    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices allDims;
    for (int64_t i = 0; i < memrefTy.getRank(); i++)
      allDims.push_back(i);
    reassociation.push_back(allDims);

    // Create the collapsed (1D) memref type
    MemRefType flatMemrefTy =
        MemRefType::get({totalElements}, memrefTy.getElementType(),
                        MemRefLayoutAttrInterface{}, memrefTy.getMemorySpace());

    // Create memref.collapse_shape op
    Value flatMemref = memref::CollapseShapeOp::create(
        rewriter, loc, flatMemrefTy, memref, reassociation);

    // ========== UPDATE MEMREF OPERAND IN PLACE ==========
    // Find the memref operand index (after async dependencies and indices)
    // Operand layout: [async_dependencies..., indices..., src, ...]
    int memrefOperandOffset = cast<air::AsyncOpInterface>(op.getOperation())
                                  .getAsyncDependencies()
                                  .size() +
                              op.getIndices().size();
    op->setOperand(memrefOperandOffset, flatMemref);

    // ========== TILE THE NOW-1D CHANNEL OP ==========
    scf::SCFTilingOptions options;
    options.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

    // Tile the 1D memref with cascadeTileSize
    SmallVector<OpFoldResult> tileSizesOfr = {
        rewriter.getIndexAttr(cascadeTileSize)};
    options.setTileSizes(tileSizesOfr);

    if (auto put = dyn_cast_if_present<air::ChannelPutOp>(op.getOperation())) {
      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, put, options);
      if (failed(tilingResult))
        return failure();
      return dyn_cast_if_present<air::ChannelInterface>(
          tilingResult->tiledOps.front());
    } else if (auto get =
                   dyn_cast_if_present<air::ChannelGetOp>(op.getOperation())) {
      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, get, options);
      if (failed(tilingResult))
        return failure();
      return dyn_cast_if_present<air::ChannelInterface>(
          tilingResult->tiledOps.front());
    }
    return failure();
  }

  AIE::ShimDMAOp getShimDMAOp(AIE::TileLike tile) {
    auto users = tile->getResult(0).getUsers();
    for (auto user : users)
      if (auto shimDMAOp = dyn_cast_if_present<AIE::ShimDMAOp>(*user))
        return shimDMAOp;
    return nullptr;
  }

  AIE::MemTileDMAOp getMemTileDMAOp(AIE::TileLike tile) {
    auto users = tile->getResult(0).getUsers();
    for (auto user : users)
      if (auto memTileDMAOp = dyn_cast_if_present<AIE::MemTileDMAOp>(*user))
        return memTileDMAOp;
    return nullptr;
  }

  // Lower mmio-typed channels into runtime-sequence MMIO writes.
  //
  // For each `air.channel @c [...] {channel_type = "npu_mmio"}`:
  //   * each `air.channel.get @c` inside an `aie.core` is replaced by an
  //     erase — the destination L1 `aie.buffer` is populated by the host
  //     before the core runs, so the get is a no-op;
  //   * each `air.channel.put @c` outside the device (i.e. living in the
  //     L3/launch-side `func.func` that becomes the runtime sequence) is
  //     rewritten to `aiex.npu.blockwrite` targeting the L1 buffer's
  //     symbol with the put's source memref as the data payload.
  //
  // V1 restrictions enforced here:
  //   * the put's source must be a `memref.get_global` of a constant
  //     `memref.global`, because `aiex.npu.blockwrite` encodes the data
  //     directly in the instruction stream;
  //   * the get's destination must resolve to an `aie.buffer` (it must be
  //     an L1 allocation, already converted by air-to-aie);
  //   * non-broadcast pairs match by equal constant indices; non-constant
  //     indices and host-side puts with no matching get are hard errors.
  //   * the source `memref.global` must have no users outside the put's
  //     enclosing func (V1 limitation; see Comment 2 in PR #1568).
  LogicalResult lowerAIRMMIOChannelOps(AIE::DeviceOp device) {
    IRRewriter rewriter(device->getContext());

    // Collect all mmio channel decls reachable from this device.
    SmallVector<air::ChannelOp> mmioChannels;
    auto collectMMIO = [&](Operation *root) {
      root->walk([&](air::ChannelOp chan) {
        if (chan.getChannelType() == "npu_mmio")
          if (!llvm::is_contained(mmioChannels, chan))
            mmioChannels.push_back(chan);
      });
    };
    collectMMIO(device);
    if (auto module = device->getParentOfType<ModuleOp>())
      collectMMIO(module);

    // Helper to compare two ValueRange of indices for constant equality.
    auto constIndices = [](ValueRange v) {
      return getConstantIntValues(getAsOpFoldResult(v));
    };
    auto sameConstIndices = [&](ValueRange a, ValueRange b) {
      auto av = constIndices(a), bv = constIndices(b);
      return av && bv && *av == *bv;
    };

    // Helper: trace a memref Value back through launch/segment/herd block
    // arguments and trivial view ops to a defining `memref.get_global`.
    // The put often appears inside an `air.launch` body, where its source
    // is a kernel block-arg whose backing operand outside the launch is
    // the actual `memref.get_global`.
    auto getSourceGlobal = [](Value v) -> memref::GetGlobalOp {
      while (v) {
        if (auto gg = v.getDefiningOp<memref::GetGlobalOp>())
          return gg;
        if (Operation *def = v.getDefiningOp()) {
          if (auto cs = dyn_cast<memref::CastOp>(def)) {
            v = cs.getSource();
            continue;
          }
          return nullptr;
        }
        // No defining op — `v` is a block argument. Walk to the
        // corresponding kernel operand of the enclosing launch/segment/herd.
        auto ba = dyn_cast<BlockArgument>(v);
        if (!ba)
          return nullptr;
        Operation *parent = ba.getOwner()->getParentOp();
        Value next = nullptr;
        if (auto launch = dyn_cast_if_present<air::LaunchOp>(parent)) {
          auto args = launch.getKernelArguments();
          auto operands = launch.getKernelOperands();
          for (size_t i = 0; i < args.size(); i++)
            if (args[i] == ba) {
              next = operands[i];
              break;
            }
        } else if (auto seg = dyn_cast_if_present<air::SegmentOp>(parent)) {
          auto args = seg.getKernelArguments();
          auto operands = seg.getKernelOperands();
          for (size_t i = 0; i < args.size(); i++)
            if (args[i] == ba) {
              next = operands[i];
              break;
            }
        } else if (auto herd = dyn_cast_if_present<air::HerdOp>(parent)) {
          auto args = herd.getKernelArguments();
          auto operands = herd.getKernelOperands();
          for (size_t i = 0; i < args.size(); i++)
            if (args[i] == ba) {
              next = operands[i];
              break;
            }
        }
        if (!next)
          return nullptr;
        v = next;
      }
      return nullptr;
    };

    // Helper: trace a memref Value back through trivial view ops to its
    // defining aie.buffer, if any.
    auto getDefiningBuffer = [](Value v) -> AIE::BufferOp {
      while (v) {
        if (auto buf = v.getDefiningOp<AIE::BufferOp>())
          return buf;
        // Peel through common view-style ops that don't change the buffer
        // identity.
        Operation *def = v.getDefiningOp();
        if (!def)
          return nullptr;
        if (auto sv = dyn_cast<memref::SubViewOp>(def)) {
          v = sv.getSource();
          continue;
        }
        if (auto cs = dyn_cast<memref::CastOp>(def)) {
          v = cs.getSource();
          continue;
        }
        return nullptr;
      }
      return nullptr;
    };

    Operation *root = device->getParentOp();
    while (root && !isa<ModuleOp>(root))
      root = root->getParentOp();
    if (!root)
      root = device->getParentOp();

    for (auto chan : mmioChannels) {
      auto chanName = chan.getSymName();

      SmallVector<air::ChannelPutOp> hostPuts;
      SmallVector<air::ChannelPutOp> devicePuts;
      SmallVector<air::ChannelGetOp> deviceGets;
      SmallVector<air::ChannelGetOp> hostGets;
      // air-to-aie's pre-processing clones each L3 channel.put into the
      // device with its source rebound to an aie.external_buffer (so that
      // later flow allocation can route through shim DMA). For mmio we
      // bypass shim DMA entirely, so those device-side puts are pure
      // artifacts and must be discarded — keep only puts in the L3 control
      // func, where the original `memref.get_global` source is reachable
      // via launch / segment / herd kernel args.
      root->walk([&](air::ChannelPutOp put) {
        if (put.getChanName() != chanName)
          return;
        if (put->getParentOfType<AIE::DeviceOp>())
          devicePuts.push_back(put);
        else
          hostPuts.push_back(put);
      });
      // Gets exist in two places after air-to-aie outlining:
      //   * inside `aie.core` ops within the device (the lowered form,
      //     where the destination is an `aie.buffer` whose sym_name is
      //     what blockwrite needs to reference),
      //   * inside the original `air.herd` body in the L3 control func
      //     (kept around for AIRLoweringPass; destination is a plain
      //     `memref.alloc`, no aie.buffer to reference).
      // Both must go for mmio. Use the device-side gets for buffer lookup;
      // erase the host-side gets without any further processing.
      device.walk([&](air::ChannelGetOp get) {
        if (get.getChanName() == chanName)
          deviceGets.push_back(get);
      });
      root->walk([&](air::ChannelGetOp get) {
        if (get.getChanName() != chanName)
          return;
        if (get->getParentOfType<AIE::DeviceOp>())
          return;
        hostGets.push_back(get);
      });

      if (hostPuts.empty() && devicePuts.empty() && deviceGets.empty() &&
          hostGets.empty())
        continue;

      // mmio has no hardware broadcast. When the channel is declared with a
      // `broadcast_shape`, the lowering instead emits one blockwrite per
      // destination buffer carrying the same payload — the controller
      // writes each tile individually. This matches the shape of
      // `q_n8_w128` in MMIO_BENCHMARK.md, which proved sub-microsecond per
      // write at LLaMA-3.2-1B Q sizes.
      bool isBcast = chan.isBroadcast();

      // Non-broadcast pairs match by constant-index equality. A non-
      // constant index would silently fail every match and the put would
      // be erased below with no blockwrite emitted — reject up front.
      if (!isBcast) {
        auto rejectIfNonConst = [&](Operation *op, ValueRange indices,
                                    StringRef kind) -> LogicalResult {
          if (constIndices(indices))
            return success();
          return op->emitOpError("channel_type=\"npu_mmio\" non-broadcast ")
                 << kind << " requires compile-time constant indices";
        };
        for (auto put : hostPuts)
          if (failed(rejectIfNonConst(put, put.getIndices(), "put")))
            return failure();
        for (auto get : deviceGets)
          if (failed(rejectIfNonConst(get, get.getIndices(), "get")))
            return failure();
      }

      // For each L3-side put, find every matching get and stamp the
      // source data onto the destination L1 buffer's `initial_value`
      // attribute. Match rule:
      //   * non-broadcast: indices must be constant-equal between put/get;
      //   * broadcast: every device-side get on this channel matches every
      //     put (one put fans out to all destinations).
      //
      // The aie.buffer initial_value is loaded into the tile by
      // AIERTControl::initBuffers (XAie_DataMemBlockWrite) at device-init
      // time — before any core starts. This eliminates the host↔core
      // race that would arise from placing the data delivery in the
      // runtime sequence (where blockwrites would race CDO-started cores).
      // It also handles bf16/other float types natively, so no i32
      // repack is required.
      for (auto put : hostPuts) {
        Value src = put.getMemref();
        memref::GetGlobalOp getGlobalOp = getSourceGlobal(src);
        if (!getGlobalOp)
          return put.emitOpError(
              "channel_type=\"npu_mmio\" put requires source memref defined by "
              "memref.get_global of a constant memref.global");

        StringAttr origName = getGlobalOp.getNameAttr().getAttr();
        Operation *moduleOp = device->getParentOp();
        while (moduleOp && !isa<ModuleOp>(moduleOp))
          moduleOp = moduleOp->getParentOp();
        auto moduleGlobal = dyn_cast_if_present<memref::GlobalOp>(
            moduleOp ? SymbolTable::lookupSymbolIn(moduleOp, origName)
                     : nullptr);
        if (!moduleGlobal)
          return getGlobalOp.emitOpError(
              "channel_type=\"npu_mmio\" lowering: cannot find memref.global "
              "for the put source at module scope");

        auto initOpt = moduleGlobal.getInitialValue();
        auto initDense =
            initOpt ? dyn_cast<DenseElementsAttr>(*initOpt) : nullptr;
        if (!initDense)
          return put.emitOpError(
              "channel_type=\"npu_mmio\" source memref.global must have a "
              "DenseElementsAttr initializer");

        unsigned matchCount = 0;
        for (auto get : deviceGets) {
          if (!isBcast && !sameConstIndices(put.getIndices(), get.getIndices()))
            continue;
          AIE::BufferOp bufferOp = getDefiningBuffer(get.getMemref());
          if (!bufferOp)
            return get.emitOpError(
                "channel_type=\"npu_mmio\" get destination does not resolve to "
                "an aie.buffer (must be an L1 allocation)");

          // Element type and total element count must match between source
          // and destination so the DenseElementsAttr is valid for the
          // buffer's memref type.
          auto bufMemTy = bufferOp.getType();
          auto srcMemTy = cast<MemRefType>(getGlobalOp.getType());
          if (bufMemTy.getElementType() != srcMemTy.getElementType())
            return get.emitOpError("channel_type=\"npu_mmio\" "
                                   "source/destination element type "
                                   "mismatch (source: ")
                   << srcMemTy.getElementType()
                   << ", destination: " << bufMemTy.getElementType() << ")";
          if (bufMemTy.getNumElements() != srcMemTy.getNumElements())
            return get.emitOpError("channel_type=\"npu_mmio\" "
                                   "source/destination element count "
                                   "mismatch (source: ")
                   << srcMemTy.getNumElements()
                   << ", destination: " << bufMemTy.getNumElements() << ")";

          // Reshape the source DenseElementsAttr to match the destination
          // buffer's tensor shape (same bytes, possibly different rank).
          auto bufTensorTy = RankedTensorType::get(bufMemTy.getShape(),
                                                   bufMemTy.getElementType());
          auto reshapedInit = initDense.reshape(bufTensorTy);

          if (auto existing = bufferOp.getInitialValue())
            return bufferOp.emitOpError(
                "channel_type=\"npu_mmio\" destination aie.buffer already has "
                "an "
                "initial_value; cannot stamp two sources into one buffer");
          bufferOp.setInitialValueAttr(reshapedInit);
          ++matchCount;
        }
        if (matchCount == 0)
          return put.emitOpError(
              "channel_type=\"npu_mmio\" put has no matching "
              "device-side air.channel.get");
      }

      // Erase all mmio puts (host-side ones have been replaced with
      // blockwrite; device-side ones are pure pre-processing artifacts
      // and have no representation in the lowered IR).
      auto erasePut = [](air::ChannelPutOp put) {
        if (auto async = dyn_cast<air::AsyncOpInterface>(put.getOperation())) {
          if (async.getAsyncToken()) {
            OpBuilder b(put);
            put->replaceAllUsesWith(air::WaitAllOp::create(
                b, put.getLoc(), air::AsyncTokenType::get(put.getContext()),
                async.getAsyncDependencies()));
          }
        }
        put->erase();
      };
      for (auto put : hostPuts)
        erasePut(put);
      for (auto put : devicePuts)
        erasePut(put);
      // Erase all mmio gets — the destination L1 buffer is populated by
      // the blockwrite issued from the host before the core starts.
      auto eraseGet = [](air::ChannelGetOp get) {
        if (auto async = dyn_cast<air::AsyncOpInterface>(get.getOperation())) {
          if (async.getAsyncToken()) {
            OpBuilder b(get);
            get->replaceAllUsesWith(air::WaitAllOp::create(
                b, get.getLoc(), air::AsyncTokenType::get(get.getContext()),
                async.getAsyncDependencies()));
          }
        }
        get->erase();
      };
      for (auto get : deviceGets)
        eraseGet(get);
      for (auto get : hostGets)
        eraseGet(get);
    }
    return success();
  }

  template <typename T>
  LogicalResult lowerAIRMemcpyOp(AIE::DeviceOp device,
                                 air::ShimDMAAllocator &shimDmaAlloc,
                                 AIRToAIEConversionOptions options) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : device.getOps<AIE::CoreOp>())
      cores.push_back(c);

    const auto &target_model = device.getTargetModel();
    // OpBuilder builder(device);
    IRRewriter rewriter(device->getContext());

    // Unlike shimDmaAlloc, tileDmaAlloc is local to device because it does not
    // need to export to airrt.metadata
    air::TileDMAAllocator tileDmaAlloc(device);
    air::MemTileDMAAllocator memTileDmaAlloc(device);
    air::CascadeAllocator core_cascade_alloc(device);

    // Place memcpy ops onto DMA tiles, channels and flows
    auto r = placeDMAChannelsAndRouteFlows<T>(device, shimDmaAlloc,
                                              memTileDmaAlloc, tileDmaAlloc,
                                              core_cascade_alloc, options);
    if (failed(r))
      return r;

    for (AIE::CoreOp core : cores) {
      AIE::TileOp tile = core.getTileOp();

      // emit the acquire and release of the L1 buffer locks
      // lock_allocation_list lock_allocs;
      llvm::SetVector<Operation *> allocs_to_remap;

      for (auto &alloc : tileDmaAlloc.mm2s_allocs) {
        if (!alloc.foundAlloc(tile))
          continue;
        for (auto o : alloc.memcpyOps) {
          if (!o)
            continue;
          auto memcpyOpIf = dyn_cast_if_present<air::MemcpyInterface>(o);
          if (!memcpyOpIf)
            return o->emitOpError("does not have air::MemcpyInterface");
          if (failed(allocateCoreLocksPerMemcpyOp(rewriter, memcpyOpIf,
                                                  allocs_to_remap, target_model,
                                                  tileDmaAlloc, tile))) {
            return o->emitOpError("failed to allocate core locks");
          }
        }
      }
      for (auto &alloc : tileDmaAlloc.s2mm_allocs) {
        if (!alloc.foundAlloc(tile))
          continue;
        for (auto o : alloc.memcpyOps) {
          if (!o)
            continue;
          auto memcpyOpIf = dyn_cast_if_present<air::MemcpyInterface>(o);
          if (!memcpyOpIf)
            return o->emitOpError("does not have air::MemcpyInterface");
          if (failed(allocateCoreLocksPerMemcpyOp(rewriter, memcpyOpIf,
                                                  allocs_to_remap, target_model,
                                                  tileDmaAlloc, tile))) {
            return o->emitOpError("failed to allocate core locks");
          }
        }
      }

      for (auto o : allocs_to_remap) {
        Value alloc = o->getResult(0);
        for (auto u : alloc.getUsers()) {
          if (auto dealloc = dyn_cast_if_present<memref::DeallocOp>(u)) {
            dealloc.erase();
            break;
          }
        }
        if (isa<memref::AllocOp>(o))
          o->erase();
      }

      // Generate the TileDMA bd program. That is, generate the aie.mem
      // body for the tile. Above we collected per channel lists of dma
      // copy operations. We'll assume these lists are in the correct
      // execution order and generate a aie.mem program to loop over
      // each list.

      // Collect memcpy ops wrt each DMA channel from chessboard; make aie.mem
      // dmabd program
      llvm::MapVector<std::pair<AIE::DMAChannelDir, int>,
                      std::vector<Operation *>>
          tile_dma_memcpys;

      for (auto &alloc : tileDmaAlloc.mm2s_allocs) {
        if (!alloc.foundAlloc(tile))
          continue;
        std::pair<AIE::DMAChannelDir, int> mm2s_chan = {
            alloc.dma_channel.direction, alloc.dma_channel.channel};
        for (auto &o : alloc.memcpyOps) {
          tile_dma_memcpys[mm2s_chan].push_back(o);
        }
      }
      for (auto &alloc : tileDmaAlloc.s2mm_allocs) {
        if (!alloc.foundAlloc(tile))
          continue;
        std::pair<AIE::DMAChannelDir, int> s2mm_chan = {
            alloc.dma_channel.direction, alloc.dma_channel.channel};
        for (auto &o : alloc.memcpyOps) {
          tile_dma_memcpys[s2mm_chan].push_back(o);
        }
      }

      auto loc = core->getLoc();

      // make a aie.mem for the tile dma
      auto mem = tile.getMemOp();
      if (!mem && tile_dma_memcpys.size()) {
        rewriter.setInsertionPoint(core);
        mem = AIE::MemOp::create(rewriter, loc, tile);
      }

      if (failed(generateDmaBdProgram<air::TileDMAAllocator, AIE::BufferOp,
                                      AIE::MemOp>(
              rewriter, target_model, tile_dma_memcpys, tileDmaAlloc, loc, mem,
              tile))) {
        mem->emitOpError("failed to generate dma bd program.");
        return failure();
      }

      // Materialize cascade put and get allocated on cores into put_ and
      // get_cascade ops.
      for (auto *allocList : {&core_cascade_alloc.cascade_put_allocs,
                              &core_cascade_alloc.cascade_get_allocs}) {
        for (auto &alloc : *allocList) {
          if (!alloc.foundAlloc(tile))
            continue;
          for (auto o : alloc.memcpyOps) {
            if (!o)
              continue;
            auto channelOpIf = dyn_cast_if_present<air::ChannelInterface>(o);
            if (!channelOpIf)
              return o->emitOpError("does not have air::ChannelInterface");
            auto tiledChannelIf = TileCascadeChannelIfUsingScfFor(
                rewriter, channelOpIf,
                target_model.getAccumulatorCascadeSize());
            if (failed(tiledChannelIf))
              continue;
            if (failed(
                    ConvertCascadeChannelIfToAIE(rewriter, *tiledChannelIf))) {
              return o->emitOpError("failed to generate cascade program");
            }
          }
        }
      }
    }

    // Generate L3 DMA program

    // Gather all shim tiles and memtiles used in design. Both physical
    // (AIE::TileOp) and unplaced (AIE::LogicalTileOp) entries flow through
    // here uniformly via TileLike; the downstream aie.shim_dma /
    // aie.memtile_dma ops accept any Index-typed tile operand.
    std::vector<AIE::TileLike> shimtiles;
    std::vector<AIE::TileLike> memTileTiles;
    for (auto &alloc : shimDmaAlloc.mm2s_allocs) {
      auto tile = alloc.getDmaTile();
      if (tile.isShimTile())
        push_back_if_unique<AIE::TileLike>(shimtiles, tile);
      else {
        tile->emitOpError(
            "tile is logged for shim DMA allocation, but is not shim tile.");
        return failure();
      }
    }
    for (auto &alloc : memTileDmaAlloc.mm2s_allocs) {
      auto tile = alloc.getDmaTile();
      if (tile.isMemTile())
        push_back_if_unique<AIE::TileLike>(memTileTiles, tile);
      else {
        tile->emitOpError(
            "tile is logged for memtile DMA allocation, but is not memtile.");
        return failure();
      }
    }

    // Disable generation of shim dma program if generate_shim_dma unset
    if (!options.generate_shim_dma)
      shimtiles.clear();

    for (auto tile : shimtiles) {
      // Collect memcpy ops wrt each DMA channel
      llvm::MapVector<std::pair<AIE::DMAChannelDir, int>,
                      std::vector<Operation *>>
          shim_dma_memcpys;

      for (auto &alloc : shimDmaAlloc.mm2s_allocs) {
        if (alloc.foundAlloc(tile)) {
          std::pair<AIE::DMAChannelDir, int> mm2s_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            shim_dma_memcpys[mm2s_chan].push_back(o);
          }
        }
      }
      for (auto &alloc : shimDmaAlloc.s2mm_allocs) {
        if (alloc.foundAlloc(tile)) {
          std::pair<AIE::DMAChannelDir, int> s2mm_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            shim_dma_memcpys[s2mm_chan].push_back(o);
          }
        }
      }

      // Generate aie.shim_dma op
      AIE::ShimDMAOp shimDMA = getShimDMAOp(tile);
      if (!shimDMA) {
        rewriter.setInsertionPoint(device.getBody()->getTerminator());
        shimDMA =
            AIE::ShimDMAOp::create(rewriter, rewriter.getUnknownLoc(),
                                   rewriter.getIndexType(), tile->getResult(0));
      }

      auto loc = rewriter.getUnknownLoc();

      // Generate DMA BD program
      if (failed(generateDmaBdProgram<air::ShimDMAAllocator,
                                      AIE::ExternalBufferOp, AIE::ShimDMAOp>(
              rewriter, target_model, shim_dma_memcpys, shimDmaAlloc, loc,
              shimDMA, tile))) {
        shimDMA->emitOpError("failed to generate dma bd program.");
        return failure();
      }
    }

    // Generate L2 DMA program

    for (auto tile : memTileTiles) {
      // Collect memcpy ops wrt each DMA channel from chessboard; make aie.mem
      // dmabd program
      llvm::MapVector<std::pair<AIE::DMAChannelDir, int>,
                      std::vector<Operation *>>
          memtile_dma_memcpys;

      for (auto &alloc : memTileDmaAlloc.mm2s_allocs) {
        if (alloc.foundAlloc(tile)) {
          std::pair<AIE::DMAChannelDir, int> mm2s_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            memtile_dma_memcpys[mm2s_chan].push_back(o);
          }
        }
      }
      for (auto &alloc : memTileDmaAlloc.s2mm_allocs) {
        if (alloc.foundAlloc(tile)) {
          std::pair<AIE::DMAChannelDir, int> s2mm_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            memtile_dma_memcpys[s2mm_chan].push_back(o);
          }
        }
      }

      // Generate aie.memtile_dma op
      AIE::MemTileDMAOp memTileDMA = getMemTileDMAOp(tile);
      if (!memTileDMA) {
        rewriter.setInsertionPoint(device.getBody()->getTerminator());
        memTileDMA = AIE::MemTileDMAOp::create(
            rewriter, rewriter.getUnknownLoc(), rewriter.getIndexType(),
            tile->getResult(0));
      }

      auto loc = rewriter.getUnknownLoc();

      // Generate DMA BD program
      if (failed(generateDmaBdProgram<air::MemTileDMAAllocator, AIE::BufferOp,
                                      AIE::MemTileDMAOp>(
              rewriter, target_model, memtile_dma_memcpys, memTileDmaAlloc, loc,
              memTileDMA, tile, options.use_lock_race_condition_fix,
              options.use_lock_race_condition_fix_v2))) {
        memTileDMA->emitOpError("failed to generate dma bd program.");
        return failure();
      }
    }

    // Clear air::allocation_info_t allocations' memcpyOps field
    for (auto &alloc : shimDmaAlloc.mm2s_allocs)
      alloc.memcpyOps.clear();
    for (auto &alloc : shimDmaAlloc.s2mm_allocs)
      alloc.memcpyOps.clear();
    for (auto &alloc : memTileDmaAlloc.mm2s_allocs)
      alloc.memcpyOps.clear();
    for (auto &alloc : memTileDmaAlloc.s2mm_allocs)
      alloc.memcpyOps.clear();
    for (auto &alloc : tileDmaAlloc.mm2s_allocs)
      alloc.memcpyOps.clear();
    for (auto &alloc : tileDmaAlloc.s2mm_allocs)
      alloc.memcpyOps.clear();

    // Lower channel_type="npu_mmio" puts/gets into runtime-sequence blockwrites
    // before the generic erase loop below removes the underlying air ops.
    // Only meaningful for the ChannelInterface specialization; for the
    // DmaMemcpyNd specialization there are no air.channel ops to convert.
    if constexpr (std::is_same_v<T, air::ChannelInterface>) {
      if (failed(lowerAIRMMIOChannelOps(device)))
        return failure();
    }

    // erase the memcpy operations in aie.device
    std::vector<Operation *> memcpy_ops;
    getAIRMemcpyOpInRegion<T>(device.getRegion(), memcpy_ops);
    for (auto o : memcpy_ops) {
      auto a = dyn_cast_if_present<air::AsyncOpInterface>(o);
      if (a && a.getAsyncToken()) {
        OpBuilder b(o);
        o->replaceAllUsesWith(air::WaitAllOp::create(
            b, o->getLoc(), air::AsyncTokenType::get(o->getContext()),
            a.getAsyncDependencies()));
      }
      o->erase();
    }

    return success();
  }

  void createTracePacketFlow(AIE::DeviceOp device) {
    OpBuilder builder(device);
    const auto &target_model = device.getTargetModel();

    // Collect existing TileOps
    DenseMap<AIE::TileID, AIE::TileOp> tiles;
    for (auto tile : device.getOps<AIE::TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[{colIndex, rowIndex}] = tile;
    }

    // On AIE2 core tiles the only legal switchbox destinations for the
    // Trace source bundle are FIFO and South (see AIE2TargetModel::
    // isLegalTileConnection in mlir-aie's AIETargetModel.cpp). When the
    // South channels of a compute-tile switchbox are already saturated by
    // circuit-switched data flows (the tile's own outbound DMA flow plus
    // any passthrough from tiles above it in the same column), the
    // pathfinder can't route Trace through South, falls back to the
    // Trace<->DMA0 hardware mux, and ends up generating an
    // 'aie.packet_rules source=DMA:0' that collides with the circuit
    // 'aie.connect source=DMA:0' for the local data flow:
    //   'aie.packet_rules' op packet switched source DMA0 cannot match
    //   another connect or masterset operation
    // (regressed by mlir-aie #3139 centroid-ing the data shim onto the
    // herd column, which concentrates all data flows on a single column
    // and saturates every south channel of every compute tile in that
    // column.)
    //
    // Estimate the south-bound circuit pressure on each compute tile and
    // skip trace emission when the pressure would consume every South
    // channel. Pressure for tile (col, row) = local outbound circuit
    // flows that exit south + outbound circuit flows from tiles strictly
    // above (col, row) in the same column (since the pathfinder routes
    // them straight down through (col, row)'s switchbox).
    const int southCapacity = 4; // AIE2 compute tile: South ch 0..3
    DenseMap<int, SmallVector<int>> colToSouthRows;
    auto getTileLike = [](Value v) -> AIE::TileLike {
      return dyn_cast_or_null<AIE::TileLike>(v.getDefiningOp());
    };
    for (auto flow : device.getOps<AIE::FlowOp>()) {
      auto srcT = getTileLike(flow.getSource());
      auto dstT = getTileLike(flow.getDest());
      if (!srcT || !dstT)
        continue;
      if (!srcT.isCoreTile())
        continue;
      auto srcCol = srcT.tryGetCol(), srcRow = srcT.tryGetRow();
      auto dstCol = dstT.tryGetCol(), dstRow = dstT.tryGetRow();
      if (!srcCol || !srcRow)
        continue;
      // Same-column south-bound flow: dst col matches src col, dst row
      // strictly less than src row. If dst has unknown coords (logical
      // memtile awaiting placement), assume it lands on the same column
      // as its consumer core (the herd column) -- which is the typical
      // post-#3139 placement and the case that triggers the conflict.
      bool isSouth = false;
      if (dstCol && dstRow)
        isSouth = (*dstCol == *srcCol) && (*dstRow < *srcRow);
      else if (dstT.isMemTile())
        isSouth = true; // unplaced memtile assumed south of the herd
      if (!isSouth)
        continue;
      colToSouthRows[*srcCol].push_back(*srcRow);
    }
    auto southPressureAt = [&](int col, int row) -> int {
      int pressure = 0;
      auto it = colToSouthRows.find(col);
      if (it == colToSouthRows.end())
        return 0;
      for (int r : it->second)
        if (r >= row)
          ++pressure;
      return pressure;
    };

    // Create packet flows
    for (auto srcTile : device.getOps<AIE::TileOp>()) {
      int srcColIndex = srcTile.colIndex();
      int srcRowIndex = srcTile.rowIndex();
      AIE::TileOp destTile;

      if (target_model.isCoreTile(srcColIndex, srcRowIndex) ||
          target_model.isMemTile(srcColIndex, srcRowIndex)) {
        if (target_model.isCoreTile(srcColIndex, srcRowIndex) &&
            southPressureAt(srcColIndex, srcRowIndex) >= southCapacity)
          continue;
        int destColIndex = srcColIndex; // todo: allocation?
        int destRowIndex = 0;
        if (!tiles[{destColIndex, destRowIndex}]) {
          builder.setInsertionPointToStart(device.getBody());
          destTile = AIE::TileOp::create(builder, builder.getUnknownLoc(),
                                         destColIndex, destRowIndex);
          tiles[{destColIndex, destRowIndex}] = destTile;
        } else {
          destTile = tiles[{destColIndex, destRowIndex}];
        }
        int destChan = 1; // todo: allocation?

        builder.setInsertionPoint(device.getBody()->getTerminator());
        auto keep_pkt_header = builder.getBoolAttr(true);
        // Trace packets terminate at shim DMA -> draw from the shim id pool.
        // assignShimPacketID skips ids already claimed by intra-device flows
        // placed earlier in this device (latent collision: pre-redesign trace
        // could reuse an id from the device's intra-device pass).
        (void)createPacketFlowOp(
            builder, assignShimPacketID(), srcTile, AIE::WireBundle::Trace, 0,
            destTile, AIE::WireBundle::DMA, destChan, keep_pkt_header);
      }
    }
  }

  void runTestPatterns() {

    auto m = getOperation();
    auto ctx = m->getContext();

    RewritePatternSet patterns(ctx);
    std::map<AIE::BufferOp, AIE::TileLike> bufferToMemtileMap;

    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      m.emitOpError("Invalid aie.device option");
      signalPassFailure();
      return;
    }

    // Map test pattern strings to pipeline stages
    auto mapTestPatternToPipelineStage =
        [](StringRef testPattern) -> PipelineStage {
      if (testPattern.contains("after-clone-memcpys"))
        return PipelineStage::AfterCloneMemcpys;
      if (testPattern.contains("after-lower-execute"))
        return PipelineStage::AfterLowerExecute;
      if (testPattern.contains("after-specialize-channel"))
        return PipelineStage::AfterSpecializeChannel;
      if (testPattern.contains("after-alloc-buffers"))
        return PipelineStage::AfterAllocBuffers;
      if (testPattern.contains("after-renumber-memcpy"))
        return PipelineStage::AfterRenumberMemcpy;
      if (testPattern.contains("after-lower-memcpy"))
        return PipelineStage::AfterLowerAIRMemcpy;
      if (testPattern.contains("after-trace-packet-flow"))
        return PipelineStage::AfterTracePacketFlow;
      if (testPattern.contains("after-lower-memref-copy"))
        return PipelineStage::AfterLowerMemRefCopy;
      if (testPattern.contains("to-aie-full") ||
          testPattern.contains("complete"))
        return PipelineStage::Complete;
      // Default for "to-aie-mlir": stop right after
      // createAIEModulesAndOutlineCores This matches the original behavior of
      // printing the module before running any further transformations.
      return PipelineStage::AfterCreateAIEModules;
    };

    if (clTestPatterns.find("to-aie-mlir") != std::string::npos) {
      std::vector<
          std::tuple<AIE::DeviceOp, air::HerdOp, AIRToAIEConversionOptions>>
          aie_modules;
      AIRToAIEConversionOptions options = {
          /*.col_offset = */ clColOffset,
          /*.row_offset = */ clRowOffset,
          /*.emit_while = */ clEmitWhileLoop,
          /*.emit_herd_lock = */ clEmitHerdLock,
          /*.generate_shim_dma = */ clGenerateShimDMA,
          /*.insert_trace_packet_flow = */ clInsertTracePacketFlow,
          /*.use_lock_race_condition_fix = */ clUseLockRaceConditionFix,
          /*.use_lock_race_condition_fix_v2 = */ clUseLockRaceConditionFixV2,
          /*.device = */ *device,
          /*.stack_size = */ clStackSize};

      // Pre-pipeline: renumber memcpy ops at module level
      air::renumberMemcpyIfOps(&m.getRegion());

      if (failed(createAIEModulesAndOutlineCores(m, aie_modules, options))) {
        signalPassFailure();
        return;
      }

      // Determine the pipeline stage to stop at based on test pattern flags
      PipelineStage stopStage =
          mapTestPatternToPipelineStage(StringRef(clTestPatterns));

      std::set<AIE::DeviceOp> seen;
      for (auto &p : aie_modules) {
        auto d = std::get<0>(p);
        auto h = std::get<1>(p);
        auto device_options = std::get<2>(p);
        auto parentModule = d->getParentOfType<ModuleOp>();

        if (seen.find(d) != seen.end())
          continue;
        seen.insert(d);

        // Run shared pipeline with the specified breakpoint
        if (failed(runDevicePipeline(d, parentModule, h, bufferToMemtileMap,
                                     device_options, clUseObjFifo,
                                     stopStage))) {
          signalPassFailure();
          return;
        }

        // Print the module after reaching the breakpoint for debugging
        if (stopStage == PipelineStage::AfterCreateAIEModules) {
          // Legacy "to-aie-mlir" behavior: print parent module after outline
          parentModule.print(llvm::outs());
          llvm::outs() << "\n";
        }
      }
    }

    if (clTestPatterns.find("lower-air-execute") != std::string::npos)
      patterns.insert<LowerAIRExecutePattern>(ctx);
    if (clTestPatterns.find("alloc-l1-buffers") != std::string::npos)
      patterns.insert<AllocL1BuffersPattern, AllocL1BuffersPattern>(ctx,
                                                                    BufferId);
    if (clTestPatterns.find("specialize-affine-if") != std::string::npos)
      patterns.insert<SpecializeAffineIfPattern>(ctx);
    if (clTestPatterns.find("specialize-scf-if") != std::string::npos)
      patterns.insert<SpecializeScfIfPattern>(ctx);
    if (clTestPatterns.find("lower-scf-tokens") != std::string::npos)
      patterns.insert<LowerScfTokenPattern>(ctx);

    ShimTileAllocator shimTileAlloc(AIE::getTargetModel(*device));
    std::map<Operation *, AIE::ObjectFifoCreateOp> linksToComplete;
    if (clTestPatterns.find("lower-air-channels") != std::string::npos) {
      patterns.insert<LowerAIRChannelsPattern>(
          ctx, shimTileAlloc, bufferToMemtileMap, linksToComplete);
    }
    if (clTestPatterns.find("lower-air-ping-pong") != std::string::npos) {
      patterns.insert<LowerAIRPingPongPattern>(ctx);
    }
    std::map<std::string, std::string> chan_to_chan_map;
    if (clTestPatterns.find("specialize-channel-bundle") != std::string::npos) {
      // Enforce max size constraint
      int maxSize =
          isa<AIE::AIE1TargetModel>(AIE::getTargetModel(*device)) ? -1 : 1023;
      patterns.insert<SpecializeChannelBundlePattern>(ctx, chan_to_chan_map,
                                                      maxSize);
    }

    if (patterns.getNativePatterns().size())
      (void)applyPatternsGreedily(m, std::move(patterns));

    // Shim LTOs emitted by the test-path LowerAIRChannelsPattern are left
    // unresolved here, matching the production path. Downstream
    // `aie-place-tiles` resolves them with full objfifo connectivity.
  }

  void runOnOperation() override {

    if (!clTestPatterns.empty()) {
      runTestPatterns();
      return;
    }

    auto module = getOperation();
    OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());

    // v1/v2 mutual exclusion: they apply different fixes to overlapping
    // problem areas, and combining them would interleave dummy-op
    // insertion (v1) with chain-lock allocation (v2) in ways that don't
    // have a coherent semantics. Force the user to pick one.
    if (clUseLockRaceConditionFix && clUseLockRaceConditionFixV2) {
      module.emitOpError(
          "use-lock-race-condition-fix and use-lock-race-condition-fix-v2 are "
          "mutually exclusive; enable at most one");
      signalPassFailure();
      return;
    }

    auto loc = builder.getUnknownLoc();
    auto module_meta = airrt::ModuleMetadataOp::create(builder, loc);
    builder.createBlock(&module_meta.getSegments());
    airrt::ModuleMetadataTerminatorOp::create(builder, loc);

    // If we have multiple herds then we must emit them into different aie
    // modules to avoid resource conflicts in the AIE physical dialect.
    std::vector<
        std::tuple<AIE::DeviceOp, air::HerdOp, AIRToAIEConversionOptions>>
        aie_devices;

    std::map<AIE::BufferOp, AIE::TileLike> bufferToMemtileMap;
    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      module.emitOpError("Invalid aie.device option");
      signalPassFailure();
      return;
    }
    air::renumberMemcpyIfOps(&module.getRegion());
    AIRToAIEConversionOptions options = {
        /* .col_offset = */ clColOffset,
        /* .row_offset = */ clRowOffset,
        /* .emit_while = */ clEmitWhileLoop,
        /* .emit_herd_lock = */ clEmitHerdLock,
        /* .generate_shim_dma = */ clGenerateShimDMA,
        /* .insert_trace_packet_flow = */ clInsertTracePacketFlow,
        /* .use_lock_race_condition_fix = */ clUseLockRaceConditionFix,
        /* .use_lock_race_condition_fix_v2 = */ clUseLockRaceConditionFixV2,
        /* .device = */ *device,
        /* .stack_size = */ clStackSize};
    if (failed(createAIEModulesAndOutlineCores(module, aie_devices, options))) {
      signalPassFailure();
      return;
    }

    std::set<AIE::DeviceOp> seen;
    DenseSet<func::FuncOp> shimUnrolledFuncs;
    for (auto &p : aie_devices) {
      auto device = std::get<0>(p);
      air::HerdOp h = std::get<1>(p);
      auto device_options = std::get<2>(p);
      auto ctx = device->getContext();

      if (seen.find(device) != seen.end())
        continue;
      seen.insert(device);

      // Reset per-device packet-flow tracking for segment unroll. Each
      // isolated device assigns its own pkt_ids; shim ids are drawn from a
      // global counter (nextGlobalShimPacketID) for cross-device uniqueness.
      packetIDForChannelName.clear();
      claimedPacketIDs.clear();

      // The shim tile allocation is not unified for dma and channel lowering
      // so we disallow a mix of dma and channel ops.
      bool hasDma = false;
      bool hasChan = false;
      device.walk([&](Operation *o) {
        hasDma |= isa<air::DmaMemcpyNdOp>(o);
        hasChan |= isa<air::ChannelInterface>(o);
      });
      if (hasDma && hasChan) {
        device.emitOpError(
            ": lowering of segments containing both dma copies and "
            "channels is not supported");
        signalPassFailure();
        return;
      }

      air::ShimDMAAllocator shimDmaAlloc(device);
      std::map<int, int> chan_renumber_reverse_map;
      ShimTileAllocator shimTileAlloc(device.getTargetModel());
      std::map<std::string, std::string> chan_to_chan_map;

      // Get the parent launch for this herd to filter memcpy ops
      air::LaunchOp targetLaunch = h->getParentOfType<air::LaunchOp>();

      if (clUseObjFifo) {
        cloneL2AndL3MemcpysToDeviceOp(
            builder, device, module, /*clone_l2*/ true, /*clone_l3*/ false,
            /*use_lock_race_cond_fix*/
            device_options.use_lock_race_condition_fix, targetLaunch);
        specializeHerdAffineIf(device);
        lowerAirExecute(device);
        lowerScfAirTokens(device);
        specializeChannelBundle(device, chan_to_chan_map);
        // Only remove orphaned channels when segment unroll is active
        if (device->hasAttr("segment_unroll_x") ||
            device->hasAttr("segment_unroll_y"))
          removeOrphanedChannels(device);
        air::renumberMemcpyIfOps(&device.getRegion());
        LowerAIRPingPong(device);
        allocL2Buffers(device, bufferToMemtileMap, BufferId);
        if (failed(
                lowerAIRChannels(device, shimTileAlloc, bufferToMemtileMap))) {
          signalPassFailure();
          return;
        }
        allocL1Buffers(device, BufferId);
      } else {
        cloneL2AndL3MemcpysToDeviceOp(
            builder, device, module, /*clone_l2*/ true, /*clone_l3*/ true,
            /*use_lock_race_cond_fix*/
            device_options.use_lock_race_condition_fix, targetLaunch);
        specializeHerdAffineIf(device);
        lowerAirExecute(device);
        lowerScfAirTokens(device);
        specializeChannelBundle(device, chan_to_chan_map);
        // Only remove orphaned channels when segment unroll is active
        if (device->hasAttr("segment_unroll_x") ||
            device->hasAttr("segment_unroll_y"))
          removeOrphanedChannels(device);
        specializeL2MemrefsIntoMemtiles(device);
        allocL1Buffers(device, BufferId);
        allocL2Buffers(device, bufferToMemtileMap, BufferId);
        air::renumberMemcpyIfOps(&device.getRegion(),
                                 chan_renumber_reverse_map);
        if (failed(lowerAIRMemcpyOp<air::ChannelInterface>(device, shimDmaAlloc,
                                                           device_options))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(lowerAIRMemcpyOp<air::DmaMemcpyNdOp>(device, shimDmaAlloc,
                                                      device_options))) {
        signalPassFailure();
        return;
      }

      if (device_options.insert_trace_packet_flow)
        createTracePacketFlow(device);

      // Lower L1-to-L1 memref.copy to loops before removing remaining copies.
      // AIE cores don't have native memcpy, so L1-to-L1 copies must be
      // converted to explicit load/store loops via linalg.copy.
      lowerMemRefCopyToLoops(device);

      SmallVector<air::HerdOp, 4> herds;
      SmallVector<air::SegmentOp, 4> segs;
      std::set<int64_t> dma_ids;
      if (auto p = h->getParentOfType<air::SegmentOp>()) {
        p.walk([&](air::HerdOp h) { herds.push_back(h); });
        segs.push_back(p);
      } else {
        herds.push_back(h);
      }

      auto segment_name =
          device->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
              .getValue();
      auto segment_meta = getOrCreateSegmentMetadata(module_meta, segment_name);
      for (auto herd : herds) {
        auto herd_meta = createHerdMetadata(segment_meta, herd);

        std::vector<Attribute> dma_allocations;

        // AIE1 dma metadata format
        getDmaAllocationMetadata(builder, ctx, herd, shimDmaAlloc.s2mm_allocs,
                                 AIE::DMAChannelDir::S2MM,
                                 chan_renumber_reverse_map, dma_allocations);
        getDmaAllocationMetadata(builder, ctx, herd, shimDmaAlloc.mm2s_allocs,
                                 AIE::DMAChannelDir::MM2S,
                                 chan_renumber_reverse_map, dma_allocations);

        herd_meta->setAttr("dma_allocations",
                           ArrayAttr::get(ctx, dma_allocations));
      }
      for (auto seg : segs) {
        std::vector<Attribute> dma_allocations;

        // AIE1 memtile dma metadata format
        getDmaAllocationMetadata(builder, ctx, seg, shimDmaAlloc.mm2s_allocs,
                                 AIE::DMAChannelDir::MM2S,
                                 chan_renumber_reverse_map, dma_allocations);

        auto segment_name =
            device->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                .getValue();
        auto segment_meta =
            getOrCreateSegmentMetadata(module_meta, segment_name);
        segment_meta->setAttr("dma_allocations",
                              ArrayAttr::get(ctx, dma_allocations));
      }

      if (isa<AIE::AIE2TargetModel>(device.getTargetModel()) && !clUseObjFifo) {
        bool isSegmentUnrolled = device->hasAttr("segment_unroll_x") ||
                                 device->hasAttr("segment_unroll_y");
        // Process shim DMA metadata per-device. For segment-unrolled designs,
        // each device processes its own allocations independently. Shim-side
        // ops belonging to other devices are silently skipped (skipUnlinked).
        auto func = h->getParentOfType<func::FuncOp>();

        // Unroll scf.parallel loops around shim-side channel ops (at func
        // level) so each tile gets a discrete channel put/get. This is needed
        // when air-dma-to-channel wraps channel ops in scf.parallel; without
        // unrolling, all tiles share one channel op and get the same packet ID.
        // Guard with shimUnrolledFuncs to avoid redundant rewrites when
        // multiple devices share the same parent func.
        if (!shimUnrolledFuncs.contains(func)) {
          shimUnrolledFuncs.insert(func);
          RewritePatternSet shimUnrollPatterns(ctx);
          air::populateAIRunrollAIRChannelPutGetInScfParallelPatterns(
              shimUnrollPatterns);
          if (failed(
                  applyPatternsGreedily(func, std::move(shimUnrollPatterns)))) {
            func->emitOpError(
                "failed to unroll scf.parallel around shim channel ops");
            signalPassFailure();
            return;
          }
        }

        std::vector<air::MemcpyInterface> shimMemcpyIfOps;
        func.walk([&](air::ChannelInterface o) {
          auto parentLaunch = o->getParentOfType<air::LaunchOp>();
          if (parentLaunch && parentLaunch != targetLaunch)
            return;
          auto memrefTy =
              dyn_cast_if_present<BaseMemRefType>(o.getMemref().getType());
          if (memrefTy && air::isL3(memrefTy))
            shimMemcpyIfOps.push_back(
                dyn_cast_if_present<air::MemcpyInterface>(o.getOperation()));
        });
        func.walk([&](air::DmaMemcpyNdOp o) {
          auto parentLaunch = o->getParentOfType<air::LaunchOp>();
          if (parentLaunch && parentLaunch != targetLaunch)
            return;
          auto srcMemrefTy =
              dyn_cast_if_present<BaseMemRefType>(o.getSrcMemref().getType());
          if (srcMemrefTy && air::isL3(srcMemrefTy))
            shimMemcpyIfOps.push_back(o);
          auto dstMemrefTy =
              dyn_cast_if_present<BaseMemRefType>(o.getDstMemref().getType());
          if (dstMemrefTy && air::isL3(dstMemrefTy))
            shimMemcpyIfOps.push_back(o);
        });
        builder.setInsertionPoint(device.getBody()->getTerminator());
        if (failed(createShimDMAAllocationOps(builder, ctx, shimMemcpyIfOps,
                                              shimDmaAlloc,
                                              chan_renumber_reverse_map,
                                              /*skipUnlinked=*/
                                              isSegmentUnrolled))) {
          signalPassFailure();
          return;
        }
      }

      else if (clUseObjFifo) {
        // ObjectFifo metadata linkage
        auto f = h->getParentOfType<func::FuncOp>();

        std::vector<air::ChannelInterface> channel_ops;
        f.walk([&](air::ChannelInterface o) {
          if (!o->getParentOfType<air::HerdOp>())
            channel_ops.push_back(o);
        });
        for (auto &n : shimTileAlloc.chan_names)
          labelAIRDmaOpsWithMetadataObjFifo(channel_ops, n, chan_to_chan_map);
      }

      RewritePatternSet patterns(ctx);
      air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsGreedily(device, std::move(patterns));

      // Remove ops via rewrite patterns.
      RewritePatternSet removepatterns(ctx);
      removepatterns.add<OpRemovalPattern<memref::DeallocOp>,
                         OpRemovalPattern<air::WaitAllOp>,
                         OpRemovalPattern<memref::CopyOp>,
                         OpRemovalPattern<memref::AssumeAlignmentOp>>(ctx);
      ConversionTarget target(*ctx);
      target.addIllegalOp<memref::DeallocOp, air::WaitAllOp, memref::CopyOp,
                          memref::AssumeAlignmentOp>();
      if (failed(applyPartialConversion(device, target,
                                        std::move(removepatterns))))
        signalPassFailure();

      // Clean up dead memref.get_global/memref.global left by outlineAIECores
      // after DMA/channel lowering consumed their users.
      removeDeadGlobalOps(device);
    }
  }

  // Packet-flow id tracking.
  //
  // Per-device (reset each device):
  //   packetIDForChannelName: air.channel symbol -> assigned pkt_id. Drives
  //     reuse for broadcast (one source, many receivers) and lookup from
  //     downstream lowering. Keyed by symbol name because air.channel decls
  //     are duplicated under aie.device and its parent module.
  //   claimedPacketIDs: pkt_ids already assigned in this device. Drives gap-
  //     finding for intra-device flows and skip-on-collision for trace/shim.
  //
  // Global (persists across devices):
  //   nextGlobalShimPacketID: monotonic source for shim (and trace) pkt_ids.
  //     Shim flows require global uniqueness because the host runtime keys
  //     packet identification on (shim tile, pkt_id).
  llvm::StringMap<int> packetIDForChannelName;
  llvm::SmallSet<int, 32> claimedPacketIDs;
  int nextGlobalShimPacketID = 0;

private:
  // Collapses the innermost `numDims` dimensions of a MemRef `val` into a
  // single dimension.
  //
  // Example:
  //   Input shape:  [2, 3, 4]
  //   numDims:      2
  //   Result shape: [2, 12]
  //
  // This is achieved by multiplying the sizes of the innermost `numDims`
  // dimensions and creating a `memref.collapse_shape` operation with the
  // appropriate reassociation indices.
  Value collapseInnermostNDims(RewriterBase &b, Location loc, int numDims,
                               Value val) {
    if (numDims < 2)
      return val;
    // Ensure the value has a MemRefType.
    auto memRefTy = dyn_cast_if_present<MemRefType>(val.getType());
    if (!memRefTy)
      return nullptr;

    // Get the original shape of the memref.
    auto shape = memRefTy.getShape();

    // Compute the new collapsed dimension as the product of the last `numDims`
    // sizes.
    int64_t newInnerMostDim = std::accumulate(
        shape.end() - numDims, shape.end(), 1, std::multiplies<>());

    // Build the new shape: same as original except that the last `numDims` are
    // collapsed into one.
    SmallVector<int64_t, 4> newShape{shape.begin(), shape.end() - numDims + 1};
    newShape[shape.size() - numDims] = newInnerMostDim;

    // Compute reassociation indices required by `memref.collapse_shape`.
    auto reassocIndices =
        getReassociationIndicesForCollapse(shape, newShape).value();

    // Create a `memref.collapse_shape` op at the specified location.
    ImplicitLocOpBuilder iBuilder(loc, b);
    return memref::CollapseShapeOp::create(iBuilder, val, reassocIndices);
  }
};

class SplitAIEDevicesPass
    : public air::impl::AIRSplitDevicesBase<SplitAIEDevicesPass> {

public:
  SplitAIEDevicesPass() = default;
  SplitAIEDevicesPass(const SplitAIEDevicesPass &pass) {}
  void runOnOperation() override {
    ModuleOp m = getOperation();
    auto ctx = &getContext();

    SmallVector<AIE::DeviceOp> deviceOps;
    m.walk([&](AIE::DeviceOp d) { deviceOps.push_back(d); });

    unsigned segment_number = 0;
    OpBuilder builder(ctx);
    for (auto device : deviceOps) {

      std::string segment_name;
      if (auto attr = device->getAttrOfType<StringAttr>(
              SymbolTable::getSymbolAttrName())) {
        segment_name = attr.getValue().str();
      } else {
        segment_name = "segment_" + std::to_string(segment_number++);
      }
      std::string aie_module_name = "aie." + segment_name;

      ModuleOp aie_module =
          ModuleOp::create(builder.getUnknownLoc(), StringRef(aie_module_name));
      builder.setInsertionPointToStart(aie_module.getBody());
      IRMapping remap;
      for (auto &o : m.getBody()->getOperations()) {

        // if it's not the current device op, don't clone it
        if (isa<AIE::DeviceOp>(o) && &o != device.getOperation())
          continue;

        // if it's a function without a use in the device op, don't clone it
        if (isa<func::FuncOp>(o)) {
          bool has_use = false;
          for (auto u : o.getUsers()) {
            has_use |= (u->getParentOfType<AIE::DeviceOp>() == device);
          }
          if (!has_use)
            continue;
        }

        // clone op into the new module
        builder.clone(o, remap);
      }

      // run lowering patterns
      //
      RewritePatternSet removepatterns(ctx);
      removepatterns.add<OpRemovalPattern<airrt::ModuleMetadataOp>>(ctx);

      ConversionTarget target(*ctx);
      target.addIllegalDialect<airrt::AIRRtDialect>();
      if (failed(applyPartialConversion(aie_module, target,
                                        std::move(removepatterns))))
        signalPassFailure();

      // write module to stdout or file
      //
      if (clOutputPrefix != "-") {
        if (clOutputPrefix != "/dev/null") {
          std::error_code EC;
          std::string fname = clOutputPrefix + aie_module_name + ".mlir";
          llvm::raw_fd_ostream aie_ostream(fname, EC);
          aie_module.print(aie_ostream);
        }
      } else {
        aie_module.print(llvm::outs());
      }
    }

    for (auto device : deviceOps)
      device.erase();
  }
};

// Custom version of LinalgOpToLibraryCallRewrite
class AIRLinalgOpToLibraryCallRewrite
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  AIRLinalgOpToLibraryCallRewrite(MLIRContext *ctx, std::string &linkWith)
      : OpInterfaceRewritePattern(ctx), linkWith(linkWith) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto fnName = op.getLibraryCallName();
    if (fnName.empty())
      return failure();

    // Function to get operands of the library call that will
    // replace the given linalg op.
    auto getLibFnOperands = [](linalg::LinalgOp op) {
      SmallVector<Value> operands;
      for (auto operand : op->getOperands()) {
        auto operation = operand.getDefiningOp();
        if (isa_and_present<memref::ReshapeOp, memref::ExpandShapeOp,
                            memref::CollapseShapeOp>(operation)) {
          operands.push_back(operation->getOperand(0));
          continue;
        }
        operands.push_back(operand);
      }
      return operands;
    };

    auto libFnOperands = getLibFnOperands(op);

    // fnName is a dynamic std::string, unique it via a SymbolRefAttr.
    FlatSymbolRefAttr fnNameAttr =
        SymbolRefAttr::get(rewriter.getContext(), fnName);
    auto module = op->getParentOfType<ModuleOp>();
    auto sym = module.lookupSymbol(fnNameAttr.getAttr());
    if (!sym) {
      auto libFnType = rewriter.getFunctionType(
          ValueRange(ArrayRef<Value>(libFnOperands)).getTypes(), {});
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(module.getBody(),
                                 std::prev(module.getBody()->end()));
      func::FuncOp funcOp = func::FuncOp::create(
          rewriter, op->getLoc(), fnNameAttr.getValue(), libFnType);
      // Insert a function attribute that will trigger the emission of the
      // corresponding `_mlir_ciface_xxx` interface so that external libraries
      // see a normalized ABI. This interface is added during std to llvm
      // conversion.
      funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                      UnitAttr::get(op->getContext()));
      if (linkWith != "") {
        // Insert a function attribute that will link to the compiled kernel
        // object file (.o).
        funcOp->setAttr("link_with",
                        StringAttr::get(rewriter.getContext(), linkWith));
      }
      funcOp.setPrivate();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fnNameAttr.getValue(), TypeRange(),
        ValueRange(ArrayRef<Value>(libFnOperands)));

    if (auto herd = op->getParentOfType<air::HerdOp>())
      rewriter.modifyOpInPlace(herd, [&]() {
        herd->setAttr("link_with",
                      StringAttr::get(rewriter.getContext(), linkWith));
      });

    return success();
  }

private:
  std::string &linkWith;
};

struct AIRLinalgToFuncPass
    : public air::impl::AIRLinalgToFuncBase<AIRLinalgToFuncPass> {
  void runOnOperation() override;
};

void AIRLinalgToFuncPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect,
                         scf::SCFDialect, air::airDialect, AIE::AIEDialect,
                         cf::ControlFlowDialect, ub::UBDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  RewritePatternSet patterns(&getContext());
  patterns.insert<AIRLinalgOpToLibraryCallRewrite>(&getContext(), clLinkWith);
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace air
} // namespace xilinx

namespace xilinx {
namespace air {

FailureOr<ModuleOp> convertAIRToAIE(mlir::RewriterBase &rewriter,
                                    air::SegmentOp p) {
  uint64_t BufferId{0};
  std::string segment_name = "segment_0";
  if (auto attr =
          p->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    segment_name = attr.getValue().str();

  std::string aie_module_name = "aie." + segment_name;
  ModuleOp aie_module =
      ModuleOp::create(rewriter.getUnknownLoc(), StringRef(aie_module_name));

  auto device = AIE::symbolizeAIEDevice("xcvc1902");
  if (!device) {
    p->emitOpError("Invalid aie.device option");
    return failure();
  }
  AIRToAIEConversionOptions options = {
      /* .col_offset = */ 7,
      /* .row_offset = */ 2,
      /* .emit_while = */ false,
      /* .emit_herd_lock = */ false,
      /* .generate_shim_dma = */ false,
      /* .insert_trace_packet_flow = */ false,
      /* .use_lock_race_condition_fix = */ true,
      /* .use_lock_race_condition_fix_v2 = */ false,
      /* .device = */ *device};
  std::vector<std::pair<ModuleOp, air::HerdOp>> aie_modules;
  p.walk([&](air::HerdOp h) { aie_modules.push_back({aie_module, h}); });
  for (auto &p : aie_modules) {
    ModuleOp aie_module = std::get<0>(p);
    air::HerdOp h = std::get<1>(p);
    rewriter.setInsertionPointToStart(aie_module.getBody());
    auto devOp = AIE::DeviceOp::create(
        rewriter, aie_module.getLoc(),
        AIE::AIEDeviceAttr::get(rewriter.getContext(), options.device));
    setAIEDeviceDataLayout(rewriter, devOp);
    AIE::DeviceOp::ensureTerminator(devOp.getRegion(), rewriter,
                                    devOp.getLoc());
    if (failed(outlineAIECores(rewriter, devOp, h, options)))
      return failure();

    auto ctx = aie_module->getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<SpecializeAffineIfPattern>(ctx);
    patterns.insert<SpecializeScfIfPattern>(ctx);
    patterns.insert<LowerAIRExecutePattern>(ctx);
    patterns.insert<AllocL1BuffersPattern>(ctx, BufferId);
    air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsGreedily(aie_module, std::move(patterns));
  }

  return aie_module;
}

std::unique_ptr<mlir::Pass> createAIRToAIEPass() {
  return std::make_unique<AIRToAIEPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRToAIEPass(const AIRToAIEOptions &options) {
  return std::make_unique<AIRToAIEPass>(options);
}

std::unique_ptr<mlir::Pass> createAIRSplitDevicesPass() {
  return std::make_unique<SplitAIEDevicesPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAIRLinalgToFuncPass() {
  return std::make_unique<AIRLinalgToFuncPass>();
}

} // namespace air
} // namespace xilinx
