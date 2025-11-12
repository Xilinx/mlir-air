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

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  bool use_packet_flow_at_shim_dmas;
  bool use_lock_race_condition_fix;
  AIE::AIEDevice device;
};

// get memcpy operation volumn (elements) as int
int getMemcpySizesAsInt(Value memref, SmallVector<Value> sizes) {
  BaseMemRefType memTy = llvm::cast<BaseMemRefType>(memref.getType());
  if (sizes.empty())
    return air::getTensorVolume(memTy);
  else {
    int output = 1;
    for (auto s : sizes) {
      auto c = dyn_cast<arith::ConstantIndexOp>(s.getDefiningOp());
      if (!c) {
        output = -1;
        break;
      }
      output *= c.value();
    }
    return output;
  }
}

struct ShimTileAllocator {

  std::vector<int> shim_columns;
  int shim_dma_channels;
  const AIE::AIETargetModel &aie_target;

  struct shim_allocation_info_t {
    int shim_col;
    int available_channels;
    std::vector<std::string> chan_names;
  };

  std::vector<shim_allocation_info_t> mm2s_allocs, s2mm_allocs;

  ShimTileAllocator(const AIE::AIETargetModel &target) : aie_target(target) {
    for (int i = 0, e = aie_target.columns(); i < e; i++) {
      if (aie_target.isShimNOCTile(i, 0)) {
        shim_columns.push_back(i);
        shim_dma_channels = aie_target.getNumDestSwitchboxConnections(
            i, 0, AIE::WireBundle::FIFO);
      }
    }
  }

  AIE::TileOp getShimTile(AIE::DeviceOp aie_device, int src_memory_space,
                          int dst_memory_space, std::string chan_name) {
    bool isMM2S = (src_memory_space < dst_memory_space);
    auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

    // return first available shim tile with a free channel
    for (auto &t : *allocs) {
      if (t.available_channels > 0) {
        t.available_channels -= 1;
        t.chan_names.push_back(chan_name);
        return air::getPhysTileOp(aie_device, t.shim_col, 0);
      }
    }
    auto shim_col = shim_columns[allocs->size()];
    auto shim_tile = air::getPhysTileOp(aie_device, shim_col, 0);
    allocs->push_back({shim_col, shim_dma_channels - 1, {chan_name}});

    return shim_tile;
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

AIE::BufferOp allocateBufferOp(uint64_t &BufferId, MemRefType memrefTy,
                               AIE::TileOp tile,
                               mlir::StringAttr attr = nullptr, int x = -1,
                               int y = -1) {

  OpBuilder builder(tile);
  Operation *t = tile.getOperation();
  while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
    t = t->getNextNode();
  builder.setInsertionPointAfter(t);
  AIE::BufferOp bufferOp = builder.create<AIE::BufferOp>(
      tile->getLoc(), memrefTy, tile, /*sym_name*/ nullptr,
      /*address*/ nullptr, /*initial_value*/ nullptr,
      /*mem_bank*/ nullptr);

  std::stringstream ss =
      air::generateBufferNameInStringStream("buf", BufferId, attr, x, y);
  bufferOp->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(tile->getContext(), ss.str()));

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

void outlineAIECores(OpBuilder &builder, AIE::DeviceOp aie_device,
                     air::HerdOp h,
                     std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
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

  // use the command line offsets unless the attribute is present
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

      // make the aie.tile
      auto tile = air::getPhysTileOp(aie_device, phys_x, phys_y);

      Operation *t = tile.getOperation();
      while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
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
        core = builder.create<AIE::CoreOp>(hloc, tile);
        tileToHerdMap[tile] = h;
        if (auto a = h->getAttrOfType<StringAttr>("link_with"))
          core->setAttr("link_with", a);
      }

      int64_t rtp_buffer_size = 0; // size in i32s
      for (unsigned ki = 0, ke = h.getNumKernelOperands(); ki < ke; ki++) {
        BlockArgument karg = h.getKernelArgument(ki);
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
        core_builder.create<cf::BranchOp>(hloc, core_bb);
        core_builder.setInsertionPointToEnd(core_bb);
      } else {
        // extending upon the existing bb chain
        for (auto &b : core.getBody().getBlocks())
          if (b.isEntryBlock())
            entry_bb = &b;
        Block *prev_bb_back = &core.getBody().back();
        auto prev_bb_branch =
            dyn_cast<cf::BranchOp>(prev_bb_back->getTerminator());
        auto prev_bb_end = dyn_cast<AIE::EndOp>(prev_bb_back->getTerminator());
        core_bb = core_builder.createBlock(&core.getBody());
        if (prev_bb_branch)
          prev_bb_branch.setDest(core_bb);
        else if (prev_bb_end) {
          core_builder.setInsertionPoint(prev_bb_end);
          core_builder.create<cf::BranchOp>(hloc, core_bb);
          prev_bb_end->erase();
        }
        core_builder.setInsertionPointToEnd(core_bb);
      }

      // map the tile ids and herd size to constants
      remap.map(h.getIds()[0],
                core_builder.create<arith::ConstantIndexOp>(hloc, x));
      remap.map(h.getIds()[1],
                core_builder.create<arith::ConstantIndexOp>(hloc, y));
      remap.map(h.getSize()[0],
                core_builder.create<arith::ConstantIndexOp>(hloc, herd_size_x));
      remap.map(h.getSize()[1],
                core_builder.create<arith::ConstantIndexOp>(hloc, herd_size_y));

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
        core_builder.create<AIE::UseLockOp>(core_builder.getUnknownLoc(),
                                            herd_lock, lockAction, lockValue);
      }

      for (unsigned ki = 0, ke = h.getNumKernelOperands(); ki < ke; ki++) {
        BlockArgument karg = h.getKernelArgument(ki);

        // Remap the kernel operands to the rtp buffer.
        // For each kernel operand of a supported type, load the data from the
        // rtp buffer and remap uses of the kernel operand to the loaded value.
        if (llvm::isa<IntegerType, IndexType, FloatType>(karg.getType())) {

          // load from rtp buffer
          SmallVector<Value> offsets{
              core_builder.create<arith::ConstantIndexOp>(hloc, ki)};
          auto load = core_builder.create<memref::LoadOp>(
              hloc, IntegerType::get(ctx, 32), rtp_buffer, offsets);

          // truncate, extend or bitcast the value to the correct type
          Value rtp = nullptr;
          llvm::TypeSwitch<Type>(karg.getType())
              .Case<IntegerType>([&](IntegerType ity) {
                unsigned int width = ity.getWidth();
                if (width < 32)
                  rtp = core_builder.create<arith::TruncIOp>(hloc, ity, load);
                else if (width > 32)
                  rtp = core_builder.create<arith::ExtUIOp>(hloc, ity, load);
                else
                  rtp = load;
              })
              .Case<IndexType>([&](IndexType ity) {
                rtp = core_builder.create<arith::IndexCastOp>(hloc, ity, load);
              })
              .Case<FloatType>([&](FloatType fty) {
                if (fty.getWidth() == 32) {
                  rtp = core_builder.create<arith::BitcastOp>(hloc, fty, load);
                } else if (fty.getWidth() == 16) {
                  auto ity = IntegerType::get(ctx, 16);
                  auto tr =
                      core_builder.create<arith::TruncIOp>(hloc, ity, load);
                  rtp = core_builder.create<arith::BitcastOp>(hloc, fty, tr);
                }
              });

          // remap the kernel operand
          if (rtp)
            remap.map(karg, rtp);
          else
            h.emitWarning("Unsupported runtime parmeter int or float type");
        }

        auto memrefTy = llvm::dyn_cast<MemRefType>(karg.getType());
        if (!memrefTy)
          continue;

        if (memrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L1) {
          // fused herds sometimes have L1 memref allocation outside of herds.
          // mapping them back
          remap.map(karg, h.getKernelOperand(ki));
          continue;
        }

        std::string sym_name = createSymbolName(aie_device, "__air_herd_arg");
        builder.create<memref::GlobalOp>(builder.getUnknownLoc(), sym_name,
                                         builder.getStringAttr("public"),
                                         memrefTy, nullptr, false, nullptr);

        auto m = core_builder.create<memref::GetGlobalOp>(
            hloc, SmallVector<Type, 1>{karg.getType()}, sym_name);
        remap.map(karg, m);
      }

      Region &r = h.getRegion();
      r.cloneInto(&core.getBody(), remap);

      Block *launch_bb = remap.lookup(&r.front());
      core_builder.create<cf::BranchOp>(hloc, launch_bb);
      core_builder.setInsertionPoint(launch_bb->getTerminator());
      if (options.emit_herd_lock) {
        if (aie_device.getTargetModel().hasProperty(
                AIE::AIETargetModel::UsesSemaphoreLocks)) {
          // we could release something, but we don't have to a way to observe
          // it yet in NPU
        } else {
          core_builder.create<AIE::UseLockOp>(core_builder.getUnknownLoc(),
                                              herd_lock,
                                              AIE::LockAction::Release, 0);
        }
      }

      if (options.emit_while) {
        auto entry_bb_br = dyn_cast<cf::BranchOp>(entry_bb->getTerminator());
        core_builder.create<cf::BranchOp>(hloc, entry_bb_br.getDest());
      } else
        core_builder.create<AIE::EndOp>(hloc);

      core.walk([&](Operation *op) {
        if (auto call = dyn_cast<func::CallOp>(op)) {
          auto fn = aie_device.lookupSymbol<func::FuncOp>(call.getCallee());
          if (!fn) {
            fn = func::FuncOp::create(aie_device.getLoc(), call.getCallee(),
                                      call.getCalleeType());
            fn.setPrivate();
            aie_device.insert(aie_device.getBody()->getTerminator(), fn);
          }
        }
      });

      // erase air.herd_termintor ops
      launch_bb->walk([&](air::HerdTerminatorOp op) { op->erase(); });
    }
  }
}

// Get all tile ops representing memtiles from device op.
std::vector<AIE::TileOp> getMemtilesFromDeviceOp(AIE::DeviceOp d) {
  std::vector<AIE::TileOp> memtiles;
  for (auto t : d.getOps<AIE::TileOp>()) {
    if (t.isMemTile()) {
      memtiles.push_back(t);
    }
  }
  return memtiles;
}

void outlineAIEMemtiles(OpBuilder &builder, AIE::DeviceOp aie_device,
                        air::SegmentOp seg,
                        AIRToAIEConversionOptions &options) {
  builder.setInsertionPointToStart(aie_device.getBody());

  int64_t seg_size_x = 1;
  if (auto num_cols = seg.getNumCols()) {
    seg_size_x = *num_cols;
  }

  seg.walk([&](air::ChannelInterface op) {
    if (!aie_device.lookupSymbol(op.getChanName())) {
      auto ch = air::getChannelDeclarationThroughSymbol(op);
      builder.clone(*ch.getOperation());
    }
  });

  // use the command line offsets unless the attribute is present
  int64_t col_offset = options.col_offset;

  for (auto x = 0; x < seg_size_x; x++) {
    // auto segloc = seg.getLoc();
    auto phys_x = x + col_offset;
    // TODO: Hard coded memtile row to be 1 here.
    auto phys_y = 1;

    // make the aie.tile
    AIE::TileOp tile = air::getPhysTileOp(aie_device, phys_x, phys_y);

    // Create a temporary buffer allocated to the memtile. This prevents
    // an unused memtile from being folded away before the L2 allocation pass.
    auto memrefTy =
        MemRefType::get(SmallVector<int64_t>{1}, builder.getI8Type());
    static uint64_t BufferId = 0;
    allocateBufferOp(BufferId, memrefTy, tile,
                     builder.getStringAttr("__L2_tmp"));
  }
}

template <typename T>
void push_back_if_unique(std::vector<T> &vec, T entry) {
  if (std::find(vec.begin(), vec.end(), entry) == vec.end())
    vec.push_back(entry);
}

void createAIEModulesAndOutlineCores(
    ModuleOp module,
    std::vector<std::pair<AIE::DeviceOp, air::HerdOp>> &aie_modules,
    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
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
    std::string segment_name;
    if (auto attr =
            seg->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      segment_name = attr.getValue().str();
    else
      segment_name = "segment_" + std::to_string(aie_modules.size());
    std::string aie_module_name = "aie." + segment_name;
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    auto aie_dev = builder.create<AIE::DeviceOp>(
        module.getLoc(),
        AIE::AIEDeviceAttr::get(builder.getContext(), options.device));
    aie_dev->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(builder.getContext(), segment_name));
    setAIEDeviceDataLayout(builder, aie_dev);
    AIE::DeviceOp::ensureTerminator(aie_dev.getRegion(), builder,
                                    aie_dev.getLoc());
    seg.walk([&](air::HerdOp h) { aie_modules.push_back({aie_dev, h}); });
    // If the device has memtiles, then outline memtiles
    if (aie_dev.getTargetModel().getNumMemTileRows()) {
      outlineAIEMemtiles(builder, aie_dev, seg, options);
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
    auto aie_dev = builder.create<AIE::DeviceOp>(
        module.getLoc(),
        AIE::AIEDeviceAttr::get(builder.getContext(), options.device));
    aie_dev->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(builder.getContext(), segment_name));
    setAIEDeviceDataLayout(builder, aie_dev);
    AIE::DeviceOp::ensureTerminator(aie_dev.getRegion(), builder,
                                    aie_dev.getLoc());
    aie_modules.push_back({aie_dev, herd});
  };
  for (auto &p : aie_modules) {
    auto aie_dev = std::get<0>(p);
    auto h = std::get<1>(p);
    OpBuilder builder(aie_dev);
    outlineAIECores(builder, aie_dev, h, tileToHerdMap, options);
  }
  // Outline any L1 memref allocs used by herds but located outside of any herd
  std::vector<Value> sharedL1Memrefs;
  for (auto &p : aie_modules) {
    // for (auto h : herds) {
    auto h = std::get<1>(p);
    for (unsigned i = 0; i < h.getNumKernelOperands(); i++) {
      auto oper = h.getKernelOperand(i);
      if (!oper.getDefiningOp())
        continue;
      auto memrefTy = llvm::dyn_cast<MemRefType>(oper.getType());
      if (!memrefTy)
        continue;
      if (memrefTy.getMemorySpaceAsInt() != (int)air::MemorySpace::L1)
        continue;
      push_back_if_unique<Value>(sharedL1Memrefs, oper);
    }
  }
  std::map<Operation *, llvm::SmallSet<AIE::CoreOp, 2>> sharedL1AllocsToCoreMap;
  for (auto memref : sharedL1Memrefs) {
    for (auto user : memref.getUsers()) {
      auto coreOp = user->getParentOfType<AIE::CoreOp>();
      if (!coreOp)
        continue;
      sharedL1AllocsToCoreMap[memref.getDefiningOp()].insert(coreOp);
    }
  }
  for (auto memref : sharedL1Memrefs) {
    for (auto coreOp : sharedL1AllocsToCoreMap[memref.getDefiningOp()]) {
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

bool isInSet(IntegerSet is) {
  auto constraints = is.getConstraints();
  auto eqFlags = is.getEqFlags();

  int i = 0;
  for (auto c : constraints) {
    auto expr = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(c, 0, 1));
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

    auto core = op->getParentOfType<AIE::CoreOp>();
    if (!core)
      return failure();

    bool in_set = false;
    if (op.getNumOperands() == 2) {
      SmallVector<int64_t, 2> operands;
      for (auto o : op.getOperands()) {
        if (auto v = dyn_cast<arith::ConstantIndexOp>(o.getDefiningOp()))
          operands.push_back(v.value());
        else if (auto v = dyn_cast<arith::RemSIOp>(o.getDefiningOp())) {
          if (mlir::getConstantIntValue(v.getLhs()) &&
              mlir::getConstantIntValue(v.getRhs())) {
            int lhs = *mlir::getConstantIntValue(v.getLhs());
            int rhs = *mlir::getConstantIntValue(v.getRhs());
            operands.push_back(llvm::mod(lhs, rhs));
          } else
            return failure();
        } else if (auto v = dyn_cast<arith::DivSIOp>(o.getDefiningOp())) {
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

void specializeHerdAffineIf(AIE::DeviceOp m) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<SpecializeAffineIfPattern>(ctx);
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
      rewriter.create<air::WaitAllOp>(op->getLoc(), Type{},
                                      op.getAsyncDependencies());
    }
    if (op.getNumResults() > 0) {
      rewriter.setInsertionPointAfter(op);
      auto w = rewriter.create<air::WaitAllOp>(
          op->getLoc(), air::AsyncTokenType::get(op->getContext()),
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
    auto new_fop = rewriter.create<scf::ForOp>(
        fop->getLoc(), fop.getLowerBound(), fop.getUpperBound(), fop.getStep(),
        iter_args);
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

    // use the new for op's results
    int idx = 0;
    for (auto r : fop.getResults()) {
      if (llvm::isa<air::AsyncTokenType>(r.getType()))
        r.replaceAllUsesWith(
            rewriter
                .create<air::WaitAllOp>(
                    fop->getLoc(), air::AsyncTokenType::get(fop->getContext()),
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
    rewriter.create<air::WaitAllOp>(fop->getLoc(), SmallVector<Type, 1>{},
                                    token_operands);
    rewriter.create<scf::YieldOp>(yield->getLoc(), yield_operands);
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

  AllocL1BuffersPattern(MLIRContext *ctx,
                        std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
                        uint64_t &bufferId)
      : OpRewritePattern(ctx), tileToHerdMap(tileToHerdMap),
        BufferId(bufferId) {}

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

    if (memrefTy.getMemorySpaceAsInt() != (int)air::MemorySpace::L1)
      return failure();

    auto herd = tileToHerdMap[tile];
    int64_t col_offset = 0;
    int64_t row_offset = 0;
    if (herd) {
      auto c = herd.getColOffset();
      auto r = herd.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    }

    auto buffer = allocateBufferOp(
        BufferId, memrefTy, tile,
        alloc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        tile.getCol() - col_offset, tile.getRow() - row_offset);

    rewriter.replaceOp(alloc, buffer->getResults());
    return success();
  }

private:
  std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap;
  uint64_t &BufferId;
};

struct AllocL2BuffersPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  AllocL2BuffersPattern(
      MLIRContext *ctx, std::map<memref::AllocOp, AIE::TileOp> &memrefToTileMap,
      std::map<AIE::BufferOp, AIE::TileOp> &bufferToMemtileMap,
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

    if (memrefTy.getMemorySpaceAsInt() != (int)air::MemorySpace::L2)
      return failure();

    // Allocation of L2 memrefs in segment to buffer ops
    if (!memrefToTileMap.count(alloc)) {
      alloc->emitOpError("alloc not found in memrefToTileMap.");
      return failure();
    }
    AIE::TileOp tile = memrefToTileMap[alloc];
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
    AIE::BufferOp buffer = allocateBufferOp(
        BufferId, memrefTy, tile,
        alloc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        tile.getCol() - col_offset, tile.getRow() - row_offset);

    rewriter.replaceOp(alloc, buffer->getResults());
    bufferToMemtileMap[buffer] = tile;
    return success();
  }

private:
  std::map<memref::AllocOp, AIE::TileOp> &memrefToTileMap;
  uint64_t &BufferId;
  std::map<AIE::BufferOp, AIE::TileOp> &bufferToMemtileMap;
};

void allocL1Buffers(AIE::DeviceOp m,
                    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
                    uint64_t &BufferId) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<AllocL1BuffersPattern>(ctx, tileToHerdMap, BufferId);
  // AllocL1TensorsPattern
  (void)applyPatternsGreedily(m, std::move(patterns));
}

bool areReferencedByTheSameAIRChannel(Value memref_a, Value memref_b) {
  for (auto user_a : memref_a.getUsers()) {
    for (auto user_b : memref_b.getUsers()) {
      auto chan_user_a = dyn_cast<air::ChannelInterface>(user_a);
      auto chan_user_b = dyn_cast<air::ChannelInterface>(user_b);
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
    std::map<memref::AllocOp, AIE::TileOp> &memrefToMemTileMap) {
  std::vector<memref::AllocOp> allocs;
  m.walk([&](memref::AllocOp alloc) {
    if (llvm::cast<MemRefType>(alloc.getMemref().getType())
            .getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      allocs.push_back(alloc);
    }
  });
  std::vector<AIE::TileOp> memtiles = getMemtilesFromDeviceOp(m);

  // Allocation of L2 memrefs in segment to (memtile) tile ops
  std::map<AIE::TileOp, uint32_t> memtileToSizeMap;
  for (auto t : memtiles) {
    memtileToSizeMap[t] = m.getTargetModel().getMemTileSize();
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
  // Second stage in memref placement: placing memref groups to memtiles.
  int memtile_id = 0;
  for (auto &bucket : memref_buckets) {
    for (auto bucket_elem : bucket) {
      MemRefType ty = llvm::cast<MemRefType>(bucket_elem.getMemref().getType());
      auto memref_vol =
          air::getElementSizeInBytes(ty) * air::getTensorVolume(ty);
      memtileToSizeMap[memtiles[memtile_id]] -= memref_vol;
      memrefToMemTileMap[bucket_elem] = memtiles[memtile_id];
    }
    memtile_id++;
    memtile_id %= memtiles.size();
  }
}

void allocL2Buffers(AIE::DeviceOp m,
                    std::map<AIE::BufferOp, AIE::TileOp> &bufferToMemtileMap,
                    uint64_t &BufferId) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  if (m.getTargetModel().getNumMemTileRows()) {
    std::map<memref::AllocOp, AIE::TileOp> memrefToTileMap;
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
      std::map<AIE::BufferOp, AIE::TileOp> &bufferToMemtileMap,
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

    channel->print(llvm::outs());
    llvm::outs() << "channelPuts" << channelPuts.size() << "\n";
    llvm::outs() << "channelGets" << channelGets.size() << "\n";

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
      int mem_space = memref.getMemorySpaceAsInt();
      if (mem_space == (int)air::MemorySpace::L2) {
        if (linksToComplete.find(channelPuts[0].getOperation()) !=
            linksToComplete.end()) {
          endOfLink = channelPuts[0].getOperation();
          linkFound = true;
        } else {
          AIE::BufferOp buff = dyn_cast<AIE::BufferOp>(
              channelPuts[0].getMemref().getDefiningOp());
          for (auto user : buff->getUsers()) {
            if (auto pairedGet = dyn_cast<air::ChannelGetOp>(user)) {
              endOfLink = pairedGet.getOperation();
              linkToComplete = true;
            }
          }
        }
      }
    } else {
      // put from L3
      producerTile = shimTileAlloc.getShimTile(
          device, (int)air::MemorySpace::L3, (int)air::MemorySpace::L1,
          channel.getName().str());
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
      int mem_space = memref.getMemorySpaceAsInt();
      if (mem_space == (int)air::MemorySpace::L2) {
        if (linksToComplete.find(get.getOperation()) != linksToComplete.end()) {
          endOfLink = get.getOperation();
          linkFound = true;
        } else {
          AIE::BufferOp buff =
              dyn_cast<AIE::BufferOp>(get.getMemref().getDefiningOp());
          for (auto user : buff->getUsers()) {
            if (auto pairedPut = dyn_cast<air::ChannelPutOp>(user)) {
              endOfLink = pairedPut.getOperation();
              linkToComplete = true;
            }
          }
        }
      }
    }
    for (int i = 0; i < expectedGets - (int)channelGets.size(); i++) {
      // get from L3
      consumerTile = shimTileAlloc.getShimTile(
          device, (int)air::MemorySpace::L1, (int)air::MemorySpace::L3,
          channel.getName().str());
      consumers.push_back(consumerTile);
    }

    if (!datatype)
      return failure();

    // create objFifo
    rewriter.setInsertionPoint(*(device.getOps<AIE::CoreOp>().begin()));
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
        rewriter.create<AIE::ObjectFifoLinkOp>(
            rewriter.getUnknownLoc(),
            rewriter.getArrayAttr({SymbolRefAttr::get(ctx, objFifo.name())}),
            rewriter.getArrayAttr(
                {SymbolRefAttr::get(ctx, producerFifo.name())}),
            rewriter.getArrayAttr({}), rewriter.getArrayAttr({}));
      else
        rewriter.create<AIE::ObjectFifoLinkOp>(
            rewriter.getUnknownLoc(),
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
          if (auto async_u = dyn_cast<air::AsyncOpInterface>(u))
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
          if (auto async_u = dyn_cast<air::AsyncOpInterface>(u))
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
    int mem_space = memref.getMemorySpaceAsInt();
    *datatype = AIE::AIEObjectFifoType::get(
        MemRefType::get(memref.getShape(), memref.getElementType()));
    if (mem_space == (int)air::MemorySpace::L1) {
      AIE::CoreOp core = op->template getParentOfType<AIE::CoreOp>();
      if (!core)
        return op.emitOpError("could not retrieve core for channel put/get op");
      *tile = core.getTileOp();
      return success();
    } else if (mem_space == (int)air::MemorySpace::L2) {
      if (bufferToMemtileMap.find(dyn_cast<AIE::BufferOp>(
              op.getMemref().getDefiningOp())) != bufferToMemtileMap.end()) {
        *tile = bufferToMemtileMap[dyn_cast<AIE::BufferOp>(
            op.getMemref().getDefiningOp())];
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
    AIE::ObjectFifoCreateOp fifo = builder.create<AIE::ObjectFifoCreateOp>(
        builder.getUnknownLoc(), builder.getStringAttr(name), prodTile,
        consTile, builder.getIntegerAttr(builder.getI32Type(), depth),
        datatype);
    return fifo;
  }

  template <typename MyOp>
  void
  rewriteChannelAllocs(PatternRewriter &rewriter, MyOp op,
                       AIE::ObjectFifoCreateOp objFifo,
                       AIE::ObjectFifoPort port,
                       llvm::SmallSet<Operation *, 2> &erased_allocs) const {
    BaseMemRefType memref = cast<BaseMemRefType>(op.getMemref().getType());
    int mem_space = memref.getMemorySpaceAsInt();
    if (mem_space == (int)air::MemorySpace::L2) {
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
        rewriter.create<AIE::ObjectFifoAcquireOp>(
            rewriter.getUnknownLoc(), acqType, port, objFifo.getName(), 1);
    rewriter.setInsertionPointAfter(producerAcq);
    AIE::ObjectFifoSubviewAccessOp producerAccess =
        rewriter.create<AIE::ObjectFifoSubviewAccessOp>(
            rewriter.getUnknownLoc(), elementType, producerAcq.getSubview(),
            rewriter.getIntegerAttr(rewriter.getI32Type(), 0));

    // replace uses of alloc with result of acquire
    if (auto a = dyn_cast<memref::AllocOp>(op.getMemref().getDefiningOp()))
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
    int mem_space = memref.getMemorySpaceAsInt();
    if (mem_space == (int)air::MemorySpace::L2) {
      return;
    }
    for (auto u : op.getMemref().getDefiningOp()->getUsers()) {
      if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
        rewriter.setInsertionPoint(&op->getBlock()->back());
        rewriter.create<AIE::ObjectFifoReleaseOp>(dealloc->getLoc(), port,
                                                  objFifo.getName(), 1);
        // Delete ops at the end of the rewrite pattern to avoid repeatedly
        // deleting the same op
        erased_deallocs.insert(dealloc.getOperation());
      }
    }
  }

  ShimTileAllocator &shimTileAlloc;
  std::map<AIE::BufferOp, AIE::TileOp> &bufferToMemtileMap;
  std::map<Operation *, AIE::ObjectFifoCreateOp> &linksToComplete;
};

// This function replaces ChannelPutOp/ChannelGetOp with AIE_CreateObjectFifoOps
// and with ObjectFifoAcquireOp<Producer/Consumer>. It also erases memref allocs
// as the objFifo lowering allocates its own memory. It replaces the associated
// memref deallocs with ObjectFifoReleaseOps.
void lowerAIRChannels(
    AIE::DeviceOp &d, ShimTileAllocator &s,
    std::map<AIE::BufferOp, AIE::TileOp> &bufferToMemtileMap) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  std::map<Operation *, AIE::ObjectFifoCreateOp> linksToComplete;
  patterns.insert<LowerAIRChannelsPattern>(ctx, s, bufferToMemtileMap,
                                           linksToComplete);
  (void)applyPatternsGreedily(d, std::move(patterns));
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
      auto new_chan = rewriter.create<air::ChannelOp>(
          channel->getLoc(), cname, rewriter.getI64ArrayAttr(channel_sizes),
          channel.getChannelType());
      if (channel->hasAttr("broadcast_shape")) {
        auto broadcast_shape = specializeBroadcastShape(rewriter, channel);
        new_chan->setAttr("broadcast_shape",
                          rewriter.getArrayAttr(ArrayRef(broadcast_shape)));
      }
      std::vector<unsigned> position =
          air::getMDVectorFromIterator(bundle_size_stdvec, iter);
      for (auto put : channelPuts) {
        auto indices_uint =
            air::convertVecOfConstIndexToVecOfUInt(put.getIndices());
        if (areIdenticalVectors(indices_uint, position)) {
          // Found channel put for this channel
          rewriter.setInsertionPoint(put);
          auto new_put = createChannelPutGetWithoutBundle(rewriter, new_chan,
                                                          put, maxSize);
          auto async_new_put =
              dyn_cast<air::AsyncOpInterface>(new_put.getOperation());
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
        if (areIdenticalVectors(indices_uint, position)) {
          // Found channel get for this channel
          rewriter.setInsertionPoint(get);
          auto new_get = createChannelPutGetWithoutBundle(rewriter, new_chan,
                                                          get, maxSize);
          auto async_new_get =
              dyn_cast<air::AsyncOpInterface>(new_get.getOperation());
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
    auto asyncOp = dyn_cast<air::AsyncOpInterface>(ci.getOperation());
    if (asyncOp.getAsyncToken()) {
      tys.push_back(air::AsyncTokenType::get(builder.getContext()));
      deps = asyncOp.getAsyncDependencies();
    }
    SmallVector<Value, 4> indices = {};
    // Canonicalize wrap and stride lists after specialization
    SmallVector<Value> offsets = ci.getOffsets();
    SmallVector<Value> wraps = ci.getSizes();
    SmallVector<Value> strides = ci.getStrides();
    (void)air::canonicalizeWrapAndStrideList(
        builder, offsets, wraps, strides,
        air::getTensorVolume(ci.getMemref().getType()), maxSize);
    air::ChannelInterface new_ci = nullptr;
    if (isa<air::ChannelPutOp>(ci))
      new_ci = builder.create<air::ChannelPutOp>(
          ci->getLoc(), tys, deps, chan.getSymName(), indices, ci.getMemref(),
          offsets, wraps, strides);
    else if (isa<air::ChannelGetOp>(ci))
      new_ci = builder.create<air::ChannelGetOp>(
          ci->getLoc(), tys, deps, chan.getSymName(), indices, ci.getMemref(),
          offsets, wraps, strides);
    new_ci->setAttrs(ci->getDiscardableAttrDictionary());
    return new_ci;
  }

  std::vector<Attribute> specializeBroadcastShape(OpBuilder builder,
                                                  air::ChannelOp chan) const {
    auto broadcast_shape = chan.getBroadcastShape();
    int diffDimension = chan.getBroadcastDimension();
    std::vector<Attribute> new_shape;
    for (int i = 0; i < (int)broadcast_shape.size(); i++) {
      if (i == diffDimension) {
        auto broadcast_dim = dyn_cast<IntegerAttr>(broadcast_shape[i]).getInt();
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
      if (auto get = dyn_cast<air::ChannelGetOp>(op)) {
        auto chan_op = air::getChannelDeclarationThroughSymbol(get);
        chan_op->setAttr(
            "buffer_resources",
            IntegerAttr::get(IntegerType::get(chan_op->getContext(), 32),
                             unroll_factor));
      } else if (auto put = dyn_cast<air::ChannelPutOp>(op)) {
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
            rewriter
                .create<air::WaitAllOp>(
                    op->getLoc(), air::AsyncTokenType::get(op->getContext()),
                    air::getAsyncDependenciesFromOp(op))
                .getAsyncToken());
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRToAIEPass : public air::impl::AIRToAIEBase<AIRToAIEPass> {

  uint64_t BufferId = 0;

public:
  AIRToAIEPass() = default;
  AIRToAIEPass(const AIRToAIEPass &pass) {}
  AIRToAIEPass(const air::AIRToAIEOptions &options) : AIRToAIEBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<airrt::AIRRtDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<xilinx::AIEX::AIEXDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<DLTIDialect>();
  }

  // Circuit-switched flow.
  AIE::FlowOp getFlowOp(AIE::DeviceOp aie_device, mlir::Value source,
                        xilinx::AIE::WireBundle sourceBundle,
                        uint32_t sourceChannel, mlir::Value dest,
                        xilinx::AIE::WireBundle destBundle,
                        uint32_t destChannel) {
    AIE::FlowOp flowOp = nullptr;
    aie_device.walk([&](Operation *op) {
      if (auto fop = dyn_cast<AIE::FlowOp>(op))
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
    return builder.create<AIE::FlowOp>(builder.getUnknownLoc(), source,
                                       sourceBundle, sourceChannel, dest,
                                       destBundle, destChannel);
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
  createPacketFlowOp(OpBuilder &builder, int &flowID, Value source,
                     xilinx::AIE::WireBundle sourceBundle,
                     uint32_t sourceChannel, Value dest,
                     xilinx::AIE::WireBundle destBundle, uint32_t destChannel,
                     mlir::BoolAttr keep_pkt_header = nullptr) {
    AIE::PacketFlowOp pktFlow = builder.create<AIE::PacketFlowOp>(
        builder.getUnknownLoc(), flowID++, keep_pkt_header, nullptr);
    Region &r_pktFlow = pktFlow.getPorts();
    Block *b_pktFlow = builder.createBlock(&r_pktFlow);
    builder.setInsertionPointToStart(b_pktFlow);
    builder.create<AIE::PacketSourceOp>(builder.getUnknownLoc(), source,
                                        sourceBundle, sourceChannel);
    builder.create<AIE::PacketDestOp>(builder.getUnknownLoc(), dest, destBundle,
                                      destChannel);
    builder.create<AIE::EndOp>(builder.getUnknownLoc());
    return pktFlow;
  }

  // This method generates broadcast packet flow if found multiple flows with
  // the same source. TODO: packet flows sharing source do not always mean
  // broadcast.
  AIE::PacketFlowOp getPacketFlowOp(AIE::DeviceOp aie_device,
                                    mlir::Value source,
                                    xilinx::AIE::WireBundle sourceBundle,
                                    uint32_t sourceChannel, mlir::Value dest,
                                    xilinx::AIE::WireBundle destBundle,
                                    uint32_t destChannel, int &flowID) {
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
      builder.create<AIE::PacketDestOp>(builder.getUnknownLoc(), dest,
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
    auto chanIfOp = dyn_cast<air::ChannelInterface>(memcpyOp.getOperation());
    if (!chanIfOp)
      return AIE::PacketFlowOp(); // Only air.channel_interface ops support
                                  // packet-flow routing.

    // Convert flowOpToFlowIdMap from Operation pointers to channel symbol
    // names. This is necessary because air.channel declarations are duplicated
    // under aie.device op and its parent module op, requiring symbol-based
    // matching.
    std::vector<std::string> flowOpStringsToFlowIdMap;
    for (auto op : flowOpToFlowIdMap) {
      auto flowChanOp = dyn_cast<air::ChannelOp>(op);
      if (!flowChanOp) {
        flowOpStringsToFlowIdMap.push_back("");
        continue;
      }
      flowOpStringsToFlowIdMap.push_back(flowChanOp.getSymName().str());
    }

    // Find the flowID by matching the channel name
    auto it =
        llvm::find(flowOpStringsToFlowIdMap, chanIfOp.getChanName().str());
    if (it == flowOpStringsToFlowIdMap.end()) {
      return AIE::PacketFlowOp();
    }
    int flowID = std::distance(flowOpStringsToFlowIdMap.begin(), it);

    // Search for the packet flow with matching source and flowID
    return findPacketFlowOp(source, sourceBundle, sourceChannel,
                            /*checkFlowID=*/true, flowID);
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
      if (auto fop = dyn_cast<AIE::CascadeFlowOp>(op))
        if (source == fop.getSourceTile() && dest == fop.getDestTile())
          flowOp = fop;
    });

    // If found, return the existing flow.
    if (flowOp)
      return flowOp;

    // Otherwise, create a new CascadeFlowOp at the end of the device body.
    OpBuilder builder(aie_device);
    builder.setInsertionPoint(aie_device.getBody()->getTerminator());
    return builder.create<AIE::CascadeFlowOp>(builder.getUnknownLoc(), source,
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
    // Map from L2 memref -> (list of puts, list of gets)
    llvm::DenseMap<Value, std::pair<llvm::SmallVector<air::ChannelPutOp>,
                                    llvm::SmallVector<air::ChannelGetOp>>>
        l2MemrefPutsGets;

    // Walk all ChannelInterface ops under the device and categorize puts/gets
    // on L2 memrefs
    aieDevice.walk<mlir::WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [&](air::ChannelInterface chanI) {
          auto memrefTy = dyn_cast<BaseMemRefType>(chanI.getMemref().getType());
          if (!memrefTy || memrefTy.getMemorySpaceAsInt() !=
                               static_cast<int>(air::MemorySpace::L2))
            return mlir::WalkResult::advance();

          if (auto chanPut = dyn_cast<air::ChannelPutOp>(chanI.getOperation()))
            l2MemrefPutsGets[chanI.getMemref()].first.push_back(chanPut);
          else if (auto chanGet =
                       dyn_cast<air::ChannelGetOp>(chanI.getOperation()))
            l2MemrefPutsGets[chanI.getMemref()].second.push_back(chanGet);

          return mlir::WalkResult::advance();
        });

    // Balance puts and gets by inserting dummy ops
    for (auto &[memref, putsAndGets] : l2MemrefPutsGets) {
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
          builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
      Value oneIdx =
          builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);

      // Use the original op as a template to emit new dummy ops
      auto templateAsyncIf = dyn_cast<air::AsyncOpInterface>(templateOp);
      auto templateChanIf = dyn_cast<air::ChannelInterface>(templateOp);
      assert(templateAsyncIf && templateChanIf &&
             "Expected valid async/channel op");

      for (unsigned i = 0; i < numOpsToClone; ++i) {
        if (isa<air::ChannelPutOp>(templateOp)) {
          builder.create<air::ChannelPutOp>(
              templateOp->getLoc(), templateOp->getResultTypes(),
              templateAsyncIf.getAsyncDependencies(),
              templateChanIf.getChanName(), templateChanIf.getIndices(),
              templateChanIf.getMemref(),
              /*sizes*/ SmallVector<Value>{zeroIdx},
              /*offsets*/ SmallVector<Value>{zeroIdx},
              /*steps*/ SmallVector<Value>{oneIdx});
        } else if (isa<air::ChannelGetOp>(templateOp)) {
          builder.create<air::ChannelGetOp>(
              templateOp->getLoc(), templateOp->getResultTypes(),
              templateAsyncIf.getAsyncDependencies(),
              templateChanIf.getChanName(), templateChanIf.getIndices(),
              templateChanIf.getMemref(),
              /*sizes*/ SmallVector<Value>{zeroIdx},
              /*offsets*/ SmallVector<Value>{zeroIdx},
              /*steps*/ SmallVector<Value>{oneIdx});
        }
      }
    }
  }

  // Clone data movement ops to and from memtile and shim tile DMAs
  void cloneL2AndL3MemcpysToDeviceOp(OpBuilder &builder,
                                     AIE::DeviceOp aie_device, ModuleOp module,
                                     bool clone_l2, bool clone_l3,
                                     bool lock_race_condition_fix = true) {

    if (!clone_l2 && !clone_l3)
      return;

    auto ctx = builder.getContext();

    Operation *t = nullptr;
    for (auto tile_op : aie_device.getBody()->getOps<AIE::TileOp>()) {
      t = tile_op.getOperation();
    }
    builder.setInsertionPointAfter(t);
    IRMapping remap;

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
                    func::ReturnOp>(op))
              return WalkResult::advance();
            bool hasParentSegmentOp = op->getParentOfType<air::SegmentOp>();
            if (!clone_l3 && !hasParentSegmentOp)
              return WalkResult::advance();
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
              b.create<air::WaitAllOp>(hierOp->getLoc(),
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
              oper, b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0));
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
      int firstOffset = *getConstantIntValue(op.getOffsets().front());
      push_back_if_unique<int>(keys, firstOffset);
      if (!chanOpPartitions.count(firstOffset))
        chanOpPartitions[firstOffset] = SmallVector<air::ChannelInterface>{op};
      else
        chanOpPartitions[firstOffset].push_back(op);
    }
    for (auto op : gets) {
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
      Value newMemref = builder.create<memref::AllocOp>(
          loc,
          MemRefType::get(newMemrefShape, ty.getElementType(),
                          ty.getLayout().getAffineMap(), ty.getMemorySpace()));
      for (auto op : chanOpPartitions[key]) {
        int memrefOperandOffset =
            dyn_cast<air::AsyncOpInterface>(op.getOperation())
                .getAsyncDependencies()
                .size();
        auto &memrefOpOper = op->getOpOperand(memrefOperandOffset);
        memrefOpOper.assign(newMemref);
        int firstOffsetOperandOffset = memrefOperandOffset + 1;
        auto &firstOffsetOpOper = op->getOpOperand(firstOffsetOperandOffset);
        firstOffsetOpOper.assign(
            builder.create<arith::ConstantIndexOp>(loc, 0));
        // Update strides (contiguous, row-major) after memref tiling.
        SmallVector<Value> offsets;
        SmallVector<Value> wraps;
        SmallVector<Value> strides;
        // One dimensional default stride value.
        if (op.getSizes().size() == 1)
          strides.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));
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
    std::vector<AIE::TileOp> memtiles = getMemtilesFromDeviceOp(d);
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
      if (memrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
        // Count the number of unique incoming and outgoing channels.
        std::vector<std::string> uniqueS2MMChannels;
        std::vector<std::string> uniqueMM2SChannels;
        for (auto user : memref.getUsers()) {
          if (auto get = dyn_cast<air::ChannelGetOp>(user))
            push_back_if_unique<std::string>(uniqueS2MMChannels,
                                             get.getChanName().str());
          else if (auto put = dyn_cast<air::ChannelPutOp>(user))
            push_back_if_unique<std::string>(uniqueMM2SChannels,
                                             put.getChanName().str());
        }
        bool tooManyChannelConnections =
            (int)uniqueS2MMChannels.size() > maxMemtileDstConnections ||
            (int)uniqueMM2SChannels.size() > maxMemtileSrcConnections;
        if (tooManyChannelConnections) {
          if (auto exec = dyn_cast<air::ExecuteOp>(allocOp->getParentOp()))
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
        if (auto put = dyn_cast<air::ChannelPutOp>(user))
          puts.push_back(put);
        else if (auto get = dyn_cast<air::ChannelGetOp>(user))
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
      if (auto dma = dyn_cast<air::DmaMemcpyNdOp>(o)) {
        // DMA memcpy always creates a new flow bundle.
        air::MemcpyBundleAsFlow flow = air::MemcpyBundleAsFlow(dma);
        if (failed(flow.pushBackMemcpyOpToBundle(dma)))
          return failure();
        memcpy_flows.push_back(flow);
      } else if (auto putget = dyn_cast<air::ChannelInterface>(o)) {
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
          auto air_flow_op_chan = dyn_cast<air::ChannelOp>(f.air_flow_op);
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

    // Step 2: Allocate tile DMA channels, shim DMA channels and shim tiles
    auto r = simpleDMAChannelAllocation(memcpy_flows, shim_dma_alloc,
                                        memtile_dma_alloc, tile_dma_alloc,
                                        core_cascade_alloc);
    if (failed(r))
      return r;

    // Step 3: Sort all ops being allocated to each DMA channel, to avoid
    // ping-pong deadlock.
    tile_dma_alloc.sortMemcpyOps(dma_memcpy_ops);

    // Step 4: Connect flows
    for (auto &f : memcpy_flows) {
      for (int i = 0; i < f.numS2MMAllocs; i++) {
        if (options.use_packet_flow_at_shim_dmas &&
            f.MM2S_alloc.getDmaTile().isShimNOCorPLTile()) {
          // use_packet_flow_at_shim_dmas mode: use packet flow for all shim dma
          // mm2s, to enable dma channel sharing with control packets
          flowOpToFlowIdMap.insert(f.air_flow_op);
          auto it = llvm::find(flowOpToFlowIdMap, f.air_flow_op);
          int flowID = std::distance(flowOpToFlowIdMap.begin(), it);
          getPacketFlowOp(
              aie_device, f.MM2S_alloc.getDmaTile(), AIE::WireBundle::DMA,
              (uint32_t)f.MM2S_alloc.dma_channel.channel,
              f.S2MM_alloc[i].getDmaTile(), AIE::WireBundle::DMA,
              (uint32_t)f.S2MM_alloc[i].dma_channel.channel, flowID);
          // Update global packet flow ID following the local packet assignment.
          globalFlowID = std::max(globalFlowID, flowID);
        } else if (f.memcpyResourceType == "dma_packet") {
          flowOpToFlowIdMap.insert(f.air_flow_op);
          auto it = llvm::find(flowOpToFlowIdMap, f.air_flow_op);
          int flowID = std::distance(flowOpToFlowIdMap.begin(), it);
          getPacketFlowOp(
              aie_device, f.MM2S_alloc.getDmaTile(), AIE::WireBundle::DMA,
              (uint32_t)f.MM2S_alloc.dma_channel.channel,
              f.S2MM_alloc[i].getDmaTile(), AIE::WireBundle::DMA,
              (uint32_t)f.S2MM_alloc[i].dma_channel.channel, flowID);
          // Update global packet flow ID following the local packet assignment.
          globalFlowID = std::max(globalFlowID, flowID);
        } else if (f.memcpyResourceType == "dma_stream")
          getFlowOp(aie_device, f.MM2S_alloc.getDmaTile(), AIE::WireBundle::DMA,
                    (uint32_t)f.MM2S_alloc.dma_channel.channel,
                    f.S2MM_alloc[i].getDmaTile(), AIE::WireBundle::DMA,
                    (uint32_t)f.S2MM_alloc[i].dma_channel.channel);
        else if (f.memcpyResourceType == "cascade") {
          getCascadeFlowOp(aie_device, f.MM2S_alloc.getDmaTile(),
                           AIE::WireBundle::DMA,
                           (uint32_t)f.MM2S_alloc.dma_channel.channel,
                           f.S2MM_alloc[i].getDmaTile(), AIE::WireBundle::DMA,
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
    if (auto herd = dyn_cast<air::HerdOp>(op.getOperation())) {
      auto c = herd.getColOffset();
      auto r = herd.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    } else if (auto seg = dyn_cast<air::SegmentOp>(op.getOperation())) {
      auto c = seg.getColOffset();
      auto r = seg.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    } else {
      return; // failure();
    }

    for (auto &t : allocs) {
      AIE::TileOp tileOp = t.getDmaTile();
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
        attrs.push_back(
            NamedAttribute(StringAttr::get(ctx, "location"),
                           builder.getI64IntegerAttr(tileOp.getCol())));
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
            auto internalBlockArg = dyn_cast<BlockArgument>(internalIdx);
            auto externalBlockArg = dyn_cast<BlockArgument>(externalIdx);
            SmallVector<int> internalSteps, externalSteps;
            if (internalBlockArg) {
              if (LoopLikeOpInterface loopLikeOwner =
                      dyn_cast<LoopLikeOpInterface>(
                          internalBlockArg.getOwner()->getParentOp()))
                internalSteps = getAllStaticStepsInLoopLike(loopLikeOwner,
                                                            internalBlockArg);
              else if (air::HerdOp herdOwner = dyn_cast<air::HerdOp>(
                           internalBlockArg.getOwner()->getParentOp()))
                internalSteps =
                    getAllStaticStepsInAIRHerd(herdOwner, internalBlockArg);
            } else if (constInternalIdx)
              internalSteps.push_back(*constInternalIdx);
            if (externalBlockArg) {
              if (LoopLikeOpInterface loopLikeOwner =
                      dyn_cast<LoopLikeOpInterface>(
                          externalBlockArg.getOwner()->getParentOp()))
                externalSteps = getAllStaticStepsInLoopLike(loopLikeOwner,
                                                            externalBlockArg);
              else if (air::HerdOp herdOwner = dyn_cast<air::HerdOp>(
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
  // information, specifically for MM2S (host-to-AIE) directions.
  LogicalResult labelMemcpyOpsWithPacketFlow(air::MemcpyInterface memcpyOpIf,
                                             StringAttr dmaNameAttr,
                                             AIE::TileOp tileOp, int channel) {
    auto pktFlowOp = getExistingPacketFlowOpFromRuntime(
        tileOp, AIE::WireBundle::DMA, channel);
    if (!pktFlowOp)
      return success();

    // If memcpy op is air.channel: filter out channel bundles based on
    // metadata; get metadata from metadataArray based on channel indices.
    if (auto ci = dyn_cast<air::ChannelInterface>(memcpyOpIf.getOperation())) {
      // Get index to metadataArray based on channel indices.
      auto iter = air::getIndexToMetadataArrayFromChannelIndices(ci);
      if (!iter) {
        ci->emitOpError("channel indices failed to convert to convert to "
                        "metadataArray index.");
        return failure();
      }
      // Get metadata from metadataArray.
      auto metadataArray = ci->getAttrOfType<ArrayAttr>("metadataArray");
      if (!metadataArray || (size_t)*iter >= metadataArray.size())
        return success();
      auto dictAttr = dyn_cast<DictionaryAttr>(metadataArray[*iter]);
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
    if (auto dma = dyn_cast<air::DmaMemcpyNdOp>(dmaOp.getOperation()))
      return cast<MemRefType>((dir == AIE::DMAChannelDir::MM2S)
                                  ? dma.getDstMemref().getType()
                                  : dma.getSrcMemref().getType());
    if (auto chan = dyn_cast<air::ChannelInterface>(dmaOp.getOperation())) {
      air::ChannelInterface tileSideChannelOp =
          air::getTheOtherChannelOpThroughSymbol(chan).front();
      return cast<MemRefType>(tileSideChannelOp.getMemref().getType());
    }
    return nullptr;
  }

  // Create shim DMA allocation ops and annotate the corresponding memcpy
  // operations with symbolic metadata.
  LogicalResult createShimDMAAllocationOps(
      OpBuilder builder, MLIRContext *ctx,
      std::vector<air::MemcpyInterface> shimSideMemcpyIfOps,
      air::ShimDMAAllocator &shimDmaAllocs,
      std::map<int, int> chanRenumberReverseMap) {
    std::vector<air::MemcpyInterface> shimMemcpyS2MMOps, shimMemcpyMM2SOps;

    // Separate memcpy ops into S2MM and MM2S based on direction.
    for (auto memcpyIf : shimSideMemcpyIfOps) {
      if (auto put = dyn_cast<air::ChannelPutOp>(memcpyIf.getOperation()))
        shimMemcpyMM2SOps.push_back(memcpyIf);
      if (auto get = dyn_cast<air::ChannelGetOp>(memcpyIf.getOperation()))
        shimMemcpyS2MMOps.push_back(memcpyIf);
      if (auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(memcpyIf.getOperation())) {
        auto srcMemrefTy =
            dyn_cast<BaseMemRefType>(dmaOp.getSrcMemref().getType());
        auto dstMemrefTy =
            dyn_cast<BaseMemRefType>(dmaOp.getDstMemref().getType());
        if (srcMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
          shimMemcpyMM2SOps.push_back(memcpyIf);
        if (dstMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
          shimMemcpyS2MMOps.push_back(memcpyIf);
      }
    }

    // Create shim-side S2MM DMA allocs and annotate corresponding ops.
    if (failed(createShimDMAAllocationOpsImpl(
            builder, ctx, shimMemcpyS2MMOps, shimDmaAllocs.s2mm_allocs,
            AIE::DMAChannelDir::S2MM, chanRenumberReverseMap))) {
      return failure();
    }

    // Create shim-side MM2S DMA allocs and annotate corresponding ops.
    if (failed(createShimDMAAllocationOpsImpl(
            builder, ctx, shimMemcpyMM2SOps, shimDmaAllocs.mm2s_allocs,
            AIE::DMAChannelDir::MM2S, chanRenumberReverseMap))) {
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
      std::map<int, int> chanRenumberReverseMap) {

    // Helper function getting dma_name from the air::MemcpyInterface op.
    auto getDmaNameFromMemcpyIfOp = [](air::MemcpyInterface memcpyIfOp) {
      std::string dma_name = "";
      if (auto ci = dyn_cast<air::ChannelInterface>(memcpyIfOp.getOperation()))
        return "air_" + ci.getChanName().str();
      else if (auto dmaOp =
                   dyn_cast<air::DmaMemcpyNdOp>(memcpyIfOp.getOperation()))
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
            if (auto ci = dyn_cast<air::ChannelInterface>(
                    memcpyIfOp.getOperation())) {
              for (auto tileSideChannelOp :
                   air::getTheOtherChannelOpThroughSymbol(ci)) {
                auto linkedTileSideDmaIds =
                    getOriginalTileSideDmaIds(t, chanRenumberReverseMap);
                if (llvm::is_contained(linkedTileSideDmaIds,
                                       tileSideChannelOp.getId()))
                  return true;
              }
            } else if (auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(
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

    // Capture errors when any shim memcpy op fails to link to shim allocation.
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

    // Create shim dma allocation ops.
    for (auto memcpyIfOp : shimSideMemcpyIfOps) {
      std::string dma_name = getDmaNameFromMemcpyIfOp(memcpyIfOp);
      int t_idx = 0;
      for (air::allocation_info_t &t : shimChanSymbolToAlloc[dma_name]) {
        auto deviceOp = t.getDmaTile()->getParentOfType<AIE::DeviceOp>();
        // Create shim allocation symbol name shim_name_attr.
        std::string shim_name = dma_name;
        if (shimChanSymbolToAlloc[dma_name].size() > 1)
          shim_name += "_" + std::to_string(t_idx);
        StringAttr shim_name_attr = builder.getStringAttr(shim_name);

        // Create shim allocation op.
        if (!SymbolTable::lookupSymbolIn(deviceOp, shim_name)) {
          auto shimAllocationOp = builder.create<AIE::ShimDMAAllocationOp>(
              builder.getUnknownLoc(), shim_name_attr,
              AIE::DMAChannelDirAttr::get(ctx, dir),
              builder.getI64IntegerAttr(t.dma_channel.channel),
              builder.getI64IntegerAttr(t.getDmaTile().getCol()),
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
          if (failed(labelMemcpyOpsWithPacketFlow(memcpyIfOp, shim_name_attr,
                                                  t.getDmaTile(),
                                                  t.dma_channel.channel)))
            return failure();
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
    auto segment_meta = builder.create<airrt::SegmentMetadataOp>(loc, name);
    builder.createBlock(&segment_meta.getHerds());
    builder.create<airrt::SegmentMetadataTerminatorOp>(loc);

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

    auto herd_meta = builder.create<airrt::HerdMetadataOp>(loc, name);
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
      std::unordered_set<Operation *> &allocs_to_remap,
      const AIE::AIETargetModel &targetModel,
      air::TileDMAAllocator &tileDmaAlloc, int x, int y) {
    bool UsesSemaphoreLocks =
        targetModel.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks);
    auto dma_alloc = tileDmaAlloc.lookupDMAAllocation(x, y, memcpyOpIf);
    if (failed(dma_alloc)) {
      return memcpyOpIf->emitOpError("failed to look up dma allocation.");
    }
    auto tile_channel = dma_alloc.value().dma_channel;
    auto bufferOp = tileDmaAlloc.getBuffer(BufferId, x, y, memcpyOpIf);
    if (failed(bufferOp)) {
      return memcpyOpIf->emitOpError("failed to get buffer.");
    }
    auto locks = tileDmaAlloc.getLockForDMA(memcpyOpIf, x, y,
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
    auto tileInbound = isTileInbound(memcpyOpIf, (int)air::MemorySpace::L1);
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

    if (auto bco = dyn_cast<bufferization::ToBufferOp>(alloc.getDefiningOp()))
      builder.setInsertionPoint(bco.getOperand().getDefiningOp());
    else if (isa<memref::AllocaOp>(alloc.getDefiningOp()))
      builder.setInsertionPoint(alloc.getDefiningOp());
    else if (!tileInbound.value() &&
             isa<AIE::BufferOp>(alloc.getDefiningOp())) {
      auto br = dyn_cast<cf::BranchOp>(memcpyOpIf->getBlock()->getTerminator());
      if (br)
        builder.setInsertionPointToStart(br.getDest());
      else
        builder.setInsertionPointToStart(memcpyOpIf->getBlock());
    } else
      builder.setInsertionPoint(memcpyOpIf);

    builder.create<AIE::UseLockOp>(memcpyOpIf->getLoc(), acqLockOp,
                                   UsesSemaphoreLocks
                                       ? AIE::LockAction::AcquireGreaterEqual
                                       : AIE::LockAction::Acquire,
                                   lockAqValue);

    // Try to find the end of lifetime for the data copied by memcpyOpIf, and
    // put the unlock.
    if (auto nextWriter = findNextDmaWriteOp(memcpyOpIf, alloc)) {
      // Lifetime ends if dma writes into the same buffer.
      builder.setInsertionPoint(nextWriter);
      builder.create<AIE::UseLockOp>(nextWriter->getLoc(), relLockOp,
                                     AIE::LockAction::Release, lockRelValue);
    } else if (auto lastAccessOp = findLastReadOrWriteOp(memcpyOpIf, alloc)) {
      // Lifetime ends after the last read/write access to buffer.
      builder.setInsertionPointAfter(lastAccessOp);
      builder.create<AIE::UseLockOp>(lastAccessOp->getLoc(), relLockOp,
                                     AIE::LockAction::Release, lockRelValue);
    } else {
      // Lifetime ends at end of block.
      auto t = memcpyOpIf->getBlock()->getTerminator();
      builder.setInsertionPoint(t);
      builder.create<AIE::UseLockOp>(t->getLoc(), relLockOp,
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
                       int x, int y, bool lockRaceConditionFix = false) {

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
          auto chanIf = dyn_cast<air::ChannelInterface>(op);
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
        end_bb_builder.create<AIE::EndOp>(loc);

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
              b.create<AIE::NextBDOp>(loc, first_bd);
            else
              b.create<AIE::NextBDOp>(loc, end_bb);
          } else {
            next_bd = new Block();
            next_bd->insertBefore(end_bb);
            b.create<AIE::NextBDOp>(loc, next_bd);
          }
          auto bufferOp = dmaAlloc.getBuffer(BufferId, x, y, memcpyOp);
          if (failed(bufferOp)) {
            memcpyOp->emitOpError("failed to get buffer.");
            return failure();
          }
          auto locks = dmaAlloc.getLockForDMA(memcpyOp, x, y,
                                              bufferOp.value().getOperation(),
                                              lockRaceConditionFix);
          if (failed(locks))
            return memcpyOp->emitOpError("failed to get lock for dma.");
          auto newBD = generateDmaBd<bufferOpTy>(loc, dir, locks.value(), x, y,
                                                 targetModel, bd, memcpyOp,
                                                 bufferOp.value(), chan);
          // Attribute task_id is necessary to ensure that BDs do not get shared
          // across tasks, otherwise MLIR may fold BDs and cause BD sharing
          // across tasks.
          if (failed(newBD))
            return bufferOp.value()->emitOpError("failed to generate dma bd.");
          newBD.value()->setAttr(
              "task_id",
              IntegerAttr::get(IntegerType::get(b.getContext(), 32), taskId));
        }

        AIE::DMAStartOp startOp = nullptr;
        if (infiniteBDLoopMode)
          rep = 0;
        if (!channel_head) {
          channel_head = start_bb;
          auto b = OpBuilder::atBlockBegin(channel_head);
          startOp =
              b.create<AIE::DMAStartOp>(loc, dir, chan, rep, first_bd, end_bb);
        } else {
          auto b = OpBuilder::atBlockBegin(start_bb);
          startOp = b.create<AIE::DMAStartOp>(
              loc, dir, chan, rep, first_bd,
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
                std::pair<AIE::LockOp, AIE::LockOp> locks, int x, int y,
                const AIE::AIETargetModel &targetModel, Block *bd,
                air::MemcpyInterface memcpyOp, bufferOpTy bufferOp, int chan) {
    bool UsesSemaphoreLocks =
        targetModel.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks);
    bool isMM2S = (dir == AIE::DMAChannelDir::MM2S);

    auto b = OpBuilder::atBlockEnd(bd);
    auto acqLockOp = isMM2S ? locks.first : locks.second;
    auto relLockOp = isMM2S ? locks.second : locks.first;
    b.setInsertionPointToStart(bd);
    int64_t lockAqValue = -1;
    int64_t lockRelValue = -1;
    auto aie2LockVal =
        air::getLockValuePair(targetModel, bufferOp->getResult(0));
    if (!isMM2S) {
      lockAqValue = UsesSemaphoreLocks ? aie2LockVal.first : 0;
      lockRelValue = UsesSemaphoreLocks ? aie2LockVal.first : 1;
    } else {
      lockAqValue = UsesSemaphoreLocks ? aie2LockVal.second : 1;
      lockRelValue = UsesSemaphoreLocks ? aie2LockVal.second : 0;
    }
    auto ndcpy = cast<air::MemcpyInterface>(memcpyOp);

    if (failed(isTileInbound(ndcpy, (int)air::MemorySpace::L1)))
      return failure();

    Value memref = isTileInbound(ndcpy, (int)air::MemorySpace::L1).value()
                       ? ndcpy.getDstMemref()
                       : ndcpy.getSrcMemref();
    SmallVector<Value> sizes =
        isTileInbound(ndcpy, (int)air::MemorySpace::L1).value()
            ? ndcpy.getDstSizes()
            : ndcpy.getSrcSizes();
    SmallVector<Value> offsets =
        isTileInbound(ndcpy, (int)air::MemorySpace::L1).value()
            ? ndcpy.getDstOffsets()
            : ndcpy.getSrcOffsets();
    SmallVector<Value> strides =
        isTileInbound(ndcpy, (int)air::MemorySpace::L1).value()
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
        b.create<arith::ConstantIndexOp>(memcpyOp.getLoc(), len)->getResult(0);
    b.create<AIE::UseLockOp>(loc, acqLockOp,
                             UsesSemaphoreLocks
                                 ? AIE::LockAction::AcquireGreaterEqual
                                 : AIE::LockAction::Acquire,
                             lockAqValue);

    // Packet flow routing: get packet flow id.
    auto aie_device = bufferOp->template getParentOfType<AIE::DeviceOp>();
    auto tileOp = air::getPhysTileOpOrNull(aie_device, x, y);
    auto pktFlowOp = getExistingPacketFlowOpFromDevice(
        tileOp, AIE::WireBundle::DMA, chan, memcpyOp);
    AIE::PacketInfoAttr pktInfoAttr = nullptr;
    if (isMM2S && pktFlowOp) {
      auto packetID = pktFlowOp.getID();
      pktInfoAttr = AIE::PacketInfoAttr::get(ndcpy->getContext(), 0, packetID);
    }

    std::vector<AIE::BDDimLayoutAttr> dims =
        air::getWrapsAndStrides(sizes, strides, ndcpy->getContext());
    auto wraps_and_strides =
        AIE::BDDimLayoutArrayAttr::get(ndcpy->getContext(), ArrayRef(dims));
    bool useDefaultDataAccessPattern =
        UsesSemaphoreLocks ? air::isDefaultDataAccessPattern(sizes, strides)
                           : true;
    AIE::DMABDOp aieDmaBdOp = nullptr;
    if (wraps_and_strides.getValue().empty() || useDefaultDataAccessPattern)
      aieDmaBdOp = b.create<AIE::DMABDOp>(
          loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value());
    else
      aieDmaBdOp = b.create<AIE::DMABDOp>(
          loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value(),
          wraps_and_strides);
    if (pktInfoAttr)
      aieDmaBdOp->setAttr("packet", pktInfoAttr);
    b.create<AIE::UseLockOp>(loc, relLockOp, AIE::LockAction::Release,
                             lockRelValue);
    return aieDmaBdOp;
  }

  // Converts an air.channel.put/get operation with channel_type = "cascade"
  // into aie.get/put_cascade + vector.transfer_read/write sequence.
  // The conversion flattens the entire memref into a 1-D vector to match
  // the cascade data format expected by the AIE put/get_cascade ops.
  LogicalResult ConvertCascadeChannelIfToAIE(RewriterBase &rewriter,
                                             air::ChannelInterface op) {
    // Match only if the associated channel has channel_type = "cascade".
    auto chan = air::getChannelDeclarationThroughSymbol(op);
    if (!chan)
      return op->emitOpError("cannot resolve channel symbol");

    if (chan.getChannelType().str() != "cascade")
      return op->emitOpError("channel_type is not cascade");

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
          rewriter.create<AIE::GetCascadeOp>(loc, collapsedVecTy);

      // Collapse the destination memref into a 1D memref to match the data
      // layout.
      memref =
          collapseInnermostNDims(rewriter, loc, memrefTy.getRank(), memref);

      // Create a constant 0 index for writing at the beginning of the memref
      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      // Write the cascade vector into the memref.
      rewriter.create<vector::TransferWriteOp>(
          loc, cascadeData, memref, ValueRange(c0),
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
      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      // Read the entire data payload as a 1D vector.
      Value cascadeData = rewriter.create<vector::TransferReadOp>(
          loc, collapsedVecTy, memref, ValueRange(c0), /*padding*/
          arith::getZeroConstant(rewriter, loc, memrefTy.getElementType()),
          /*inBounds*/ SmallVector<bool>{true});

      // Send the vector data via AIE put_cascade.
      rewriter.create<AIE::PutCascadeOp>(loc, cascadeData);
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
    // Match only if the associated channel has channel_type = "cascade".
    auto chan = air::getChannelDeclarationThroughSymbol(op);
    if (!chan)
      return op->emitOpError("cannot resolve channel symbol");

    if (chan.getChannelType().str() != "cascade")
      return op->emitOpError("channel_type is not cascade");

    Value memref = op.getMemref();
    MemRefType memrefTy = cast<MemRefType>(memref.getType());

    // Create the aie.get_cascade operation
    rewriter.setInsertionPoint(op);

    scf::SCFTilingOptions options;
    options.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

    SmallVector<int64_t> tileSizes;
    unsigned innermostTileableDim = 0;
    if (op.getSizes().empty()) {
      for (auto dim : llvm::seq<int64_t>(0, memrefTy.getRank())) {
        tileSizes.push_back(1);
        if (memrefTy.getShape()[dim] > 1)
          innermostTileableDim = dim;
      }
    } else {
      for (auto dim : llvm::seq<int64_t>(0, op.getSizes().size())) {
        tileSizes.push_back(1);
        auto constSize = *getConstantIntValue(op.getSizes()[dim]);
        if (constSize > 1)
          innermostTileableDim = dim;
      }
    }
    unsigned elementWidth = memrefTy.getElementType().getIntOrFloatBitWidth();
    tileSizes[innermostTileableDim] =
        llvm::divideCeil(cascadeWidth, elementWidth);
    SmallVector<OpFoldResult> tileSizesOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);
    options.setTileSizes(tileSizesOfr);

    if (auto put = dyn_cast<air::ChannelPutOp>(op.getOperation())) {
      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, put, options);
      return dyn_cast<air::ChannelInterface>(tilingResult->tiledOps.front());
    } else if (auto get = dyn_cast<air::ChannelGetOp>(op.getOperation())) {
      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, get, options);
      return dyn_cast<air::ChannelInterface>(tilingResult->tiledOps.front());
    }
    return failure();
  }

  AIE::ShimDMAOp getShimDMAOp(AIE::TileOp tile) {
    auto users = tile.getResult().getUsers();
    for (auto user : users)
      if (auto shimDMAOp = dyn_cast<AIE::ShimDMAOp>(*user))
        return shimDMAOp;
    return nullptr;
  }

  AIE::MemTileDMAOp getMemTileDMAOp(AIE::TileOp tile) {
    auto users = tile.getResult().getUsers();
    for (auto user : users)
      if (auto memTileDMAOp = dyn_cast<AIE::MemTileDMAOp>(*user))
        return memTileDMAOp;
    return nullptr;
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
      auto x = tile.getCol();
      auto y = tile.getRow();

      // emit the acquire and release of the L1 buffer locks
      // lock_allocation_list lock_allocs;
      std::unordered_set<Operation *> allocs_to_remap;

      for (auto &alloc : tileDmaAlloc.mm2s_allocs) {
        if (!alloc.foundAlloc(x, y))
          continue;
        for (auto o : alloc.memcpyOps) {
          if (!o)
            continue;
          auto memcpyOpIf = dyn_cast<air::MemcpyInterface>(o);
          if (!memcpyOpIf)
            return o->emitOpError("does not have air::MemcpyInterface");
          if (failed(allocateCoreLocksPerMemcpyOp(rewriter, memcpyOpIf,
                                                  allocs_to_remap, target_model,
                                                  tileDmaAlloc, x, y))) {
            return o->emitOpError("failed to allocate core locks");
          }
        }
      }
      for (auto &alloc : tileDmaAlloc.s2mm_allocs) {
        if (!alloc.foundAlloc(x, y))
          continue;
        for (auto o : alloc.memcpyOps) {
          if (!o)
            continue;
          auto memcpyOpIf = dyn_cast<air::MemcpyInterface>(o);
          if (!memcpyOpIf)
            return o->emitOpError("does not have air::MemcpyInterface");
          if (failed(allocateCoreLocksPerMemcpyOp(rewriter, memcpyOpIf,
                                                  allocs_to_remap, target_model,
                                                  tileDmaAlloc, x, y))) {
            return o->emitOpError("failed to allocate core locks");
          }
        }
      }

      for (auto o : allocs_to_remap) {
        Value alloc = o->getResult(0);
        for (auto u : alloc.getUsers()) {
          if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
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
        if (!alloc.foundAlloc(x, y))
          continue;
        std::pair<AIE::DMAChannelDir, int> mm2s_chan = {
            alloc.dma_channel.direction, alloc.dma_channel.channel};
        for (auto &o : alloc.memcpyOps) {
          tile_dma_memcpys[mm2s_chan].push_back(o);
        }
      }
      for (auto &alloc : tileDmaAlloc.s2mm_allocs) {
        if (!alloc.foundAlloc(x, y))
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
        mem = rewriter.create<AIE::MemOp>(loc, tile);
      }

      if (failed(generateDmaBdProgram<air::TileDMAAllocator, AIE::BufferOp,
                                      AIE::MemOp>(
              rewriter, target_model, tile_dma_memcpys, tileDmaAlloc, loc, mem,
              x, y))) {
        mem->emitOpError("failed to generate dma bd program.");
        return failure();
      }

      // Materialize cascade put and get allocated on cores into put_ and
      // get_cascade ops.
      for (auto *allocList : {&core_cascade_alloc.cascade_put_allocs,
                              &core_cascade_alloc.cascade_get_allocs}) {
        for (auto &alloc : *allocList) {
          if (!alloc.foundAlloc(x, y))
            continue;
          for (auto o : alloc.memcpyOps) {
            if (!o)
              continue;
            auto channelOpIf = dyn_cast<air::ChannelInterface>(o);
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

    // Gather all shim tiles and memtiles used in design
    std::vector<AIE::TileOp> shimtiles;
    std::vector<AIE::TileOp> memTileTiles;
    for (auto &alloc : shimDmaAlloc.mm2s_allocs) {
      auto tile = alloc.getDmaTile();
      if (tile.isShimTile())
        push_back_if_unique<AIE::TileOp>(shimtiles, tile);
      else {
        tile->emitOpError(
            "tile is logged for shim DMA allocation, but is not shim tile.");
        return failure();
      }
    }
    for (auto &alloc : memTileDmaAlloc.mm2s_allocs) {
      auto tile = alloc.getDmaTile();
      if (tile.isMemTile())
        push_back_if_unique<AIE::TileOp>(memTileTiles, tile);
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
      auto x = tile.getCol();
      auto y = tile.getRow();

      // Collect memcpy ops wrt each DMA channel
      llvm::MapVector<std::pair<AIE::DMAChannelDir, int>,
                      std::vector<Operation *>>
          shim_dma_memcpys;

      for (auto &alloc : shimDmaAlloc.mm2s_allocs) {
        if (alloc.foundAlloc(x, y)) {
          std::pair<AIE::DMAChannelDir, int> mm2s_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            shim_dma_memcpys[mm2s_chan].push_back(o);
          }
        }
      }
      for (auto &alloc : shimDmaAlloc.s2mm_allocs) {
        if (alloc.foundAlloc(x, y)) {
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
        shimDMA = rewriter.create<AIE::ShimDMAOp>(
            rewriter.getUnknownLoc(), rewriter.getIndexType(), tile);
      }

      auto loc = rewriter.getUnknownLoc();

      // Generate DMA BD program
      if (failed(generateDmaBdProgram<air::ShimDMAAllocator,
                                      AIE::ExternalBufferOp, AIE::ShimDMAOp>(
              rewriter, target_model, shim_dma_memcpys, shimDmaAlloc, loc,
              shimDMA, x, y))) {
        shimDMA->emitOpError("failed to generate dma bd program.");
        return failure();
      }
    }

    // Generate L2 DMA program

    for (auto tile : memTileTiles) {
      auto x = tile.getCol();
      auto y = tile.getRow();

      // Collect memcpy ops wrt each DMA channel from chessboard; make aie.mem
      // dmabd program
      llvm::MapVector<std::pair<AIE::DMAChannelDir, int>,
                      std::vector<Operation *>>
          memtile_dma_memcpys;

      for (auto &alloc : memTileDmaAlloc.mm2s_allocs) {
        if (alloc.foundAlloc(x, y)) {
          std::pair<AIE::DMAChannelDir, int> mm2s_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            memtile_dma_memcpys[mm2s_chan].push_back(o);
          }
        }
      }
      for (auto &alloc : memTileDmaAlloc.s2mm_allocs) {
        if (alloc.foundAlloc(x, y)) {
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
        memTileDMA = rewriter.create<AIE::MemTileDMAOp>(
            rewriter.getUnknownLoc(), rewriter.getIndexType(), tile);
      }

      auto loc = rewriter.getUnknownLoc();

      // Generate DMA BD program
      if (failed(generateDmaBdProgram<air::MemTileDMAAllocator, AIE::BufferOp,
                                      AIE::MemTileDMAOp>(
              rewriter, target_model, memtile_dma_memcpys, memTileDmaAlloc, loc,
              memTileDMA, x, y, options.use_lock_race_condition_fix))) {
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

    // erase the memcpy operations in aie.device
    std::vector<Operation *> memcpy_ops;
    getAIRMemcpyOpInRegion<T>(device.getRegion(), memcpy_ops);
    for (auto o : memcpy_ops) {
      auto a = dyn_cast<air::AsyncOpInterface>(o);
      if (a && a.getAsyncToken()) {
        OpBuilder b(o);
        o->replaceAllUsesWith(b.create<air::WaitAllOp>(
            o->getLoc(), air::AsyncTokenType::get(o->getContext()),
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

    // Create packet flows
    for (auto srcTile : device.getOps<AIE::TileOp>()) {
      int srcColIndex = srcTile.colIndex();
      int srcRowIndex = srcTile.rowIndex();
      AIE::TileOp destTile;

      if (target_model.isCoreTile(srcColIndex, srcRowIndex) ||
          target_model.isMemTile(srcColIndex, srcRowIndex)) {
        int destColIndex = srcColIndex; // todo: allocation?
        int destRowIndex = 0;
        if (!tiles[{destColIndex, destRowIndex}]) {
          builder.setInsertionPointToStart(device.getBody());
          destTile = builder.create<AIE::TileOp>(builder.getUnknownLoc(),
                                                 destColIndex, destRowIndex);
          tiles[{destColIndex, destRowIndex}] = destTile;
        } else {
          destTile = tiles[{destColIndex, destRowIndex}];
        }
        int destChan = 1; // todo: allocation?

        builder.setInsertionPoint(device.getBody()->getTerminator());
        auto keep_pkt_header = builder.getBoolAttr(true);
        (void)createPacketFlowOp(
            builder, globalFlowID, srcTile, AIE::WireBundle::Trace, 0, destTile,
            AIE::WireBundle::DMA, destChan, keep_pkt_header);
      }
    }
  }

  void runTestPatterns() {

    auto m = getOperation();
    auto ctx = m->getContext();

    RewritePatternSet patterns(ctx);
    std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
    std::map<AIE::BufferOp, AIE::TileOp> bufferToMemtileMap;

    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      m.emitOpError("Invalid aie.device option");
      signalPassFailure();
      return;
    }

    if (clTestPatterns.find("to-aie-mlir") != std::string::npos) {
      std::vector<std::pair<AIE::DeviceOp, air::HerdOp>> aie_modules;
      std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
      AIRToAIEConversionOptions options = {
          /*.col_offset = */ clColOffset,
          /*.row_offset = */ clRowOffset,
          /*.emit_while = */ clEmitWhileLoop,
          /*.emit_herd_lock = */ clEmitHerdLock,
          /*.generate_shim_dma = */ clGenerateShimDMA,
          /*.insert_trace_packet_flow = */ clInsertTracePacketFlow,
          /*.use_packet_flow_at_shim_dmas = */ clUsePktFlowsAtShimDma,
          /*.use_lock_race_condition_fix = */ clUseLockRaceConditionFix,
          /*.device = */ *device};
      createAIEModulesAndOutlineCores(m, aie_modules, tileToHerdMap, options);
      std::set<ModuleOp> seen;
      for (auto &p : aie_modules) {
        auto d = std::get<0>(p);
        auto m = d->getParentOfType<ModuleOp>();
        if (seen.find(m) == seen.end()) {
          seen.insert(m);
          m.print(llvm::outs());
          llvm::outs() << "\n";
        }
        if (options.generate_shim_dma) {
          OpBuilder builder(d);
          cloneL2AndL3MemcpysToDeviceOp(builder, d, m, /*clone_l2*/ true,
                                        /*clone_l3*/ true,
                                        /*use_lock_race_cond_fix*/ true);
          specializeHerdAffineIf(d);
          lowerAirExecute(d);
          lowerScfAirTokens(d);
          std::map<std::string, std::string> chan_to_chan_map;
          specializeChannelBundle(d, chan_to_chan_map);
          specializeL2MemrefsIntoMemtiles(d);
          allocL1Buffers(d, tileToHerdMap, BufferId);
          allocL2Buffers(d, bufferToMemtileMap, BufferId);
          std::map<int, int> chan_renumber_reverse_map;
          air::renumberMemcpyIfOps(&d.getRegion(), chan_renumber_reverse_map);
        }
        if (options.insert_trace_packet_flow)
          createTracePacketFlow(d);
      }
    }

    if (clTestPatterns.find("lower-air-execute") != std::string::npos)
      patterns.insert<LowerAIRExecutePattern>(ctx);
    if (clTestPatterns.find("alloc-l1-buffers") != std::string::npos)
      patterns.insert<AllocL1BuffersPattern, AllocL1BuffersPattern>(
          ctx, tileToHerdMap, BufferId);
    if (clTestPatterns.find("specialize-affine-if") != std::string::npos)
      patterns.insert<SpecializeAffineIfPattern>(ctx);
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
  }

  void runOnOperation() override {

    if (!clTestPatterns.empty()) {
      runTestPatterns();
      return;
    }

    auto module = getOperation();
    OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());

    auto loc = builder.getUnknownLoc();
    auto module_meta = builder.create<airrt::ModuleMetadataOp>(loc);
    builder.createBlock(&module_meta.getSegments());
    builder.create<airrt::ModuleMetadataTerminatorOp>(loc);

    // If we have multiple herds then we must emit them into different aie
    // modules to avoid resource conflicts in the AIE physical dialect.
    std::vector<std::pair<AIE::DeviceOp, air::HerdOp>> aie_devices;

    std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
    std::map<AIE::BufferOp, AIE::TileOp> bufferToMemtileMap;
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
        /* .use_packet_flow_at_shim_dmas = */ clUsePktFlowsAtShimDma,
        /* .use_lock_race_condition_fix = */ clUseLockRaceConditionFix,
        /* .device = */ *device};
    createAIEModulesAndOutlineCores(module, aie_devices, tileToHerdMap,
                                    options);

    std::set<AIE::DeviceOp> seen;
    for (auto &p : aie_devices) {
      auto device = std::get<0>(p);
      air::HerdOp h = std::get<1>(p);
      auto ctx = device->getContext();

      if (seen.find(device) != seen.end())
        continue;
      seen.insert(device);

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

      if (clUseObjFifo) {
        cloneL2AndL3MemcpysToDeviceOp(
            builder, device, module, /*clone_l2*/ true, /*clone_l3*/ false,
            /*use_lock_race_cond_fix*/ options.use_lock_race_condition_fix);
        specializeHerdAffineIf(device);
        lowerAirExecute(device);
        lowerScfAirTokens(device);
        specializeChannelBundle(device, chan_to_chan_map);
        air::renumberMemcpyIfOps(&device.getRegion());
        LowerAIRPingPong(device);
        allocL2Buffers(device, bufferToMemtileMap, BufferId);
        lowerAIRChannels(device, shimTileAlloc, bufferToMemtileMap);
        allocL1Buffers(device, tileToHerdMap, BufferId);
      } else {
        cloneL2AndL3MemcpysToDeviceOp(
            builder, device, module, /*clone_l2*/ true, /*clone_l3*/ true,
            /*use_lock_race_cond_fix*/ options.use_lock_race_condition_fix);
        specializeHerdAffineIf(device);
        lowerAirExecute(device);
        lowerScfAirTokens(device);
        specializeChannelBundle(device, chan_to_chan_map);
        specializeL2MemrefsIntoMemtiles(device);
        allocL1Buffers(device, tileToHerdMap, BufferId);
        allocL2Buffers(device, bufferToMemtileMap, BufferId);
        air::renumberMemcpyIfOps(&device.getRegion(),
                                 chan_renumber_reverse_map);
        if (failed(lowerAIRMemcpyOp<air::ChannelInterface>(device, shimDmaAlloc,
                                                           options))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(lowerAIRMemcpyOp<air::DmaMemcpyNdOp>(device, shimDmaAlloc,
                                                      options))) {
        signalPassFailure();
        return;
      }

      if (options.insert_trace_packet_flow)
        createTracePacketFlow(device);

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

        // Control packet generation for AIE1 is not yet implemented.
        if (isa<AIE::AIE1TargetModel>(device.getTargetModel()) &&
            options.use_packet_flow_at_shim_dmas)
          herd->emitOpError("control packet flow generation is not yet "
                            "supported for AIE1.");
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

        // Control packet generation for AIE1 is not yet implemented.
        if (isa<AIE::AIE1TargetModel>(device.getTargetModel()) &&
            options.use_packet_flow_at_shim_dmas)
          seg->emitOpError("control packet flow generation is not yet "
                           "supported for AIE1.");
      }

      if (isa<AIE::AIE2TargetModel>(device.getTargetModel()) && !clUseObjFifo) {
        // Shim dma allocation metadata linkage (AIE2)
        auto func = h->getParentOfType<func::FuncOp>();
        std::vector<air::MemcpyInterface> shimMemcpyIfOps;
        func.walk([&](air::ChannelInterface o) {
          auto memrefTy = dyn_cast<BaseMemRefType>(o.getMemref().getType());
          if (memrefTy &&
              memrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
            shimMemcpyIfOps.push_back(
                dyn_cast<air::MemcpyInterface>(o.getOperation()));
        });
        func.walk([&](air::DmaMemcpyNdOp o) {
          auto srcMemrefTy =
              dyn_cast<BaseMemRefType>(o.getSrcMemref().getType());
          if (srcMemrefTy &&
              srcMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
            shimMemcpyIfOps.push_back(o);
          auto dstMemrefTy =
              dyn_cast<BaseMemRefType>(o.getDstMemref().getType());
          if (dstMemrefTy &&
              dstMemrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
            shimMemcpyIfOps.push_back(o);
        });
        builder.setInsertionPoint(device.getBody()->getTerminator());
        if (failed(createShimDMAAllocationOps(builder, ctx, shimMemcpyIfOps,
                                              shimDmaAlloc,
                                              chan_renumber_reverse_map))) {
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
        for (auto &t : shimTileAlloc.s2mm_allocs)
          for (auto n : t.chan_names)
            labelAIRDmaOpsWithMetadataObjFifo(channel_ops, n, chan_to_chan_map);
        for (auto &t : shimTileAlloc.mm2s_allocs)
          for (auto n : t.chan_names)
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
    }
  }

  // Static flow id to ensure unique packet header per packet flow.
  int globalFlowID = 0;
  SetVector<Operation *>
      flowOpToFlowIdMap; // Ordered set of memcpy_flows.air_flow_op, flowID is
                         // index.

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
    auto memRefTy = dyn_cast<MemRefType>(val.getType());
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
      func::FuncOp funcOp = rewriter.create<func::FuncOp>(
          op->getLoc(), fnNameAttr.getValue(), libFnType);
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
      /* .trace_size = */ 0,
      /* .ctrl_packet = */ false,
      /* .use_lock_race_condition_fix = */ true,
      /* .device = */ *device};
  std::vector<std::pair<ModuleOp, air::HerdOp>> aie_modules;
  p.walk([&](air::HerdOp h) { aie_modules.push_back({aie_module, h}); });
  std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
  for (auto &p : aie_modules) {
    ModuleOp aie_module = std::get<0>(p);
    air::HerdOp h = std::get<1>(p);
    rewriter.setInsertionPointToStart(aie_module.getBody());
    auto devOp = rewriter.create<AIE::DeviceOp>(
        aie_module.getLoc(),
        AIE::AIEDeviceAttr::get(rewriter.getContext(), options.device));
    setAIEDeviceDataLayout(rewriter, devOp);
    AIE::DeviceOp::ensureTerminator(devOp.getRegion(), rewriter,
                                    devOp.getLoc());
    outlineAIECores(rewriter, devOp, h, tileToHerdMap, options);

    auto ctx = aie_module->getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<SpecializeAffineIfPattern>(ctx);
    patterns.insert<LowerAIRExecutePattern>(ctx);
    patterns.insert<AllocL1BuffersPattern>(ctx, tileToHerdMap, BufferId);
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
