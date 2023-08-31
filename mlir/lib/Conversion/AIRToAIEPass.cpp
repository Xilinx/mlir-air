//===- AIRToAIEPass.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
using namespace xilinx;
using namespace xilinx::air;

namespace {

struct AIRToAIEOptions {
  int64_t col_offset;
  int64_t row_offset;
  bool emit_while;
  bool emit_herd_lock;
  AIE::AIEDevice device;
};

AIE::TileOp getPhysTileOpOrNull(AIE::DeviceOp aie_device, int col, int row) {
  for (auto t : aie_device.getOps<AIE::TileOp>()) {
    if (t.colIndex() == col && t.rowIndex() == row)
      return t;
  }
  return nullptr;
}

// get tileop using physical coordinates
AIE::TileOp getPhysTileOp(AIE::DeviceOp aie_device, int col, int row) {
  auto t = getPhysTileOpOrNull(aie_device, col, row);
  if (t)
    return t;

  OpBuilder builder(aie_device);

  builder.setInsertionPointToStart(aie_device.getBody());
  for (auto &o : aie_device.getBody()->getOperations()) {
    if (isa<AIE::TileOp>(o))
      builder.setInsertionPointAfter(&o);
    else
      break;
  }
  return builder.create<AIE::TileOp>(UnknownLoc::get(aie_device.getContext()),
                                     col, row);
}

// get memcpy operation volumn (elements) as int
int getMemcpySizesAsInt(Value memref, SmallVector<Value> sizes) {
  MemRefType memTy = memref.getType().cast<MemRefType>();
  if (sizes.empty())
    return getTensorVolume(memTy);
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
    AIE::TileOp shim_tile;
    int available_channels;
  };

  std::vector<shim_allocation_info_t> mm2s_allocs, s2mm_allocs;

  ShimTileAllocator(const AIE::AIETargetModel &target) : aie_target(target) {
    shim_dma_channels = 2;
    for (int i = 0, e = aie_target.columns(); i < e; i++) {
      if (aie_target.isShimNOCTile(i, 0))
        shim_columns.push_back(i);
    }
  }

  AIE::TileOp getShimTile(AIE::DeviceOp aie_device, int src_memory_space,
                          int dst_memory_space) {
    bool isMM2S = (src_memory_space < dst_memory_space);
    auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

    // return first available shim tile with a free channel
    for (auto &t : *allocs) {
      if (t.available_channels > 0) {
        t.available_channels -= 1;
        return t.shim_tile;
      }
    }
    auto shim_col = shim_columns[allocs->size()];
    auto shim_tile = getPhysTileOp(aie_device, shim_col, 0);
    allocs->push_back({shim_tile, shim_dma_channels - 1});

    return shim_tile;
  }
};

bool isMM2S(AIE::DMAChannel channel) {
  return (channel.first == AIE::DMAChannelDir::MM2S);
}
bool isTileInbound(air::MemcpyInterface memcpyOp) {
  if (memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    auto src_memory_space = memcpyOp.getSrcMemref()
                                .getType()
                                .cast<MemRefType>()
                                .getMemorySpaceAsInt();
    auto dst_memory_space = memcpyOp.getDstMemref()
                                .getType()
                                .cast<MemRefType>()
                                .getMemorySpaceAsInt();
    assert(src_memory_space != dst_memory_space);
    return src_memory_space < dst_memory_space;
  } else if (!memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    return true;
  } else
    return false;
}
bool isTileOutbound(air::MemcpyInterface memcpyOp) {
  return !isTileInbound(memcpyOp);
}
bool isLegalMemorySpace(air::MemcpyInterface memcpyOp) {
  if (memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    if (getMemorySpaceAsString(memcpyOp.getSrcMemref()) == "L1" &&
        getMemorySpaceAsString(memcpyOp.getDstMemref()) == "L3") {
      return true;
    } else if (getMemorySpaceAsString(memcpyOp.getSrcMemref()) == "L3" &&
               getMemorySpaceAsString(memcpyOp.getDstMemref()) == "L1") {
      return true;
    } else
      return false;
  } else if (memcpyOp.getSrcMemref() &&
             getMemorySpaceAsString(memcpyOp.getSrcMemref()) == "L1") {
    return true;
  } else if (memcpyOp.getDstMemref() &&
             getMemorySpaceAsString(memcpyOp.getDstMemref()) == "L1") {
    return true;
  } else
    return false;
}

struct DMAAllocator {

  std::vector<int> dma_columns;
  int dma_channels;
  const AIE::AIETargetModel &aie_target;

  struct allocation_info_t {
    AIE::TileOp dma_tile;
    int64_t col;
    int64_t row;
    int64_t dma_channel;
    int64_t tile_channel;
    std::vector<int32_t> dma_id;
  };

  std::vector<allocation_info_t> mm2s_allocs, s2mm_allocs;

  DMAAllocator(const AIE::AIETargetModel &target) : aie_target(target) {
    dma_channels = 2;
    for (int i = 0, e = aie_target.columns(); i < e; i++) {
      if (aie_target.isShimNOCTile(i, 0))
        dma_columns.push_back(i);
    }
  }

  AIE::TileOp getTile(AIE::DeviceOp aie_device, air::MemcpyInterface &memcpyOp,
                      int64_t tile_channel, int64_t col, int64_t row) {
    auto allocs = isTileInbound(memcpyOp) ? &mm2s_allocs : &s2mm_allocs;

    for (auto &t : *allocs) {
      if (col == t.col && row == t.row) {
        for (auto id : t.dma_id)
          if (memcpyOp.getId() == id)
            return t.dma_tile;
        if (tile_channel == t.tile_channel) {
          t.dma_id.push_back(memcpyOp.getId());
          return t.dma_tile;
        }
      }
    }
    auto dma_col = dma_columns[allocs->size() / dma_channels];
    auto dma_channel = allocs->size() % dma_channels;
    auto dma_tile = getPhysTileOp(aie_device, dma_col, 0);
    allocs->push_back({dma_tile,
                       col,
                       row,
                       (int64_t)dma_channel,
                       tile_channel,
                       {memcpyOp.getId()}});
    LLVM_DEBUG(llvm::outs() << "isTileInbound = " << isTileInbound(memcpyOp)
                            << " " << memcpyOp.getId() << ", col =" << col
                            << ", row = " << row << ", dma_col =" << dma_col
                            << ", dma_chan =" << dma_channel << "\n");

    return dma_tile;
  }

  AIE::DMAChannel getChannel(AIE::DeviceOp aie_device,
                             air::MemcpyInterface &memcpyOp,
                             AIE::DMAChannel tile_channel, int64_t col,
                             int64_t row) {
    bool isMM2S = isTileInbound(memcpyOp);
    auto allocs = isTileInbound(memcpyOp) ? &mm2s_allocs : &s2mm_allocs;

    int64_t chan = -1;
    for (auto &t : *allocs) {
      LLVM_DEBUG(llvm::outs()
                 << "gSDC: op " << t.dma_tile << ", col" << t.col << ", row "
                 << t.row << ", chan " << t.dma_channel << "\n");
      if (col == t.col && row == t.row) {
        for (auto id : t.dma_id)
          if (memcpyOp.getId() == id)
            chan = t.dma_channel;
        if (tile_channel.second == t.tile_channel) {
          chan = t.dma_channel;
        }
      }
    }
    assert(chan != -1);

    LLVM_DEBUG(llvm::outs() << "isMM2S = " << isMM2S << ", col =" << col
                            << ", row = " << row << " chan =" << chan << "\n");

    if (isMM2S)
      return std::make_pair(AIE::DMAChannelDir::MM2S, (int)chan);
    else
      return std::make_pair(AIE::DMAChannelDir::S2MM, (int)chan);
  }
};

AIE::LockOp allocateLockOp(AIE::DeviceOp aie_device, AIE::TileOp tile,
                           int init = 0, int id = -1) {
  AIE::LockOp lock = nullptr;
  std::set<int> ids;
  aie_device.walk([&](AIE::LockOp l) {
    if (cast<AIE::TileOp>(l.getTile().getDefiningOp()) == tile) {
      auto i = l.getLockIDValue();
      if (i == id)
        lock = l;
      ids.insert(i);
    }
  });

  if (lock)
    return lock;

  int new_id = 0;
  if (id > 0)
    new_id = id;
  else {
    while (ids.count(new_id))
      new_id++;
  }

  OpBuilder b(aie_device);
  Operation *t = tile.getOperation();
  while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
    t = t->getNextNode();
  b.setInsertionPointAfter(t);
  return b.create<AIE::LockOp>(tile.getLoc(), tile, new_id, init);
}

void outlineAIECores(OpBuilder &builder, AIE::DeviceOp aie_device,
                     xilinx::air::HerdOp h,
                     std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
                     AIRToAIEOptions &options) {
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
  auto col_name = xilinx::air::HerdOp::getColOffsetAttrName();
  auto row_name = xilinx::air::HerdOp::getRowOffsetAttrName();
  if (auto co = h.getColOffset())
    col_offset = *co;
  else
    h->setAttr(col_name, IntegerAttr::get(IntegerType::get(h->getContext(), 32),
                                          col_offset));
  if (auto ro = h.getRowOffset())
    row_offset = *ro;
  else
    h->setAttr(row_name, IntegerAttr::get(IntegerType::get(h->getContext(), 32),
                                          row_offset));

  for (auto y = 0; y < herd_size_y; y++) {
    for (auto x = 0; x < herd_size_x; x++) {
      auto hloc = h.getLoc();
      IRMapping remap;
      auto phys_x = x + col_offset;
      auto phys_y = y + row_offset;

      // make the AIE.tile
      auto tile = getPhysTileOp(aie_device, phys_x, phys_y);

      Operation *t = tile.getOperation();
      while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
        t = t->getNextNode();
      builder.setInsertionPointAfter(t);

      // make the AIE.core for the tile core
      auto core = tile.getCoreOp();
      if (!core) {
        core = builder.create<AIE::CoreOp>(hloc, tile);
        tileToHerdMap[tile] = h;
        auto herd_name =
            aie_device
                ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                .getValue()
                .str();
        core->setAttr("elf_file",
                      StringAttr::get(aie_device.getContext(),
                                      herd_name + "_core_" +
                                          std::to_string(phys_x) + "_" +
                                          std::to_string(phys_y) + ".elf"));
        if (auto a = h->getAttrOfType<StringAttr>("link_with"))
          core->setAttr("link_with", a);
      }

      Value herd_lock = nullptr;
      if (options.emit_herd_lock)
        herd_lock = allocateLockOp(aie_device, tile, /*init=*/0, /*id=*/0);

      // the buffers and locks created below need to go before the core and
      // mem
      builder.setInsertionPoint(core);

      assert((h.getBody().getBlocks().size() == 1) &&
             "Launch body can only contain one Block");

      // generate the AIE.core body
      //
      OpBuilder core_builder(core);
      Block *core_bb = core_builder.createBlock(&core.getBody());

      Block *entry_bb = core_builder.createBlock(core_bb);
      core_builder.setInsertionPointToEnd(entry_bb);
      core_builder.create<cf::BranchOp>(hloc, core_bb);
      core_builder.setInsertionPointToEnd(core_bb);

      // map the tile ids and herd size to constants
      remap.map(h.getIds()[0],
                core_builder.create<arith::ConstantIndexOp>(hloc, x));
      remap.map(h.getIds()[1],
                core_builder.create<arith::ConstantIndexOp>(hloc, y));
      remap.map(h.getSize()[0],
                core_builder.create<arith::ConstantIndexOp>(hloc, herd_size_x));
      remap.map(h.getSize()[1],
                core_builder.create<arith::ConstantIndexOp>(hloc, herd_size_y));

      for (auto a : h.getKernelArguments()) {
        auto memrefTy = a.getType().dyn_cast<MemRefType>();
        if (!memrefTy)
          continue;

        OpBuilder b(aie_device);
        b.setInsertionPoint(core);

        int which_try = 0;
        std::string sym_name = "__air_herd_arg_0";
        while (aie_device.lookupSymbol(sym_name))
          sym_name = "__air_herd_arg_" + std::to_string(++which_try);
        b.create<memref::GlobalOp>(builder.getUnknownLoc(), sym_name,
                                   builder.getStringAttr("public"), memrefTy,
                                   nullptr, false, nullptr);

        auto m = core_builder.create<memref::GetGlobalOp>(
            hloc, SmallVector<Type, 1>{a.getType()}, sym_name);
        remap.map(a, m);
      }

      if (options.emit_herd_lock)
        core_builder.create<AIE::UseLockOp>(core_builder.getUnknownLoc(),
                                            herd_lock, 0,
                                            AIE::LockAction::Acquire);

      Region &r = h.getRegion();
      r.cloneInto(&core.getBody(), remap);

      Block *launch_bb = remap.lookup(&r.front());
      core_builder.create<cf::BranchOp>(hloc, launch_bb);
      core_builder.setInsertionPoint(launch_bb->getTerminator());
      if (options.emit_herd_lock)
        core_builder.create<AIE::UseLockOp>(core_builder.getUnknownLoc(),
                                            herd_lock, 0,
                                            AIE::LockAction::Release);

      if (options.emit_while)
        core_builder.create<cf::BranchOp>(hloc, core_bb);
      else
        core_builder.create<AIE::EndOp>(hloc);

      core.walk([&](Operation *op) {
        if (auto call = dyn_cast<func::CallOp>(op)) {
          auto fn = aie_device.lookupSymbol<func::FuncOp>(call.getCallee());
          if (!fn) {
            fn = func::FuncOp::create(aie_device.getLoc(), call.getCallee(),
                                      call.getCalleeType());
            fn.setPrivate();
            aie_device.push_back(fn);
          }
        }
      });

      // erase air.herd_termintor ops
      launch_bb->walk([&](air::HerdTerminatorOp op) { op->erase(); });
    }
  }
}

void createAIEModulesAndOutlineCores(
    ModuleOp module,
    std::vector<std::pair<AIE::DeviceOp, xilinx::air::HerdOp>> &aie_modules,
    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
    AIRToAIEOptions &options) {

  SmallVector<air::SegmentOp> segments;
  SmallVector<air::HerdOp> herds;
  module.walk([&](xilinx::air::SegmentOp s) { segments.push_back(s); });
  module.walk([&](xilinx::air::HerdOp h) {
    if (h->getParentOfType<xilinx::air::SegmentOp>())
      return;
    herds.push_back(h);
  });

  for (auto p : segments) {
    std::string segment_name;
    if (auto attr =
            p->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
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

    aie_dev.getRegion().emplaceBlock();
    p.walk([&](xilinx::air::HerdOp h) { aie_modules.push_back({aie_dev, h}); });
  };

  for (auto h : herds) {
    std::string segment_name;
    segment_name = "segment_" + std::to_string(aie_modules.size());
    std::string aie_module_name = "aie." + segment_name;
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    auto aie_dev = builder.create<AIE::DeviceOp>(
        module.getLoc(),
        AIE::AIEDeviceAttr::get(builder.getContext(), options.device));
    aie_dev->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(builder.getContext(), segment_name));
    aie_dev.getRegion().emplaceBlock();
    aie_modules.push_back({aie_dev, h});
  };
  for (auto &p : aie_modules) {
    auto aie_dev = std::get<0>(p);
    auto h = std::get<1>(p);
    OpBuilder builder(aie_dev);
    outlineAIECores(builder, aie_dev, h, tileToHerdMap, options);
  }
}

AIE::BufferOp allocateBufferOp(MemRefType memrefTy, AIE::TileOp tile,
                               mlir::StringAttr attr = nullptr, int x = -1,
                               int y = -1) {

  static uint64_t BufferId = 0;

  OpBuilder builder(tile);
  Operation *t = tile.getOperation();
  while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
    t = t->getNextNode();
  builder.setInsertionPointAfter(t);
  AIE::BufferOp bufferOp =
      builder.create<AIE::BufferOp>(tile->getLoc(), memrefTy, tile);

  // if a symbol name was passed in, use it to make
  // the buffer symbol name as "sym_name_x_y",
  // otherwise we'll make a generic symbol name "bufN"
  std::stringstream ss;
  if (attr) {
    if (x >= 0 && y >= 0)
      ss << attr.getValue().str() << "_" << x << "_" << y;
    else
      ss << attr.getValue().str() << BufferId++;
  } else {
    ss << "buf" << BufferId++;
  }
  bufferOp->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(tile->getContext(), ss.str()));

  return bufferOp;
}

bool isInSet(IntegerSet is) {
  auto constraints = is.getConstraints();
  auto eqFlags = is.getEqFlags();

  int i = 0;
  for (auto c : constraints) {
    auto expr = simplifyAffineExpr(c, 0, 1).dyn_cast<AffineConstantExpr>();
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

bool isInSet(int64_t x, int64_t y, AffineIfOp aif) {
  auto is = aif.getIntegerSet();
  if (is.getConstraints().size() != 2)
    return false;

  SmallVector<AffineExpr, 2> dims{
      getAffineConstantExpr(x, aif->getContext()),
      getAffineConstantExpr(y, aif->getContext()),
  };

  auto newIs = is.replaceDimsAndSymbols({}, dims, 0, 2);
  return isInSet(newIs);
}

struct SpecializeAffineIfPattern : public OpRewritePattern<AffineIfOp> {
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  SpecializeAffineIfPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {

    auto core = op->getParentOfType<AIE::CoreOp>();
    if (!core)
      return failure();

    bool in_set = false;
    if (op.getNumOperands() == 2) {
      SmallVector<int64_t, 2> operands;
      for (auto o : op.getOperands()) {
        auto v = dyn_cast<arith::ConstantIndexOp>(o.getDefiningOp());
        if (!v)
          return failure();
        operands.push_back(v.value());
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
  (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
}

struct LowerAIRExecutePattern : public OpRewritePattern<air::ExecuteOp> {
  using OpRewritePattern<air::ExecuteOp>::OpRewritePattern;

  LowerAIRExecutePattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::ExecuteOp op,
                                PatternRewriter &rewriter) const override {
    auto &bb = op.getBody().front();
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
  patterns.insert<LowerAIRExecutePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(d, std::move(patterns));
}

struct LowerScfTokenPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LowerScfTokenPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp fop,
                                PatternRewriter &rewriter) const override {

    if (!fop.getNumIterOperands())
      return failure();

    SmallVector<Value, 4> iter_args;
    BitVector iter_args_idx(fop.getNumOperands());

    // erase air.event from the iter args
    for (OpOperand &oper : fop.getIterOpOperands()) {
      Value v = oper.get();
      BlockArgument block_arg = fop.getRegionIterArgForOpOperand(oper);
      if (v.getType().isa<xilinx::air::AsyncTokenType>()) {
        block_arg.replaceAllUsesWith(v);
        iter_args_idx.set(block_arg.getArgNumber());
      } else {
        iter_args.push_back(v);
      }
    }

    // if none of the iter args were air.async.token, return
    if (iter_args.size() == fop.getNumIterOperands())
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
      if (r.getType().isa<xilinx::air::AsyncTokenType>())
        r.replaceAllUsesWith(
            rewriter
                .create<xilinx::air::WaitAllOp>(
                    fop->getLoc(),
                    xilinx::air::AsyncTokenType::get(fop->getContext()),
                    SmallVector<Value, 1>{})
                .getResult(0));
      else
        r.replaceAllUsesWith(new_fop.getResult(idx++));
    }

    // remove air.async.token from the yield op
    auto yield = new_region.back().getTerminator();
    assert(isa<scf::YieldOp>(yield));
    rewriter.setInsertionPoint(yield);
    SmallVector<Value, 4> yield_operands;
    SmallVector<Value, 4> token_operands;
    for (auto o : yield->getOperands()) {
      if (o.getType().isa<xilinx::air::AsyncTokenType>())
        token_operands.push_back(o);
      else
        yield_operands.push_back(o);
    }
    rewriter.create<xilinx::air::WaitAllOp>(
        fop->getLoc(), SmallVector<Type, 1>{}, token_operands);
    rewriter.create<scf::YieldOp>(yield->getLoc(), yield_operands);
    rewriter.eraseOp(yield);

    rewriter.eraseOp(fop);
    return success();
  }
};

void lowerScfAirTokens(AIE::DeviceOp m) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<LowerScfTokenPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
}

struct LowerPipeGetPutPattern : public OpRewritePattern<air::PipelinePutOp> {
  using OpRewritePattern<air::PipelinePutOp>::OpRewritePattern;

  LowerPipeGetPutPattern(MLIRContext *ctx,
                         std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap)
      : OpRewritePattern(ctx), tileToHerdMap(tileToHerdMap) {}

  LogicalResult matchAndRewrite(air::PipelinePutOp put,
                                PatternRewriter &rewriter) const override {
    auto aie_device = put->getParentOfType<AIE::DeviceOp>();
    auto core = put->getParentOfType<AIE::CoreOp>();
    assert(aie_device && core);

    auto herd = tileToHerdMap[core.getTileOp()];
    auto c = herd.getColOffset();
    auto r = herd.getRowOffset();
    auto col_offset = c ? *c : 0;
    auto row_offset = r ? *r : 0;

    auto other_x = cast<arith::ConstantIndexOp>(put.getDst0().getDefiningOp());
    auto other_y = cast<arith::ConstantIndexOp>(put.getDst1().getDefiningOp());
    auto other_core = getPhysTileOp(aie_device, other_x.value() + col_offset,
                                    other_y.value() + row_offset)
                          .getCoreOp();
    assert(other_core);

    air::PipelineGetOp get;
    other_core.walk([&](air::PipelineGetOp pgo) { get = pgo; });
    assert(get && get->getNumResults() == (put->getNumOperands() - 2));

    for (auto p :
         llvm::zip(put->getOperands().drop_front(2), get->getResults())) {

      auto o = std::get<0>(p); // operand of put
      auto r = std::get<1>(p); // result of get
      // for each ranked tensor put (yielded) by the tile
      if (RankedTensorType tt = o.getType().dyn_cast<RankedTensorType>()) {
        auto memrefTy = MemRefType::get(tt.getShape(), tt.getElementType(), {},
                                        (int)air::MemorySpace::L1);
        // allocate buffer+lock
        auto buf = allocateBufferOp(
            memrefTy, core.getTileOp(),
            StringAttr::get(aie_device.getContext(), "pipebuf"));
        auto lockOp = allocateLockOp(aie_device, core.getTileOp());

        // acquire the lock for write on the put side
        rewriter.setInsertionPoint(put);
        rewriter.create<AIE::UseLockOp>(put->getLoc(), lockOp, 0,
                                        AIE::LockAction::Acquire);
        rewriter.create<memref::TensorStoreOp>(put->getLoc(), o, buf);
        rewriter.create<AIE::UseLockOp>(put->getLoc(), lockOp, 1,
                                        AIE::LockAction::Release);

        // acquire the lock for read on the get side
        rewriter.setInsertionPoint(get);
        rewriter.create<AIE::UseLockOp>(get->getLoc(), lockOp, 1,
                                        AIE::LockAction::Acquire);
        auto loadOp =
            rewriter.create<bufferization::ToTensorOp>(get->getLoc(), buf);
        rewriter.create<AIE::UseLockOp>(get->getLoc(), lockOp, 0,
                                        AIE::LockAction::Release);
        r.replaceAllUsesWith(loadOp.getResult());
      } else {
        llvm::errs() << "error, unsupported air.pipeline.yield operand type\n";
        assert(0 && "Unsupported");
        return failure();
      }
    }
    rewriter.eraseOp(get);
    rewriter.eraseOp(put);
    return success();
  }

private:
  std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap;
};

// This function replaces PipelinePutOp/PipelineGetOp pairs with a
// shared AIE.buffer + AIE.lock. This is a single-buffered implementation
// with exclusive access to the buffer controlled by the lock. i.e. FIXME.
void lowerPipelineGetPut(AIE::DeviceOp &m,
                         std::map<AIE::TileOp, air::HerdOp> tileToHerdMap) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<LowerPipeGetPutPattern>(ctx, tileToHerdMap);
  (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
}

struct AllocL1TensorsPattern
    : public OpRewritePattern<bufferization::ToMemrefOp> {
  using OpRewritePattern<bufferization::ToMemrefOp>::OpRewritePattern;

  AllocL1TensorsPattern(MLIRContext *ctx,
                        std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap)
      : OpRewritePattern(ctx), tileToHerdMap(tileToHerdMap) {}

  LogicalResult matchAndRewrite(bufferization::ToMemrefOp cast,
                                PatternRewriter &rewriter) const override {

    AIE::CoreOp core = cast->getParentOfType<AIE::CoreOp>();
    if (!core)
      return failure();

    AIE::TileOp tile = core.getTileOp();
    if (!tile)
      return failure();

    MemRefType memrefTy = nullptr;
    memrefTy = cast.getType().cast<MemRefType>();

    if (memrefTy.getMemorySpaceAsInt() != (int)air::MemorySpace::L1)
      return failure();

    rewriter.setInsertionPointAfter(tile);
    auto herd = tileToHerdMap[core.getTileOp()];
    int64_t col_offset = 0;
    int64_t row_offset = 0;
    if (herd) {
      auto c = herd.getColOffset();
      auto r = herd.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    }
    auto buffer = allocateBufferOp(
        memrefTy, tile,
        cast->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        tile.getCol() - col_offset, tile.getRow() - row_offset);

    rewriter.setInsertionPoint(cast);
    rewriter.create<memref::TensorStoreOp>(cast.getLoc(), cast.getOperand(),
                                           buffer);
    rewriter.replaceOp(cast, buffer->getResults());
    return success();
  }

private:
  std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap;
};

struct AllocL1BuffersPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  AllocL1BuffersPattern(MLIRContext *ctx,
                        std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap)
      : OpRewritePattern(ctx), tileToHerdMap(tileToHerdMap) {}

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

    rewriter.setInsertionPointAfter(tile);
    auto herd = tileToHerdMap[core.getTileOp()];
    int64_t col_offset = 0;
    int64_t row_offset = 0;
    if (herd) {
      auto c = herd.getColOffset();
      auto r = herd.getRowOffset();
      col_offset = c ? *c : 0;
      row_offset = r ? *r : 0;
    }

    auto buffer = allocateBufferOp(
        memrefTy, tile,
        alloc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        tile.getCol() - col_offset, tile.getRow() - row_offset);

    rewriter.setInsertionPoint(alloc);
    rewriter.replaceOp(alloc, buffer->getResults());
    return success();
  }

private:
  std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap;
};

void allocL1Buffers(AIE::DeviceOp m,
                    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<AllocL1BuffersPattern, AllocL1TensorsPattern>(ctx,
                                                                tileToHerdMap);
  (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
}

struct LowerAIRChannelsPattern : public OpRewritePattern<air::ChannelOp> {
  using OpRewritePattern<air::ChannelOp>::OpRewritePattern;

  LowerAIRChannelsPattern(MLIRContext *ctx, ShimTileAllocator &shimTileAlloc)
      : OpRewritePattern(ctx), shimTileAlloc(shimTileAlloc) {}

  LogicalResult matchAndRewrite(air::ChannelOp channel,
                                PatternRewriter &rewriter) const override {
    auto device = channel->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    // SpecializeChannelBundlePattern should have removed them
    if (channel.getBundleSize() > 1)
      return failure();

    AIE::AIEObjectFifoType datatype;
    std::vector<ChannelPutOp> channelPuts =
        getChannelPutOpThroughSymbol(channel, device);
    std::vector<ChannelGetOp> channelGets =
        getChannelGetOpThroughSymbol(channel, device);

    // put/get come in pairs, if one is missing then it's L3
    Value producerTile;
    if (channelPuts.size() > 0) {
      if (channelPuts.size() > 1)
        return channel.emitOpError(
            "channel lowering currently does not support many-to-one/many");
      auto res = findChannelPutGetTile<ChannelPutOp>(channelPuts[0],
                                                     &producerTile, &datatype);
      if (res.failed())
        return res;
    } else {
      // put from L3
      producerTile = shimTileAlloc.getShimTile(
          device, (int)air::MemorySpace::L3, (int)air::MemorySpace::L1);
    }

    // put/get come in pairs, if one is missing then it's L3
    std::vector<Value> consumers;
    Value consumerTile;
    if (channelGets.size() > 1 && !channel.isBroadcast())
      return channel.emitOpError("has multiple gets but no broadcast shape");

    int expectedGets = channel.isBroadcast() ? channel.getBroadcastNum() : 1;
    for (auto get : channelGets) {
      auto res =
          findChannelPutGetTile<ChannelGetOp>(get, &consumerTile, &datatype);
      if (res.failed())
        return res;
      consumers.push_back(consumerTile);
    }
    for (int i = 0; i < expectedGets - (int)channelGets.size(); i++) {
      // get from L3
      consumerTile = shimTileAlloc.getShimTile(
          device, (int)air::MemorySpace::L1, (int)air::MemorySpace::L3);
      consumers.push_back(consumerTile);
    }

    if (!datatype)
      return failure();

    // create objFifo
    rewriter.setInsertionPoint(*(device.getOps<AIE::CoreOp>().begin()));
    AIE::ObjectFifoCreateOp objFifo = createObjectFifo(
        rewriter, datatype, producerTile, consumers,
        channel.getBufferResources(), "air_" + channel.getName().str());

    // replace put/get and any associated memref alloc/dealloc
    std::vector<Operation *> erased_deallocs;
    for (auto put : channelPuts) {
      rewriteChannelAllocs<ChannelPutOp>(rewriter, put, objFifo,
                                         AIE::ObjectFifoPort::Produce);
      rewriteChannelDeallocs<ChannelPutOp>(rewriter, put, objFifo,
                                           AIE::ObjectFifoPort::Produce,
                                           erased_deallocs);
      // clear any dependence to put
      if (put.getAsyncToken())
        for (auto u : put.getAsyncToken().getUsers())
          if (auto async_u = dyn_cast<air::AsyncOpInterface>(u))
            air::eraseAsyncDependencyFromAsyncOp(async_u, put.getAsyncToken());
      // TODO: complete else
    }
    for (auto get : channelGets) {
      rewriteChannelAllocs<ChannelGetOp>(rewriter, get, objFifo,
                                         AIE::ObjectFifoPort::Consume);
      rewriteChannelDeallocs<ChannelGetOp>(rewriter, get, objFifo,
                                           AIE::ObjectFifoPort::Consume,
                                           erased_deallocs);
      if (get.getAsyncToken())
        // clear any dependence to get
        for (auto u : get.getAsyncToken().getUsers())
          if (auto async_u = dyn_cast<air::AsyncOpInterface>(u))
            air::eraseAsyncDependencyFromAsyncOp(async_u, get.getAsyncToken());
      // TODO: complete else
    }
    // erase deallocs
    for (auto o : erased_deallocs)
      rewriter.eraseOp(o);

    // erase channel puts and gets
    for (auto get : channelGets)
      rewriter.eraseOp(get);
    for (auto put : channelPuts)
      rewriter.eraseOp(put);

    // erase the channel
    rewriter.eraseOp(channel);
    return success();
  }

private:
  // find AIE cores and their tiles based on memory hierarchy levels
  template <typename MyOp>
  LogicalResult findChannelPutGetTile(MyOp op, Value *tile,
                                      AIE::AIEObjectFifoType *datatype) const {
    MemRefType memref = op.getMemref().getType().template cast<MemRefType>();
    int mem_space = memref.getMemorySpaceAsInt();
    *datatype = AIE::AIEObjectFifoType::get(
        MemRefType::get(memref.getShape(), memref.getElementType()));
    if (mem_space == (int)air::MemorySpace::L1) {
      AIE::CoreOp core = op->template getParentOfType<AIE::CoreOp>();
      if (!core)
        return op.emitOpError("could not retrieve core for channel put/get op");
      *tile = core.getTileOp();
      return success();
    } else {
      // TODO: complete for L2
      return op.emitOpError("unsupported memory space");
    }
  }

  AIE::ObjectFifoCreateOp createObjectFifo(OpBuilder &builder,
                                           AIE::AIEObjectFifoType datatype,
                                           Value prodTile,
                                           const std::vector<Value> &consTile,
                                           int depth, StringRef name) const {
    AIE::ObjectFifoCreateOp fifo = builder.create<AIE::ObjectFifoCreateOp>(
        builder.getUnknownLoc(), name, prodTile, consTile,
        builder.getIntegerAttr(builder.getI32Type(), depth), datatype);
    return fifo;
  }

  template <typename MyOp>
  void rewriteChannelAllocs(PatternRewriter &rewriter, MyOp op,
                            AIE::ObjectFifoCreateOp objFifo,
                            AIE::ObjectFifoPort port) const {

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
      rewriter.replaceOp(a.getOperation(), producerAccess.getOutput());
  }

  template <typename T>
  void push_back_if_unique(std::vector<T> &vec, T entry) const {
    if (std::find(vec.begin(), vec.end(), entry) == vec.end())
      vec.push_back(entry);
  }

  template <typename MyOp>
  void rewriteChannelDeallocs(PatternRewriter &rewriter, MyOp op,
                              AIE::ObjectFifoCreateOp objFifo,
                              AIE::ObjectFifoPort port,
                              std::vector<Operation *> &erased_deallocs) const {
    for (auto u : op.getMemref().getDefiningOp()->getUsers()) {
      if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
        rewriter.setInsertionPoint(&op->getBlock()->back());
        rewriter.create<AIE::ObjectFifoReleaseOp>(dealloc->getLoc(), port,
                                                  objFifo.getName(), 1);
        // Delete ops at the end of the rewrite pattern to avoid repeatedly
        // deleting the same op
        push_back_if_unique<Operation *>(erased_deallocs,
                                         dealloc.getOperation());
      }
    }
  }

  ShimTileAllocator &shimTileAlloc;
};

// This function replaces ChannelPutOp/ChannelGetOp with AIE_CreateObjectFifoOps
// and with ObjectFifoAcquireOp<Producer/Consumer>. It also erases memref allocs
// as the objFifo lowering allocates its own memory. It replaces the associated
// memref deallocs with ObjectFifoReleaseOps.
void lowerAIRChannels(AIE::DeviceOp &d, ShimTileAllocator &a) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<LowerAIRChannelsPattern>(ctx, a);
  (void)applyPatternsAndFoldGreedily(d, std::move(patterns));
}

struct SpecializeChannelBundlePattern
    : public OpRewritePattern<air::ChannelOp> {
  using OpRewritePattern<air::ChannelOp>::OpRewritePattern;

  SpecializeChannelBundlePattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::ChannelOp channel,
                                PatternRewriter &rewriter) const override {

    auto device = channel->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    if (channel.getBundleSize() <= 1)
      return failure();

    std::vector<ChannelPutOp> channelPuts =
        getChannelPutOpThroughSymbol(channel, device);
    std::vector<ChannelGetOp> channelGets =
        getChannelGetOpThroughSymbol(channel, device);

    // Walk through each element in a channel bundle
    auto bundle_size = extractFromI64ArrayAttr(channel.getSize());
    auto bundle_size_stdvec = convertToStdVec(bundle_size);
    for (unsigned iter = 0; iter < (unsigned)channel.getBundleSize(); iter++) {
      rewriter.setInsertionPoint(channel);
      auto cname = createChannelName(device.getOperation());
      SmallVector<int64_t, 2> channel_sizes = {1, 1};
      auto new_chan = rewriter.create<air::ChannelOp>(
          channel->getLoc(), cname, rewriter.getI64ArrayAttr(channel_sizes));
      if (channel->hasAttr("broadcast_shape")) {
        auto broadcast_shape = specializeBroadcastShape(rewriter, channel);
        new_chan->setAttr("broadcast_shape",
                          rewriter.getArrayAttr(ArrayRef(broadcast_shape)));
      }
      std::vector<unsigned> position =
          getMDVectorFromIterator(bundle_size_stdvec, iter);
      for (auto put : channelPuts) {
        auto indices_uint = convertVecOfConstIndexToVecOfUInt(put.getIndices());
        if (areIdenticalVectors(indices_uint, position)) {
          // Found channel put for this channel
          rewriter.setInsertionPoint(put);
          auto new_put =
              createChannelPutGetWithoutBundle(rewriter, new_chan, put);
          if (put.getAsyncToken()) {
            replaceAllUsesInRegionWith(put.getAsyncToken(),
                                       new_put.getAsyncToken(),
                                       device.getRegion());
          }
        }
      }
      for (auto get : channelGets) {
        auto indices_uint = convertVecOfConstIndexToVecOfUInt(get.getIndices());
        if (areIdenticalVectors(indices_uint, position)) {
          // Found channel get for this channel
          rewriter.setInsertionPoint(get);
          auto new_get =
              createChannelPutGetWithoutBundle(rewriter, new_chan, get);
          if (get.getAsyncToken()) {
            replaceAllUsesInRegionWith(get.getAsyncToken(),
                                       new_get.getAsyncToken(),
                                       device.getRegion());
          }
        }
      }
    }

    // Erase bundled channel ops and their corresponding put/get ops
    for (auto put : channelPuts) {
      rewriter.eraseOp(put);
    }
    for (auto get : channelGets) {
      rewriter.eraseOp(get);
    }
    rewriter.eraseOp(channel);

    return success();
  }

private:
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

  std::vector<unsigned> convertToStdVec(SmallVector<long int, 4> vec) const {
    std::vector<unsigned> output;
    for (auto v : vec) {
      output.push_back((unsigned)v);
    }
    return output;
  }

  // Create channel name as string
  std::string createChannelName(Operation *scope) const {
    if (!scope->hasTrait<OpTrait::SymbolTable>()) {
      scope->emitOpError("has no symbol table trait");
    }
    std::string new_cname = "channel_0";
    std::string cname = "channel";
    int which_try = 0;
    while (mlir::SymbolTable::lookupSymbolIn(scope, new_cname))
      new_cname = cname + "_" + std::to_string(++which_try);
    cname = new_cname;
    return cname;
  }

  air::ChannelPutOp
  createChannelPutGetWithoutBundle(OpBuilder builder, air::ChannelOp chan,
                                   air::ChannelPutOp put) const {
    SmallVector<Type, 4> tys = {};
    SmallVector<Value, 4> deps = {};
    if (put.getAsyncToken()) {
      tys.push_back(air::AsyncTokenType::get(put->getContext()));
      deps = put.getAsyncDependencies();
    }
    SmallVector<Value, 4> indices = {};
    auto new_put = builder.create<air::ChannelPutOp>(
        put->getLoc(), tys, deps, chan.getSymName(), indices, put.getSrc(),
        put.getSrcOffsets(), put.getSrcSizes(), put.getSrcStrides());
    return new_put;
  }

  air::ChannelGetOp
  createChannelPutGetWithoutBundle(OpBuilder builder, air::ChannelOp chan,
                                   air::ChannelGetOp get) const {
    SmallVector<Type, 4> tys = {};
    SmallVector<Value, 4> deps = {};
    if (get.getAsyncToken()) {
      tys.push_back(air::AsyncTokenType::get(get->getContext()));
      deps = get.getAsyncDependencies();
    }
    SmallVector<Value, 4> indices = {};
    auto new_get = builder.create<air::ChannelGetOp>(
        get->getLoc(), tys, deps, chan.getSymName(), indices, get.getDst(),
        get.getDstOffsets(), get.getDstSizes(), get.getDstStrides());
    return new_get;
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
void specializeChannelBundle(AIE::DeviceOp &d) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<SpecializeChannelBundlePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(d, std::move(patterns));
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
  (void)applyPatternsAndFoldGreedily(d, std::move(patterns));
}

class AIRToAIEPass : public AIRToAIEBase<AIRToAIEPass> {

public:
  AIRToAIEPass() = default;
  AIRToAIEPass(const AIRToAIEPass &pass) {}

  typedef std::vector<
      std::tuple<AIE::BufferOp, AIE::DMAChannel, AIE::LockOp, AIE::LockOp>>
      lock_allocation_list;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
    registry.insert<xilinx::airrt::AIRRtDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  const int tile_dma_channels = 2;
  std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>>
      tile_dma_S2MM_allocs;
  std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>>
      tile_dma_MM2S_allocs;

  // A very simple scheme to allocate channels for dma operations:
  //  <description>
  AIE::DMAChannel getTileDMAChannel(AIE::DeviceOp aie_device,
                                    air::MemcpyInterface &memcpyOp, int col,
                                    int row) {
    bool isMM2S = isTileOutbound(memcpyOp);
    auto all_tile_dma_allocs =
        isMM2S ? &tile_dma_MM2S_allocs : &tile_dma_S2MM_allocs;

    int64_t chan = -1;

    unsigned num_allocs = 0;
    for (auto &t : *all_tile_dma_allocs) {
      if (col == std::get<1>(t) && row == std::get<2>(t)) {
        if (memcpyOp.getId() == std::get<0>(t))
          chan = std::get<3>(t);
        num_allocs++;
      }
    }
    if (chan == -1) {
      // Need to allocate a new one
      chan = num_allocs % tile_dma_channels;
      all_tile_dma_allocs->push_back({memcpyOp.getId(), col, row, chan});
      LLVM_DEBUG(llvm::outs()
                 << "  1 tile isMM2S = " << isMM2S << ", col =" << col
                 << ", row = " << row << ", tile chan =" << chan << "\n");
    }

    LLVM_DEBUG(llvm::outs()
               << "  2 tile isMM2S = " << isMM2S << ", col =" << col
               << ", row = " << row << ", tile chan =" << chan << "\n");

    if (isMM2S)
      return std::make_pair(AIE::DMAChannelDir::MM2S, (int)chan);
    else
      return std::make_pair(AIE::DMAChannelDir::S2MM, (int)chan);
  }

  AIE::BufferOp getBufferForTileDMA(AIE::DeviceOp aie_device,
                                    air::MemcpyInterface &memcpyOp, int col,
                                    int row) {
    Value buffer = isTileInbound(memcpyOp) ? (memcpyOp.getDstMemref())
                                           : (memcpyOp.getSrcMemref());
    AIE::BufferOp bufferOp = buffer.getDefiningOp<AIE::BufferOp>();
    return bufferOp;
  }

  // Allocate a reader/writer lock pair. These may be the same or different
  // locks depending on the target device.
  std::pair<AIE::LockOp, AIE::LockOp>
  getLockForTileDMA(AIE::DeviceOp device, air::MemcpyInterface &memcpyOp,
                    lock_allocation_list &info, int col, int row) {
    AIE::BufferOp bufferOp = getBufferForTileDMA(device, memcpyOp, col, row);
    AIE::DMAChannel channel = getTileDMAChannel(device, memcpyOp, col, row);
    assert(bufferOp);

    for (size_t i = 0; i < info.size(); i++) {
      if ((std::get<0>(info[i]) == bufferOp) &&
          (std::get<1>(info[i]) == channel)) {
        return {std::get<2>(info[i]), std::get<3>(info[i])};
      }
    }
    const auto &target_model = device.getTargetModel();
    bool isAIE2 = (target_model.getTargetArch() == AIE::AIEArch::AIE2);
    auto init = isAIE2 ? 1 : 0;

    OpBuilder builder(bufferOp);
    auto rlock = allocateLockOp(device, bufferOp.getTileOp(), 0);
    auto wlock =
        isAIE2 ? allocateLockOp(device, bufferOp.getTileOp(), init) : rlock;
    info.push_back({bufferOp, channel, rlock, wlock});
    return {rlock, wlock};
  }

  // get tileop using segment-relative coordinates
  AIE::TileOp getTileOp(AIE::DeviceOp aie_device, int herd_col, int herd_row) {
    int col = herd_col;
    int row = herd_row;
    return getPhysTileOp(aie_device, col, row);
  }

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
            sourceChannel == fop.getSourceChannel() &&
            destChannel == fop.getDestChannel())
          flowOp = fop;
    });
    if (flowOp)
      return flowOp;

    OpBuilder builder(aie_device);
    builder.setInsertionPointToEnd(aie_device.getBody());
    return builder.create<AIE::FlowOp>(builder.getUnknownLoc(), source,
                                       sourceBundle, sourceChannel, dest,
                                       destBundle, destChannel);
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

  template <typename T>
  std::map<AIE::DMAChannel, std::vector<Operation *>>
  getDmaSchedules(AIE::CoreOp core, int x, int y, DMAAllocator &shim_dma_alloc,
                  std::vector<AIE::TileOp> &shim_dma_inits) {

    std::map<AIE::DMAChannel, std::vector<Operation *>> tile_dma_copies;
    std::vector<Operation *> dma_memcpy_ops;
    getAIRMemcpyOpInRegion<T>(core.getBody(), dma_memcpy_ops);

    auto aie_device = core->getParentOfType<AIE::DeviceOp>();
    auto tile = core.getTileOp();

    for (auto o : dma_memcpy_ops) {

      auto memcpyOpIf = cast<air::MemcpyInterface>(o);
      if (!isLegalMemorySpace(memcpyOpIf)) {
        o->erase();
        continue;
      } else {

        AIE::DMAChannel tile_channel =
            getTileDMAChannel(aie_device, memcpyOpIf, x, y);

        // copy between L1 and external memory, use shim dma
        tile_channel = getTileDMAChannel(aie_device, memcpyOpIf, x, y);
        AIE::TileOp shim_tile = shim_dma_alloc.getTile(
            aie_device, memcpyOpIf, (int64_t)tile_channel.second, x, y);
        AIE::DMAChannel shim_channel = shim_dma_alloc.getChannel(
            aie_device, memcpyOpIf, tile_channel, x, y);

        LLVM_DEBUG(llvm::outs()
                   << "Shim channel is " << (uint64_t)shim_channel.second
                   << " for x=" << x << ", y=" << y << "\n");

        if ((shim_channel.first == AIE::DMAChannelDir::S2MM) &&
            ((uint64_t)shim_channel.second <
             (uint64_t)shim_dma_alloc.dma_channels)) {
          getFlowOp(
              aie_device, tile, AIE::WireBundle::DMA,
              (uint32_t)tile_channel.second, shim_tile, AIE::WireBundle::DMA,
              ((uint32_t)shim_channel.second) % shim_dma_alloc.dma_channels);
        } else {
          getFlowOp(aie_device, shim_tile, AIE::WireBundle::DMA,
                    ((uint32_t)shim_channel.second) %
                        shim_dma_alloc.dma_channels,
                    tile, AIE::WireBundle::DMA, (uint32_t)tile_channel.second);
        }
        tile_dma_copies[tile_channel].push_back(memcpyOpIf);
      }
    }
    return tile_dma_copies;
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
    return herd_meta;
  }

  template <typename T>
  void lowerAIRMemcpyOp(AIE::DeviceOp device, DMAAllocator &shimDmaAlloc) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : device.getOps<AIE::CoreOp>())
      cores.push_back(c);

    const auto &target_model = device.getTargetModel();
    bool isAIE2 = (target_model.getTargetArch() == AIE::AIEArch::AIE2);
    OpBuilder builder(device);

    for (AIE::CoreOp core : cores) {
      AIE::TileOp tile = core.getTileOp();
      auto x = tile.getCol();
      auto y = tile.getRow();

      std::vector<AIE::TileOp> shim_dma_inits;

      // collect dma operations and generate a schedule
      std::map<AIE::DMAChannel, std::vector<Operation *>> tile_dma_copies =
          getDmaSchedules<T>(core, x, y, shimDmaAlloc, shim_dma_inits);

      // emit the acquire and release of the L1 buffer locks
      lock_allocation_list lock_allocs;
      std::unordered_set<Operation *> allocs_to_remap;
      for (auto p : tile_dma_copies) {
        for (auto o : p.second) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          if (!memcpyOpIf)
            o->emitOpError("does not have air::MemcpyInterface");
          AIE::DMAChannel tile_channel =
              getTileDMAChannel(device, memcpyOpIf, x, y);
          auto locks = getLockForTileDMA(device, memcpyOpIf, lock_allocs, x, y);
          auto acqLockOp = isMM2S(tile_channel) ? locks.second : locks.first;
          auto relLockOp = isMM2S(tile_channel) ? locks.first : locks.second;
          int64_t lockAqValue = -1;
          int64_t lockRelValue = -1;
          Value alloc = nullptr;
          if (isTileInbound(memcpyOpIf)) {
            lockAqValue = isAIE2 ? 1 : 1;
            lockRelValue = isAIE2 ? 1 : 0;
            alloc = memcpyOpIf.getDstMemref();
          } else {
            lockAqValue = isAIE2 ? 1 : 0;
            lockRelValue = isAIE2 ? 1 : 1;
            alloc = memcpyOpIf.getSrcMemref();
          }

          if (auto bco =
                  dyn_cast<bufferization::ToMemrefOp>(alloc.getDefiningOp()))
            builder.setInsertionPoint(bco.getOperand().getDefiningOp());
          else if (auto a = dyn_cast<memref::AllocaOp>(alloc.getDefiningOp()))
            builder.setInsertionPoint(alloc.getDefiningOp());
          else
            builder.setInsertionPoint(&memcpyOpIf->getBlock()->front());

          builder.create<AIE::UseLockOp>(
              o->getLoc(), acqLockOp, lockAqValue,
              isAIE2 ? AIE::LockAction::AcquireGreaterEqual
                     : AIE::LockAction::Acquire);
          // try to find a place to put the unlock. If there are deallocs,
          // replace them with unlock. Otherwise, put them at the end.
          bool need_unlock = true;
          for (auto u : alloc.getUsers()) {
            if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
              builder.setInsertionPoint(dealloc);
              builder.create<AIE::UseLockOp>(dealloc->getLoc(), relLockOp,
                                             lockRelValue,
                                             AIE::LockAction::Release);
              // assume that the deallocs will take care of it when
              // deallocs are present
              need_unlock = false;
            }
          }
          if (need_unlock) {
            auto t = memcpyOpIf->getBlock()->getTerminator();
            builder.setInsertionPoint(t);
            builder.create<AIE::UseLockOp>(t->getLoc(), relLockOp, lockRelValue,
                                           AIE::LockAction::Release);
          }
          allocs_to_remap.insert(alloc.getDefiningOp());
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

      // Generate the TileDMA bd program. That is, generate the AIE.mem
      // body for the tile. Above we collected per channel lists of dma
      // copy operations. We'll assume these lists are in the correct
      // execution order and generate a AIE.mem program to loop over
      // each list.

      // The first block
      Block *channel_head = nullptr;
      Block *end_bb = nullptr;

      auto loc = core->getLoc();

      // make a AIE.mem for the tile dma
      auto mem = tile.getMemOp();
      if (!mem && tile_dma_copies.size()) {
        builder.setInsertionPoint(core);
        mem = builder.create<AIE::MemOp>(loc, tile);
      }
      for (auto &p : tile_dma_copies) {
        auto channel = p.first;

        LLVM_DEBUG(llvm::outs()
                   << " TILE dma channel is " << (uint64_t)channel.second
                   << " for x=" << x << ", y=" << y << "\n");

        Block *start_bb = new Block();
        mem.getBody().push_back(start_bb);

        Block *first_bd = new Block();
        mem.getBody().push_back(first_bd);
        auto memcpyOps = p.second;
        Block *next_bd = nullptr;
        for (size_t i = 0; i < memcpyOps.size(); i++) {
          auto memcpyOp = cast<air::MemcpyInterface>(memcpyOps[i]);
          Block *bd;
          if (i == 0)
            bd = first_bd;
          else
            bd = next_bd;
          auto b = OpBuilder::atBlockEnd(bd);
          if (i == memcpyOps.size() - 1) {
            b.create<AIE::NextBDOp>(loc, first_bd);
          } else {
            next_bd = new Block();
            mem.getBody().push_back(next_bd);
            b.create<AIE::NextBDOp>(loc, next_bd);
          }
          AIE::BufferOp bufferOp = getBufferForTileDMA(device, memcpyOp, x, y);
          auto locks = getLockForTileDMA(device, memcpyOp, lock_allocs, x, y);
          auto acqLockOp = isMM2S(channel) ? locks.first : locks.second;
          auto relLockOp = isMM2S(channel) ? locks.second : locks.first;
          b.setInsertionPointToStart(bd);
          int64_t lockAqValue = -1;
          int64_t lockRelValue = -1;
          if (!isMM2S(channel)) {
            lockAqValue = isAIE2 ? 1 : 0;
            lockRelValue = isAIE2 ? 1 : 1;
          } else {
            lockAqValue = isAIE2 ? 1 : 1;
            lockRelValue = isAIE2 ? 1 : 0;
          }
          auto ndcpy = cast<air::MemcpyInterface>(memcpyOps[i]);

          int64_t len = isTileInbound(ndcpy)
                            ? getMemcpySizesAsInt(ndcpy.getDstMemref(),
                                                  ndcpy.getDstSizes())
                            : getMemcpySizesAsInt(ndcpy.getSrcMemref(),
                                                  ndcpy.getSrcSizes());

          Value length =
              b.create<arith::ConstantIndexOp>(memcpyOp.getLoc(), len)
                  ->getResult(0);
          b.create<AIE::UseLockOp>(loc, acqLockOp, lockAqValue,
                                   isAIE2 ? AIE::LockAction::AcquireGreaterEqual
                                          : AIE::LockAction::Acquire);
          b.create<AIE::DMABDOp>(
              loc, bufferOp, 0,
              cast<arith::ConstantIndexOp>(length.getDefiningOp()).value(), 0);
          b.create<AIE::UseLockOp>(loc, relLockOp, lockRelValue,
                                   AIE::LockAction::Release);
        }
        if (!channel_head) {
          channel_head = start_bb;
          end_bb = new Block();
          mem.getBody().push_back(end_bb);
          auto b = OpBuilder::atBlockBegin(channel_head);
          b.create<AIE::DMAStartOp>(loc, channel.first, channel.second,
                                    first_bd, end_bb);
          b.setInsertionPointToEnd(end_bb);
          b.create<AIE::EndOp>(loc);
        } else {
          auto b = OpBuilder::atBlockBegin(start_bb);
          b.create<AIE::DMAStartOp>(
              loc, channel.first, channel.second, first_bd,
              channel_head->getTerminator()->getSuccessor(1));
          channel_head->getTerminator()->setSuccessor(start_bb, 1);
        }
      }

      // erase the dma copy operations
      for (auto p : tile_dma_copies) {
        for (auto o : p.second) {
          auto a = cast<xilinx::air::AsyncOpInterface>(o);
          if (a.getAsyncToken()) {
            OpBuilder b(o);
            o->replaceAllUsesWith(b.create<xilinx::air::WaitAllOp>(
                o->getLoc(), air::AsyncTokenType::get(o->getContext()),
                a.getAsyncDependencies()));
          }
          o->erase();
        }
      }
    }
  }

  void runTestPatterns() {

    auto m = getOperation();
    auto ctx = m->getContext();

    RewritePatternSet patterns(ctx);
    std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;

    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      m.emitOpError("Invalid AIE.device option");
      signalPassFailure();
      return;
    }

    if (clTestPatterns.find("to-aie-mlir") != std::string::npos) {
      std::vector<std::pair<AIE::DeviceOp, air::HerdOp>> aie_modules;
      std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
      AIRToAIEOptions options = {.col_offset = clColOffset,
                                 .row_offset = clRowOffset,
                                 .emit_while = clEmitWhileLoop,
                                 .emit_herd_lock = clEmitHerdLock,
                                 .device = *device};
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
      }
    }

    if (clTestPatterns.find("lower-air-execute") != std::string::npos)
      patterns.insert<LowerAIRExecutePattern>(ctx);
    if (clTestPatterns.find("alloc-l1-buffers") != std::string::npos)
      patterns.insert<AllocL1BuffersPattern, AllocL1BuffersPattern>(
          ctx, tileToHerdMap);
    if (clTestPatterns.find("specialize-affine-if") != std::string::npos)
      patterns.insert<SpecializeAffineIfPattern>(ctx);
    if (clTestPatterns.find("lower-pipe-get-put") != std::string::npos)
      patterns.insert<LowerPipeGetPutPattern>(ctx, tileToHerdMap);
    if (clTestPatterns.find("lower-scf-tokens") != std::string::npos)
      patterns.insert<LowerScfTokenPattern>(ctx);

    OpBuilder builder(ctx);
    AIE::DeviceOp deviceOp = builder.create<AIE::DeviceOp>(
        builder.getUnknownLoc(),
        AIE::AIEDeviceAttr::get(builder.getContext(), *device));
    ShimTileAllocator shimTileAlloc(deviceOp.getTargetModel());
    if (clTestPatterns.find("lower-air-channels") != std::string::npos) {
      patterns.insert<LowerAIRChannelsPattern>(ctx, shimTileAlloc);
    }
    if (clTestPatterns.find("lower-air-ping-pong") != std::string::npos) {
      patterns.insert<LowerAIRPingPongPattern>(ctx);
    }
    if (clTestPatterns.find("specialize-channel-bundle") != std::string::npos) {
      patterns.insert<SpecializeChannelBundlePattern>(ctx);
    }

    if (patterns.getNativePatterns().size())
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
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
    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      module.emitOpError("Invalid AIE.device option");
      signalPassFailure();
      return;
    }
    AIRToAIEOptions options = {.col_offset = clColOffset,
                               .row_offset = clRowOffset,
                               .emit_while = clEmitWhileLoop,
                               .emit_herd_lock = clEmitHerdLock,
                               .device = *device};
    createAIEModulesAndOutlineCores(module, aie_devices, tileToHerdMap,
                                    options);

    std::set<AIE::DeviceOp> seen;
    for (auto &p : aie_devices) {
      auto device = std::get<0>(p);
      xilinx::air::HerdOp h = std::get<1>(p);
      auto ctx = device->getContext();

      if (seen.find(device) != seen.end())
        continue;
      seen.insert(device);

      specializeHerdAffineIf(device);
      lowerAirExecute(device);
      lowerScfAirTokens(device);

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

      DMAAllocator shimDmaAlloc(device.getTargetModel());

      specializeChannelBundle(device);

      if (clUseObjFifo) {
        LowerAIRPingPong(device);
        ShimTileAllocator shimTileAlloc(device.getTargetModel());
        lowerAIRChannels(device, shimTileAlloc);
        allocL1Buffers(device, tileToHerdMap);
      } else {
        allocL1Buffers(device, tileToHerdMap);
        lowerAIRMemcpyOp<air::DmaMemcpyNdOp>(device, shimDmaAlloc);
        lowerAIRMemcpyOp<air::ChannelInterface>(device, shimDmaAlloc);
      }

      lowerPipelineGetPut(device, tileToHerdMap);

      SmallVector<air::HerdOp, 4> herds;
      if (auto p = h->getParentOfType<air::SegmentOp>()) {
        auto hops = p.getOps<air::HerdOp>();
        herds.append(hops.begin(), hops.end());
      } else {
        herds.push_back(h);
      }

      for (auto herd : herds) {
        std::set<int64_t> dma_ids;
        herd.walk([&](Operation *o) {
          if (auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(o))
            dma_ids.insert(dmaOp.getId());
        });
        auto c = herd.getColOffset();
        auto r = herd.getRowOffset();
        int64_t col_offset = c ? *c : 0;
        int64_t row_offset = r ? *r : 0;

        // createAIRRtMetadata(module_meta, shimDmaAlloc);
        std::vector<Attribute> dma_allocations;
        for (auto &t : shimDmaAlloc.s2mm_allocs) {
          auto tileOp = t.dma_tile;
          int64_t col = t.col - col_offset;
          int64_t row = t.row - row_offset;
          int64_t chan = t.dma_channel;

          for (int64_t id : t.dma_id) {
            if (dma_ids.count(id) == 0)
              continue;
            SmallVector<NamedAttribute, 5> attrs;
            attrs.push_back(NamedAttribute(StringAttr::get(ctx, "id"),
                                           builder.getI64IntegerAttr(id)));
            attrs.push_back(NamedAttribute(StringAttr::get(ctx, "row"),
                                           builder.getI64IntegerAttr(row)));
            attrs.push_back(NamedAttribute(StringAttr::get(ctx, "col"),
                                           builder.getI64IntegerAttr(col)));
            attrs.push_back(NamedAttribute(StringAttr::get(ctx, "channel"),
                                           builder.getI64IntegerAttr(chan)));
            attrs.push_back(
                NamedAttribute(StringAttr::get(ctx, "location"),
                               builder.getI64IntegerAttr(tileOp.getCol())));
            dma_allocations.push_back(DictionaryAttr::get(ctx, attrs));
          }
        }
        for (auto &t : shimDmaAlloc.mm2s_allocs) {
          auto tileOp = t.dma_tile;
          int64_t col = t.col - col_offset;
          int64_t row = t.row - row_offset;
          int64_t chan = t.dma_channel;
          for (int64_t id : t.dma_id) {
            if (dma_ids.count(id) == 0)
              continue;
            SmallVector<NamedAttribute, 5> attrs;
            attrs.push_back(NamedAttribute(StringAttr::get(ctx, "id"),
                                           builder.getI64IntegerAttr(id)));
            attrs.push_back(NamedAttribute(StringAttr::get(ctx, "row"),
                                           builder.getI64IntegerAttr(row)));
            attrs.push_back(NamedAttribute(StringAttr::get(ctx, "col"),
                                           builder.getI64IntegerAttr(col)));
            attrs.push_back(
                NamedAttribute(StringAttr::get(ctx, "channel"),
                               builder.getI64IntegerAttr(chan + 2)));
            attrs.push_back(
                NamedAttribute(StringAttr::get(ctx, "location"),
                               builder.getI64IntegerAttr(tileOp.getCol())));
            dma_allocations.push_back(DictionaryAttr::get(ctx, attrs));
          }
        }
        auto segment_name =
            device->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                .getValue();
        auto segment_meta =
            getOrCreateSegmentMetadata(module_meta, segment_name);
        auto herd_meta = createHerdMetadata(segment_meta, herd);
        herd_meta->setAttr("dma_allocations",
                           ArrayAttr::get(ctx, dma_allocations));
      }
      tile_dma_S2MM_allocs.clear();
      tile_dma_MM2S_allocs.clear();

      RewritePatternSet patterns(ctx);
      air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsAndFoldGreedily(device, std::move(patterns));
    }
  }
};

template <typename OpT>
struct OpRemovalPattern : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpT::Adaptor;

  OpRemovalPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<OpT>(context, benefit) {}

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class SplitAIEDevicesPass : public AIRSplitDevicesBase<SplitAIEDevicesPass> {

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
      target.addIllegalDialect<xilinx::airrt::AIRRtDialect>();
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

} // namespace

namespace xilinx {
namespace air {

FailureOr<ModuleOp> convertAIRToAIE(mlir::RewriterBase &rewriter,
                                    air::SegmentOp p) {
  std::string segment_name = "segment_0";
  if (auto attr =
          p->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    segment_name = attr.getValue().str();

  std::string aie_module_name = "aie." + segment_name;
  ModuleOp aie_module =
      ModuleOp::create(rewriter.getUnknownLoc(), StringRef(aie_module_name));

  auto device = AIE::symbolizeAIEDevice("xcvc1902");
  if (!device) {
    p->emitOpError("Invalid AIE.device option");
    return failure();
  }
  AIRToAIEOptions options = {
      .col_offset = 7, .row_offset = 2, .emit_while = false, .device = *device};
  std::vector<std::pair<ModuleOp, xilinx::air::HerdOp>> aie_modules;
  p.walk([&](xilinx::air::HerdOp h) {
    aie_modules.push_back({aie_module, h});
  });
  std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
  for (auto &p : aie_modules) {
    ModuleOp aie_module = std::get<0>(p);
    xilinx::air::HerdOp h = std::get<1>(p);
    rewriter.setInsertionPointToStart(aie_module.getBody());
    auto devOp = rewriter.create<AIE::DeviceOp>(
        aie_module.getLoc(),
        AIE::AIEDeviceAttr::get(rewriter.getContext(), options.device));
    devOp.getRegion().emplaceBlock();
    outlineAIECores(rewriter, devOp, h, tileToHerdMap, options);

    auto ctx = aie_module->getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<SpecializeAffineIfPattern>(ctx);
    patterns.insert<LowerAIRExecutePattern>(ctx);
    patterns.insert<AllocL1BuffersPattern>(ctx, tileToHerdMap);
    air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(aie_module, std::move(patterns));
  }

  return aie_module;
}

std::unique_ptr<mlir::Pass> createAIRToAIEPass() {
  return std::make_unique<AIRToAIEPass>();
}

std::unique_ptr<mlir::Pass> createAIRSplitDevicesPass() {
  return std::make_unique<SplitAIEDevicesPass>();
}

} // namespace air
} // namespace xilinx
