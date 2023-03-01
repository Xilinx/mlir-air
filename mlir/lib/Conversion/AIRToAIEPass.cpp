//===- AIRToAIEPass.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

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
};

// std::vector<int> s80_nmu_col_list{0, 0, 1, 1, 0, 0, 1, 1,
//                                   0, 0, 1, 1, 0, 0, 0, 0,
//                                   0, 0, 1, 1, 0, 0, 0, 0,
//                                   0, 0, 1, 1, 0, 0, 0, 0,
//                                   0, 0, 1, 1, 0, 0, 0, 0,
//                                   0, 0, 1, 1, 0, 0, 1, 1,
//                                   0, 0};
std::vector<int> shim_dma_cols{2,  3,  6,  7,  10, 11, 18, 19,
                               26, 27, 34, 35, 42, 43, 46, 47};
const int shim_dma_channels = 2;

AIE::TileOp getPhysTileOpOrNull(ModuleOp aie_module, int col, int row) {
  for (auto t : aie_module.getOps<AIE::TileOp>()) {
    if (t.colIndex() == col && t.rowIndex() == row)
      return t;
  }
  return nullptr;
}

// get tileop using physical coordinates
AIE::TileOp getPhysTileOp(ModuleOp aie_module, int col, int row) {
  auto t = getPhysTileOpOrNull(aie_module, col, row);
  if (t)
    return t;

  OpBuilder builder(aie_module);

  builder.setInsertionPointToStart(aie_module.getBody());
  for (auto &o : aie_module.getBody()->getOperations()) {
    if (isa<AIE::TileOp>(o))
      builder.setInsertionPointAfter(&o);
    else
      break;
  }
  return builder.create<AIE::TileOp>(UnknownLoc::get(aie_module.getContext()),
                                     col, row);
}

struct ShimTileAllocator {

  std::vector<int> shim_columns;
  int shim_dma_channels;

  struct shim_allocation_info_t {
    AIE::TileOp shim_tile;
    int available_channels;
  };

  std::vector<shim_allocation_info_t> mm2s_allocs, s2mm_allocs;

  ShimTileAllocator(std::vector<int> cols, int channels)
      : shim_columns(cols), shim_dma_channels(channels) {}

  AIE::TileOp getShimTile(ModuleOp aie_module, int src_memory_space,
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
    auto shim_tile = getPhysTileOp(aie_module, shim_col, 0);
    allocs->push_back({shim_tile, shim_dma_channels - 1});

    return shim_tile;
  }
};

bool isMM2S(AIE::DMAChannel channel) {
  return (channel.first == AIE::DMAChannelDir::MM2S);
}

struct DMAAllocator {

  std::vector<int> dma_columns;
  int dma_channels;

  struct allocation_info_t {
    AIE::TileOp dma_tile;
    int64_t col;
    int64_t row;
    int64_t dma_channel;
    int64_t tile_channel;
    std::vector<int32_t> dma_id;
  };

  std::vector<allocation_info_t> mm2s_allocs, s2mm_allocs;

  DMAAllocator(std::vector<int> cols, int channels)
      : dma_columns(cols), dma_channels(channels) {}

  AIE::TileOp getTile(ModuleOp aie_module, air::DmaMemcpyInterface &dmaOp,
                      int64_t tile_channel, int64_t col, int64_t row) {
    auto src_memory_space =
        dmaOp.getSrcMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    auto dst_memory_space =
        dmaOp.getDstMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    assert(src_memory_space != dst_memory_space);

    bool isMM2S = (src_memory_space < dst_memory_space);
    auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

    for (auto &t : *allocs) {
      if (col == t.col && row == t.row) {
        for (auto id : t.dma_id)
          if (dmaOp.getId() == id)
            return t.dma_tile;
        if (tile_channel == t.tile_channel) {
          t.dma_id.push_back(dmaOp.getId());
          return t.dma_tile;
        }
      }
    }
    auto dma_col = dma_columns[allocs->size() / dma_channels];
    auto dma_channel = allocs->size() % dma_channels;
    auto dma_tile = getPhysTileOp(aie_module, dma_col, 0);
    allocs->push_back({dma_tile,
                       col,
                       row,
                       (int64_t)dma_channel,
                       tile_channel,
                       {dmaOp.getId()}});
    LLVM_DEBUG(llvm::outs() << "isMM2S = " << isMM2S << " " << dmaOp.getId()
                            << ", col =" << col << ", row = " << row
                            << ", dma_col =" << dma_col
                            << ", dma_chan =" << dma_channel << "\n");

    return dma_tile;
  }

  AIE::DMAChannel getChannel(ModuleOp aie_module,
                             air::DmaMemcpyInterface &dmaOp,
                             AIE::DMAChannel tile_channel, int64_t col,
                             int64_t row) {
    auto src_memory_space =
        dmaOp.getSrcMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    auto dst_memory_space =
        dmaOp.getDstMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    assert(src_memory_space != dst_memory_space);

    bool isMM2S = (src_memory_space < dst_memory_space);
    auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

    int64_t chan = -1;
    for (auto &t : *allocs) {
      LLVM_DEBUG(llvm::outs()
                 << "gSDC: op " << t.dma_tile << ", col" << t.col << ", row "
                 << t.row << ", chan " << t.dma_channel << "\n");
      if (col == t.col && row == t.row) {
        for (auto id : t.dma_id)
          if (dmaOp.getId() == id)
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

AIE::LockOp allocateLockOp(ModuleOp aie_module, AIE::TileOp tile, int id = -1) {
  AIE::LockOp lock = nullptr;
  std::set<int> ids;
  aie_module.walk([&](AIE::LockOp l) {
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

  OpBuilder b(aie_module);
  Operation *t = tile.getOperation();
  while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
    t = t->getNextNode();
  b.setInsertionPointAfter(t);
  return b.create<AIE::LockOp>(tile.getLoc(), tile, new_id);
}

void outlineAIECores(OpBuilder &builder, ModuleOp aie_module,
                     xilinx::air::HerdOp h,
                     std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
                     AIRToAIEOptions &options) {
  builder.setInsertionPointToStart(aie_module.getBody());

  int64_t herd_size_x = h.getNumCols();
  int64_t herd_size_y = h.getNumRows();

  h.walk([&](air::ChannelInterface op) {
    if (!aie_module.lookupSymbol(op.getChanName())) {
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
      auto tile = getPhysTileOp(aie_module, phys_x, phys_y);

      Operation *t = tile.getOperation();
      while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
        t = t->getNextNode();
      builder.setInsertionPointAfter(t);

      // make the AIE.core for the tile core
      auto core = tile.getCoreOp();
      if (!core) {
        core = builder.create<AIE::CoreOp>(hloc, tile);
        tileToHerdMap[tile] = h;
        std::string herd_name =
            aie_module.getName()->str().substr(strlen("aie."));
        core->setAttr("elf_file",
                      StringAttr::get(aie_module.getContext(),
                                      herd_name + "_core_" +
                                          std::to_string(phys_x) + "_" +
                                          std::to_string(phys_y) + ".elf"));
        if (auto a = h->getAttrOfType<StringAttr>("link_with"))
          core->setAttr("link_with", a);
      }

      Value herd_lock = nullptr;
      if (options.emit_herd_lock)
        herd_lock = allocateLockOp(aie_module, tile, 0);

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

        OpBuilder b(aie_module);
        b.setInsertionPoint(core);

        int which_try = 0;
        std::string sym_name = "__air_herd_arg_0";
        while (aie_module.lookupSymbol(sym_name))
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
          auto fn = aie_module.lookupSymbol<func::FuncOp>(call.getCallee());
          if (!fn) {
            fn = func::FuncOp::create(aie_module.getLoc(), call.getCallee(),
                                      call.getCalleeType());
            fn.setPrivate();
            aie_module.push_back(fn);
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
    std::vector<std::pair<ModuleOp, xilinx::air::HerdOp>> &aie_modules,
    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
    AIRToAIEOptions &options) {

  module.walk([&](xilinx::air::PartitionOp p) {
    std::string partition_name;
    if (auto attr =
            p->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      partition_name = attr.getValue().str();
    else
      partition_name = "partition_" + std::to_string(aie_modules.size());
    std::string aie_module_name = "aie." + partition_name;
    ModuleOp aie_module =
        ModuleOp::create(module.getLoc(), StringRef(aie_module_name));

    p.walk([&](xilinx::air::HerdOp h) {
      aie_modules.push_back({aie_module, h});
    });
  });

  module.walk([&](xilinx::air::HerdOp h) {
    if (h->getParentOfType<xilinx::air::PartitionOp>())
      return;
    std::string partition_name;
    partition_name = "partition_" + std::to_string(aie_modules.size());
    std::string aie_module_name = "aie." + partition_name;
    ModuleOp aie_module =
        ModuleOp::create(module.getLoc(), StringRef(aie_module_name));

    aie_modules.push_back({aie_module, h});
  });

  for (auto &p : aie_modules) {
    ModuleOp aie_module = std::get<0>(p);
    xilinx::air::HerdOp h = std::get<1>(p);
    OpBuilder builder(aie_module);
    outlineAIECores(builder, aie_module, h, tileToHerdMap, options);
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

void specializeHerdAffineIf(ModuleOp m) {
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

void lowerAirExecute(ModuleOp m) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<LowerAIRExecutePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
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

void lowerScfAirTokens(ModuleOp m) {
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
    auto aie_module = put->getParentOfType<ModuleOp>();
    auto core = put->getParentOfType<AIE::CoreOp>();
    assert(aie_module && core);

    auto herd = tileToHerdMap[core.getTileOp()];
    auto c = herd.getColOffset();
    auto r = herd.getRowOffset();
    auto col_offset = c ? *c : 0;
    auto row_offset = r ? *r : 0;

    auto other_x = cast<arith::ConstantIndexOp>(put.getDst0().getDefiningOp());
    auto other_y = cast<arith::ConstantIndexOp>(put.getDst1().getDefiningOp());
    auto other_core = getPhysTileOp(aie_module, other_x.value() + col_offset,
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
            StringAttr::get(aie_module.getContext(), "pipebuf"));
        auto lockOp = allocateLockOp(aie_module, core.getTileOp());

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
void lowerPipelineGetPut(ModuleOp &m,
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

void allocL1Buffers(ModuleOp m,
                    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<AllocL1BuffersPattern, AllocL1TensorsPattern>(ctx,
                                                                tileToHerdMap);
  (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
}

void
createObjectFifo(OpBuilder &builder, AIE::AIEObjectFifoType datatype,
                 std::map<std::pair<int, int>, AIE::ObjectFifoCreateOp> &objFifos,
                 std::map<std::pair<int, int>, Value> &prodTiles, 
                 std::map<std::pair<int, int>, Value> &consTiles, 
                 int depth, bool broadcast) {
  // channels and objFifos currently don't support many-to-many/one broadcast
  if (broadcast && prodTiles.size() > 1)
    return;
  
  for (auto const& [coord, tile] : prodTiles) {
    AIE::ObjectFifoCreateOp fifo = builder.create<AIE::ObjectFifoCreateOp>(
      builder.getUnknownLoc(), datatype, prodTiles[coord], consTiles[coord], depth);
    objFifos[coord] = fifo;
  } 
}

template <typename MyOp>
void rewriteChannelAllocs(PatternRewriter &rewriter, MyOp op,
                          AIE::ObjectFifoCreateOp objFifo,
                          AIE::ObjectFifoPort port) {
  auto elementType =
      objFifo.getType().dyn_cast<AIE::AIEObjectFifoType>().getElementType();
  auto acqType = AIE::AIEObjectFifoSubviewType::get(elementType);

  rewriter.setInsertionPoint(&op->getBlock()->front());
  AIE::ObjectFifoAcquireOp producerAcq =
      rewriter.create<AIE::ObjectFifoAcquireOp>(rewriter.getUnknownLoc(),
                                                acqType, port, objFifo, 1);
  rewriter.setInsertionPointAfter(producerAcq);
  AIE::ObjectFifoSubviewAccessOp producerAccess =
      rewriter.create<AIE::ObjectFifoSubviewAccessOp>(
          rewriter.getUnknownLoc(), elementType, producerAcq.getSubview(),
          rewriter.getIntegerAttr(rewriter.getI32Type(), 0));

  // replace uses of alloc with result of acquire
  if (auto a = dyn_cast<memref::AllocOp>(op.getMemref().getDefiningOp()))
    rewriter.replaceOp(a.getOperation(), producerAccess.getOutput());
}

template <typename MyOp>
void rewriteChannelDeallocs(PatternRewriter &rewriter, MyOp op,
                            AIE::ObjectFifoCreateOp objFifo,
                            AIE::ObjectFifoPort port) {
  for (auto u : op.getMemref().getDefiningOp()->getUsers()) {
    if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
      rewriter.setInsertionPoint(&op->getBlock()->back());
      rewriter.create<AIE::ObjectFifoReleaseOp>(dealloc->getLoc(), port,
                                                objFifo, 1);
      rewriter.eraseOp(dealloc);
    }
  }
}

struct LowerAIRChannelsPattern : public OpRewritePattern<air::ChannelOp> {
  using OpRewritePattern<air::ChannelOp>::OpRewritePattern;

  LowerAIRChannelsPattern(MLIRContext *ctx, ShimTileAllocator &shimTileAlloc)
      : OpRewritePattern(ctx), shimTileAlloc(shimTileAlloc) {}

  LogicalResult matchAndRewrite(air::ChannelOp channel,
                                PatternRewriter &rewriter) const override {
    // for now, the lowering does not support broadcast
    if (channel.isBroadcast()) 
      return failure();

    auto aie_module = channel->getParentOfType<ModuleOp>();
    int numSubchannels = channel.getSubchannelSize();
    std::vector<ChannelPutOp> channelPuts =
        getChannelPutOpThroughSymbol(channel);
    std::vector<ChannelGetOp> channelGets =
        getChannelGetOpThroughSymbol(channel);

    if ((int)channelPuts.size() > numSubchannels ||
        (int)channelGets.size() > numSubchannels)
        return failure(); // TODO: assert instead?

    std::map<std::pair<int, int>, Value> producerTiles;
    std::map<std::pair<int, int>, Value> consumerTiles;
    std::map<std::pair<int, int>, AIE::ObjectFifoCreateOp> objFifos;
    AIE::AIEObjectFifoType datatype;

    // put/get come in pairs, if one is missing then it's L3
    MemRefType srcMemref;
    int src_space = (int)air::MemorySpace::L3;
    Value producerTile;
    if (channelPuts.size() > 0) {
      for (auto put : channelPuts) {
        // find AIE tiles and their cores based on memory hierarchy levels
        srcMemref = put.getSrc().getType().cast<MemRefType>();
        src_space = srcMemref.getMemorySpaceAsInt();
        datatype = AIE::AIEObjectFifoType::get(MemRefType::get(srcMemref.getShape(), srcMemref.getElementType()));
        if (src_space == (int)air::MemorySpace::L1) {
          AIE::CoreOp producerCore = put->getParentOfType<AIE::CoreOp>();
          if (!producerCore)
            return failure();
          producerTile = producerCore.getTileOp();
          if (!producerTile)
            return failure();
          
          int coord_x = 0;
          int coord_y = 0;
          if (put.getIndices().size() > 0) {
            coord_x = put.getSizeX();
            coord_y = put.getSizeY();
          }
          producerTiles[std::make_pair(coord_x, coord_y)] = producerTile;
        } else {
          return failure();
        }
      }
    } else {
      // put from L3
      producerTile = shimTileAlloc.getShimTile(
          aie_module, src_space, (int)air::MemorySpace::L1);
      producerTiles[{0, 0}] = producerTile;
    }

    // put/get come in pairs, if one is missing then it's L3
    MemRefType dstMemref;
    int dst_space = (int)air::MemorySpace::L3;
    Value consumerTile;
    if (channelGets.size() > 0) {
      for (auto get : channelGets) {
        // find AIE tiles and their cores based on memory hierarchy levels
        dstMemref = get.getDst().getType().cast<MemRefType>();
        dst_space = dstMemref.getMemorySpaceAsInt();
        datatype = AIE::AIEObjectFifoType::get(MemRefType::get(dstMemref.getShape(), dstMemref.getElementType()));
        if (dst_space == (int)air::MemorySpace::L1) {
          AIE::CoreOp consumerCore = get->getParentOfType<AIE::CoreOp>();
          if (!consumerCore)
            return failure();
          consumerTile = consumerCore.getTileOp();
          if (!consumerTile)
            return failure();
          
          int coord_x = 0;
          int coord_y = 0;
          if (get.getIndices().size() > 0) {
            coord_x = get.getSizeX();
            coord_y = get.getSizeY();
          }
          consumerTiles[std::make_pair(coord_x, coord_y)] = consumerTile;
        } else {
          return failure();
        }
      }
    } else {
      // get from L3
      consumerTile = shimTileAlloc.getShimTile(
          aie_module, (int)air::MemorySpace::L1, dst_space);
      consumerTiles[{0, 0}] = consumerTile;
    }

    if (!datatype)
      return failure();

    // create objFifo
    // For now, number of memory elements in OF is hardcoded to 1
    // (single buffer). FIXME
    rewriter.setInsertionPoint(*(aie_module.getOps<AIE::CoreOp>().begin()));
    createObjectFifo(rewriter, datatype, objFifos, producerTiles, consumerTiles, 1, false);

    // replace put/get and the associated memref alloc/dealloc
    for (auto put : channelPuts) {
      int coord_x = 0;
      int coord_y = 0;
      if (put.getIndices().size() > 0) {
        coord_x = put.getSizeX();
        coord_y = put.getSizeY();
      }
      rewriteChannelAllocs<ChannelPutOp>(rewriter, put, 
                                         objFifos[std::make_pair(coord_x, coord_y)],
                                         AIE::ObjectFifoPort::Produce);
      rewriteChannelDeallocs<ChannelPutOp>(rewriter, put, 
                                           objFifos[std::make_pair(coord_x, coord_y)],
                                           AIE::ObjectFifoPort::Produce);
      // erase put
      rewriter.eraseOp(put);
    }
    for (auto get : channelGets) {
      int coord_x = 0;
      int coord_y = 0;
      if (get.getIndices().size() > 0) {
        coord_x = get.getSizeX();
        coord_y = get.getSizeY();
      }
      rewriteChannelAllocs<ChannelGetOp>(rewriter, get, 
                                         objFifos[std::make_pair(coord_x, coord_y)],
                                         AIE::ObjectFifoPort::Consume);
      rewriteChannelDeallocs<ChannelGetOp>(rewriter, get, 
                                           objFifos[std::make_pair(coord_x, coord_y)],
                                           AIE::ObjectFifoPort::Consume);
      // erase get
      rewriter.eraseOp(get);
    }
    // erase the channel
    rewriter.eraseOp(channel);
    return success();
  }

private:
  ShimTileAllocator &shimTileAlloc;
};

// This function replaces ChannelPutOp/ChannelGetOp with AIE_CreateObjectFifoOps
// and with ObjectFifoAcquireOp<Producer/Consumer>. It also erases memref allocs
// as the objFifo lowering allocates its own memory. It replaces the associated
// memref deallocs with ObjectFifoReleaseOps. void lowerAIRChannels(ModuleOp &m)
// {
//   auto ctx = m->getContext();
//   RewritePatternSet patterns(ctx);
//   patterns.insert<LowerAIRChannelsPattern>(ctx);
//   (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
// }

class AIRToAIEPass : public AIRToAIEBase<AIRToAIEPass> {

public:
  AIRToAIEPass() = default;
  AIRToAIEPass(const AIRToAIEPass &pass) {}

  typedef std::vector<std::tuple<AIE::BufferOp, AIE::LockOp, AIE::DMAChannel>>
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
  AIE::DMAChannel getTileDMAChannel(ModuleOp aie_module,
                                    air::DmaMemcpyInterface &dmaOp, int col,
                                    int row) {
    auto src_memory_space =
        dmaOp.getSrcMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    auto dst_memory_space =
        dmaOp.getDstMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();

    bool isMM2S =
        (src_memory_space >
         dst_memory_space); // This is the tile DMA pushing onto a stream from
                            // its own memory, e.g if the DMA is from 2 (src,
                            // tile memory) to 0 (dst, ext memory)
    auto all_tile_dma_allocs =
        isMM2S ? &tile_dma_MM2S_allocs : &tile_dma_S2MM_allocs;

    int64_t chan = -1;

    unsigned num_allocs = 0;
    for (auto &t : *all_tile_dma_allocs) {
      if (col == std::get<1>(t) && row == std::get<2>(t)) {
        if (dmaOp.getId() == std::get<0>(t))
          chan = std::get<3>(t);
        num_allocs++;
      }
    }
    if (chan == -1) {
      // Need to allocate a new one
      chan = num_allocs % tile_dma_channels;
      all_tile_dma_allocs->push_back({dmaOp.getId(), col, row, chan});
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

  AIE::BufferOp getBufferForTileDMA(ModuleOp aie_module,
                                    air::DmaMemcpyInterface &dmaOp, int col,
                                    int row) {
    AIE::DMAChannel channel = getTileDMAChannel(aie_module, dmaOp, col, row);
    Value buffer;
    if (isMM2S(channel)) {
      buffer = dmaOp.getSrcMemref();
    } else {
      buffer = dmaOp.getDstMemref();
    }
    AIE::BufferOp bufferOp = buffer.getDefiningOp<AIE::BufferOp>();
    // if (!bufferOp)
    //   buffer.dump();
    return bufferOp;
  }

  AIE::LockOp getLockForTileDMA(ModuleOp aie_module,
                                air::DmaMemcpyInterface &dmaOp,
                                lock_allocation_list &info, int col, int row) {
    AIE::BufferOp bufferOp = getBufferForTileDMA(aie_module, dmaOp, col, row);
    AIE::DMAChannel channel = getTileDMAChannel(aie_module, dmaOp, col, row);
    assert(bufferOp);
    AIE::LockOp lockOp = nullptr;
    for (size_t i = 0; i < info.size(); i++) {
      if ((std::get<0>(info[i]) == bufferOp) &&
          (std::get<2>(info[i]) == channel)) {
        lockOp = std::get<1>(info[i]);
        break;
      }
    }
    if (!lockOp) {
      OpBuilder builder(bufferOp);
      lockOp = allocateLockOp(aie_module, bufferOp.getTileOp());
      info.push_back({bufferOp, lockOp, channel});
    }
    return lockOp;
  }

  // get tileop using partition-relative coordinates
  AIE::TileOp getTileOp(ModuleOp aie_module, int herd_col, int herd_row) {
    int col = herd_col;
    int row = herd_row;
    return getPhysTileOp(aie_module, col, row);
  }

  AIE::FlowOp getFlowOp(ModuleOp aie_module, mlir::Value source,
                        xilinx::AIE::WireBundle sourceBundle,
                        uint32_t sourceChannel, mlir::Value dest,
                        xilinx::AIE::WireBundle destBundle,
                        uint32_t destChannel) {
    AIE::FlowOp flowOp = nullptr;
    aie_module.walk([&](Operation *op) {
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

    OpBuilder builder(aie_module);
    builder.setInsertionPointToEnd(aie_module.getBody());
    return builder.create<AIE::FlowOp>(builder.getUnknownLoc(), source,
                                       sourceBundle, sourceChannel, dest,
                                       destBundle, destChannel);
  }

  void getAIRDmaMemcpyInBlock(Block &b, std::vector<Operation *> &output) {
    for (Operation &o : b.getOperations()) {
      if (isa<air::DmaMemcpyInterface>(&o))
        output.push_back(&o);
      for (Region &r : o.getRegions())
        getAIRDmaMemcpyInRegion(r, output);
    }
  }

  void getAIRDmaMemcpyInRegion(Region &r, std::vector<Operation *> &output) {
    for (Block &b : r.getBlocks())
      getAIRDmaMemcpyInBlock(b, output);
  }

  std::map<AIE::DMAChannel, std::vector<Operation *>>
  getDmaSchedules(AIE::CoreOp core, int x, int y, DMAAllocator &shim_dma_alloc,
                  std::vector<AIE::TileOp> &shim_dma_inits) {

    std::map<AIE::DMAChannel, std::vector<Operation *>> tile_dma_copies;
    std::vector<Operation *> dma_memcpy_ops;
    getAIRDmaMemcpyInRegion(core.getBody(), dma_memcpy_ops);

    auto aie_module = core->getParentOfType<ModuleOp>();
    auto tile = core.getTileOp();

    for (auto o : dma_memcpy_ops) {

      auto dmaOpIf = cast<air::DmaMemcpyInterface>(o);

      int src_space = dmaOpIf.getSrcMemref()
                          .getType()
                          .cast<MemRefType>()
                          .getMemorySpaceAsInt();
      int dst_space = dmaOpIf.getDstMemref()
                          .getType()
                          .cast<MemRefType>()
                          .getMemorySpaceAsInt();

      if ((src_space == (int)air::MemorySpace::L2 &&
           dst_space == (int)air::MemorySpace::L3) ||
          (src_space == (int)air::MemorySpace::L3 &&
           dst_space == (int)air::MemorySpace::L2) ||
          (src_space == (int)air::MemorySpace::L1 &&
           dst_space == (int)air::MemorySpace::L2) ||
          (src_space == (int)air::MemorySpace::L2 &&
           dst_space == (int)air::MemorySpace::L1)) {
        o->erase();
        continue;
      }

      AIE::DMAChannel tile_channel =
          getTileDMAChannel(aie_module, dmaOpIf, x, y);

      if ((src_space == (int)air::MemorySpace::L3 &&
           dst_space == (int)air::MemorySpace::L1) ||
          (src_space == (int)air::MemorySpace::L1 &&
           dst_space == (int)air::MemorySpace::L3)) {

        // copy between L1 and external memory, use shim dma
        tile_channel = getTileDMAChannel(aie_module, dmaOpIf, x, y);
        AIE::TileOp shim_tile = shim_dma_alloc.getTile(
            aie_module, dmaOpIf, (int64_t)tile_channel.second, x, y);
        AIE::DMAChannel shim_channel =
            shim_dma_alloc.getChannel(aie_module, dmaOpIf, tile_channel, x, y);

        LLVM_DEBUG(llvm::outs()
                   << "Shim channel is " << (uint64_t)shim_channel.second
                   << " for x=" << x << ", y=" << y << "\n");

        if ((shim_channel.first == AIE::DMAChannelDir::S2MM) &&
            ((uint64_t)shim_channel.second < (uint64_t)shim_dma_channels)) {
          getFlowOp(aie_module, tile, AIE::WireBundle::DMA,
                    (uint32_t)tile_channel.second, shim_tile,
                    AIE::WireBundle::DMA,
                    ((uint32_t)shim_channel.second) % shim_dma_channels);
        } else {
          getFlowOp(aie_module, shim_tile, AIE::WireBundle::DMA,
                    ((uint32_t)shim_channel.second) % shim_dma_channels, tile,
                    AIE::WireBundle::DMA, (uint32_t)tile_channel.second);
        }
      } else {
        llvm_unreachable("Unhandled dma transfer type");
      }

      tile_dma_copies[tile_channel].push_back(dmaOpIf);
    }
    return tile_dma_copies;
  }

  airrt::PartitionMetadataOp
  getOrCreatePartitionMetadata(airrt::ModuleMetadataOp module_meta,
                               StringRef name) {

    for (auto pm : module_meta.getPartitions()
                       .front()
                       .getOps<airrt::PartitionMetadataOp>())
      if (name == pm.getSymName().str())
        return pm;

    auto builder = OpBuilder::atBlockTerminator(module_meta.getBody());
    auto loc = builder.getUnknownLoc();
    auto partition_meta = builder.create<airrt::PartitionMetadataOp>(loc, name);
    builder.createBlock(&partition_meta.getHerds());
    builder.create<airrt::PartitionMetadataTerminatorOp>(loc);

    return partition_meta;
  }

  airrt::HerdMetadataOp
  createHerdMetadata(airrt::PartitionMetadataOp partition_meta,
                     air::HerdOp herd) {
    auto builder = OpBuilder::atBlockTerminator(partition_meta.getBody());
    auto loc = builder.getUnknownLoc();

    std::string name = "herd";
    if (auto attr =
            herd->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      name = attr.getValue().str();

    auto herd_meta = builder.create<airrt::HerdMetadataOp>(loc, name);
    return herd_meta;
  }

  void lowerAirDmaMemcpy(ModuleOp module, DMAAllocator &shimDmaAlloc) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : module.getOps<AIE::CoreOp>())
      cores.push_back(c);

    OpBuilder builder(module);

    for (AIE::CoreOp core : cores) {
      AIE::TileOp tile = core.getTileOp();
      auto x = tile.getCol();
      auto y = tile.getRow();

      std::vector<AIE::TileOp> shim_dma_inits;

      // collect dma operations and generate a schedule
      std::map<AIE::DMAChannel, std::vector<Operation *>> tile_dma_copies =
          getDmaSchedules(core, x, y, shimDmaAlloc, shim_dma_inits);

      // emit the acquire and release of the L1 buffer locks
      lock_allocation_list lock_allocs;
      std::unordered_set<Operation *> allocs_to_remap;
      for (auto p : tile_dma_copies) {
        for (auto o : p.second) {
          auto dmaOpIf = cast<air::DmaMemcpyInterface>(o);
          AIE::DMAChannel tile_channel =
              getTileDMAChannel(module, dmaOpIf, x, y);
          AIE::LockOp lockOp =
              getLockForTileDMA(module, dmaOpIf, lock_allocs, x, y);
          int64_t lockAqValue = -1;
          int64_t lockRelValue = -1;
          Value alloc = nullptr;
          if (!isMM2S(tile_channel)) {
            lockAqValue = 1;
            lockRelValue = 0;
            alloc = dmaOpIf.getDstMemref();
          } else {
            lockAqValue = 0;
            lockRelValue = 1;
            alloc = dmaOpIf.getSrcMemref();
          }

          if (auto bco =
                  dyn_cast<bufferization::ToMemrefOp>(alloc.getDefiningOp()))
            builder.setInsertionPoint(bco.getOperand().getDefiningOp());
          else if (auto a = dyn_cast<memref::AllocaOp>(alloc.getDefiningOp()))
            builder.setInsertionPoint(alloc.getDefiningOp());
          else
            builder.setInsertionPoint(&dmaOpIf->getBlock()->front());

          builder.create<AIE::UseLockOp>(o->getLoc(), lockOp, lockAqValue,
                                         AIE::LockAction::Acquire);
          // try to find a place to put the unlock. If there are deallocs,
          // replace them with unlock. Otherwise, put them at the end.
          bool need_unlock = true;
          for (auto u : alloc.getUsers()) {
            if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
              builder.setInsertionPoint(dealloc);
              builder.create<AIE::UseLockOp>(dealloc->getLoc(), lockOp,
                                             lockRelValue,
                                             AIE::LockAction::Release);
              // assume that the deallocs will take care of it when
              // deallocs are present
              need_unlock = false;
            }
          }
          if (need_unlock) {
            auto t = dmaOpIf->getBlock()->getTerminator();
            builder.setInsertionPoint(t);
            builder.create<AIE::UseLockOp>(t->getLoc(), lockOp, lockRelValue,
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
        auto dmaOps = p.second;
        Block *next_bd = nullptr;
        for (size_t i = 0; i < dmaOps.size(); i++) {
          auto dmaOp = cast<air::DmaMemcpyInterface>(dmaOps[i]);
          Block *bd;
          if (i == 0)
            bd = first_bd;
          else
            bd = next_bd;
          auto b = OpBuilder::atBlockEnd(bd);
          if (i == dmaOps.size() - 1) {
            b.create<AIE::NextBDOp>(loc, first_bd);
          } else {
            next_bd = new Block();
            mem.getBody().push_back(next_bd);
            b.create<AIE::NextBDOp>(loc, next_bd);
          }
          AIE::BufferOp bufferOp = getBufferForTileDMA(module, dmaOp, x, y);
          AIE::LockOp lockOp =
              getLockForTileDMA(module, dmaOp, lock_allocs, x, y);
          b.setInsertionPointToStart(bd);
          int64_t lockAqValue = -1;
          int64_t lockRelValue = -1;
          if (!isMM2S(channel)) {
            lockAqValue = 0;
            lockRelValue = 1;
          } else {
            lockAqValue = 1;
            lockRelValue = 0;
          }
          Value length = dmaOp.getLength();
          if (!length) {
            auto ndcpy = cast<air::DmaMemcpyNdOp>(dmaOps[i]);
            auto src_memory_space = ndcpy.getSrcMemref()
                                        .getType()
                                        .cast<MemRefType>()
                                        .getMemorySpaceAsInt();
            auto dst_memory_space = ndcpy.getDstMemref()
                                        .getType()
                                        .cast<MemRefType>()
                                        .getMemorySpaceAsInt();
            auto sizes = src_memory_space > dst_memory_space
                             ? ndcpy.getDstSizes()
                             : ndcpy.getSrcSizes();
            int64_t size = 1;
            for (auto s : sizes) {
              auto c = dyn_cast<arith::ConstantIndexOp>(s.getDefiningOp());
              if (!c) {
                size = -1;
                break;
              }
              size = size * c.value();
            }
            length = b.create<arith::ConstantIndexOp>(dmaOp.getLoc(), size)
                         ->getResult(0);
          }
          b.create<AIE::UseLockOp>(loc, lockOp, lockAqValue,
                                   AIE::LockAction::Acquire);
          b.create<AIE::DMABDOp>(
              loc, bufferOp, 0,
              cast<arith::ConstantIndexOp>(length.getDefiningOp()).value(), 0);
          b.create<AIE::UseLockOp>(loc, lockOp, lockRelValue,
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
    ShimTileAllocator shimTileAlloc(shim_dma_cols, shim_dma_channels);

    if (clTestPatterns.find("to-aie-mlir") != std::string::npos) {
      std::vector<std::pair<ModuleOp, air::HerdOp>> aie_modules;
      std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
      AIRToAIEOptions options = {.col_offset = clColOffset,
                                 .row_offset = clRowOffset,
                                 .emit_while = clEmitWhileLoop,
                                 .emit_herd_lock = clEmitHerdLock};
      createAIEModulesAndOutlineCores(m, aie_modules, tileToHerdMap, options);
      std::set<ModuleOp> seen;
      for (auto &p : aie_modules) {
        ModuleOp m = std::get<0>(p);
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
    if (clTestPatterns.find("lower-air-channels") != std::string::npos)
      patterns.insert<LowerAIRChannelsPattern>(ctx, shimTileAlloc);

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
    builder.createBlock(&module_meta.getPartitions());
    builder.create<airrt::ModuleMetadataTerminatorOp>(loc);

    // If we have multiple herds then we must emit them into different aie
    // modules to avoid resource conflicts in the AIE physical dialect.
    std::vector<std::pair<ModuleOp, air::HerdOp>> aie_modules;

    std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
    AIRToAIEOptions options = {.col_offset = clColOffset,
                               .row_offset = clRowOffset,
                               .emit_while = clEmitWhileLoop,
                               .emit_herd_lock = clEmitHerdLock};
    createAIEModulesAndOutlineCores(module, aie_modules, tileToHerdMap,
                                    options);

    std::set<ModuleOp> seen;
    for (auto &p : aie_modules) {
      ModuleOp m = std::get<0>(p);
      xilinx::air::HerdOp h = std::get<1>(p);
      auto ctx = m->getContext();

      if (seen.find(m) == seen.end()) {
        seen.insert(m);

        specializeHerdAffineIf(m);
        lowerAirExecute(m);
        lowerScfAirTokens(m);

        allocL1Buffers(m, tileToHerdMap);

        DMAAllocator shimDmaAlloc(shim_dma_cols, shim_dma_channels);

        lowerAirDmaMemcpy(m, shimDmaAlloc);
        lowerPipelineGetPut(m, tileToHerdMap);

        SmallVector<air::HerdOp, 4> herds;
        if (auto p = h->getParentOfType<air::PartitionOp>()) {
          auto hops = p.getOps<air::HerdOp>();
          herds.append(hops.begin(), hops.end());
        } else {
          herds.push_back(h);
        }

        for (auto herd : herds) {
          std::set<int64_t> dma_ids;
          herd.walk([&](Operation *o) {
            if (auto dmaOp = dyn_cast<air::DmaMemcpyInterface>(o))
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
          auto partition_meta = getOrCreatePartitionMetadata(
              module_meta, m.getName()->split('.').second);
          auto herd_meta = createHerdMetadata(partition_meta, herd);
          herd_meta->setAttr("dma_allocations",
                             ArrayAttr::get(ctx, dma_allocations));
        }
        tile_dma_S2MM_allocs.clear();
        tile_dma_MM2S_allocs.clear();
      }

      RewritePatternSet patterns(ctx);
      air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    };

    // emit aie_modules to files or to stdout
    seen.clear();
    for (auto p : aie_modules) {
      auto aie_module = std::get<0>(p);
      if (seen.find(aie_module) == seen.end())
        seen.insert(aie_module);
      else
        continue;
      if (clOutputPrefix != "-") {
        if (clOutputPrefix != "/dev/null") {
          std::error_code EC;
          std::string fname =
              clOutputPrefix + aie_module.getName()->str() + ".mlir";
          llvm::raw_fd_ostream aie_ostream(fname, EC);
          aie_module.print(aie_ostream);
        }
      } else {
        aie_module.print(llvm::outs());
      }
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

FailureOr<ModuleOp> convertAIRToAIE(mlir::RewriterBase &rewriter,
                                    air::PartitionOp p) {
  std::string partition_name = "partition_0";
  if (auto attr =
          p->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    partition_name = attr.getValue().str();

  std::string aie_module_name = "aie." + partition_name;
  ModuleOp aie_module =
      ModuleOp::create(rewriter.getUnknownLoc(), StringRef(aie_module_name));

  AIRToAIEOptions options = {
      .col_offset = 7, .row_offset = 2, .emit_while = false};
  std::vector<std::pair<ModuleOp, xilinx::air::HerdOp>> aie_modules;
  p.walk([&](xilinx::air::HerdOp h) {
    aie_modules.push_back({aie_module, h});
  });
  std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
  for (auto &p : aie_modules) {
    ModuleOp aie_module = std::get<0>(p);
    xilinx::air::HerdOp h = std::get<1>(p);

    outlineAIECores(rewriter, aie_module, h, tileToHerdMap, options);

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

} // namespace air
} // namespace xilinx
