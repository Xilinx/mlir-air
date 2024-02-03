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
#include "mlir/IR/AffineExpr.h"
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

struct AIRToAIEConversionOptions {
  int64_t col_offset;
  int64_t row_offset;
  bool emit_while;
  bool emit_herd_lock;
  bool generate_shim_dma;
  AIE::AIEDevice device;
};

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
    std::vector<std::string> chan_names;
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
                          int dst_memory_space, std::string chan_name) {
    bool isMM2S = (src_memory_space < dst_memory_space);
    auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

    // return first available shim tile with a free channel
    for (auto &t : *allocs) {
      if (t.available_channels > 0) {
        t.available_channels -= 1;
        t.chan_names.push_back(chan_name);
        return t.shim_tile;
      }
    }
    auto shim_col = shim_columns[allocs->size()];
    auto shim_tile = getPhysTileOp(aie_device, shim_col, 0);
    allocs->push_back({shim_tile, shim_dma_channels - 1, {chan_name}});

    return shim_tile;
  }
};

bool isMM2S(AIE::DMAChannel channel) {
  return (channel.direction == AIE::DMAChannelDir::MM2S);
}
bool isLegalMemorySpace(air::MemcpyInterface memcpyOp, AIE::AIEArch arch) {
  switch (arch) {
  case xilinx::AIE::AIEArch::AIE1: {
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
    }
    return false;
  }
  case xilinx::AIE::AIEArch::AIE2: {
    // todo for AIE2: add memtile data movement support
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
    }
    return false;
  }
  }
  return false;
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
  AIE::BufferOp bufferOp = builder.create<AIE::BufferOp>(
      tile->getLoc(), memrefTy, tile, nullptr, nullptr);

  std::stringstream ss =
      generateBufferNameInStringStream("buf", BufferId, attr, x, y);
  bufferOp->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(tile->getContext(), ss.str()));

  return bufferOp;
}

void outlineAIECores(OpBuilder &builder, AIE::DeviceOp aie_device,
                     xilinx::air::HerdOp h,
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

      // make the aie.tile
      auto tile = getPhysTileOp(aie_device, phys_x, phys_y);

      Operation *t = tile.getOperation();
      while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
        t = t->getNextNode();
      builder.setInsertionPointAfter(t);

      // make the aie.core for the tile core
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

      // generate the aie.core body
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
                                            herd_lock, AIE::LockAction::Acquire,
                                            0);

      Region &r = h.getRegion();
      r.cloneInto(&core.getBody(), remap);

      Block *launch_bb = remap.lookup(&r.front());
      core_builder.create<cf::BranchOp>(hloc, launch_bb);
      core_builder.setInsertionPoint(launch_bb->getTerminator());
      if (options.emit_herd_lock)
        core_builder.create<AIE::UseLockOp>(core_builder.getUnknownLoc(),
                                            herd_lock, AIE::LockAction::Release,
                                            0);

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

void outlineAIEMemtiles(OpBuilder &builder, AIE::DeviceOp aie_device,
                        xilinx::air::SegmentOp seg,
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
    getPhysTileOp(aie_device, phys_x, phys_y);
  }
}

void createAIEModulesAndOutlineCores(
    ModuleOp module,
    std::vector<std::pair<AIE::DeviceOp, xilinx::air::HerdOp>> &aie_modules,
    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
    AIRToAIEConversionOptions &options) {

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

    // If the device has memtiles, then outline memtiles
    if (aie_dev.getTargetModel().getNumMemTileRows()) {
      outlineAIEMemtiles(builder, aie_dev, p, options);
    }
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
            operands.push_back(mlir::mod(lhs, rhs));
          } else
            return failure();
        } else if (auto v = dyn_cast<arith::DivSIOp>(o.getDefiningOp())) {
          if (mlir::getConstantIntValue(v.getLhs()) &&
              mlir::getConstantIntValue(v.getRhs())) {
            int lhs = *mlir::getConstantIntValue(v.getLhs());
            int rhs = *mlir::getConstantIntValue(v.getRhs());
            operands.push_back(mlir::floorDiv(lhs, rhs));
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

    if (!fop.getInitArgs().size())
      return failure();

    SmallVector<Value, 4> iter_args;
    BitVector iter_args_idx(fop.getNumOperands());

    // erase air.event from the iter args
    for (auto block_arg : fop.getRegionIterArgs()) {
      Value v =
          fop.getOperand(block_arg.getArgNumber() - fop.getNumInductionVars() +
                         fop.getNumControlOperands());
      if (v.getType().isa<xilinx::air::AsyncTokenType>()) {
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

// struct LowerPipeGetPutPattern : public OpRewritePattern<air::PipelinePutOp> {
//   using OpRewritePattern<air::PipelinePutOp>::OpRewritePattern;

//   LowerPipeGetPutPattern(MLIRContext *ctx,
//                          std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap)
//       : OpRewritePattern(ctx), tileToHerdMap(tileToHerdMap) {}

//   LogicalResult matchAndRewrite(air::PipelinePutOp put,
//                                 PatternRewriter &rewriter) const override {
//     auto aie_device = put->getParentOfType<AIE::DeviceOp>();
//     auto core = put->getParentOfType<AIE::CoreOp>();
//     assert(aie_device && core);

//     auto herd = tileToHerdMap[core.getTileOp()];
//     auto c = herd.getColOffset();
//     auto r = herd.getRowOffset();
//     auto col_offset = c ? *c : 0;
//     auto row_offset = r ? *r : 0;

//     auto other_x =
//     cast<arith::ConstantIndexOp>(put.getDst0().getDefiningOp()); auto other_y
//     = cast<arith::ConstantIndexOp>(put.getDst1().getDefiningOp()); auto
//     other_core = getPhysTileOp(aie_device, other_x.value() + col_offset,
//                                     other_y.value() + row_offset)
//                           .getCoreOp();
//     assert(other_core);

//     air::PipelineGetOp get;
//     other_core.walk([&](air::PipelineGetOp pgo) { get = pgo; });
//     assert(get && get->getNumResults() == (put->getNumOperands() - 2));

//     for (auto p :
//          llvm::zip(put->getOperands().drop_front(2), get->getResults())) {

//       auto o = std::get<0>(p); // operand of put
//       auto r = std::get<1>(p); // result of get
//       // for each ranked tensor put (yielded) by the tile
//       if (RankedTensorType tt = o.getType().dyn_cast<RankedTensorType>()) {
//         auto memrefTy = MemRefType::get(tt.getShape(), tt.getElementType(),
//         {},
//                                         (int)air::MemorySpace::L1);
//         // allocate buffer+lock
//         auto buf = allocateBufferOp(
//             memrefTy, core.getTileOp(),
//             StringAttr::get(aie_device.getContext(), "pipebuf"));
//         auto lockOp = allocateLockOp(aie_device, core.getTileOp());

//         // acquire the lock for write on the put side
//         rewriter.setInsertionPoint(put);
//         rewriter.create<AIE::UseLockOp>(put->getLoc(), lockOp, 0,
//                                         AIE::LockAction::Acquire);
//         rewriter.create<memref::TensorStoreOp>(put->getLoc(), o, buf);
//         rewriter.create<AIE::UseLockOp>(put->getLoc(), lockOp, 1,
//                                         AIE::LockAction::Release);

//         // acquire the lock for read on the get side
//         rewriter.setInsertionPoint(get);
//         rewriter.create<AIE::UseLockOp>(get->getLoc(), lockOp, 1,
//                                         AIE::LockAction::Acquire);
//         auto loadOp =
//             rewriter.create<bufferization::ToTensorOp>(get->getLoc(), buf);
//         rewriter.create<AIE::UseLockOp>(get->getLoc(), lockOp, 0,
//                                         AIE::LockAction::Release);
//         r.replaceAllUsesWith(loadOp.getResult());
//       } else {
//         llvm::errs() << "error, unsupported air.pipeline.yield operand
//         type\n"; assert(0 && "Unsupported"); return failure();
//       }
//     }
//     rewriter.eraseOp(get);
//     rewriter.eraseOp(put);
//     return success();
//   }

// private:
//   std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap;
// };

// This function replaces PipelinePutOp/PipelineGetOp pairs with a
// shared aie.buffer + aie.lock. This is a single-buffered implementation
// with exclusive access to the buffer controlled by the lock. i.e. FIXME.
// void lowerPipelineGetPut(AIE::DeviceOp &m,
//                          std::map<AIE::TileOp, air::HerdOp> tileToHerdMap) {
//   auto ctx = m->getContext();
//   RewritePatternSet patterns(ctx);
//   patterns.insert<LowerPipeGetPutPattern>(ctx, tileToHerdMap);
//   (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
// }

// struct AllocL1TensorsPattern
//     : public OpRewritePattern<bufferization::ToMemrefOp> {
//   using OpRewritePattern<bufferization::ToMemrefOp>::OpRewritePattern;

//   AllocL1TensorsPattern(MLIRContext *ctx,
//                         std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap)
//       : OpRewritePattern(ctx), tileToHerdMap(tileToHerdMap) {}

//   LogicalResult matchAndRewrite(bufferization::ToMemrefOp cast,
//                                 PatternRewriter &rewriter) const override {

//     AIE::CoreOp core = cast->getParentOfType<AIE::CoreOp>();
//     if (!core)
//       return failure();

//     AIE::TileOp tile = core.getTileOp();
//     if (!tile)
//       return failure();

//     MemRefType memrefTy = nullptr;
//     memrefTy = cast.getType().cast<MemRefType>();

//     if (memrefTy.getMemorySpaceAsInt() != (int)air::MemorySpace::L1)
//       return failure();

//     rewriter.setInsertionPointAfter(tile);
//     auto herd = tileToHerdMap[core.getTileOp()];
//     int64_t col_offset = 0;
//     int64_t row_offset = 0;
//     if (herd) {
//       auto c = herd.getColOffset();
//       auto r = herd.getRowOffset();
//       col_offset = c ? *c : 0;
//       row_offset = r ? *r : 0;
//     }
//     auto buffer = allocateBufferOp(
//         memrefTy, tile,
//         cast->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
//         tile.getCol() - col_offset, tile.getRow() - row_offset);

//     rewriter.setInsertionPoint(cast);
//     rewriter.create<memref::TensorStoreOp>(cast.getLoc(), cast.getOperand(),
//                                            buffer);
//     rewriter.replaceOp(cast, buffer->getResults());
//     return success();
//   }

// private:
//   std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap;
// };

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

struct AllocL2BuffersPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  AllocL2BuffersPattern(MLIRContext *ctx,
                        std::map<memref::AllocOp, AIE::TileOp> &memrefToTileMap)
      : OpRewritePattern(ctx), memrefToTileMap(memrefToTileMap) {}

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
    assert(memrefToTileMap.count(alloc));
    AIE::TileOp tile = memrefToTileMap[alloc];
    if (!tile)
      return failure();

    rewriter.setInsertionPointAfter(tile);
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
        memrefTy, tile,
        alloc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        tile.getCol() - col_offset, tile.getRow() - row_offset);

    rewriter.setInsertionPoint(alloc);
    rewriter.replaceOp(alloc, buffer->getResults());
    return success();
  }

private:
  std::map<memref::AllocOp, AIE::TileOp> &memrefToTileMap;
};

void allocL1Buffers(AIE::DeviceOp m,
                    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<AllocL1BuffersPattern>(ctx, tileToHerdMap);
  // AllocL1TensorsPattern
  (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
}

void L2MemrefToMemTileMap(
    AIE::DeviceOp m,
    std::map<memref::AllocOp, AIE::TileOp> &memrefToMemTileMap) {
  std::vector<memref::AllocOp> allocs;
  m.walk([&](memref::AllocOp alloc) {
    if (alloc.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt() ==
        (int)air::MemorySpace::L2) {
      allocs.push_back(alloc);
    }
  });
  std::vector<AIE::TileOp> memtiles;
  for (auto t : m.getOps<AIE::TileOp>()) {
    // TODO: Hard coded memtile row as 1 here
    if (t.rowIndex() == 1) {
      memtiles.push_back(t);
    }
  }

  // Allocation of L2 memrefs in segment to (memtile) tile ops
  // // Strategy: when one memtile is full, move on to the next one
  std::map<AIE::TileOp, uint32_t> memtileToSizeMap;
  for (auto t : memtiles) {
    memtileToSizeMap[t] = m.getTargetModel().getMemTileSize();
  }
  int memtile_id = 0;
  for (auto alloc : allocs) {
    MemRefType ty = alloc.getMemref().getType().cast<MemRefType>();
    auto memref_vol = getElementSizeInBytes(ty) * getTensorVolume(ty);
    while (memtileToSizeMap[memtiles[memtile_id]] < memref_vol) {
      memtile_id++;
      assert(memtile_id < memtiles.size());
    }
    memtileToSizeMap[memtiles[memtile_id]] -= memref_vol;
    memrefToMemTileMap[alloc] = memtiles[memtile_id];
  }
  // Strategy: round robin, with memory usage awareness
  // std::map<AIE::TileOp, uint32_t> memtileToSizeMap;
  // for (auto t : memtiles) {
  //   memtileToSizeMap[t] = m.getTargetModel().getMemTileSize();
  // }
  // int memtile_id = 0;
  // memref::AllocOp previous_alloc = nullptr;
  // for (auto alloc : allocs) {
  //   MemRefType ty = alloc.getMemref().getType().cast<MemRefType>();
  //   auto memref_vol = getElementSizeInBytes(ty) * getTensorVolume(ty);
  //   int skip_count = 0;
  //   while (memtileToSizeMap[memtiles[memtile_id]] < memref_vol) {
  //     memtile_id++;
  //     memtile_id %= memtiles.size();
  //     skip_count++;
  //     assert(skip_count < (int)memtiles.size());
  //   }
  //   memtileToSizeMap[memtiles[memtile_id]] -= memref_vol;
  //   memrefToMemTileMap[alloc] = memtiles[memtile_id];
  //   if (previous_alloc && areReferencedByTheSameAIRChannel(alloc.getMemref(),
  //   previous_alloc.getMemref())){
  //     // Memrefs referenced by the same channel must be located in the same
  //     tile
  //   }
  //   else memtile_id++;
  //   // memtile_id++;
  //   memtile_id %= memtiles.size();
  //   previous_alloc = alloc;
  // }
}

void allocL2Buffers(AIE::DeviceOp m) {
  auto ctx = m->getContext();
  RewritePatternSet patterns(ctx);
  if (m.getTargetModel().getNumMemTileRows()) {
    std::map<memref::AllocOp, AIE::TileOp> memrefToTileMap;
    L2MemrefToMemTileMap(m, memrefToTileMap);
    patterns.insert<AllocL2BuffersPattern>(ctx, memrefToTileMap);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
  }
}

template <typename T> void push_back_if_unique(std::vector<T> &vec, T entry) {
  if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
    vec.push_back(entry);
  }
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
      auto res =
          findChannelPutGetTile<ChannelGetOp>(get, &consumerTile, &datatype);
      if (res.failed())
        return res;
      consumers.push_back(consumerTile);
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
        builder.getUnknownLoc(), builder.getStringAttr(name), prodTile,
        consTile, builder.getIntegerAttr(builder.getI32Type(), depth),
        datatype);
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

// Get owner (scf.parallelop) of channel indices
scf::ParallelOp getChannelIndicesOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return scf::ParallelOp();
  if (!ivArg.getOwner()) {
    val.getDefiningOp()->emitOpError("unlinked block argument");
    return scf::ParallelOp();
  }
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<scf::ParallelOp>(containingOp);
}
scf::ParallelOp getChannelIndicesOwner(Operation *op) {
  if (!op)
    return scf::ParallelOp();
  auto putget = dyn_cast<air::ChannelInterface>(op);
  if (!putget)
    return scf::ParallelOp();
  for (auto index : putget.getIndices()) {
    if (auto par = getChannelIndicesOwner(index)) {
      return par;
    }
  }
  return scf::ParallelOp();
}

struct SpecializeChannelBundlePattern
    : public OpRewritePattern<air::ChannelOp> {
  using OpRewritePattern<air::ChannelOp>::OpRewritePattern;

  SpecializeChannelBundlePattern(
      MLIRContext *ctx, std::map<std::string, std::string> &chan_to_chan_map)
      : OpRewritePattern(ctx), chan_to_chan_map(chan_to_chan_map) {}

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
    auto bundle_size = extractFromIntegerArrayAttr<int64_t>(channel.getSize());
    auto bundle_size_stdvec = convertToStdVec(bundle_size);
    for (unsigned iter = 0; iter < (unsigned)channel.getBundleSize(); iter++) {
      rewriter.setInsertionPoint(channel);
      auto cname = createChannelName(device.getOperation());
      // Add chan name to chan name map
      chan_to_chan_map[cname] = channel.getName().str();
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
            clearAsyncDependenciesOfAsyncOp(new_put);
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
            clearAsyncDependenciesOfAsyncOp(new_get);
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
  std::map<std::string, std::string> &chan_to_chan_map;
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
    new_put->setAttr(
        "id",
        IntegerAttr::get(IntegerType::get(put->getContext(), 32), put.getId()));
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
    new_get->setAttr(
        "id",
        IntegerAttr::get(IntegerType::get(get->getContext(), 32), get.getId()));
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
void specializeChannelBundle(
    AIE::DeviceOp &d, std::map<std::string, std::string> &chan_to_chan_map) {
  auto ctx = d->getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<SpecializeChannelBundlePattern>(ctx, chan_to_chan_map);
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

class AIRToAIEPass : public air::impl::AIRToAIEBase<AIRToAIEPass> {

public:
  AIRToAIEPass() = default;
  AIRToAIEPass(const AIRToAIEPass &pass) {}
  AIRToAIEPass(const ::xilinx::air::AIRToAIEOptions &options)
      : AIRToAIEBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
    registry.insert<xilinx::airrt::AIRRtDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<cf::ControlFlowDialect>();
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
            static_cast<int64_t>(sourceChannel) == fop.getSourceChannel() &&
            static_cast<int64_t>(destChannel) == fop.getDestChannel())
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

  // Clone data movement ops to and from memtile and shim tile DMAs
  void cloneL2AndL3MemcpysToDeviceOp(OpBuilder &builder,
                                     AIE::DeviceOp aie_device, ModuleOp module,
                                     bool clone_l2, bool clone_l3) {

    if (!clone_l2 && !clone_l3)
      return;

    std::vector<xilinx::air::MemcpyInterface> memcpyOps;
    module.walk([&](xilinx::air::MemcpyInterface memcpyOp) {
      auto hasParentHerdOp = memcpyOp->getParentOfType<air::HerdOp>();
      auto hasParentSegmentOp = memcpyOp->getParentOfType<air::SegmentOp>();
      auto hasParentDeviceOp = memcpyOp->getParentOfType<AIE::DeviceOp>();
      if (clone_l2) {
        if (!hasParentHerdOp && hasParentSegmentOp && !hasParentDeviceOp) {
          memcpyOps.push_back(memcpyOp);
        }
      }
      if (clone_l3) {
        if (!hasParentHerdOp && !hasParentSegmentOp && !hasParentDeviceOp) {
          memcpyOps.push_back(memcpyOp);
        }
      }
    });

    Operation *t = nullptr;
    for (auto tile_op : aie_device.getBody()->getOps<AIE::TileOp>()) {
      t = tile_op.getOperation();
    }
    builder.setInsertionPointAfter(t);
    IRMapping remap;

    // Get defining ops to memcpyOp's operands copied over together with
    // memcpyOp
    std::vector<Operation *> operandOps;
    for (auto o : memcpyOps) {
      for (auto operand : o->getOperands()) {
        if (operand.getDefiningOp() &&
            isa<arith::ConstantIndexOp>(operand.getDefiningOp())) {
          operandOps.push_back(operand.getDefiningOp());
        }
        // Substituting index operands, such as strides and offsets, to constant
        // zero for convenience. TODO: generalize this
        else if (operand.getType().isa<IndexType>()) {
          remap.map(operand, builder.create<arith::ConstantIndexOp>(
                                 builder.getUnknownLoc(), 0));
        }
      }
    }

    for (auto o : operandOps) {
      builder.clone(*o, remap);
    }
    std::vector<Value> cloned_memrefs;
    for (auto o : memcpyOps) {
      if (auto memref = o.getSrcMemref()) {
        push_back_if_unique<Value>(cloned_memrefs, memref);
      }
      if (auto memref = o.getDstMemref()) {
        push_back_if_unique<Value>(cloned_memrefs, memref);
      }
    }
    for (auto memref : cloned_memrefs) {
      if (auto memalloc = memref.getDefiningOp()) {
        auto cloned_alloc = builder.clone(*memalloc, remap);
        clearAsyncDependenciesOfAsyncOp(cloned_alloc);
      } else {
        MemRefType ty = memref.getType().cast<MemRefType>();
        auto alloc_op = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(),
            MemRefType::get(ty.getShape(), ty.getElementType(),
                            ty.getLayout().getAffineMap(),
                            ty.getMemorySpaceAsInt()));
        remap.map(memref, alloc_op.getMemref());
      }
    }
    for (auto o : memcpyOps) {
      // Clone memcpy op
      if (auto par = getChannelIndicesOwner(o)) {

        SmallVector<int, 2> lbs_spatial, ubs_spatial;
        getSizesFromSpatialLoop(par.getOperation(), lbs_spatial, ubs_spatial);
        std::vector<unsigned> par_size;
        unsigned par_vol = 1;
        for (unsigned i = 0; i < lbs_spatial.size(); i++) {
          par_size.push_back(ubs_spatial[i] - lbs_spatial[i] + 1);
          par_vol *= ubs_spatial[i] - lbs_spatial[i] + 1;
        }
        for (unsigned iter = 0; iter < par_vol; iter++) {
          std::vector<unsigned> position =
              getMDVectorFromIterator(par_size, iter);
          SmallVector<Value, 4> emptyVec = {};
          SmallVector<Type, 4> tys = {};
          if (auto putget = dyn_cast<air::ChannelInterface>(o.getOperation())) {
            for (unsigned i = 0; i < putget.getIndices().size(); i++)
              remap.map(putget.getIndices()[i],
                        builder.create<arith::ConstantIndexOp>(
                            builder.getUnknownLoc(), position[i]));
          }
          // Specialize any affine apply mappings to operand
          for (auto oper : o->getOperands()) {
            if (oper.getDefiningOp()) {
              mlir::affine::AffineApplyOp position_apply = nullptr;
              if (auto apply_op = dyn_cast<mlir::affine::AffineApplyOp>(
                      oper.getDefiningOp()))
                position_apply = apply_op;
              else if (auto exec =
                           dyn_cast<air::ExecuteOp>(oper.getDefiningOp())) {
                auto child_op = &exec.getBody().front().getOperations().front();
                if (auto apply_op =
                        dyn_cast<mlir::affine::AffineApplyOp>(child_op))
                  position_apply = apply_op;
              }
              int position_iv = -1;
              if (position_apply)
                for (unsigned i = 0; i < par.getInductionVars().size(); i++)
                  for (auto map_o : position_apply.getMapOperands())
                    if (par.getInductionVars()[i] == map_o)
                      position_iv = i;
              if (position_apply && position_iv >= 0) {
                // Evaluate the affine expression and specialize the operand
                auto c = position_apply.getAffineMap().getResult(0);
                SmallVector<AffineExpr, 1> const_syms{
                    getAffineConstantExpr(position[position_iv],
                                          builder.getContext()),
                };
                auto newC = c.replaceSymbols(const_syms);
                auto expr = dyn_cast<AffineConstantExpr>(
                    simplifyAffineExpr(newC, 0, 1));
                assert(expr);
                int result = expr.getValue();
                remap.map(oper, builder.create<arith::ConstantIndexOp>(
                                    builder.getUnknownLoc(), result));
              }
            }
          }
          auto new_memcpy = builder.clone(*o, remap);
          clearAsyncDependenciesOfAsyncOp(new_memcpy);
        }
      } else {
        auto new_memcpy = builder.clone(*o, remap);
        clearAsyncDependenciesOfAsyncOp(new_memcpy);
      }
    }

    // Clone channel declaration ops
    for (auto o : memcpyOps) {
      if (auto chan_op = dyn_cast<air::ChannelInterface>(o.getOperation())) {
        if (!aie_device.lookupSymbol(chan_op.getChanName())) {
          auto ch = air::getChannelDeclarationThroughSymbol(chan_op);
          builder.clone(*ch.getOperation());
        }
      }
    }
  }

  // Verify data movement legality for the given device architecture
  void verifyMemcpyOps(std::vector<Operation *> &dma_memcpy_ops,
                       AIE::AIEArch arch) {
    for (auto o = dma_memcpy_ops.begin(); o != dma_memcpy_ops.end();) {
      auto memcpyOpIf = cast<air::MemcpyInterface>(*o);
      if (!isLegalMemorySpace(memcpyOpIf, arch)) {
        o = dma_memcpy_ops.erase(o);
        (*o)->emitOpError("is an illegal data movement for architecture");
        (*o)->erase();
      } else
        ++o;
    }
  }

  template <typename T>
  void placeDMAChannelsAndRouteFlows(AIE::DeviceOp aie_device,
                                     ShimDMAAllocator &shim_dma_alloc,
                                     MemTileDMAAllocator &memtile_dma_alloc,
                                     TileDMAAllocator &tile_dma_alloc,
                                     bool generate_shim_dma) {

    std::vector<Operation *> dma_memcpy_ops;

    aie_device.walk(
        [&](T memcpyOp) { dma_memcpy_ops.push_back(memcpyOp.getOperation()); });

    // Step 1: Verify data movement legality for the given device architecture
    // verifyMemcpyOps(dma_memcpy_ops,
    //                 aie_device.getTargetModel().getTargetArch());

    // Step 2: Pair up memcpy ops into flow ops. Each entry in memcpy_flows is a
    // bundle of memcpy ops which share the same aie.flow.
    std::vector<MemcpyBundleAsFlow> memcpy_flows;
    for (auto o : dma_memcpy_ops) {
      if (auto dma = dyn_cast<air::DmaMemcpyNdOp>(o)) {
        MemcpyBundleAsFlow flow = MemcpyBundleAsFlow(dma);
        flow.pushBackMemcpyOpToBundle(dma);
        memcpy_flows.push_back(flow);
      } else if (auto putget = dyn_cast<air::ChannelInterface>(o)) {
        auto chan = air::getChannelDeclarationThroughSymbol(putget);
        assert(chan);
        std::string chan_name = putget.getChanName().str();
        // Check if new pair
        bool found_in_flows = false;
        for (auto &f : memcpy_flows) {
          if (auto air_flow_op_chan = dyn_cast<air::ChannelOp>(f.air_flow_op)) {
            if (chan_name == air_flow_op_chan.getSymName().str()) {
              f.pushBackMemcpyOpToBundle(putget);
              found_in_flows = true;
            }
          }
        }
        if (!found_in_flows) {
          // Create new entry in memcpy_flows
          MemcpyBundleAsFlow flow = MemcpyBundleAsFlow(chan);
          flow.pushBackMemcpyOpToBundle(putget);
          memcpy_flows.push_back(flow);
        }
      } else {
        o->emitOpError(
            "unknown memcpy op type. Expected air::MemcpyInterface.");
      }
    }

    // Step 3: Allocate tile DMA channels, shim DMA channels and shim tiles
    // AIR channel to AIE flow mapping strategy: allocate L1 DMAs first,
    // followed by L2 and then L3, where outer memory hierarchies reuse existing
    // AIE flows as possible.
    // if (groupingMemcpysByLoop(memcpy_flows))
    //   groupedByLoopDMAChannelAllocation(memcpy_flows, shim_dma_alloc,
    //                                     memtile_dma_alloc, tile_dma_alloc);
    // else
    //   simpleDMAChannelAllocation(memcpy_flows, shim_dma_alloc,
    //                              memtile_dma_alloc, tile_dma_alloc);
    simpleDMAChannelAllocation(memcpy_flows, shim_dma_alloc, memtile_dma_alloc,
                               tile_dma_alloc);

    // Step 3.5: Sort all ops being allocated to each DMA channel, to avoid
    // ping-pong deadlock.
    tile_dma_alloc.sortMemcpyOps(dma_memcpy_ops);

    // Step 4: Connect flows
    for (auto &f : memcpy_flows) {
      for (int i = 0; i < f.numS2MMAllocs; i++) {
        assert(f.MM2S_alloc.dma_tile);
        assert(f.S2MM_alloc[i].dma_tile);
        getFlowOp(aie_device, f.MM2S_alloc.dma_tile, AIE::WireBundle::DMA,
                  (uint32_t)f.MM2S_alloc.dma_channel.channel,
                  f.S2MM_alloc[i].dma_tile, AIE::WireBundle::DMA,
                  (uint32_t)f.S2MM_alloc[i].dma_channel.channel);
      }
    }
  }

  // Get herd dma allocation info for airrt herd metadata
  void getHerdDmaAllocations(OpBuilder builder, MLIRContext *ctx,
                             air::HerdOp herd,
                             std::vector<allocation_info_t> allocs, bool isMM2S,
                             std::map<int, int> chan_renumber_reverse_map,
                             std::vector<Attribute> &dma_allocations) {
    std::set<int64_t> dma_ids;
    herd.walk([&](air::MemcpyInterface o) { dma_ids.insert(o.getId()); });

    auto c = herd.getColOffset();
    auto r = herd.getRowOffset();
    int64_t col_offset = c ? *c : 0;
    int64_t row_offset = r ? *r : 0;

    for (auto &t : allocs) {
      auto tileOp = t.dma_tile;
      int64_t col = t.col - col_offset;
      int64_t row = t.row - row_offset;
      int64_t chan = isMM2S ? t.dma_channel.channel + 2 : t.dma_channel.channel;

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

  // Get segment dma allocation info for airrt segment metadata
  void getSegmentDmaAllocations(OpBuilder builder, MLIRContext *ctx,
                                air::SegmentOp seg,
                                std::vector<allocation_info_t> allocs,
                                bool isMM2S,
                                std::map<int, int> chan_renumber_reverse_map,
                                std::vector<Attribute> &dma_allocations) {
    std::set<int64_t> dma_ids;
    seg.walk([&](air::MemcpyInterface o) {
      if (!o->getParentOfType<air::HerdOp>())
        dma_ids.insert(o.getId());
    });

    auto c = seg.getColOffset();
    auto r = seg.getRowOffset();
    int64_t col_offset = c ? *c : 0;
    int64_t row_offset = r ? *r : 0;

    for (auto &t : allocs) {
      auto tileOp = t.dma_tile;
      int64_t col = t.col - col_offset;
      int64_t row = t.row - row_offset;
      int64_t chan = isMM2S ? t.dma_channel.channel + 2 : t.dma_channel.channel;

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

  void annotateMetadataPerShimAIRChannel(air::ChannelInterface chan_o,
                                         MemRefType memref_ty,
                                         StringAttr dma_name_attr) {
    for (auto the_other_chan_o : getTheOtherChannelOpThroughSymbol(chan_o)) {
      the_other_chan_o->setAttr(
          "metadata", FlatSymbolRefAttr::get(the_other_chan_o->getContext(),
                                             dma_name_attr));
    }
  }

  // AIE2 metadata format is symbolic linked to shim dma ops
  void labelAIRDmaOpsWithMetadata(air::HierarchyInterface hier,
                                  int target_op_id, StringAttr dma_name_attr,
                                  MemRefType memref_ty) {
    // Label air.dmamemcpynd ops with symbolic ref. to shimdmaalloc op
    hier.walk([&](air::MemcpyInterface o) {
      if (o.getId() == target_op_id) {
        if (isa<air::DmaMemcpyNdOp>(o.getOperation()))
          o->setAttr("metadata",
                     FlatSymbolRefAttr::get(hier->getContext(), dma_name_attr));
        else if (auto chan_o =
                     dyn_cast<air::ChannelInterface>(o.getOperation()))
          annotateMetadataPerShimAIRChannel(chan_o, memref_ty, dma_name_attr);
      }
    });
  }

  void labelAIRDmaOpsWithMetadata(
      std::vector<air::ChannelInterface> channel_ops,
      std::string specializedChanName,
      std::map<std::string, std::string> chan_to_chan_map) {
    for (auto o : channel_ops) {
      if (o.getChanName().str() == specializedChanName) {
        auto dma_name_attr =
            StringAttr::get(o->getContext(), "air_" + specializedChanName);
        o->setAttr("metadata",
                   FlatSymbolRefAttr::get(o->getContext(), dma_name_attr));

      } else if (o.getChanName().str() ==
                 chan_to_chan_map[specializedChanName]) {
        auto dma_name_attr =
            StringAttr::get(o->getContext(), "air_" + specializedChanName);
        o->setAttr("metadata",
                   FlatSymbolRefAttr::get(o->getContext(), dma_name_attr));
      }
    }
  }

  // Create channel name as string, in case if repetition due to channel
  // specialization
  std::string createChannelSubName(AIE::DeviceOp device, std::string dma_name) {
    std::string new_cname = dma_name;
    std::string cname = "";
    int which_try = 0;
    while (device.lookupSymbol(new_cname))
      new_cname = dma_name + "_" + std::to_string(++which_try);
    cname = new_cname;
    return cname;
  }

  // AIE2: Get herd dma allocation info, and write as AIE::ShimDMAAllocationOp
  void createShimDMAAllocationOpsFromHerd(
      OpBuilder builder, MLIRContext *ctx, air::HerdOp herd,
      std::vector<allocation_info_t> allocs, bool isMM2S,
      std::map<int, int> chan_renumber_reverse_map) {
    std::set<int64_t> dma_ids;
    herd.walk([&](air::MemcpyInterface o) { dma_ids.insert(o.getId()); });

    for (auto &t : allocs) {
      auto tileOp = t.dma_tile;
      int64_t chan = t.dma_channel.channel;
      AIE::DMAChannelDir dir =
          isMM2S ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;

      for (int64_t id : t.dma_id) {
        int original_id = chan_renumber_reverse_map.size()
                              ? chan_renumber_reverse_map[id]
                              : id;
        if (dma_ids.count(original_id) == 0)
          continue;
        original_id = std::max(original_id, 0); // If id is -1, change to 0.
        std::string dma_name = "airMemcpyId" + std::to_string(original_id);
        dma_name = createChannelSubName(
            tileOp->getParentOfType<AIE::DeviceOp>(), dma_name);
        auto dma_name_attr = builder.getStringAttr(dma_name);

        builder.create<AIE::ShimDMAAllocationOp>(
            builder.getUnknownLoc(), SymbolRefAttr::get(ctx, dma_name_attr),
            AIE::DMAChannelDirAttr::get(ctx, dir),
            builder.getI64IntegerAttr(chan),
            builder.getI64IntegerAttr(tileOp.getCol()));

        air::MemcpyInterface tile_side_memcpy = nullptr;
        herd.walk([&](air::MemcpyInterface o) {
          if (o.getId() == original_id)
            tile_side_memcpy = o;
        });

        // Create memref.global op with memref shape
        MemRefType memref_ty;
        if (tile_side_memcpy) {
          if (auto tile_side_dmamemcpy = dyn_cast<air::DmaMemcpyNdOp>(
                  tile_side_memcpy.getOperation())) {
            if (isMM2S)
              memref_ty =
                  tile_side_memcpy.getDstMemref().getType().cast<MemRefType>();
            else
              memref_ty =
                  tile_side_memcpy.getSrcMemref().getType().cast<MemRefType>();
          } else if (auto tile_side_chan = dyn_cast<air::ChannelInterface>(
                         tile_side_memcpy.getOperation())) {
            memref_ty = tile_side_chan.getMemref().getType().cast<MemRefType>();
          }

          builder.create<memref::GlobalOp>(builder.getUnknownLoc(), dma_name,
                                           builder.getStringAttr("public"),
                                           memref_ty, nullptr, false, nullptr);
        }

        // Label airrt.dmamemcpynd ops with symbolic ref. to shimdmaalloc op
        labelAIRDmaOpsWithMetadata(
            herd, original_id,
            builder.getStringAttr("airMemcpyId" + std::to_string(original_id)),
            memref_ty);
      }
    }
  }
  void createShimDMAAllocationOpsFromSegment(
      OpBuilder builder, MLIRContext *ctx, air::SegmentOp seg,
      std::vector<allocation_info_t> allocs, bool isMM2S,
      std::map<int, int> chan_renumber_reverse_map) {
    std::set<int64_t> dma_ids;
    seg.walk([&](air::MemcpyInterface o) {
      if (!o->getParentOfType<air::HerdOp>())
        dma_ids.insert(o.getId());
    });

    for (auto &t : allocs) {
      auto tileOp = t.dma_tile;
      int64_t chan = t.dma_channel.channel;
      AIE::DMAChannelDir dir =
          isMM2S ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;

      for (int64_t id : t.dma_id) {
        int original_id = chan_renumber_reverse_map.size()
                              ? chan_renumber_reverse_map[id]
                              : id;
        if (dma_ids.count(original_id) == 0)
          continue;
        original_id = std::max(original_id, 0); // If id is -1, change to 0.
        std::string dma_name = "airMemcpyId" + std::to_string(original_id);
        auto dma_name_attr = builder.getStringAttr(dma_name);

        // Avoid redeclaration of the same metadata
        auto dev = tileOp->getParentOfType<AIE::DeviceOp>();
        auto sym = dev.lookupSymbol(dma_name_attr);
        if (sym)
          continue;

        builder.create<AIE::ShimDMAAllocationOp>(
            builder.getUnknownLoc(), SymbolRefAttr::get(ctx, dma_name_attr),
            AIE::DMAChannelDirAttr::get(ctx, dir),
            builder.getI64IntegerAttr(chan),
            builder.getI64IntegerAttr(tileOp.getCol()));

        // Create memref.global op with memref shape
        air::MemcpyInterface tile_side_memcpy = nullptr;
        MemRefType memref_ty;
        seg.walk([&](air::MemcpyInterface o) {
          if (o.getId() == original_id)
            tile_side_memcpy = o;
        });
        if (tile_side_memcpy) {
          if (auto tile_side_dmamemcpy = dyn_cast<air::DmaMemcpyNdOp>(
                  tile_side_memcpy.getOperation())) {
            if (isMM2S)
              memref_ty =
                  tile_side_memcpy.getDstMemref().getType().cast<MemRefType>();
            else
              memref_ty =
                  tile_side_memcpy.getSrcMemref().getType().cast<MemRefType>();
          } else if (auto tile_side_chan = dyn_cast<air::ChannelInterface>(
                         tile_side_memcpy.getOperation())) {
            memref_ty = tile_side_chan.getMemref().getType().cast<MemRefType>();
          }

          builder.create<memref::GlobalOp>(builder.getUnknownLoc(), dma_name,
                                           builder.getStringAttr("public"),
                                           memref_ty, nullptr, false, nullptr);
        }
        // Label airrt.dmamemcpynd ops with symbolic ref. to shimdmaalloc op
        labelAIRDmaOpsWithMetadata(seg, original_id, dma_name_attr, memref_ty);
      }
    }
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

  void allocateCoreLocksPerMemcpyOp(
      OpBuilder builder, air::MemcpyInterface memcpyOpIf,
      std::unordered_set<Operation *> &allocs_to_remap, AIE::AIEArch arch,
      TileDMAAllocator &tileDmaAlloc, int x, int y) {
    bool isAIE2 = (arch == AIE::AIEArch::AIE2);
    AIE::DMAChannel tile_channel =
        tileDmaAlloc.lookupDMAAllocation(x, y, memcpyOpIf).dma_channel;
    AIE::BufferOp bufferOp = tileDmaAlloc.getBuffer(x, y, memcpyOpIf);
    auto locks =
        tileDmaAlloc.getLockForDMA(memcpyOpIf, x, y, bufferOp.getOperation());
    auto acqLockOp = isMM2S(tile_channel) ? locks.second : locks.first;
    auto relLockOp = isMM2S(tile_channel) ? locks.first : locks.second;
    int64_t lockAqValue = -1;
    int64_t lockRelValue = -1;
    Value alloc = nullptr;
    if (isTileInbound(memcpyOpIf, (int)air::MemorySpace::L1)) {
      lockAqValue = isAIE2 ? 1 : 1;
      lockRelValue = isAIE2 ? 1 : 0;
      alloc = memcpyOpIf.getDstMemref();
    } else {
      lockAqValue = isAIE2 ? 1 : 0;
      lockRelValue = isAIE2 ? 1 : 1;
      alloc = memcpyOpIf.getSrcMemref();
    }

    if (auto bco = dyn_cast<bufferization::ToMemrefOp>(alloc.getDefiningOp()))
      builder.setInsertionPoint(bco.getOperand().getDefiningOp());
    else if (auto a = dyn_cast<memref::AllocaOp>(alloc.getDefiningOp()))
      builder.setInsertionPoint(alloc.getDefiningOp());
    else
      builder.setInsertionPoint(memcpyOpIf);

    builder.create<AIE::UseLockOp>(memcpyOpIf->getLoc(), acqLockOp,
                                   isAIE2 ? AIE::LockAction::AcquireGreaterEqual
                                          : AIE::LockAction::Acquire,
                                   lockAqValue);
    // try to find a place to put the unlock. If there are deallocs,
    // replace them with unlock. Otherwise, put them at the end.
    bool need_unlock = true;
    for (auto u : alloc.getUsers()) {
      if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
        builder.setInsertionPoint(dealloc);
        builder.create<AIE::UseLockOp>(dealloc->getLoc(), relLockOp,
                                       AIE::LockAction::Release, lockRelValue);
        // assume that the deallocs will take care of it when
        // deallocs are present
        need_unlock = false;
      }
    }
    if (need_unlock) {
      auto t = memcpyOpIf->getBlock()->getTerminator();
      builder.setInsertionPoint(t);
      builder.create<AIE::UseLockOp>(t->getLoc(), relLockOp,
                                     AIE::LockAction::Release, lockRelValue);
    }
    allocs_to_remap.insert(alloc.getDefiningOp());
  }

  template <typename dmaAllocatorTy, typename bufferOpTy, typename memOpTy>
  void generateDmaBdProgram(
      OpBuilder builder, AIE::AIEArch arch,
      std::map<std::pair<AIE::DMAChannelDir, int>, std::vector<Operation *>>
          dma_memcpys,
      dmaAllocatorTy dmaAlloc, mlir::Location loc, memOpTy mem, int x, int y) {

    // The first block
    Block *channel_head = nullptr;
    Block *end_bb = nullptr;

    for (auto &p : dma_memcpys) {
      AIE::DMAChannelDir dir = p.first.first;
      int chan = p.first.second;
      Block *start_bb = new Block();
      mem.getBody().push_back(start_bb);

      Block *first_bd = new Block();
      mem.getBody().push_back(first_bd);
      Block *next_bd = nullptr;
      for (size_t i = 0; i < p.second.size(); i++) {
        auto memcpyOp = cast<air::MemcpyInterface>(p.second[i]);
        Block *bd;
        if (i == 0)
          bd = first_bd;
        else
          bd = next_bd;
        auto b = OpBuilder::atBlockEnd(bd);
        if (i == p.second.size() - 1) {
          b.create<AIE::NextBDOp>(loc, first_bd);
        } else {
          next_bd = new Block();
          mem.getBody().push_back(next_bd);
          b.create<AIE::NextBDOp>(loc, next_bd);
        }
        bufferOpTy bufferOp = dmaAlloc.getBuffer(x, y, memcpyOp);
        auto locks =
            dmaAlloc.getLockForDMA(memcpyOp, x, y, bufferOp.getOperation());
        generateDmaBd<bufferOpTy>(loc, dir, locks, x, y, arch, bd, memcpyOp,
                                  bufferOp);
      }
      if (!channel_head) {
        channel_head = start_bb;
        end_bb = new Block();
        mem.getBody().push_back(end_bb);
        auto b = OpBuilder::atBlockBegin(channel_head);
        b.create<AIE::DMAStartOp>(loc, dir, chan, first_bd, end_bb);
        b.setInsertionPointToEnd(end_bb);
        b.create<AIE::EndOp>(loc);
      } else {
        auto b = OpBuilder::atBlockBegin(start_bb);
        b.create<AIE::DMAStartOp>(
            loc, dir, chan, first_bd,
            channel_head->getTerminator()->getSuccessor(1));
        channel_head->getTerminator()->setSuccessor(start_bb, 1);
      }
    }
  }

  template <typename bufferOpTy>
  void generateDmaBd(mlir::Location loc, AIE::DMAChannelDir dir,
                     std::pair<AIE::LockOp, AIE::LockOp> locks, int x, int y,
                     AIE::AIEArch arch, Block *bd,
                     air::MemcpyInterface memcpyOp, bufferOpTy bufferOp) {
    bool isAIE2 = (arch == AIE::AIEArch::AIE2);
    bool isMM2S = (dir == AIE::DMAChannelDir::MM2S);

    auto b = OpBuilder::atBlockEnd(bd);
    auto acqLockOp = isMM2S ? locks.first : locks.second;
    auto relLockOp = isMM2S ? locks.second : locks.first;
    b.setInsertionPointToStart(bd);
    int64_t lockAqValue = -1;
    int64_t lockRelValue = -1;
    auto aie2LockVal = getLockValuePair(arch, bufferOp->getResult(0));
    if (!isMM2S) {
      lockAqValue = isAIE2 ? aie2LockVal.first : 0;
      lockRelValue = isAIE2 ? aie2LockVal.first : 1;
    } else {
      lockAqValue = isAIE2 ? aie2LockVal.second : 1;
      lockRelValue = isAIE2 ? aie2LockVal.second : 0;
    }
    auto ndcpy = cast<air::MemcpyInterface>(memcpyOp);

    Value memref = isTileInbound(ndcpy, (int)air::MemorySpace::L1)
                       ? ndcpy.getDstMemref()
                       : ndcpy.getSrcMemref();
    SmallVector<Value> sizes = isTileInbound(ndcpy, (int)air::MemorySpace::L1)
                                   ? ndcpy.getDstSizes()
                                   : ndcpy.getSrcSizes();
    SmallVector<Value> offsets = isTileInbound(ndcpy, (int)air::MemorySpace::L1)
                                     ? ndcpy.getDstOffsets()
                                     : ndcpy.getSrcOffsets();
    SmallVector<Value> strides = isTileInbound(ndcpy, (int)air::MemorySpace::L1)
                                     ? ndcpy.getDstStrides()
                                     : ndcpy.getSrcStrides();

    int64_t len = getMemcpySizesAsInt(memref, sizes);
    int64_t offset =
        get1DOffset(offsets, strides, getElementSizeInBytes(memref.getType()));

    Value length =
        b.create<arith::ConstantIndexOp>(memcpyOp.getLoc(), len)->getResult(0);
    b.create<AIE::UseLockOp>(loc, acqLockOp,
                             isAIE2 ? AIE::LockAction::AcquireGreaterEqual
                                    : AIE::LockAction::Acquire,
                             lockAqValue);

    std::vector<AIE::BDDimLayoutAttr> dims =
        getWrapsAndStrides(sizes, strides, ndcpy->getContext());

    auto wraps_and_strides =
        AIE::BDDimLayoutArrayAttr::get(ndcpy->getContext(), ArrayRef(dims));
    bool useDefaultDataAccessPattern =
        isAIE2 ? isDefaultDataAccessPattern(sizes, strides, memref) : true;
    if (wraps_and_strides.getValue().empty() || useDefaultDataAccessPattern)
      b.create<AIE::DMABDOp>(
          loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value());
    else
      b.create<AIE::DMABDOp>(
          loc, bufferOp, offset,
          cast<arith::ConstantIndexOp>(length.getDefiningOp()).value(),
          wraps_and_strides);
    b.create<AIE::UseLockOp>(loc, relLockOp, AIE::LockAction::Release,
                             lockRelValue);
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
  void lowerAIRMemcpyOp(AIE::DeviceOp device, ShimDMAAllocator &shimDmaAlloc,
                        AIRToAIEConversionOptions options) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : device.getOps<AIE::CoreOp>())
      cores.push_back(c);

    const auto &target_model = device.getTargetModel();
    OpBuilder builder(device);

    // Unlike shimDmaAlloc, tileDmaAlloc is local to device because it does not
    // need to export to airrt.metadata
    TileDMAAllocator tileDmaAlloc(device);
    MemTileDMAAllocator memTileDmaAlloc(device);

    // Place memcpy ops onto DMA tiles, channels and flows
    placeDMAChannelsAndRouteFlows<T>(device, shimDmaAlloc, memTileDmaAlloc,
                                     tileDmaAlloc, options.generate_shim_dma);

    for (AIE::CoreOp core : cores) {
      AIE::TileOp tile = core.getTileOp();
      auto x = tile.getCol();
      auto y = tile.getRow();

      // emit the acquire and release of the L1 buffer locks
      // lock_allocation_list lock_allocs;
      std::unordered_set<Operation *> allocs_to_remap;

      for (auto &alloc : tileDmaAlloc.mm2s_allocs) {
        if (alloc.foundAlloc(x, y)) {
          for (auto o : alloc.memcpyOps) {
            assert(o);
            auto memcpyOpIf = dyn_cast<air::MemcpyInterface>(o);
            if (!memcpyOpIf)
              o->emitOpError("does not have air::MemcpyInterface");
            allocateCoreLocksPerMemcpyOp(builder, memcpyOpIf, allocs_to_remap,
                                         target_model.getTargetArch(),
                                         tileDmaAlloc, x, y);
          }
        }
      }
      for (auto &alloc : tileDmaAlloc.s2mm_allocs) {
        if (alloc.foundAlloc(x, y)) {
          for (auto o : alloc.memcpyOps) {
            assert(o);
            auto memcpyOpIf = dyn_cast<air::MemcpyInterface>(o);
            if (!memcpyOpIf)
              o->emitOpError("does not have air::MemcpyInterface");
            allocateCoreLocksPerMemcpyOp(builder, memcpyOpIf, allocs_to_remap,
                                         target_model.getTargetArch(),
                                         tileDmaAlloc, x, y);
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
      std::map<std::pair<AIE::DMAChannelDir, int>, std::vector<Operation *>>
          tile_dma_memcpys;

      for (auto &alloc : tileDmaAlloc.mm2s_allocs) {
        if (alloc.foundAlloc(x, y)) {
          std::pair<AIE::DMAChannelDir, int> mm2s_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            tile_dma_memcpys[mm2s_chan].push_back(o);
          }
        }
      }
      for (auto &alloc : tileDmaAlloc.s2mm_allocs) {
        if (alloc.foundAlloc(x, y)) {
          std::pair<AIE::DMAChannelDir, int> s2mm_chan = {
              alloc.dma_channel.direction, alloc.dma_channel.channel};
          for (auto &o : alloc.memcpyOps) {
            tile_dma_memcpys[s2mm_chan].push_back(o);
          }
        }
      }

      auto loc = core->getLoc();

      // make a aie.mem for the tile dma
      auto mem = tile.getMemOp();
      if (!mem && tile_dma_memcpys.size()) {
        builder.setInsertionPoint(core);
        mem = builder.create<AIE::MemOp>(loc, tile);
      }

      generateDmaBdProgram<TileDMAAllocator, AIE::BufferOp, AIE::MemOp>(
          builder, target_model.getTargetArch(), tile_dma_memcpys, tileDmaAlloc,
          loc, mem, x, y);
    }

    // Generate L3 DMA program

    // Gather all shim tiles and memtiles used in design
    std::vector<AIE::TileOp> shimtiles;
    std::vector<AIE::TileOp> memTileTiles;
    for (auto &alloc : shimDmaAlloc.mm2s_allocs) {
      auto tile = alloc.dma_tile;
      if (tile.isShimTile())
        push_back_if_unique<AIE::TileOp>(shimtiles, tile);
      else
        assert(false);
    }
    for (auto &alloc : memTileDmaAlloc.mm2s_allocs) {
      auto tile = alloc.dma_tile;
      if (tile.isMemTile())
        push_back_if_unique<AIE::TileOp>(memTileTiles, tile);
      else
        assert(false);
    }

    // Disable generation of shim dma program if generate_shim_dma unset
    if (!options.generate_shim_dma) {
      shimtiles.clear();
    }

    for (auto tile : shimtiles) {
      auto x = tile.getCol();
      auto y = tile.getRow();

      // Collect memcpy ops wrt each DMA channel
      std::map<std::pair<AIE::DMAChannelDir, int>, std::vector<Operation *>>
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
        builder.setInsertionPointToEnd(device.getBody());
        shimDMA = builder.create<AIE::ShimDMAOp>(builder.getUnknownLoc(),
                                                 builder.getIndexType(), tile);
      }

      auto loc = builder.getUnknownLoc();

      // Generate DMA BD program
      generateDmaBdProgram<ShimDMAAllocator, AIE::ExternalBufferOp,
                           AIE::ShimDMAOp>(
          builder, target_model.getTargetArch(), shim_dma_memcpys, shimDmaAlloc,
          loc, shimDMA, x, y);
    }

    // Generate L2 DMA program

    for (auto tile : memTileTiles) {
      auto x = tile.getCol();
      auto y = tile.getRow();

      // Collect memcpy ops wrt each DMA channel from chessboard; make aie.mem
      // dmabd program
      std::map<std::pair<AIE::DMAChannelDir, int>, std::vector<Operation *>>
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
        builder.setInsertionPointToEnd(device.getBody());
        memTileDMA = builder.create<AIE::MemTileDMAOp>(
            builder.getUnknownLoc(), builder.getIndexType(), tile);
      }

      auto loc = builder.getUnknownLoc();

      // Generate DMA BD program
      generateDmaBdProgram<MemTileDMAAllocator, AIE::BufferOp,
                           AIE::MemTileDMAOp>(
          builder, target_model.getTargetArch(), memtile_dma_memcpys,
          memTileDmaAlloc, loc, memTileDMA, x, y);
    }

    // Clear allocation_info_t allocations' memcpyOps field
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

    // erase the memcpy operations
    for (AIE::CoreOp core : cores) {
      (void)core;
      std::vector<Operation *> memcpy_ops;
      getAIRMemcpyOpInRegion<T>(device.getRegion(), memcpy_ops);

      for (auto o : memcpy_ops) {
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

  void runTestPatterns() {

    auto m = getOperation();
    auto ctx = m->getContext();

    RewritePatternSet patterns(ctx);
    std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;

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
          cloneL2AndL3MemcpysToDeviceOp(builder, d, m, true, true);
          specializeHerdAffineIf(d);
          lowerAirExecute(d);
          lowerScfAirTokens(d);
          allocL1Buffers(d, tileToHerdMap);
          allocL2Buffers(d);
          builder.setInsertionPointToStart(d.getBody());
          std::map<std::string, std::string> chan_to_chan_map;
          specializeChannelBundle(d, chan_to_chan_map);
          renumberChannelOps(d.getBody());
          // ShimDMAAllocator shimDmaAlloc(d);
          // lowerAIRMemcpyOp<air::ChannelInterface>(d, shimDmaAlloc, options);
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
    // if (clTestPatterns.find("lower-pipe-get-put") != std::string::npos)
    //   patterns.insert<LowerPipeGetPutPattern>(ctx, tileToHerdMap);
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
    std::map<std::string, std::string> chan_to_chan_map;
    if (clTestPatterns.find("specialize-channel-bundle") != std::string::npos) {
      patterns.insert<SpecializeChannelBundlePattern>(ctx, chan_to_chan_map);
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
      module.emitOpError("Invalid aie.device option");
      signalPassFailure();
      return;
    }
    AIRToAIEConversionOptions options = {
        /* .col_offset = */ clColOffset,
        /* .row_offset = */ clRowOffset,
        /* .emit_while = */ clEmitWhileLoop,
        /* .emit_herd_lock = */ clEmitHerdLock,
        /* .generate_shim_dma = */ clGenerateShimDMA,
        /* .device = */ *device};
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

      ShimDMAAllocator shimDmaAlloc(device);
      std::map<int, int> chan_renumber_reverse_map;
      ShimTileAllocator shimTileAlloc(device.getTargetModel());
      std::map<std::string, std::string> chan_to_chan_map;

      if (clUseObjFifo) {

        specializeHerdAffineIf(device);
        lowerAirExecute(device);
        lowerScfAirTokens(device);
        specializeChannelBundle(device, chan_to_chan_map);
        renumberChannelOps(device.getBody());
        LowerAIRPingPong(device);
        lowerAIRChannels(device, shimTileAlloc);
        allocL1Buffers(device, tileToHerdMap);
      } else {

        cloneL2AndL3MemcpysToDeviceOp(builder, device, module, true, true);
        specializeHerdAffineIf(device);
        lowerAirExecute(device);
        lowerScfAirTokens(device);

        allocL1Buffers(device, tileToHerdMap);
        allocL2Buffers(device);

        // Copy over L2 and L3 memcpy ops into device op
        builder.setInsertionPointToStart(device.getBody());
        specializeChannelBundle(device, chan_to_chan_map);
        renumberChannelOps(&device.getBodyRegion().front(),
                           chan_renumber_reverse_map);
        lowerAIRMemcpyOp<air::ChannelInterface>(device, shimDmaAlloc, options);
      }

      lowerAIRMemcpyOp<air::DmaMemcpyNdOp>(device, shimDmaAlloc, options);

      // lowerPipelineGetPut(device, tileToHerdMap);

      SmallVector<air::HerdOp, 4> herds;
      SmallVector<air::SegmentOp, 4> segs;
      std::set<int64_t> dma_ids;
      if (auto p = h->getParentOfType<air::SegmentOp>()) {
        auto hops = p.getOps<air::HerdOp>();
        herds.append(hops.begin(), hops.end());
        segs.push_back(p);
      } else {
        herds.push_back(h);
      }

      for (auto herd : herds) {
        std::vector<Attribute> dma_allocations;
        if (device.getTargetModel().getTargetArch() == AIE::AIEArch::AIE1) {
          // AIE1 dma metadata format
          getHerdDmaAllocations(builder, ctx, herd, shimDmaAlloc.s2mm_allocs,
                                false, chan_renumber_reverse_map,
                                dma_allocations);
          getHerdDmaAllocations(builder, ctx, herd, shimDmaAlloc.mm2s_allocs,
                                true, chan_renumber_reverse_map,
                                dma_allocations);

          auto segment_name =
              device
                  ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                  .getValue();
          auto segment_meta =
              getOrCreateSegmentMetadata(module_meta, segment_name);
          auto herd_meta = createHerdMetadata(segment_meta, herd);
          herd_meta->setAttr("dma_allocations",
                             ArrayAttr::get(ctx, dma_allocations));
        } else if (device.getTargetModel().getTargetArch() ==
                   AIE::AIEArch::AIE2) {
          // AIE2 dma metadata format
          builder.setInsertionPointToEnd(device.getBody());
          createShimDMAAllocationOpsFromHerd(builder, ctx, herd,
                                             shimDmaAlloc.s2mm_allocs, false,
                                             chan_renumber_reverse_map);
          createShimDMAAllocationOpsFromHerd(builder, ctx, herd,
                                             shimDmaAlloc.mm2s_allocs, true,
                                             chan_renumber_reverse_map);
        }
      }
      for (auto seg : segs) {
        std::vector<Attribute> dma_allocations;
        if (device.getTargetModel().getTargetArch() == AIE::AIEArch::AIE1) {
          // AIE1 memtile dma metadata format
          getSegmentDmaAllocations(builder, ctx, seg, shimDmaAlloc.mm2s_allocs,
                                   true, chan_renumber_reverse_map,
                                   dma_allocations);

          auto segment_name =
              device
                  ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                  .getValue();
          auto segment_meta =
              getOrCreateSegmentMetadata(module_meta, segment_name);
          segment_meta->setAttr("dma_allocations",
                                ArrayAttr::get(ctx, dma_allocations));
        } else if (device.getTargetModel().getTargetArch() ==
                   AIE::AIEArch::AIE2) {
          // AIE2 memtile dma metadata format
          builder.setInsertionPointToEnd(device.getBody());
          createShimDMAAllocationOpsFromSegment(builder, ctx, seg,
                                                shimDmaAlloc.s2mm_allocs, false,
                                                chan_renumber_reverse_map);
          createShimDMAAllocationOpsFromSegment(builder, ctx, seg,
                                                shimDmaAlloc.mm2s_allocs, true,
                                                chan_renumber_reverse_map);
        }
      }

      // ObjectFifo metadata linkage
      auto f = h->getParentOfType<func::FuncOp>();

      std::vector<air::ChannelInterface> channel_ops;
      f.walk([&](air::ChannelInterface o) {
        if (!o->getParentOfType<air::HerdOp>())
          channel_ops.push_back(o);
      });
      for (auto &t : shimTileAlloc.s2mm_allocs)
        for (auto n : t.chan_names)
          labelAIRDmaOpsWithMetadata(channel_ops, n, chan_to_chan_map);
      for (auto &t : shimTileAlloc.mm2s_allocs)
        for (auto n : t.chan_names)
          labelAIRDmaOpsWithMetadata(channel_ops, n, chan_to_chan_map);

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
    p->emitOpError("Invalid aie.device option");
    return failure();
  }
  AIRToAIEConversionOptions options = {/* .col_offset = */ 7,
                                       /* .row_offset = */ 2,
                                       /* .emit_while = */ false,
                                       /* .emit_herd_lock = */ false,
                                       /* .generate_shim_dma = */ false,
                                       /* .device = */ *device};
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRToAIEPass(const AIRToAIEOptions &options) {
  return std::make_unique<AIRToAIEPass>(options);
}

std::unique_ptr<mlir::Pass> createAIRSplitDevicesPass() {
  return std::make_unique<SplitAIEDevicesPass>();
}

} // namespace air
} // namespace xilinx
