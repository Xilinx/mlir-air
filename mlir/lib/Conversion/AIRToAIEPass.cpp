// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"
#include "aie/AIEDialect.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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

bool isMM2S(AIE::DMAChan channel) {
  if ((channel == AIE::DMAChan::MM2S0) || (channel == AIE::DMAChan::MM2S1))
    return true;
  else
    return false;
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
                            << ", l2 col =" << dma_col
                            << ", l2 chan =" << dma_channel << "\n");

    return dma_tile;
  }

  AIE::DMAChan getChannel(ModuleOp aie_module, air::DmaMemcpyInterface &dmaOp,
                          int64_t tile_channel, int64_t col, int64_t row) {
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
        if (tile_channel == t.tile_channel) {
          chan = t.dma_channel;
        }
      }
    }
    assert(chan != -1);

    LLVM_DEBUG(llvm::outs() << "isMM2S = " << isMM2S << ", col =" << col
                            << ", row = " << row << " chan =" << chan << "\n");

    if (isMM2S)
      return (AIE::DMAChan)((uint64_t)AIE::DMAChan::MM2S0 + chan);
    else
      return (AIE::DMAChan)((uint64_t)AIE::DMAChan::S2MM0 + chan);
  }
};

class AIRToAIEPass : public AIRToAIEBase<AIRToAIEPass> {

public:
  AIRToAIEPass() = default;
  AIRToAIEPass(const AIRToAIEPass &pass) {}

  Option<std::string> AIRToAIEModulePrefix{
      *this, "output-prefix",
      llvm::cl::desc("Output filename prefix for AIE module"),
      llvm::cl::init("-")};

  Option<std::string> AIRToAIEELFFilename{
      *this, "elf-file",
      llvm::cl::desc("Specify elf file to add as an attribute of AIE.core"),
      llvm::cl::init("-")};

  Option<int> AIRToAIERowOffset{
      *this, "row-offset",
      llvm::cl::desc("The start row for any output herds"), llvm::cl::init(0)};

  Option<int> AIRToAIEColOffset{
      *this, "col-offset",
      llvm::cl::desc("The start col for any output herds"), llvm::cl::init(0)};

  Option<bool> AIRToAIEEmitWhileLoop{
      *this, "emit-while-loop",
      llvm::cl::desc("Emit while(1) around AIE code"), llvm::cl::init(false)};

  typedef std::vector<std::tuple<AIE::BufferOp, AIE::LockOp, AIE::DMAChan>>
      lock_allocation_list;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
    registry.insert<xilinx::airrt::AIRRtDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  uint64_t BufferId;

  const int tile_dma_channels = 2;
  std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>>
      tile_dma_S2MM_allocs;
  std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>>
      tile_dma_MM2S_allocs;

  // A very simple scheme to allocate channels for dma operations:
  //  <description>
  AIE::DMAChan getTileDMAChannel(ModuleOp aie_module,
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
      return (AIE::DMAChan)((uint64_t)AIE::DMAChan::MM2S0 + chan);
    else
      return (AIE::DMAChan)((uint64_t)AIE::DMAChan::S2MM0 + chan);
  }

  AIE::BufferOp getBufferForTileDMA(ModuleOp aie_module,
                                    air::DmaMemcpyInterface &dmaOp,
                                    BlockAndValueMapping &map, int col,
                                    int row) {
    AIE::DMAChan channel = getTileDMAChannel(aie_module, dmaOp, col, row);
    Value buffer;
    if (isMM2S(channel)) {
      buffer = map.lookupOrDefault(dmaOp.getSrcMemref());
    } else {
      buffer = map.lookupOrDefault(dmaOp.getDstMemref());
    }
    AIE::BufferOp bufferOp = buffer.getDefiningOp<AIE::BufferOp>();
    // if (!bufferOp)
    //   buffer.dump();
    return bufferOp;
  }

  AIE::LockOp allocateLockOp(ModuleOp aie_module, AIE::TileOp tile) {
    std::set<int> ids;
    aie_module.walk([&](AIE::LockOp lock) {
      if (cast<xilinx::AIE::TileOp>(lock.tile().getDefiningOp()) == tile)
        ids.insert(lock.getLockID());
    });
    int new_id = 0;
    while (ids.count(new_id))
      new_id++;
    OpBuilder b(aie_module);
    Operation *t = tile.getOperation();
    while (dyn_cast_or_null<AIE::TileOp>(t->getNextNode()))
      t = t->getNextNode();
    b.setInsertionPointAfter(t);
    return b.create<AIE::LockOp>(tile.getLoc(), tile, new_id);
  }

  AIE::BufferOp allocateBufferOp(ModuleOp module, MemRefType memrefTy,
                                 AIE::TileOp tile,
                                 mlir::StringAttr attr = nullptr, int x = -1,
                                 int y = -1) {
    OpBuilder builder(module);
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
                      StringAttr::get(module.getContext(), ss.str()));

    return bufferOp;
  }

  AIE::LockOp getLockForTileDMA(ModuleOp aie_module,
                                air::DmaMemcpyInterface &dmaOp,
                                lock_allocation_list &info,
                                BlockAndValueMapping &map, int col, int row) {
    AIE::BufferOp bufferOp =
        getBufferForTileDMA(aie_module, dmaOp, map, col, row);
    AIE::DMAChan channel = getTileDMAChannel(aie_module, dmaOp, col, row);
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
        if (source == fop.source() && dest == fop.dest() &&
            sourceBundle == fop.sourceBundle() &&
            destBundle == fop.destBundle() &&
            sourceChannel == fop.sourceChannel() &&
            destChannel == fop.destChannel())
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

  std::vector<int> l2_dma_cols{7, 8, 9, 10};
  const int l2_dma_channels = 2;

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

  std::map<AIE::DMAChan, std::vector<Operation *>>
  getDmaSchedules(AIE::CoreOp core, int x, int y, DMAAllocator &shim_dma_alloc,
                  DMAAllocator &l2_dma_alloc,
                  std::vector<AIE::TileOp> &shim_dma_inits,
                  std::vector<AIE::TileOp> &l2_dma_tiles) {

    std::map<AIE::DMAChan, std::vector<Operation *>> tile_dma_copies;
    std::vector<Operation *> dma_memcpy_ops;
    getAIRDmaMemcpyInRegion(core.body(), dma_memcpy_ops);

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
           dst_space == (int)air::MemorySpace::L2)) {
        o->erase();
        continue;
      }

      AIE::DMAChan tile_channel = getTileDMAChannel(aie_module, dmaOpIf, x, y);

      if ((src_space == (int)air::MemorySpace::L3 &&
           dst_space == (int)air::MemorySpace::L1) ||
          (src_space == (int)air::MemorySpace::L1 &&
           dst_space == (int)air::MemorySpace::L3)) {

        // copy between L1 and external memory, use shim dma
        tile_channel = getTileDMAChannel(aie_module, dmaOpIf, x, y);
        AIE::TileOp shim_tile = shim_dma_alloc.getTile(
            aie_module, dmaOpIf, (int64_t)tile_channel, x, y);
        AIE::DMAChan shim_channel = shim_dma_alloc.getChannel(
            aie_module, dmaOpIf, (int64_t)tile_channel, x, y);

        LLVM_DEBUG(llvm::outs() << "Shim channel is " << (uint64_t)shim_channel
                                << " for x=" << x << ", y=" << y << "\n");

        if (((uint64_t)shim_channel >= (uint64_t)AIE::DMAChan::S2MM0) &&
            ((uint64_t)shim_channel <
             ((uint64_t)AIE::DMAChan::S2MM0 + shim_dma_channels))) {
          getFlowOp(aie_module, tile, AIE::WireBundle::DMA,
                    (uint32_t)tile_channel - 2, shim_tile, AIE::WireBundle::DMA,
                    ((uint32_t)shim_channel) % shim_dma_channels);
        } else {
          getFlowOp(aie_module, shim_tile, AIE::WireBundle::DMA,
                    ((uint32_t)shim_channel) % shim_dma_channels, tile,
                    AIE::WireBundle::DMA, (uint32_t)tile_channel);
        }

      } else if ((src_space == (int)air::MemorySpace::L2 &&
                  dst_space == (int)air::MemorySpace::L1) ||
                 (src_space == (int)air::MemorySpace::L1 &&
                  dst_space == (int)air::MemorySpace::L2)) {
        // copy between L1 and L2
        tile_channel = getTileDMAChannel(aie_module, dmaOpIf, x, y);
        AIE::TileOp l2_tile = l2_dma_alloc.getTile(aie_module, dmaOpIf,
                                                   (int64_t)tile_channel, x, y);
        AIE::DMAChan l2_channel = l2_dma_alloc.getChannel(
            aie_module, dmaOpIf, (int64_t)tile_channel, x, y);

        OpBuilder builder(aie_module);
        builder.setInsertionPointToEnd(&(aie_module.getBodyRegion().front()));

        if (((uint64_t)l2_channel >= (uint64_t)AIE::DMAChan::S2MM0) &&
            ((uint64_t)l2_channel <
             ((uint64_t)AIE::DMAChan::S2MM0 + l2_dma_channels))) {
          getFlowOp(aie_module, tile, AIE::WireBundle::DMA,
                    (uint32_t)tile_channel - 2, l2_tile, AIE::WireBundle::PLIO,
                    ((uint32_t)l2_channel) % l2_dma_channels);
        } else {
          getFlowOp(aie_module, l2_tile, AIE::WireBundle::PLIO,
                    ((uint32_t)l2_channel) % l2_dma_channels + 4, tile,
                    AIE::WireBundle::DMA, (uint32_t)tile_channel);
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
                               air::PartitionOp partition) {

    std::string name = "partition_0";
    if (partition)
      if (auto attr = partition->getAttrOfType<StringAttr>(
              SymbolTable::getSymbolAttrName()))
        name = attr.getValue().str();

    for (auto pm :
         module_meta.partitions().front().getOps<airrt::PartitionMetadataOp>())
      if (name == pm.sym_name().str())
        return pm;

    auto builder = OpBuilder::atBlockTerminator(module_meta.getBody());
    auto loc = builder.getUnknownLoc();
    auto partition_meta = builder.create<airrt::PartitionMetadataOp>(loc, name);
    builder.createBlock(&partition_meta.herds());
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

  bool isInSet(int64_t x, int64_t y, AffineIfOp aif) {
    auto is = aif.getIntegerSet();
    if (is.getConstraints().size() != 2)
      return false;

    SmallVector<AffineExpr, 2> dims{
        getAffineConstantExpr(x, aif->getContext()),
        getAffineConstantExpr(y, aif->getContext()),
    };

    auto newIs = is.replaceDimsAndSymbols({}, dims, 0, 2);
    auto constraints = newIs.getConstraints();
    auto eqFlags = newIs.getEqFlags();

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

  void specializeHerdAffineIf(AIE::CoreOp core) {
    SmallVector<Operation *, 8> erased;
    core.walk([&](AffineIfOp aif) {
      SmallVector<int64_t, 2> operands;
      for (auto o : aif.getOperands()) {
        auto v = dyn_cast<arith::ConstantIndexOp>(o.getDefiningOp());
        if (!v)
          return;
        operands.push_back(v.value());
      }
      auto x = operands[0];
      auto y = operands[1];
      Block *bb = nullptr;
      if (isInSet(x, y, aif)) {
        bb = aif.getThenBlock();
      } else if (aif.hasElse()) {
        bb = aif.getElseBlock();
      }
      if (bb) {
        auto t = bb->getTerminator();
        auto &ops = bb->getOperations();
        aif->getBlock()->getOperations().splice(Block::iterator(aif), ops,
                                                ops.begin(), --ops.end());
        for (int i = 0, e = aif.getNumResults(); i < e; i++)
          aif.getResult(i).replaceAllUsesWith(t->getOperand(i));
      }
      erased.push_back(aif);
    });
    for (auto a : erased)
      a->erase();
  }

  // remove any remaining air.execute, turning them
  // back into sequential code
  void lowerAirRegions(AIE::CoreOp core) {
    SmallVector<Operation *, 8> erased;
    core.walk([&](xilinx::air::ExecuteOp rop) {
      auto &bb = rop.body().front();
      unsigned idx = 0;
      for (auto &arg : bb.getArguments()) {
        arg.replaceAllUsesWith(rop.getOperand(idx));
        idx++;
      }
      if (rop.getNumResults() > 0) {
        OpBuilder builder(rop);
        auto w = builder.create<xilinx::air::WaitAllOp>(
            core.getLoc(), air::AsyncTokenType::get(rop->getContext()),
            SmallVector<Value, 1>{});
        rop.getResult(0).replaceAllUsesWith(w.getResult(0));
      }
      rop.walk([&](xilinx::air::ExecuteTerminatorOp t) {
        int resultIdx = 1;
        for (auto r : t->getOperands())
          rop.getResult(resultIdx++).replaceAllUsesWith(r);
      });
      auto &ops = bb.getOperations();
      rop->getBlock()->getOperations().splice(Block::iterator(rop), ops,
                                              ops.begin(), --ops.end());
      erased.push_back(rop);
    });
    for (auto a : erased)
      a->erase();
  }

  // remove air.async.token from scf.for
  void lowerScfAirEvents(AIE::CoreOp core) {
    SmallVector<Operation *, 8> erased;
    core.walk([&](scf::ForOp fop) {
      OpBuilder builder(fop);
      if (!fop.getNumIterOperands())
        return;
      SmallVector<Value, 4> iter_args;
      SmallVector<unsigned, 4> iter_args_idx;

      // erase air.event from the iter args
      for (OpOperand &oper : fop.getIterOpOperands()) {
        Value v = oper.get();
        BlockArgument block_arg = fop.getRegionIterArgForOpOperand(oper);
        if (v.getType().isa<xilinx::air::AsyncTokenType>()) {
          block_arg.replaceAllUsesWith(v);
          iter_args_idx.push_back(block_arg.getArgNumber());
        } else {
          iter_args.push_back(v);
        }
      }

      // if none of the iter args were air.async.token, return
      if (iter_args.size() == fop.getNumIterOperands())
        return;

      // make a new scf.for without air.async.token
      BlockAndValueMapping remap;
      auto new_fop = builder.create<scf::ForOp>(
          fop->getLoc(), fop.getLowerBound(), fop.getUpperBound(),
          fop.getStep(), iter_args);
      auto &new_region = new_fop.getRegion();
      fop.getRegion().cloneInto(&new_region, new_region.begin(), remap);
      new_region.back().erase();
      new_region.front().eraseArguments(iter_args_idx);

      // use the new for op's results
      int idx = 0;
      for (auto r : fop.getResults()) {
        if (r.getType().isa<xilinx::air::AsyncTokenType>())
          r.replaceAllUsesWith(
              builder
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
      builder.setInsertionPoint(yield);
      SmallVector<Value, 4> yield_operands;
      for (auto o : yield->getOperands()) {
        if (!o.getType().isa<xilinx::air::AsyncTokenType>())
          yield_operands.push_back(o);
      }
      builder.create<scf::YieldOp>(yield->getLoc(), yield_operands);
      yield->erase();

      erased.push_back(fop);
    });
    for (auto e : erased)
      e->erase();
  }

  // This function replaces PipelinePutOp/PipelineGetOp pairs with a
  // shared AIE.buffer + AIE.lock. This is a single-buffered implementation
  // with exclusive access to the buffer controlled by the lock. i.e. FIXME.
  void lowerPipelineGetPut(ModuleOp &aie_module,
                           std::map<AIE::TileOp, air::HerdOp> tileToHerdMap) {
    aie_module.walk([&](air::PipelinePutOp putOp) {
      auto core = putOp->getParentOfType<AIE::CoreOp>();
      assert(core);

      auto herd = tileToHerdMap[core.getTileOp()];
      int64_t col_offset = herd.getColOffset();
      int64_t row_offset = herd.getRowOffset();

      auto other_x = cast<arith::ConstantIndexOp>(putOp.dst0().getDefiningOp());
      auto other_y = cast<arith::ConstantIndexOp>(putOp.dst1().getDefiningOp());
      auto other_core = getPhysTileOp(aie_module, other_x.value() + col_offset,
                                      other_y.value() + row_offset)
                            .getCoreOp();
      assert(other_core);

      air::PipelineGetOp getOp;
      other_core.walk([&](air::PipelineGetOp pgo) { getOp = pgo; });
      assert(getOp && getOp->getNumResults() == (putOp->getNumOperands() - 2));

      for (auto p :
           llvm::zip(putOp->getOperands().drop_front(2), getOp->getResults())) {
        OpBuilder b(core);
        auto o = std::get<0>(p); // operand of put
        auto r = std::get<1>(p); // result of get
        // for each ranked tensor put (yielded) by the tile
        if (RankedTensorType tt = o.getType().dyn_cast<RankedTensorType>()) {
          auto memrefTy = MemRefType::get(tt.getShape(), tt.getElementType(),
                                          {}, (int)air::MemorySpace::L1);
          // allocate buffer+lock
          auto buf = allocateBufferOp(
              aie_module, memrefTy, core.getTileOp(),
              StringAttr::get(aie_module.getContext(), "pipebuf"));
          auto lockOp = allocateLockOp(aie_module, core.getTileOp());

          // acquire the lock for write on the put side
          b.setInsertionPoint(putOp);
          b.create<AIE::UseLockOp>(putOp->getLoc(), lockOp, 0,
                                   AIE::LockAction::Acquire);
          b.create<memref::TensorStoreOp>(putOp->getLoc(), o, buf);
          b.create<AIE::UseLockOp>(putOp->getLoc(), lockOp, 1,
                                   AIE::LockAction::Release);

          // acquire the lock for read on the get side
          b.setInsertionPoint(getOp);
          b.create<AIE::UseLockOp>(getOp->getLoc(), lockOp, 1,
                                   AIE::LockAction::Acquire);
          auto loadOp =
              b.create<bufferization::ToTensorOp>(getOp->getLoc(), buf);
          b.create<AIE::UseLockOp>(getOp->getLoc(), lockOp, 0,
                                   AIE::LockAction::Release);
          r.replaceAllUsesWith(loadOp.getResult());
        } else {
          llvm::errs()
              << "error, unsupported air.pipeline.yield operand type\n";
          assert(0 && "Unsupported");
        }
      }
      putOp->erase();
    });
  }

  void cleanupPipelineOps(ModuleOp aie_module) {
    // erase air.pipeline.terminator ops
    aie_module.walk([&](air::PipelineTerminatorOp op) { op->erase(); });

    // yank the ops out of the pipeline op and erase it
    aie_module.walk([&](air::HerdPipelineOp op) {
      op->getBlock()->getOperations().splice(Block::iterator(op),
                                             op.body().front().getOperations());
      op->erase();
    });
  }

  void lowerAirDmaMemcpy(ModuleOp module, DMAAllocator &shimDmaAlloc,
                         DMAAllocator &L2DmaAlloc,
                         BlockAndValueMapping &remap) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : module.getOps<AIE::CoreOp>())
      cores.push_back(c);

    OpBuilder builder(module);

    for (AIE::CoreOp core : cores) {
      AIE::TileOp tile = core.getTileOp();
      auto x = tile.col();
      auto y = tile.row();

      // BlockAndValueMapping remap;

      std::vector<AIE::TileOp> shim_dma_inits;
      std::vector<AIE::TileOp> l2_dma_tiles;

      // collect dma operations and generate a schedule
      std::map<AIE::DMAChan, std::vector<Operation *>> tile_dma_copies =
          getDmaSchedules(core, x, y, shimDmaAlloc, L2DmaAlloc, shim_dma_inits,
                          l2_dma_tiles);

      // emit the acquire and release of the L1 buffer locks
      lock_allocation_list lock_allocs;
      std::unordered_set<Operation *> allocs_to_remap;
      for (auto p : tile_dma_copies) {
        for (auto o : p.second) {
          auto dmaOpIf = cast<air::DmaMemcpyInterface>(o);
          AIE::DMAChan tile_channel = getTileDMAChannel(module, dmaOpIf, x, y);
          AIE::LockOp lockOp =
              getLockForTileDMA(module, dmaOpIf, lock_allocs, remap, x, y);
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
          else
            builder.setInsertionPoint(alloc.getDefiningOp());
          //          builder.setInsertionPoint(dmaOpIf);

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
            auto t = alloc.getParentBlock()->getTerminator();
            builder.setInsertionPoint(t);
            builder.create<AIE::UseLockOp>(t->getLoc(), lockOp, lockRelValue,
                                           AIE::LockAction::Release);
          }
          allocs_to_remap.insert(alloc.getDefiningOp());
        }
      }
      for (auto o : allocs_to_remap) {
        Value alloc = o->getResult(0);
        for (auto u : alloc.getUsers())
          if (auto dealloc = dyn_cast<memref::DeallocOp>(u)) {
            dealloc.erase();
            break;
          }
        alloc.replaceAllUsesWith(remap.lookup(alloc));
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

        LLVM_DEBUG(llvm::outs() << " TILE dma channel is " << (uint64_t)channel
                                << " for x=" << x << ", y=" << y << "\n");

        Block *start_bb = new Block();
        mem.body().push_back(start_bb);

        Block *first_bd = new Block();
        mem.body().push_back(first_bd);
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
            b.create<cf::BranchOp>(loc, first_bd);
          } else {
            next_bd = new Block();
            mem.body().push_back(next_bd);
            b.create<cf::BranchOp>(loc, next_bd);
          }
          AIE::BufferOp bufferOp =
              getBufferForTileDMA(module, dmaOp, remap, x, y);
          AIE::LockOp lockOp =
              getLockForTileDMA(module, dmaOp, lock_allocs, remap, x, y);
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
                             ? ndcpy.dst_sizes()
                             : ndcpy.src_sizes();
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
          mem.body().push_back(end_bb);
          auto b = OpBuilder::atBlockBegin(channel_head);
          b.create<AIE::DMAStartOp>(loc, channel, first_bd, end_bb);
          b.setInsertionPointToEnd(end_bb);
          b.create<AIE::EndOp>(loc);
        } else {
          auto b = OpBuilder::atBlockBegin(start_bb);
          b.create<AIE::DMAStartOp>(
              loc, channel, first_bd,
              channel_head->getTerminator()->getSuccessor(1));
          channel_head->getTerminator()->setSuccessor(start_bb, 1);
        }
      }

      // erase the dma copy operations
      for (auto p : tile_dma_copies)
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

      // replace all remaining uses of AllocOps with the
      // corresponding AIE.buffer from the remapping.
      // erase all AllocOps
      std::vector<Operation *> erase_ops;
      core.walk([&](Operation *op) {
        if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
          if (alloc.getType().getMemorySpaceAsInt() ==
              (int)air::MemorySpace::L1)
            alloc->replaceAllUsesWith(
                remap.lookup(alloc.getResult()).getDefiningOp());
          erase_ops.push_back(op);
        }
      });
      for (auto o : erase_ops)
        o->erase();
    }
  }

  void allocL1Buffers(ModuleOp module, BlockAndValueMapping &remap,
                      std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap) {

    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : module.getOps<AIE::CoreOp>())
      cores.push_back(c);

    OpBuilder builder(module);

    for (AIE::CoreOp core : cores) {
      AIE::TileOp tile = core.getTileOp();
      // generate a buffer for each alloc into L1
      core.walk([&](Operation *op) {
        auto alloc = dyn_cast<memref::AllocOp>(op);
        auto cast = dyn_cast<bufferization::ToMemrefOp>(op);
        if (!(alloc || cast))
          return;

        MemRefType memrefTy = nullptr;
        if (alloc)
          memrefTy = alloc.getType();
        if (cast)
          memrefTy = cast.getType().cast<MemRefType>();

        if (memrefTy.getMemorySpaceAsInt() != (int)air::MemorySpace::L1)
          return;

        builder.setInsertionPointAfter(tile);
        auto herd = tileToHerdMap[core.getTileOp()];
        int64_t col_offset = herd.getColOffset();
        int64_t row_offset = herd.getRowOffset();

        auto buffer = allocateBufferOp(
            module, memrefTy, tile,
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
            tile.col() - col_offset, tile.row() - row_offset);
        // map uses of the alloc to the new buffer
        remap.map(op->getResult(0), buffer);

        builder.setInsertionPoint(op);
        if (cast)
          builder.create<memref::TensorStoreOp>(cast.getLoc(),
                                                cast.getOperand(), buffer);
      });
    }
  }

  void specializeHerdAffineIf(ModuleOp module) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : module.getOps<AIE::CoreOp>())
      cores.push_back(c);
    for (AIE::CoreOp core : cores)
      specializeHerdAffineIf(core);
  }

  void lowerAirRegions(ModuleOp module) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : module.getOps<AIE::CoreOp>())
      cores.push_back(c);
    for (AIE::CoreOp core : cores)
      lowerAirRegions(core);
  }

  void lowerScfAirEvents(ModuleOp module) {
    SmallVector<AIE::CoreOp, 32> cores;
    for (auto c : module.getOps<AIE::CoreOp>())
      cores.push_back(c);
    for (AIE::CoreOp core : cores)
      lowerScfAirEvents(core);
  }

  void createAIEModulesAndOutlineCores(
      ModuleOp module,
      std::vector<std::pair<ModuleOp, xilinx::air::HerdOp>> &aie_modules,
      std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap) {

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
      builder.setInsertionPointToStart(aie_module.getBody());

      SmallVector<Value, 2> herd_size = h.getSizeOperands();
      if (!isa<arith::ConstantIndexOp>(herd_size[0].getDefiningOp()) ||
          !isa<arith::ConstantIndexOp>(herd_size[1].getDefiningOp())) {
        llvm::errs() << "Only constant sized herds are supported";
        return;
      }

      int64_t herd_size_x =
          cast<arith::ConstantIndexOp>(herd_size[0].getDefiningOp()).value();
      int64_t herd_size_y =
          cast<arith::ConstantIndexOp>(herd_size[1].getDefiningOp()).value();

      // use the command line offsets unless the attribute is present
      int64_t col_offset = AIRToAIEColOffset;
      int64_t row_offset = AIRToAIERowOffset;
      auto col_name = xilinx::air::HerdOp::getColOffsetAttrName();
      auto row_name = xilinx::air::HerdOp::getRowOffsetAttrName();
      if (h->getAttrOfType<IntegerAttr>(col_name))
        col_offset = h.getColOffset();
      else
        h->setAttr(col_name,
                   IntegerAttr::get(IntegerType::get(h->getContext(), 32),
                                    col_offset));
      if (h->getAttrOfType<IntegerAttr>(row_name))
        row_offset = h.getRowOffset();
      else
        h->setAttr(row_name,
                   IntegerAttr::get(IntegerType::get(h->getContext(), 32),
                                    row_offset));

      auto ctx = module.getContext();
      for (auto y = 0; y < herd_size_y; y++) {
        for (auto x = 0; x < herd_size_x; x++) {
          auto hloc = h.getLoc();
          BlockAndValueMapping remap;
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
            core->setAttr(
                "elf_file",
                StringAttr::get(ctx, herd_name + "_core_" +
                                         std::to_string(phys_x) + "_" +
                                         std::to_string(phys_y) + ".elf"));
            if (auto a = h->getAttrOfType<StringAttr>("link_with"))
              core->setAttr("link_with", a);
          }

          // the buffers and locks created below need to go before the core and
          // mem
          builder.setInsertionPoint(core);

          assert((h.body().getBlocks().size() == 1) &&
                 "Launch body can only contain one Block");

          // generate the AIE.core body
          //
          OpBuilder core_builder(core);
          Block *core_bb = core_builder.createBlock(&core.body());

          Block *entry_bb = core_builder.createBlock(core_bb);
          core_builder.setInsertionPointToEnd(entry_bb);
          core_builder.create<cf::BranchOp>(hloc, core_bb);
          core_builder.setInsertionPointToEnd(core_bb);

          // map the tile ids and herd size to constants
          remap.map(h.getIds()[0],
                    core_builder.create<arith::ConstantIndexOp>(hloc, x));
          remap.map(h.getIds()[1],
                    core_builder.create<arith::ConstantIndexOp>(hloc, y));
          remap.map(h.getSize()[0], core_builder.create<arith::ConstantIndexOp>(
                                        hloc, herd_size_x));
          remap.map(h.getSize()[1], core_builder.create<arith::ConstantIndexOp>(
                                        hloc, herd_size_y));

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
                                       builder.getStringAttr("public"),
                                       memrefTy, nullptr, false, nullptr);

            auto m = core_builder.create<memref::GetGlobalOp>(
                hloc, SmallVector<Type, 1>{a.getType()}, sym_name);
            remap.map(a, m);
          }

          Region &r = h.getRegion();
          r.cloneInto(&core.body(), remap);

          Block *launch_bb = remap.lookup(&r.front());
          core_builder.create<cf::BranchOp>(hloc, launch_bb);
          core_builder.setInsertionPoint(launch_bb->getTerminator());

          if (AIRToAIEEmitWhileLoop)
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
  }

  void
  cleanupAIEModules(std::vector<std::pair<ModuleOp, air::HerdOp>> aie_modules) {
    for (auto p : aie_modules) {
      // quick n dirty dce
      auto aie_module = std::get<0>(p);
      size_t n = 0;
      do {
        std::vector<Operation *> dead_code;
        aie_module.walk([&](AIE::CoreOp core) {
          core.walk([&](Operation *op) {
            // this avoids terminators and loops
            if (isa<xilinx::air::DmaMemcpyNdOp>(op))
              return;
            if (op->getNumResults() == 0)
              return;
            if (op->getNumRegions())
              return;
            bool used = false;
            for (unsigned i = 0, e = op->getNumResults(); i != e; ++i)
              used |= !op->getResult(i).use_empty();
            if (!used) {
              auto end = std::end(dead_code);
              if (std::find(std::begin(dead_code), end, op) == end)
                dead_code.push_back(op);
            }
            for (Block &b : core.getRegion().getBlocks()) {
              std::vector<unsigned> erased_arg_idx;
              for (int i = 0, e = b.getNumArguments(); i < e; i++) {
                if (b.getArgument(i).use_empty())
                  erased_arg_idx.push_back(i);
              }
              b.eraseArguments(erased_arg_idx);
            }
          });
        });
        n = dead_code.size();
        for (auto op : dead_code) {
          op->erase();
        }
      } while (n);
    }
  }

  void runOnOperation() override {

    auto module = getOperation();
    OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());

    auto loc = builder.getUnknownLoc();
    auto module_meta = builder.create<airrt::ModuleMetadataOp>(loc);
    builder.createBlock(&module_meta.partitions());
    builder.create<airrt::ModuleMetadataTerminatorOp>(loc);

    // If we have multiple herds then we must emit them into different aie
    // modules to avoid resource conflicts in the AIE physical dialect.
    std::vector<std::pair<ModuleOp, air::HerdOp>> aie_modules;

    std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
    createAIEModulesAndOutlineCores(module, aie_modules, tileToHerdMap);

    std::set<ModuleOp> seen;
    for (auto &p : aie_modules) {
      ModuleOp m = std::get<0>(p);
      xilinx::air::HerdOp h = std::get<1>(p);
      auto ctx = m->getContext();

      if (seen.find(m) == seen.end()) {
        seen.insert(m);

        specializeHerdAffineIf(m);
        lowerAirRegions(m);
        lowerScfAirEvents(m);

        // buffer_map maps allocs to their AIE.buffer replacement
        BlockAndValueMapping buffer_map;
        allocL1Buffers(m, buffer_map, tileToHerdMap);

        DMAAllocator shimDmaAlloc(shim_dma_cols, shim_dma_channels);
        DMAAllocator L2DmaAlloc(l2_dma_cols, l2_dma_channels);

        lowerAirDmaMemcpy(m, shimDmaAlloc, L2DmaAlloc, buffer_map);
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
          int64_t col_offset = herd.getColOffset();
          int64_t row_offset = herd.getRowOffset();

          // createAIRRtMetadata(module_meta, shimDmaAlloc, L2DmaAlloc);
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
                                 builder.getI64IntegerAttr(tileOp.col())));
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
                                 builder.getI64IntegerAttr(tileOp.col())));
              dma_allocations.push_back(DictionaryAttr::get(ctx, attrs));
            }
          }
          for (auto &t : L2DmaAlloc.s2mm_allocs) {
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
                                 builder.getI64IntegerAttr(tileOp.col())));
              dma_allocations.push_back(DictionaryAttr::get(ctx, attrs));
            }
          }
          for (auto &t : L2DmaAlloc.mm2s_allocs) {
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
                                 builder.getI64IntegerAttr(tileOp.col())));
              dma_allocations.push_back(DictionaryAttr::get(ctx, attrs));
            }
          }
          auto partition_meta = getOrCreatePartitionMetadata(
              module_meta, h->getParentOfType<air::PartitionOp>());
          auto herd_meta = createHerdMetadata(partition_meta, herd);
          herd_meta->setAttr("dma_allocations",
                             ArrayAttr::get(ctx, dma_allocations));
        }
        tile_dma_S2MM_allocs.clear();
        tile_dma_MM2S_allocs.clear();
      }
    }
    cleanupAIEModules(aie_modules);

    // emit aie_modules to files or to stdout
    seen.clear();
    for (auto p : aie_modules) {
      auto aie_module = std::get<0>(p);
      if (seen.find(aie_module) == seen.end())
        seen.insert(aie_module);
      else
        continue;
      if (AIRToAIEModulePrefix != "-") {
        if (AIRToAIEModulePrefix != "/dev/null") {
          std::error_code EC;
          std::string fname =
              AIRToAIEModulePrefix + aie_module.getName()->str() + ".mlir";
          llvm::raw_fd_ostream aie_ostream(fname, EC);
          aie_module.print(aie_ostream);
        }
      } else {
        aie_module.print(llvm::outs());
      }
    }
  }

private:
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToAIEPass() {
  return std::make_unique<AIRToAIEPass>();
}

} // namespace air
} // namespace xilinx
