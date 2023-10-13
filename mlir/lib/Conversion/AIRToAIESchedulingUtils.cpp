//===- AIRToAIESchedulingUtils.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToAIESchedulingUtils.h"
#include "air/Util/Util.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "air-to-aie-scheduling-utils"

using namespace mlir;
using namespace xilinx;

bool air::isTileInbound(air::MemcpyInterface memcpyOp, int tileMemSpaceAsInt) {
  if (memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    int src_memory_space = memcpyOp.getSrcMemref()
                               .getType()
                               .cast<MemRefType>()
                               .getMemorySpaceAsInt();
    int dst_memory_space = memcpyOp.getDstMemref()
                               .getType()
                               .cast<MemRefType>()
                               .getMemorySpaceAsInt();
    assert(src_memory_space !=
           dst_memory_space); // air.dmaMemcpyNdOp isn't meant to represent
                              // core-to-core communication
    if (src_memory_space == tileMemSpaceAsInt)
      return false;
    else if (dst_memory_space == tileMemSpaceAsInt)
      return true;
    else
      assert(false);
    return src_memory_space < dst_memory_space;
  } else if (!memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    return true;
  } else
    return false;
}
bool air::isTileOutbound(air::MemcpyInterface memcpyOp, int tileMemSpaceAsInt) {
  return !isTileInbound(memcpyOp, tileMemSpaceAsInt);
}

AIE::TileOp air::getPhysTileOpOrNull(AIE::DeviceOp aie_device, int col,
                                     int row) {
  for (auto t : aie_device.getOps<AIE::TileOp>()) {
    if (t.colIndex() == col && t.rowIndex() == row)
      return t;
  }
  return nullptr;
}

// get tileop using physical coordinates
AIE::TileOp air::getPhysTileOp(AIE::DeviceOp aie_device, int col, int row) {
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

AIE::LockOp air::allocateLockOp(AIE::DeviceOp aie_device, AIE::TileOp tile,
                                int init, int id) {
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

std::stringstream air::generateBufferNameInStringStream(std::string prefix,
                                                        uint64_t &BufferId,
                                                        mlir::StringAttr attr,
                                                        int x, int y) {

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
    ss << prefix << BufferId++;
  }
  return ss;
}

AIE::ExternalBufferOp air::allocateExternalBufferOp(MemRefType memrefTy,
                                                    AIE::DeviceOp device,
                                                    mlir::StringAttr attr,
                                                    int x, int y) {

  static uint64_t BufferId = 0;

  auto builder = OpBuilder::atBlockBegin(device.getBody());
  AIE::ExternalBufferOp bufferOp =
      builder.create<AIE::ExternalBufferOp>(builder.getUnknownLoc(), memrefTy);

  std::stringstream ss =
      generateBufferNameInStringStream("extBuf", BufferId, attr, x, y);
  bufferOp->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(device->getContext(), ss.str()));

  return bufferOp;
}

std::vector<unsigned> air::convertToStdVec(SmallVector<long int, 4> vec) {
  return {vec.begin(), vec.end()};
}

bool air::areIdenticalVectors(std::vector<unsigned> &a,
                              std::vector<unsigned> &b) {
  return a == b;
}

// allocation_info_t impl.

namespace xilinx::air {

bool allocation_info_t::foundAlloc(uint32_t col, uint32_t row,
                                   air::MemcpyInterface memcpyOp) {
  if (col == dma_tile.getCol() && row == dma_tile.getRow())
    for (auto o : memcpyOps)
      if (memcpyOp.getOperation() == o)
        return true;
  return false;
}
bool allocation_info_t::foundAlloc(uint32_t col, uint32_t row, int chan) {
  if (col == dma_tile.getCol() && row == dma_tile.getRow() &&
      chan == dma_channel.second)
    return true;
  return false;
}
bool allocation_info_t::foundAlloc(uint32_t col, uint32_t row) {
  if (col == dma_tile.getCol() && row == dma_tile.getRow())
    return true;
  return false;
}
bool allocation_info_t::foundAlloc(AIE::TileOp tile, AIE::DMAChannel channel) {
  if (tile == dma_tile && channel.first == dma_channel.first &&
      channel.second == dma_channel.second)
    return true;
  else
    return false;
}

} // namespace xilinx::air

// DMAAllocator impl.

// A simple selection sorting implementation.
static inline void swap(std::vector<Operation *> &a, int i, int j) {
  Operation *t = a[i];
  a[i] = a[j];
  a[j] = t;
}

static void selection(std::vector<Operation *> &a) {
  size_t i, j, min;
  for (i = 0; i < a.size() - 1; i++) {
    min = i;
    for (j = i + 1; j < a.size(); j++) {
      auto a_j = dyn_cast<air::MemcpyInterface>(a[j]);
      auto a_min = dyn_cast<air::MemcpyInterface>(a[min]);
      if (a_j.getId() < a_min.getId())
        min = j;
    }
    swap(a, min, i);
  }
}

namespace xilinx::air {

allocation_info_t
DMAAllocator::lookupDMAAllocation(int64_t col, int64_t row,
                                  air::MemcpyInterface &memcpyOp) {

  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;
  for (auto &t : *allocs) {
    if (t.foundAlloc(col, row, memcpyOp))
      return t;
  }
  assert(false);
  return {nullptr, -1, -1, AIE::DMAChannel(), -1, {}, {}};
}

// Allocate a reader/writer lock pair. These may be the same or different
// locks depending on the target device.
std::pair<AIE::LockOp, AIE::LockOp>
DMAAllocator::getLockForDMA(air::MemcpyInterface &memcpyOp, int col, int row,
                            Operation *bufferOp) {
  allocation_info_t alloc = lookupDMAAllocation(col, row, memcpyOp);
  AIE::DMAChannel channel = alloc.dma_channel;
  AIE::TileOp tile = alloc.dma_tile;

  for (size_t i = 0; i < lock_allocation_list.size(); i++) {
    if ((std::get<0>(lock_allocation_list[i]) == bufferOp) &&
        (std::get<1>(lock_allocation_list[i]) == channel)) {
      return {std::get<2>(lock_allocation_list[i]),
              std::get<3>(lock_allocation_list[i])};
    }
  }
  const auto &target_model = device.getTargetModel();
  bool isAIE2 = (target_model.getTargetArch() == AIE::AIEArch::AIE2);
  auto init = isAIE2 ? 1 : 0;

  OpBuilder builder(bufferOp);
  auto rlock = allocateLockOp(device, tile, 0);
  auto wlock = isAIE2 ? allocateLockOp(device, tile, init) : rlock;
  lock_allocation_list.push_back({bufferOp, channel, rlock, wlock});
  return {rlock, wlock};
}

// Allocate a new DMA channel
allocation_info_t
DMAAllocator::allocNewDmaChannel(air::MemcpyInterface &memcpyOp,
                                 AIE::TileOp tile, int chan, int col = -1,
                                 int row = -1, std::vector<int> dma_id = {}) {
  assert(tile);
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;
  AIE::DMAChannel aie_chan =
      (isMM2S) ? (std::make_pair(AIE::DMAChannelDir::MM2S, chan))
               : (std::make_pair(AIE::DMAChannelDir::S2MM, chan));
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile.getCol(), tile.getRow())) {
      if (t.dma_channel.first == aie_chan.first &&
          t.dma_channel.second == aie_chan.second) {
        t.memcpyOps.push_back(memcpyOp.getOperation());
        return t;
      }
    }
  }
  allocation_info_t output = {
      tile, col, row, aie_chan, chan, dma_id, {memcpyOp.getOperation()}};
  allocs->push_back(output);
  return output;
}

// Sort all ops being allocated to each DMA channel (based on id which indicates
// op sequence), to avoid ping-pong deadlock.
void DMAAllocator::sortMemcpyOps(std::vector<Operation *> dma_memcpy_ops) {
  for (auto &alloc : mm2s_allocs) {
    selection(alloc.memcpyOps);
  }
  for (auto &alloc : s2mm_allocs) {
    selection(alloc.memcpyOps);
  }
}

// TileDMAAllocator impl.

// A very simple scheme to allocate channels for dma operations:
//  <description>
allocation_info_t
TileDMAAllocator::simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp, int col,
                                        int row, int chan = -1) {
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

  // Search for existing dma channel allocation
  unsigned num_allocs = 0;
  for (auto &t : *allocs) {
    if (t.foundAlloc(col, row))
      num_allocs++;
    if (t.foundAlloc(col, row, memcpyOp))
      return t;
    if (t.foundAlloc(col, row, chan)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }
  // Need to allocate a new one
  auto tile = getPhysTileOpOrNull(device, col, row);
  assert(tile);
  int tile_dma_channels =
      isMM2S ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
             : tile.getNumDestConnections(AIE::WireBundle::DMA);
  if (chan == -1)
    chan = num_allocs % tile_dma_channels;
  return DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

AIE::BufferOp TileDMAAllocator::getBuffer(int64_t col, int64_t row,
                                          air::MemcpyInterface &memcpyOp) {
  Value buffer = isTileInbound(memcpyOp, DMAMemorySpaceAsInt)
                     ? (memcpyOp.getDstMemref())
                     : (memcpyOp.getSrcMemref());
  AIE::BufferOp bufferOp = buffer.getDefiningOp<AIE::BufferOp>();
  return bufferOp;
}

// ShimDMAAllocator impl.

ShimDMAAllocator::ShimDMAAllocator(AIE::DeviceOp device)
    : DMAAllocator(device, (int)air::MemorySpace::L3) {
  const auto &aie_target = device.getTargetModel();
  shim_dma_channels = 2;
  for (int i = 0, e = aie_target.columns(); i < e; i++) {
    if (aie_target.isShimNOCTile(i, 0))
      dma_columns.push_back(i);
  }
}

allocation_info_t
ShimDMAAllocator::allocNewDmaChannel(air::MemcpyInterface &memcpyOp, int col,
                                     int row,
                                     std::vector<Operation *> &dma_ops) {
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

  auto dma_col = dma_columns[allocs->size() / shim_dma_channels];
  auto dma_channel = allocs->size() % shim_dma_channels;
  auto tile = getPhysTileOp(device, dma_col, 0);
  assert(tile);
  // For shim dma allocations, the col, row and dma_id fields record the other
  // side of the flows, for airrt metadata
  std::vector<int> dma_ops_get_id;
  for (auto op : dma_ops) {
    if (op->hasAttr("id"))
      dma_ops_get_id.push_back(op->getAttrOfType<IntegerAttr>("id").getInt());
    else
      dma_ops_get_id.push_back(-1);
  }
  return DMAAllocator::allocNewDmaChannel(memcpyOp, tile, dma_channel, col, row,
                                          dma_ops_get_id);
}

allocation_info_t
ShimDMAAllocator::allocNewDmaChannel(air::MemcpyInterface &memcpyOp,
                                     allocation_info_t existing_alloc,
                                     std::vector<Operation *> &dma_ops) {
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

  std::vector<int> dma_ops_get_id;
  for (auto op : dma_ops) {
    if (op->hasAttr("id"))
      dma_ops_get_id.push_back(op->getAttrOfType<IntegerAttr>("id").getInt());
    else
      dma_ops_get_id.push_back(-1);
  }

  for (auto &t : *allocs) {
    if (t.foundAlloc(existing_alloc.dma_tile, existing_alloc.dma_channel)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      for (auto id : dma_ops_get_id)
        t.dma_id.push_back(id);
      return t;
    }
  }
  assert(false);
  return DMAAllocator::allocNewDmaChannel(memcpyOp, existing_alloc.dma_tile,
                                          existing_alloc.dma_channel.second);
}

AIE::ExternalBufferOp
ShimDMAAllocator::getBuffer(int64_t col, int64_t row,
                            air::MemcpyInterface &memcpyOp) {
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  // Allocate external buffers
  auto memref =
      (isMM2S) ? (memcpyOp.getSrcMemref()) : (memcpyOp.getDstMemref());
  assert(memref);
  MemRefType memrefTy = memref.getType().cast<MemRefType>();
  // External buffers have memory space L3
  memrefTy = MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(), {},
                             DMAMemorySpaceAsInt);
  AIE::ExternalBufferOp bufferOp = allocateExternalBufferOp(
      memrefTy, device,
      memcpyOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
      col, row);
  return bufferOp;
}

} // namespace xilinx::air

// MemTileDMAAllocator impl.

namespace xilinx::air {

MemTileDMAAllocator::MemTileDMAAllocator(AIE::DeviceOp device)
    : DMAAllocator(device, (int)air::MemorySpace::L2) {
  const auto &aie_target = device.getTargetModel();
  for (int i = 0, e = aie_target.columns(); i < e; i++) {
    memtile_dma_columns.push_back(i);
  }
}

allocation_info_t
MemTileDMAAllocator::simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp) {
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

  AIE::BufferOp buffer = getBuffer(-1, -1, memcpyOp);
  auto tile = buffer.getTileOp();
  assert(tile);

  // Search for existing dma channel allocation
  unsigned num_allocs = 0;
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile.getCol(), tile.getRow()))
      num_allocs++;
    if (t.foundAlloc(tile.getCol(), tile.getRow(), memcpyOp))
      return t;
  }
  // Need to allocate a new one
  int memtile_dma_channels =
      isMM2S ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
             : tile.getNumDestConnections(AIE::WireBundle::DMA);
  int chan = num_allocs % memtile_dma_channels;
  return DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

AIE::BufferOp MemTileDMAAllocator::getBuffer(int64_t col, int64_t row,
                                             air::MemcpyInterface &memcpyOp) {
  Value buffer = isTileInbound(memcpyOp, DMAMemorySpaceAsInt)
                     ? (memcpyOp.getDstMemref())
                     : (memcpyOp.getSrcMemref());
  AIE::BufferOp bufferOp = buffer.getDefiningOp<AIE::BufferOp>();
  assert(bufferOp);
  return bufferOp;
}

// MemcpyBundleAsFlow impl.

void MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::DmaMemcpyNdOp memcpyOp) {
  // air::DmaMemcpyNdOp is a complete memcpy with both src and dst
  S2MM[0].push_back(memcpyOp.getOperation());
  S2MM_memspace_as_int = memcpyOp.getDstMemref()
                             .getType()
                             .cast<MemRefType>()
                             .getMemorySpaceAsInt();
  MM2S.push_back(memcpyOp.getOperation());
  MM2S_memspace_as_int = memcpyOp.getSrcMemref()
                             .getType()
                             .cast<MemRefType>()
                             .getMemorySpaceAsInt();
}

void MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::ChannelGetOp memcpyOp) {
  auto chan = air::getChannelDeclarationThroughSymbol(memcpyOp);
  int alloc_id = 0;
  if (chan->hasAttr("broadcast_shape")) {
    // Walk through each element in broadcast_shape
    auto bcast_sizes = extractFromI64ArrayAttr(
        chan->getAttrOfType<mlir::ArrayAttr>("broadcast_shape"));
    auto bcast_sizes_stdvec = convertToStdVec(bcast_sizes);
    for (int iter = 0; iter < numS2MMAllocs; iter++) {
      std::vector<unsigned> position =
          getMDVectorFromIterator(bcast_sizes_stdvec, iter);
      auto indices_uint =
          convertVecOfConstIndexToVecOfUInt(memcpyOp.getIndices());
      // Remove position coord offset
      for (unsigned dim = 0; dim < indices_uint.size(); dim++) {
        if (bcast_sizes_stdvec[dim] == 1) {
          // Offset dimension
          indices_uint[dim] = 0;
        }
      }
      assert(indices_uint[0] != 1 || indices_uint[1] != 1);
      if (areIdenticalVectors(indices_uint, position)) {
        alloc_id = iter;
      }
    }
  }
  air_flow_op = chan.getOperation();
  S2MM[alloc_id].push_back(memcpyOp.getOperation());
  S2MM_memspace_as_int =
      memcpyOp.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();
}

void MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::ChannelPutOp memcpyOp) {
  auto chan = air::getChannelDeclarationThroughSymbol(memcpyOp);
  air_flow_op = chan.getOperation();
  MM2S.push_back(memcpyOp.getOperation());
  MM2S_memspace_as_int =
      memcpyOp.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt();
}

void MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(
    air::ChannelInterface memcpyOp) {
  if (auto get = dyn_cast<air::ChannelGetOp>(memcpyOp.getOperation()))
    pushBackMemcpyOpToBundle(get);
  else if (auto put = dyn_cast<air::ChannelPutOp>(memcpyOp.getOperation()))
    pushBackMemcpyOpToBundle(put);
  else
    memcpyOp->emitOpError("unknown op type in air::ChannelInterface");
}

MemcpyBundleAsFlow::MemcpyBundleAsFlow(air::DmaMemcpyNdOp dmaMemcpyOp) {
  air_flow_op = dmaMemcpyOp.getOperation();
  numS2MMAllocs = 1;
  numMM2SAllocs = 1;
  std::vector<std::vector<Operation *>> v1(numS2MMAllocs,
                                           std::vector<Operation *>());
  S2MM = v1;
  S2MM_alloc = std::vector<allocation_info_t>(numS2MMAllocs);
}

MemcpyBundleAsFlow::MemcpyBundleAsFlow(air::ChannelOp chan) {
  air_flow_op = chan.getOperation();
  int num_bcast_dests = 1;
  if (chan->hasAttr("broadcast_shape")) {
    auto bsize = extractFromI64ArrayAttr(
        chan->getAttrOfType<mlir::ArrayAttr>("broadcast_shape"));
    for (auto &s : bsize) {
      num_bcast_dests *= s;
    }
  }
  numS2MMAllocs = num_bcast_dests;
  numMM2SAllocs = 1;
  std::vector<std::vector<Operation *>> v1(numS2MMAllocs,
                                           std::vector<Operation *>());
  S2MM = v1;
  S2MM_alloc = std::vector<allocation_info_t>(numS2MMAllocs);
}

} // namespace xilinx::air

// Search for opportunities where air channels can reuse flow op via time
// multiplexing
std::optional<air::allocation_info_t>
air::foundFlowReuseOpportunity(std::vector<MemcpyBundleAsFlow> memcpy_flows,
                               air::allocation_info_t alloc, bool isMM2S) {
  std::optional<allocation_info_t> output = std::nullopt;
  for (auto &f : memcpy_flows) {
    if (isMM2S) {
      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].dma_tile == alloc.dma_tile &&
            f.S2MM_alloc[i].dma_channel.first == alloc.dma_channel.first &&
            f.S2MM_alloc[i].dma_channel.second == alloc.dma_channel.second) {
          if (f.MM2S_alloc.dma_tile && f.MM2S_alloc.dma_tile.isShimTile()) {
            output = f.MM2S_alloc;
            return output;
          }
        }
      }
    } else if (!isMM2S && f.MM2S_alloc.dma_tile == alloc.dma_tile &&
               f.MM2S_alloc.dma_channel.first == alloc.dma_channel.first &&
               f.MM2S_alloc.dma_channel.second == alloc.dma_channel.second) {

      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].dma_tile && f.S2MM_alloc[i].dma_tile.isShimTile()) {
          output = f.S2MM_alloc[i];
          return output;
        }
      }
    }
  }
  return output;
}

std::optional<air::allocation_info_t>
air::foundFlowReuseOpportunity(std::vector<MemcpyBundleAsFlow> memcpy_flows,
                               std::vector<air::allocation_info_t> allocs,
                               bool isMM2S) {
  std::optional<allocation_info_t> output = std::nullopt;
  for (auto alloc : allocs) {
    output = foundFlowReuseOpportunity(memcpy_flows, alloc, isMM2S);
    if (output.has_value())
      return output;
  }
  return output;
}

// AIR channel to AIE flow scheduling strategy 1: round robin
// Problem: no awareness wrt channel put and get pattern, leading to deadlocks
void air::simpleDMAChannelAllocation(
    std::vector<MemcpyBundleAsFlow> &memcpy_flows,
    ShimDMAAllocator &shim_dma_alloc, MemTileDMAAllocator &memtile_dma_alloc,
    TileDMAAllocator &tile_dma_alloc) {
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L1) {
      for (auto o : f.MM2S) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        auto core = o->getParentOfType<AIE::CoreOp>();
        assert(core);
        auto tile = core.getTileOp();
        int x = tile.getCol();
        int y = tile.getRow();

        f.MM2S_alloc = tile_dma_alloc.simpleDmaChannelAlloc(
            memcpyOpIf, x, y, f.MM2S_alloc.dma_channel.second);
        assert(f.MM2S_alloc.dma_tile);
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L1) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          auto core = o->getParentOfType<AIE::CoreOp>();
          assert(core);
          auto tile = core.getTileOp();
          int x = tile.getCol();
          int y = tile.getRow();

          f.S2MM_alloc[i] = tile_dma_alloc.simpleDmaChannelAlloc(
              memcpyOpIf, x, y, f.S2MM_alloc[i].dma_channel.second);
          assert(f.S2MM_alloc[i].dma_tile);
        }
      }
    }
  }
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L2) {
      for (auto o : f.MM2S) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        f.MM2S_alloc = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L2) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (unsigned i = 0; i < f.S2MM.size(); i++) {
          for (auto o : f.S2MM[i]) {
            auto memcpyOpIf = cast<air::MemcpyInterface>(o);
            f.S2MM_alloc[i] =
                memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
          }
        }
      }
    }
  }
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L3) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        auto alloc =
            foundFlowReuseOpportunity(memcpy_flows, f.S2MM_alloc[i], true);
        if (alloc.has_value()) {
          for (auto o : f.MM2S) {
            auto memcpyOpIf = cast<air::MemcpyInterface>(o);
            f.MM2S_alloc = shim_dma_alloc.allocNewDmaChannel(memcpyOpIf, *alloc,
                                                             f.S2MM[i]);
          }
        } else {
          for (auto o : f.MM2S) {
            auto memcpyOpIf = cast<air::MemcpyInterface>(o);
            f.MM2S_alloc = shim_dma_alloc.allocNewDmaChannel(
                memcpyOpIf, f.S2MM_alloc[i].dma_tile.getCol(),
                f.S2MM_alloc[i].dma_tile.getRow(), f.S2MM[i]);
          }
        }
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L3) {
      // L3 shim tiles assumed to not be target for broadcast
      auto alloc = foundFlowReuseOpportunity(memcpy_flows, f.MM2S_alloc, false);
      if (alloc.has_value()) {
        for (auto o : f.S2MM[0]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.S2MM_alloc[0] =
              shim_dma_alloc.allocNewDmaChannel(memcpyOpIf, *alloc, f.MM2S);
        }
      } else {
        for (auto o : f.S2MM[0]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.S2MM_alloc[0] = shim_dma_alloc.allocNewDmaChannel(
              memcpyOpIf, f.MM2S_alloc.dma_tile.getCol(),
              f.MM2S_alloc.dma_tile.getRow(), f.MM2S);
        }
      }
    }
  }
}

// If found item in vector, return index; else return -1.
template <typename T> int air::foundInVector(T item, std::vector<T> vec) {
  auto it = std::find(vec.begin(), vec.end(), item);
  int index = it - vec.begin();
  return index;
}

// AIR channel to AIE flow scheduling strategy 2: grouped by for loop
// Only those air channel puts and gets which share the same for loop region can
// share the same AIE DMA channel
bool air::groupingMemcpysByLoop(std::vector<MemcpyBundleAsFlow> &memcpy_flows) {
  // Group memcpy_flows based on L1-side puts/gets' loop structure
  std::map<AIE::CoreOp, std::vector<scf::ForOp>> for_loops_log_mm2s,
      for_loops_log_s2mm;
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L1) {
      for (auto o : f.MM2S) {
        auto core = o->getParentOfType<AIE::CoreOp>();
        f.flow_op_group = foundInVector<scf::ForOp>(
            o->getParentOfType<scf::ForOp>(), for_loops_log_mm2s[core]);
        if ((size_t)f.flow_op_group == for_loops_log_mm2s[core].size()) {
          for_loops_log_mm2s[core].push_back(o->getParentOfType<scf::ForOp>());
        }
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L1) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
          auto core = o->getParentOfType<AIE::CoreOp>();
          f.flow_op_group = foundInVector<scf::ForOp>(
              o->getParentOfType<scf::ForOp>(), for_loops_log_s2mm[core]);
          if ((size_t)f.flow_op_group == for_loops_log_s2mm[core].size()) {
            for_loops_log_s2mm[core].push_back(
                o->getParentOfType<scf::ForOp>());
          }
        }
      }
    }
  }
  // Toggle scheduling strategy
  int flow_op_group_max = 0;
  for (auto &f : memcpy_flows) {
    flow_op_group_max = std::max(flow_op_group_max, f.flow_op_group);
  }
  return flow_op_group_max;
}

void air::groupedByLoopDMAChannelAllocation(
    std::vector<MemcpyBundleAsFlow> &memcpy_flows,
    ShimDMAAllocator &shim_dma_alloc, MemTileDMAAllocator &memtile_dma_alloc,
    TileDMAAllocator &tile_dma_alloc) {

  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L1) {
      for (auto o : f.MM2S) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        auto core = o->getParentOfType<AIE::CoreOp>();
        assert(core);
        auto tile = core.getTileOp();
        int x = tile.getCol();
        int y = tile.getRow();

        f.MM2S_alloc = tile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf, x, y,
                                                            f.flow_op_group);
        assert(f.MM2S_alloc.dma_tile);
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L1) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          auto core = o->getParentOfType<AIE::CoreOp>();
          assert(core);
          auto tile = core.getTileOp();
          int x = tile.getCol();
          int y = tile.getRow();

          f.S2MM_alloc[i] = tile_dma_alloc.simpleDmaChannelAlloc(
              memcpyOpIf, x, y, f.flow_op_group);
          assert(f.S2MM_alloc[i].dma_tile);
        }
      }
    }
  }
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L2) {
      for (auto o : f.MM2S) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        f.MM2S_alloc = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L2) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (unsigned i = 0; i < f.S2MM.size(); i++) {
          for (auto o : f.S2MM[i]) {
            auto memcpyOpIf = cast<air::MemcpyInterface>(o);
            f.S2MM_alloc[i] =
                memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
          }
        }
      }
    }
  }
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L3) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        auto alloc =
            foundFlowReuseOpportunity(memcpy_flows, f.S2MM_alloc[i], true);
        if (alloc.has_value()) {
          for (auto o : f.MM2S) {
            auto memcpyOpIf = cast<air::MemcpyInterface>(o);
            f.MM2S_alloc = shim_dma_alloc.allocNewDmaChannel(memcpyOpIf, *alloc,
                                                             f.S2MM[i]);
          }
        } else {
          for (auto o : f.MM2S) {
            auto memcpyOpIf = cast<air::MemcpyInterface>(o);
            f.MM2S_alloc = shim_dma_alloc.allocNewDmaChannel(
                memcpyOpIf, f.S2MM_alloc[i].dma_tile.getCol(),
                f.S2MM_alloc[i].dma_tile.getRow(), f.S2MM[i]);
          }
        }
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L3) {
      // L3 shim tiles assumed to not be target for broadcast
      auto alloc = foundFlowReuseOpportunity(memcpy_flows, f.MM2S_alloc, false);
      if (alloc.has_value()) {
        for (auto o : f.S2MM[0]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.S2MM_alloc[0] =
              shim_dma_alloc.allocNewDmaChannel(memcpyOpIf, *alloc, f.MM2S);
        }
      } else {
        for (auto o : f.S2MM[0]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.S2MM_alloc[0] = shim_dma_alloc.allocNewDmaChannel(
              memcpyOpIf, f.MM2S_alloc.dma_tile.getCol(),
              f.MM2S_alloc.dma_tile.getRow(), f.MM2S);
        }
      }
    }
  }
}
