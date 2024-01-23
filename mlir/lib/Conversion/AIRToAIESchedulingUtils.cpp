//===- AIRToAIESchedulingUtils.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToAIESchedulingUtils.h"
#include "air/Util/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/MathExtras.h"
#include <set>

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
  AIE::ExternalBufferOp bufferOp = builder.create<AIE::ExternalBufferOp>(
      builder.getUnknownLoc(), memrefTy, nullptr);

  std::stringstream ss =
      generateBufferNameInStringStream("extBuf", BufferId, attr, x, y);
  bufferOp->setAttr(SymbolTable::getSymbolAttrName(),
                    StringAttr::get(device->getContext(), ss.str()));

  return bufferOp;
}

std::vector<unsigned> air::convertToStdVec(SmallVector<int64_t, 6> vec) {
  return {vec.begin(), vec.end()};
}

bool air::areIdenticalVectors(std::vector<unsigned> &a,
                              std::vector<unsigned> &b) {
  return a == b;
}

int64_t air::get1DOffset(SmallVector<Value> memcpy_offsets,
                         SmallVector<Value> memcpy_strides,
                         int byte_count_per_elem) {
  if (memcpy_offsets.empty())
    return 0;

  int64_t one_d_offset = 0;
  for (int i = memcpy_offsets.size() - 1; i >= 0; i--) {
    auto offset = mlir::getConstantIntValue(memcpy_offsets[i]);
    if (!offset)
      assert(false && "non-static offset in memcpy op");
    if (i == memcpy_offsets.size() - 1)
      one_d_offset += *offset;
    else {
      if (auto stride_i = mlir::getConstantIntValue(memcpy_strides[i])) {
        one_d_offset += (*offset) * (*stride_i);
      } else
        assert(false && "non-static size in memcpy op");
    }
  }
  return one_d_offset * byte_count_per_elem;
}

std::vector<AIE::BDDimLayoutAttr>
air::getWrapsAndStrides(SmallVector<Value> memcpy_sizes,
                        SmallVector<Value> memcpy_strides, MLIRContext *ctx) {
  if (memcpy_sizes.empty() || memcpy_strides.empty())
    return std::vector<AIE::BDDimLayoutAttr>{};
  assert(memcpy_sizes.size() == memcpy_strides.size() &&
         "unequal sizes between wrap list and stride list");
  std::vector<AIE::BDDimLayoutAttr> output = {};
  for (unsigned i = 0; i < memcpy_sizes.size(); i++) {
    auto stepsize = mlir::getConstantIntValue(memcpy_strides[i]);
    assert(stepsize && "non-static stride");
    auto wrap = mlir::getConstantIntValue(memcpy_sizes[i]);
    assert(wrap && "non-static wrap");
    auto tuple = AIE::BDDimLayoutAttr::get(ctx, *wrap, *stepsize);
    output.push_back(tuple);
  }
  return output;
}

bool air::isDefaultDataAccessPattern(SmallVector<Value> memcpy_sizes,
                                     SmallVector<Value> memcpy_strides,
                                     Value memref) {
  if (memcpy_sizes.empty() || memcpy_strides.empty())
    return false;
  // If the sizes and strides were already accessing the memref in default
  // order, then wraps and strides are not needed
  SmallVector<int> memref_shape = getTensorShape(memref.getType());
  if (memcpy_sizes.size() != memref_shape.size())
    return false;
  unsigned stride_factor = 1;
  for (int i = memcpy_sizes.size() - 1; i >= 0; i--) {
    auto stepsize = mlir::getConstantIntValue(memcpy_strides[i]);
    assert(stepsize && "non-static stride");
    auto wrap = mlir::getConstantIntValue(memcpy_sizes[i]);
    assert(wrap && "non-static wrap");
    if (*stepsize != stride_factor)
      return false;
    if (*wrap != memref_shape[i])
      return false;
    stride_factor *= *wrap;
  }
  return true;
}

std::pair<int64_t, int64_t> air::getLockValuePair(AIE::AIEArch arch,
                                                  Value buffer_memref) {
  bool isAIE2 = (arch == AIE::AIEArch::AIE2);
  if (!isAIE2)
    return std::make_pair(0, 0);

  // Infer semaphore lock values using buffer op
  // TODO: What if a buffer memref is read or written by multiple channels?
  if (!buffer_memref.getType().isa<MemRefType>())
    return std::make_pair(-1, -1);
  int read_counter = 0;
  int write_counter = 0;
  for (auto user : buffer_memref.getUsers()) {
    if (auto memcpyOp = dyn_cast<air::MemcpyInterface>(user)) {
      if (buffer_memref == memcpyOp.getSrcMemref())
        read_counter++;
      else if (buffer_memref == memcpyOp.getDstMemref())
        write_counter++;
    } else if (isa<affine::AffineLoadOp>(user))
      read_counter++;
    else if (isa<affine::AffineStoreOp>(user))
      write_counter++;
    else if (auto linalgop = dyn_cast<linalg::LinalgOp>(user)) {
      for (auto opoperand : linalgop.getDpsInputOperands())
        if (opoperand->is(buffer_memref))
          read_counter++;
      for (auto &opoperand : linalgop.getDpsInitsMutable())
        if (opoperand.is(buffer_memref)) {
          read_counter++;
          write_counter++;
        }
    }
  }
  if (!read_counter || !write_counter)
    return std::make_pair(1, 1);
  if (read_counter >= write_counter)
    return std::make_pair(mlir::ceilDiv(read_counter, write_counter), 1);
  else
    return std::make_pair(1, mlir::ceilDiv(write_counter, read_counter));
}

std::pair<int64_t, int64_t> air::getLockValuePair(AIE::AIEArch arch,
                                                  Value buffer_memref,
                                                  air::ChannelOp air_chan) {
  bool isAIE2 = (arch == AIE::AIEArch::AIE2);
  if (!isAIE2)
    return std::make_pair(0, 0);
  if (!buffer_memref.getType().isa<MemRefType>())
    return std::make_pair(-1, -1);

  if (!air_chan)
    return getLockValuePair(arch, buffer_memref);

  // Infer semaphore lock values using air.channel
  int read_counter = 0;
  int write_counter = 0;
  for (auto get : getChannelGetOpThroughSymbol(air_chan)) {
    if (isa<AIE::ExternalBufferOp>(buffer_memref.getDefiningOp())) {
      // Shim DMA locks
      write_counter = 1;
    } else if (auto core_op = get->getParentOfType<AIE::CoreOp>()) {
      if (core_op.getTileOp().getResult() ==
          buffer_memref.getDefiningOp()->getOperand(0)) {
        write_counter++;
      }
    }
  }
  for (auto put : getChannelPutOpThroughSymbol(air_chan)) {
    if (isa<AIE::ExternalBufferOp>(buffer_memref.getDefiningOp())) {
      // Shim DMA locks
      read_counter = 1;
    } else if (auto core_op = put->getParentOfType<AIE::CoreOp>()) {
      if (core_op.getTileOp().getResult() ==
          buffer_memref.getDefiningOp()->getOperand(0)) {
        read_counter++;
      }
    }
  }
  return std::make_pair(read_counter, write_counter);
}

// allocation_info_t impl.

namespace xilinx::air {

bool allocation_info_t::foundAlloc(air::ChannelOp channel_op) {
  if (channel_op) {
    for (auto o : memcpyOps) {
      if (auto chan_op = dyn_cast<air::ChannelInterface>(o)) {
        auto chan_declr = getChannelDeclarationThroughSymbol(chan_op);
        if (chan_declr == channel_op)
          return true;
      }
    }
  }
  return false;
}
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
      chan == dma_channel.channel)
    return true;
  return false;
}
bool allocation_info_t::foundAlloc(uint32_t col, uint32_t row) {
  if (col == dma_tile.getCol() && row == dma_tile.getRow())
    return true;
  return false;
}
bool allocation_info_t::foundAlloc(AIE::TileOp tile, AIE::DMAChannel channel) {
  if (tile == dma_tile && channel.direction == dma_channel.direction &&
      channel.channel == dma_channel.channel)
    return true;
  else
    return false;
}
bool allocation_info_t::foundAlloc(uint32_t col, uint32_t row,
                                   air::ChannelOp channel_op) {
  if (col == dma_tile.getCol() && row == dma_tile.getRow() && channel_op) {
    for (auto o : memcpyOps) {
      if (auto chan_op = dyn_cast<air::ChannelInterface>(o)) {
        auto chan_declr = getChannelDeclarationThroughSymbol(chan_op);
        if (chan_declr == channel_op)
          return true;
      }
    }
  }
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
  air::ChannelOp air_chan = nullptr;
  if (auto air_chan_op =
          dyn_cast<air::ChannelInterface>(memcpyOp.getOperation())) {
    air_chan = getChannelDeclarationThroughSymbol(air_chan_op);
  }
  const auto &target_model = device.getTargetModel();
  bool isAIE2 = (target_model.getTargetArch() == AIE::AIEArch::AIE2);
  bool isAIE1 = (target_model.getTargetArch() == AIE::AIEArch::AIE1);

  if (isAIE1) {
    for (size_t i = 0; i < lock_allocation_list.size(); i++) {
      // If multiple bds reference the same buffer and DMA channel
      if ((std::get<0>(lock_allocation_list[i]) == bufferOp) &&
          (std::get<2>(lock_allocation_list[i]) == channel)) {
        return {std::get<3>(lock_allocation_list[i]),
                std::get<4>(lock_allocation_list[i])};
      }
    }
  }

  else if (isAIE2) {
    if (air_chan) {
      // AIE2's semaphore locks may share by air.channels
      for (size_t i = 0; i < lock_allocation_list.size(); i++) {
        if (target_model.isMemTile(col, row)) {
          // If memtile, and multiple bds reference the same buffer op, but
          // different DMA channels, then we assume the scenario of having two
          // bds, one S2MM and the other MM2S. This scenario is almost always
          // true due to memtile having no core to communicate data with.
          if (std::get<0>(lock_allocation_list[i]) == bufferOp) {
            return {std::get<3>(lock_allocation_list[i]),
                    std::get<4>(lock_allocation_list[i])};
          }
        } else if ((std::get<1>(lock_allocation_list[i]) == air_chan) &&
                   (std::get<0>(lock_allocation_list[i])->getOperand(0) ==
                    bufferOp->getOperand(0)) &&
                   (std::get<2>(lock_allocation_list[i]) == channel)) {
          return {std::get<3>(lock_allocation_list[i]),
                  std::get<4>(lock_allocation_list[i])};
        }
      }
    } else {
      for (size_t i = 0; i < lock_allocation_list.size(); i++) {
        if ((std::get<0>(lock_allocation_list[i]) == bufferOp) &&
            (std::get<2>(lock_allocation_list[i]) == channel)) {
          return {std::get<3>(lock_allocation_list[i]),
                  std::get<4>(lock_allocation_list[i])};
        }
        // Else if memtile, and multiple bds reference the same buffer, but
        // different DMA channels, then we assume the scenario of having two
        // bds, one S2MM and the other MM2S. This scenario is almost always true
        // due to memtile having no core to communicate data with.
        else if (target_model.isMemTile(col, row) &&
                 std::get<0>(lock_allocation_list[i]) == bufferOp) {
          return {std::get<3>(lock_allocation_list[i]),
                  std::get<4>(lock_allocation_list[i])};
        }
      }
    }
  }
  std::pair<int64_t, int64_t> init_pair;
  if (target_model.isMemTile(col, row))
    init_pair =
        getLockValuePair(target_model.getTargetArch(), bufferOp->getResult(0));
  else
    init_pair = getLockValuePair(target_model.getTargetArch(),
                                 bufferOp->getResult(0), air_chan);
  auto init = std::max(init_pair.first, init_pair.second);

  OpBuilder builder(bufferOp);
  auto rlock = allocateLockOp(device, tile, 0);
  auto wlock = isAIE2 ? allocateLockOp(device, tile, init) : rlock;
  lock_allocation_list.push_back({bufferOp, air_chan, channel, rlock, wlock});
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
  AIE::DMAChannel aie_chan;
  aie_chan.direction =
      isMM2S ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;
  aie_chan.channel = chan;
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile.getCol(), tile.getRow())) {
      if (t.dma_channel.direction == aie_chan.direction &&
          t.dma_channel.channel == aie_chan.channel) {
        t.memcpyOps.push_back(memcpyOp.getOperation());
        return t;
      }
    }
    if (t.foundAlloc(
            tile.getCol(), tile.getRow(),
            getChannelDeclarationThroughSymbol(
                dyn_cast<air::ChannelInterface>(memcpyOp.getOperation())))) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
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

  // Search for existing dma channel allocation
  for (auto &t : *allocs) {
    if (t.foundAlloc(getChannelDeclarationThroughSymbol(
            dyn_cast<air::ChannelInterface>(memcpyOp.getOperation())))) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }

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
                                          existing_alloc.dma_channel.channel);
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

// Search for opportunities where air channels can reuse flow op via time
// multiplexing
std::optional<air::allocation_info_t>
ShimDMAAllocator::foundFlowReuseOpportunity(
    std::vector<MemcpyBundleAsFlow> memcpy_flows, air::allocation_info_t alloc,
    bool isMM2S) {
  std::optional<allocation_info_t> output = std::nullopt;
  for (auto &f : memcpy_flows) {
    if (isMM2S) {
      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].dma_tile == alloc.dma_tile &&
            f.S2MM_alloc[i].dma_channel.direction ==
                alloc.dma_channel.direction &&
            f.S2MM_alloc[i].dma_channel.channel == alloc.dma_channel.channel) {
          if (f.MM2S_alloc.dma_tile && f.MM2S_alloc.dma_tile.isShimTile()) {
            output = f.MM2S_alloc;
            return output;
          }
        }
      }
    } else if (!isMM2S && f.MM2S_alloc.dma_tile == alloc.dma_tile &&
               f.MM2S_alloc.dma_channel.direction ==
                   alloc.dma_channel.direction &&
               f.MM2S_alloc.dma_channel.channel == alloc.dma_channel.channel) {

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
MemTileDMAAllocator::simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp,
                                           int chan = -1) {
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
  if (chan == -1)
    chan = num_allocs % memtile_dma_channels;
  return DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

allocation_info_t
MemTileDMAAllocator::simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp,
                                           allocation_info_t &existing_alloc) {
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

  AIE::BufferOp buffer = getBuffer(-1, -1, memcpyOp);
  auto tile = buffer.getTileOp();
  assert(tile);

  for (auto &t : *allocs) {
    if (t.foundAlloc(existing_alloc.dma_tile, existing_alloc.dma_channel)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }
  assert(false);
  int chan = -1;
  return DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

// Search for opportunities where air channels can reuse flow op via time
// multiplexing
std::optional<air::allocation_info_t>
MemTileDMAAllocator::foundFlowReuseOpportunity(
    std::vector<MemcpyBundleAsFlow> memcpy_flows, air::allocation_info_t alloc,
    bool isMM2S) {
  std::optional<allocation_info_t> output = std::nullopt;
  for (auto &f : memcpy_flows) {
    if (!isMM2S) {
      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].dma_tile == alloc.dma_tile &&
            f.S2MM_alloc[i].dma_channel.direction ==
                alloc.dma_channel.direction &&
            f.S2MM_alloc[i].dma_channel.channel == alloc.dma_channel.channel) {
          if (f.MM2S_alloc.dma_tile && f.MM2S_alloc.dma_tile.isMemTile()) {
            output = f.MM2S_alloc;
            return output;
          }
        }
      }
    } else if (isMM2S && f.MM2S_alloc.dma_tile == alloc.dma_tile &&
               f.MM2S_alloc.dma_channel.direction ==
                   alloc.dma_channel.direction &&
               f.MM2S_alloc.dma_channel.channel == alloc.dma_channel.channel) {

      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].dma_tile && f.S2MM_alloc[i].dma_tile.isMemTile()) {
          output = f.S2MM_alloc[i];
          return output;
        }
      }
    }
  }
  return output;
}

int MemTileDMAAllocator::forecastChannelAlloc(air::MemcpyInterface &memcpyOp) {
  bool isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  auto allocs = isMM2S ? &mm2s_allocs : &s2mm_allocs;

  AIE::BufferOp buffer = getBuffer(-1, -1, memcpyOp);
  auto tile = buffer.getTileOp();

  // Search for existing dma channel allocation
  unsigned num_allocs = 0;
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile.getCol(), tile.getRow()))
      num_allocs++;
    if (t.foundAlloc(tile.getCol(), tile.getRow(), memcpyOp))
      return t.tile_channel;
  }
  int memtile_dma_channels =
      isMM2S ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
             : tile.getNumDestConnections(AIE::WireBundle::DMA);
  return num_allocs % memtile_dma_channels;
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
    auto bcast_sizes = extractFromIntegerArrayAttr<int64_t>(
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
    auto bsize = extractFromIntegerArrayAttr<int64_t>(
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
            memcpyOpIf, x, y, f.MM2S_alloc.dma_channel.channel);
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
              memcpyOpIf, x, y, f.S2MM_alloc[i].dma_channel.channel);
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
        assert(f.MM2S_alloc.dma_tile);
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L2) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.S2MM_alloc[i] = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
          assert(f.S2MM_alloc[i].dma_tile);
        }
      }
    }
  }
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L3) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.MM2S) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.MM2S_alloc = shim_dma_alloc.allocNewDmaChannel(
              memcpyOpIf, f.S2MM_alloc[i].dma_tile.getCol(),
              f.S2MM_alloc[i].dma_tile.getRow(), f.S2MM[i]);
          assert(f.MM2S_alloc.dma_tile);
        }
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L3) {
      // L3 shim tiles assumed to not be target for broadcast
      assert(f.S2MM.size() <= 1);
      for (auto o : f.S2MM[0]) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        f.S2MM_alloc[0] = shim_dma_alloc.allocNewDmaChannel(
            memcpyOpIf, f.MM2S_alloc.dma_tile.getCol(),
            f.MM2S_alloc.dma_tile.getRow(), f.MM2S);
        assert(f.S2MM_alloc[0].dma_tile);
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

int air::getSCFForLoopDepth(Operation *o) {
  int for_loop_depth = 0;
  Operation *parentFor = o->getParentOfType<scf::ForOp>();
  while (parentFor) {
    for_loop_depth++;
    parentFor = parentFor->getParentOfType<scf::ForOp>();
  }
  return for_loop_depth;
}

// AIR channel to AIE flow scheduling strategy 2: grouped by for loop
// Only those air channel puts and gets which share the same for loop level can
// share the same AIE DMA channel. TODO: what if same level but different parent
// loops?
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
      std::optional<allocation_info_t> reusable_alloc = std::nullopt;
      for (auto &the_other_alloc : f.S2MM_alloc) {
        if (auto alloc = memtile_dma_alloc.foundFlowReuseOpportunity(
                memcpy_flows, the_other_alloc, false)) {
          reusable_alloc = alloc;
        }
      }

      if (reusable_alloc.has_value()) {
        // If found channel reuse opportunity on the opposite of flow
        for (auto o : f.MM2S) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.MM2S_alloc = memtile_dma_alloc.simpleDmaChannelAlloc(
              memcpyOpIf, *reusable_alloc);
        }
      } else {
        for (auto o : f.MM2S) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          f.MM2S_alloc = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
        }
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L2) {
      std::optional<allocation_info_t> reusable_alloc = std::nullopt;
      if (auto alloc = memtile_dma_alloc.foundFlowReuseOpportunity(
              memcpy_flows, f.MM2S_alloc, true)) {
        reusable_alloc = alloc;
      }
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        if (reusable_alloc.has_value()) {
          // If found channel reuse opportunity on the opposite of flow
          for (auto o : f.S2MM[i]) {
            auto memcpyOpIf = cast<air::MemcpyInterface>(o);
            f.S2MM_alloc[i] = memtile_dma_alloc.simpleDmaChannelAlloc(
                memcpyOpIf, *reusable_alloc);
          }
        } else {
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
        auto alloc = shim_dma_alloc.foundFlowReuseOpportunity(
            memcpy_flows, f.S2MM_alloc[i], true);
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
      auto alloc = shim_dma_alloc.foundFlowReuseOpportunity(
          memcpy_flows, f.MM2S_alloc, false);
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
