//===- AIRToAIESchedulingUtils.h --------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
#ifndef AIR_TO_AIE_SCHEDULING_UTILS_H
#define AIR_TO_AIE_SCHEDULING_UTILS_H

#include "air/Conversion/PassDetail.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace xilinx {
namespace air {

FailureOr<bool> isTileInbound(air::MemcpyInterface memcpyOp,
                              air::MemorySpace tileMemSpace);
FailureOr<bool> isTileOutbound(air::MemcpyInterface memcpyOp,
                               air::MemorySpace tileMemSpace);

AIE::TileOp getPhysTileOpOrNull(AIE::DeviceOp aie_device, int col, int row);

// get tileop using physical coordinates
AIE::TileOp getPhysTileOp(AIE::DeviceOp aie_device, int col, int row);

AIE::LockOp allocateLockOp(AIE::DeviceOp aie_device, AIE::TileLike tile,
                           int init = 0, int id = -1,
                           StringAttr name = nullptr);

std::stringstream
generateBufferNameInStringStream(StringRef prefix, uint64_t &BufferId,
                                 mlir::StringAttr attr = nullptr, int x = -1,
                                 int y = -1);

AIE::ExternalBufferOp allocateExternalBufferOp(uint64_t &BufferId,
                                               MemRefType memrefTy,
                                               AIE::DeviceOp device,
                                               mlir::StringAttr attr = nullptr,
                                               int x = -1, int y = -1);

std::vector<unsigned> convertToStdVec(SmallVector<int64_t, 6> vec);

bool areIdenticalVectors(std::vector<unsigned> &a, std::vector<unsigned> &b);

int64_t get1DOffset(SmallVector<Value> memcpy_offsets,
                    SmallVector<Value> memcpy_strides);

// Given a vector of memcpy operations, return a map of their repeat counts,
// relative to a common ancestor region.
llvm::MapVector<int, llvm::SetVector<Operation *>>
getRepeatCounts(std::vector<Operation *> memcpy_ops);

std::vector<AIE::BDDimLayoutAttr>
getWrapsAndStrides(SmallVector<Value> memcpy_sizes,
                   SmallVector<Value> memcpy_strides, MLIRContext *ctx);

std::pair<int64_t, int64_t>
getLockValuePair(const AIE::AIETargetModel &targetModel, Value buffer_memref);

std::pair<int64_t, int64_t>
getLockValuePair(const AIE::AIETargetModel &targetModel, Value buffer_memref,
                 air::ChannelOp air_chan);

struct allocation_info_t {
  // dma_tile is the SSA value of the (logical or physical) AIE tile that owns
  // this DMA allocation. Stored as TileLike (op interface) so it works for
  // both AIE::TileOp (post-placement) and AIE::LogicalTileOp (pre-placement).
  // Pointer-equality on the underlying Operation* gives the same answer as
  // (col, row) integer comparison without dependence on physical placement.
  AIE::TileLike dma_tile = nullptr;
  int64_t col = -1;
  int64_t row = -1;
  AIE::DMAChannel dma_channel = {AIE::DMAChannelDir::MM2S, -1};
  int64_t tile_channel = -1;
  int packet_flow_id = -1; // Packet flow ID assigned during flow creation
  std::vector<int32_t> dma_id;
  std::vector<Operation *> memcpyOps;
  bool valid();
  AIE::TileLike getDmaTile();
  bool foundAlloc(AIE::TileLike tile);
  bool foundAlloc(AIE::TileLike tile, air::MemcpyInterface memcpyOp);
  bool foundAlloc(AIE::TileLike tile, air::ChannelOp channel_op);
  bool foundAlloc(AIE::TileLike tile, AIE::DMAChannel channel);
  bool foundPacketFlowAllocInTile(AIE::TileLike tile);

  bool foundAlloc(air::ChannelOp channel_op);
  bool foundAlloc(AIE::DMAChannel channel);

  // Column-keyed; row is implied (shim is always row 0). Returns false for
  // unplaced tiles (tryGetCol() == nullopt) — column-keyed lookups are only
  // meaningful when the tile has a known column.
  bool foundAllocInColumn(int32_t col);
  bool foundAllocInColumn(int32_t col, AIE::DMAChannel channel);
  bool foundPacketFlowAllocInColumn(int32_t col);

  bool operator==(const allocation_info_t &other) const {
    // op interface getOperation() isn't const-qualified; cast away the
    // top-level const for the pointer-equality comparison.
    auto thisOp =
        const_cast<allocation_info_t *>(this)->dma_tile.getOperation();
    auto otherOp =
        const_cast<allocation_info_t &>(other).dma_tile.getOperation();
    return thisOp == otherOp && col == other.col && row == other.row &&
           dma_channel == other.dma_channel &&
           tile_channel == other.tile_channel;
  }
};

// Bundling up memcpy ops into MM2S and S2MM ops sharing the same aie.flow
struct MemcpyBundleAsFlow {
  Operation *air_flow_op; // Either air::DmaMemcpyNdOp or air::ChannelOp
  int flow_op_group = -1; // Scheduling group index; (in scheduling strategy 2,
                          // flows of the same index can share DMA channels)
  std::vector<allocation_info_t> S2MM_alloc;
  std::vector<std::vector<Operation *>> S2MM;
  allocation_info_t MM2S_alloc;
  std::vector<Operation *> MM2S; // air::ChannelPuts
  air::MemorySpace MM2S_memspace;
  air::MemorySpace S2MM_memspace;
  int numMM2SAllocs = 0;
  int numS2MMAllocs = 0;
  std::string
      memcpyResourceType; // The type of mechanism used for the memcpy op,
                          // including dma_stream, dma_packet, and cascade.
  LogicalResult pushBackMemcpyOpToBundle(air::DmaMemcpyNdOp memcpyOp);
  LogicalResult pushBackMemcpyOpToBundle(air::ChannelGetOp memcpyOp);
  LogicalResult pushBackMemcpyOpToBundle(air::ChannelPutOp memcpyOp);
  LogicalResult pushBackMemcpyOpToBundle(air::ChannelInterface memcpyOp);
  MemcpyBundleAsFlow(air::DmaMemcpyNdOp dmaMemcpyOp);
  MemcpyBundleAsFlow(air::ChannelOp chan);
};

class DMAAllocator {

public:
  DMAAllocator() = delete;
  DMAAllocator(AIE::DeviceOp device, air::MemorySpace dmaMemorySpace)
      : device(device), dmaMemorySpace(dmaMemorySpace) {}

  FailureOr<allocation_info_t>
  lookupDMAAllocation(AIE::TileLike tile, air::MemcpyInterface &memcpyOp);
  FailureOr<std::pair<AIE::LockOp, AIE::LockOp>>
  getLockForDMA(air::MemcpyInterface &memcpyOp, AIE::TileLike tile,
                Operation *bufferOp, bool lockRaceConditionFix = false);
  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp, AIE::TileLike tile,
                     int chan, int col, int row, std::vector<int> dma_id);
  void sortMemcpyOps(std::vector<Operation *> dma_memcpy_ops);

protected:
  AIE::DeviceOp device;
  air::MemorySpace dmaMemorySpace;

public:
  std::vector<allocation_info_t> mm2s_allocs, s2mm_allocs;
  std::vector<std::tuple<Operation *, air::ChannelOp, AIE::DMAChannel,
                         AIE::LockOp, AIE::LockOp>>
      lock_allocation_list;
  DenseMap<Value, std::pair<int, int>> passiveSideBufferUseCounters;
};

class TileDMAAllocator : public DMAAllocator {

public:
  TileDMAAllocator(AIE::DeviceOp device)
      : DMAAllocator(device, air::MemorySpace::L1) {}

  // A very simple scheme to allocate channels for dma operations:
  //  <description>
  FailureOr<allocation_info_t>
  simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp, AIE::TileOp tile,
                        int chan = -1);

  FailureOr<AIE::BufferOp> getBuffer(uint64_t, AIE::TileOp tile,
                                     air::MemcpyInterface &memcpyOp);
};

class ShimDMAAllocator : public DMAAllocator {

public:
  // Per-shim DMA channel count (2 MM2S + 2 S2MM on all current targets).
  // Used by allocNewDmaChannel for round-robin channel-index assignment;
  // the placer's per-tile DMA channel budget then spreads logical shim
  // tiles across physical shim columns so channel demand per column is
  // honored.
  int shim_dma_channels;

  ShimDMAAllocator(AIE::DeviceOp device);

  // Allocate a new shim DMA channel. The shim tile is emitted as an
  // unconstrained aie.logical_tile<ShimNOCTile>(?, ?); mlir-aie's
  // aie-place-tiles pass picks the physical column from flow adjacency to
  // placed core peers and respects per-shim DMA channel capacity. The col
  // and row int args record the OTHER side (compute side) of the flow
  // for airrt metadata; they have nothing to do with the shim's eventual
  // physical placement. (RFC #1567: subsumes the deletion of the
  // `colAllocConstraint == "same_column"` heuristic, formerly attempted
  // standalone in #1605 — that PR couldn't compile multi-column workloads
  // because shim tiles were still pre-pinned via createTileViaPlacer.)
  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp, int col, int row,
                     std::vector<Operation *> &dma_ops);

  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp,
                     allocation_info_t existing_alloc,
                     std::vector<Operation *> &dma_ops);

  FailureOr<AIE::ExternalBufferOp> getBuffer(uint64_t &BufferId,
                                             AIE::TileOp tile,
                                             air::MemcpyInterface &memcpyOp);

  FailureOr<air::allocation_info_t>
  foundFlowReuseOpportunity(std::vector<MemcpyBundleAsFlow> memcpy_flows,
                            air::allocation_info_t alloc, bool isMM2S);
};

class MemTileDMAAllocator : public DMAAllocator {

public:
  std::vector<int> memtile_dma_columns;

  MemTileDMAAllocator(AIE::DeviceOp device);

  FailureOr<allocation_info_t>
  simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp, int chan = -1);
  FailureOr<allocation_info_t>
  simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp,
                        allocation_info_t &existing_alloc);

  // tile derived from memcpyOp's buffer; param kept for signature uniformity.
  FailureOr<AIE::BufferOp> getBuffer(uint64_t, AIE::TileOp tile,
                                     air::MemcpyInterface &memcpyOp);

  FailureOr<air::allocation_info_t>
  foundFlowReuseOpportunity(std::vector<MemcpyBundleAsFlow> memcpy_flows,
                            air::allocation_info_t alloc, bool isMM2S);
};

class CascadeAllocator {

public:
  CascadeAllocator() = delete;
  CascadeAllocator(AIE::DeviceOp device)
      : device(device), dmaMemorySpace(air::MemorySpace::L1) {}
  FailureOr<allocation_info_t> coreCascadeAlloc(air::MemcpyInterface &memcpyOp);
  FailureOr<allocation_info_t> allocNewCascade(air::MemcpyInterface &memcpyOp,
                                               AIE::TileOp tile);

  // tile derived from memcpyOp's buffer; param kept for signature uniformity.
  FailureOr<AIE::BufferOp> getBuffer(uint64_t, AIE::TileOp tile,
                                     air::MemcpyInterface &memcpyOp);

protected:
  AIE::DeviceOp device;
  air::MemorySpace dmaMemorySpace;

public:
  std::vector<allocation_info_t> cascade_put_allocs, cascade_get_allocs;
};

LogicalResult
simpleDMAChannelAllocation(std::vector<MemcpyBundleAsFlow> &memcpy_flows,
                           ShimDMAAllocator &shim_dma_alloc,
                           MemTileDMAAllocator &memtile_dma_alloc,
                           TileDMAAllocator &tile_dma_alloc,
                           air::CascadeAllocator &core_cascade_alloc);
template <typename T>
int foundInVector(T item, std::vector<T> vec);
int getSCFForLoopDepth(Operation *o);
bool groupingMemcpysByLoop(std::vector<MemcpyBundleAsFlow> &memcpy_flows);

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_SCHEDULING_UTILS_H
