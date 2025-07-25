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
                              int tileMemSpaceAsInt);
FailureOr<bool> isTileOutbound(air::MemcpyInterface memcpyOp,
                               int tileMemSpaceAsInt);

AIE::TileOp getPhysTileOpOrNull(AIE::DeviceOp aie_device, int col, int row);

// get tileop using physical coordinates
AIE::TileOp getPhysTileOp(AIE::DeviceOp aie_device, int col, int row);

AIE::LockOp allocateLockOp(AIE::DeviceOp aie_device, AIE::TileOp tile,
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
  AIE::TileOp dma_tile = nullptr;
  int64_t col = -1;
  int64_t row = -1;
  AIE::DMAChannel dma_channel = {AIE::DMAChannelDir::MM2S, -1};
  int64_t tile_channel = -1;
  std::vector<int32_t> dma_id;
  std::vector<Operation *> memcpyOps;
  bool valid();
  AIE::TileOp getDmaTile();
  bool foundAlloc(air::ChannelOp channel_op);
  bool foundAlloc(int32_t col, int32_t row, air::MemcpyInterface memcpyOp);
  bool foundAlloc(int32_t col, int32_t row, int chan);
  bool foundAlloc(AIE::DMAChannel channel);
  bool foundAlloc(int32_t col, int32_t row, AIE::DMAChannel channel);
  bool foundAlloc(int32_t col, int32_t row);
  bool foundAlloc(int32_t col, int32_t row, air::ChannelOp channel_op);
  bool foundAlloc(AIE::TileOp tile, AIE::DMAChannel channel);
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
  int MM2S_memspace_as_int;
  int S2MM_memspace_as_int;
  int numMM2SAllocs = 0;
  int numS2MMAllocs = 0;
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
  DMAAllocator(AIE::DeviceOp device, int dmaMemorySpaceAsInt)
      : device(device), DMAMemorySpaceAsInt(dmaMemorySpaceAsInt) {}

  FailureOr<allocation_info_t>
  lookupDMAAllocation(int64_t col, int64_t row, air::MemcpyInterface &memcpyOp);
  FailureOr<std::pair<AIE::LockOp, AIE::LockOp>>
  getLockForDMA(air::MemcpyInterface &memcpyOp, int col, int row,
                Operation *bufferOp, bool lockRaceConditionFix = false);
  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp, AIE::TileOp tile, int chan,
                     int col, int row, std::vector<int> dma_id);
  void sortMemcpyOps(std::vector<Operation *> dma_memcpy_ops);

protected:
  AIE::DeviceOp device;
  int DMAMemorySpaceAsInt;

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
      : DMAAllocator(device, (int)air::MemorySpace::L1) {}

  // A very simple scheme to allocate channels for dma operations:
  //  <description>
  FailureOr<allocation_info_t>
  simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp, int col, int row,
                        int chan);

  FailureOr<AIE::BufferOp> getBuffer(uint64_t, int64_t col, int64_t row,
                                     air::MemcpyInterface &memcpyOp);
};

class ShimDMAAllocator : public DMAAllocator {

public:
  std::vector<int> dma_columns;
  int shim_dma_channels;

  ShimDMAAllocator(AIE::DeviceOp device);

  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp, int col, int row,
                     std::vector<Operation *> &dma_ops,
                     std::string colAllocConstraint);

  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp,
                     allocation_info_t existing_alloc,
                     std::vector<Operation *> &dma_ops);

  FailureOr<AIE::ExternalBufferOp> getBuffer(uint64_t &BufferId, int64_t col,
                                             int64_t row,
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
  simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp, int chan);
  FailureOr<allocation_info_t>
  simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp,
                        allocation_info_t &existing_alloc);

  FailureOr<AIE::BufferOp> getBuffer(uint64_t, int64_t col, int64_t row,
                                     air::MemcpyInterface &memcpyOp);

  FailureOr<air::allocation_info_t>
  foundFlowReuseOpportunity(std::vector<MemcpyBundleAsFlow> memcpy_flows,
                            air::allocation_info_t alloc, bool isMM2S);
};

LogicalResult
simpleDMAChannelAllocation(std::vector<MemcpyBundleAsFlow> &memcpy_flows,
                           ShimDMAAllocator &shim_dma_alloc,
                           MemTileDMAAllocator &memtile_dma_alloc,
                           TileDMAAllocator &tile_dma_alloc);
template <typename T>
int foundInVector(T item, std::vector<T> vec);
int getSCFForLoopDepth(Operation *o);
bool groupingMemcpysByLoop(std::vector<MemcpyBundleAsFlow> &memcpy_flows);

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_SCHEDULING_UTILS_H
