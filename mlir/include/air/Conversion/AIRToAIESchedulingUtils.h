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

// v2 chain-lock helpers (use_lock_race_condition_fix_v2).
//
// An L2 (memtile) buffer is a chain-lock candidate when its access pattern
// is fan-in (>1 writers + 1 reader) or fan-out (1 writer + >1 readers).
// The chain-lock template emits 1 capacity lock (init = # ping-pong slots)
// plus N init=0 signal locks, daisy-chained across the writer (or reader)
// stages, replacing the legacy `1 cap (init=N) + 1 done counter` template
// that allows concurrent stage firing and races on the memtile DMA.
//
// `isChainLockCandidate` is a pure structural predicate; the caller is
// responsible for gating it on the `use_lock_race_condition_fix_v2` pass
// option.
bool isChainLockCandidate(AIE::BufferOp buf);

// Classify a chain-lock buffer's access shape. Counts distinct memcpy
// users (channel puts/gets) on the buffer's underlying memref result.
// Writes/reads from the buffer's perspective:
//   - writers = ops where the buffer's memref is the DST (S2MM into memtile)
//   - readers = ops where the buffer's memref is the SRC (MM2S out of memtile)
// Precondition: `isChainLockCandidate(buf)` returned true.
void classifyChainBuffer(AIE::BufferOp buf, int &numWriters, int &numReaders);

// Deterministic stage index (0..N-1) for one memcpy op on a chain-lock
// buffer. Order = position of `memcpyOp` in the buffer's user list,
// filtered to the matching direction (writers-only for an S2MM op,
// readers-only for an MM2S op).
int computeStageIndexForMemcpyOp(Operation *memcpyOp, AIE::BufferOp buf);

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
  // The other-side LTO (Operation*) of the flow this allocation belongs to.
  // For a shim allocation, this is the memtile (or compute-core) LTO at the
  // far end of the flow; for tile/memtile allocations it is unused. Used as
  // the shim DMA bucket key so that one shim LTO never bundles flows whose
  // far-side LTOs differ — keying on TileLike Operation* identity is lossless
  // even when the far-side LTO is unplaced and its col is unknown (Path B,
  // RFC #1567). Pre-Path-B the bucket keyed on `col`, which was a lossless
  // proxy because each LTO had a unique col; with unhinted LTOs every flow
  // collapsed to col=-1 and one shim LTO swallowed every memtile-side flow.
  Operation *otherSideLTO = nullptr;
  std::vector<int32_t> dma_id;
  std::vector<Operation *> memcpyOps;
  bool valid();
  AIE::TileLike getDmaTile();
  bool foundAlloc(AIE::TileLike tile);
  bool foundAlloc(AIE::TileLike tile, air::MemcpyInterface memcpyOp);
  bool foundAlloc(AIE::TileLike tile, air::ChannelOp channel_op);
  bool foundAlloc(AIE::TileLike tile, AIE::DMAChannel channel);
  bool foundPacketFlowAllocInTile(AIE::TileLike tile);
  // True if a packet-flow memcpy on `tile` belongs to the SAME logical flow as
  // `memcpyOp` (same channel decl + same constant bundle indices). Used for
  // S2MM-side collapse discrimination: distinct sources must not share a
  // physical channel. A non-constant index cannot be proven equal and is
  // treated as distinct (safe: forces a separate channel).
  bool foundSamePacketFlowInTile(AIE::TileLike tile,
                                 air::MemcpyInterface memcpyOp);

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
  // MM2S is symmetric to S2MM: one allocation per distinct producer tile, to
  // support N producers converging on a single destination S2MM (multi-producer
  // packet flow). MM2S_alloc is indexed by producer tile in [0, numMM2SAllocs);
  // MM2S holds one entry per put op, so several MM2S ops from the same producer
  // tile map to one MM2S_alloc entry (the index spaces do NOT align). Grouping
  // of puts to producer-tile indices happens in simpleDMAChannelAllocation.
  // Single-producer flows keep numMM2SAllocs == 1.
  std::vector<allocation_info_t> MM2S_alloc;
  std::vector<Operation *> MM2S; // air::ChannelPuts (one entry per put op)
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

// v2: chain-lock allocation record for one shared L2 buffer.
//   cap_lock: capacity (init = pp_slots, the number of ping-pong buffer
//             instances). When pp_slots == 2 the cap admits two
//             concurrent stage-K writes (one per buffer) before blocking,
//             matching a shared-L2 2-slot producer-consumer pattern.
//   sig_locks: N init=0 locks shared across BOTH ping/pong buffer
//             instances. One per writer→writer (or reader→reader)
//             transition + the writer→reader (or reader→writer) handoff.
//             For fan-in (N writers + 1 reader): sig_locks[i] signals
//             "writer i done"; writer i+1 acquires sig_locks[i]; the
//             reader acquires sig_locks[N-1] and releases cap_lock.
//             For fan-out (1 writer + N readers): sig_locks[0] signals
//             "writer done"; reader 0 acquires sig_locks[0]; reader i+1
//             acquires sig_locks[i+1] released by reader i; the last
//             reader releases cap_lock.
//   primary_buf / twin_buf: the two ping-pong buffer instances. When
//             pp_slots == 1, twin_buf is null and only primary_buf is
//             used. Both buffers share the cap_lock + sig_locks above.
//   pp_slots: 1 = single-buffer chain (no ping-pong overlap),
//             2 = 2-buffer ping-pong (default under v2).
struct ChainLockSet {
  AIE::LockOp cap_lock;
  SmallVector<AIE::LockOp> sig_locks;
  AIE::BufferOp primary_buf = nullptr;
  AIE::BufferOp twin_buf = nullptr;
  int n_writers = 0;
  int n_readers = 0;
  int pp_slots = 1;
  bool isFanIn() const { return n_writers > 1 && n_readers == 1; }
  bool isFanOut() const { return n_writers == 1 && n_readers > 1; }
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
                Operation *bufferOp, bool lockRaceConditionFix = false,
                bool lockRaceConditionFixV2 = false);
  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp, AIE::TileLike tile,
                     int chan, int col, int row, std::vector<int> dma_id);
  void sortMemcpyOps(std::vector<Operation *> dma_memcpy_ops);

  // v2: get-or-create the chain-lock allocation for a shared L2 buffer.
  // Allocates `cap_lock` + N signal locks on first call for `buf`; reuses
  // the cached set on subsequent calls. Returns failure if the buffer is
  // not a chain-lock candidate.
  FailureOr<ChainLockSet *> getOrCreateChainLockSet(AIE::BufferOp buf,
                                                    AIE::TileLike tile);

  // v2: promote a chain-lock set to 2-slot ping-pong. Records the twin buffer,
  // sets pp_slots = 2, and bumps cap_lock init to 2 together so the slot count
  // and buffer-instance count stay in sync.
  void activateChainPingPong(ChainLockSet &cls, AIE::BufferOp twin);

  // v2: pick the (acquire, release) lock pair for one BD's position in
  // the chain. `stage` is the per-direction stage index returned by
  // `computeStageIndexForMemcpyOp`.
  std::pair<AIE::LockOp, AIE::LockOp>
  pickChainBdLocks(const ChainLockSet &cls, AIE::DMAChannelDir dir, int stage);

protected:
  AIE::DeviceOp device;
  air::MemorySpace dmaMemorySpace;

public:
  std::vector<allocation_info_t> mm2s_allocs, s2mm_allocs;
  std::vector<std::tuple<Operation *, air::ChannelOp, AIE::DMAChannel,
                         AIE::LockOp, AIE::LockOp>>
      lock_allocation_list;
  DenseMap<Value, std::pair<int, int>> passiveSideBufferUseCounters;
  // v2: one chain-lock set per shared L2 buffer (keyed on aie.buffer op).
  DenseMap<Operation *, ChainLockSet> chain_lock_sets;
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
  // Caps how many channels AIR may pack onto one shim LTO before opening
  // a new LTO; aie-place-tiles (with merge-ltos=false) then maps each LTO
  // to its own physical shim col.
  int shim_dma_channels;

  ShimDMAAllocator(AIE::DeviceOp device);

  // Allocate a new shim DMA channel. The shim tile is emitted as an
  // unconstrained aie.logical_tile<ShimNOCTile>(?, ?). aie-place-tiles
  // assigns the physical column from flow adjacency to placed core peers.
  // `otherSide` is the LTO (or physical tile) at the OTHER end of the flow
  // (memtile or core); its Operation* identity is the bucket key used to
  // group shim allocations so flows targeting distinct far-side LTOs land
  // on distinct shim LTOs. col/row are kept for airrt metadata only and
  // may be -1 when otherSide is an unhinted LTO.
  FailureOr<allocation_info_t>
  allocNewDmaChannel(air::MemcpyInterface &memcpyOp, AIE::TileLike otherSide,
                     int col, int row, std::vector<Operation *> &dma_ops);

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

// Shared-MM2S packet flow ordering. Receiver mem chain + core follow
// herd-source order; sender pkt_ids + dispatch order must too. Call
// sortPacketShimFlowsByReceiverOrder before simpleDMAChannelAllocation,
// reorderL3PacketPutsByFlowOrder after.
bool isPacketShimFlow(const MemcpyBundleAsFlow &f);
void sortPacketShimFlowsByReceiverOrder(
    std::vector<MemcpyBundleAsFlow> &memcpy_flows, AIE::DeviceOp aie_device);
void reorderL3PacketPutsByFlowOrder(
    AIE::DeviceOp aie_device,
    const std::vector<MemcpyBundleAsFlow> &memcpy_flows);
template <typename T>
int foundInVector(T item, std::vector<T> vec);
int getSCFForLoopDepth(Operation *o);
bool groupingMemcpysByLoop(std::vector<MemcpyBundleAsFlow> &memcpy_flows);

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_SCHEDULING_UTILS_H
