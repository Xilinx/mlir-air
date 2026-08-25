//===- AIRToAIESchedulingUtils.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToAIESchedulingUtils.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

#include <limits>
#include <mutex>
#include <set>

#define DEBUG_TYPE "air-to-aie-scheduling-utils"

using namespace mlir;

namespace {
// A channel declared with `air.dedicated_dma_channel` must get its OWN physical
// DMA channel and never be time-multiplexed (packet-collapsed) with any other
// flow on the same tile, in either direction. Used to keep a latency-critical
// packet flow off a shared DMA channel that another (e.g. later-phase) flow
// would otherwise collapse onto and BD-order ahead of -- starving the first
// flow's consumer. Honored by the packet-reuse branches of both the shim and
// MemTile DMA allocators.
bool memcpyIsDedicatedChannel(xilinx::air::MemcpyInterface mc) {
  auto chan = mlir::dyn_cast_if_present<xilinx::air::ChannelInterface>(
      mc.getOperation());
  if (!chan)
    return false;
  auto decl = xilinx::air::getChannelDeclarationThroughSymbol(chan);
  return decl && decl->hasAttr(xilinx::air::attrs::DedicatedDmaChannel);
}
} // namespace

namespace xilinx {

FailureOr<bool> air::isTileInbound(air::MemcpyInterface memcpyOp,
                                   air::MemorySpace tileMemSpace) {
  if (memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    auto src_memory_space = air::getMemorySpace(
        llvm::cast<BaseMemRefType>(memcpyOp.getSrcMemref().getType()));
    auto dst_memory_space = air::getMemorySpace(
        llvm::cast<BaseMemRefType>(memcpyOp.getDstMemref().getType()));
    if (src_memory_space && *src_memory_space == tileMemSpace)
      return false;
    else if (dst_memory_space && *dst_memory_space == tileMemSpace)
      return true;
    memcpyOp->emitOpError(
        "neither src nor dst use the tile's memory space, indicating a "
        "potential error in the compilation workflow.");
    return failure();
  } else if (!memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    return true;
  } else
    return false;
}
FailureOr<bool> air::isTileOutbound(air::MemcpyInterface memcpyOp,
                                    air::MemorySpace tileMemSpace) {
  auto isTileInbRes = isTileInbound(memcpyOp, tileMemSpace);
  if (failed(isTileInbRes))
    return failure();
  return !(*isTileInbRes);
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
    // Skip past both physical and logical tile ops so the new TileOp lands
    // after them (preserves stable ordering for downstream consumers like
    // getMemtilesFromDeviceOp that index by IR position).
    if (isa<AIE::TileOp, AIE::LogicalTileOp>(o))
      builder.setInsertionPointAfter(&o);
    else
      break;
  }
  return AIE::TileOp::create(builder, UnknownLoc::get(aie_device.getContext()),
                             col, row);
}

AIE::LockOp air::allocateLockOp(AIE::DeviceOp aie_device, AIE::TileLike tile,
                                int init, int id, StringAttr name) {
  AIE::LockOp lock = nullptr;
  std::set<int> ids;
  Operation *tileOp = tile.getOperation();
  // Each (logical or physical) tile owns its own lock-ID space. The
  // aie-place-tiles pass is invoked with merge-ltos=false from aircc, so
  // distinct LTOs never collapse onto a shared physical tile — no need
  // to reserve IDs across other LTOs.
  aie_device.walk([&](AIE::LockOp l) {
    if (l.getTile().getDefiningOp() != tileOp)
      return;
    if (!l.getLockID().has_value())
      return;
    auto i = l.getLockIDValue();
    if (i == id)
      lock = l;
    ids.insert(i);
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
  Operation *t = tileOp;
  // Walk past contiguous tile defining ops (TileOp or LogicalTileOp) so the
  // new lock lands after them.
  while (t->getNextNode() &&
         isa<AIE::TileOp, AIE::LogicalTileOp>(t->getNextNode()))
    t = t->getNextNode();
  b.setInsertionPointAfter(t);
  auto lockOp = AIE::LockOp::create(b, tileOp->getLoc(), tileOp->getResult(0),
                                    new_id, init);
  if (name)
    lockOp->setAttr(SymbolTable::getSymbolAttrName(), name);
  return lockOp;
}

std::stringstream air::generateBufferNameInStringStream(StringRef prefix,
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
    ss << prefix.str() << BufferId++;
  }
  return ss;
}

AIE::ExternalBufferOp air::allocateExternalBufferOp(uint64_t &BufferId,
                                                    MemRefType memrefTy,
                                                    AIE::DeviceOp device,
                                                    mlir::StringAttr attr,
                                                    int x, int y) {

  auto builder = OpBuilder::atBlockBegin(device.getBody());
  AIE::ExternalBufferOp bufferOp = AIE::ExternalBufferOp::create(
      builder, builder.getUnknownLoc(), memrefTy, nullptr, nullptr);

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

int64_t air::get1DOffset(ArrayRef<OpFoldResult> memcpy_offsets,
                         ArrayRef<OpFoldResult> memcpy_strides) {
  if (memcpy_offsets.empty())
    return 0;

  int64_t one_d_offset = 0;
  for (int i = memcpy_offsets.size() - 1; i >= 0; i--) {
    auto offset = mlir::getConstantIntValue(memcpy_offsets[i]);
    if ((unsigned)i == memcpy_offsets.size() - 1)
      one_d_offset += *offset;
    else {
      auto stride_i = mlir::getConstantIntValue(memcpy_strides[i]);
      one_d_offset += (*offset) * (*stride_i);
    }
  }
  return one_d_offset;
}

// Fast path: if all memcpy_ops map to an equivalent BD (same memref and
// matching offsets/sizes/strides), collapse them into a single BD task
// (one repeat-count entry).
static bool chansMappedToEquivalentBDs(air::ChannelInterface chanA,
                                       air::ChannelInterface chanB) {
  // Two transfers on different channels are different FLOWS: they carry
  // different packet ids and land on different destinations, so their BDs are
  // not interchangeable however alike the access patterns look. Folding them
  // would keep one BD and silently drop the other flow's routing -- every
  // packet would then go wherever the surviving BD points, and the other
  // consumers would wait forever. (chansPartOfSameRotation below already keys
  // on the declaration for the same reason.)
  if (air::getChannelDeclarationThroughSymbol(chanA) !=
      air::getChannelDeclarationThroughSymbol(chanB))
    return false;
  if (chanA.getMemref() != chanB.getMemref())
    return false;
  auto offsetsA = chanA.getMixedOffsets(), offsetsB = chanB.getMixedOffsets();
  auto sizesA = chanA.getMixedSizes(), sizesB = chanB.getMixedSizes();
  auto stridesA = chanA.getMixedStrides(), stridesB = chanB.getMixedStrides();
  if (offsetsA.size() != offsetsB.size() || sizesA.size() != sizesB.size() ||
      stridesA.size() != stridesB.size())
    return false;
  auto zipped_operands =
      llvm::zip_equal(llvm::concat<OpFoldResult>(offsetsA, sizesA, stridesA),
                      llvm::concat<OpFoldResult>(offsetsB, sizesB, stridesB));
  bool wrapsAndStridesAllEquivalent = llvm::all_of(
      zipped_operands, [](std::tuple<OpFoldResult, OpFoldResult> pair) {
        return isEqualConstantIntOrValue(std::get<0>(pair), std::get<1>(pair));
      });
  return wrapsAndStridesAllEquivalent;
}

// Check if two channel operations are part of an N-buffer rotation pattern.
// They are part of the same rotation if:
// 1. They belong to the same air.channel declaration
// 2. Their memrefs have the same type (shape, element type, memory space)
// 3. Their sizes and strides are equivalent (access pattern match)
// Note: Unlike chansMappedToEquivalentBDs, this allows different buffer
// values as long as they have the same type and access pattern.
bool air::chansPartOfSameRotation(air::ChannelInterface chanA,
                                  air::ChannelInterface chanB) {
  // Must use same channel declaration
  auto chanDeclA = air::getChannelDeclarationThroughSymbol(chanA);
  auto chanDeclB = air::getChannelDeclarationThroughSymbol(chanB);
  if (chanDeclA != chanDeclB)
    return false;

  // Memrefs must have same type (but can be different buffer values)
  auto memrefTypeA = llvm::cast<MemRefType>(chanA.getMemref().getType());
  auto memrefTypeB = llvm::cast<MemRefType>(chanB.getMemref().getType());
  if (memrefTypeA != memrefTypeB)
    return false;

  // Sizes and strides must match (ignoring offsets which vary per buffer)
  auto sizesA = chanA.getMixedSizes(), sizesB = chanB.getMixedSizes();
  auto stridesA = chanA.getMixedStrides(), stridesB = chanB.getMixedStrides();
  if (sizesA.size() != sizesB.size() || stridesA.size() != stridesB.size())
    return false;

  auto zipped = llvm::zip_equal(llvm::concat<OpFoldResult>(sizesA, stridesA),
                                llvm::concat<OpFoldResult>(sizesB, stridesB));
  return llvm::all_of(zipped, [](std::tuple<OpFoldResult, OpFoldResult> pair) {
    return isEqualConstantIntOrValue(std::get<0>(pair), std::get<1>(pair));
  });
}

static bool dmasMappedToEquivalentBDs(air::DmaMemcpyNdOp dmaA,
                                      air::DmaMemcpyNdOp dmaB) {
  return OperationEquivalence::isEquivalentTo(
      dmaA, dmaB, OperationEquivalence::IgnoreLocations);
}

bool air::memcpyIMappedToEquivalentBDs(Operation *opA, Operation *opB) {
  if (auto chanA = dyn_cast_if_present<air::ChannelInterface>(opA))
    if (auto chanB = dyn_cast_if_present<air::ChannelInterface>(opB))
      return chansMappedToEquivalentBDs(chanA, chanB);
  if (auto dmaA = dyn_cast_if_present<air::DmaMemcpyNdOp>(opA))
    if (auto dmaB = dyn_cast_if_present<air::DmaMemcpyNdOp>(opB))
      return dmasMappedToEquivalentBDs(dmaA, dmaB);
  return false; // Unknown or different air::MemcpyInterface op types.
}

// Canonicalize a chain of memcpy ops as candidates to map to dma bds, by
// removing repetitive patterns. Returns the unique repeating unit, or an empty
// vector when the chain does not repeat.
llvm::SetVector<Operation *>
air::getUniqueBDPattern(llvm::SetVector<Operation *> memcpyIOps) {
  // Get a vector of unique BDs.
  llvm::SetVector<Operation *> uniqueBDPattern;
  auto opIt = memcpyIOps.begin();
  while (opIt != memcpyIOps.end() &&
         llvm::none_of(uniqueBDPattern, [opIt](Operation *op1) {
           return memcpyIMappedToEquivalentBDs(*opIt, op1);
         })) {
    uniqueBDPattern.insert(*opIt);
    opIt++;
  }

  unsigned idx = 0;
  while (opIt != memcpyIOps.end()) {
    // BD repetition found. Check if repeating pattern.
    if (!memcpyIMappedToEquivalentBDs(*opIt, uniqueBDPattern[idx]))
      return llvm::SetVector<Operation *>(); // Chain isn't repeating. Return
                                             // an empty vector.
    opIt++;
    idx++;
    idx %= uniqueBDPattern.size();
  }

  // Repeating BD chain successfully detected.
  return uniqueBDPattern;
}

// Given a vector of memcpy operations, split them into BD tasks paired with the
// repeat count each task runs at, relative to a common ancestor region. See the
// header for why adjacency in program order gates the merge.
SmallVector<std::pair<int, llvm::SetVector<Operation *>>>
air::getRepeatCounts(std::vector<Operation *> memcpy_ops) {
  SmallVector<std::pair<int, llvm::SetVector<Operation *>>> repeatCounts;
  llvm::SetVector<Operation *> memcpyIOps;
  for (auto o : memcpy_ops) {
    memcpyIOps.insert(o);
  }

  auto uniqueMemcpyIPattern = getUniqueBDPattern(memcpyIOps);
  if (!uniqueMemcpyIPattern.empty())
    memcpyIOps = uniqueMemcpyIPattern;

  // Handle "prefix + repeating suffix" pattern (e.g., [Q, K, K, K...K]).
  // Collapse to [Q, K] circular chain (2 BDs instead of N+1), avoiding
  // memtile BD exhaustion for large chunks_per_stage.
  // Minimum number of repeated suffix ops before collapsing. Small counts
  // (e.g., 3 ops from lock race condition fix) are intentional and must not
  // be collapsed. The prefix+suffix pattern targets flash attention with
  // chunks_per_stage >> 4, where BD exhaustion is a real risk.
  constexpr unsigned kMinSuffixOpsForCollapse = 3;
  if (uniqueMemcpyIPattern.empty() &&
      memcpyIOps.size() > kMinSuffixOpsForCollapse + 1) {
    llvm::SetVector<Operation *> suffix;
    auto it = memcpyIOps.begin();
    ++it;
    while (it != memcpyIOps.end()) {
      suffix.insert(*it);
      ++it;
    }
    auto suffixPattern = getUniqueBDPattern(suffix);
    if (!suffixPattern.empty() && suffixPattern.size() == 1) {
      llvm::SetVector<Operation *> prefixPlusSuffix;
      prefixPlusSuffix.insert(*memcpyIOps.begin());
      prefixPlusSuffix.insert(*suffixPattern.begin());
      memcpyIOps = prefixPlusSuffix;
    }
  }

  // Detect if all operations form an N-buffer rotation pattern.
  // For N-buffer rotation (e.g., 4-buffer sliding window), we need to generate
  // a single circular BD chain even if operations have different loop contexts.
  auto detectNBufferRotation =
      [](const llvm::SetVector<Operation *> &ops) -> bool {
    if (ops.size() < 2)
      return false;

    // Check all ops are channel operations sharing same rotation pattern
    auto *firstOp = *ops.begin();
    auto firstChan = dyn_cast_if_present<air::ChannelInterface>(firstOp);
    if (!firstChan)
      return false;

    // Count unique buffers
    llvm::DenseSet<Value> uniqueBuffers;
    for (auto *op : ops) {
      auto chanOp = dyn_cast_if_present<air::ChannelInterface>(op);
      if (!chanOp || !chansPartOfSameRotation(firstChan, chanOp))
        return false;
      uniqueBuffers.insert(chanOp.getMemref());
    }

    // A genuine rotation interleaves buffers inside a shared loop (a peeled
    // steady-state loop unrolled across the buffer pool), so at least two sites
    // share one enclosing loop. Sites that each sit alone in their own loop are
    // time-multiplexed block consumers, NOT a rotation -- a single circular BD
    // chain would mis-deliver (each block would see only every Nth buffer). Let
    // those fall through to per-op sequential BDs.
    // Note: this is coarse -- any two sites sharing a loop marks the whole set
    // as a rotation; a channel/tile mixing a rotation with block consumers on
    // one DMA channel (not observed in practice) would be over-collapsed.
    llvm::DenseMap<Operation *, unsigned> loopSiteCount;
    bool anySharedLoop = false;
    for (auto *op : ops)
      if (auto loop = op->getParentOfType<LoopLikeOpInterface>())
        if (++loopSiteCount[loop.getOperation()] >= 2)
          anySharedLoop = true;
    if (!anySharedLoop)
      return false;

    // Valid rotation: multiple unique buffers, total ops divisible by buffer
    // count
    unsigned numBuffers = uniqueBuffers.size();
    return numBuffers >= 2 && ops.size() % numBuffers == 0;
  };

  // If N-buffer rotation pattern detected, return all ops with same repeat
  // count. This ensures generateDmaBdProgram() creates a single circular BD
  // chain (infiniteBDLoopMode = true) instead of separate terminated tasks.
  if (detectNBufferRotation(memcpyIOps)) {
    SmallVector<Operation *> opVec = memcpyIOps.takeVector();
    repeatCounts.emplace_back(0, llvm::SetVector<Operation *>());
    for (auto *op : opVec)
      repeatCounts.back().second.insert(op);
    return repeatCounts;
  }

  // Get the deepest region which is ancestor to all memcpyIOps.
  SmallVector<Operation *> memcpyIOpVec = memcpyIOps.takeVector();
  Region *commonRegion =
      air::findCommonRegionContainingAllAncestors(memcpyIOpVec);
  if (!commonRegion)
    return repeatCounts;

  // Get each memcpy op's repeat count, relative to the common region, and open
  // a new task whenever the count changes. memcpyIOpVec is in program order, so
  // a maximal run of equal counts is exactly the set of transfers that may
  // share one task without any of them jumping over another.
  //
  // Two separate loops with the same trip count are NOT one task. Under the old
  // grouping-by-count they became one, which spliced the later loop's BDs into
  // the earlier loop's chain and ran them before every task in between -- so a
  // three-phase egress [A x4, B x32, C x4] emitted A and C interleaved up front
  // and B afterwards. Counts stayed right, order did not.
  for (auto o : memcpyIOpVec) {
    int tripCount = 1;
    Region *currRegion = o->getParentRegion();
    while (commonRegion->isAncestor(currRegion)) {
      Operation *parent = currRegion->getParentOp();
      currRegion = currRegion->getParentRegion();
      auto affineFor = dyn_cast_if_present<affine::AffineForOp>(parent);
      auto scfFor = dyn_cast_if_present<scf::ForOp>(parent);
      if (affineFor && affineFor.hasConstantBounds()) {
        tripCount *= *air::getStaticAffineForTripCountAsInt(affineFor);
      } else if (scfFor && air::getStaticScfForTripCountAsInt(scfFor)) {
        tripCount *= *air::getStaticScfForTripCountAsInt(scfFor);
      }
    }
    // In English, repeat count is trip count minus one.
    int rep = tripCount - 1;
    if (repeatCounts.empty() || repeatCounts.back().first != rep)
      repeatCounts.emplace_back(rep, llvm::SetVector<Operation *>());
    repeatCounts.back().second.insert(o);
  }

  return repeatCounts;
}

std::vector<AIE::BDDimLayoutAttr>
air::getWrapsAndStrides(ArrayRef<OpFoldResult> memcpy_sizes,
                        ArrayRef<OpFoldResult> memcpy_strides,
                        MLIRContext *ctx) {
  if (memcpy_sizes.empty() || memcpy_strides.empty())
    return std::vector<AIE::BDDimLayoutAttr>();
  std::vector<AIE::BDDimLayoutAttr> output;
  for (auto [wrapVal, stepsizeVal] :
       llvm::zip_equal(memcpy_sizes, memcpy_strides)) {
    auto stepsize = mlir::getConstantIntValue(stepsizeVal);
    auto wrap = mlir::getConstantIntValue(wrapVal);
    auto tuple = AIE::BDDimLayoutAttr::get(ctx, *wrap, *stepsize);
    output.push_back(tuple);
  }
  return output;
}

std::pair<int64_t, int64_t>
air::getLockValuePair(const AIE::AIETargetModel &targetModel,
                      Value buffer_memref) {
  if (!targetModel.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks))
    return std::make_pair(0, 0);

  // Infer semaphore lock values using buffer op
  // TODO: What if a buffer memref is read or written by multiple channels?
  if (!llvm::isa<BaseMemRefType>(buffer_memref.getType()))
    return std::make_pair(-1, -1);
  int read_counter = 0;
  int write_counter = 0;
  for (auto user : buffer_memref.getUsers()) {
    if (auto memcpyOp = dyn_cast_if_present<air::MemcpyInterface>(user)) {
      if (buffer_memref == memcpyOp.getSrcMemref())
        read_counter++;
      else if (buffer_memref == memcpyOp.getDstMemref())
        write_counter++;
    } else if (isa<affine::AffineLoadOp>(user))
      read_counter++;
    else if (isa<affine::AffineStoreOp>(user))
      write_counter++;
    else if (auto linalgop = dyn_cast_if_present<linalg::LinalgOp>(user)) {
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
    return std::make_pair(llvm::divideCeilSigned(read_counter, write_counter),
                          1);
  else
    return std::make_pair(1,
                          llvm::divideCeilSigned(write_counter, read_counter));
}

// ----------------------------------------------------------------------
// v2 chain-lock helpers (use_lock_race_condition_fix_v2). See the header
// for the semantic description.
// ----------------------------------------------------------------------

// Endpoint identity for a channel memcpy: (channel symbol, constant indices).
// Returns nullopt if any bundle index is non-constant — such endpoints cannot
// be proven equal, so they must NOT be deduped (collapsing distinct dynamic
// endpoints would undercount writers/readers and mis-size the chain).
using ChainEndpointKey = std::pair<StringRef, SmallVector<int64_t, 4>>;
static std::optional<ChainEndpointKey>
getChainEndpointKey(air::ChannelInterface chan) {
  SmallVector<int64_t, 4> idx;
  for (auto v : chan.getIndices()) {
    auto c = getConstantIntValue(v);
    if (!c)
      return std::nullopt;
    idx.push_back(*c);
  }
  return ChainEndpointKey{chan.getChanName(), idx};
}

// Ordered list of one representative memcpy op per chain stage on `buf`, in
// use-list order, filtered to one direction (writers = buffer is the DST/S2MM;
// readers = buffer is the SRC/MM2S). Channel endpoints sharing the same
// (symbol, constant-indices) key collapse to one stage (dedupes scf.for unroll
// / ping-pong duplication that would otherwise inflate the fan-in/out counts).
// Endpoints without a provable key — non-channel memcpy (legacy
// air.dma_memcpy_nd) or a channel with any dynamic index — are each their own
// stage. Single source of truth: the stage count is the list size and a memcpy
// op's stage index is its representative's position.
static SmallVector<Operation *> getOrderedChainEndpoints(AIE::BufferOp buf,
                                                         bool writers) {
  SmallVector<Operation *> stages;
  llvm::SetVector<ChainEndpointKey> seenKeys;
  for (auto user : buf.getResult().getUsers()) {
    auto memcpyOp = dyn_cast<air::MemcpyInterface>(user);
    if (!memcpyOp)
      continue;
    bool isWriter = (buf.getResult() == memcpyOp.getDstMemref());
    bool isReader = (buf.getResult() == memcpyOp.getSrcMemref());
    if ((writers && !isWriter) || (!writers && !isReader))
      continue;
    std::optional<ChainEndpointKey> key;
    if (auto chan = dyn_cast<air::ChannelInterface>(user))
      key = getChainEndpointKey(chan);
    // Dedupe only provably-equal keyed endpoints; everything else is its own
    // stage.
    if (!key || seenKeys.insert(*key))
      stages.push_back(user);
  }
  return stages;
}

static void countChainBufferRoles(AIE::BufferOp buf, int &numWriters,
                                  int &numReaders) {
  numWriters = getOrderedChainEndpoints(buf, /*writers=*/true).size();
  numReaders = getOrderedChainEndpoints(buf, /*writers=*/false).size();
}

// Eligibility guard shared by the memtile-specific lock predicates: only a
// non-null L2 (memtile) buffer qualifies.
static bool isL2MemtileBuffer(AIE::BufferOp buf) {
  if (!buf)
    return false;
  auto memrefTy = dyn_cast<MemRefType>(buf.getResult().getType());
  return memrefTy && air::isL2(memrefTy);
}

bool air::isChainLockCandidate(AIE::BufferOp buf) {
  // Predicate is shape-based on the buffer's user list. Only L2 memtile
  // buffers are eligible (this is a memtile-specific lock pattern).
  if (!isL2MemtileBuffer(buf))
    return false;
  int nW = 0, nR = 0;
  countChainBufferRoles(buf, nW, nR);
  // Fan-in: N writers (N>1) + 1 reader. The chain-lock is required here to
  // prevent write-side corruption, so the opt-out below is NOT honored.
  if (nW > 1 && nR == 1)
    return true;
  // Fan-out: 1 writer + N readers (N>1).
  if (nW == 1 && nR > 1) {
    // Opt-out: a buffer explicitly pinned with `air.no_chain_lock` keeps the
    // legacy counted-lock template. Used for fan-out broadcast buffers whose N
    // readers are independent compute cores (e.g. a per-column weight fan):
    // concurrent reads never conflict, so the daisy-chain only over-serializes
    // them and can deadlock against a competing fan-in chain under multi-block
    // streaming. Scoped to fan-out: reverting fan-in to the counted lock would
    // reintroduce the very race the chain-lock fixes.
    if (buf->hasAttr(air::attrs::NoChainLock))
      return false;
    return true;
  }
  // Single-writer/single-reader (legacy 1:1) or MIMO (M writers + N
  // readers) are NOT chain-lock candidates; legacy lock template
  // applies.
  return false;
}

// True iff `buf` is an L2 memtile buffer that is FILLED by DMA (>=1 writer
// endpoint) but NEVER READ (0 reader endpoints) within the segment -- a "pure
// drain" (e.g. a readback whose data is discarded). Such a buffer's receiving
// BD must SELF-RECYCLE on a single lock: there is no consumer to release a
// separate producer lock or acquire a separate handoff lock, so the legacy
// producer/consumer pair (acquire wlock / release rlock) fires exactly once
// then DEADLOCKS on the next dispatch re-acquiring wlock.
static bool isConsumerlessMemtileDrain(AIE::BufferOp buf) {
  if (!isL2MemtileBuffer(buf))
    return false;
  int nW = 0, nR = 0;
  air::classifyChainBuffer(buf, nW, nR);
  return nW >= 1 && nR == 0;
}

void air::classifyChainBuffer(AIE::BufferOp buf, int &numWriters,
                              int &numReaders) {
  countChainBufferRoles(buf, numWriters, numReaders);
}

int air::computeStageIndexForMemcpyOp(Operation *memcpyOp, AIE::BufferOp buf) {
  auto mc = dyn_cast<air::MemcpyInterface>(memcpyOp);
  if (!mc || !buf)
    return -1;
  bool isWriter = (buf.getResult() == mc.getDstMemref());
  auto stages = getOrderedChainEndpoints(buf, /*writers=*/isWriter);
  auto myChan = dyn_cast<air::ChannelInterface>(memcpyOp);
  std::optional<ChainEndpointKey> myKey =
      myChan ? getChainEndpointKey(myChan) : std::nullopt;
  for (auto [i, rep] : llvm::enumerate(stages)) {
    if (myKey) {
      // Identically-keyed ops (scf.for unroll / ping-pong duplication) share a
      // stage, so match on the endpoint key rather than op identity.
      auto repChan = dyn_cast<air::ChannelInterface>(rep);
      auto repKey = repChan ? getChainEndpointKey(repChan) : std::nullopt;
      if (repKey && *repKey == *myKey)
        return static_cast<int>(i);
    } else if (rep == memcpyOp)
      // Unkeyed (non-channel or dynamic-index): match by op identity.
      return static_cast<int>(i);
  }
  return -1;
}

FailureOr<air::ChainLockSet *>
air::DMAAllocator::getOrCreateChainLockSet(AIE::BufferOp buf,
                                           AIE::TileLike tile) {
  if (!buf || !isChainLockCandidate(buf))
    return failure();
  auto it = chain_lock_sets.find(buf.getOperation());
  if (it != chain_lock_sets.end())
    return &it->second;

  int nW = 0, nR = 0;
  classifyChainBuffer(buf, nW, nR);
  int nStages = (nW > 1) ? nW : nR; // fan-in or fan-out

  ChainLockSet cls;
  cls.n_writers = nW;
  cls.n_readers = nR;
  cls.primary_buf = buf;

  // Start single-slot (cap init = 1), which is safe for any chain shape. If
  // generateDmaBdProgram later allocates a twin buffer it calls
  // activateChainPingPong, which bumps the cap and pp_slots together so the
  // slot count and buffer count never diverge.
  cls.pp_slots = 1;
  // A refeed buffer (air.refeed_count=N, single-buffer count-free re-broadcast)
  // needs the cap_lock primed to N: the first writer acquires cap >= N, and the
  // single reader releases cap by 1 per re-send (N sends drain sig[last]=N and
  // restore cap=N). Default init=1 would deadlock the first writer's acq>=N.
  int capInit = static_cast<int>(
      std::max<int64_t>(1, air::getRefeedCount(buf.getOperation())));
  cls.cap_lock = allocateLockOp(device, tile, /*init=*/capInit);

  // N init=0 signal locks for the writer→writer (or reader→reader)
  // transitions plus the producer→consumer (or last-reader→producer)
  // handoff. Shared across both ping/pong instances.
  cls.sig_locks.reserve(nStages);
  for (int i = 0; i < nStages; i++)
    cls.sig_locks.push_back(allocateLockOp(device, tile, 0));

  auto inserted = chain_lock_sets.insert({buf.getOperation(), std::move(cls)});
  return &inserted.first->second;
}

void air::DMAAllocator::activateChainPingPong(ChainLockSet &cls,
                                              AIE::BufferOp twin) {
  // Bump the twin buffer, slot count, and cap-lock init together: the cap
  // (slot count) must always equal the number of buffer instances, so these
  // updates are one atomic operation rather than three scattered writes.
  cls.twin_buf = twin;
  cls.pp_slots = 2;
  cls.cap_lock->setAttr(
      "init",
      IntegerAttr::get(IntegerType::get(cls.cap_lock->getContext(), 32), 2));
}

std::pair<AIE::LockOp, AIE::LockOp>
air::DMAAllocator::pickChainBdLocks(const ChainLockSet &cls,
                                    AIE::DMAChannelDir dir, int stage) {
  // generateDmaBd interprets the returned pair as (rlock, wlock) — the
  // legacy producer-consumer convention — and direction-dependently
  // chooses which is acquired vs released:
  //   S2MM (writer): acquireLock = pair.second (wlock),
  //                  releaseLock = pair.first  (rlock)
  //   MM2S (reader): acquireLock = pair.first  (rlock),
  //                  releaseLock = pair.second (wlock)
  // We map our chain-lock semantics onto this convention by populating
  // `first` / `second` so that the direction-dependent acquire/release
  // gives the correct chain semantics.
  AIE::LockOp toAcquire, toRelease;

  if (cls.isFanIn()) {
    // Writers serialized W0 → W1 → ... → W{N-1} → Reader → Cap → W0
    if (dir == AIE::DMAChannelDir::S2MM) {
      // Writer stage `stage` (0..N-1)
      toAcquire = (stage == 0) ? cls.cap_lock : cls.sig_locks[stage - 1];
      toRelease = cls.sig_locks[stage];
    } else {
      // The single reader: acquire last signal lock, release cap lock.
      toAcquire = cls.sig_locks[cls.n_writers - 1];
      toRelease = cls.cap_lock;
    }
  } else {
    // Fan-out: writer → Reader0 → Reader1 → ... → Reader{N-1} → Cap → writer
    if (dir == AIE::DMAChannelDir::MM2S) {
      // Reader stage `stage` (0..N-1)
      toAcquire = cls.sig_locks[stage];
      toRelease = (stage == cls.n_readers - 1) ? cls.cap_lock
                                               : cls.sig_locks[stage + 1];
    } else {
      // The single writer: acquire cap lock, release first signal lock.
      toAcquire = cls.cap_lock;
      toRelease = cls.sig_locks[0];
    }
  }

  // Map (toAcquire, toRelease) to (pair.first, pair.second) consistent
  // with generateDmaBd's direction-dependent mapping:
  //   S2MM: acquire = pair.second, release = pair.first
  //   MM2S: acquire = pair.first,  release = pair.second
  if (dir == AIE::DMAChannelDir::S2MM)
    return std::make_pair(toRelease, toAcquire);
  return std::make_pair(toAcquire, toRelease);
}

std::pair<int64_t, int64_t>
air::getLockValuePair(const AIE::AIETargetModel &targetModel,
                      Value buffer_memref, air::ChannelOp air_chan) {
  if (!targetModel.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks))
    return std::make_pair(0, 0);

  if (!llvm::isa<BaseMemRefType>(buffer_memref.getType()))
    return std::make_pair(-1, -1);

  if (!air_chan)
    return getLockValuePair(targetModel, buffer_memref);

  // Infer semaphore lock values using air.channel. This method enables
  // ping-pong compute-communication overlap.
  llvm::SmallSet<Operation *, 2> unique_write_buffers;
  llvm::SmallSet<Operation *, 2> unique_read_buffers;
  for (auto get : getChannelGetOpThroughSymbol(air_chan)) {
    if (isa<AIE::ExternalBufferOp>(buffer_memref.getDefiningOp())) {
      // Shim DMA locks
      unique_write_buffers.clear();
      unique_write_buffers.insert(buffer_memref.getDefiningOp());
      break;
    } else if (auto core_op = get->getParentOfType<AIE::CoreOp>()) {
      if (core_op.getTileOp().getResult() ==
          buffer_memref.getDefiningOp()->getOperand(0)) {
        unique_write_buffers.insert(get.getMemref().getDefiningOp());
      }
    }
  }
  for (auto put : getChannelPutOpThroughSymbol(air_chan)) {
    if (isa<AIE::ExternalBufferOp>(buffer_memref.getDefiningOp())) {
      // Shim DMA locks
      unique_read_buffers.clear();
      unique_read_buffers.insert(buffer_memref.getDefiningOp());
      break;
    } else if (auto core_op = put->getParentOfType<AIE::CoreOp>()) {
      if (core_op.getTileOp().getResult() ==
          buffer_memref.getDefiningOp()->getOperand(0)) {
        unique_read_buffers.insert(put.getMemref().getDefiningOp());
      }
    }
  }
  return std::make_pair(unique_read_buffers.size(),
                        unique_write_buffers.size());
}

// Helper function that tries to retrieve the underlying AIE::BufferOp by
// unwrapping common memref wrappers (cast or subview)
AIE::BufferOp getUnderlyingBufferOp(Value buffer) {
  // Case 1: Directly defined by an AIE::BufferOp
  if (auto bufferOp = buffer.getDefiningOp<AIE::BufferOp>())
    return bufferOp;

  // Case 2: Defined by a cast (e.g., memref.cast)
  if (auto castOp = buffer.getDefiningOp<CastOpInterface>())
    if (auto innerBuffer = castOp->getOperand(0).getDefiningOp<AIE::BufferOp>())
      return innerBuffer;

  // Case 3: Defined by a view-like op (e.g., memref.subview)
  if (auto viewLikeOp = buffer.getDefiningOp<ViewLikeOpInterface>())
    if (auto innerBuffer =
            viewLikeOp->getOperand(0).getDefiningOp<AIE::BufferOp>())
      return innerBuffer;

  // No underlying BufferOp found
  return nullptr;
}

// allocation_info_t impl.

bool xilinx::air::allocation_info_t::valid() {
  return dma_tile.getOperation() != nullptr;
}

AIE::TileLike xilinx::air::allocation_info_t::getDmaTile() { return dma_tile; }

bool xilinx::air::allocation_info_t::foundAlloc(air::ChannelOp channel_op) {
  if (channel_op) {
    for (auto o : memcpyOps) {
      if (auto chan_op = dyn_cast_if_present<air::ChannelInterface>(o)) {
        auto chan_declr = getChannelDeclarationThroughSymbol(chan_op);
        if (chan_declr == channel_op)
          return true;
      }
    }
  }
  return false;
}

bool xilinx::air::allocation_info_t::foundAllocInColumn(int32_t col) {
  if (!getDmaTile())
    return false;
  auto tileCol = getDmaTile().tryGetCol();
  return tileCol && *tileCol == col;
}

bool xilinx::air::allocation_info_t::foundAlloc(AIE::DMAChannel channel) {
  if (channel.direction == dma_channel.direction &&
      channel.channel == dma_channel.channel)
    return true;
  else
    return false;
}

bool xilinx::air::allocation_info_t::foundAllocInColumn(
    int32_t col, AIE::DMAChannel channel) {
  return foundAllocInColumn(col) && foundAlloc(channel);
}

bool xilinx::air::allocation_info_t::foundAlloc(AIE::TileLike tile,
                                                AIE::DMAChannel channel) {
  if (tile.getOperation() == getDmaTile().getOperation() && foundAlloc(channel))
    return true;
  else
    return false;
}

// Is there a packet-flow allocation owned by a tile in the given column?
bool xilinx::air::allocation_info_t::foundPacketFlowAllocInColumn(int32_t col) {
  if (!foundAllocInColumn(col))
    return false;
  for (auto o : memcpyOps) {
    auto memcpy_op = dyn_cast_if_present<air::MemcpyInterface>(o);
    if (!memcpy_op)
      continue;
    auto chanTypeRes = air::getChannelType(memcpy_op);
    if (succeeded(chanTypeRes))
      return chanTypeRes.value() == "npu_dma_packet";
  }
  return false;
}

// TileLike-keyed overloads (RFC #1567). Pointer-equality on the underlying
// Operation* of dma_tile replaces (col, row) integer comparison; same answer,
// no dependence on physical placement coordinates. Works for both AIE::TileOp
// and AIE::LogicalTileOp.
bool xilinx::air::allocation_info_t::foundAlloc(AIE::TileLike tile) {
  return tile && tile.getOperation() == getDmaTile().getOperation();
}

bool xilinx::air::allocation_info_t::foundAlloc(AIE::TileLike tile,
                                                air::MemcpyInterface memcpyOp) {
  if (!foundAlloc(tile))
    return false;
  for (auto o : memcpyOps)
    if (memcpyOp.getOperation() == o)
      return true;
  return false;
}

bool xilinx::air::allocation_info_t::foundAlloc(AIE::TileLike tile,
                                                air::ChannelOp channel_op) {
  return foundAlloc(tile) && foundAlloc(channel_op);
}

bool xilinx::air::allocation_info_t::foundPacketFlowAllocInTile(
    AIE::TileLike tile) {
  if (!foundAlloc(tile))
    return false;
  for (auto o : memcpyOps) {
    auto memcpy_op = dyn_cast_if_present<air::MemcpyInterface>(o);
    if (!memcpy_op)
      continue;
    auto chanTypeRes = air::getChannelType(memcpy_op);
    if (succeeded(chanTypeRes))
      return chanTypeRes.value() == "npu_dma_packet";
  }
  return false;
}

// Same-logical-flow test: same channel declaration and same constant bundle
// indices. A non-constant index cannot be proven equal, so it is distinct.
static bool isSameLogicalFlowEndpoint(air::MemcpyInterface a,
                                      air::MemcpyInterface b) {
  auto chanA = dyn_cast_if_present<air::ChannelInterface>(a.getOperation());
  auto chanB = dyn_cast_if_present<air::ChannelInterface>(b.getOperation());
  if (!chanA || !chanB)
    return false;
  if (air::getChannelDeclarationThroughSymbol(chanA) !=
      air::getChannelDeclarationThroughSymbol(chanB))
    return false;
  auto idxA = chanA.getIndices();
  auto idxB = chanB.getIndices();
  if (idxA.size() != idxB.size())
    return false;
  for (auto [va, vb] : llvm::zip_equal(idxA, idxB)) {
    auto ca = getConstantIntValue(va);
    auto cb = getConstantIntValue(vb);
    if (!ca || !cb || *ca != *cb)
      return false;
  }
  return true;
}

bool xilinx::air::allocation_info_t::foundSameLogicalFlowInTile(
    AIE::TileLike tile, air::MemcpyInterface memcpyOp) {
  if (!foundAlloc(tile))
    return false;
  for (auto o : memcpyOps) {
    auto existingMc = dyn_cast_if_present<air::MemcpyInterface>(o);
    if (existingMc && isSameLogicalFlowEndpoint(existingMc, memcpyOp))
      return true;
  }
  return false;
}

bool xilinx::air::allocation_info_t::containsDedicatedChannel() {
  for (auto o : memcpyOps) {
    auto existingMc = dyn_cast_if_present<air::MemcpyInterface>(o);
    if (existingMc && memcpyIsDedicatedChannel(existingMc))
      return true;
  }
  return false;
}

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
      auto a_j = dyn_cast_if_present<air::MemcpyInterface>(a[j]);
      auto a_min = dyn_cast_if_present<air::MemcpyInterface>(a[min]);
      if (a_j.getId() < a_min.getId())
        min = j;
    }
    swap(a, min, i);
  }
}

} // namespace xilinx

namespace xilinx {

FailureOr<air::allocation_info_t>
air::DMAAllocator::lookupDMAAllocation(AIE::TileLike tile,
                                       air::MemcpyInterface &memcpyOp) {

  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile, memcpyOp))
      return t;
  }
  return memcpyOp.emitOpError(
      "failed to look up a DMA allocation. This potentially "
      "indicates a failure in the compilation flow.");
}

// Allocate a reader/writer lock pair. These may be the same or different
// locks depending on the target device.
FailureOr<std::pair<AIE::LockOp, AIE::LockOp>> air::DMAAllocator::getLockForDMA(
    air::MemcpyInterface &memcpyOp, AIE::TileLike tile, Operation *bufferOp,
    bool lockRaceConditionFix, bool lockRaceConditionFixV2) {
  auto alloc = lookupDMAAllocation(tile, memcpyOp);
  if (failed(alloc))
    return memcpyOp->emitOpError("failed to look up dma allocation.");
  AIE::DMAChannel channel = alloc.value().dma_channel;
  // Tile-type predicates derived from TileLike (works for placed and unplaced
  // tiles alike). Avoids depending on physical (col, row) coordinates.
  bool tileIsMemTile = tile.isMemTile();
  air::ChannelOp air_chan = nullptr;
  if (auto air_chan_op =
          dyn_cast_if_present<air::ChannelInterface>(memcpyOp.getOperation())) {
    air_chan = getChannelDeclarationThroughSymbol(air_chan_op);
  }
  const auto &target_model = device.getTargetModel();
  bool UsesSemaphoreLocks =
      target_model.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks);

  // 3-way shared L1 branch (compute tile only). allocateSharedL1BufferLocks
  // stamped this channel put/get op with prod/cons lock symbol-refs because
  // its buffer is shared across N writer cores AND this DMA participant.
  // Resolve and reuse that exact pair so all participants synchronize on the
  // same locks, instead of allocating a private channel-put pair (which would
  // silently break the 3-way share). Keyed on the op's attribute (only present
  // on genuine 3-way ops), so it takes precedence for those ops and leaves all
  // other callers unaffected.
  if (!tileIsMemTile && UsesSemaphoreLocks) {
    Operation *op = memcpyOp.getOperation();
    auto prodRef = op->getAttrOfType<FlatSymbolRefAttr>("air.shared_prod_lock");
    auto consRef = op->getAttrOfType<FlatSymbolRefAttr>("air.shared_cons_lock");
    if (prodRef && consRef) {
      auto prodLock = dyn_cast_or_null<AIE::LockOp>(
          SymbolTable::lookupSymbolIn(device, prodRef.getAttr()));
      auto consLock = dyn_cast_or_null<AIE::LockOp>(
          SymbolTable::lookupSymbolIn(device, consRef.getAttr()));
      if (prodLock && consLock) {
        // Return (cons, prod). generateDmaBd selects
        // acq = isMM2S ? first : second, rel = isMM2S ? second : first.
        // So MM2S reader acquires cons / releases prod; S2MM writer
        // acquires prod / releases cons. Both directions are correct with
        // this single ordering.
        std::pair<AIE::LockOp, AIE::LockOp> pair = {consLock, prodLock};
        lock_allocation_list.push_back(
            {bufferOp, air_chan, channel, pair.first, pair.second});
        return pair;
      }
    }
  }

  // v2: chain-lock branch. Memtile-only; requires the buffer to be a
  // shared L2 with fan-in or fan-out shape (predicate is shape-based).
  // Takes precedence over the legacy / v1 paths when it applies.
  if (lockRaceConditionFixV2 && tileIsMemTile && UsesSemaphoreLocks) {
    auto buf = dyn_cast_or_null<AIE::BufferOp>(bufferOp);
    if (buf && isChainLockCandidate(buf)) {
      auto clsOrFail = getOrCreateChainLockSet(buf, tile);
      if (failed(clsOrFail))
        return memcpyOp->emitOpError(
            "v2 chain-lock: failed to allocate chain lock set");
      ChainLockSet *cls = clsOrFail.value();
      int stage = computeStageIndexForMemcpyOp(memcpyOp.getOperation(), buf);
      if (stage < 0)
        return memcpyOp->emitOpError(
            "v2 chain-lock: failed to determine BD stage index");
      auto pair = pickChainBdLocks(*cls, channel.direction, stage);
      // Register a lock_allocation_list entry so subsequent reuse-lookup
      // queries on the same (buffer, channel) find the same pair —
      // matches the legacy reuse model for the rare same-channel multi-BD
      // case. Note: the chain-lock branch returns DIFFERENT pairs for
      // different memcpy ops on the same buffer (one per stage), so the
      // legacy "same buffer → same lock" reuse logic in the loop below
      // does NOT apply for v2; we just record this BD's pair.
      lock_allocation_list.push_back(
          {bufferOp, air_chan, channel, pair.first, pair.second});
      return pair;
    }
    // Fall through to the legacy path for non-candidate buffers (e.g.
    // 1:1 single-writer single-reader L2 buffers, or MIMO buffers).
  }

  if (UsesSemaphoreLocks) {
    if (air_chan) {
      // AIE2's semaphore locks may share by air.channels
      for (size_t i = 0; i < lock_allocation_list.size(); i++) {
        if (tileIsMemTile) {
          if (!lockRaceConditionFix) {
            // If memtile, and multiple bds reference the same buffer op, but
            // different DMA channels, then we assume the scenario of having two
            // bds, one S2MM and the other MM2S. This scenario is almost always
            // true due to memtile having no core to communicate data with.
            if (std::get<0>(lock_allocation_list[i]) == bufferOp) {
              return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                    std::get<4>(lock_allocation_list[i]));
            }
          } else {
            // Determine the opposite direction of the given DMA channel.
            // MM2S (Memory-to-Stream) ↔ S2MM (Stream-to-Memory)
            AIE::DMAChannelDir oppo_channel_dir =
                channel.direction == AIE::DMAChannelDir::MM2S
                    ? AIE::DMAChannelDir::S2MM
                    : AIE::DMAChannelDir::MM2S;
            // Case 1: Exact match on (channel symbol, physical channel number).
            if (air_chan &&
                (std::get<1>(lock_allocation_list[i]) == air_chan) &&
                (std::get<2>(lock_allocation_list[i]) == channel)) {
              // Reuse the existing lock entry by appending a new BD with the
              // same locks.
              auto entry =
                  std::make_tuple(bufferOp, air_chan, channel,
                                  std::get<3>(lock_allocation_list[i]),
                                  std::get<4>(lock_allocation_list[i]));
              lock_allocation_list.push_back(entry);
              // Return the (acquire, release) lock pair for this op.
              return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                    std::get<4>(lock_allocation_list[i]));
            }
            // Case 2: Passive-direction DMA op on same buffer (i.e. the
            // direction that may come with dummy channels).
            else if ((std::get<0>(lock_allocation_list[i]) == bufferOp) &&
                     (std::get<2>(lock_allocation_list[i]).direction ==
                      oppo_channel_dir)) {
              // First time we see this on the passive side
              if (!passiveSideBufferUseCounters.count(bufferOp->getResult(0))) {
                passiveSideBufferUseCounters[bufferOp->getResult(0)] =
                    std::make_pair(1, 0); // (activeCount, passiveCount)
                return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                      std::get<4>(lock_allocation_list[i]));
              }
              // All previous passive users have matched active counts (balanced
              // so far)
              else if (passiveSideBufferUseCounters[bufferOp->getResult(0)]
                           .first ==
                       passiveSideBufferUseCounters[bufferOp->getResult(0)]
                           .second) {
                passiveSideBufferUseCounters[bufferOp->getResult(0)].first++;
                passiveSideBufferUseCounters[bufferOp->getResult(0)].second = 0;
                return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                      std::get<4>(lock_allocation_list[i]));
              } else {
                // Still have unmatched passive users — increment passive side
                // count
                passiveSideBufferUseCounters[bufferOp->getResult(0)].second++;
                continue; // Try next entry in lock_allocation_list
              }
            }
          }
        } else if ((std::get<1>(lock_allocation_list[i]) == air_chan) &&
                   (std::get<0>(lock_allocation_list[i])->getOperand(0) ==
                    bufferOp->getOperand(0)) &&
                   (std::get<2>(lock_allocation_list[i]) == channel)) {
          return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                std::get<4>(lock_allocation_list[i]));
        } else if ((std::get<0>(lock_allocation_list[i]) == bufferOp) &&
                   (std::get<2>(lock_allocation_list[i]) == channel)) {
          // Same physical buffer and same DMA channel but different
          // air.channel symbols. This handles multiple outbound puts sharing
          // a staging buffer (e.g., K and V writeback through the same L1
          // buffer).
          return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                std::get<4>(lock_allocation_list[i]));
        }
      }
    } else {
      for (size_t i = 0; i < lock_allocation_list.size(); i++) {
        if ((std::get<0>(lock_allocation_list[i]) == bufferOp) &&
            (std::get<2>(lock_allocation_list[i]) == channel)) {
          return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                std::get<4>(lock_allocation_list[i]));
        }
        // Else if memtile, and multiple bds reference the same buffer, but
        // different DMA channels, then we assume the scenario of having two
        // bds, one S2MM and the other MM2S. This scenario is almost always true
        // due to memtile having no core to communicate data with.
        else if (tileIsMemTile &&
                 std::get<0>(lock_allocation_list[i]) == bufferOp) {
          return std::make_pair(std::get<3>(lock_allocation_list[i]),
                                std::get<4>(lock_allocation_list[i]));
        }
      }
    }
  } else {
    for (size_t i = 0; i < lock_allocation_list.size(); i++) {
      // If multiple bds reference the same buffer and DMA channel
      if ((std::get<0>(lock_allocation_list[i]) == bufferOp) &&
          (std::get<2>(lock_allocation_list[i]) == channel)) {
        return std::make_pair(std::get<3>(lock_allocation_list[i]),
                              std::get<4>(lock_allocation_list[i]));
      }
    }
  }
  if (!bufferOp) {
    return memcpyOp->emitOpError(
        "failed to materialize src/dst memref into AIE.BufferOp.");
  }
  std::pair<int64_t, int64_t> init_pair;
  if (tileIsMemTile)
    init_pair = getLockValuePair(target_model, bufferOp->getResult(0));
  else
    init_pair =
        getLockValuePair(target_model, bufferOp->getResult(0), air_chan);
  auto init = std::max(init_pair.first, init_pair.second);

  // Consumerless memtile drain: emit ONE self-recycling lock (returned as both
  // pair elements) so the receiving BD acquires AND releases the same lock and
  // re-arms every dispatch. A distinct producer/consumer pair would deadlock on
  // the 2nd dispatch (no consumer to release the producer lock). generateDmaBd
  // emits acquire(self,1)/release(self,1) automatically when both pair elements
  // are the same lock. Scoped by nR==0 -> mutually exclusive with the
  // chain-lock / shared-L1 / producer->consumer paths above.
  if (UsesSemaphoreLocks && tileIsMemTile &&
      isConsumerlessMemtileDrain(dyn_cast_or_null<AIE::BufferOp>(bufferOp))) {
    auto selfLock = allocateLockOp(
        device, tile, static_cast<int>(std::max<int64_t>(init, 1)));
    lock_allocation_list.push_back(
        {bufferOp, air_chan, channel, selfLock, selfLock});
    return std::make_pair(selfLock, selfLock);
  }

  OpBuilder builder(bufferOp);
  auto rlock = allocateLockOp(device, tile, 0);
  // air.refeed_count=N (single-buffer count-free re-broadcast): the fill (S2MM)
  // does AcquireGreaterEqual N on the empty/write lock so that ONE fill enables
  // N count-free MM2S re-broadcasts (generateDmaBd sets acq/rel = N for the
  // fill BD when the buffer carries air.refeed_count). The write lock must
  // therefore INIT to N -- with the default slot-count init the first
  // AcquireGreaterEqual N can never fire and the producer feeding the buffer
  // deadlocks.
  // getRefeedCount guarantees 1 <= count <= INT32_MAX, so this init fits the
  // 32-bit lock without truncation.
  int64_t wlockInit = init;
  if (UsesSemaphoreLocks)
    wlockInit = std::max(wlockInit, air::getRefeedCount(bufferOp));
  auto wlock = UsesSemaphoreLocks
                   ? allocateLockOp(device, tile, static_cast<int>(wlockInit))
                   : rlock;
  lock_allocation_list.push_back({bufferOp, air_chan, channel, rlock, wlock});
  return std::make_pair(rlock, wlock);
}

// Allocate a new DMA channel
FailureOr<air::allocation_info_t> air::DMAAllocator::allocNewDmaChannel(
    air::MemcpyInterface &memcpyOp, AIE::TileLike tile, int chan, int col = -1,
    int row = -1, std::vector<int> dma_id = {}) {
  if (!tile) {
    return memcpyOp.emitOpError("failed to get the AIE tile. This indicates a "
                                "potential error in the compilation flow.");
  }
  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;
  AIE::DMAChannel aie_chan;
  aie_chan.direction =
      isMM2S.value() ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;
  aie_chan.channel = chan;
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile)) {
      if (t.dma_channel.direction == aie_chan.direction &&
          t.dma_channel.channel == aie_chan.channel) {
        t.memcpyOps.push_back(memcpyOp.getOperation());
        return t;
      }
    }
    if (t.foundAlloc(tile, getChannelDeclarationThroughSymbol(
                               dyn_cast_if_present<air::ChannelInterface>(
                                   memcpyOp.getOperation())))) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }
  air::allocation_info_t output = {tile,
                                   col,
                                   row,
                                   aie_chan,
                                   chan,
                                   /*packet_flow_id=*/-1,
                                   /*otherSideLTO=*/nullptr,
                                   dma_id,
                                   {memcpyOp.getOperation()}};
  allocs->push_back(output);
  return output;
}

// Sort all ops being allocated to each DMA channel (based on id which indicates
// op sequence), to avoid ping-pong deadlock.
void air::DMAAllocator::sortMemcpyOps(std::vector<Operation *> dma_memcpy_ops) {
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
FailureOr<air::allocation_info_t>
air::TileDMAAllocator::simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp,
                                             AIE::TileOp tile, int chan) {
  if (!tile) {
    return memcpyOp.emitOpError(
        "failed to get a tile. This indicates a potential compilation "
        "failure.");
  }
  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

  // Check if allocating for a packet flow (packet flow supports channel time
  // multiplexing)
  bool isPacketFlowOp = false;
  auto chanTypeRes = getChannelType(memcpyOp);
  if (succeeded(chanTypeRes)) {
    isPacketFlowOp = chanTypeRes.value() == "npu_dma_packet";
  }

  // Compute-tile DMA channel pin: a channel decl carrying an
  // `air.tile_dma_channel` IntegerAttr forces this flow onto that physical DMA
  // channel index (the compute-tile analogue of the memtile
  // `air.memtile_dma_channel_min` floor). Used when two flows on the same tile
  // must keep fixed, distinct physical channels because their routes would
  // otherwise collide. The pin is an explicit override: apply it even when a
  // channel index was already chosen by the flow-level allocation phase
  // (callers may pass a pre-set `chan`), otherwise convergent channels whose
  // channel is fixed before this call would ignore the pin.
  bool chanPinned = false;
  if (auto chanIf =
          dyn_cast_if_present<air::ChannelInterface>(memcpyOp.getOperation()))
    if (auto chanDecl = getChannelDeclarationThroughSymbol(chanIf))
      if (auto a = chanDecl->getAttrOfType<mlir::IntegerAttr>(
              air::attrs::TileDmaChannel)) {
        int pinned = (int)a.getInt();
        // Validate against the tile's available DMA channels for this
        // direction; an out-of-range pin would otherwise create an invalid
        // allocation.
        int numChans = isMM2S.value()
                           ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
                           : tile.getNumDestConnections(AIE::WireBundle::DMA);
        if (pinned < 0 || pinned >= numChans)
          return memcpyOp.emitOpError("air.tile_dma_channel = ")
                 << pinned << " is out of range [0, " << numChans
                 << ") for the " << (isMM2S.value() ? "MM2S" : "S2MM")
                 << " DMA channels of tile (" << tile.getCol() << ", "
                 << tile.getRow() << ")";
        chan = pinned;
        chanPinned = true;
      }

  // Search for existing dma channel allocation
  unsigned num_allocs = 0;
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile))
      num_allocs++;
    if (t.foundAlloc(tile, memcpyOp))
      return t;
    if (t.foundAlloc(tile,
                     AIE::DMAChannel{isMM2S.value() ? AIE::DMAChannelDir::MM2S
                                                    : AIE::DMAChannelDir::S2MM,
                                     chan})) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
    // Search for existing packet-flow allocations on this tile, and try to
    // reuse the channel allocation. Skipped when this flow is channel-pinned:
    // the pin dictates the physical channel, and same-channel reuse is already
    // handled above -- cross-channel packet reuse would silently override the
    // pin.
    if (isPacketFlowOp && !chanPinned && t.foundPacketFlowAllocInTile(tile)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }
  // Need to allocate a new one
  int tile_dma_channels =
      isMM2S.value() ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
                     : tile.getNumDestConnections(AIE::WireBundle::DMA);
  if (chan == -1)
    chan = num_allocs % tile_dma_channels;
  return air::DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

// Two transfers occupy the same slot of the emitted ring if the emitter would
// fold them onto one BD. That is either strict BD equivalence, or -- because
// getRepeatCounts also folds N-buffer rotations -- two slots of one rotation.
static bool sameBDSlot(Operation *a, Operation *b) {
  if (air::memcpyIMappedToEquivalentBDs(a, b))
    return true;
  auto ca = dyn_cast_if_present<air::ChannelInterface>(a);
  auto cb = dyn_cast_if_present<air::ChannelInterface>(b);
  return ca && cb && air::chansPartOfSameRotation(ca, cb);
}

// The conditional-arm path a chain op sits on, as the list of enclosing
// branching ops paired with the region index taken to reach the op. Two ops
// are on mutually exclusive paths iff their keys agree on some branching op
// but disagree on its region index.
static SmallVector<std::pair<Operation *, unsigned>>
getCondPathKey(Operation *op, Operation *stopAt) {
  SmallVector<std::pair<Operation *, unsigned>> key;
  Region *r = op->getParentRegion();
  while (r && r->getParentOp() && r->getParentOp() != stopAt) {
    Operation *parent = r->getParentOp();
    if (isa<scf::IfOp, scf::IndexSwitchOp>(parent))
      key.push_back({parent, r->getRegionNumber()});
    r = parent->getParentRegion();
  }
  std::reverse(key.begin(), key.end());
  return key;
}

// A BD ring is walked strictly in order; a packet header routes a transfer to
// the tile but does not select which BD receives it. So every feasible control
// path through the consumer must present the SAME sequence of BDs -- otherwise
// one path delivers a different number (or a different order) of transfers
// than the ring was built for, the BD pointer slips, and a transfer eventually
// meets a BD belonging to another flow.
//
// Checked per branching op: for each region of that op, project the chain onto
// the transfers reachable through it. Regions contributing no transfer give an
// empty word, which is the "hole" case (one arm forgets to consume a flow the
// other arm does).
//
// `emptyPathsOk` drops the hole case, which is the difference between the two
// directions. A CONSUMER that skips an arrival still receives it -- the packet
// arrives, lands on whatever BD the pointer sits on, and the ring is out of
// step from then on. A PRODUCER that takes an arm issuing nothing simply does
// not advance the ring: it is where it was, still aligned, provided the arms
// that do issue each cover a whole cycle -- which the equivalence check on the
// remaining non-empty words is what establishes.
static bool allPathWordsEquivalent(ArrayRef<Operation *> ops, Operation *stopAt,
                                   bool emptyPathsOk = false) {
  llvm::SetVector<Operation *> branchOps;
  llvm::DenseMap<Operation *, SmallVector<std::pair<Operation *, unsigned>>>
      keys;
  for (auto *o : ops) {
    keys[o] = getCondPathKey(o, stopAt);
    for (auto &[brOp, regionIdx] : keys[o])
      branchOps.insert(brOp);
  }

  for (auto *brOp : branchOps) {
    // Word per region of this branching op, in chain order.
    SmallVector<SmallVector<Operation *>> words(brOp->getNumRegions());
    for (auto *o : ops)
      for (auto &[k, regionIdx] : keys[o])
        if (k == brOp)
          words[regionIdx].push_back(o);

    // Every region of an scf.if / scf.index_switch is reachable, so all of
    // them must match -- including an absent else, whose empty word is the
    // hole case (one path forgets to consume a flow the other one does).
    SmallVector<Operation *> *ref = nullptr;
    for (auto [i, w] : llvm::enumerate(words)) {
      if (emptyPathsOk && w.empty())
        continue;
      if (!ref) {
        ref = &words[i];
        continue;
      }
      if (w.size() != ref->size())
        return false;
      for (auto [a, b] : llvm::zip_equal(w, *ref))
        if (!sameBDSlot(a, b))
          return false;
    }
  }
  return true;
}

// Why a chain's BD ring cannot stay in step with the transfers crossing it, or
// an empty string when it can. Direction-neutral: the ring walks in order and
// the header does not select a BD, so the same reasoning bounds both the S2MM
// arrivals and the MM2S departures.
static std::string diagnoseBDChain(ArrayRef<Operation *> ops, Operation *stopAt,
                                   bool emptyPathsOk = false) {
  if (ops.size() <= 1)
    return "";

  llvm::SetVector<Operation *> opSet;
  for (auto *o : ops)
    opSet.insert(o);

  // Homogeneous ring: every BD is interchangeable, so no arrival can land on a
  // BD meant for something else no matter where the pointer is.
  if (air::getUniqueBDPattern(opSet).size() == 1)
    return "";

  // Note the emitter's BD-task count is deliberately NOT a criterion: several
  // repeat-count buckets lower to a sequence of finite tasks that is re-armed
  // per dispatch, so the chain still covers exactly one round of arrivals.
  // Drift comes only from paths that deliver different rounds.
  if (!allPathWordsEquivalent(ops, stopAt, emptyPathsOk))
    return "control-flow paths deliver different BD sequences: the ring was "
           "built for one path's transfers and will slip on the others";

  // Not checked: that independent producers cannot reorder the arrivals
  // relative to the consumer's program order. That needs cross-herd ordering
  // the pass does not have, so convergent flows are trusted to be time-
  // disjoint.
  return "";
}

void air::TileDMAAllocator::repairS2MMChains(
    std::vector<MemcpyBundleAsFlow> &memcpy_flows) {
  // One BD chain is built per (tile, channel) from the concatenation of every
  // allocation mapped to it, in s2mm_allocs order -- mirror that grouping
  // exactly (see the tile_dma_memcpys construction in AIRToAIEPass.cpp), since
  // allocations that each fold in isolation can still concatenate into a chain
  // that does not.
  llvm::MapVector<std::pair<Operation *, int>, SmallVector<size_t>> chains;
  llvm::DenseMap<Operation *, llvm::SmallDenseSet<int>> usedChansPerTile;
  for (auto [i, alloc] : llvm::enumerate(s2mm_allocs))
    if (alloc.dma_tile) {
      Operation *t = alloc.dma_tile.getOperation();
      chains[{t, alloc.dma_channel.channel}].push_back(i);
      usedChansPerTile[t].insert(alloc.dma_channel.channel);
    }

  auto declOf = [](Operation *o) -> Operation * {
    auto chan = dyn_cast_if_present<air::ChannelInterface>(o);
    if (!chan)
      return nullptr;
    auto decl = air::getChannelDeclarationThroughSymbol(chan);
    return decl ? decl.getOperation() : nullptr;
  };

  for (auto &[key, allocIdxs] : chains) {
    auto tile = s2mm_allocs[allocIdxs.front()].dma_tile;
    std::vector<Operation *> ops;
    for (size_t i : allocIdxs)
      llvm::append_range(ops, s2mm_allocs[i].memcpyOps);
    if (ops.size() <= 1)
      continue;

    // Only a chain hosting more than one logical flow can mis-deliver ACROSS
    // flows, which is the deadlock at issue. A single-flow chain that slips
    // writes its own slices out of order -- a different matter, and where it
    // happens today, deliberate.
    llvm::SetVector<Operation *> decls;
    for (auto *o : ops) {
      auto *d = declOf(o);
      if (!d) {
        decls.clear(); // Unkeyed transfer: not attributable to a flow.
        break;
      }
      decls.insert(d);
    }
    if (decls.size() < 2)
      continue;

    // Bound the control-flow walk at the core body: transfers in different
    // herds are different consumers and share no BD ring.
    Operation *stopAt = ops.front()->getParentOfType<air::HerdOp>();
    std::string why = diagnoseBDChain(ops, stopAt);
    if (why.empty())
      continue;

    // Peel the one flow whose removal leaves BOTH halves in step. A flow that
    // was pinned or marked dedicated already states where it belongs, so it is
    // not ours to move.
    Operation *peelDecl = nullptr;
    std::vector<Operation *> moved, rest;
    for (auto *d : decls) {
      if (d->hasAttr(air::attrs::TileDmaChannel) ||
          d->hasAttr(air::attrs::DedicatedDmaChannel))
        continue;
      std::vector<Operation *> m, r;
      for (auto *o : ops)
        (declOf(o) == d ? m : r).push_back(o);
      if (m.empty() || r.empty())
        continue;
      // A half carrying one flow is out of scope for the same reason the whole
      // chain would be: it has no other flow's BD to land on.
      auto halfInStep = [&](ArrayRef<Operation *> half) {
        llvm::SetVector<Operation *> hd;
        for (auto *o : half)
          hd.insert(declOf(o));
        return hd.size() < 2 || diagnoseBDChain(half, stopAt).empty();
      };
      if (!halfInStep(m) || !halfInStep(r))
        continue;
      peelDecl = d;
      moved = m;
      rest = r;
      break;
    }

    auto &used = usedChansPerTile[key.first];
    int numChans = tile.getNumDestConnections(AIE::WireBundle::DMA);
    int freeChan = -1;
    for (int c = 0; c < numChans; c++)
      if (!used.count(c)) {
        freeChan = c;
        break;
      }

    if (!peelDecl || freeChan < 0) {
      // Nothing to move, or nowhere to move it. Emitting a design that is
      // expected to hang is worth saying out loud.
      auto diag = ops.front()->emitWarning()
                  << "compute-tile S2MM channel " << key.second
                  << " multiplexes " << decls.size() << " flows over "
                  << ops.size() << " transfers, but " << why
                  << ". The receiver is expected to deadlock after the first "
                     "dispatch";
      if (!peelDecl)
        diag << ", and no single flow can be peeled off to fix it. Equalize "
                "the paths so each delivers the same transfers";
      else
        diag << ", and the tile has no spare S2MM channel to move @"
             << cast<air::ChannelOp>(peelDecl).getSymName()
             << " onto. Equalize the paths so each delivers the same "
                "transfers";
      for (auto *d : decls)
        diag.attachNote(d->getLoc()) << "flow on this chain";
      continue;
    }

    // Retarget an allocation whose transfers all move; split the rest.
    SmallVector<allocation_info_t> splits;
    for (size_t i : allocIdxs) {
      std::vector<Operation *> keepOps, moveOps;
      for (auto *o : s2mm_allocs[i].memcpyOps)
        (declOf(o) == peelDecl ? moveOps : keepOps).push_back(o);
      if (moveOps.empty())
        continue;
      s2mm_allocs[i].packet_flow_id = -1; // reassigned on flow connection
      if (keepOps.empty()) {
        s2mm_allocs[i].dma_channel = {AIE::DMAChannelDir::S2MM, freeChan};
        s2mm_allocs[i].tile_channel = freeChan;
        continue;
      }
      allocation_info_t split = s2mm_allocs[i];
      split.dma_channel = {AIE::DMAChannelDir::S2MM, freeChan};
      split.tile_channel = freeChan;
      split.memcpyOps = moveOps;
      s2mm_allocs[i].memcpyOps = keepOps;
      splits.push_back(split);
    }
    llvm::append_range(s2mm_allocs, splits);
    used.insert(freeChan);

    // The flows are connected from the bundle's own copy of the allocation, so
    // leaving it behind would route the packet to the channel the BDs just
    // left.
    for (auto &f : memcpy_flows) {
      if (f.air_flow_op != peelDecl)
        continue;
      for (auto &fa : f.S2MM_alloc) {
        if (!fa.dma_tile || fa.dma_tile.getOperation() != key.first)
          continue;
        if (fa.dma_channel.direction != AIE::DMAChannelDir::S2MM ||
            fa.dma_channel.channel != key.second)
          continue;
        fa.dma_channel.channel = freeChan;
        fa.tile_channel = freeChan;
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "repaired S2MM chain on tile " << tile.getOperation()
               << ": moved " << moved.size() << " of " << ops.size()
               << " transfers to channel " << freeChan << "\n");
  }
}

void air::TileDMAAllocator::spreadCollapsedPacketChannels(
    std::vector<MemcpyBundleAsFlow> &memcpy_flows) {
  auto declOf = [](Operation *o) -> Operation * {
    auto chan = dyn_cast_if_present<air::ChannelInterface>(o);
    if (!chan)
      return nullptr;
    auto decl = air::getChannelDeclarationThroughSymbol(chan);
    return decl ? decl.getOperation() : nullptr;
  };
  // A decl the front end has already placed on this tile. Unlike at the shim,
  // `air.tile_dma_channel` IS about this tile's channel, so it is honoured here
  // as the explicit override it is documented to be. `broadcast_shape` is not
  // consulted: a broadcast constrains the SOURCE port it fans out from, which
  // says nothing about which channel each receiving core takes it in on.
  auto isImmovable = [](Operation *decl) {
    return decl->hasAttr(air::attrs::TileDmaChannel) ||
           decl->hasAttr(air::attrs::DedicatedDmaChannel);
  };
  // Whether the flow this transfer belongs to travels as packets. Read from the
  // memcpy op, the same way simpleDmaChannelAlloc decides whether to multiplex,
  // so the two agree on what a packet flow is even after the pass has converted
  // a circuit channel into one.
  auto isPacket = [](Operation *o) {
    auto mc = dyn_cast_if_present<air::MemcpyInterface>(o);
    if (!mc)
      return false;
    auto ct = air::getChannelType(mc);
    return succeeded(ct) && *ct == "npu_dma_packet";
  };

  // The tiles a flow is produced by. Taken from the bundle: an allocation only
  // knows its own side of the flow.
  auto sourceTilesOf = [&memcpy_flows](Operation *decl) {
    llvm::SmallPtrSet<Operation *, 4> tiles;
    for (auto &f : memcpy_flows) {
      if (f.air_flow_op != decl)
        continue;
      for (auto &fa : f.MM2S_alloc)
        if (fa.dma_tile)
          tiles.insert(fa.dma_tile.getOperation());
    }
    return tiles;
  };

  // What KIND of thing produces a flow: the shim, or somewhere on the array.
  // This is the grain the arrival-order question is really asked at. Shim
  // traffic is DDR-backed and its timing depends on the host and the NoC, so
  // nothing on the array bounds when it turns up; two on-array producers are
  // driven by the same lock protocol as the consumer. So a ring may hold
  // several on-array producers and still behave, while mixing shim traffic with
  // on-array traffic on one ring is where the order actually comes apart. It is
  // also the split the qwen2.5-3b pins encode by hand: {rmsIn, rmsW} from the
  // host on one channel, {orms, outY} from two memtiles on the other.
  auto producedOffArray = [&](Operation *decl) {
    for (auto *t : sourceTilesOf(decl)) {
      if (auto lt = dyn_cast<AIE::LogicalTileOp>(t))
        return lt.getTileType() == AIE::AIETileType::ShimNOCTile;
      if (auto pt = dyn_cast<AIE::TileOp>(t))
        return pt.getRow() == 0;
    }
    return false;
  };

  // One chain as the emitter sees it: every allocation mapped to a
  // (tile, channel), the flows on it in a deterministic order, and which of
  // those flows are packet flows.
  struct Chain {
    SmallVector<size_t> allocIdxs;
    SmallVector<Operation *> order;
    llvm::SmallPtrSet<Operation *, 4> packetDecls;
    std::vector<Operation *> ops;
    bool unkeyed = false;
  };

  for (auto *allocs : {&mm2s_allocs, &s2mm_allocs}) {
    if (allocs->empty())
      continue;
    AIE::DMAChannelDir dir = allocs->front().dma_channel.direction;
    bool isMM2S = dir == AIE::DMAChannelDir::MM2S;

    // Where each allocation sat before the pass ran. A split has to be APPENDED
    // when it is made -- inserting into the vector mid-pass would invalidate
    // the chain indices still to be visited -- but appending also reorders the
    // allocations, and this vector is the order flows are later created in.
    // The pathfinder is greedy, so that order decides routes: llama-3.2-1b
    // routes under one order of the rope tile's two packet appends and fails to
    // route under the other, on identical channels. Changing which channel a
    // flow uses is this pass's business; changing the order they are emitted in
    // is not. So a split remembers the position of the allocation it came from,
    // and the vector is put back in that order at the end, each split sitting
    // immediately after its parent.
    SmallVector<size_t> origPos(allocs->size());
    std::iota(origPos.begin(), origPos.end(), 0);

    // Chains are keyed exactly as the emitter groups them: one per
    // (tile, channel), over every allocation mapped to it. Rebuilt between
    // phases, since moving a flow changes the grouping.
    auto buildChains = [&]() {
      llvm::MapVector<std::pair<Operation *, int>, Chain> chains;
      for (auto [i, alloc] : llvm::enumerate(*allocs)) {
        if (!alloc.dma_tile)
          continue;
        auto &c =
            chains[{alloc.dma_tile.getOperation(), alloc.dma_channel.channel}];
        c.allocIdxs.push_back(i);
        for (auto *o : alloc.memcpyOps) {
          c.ops.push_back(o);
          auto *d = declOf(o);
          if (!d) {
            c.unkeyed = true; // not attributable to a flow: leave it alone
            continue;
          }
          if (!llvm::is_contained(c.order, d))
            c.order.push_back(d);
          if (isPacket(o))
            c.packetDecls.insert(d);
        }
      }
      return chains;
    };
    auto channelsInUse = [&](Operation *tileOp) {
      llvm::SmallDenseSet<int> used;
      for (auto &alloc : *allocs)
        if (alloc.dma_tile && alloc.dma_tile.getOperation() == tileOp)
          used.insert(alloc.dma_channel.channel);
      return used;
    };

    // (tile, channel) pairs this pass has written to, so program order can be
    // restored on exactly those and nothing else.
    llvm::SetVector<std::pair<Operation *, int>> touched;

    // Move every transfer of `moveDecl` currently on (tileOp, fromChan) to
    // `toChan`: retarget an allocation whose transfers all move, split the
    // rest, and follow with the bundle's own copy -- the flows are connected
    // from that copy, so leaving it behind would route the packet to the
    // channel the BDs just left.
    auto moveFlow = [&](ArrayRef<size_t> allocIdxs, Operation *moveDecl,
                        Operation *tileOp, int fromChan, int toChan) {
      touched.insert({tileOp, fromChan});
      touched.insert({tileOp, toChan});
      SmallVector<std::pair<size_t, allocation_info_t>> splits;
      for (size_t i : allocIdxs) {
        std::vector<Operation *> keepOps, moveOps;
        for (auto *o : (*allocs)[i].memcpyOps)
          (declOf(o) == moveDecl ? moveOps : keepOps).push_back(o);
        if (moveOps.empty())
          continue;
        (*allocs)[i].packet_flow_id = -1; // reassigned on flow connection
        if (keepOps.empty()) {
          (*allocs)[i].dma_channel = {dir, toChan};
          (*allocs)[i].tile_channel = toChan;
          continue;
        }
        allocation_info_t split = (*allocs)[i];
        split.dma_channel = {dir, toChan};
        split.tile_channel = toChan;
        split.memcpyOps = moveOps;
        (*allocs)[i].memcpyOps = keepOps;
        splits.push_back({origPos[i], split});
      }
      for (auto &[parentPos, split] : splits) {
        allocs->push_back(split);
        origPos.push_back(parentPos);
      }

      for (auto &f : memcpy_flows) {
        if (f.air_flow_op != moveDecl)
          continue;
        for (auto *side : {&f.MM2S_alloc, &f.S2MM_alloc})
          for (auto &fa : *side) {
            if (!fa.dma_tile || fa.dma_tile.getOperation() != tileOp)
              continue;
            if (fa.dma_channel.direction != dir ||
                fa.dma_channel.channel != fromChan)
              continue;
            fa.dma_channel.channel = toChan;
            fa.tile_channel = toChan;
            fa.packet_flow_id = -1;
          }
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "TileDMAAllocator::spreadCollapsedPacketChannels: moved @"
                 << cast<air::ChannelOp>(moveDecl).getSymName() << " to "
                 << (isMM2S ? "MM2S" : "S2MM") << " " << toChan << "\n");
    };

    // Is sharing this ring a hazard, or merely a choice? Collapse is not free
    // to undo -- several flows on one channel is how the emitter folds a
    // repeating chain into a single BD with a repeat count -- so it is only
    // worth breaking where the sharing is unsound. Two things make it so, and
    // both are decided by rules already in this file rather than guessed at
    // here.
    //
    // Deliberately NOT a criterion: that the flows come from independent
    // producers. diagnoseBDChain names it as the thing it does not check, and
    // it is what some of the remaining hand-written pins guard, but "two source
    // tiles" does not imply "unordered": every AIR matmul feeds a core one A
    // tile and one B tile per iteration from two memtiles, onto a ring that
    // repeats in lockstep with them. Splitting on source count alone was tried
    // and spends both S2MM channels of every compute tile in the project.
    // Does the ring visit each of its flows exactly once per cycle? That is the
    // condition under which a shared in-order ring stays synchronised with
    // producers that are not synchronised with each other: if every flow owns
    // exactly one BD of the repeating unit, then a flow's k-th transfer always
    // meets that same BD, whatever order the arrivals interleave in. Let one
    // flow own two BDs of a cycle and the ring position stops being a function
    // of how many rounds have happened, so an arrival can land on a BD bound to
    // another flow's buffer and lock.
    //
    // getUniqueBDPattern is the BD emitter's own notion of the repeating unit,
    // so this asks the question in the emitter's terms rather than inventing a
    // second one.
    auto isRoundRobin = [&](const Chain &chain) {
      llvm::SetVector<Operation *> opSet;
      for (auto *o : chain.ops)
        opSet.insert(o);
      auto unit = air::getUniqueBDPattern(opSet);
      if (unit.empty())
        return false; // does not repeat at all
      llvm::SmallPtrSet<Operation *, 4> seen;
      for (auto *o : unit)
        if (!seen.insert(declOf(o)).second)
          return false; // a flow owning two BDs of one cycle
      return seen.size() == chain.order.size();
    };

    auto isHazard = [&](const Chain &chain) {
      // 1. A switchbox port is either statically connected or
      // packet-arbitrated,
      //    never both, and a DMA channel is one port. Mixing the kinds asks the
      //    same port for a static connection AND an arbiter.
      if (!chain.packetDecls.empty() &&
          chain.packetDecls.size() != chain.order.size())
        return true;
      // 2. Independent producers on a ring that is not a round-robin. Neither
      //    half is sufficient on its own, which is why both are here:
      //
      //    - producers alone would split every AIR matmul, where two memtiles
      //      feed a core one A tile and one B tile per iteration. That ring IS
      //      a round-robin (one BD each) and is safe however the two
      //      interleave.
      //    - a non-round-robin ring alone is not a hazard either, as long as
      //    one
      //      producer emits the whole cycle: its own BD order then fixes the
      //      arrival order, which is what lets @appendK and @appendV share.
      //
      //    This is the case diagnoseBDChain explicitly declines to judge
      //    ("convergent flows are trusted to be time-disjoint"). The qwen2.5-3b
      //    rms core is where that trust is misplaced: four flows, six
      //    transfers, from the shim and two different memtiles, and it
      //    deadlocks on device.
      llvm::SmallPtrSet<Operation *, 4> srcs;
      for (auto *d : chain.order)
        for (auto *t : sourceTilesOf(d))
          srcs.insert(t);
      if (srcs.size() >= 2 && !isRoundRobin(chain))
        return true;
      // 2b. The ring cannot serve its flows at the rates they DEMAND. Unlike 2,
      //     this bites with a SINGLE producer, so the argument above -- "one
      //     producer's own BD order fixes the arrival order" -- does not cover
      //     it: that is right about ORDER and silent about RATE.
      //
      //     Two rates per flow, and the hazard is the mismatch BETWEEN them:
      //
      //       demand(d) = firings the flow needs to complete one design step
      //       service(d) = firings one pass of the ring gives it
      //                  = BDs of the repeating unit it owns
      //
      //     The ring stays in step exactly when demand/service is equal across
      //     its flows: a flow needs demand/service passes to finish a step, so
      //     if those counts differ the flow needing fewer runs out of credit
      //     first and blocks. Because the ring is in-order, nothing behind it
      //     proceeds either, and the design deadlocks having written nothing.
      //
      //     Now evaluate that ratio. A flow owning k BDs of the unit is emitted
      //     k times per step and re-fed air.refeed_count times per emission, so
      //
      //       demand(d)  = refeed_count(d) * k(d)
      //       service(d) = k(d)
      //       ratio      = refeed_count(d)
      //
      //     The BD count CANCELS, and comparing bare re-feed counts is not a
      //     proxy for the rate condition -- it IS the rate condition, exactly.
      //     Carrying k explicitly is not more general, it is wrong: it double
      //     counts, and scoring a flow with two BDs as "served twice as fast"
      //     splits rings that are perfectly in step (mm2s_flows_program_order
      //     and tile_s2mm_chain_arrival_order are two such, and both fail if
      //     the ratio is formed that way).
      //
      //     This is also why the condition is independent of isRoundRobin above
      //     rather than layered on it: k cancels whether it is 1 or not, so 2b
      //     needs no round-robin precondition, and 2's round-robin test answers
      //     a different question (ORDER under multiple producers, not RATE).
      //
      //     The rate is air.refeed_count and NOT the op count or the trip
      //     count: air-annotate-refeed has already collapsed the re-feed loop
      //     by the time this runs, so both of those read 1 and miss it.
      //
      //     Measured on the LFM2 ShortConv mixer, whose o-proj X is a re-feed
      //     and whose carried state goes out once, both packet flows on one
      //     tile that simpleDmaChannelAlloc multiplexes onto one MM2S.
      {
        llvm::MapVector<Operation *, int> rateOf;
        for (auto *o : chain.ops) {
          auto *d = declOf(o);
          if (!d)
            continue;
          // Read through the ChannelInterface overload, which honours a
          // per-emission override AND falls back to the channel declaration.
          // The bare Operation* overload reads only an attribute sitting on
          // this op, so a count declared once on the air.channel -- the usual
          // way to spell it -- would read as 1 here and the mismatch would be
          // invisible.
          //
          // max, not first: the re-feed count is a property of the flow, so if
          // two emissions of it disagree the largest is the one whose credits
          // must be satisfied for the design to make progress.
          int rate = 1;
          if (auto ci = dyn_cast<air::ChannelInterface>(o))
            rate = (int)air::getRefeedCount(ci);
          else
            rate = (int)air::getRefeedCount(o);
          auto it = rateOf.find(d);
          if (it == rateOf.end())
            rateOf[d] = rate;
          else
            it->second = std::max(it->second, rate);
        }
        if (rateOf.size() >= 2) {
          int first = rateOf.front().second;
          for (auto &kv : rateOf)
            if (kv.second != first)
              return true;
        }
      }
      // 3. The ring cannot stay in step with the transfers crossing it, by the
      //    emitter's own foldability oracle. repairS2MMChains already acts on
      //    this; verifyMM2SChains only refuses, and being refused is what made
      //    a front end name a channel by hand.
      Operation *stopAt = chain.ops.front()->getParentOfType<air::HerdOp>();
      return !diagnoseBDChain(chain.ops, stopAt, /*emptyPathsOk=*/isMM2S)
                  .empty();
    };

    // Every channel of a tile that is homogeneous in one switching kind, so an
    // evicted group has somewhere legal to land.
    auto homogeneousChans = [&](Operation *tileOp, bool wantPkt, int notChan) {
      SmallVector<int> out;
      for (auto &[k, c] : buildChains()) {
        if (k.first != tileOp || k.second == notChan || c.unkeyed ||
            c.order.empty())
          continue;
        if ((c.packetDecls.size() == c.order.size()) == wantPkt)
          out.push_back(k.second);
      }
      llvm::sort(out);
      return out;
    };

    // What has to leave a mixed chain, and where it goes. The channel must end
    // up homogeneous, and there are exactly two ways to get there: move the
    // packet flows off, or move the circuit flows off. Which one is not a
    // preference -- it is whichever the tile can actually accommodate, on a
    // free channel or one already homogeneous in that kind. Both feasible, the
    // smaller group moves, because disturbing fewer flows is the weaker change.
    //
    // Getting this backwards is not harmless: the qwen2.5-3b rms core ends up
    // with @rmsIn(circuit) sharing with @orms/@outY(packet) while
    // @rmsW(circuit) sits alone on the other channel. Evicting the two packets
    // has nowhere to go and leaves the mix; moving the ONE circuit flow next to
    // @rmsW makes both channels homogeneous, and is what the hand-written pins
    // encoded.
    struct Plan {
      SmallVector<Operation *> movers;
      int dest;
    };
    auto planFor = [&](const Chain &chain, Operation *tileOp, int chan,
                       const llvm::SmallDenseSet<int> &used, int numChans) {
      SmallVector<Operation *> pkt, circ;
      for (auto *d : chain.order)
        (chain.packetDecls.contains(d) ? pkt : circ).push_back(d);
      bool pktFirst = pkt.size() <= circ.size();
      SmallVector<Plan> out;
      for (bool wantPkt : {pktFirst, !pktFirst}) {
        auto &group = wantPkt ? pkt : circ;
        if (group.empty() || llvm::any_of(group, isImmovable))
          continue;
        int dest = -1;
        for (int c = 0; c < numChans && dest < 0; c++)
          if (!used.count(c))
            dest = c; // a channel nobody is using takes the group whole
        if (dest < 0) {
          auto same = homogeneousChans(tileOp, wantPkt, chan);
          if (!same.empty())
            dest = same.front();
        }
        if (dest >= 0)
          out.push_back({group, dest});
      }
      return out;
    };

    for (auto &[key, chain] : buildChains()) {
      if (chain.unkeyed || chain.order.size() < 2 || !isHazard(chain))
        continue;
      Operation *tileOp = key.first;
      auto tile = (*allocs)[chain.allocIdxs.front()].dma_tile;
      auto used = channelsInUse(tileOp);
      int numChans = isMM2S ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
                            : tile.getNumDestConnections(AIE::WireBundle::DMA);

      bool mixed = !chain.packetDecls.empty() &&
                   chain.packetDecls.size() != chain.order.size();
      if (mixed) {
        auto plans = planFor(chain, tileOp, key.second, used, numChans);
        if (plans.empty())
          continue; // no legal home; the router will have the last word
        for (auto *d : plans.front().movers)
          moveFlow(chain.allocIdxs, d, tileOp, key.second, plans.front().dest);
        continue;
      }

      // Unsynchronised producers on a ring that is not a round-robin. The
      // repair is not a peel but a PARTITION BY PRODUCER: give each producer
      // its own channel, and the arrival order on each ring is then fixed by
      // that producer's own BD order, which is the property the ring needs.
      // Peeling one flow instead leaves the rest of the group sharing with a
      // stranger -- on the qwen2.5-3b rms core that leaves three flows and two
      // producers on one channel, having spent the only spare.
      llvm::SmallPtrSet<Operation *, 4> srcs;
      for (auto *d : chain.order)
        for (auto *t : sourceTilesOf(d))
          srcs.insert(t);
      if (srcs.size() >= 2 && !isRoundRobin(chain)) {
        SmallVector<
            std::pair<SmallVector<Operation *>, SmallVector<Operation *>>>
            groups; // (producer signature, flows produced there)
        for (auto *d : chain.order) {
          auto st = sourceTilesOf(d);
          SmallVector<Operation *> sig(st.begin(), st.end());
          llvm::sort(sig);
          bool placed = false;
          for (auto &g : groups)
            if (g.first == sig) {
              g.second.push_back(d);
              placed = true;
              break;
            }
          if (!placed)
            groups.push_back({sig, {d}});
        }

        int spare = 0;
        for (int c = 0; c < numChans; c++)
          if (!used.count(c))
            spare++;
        // More producers than channels: something must double up, and the
        // choice is not free. Coalesce the ON-ARRAY producers and keep the shim
        // apart -- shim traffic is DDR-backed and its timing depends on the
        // host and the NoC, so nothing on the array bounds when it turns up,
        // whereas two on-array producers run under the same lock protocol as
        // the consumer. This is the merge the qwen2.5-3b rms core needs: three
        // producers, two channels, and the working split is {rmsIn, rmsW} from
        // the host against {orms, outY} from two different memtiles.
        if ((int)groups.size() > spare + 1) {
          SmallVector<Operation *> onArray;
          SmallVector<
              std::pair<SmallVector<Operation *>, SmallVector<Operation *>>>
              merged;
          for (auto &g : groups) {
            if (g.second.empty())
              continue;
            if (producedOffArray(g.second.front()))
              merged.push_back(g);
            else
              llvm::append_range(onArray, g.second);
          }
          if (!onArray.empty())
            merged.push_back({{}, onArray});
          // Keep the group holding the first-allocated flow in front, so it is
          // the one that keeps the channel it already has.
          for (size_t i = 0; i < merged.size(); i++)
            if (llvm::is_contained(merged[i].second, chain.order.front())) {
              std::swap(merged[0], merged[i]);
              break;
            }
          groups = merged;
        }

        // Which group KEEPS the channel it is on? Not simply the one that got
        // here first. A group whose flows are each fed by ONE producer already
        // has its arrival order fixed by that producer's BD order -- it is the
        // well-behaved party. A CONVERGENT flow, one channel fed by several
        // producers, is the party nothing orders, so that is what gets moved to
        // the fresh channel and what the well-behaved group is protected from.
        //
        // This is also the choice the hand-written pins made. On the llama rms
        // core @xnorm is convergent (rms core L1 + down_buffer L2) and
        // @layerOut is not, and the pin moved @xnorm. Moving @layerOut instead
        // leaves the same partition on swapped indices, which routes on some
        // designs and not on others -- llama-3.2-1b, llama-3.1-8b and phi4-mini
        // all fail to route (2,2) DMA0 -> (1,1) DMA2 that way.
        llvm::stable_sort(groups, [](const auto &a, const auto &b) {
          return a.first.size() < b.first.size();
        });

        // The keeper is now groups[0]; the others take the channels the tile
        // was not using, one group each while they last.
        for (size_t gi = 1; gi < groups.size(); gi++) {
          if (llvm::any_of(groups[gi].second, isImmovable))
            continue;
          int dest = -1;
          for (int c = 0; c < numChans && dest < 0; c++)
            if (!used.count(c))
              dest = c;
          if (dest < 0)
            break;
          for (auto *d : groups[gi].second)
            moveFlow(chain.allocIdxs, d, tileOp, key.second, dest);
          used.insert(dest);
        }
        continue;
      }

      // Out of step by the emitter's oracle: peel the one flow whose removal
      // leaves BOTH halves in step, exactly as repairS2MMChains chooses, onto a
      // channel the tile was not using. A half carrying a single flow is out of
      // scope for the same reason the whole chain would be -- it has no other
      // flow's BD to land on.
      Operation *stopAt = chain.ops.front()->getParentOfType<air::HerdOp>();
      auto halfInStep = [&](ArrayRef<Operation *> half) {
        llvm::SetVector<Operation *> hd;
        for (auto *o : half)
          hd.insert(declOf(o));
        return hd.size() < 2 ||
               diagnoseBDChain(half, stopAt, /*emptyPathsOk=*/isMM2S).empty();
      };
      int freeChan = -1;
      for (int c = 0; c < numChans && freeChan < 0; c++)
        if (!used.count(c))
          freeChan = c;
      if (freeChan < 0)
        continue;
      for (auto *d : chain.order) {
        if (isImmovable(d))
          continue;
        std::vector<Operation *> m, r;
        for (auto *o : chain.ops)
          (declOf(o) == d ? m : r).push_back(o);
        if (m.empty() || r.empty() || !halfInStep(m) || !halfInStep(r))
          continue;
        moveFlow(chain.allocIdxs, d, tileOp, key.second, freeChan);
        break;
      }
    }

    // A ring is walked strictly in order, and the emitter builds it by
    // concatenating the allocations on a channel in the order they appear here.
    // Splitting a flow off a shared allocation appends the remainder, so a flow
    // that the core issues FIRST can end up emitted last -- the transfer then
    // waits on a BD bound to the other flow's buffer and lock, which serializes
    // the two and hangs outright when the consumer is blocking. Restore the
    // order the core issues in, keyed on the same memcpy id `selection` sorts
    // by within an allocation.
    //
    // Each touched group is permuted among ITS OWN slots, so allocations this
    // pass never looked at keep their positions exactly, and a design where
    // nothing moved is emitted byte for byte as before.
    auto minId = [](const allocation_info_t &a) {
      int64_t m = std::numeric_limits<int64_t>::max();
      for (auto *o : a.memcpyOps)
        if (auto mc = dyn_cast_if_present<air::MemcpyInterface>(o))
          m = std::min(m, (int64_t)mc.getId());
      return m;
    };
    for (auto &key : touched) {
      SmallVector<size_t> slots;
      for (auto [i, alloc] : llvm::enumerate(*allocs))
        if (alloc.dma_tile && alloc.dma_tile.getOperation() == key.first &&
            alloc.dma_channel.channel == key.second)
          slots.push_back(i);
      if (slots.size() < 2)
        continue;
      SmallVector<allocation_info_t> group;
      for (size_t i : slots)
        group.push_back((*allocs)[i]);
      llvm::stable_sort(
          group, [&](const allocation_info_t &a, const allocation_info_t &b) {
            return minId(a) < minId(b);
          });
      for (auto [j, i] : llvm::enumerate(slots))
        (*allocs)[i] = group[j];
    }
  }
}
LogicalResult air::TileDMAAllocator::verifyMM2SChains() {
  // Group exactly as the emitter does: one BD chain per (tile, channel), from
  // the concatenation of every allocation mapped to it in mm2s_allocs order.
  llvm::MapVector<std::pair<Operation *, int>, SmallVector<size_t>> chains;
  for (auto [i, alloc] : llvm::enumerate(mm2s_allocs))
    if (alloc.dma_tile)
      chains[{alloc.dma_tile.getOperation(), alloc.dma_channel.channel}]
          .push_back(i);

  auto declOf = [](Operation *o) -> Operation * {
    auto chan = dyn_cast_if_present<air::ChannelInterface>(o);
    if (!chan)
      return nullptr;
    auto decl = air::getChannelDeclarationThroughSymbol(chan);
    return decl ? decl.getOperation() : nullptr;
  };

  LogicalResult result = success();
  for (auto &[key, allocIdxs] : chains) {
    std::vector<Operation *> ops;
    for (size_t i : allocIdxs)
      llvm::append_range(ops, mm2s_allocs[i].memcpyOps);
    if (ops.size() <= 1)
      continue;

    // Only a chain carrying more than one flow can mis-route. A single-flow
    // chain that slips sends its own slices out of order -- a different
    // matter, and not one this check owns.
    llvm::SetVector<Operation *> decls;
    for (auto *o : ops) {
      auto *d = declOf(o);
      if (!d) {
        decls.clear(); // Unkeyed transfer: not attributable to a flow.
        break;
      }
      decls.insert(d);
    }
    if (decls.size() < 2)
      continue;

    Operation *stopAt = ops.front()->getParentOfType<air::HerdOp>();
    // On the producer side an arm that issues nothing is not a hole: the ring
    // only moves when a transfer goes out, so skipping an arm leaves it
    // aligned.
    std::string why = diagnoseBDChain(ops, stopAt, /*emptyPathsOk=*/true);
    if (why.empty())
      continue;

    // The ring advances one BD per transfer no matter which branch the core
    // took, so keeping it in step needs the branch sequence to cycle in
    // lockstep with the ring. Nothing in the IR states that, and guessing
    // wrong routes a packet to another flow's destination -- a silent hang.
    // Emitting such a design is worse than refusing it.
    std::string names;
    for (auto *d : decls) {
      if (!names.empty())
        names += ", ";
      names += ("@" + cast<air::ChannelOp>(d).getSymName()).str();
    }
    auto diag = ops.front()->emitOpError()
                << "compute-tile MM2S channel " << key.second << " multiplexes "
                << decls.size() << " flows (" << names << ") over "
                << ops.size() << " transfers, but " << why
                << ". Put each flow's transfers in its own unconditional loop, "
                   "so the BD ring follows program order";
    for (auto *d : decls)
      diag.attachNote(d->getLoc())
          << "flow @" << cast<air::ChannelOp>(d).getSymName()
          << " on this chain";
    result = failure();
  }
  return result;
}

FailureOr<AIE::BufferOp>
air::TileDMAAllocator::getBuffer(uint64_t, AIE::TileOp tile,
                                 air::MemcpyInterface &memcpyOp) {
  auto isInbound = isTileInbound(memcpyOp, dmaMemorySpace);
  if (failed(isInbound))
    return failure();
  Value buffer =
      isInbound.value() ? (memcpyOp.getDstMemref()) : (memcpyOp.getSrcMemref());
  auto bufferOp = getUnderlyingBufferOp(buffer);
  if (!bufferOp)
    return failure();
  return bufferOp;
}

// ShimDMAAllocator impl.

// Collect the integer "id" attribute from each dma op (or -1 if missing).
// Used to populate allocation_info_t::dma_id when recording a new shim
// alloc entry. Returned as std::vector<int> to match the downstream
// allocation_info_t::dma_id field type.
static std::vector<int> collectDmaIds(ArrayRef<Operation *> dma_ops) {
  auto idOrSentinel = llvm::map_range(dma_ops, [](Operation *op) -> int {
    auto idAttr = op->getAttrOfType<IntegerAttr>("id");
    return idAttr ? (int)idAttr.getInt() : -1;
  });
  return {idOrSentinel.begin(), idOrSentinel.end()};
}

// Derive the eventual physical column for an unhinted MemTile LTO by walking
// to its downstream cores via the L2 buffer use-chain. The memtile carries
// AIE::BufferOp(s); each buffer is used by air.channel puts/gets whose
// symbol-peers live in aie.core ops with an already-placed column. Returns
// -1 if no peer core can be resolved.
//
// Rationale: under Path B, AIRToAIE emits memtile LTOs with col=row=? and
// defers col selection to aie-place-tiles. ShimDMAAllocator runs earlier
// though and needs *some* col-equivalent key to bucket shim allocations that
// will eventually share a column. Pre-Path B that key was the memtile's
// physical col; today the memtile is column-less but the cores it serves
// are not, so derive the col from them. Single source of truth: the cores
// authoritatively dictate which column a memtile-anchored flow lives in.
static int effectiveColForMemTileLTO(AIE::LogicalTileOp mtLTO) {
  if (!mtLTO || mtLTO.getTileType() != AIE::AIETileType::MemTile)
    return -1;
  std::set<int> cols;
  for (Operation *user : mtLTO.getResult().getUsers()) {
    auto buf = dyn_cast<AIE::BufferOp>(user);
    if (!buf)
      continue;
    for (Operation *bufUser : buf.getResult().getUsers()) {
      auto chan = dyn_cast<air::ChannelInterface>(bufUser);
      if (!chan)
        continue;
      for (air::ChannelInterface peer :
           air::getTheOtherChannelOpThroughSymbol(chan)) {
        auto core = peer->getParentOfType<AIE::CoreOp>();
        if (!core)
          continue;
        auto t = dyn_cast_or_null<AIE::TileOp>(core.getTile().getDefiningOp());
        if (t)
          cols.insert(t.getCol());
      }
    }
  }
  // Only return a col when the memtile unambiguously serves ONE column.
  // A fan-out memtile (e.g. after aggressive-mode L1/L2/L3 channel fusion
  // that broadcasts L3 data to multiple compute columns through a single
  // memtile) has no single "right" bucket col; collapsing two such
  // memtiles by an arbitrary tiebreaker would erase the Op*-bucket
  // anti-collapse guard the caller still needs for the fan-out case. Fall
  // through to -1 and let `sameBucket` re-fall-back to Op* identity.
  if (cols.size() != 1)
    return -1;
  return *cols.begin();
}

air::ShimDMAAllocator::ShimDMAAllocator(AIE::DeviceOp device)
    : air::DMAAllocator(device, air::MemorySpace::L3) {
  shim_dma_channels = 2;
}

FailureOr<air::allocation_info_t> air::ShimDMAAllocator::allocNewDmaChannel(
    air::MemcpyInterface &memcpyOp, AIE::TileLike otherSide, int col, int row,
    std::vector<Operation *> &dma_ops) {
  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;
  AIE::DMAChannelDir dir =
      isMM2S.value() ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;

  // The shim receiving (S2MM) means this flow ends off-chip: a readback to host
  // DDR. `otherSide`/`col` then describe the PRODUCER core, not a destination.
  bool isHostReadback = !isMM2S.value();

  // Check if allocating for a packet flow (packet flow supports channel time
  // multiplexing at the shim DMA level)
  bool isPacketFlowOp = false;
  auto chanTypeRes = getChannelType(memcpyOp);
  if (succeeded(chanTypeRes)) {
    isPacketFlowOp = chanTypeRes.value() == "npu_dma_packet";
  }

  // Search for existing dma channel allocation by air.channel symbol.
  for (auto &t : *allocs) {
    if (t.foundAlloc(getChannelDeclarationThroughSymbol(
            dyn_cast_if_present<air::ChannelInterface>(
                memcpyOp.getOperation())))) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }

  std::vector<int> dma_ops_get_id = collectDmaIds(dma_ops);

  // Single channel-decl lookup for the two attrs that steer shim bucketing:
  //   `broadcast_shape`: L3-direct broadcasts bucket by their first-dest's
  //     incidental col/Op, giving each broadcast its own shim LTO and
  //     overflowing the ShimNOC col count; spread them across existing shim
  //     LTOs instead (see fallback below).
  //   `air.shim_col`: pin this flow's shim LogicalTileOp to a physical column
  //     (applied to bucketCol below, after it is derived).
  bool isBroadcastL3Put = false;
  int shimColPin = -1;
  if (auto chanIf =
          dyn_cast_if_present<air::ChannelInterface>(memcpyOp.getOperation())) {
    if (auto chanDecl = getChannelDeclarationThroughSymbol(chanIf)) {
      isBroadcastL3Put = chanDecl->hasAttr("broadcast_shape");
      if (auto a = chanDecl->getAttrOfType<mlir::IntegerAttr>("air.shim_col")) {
        int pin = (int)a.getInt();
        int numCols = device.getTargetModel().columns();
        if (pin < 0 || pin >= numCols)
          return memcpyOp.emitOpError("air.shim_col column ")
                 << pin << " is out of range [0, " << numCols << ")";
        shimColPin = pin;
      }
    }
  }

  // Bucket key: the far-side col when known, else derive it from a memtile
  // LTO's downstream cores (Path B: memtiles are emitted column-less but
  // their consumer cores are placed). Distinct memtile LTOs that resolve
  // to the same col land in the same bucket and may share one shim; that
  // is the exact merge the packet-multiplex branch below depends on. Falls
  // through to Operation* identity only when no col can be recovered, so
  // truly-unhinted LTOs still each get their own shim — preserving the
  // anti-collapse property the pre-derivation key gave for L3->memtile
  // flows (cross-column routing failure when many memtile flows piled
  // onto one col=-1 bucket and produced too few shim LTOs).
  Operation *otherSideOp = otherSide ? otherSide.getOperation() : nullptr;
  auto bucketColFor = [](int knownCol, Operation *otherOp) -> int {
    if (knownCol >= 0)
      return knownCol;
    if (auto lt = dyn_cast_or_null<AIE::LogicalTileOp>(otherOp))
      return effectiveColForMemTileLTO(lt);
    return -1;
  };
  int bucketCol = bucketColFor(col, otherSideOp);
  // A shim-col pin forces the flow into its own bucket keyed on the pinned
  // column and pins the opened shim LogicalTileOp there (the placer honors the
  // col attr via tryGetCol). Same-pin flows share that bucket/column; without
  // the pin a separate bucket alone yields a col-less LTO whose centroid falls
  // on the (saturated) producer column.
  if (shimColPin >= 0)
    bucketCol = shimColPin;
  // Channel declaration behind a memcpy, or null. Sub-channels of one bundled
  // decl (e.g. @outD [2,2]) share it; independent channels do not.
  auto declOf = [](Operation *op) -> Operation * {
    auto chanIf = dyn_cast_if_present<air::ChannelInterface>(op);
    if (!chanIf)
      return nullptr;
    auto decl = getChannelDeclarationThroughSymbol(chanIf);
    return decl ? decl.getOperation() : nullptr;
  };
  Operation *thisDecl = declOf(memcpyOp.getOperation());

  auto sameBucket = [&](const allocation_info_t &t) {
    int tCol = bucketColFor(t.col, t.otherSideLTO);
    if (bucketCol >= 0 && tCol >= 0)
      return tCol == bucketCol;
    return t.otherSideLTO == otherSideOp;
  };
  auto walkBucketLTOs = [&](auto fn) {
    llvm::SmallPtrSet<Operation *, 8> seen;
    for (auto &t : llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs)) {
      if (!sameBucket(t))
        continue;
      auto lt = dyn_cast<AIE::LogicalTileOp>(t.dma_tile.getOperation());
      if (!lt || lt.getTileType() != AIE::AIETileType::ShimNOCTile)
        continue;
      if (!seen.insert(lt.getOperation()).second)
        continue;
      if (fn(lt))
        return;
    }
  };

  auto channelsUsedOn = [&](AIE::LogicalTileOp lt) {
    std::set<int> used;
    for (auto &t : llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs))
      if (t.dma_tile.getOperation() == lt.getOperation() &&
          t.dma_channel.direction == dir)
        used.insert((int)t.dma_channel.channel);
    return used;
  };

  // For packet flows: reuse the bucket's existing packet channel if any.
  // EXCEPT `air.dedicated_dma_channel` (mirrors MemTileDMAAllocator), which is
  // never collapsed in either direction: a dedicated flow does not reuse an
  // existing packet channel (guarded here), and an unmarked flow does not reuse
  // a channel that already hosts a dedicated flow (skipped in the walk below).
  // This lets a column host BOTH a packet-multiplexed channel and a separate
  // dedicated channel on the same column's other DMA channel, regardless of
  // allocation order.
  if (isPacketFlowOp && !memcpyIsDedicatedChannel(memcpyOp)) {
    AIE::LogicalTileOp packetLT = nullptr;
    int packetCh = -1;
    walkBucketLTOs([&](AIE::LogicalTileOp lt) {
      // When this flow is shim-col-pinned, only reuse a packet channel whose
      // LogicalTileOp sits on the pinned column -- otherwise a previously
      // opened (unpinned, off-column) packet LTO would capture this flow and
      // silently ignore the pin.
      if (shimColPin >= 0) {
        auto ltCol = lt.tryGetCol();
        if (!ltCol || (int)*ltCol != shimColPin)
          return false;
      }
      for (auto &t :
           llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs)) {
        if (t.dma_tile.getOperation() != lt.getOperation())
          continue;
        if (t.dma_channel.direction != dir)
          continue;
        bool tPacket = false;
        for (auto o : t.memcpyOps) {
          auto mc = dyn_cast_if_present<air::MemcpyInterface>(o);
          if (!mc)
            continue;
          auto ct = air::getChannelType(mc);
          if (succeeded(ct) && ct.value() == "npu_dma_packet") {
            tPacket = true;
            break;
          }
        }
        // Never collapse two INDEPENDENT readbacks onto one shim packet
        // channel. A readback ends off-chip, so its bucket key is the producer
        // core's column -- which says nothing about where the transfer should
        // leave the array, and lands every readback out of a herd in the same
        // bucket. Packet reuse then puts them on one channel: llama-1b's
        // appendK, appendV and layerOut all became (2,0) S2MM 0, which the
        // pathfinder cannot route, and which air.shim_col was pinning apart by
        // hand. Sub-channels of ONE bundled decl still multiplex -- that is a
        // single logical transfer and is what the packing exists for.
        if (isHostReadback && t.isHostReadback &&
            declOf(t.memcpyOps.empty() ? nullptr : t.memcpyOps.front()) !=
                thisDecl)
          continue;
        // Never collapse onto a channel that hosts a dedicated flow.
        if (tPacket && !t.containsDedicatedChannel()) {
          packetLT = lt;
          packetCh = (int)t.dma_channel.channel;
          return true;
        }
      }
      return false;
    });
    if (packetLT) {
      AIE::DMAChannel aie_chan = {dir, packetCh};
      allocs->push_back({packetLT,
                         col,
                         row,
                         aie_chan,
                         packetCh,
                         /*packet_flow_id=*/-1,
                         /*otherSideLTO=*/otherSideOp,
                         dma_ops_get_id,
                         {memcpyOp.getOperation()}});
      return allocs->back();
    }
  }

  // Find a bucket LTO with a free channel in this direction; else open
  // a new unhinted shim LTO. When shim-col-pinned, only reuse an LTO already
  // on the pinned column -- otherwise a col-less (or off-column) bucket LTO
  // would capture this flow on the non-packet/dedicated paths and silently
  // drop the pin.
  // Does `lt` already carry a PACKET readback from a DIFFERENT channel decl?
  // Packet matters: only packet flows time-multiplex a shim channel, so only
  // they can collapse onto one. A circuit readback holds a channel of its own
  // and sharing a tile with it is ordinary packing. Same decl is excluded too
  // -- its sub-channels are one logical transfer and are meant to multiplex.
  auto hostsOtherPacketReadback = [&](AIE::LogicalTileOp lt) {
    for (auto &t : llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs)) {
      if (t.dma_tile.getOperation() != lt.getOperation() || !t.isHostReadback)
        continue;
      if (declOf(t.memcpyOps.empty() ? nullptr : t.memcpyOps.front()) ==
          thisDecl)
        continue;
      for (auto o : t.memcpyOps) {
        auto mc = dyn_cast_if_present<air::MemcpyInterface>(o);
        if (!mc)
          continue;
        auto ct = air::getChannelType(mc);
        if (succeeded(ct) && ct.value() == "npu_dma_packet")
          return true;
      }
    }
    return false;
  };

  // A readback ends off-chip, so it has no column affinity at all: its bucket
  // key is the PRODUCER's column, which only says where the data came from.
  // Restricting it to that bucket puts every readback out of a herd on one
  // shim tile -- llama-1b's appendK/appendV/layerOut -- which does not route,
  // and is what air.shim_col was pinning apart by hand. Give it the SPARSEST
  // existing shim LTO with a free channel instead, so readbacks spread over
  // tiles the design already owns.
  //
  // Spreading this way rather than by opening a fresh LTO per readback is
  // deliberate: the AIR pipeline runs aie-place-tiles with
  // merge-logical-tiles=false, so every LTO costs a whole shim tile and NPU2
  // only has 8. One-LTO-per-readback needs 9 for this design and fails to
  // place. Joining another column's existing bucket is exactly what a
  // shim_col pin achieves (appendK pinned to col 3 shares the inKV LTO).
  // Bucket column of an LTO, from any allocation that owns it, or -1.
  auto ltoBucketCol = [&](AIE::LogicalTileOp lt) {
    for (auto &t : llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs))
      if (t.dma_tile.getOperation() == lt.getOperation()) {
        int c = bucketColFor(t.col, t.otherSideLTO);
        if (c >= 0)
          return c;
      }
    return -1;
  };
  // Nearest LTO to the producer that has a free channel and carries no packet
  // readback from another decl (same-decl sub-channels may still share it).
  // Distance still matters -- picking purely the emptiest tile scatters
  // readbacks away from their producers and cost ~1.8% decode throughput on
  // llama-1b. Rank by (distance from the producer column, channels used).
  auto spreadShimLTO = [&]() -> AIE::LogicalTileOp {
    AIE::LogicalTileOp best = nullptr;
    std::pair<int, int> bestKey = {std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max()};
    llvm::SmallPtrSet<Operation *, 8> seen;
    for (auto &t : llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs)) {
      auto lt = dyn_cast<AIE::LogicalTileOp>(t.dma_tile.getOperation());
      if (!lt || lt.getTileType() != AIE::AIETileType::ShimNOCTile)
        continue;
      if (!seen.insert(lt.getOperation()).second)
        continue;
      if (hostsOtherPacketReadback(lt))
        continue;
      int used = (int)channelsUsedOn(lt).size();
      if (used >= shim_dma_channels)
        continue;
      int ltCol = ltoBucketCol(lt);
      int dist = (ltCol < 0 || col < 0) ? shim_dma_channels
                                        : std::abs(ltCol - (int)col);
      std::pair<int, int> key = {dist, used};
      if (key < bestKey) {
        best = lt;
        bestKey = key;
      }
    }
    return best;
  };

  AIE::LogicalTileOp tileLT = nullptr;
  if (shimColPin < 0 && isPacketFlowOp && isHostReadback)
    tileLT = spreadShimLTO();
  if (!tileLT)
    walkBucketLTOs([&](AIE::LogicalTileOp lt) {
      if (shimColPin >= 0) {
        auto ltCol = lt.tryGetCol();
        if (!ltCol || (int)*ltCol != shimColPin)
          return false;
      }
      if ((int)channelsUsedOn(lt).size() < shim_dma_channels) {
        tileLT = lt;
        return true;
      }
      return false;
    });
  // Broadcast fallback: reuse the sparsest existing shim LTO across all
  // buckets before opening a new one.
  if (!tileLT && isBroadcastL3Put && !isPacketFlowOp) {
    AIE::LogicalTileOp best = nullptr;
    int bestUsed = std::numeric_limits<int>::max();
    llvm::SmallPtrSet<Operation *, 8> seen;
    for (auto &t : llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs)) {
      auto lt = dyn_cast<AIE::LogicalTileOp>(t.dma_tile.getOperation());
      if (!lt || lt.getTileType() != AIE::AIETileType::ShimNOCTile)
        continue;
      if (!seen.insert(lt.getOperation()).second)
        continue;
      int used = (int)channelsUsedOn(lt).size();
      if (used >= shim_dma_channels)
        continue;
      if (used < bestUsed) {
        best = lt;
        bestUsed = used;
      }
    }
    if (best)
      tileLT = best;
  }
  if (!tileLT) {
    OpBuilder b(device);
    b.setInsertionPointToStart(device.getBody());
    for (auto &op : device.getBody()->getOperations()) {
      if (isa<AIE::TileOp, AIE::LogicalTileOp>(op))
        b.setInsertionPointAfter(&op);
      else
        break;
    }
    // Order shim LTOs to mirror the IR order of their target memtile LTO.
    // SequentialPlacer packs both pools in IR order from col 0, so without
    // this the k-th shim ends up at col k but its connected memtile may be
    // at a different col, producing cross-column flows that overload the
    // switchbox on narrow devices (NPU1, 4 cols). Insertion point is moved
    // to just before the first existing shim LTO whose target memtile has
    // a strictly larger IR index than this flow's target memtile.
    auto otherSideMem = dyn_cast_or_null<AIE::LogicalTileOp>(otherSideOp);
    if (otherSideMem &&
        otherSideMem.getTileType() == AIE::AIETileType::MemTile) {
      SmallVector<AIE::LogicalTileOp> memtileLTOs;
      for (auto lt : device.getBody()->getOps<AIE::LogicalTileOp>())
        if (lt.getTileType() == AIE::AIETileType::MemTile)
          memtileLTOs.push_back(lt);
      int targetJ = -1;
      for (int i = 0; i < (int)memtileLTOs.size(); i++) {
        if (memtileLTOs[i].getOperation() == otherSideOp) {
          targetJ = i;
          break;
        }
      }
      auto shimTargetJ = [&](AIE::LogicalTileOp shim) -> int {
        for (auto &t :
             llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs)) {
          if (t.dma_tile.getOperation() != shim.getOperation())
            continue;
          if (!t.otherSideLTO)
            continue;
          for (int i = 0; i < (int)memtileLTOs.size(); i++)
            if (memtileLTOs[i].getOperation() == t.otherSideLTO)
              return i;
        }
        return std::numeric_limits<int>::max();
      };
      if (targetJ >= 0) {
        for (auto lt : device.getBody()->getOps<AIE::LogicalTileOp>()) {
          if (lt.getTileType() != AIE::AIETileType::ShimNOCTile)
            continue;
          if (shimTargetJ(lt) > targetJ) {
            b.setInsertionPoint(lt);
            break;
          }
        }
      }
    }
    tileLT = AIE::LogicalTileOp::create(
        b, device.getLoc(), AIE::AIETileType::ShimNOCTile,
        /*col=*/shimColPin >= 0 ? b.getI32IntegerAttr(shimColPin)
                                : IntegerAttr(),
        /*row=*/IntegerAttr(),
        /*allocation_scheme=*/StringAttr());
  }

  auto usedChans = channelsUsedOn(tileLT);
  int dma_channel = -1;
  for (int ch = 0; ch < shim_dma_channels; ch++) {
    if (!usedChans.count(ch)) {
      dma_channel = ch;
      break;
    }
  }
  if (dma_channel < 0)
    return memcpyOp.emitOpError("out of shim DMA channels");

  // When shim-col-pinned, record the pinned col as this entry's col so
  // sameBucket (which keys on t.col) groups same-pin packet flows together --
  // letting two pinned packet channels packet-multiplex onto ONE shim
  // LTO/channel at the pinned column instead of each opening its own LTO there.
  int baseCol = shimColPin >= 0 ? shimColPin : col;
  auto baseRes = air::DMAAllocator::allocNewDmaChannel(
      memcpyOp, tileLT, dma_channel, baseCol, row, dma_ops_get_id);
  if (failed(baseRes))
    return baseRes;
  // Stamp the bucket key on the record the base allocator just pushed.
  // The base allocator returns either the matched reused entry or
  // `allocs->back()`; in both cases the matching record lives in
  // mm2s_allocs/s2mm_allocs and we update both copies (returned + stored)
  // to keep walkBucketLTOs's view consistent.
  // getOperation() isn't const-qualified on the op interface; cast away
  // const for the pointer-equality compare.
  Operation *baseOp =
      const_cast<allocation_info_t &>(*baseRes).dma_tile.getOperation();
  auto matchesReturned = [&](allocation_info_t &t) {
    return t.dma_tile.getOperation() == baseOp &&
           t.dma_channel == baseRes->dma_channel;
  };
  for (auto &t : llvm::concat<allocation_info_t>(mm2s_allocs, s2mm_allocs)) {
    if (matchesReturned(t)) {
      t.otherSideLTO = otherSideOp;
      t.isHostReadback = isHostReadback;
    }
  }
  baseRes->otherSideLTO = otherSideOp;
  baseRes->isHostReadback = isHostReadback;
  return baseRes;
}

FailureOr<air::allocation_info_t>
air::ShimDMAAllocator::allocNewDmaChannel(air::MemcpyInterface &memcpyOp,
                                          air::allocation_info_t existing_alloc,
                                          std::vector<Operation *> &dma_ops) {
  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

  std::vector<int> dma_ops_get_id = collectDmaIds(dma_ops);

  for (auto &t : *allocs) {
    if (t.foundAlloc(existing_alloc.getDmaTile(), existing_alloc.dma_channel)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      for (auto id : dma_ops_get_id)
        t.dma_id.push_back(id);
      return t;
    }
  }
  // Code shouldn't have proceeded to this stage.
  return air::DMAAllocator::allocNewDmaChannel(
      memcpyOp, existing_alloc.getDmaTile(),
      existing_alloc.dma_channel.channel);
}

FailureOr<AIE::ExternalBufferOp>
air::ShimDMAAllocator::getBuffer(uint64_t &BufferId, AIE::TileOp tile,
                                 air::MemcpyInterface &memcpyOp) {
  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  // Allocate external buffers
  auto memref =
      (isMM2S.value()) ? (memcpyOp.getSrcMemref()) : (memcpyOp.getDstMemref());
  MemRefType memrefTy = llvm::cast<MemRefType>(memref.getType());
  // External buffers have memory space L3
  mlir::Attribute memSpaceAttr =
      air::MemorySpaceAttr::get(memcpyOp->getContext(), dmaMemorySpace);
  memrefTy = MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                             AffineMap(), memSpaceAttr);
  // Names use shim coords when known: tile is the shim NOC tile that owns the
  // external buffer's DMA program (the L3 buffer itself has no tile, but its
  // name ties it to the shim that drives it). For unplaced shim tiles
  // (LogicalTileOp(?, ?)) the col/row are -1 in the printed name; the symbol
  // suffix in generateBufferNameInStringStream still keeps it unique.
  AIE::TileLike tileLike =
      dyn_cast_if_present<AIE::TileLike>(tile.getOperation());
  int shimCol = tileLike ? tileLike.tryGetCol().value_or(-1) : -1;
  int shimRow = tileLike ? tileLike.tryGetRow().value_or(-1) : -1;
  AIE::ExternalBufferOp bufferOp = allocateExternalBufferOp(
      BufferId, memrefTy, device,
      memcpyOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
      shimCol, shimRow);
  return bufferOp;
}

// Search for opportunities where air channels can reuse flow op via time
// multiplexing
FailureOr<air::allocation_info_t>
air::ShimDMAAllocator::foundFlowReuseOpportunity(
    std::vector<air::MemcpyBundleAsFlow> memcpy_flows,
    air::allocation_info_t alloc, bool isMM2S) {
  for (auto &f : memcpy_flows) {
    if (isMM2S) {
      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].getDmaTile() == alloc.getDmaTile() &&
            f.S2MM_alloc[i].dma_channel.direction ==
                alloc.dma_channel.direction &&
            f.S2MM_alloc[i].dma_channel.channel == alloc.dma_channel.channel) {
          for (auto &mm2s : f.MM2S_alloc) {
            if (mm2s.getDmaTile() && mm2s.getDmaTile().isShimTile()) {
              return mm2s;
            }
          }
        }
      }
    } else {
      for (auto &mm2s : f.MM2S_alloc) {
        if (mm2s.getDmaTile() == alloc.getDmaTile() &&
            mm2s.dma_channel.direction == alloc.dma_channel.direction &&
            mm2s.dma_channel.channel == alloc.dma_channel.channel) {
          for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
            if (f.S2MM_alloc[i].getDmaTile() &&
                f.S2MM_alloc[i].getDmaTile().isShimTile()) {
              return f.S2MM_alloc[i];
            }
          }
        }
      }
    }
  }
  return failure();
}

void air::ShimDMAAllocator::spreadCollapsedPacketChannels(
    std::vector<MemcpyBundleAsFlow> &memcpy_flows) {
  auto declOf = [](Operation *o) -> Operation * {
    auto chan = dyn_cast_if_present<air::ChannelInterface>(o);
    if (!chan)
      return nullptr;
    auto decl = air::getChannelDeclarationThroughSymbol(chan);
    return decl ? decl.getOperation() : nullptr;
  };
  // A decl whose collapse is load-bearing, or whose channel the front end has
  // already chosen. Broadcasts fan out from ONE port by construction, and a
  // pinned or dedicated flow states where it belongs; neither is ours to move.
  // `air.tile_dma_channel` is deliberately NOT consulted: it pins a COMPUTE
  // TILE's DMA channel and says nothing about which shim port the flow leaves
  // from. Treating it as immovable here is what made this pass pick the wrong
  // flow on llama-1b and phi4-mini -- air-annotate-packet-ids derives that pin
  // on @rmsW for those two, so the peel fell through to the rope LUT, the one
  // flow on the chain whose destination differs from the rest, and the column
  // lost the port diversity an unrelated convergent group needed.
  auto isImmovable = [](Operation *decl) {
    return decl->hasAttr("broadcast_shape") ||
           decl->hasAttr(air::attrs::DedicatedDmaChannel);
  };

  for (auto *allocs : {&mm2s_allocs, &s2mm_allocs}) {
    if (allocs->empty())
      continue;
    AIE::DMAChannelDir dir = allocs->front().dma_channel.direction;

    // Chains are keyed exactly as the emitter groups them: one per
    // (tile, channel), over every allocation mapped to it.
    llvm::MapVector<std::pair<Operation *, int>, SmallVector<size_t>> chains;
    llvm::DenseMap<Operation *, llvm::SmallDenseSet<int>> usedChansPerTile;
    for (auto [i, alloc] : llvm::enumerate(*allocs)) {
      if (!alloc.dma_tile)
        continue;
      // Shim only: a LogicalTileOp that is not a ShimNOC tile, or a physical
      // tile of another kind, belongs to a different allocator.
      auto lt = dyn_cast<AIE::LogicalTileOp>(alloc.dma_tile.getOperation());
      if (lt && lt.getTileType() != AIE::AIETileType::ShimNOCTile)
        continue;
      Operation *t = alloc.dma_tile.getOperation();
      chains[{t, alloc.dma_channel.channel}].push_back(i);
      usedChansPerTile[t].insert(alloc.dma_channel.channel);
    }

    for (auto &[key, allocIdxs] : chains) {
      Operation *tileOp = key.first;
      auto &used = usedChansPerTile[tileOp];

      // Decls on this chain, in a deterministic order.
      SmallVector<Operation *> order;
      llvm::SmallPtrSet<Operation *, 8> seen;
      bool unkeyed = false;
      for (size_t i : allocIdxs) {
        for (auto *o : (*allocs)[i].memcpyOps) {
          auto *d = declOf(o);
          if (!d) {
            unkeyed = true; // not attributable to a flow: leave the chain alone
            break;
          }
          if (seen.insert(d).second)
            order.push_back(d);
        }
        if (unkeyed)
          break;
      }
      if (unkeyed || order.size() < 2)
        continue;

      // Which decl KEEPS the channel? An immovable one has to, so if the chain
      // carries any, the first of them is the keeper and every other decl --
      // including those allocated BEFORE it -- becomes a candidate. Keeping
      // order[0] unconditionally instead would strand the chain whenever every
      // decl after the first is immovable: nothing would be eligible to move
      // and the collapse would survive with a free channel sitting idle.
      // Otherwise the decl that got here first keeps the channel, so a chain of
      // N flows spreads to min(N, free+1) channels without shuffling the flow
      // that was already placed.
      size_t keepIdx = 0;
      for (size_t i = 0; i < order.size(); i++)
        if (isImmovable(order[i])) {
          keepIdx = i;
          break;
        }

      for (size_t oi = 0; oi < order.size(); oi++) {
        if (oi == keepIdx)
          continue;
        Operation *moveDecl = order[oi];
        if (isImmovable(moveDecl))
          continue;
        // Lowest free channel in this direction on this tile.
        int freeChan = -1;
        for (int c = 0; c < shim_dma_channels; c++)
          if (!used.count(c)) {
            freeChan = c;
            break;
          }
        if (freeChan < 0)
          break; // fully subscribed: the remaining flows keep multiplexing

        // Retarget an allocation whose transfers all move; split the rest.
        SmallVector<allocation_info_t> splits;
        for (size_t i : allocIdxs) {
          std::vector<Operation *> keepOps, moveOps;
          for (auto *o : (*allocs)[i].memcpyOps)
            (declOf(o) == moveDecl ? moveOps : keepOps).push_back(o);
          if (moveOps.empty())
            continue;
          (*allocs)[i].packet_flow_id = -1; // reassigned on flow connection
          if (keepOps.empty()) {
            (*allocs)[i].dma_channel = {dir, freeChan};
            (*allocs)[i].tile_channel = freeChan;
            continue;
          }
          allocation_info_t split = (*allocs)[i];
          split.dma_channel = {dir, freeChan};
          split.tile_channel = freeChan;
          split.memcpyOps = moveOps;
          (*allocs)[i].memcpyOps = keepOps;
          splits.push_back(split);
        }
        llvm::append_range(*allocs, splits);
        used.insert(freeChan);

        // The bundles hold their own copy of the allocation and the flows are
        // connected from that copy.
        for (auto &f : memcpy_flows) {
          if (f.air_flow_op != moveDecl)
            continue;
          for (auto *side : {&f.MM2S_alloc, &f.S2MM_alloc})
            for (auto &fa : *side) {
              if (!fa.dma_tile || fa.dma_tile.getOperation() != tileOp)
                continue;
              if (fa.dma_channel.direction != dir ||
                  fa.dma_channel.channel != key.second)
                continue;
              fa.dma_channel.channel = freeChan;
              fa.tile_channel = freeChan;
              fa.packet_flow_id = -1;
            }
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "spreadCollapsedPacketChannels: moved @"
                   << cast<air::ChannelOp>(moveDecl).getSymName() << " to "
                   << (dir == AIE::DMAChannelDir::MM2S ? "MM2S" : "S2MM") << " "
                   << freeChan << "\n");
      }
    }
  }
}

} // namespace xilinx

// MemTileDMAAllocator impl.

namespace xilinx {

air::MemTileDMAAllocator::MemTileDMAAllocator(AIE::DeviceOp device)
    : air::DMAAllocator(device, air::MemorySpace::L2) {
  const auto &aie_target = device.getTargetModel();
  for (int i = 0, e = aie_target.columns(); i < e; i++) {
    memtile_dma_columns.push_back(i);
  }
}

FailureOr<air::allocation_info_t>
air::MemTileDMAAllocator::simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp,
                                                int chan) {
  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

  const int dummy{0};
  auto buffer = getBuffer(dummy, /*tile=*/nullptr, memcpyOp);
  if (failed(buffer)) {
    return memcpyOp->emitOpError("failed to get buffer.");
  }
  // TileLike instead of TileOp: the underlying tile may be a logical tile
  // before aie-place-tiles runs.
  auto tile = dyn_cast_if_present<AIE::TileLike>(
      buffer.value().getTile().getDefiningOp());
  if (!tile) {
    return buffer.value()->emitOpError("failed to get an AIE tile.");
  }

  // Check if allocating for a packet flow (packet flow supports channel time
  // multiplexing)
  bool isPacketFlowOp = false;
  auto chanTypeRes = getChannelType(memcpyOp);
  if (succeeded(chanTypeRes)) {
    isPacketFlowOp = chanTypeRes.value() == "npu_dma_packet";
  }

  // Search for existing dma channel allocation
  unsigned num_allocs = 0;
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile))
      num_allocs++;
    if (t.foundAlloc(tile, memcpyOp)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
    // Reuse an existing DMA channel on this tile instead of allocating a new
    // one. Never collapse a channel marked air.dedicated_dma_channel, nor
    // collapse onto an allocation that already hosts one (either direction).
    //   - MM2S (source) side collapses promiscuously onto a packet-flow
    //     channel (broadcast fan-out / pkt_id multiplexing rely on it), and
    //     otherwise onto a proven-identical endpoint, same as S2MM: repeated
    //     drains of one L2 buffer must time-multiplex ONE channel. Without
    //     this they round-robin, so a memtile holding two output endpoints
    //     interleaves their BDs on a shared channel and the switchbox
    //     multicasts each endpoint's data to both destinations.
    //   - S2MM (receiver) side collapses only flows proven identical (same
    //     channel decl + constant bundle indices), for BOTH packet and circuit
    //     flows: distinct sources fanning into one L2 buffer each need their
    //     own physical S2MM channel, but repeated invocations of one flow
    //     (e.g. an L2 buffer re-filled at one endpoint across loop iterations)
    //     time-multiplex a single channel as sequential BD tasks. Collapsing
    //     circuit flows too keeps a memtile that also carries a wide broadcast
    //     from exhausting its S2MM channels on same-flow refills.
    if (!memcpyIsDedicatedChannel(memcpyOp) && !t.containsDedicatedChannel()) {
      bool canCollapse =
          isMM2S.value()
              ? ((isPacketFlowOp && t.foundPacketFlowAllocInTile(tile)) ||
                 t.foundSameLogicalFlowInTile(tile, memcpyOp))
              : t.foundSameLogicalFlowInTile(tile, memcpyOp);
      if (canCollapse) {
        t.memcpyOps.push_back(memcpyOp.getOperation());
        return t;
      }
    }
  }
  // Need to allocate a new one. TileLike.getNumSourceConnections /
  // getNumDestConnections is interface-defined and works for both physical
  // TileOp and LogicalTileOp (LogicalTileOp consults the targetModel via
  // its tile_type).
  int memtile_dma_channels =
      isMM2S.value() ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
                     : tile.getNumDestConnections(AIE::WireBundle::DMA);
  if (chan == -1) {
    // Channel-floor steer: a memtile DMA buffer whose defining op carries
    // `air.memtile_dma_channel_min = N` reserves physical channels [0, N) on
    // this memtile, so its flows land on [N, ...). Used when a broadcast's
    // route on a low physical channel collides with another column flow
    // transiting this memtile's switchbox. The attr rides on the memcpy
    // (air.channel.put/get) op itself, set by the front end and preserved
    // through the AIR pipeline (copyPaddingAttributes /
    // ComposeMemrefOpOnChannelOp), so it is available here regardless of how
    // the underlying buffer was lowered.
    int minCh = 0;
    if (auto a = memcpyOp->getAttrOfType<IntegerAttr>(
            air::attrs::MemtileDmaChannelMin)) {
      minCh = static_cast<int>(a.getInt());
      // Validate the floor: it must leave at least one usable channel
      // [minCh, memtile_dma_channels). An out-of-range floor would otherwise be
      // silently ignored (falling back to round-robin), defeating the steer.
      if (minCh < 0 || minCh >= memtile_dma_channels)
        return memcpyOp.emitOpError("air.memtile_dma_channel_min = ")
               << minCh << " is out of range [0, " << memtile_dma_channels
               << ") for the " << (isMM2S.value() ? "MM2S" : "S2MM")
               << " DMA channels of this memtile";
    }
    int avail = memtile_dma_channels - minCh;
    chan = minCh + (num_allocs % avail);
  }
  return air::DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

FailureOr<air::allocation_info_t>
air::MemTileDMAAllocator::simpleDmaChannelAlloc(
    air::MemcpyInterface &memcpyOp, air::allocation_info_t &existing_alloc) {
  auto isMM2S = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

  const int dummy{0};
  auto buffer = getBuffer(dummy, /*tile=*/nullptr, memcpyOp);
  if (failed(buffer)) {
    return memcpyOp->emitOpError("failed to get buffer.");
  }
  auto tile = dyn_cast_if_present<AIE::TileLike>(
      buffer.value().getTile().getDefiningOp());
  if (!tile) {
    return buffer.value()->emitOpError("failed to get AIE tile.");
  }

  for (auto &t : *allocs) {
    if (t.foundAlloc(existing_alloc.getDmaTile(), existing_alloc.dma_channel)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }
  // Code shouldn't have proceeded to this stage.
  int chan = -1;
  return air::DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

// Search for opportunities where air channels can reuse flow op via time
// multiplexing
FailureOr<air::allocation_info_t>
air::MemTileDMAAllocator::foundFlowReuseOpportunity(
    std::vector<air::MemcpyBundleAsFlow> memcpy_flows,
    air::allocation_info_t alloc, bool isMM2S) {
  for (auto &f : memcpy_flows) {
    if (!isMM2S) {
      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].getDmaTile() == alloc.getDmaTile() &&
            f.S2MM_alloc[i].dma_channel.direction ==
                alloc.dma_channel.direction &&
            f.S2MM_alloc[i].dma_channel.channel == alloc.dma_channel.channel) {
          for (auto &mm2s : f.MM2S_alloc) {
            if (mm2s.getDmaTile() && mm2s.getDmaTile().isMemTile()) {
              return mm2s;
            }
          }
        }
      }
    } else {
      for (auto &mm2s : f.MM2S_alloc) {
        if (mm2s.getDmaTile() == alloc.getDmaTile() &&
            mm2s.dma_channel.direction == alloc.dma_channel.direction &&
            mm2s.dma_channel.channel == alloc.dma_channel.channel) {
          for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
            if (f.S2MM_alloc[i].getDmaTile() &&
                f.S2MM_alloc[i].getDmaTile().isMemTile()) {
              return f.S2MM_alloc[i];
            }
          }
        }
      }
    }
  }
  return failure();
}

FailureOr<AIE::BufferOp>
air::MemTileDMAAllocator::getBuffer(uint64_t, AIE::TileOp,
                                    air::MemcpyInterface &memcpyOp) {
  auto isInbound = isTileInbound(memcpyOp, dmaMemorySpace);
  if (failed(isInbound))
    return failure();
  Value buffer =
      isInbound.value() ? (memcpyOp.getDstMemref()) : (memcpyOp.getSrcMemref());
  auto bufferOp = getUnderlyingBufferOp(buffer);
  if (!bufferOp)
    return failure();
  return bufferOp;
}

// CascadeAllocator impl.

// Attempts to allocate (or reuse) a cascade flow for the given memcpyOp.
FailureOr<air::allocation_info_t>
air::CascadeAllocator::coreCascadeAlloc(air::MemcpyInterface &memcpyOp) {
  // Determine if the operation is a cascade put (outbound)
  auto isCascadePut = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isCascadePut))
    return failure();

  // Select allocation list based on direction
  auto allocs =
      isCascadePut.value() ? &cascade_put_allocs : &cascade_get_allocs;

  // Retrieve the buffer and the tile where this memcpyOp operates
  const int dummy{0};
  auto buffer = getBuffer(dummy, /*tile=*/nullptr, memcpyOp);
  if (failed(buffer)) {
    return memcpyOp->emitOpError("failed to get buffer.");
  }
  auto tile = buffer.value().getTileOp();
  if (!tile) {
    return buffer.value()->emitOpError("failed to get AIE tile.");
  }

  // Search for an existing allocation for this tile and memcpyOp
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile, memcpyOp))
      return t;
  }

  // No existing allocation found, create a new one
  return air::CascadeAllocator::allocNewCascade(memcpyOp, tile);
}

// Creates a new cascade allocation entry when no matching allocation exists.
FailureOr<air::allocation_info_t>
air::CascadeAllocator::allocNewCascade(air::MemcpyInterface &memcpyOp,
                                       AIE::TileOp tile) {
  if (!tile) {
    return memcpyOp.emitOpError("failed to get the AIE tile. This indicates a "
                                "potential error in the compilation flow.");
  }

  // Determine if this is a cascade put or get
  auto isCascadePut = isTileOutbound(memcpyOp, dmaMemorySpace);
  if (failed(isCascadePut))
    return failure();
  auto allocs =
      isCascadePut.value() ? &cascade_put_allocs : &cascade_get_allocs;

  // Check if allocation already exists for this tile
  for (auto &t : *allocs) {
    if (t.foundAlloc(tile)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
    // Also check for an allocation tied to the channel declaration
    if (t.foundAlloc(tile, getChannelDeclarationThroughSymbol(
                               dyn_cast_if_present<air::ChannelInterface>(
                                   memcpyOp.getOperation())))) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }

  // Create a new allocation_info_t entry for this tile
  air::allocation_info_t output = {tile,
                                   /*col*/ -1,
                                   /*row*/ -1,
                                   /*aie_chan*/ AIE::DMAChannel(),
                                   /*chan*/ -1,
                                   /*packet_flow_id=*/-1,
                                   /*otherSideLTO=*/nullptr,
                                   /*dma_id*/ std::vector<int>{},
                                   {memcpyOp.getOperation()}};
  allocs->push_back(output);
  return output;
}

// Retrieves the underlying AIE::BufferOp associated with the given memcpyOp.
FailureOr<AIE::BufferOp>
air::CascadeAllocator::getBuffer(uint64_t, AIE::TileOp,
                                 air::MemcpyInterface &memcpyOp) {
  auto isInbound = isTileInbound(memcpyOp, dmaMemorySpace);
  if (failed(isInbound))
    return failure();

  // Select source or destination buffer depending on inbound/outbound
  Value buffer =
      isInbound.value() ? (memcpyOp.getDstMemref()) : (memcpyOp.getSrcMemref());

  // Resolve the actual underlying buffer op
  auto bufferOp = getUnderlyingBufferOp(buffer);
  if (!bufferOp)
    return failure();
  return bufferOp;
}

// MemcpyBundleAsFlow impl.

LogicalResult
air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::DmaMemcpyNdOp memcpyOp) {
  // air::DmaMemcpyNdOp is a complete memcpy with both src and dst
  S2MM[0].push_back(memcpyOp.getOperation());
  auto dstMS = air::getMemorySpace(
      llvm::cast<BaseMemRefType>(memcpyOp.getDstMemref().getType()));
  auto srcMS = air::getMemorySpace(
      llvm::cast<BaseMemRefType>(memcpyOp.getSrcMemref().getType()));
  if (!dstMS || !srcMS)
    return memcpyOp->emitOpError("unrecognized memory space on memref");
  S2MM_memspace = *dstMS;
  MM2S.push_back(memcpyOp.getOperation());
  MM2S_memspace = *srcMS;
  memcpyResourceType = "npu_dma_stream";
  return success();
}

LogicalResult
air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::ChannelGetOp memcpyOp) {
  auto chan = air::getChannelDeclarationThroughSymbol(memcpyOp);
  int alloc_id = 0;
  // mmio channels reserve no DMA allocations and don't need the
  // broadcast/index-matching logic below, which assumes hardware fanout.
  // Record the resource type (so downstream code can skip mmio bundles)
  // and return — the dedicated mmio lowering pass handles the rest.
  if (chan.getChannelType() == "npu_mmio") {
    air_flow_op = chan.getOperation();
    S2MM[alloc_id].push_back(memcpyOp.getOperation());
    auto getMS = air::getMemorySpace(
        llvm::cast<BaseMemRefType>(memcpyOp.getMemref().getType()));
    if (!getMS)
      return memcpyOp->emitOpError("unrecognized memory space on memref");
    S2MM_memspace = *getMS;
    memcpyResourceType = "npu_mmio";
    return success();
  }
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
      if (areIdenticalVectors(indices_uint, position)) {
        alloc_id = iter;
      }
    }
  }
  air_flow_op = chan.getOperation();
  S2MM[alloc_id].push_back(memcpyOp.getOperation());
  auto getMS = air::getMemorySpace(
      llvm::cast<BaseMemRefType>(memcpyOp.getMemref().getType()));
  if (!getMS)
    return memcpyOp->emitOpError("unrecognized memory space on memref");
  S2MM_memspace = *getMS;
  memcpyResourceType = chan.getChannelType().str();
  return success();
}

LogicalResult
air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::ChannelPutOp memcpyOp) {
  auto chan = air::getChannelDeclarationThroughSymbol(memcpyOp);
  air_flow_op = chan.getOperation();
  MM2S.push_back(memcpyOp.getOperation());
  auto putMS = air::getMemorySpace(
      llvm::cast<BaseMemRefType>(memcpyOp.getMemref().getType()));
  if (!putMS)
    return memcpyOp->emitOpError("unrecognized memory space on memref");
  MM2S_memspace = *putMS;
  memcpyResourceType = chan.getChannelType().str();
  // numMM2SAllocs (number of DISTINCT producer tiles) is computed during DMA
  // channel allocation, where each put's tile is resolved -- a single producer
  // doing multiple puts (loop / ping-pong) stays ONE producer, while distinct
  // producer tiles (packet fan-in convergence) become N. See
  // simpleDMAChannelAllocation.
  return success();
}

LogicalResult air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(
    air::ChannelInterface memcpyOp) {
  if (auto get =
          dyn_cast_if_present<air::ChannelGetOp>(memcpyOp.getOperation()))
    return pushBackMemcpyOpToBundle(get);
  else if (auto put =
               dyn_cast_if_present<air::ChannelPutOp>(memcpyOp.getOperation()))
    return pushBackMemcpyOpToBundle(put);
  else
    return memcpyOp->emitOpError("unknown op type in air::ChannelInterface");
  return success();
}

air::MemcpyBundleAsFlow::MemcpyBundleAsFlow(air::DmaMemcpyNdOp dmaMemcpyOp) {
  air_flow_op = dmaMemcpyOp.getOperation();
  numS2MMAllocs = 1;
  numMM2SAllocs = 1;
  std::vector<std::vector<Operation *>> v1(numS2MMAllocs,
                                           std::vector<Operation *>());
  S2MM = v1;
  S2MM_alloc = std::vector<air::allocation_info_t>(numS2MMAllocs);
  MM2S_alloc = std::vector<air::allocation_info_t>(numMM2SAllocs);
  memcpyResourceType = "npu_dma_stream";
}

air::MemcpyBundleAsFlow::MemcpyBundleAsFlow(air::ChannelOp chan) {
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
  S2MM_alloc = std::vector<air::allocation_info_t>(numS2MMAllocs);
  MM2S_alloc = std::vector<air::allocation_info_t>(numMM2SAllocs);
  memcpyResourceType = chan.getChannelType().str();
  keep_pkt_header = chan->hasAttr(air::attrs::KeepPktHeader);
}

} // namespace xilinx

namespace xilinx {

bool air::isPacketShimFlow(const air::MemcpyBundleAsFlow &f) {
  return f.memcpyResourceType == "npu_dma_packet" &&
         f.MM2S_memspace == air::MemorySpace::L3;
}

// Identify the unique parent core (== receiver tile) of a flow's S2MM
// receivers. Returns null if there isn't exactly one. Reordering across
// flows that target distinct receiver tiles is unsafe -- the bug we fix
// only manifests when N flows demux into a single tile's S2MM port.
static AIE::CoreOp uniqueReceiverCore(const air::MemcpyBundleAsFlow &f) {
  AIE::CoreOp uniq = nullptr;
  for (int i = 0; i < f.numS2MMAllocs && i < (int)f.S2MM.size(); ++i)
    for (auto *recvOp : f.S2MM[i]) {
      auto c = recvOp->getParentOfType<AIE::CoreOp>();
      if (!c)
        return nullptr;
      if (!uniq)
        uniq = c;
      else if (uniq != c)
        return nullptr;
    }
  return uniq;
}

void air::sortPacketShimFlowsByReceiverOrder(
    std::vector<air::MemcpyBundleAsFlow> &memcpy_flows,
    AIE::DeviceOp aie_device) {
  // Group eligible flows by receiver core. Only reorder within groups of
  // >=2 flows that share a single core -- that's the bug pattern (N flows
  // demuxing into one tile S2MM port). Reordering across cores would
  // disturb routing for unrelated flows (dual-herd, multi-column, etc.).
  llvm::MapVector<AIE::CoreOp, SmallVector<size_t>> groupByCore;
  for (size_t i = 0; i < memcpy_flows.size(); ++i) {
    if (!isPacketShimFlow(memcpy_flows[i]))
      continue;
    if (auto core = uniqueReceiverCore(memcpy_flows[i]))
      groupByCore[core].push_back(i);
  }
  bool anyGroup = false;
  for (auto &kv : groupByCore)
    if (kv.second.size() >= 2) {
      anyGroup = true;
      break;
    }
  if (!anyGroup)
    return;

  DenseMap<Operation *, unsigned> walkPos;
  unsigned pos = 0;
  Operation *root = aie_device->getParentOfType<ModuleOp>();
  if (!root)
    root = aie_device;
  root->walk([&](Operation *op) { walkPos[op] = pos++; });

  auto firstUsePos = [&](const air::MemcpyBundleAsFlow &f) {
    unsigned minPos = std::numeric_limits<unsigned>::max();
    for (int i = 0; i < f.numS2MMAllocs && i < (int)f.S2MM.size(); ++i)
      for (auto *recvOp : f.S2MM[i]) {
        auto it = walkPos.find(recvOp);
        if (it != walkPos.end() && it->second < minPos)
          minPos = it->second;
      }
    return minPos;
  };

  // Sort each group's subset in place (stable_sort on the subset keeps
  // the comparator strict-weak).
  for (auto &kv : groupByCore) {
    auto &idx = kv.second;
    if (idx.size() < 2)
      continue;
    SmallVector<air::MemcpyBundleAsFlow> group;
    group.reserve(idx.size());
    for (auto i : idx)
      group.push_back(std::move(memcpy_flows[i]));
    llvm::stable_sort(group, [&](const air::MemcpyBundleAsFlow &a,
                                 const air::MemcpyBundleAsFlow &b) {
      return firstUsePos(a) < firstUsePos(b);
    });
    for (size_t k = 0; k < idx.size(); ++k)
      memcpy_flows[idx[k]] = std::move(group[k]);
  }
}

void air::reorderL3PacketPutsByFlowOrder(
    AIE::DeviceOp aie_device,
    const std::vector<air::MemcpyBundleAsFlow> &memcpy_flows) {
  // L3 puts live in the launch's func.func outside aie.device, reached via
  // the parent module. AIRRtToNpuPass emits dma_start_task in IR walk order
  // of these puts, so IR order must match flow order.
  auto parentModule = aie_device->getParentOfType<ModuleOp>();
  if (!parentModule)
    return;

  DenseMap<StringAttr, SmallVector<air::ChannelPutOp>> putsByChan;
  parentModule.walk([&](air::ChannelPutOp put) {
    auto memrefTy = dyn_cast<BaseMemRefType>(put.getMemref().getType());
    if (!memrefTy || !air::isL3(memrefTy))
      return;
    auto chan = air::getChannelDeclarationThroughSymbol(put);
    if (!chan || chan.getChannelType().str() != "npu_dma_packet")
      return;
    putsByChan[chan.getSymNameAttr()].push_back(put);
  });

  SmallVector<air::ChannelPutOp> sortedPuts;
  for (auto &f : memcpy_flows) {
    if (!isPacketShimFlow(f))
      continue;
    auto chan = dyn_cast_if_present<air::ChannelOp>(f.air_flow_op);
    if (!chan)
      continue;
    auto it = putsByChan.find(chan.getSymNameAttr());
    if (it == putsByChan.end())
      continue;
    for (auto put : it->second)
      sortedPuts.push_back(put);
  }

  // Per-block prev anchors handle launch clones (e.g. main + lightweight
  // reset device for load_pdi); each clone gets its own moveAfter chain.
  DenseMap<Block *, Operation *> prevByBlock;
  for (auto put : sortedPuts) {
    Block *blk = put->getBlock();
    auto it = prevByBlock.find(blk);
    if (it == prevByBlock.end()) {
      prevByBlock[blk] = put;
      continue;
    }
    // Move the put after its flow-order predecessor, carrying its wrap-and-
    // stride operand slice (e.g. an arith.addi computing a per-iteration DDR
    // offset) so SSA dominance holds. A put whose slice contains a side-
    // effecting op is left in place rather than reordered unsafely.
    (void)air::moveWithPureBackwardSlice(put.getOperation(), it->second,
                                         /*after=*/true);
    prevByBlock[blk] = put;
  }
}

// Resolve the memory space of one endpoint (source when isSource, else
// destination) of a memcpy op via the air::MemcpyInterface, which already
// abstracts the src/dst memref across channel puts/gets and dma_memcpy_nd. This
// keeps the per-endpoint dispatch working uniformly for any MemcpyInterface op.
static std::optional<air::MemorySpace>
getMemcpyEndpointMemorySpace(Operation *o, bool isSource) {
  auto memcpyOpIf = dyn_cast_if_present<air::MemcpyInterface>(o);
  if (!memcpyOpIf)
    return std::nullopt;
  Value ref = isSource ? memcpyOpIf.getSrcMemref() : memcpyOpIf.getDstMemref();
  if (!ref)
    return std::nullopt;
  return air::getMemorySpace(llvm::cast<BaseMemRefType>(ref.getType()));
}

// Resolve the memory space of a single S2MM receiver op (the destination of
// the data movement). A broadcast/demux flow may fan out to receivers in
// DIFFERENT memory spaces (e.g. one dest is an L1 compute tile, others are L2
// memtile relays), so allocation must dispatch per-receiver rather than on the
// flow's aggregate S2MM_memspace (which only records the last receiver seen).
static std::optional<air::MemorySpace>
getS2MMReceiverMemorySpace(Operation *o) {
  return getMemcpyEndpointMemorySpace(o, /*isSource=*/false);
}

// Resolve the memory space of a single MM2S producer op (the source of the data
// movement). A convergent flow may have producers in DIFFERENT memory spaces
// (e.g. an L1 compute core + an L2 memtile both feeding one S2MM), so MM2S
// allocation must dispatch per-producer rather than
// on the flow's aggregate MM2S_memspace.
static std::optional<air::MemorySpace>
getMM2SProducerMemorySpace(Operation *o) {
  return getMemcpyEndpointMemorySpace(o, /*isSource=*/true);
}

// The producer tile of an MM2S op, used as the grouping key that collapses
// multiple puts from ONE producer tile (loop / ping-pong) onto a single MM2S
// allocation while keeping DISTINCT producer tiles apart. Key identity:
//   - L1 producer: the parent aie.core's tile op.
//   - L2 producer: the owning tile of the source buffer -- a physical
//     AIE::TileOp once placed, or the logical-tile op pre-placement.
// The returned op is used only for pointer-identity comparison within one
// flow's producer list, so a physical or logical tile op is equally valid as
// long as one producer tile yields one stable key. Returns null when the tile
// cannot be resolved (caller treats a null key as "ungroupable").
static Operation *getMM2SProducerTileKey(Operation *o) {
  if (auto ms = getMM2SProducerMemorySpace(o)) {
    if (*ms == air::MemorySpace::L1) {
      if (auto core = o->getParentOfType<AIE::CoreOp>())
        return core.getTileOp().getOperation();
    } else if (*ms == air::MemorySpace::L2) {
      auto memcpyOpIf = cast<air::MemcpyInterface>(o);
      if (auto buf = getUnderlyingBufferOp(memcpyOpIf.getSrcMemref()))
        return buf.getTile().getDefiningOp();
    }
  }
  return nullptr;
}

// AIR channel to AIE flow scheduling strategy 1: round robin
// Problem: no awareness wrt channel put and get pattern, leading to deadlocks
LogicalResult air::simpleDMAChannelAllocation(
    std::vector<air::MemcpyBundleAsFlow> &memcpy_flows,
    air::ShimDMAAllocator &shim_dma_alloc,
    air::MemTileDMAAllocator &memtile_dma_alloc,
    TileDMAAllocator &tile_dma_alloc,
    air::CascadeAllocator &core_cascade_alloc) {
  for (auto &f : memcpy_flows) {
    // MMIO channels carry data via host-side runtime-sequence blockwrites,
    // not DMA. They consume no DMA channel, BD, or routing resource and
    // bypass allocation entirely. Their put/get pairs are converted by a
    // dedicated late pass (see lowerAIRMMIOChannelOps).
    if (f.memcpyResourceType == "npu_mmio")
      continue;
    {
      // Allocate MM2S producers. Dispatch PER-PRODUCER by memory space so a
      // convergent packet flow with mixed-memspace producers (e.g. an L1
      // compute core + an L2 memtile both feeding one S2MM) gets each its right
      // DMA resource: L1 -> tile
      // DMA channel (requires an aie.core parent), L2 -> memtile DMA channel.
      // PACKET channels group producers by DISTINCT tile (L1 or L2), so each
      // producer tile gets one packet_flow / MM2S alloc index. A single
      // producer doing multiple puts (loop / ping-pong) stays ONE alloc.
      // Non-packet (circuit/cascade) keeps single-alloc behavior (idx==0). L3
      // (shim) producers are handled in the dedicated shim passes below.
      bool isPacket = f.memcpyResourceType == "npu_dma_packet";
      SmallVector<Operation *> producerTiles;
      for (auto o : f.MM2S) {
        auto pms = getMM2SProducerMemorySpace(o);
        if (pms != air::MemorySpace::L1 && pms != air::MemorySpace::L2)
          continue;
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        AIE::TileOp coreTile; // valid for L1 producers
        if (pms == air::MemorySpace::L1) {
          auto core = memcpyOpIf->getParentOfType<AIE::CoreOp>();
          if (!core)
            return memcpyOpIf->emitOpError(
                "memcpy op not outlined in an aie.core op.");
          coreTile = core.getTileOp();
        }
        int idx = 0;
        // Only PACKET flows group producers per distinct tile (each grouped
        // producer gets its own packet_flow / MM2S alloc, all converging on the
        // shared dest S2MM via a common pkt id). Non-packet (circuit / cascade)
        // flows keep idx==0: their multiple MM2S ops are per-index / broadcast
        // endpoints resolved elsewhere, not fan-in convergence onto one alloc.
        if (isPacket) {
          Operation *tileKey = getMM2SProducerTileKey(o);
          idx = -1;
          for (int k = 0; k < (int)producerTiles.size(); k++)
            if (producerTiles[k] == tileKey) {
              idx = k;
              break;
            }
          if (idx < 0) {
            idx = (int)producerTiles.size();
            producerTiles.push_back(tileKey);
          }
        }
        if ((int)f.MM2S_alloc.size() <= idx)
          f.MM2S_alloc.resize(idx + 1);

        FailureOr<air::allocation_info_t> alloc_res;
        if (pms == air::MemorySpace::L1) {
          if (f.memcpyResourceType == "npu_dma_stream" ||
              f.memcpyResourceType == "npu_dma_packet") {
            alloc_res = tile_dma_alloc.simpleDmaChannelAlloc(
                memcpyOpIf, coreTile, f.MM2S_alloc[idx].dma_channel.channel);
          } else if (f.memcpyResourceType == "npu_cascade") {
            alloc_res = core_cascade_alloc.coreCascadeAlloc(memcpyOpIf);
          }
        } else { // L2 (memtile) producer
          if (f.memcpyResourceType != "npu_dma_stream" &&
              f.memcpyResourceType != "npu_dma_packet")
            return memcpyOpIf->emitOpError(
                "only supports npu_dma_stream or npu_dma_packet "
                "connections at L2 memory");
          alloc_res = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
        }
        if (failed(alloc_res))
          return failure();
        f.MM2S_alloc[idx] = alloc_res.value();
        if (!f.MM2S_alloc[idx].valid())
          return failure();
      }
      if (isPacket && !producerTiles.empty())
        f.numMM2SAllocs = (int)producerTiles.size();
    }
    // Allocate tile DMA channels for L1 receivers. Dispatch per-receiver (not
    // on f.S2MM_memspace) so a broadcast/demux flow with mixed-memspace dests
    // gets each L1 dest a tile DMA channel here; its L2 dests are allocated a
    // memtile channel in the second pass below.
    for (size_t i = 0; i < f.S2MM.size(); i++) {
      for (auto o : f.S2MM[i]) {
        if (getS2MMReceiverMemorySpace(o) != air::MemorySpace::L1)
          continue;
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        auto core = memcpyOpIf->getParentOfType<AIE::CoreOp>();
        if (!core) {
          return memcpyOpIf->emitOpError(
              "memcpy op not outlined in an aie.core op.");
        }
        auto tile = core.getTileOp();

        FailureOr<air::allocation_info_t> alloc_res;
        if (f.memcpyResourceType == "npu_dma_stream" ||
            f.memcpyResourceType == "npu_dma_packet") {
          alloc_res = tile_dma_alloc.simpleDmaChannelAlloc(
              memcpyOpIf, tile, f.S2MM_alloc[i].dma_channel.channel);
          if (failed(alloc_res))
            return failure();
        } else if (f.memcpyResourceType == "npu_cascade") {
          alloc_res = core_cascade_alloc.coreCascadeAlloc(memcpyOpIf);
          if (failed(alloc_res))
            return failure();
        }

        f.S2MM_alloc[i] = alloc_res.value();
        if (!f.S2MM_alloc[i].valid())
          return failure();
      }
    }
  }
  for (auto &f : memcpy_flows) {
    // MMIO channels are not allocated to any DMA resource at L2 either.
    if (f.memcpyResourceType == "npu_mmio")
      continue;
    // (L2 (memtile) MM2S producers are allocated in the unified per-producer
    // MM2S pass above, which dispatches L1 -> tile DMA and L2 -> memtile DMA
    // and groups packet producers across both memory spaces.) Allocate memtile
    // DMA channels for L2 receivers. Dispatch per-receiver so a broadcast/demux
    // flow with mixed-memspace dests gets each L2 dest a memtile channel here
    // (its L1 dests were allocated a tile channel above).
    // Ordering invariant: moving L2 MM2S allocation into the first loop is safe
    // because MM2S and S2MM draw from independent per-tile channel pools (a
    // memtile has separate MM2S and S2MM DMA channel sets), so allocating all
    // MM2S before all S2MM does not perturb S2MM channel assignments relative
    // to the old interleaved order.
    for (size_t i = 0; i < f.S2MM.size(); i++) {
      for (auto o : f.S2MM[i]) {
        if (getS2MMReceiverMemorySpace(o) != air::MemorySpace::L2)
          continue;
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        // Report error if the data movement lowers to neither dma stream
        // (aie.flow) nor dma packet flow (aie.packet_flow).
        if (f.memcpyResourceType != "npu_dma_stream" &&
            f.memcpyResourceType != "npu_dma_packet")
          return memcpyOpIf->emitOpError(
              "only supports npu_dma_stream or npu_dma_packet "
              "connections at L2 memory");
        auto alloc_res = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
        if (failed(alloc_res) || !alloc_res->valid())
          return failure();
        f.S2MM_alloc[i] = alloc_res.value();
      }
    }
  }
  // Detect L3 MM2S puts whose air.channel decl carries `broadcast_shape`.
  // These are column-flexible — their far side is a fan-out to many cores,
  // so they can land on any shim col with free MM2S. Other L3 flows are
  // column-rigid (paired to a specific memtile LTO or a placed core).
  auto isBroadcastL3MM2S = [](const MemcpyBundleAsFlow &f) {
    if (f.MM2S_memspace != air::MemorySpace::L3)
      return false;
    for (auto o : f.MM2S) {
      auto chanIf = dyn_cast_if_present<air::ChannelInterface>(o);
      if (!chanIf)
        continue;
      auto decl = getChannelDeclarationThroughSymbol(chanIf);
      if (decl && decl->hasAttr("broadcast_shape"))
        return true;
    }
    return false;
  };

  // L3 shim allocation is bin-packing onto a fixed set of ShimNOC cols
  // (hard cap = device.getNumShimNOCCols(), per-bin cap = 2 MM2S + 2 S2MM).
  // Process flows in rigidity-decreasing order so that rigid flows establish
  // the bins and flexible flows pack into the gaps:
  //   pass 1 — rigid (non-broadcast L3 MM2S + all L3 S2MM)
  //   pass 2 — flexible (broadcast L3 MM2S), reusing existing bins via the
  //            broadcast cross-bucket fallback in ShimDMAAllocator
  // This avoids the order-of-allocation pitfall where a flexible flow opens
  // its own bin before the complementary-direction rigid bin has been
  // created, exceeding the device's ShimNOC col count.
  auto allocateL3 = [&](MemcpyBundleAsFlow &f) -> LogicalResult {
    if (f.memcpyResourceType == "npu_mmio")
      return success();
    if (f.MM2S_memspace == air::MemorySpace::L3) {
      // L3 (shim/host) producers stay single-producer (numMM2SAllocs==1).
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.MM2S) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          if (f.memcpyResourceType != "npu_dma_stream" &&
              f.memcpyResourceType != "npu_dma_packet")
            return memcpyOpIf->emitOpError(
                "only supports npu_dma_stream or npu_dma_packet "
                "connections at L3 memory");
          if (!f.S2MM_alloc[i].getDmaTile())
            return memcpyOpIf->emitOpError(
                "failed to get S2MM tile for L3 allocation.");
          auto s2mmTile = f.S2MM_alloc[i].getDmaTile();
          auto alloc_res = shim_dma_alloc.allocNewDmaChannel(
              memcpyOpIf, s2mmTile, s2mmTile.tryGetCol().value_or(-1),
              s2mmTile.tryGetRow().value_or(-1), f.S2MM[i]);
          if (failed(alloc_res) || !alloc_res->valid())
            return failure();
          f.MM2S_alloc[0] = alloc_res.value();
        }
      }
    }
    if (f.S2MM_memspace == air::MemorySpace::L3) {
      if (f.S2MM.size() > 1) {
        return f.S2MM.front().front()->emitOpError(
            "found multiple inputs for an aie.flow. Fan-in for aie.flow isn't "
            "supported in current architecture.");
      }
      for (auto o : f.S2MM.front()) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        if (f.memcpyResourceType != "npu_dma_stream" &&
            f.memcpyResourceType != "npu_dma_packet")
          return memcpyOpIf->emitOpError(
              "only supports npu_dma_stream or npu_dma_packet "
              "connections at L3 memory");
        if (f.MM2S_alloc.empty() || !f.MM2S_alloc[0].getDmaTile())
          return memcpyOpIf->emitOpError(
              "failed to get MM2S tile for L3 allocation.");
        // L3 (shim) S2MM consumer is single-producer (fan-in to a shim dest is
        // unsupported); use the sole producer alloc.
        auto mm2sTile = f.MM2S_alloc[0].getDmaTile();
        auto alloc_res = shim_dma_alloc.allocNewDmaChannel(
            memcpyOpIf, mm2sTile, mm2sTile.tryGetCol().value_or(-1),
            mm2sTile.tryGetRow().value_or(-1), f.MM2S);
        if (failed(alloc_res) || !alloc_res->valid())
          return failure();
        f.S2MM_alloc.front() = alloc_res.value();
      }
    }
    return success();
  };
  // Pass 1: rigid flows.
  for (auto &f : memcpy_flows)
    if (!isBroadcastL3MM2S(f))
      if (failed(allocateL3(f)))
        return failure();
  // Pass 2: flexible (broadcast) flows.
  for (auto &f : memcpy_flows)
    if (isBroadcastL3MM2S(f))
      if (failed(allocateL3(f)))
        return failure();
  return success();
}

// If found item in vector, return index; else return -1.
template <typename T>
int air::foundInVector(T item, std::vector<T> vec) {
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
bool air::groupingMemcpysByLoop(
    std::vector<air::MemcpyBundleAsFlow> &memcpy_flows) {
  // Group memcpy_flows based on L1-side puts/gets' loop structure
  std::map<AIE::CoreOp, std::vector<scf::ForOp>> for_loops_log_mm2s,
      for_loops_log_s2mm;
  for (auto &f : memcpy_flows) {
    {
      // Group by loop only for L1 producers (dispatch per-producer, since a
      // convergent flow may also have L2 memtile producers with no aie.core).
      for (auto o : f.MM2S) {
        if (getMM2SProducerMemorySpace(o) != air::MemorySpace::L1)
          continue;
        auto core = o->getParentOfType<AIE::CoreOp>();
        f.flow_op_group = foundInVector<scf::ForOp>(
            o->getParentOfType<scf::ForOp>(), for_loops_log_mm2s[core]);
        if ((size_t)f.flow_op_group == for_loops_log_mm2s[core].size()) {
          for_loops_log_mm2s[core].push_back(o->getParentOfType<scf::ForOp>());
        }
      }
    }
    {
      // Group by loop only for L1 receivers (dispatch per-receiver, since a
      // broadcast/demux flow may also have L2 dests with no aie.core parent).
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
          if (getS2MMReceiverMemorySpace(o) != air::MemorySpace::L1)
            continue;
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

} // namespace xilinx
