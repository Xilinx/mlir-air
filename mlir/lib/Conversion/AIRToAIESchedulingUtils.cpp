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

#include "llvm/ADT/SmallSet.h"

#include <limits>
#include <mutex>
#include <set>

#define DEBUG_TYPE "air-to-aie-scheduling-utils"

using namespace mlir;

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

int64_t air::get1DOffset(SmallVector<Value> memcpy_offsets,
                         SmallVector<Value> memcpy_strides) {
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

// Given a vector of memcpy operations, return a map of their repeat counts,
// relative to a common ancestor region.
llvm::MapVector<int, llvm::SetVector<Operation *>>
air::getRepeatCounts(std::vector<Operation *> memcpy_ops) {
  llvm::MapVector<int, llvm::SetVector<Operation *>> repeatCounts;
  llvm::SetVector<Operation *> memcpyIOps;
  for (auto o : memcpy_ops) {
    memcpyIOps.insert(o);
  }

  // Two channel ops map to the same shim BD iff memref + offsets/sizes/
  // strides all match.
  auto chansMappedToEquivalentBDs = [](air::ChannelInterface chanA,
                                       air::ChannelInterface chanB) {
    if (chanA.getMemref() != chanB.getMemref())
      return false;
    if (chanA.getOffsets().size() != chanB.getOffsets().size() ||
        chanA.getSizes().size() != chanB.getSizes().size() ||
        chanA.getStrides().size() != chanB.getStrides().size())
      return false;
    auto zipped_operands = llvm::zip_equal(
        llvm::concat<Value>(chanA.getOffsets(), chanA.getSizes(),
                            chanA.getStrides()),
        llvm::concat<Value>(chanB.getOffsets(), chanB.getSizes(),
                            chanB.getStrides()));
    return llvm::all_of(zipped_operands, [](std::tuple<Value, Value> pair) {
      return isEqualConstantIntOrValue(std::get<0>(pair), std::get<1>(pair));
    });
  };

  // Check if two channel operations are part of an N-buffer rotation pattern.
  // They are part of the same rotation if:
  // 1. They belong to the same air.channel declaration
  // 2. Their memrefs have the same type (shape, element type, memory space)
  // 3. Their sizes and strides are equivalent (access pattern match)
  // Note: Unlike chansMappedToEquivalentBDs, this allows different buffer
  // values as long as they have the same type and access pattern.
  auto chansPartOfSameRotation = [](air::ChannelInterface chanA,
                                    air::ChannelInterface chanB) -> bool {
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
    if (chanA.getSizes().size() != chanB.getSizes().size() ||
        chanA.getStrides().size() != chanB.getStrides().size())
      return false;

    auto zipped = llvm::zip_equal(
        llvm::concat<Value>(chanA.getSizes(), chanA.getStrides()),
        llvm::concat<Value>(chanB.getSizes(), chanB.getStrides()));
    return llvm::all_of(zipped, [](std::tuple<Value, Value> pair) {
      return isEqualConstantIntOrValue(std::get<0>(pair), std::get<1>(pair));
    });
  };

  auto dmasMappedToEquivalentBDs = [](air::DmaMemcpyNdOp dmaA,
                                      air::DmaMemcpyNdOp dmaB) {
    return OperationEquivalence::isEquivalentTo(
        dmaA, dmaB, OperationEquivalence::IgnoreLocations);
  };
  auto memcpyIMappedToEquivalentBDs =
      [chansMappedToEquivalentBDs, dmasMappedToEquivalentBDs](Operation *opA,
                                                              Operation *opB) {
        if (auto chanA = dyn_cast_if_present<air::ChannelInterface>(opA))
          if (auto chanB = dyn_cast_if_present<air::ChannelInterface>(opB))
            return chansMappedToEquivalentBDs(chanA, chanB);
        if (auto dmaA = dyn_cast_if_present<air::DmaMemcpyNdOp>(opA))
          if (auto dmaB = dyn_cast_if_present<air::DmaMemcpyNdOp>(opB))
            return dmasMappedToEquivalentBDs(dmaA, dmaB);
        return false; // Unknown or different air::MemcpyInterface op types.
      };

  // Canonicalize a chain of memcpy ops as candidates to map to dma bds, by
  // removing repetitive patterns.
  auto getUniqueBDPattern = [memcpyIMappedToEquivalentBDs](
                                llvm::SetVector<Operation *> memcpyIOps) {
    // Get a vector of unique BDs.
    llvm::SetVector<Operation *> uniqueBDPattern;
    auto opIt = memcpyIOps.begin();
    while (opIt != memcpyIOps.end() &&
           llvm::none_of(uniqueBDPattern,
                         [opIt, memcpyIMappedToEquivalentBDs](Operation *op1) {
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
  };

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
      [&chansPartOfSameRotation](
          const llvm::SetVector<Operation *> &ops) -> bool {
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
    for (auto *op : opVec) {
      repeatCounts[0].insert(op);
    }
    return repeatCounts;
  }

  // Get the deepest region which is ancestor to all memcpyIOps.
  SmallVector<Operation *> memcpyIOpVec = memcpyIOps.takeVector();
  Region *commonRegion =
      air::findCommonRegionContainingAllAncestors(memcpyIOpVec);
  if (!commonRegion)
    return repeatCounts;

  // Get each memcpy op's repeat count, relative to the common region.
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
    repeatCounts[tripCount - 1].insert(o);
  }

  return repeatCounts;
}

std::vector<AIE::BDDimLayoutAttr>
air::getWrapsAndStrides(SmallVector<Value> memcpy_sizes,
                        SmallVector<Value> memcpy_strides, MLIRContext *ctx) {
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
FailureOr<std::pair<AIE::LockOp, AIE::LockOp>>
air::DMAAllocator::getLockForDMA(air::MemcpyInterface &memcpyOp,
                                 AIE::TileLike tile, Operation *bufferOp,
                                 bool lockRaceConditionFix) {
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

  OpBuilder builder(bufferOp);
  auto rlock = allocateLockOp(device, tile, 0);
  auto wlock = UsesSemaphoreLocks ? allocateLockOp(device, tile, init) : rlock;
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
    // reuse the channel allocation.
    if (isPacketFlowOp && t.foundPacketFlowAllocInTile(tile)) {
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
// alloc entry.
static std::vector<int> collectDmaIds(ArrayRef<Operation *> dma_ops) {
  std::vector<int> ids;
  ids.reserve(dma_ops.size());
  for (auto *op : dma_ops) {
    if (op->hasAttr("id"))
      ids.push_back(op->getAttrOfType<IntegerAttr>("id").getInt());
    else
      ids.push_back(-1);
  }
  return ids;
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

  // L3-direct broadcasts (channel decl carries `broadcast_shape`) bucket
  // by their first-dest's incidental col/Op, which gives each broadcast
  // its own shim LTO and overflows the ShimNOC col count. Spread them
  // across existing shim LTOs instead (see fallback below).
  bool isBroadcastL3Put = false;
  if (auto chanIf =
          dyn_cast_if_present<air::ChannelInterface>(memcpyOp.getOperation())) {
    if (auto chanDecl = getChannelDeclarationThroughSymbol(chanIf))
      isBroadcastL3Put = chanDecl->hasAttr("broadcast_shape");
  }

  // Bucket key: the far-side col when known, else the far-side LTO's
  // Operation*. Col is authoritative whenever it's known (>= 0) because two
  // flows targeting the same physical col should share one shim so the shim
  // can sit adjacent to that col. When the far side is an unhinted LTO
  // (col == -1 under Path B) we fall back to Operation* identity, so each
  // distinct unhinted LTO still gets its own shim LTO — preventing the pre-
  // fix collapse where every memtile-side flow piled into one col=-1 bucket
  // and produced too-few shim LTOs (cross-column routing failure).
  Operation *otherSideOp = otherSide ? otherSide.getOperation() : nullptr;
  auto sameBucket = [&](const allocation_info_t &t) {
    if (col >= 0)
      return t.col == col;
    return t.otherSideLTO == otherSideOp;
  };
  auto walkBucketLTOs = [&](auto fn) {
    llvm::SmallPtrSet<Operation *, 8> seen;
    for (auto *side : {&mm2s_allocs, &s2mm_allocs}) {
      for (auto &t : *side) {
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
    }
  };

  auto channelsUsedOn = [&](AIE::LogicalTileOp lt) {
    std::set<int> used;
    for (auto *side : {&mm2s_allocs, &s2mm_allocs})
      for (auto &t : *side)
        if (t.dma_tile.getOperation() == lt.getOperation() &&
            t.dma_channel.direction == dir)
          used.insert((int)t.dma_channel.channel);
    return used;
  };

  // For packet flows: reuse the bucket's existing packet channel if any.
  if (isPacketFlowOp) {
    AIE::LogicalTileOp packetLT = nullptr;
    int packetCh = -1;
    walkBucketLTOs([&](AIE::LogicalTileOp lt) {
      for (auto *side : {&mm2s_allocs, &s2mm_allocs}) {
        for (auto &t : *side) {
          if (t.dma_tile.getOperation() != lt.getOperation())
            continue;
          if (t.dma_channel.direction != dir)
            continue;
          for (auto o : t.memcpyOps) {
            auto mc = dyn_cast_if_present<air::MemcpyInterface>(o);
            if (!mc)
              continue;
            auto ct = air::getChannelType(mc);
            if (succeeded(ct) && ct.value() == "npu_dma_packet") {
              packetLT = lt;
              packetCh = (int)t.dma_channel.channel;
              return true;
            }
          }
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
  // a new unhinted shim LTO.
  AIE::LogicalTileOp tileLT = nullptr;
  walkBucketLTOs([&](AIE::LogicalTileOp lt) {
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
    for (auto *side : {&mm2s_allocs, &s2mm_allocs}) {
      for (auto &t : *side) {
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
      for (auto &op : device.getBody()->getOperations())
        if (auto lt = dyn_cast<AIE::LogicalTileOp>(op))
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
        for (auto *side : {&mm2s_allocs, &s2mm_allocs})
          for (auto &t : *side) {
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
        for (auto &op : device.getBody()->getOperations()) {
          auto lt = dyn_cast<AIE::LogicalTileOp>(op);
          if (!lt || lt.getTileType() != AIE::AIETileType::ShimNOCTile)
            continue;
          if (shimTargetJ(lt) > targetJ) {
            b.setInsertionPoint(lt);
            break;
          }
        }
      }
    }
    tileLT = AIE::LogicalTileOp::create(b, device.getLoc(),
                                        AIE::AIETileType::ShimNOCTile,
                                        /*col=*/IntegerAttr(),
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

  auto baseRes = air::DMAAllocator::allocNewDmaChannel(
      memcpyOp, tileLT, dma_channel, col, row, dma_ops_get_id);
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
  for (auto *side : {&mm2s_allocs, &s2mm_allocs}) {
    for (auto &t : *side) {
      if (matchesReturned(t))
        t.otherSideLTO = otherSideOp;
    }
  }
  baseRes->otherSideLTO = otherSideOp;
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
          if (f.MM2S_alloc.getDmaTile() &&
              f.MM2S_alloc.getDmaTile().isShimTile()) {
            return f.MM2S_alloc;
          }
        }
      }
    } else if (!isMM2S && f.MM2S_alloc.getDmaTile() == alloc.getDmaTile() &&
               f.MM2S_alloc.dma_channel.direction ==
                   alloc.dma_channel.direction &&
               f.MM2S_alloc.dma_channel.channel == alloc.dma_channel.channel) {

      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].getDmaTile() &&
            f.S2MM_alloc[i].getDmaTile().isShimTile()) {
          return f.S2MM_alloc[i];
        }
      }
    }
  }
  return failure();
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
    // Search for existing packet-flow allocations on this tile, and try to
    // reuse the channel allocation.
    if (isPacketFlowOp && t.foundPacketFlowAllocInTile(tile)) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }
  // Need to allocate a new one. TileLike.getNumSourceConnections /
  // getNumDestConnections is interface-defined and works for both physical
  // TileOp and LogicalTileOp (LogicalTileOp consults the targetModel via
  // its tile_type).
  int memtile_dma_channels =
      isMM2S.value() ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
                     : tile.getNumDestConnections(AIE::WireBundle::DMA);
  if (chan == -1)
    chan = num_allocs % memtile_dma_channels;
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
          if (f.MM2S_alloc.getDmaTile() &&
              f.MM2S_alloc.getDmaTile().isMemTile()) {
            return f.MM2S_alloc;
          }
        }
      }
    } else if (isMM2S && f.MM2S_alloc.getDmaTile() == alloc.getDmaTile() &&
               f.MM2S_alloc.dma_channel.direction ==
                   alloc.dma_channel.direction &&
               f.MM2S_alloc.dma_channel.channel == alloc.dma_channel.channel) {

      for (unsigned i = 0; i < f.S2MM_alloc.size(); i++) {
        if (f.S2MM_alloc[i].getDmaTile() &&
            f.S2MM_alloc[i].getDmaTile().isMemTile()) {
          return f.S2MM_alloc[i];
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
  memcpyResourceType = chan.getChannelType().str();
}

} // namespace xilinx

namespace xilinx {

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
    if (f.MM2S_memspace == air::MemorySpace::L1) {
      for (auto o : f.MM2S) {
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
              memcpyOpIf, tile, f.MM2S_alloc.dma_channel.channel);
          if (failed(alloc_res))
            return failure();
        } else if (f.memcpyResourceType == "npu_cascade") {
          alloc_res = core_cascade_alloc.coreCascadeAlloc(memcpyOpIf);
          if (failed(alloc_res))
            return failure();
        }

        f.MM2S_alloc = alloc_res.value();
        if (!f.MM2S_alloc.valid())
          return failure();
      }
    }
    if (f.S2MM_memspace == air::MemorySpace::L1) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
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
  }
  for (auto &f : memcpy_flows) {
    // MMIO channels are not allocated to any DMA resource at L2 either.
    if (f.memcpyResourceType == "npu_mmio")
      continue;
    if (f.MM2S_memspace == air::MemorySpace::L2) {
      for (auto o : f.MM2S) {
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
        f.MM2S_alloc = alloc_res.value();
      }
    }
    if (f.S2MM_memspace == air::MemorySpace::L2) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
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
          f.MM2S_alloc = alloc_res.value();
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
        if (!f.MM2S_alloc.getDmaTile())
          return memcpyOpIf->emitOpError(
              "failed to get MM2S tile for L3 allocation.");
        auto mm2sTile = f.MM2S_alloc.getDmaTile();
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
    if (f.MM2S_memspace == air::MemorySpace::L1) {
      for (auto o : f.MM2S) {
        auto core = o->getParentOfType<AIE::CoreOp>();
        f.flow_op_group = foundInVector<scf::ForOp>(
            o->getParentOfType<scf::ForOp>(), for_loops_log_mm2s[core]);
        if ((size_t)f.flow_op_group == for_loops_log_mm2s[core].size()) {
          for_loops_log_mm2s[core].push_back(o->getParentOfType<scf::ForOp>());
        }
      }
    }
    if (f.S2MM_memspace == air::MemorySpace::L1) {
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

} // namespace xilinx
