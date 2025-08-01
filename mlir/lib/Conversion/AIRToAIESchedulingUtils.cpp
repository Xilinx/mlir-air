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

#include "llvm/ADT/SmallSet.h"

#include <mutex>
#include <set>

#define DEBUG_TYPE "air-to-aie-scheduling-utils"

using namespace mlir;

namespace xilinx {

FailureOr<bool> air::isTileInbound(air::MemcpyInterface memcpyOp,
                                   int tileMemSpaceAsInt) {
  if (memcpyOp.getSrcMemref() && memcpyOp.getDstMemref()) {
    int src_memory_space =
        llvm::cast<BaseMemRefType>(memcpyOp.getSrcMemref().getType())
            .getMemorySpaceAsInt();
    int dst_memory_space =
        llvm::cast<BaseMemRefType>(memcpyOp.getDstMemref().getType())
            .getMemorySpaceAsInt();
    if (src_memory_space == tileMemSpaceAsInt)
      return false;
    else if (dst_memory_space == tileMemSpaceAsInt)
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
                                    int tileMemSpaceAsInt) {
  auto isTileInbRes = isTileInbound(memcpyOp, tileMemSpaceAsInt);
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
    if (isa<AIE::TileOp>(o))
      builder.setInsertionPointAfter(&o);
    else
      break;
  }
  return builder.create<AIE::TileOp>(UnknownLoc::get(aie_device.getContext()),
                                     col, row);
}

AIE::LockOp air::allocateLockOp(AIE::DeviceOp aie_device, AIE::TileOp tile,
                                int init, int id, StringAttr name) {
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
  auto lockOp = b.create<AIE::LockOp>(tile.getLoc(), tile, new_id, init);
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
  AIE::ExternalBufferOp bufferOp = builder.create<AIE::ExternalBufferOp>(
      builder.getUnknownLoc(), memrefTy, nullptr, nullptr);

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

  // Check if all of memcpy_ops only map to one same dma bd. If true, then
  // return that there is only one single repeat count, i.e. a single bd task.
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
    bool wrapsAndStridesAllEquivalent =
        llvm::all_of(zipped_operands, [](std::tuple<Value, Value> pair) {
          return isEqualConstantIntOrValue(std::get<0>(pair),
                                           std::get<1>(pair));
        });
    return wrapsAndStridesAllEquivalent;
  };
  auto dmasMappedToEquivalentBDs = [](air::DmaMemcpyNdOp dmaA,
                                      air::DmaMemcpyNdOp dmaB) {
    return OperationEquivalence::isEquivalentTo(
        dmaA, dmaB, OperationEquivalence::IgnoreLocations);
  };
  auto memcpyIMappedToEquivalentBDs =
      [chansMappedToEquivalentBDs, dmasMappedToEquivalentBDs](Operation *opA,
                                                              Operation *opB) {
        if (auto chanA = dyn_cast<air::ChannelInterface>(opA))
          if (auto chanB = dyn_cast<air::ChannelInterface>(opB))
            return chansMappedToEquivalentBDs(chanA, chanB);
        if (auto dmaA = dyn_cast<air::DmaMemcpyNdOp>(opA))
          if (auto dmaB = dyn_cast<air::DmaMemcpyNdOp>(opB))
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
      auto affineFor = dyn_cast<affine::AffineForOp>(parent);
      auto scfFor = dyn_cast<scf::ForOp>(parent);
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

// allocation_info_t impl.

bool xilinx::air::allocation_info_t::valid() { return dma_tile != nullptr; }

AIE::TileOp xilinx::air::allocation_info_t::getDmaTile() { return dma_tile; }

bool xilinx::air::allocation_info_t::foundAlloc(air::ChannelOp channel_op) {
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

bool xilinx::air::allocation_info_t::foundAlloc(int32_t col, int32_t row) {
  if (col == getDmaTile().getCol() && row == getDmaTile().getRow())
    return true;
  return false;
}

bool xilinx::air::allocation_info_t::foundAlloc(int32_t col, int32_t row,
                                                air::MemcpyInterface memcpyOp) {
  if (foundAlloc(col, row))
    for (auto o : memcpyOps)
      if (memcpyOp.getOperation() == o)
        return true;
  return false;
}

bool xilinx::air::allocation_info_t::foundAlloc(int32_t col, int32_t row,
                                                int chan) {
  return foundAlloc(col, row) && (chan == dma_channel.channel);
}

bool xilinx::air::allocation_info_t::foundAlloc(AIE::DMAChannel channel) {
  if (channel.direction == dma_channel.direction &&
      channel.channel == dma_channel.channel)
    return true;
  else
    return false;
}

bool xilinx::air::allocation_info_t::foundAlloc(int32_t col, int32_t row,
                                                AIE::DMAChannel channel) {
  return foundAlloc(col, row) && foundAlloc(channel);
}

bool xilinx::air::allocation_info_t::foundAlloc(AIE::TileOp tile,
                                                AIE::DMAChannel channel) {
  if (tile == getDmaTile() && foundAlloc(channel))
    return true;
  else
    return false;
}

bool xilinx::air::allocation_info_t::foundAlloc(int32_t col, int32_t row,
                                                air::ChannelOp channel_op) {

  return foundAlloc(col, row) && foundAlloc(channel_op);
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
      auto a_j = dyn_cast<air::MemcpyInterface>(a[j]);
      auto a_min = dyn_cast<air::MemcpyInterface>(a[min]);
      if (a_j.getId() < a_min.getId())
        min = j;
    }
    swap(a, min, i);
  }
}

} // namespace xilinx

namespace xilinx {

FailureOr<air::allocation_info_t>
air::DMAAllocator::lookupDMAAllocation(int64_t col, int64_t row,
                                       air::MemcpyInterface &memcpyOp) {

  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;
  for (auto &t : *allocs) {
    if (t.foundAlloc(col, row, memcpyOp))
      return t;
  }
  return memcpyOp.emitOpError(
      "failed to look up a DMA allocation. This potentially "
      "indicates a failure in the compilation flow.");
}

// Allocate a reader/writer lock pair. These may be the same or different
// locks depending on the target device.
FailureOr<std::pair<AIE::LockOp, AIE::LockOp>>
air::DMAAllocator::getLockForDMA(air::MemcpyInterface &memcpyOp, int col,
                                 int row, Operation *bufferOp,
                                 bool lockRaceConditionFix) {
  auto alloc = lookupDMAAllocation(col, row, memcpyOp);
  if (failed(alloc))
    return memcpyOp->emitOpError("failed to look up dma allocation.");
  AIE::DMAChannel channel = alloc.value().dma_channel;
  AIE::TileOp tile = alloc.value().getDmaTile();
  air::ChannelOp air_chan = nullptr;
  if (auto air_chan_op =
          dyn_cast<air::ChannelInterface>(memcpyOp.getOperation())) {
    air_chan = getChannelDeclarationThroughSymbol(air_chan_op);
  }
  const auto &target_model = device.getTargetModel();
  bool UsesSemaphoreLocks =
      target_model.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks);

  if (UsesSemaphoreLocks) {
    if (air_chan) {
      // AIE2's semaphore locks may share by air.channels
      for (size_t i = 0; i < lock_allocation_list.size(); i++) {
        if (target_model.isMemTile(col, row)) {
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
        else if (target_model.isMemTile(col, row) &&
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
  if (target_model.isMemTile(col, row))
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
    air::MemcpyInterface &memcpyOp, AIE::TileOp tile, int chan, int col = -1,
    int row = -1, std::vector<int> dma_id = {}) {
  if (!tile) {
    return memcpyOp.emitOpError("failed to get the AIE tile. This indicates a "
                                "potential error in the compilation flow.");
  }
  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;
  AIE::DMAChannel aie_chan;
  aie_chan.direction =
      isMM2S.value() ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;
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
  air::allocation_info_t output = {
      tile, col, row, aie_chan, chan, dma_id, {memcpyOp.getOperation()}};
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
                                             int col, int row, int chan = -1) {
  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

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
  if (!tile) {
    return memcpyOp.emitOpError(
        "failed to get a tile at specified col and row. This "
        "indicates a potential compilation failre.");
  }
  int tile_dma_channels =
      isMM2S.value() ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
                     : tile.getNumDestConnections(AIE::WireBundle::DMA);
  if (chan == -1)
    chan = num_allocs % tile_dma_channels;
  return air::DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

FailureOr<AIE::BufferOp>
air::TileDMAAllocator::getBuffer(uint64_t, int64_t col, int64_t row,
                                 air::MemcpyInterface &memcpyOp) {
  if (failed(isTileInbound(memcpyOp, DMAMemorySpaceAsInt)))
    return failure();
  Value buffer = isTileInbound(memcpyOp, DMAMemorySpaceAsInt).value()
                     ? (memcpyOp.getDstMemref())
                     : (memcpyOp.getSrcMemref());
  AIE::BufferOp bufferOp = buffer.getDefiningOp<AIE::BufferOp>();
  // Memref cast
  memref::CastOp castOp = buffer.getDefiningOp<memref::CastOp>();
  if (!bufferOp && castOp)
    bufferOp = castOp.getOperand().getDefiningOp<AIE::BufferOp>();
  return bufferOp;
}

// ShimDMAAllocator impl.

air::ShimDMAAllocator::ShimDMAAllocator(AIE::DeviceOp device)
    : air::DMAAllocator(device, (int)air::MemorySpace::L3) {
  const auto &aie_target = device.getTargetModel();
  shim_dma_channels = 2;
  for (int i = 0, e = aie_target.columns(); i < e; i++) {
    if (aie_target.isShimNOCTile(i, 0))
      dma_columns.push_back(i);
  }
}

FailureOr<air::allocation_info_t> air::ShimDMAAllocator::allocNewDmaChannel(
    air::MemcpyInterface &memcpyOp, int col, int row,
    std::vector<Operation *> &dma_ops,
    std::string colAllocConstraint = "same_column") {
  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;
  AIE::DMAChannelDir dir =
      isMM2S.value() ? AIE::DMAChannelDir::MM2S : AIE::DMAChannelDir::S2MM;

  // Search for existing dma channel allocation
  for (auto &t : *allocs) {
    if (t.foundAlloc(getChannelDeclarationThroughSymbol(
            dyn_cast<air::ChannelInterface>(memcpyOp.getOperation())))) {
      t.memcpyOps.push_back(memcpyOp.getOperation());
      return t;
    }
  }
  AIE::TileOp tile = nullptr;
  int colIdx = 0;
  if (colAllocConstraint == "same_column") {
    // Attempt to use shim dma channels within the same column.
    auto it = find(dma_columns.begin(), dma_columns.end(), col);
    if (it != dma_columns.end())
      colIdx = it - dma_columns.begin();
  }
  int dma_col = dma_columns[colIdx];
  int dma_channel = 0;
  int colTripCount = 0;
  while (any_of(allocs->begin(), allocs->end(), [&](air::allocation_info_t &a) {
    return a.foundAlloc(dma_col, 0, AIE::DMAChannel{dir, dma_channel});
  })) {
    dma_channel++;
    if (dma_channel >= shim_dma_channels) {
      dma_channel = 0;
      dma_col = dma_columns[colIdx++ % dma_columns.size()];
      colTripCount++;
      if (colTripCount > (int)dma_columns.size()) {
        return memcpyOp->emitOpError(
            "failed to map to shim dma channels: out of channels.");
      }
    }
  }
  if (dma_channel >= shim_dma_channels) {
    return memcpyOp.emitOpError("out of shim dma channels.");
  }
  tile = getPhysTileOp(device, dma_col, 0);
  if (!tile) {
    return memcpyOp.emitOpError(
        "failed to get shim tile for the newly allocated shim dma channel.");
  }
  // For shim dma allocations, the col, row and dma_id fields record the other
  // side of the flows, for airrt metadata
  std::vector<int> dma_ops_get_id;
  for (auto op : dma_ops) {
    if (op->hasAttr("id"))
      dma_ops_get_id.push_back(op->getAttrOfType<IntegerAttr>("id").getInt());
    else
      dma_ops_get_id.push_back(-1);
  }
  return air::DMAAllocator::allocNewDmaChannel(memcpyOp, tile, dma_channel, col,
                                               row, dma_ops_get_id);
}

FailureOr<air::allocation_info_t>
air::ShimDMAAllocator::allocNewDmaChannel(air::MemcpyInterface &memcpyOp,
                                          air::allocation_info_t existing_alloc,
                                          std::vector<Operation *> &dma_ops) {
  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

  std::vector<int> dma_ops_get_id;
  for (auto op : dma_ops) {
    if (op->hasAttr("id"))
      dma_ops_get_id.push_back(op->getAttrOfType<IntegerAttr>("id").getInt());
    else
      dma_ops_get_id.push_back(-1);
  }

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
air::ShimDMAAllocator::getBuffer(uint64_t &BufferId, int64_t col, int64_t row,
                                 air::MemcpyInterface &memcpyOp) {
  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  // Allocate external buffers
  auto memref =
      (isMM2S.value()) ? (memcpyOp.getSrcMemref()) : (memcpyOp.getDstMemref());
  MemRefType memrefTy = llvm::cast<MemRefType>(memref.getType());
  // External buffers have memory space L3
  mlir::IntegerType i32Ty = mlir::IntegerType::get(memcpyOp->getContext(), 32);
  mlir::Attribute memSpaceAttr =
      mlir::IntegerAttr::get(i32Ty, DMAMemorySpaceAsInt);
  memrefTy = MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                             AffineMap(), memSpaceAttr);
  AIE::ExternalBufferOp bufferOp = allocateExternalBufferOp(
      BufferId, memrefTy, device,
      memcpyOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
      col, row);
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
    : air::DMAAllocator(device, (int)air::MemorySpace::L2) {
  const auto &aie_target = device.getTargetModel();
  for (int i = 0, e = aie_target.columns(); i < e; i++) {
    memtile_dma_columns.push_back(i);
  }
}

FailureOr<air::allocation_info_t>
air::MemTileDMAAllocator::simpleDmaChannelAlloc(air::MemcpyInterface &memcpyOp,
                                                int chan = -1) {
  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

  const int dummy{0};
  auto buffer = getBuffer(dummy, -1, -1, memcpyOp);
  if (failed(buffer)) {
    return memcpyOp->emitOpError("failed to get buffer.");
  }
  auto tile = buffer.value().getTileOp();
  if (!tile) {
    return buffer.value()->emitOpError("failed to get an AIE tile.");
  }

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
      isMM2S.value() ? tile.getNumSourceConnections(AIE::WireBundle::DMA)
                     : tile.getNumDestConnections(AIE::WireBundle::DMA);
  if (chan == -1)
    chan = num_allocs % memtile_dma_channels;
  return air::DMAAllocator::allocNewDmaChannel(memcpyOp, tile, chan);
}

FailureOr<air::allocation_info_t>
air::MemTileDMAAllocator::simpleDmaChannelAlloc(
    air::MemcpyInterface &memcpyOp, air::allocation_info_t &existing_alloc) {
  auto isMM2S = isTileOutbound(memcpyOp, DMAMemorySpaceAsInt);
  if (failed(isMM2S))
    return failure();
  auto allocs = isMM2S.value() ? &mm2s_allocs : &s2mm_allocs;

  const int dummy{0};
  auto buffer = getBuffer(dummy, -1, -1, memcpyOp);
  if (failed(buffer)) {
    return memcpyOp->emitOpError("failed to get buffer.");
  }
  auto tile = buffer.value().getTileOp();
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
air::MemTileDMAAllocator::getBuffer(uint64_t, int64_t col, int64_t row,
                                    air::MemcpyInterface &memcpyOp) {
  if (failed(isTileInbound(memcpyOp, DMAMemorySpaceAsInt)))
    return failure();
  Value buffer = isTileInbound(memcpyOp, DMAMemorySpaceAsInt).value()
                     ? (memcpyOp.getDstMemref())
                     : (memcpyOp.getSrcMemref());
  AIE::BufferOp bufferOp = buffer.getDefiningOp<AIE::BufferOp>();
  // Memref cast
  memref::CastOp castOp = buffer.getDefiningOp<memref::CastOp>();
  if (!bufferOp && castOp)
    bufferOp = castOp.getOperand().getDefiningOp<AIE::BufferOp>();
  return bufferOp;
}

// MemcpyBundleAsFlow impl.

LogicalResult
air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::DmaMemcpyNdOp memcpyOp) {
  // air::DmaMemcpyNdOp is a complete memcpy with both src and dst
  S2MM[0].push_back(memcpyOp.getOperation());
  S2MM_memspace_as_int =
      llvm::cast<BaseMemRefType>(memcpyOp.getDstMemref().getType())
          .getMemorySpaceAsInt();
  MM2S.push_back(memcpyOp.getOperation());
  MM2S_memspace_as_int =
      llvm::cast<BaseMemRefType>(memcpyOp.getSrcMemref().getType())
          .getMemorySpaceAsInt();
  return success();
}

LogicalResult
air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::ChannelGetOp memcpyOp) {
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
      if (areIdenticalVectors(indices_uint, position)) {
        alloc_id = iter;
      }
    }
  }
  air_flow_op = chan.getOperation();
  S2MM[alloc_id].push_back(memcpyOp.getOperation());
  S2MM_memspace_as_int =
      llvm::cast<BaseMemRefType>(memcpyOp.getMemref().getType())
          .getMemorySpaceAsInt();
  return success();
}

LogicalResult
air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(air::ChannelPutOp memcpyOp) {
  auto chan = air::getChannelDeclarationThroughSymbol(memcpyOp);
  air_flow_op = chan.getOperation();
  MM2S.push_back(memcpyOp.getOperation());
  MM2S_memspace_as_int =
      llvm::cast<BaseMemRefType>(memcpyOp.getMemref().getType())
          .getMemorySpaceAsInt();
  return success();
}

LogicalResult air::MemcpyBundleAsFlow::pushBackMemcpyOpToBundle(
    air::ChannelInterface memcpyOp) {
  if (auto get = dyn_cast<air::ChannelGetOp>(memcpyOp.getOperation()))
    return pushBackMemcpyOpToBundle(get);
  else if (auto put = dyn_cast<air::ChannelPutOp>(memcpyOp.getOperation()))
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
}

} // namespace xilinx

namespace xilinx {

// AIR channel to AIE flow scheduling strategy 1: round robin
// Problem: no awareness wrt channel put and get pattern, leading to deadlocks
LogicalResult air::simpleDMAChannelAllocation(
    std::vector<air::MemcpyBundleAsFlow> &memcpy_flows,
    air::ShimDMAAllocator &shim_dma_alloc,
    air::MemTileDMAAllocator &memtile_dma_alloc,
    TileDMAAllocator &tile_dma_alloc) {
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L1) {
      for (auto o : f.MM2S) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        auto core = memcpyOpIf->getParentOfType<AIE::CoreOp>();
        if (!core) {
          return memcpyOpIf->emitOpError(
              "memcpy op not outlined in an aie.core op.");
        }
        auto tile = core.getTileOp();
        int x = tile.getCol();
        int y = tile.getRow();

        auto alloc_res = tile_dma_alloc.simpleDmaChannelAlloc(
            memcpyOpIf, x, y, f.MM2S_alloc.dma_channel.channel);
        if (failed(alloc_res))
          return failure();

        f.MM2S_alloc = alloc_res.value();
        if (!f.MM2S_alloc.valid())
          return failure();
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L1) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          auto core = memcpyOpIf->getParentOfType<AIE::CoreOp>();
          if (!core) {
            return memcpyOpIf->emitOpError(
                "memcpy op not outlined in an aie.core op.");
          }
          auto tile = core.getTileOp();
          int x = tile.getCol();
          int y = tile.getRow();

          auto alloc_res = tile_dma_alloc.simpleDmaChannelAlloc(
              memcpyOpIf, x, y, f.S2MM_alloc[i].dma_channel.channel);
          if (failed(alloc_res))
            return failure();
          f.S2MM_alloc[i] = alloc_res.value();
          if (!f.S2MM_alloc[i].valid())
            return failure();
        }
      }
    }
  }
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L2) {
      for (auto o : f.MM2S) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        auto alloc_res = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
        if (failed(alloc_res) || !alloc_res->valid())
          return failure();
        f.MM2S_alloc = alloc_res.value();
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L2) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.S2MM[i]) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          auto alloc_res = memtile_dma_alloc.simpleDmaChannelAlloc(memcpyOpIf);
          if (failed(alloc_res) || !alloc_res->valid())
            return failure();
          f.S2MM_alloc[i] = alloc_res.value();
        }
      }
    }
  }
  for (auto &f : memcpy_flows) {
    if (f.MM2S_memspace_as_int == (int)air::MemorySpace::L3) {
      for (size_t i = 0; i < f.S2MM.size(); i++) {
        for (auto o : f.MM2S) {
          auto memcpyOpIf = cast<air::MemcpyInterface>(o);
          auto alloc_res = shim_dma_alloc.allocNewDmaChannel(
              memcpyOpIf, f.S2MM_alloc[i].getDmaTile().getCol(),
              f.S2MM_alloc[i].getDmaTile().getRow(), f.S2MM[i]);
          if (failed(alloc_res) || !alloc_res->valid())
            return failure();
          f.MM2S_alloc = alloc_res.value();
        }
      }
    }
    if (f.S2MM_memspace_as_int == (int)air::MemorySpace::L3) {
      // L3 shim tiles assumed to not be target for broadcast
      if (f.S2MM.size() > 1) {
        return f.S2MM.front().front()->emitOpError(
            "found multiple inputs for an aie.flow. Fan-in for aie.flow isn't "
            "supported in current architecture.");
      }
      for (auto o : f.S2MM.front()) {
        auto memcpyOpIf = cast<air::MemcpyInterface>(o);
        auto alloc_res = shim_dma_alloc.allocNewDmaChannel(
            memcpyOpIf, f.MM2S_alloc.getDmaTile().getCol(),
            f.MM2S_alloc.getDmaTile().getRow(), f.MM2S);
        if (failed(alloc_res) || !alloc_res->valid())
          return failure();
        f.S2MM_alloc.front() = alloc_res.value();
      }
    }
  }
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

} // namespace xilinx
