//===- AIRDialect.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIR_DIALECT_H
#define MLIR_AIR_DIALECT_H

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"

#include <map>

using namespace mlir;

namespace xilinx {
namespace air {

// Attribute names for the DMA-steering / runtime-sequence-ordering markers.
// Centralized so producers and consumers agree (a mistyped literal silently
// no-ops). See the AIRRtToNpu / AIRToAIE passes for their semantics.
namespace attrs {
constexpr StringLiteral RuntimeHoist = "air.runtime_hoist";
constexpr StringLiteral AwaitAppends = "air.await_appends";
constexpr StringLiteral AppendBarrier = "air.append_barrier";
constexpr StringLiteral PreserveShimDmaOrder = "air.preserve_shim_dma_order";
// Marker on a shim MM2S configure task produced by the coalesce-shim-dma merge:
// its BD covers a whole contiguous run merged from several smaller feeds, so
// consecutive coalesced tasks on the same channel may feed distinct downstream
// consumers and must NOT be kept in flight together. The backpressure paces
// such tasks with a cross-channel barrier (drain a group of coalesced transfers
// before starting the next group), reproducing the drain schedule of the
// un-coalesced feed.
constexpr StringLiteral CoalescedShimFeed = "air.coalesced_shim_feed";
constexpr StringLiteral TileDmaChannel = "air.tile_dma_channel";
constexpr StringLiteral MemtileDmaChannelMin = "air.memtile_dma_channel_min";
constexpr StringLiteral DedicatedDmaChannel = "air.dedicated_dma_channel";
// Single-buffer count-free re-broadcast: N (>= 1) re-sends of one resident
// buffer per production. Carried on the air.channel declaration, on a channel
// put/get, or on the source memref.alloc of a memtile producer; read via
// air::getRefeedCount. Written by air-annotate-refeed from a re-feed loop's
// trip count, not by the front end. Verified on air.channel and on put/get.
constexpr StringLiteral RefeedCount = "air.refeed_count";
// User-pinned packet routing ids on an air.channel (channel_type
// "npu_dma_packet"). One packet_flow per id: N ids to a single dest converge on
// one buffer for a downstream demux hop; N ids to N dests route dest i with
// pinned[i]. The compute core writes the id into the payload header, so the DMA
// does not stamp/filter these -- the flows only install switchbox routes. Bare
// spelling matches the broadcast_shape discardable-attr convention on
// air.channel; read via air::ChannelOp::getPacketIDs. Verified on air.channel.
constexpr StringLiteral PacketIDs = "packet_ids";
// Marks a call whose callee writes the packet routing header into the payload
// (the kernel-written-header contract of a PacketIDs channel). The pair binds
// that call to the channel it stamps for:
//   air.pkt_header_channel : SymbolRefAttr -- the air.channel being stamped
//   air.pkt_header_operand : i32           -- index of the id operand
// The ids reaching that operand and the channel's packet_ids are two spellings
// of one contract written in two places; nothing links them in the IR, so a
// mismatch is a silent routing hang. air-annotate-packet-ids compares them
// when both are statically known.
constexpr StringLiteral PktHeaderChannel = "air.pkt_header_channel";
constexpr StringLiteral PktHeaderOperand = "air.pkt_header_operand";
// The kernel writes the routing packet header into the payload itself.
// air-to-aie must not stamp a static pkt_id on the producer BD (that would
// prepend a second header word) and emits the aie.packet_flow with
// {keep_pkt_header = true} so the switchbox keeps the header at the
// destination. For a split bundle keep is per-flow (only the offset-0 bearer
// keeps it); see SrcWritesPktHeader. Bare spelling matches the packet_flow attr
// in the AIE dialect. Verified on air.channel.
constexpr StringLiteral KeepPktHeader = "keep_pkt_header";
// Bundle-wide derived marker set on every split of a KeepPktHeader channel: the
// bundle source writes its own header, so no split's producer BD may be
// stamped. Distinct from KeepPktHeader, which is per-flow (offset-0 bearer
// only).
constexpr StringLiteral SrcWritesPktHeader = "air.src_writes_pkt_header";
// Per-op launch-iteration ("wave") index (i64) on runtime-sequence ops of a
// fused multi-iteration launch. Assigned in AIRRtToNpu right after the fused
// launch loop is unrolled (program order still reflects wave membership) and
// propagated onto the ops each airrt op lowers to, so downstream per-wave
// ordering (RTP arm / set_lock / output-S2MM hoist) groups by this index
// instead of inferring wave boundaries from op positions.
constexpr StringLiteral LaunchWave = "air.launch_wave";
// Opt-out (unit attr) on a shared-L2 memref.alloc: keep the buffer on the
// legacy counted-lock template even under use-lock-race-condition-fix-v2,
// instead of the daisy-chained chain-lock. Honored only for fan-out broadcast
// buffers, whose N readers are independent compute cores: the chain-lock
// over-serializes those reads and can deadlock against a competing fan-in
// chain. Must be tagged on the alloc itself (the air.execute wrapper is already
// lowered away by the time AIRToAIE reads it); propagated onto the lowered
// AIE::BufferOp so air::isChainLockCandidate can exclude it.
constexpr StringLiteral NoChainLock = "air.no_chain_lock";
// Opt-out (unit attr) on a shared-L2 memref.alloc (or its enclosing
// air.execute): leave this L2 buffer intact instead of partitioning it. Used by
// hand-written aggregator patterns where splitting would multiply the
// launch-level shim endpoint count.
constexpr StringLiteral NoSplit = "air.no_split";
// Explicit L2 placement (i32) on a memref.alloc (or its enclosing air.execute):
// pin this buffer to that memtile column. Read by AIRToAIE's memtile bucketing,
// and by air-split-l2-memref, which leaves an explicitly-placed buffer intact.
constexpr StringLiteral MemtileCol = "air.memtile_col";
} // namespace attrs

// Largest routing id a packet header can carry. Mirrors the AIE target hook
// getMaxPacketId(); the packet id field is 5 bits. Ids allocated for a demux
// are taken DOWNWARD from here so they cannot collide with the upward
// auto-assignment air-to-aie gives every other packet flow.
constexpr int kMaxPacketID = 31;

// Copy the DMA-steering / runtime-ordering markers
// (attrs::MemtileDmaChannelMin, RuntimeHoist, AwaitAppends, AppendBarrier,
// RefeedCount, PacketIDs, KeepPktHeader) that must survive channel-op
// re-instantiation from src to dst. Single source of truth for the marker set,
// so copy sites (Util::copyPaddingAttributes, ComposeMemrefOpOnChannelOp,
// SpecializeChannelBundlePattern) cannot diverge. Both ops must be live (call
// before erasing src).
void copyChannelSteeringAttrs(Operation *src, Operation *dst);

void registerAIRRtTranslations();

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  static constexpr StringLiteral name = "xilinx.air.async_token";
};

class UniverseType : public Type::TypeBase<UniverseType, Type, TypeStorage> {
public:
  using Base::Base;
  static constexpr StringLiteral name = "xilinx.air.universe";
};

// Adds a `air.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);
// Erases a `air.async.token` at position index of the argument list.
void eraseAsyncDependency(Operation *op, unsigned index);

// Collects ops transitively reachable from `root` via async-token use chains
// into `consumers`. Follows both op-result uses and (for LoopLikeOpInterface
// ops) the tied region iter_arg, so body ops are reached. `root` is excluded.
void walkAsyncTokenConsumers(Operation *root,
                             llvm::SetVector<Operation *> &consumers);

} // namespace air
} // namespace xilinx

#include "air/Dialect/AIR/AIRDialect.h.inc"
#include "air/Dialect/AIR/AIREnums.h.inc"
#include "air/Dialect/AIR/AIROpInterfaces.h.inc"

// include TableGen generated Attribute definitions
#define GET_ATTRDEF_CLASSES
#include "air/Dialect/AIR/AIRAttrs.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIR.h.inc"

namespace xilinx {
namespace air {
// True if the PRODUCING KERNEL writes the packet routing header into the
// payload, rather than the DMA stamping a static id onto the BD. Necessary for
// any data-dependent routing: a DMA-stamped channel carries one id on one BD
// and cannot select a destination per packet.
//
// Single source of truth. Both air-to-aie (which must not stamp such a BD) and
// air-annotate-packet-ids (which uses it to gate demux classification) read
// this; two copies drifted apart once already.
bool channelKernelWritesHeader(ChannelOp chanOp);
} // namespace air
} // namespace xilinx

#endif
