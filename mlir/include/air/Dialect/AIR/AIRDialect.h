//===- AIRDialect.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIR_DIALECT_H
#define MLIR_AIR_DIALECT_H

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
constexpr StringLiteral TileDmaChannel = "air.tile_dma_channel";
constexpr StringLiteral MemtileDmaChannelMin = "air.memtile_dma_channel_min";
constexpr StringLiteral DedicatedDmaChannel = "air.dedicated_dma_channel";
// Single-buffer count-free re-broadcast: N (>= 1) re-sends of one resident
// buffer per production. Authoritative carrier is the air.channel declaration;
// read via air::getRefeedCount. Verified on air.channel.
constexpr StringLiteral RefeedCount = "air.refeed_count";
// Opt-in front-end marker (unit attr) on an scf.for / affine.for whose body is
// a single loop-invariant air.channel.put: the loop re-sends one resident
// buffer once per iteration. The air-annotate-refeed pass reads its trip count
// into attrs::RefeedCount on the channel and collapses the loop.
constexpr StringLiteral RefeedLoop = "air.refeed_loop";
// User-pinned packet routing ids on an air.channel (channel_type
// "npu_dma_packet"). One packet_flow per id: N ids to a single dest converge on
// one buffer for a downstream demux hop; N ids to N dests route dest i with
// pinned[i]. The compute core writes the id into the payload header, so the DMA
// does not stamp/filter these -- the flows only install switchbox routes. Bare
// spelling matches the broadcast_shape discardable-attr convention on
// air.channel; read via air::ChannelOp::getPacketIDs. Verified on air.channel.
constexpr StringLiteral PacketIDs = "packet_ids";
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
} // namespace attrs

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

#endif
