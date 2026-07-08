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
constexpr StringLiteral RefeedCount = "air.refeed_count";
} // namespace attrs

// Copy the DMA-steering / runtime-ordering markers
// (attrs::MemtileDmaChannelMin, RuntimeHoist, AwaitAppends, AppendBarrier,
// RefeedCount) that
// must survive channel-op re-instantiation from src to dst. Single source of
// truth for the marker set, so copy sites (Util::copyPaddingAttributes,
// ComposeMemrefOpOnChannelOp) cannot diverge. Both ops must be live (call
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
