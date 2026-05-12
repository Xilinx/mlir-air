//===- AIRGpuChannelToMgpuPass.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Lower air.channel of channel_type="gpu_symmetric_heap" plus its put/get
// pair to host-side mgpuMemcpy with peer-VA addressing through
// mgpuGetHeapBases(), with mgpuBarrier-based synchronization.
//
// Per channel:
//   - put becomes mgpuBarrier() (publish — the data is already in the
//     symmetric heap via the put's air.symmetric source memref)
//   - get becomes mgpuBarrier() followed by mgpuMemcpy(dst, peer_va(src), sz)
//     where the peer rank is the get's first index operand
//
//===-----------------------------------------------------------------------===//

#include "air/Conversion/AIRGpuChannelToMgpuPass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;

namespace {

static func::FuncOp ensureExternFunc(ModuleOp module, OpBuilder &builder,
                                     StringRef name, FunctionType type) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fn = func::FuncOp::create(builder, module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

static Value computeMemrefByteSize(OpBuilder &b, Location loc, MemRefType ty) {
  if (!ty.hasStaticShape())
    return nullptr;
  int64_t numElts = 1;
  for (int64_t d : ty.getShape())
    numElts *= d;
  unsigned eltBits = ty.getElementType().getIntOrFloatBitWidth();
  if (eltBits == 0 || (eltBits % 8) != 0)
    return nullptr;
  int64_t totalBytes = numElts * (eltBits / 8);
  return arith::ConstantOp::create(b, loc, b.getI64Type(),
                                   b.getI64IntegerAttr(totalBytes));
}

static Value extractAlignedPtr(OpBuilder &b, Location loc, Value memref) {
  Value idx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, memref);
  Value i64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), idx);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  return LLVM::IntToPtrOp::create(b, loc, ptrTy, i64);
}

struct AIRGpuChannelToMgpuPass
    : public xilinx::air::impl::AIRGpuChannelToMgpuBase<
          AIRGpuChannelToMgpuPass> {

  AIRGpuChannelToMgpuPass() = default;
  AIRGpuChannelToMgpuPass(const AIRGpuChannelToMgpuPass &) {}

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module.getContext());
    auto i32Ty = builder.getI32Type();
    auto i64Ty = builder.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(module.getContext());

    // Collect gpu_symmetric_heap channel decls and their put/get sites.
    SmallVector<air::ChannelOp> chans;
    module.walk([&](air::ChannelOp ch) {
      if (ch.getChannelType() == "gpu_symmetric_heap")
        chans.push_back(ch);
    });
    if (chans.empty())
      return;

    auto getRankFn = ensureExternFunc(module, builder, "mgpuGetRank",
                                       builder.getFunctionType({}, {i32Ty}));
    auto getBasesFn =
        ensureExternFunc(module, builder, "mgpuGetHeapBases",
                          builder.getFunctionType({}, {ptrTy}));
    auto memcpyFn = ensureExternFunc(
        module, builder, "mgpuMemcpy",
        builder.getFunctionType({ptrTy, ptrTy, i64Ty, ptrTy}, {}));
    auto barrierFn = ensureExternFunc(
        module, builder, "mgpuBarrier", builder.getFunctionType({}, {}));

    for (air::ChannelOp ch : chans) {
      StringAttr sym = ch.getSymNameAttr();

      // Find puts and gets that reference this channel symbol.
      SmallVector<air::ChannelPutOp> puts;
      SmallVector<air::ChannelGetOp> gets;
      module.walk([&](air::ChannelPutOp p) {
        if (p.getChanName() == sym.getValue())
          puts.push_back(p);
      });
      module.walk([&](air::ChannelGetOp g) {
        if (g.getChanName() == sym.getValue())
          gets.push_back(g);
      });

      if (puts.size() != 1 || gets.size() != 1) {
        ch.emitOpError()
            << "channel_type=\"gpu_symmetric_heap\" requires exactly one "
               "put and one get per channel; found "
            << puts.size() << " put(s), " << gets.size() << " get(s)";
        signalPassFailure();
        return;
      }
      air::ChannelPutOp put = puts.front();
      air::ChannelGetOp get = gets.front();

      // Restrictions
      if (put->getParentOfType<gpu::LaunchOp>() ||
          put->getParentOfType<gpu::GPUFuncOp>() ||
          get->getParentOfType<gpu::LaunchOp>() ||
          get->getParentOfType<gpu::GPUFuncOp>()) {
        ch.emitOpError("gpu_symmetric_heap put/get inside a GPU kernel is "
                       "not yet supported");
        signalPassFailure();
        return;
      }
      if (!put.getSrcOffsets().empty() || !put.getSrcSizes().empty() ||
          !put.getSrcStrides().empty() || !get.getDstOffsets().empty() ||
          !get.getDstSizes().empty() || !get.getDstStrides().empty()) {
        ch.emitOpError("gpu_symmetric_heap put/get with explicit "
                       "offsets/sizes/strides is not yet supported");
        signalPassFailure();
        return;
      }

      auto srcType = cast<MemRefType>(put.getSrc().getType());
      auto dstType = cast<MemRefType>(get.getDst().getType());
      if (srcType.getMemorySpaceAsInt() != 0 ||
          dstType.getMemorySpaceAsInt() != 0) {
        ch.emitOpError(
            "gpu_symmetric_heap put/get requires both memrefs in memory_space=0");
        signalPassFailure();
        return;
      }

      // The put's source must be air.symmetric so peers can read it.
      if (auto allocOp = put.getSrc().getDefiningOp<memref::AllocOp>())
        if (!allocOp->hasAttr("air.symmetric")) {
          ch.emitOpError("gpu_symmetric_heap put requires a memref.alloc "
                         "carrying the \"air.symmetric\" attribute");
          signalPassFailure();
          return;
        }

      if (get.getIndices().size() != 1) {
        ch.emitOpError("gpu_symmetric_heap get requires exactly one index "
                       "operand (the peer rank)");
        signalPassFailure();
        return;
      }
      Value peerRankIdx = get.getIndices().front();

      // ---- Lower put: emit barrier (publish) and erase ----
      Location putLoc = put.getLoc();
      builder.setInsertionPointAfter(put);
      func::CallOp::create(builder, putLoc, barrierFn, ValueRange{});
      if (put.getAsyncToken()) {
        Value tok = air::WaitAllOp::create(
                         builder, putLoc,
                         air::AsyncTokenType::get(builder.getContext()),
                         ValueRange{})
                        .getAsyncToken();
        put.getAsyncToken().replaceAllUsesWith(tok);
      }
      put.erase();

      // ---- Lower get: barrier + cross-rank mgpuMemcpy(dst, peer_va(src), sz) ----
      Location getLoc = get.getLoc();
      builder.setInsertionPoint(get);

      // Barrier (consume)
      func::CallOp::create(builder, getLoc, barrierFn, ValueRange{});

      Value sizeBytes = computeMemrefByteSize(builder, getLoc, srcType);
      if (!sizeBytes) {
        ch.emitOpError("gpu_symmetric_heap requires static memref shape");
        signalPassFailure();
        return;
      }
      Value nullPtr = LLVM::ZeroOp::create(builder, getLoc, ptrTy);

      Value srcLocalPtr = extractAlignedPtr(builder, getLoc, put.getSrc());
      Value dstLocalPtr = extractAlignedPtr(builder, getLoc, get.getDst());

      Value bases =
          func::CallOp::create(builder, getLoc, getBasesFn, ValueRange{})
              .getResult(0);
      Value myRankI32 =
          func::CallOp::create(builder, getLoc, getRankFn, ValueRange{})
              .getResult(0);
      Value myRankI64 =
          arith::ExtSIOp::create(builder, getLoc, i64Ty, myRankI32);
      Value myBaseAddr = LLVM::GEPOp::create(builder, getLoc, ptrTy, ptrTy,
                                              bases, ArrayRef<Value>{myRankI64});
      Value myBase = LLVM::LoadOp::create(builder, getLoc, ptrTy, myBaseAddr);

      // Peer rank: convert dynamic index operand to i64.
      Value peerRankI64;
      Type peerTy = peerRankIdx.getType();
      if (isa<IndexType>(peerTy))
        peerRankI64 = arith::IndexCastOp::create(builder, getLoc, i64Ty,
                                                  peerRankIdx);
      else if (auto intTy = dyn_cast<IntegerType>(peerTy)) {
        if (intTy.getWidth() == 64)
          peerRankI64 = peerRankIdx;
        else
          peerRankI64 =
              arith::ExtSIOp::create(builder, getLoc, i64Ty, peerRankIdx);
      } else {
        ch.emitOpError("gpu_symmetric_heap get peer-rank index must be index "
                       "or integer type");
        signalPassFailure();
        return;
      }

      Value peerBaseAddr = LLVM::GEPOp::create(
          builder, getLoc, ptrTy, ptrTy, bases, ArrayRef<Value>{peerRankI64});
      Value peerBase =
          LLVM::LoadOp::create(builder, getLoc, ptrTy, peerBaseAddr);

      Value srcLocalInt =
          LLVM::PtrToIntOp::create(builder, getLoc, i64Ty, srcLocalPtr);
      Value myBaseInt =
          LLVM::PtrToIntOp::create(builder, getLoc, i64Ty, myBase);
      Value offset =
          arith::SubIOp::create(builder, getLoc, srcLocalInt, myBaseInt);

      auto i8Ty = builder.getI8Type();
      Value peerSrc = LLVM::GEPOp::create(builder, getLoc, ptrTy, i8Ty,
                                           peerBase, ArrayRef<Value>{offset});

      func::CallOp::create(
          builder, getLoc, memcpyFn,
          ValueRange{dstLocalPtr, peerSrc, sizeBytes, nullPtr});

      if (get.getAsyncToken()) {
        Value tok = air::WaitAllOp::create(
                         builder, getLoc,
                         air::AsyncTokenType::get(builder.getContext()),
                         ValueRange{})
                        .getAsyncToken();
        get.getAsyncToken().replaceAllUsesWith(tok);
      }
      get.erase();

      // The channel symbol can now be erased.
      ch.erase();
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRGpuChannelToMgpuPass() {
  return std::make_unique<AIRGpuChannelToMgpuPass>();
}

} // namespace air
} // namespace xilinx
