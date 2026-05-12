//===- AIRCrossRankDmaToMgpuPass.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Lower air.dma_memcpy_nd ops carrying a `src_rank` or `dst_rank` integer
// attribute to host-side mgpuMemcpy calls with peer-VA addressing through
// mgpuGetHeapBases().
//
// Pattern emitted (for src_rank = R):
//   %size       = arith.constant <bytes> : i64
//   %nullptr    = llvm.mlir.zero : !llvm.ptr
//   %dst_ptr    = (extract aligned ptr from %dst memref)
//   %src_ptr    = (extract aligned ptr from %src memref)
//   %my_rank    = call @mgpuGetRank() : () -> i32
//   %bases      = call @mgpuGetHeapBases() : () -> !llvm.ptr
//   %my_base_at = llvm.getelementptr %bases[%my_rank] : ... -> !llvm.ptr, !llvm.ptr
//   %my_base    = llvm.load %my_base_at : !llvm.ptr -> !llvm.ptr
//   %src_int    = llvm.ptrtoint %src_ptr  : !llvm.ptr to i64
//   %my_base_int = llvm.ptrtoint %my_base : !llvm.ptr to i64
//   %offset     = arith.subi %src_int, %my_base_int : i64
//   %peer_base_at = llvm.getelementptr %bases[<R>] : ... -> !llvm.ptr, !llvm.ptr
//   %peer_base    = llvm.load %peer_base_at : !llvm.ptr -> !llvm.ptr
//   %peer_src     = llvm.getelementptr %peer_base[%offset] : ... -> !llvm.ptr, i8
//   call @mgpuMemcpy(%dst_ptr, %peer_src, %size, %nullptr)
//
// Initial restrictions:
//   - Both memrefs must have memory_space=0 (L3/global).
//   - Op must be at host scope (not inside a gpu.launch / gpu.func).
//   - "Entire memref" form only: empty offsets/sizes/strides on both sides.
//
//===-----------------------------------------------------------------------===//

#include "air/Conversion/AIRCrossRankDmaToMgpuPass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;

namespace {

// Ensure a private extern func declaration exists at module scope.
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

// Compute byte size of a static-shape memref as an i64 SSA value.
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

// Extract an aligned !llvm.ptr from a memref via the standard idiom.
static Value extractAlignedPtr(OpBuilder &b, Location loc, Value memref) {
  Value idx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, memref);
  Value i64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), idx);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  return LLVM::IntToPtrOp::create(b, loc, ptrTy, i64);
}

struct AIRCrossRankDmaToMgpuPass
    : public xilinx::air::impl::AIRCrossRankDmaToMgpuBase<
          AIRCrossRankDmaToMgpuPass> {

  AIRCrossRankDmaToMgpuPass() = default;
  AIRCrossRankDmaToMgpuPass(const AIRCrossRankDmaToMgpuPass &) {}

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module.getContext());
    auto i32Ty = builder.getI32Type();
    auto i64Ty = builder.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(module.getContext());

    // Collect cross-rank DMA ops.
    SmallVector<air::DmaMemcpyNdOp> crossRankDmas;
    module.walk([&](air::DmaMemcpyNdOp op) {
      if (op.hasCrossRank())
        crossRankDmas.push_back(op);
    });
    if (crossRankDmas.empty())
      return;

    // Declare the runtime ABI functions we may need.
    auto getRankFn = ensureExternFunc(module, builder, "mgpuGetRank",
                                       builder.getFunctionType({}, {i32Ty}));
    auto getBasesFn =
        ensureExternFunc(module, builder, "mgpuGetHeapBases",
                          builder.getFunctionType({}, {ptrTy}));
    auto memcpyFn = ensureExternFunc(
        module, builder, "mgpuMemcpy",
        builder.getFunctionType({ptrTy, ptrTy, i64Ty, ptrTy}, {}));

    for (air::DmaMemcpyNdOp dma : crossRankDmas) {
      Location loc = dma.getLoc();

      // Restrictions
      if (dma->getParentOfType<gpu::LaunchOp>() ||
          dma->getParentOfType<gpu::GPUFuncOp>()) {
        dma.emitOpError(
            "cross-rank DMA inside a GPU kernel is not yet supported");
        signalPassFailure();
        return;
      }
      if (!dma.getSrcOffsets().empty() || !dma.getSrcSizes().empty() ||
          !dma.getSrcStrides().empty() || !dma.getDstOffsets().empty() ||
          !dma.getDstSizes().empty() || !dma.getDstStrides().empty()) {
        dma.emitOpError("cross-rank DMA with explicit offsets/sizes/strides "
                        "is not yet supported");
        signalPassFailure();
        return;
      }

      auto srcType = cast<MemRefType>(dma.getSrcMemref().getType());
      auto dstType = cast<MemRefType>(dma.getDstMemref().getType());
      if (srcType.getMemorySpaceAsInt() != 0 ||
          dstType.getMemorySpaceAsInt() != 0) {
        dma.emitOpError(
            "cross-rank DMA requires both memrefs in memory_space=0");
        signalPassFailure();
        return;
      }

      // Determine which side has the rank attribute. (Only one is supported
      // per op for now.)
      bool srcIsPeer = dma.getSrcRank().has_value();
      bool dstIsPeer = dma.getDstRank().has_value();
      if (srcIsPeer && dstIsPeer) {
        dma.emitOpError(
            "cross-rank DMA with both src_rank and dst_rank set is not yet "
            "supported");
        signalPassFailure();
        return;
      }
      int64_t peerRank =
          srcIsPeer ? *dma.getSrcRank() : *dma.getDstRank();
      auto peerSideType = srcIsPeer ? srcType : dstType;
      Value peerMemref = srcIsPeer ? dma.getSrcMemref() : dma.getDstMemref();
      Value localMemref =
          srcIsPeer ? dma.getDstMemref() : dma.getSrcMemref();

      builder.setInsertionPoint(dma);
      Value sizeBytes = computeMemrefByteSize(builder, loc, peerSideType);
      if (!sizeBytes) {
        dma.emitOpError("cross-rank DMA requires static memref shape with "
                        "byte-aligned element type");
        signalPassFailure();
        return;
      }
      Value nullPtr = LLVM::ZeroOp::create(builder, loc, ptrTy);

      Value peerLocalPtr = extractAlignedPtr(builder, loc, peerMemref);
      Value localPtr = extractAlignedPtr(builder, loc, localMemref);

      // bases = mgpuGetHeapBases()
      Value bases = func::CallOp::create(builder, loc, getBasesFn, ValueRange{})
                       .getResult(0);

      // my_rank = mgpuGetRank() (i32 -> i64)
      Value myRankI32 =
          func::CallOp::create(builder, loc, getRankFn, ValueRange{})
              .getResult(0);
      Value myRankI64 = arith::ExtSIOp::create(builder, loc, i64Ty, myRankI32);

      // my_base = bases[my_rank]
      Value myBaseAddr = LLVM::GEPOp::create(builder, loc, ptrTy, ptrTy, bases,
                                              ArrayRef<Value>{myRankI64});
      Value myBase = LLVM::LoadOp::create(builder, loc, ptrTy, myBaseAddr);

      // peer_base = bases[<peerRank>]
      Value peerRankIdx = LLVM::ConstantOp::create(
          builder, loc, i64Ty, builder.getI64IntegerAttr(peerRank));
      Value peerBaseAddr = LLVM::GEPOp::create(
          builder, loc, ptrTy, ptrTy, bases, ArrayRef<Value>{peerRankIdx});
      Value peerBase = LLVM::LoadOp::create(builder, loc, ptrTy, peerBaseAddr);

      // offset = peerLocalPtr (as i64) - my_base (as i64)
      Value peerLocalInt =
          LLVM::PtrToIntOp::create(builder, loc, i64Ty, peerLocalPtr);
      Value myBaseInt = LLVM::PtrToIntOp::create(builder, loc, i64Ty, myBase);
      Value offset =
          arith::SubIOp::create(builder, loc, peerLocalInt, myBaseInt);

      // peer_ptr = peer_base + offset (byte-stride GEP)
      auto i8Ty = builder.getI8Type();
      Value peerPtr = LLVM::GEPOp::create(builder, loc, ptrTy, i8Ty, peerBase,
                                           ArrayRef<Value>{offset});

      // mgpuMemcpy(dst, src, size, nullptr) — substitute peerPtr on the
      // peer side.
      Value srcArg = srcIsPeer ? peerPtr : localPtr;
      Value dstArg = dstIsPeer ? peerPtr : localPtr;
      func::CallOp::create(builder, loc, memcpyFn,
                            ValueRange{dstArg, srcArg, sizeBytes, nullPtr});

      // If this DMA returned an async token, replace it with a wait_all.
      if (dma.getAsyncToken()) {
        Value tok = air::WaitAllOp::create(
                         builder, loc,
                         air::AsyncTokenType::get(builder.getContext()),
                         ValueRange{})
                        .getAsyncToken();
        dma.getAsyncToken().replaceAllUsesWith(tok);
      }
      dma.erase();
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRCrossRankDmaToMgpuPass() {
  return std::make_unique<AIRCrossRankDmaToMgpuPass>();
}

} // namespace air
} // namespace xilinx
