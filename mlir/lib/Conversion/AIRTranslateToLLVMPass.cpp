//===- AIRTranslateToLLVMPass.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Lower air.translate to memref-descriptor construction over a peer-rebased
// pointer.
//
// For each `air.translate %src, %from, %to, %bases`:
//   1. Extract the source memref's aligned pointer as `index`.
//   2. Read per-rank base addresses from the heap_bases memref:
//          from_base = bases[from]
//          to_base   = bases[to]
//      via memref.load (each element is an `index` — a pointer-width
//      integer).
//   3. Compute the peer aligned index:
//          peer_aligned = src_aligned + (to_base - from_base)
//   4. Materialize the peer aligned address as !llvm.ptr (needed only for
//      the descriptor build below — memref descriptors are LLVM structs).
//   5. Build a fresh LLVM memref descriptor (poison + insertvalue chain)
//      whose allocated/aligned pointers both reference the peer address;
//      offset = 0, sizes/strides come from the source memref's static type.
//   6. unrealized_conversion_cast the descriptor back to the result memref
//      type so downstream uses keep working through the standard
//      memref-to-llvm pipeline.
//
// Steps 1-3 use only memref + arith + index ops. The LLVM dialect appears
// only in steps 4-5 where it is unavoidable (memref descriptors *are* LLVM
// structs). The lowering is therefore valid both at host scope and inside
// `gpu.func` — the kernel just needs the heap_bases memref as an argument.
//
//===-----------------------------------------------------------------------===//

#include "air/Conversion/AIRTranslateToLLVMPass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;

namespace {

// Build a fresh LLVM memref descriptor for `memrefTy` whose
// allocated_ptr and aligned_ptr both reference `ptr`, offset is 0, and
// sizes/strides come from the static type (row-major).
//
// Mirrors buildMemrefDescriptor in AIRSymmetricAllocToMgpuPass.
static Value buildPeerDescriptor(OpBuilder &b, Location loc,
                                 MemRefType memrefTy, Value ptr) {
  ArrayRef<int64_t> shape = memrefTy.getShape();
  unsigned rank = shape.size();
  auto i64Ty = b.getI64Type();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());

  SmallVector<Type, 5> descFields;
  descFields.push_back(ptrTy);
  descFields.push_back(ptrTy);
  descFields.push_back(i64Ty);
  if (rank > 0) {
    descFields.push_back(LLVM::LLVMArrayType::get(i64Ty, rank));
    descFields.push_back(LLVM::LLVMArrayType::get(i64Ty, rank));
  }
  auto structTy = LLVM::LLVMStructType::getLiteral(b.getContext(), descFields);

  Value desc = LLVM::PoisonOp::create(b, loc, structTy);
  desc = LLVM::InsertValueOp::create(b, loc, desc, ptr, ArrayRef<int64_t>{0});
  desc = LLVM::InsertValueOp::create(b, loc, desc, ptr, ArrayRef<int64_t>{1});
  Value zero = LLVM::ConstantOp::create(b, loc, i64Ty, b.getI64IntegerAttr(0));
  desc = LLVM::InsertValueOp::create(b, loc, desc, zero, ArrayRef<int64_t>{2});

  if (rank > 0) {
    SmallVector<int64_t> strides(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];
    for (unsigned i = 0; i < rank; ++i) {
      Value sz = LLVM::ConstantOp::create(b, loc, i64Ty,
                                          b.getI64IntegerAttr(shape[i]));
      desc = LLVM::InsertValueOp::create(b, loc, desc, sz,
                                         ArrayRef<int64_t>{3, (int64_t)i});
      Value st = LLVM::ConstantOp::create(b, loc, i64Ty,
                                          b.getI64IntegerAttr(strides[i]));
      desc = LLVM::InsertValueOp::create(b, loc, desc, st,
                                         ArrayRef<int64_t>{4, (int64_t)i});
    }
  }
  return desc;
}

struct AIRTranslateToLLVMPass
    : public xilinx::air::impl::AIRTranslateToLLVMBase<AIRTranslateToLLVMPass> {

  AIRTranslateToLLVMPass() = default;
  AIRTranslateToLLVMPass(const AIRTranslateToLLVMPass &) {}

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    OpBuilder builder(ctx);
    auto i64Ty = builder.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    SmallVector<air::TranslateOp> translates;
    module.walk([&](air::TranslateOp op) { translates.push_back(op); });
    if (translates.empty())
      return;

    for (air::TranslateOp op : translates) {
      builder.setInsertionPoint(op);
      Location loc = op.getLoc();

      auto memrefTy = cast<MemRefType>(op.getSource().getType());
      if (!memrefTy.hasStaticShape()) {
        op.emitOpError("air.translate requires a static-shape source memref");
        signalPassFailure();
        return;
      }

      // Extract source aligned pointer (as index — pointer-width integer).
      Value srcAlignedIdx = memref::ExtractAlignedPointerAsIndexOp::create(
          builder, loc, op.getSource());

      // Load bases[from] / bases[to] as index values. Each element of the
      // heap_bases memref<?xindex> is a per-rank symmetric-heap base
      // address stored as a pointer-width integer.
      Value fromBaseIdx = memref::LoadOp::create(
          builder, loc, op.getHeapBases(), ValueRange{op.getFromRank()});
      Value toBaseIdx = memref::LoadOp::create(builder, loc, op.getHeapBases(),
                                               ValueRange{op.getToRank()});

      // peer_aligned_idx = srcAlignedIdx + (toBaseIdx - fromBaseIdx)
      Value diffIdx =
          arith::SubIOp::create(builder, loc, toBaseIdx, fromBaseIdx);
      Value peerAlignedIdx =
          arith::AddIOp::create(builder, loc, srcAlignedIdx, diffIdx);

      // Materialize as !llvm.ptr for the descriptor build below (the
      // descriptor's allocated/aligned-ptr fields are LLVM-typed because
      // memref descriptors are LLVM structs).
      Value peerAlignedI64 =
          arith::IndexCastOp::create(builder, loc, i64Ty, peerAlignedIdx);
      Value peerAlignedPtr =
          LLVM::IntToPtrOp::create(builder, loc, ptrTy, peerAlignedI64);

      // Build a fresh memref descriptor with the peer aligned pointer.
      Value desc = buildPeerDescriptor(builder, loc, memrefTy, peerAlignedPtr);
      Value newMemref = UnrealizedConversionCastOp::create(
                            builder, loc, TypeRange{memrefTy}, ValueRange{desc})
                            .getResult(0);

      op.getResult().replaceAllUsesWith(newMemref);
      op.erase();
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRTranslateToLLVMPass() {
  return std::make_unique<AIRTranslateToLLVMPass>();
}

} // namespace air
} // namespace xilinx
