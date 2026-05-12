//===- AIRSymmetricAllocToMgpuPass.cpp -------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Lower memref.alloc carrying the `air.symmetric` attribute to a call to the
// runtime function `mgpuSymmetricAlloc`. The returned `!llvm.ptr` is wrapped
// in an LLVM memref descriptor (struct) and projected back to the original
// memref type via `builtin.unrealized_conversion_cast` so that downstream
// uses keep working.
//
// `memref.dealloc` ops whose operand traces (through a single
// `unrealized_conversion_cast`) back to a symmetric alloc are rewritten to
// `mgpuSymmetricFree`.
//
//===-----------------------------------------------------------------------===//

#include "air/Conversion/AIRSymmetricAllocToMgpuPass.h"
#include "air/Conversion/GPUPassDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

// Compute the byte size of a static-shaped memref as an i64 SSA value.
// Returns nullptr if the memref is dynamically shaped.
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

// Build an LLVM memref descriptor struct populated with the given pointer.
// For now we support only static-shape, contiguous, identity-layout memrefs
// without an offset. For dimensions: sizes from the type, strides as
// row-major (innermost stride = 1).
static Value buildMemrefDescriptor(OpBuilder &b, Location loc,
                                   MemRefType memrefTy, Value ptr) {
  ArrayRef<int64_t> shape = memrefTy.getShape();
  unsigned rank = shape.size();
  auto i64Ty = b.getI64Type();
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());

  // Build the descriptor type: !llvm.struct<(ptr, ptr, i64, array<R x i64>,
  // array<R x i64>)>. For rank-0 memrefs, MLIR omits the size/stride arrays.
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
    // Compute row-major strides from shape (innermost = 1).
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

struct AIRSymmetricAllocToMgpuPass
    : public xilinx::air::impl::AIRSymmetricAllocToMgpuBase<
          AIRSymmetricAllocToMgpuPass> {

  AIRSymmetricAllocToMgpuPass() = default;
  AIRSymmetricAllocToMgpuPass(const AIRSymmetricAllocToMgpuPass &) {}

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module.getContext());
    auto i64Ty = builder.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(module.getContext());

    // Collect symmetric allocs.
    SmallVector<memref::AllocOp> symAllocs;
    module.walk([&](memref::AllocOp op) {
      if (op->hasAttr("air.symmetric"))
        symAllocs.push_back(op);
    });

    if (symAllocs.empty())
      return;

    auto allocFn = ensureExternFunc(
        module, builder, "mgpuSymmetricAlloc",
        builder.getFunctionType({i64Ty, ptrTy}, {ptrTy}));
    auto freeFn = ensureExternFunc(
        module, builder, "mgpuSymmetricFree",
        builder.getFunctionType({ptrTy, ptrTy}, {}));

    // Track the !llvm.ptr backing each lowered memref so deallocs can look
    // them up.
    DenseMap<Value, Value> symmetricMemrefToPtr;

    for (memref::AllocOp alloc : symAllocs) {
      auto memrefTy = alloc.getType();
      Location loc = alloc.getLoc();
      builder.setInsertionPoint(alloc);

      Value sizeBytes = computeMemrefByteSize(builder, loc, memrefTy);
      if (!sizeBytes) {
        alloc.emitOpError(
            "air.symmetric memref.alloc requires a static-shape memref with "
            "byte-aligned element type");
        signalPassFailure();
        return;
      }
      Value nullPtr = LLVM::ZeroOp::create(builder, loc, ptrTy);
      Value ptr = func::CallOp::create(builder, loc, allocFn,
                                        ValueRange{sizeBytes, nullPtr})
                       .getResult(0);

      Value desc = buildMemrefDescriptor(builder, loc, memrefTy, ptr);
      Value newMemref = UnrealizedConversionCastOp::create(
                            builder, loc, TypeRange{memrefTy}, ValueRange{desc})
                            .getResult(0);
      symmetricMemrefToPtr[newMemref] = ptr;
      alloc.getResult().replaceAllUsesWith(newMemref);
      alloc.erase();
    }

    // Lower deallocs whose operand traces back to a symmetric alloc.
    SmallVector<memref::DeallocOp> deallocs;
    module.walk([&](memref::DeallocOp op) { deallocs.push_back(op); });
    for (memref::DeallocOp d : deallocs) {
      Value src = d.getMemref();
      auto it = symmetricMemrefToPtr.find(src);
      if (it == symmetricMemrefToPtr.end())
        continue; // not a symmetric memref
      builder.setInsertionPoint(d);
      Value nullPtr = LLVM::ZeroOp::create(builder, d.getLoc(), ptrTy);
      func::CallOp::create(builder, d.getLoc(), freeFn,
                            ValueRange{it->second, nullPtr});
      d.erase();
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRSymmetricAllocToMgpuPass() {
  return std::make_unique<AIRSymmetricAllocToMgpuPass>();
}

} // namespace air
} // namespace xilinx
