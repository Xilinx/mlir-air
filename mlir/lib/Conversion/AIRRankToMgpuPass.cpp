//===- AIRRankToMgpuPass.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Lower air.rank to mgpu* runtime calls (multi-GPU process model).
//
// Each `air.rank` op is replaced by inlining its body in place, with rank
// IDs computed from `mgpuGetRank()` (delinearized into the rank's N-D
// iteration space) and rank sizes substituted from the static size operands.
//
// The pass also inserts `mgpuSymmetricHeapInit(heap_size)` at the entry of
// the enclosing `func.func` and `mgpuSymmetricHeapDestroy()` before each
// `func.return` in that function.
//
//===-----------------------------------------------------------------------===//

#include "air/Conversion/AIRRankToMgpuPass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;

namespace {

// Ensure a private extern func declaration exists at the top of the module.
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

struct AIRRankToMgpuPass
    : public xilinx::air::impl::AIRRankToMgpuBase<AIRRankToMgpuPass> {

  AIRRankToMgpuPass() = default;
  AIRRankToMgpuPass(const AIRRankToMgpuPass &pass) {}

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module.getContext());
    auto i32Ty = builder.getI32Type();
    auto i64Ty = builder.getI64Type();
    auto idxTy = builder.getIndexType();

    // Collect all air.rank ops and their parent functions.
    SmallVector<air::RankOp> rankOps;
    SetVector<func::FuncOp> rankParentFuncs;
    module.walk([&](air::RankOp op) {
      rankOps.push_back(op);
      if (auto fn = op->getParentOfType<func::FuncOp>())
        rankParentFuncs.insert(fn);
    });

    // If no air.rank ops exist, leave the module untouched.
    if (rankOps.empty())
      return;

    // Declare the mgpu* runtime ABI functions (only when needed).
    auto initFn = ensureExternFunc(module, builder, "mgpuSymmetricHeapInit",
                                   builder.getFunctionType({i64Ty}, {}));
    auto destroyFn =
        ensureExternFunc(module, builder, "mgpuSymmetricHeapDestroy",
                         builder.getFunctionType({}, {}));
    auto getRankFn = ensureExternFunc(module, builder, "mgpuGetRank",
                                      builder.getFunctionType({}, {i32Ty}));

    // For each parent function, insert mgpuSymmetricHeapInit at entry and
    // mgpuSymmetricHeapDestroy before each return.
    for (func::FuncOp fn : rankParentFuncs) {
      if (fn.empty())
        continue;
      Block &entry = fn.front();
      Location loc = fn.getLoc();
      builder.setInsertionPointToStart(&entry);
      // heapSize is uint64_t; preserve all 64 bits when materializing as
      // an i64 IntegerAttr (a static_cast<int64_t> on a value > INT64_MAX
      // would silently wrap to a negative literal in the IR).
      Value heapSizeVal = arith::ConstantOp::create(
          builder, loc, i64Ty,
          IntegerAttr::get(i64Ty, APInt(/*numBits=*/64, heapSize)));
      func::CallOp::create(builder, loc, initFn, ValueRange{heapSizeVal});

      // Insert destroy before every return op.
      SmallVector<func::ReturnOp> returns;
      fn.walk([&](func::ReturnOp r) { returns.push_back(r); });
      for (func::ReturnOp r : returns) {
        builder.setInsertionPoint(r);
        func::CallOp::create(builder, r.getLoc(), destroyFn, ValueRange{});
      }
    }

    // Lower each air.rank op.
    for (air::RankOp rankOp : rankOps) {
      builder.setInsertionPoint(rankOp);
      Location loc = rankOp.getLoc();

      // If the rank has async dependencies, insert a blocking wait before
      // proceeding.
      if (!rankOp.getAsyncDependencies().empty()) {
        air::WaitAllOp::create(builder, loc, Type{},
                               rankOp.getAsyncDependencies());
      }

      // Get the flat rank id from mgpuGetRank() and convert to index.
      Value rankI32 =
          func::CallOp::create(builder, loc, getRankFn, ValueRange{})
              .getResult(0);
      Value rankI64 = arith::ExtSIOp::create(builder, loc, i64Ty, rankI32);
      Value flatRank = arith::IndexCastOp::create(builder, loc, idxTy, rankI64);

      // Delinearize flatRank into N rank IDs using the static size operands.
      // For sizes [s0, s1, ..., sn-1]:
      //   id[0]   = flat % s0
      //   id[1]   = (flat / s0) % s1
      //   ...
      //   id[n-1] = flat / (s0 * s1 * ... * sn-2)
      auto sizeOpers = rankOp.getSizeOperands();
      unsigned n = rankOp.getNumDims();
      SmallVector<Value> ids(n);
      Value remaining = flatRank;
      for (unsigned d = 0; d < n; ++d) {
        if (d == n - 1) {
          ids[d] = remaining;
        } else {
          ids[d] =
              arith::RemSIOp::create(builder, loc, remaining, sizeOpers[d]);
          remaining =
              arith::DivSIOp::create(builder, loc, remaining, sizeOpers[d]);
        }
      }

      // Build remap and clone the body.
      IRMapping remap;
      for (unsigned d = 0; d < n; ++d) {
        remap.map(rankOp.getIds()[d], ids[d]);
        remap.map(rankOp.getSize()[d], sizeOpers[d]);
      }
      for (unsigned i = 0; i < rankOp.getNumKernelOperands(); ++i)
        remap.map(rankOp.getKernelArgument(i), rankOp.getKernelOperand(i));

      auto &ops = rankOp.getBody().front().getOperations();
      for (auto oi = ops.begin(), oe = --ops.end(); oi != oe; ++oi)
        builder.clone(*oi, remap);

      // Replace the async token (if any) with a synchronous wait_all.
      if (rankOp.getAsyncToken()) {
        auto waitAll = air::WaitAllOp::create(
            builder, loc, air::AsyncTokenType::get(builder.getContext()),
            ValueRange{});
        rankOp.getAsyncToken().replaceAllUsesWith(waitAll.getAsyncToken());
      }

      rankOp.erase();
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRRankToMgpuPass() {
  return std::make_unique<AIRRankToMgpuPass>();
}

} // namespace air
} // namespace xilinx
