//===- AIRGpuChannelToCachelinePass.cpp ------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Expand air.channel.put/get ops on channels of type "gpu_symmetric_heap"
// into the kernel-driven cache-line atomicity pattern (see
// test/gpu/multi_gpu/handwritten/cacheline.mlir for the reference shape):
//
//   put -> air.translate + cooperative memref.store (flag at lane 31)
//   get -> scf.while spin loop + gpu.shuffle idx broadcast of lane 31
//
//===-----------------------------------------------------------------------===//

#include "air/Conversion/AIRGpuChannelToCachelinePass.h"
#include "air/Conversion/GPUPassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;

namespace {

// Walk op->getParentOp() chain looking for an enclosing scf::IfOp whose
// condition is an `arith.cmpi eq, %v, %const : index` pattern. Returns
// the constant operand of the cmpi. Returns std::nullopt if no matching
// enclosing if is found.
static std::optional<int64_t> inferRankFromEnclosingIf(Operation *op) {
  Operation *cur = op->getParentOp();
  while (cur) {
    if (auto ifOp = dyn_cast<scf::IfOp>(cur)) {
      Value cond = ifOp.getCondition();
      if (auto cmp = cond.getDefiningOp<arith::CmpIOp>()) {
        if (cmp.getPredicate() == arith::CmpIPredicate::eq) {
          Value rhs = cmp.getRhs();
          if (auto cst = rhs.getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
              if (isa<IndexType>(intAttr.getType()))
                return intAttr.getInt();
            }
          }
          // also try lhs in case the user wrote the comparison flipped
          Value lhs = cmp.getLhs();
          if (auto cst = lhs.getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
              if (isa<IndexType>(intAttr.getType()))
                return intAttr.getInt();
            }
          }
        }
      }
    }
    cur = cur->getParentOp();
  }
  return std::nullopt;
}

// Walk herd.getKernelArguments() looking for the unique block arg whose
// type is `memref<?xindex, #air.symmetric_heap>` (the canonical heap_bases
// type). Returns nullptr on 0 or >1 matches.
static BlockArgument findUniqueBasesArg(air::HerdOp herd) {
  BlockArgument found;
  unsigned matchCount = 0;
  for (BlockArgument arg : herd.getKernelArguments()) {
    auto memrefTy = dyn_cast<MemRefType>(arg.getType());
    if (!memrefTy)
      continue;
    if (!isa<IndexType>(memrefTy.getElementType()))
      continue;
    if (!isa_and_nonnull<air::SymmetricHeapMemorySpaceAttr>(
            memrefTy.getMemorySpace()))
      continue;
    found = arg;
    ++matchCount;
  }
  if (matchCount != 1)
    return BlockArgument();
  return found;
}

// Verify the memref type is `memref<32xi32, #air.symmetric_heap>` (the
// initial-PR shape constraint).
static bool isCachelineMemref(Type ty) {
  auto memrefTy = dyn_cast<MemRefType>(ty);
  if (!memrefTy)
    return false;
  if (!memrefTy.hasStaticShape())
    return false;
  ArrayRef<int64_t> shape = memrefTy.getShape();
  if (shape.size() != 1 || shape[0] != 32)
    return false;
  if (!memrefTy.getElementType().isInteger(32))
    return false;
  if (!isa_and_nonnull<air::SymmetricHeapMemorySpaceAttr>(
          memrefTy.getMemorySpace()))
    return false;
  return true;
}

// Emit the put-side cacheline write inside the herd body, replacing
// the put op.
//
// Per the AIR compute model (PE = wavefront), the herd body executes once
// per PE; lanes within the PE are addressed via `gpu.lane_id`. So the
// cooperative cacheline write is across the 64 lanes of one PE, with
// lanes 0..30 writing payload and lane 31 writing the sync flag.
static void expandPutToCachelineWrite(OpBuilder &b, air::ChannelPutOp put,
                                      int64_t fromRank, int64_t toRank,
                                      Value bases) {
  Location loc = put.getLoc();
  b.setInsertionPoint(put);
  Value src = put.getSrc();
  auto memrefTy = cast<MemRefType>(src.getType());
  auto i32Ty = b.getI32Type();

  Value fromIdx = arith::ConstantOp::create(b, loc, b.getIndexAttr(fromRank));
  Value toIdx = arith::ConstantOp::create(b, loc, b.getIndexAttr(toRank));
  Value peer =
      air::TranslateOp::create(b, loc, memrefTy, src, fromIdx, toIdx, bases);

  Value c32 = arith::ConstantOp::create(b, loc, b.getIndexAttr(32));
  Value c31 = arith::ConstantOp::create(b, loc, b.getIndexAttr(31));
  Value c1I32 =
      arith::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(1));

  // Lane id within the PE (0..wavesize-1). gpu.lane_id returns Index.
  Value lane = gpu::LaneIdOp::create(b, loc, /*upperBound=*/IntegerAttr{});

  Value active =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, lane, c32);

  auto ifOp = scf::IfOp::create(b, loc, active, /*withElseRegion=*/false);
  OpBuilder thenB = ifOp.getThenBodyBuilder();
  Value payload = memref::LoadOp::create(thenB, loc, src, ValueRange{lane});
  Value isFlag =
      arith::CmpIOp::create(thenB, loc, arith::CmpIPredicate::eq, lane, c31);
  Value val = arith::SelectOp::create(thenB, loc, isFlag, c1I32, payload);
  memref::StoreOp::create(thenB, loc, val, peer, ValueRange{lane});

  put.erase();
}

// Emit the get-side spin loop inside the herd body, replacing the get op.
//
// Shape mirrors the upstream MLIR pattern for cross-kernel polling
// (mlir/test/Integration/GPU/CUDA/concurrent-kernels.mlir): a zero-result
// `scf.while` whose body uses `memref.atomic_rmw` for the observable read.
// `addi %c0` is functionally a load (returns the prior value, adds 0) but
// the atomic_rmw op carries both Read and Write memory effects, which:
//   1. Prevent the greedy rewriter in `air-to-rocdl` from DCEing the spin
//      (the Write effect breaks the all-reads check in
//      `wouldOpBeTriviallyDead`).
//   2. Encode "this read must be observable across producers" in the IR,
//      so subsequent lowering passes don't need to be trusted to keep a
//      plain `memref.load` observable.
//
// Per the AIR compute model (PE = wavefront), lane index inside the PE
// comes from `gpu.lane_id`; the gpu.shuffle width matches the wavefront.
static void expandGetToCachelineSpin(OpBuilder &b, air::ChannelGetOp get) {
  Location loc = get.getLoc();
  b.setInsertionPoint(get);
  Value dst = get.getDst();
  auto i32Ty = b.getI32Type();

  Value c32 = arith::ConstantOp::create(b, loc, b.getIndexAttr(32));
  Value c0I32 =
      arith::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(0));
  Value c1I32 =
      arith::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(1));
  Value c31I32 =
      arith::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(31));
  Value c64I32 =
      arith::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(64));

  Value lane = gpu::LaneIdOp::create(b, loc, /*upperBound=*/IntegerAttr{});
  Value active =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, lane, c32);

  scf::WhileOp::create(
      b, loc, TypeRange{}, ValueRange{},
      /*beforeBuilder=*/
      [&](OpBuilder &beforeB, Location l, ValueRange beforeArgs) {
        auto innerIf = scf::IfOp::create(beforeB, l, TypeRange{i32Ty}, active,
                                         /*withElseRegion=*/true);
        {
          OpBuilder thenB = innerIf.getThenBodyBuilder();
          Value loaded = memref::AtomicRMWOp::create(
              thenB, l, i32Ty, arith::AtomicRMWKind::addi, c0I32, dst,
              ValueRange{lane});
          scf::YieldOp::create(thenB, l, ValueRange{loaded});
        }
        {
          OpBuilder elseB = innerIf.getElseBodyBuilder();
          scf::YieldOp::create(elseB, l, ValueRange{c0I32});
        }
        Value v = innerIf.getResult(0);
        auto shuffle = gpu::ShuffleOp::create(beforeB, l, v, c31I32, c64I32,
                                              gpu::ShuffleMode::IDX);
        Value flag = shuffle.getResult(0);
        Value notReady = arith::CmpIOp::create(
            beforeB, l, arith::CmpIPredicate::ne, flag, c1I32);
        scf::ConditionOp::create(beforeB, l, notReady, ValueRange{});
      },
      /*afterBuilder=*/
      [&](OpBuilder &afterB, Location l, ValueRange afterArgs) {
        scf::YieldOp::create(afterB, l, ValueRange{});
      });

  get.erase();
}

struct AIRGpuChannelToCachelinePass
    : public xilinx::air::impl::AIRGpuChannelToCachelineBase<
          AIRGpuChannelToCachelinePass> {

  AIRGpuChannelToCachelinePass() = default;
  AIRGpuChannelToCachelinePass(const AIRGpuChannelToCachelinePass &) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // 1. Collect gpu_symmetric_heap channels.
    SmallVector<air::ChannelOp> chans;
    module.walk([&](air::ChannelOp c) {
      if (c.getChannelType() == "gpu_symmetric_heap")
        chans.push_back(c);
    });

    for (air::ChannelOp chan : chans) {
      // 2. Pair via existing util. Initial scope: exactly 1 put + 1 get.
      auto puts = air::getChannelPutOpThroughSymbol(chan);
      auto gets = air::getChannelGetOpThroughSymbol(chan);
      if (puts.size() != 1 || gets.size() != 1) {
        chan.emitOpError("air-gpu-channel-to-cacheline initial scope "
                         "supports exactly one put and one get per channel; "
                         "got ")
            << puts.size() << " puts and " << gets.size() << " gets";
        return signalPassFailure();
      }
      air::ChannelPutOp put = puts.front();
      air::ChannelGetOp get = gets.front();

      // 3. Verify both inside air.herd.
      auto putHerd = put->getParentOfType<air::HerdOp>();
      auto getHerd = get->getParentOfType<air::HerdOp>();
      if (!putHerd) {
        put.emitOpError("must be enclosed by an air.herd op for the "
                        "cacheline lowering");
        return signalPassFailure();
      }
      if (!getHerd) {
        get.emitOpError("must be enclosed by an air.herd op for the "
                        "cacheline lowering");
        return signalPassFailure();
      }

      // 4. Verify memref shape.
      if (!isCachelineMemref(put.getSrc().getType())) {
        put.emitOpError("initial-PR scope requires src memref type "
                        "`memref<32xi32, #air.symmetric_heap>`");
        return signalPassFailure();
      }
      if (!isCachelineMemref(get.getDst().getType())) {
        get.emitOpError("initial-PR scope requires dst memref type "
                        "`memref<32xi32, #air.symmetric_heap>`");
        return signalPassFailure();
      }

      // 5. Infer ranks from enclosing scf.if conditions.
      auto producerRank = inferRankFromEnclosingIf(put);
      if (!producerRank) {
        put.emitOpError("could not infer producer rank from rank-dispatch "
                        "context; expected enclosing `scf.if (arith.cmpi "
                        "eq, %rid, %const : index)`");
        return signalPassFailure();
      }
      auto consumerRank = inferRankFromEnclosingIf(get);
      if (!consumerRank) {
        get.emitOpError("could not infer consumer rank from rank-dispatch "
                        "context; expected enclosing `scf.if (arith.cmpi "
                        "eq, %rid, %const : index)`");
        return signalPassFailure();
      }

      // 6. Find %bases inside the put's herd. Get-side currently unused
      // because the get spin reads from %dst (the consumer's local view of
      // the symmetric heap region the producer publishes to).
      BlockArgument putBases = findUniqueBasesArg(putHerd);
      if (!putBases) {
        putHerd.emitOpError("no `memref<?xindex, #air.symmetric_heap>` arg "
                            "in herd; please thread `%bases` through "
                            "air.launch/segment/herd args() so the put "
                            "expansion can address peer ranks");
        return signalPassFailure();
      }

      // Per AIR compute model (PE = wavefront), the herd body runs once
      // per PE; lane index within the PE comes from `gpu.lane_id` inside
      // the expansion, not from the herd tile ids.
      expandPutToCachelineWrite(builder, put, *producerRank, *consumerRank,
                                putBases);
      expandGetToCachelineSpin(builder, get);
      chan.erase();
    }
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRGpuChannelToCachelinePass() {
  return std::make_unique<AIRGpuChannelToCachelinePass>();
}

} // namespace air
} // namespace xilinx
