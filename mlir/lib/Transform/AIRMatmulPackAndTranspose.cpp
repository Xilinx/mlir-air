//===- AIRMatmulPackAndTranspose.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulPackAndTranspose.h"
#include "air/Transform/AIRMatmulBufferizationPasses.h"
#include "air/Util/MatmulCodegenConfig.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <numeric>

#define DEBUG_TYPE "air-matmul-pack-and-transpose"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

// Apply pack_transpose to the producer of `linalgOp` operand `operandIdx`.
// Updates `linalgOp` in-place and returns the new linalg op on success.
static FailureOr<linalg::LinalgOp>
applyOperandTranspose(IRRewriter &rewriter, linalg::LinalgOp linalgOp,
                      int64_t operandIdx, ArrayRef<int64_t> outerPerm,
                      ArrayRef<int64_t> innerPerm) {
  if (outerPerm.empty() && innerPerm.empty())
    return linalgOp;
  Value operand = linalgOp->getOperand(operandIdx);
  auto packOp = operand.getDefiningOp<linalg::PackOp>();
  if (!packOp)
    return linalgOp->emitError() << "operand " << operandIdx
                                 << " is not produced by a linalg.pack op";
  // For an output operand, packTranspose also walks to the consumer unpack.
  linalg::UnPackOp maybeUnPack;
  if (operandIdx == (int64_t)linalgOp.getNumDpsInputs()) {
    for (auto user : linalgOp->getUsers()) {
      if (auto u = dyn_cast<linalg::UnPackOp>(user)) {
        maybeUnPack = u;
        break;
      }
    }
    if (!maybeUnPack)
      return linalgOp->emitError()
             << "output operand has no unpack consumer; cannot transpose";
  }
  auto res = linalg::packTranspose(rewriter, packOp, linalgOp, maybeUnPack,
                                   outerPerm, innerPerm);
  if (failed(res))
    return linalgOp->emitError() << "packTranspose failed for operand "
                                 << operandIdx;
  return cast<linalg::LinalgOp>(res->transposedLinalgOp.getOperation());
}

// Apply linalg::pack + per-operand pack_transpose to a single matmul.
static LogicalResult
runOnMatmul(linalg::LinalgOp matmulOp, ArrayRef<int64_t> packSizes,
            ArrayRef<int64_t> lhsOuter, ArrayRef<int64_t> lhsInner,
            ArrayRef<int64_t> rhsOuter, ArrayRef<int64_t> rhsInner,
            ArrayRef<int64_t> accOuter, ArrayRef<int64_t> accInner,
            StringRef marker) {
  IRRewriter rewriter(matmulOp.getContext());
  rewriter.setInsertionPoint(matmulOp);

  // Snapshot discardable attrs (e.g. air.matmul_codegen_config) before pack
  // rewrites the op into a new linalg.generic that doesn't inherit them.
  SmallVector<NamedAttribute> savedAttrs(
      matmulOp->getDiscardableAttrs().begin(),
      matmulOp->getDiscardableAttrs().end());

  // Build OpFoldResult sizes for linalg::pack.
  SmallVector<OpFoldResult> packed;
  packed.reserve(packSizes.size());
  for (int64_t s : packSizes)
    packed.push_back(rewriter.getIndexAttr(s));

  auto packResult = linalg::pack(rewriter, matmulOp, packed);
  if (failed(packResult))
    return matmulOp->emitError() << "linalg::pack failed";
  linalg::LinalgOp current = packResult->packedLinalgOp;

  // Per-operand transposes. Operand order on the packed op: 0=LHS, 1=RHS,
  // 2=accumulator (the only DPS init for matmul).
  auto step = [&](int64_t idx, ArrayRef<int64_t> outer,
                  ArrayRef<int64_t> inner) -> LogicalResult {
    auto res = applyOperandTranspose(rewriter, current, idx, outer, inner);
    if (failed(res))
      return failure();
    current = *res;
    return success();
  };
  if (failed(step(0, lhsOuter, lhsInner)))
    return failure();
  if (failed(step(1, rhsOuter, rhsInner)))
    return failure();
  if (failed(step(2, accOuter, accInner)))
    return failure();

  // Re-attach discardable attrs (the codegen config, etc.) to the final
  // packed/transposed op so downstream consumer passes can read them.
  for (NamedAttribute a : savedAttrs)
    if (!current->hasAttr(a.getName()))
      current->setAttr(a.getName(), a.getValue());

  if (!marker.empty())
    current->setAttr(marker, rewriter.getUnitAttr());
  return success();
}

class AIRMatmulPackAndTranspose
    : public impl::AIRMatmulPackAndTransposeBase<AIRMatmulPackAndTranspose> {

public:
  AIRMatmulPackAndTranspose() = default;
  AIRMatmulPackAndTranspose(const AIRMatmulPackAndTransposeOptions &opts)
      : AIRMatmulPackAndTransposeBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Find the first linalg.matmul; if none, fall back to the first
    // linalg.generic carrying the `packed_matmul` marker (= already-packed
    // matmul, eligible for a second pack level on M4 two-pack flow).
    linalg::LinalgOp target;
    func.walk([&](linalg::MatmulOp op) {
      target = cast<linalg::LinalgOp>(op.getOperation());
      return WalkResult::interrupt();
    });
    if (!target) {
      func.walk([&](linalg::GenericOp op) {
        if (op->hasAttr(clPackedMatmulMarker)) {
          target = cast<linalg::LinalgOp>(op.getOperation());
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
    if (!target) {
      // No matmul to pack; treat as a no-op (other passes may have already
      // packed it into a generic without the marker).
      return;
    }

    // Override pass-options from the codegen config when present (M3a).
    SmallVector<int64_t> packSizes(clPackSizes.begin(), clPackSizes.end());
    SmallVector<int64_t> lhsO(clLhsOuterPerm.begin(), clLhsOuterPerm.end());
    SmallVector<int64_t> lhsI(clLhsInnerPerm.begin(), clLhsInnerPerm.end());
    SmallVector<int64_t> rhsO(clRhsOuterPerm.begin(), clRhsOuterPerm.end());
    SmallVector<int64_t> rhsI(clRhsInnerPerm.begin(), clRhsInnerPerm.end());
    SmallVector<int64_t> accO(clAccOuterPerm.begin(), clAccOuterPerm.end());
    SmallVector<int64_t> accI(clAccInnerPerm.begin(), clAccInnerPerm.end());
    if (auto cfg = xilinx::air::findMatmulCodegenConfig(func)) {
      auto take = [&](StringRef key, SmallVector<int64_t> &dst) {
        auto v = xilinx::air::getI64Array(*cfg, key);
        if (!v.empty())
          dst = std::move(v);
      };
      take("pack_sizes", packSizes);
      take("lhs_outer_perm", lhsO);
      take("lhs_inner_perm", lhsI);
      take("rhs_outer_perm", rhsO);
      take("rhs_inner_perm", rhsI);
      take("acc_outer_perm", accO);
      take("acc_inner_perm", accI);
    }

    // Validate pack-sizes vs op iterator count. M2 first-pack expects 3
    // (matmul m,n,k); M4 second-pack on an already-packed op expects 6
    // (m,n,k outer + m,n,k inner) and may include zeros to leave outer
    // dims unpacked. Per-operand outer/inner rank is then determined by the
    // (already-packed) operand shape and the count of non-zero pack sizes
    // affecting that operand; rather than hand-validating, we let upstream
    // `linalg::packTranspose` enforce well-formedness when it runs.
    int64_t numIters = target.getNumLoops();
    if ((int64_t)packSizes.size() != numIters) {
      target->emitError() << "pack-sizes has " << packSizes.size()
                          << " entries; op has " << numIters
                          << " iterators";
      return signalPassFailure();
    }

    if (failed(runOnMatmul(target, packSizes, lhsO, lhsI, rhsO, rhsI, accO,
                           accI, clPackedMatmulMarker)))
      return signalPassFailure();

    // Optional tail step: bufferize the output linalg.pack into an L1 (or
    // configurable memory-space) allocation. Replaces the former standalone
    // `air-matmul-bufferize-l1-output` pass.
    if (clDoBufferizeL1Output) {
      IRRewriter rewriter(&getContext());
      if (failed(runBufferizeL1OutputImpl(func, clBufferizeL1OutputMemorySpace,
                                          clPackedMatmulMarker, rewriter)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulPackAndTransposePass() {
  return std::make_unique<AIRMatmulPackAndTranspose>();
}

std::unique_ptr<mlir::Pass> createAIRMatmulPackAndTransposePass(
    const AIRMatmulPackAndTransposeOptions &opts) {
  return std::make_unique<AIRMatmulPackAndTranspose>(opts);
}

} // namespace air
} // namespace xilinx
