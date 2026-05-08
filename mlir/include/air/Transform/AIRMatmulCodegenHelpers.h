//===- AIRMatmulCodegenHelpers.h --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Free C++ entry points for the matmul codegen transformations originally
// defined as transform.air.* op apply() bodies in AIRLinalgCodegen.cpp.
// Both the existing transform ops and the new air-matmul-* C++ passes call
// these. New helpers are added here as their corresponding apply() body is
// migrated; until migrated, the apply() retains its original logic.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MATMUL_CODEGEN_HELPERS_H
#define AIR_MATMUL_CODEGEN_HELPERS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace xilinx {
namespace air {

//===----------------------------------------------------------------------===//
// Pure utilities used by multiple codegen helpers.
//===----------------------------------------------------------------------===//

/// True if two vector.transfer_read ops read the same memref location and
/// produce the same vector type.
bool areIdenticalReads(::mlir::vector::TransferReadOp read1,
                       ::mlir::vector::TransferReadOp read2);

/// True if any operation between `firstRead` and `secondRead` (in the same
/// block) writes to `firstRead`'s base memref.
bool hasWritesBetweenReads(::mlir::vector::TransferReadOp firstRead,
                           ::mlir::vector::TransferReadOp secondRead);

/// True if `val` transitively depends on `loopIV` via affine.apply or any
/// other defining op.
bool dependsOnLoopIV(::mlir::Value val, ::mlir::Value loopIV);

/// Recursively clone `op` and the chain of operand-producers that live
/// inside `loopOp` and don't depend on `loopIV`, mapping cloned values
/// through `mapping`. Operands defined outside `loopOp` are reused. Returns
/// the cloned result Value (or null if `op` produces no results).
::mlir::Value cloneOpAndOperands(::mlir::Operation *op, ::mlir::Value loopIV,
                                 ::mlir::scf::ForOp loopOp,
                                 ::mlir::RewriterBase &rewriter,
                                 ::mlir::IRMapping &mapping);

//===----------------------------------------------------------------------===//
// Free functions backing both transform.air.* ops and air-matmul-* passes.
//===----------------------------------------------------------------------===//

/// Greedily fold unit-extent dims in linalg ops on `funcOp`, using a
/// memref-aware collapse function (rank-reducing subview for strided memrefs).
::mlir::LogicalResult runFoldUnitExtentDimsOnFunc(::mlir::func::FuncOp funcOp);

/// Walk all vector.transfer_read in `target` and replace each pair of
/// identical reads with no intervening writes by the first read. Returns
/// the number of eliminations performed.
int runEliminateRedundantVectorTransfers(::mlir::Operation *target,
                                         ::mlir::RewriterBase &rewriter);

/// Replace vector-typed iter_args of `forOp` with their 1D-flattened form,
/// inserting vector.shape_cast at the loop entry/exit and inside the loop
/// body to convert back to the original shape. Returns the (possibly new)
/// scf.for, or `forOp` unchanged if there were no vector iter_args.
::mlir::FailureOr<::mlir::scf::ForOp>
runFlattenForIterArgs(::mlir::scf::ForOp forOp, ::mlir::RewriterBase &rewriter);

/// Iteratively hoist matched vector.transfer_read/write pairs whose indices
/// are loop-invariant out of `loopOp` (which must live inside `scopeOp`),
/// threading the accumulator through a new iter_arg. Returns the new loop.
::mlir::FailureOr<::mlir::scf::ForOp>
runHoistLoopInvariantTransfers(::mlir::Operation *scopeOp,
                               ::mlir::scf::ForOp loopOp,
                               ::mlir::RewriterBase &rewriter);

/// Hoist subview/affine.apply chains for vector transfer base pointers out
/// of `forOp` when they are loop-invariant. Returns the (possibly new)
/// scf.for via the rewriter; returns success/failure.
::mlir::LogicalResult
runHoistVectorTransferPointers(::mlir::scf::ForOp forOp,
                               ::mlir::RewriterBase &rewriter);

/// Cast vector-typed operands (at `inputIndices`) and/or vector-typed results
/// (at `outputIndices`) of `target` to `targetElementType`, then re-create
/// the op with the casted operand/result types. Empty index lists mean
/// "cast all inputs and outputs". Used for BFP16-mmul emulation: cast
/// vector.contract inputs to bf16 + accumulator/output to f32.
/// Returns success even when the op needs no change; returns failure on
/// validation errors (target has no vector types, etc).
::mlir::LogicalResult runVectorTypeCastOnTarget(
    ::mlir::Operation *target, ::mlir::Type targetElementType,
    ::llvm::ArrayRef<int64_t> inputIndices,
    ::llvm::ArrayRef<int64_t> outputIndices, ::mlir::RewriterBase &rewriter);

/// Hoist an extension/truncation pair surrounding a loop iter_arg out of
/// `loopOp`: extend the init value before the loop, change the iter_arg to
/// wide type, truncate the result after the loop. `extensionOp` must be
/// arith.extsi/extui/extf and `truncationOp` the matching truncation; both
/// must live inside `loopOp`. Returns the new scf.for on success.
::mlir::FailureOr<::mlir::scf::ForOp>
runHoistCastPair(::mlir::Operation *extensionOp,
                 ::mlir::Operation *truncationOp, ::mlir::scf::ForOp loopOp,
                 ::mlir::RewriterBase &rewriter);

//===----------------------------------------------------------------------===//
// Bufferization & fusion utilities used by the air-matmul-codegen
// orchestrator phases.
//===----------------------------------------------------------------------===//

/// Apply OptimizeCopyOpPattern to remove copies whose source is uninitialized
/// (or only filled), replacing them with linalg.fill. Operates greedily on
/// `funcOp`.
::mlir::LogicalResult runRemoveUninitializedCopy(::mlir::func::FuncOp funcOp);

/// Apply EliminateIntermediateMemrefPattern to collapse cascade memcpy
/// sequences (intermediate memref alloc + double copy) on `target`.
::mlir::LogicalResult runEliminateCascadeMemcpy(::mlir::Operation *target);

/// Apply ConvertMemrefCopyToLinalgCopyPattern: rewrite memref.copy to
/// linalg.copy on `target`. Required before tile-using-for of L3->L2 copies
/// (TilingInterface lives on linalg.copy, not memref.copy).
::mlir::LogicalResult
runConvertMemrefCopyToLinalgCopy(::mlir::Operation *target);

/// Tile-and-fuse `producerOp` (a LinalgOp with one DPS init) into the first
/// memref.subview use found inside `containingOp` (typically an scf.for/forall
/// body). Returns the tiled fused op on success, nullptr on failure.
::mlir::Operation *runFuseIntoContainingMemref(::mlir::Operation *producerOp,
                                               ::mlir::Operation *containingOp,
                                               ::mlir::RewriterBase &rewriter);

/// True iff `linalgOp`'s body contains exactly one non-terminator op and that
/// op is arith.truncf. Used to identify "truncf-only" linalg ops eligible for
/// fusion into their producer.
bool containsOnlyTruncfOp(::mlir::linalg::LinalgOp linalgOp);

/// True iff `producerOp` produces a single result that is consumed by
/// `truncfOp` as one of its DPS inputs.
bool producesResultForOp(::mlir::linalg::LinalgOp producerOp,
                         ::mlir::linalg::LinalgOp truncfOp);

/// Fuse a truncf-only linalg op into its producer. The fused op accumulates
/// in the producer's wide type but yields the truncated type. If inputs are
/// 2D+ (matmul-shaped), replace the fused generic with linalg.matmul of the
/// truncated output type and return that matmul; otherwise return the fused
/// generic. Both `producerOp` and `truncfOp` are erased.
::mlir::FailureOr<::mlir::Operation *>
runFuseTruncfLinalg(::mlir::linalg::LinalgOp producerOp,
                    ::mlir::linalg::LinalgOp truncfOp,
                    ::mlir::RewriterBase &rewriter);

/// Fold affine.apply ops into `forOp`'s lower/upper bounds via
/// xilinx::air::foldAffineApplyIntoLoopBounds. Returns the (possibly new)
/// scf.for, or `forOp` unchanged if the fold did not apply. AIR-only.
::mlir::scf::ForOp runNormalizeForBounds(::mlir::scf::ForOp forOp,
                                         ::mlir::RewriterBase &rewriter);

} // namespace air
} // namespace xilinx

#endif // AIR_MATMUL_CODEGEN_HELPERS_H
