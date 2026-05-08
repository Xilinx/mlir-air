//===- MatmulCodegenConfig.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Carrier attribute + reader/writer helpers for the matmul codegen pipeline.
// External producers (autotuners, future heuristic passes) write the
// attribute on each linalg.matmul (or marker-attributed LinalgOp). The
// air-matmul-codegen orchestrator currently does NOT read this attribute
// (per-phase options are passed explicitly by the caller); this header
// remains so the schema and helpers are available to the future heuristic.
// The attribute is a `DictionaryAttr` named "air.matmul_codegen_config"
// with the following keys (any field may be missing):
//
//   tile_l3_l2_k      : i64
//   pack_sizes        : ArrayAttr<i64>     (length 3)
//   lhs_outer_perm    : ArrayAttr<i64>     (length 2; e.g. [1,0])
//   lhs_inner_perm    : ArrayAttr<i64>
//   rhs_outer_perm    : ArrayAttr<i64>
//   rhs_inner_perm    : ArrayAttr<i64>
//   acc_outer_perm    : ArrayAttr<i64>
//   acc_inner_perm    : ArrayAttr<i64>
//   tile_k_factor     : i64
//   tile_cores        : ArrayAttr<i64>
//   prologue_tile     : ArrayAttr<i64>
//   epilogue_tile     : ArrayAttr<i64>
//   fill_iter_perm    : ArrayAttr<i64>
//   vector_tile       : ArrayAttr<i64>     (length 6 for packed matmul)
//   vector_unroll_tile: ArrayAttr<i64>
//   vector_unroll_factor : i64
//   fill_vector_tile  : ArrayAttr<i64>
//   bfp16_emulation             : bool
//   fuse_output_truncf          : bool
//   bf16_output_hoist_pairs     : bool
//   three_herd_prologue_epilogue: bool
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_MATMUL_CODEGEN_CONFIG_H
#define AIR_UTIL_MATMUL_CODEGEN_CONFIG_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx {
namespace air {

/// Discardable attribute name on the linalg.matmul (or its packed marker
/// successor) carrying the codegen config dictionary.
inline llvm::StringRef getMatmulCodegenConfigAttrName() {
  return "air.matmul_codegen_config";
}

/// Find the codegen-config DictionaryAttr in `funcOp`. Looks for the first op
/// in the function carrying `getMatmulCodegenConfigAttrName()`. Returns the
/// dict (possibly empty) on success, std::nullopt if no config is attached.
std::optional<::mlir::DictionaryAttr>
findMatmulCodegenConfig(::mlir::func::FuncOp funcOp);

/// Helper: extract an `ArrayAttr<i64>` field from `cfg` as
/// `SmallVector<int64_t>`. Returns an empty vector if the field is missing or
/// the wrong type.
::llvm::SmallVector<int64_t> getI64Array(::mlir::DictionaryAttr cfg,
                                         ::llvm::StringRef key);

/// Helper: extract an i64 field from `cfg`. Returns `defaultVal` if missing.
int64_t getI64(::mlir::DictionaryAttr cfg, ::llvm::StringRef key,
               int64_t defaultVal);

/// Helper: extract a bool field from `cfg`. Returns `defaultVal` if missing.
bool getBool(::mlir::DictionaryAttr cfg, ::llvm::StringRef key,
             bool defaultVal);

/// Build (and write) a DictionaryAttr config onto the first linalg.matmul (or
/// op marked `markerName`) in `funcOp`. Existing entries in `dict` overwrite
/// any prior config. Returns true if an op was found and the attribute was
/// written; false otherwise.
bool writeMatmulCodegenConfig(::mlir::func::FuncOp funcOp,
                              ::mlir::DictionaryAttr dict,
                              ::llvm::StringRef markerName = "");

/// Build a DictionaryAttr from a list of (name, attr) pairs, dropping any
/// entries with null attrs. Convenience wrapper around DictionaryAttr::get.
::mlir::DictionaryAttr
buildMatmulCodegenConfig(::mlir::MLIRContext *ctx,
                         ::llvm::ArrayRef<::mlir::NamedAttribute> entries);

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_MATMUL_CODEGEN_CONFIG_H
