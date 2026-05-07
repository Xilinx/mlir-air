//===- AIRLinalgBufferize.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_LINALG_BUFFERIZE_H
#define AIR_LINALG_BUFFERIZE_H

#include "air/Transform/PassDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass>
createAIRresolveTensorOpOperandConflictsWithNewTensors();

/// Hoist statically-bound `memref.alloc` ops out of nested loops into the
/// function entry block. Wrapper around the file-scope template
/// `hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>`. Used both by
/// `transform.air.hoist_static_alloc` (single-shot) and the
/// `air-hoist-static-alloc` pass.
void hoistStaticAllocsInFunc(::mlir::RewriterBase &rewriter,
                             ::mlir::FunctionOpInterface funcOp);

} // namespace air
} // namespace xilinx

#endif // AIR_LINALG_BUFFERIZE_H
