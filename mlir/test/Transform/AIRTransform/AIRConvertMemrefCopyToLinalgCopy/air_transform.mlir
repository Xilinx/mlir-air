//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.convert_memref_copy_to_linalg_copy

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
        // Convert memref.copy to linalg.copy
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_op_updated = transform.air.convert_memref_copy_to_linalg_copy %func_op
    }
}
