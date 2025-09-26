//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.transpose_reduce

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
        // Find linalg.reduce operations and apply transpose optimization
        %reduce_op = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %transformed_reduce = transform.air.transpose_reduce %reduce_op
    }
}
