//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.herd_vectorize

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
        // Find air.herd operations and apply vectorization
        %herd_op = transform.structured.match ops{["air.herd"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %vectorized_herd = transform.air.herd_vectorize %herd_op
    }
}
