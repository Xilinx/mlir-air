//===- air_transform_loop.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} attributes{linearize} in %arg1 : (!transform.any_op) -> !transform.any_op
    %transformed = transform.air.linearize_vectors %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
