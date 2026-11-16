//===- air_transform_both.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %t = transform.structured.match attributes{sink} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.air.sink_operand_producers %t operand_indices = [0, 1] : !transform.any_op
    transform.yield
  }
}
