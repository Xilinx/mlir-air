//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.get_segment_for

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_addi : benefit(1) {
    %args = pdl.operands
    %results = pdl.types
    %op = pdl.operation "arith.addi"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    pdl.rewrite %op with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1 : !pdl.operation):
    %0 = pdl_match @match_addi in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = transform.air.get_segment_for %0
    transform.debug.emit_remark_at %1, "found segment" : !pdl.operation
  }
}
