//===- transform-ops.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @get_partition_for_op
func.func @get_partition_for_op(%arg0: i32, %arg1: i32) {
  // expected-remark @below {{found partition}}
  air.partition args (%arg2=%arg0, %arg3=%arg1) : i32, i32 {
    %c1 = arith.constant 1 : index
    air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args (%op0=%arg2, %op1=%arg3) : i32, i32 attributes { } {
      %2 = arith.addi %op0, %op1 : i32
      air.herd_terminator
    }
  }
  return
}

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
    // CHECK: = transform.air.get_partition_for
    %1 = transform.air.get_partition_for %0
    transform.test_print_remark_at_operand %1, "found partition" : !pdl.operation
  }
}