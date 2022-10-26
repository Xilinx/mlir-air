//===- transform-ops.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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