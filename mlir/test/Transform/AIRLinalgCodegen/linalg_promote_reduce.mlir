//===- linalg_promote_reduce.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for https://github.com/Xilinx/mlir-air/issues/1399
// Verify that linalg_promote does not leave memref.cast to dynamic types
// on promoted linalg.reduce operands.

// RUN: air-opt %s -air-transform='filename=%s' | FileCheck %s

// CHECK-LABEL: func.func @reduce_promote
// CHECK: memref.alloc() : memref<1x256xbf16, 2 : i32>
// CHECK: memref.copy
// CHECK: linalg.reduce
// CHECK-SAME: memref<1x256xbf16, 2 : i32>
// CHECK-NOT: memref<?x?xbf16
func.func @reduce_promote(%arg0: memref<2x256xbf16>) -> memref<2xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc_out = memref.alloc() : memref<2xbf16>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c2 step %c1 {
    %subview = memref.subview %arg0[%iv, 0] [1, 256] [1, 1]
        : memref<2x256xbf16> to memref<1x256xbf16, strided<[256, 1], offset: ?>>
    %subview_out = memref.subview %alloc_out[%iv] [1] [1]
        : memref<2xbf16> to memref<1xbf16, strided<[1], offset: ?>>
    linalg.fill ins(%cst : bf16) outs(%subview_out : memref<1xbf16, strided<[1], offset: ?>>)
    linalg.reduce ins(%subview : memref<1x256xbf16, strided<[256, 1], offset: ?>>)
                  outs(%subview_out : memref<1xbf16, strided<[1], offset: ?>>)
                  dimensions = [1]
      (%in: bf16, %init: bf16) {
        %add = arith.addf %in, %init : bf16
        linalg.yield %add : bf16
      }
  }
  return %alloc_out : memref<2xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.air.linalg_promote %0 {memory_space = "L1"} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
