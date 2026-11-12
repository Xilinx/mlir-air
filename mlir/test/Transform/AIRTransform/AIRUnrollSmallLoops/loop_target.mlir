//===- loop_target.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter %s | FileCheck %s

// A loop handle as the target: the loop itself stays (the result returns
// it), the short loop inside it is unrolled.
// CHECK-LABEL: @loop_target
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK-COUNT-2: vector.transfer_write
// CHECK: } {done, outer}
func.func @loop_target(%o: memref<2x2x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %z = arith.constant dense<0.0> : vector<8xf32>
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      vector.transfer_write %z, %o[%i, %j, %c0] {in_bounds = [true]} : vector<8xf32>, memref<2x2x8xf32>
    }
  } {outer}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %l = transform.structured.match ops{["scf.for"]} attributes{outer} in %arg1 : (!transform.any_op) -> !transform.any_op
    %l2 = transform.air.unroll_small_loops %l <{max_trips = 4}> : (!transform.any_op) -> !transform.any_op
    transform.annotate %l2 "done" : !transform.any_op
    transform.yield
  }
}
