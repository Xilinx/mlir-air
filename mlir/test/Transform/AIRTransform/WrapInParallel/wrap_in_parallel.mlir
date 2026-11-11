//===- wrap_in_parallel.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// A kernel to run whole on one core: its body goes into a 1x1 scf.parallel,
// buffers and copies included.

// CHECK-LABEL: func.func @kernel
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: scf.parallel ({{.*}}) = (%[[C0]], %[[C0]]) to (%[[C1]], %[[C1]])
// CHECK-NEXT: memref.alloc() : memref<64xbf16, 2>
// CHECK-NEXT: memref.copy %arg0
// CHECK: linalg.copy
// CHECK: memref.copy {{.*}}, %arg1
// CHECK-NEXT: scf.reduce
// CHECK-NEXT: }
// CHECK-NEXT: return

func.func @kernel(%in: memref<64xbf16>, %out: memref<64xbf16>) {
  %a = memref.alloc() : memref<64xbf16, 2>
  memref.copy %in, %a : memref<64xbf16> to memref<64xbf16, 2>
  %b = memref.alloc() : memref<64xbf16, 2>
  linalg.copy ins(%a : memref<64xbf16, 2>) outs(%b : memref<64xbf16, 2>)
  memref.copy %b, %out : memref<64xbf16, 2> to memref<64xbf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %p = transform.air.wrap_in_parallel %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// With stage_l2, the load fills an L2 buffer before the parallel, which the
// core copies into a buffer of its own; the store goes to an L2 buffer the
// parallel hands out, copied to the function buffer after it.

// CHECK-LABEL: func.func @kernel
// CHECK: %[[OUT:.*]] = memref.alloc() : memref<64xbf16, 1 : i32>
// CHECK: %[[IN:.*]] = memref.alloc() : memref<64xbf16, 1 : i32>
// CHECK: memref.copy %arg0, %[[IN]]
// CHECK: scf.parallel
// CHECK: %[[OWN:.*]] = memref.alloc() : memref<64xbf16, 2>
// CHECK: memref.copy %[[IN]], %[[OWN]]
// CHECK: linalg.copy ins(%[[OWN]]
// CHECK: memref.copy %{{.*}}, %[[OUT]]
// CHECK: scf.reduce
// CHECK: memref.copy %[[OUT]], %arg1
// CHECK-NEXT: return

func.func @kernel(%in: memref<64xbf16>, %out: memref<64xbf16>) {
  %a = memref.alloc() : memref<64xbf16, 2>
  memref.copy %in, %a : memref<64xbf16> to memref<64xbf16, 2>
  %b = memref.alloc() : memref<64xbf16, 2>
  linalg.copy ins(%a : memref<64xbf16, 2>) outs(%b : memref<64xbf16, 2>)
  memref.copy %b, %out : memref<64xbf16, 2> to memref<64xbf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %p = transform.air.wrap_in_parallel %f stage_l2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
