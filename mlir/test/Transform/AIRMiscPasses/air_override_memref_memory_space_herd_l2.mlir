//===- air_override_memref_memory_space_herd_l2.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-transform='filename=%s' | FileCheck %s

// A schedule that makes a herd of a parallel whose body allocates in L2 (as
// bufferization does by default) then sets the herd's allocs to L1: the L2
// ones are overridden too, since a herd cannot hold L2.

// An alloc keeps its dynamic sizes and attributes: only its type changes.

// CHECK-LABEL: func.func @kernel
// CHECK: air.herd
// CHECK: memref.alloc() : memref<32xf32, 2 : i32>
// CHECK: memref.alloc(%{{.*}}) alignment = 64 : memref<?xf32, 2 : i32>
func.func @kernel(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%x, %y) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
    %a = memref.alloc() : memref<32xf32, 1 : i32>
    %b = memref.alloc(%n) alignment = 64 : memref<?xf32, 1 : i32>
    memref.dealloc %a : memref<32xf32, 1 : i32>
    memref.dealloc %b : memref<?xf32, 1 : i32>
    scf.reduce
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %p = transform.structured.match ops{["scf.parallel"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %h = transform.air.par_to_herd %p : (!transform.any_op) -> !transform.any_op
    %h1 = transform.air.override_memref_memory_space %h <{memory_space = 2 : i32}> : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
