//===- memory_space.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// With memory_space, only the allocations in that space leave the loop; one
// in another space, and one with no space yet, stay.
// CHECK-LABEL: @one_space
// CHECK: memref.alloc() : memref<16xf32, 2 : i32>
// CHECK: scf.for
// CHECK-DAG: memref.alloc() : memref<16xf32, 1 : i32>
// CHECK-DAG: memref.alloc() : memref<16xf32>
func.func @one_space(%x: f32, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %a = memref.alloc() : memref<16xf32, 2 : i32>
    %b = memref.alloc() : memref<16xf32, 1 : i32>
    %c = memref.alloc() : memref<16xf32>
    memref.store %x, %a[%c0] : memref<16xf32, 2 : i32>
    memref.store %x, %b[%c0] : memref<16xf32, 1 : i32>
    memref.store %x, %c[%c0] : memref<16xf32>
    memref.dealloc %a : memref<16xf32, 2 : i32>
    memref.dealloc %b : memref<16xf32, 1 : i32>
    memref.dealloc %c : memref<16xf32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.air.hoist_static_alloc %f <{memory_space = 2 : i32}> : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// Without memory_space, every allocation leaves the loop.
// CHECK-LABEL: @all_spaces
// CHECK-COUNT-3: memref.alloc()
// CHECK: scf.for
// CHECK-NOT: memref.alloc()
func.func @all_spaces(%x: f32, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %a = memref.alloc() : memref<16xf32, 2 : i32>
    %b = memref.alloc() : memref<16xf32, 1 : i32>
    %c = memref.alloc() : memref<16xf32>
    memref.store %x, %a[%c0] : memref<16xf32, 2 : i32>
    memref.store %x, %b[%c0] : memref<16xf32, 1 : i32>
    memref.store %x, %c[%c0] : memref<16xf32>
    memref.dealloc %a : memref<16xf32, 2 : i32>
    memref.dealloc %b : memref<16xf32, 1 : i32>
    memref.dealloc %c : memref<16xf32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.air.hoist_static_alloc %f : (!transform.any_op) -> ()
    transform.yield
  }
}
