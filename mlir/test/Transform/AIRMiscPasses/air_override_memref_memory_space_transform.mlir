//===- air_override_memref_memory_space_transform.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test transform dialect op for air-override-memref-memory-space.

// RUN: air-opt %s -air-transform='filename=%s' | FileCheck %s

// CHECK-LABEL: func.func @test_herd_scope
// Herd alloc overridden to memory_space 2, segment alloc to memory_space 1.
// CHECK: air.segment
// CHECK:   memref.alloc() : memref<64xf32, 1 : i32>
// CHECK:   air.herd
// CHECK:     memref.alloc() : memref<32xf32, 2 : i32>
func.func @test_herd_scope(%arg0: memref<64xf32, 3>, %arg1: memref<32xf32, 3>) {
  air.launch () in () args(%a0=%arg0, %a1=%arg1) : memref<64xf32, 3>, memref<32xf32, 3> {
    air.segment @seg args(%s0=%a0, %s1=%a1) : memref<64xf32, 3>, memref<32xf32, 3> {
      %c1 = arith.constant 1 : index
      %seg_buf = memref.alloc() : memref<64xf32, 3>
      memref.copy %s0, %seg_buf : memref<64xf32, 3> to memref<64xf32, 3>
      air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%h1=%s1) : memref<32xf32, 3> {
        %herd_buf = memref.alloc() : memref<32xf32, 3>
        memref.copy %h1, %herd_buf : memref<32xf32, 3> to memref<32xf32, 3>
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @test_func_scope
// Func-level alloc overridden to memory_space 1 (exclusive: herd alloc
// untouched by func-scope override). Herd alloc overridden to memory_space 2
// by the herd-scope override.
// CHECK: memref.alloc() : memref<64xf32, 1 : i32>
// CHECK: air.herd
// CHECK:   memref.alloc() : memref<32xf32, 2 : i32>
func.func @test_func_scope(%arg0: memref<32xf32, 3>) {
  %c1 = arith.constant 1 : index
  %func_buf = memref.alloc() : memref<64xf32, 3>
  memref.dealloc %func_buf : memref<64xf32, 3>
  air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%h0=%arg0) : memref<32xf32, 3> {
    %herd_buf = memref.alloc() : memref<32xf32, 3>
    memref.copy %h0, %herd_buf : memref<32xf32, 3> to memref<32xf32, 3>
  }
  return
}

// CHECK-LABEL: func.func @test_subview_type_propagation
// Segment alloc overridden to memory_space 1, herd alloc to memory_space 2.
// Subview type propagated to match source (memory_space 1).
// CHECK: air.segment
// CHECK:   memref.alloc() : memref<64xbf16, 1 : i32>
// CHECK:   air.herd
// CHECK:     memref.alloc() : memref<32xbf16, 2 : i32>
// CHECK:     memref.subview %{{.*}}[0] [32] [1] : memref<64xbf16, 1 : i32> to memref<32xbf16, strided<[1]>, 1 : i32>
func.func @test_subview_type_propagation() {
  air.launch () in () {
    air.segment @seg {
      %c1 = arith.constant 1 : index
      %buf = memref.alloc() : memref<64xbf16, 3>
      air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%h0=%buf) : memref<64xbf16, 3> {
        %local = memref.alloc() : memref<32xbf16, 3>
        %sv = memref.subview %h0[0] [32] [1] : memref<64xbf16, 3> to memref<32xbf16, strided<[1]>, 3>
        memref.copy %sv, %local : memref<32xbf16, strided<[1]>, 3> to memref<32xbf16, 3>
        memref.dealloc %local : memref<32xbf16, 3>
      }
      memref.dealloc %buf : memref<64xbf16, 3>
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // Override herd allocs to L1 (memory_space 2)
    %herds = transform.structured.match ops{["air.herd"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %herds_updated = transform.air.override_memref_memory_space %herds {memory_space = 2 : i32}
      : (!transform.any_op) -> !transform.any_op

    // Override segment allocs to L2 (memory_space 1)
    %segments = transform.structured.match ops{["air.segment"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %segments_updated = transform.air.override_memref_memory_space %segments {memory_space = 1 : i32}
      : (!transform.any_op) -> !transform.any_op

    // Override func-level allocs to L2 (memory_space 1)
    %funcs = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %funcs_updated = transform.air.override_memref_memory_space %funcs {memory_space = 1 : i32}
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
