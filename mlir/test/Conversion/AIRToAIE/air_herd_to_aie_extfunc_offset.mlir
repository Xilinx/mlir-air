//===- air_herd_to_aie_extfunc_offset.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that outlineAIECores normalizes memref types (strips strided layout)
// and copies attributes (link_with, llvm.emit_c_interface) when recreating
// function declarations inside aie.device. This is needed when func.call
// operands have offset: ? from memref.subview (e.g. herd=1x1 matmul).

// RUN: air-opt %s -air-to-aie | FileCheck %s

// The pass creates aie.device ops in reverse order (last herd first).

// Test 3: identity-layout memref (no strided layout). Verifies no spurious
// memref.cast is inserted when operand types already match the declaration.
// CHECK-LABEL: aie.device{{.*}}@herd_2
// CHECK: aie.core
// CHECK-NOT: memref.cast
// CHECK:   call @identity_kernel(%{{.*}}) : (memref<1024xi32, 2>) -> ()
// CHECK: func.func private @identity_kernel(memref<1024xi32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}

// Test 2: mixed operand types (memref with offset: ? and scalar i32).
// Verifies that scalar arguments pass through unchanged while memref
// arguments are normalized and cast.
// CHECK-LABEL: aie.device{{.*}}@herd_1
// CHECK: aie.core
// CHECK:   memref.subview
// CHECK:   %[[CAST2:.*]] = memref.cast
// CHECK:   call @mixed_kernel(%[[CAST2]], %{{.*}}) : (memref<4x8xi32, 2>, i32) -> ()
// CHECK: func.func private @mixed_kernel(memref<4x8xi32, 2>, i32) attributes {link_with = "kernel.o", llvm.emit_c_interface}

// Test 1: herd=1x1 with subview producing offset: ? memref passed to external
// func.call. Verifies:
// 1. Declaration inside aie.device is normalized (plain memref, no strided layout)
// 2. link_with and llvm.emit_c_interface attributes are preserved
// 3. memref.cast is inserted at the call site
// CHECK-LABEL: aie.device{{.*}}@herd_0
// CHECK: aie.core
// CHECK:   memref.subview
// CHECK:   %[[CAST:.*]] = memref.cast
// CHECK:   call @extern_kernel(%[[CAST]]) : (memref<4x8xi32, 2>) -> ()
// CHECK: func.func private @extern_kernel(memref<4x8xi32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}

module {

func.func private @extern_kernel(memref<4x8xi32, strided<[8, 1], offset: ?>, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
func.func private @mixed_kernel(memref<4x8xi32, strided<[8, 1], offset: ?>, 2>, i32) attributes {link_with = "kernel.o", llvm.emit_c_interface}
func.func private @identity_kernel(memref<1024xi32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}

func.func @test_strided_memref() {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with = "kernel.o"} {
    %buf = memref.alloc() : memref<1x1x4x8xi32, 2>
    %sv = memref.subview %buf[%tx, %ty, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1]
        : memref<1x1x4x8xi32, 2> to memref<4x8xi32, strided<[8, 1], offset: ?>, 2>
    func.call @extern_kernel(%sv) : (memref<4x8xi32, strided<[8, 1], offset: ?>, 2>) -> ()
  }
  return
}

func.func @test_mixed_operands() {
  %cst1 = arith.constant 1 : index
  %scalar = arith.constant 42 : i32
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) args(%s = %scalar) : i32 attributes {link_with = "kernel.o"} {
    %buf = memref.alloc() : memref<1x1x4x8xi32, 2>
    %sv = memref.subview %buf[%tx, %ty, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1]
        : memref<1x1x4x8xi32, 2> to memref<4x8xi32, strided<[8, 1], offset: ?>, 2>
    func.call @mixed_kernel(%sv, %s) : (memref<4x8xi32, strided<[8, 1], offset: ?>, 2>, i32) -> ()
  }
  return
}

func.func @test_identity_layout() {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with = "kernel.o"} {
    %buf = memref.alloc() : memref<1024xi32, 2>
    func.call @identity_kernel(%buf) : (memref<1024xi32, 2>) -> ()
  }
  return
}

}
