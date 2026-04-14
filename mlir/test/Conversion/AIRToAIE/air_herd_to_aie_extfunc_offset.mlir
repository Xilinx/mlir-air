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

// herd=1x1 with subview producing offset: ? memref passed to external
// func.call. Verifies:
// 1. Declaration inside aie.device is normalized (plain memref, no strided layout)
// 2. link_with and llvm.emit_c_interface attributes are preserved
// 3. memref.cast is inserted at the call site to bridge offset: ? to identity layout

// CHECK-LABEL: aie.device
// CHECK: aie.core
// CHECK:   memref.subview
// CHECK:   %[[CAST:.*]] = memref.cast
// CHECK:   call @extern_kernel(%[[CAST]]) : (memref<4x8xi32, 2>) -> ()
// CHECK: func.func private @extern_kernel(memref<4x8xi32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
module {

func.func private @extern_kernel(memref<4x8xi32, strided<[8, 1], offset: ?>, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}

func.func @test1() {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with = "kernel.o"} {
    %buf = memref.alloc() : memref<1x1x4x8xi32, 2>
    %sv = memref.subview %buf[%tx, %ty, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1]
        : memref<1x1x4x8xi32, 2> to memref<4x8xi32, strided<[8, 1], offset: ?>, 2>
    func.call @extern_kernel(%sv) : (memref<4x8xi32, strided<[8, 1], offset: ?>, 2>) -> ()
  }
  return
}

}
