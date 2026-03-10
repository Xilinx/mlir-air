//===- scf_if_no_else_with_call.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Test that air-dependency correctly handles scf.if without an else block
// when the then block contains ops that get wrapped in air.execute (e.g.,
// func.call). The pass must create an else block with a wait_all + yield
// so that scf.if can yield an async token.

// CHECK-LABEL: func.func @scf_if_no_else_with_call
// CHECK: air.herd
// CHECK: %[[WA:.*]] = air.wait_all async
// CHECK: %[[IF_RESULT:.*]] = scf.if
// CHECK:   air.execute
// CHECK:     func.call @my_kernel
// CHECK:   air.wait_all async
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   air.wait_all async
// CHECK:   scf.yield
// CHECK: }
module {
  func.func private @my_kernel(memref<48xi32, 2 : i32>) attributes {link_with = "kernel.o", llvm.bareptr = true}
  func.func @scf_if_no_else_with_call() {
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %true = arith.constant true
      %alloc = memref.alloc() : memref<48xi32, 2 : i32>
      scf.if %true {
        func.call @my_kernel(%alloc) : (memref<48xi32, 2 : i32>) -> ()
      }
      memref.dealloc %alloc : memref<48xi32, 2 : i32>
    }
    return
  }
}
