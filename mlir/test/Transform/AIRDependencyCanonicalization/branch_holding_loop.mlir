//===- branch_holding_loop.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// An op after an scf.if whose region yields a loop's token depends on the
// scf.if's token, not on the loop's, which does not dominate it. Before, the
// loop's token was used: invalid IR (operand does not dominate this use).

// CHECK-LABEL: func.func @loop_in_branch
// CHECK: %[[IF:.*]] = scf.if
// CHECK:   scf.for
// CHECK: } else {
// CHECK: %[[J:.*]] = air.wait_all async [%[[IF]]{{.*}}]
// CHECK: air.execute [%[[J]]]
// CHECK-NEXT: memref.dealloc
func.func @loop_in_branch(%c: i1) {
  %c1 = arith.constant 1 : index
  %h = air.herd @h async tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%cc = %c) : i1 {
    %c0 = arith.constant 0 : index
    %c1h = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %v = arith.constant 0.0 : f32
    %t0, %a = air.execute -> (memref<4xf32, 2>) {
      %m = memref.alloc() : memref<4xf32, 2>
      air.execute_terminator %m : memref<4xf32, 2>
    }
    %r = scf.if %cc -> (!air.async.token) {
      %w = air.wait_all async [%t0]
      %f = scf.for %i = %c0 to %c4 step %c1h iter_args(%it = %w) -> (!air.async.token) {
        %e = air.execute [%it] {
          memref.store %v, %a[%i] : memref<4xf32, 2>
        }
        scf.yield %e : !air.async.token
      }
      scf.yield %f : !air.async.token
    } else {
      %w = air.wait_all async [%t0]
      scf.yield %w : !air.async.token
    }
    %j = air.wait_all async [%t0, %r]
    %d = air.execute [%j] {
      memref.dealloc %a : memref<4xf32, 2>
    }
  }
  return
}
