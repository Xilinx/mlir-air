//===- async_input.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s
// RUN: air-opt %s -air-dependency -air-dependency | FileCheck %s --check-prefix=TWICE

// Input that is already partly async. The existing air.execute is kept as it
// is. The loop and the branch already yield tokens that cover the work inside
// them, so they are left alone; the get after the loop writes the same buffer
// as the get inside it and waits for the loop's token. Running the pass again
// adds no nesting.

// CHECK-LABEL: func.func @async_input
// CHECK: %[[T0:.*]], %[[R0:.*]] = air.execute -> (memref<8xi32, 1 : i32>) {
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: air.execute_terminator
// CHECK: %[[L:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[T0]]) -> (!air.async.token) {
// CHECK: scf.if {{.*}} -> (!air.async.token) {
// CHECK: air.channel.get async [%[[L]]] @chan

// TWICE-LABEL: func.func @async_input
// TWICE: air.execute -> (memref<8xi32, 1 : i32>) {
// TWICE-NEXT: memref.alloc()
// TWICE-NEXT: air.execute_terminator
// TWICE-NOT: = air.execute
// TWICE: scf.for
module {
  air.channel @chan [1]
  func.func @async_input(%p: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%lp=%p) : i1 {
      air.segment @seg args(%go=%lp) : i1 {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1s = arith.constant 1 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %l = scf.for %i = %c0 to %c2 step %c1s iter_args(%t = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%t] @chan[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          %s = scf.if %go -> (!air.async.token) {
            %w = air.wait_all async [%g0]
            scf.yield %w : !air.async.token
          } else {
            scf.yield %g0 : !air.async.token
          }
          scf.yield %s : !air.async.token
        }
        air.channel.get @chan[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}
