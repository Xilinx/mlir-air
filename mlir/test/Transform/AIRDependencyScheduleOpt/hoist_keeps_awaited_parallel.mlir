//===- hoist_keeps_awaited_parallel.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-hoist-ops-not-using-ping-pong | FileCheck %s

// The scf.parallel gathers into the buffer the put after it sends, so it stays
// in the loop with the put rather than moving to a loop of its own.
// CHECK-LABEL: @keep_awaited_parallel
// CHECK: scf.for
// CHECK: memref.alloc() {hoist_alloc = true}
// CHECK: %[[PAR:.*]] = scf.parallel
// CHECK: air.channel.get async {{.*}} @gather
// CHECK: air.channel.put async [{{.*}}%[[PAR]]] @out
// CHECK: } {isolated = true, unroll = 2 : i32}
// CHECK-NOT: scf.for

air.channel @gather [1, 2]
air.channel @out []
func.func @keep_awaited_parallel() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
    %1 = air.segment @seg async {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %init = air.wait_all async
      %2 = scf.for %iv = %c0 to %c4 step %c1_0 iter_args(%arg = %init) -> (!air.async.token) {
        %t, %buf = air.execute -> (memref<1x2x32x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<1x2x32x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x2x32x64xbf16, 1 : i32>
        }
        %par = scf.parallel (%x) = (%c0) to (%c2) step (%c1_0) init (%arg) -> !air.async.token {
          %g = air.channel.get async [%t, %arg] @gather[%c0, %x] (%buf[%c0, %x, %c0, %c0] [%c1_0, %c1_0, %c4, %c4] [%c4, %c4, %c4, %c1_0]) : (memref<1x2x32x64xbf16, 1 : i32>)
          scf.reduce(%g : !air.async.token) {
          ^bb0(%l: !air.async.token, %r: !air.async.token):
            %w = air.wait_all async [%l, %r]
            scf.reduce.return %w : !air.async.token
          }
        }
        %p = air.channel.put async [%t, %par] @out[] (%buf[] [] []) : (memref<1x2x32x64xbf16, 1 : i32>)
        %d = air.execute [%p, %par] {
          memref.dealloc %buf : memref<1x2x32x64xbf16, 1 : i32>
        }
        scf.yield %p : !air.async.token
      } {unroll = 2 : i32}
    }
  }
  return
}
