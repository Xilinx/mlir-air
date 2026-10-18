//===- hoist_keeps_awaited_parallel.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-hoist-ops-not-using-ping-pong -split-input-file | FileCheck %s

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

// -----

// Q waits on P from inside its region: both move, and each transfer appears
// once, in the new loop.
// CHECK-LABEL: @move_captured_producer
// CHECK: scf.for
// CHECK: scf.parallel
// CHECK: air.channel.get async {{.*}} @p_in
// CHECK: scf.parallel
// CHECK: air.channel.put async {{.*}} @q_out
// CHECK: scf.for
// CHECK-NOT: @p_in
// CHECK-NOT: @q_out
// CHECK: } {isolated = true, unroll = 2 : i32}

air.channel @p_in [1, 2]
air.channel @q_out [1, 2]
func.func @move_captured_producer() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
    %1 = air.segment @seg async {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %init = air.wait_all async
      %2 = scf.for %iv = %c0 to %c4 step %c1_0 iter_args(%arg = %init) -> (!air.async.token) {
        %t, %buf = air.execute -> (memref<2x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<2x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<2x64xbf16, 1 : i32>
        }
        %pp = scf.parallel (%x) = (%c0) to (%c2) step (%c1_0) init (%arg) -> !air.async.token {
          %g = air.channel.get async [%arg] @p_in[%c0, %x] (%buf[%x, %c0] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1 : i32>)
          scf.reduce(%g : !air.async.token) {
          ^bb0(%l: !air.async.token, %r: !air.async.token):
            %w = air.wait_all async [%l, %r]
            scf.reduce.return %w : !air.async.token
          }
        }
        %qq = scf.parallel (%x) = (%c0) to (%c2) step (%c1_0) init (%arg) -> !air.async.token {
          %p = air.channel.put async [%pp] @q_out[%c0, %x] (%buf[%x, %c0] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1 : i32>)
          scf.reduce(%p : !air.async.token) {
          ^bb0(%l: !air.async.token, %r: !air.async.token):
            %w = air.wait_all async [%l, %r]
            scf.reduce.return %w : !air.async.token
          }
        }
        %d = air.execute [%t] {
          memref.dealloc %buf : memref<2x64xbf16, 1 : i32>
        }
        scf.yield %d : !air.async.token
      } {unroll = 2 : i32}
    }
  }
  return
}

// -----

// The same, but a put that stays also waits on P: P stays, and so does Q,
// which depends on it. Nothing is copied.
// CHECK-LABEL: @keep_dependent_of_awaited
// CHECK: scf.for
// CHECK: air.channel.get async {{.*}} @p_in2
// CHECK: air.channel.put async {{.*}} @q_out2
// CHECK: air.channel.put async {{.*}} @out2
// CHECK: } {isolated = true, unroll = 2 : i32}
// CHECK-NOT: @p_in2
// CHECK-NOT: @q_out2

air.channel @p_in2 [1, 2]
air.channel @q_out2 [1, 2]
air.channel @out2 []
func.func @keep_dependent_of_awaited() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
    %1 = air.segment @seg2 async {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %init = air.wait_all async
      %2 = scf.for %iv = %c0 to %c4 step %c1_0 iter_args(%arg = %init) -> (!air.async.token) {
        %t, %buf = air.execute -> (memref<2x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<2x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<2x64xbf16, 1 : i32>
        }
        %pp = scf.parallel (%x) = (%c0) to (%c2) step (%c1_0) init (%arg) -> !air.async.token {
          %g = air.channel.get async [%arg] @p_in2[%c0, %x] (%buf[%x, %c0] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1 : i32>)
          scf.reduce(%g : !air.async.token) {
          ^bb0(%l: !air.async.token, %r: !air.async.token):
            %w = air.wait_all async [%l, %r]
            scf.reduce.return %w : !air.async.token
          }
        }
        %qq = scf.parallel (%x) = (%c0) to (%c2) step (%c1_0) init (%arg) -> !air.async.token {
          %p = air.channel.put async [%pp] @q_out2[%c0, %x] (%buf[%x, %c0] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1 : i32>)
          scf.reduce(%p : !air.async.token) {
          ^bb0(%l: !air.async.token, %r: !air.async.token):
            %w = air.wait_all async [%l, %r]
            scf.reduce.return %w : !air.async.token
          }
        }
        %o = air.channel.put async [%t, %pp] @out2[] (%buf[] [] []) : (memref<2x64xbf16, 1 : i32>)
        %d = air.execute [%o] {
          memref.dealloc %buf : memref<2x64xbf16, 1 : i32>
        }
        scf.yield %o : !air.async.token
      } {unroll = 2 : i32}
    }
  }
  return
}
