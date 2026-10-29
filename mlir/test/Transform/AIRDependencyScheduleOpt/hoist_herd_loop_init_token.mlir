//===- hoist_herd_loop_init_token.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests -split-input-file | FileCheck %s

// The loop starts its token from a channel put's, which the herd waits for
// directly and through the loop's iter_arg. The hoisted herd waits for it
// once before it starts, as the herd's first run did, so the herd is hoisted.
// CHECK-LABEL: func.func @init_from_channel
// CHECK: %[[E:.*]] = air.channel.put async
// CHECK: air.herd @herd_0 async [%[[E]]{{.*}}]
// CHECK: scf.for
// CHECK-NOT: air.herd
func.func @init_from_channel(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %ta, %l2 = air.execute -> (memref<16xi32, 1>) {
        %a2 = memref.alloc() : memref<16xi32, 1>
        air.execute_terminator %a2 : memref<16xi32, 1>
      }
      %e = air.channel.put async [%ta] @feed[] (%l2[] [] []) : (memref<16xi32, 1>)
      %r = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %e) -> (!air.async.token) {
        %h = air.herd @herd_0 async [%e, %d] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%sn) : i32 {
          %c0_1 = arith.constant 0 : index
          %t0, %b = air.execute -> (memref<16xi32, 2>) {
            %al = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %al : memref<16xi32, 2>
          }
          %t1 = air.execute [%t0] {
            memref.store %q, %b[%c0_1] : memref<16xi32, 2>
          }
          %t2 = air.execute [%t1] {
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
        scf.yield %h : !air.async.token
      }
    }
  }
  return
}
air.channel @feed [1, 1]

// -----

// As above, with the token carried through two loops to the herd. The herd is
// hoisted out of both.
// CHECK-LABEL: func.func @init_through_two_loops
// CHECK: %[[E:.*]] = air.channel.put async
// CHECK: air.herd @herd_0 async [%[[E]]{{.*}}]
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: air.herd
func.func @init_through_two_loops(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %ta, %l2 = air.execute -> (memref<16xi32, 1>) {
        %a2 = memref.alloc() : memref<16xi32, 1>
        air.execute_terminator %a2 : memref<16xi32, 1>
      }
      %e = air.channel.put async [%ta] @feed[] (%l2[] [] []) : (memref<16xi32, 1>)
      %r = scf.for %i = %c0 to %c2 step %c1_0 iter_args(%d = %e) -> (!air.async.token) {
        %r2 = scf.for %j = %c0 to %c4 step %c1_0 iter_args(%d2 = %d) -> (!air.async.token) {
          %h = air.herd @herd_0 async [%e, %d2] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%sn) : i32 {
            %c0_1 = arith.constant 0 : index
            %t0, %b = air.execute -> (memref<16xi32, 2>) {
              %al = memref.alloc() : memref<16xi32, 2>
              air.execute_terminator %al : memref<16xi32, 2>
            }
            %t1 = air.execute [%t0] {
              memref.store %q, %b[%c0_1] : memref<16xi32, 2>
            }
            %t2 = air.execute [%t1] {
              memref.dealloc %b : memref<16xi32, 2>
            }
          }
          scf.yield %h : !air.async.token
        }
        scf.yield %r2 : !air.async.token
      }
    }
  }
  return
}
air.channel @feed [1, 1]

// -----

// The outer loop starts its token from the channel put's, which the herd
// waits for, but so does the inner loop, whose iter_arg the herd does not
// wait for. That counts as a wait of the inner loop's own, so the herd stays
// in the loops.
// CHECK-LABEL: func.func @inner_loop_init
// CHECK: air.channel.put async
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.herd @herd_0
func.func @inner_loop_init(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %ta, %l2 = air.execute -> (memref<16xi32, 1>) {
        %a2 = memref.alloc() : memref<16xi32, 1>
        air.execute_terminator %a2 : memref<16xi32, 1>
      }
      %e = air.channel.put async [%ta] @feed[] (%l2[] [] []) : (memref<16xi32, 1>)
      %r = scf.for %i = %c0 to %c2 step %c1_0 iter_args(%d = %e) -> (!air.async.token) {
        %r2 = scf.for %j = %c0 to %c4 step %c1_0 iter_args(%d2 = %e) -> (!air.async.token) {
          %h = air.herd @herd_0 async [%e, %d] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%sn) : i32 {
            %c0_1 = arith.constant 0 : index
            %t0, %b = air.execute -> (memref<16xi32, 2>) {
              %al = memref.alloc() : memref<16xi32, 2>
              air.execute_terminator %al : memref<16xi32, 2>
            }
            %t1 = air.execute [%t0] {
              memref.store %q, %b[%c0_1] : memref<16xi32, 2>
            }
            %t2 = air.execute [%t1] {
              memref.dealloc %b : memref<16xi32, 2>
            }
          }
          %y = air.wait_all async [%h, %d2]
          scf.yield %y : !air.async.token
        }
        scf.yield %r2 : !air.async.token
      }
    }
  }
  return
}
air.channel @feed [1, 1]
