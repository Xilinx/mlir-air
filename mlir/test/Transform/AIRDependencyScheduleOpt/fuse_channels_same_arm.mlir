//===- fuse_channels_same_arm.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" | FileCheck %s

// Two loops in one arm of an affine.if run on the same core, one after the
// other, as two loops in a herd body would. Their channels fuse in time like
// those do. (The two arms of one affine.if run on different cores; ops in
// different arms stay apart.)

// CHECK-LABEL: func.func @same_arm
// CHECK: scf.for
// CHECK-NEXT: air.channel.put {{.*}} @chan_a[]
// CHECK-NEXT: air.channel.put {{.*}} @chan_a[]
// CHECK-NOT: @chan_b
// CHECK: air.herd
// CHECK: affine.if
// CHECK: scf.for
// CHECK-NEXT: air.channel.get {{.*}} @chan_a[]
// CHECK-NEXT: air.channel.get {{.*}} @chan_a[]
// CHECK-NOT: @chan_b

#col0 = affine_set<()[s0, s1] : (s0 == 0, s1 == 0)>
module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @same_arm() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %pa = scf.for %i = %c0 to %c4 step %c1_s iter_args(%t = %ta2) -> (!air.async.token) {
          %p = air.channel.put async [%t] @chan_a[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %pb = scf.for %i = %c0 to %c4 step %c1_s iter_args(%t = %tb2) -> (!air.async.token) {
          %p = air.channel.put async [%t] @chan_b[] (%l2b[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c2, %sy=%c1_s) {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4_h = arith.constant 4 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %g = affine.if #col0()[%tx, %ty] -> !air.async.token {
            %ga = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %ta) -> (!air.async.token) {
              %x = air.channel.get async [%t] @chan_a[] (%bufa[] [] []) : (memref<32xi32, 2>)
              scf.yield %x : !air.async.token
            }
            %gb = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %ga) -> (!air.async.token) {
              %x = air.channel.get async [%t] @chan_b[] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %x : !air.async.token
            }
            affine.yield %gb : !air.async.token
          } else {
            %x = air.wait_all async [%ta]
            affine.yield %x : !air.async.token
          }
        }
      }
    }
    return
  }
}
