//===- fuse_channels_guards.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" | FileCheck %s

// One channel feeds the cores of column 0, the other those of row 0: the
// guards test different coordinates, so the gets reach different cores and
// the channels stay apart.
// CHECK-LABEL: @different_guards
// CHECK: air.channel.put {{.*}}@chan_a
// CHECK: air.channel.put {{.*}}@chan_b
#col0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#row0 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
module {
  air.channel @chan_a [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_b [1, 1] {broadcast_shape = [2, 1]}
  func.func @different_guards() {
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
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c2, %sy=%c2) {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4_h = arith.constant 4 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %g = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %ta) -> (!air.async.token) {
            %ga = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_a[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %gb = affine.if #row0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%ga] @chan_b[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%ga]
              affine.yield %x : !air.async.token
            }
            scf.yield %gb : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}
