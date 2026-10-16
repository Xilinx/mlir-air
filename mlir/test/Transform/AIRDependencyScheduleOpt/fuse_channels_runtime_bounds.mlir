//===- fuse_channels_runtime_bounds.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" | FileCheck %s

// The put nests loop to a run-time bound; the same bound value gives the same
// trip count, so the channels time-multiplex.
// CHECK-LABEL: @runtime_bound
// CHECK: air.channel.put {{.*}}@chan_a
// CHECK: air.channel.put {{.*}}@chan_a
// CHECK: air.herd
// CHECK: air.channel.get {{.*}}@chan_a
// CHECK: air.channel.get {{.*}}@chan_a
// CHECK-NOT: @chan_b[
module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @runtime_bound(%n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n) : index {
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %pa = scf.for %i = %c0 to %sn step %c1_s iter_args(%t = %ta2) -> (!air.async.token) {
          %p = air.channel.put async [%t] @chan_a[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %pb = scf.for %i = %c0 to %sn step %c1_s iter_args(%t = %tb2) -> (!air.async.token) {
          %p = air.channel.put async [%t] @chan_b[] (%l2b[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %w = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %w) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @chan_a[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gb = air.channel.get async [%ga] @chan_b[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
            scf.yield %gb : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}
