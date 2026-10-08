//===- fuse_channels_l1_alloc.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" --split-input-file | FileCheck %s

// A herd fills two L1 buffers from two channels in two loop nests, and the
// second buffer is allocated between the nests. Time-multiplexing the channels
// merges the second get into the first nest, so the second allocation moves
// above that nest.
// CHECK-LABEL: @hoist_l1_alloc
// CHECK: scf.for
// CHECK: air.channel.put {{.*}}@chan_a
// CHECK: air.channel.put {{.*}}@chan_a
// CHECK: air.herd
// CHECK: %{{.*}}, %[[A:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: %{{.*}}, %[[B:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: scf.for
// CHECK-NEXT: air.channel.get {{.*}}@chan_a[{{.*}}] (%[[A]]
// CHECK-NEXT: air.channel.get {{.*}}@chan_a[{{.*}}] (%[[B]]
module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @hoist_l1_alloc() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
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
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4_h = arith.constant 4 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %ga = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %ta) -> (!air.async.token) {
            %g = air.channel.get async [%t] @chan_a[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            scf.yield %g : !air.async.token
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %gb = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %tb) -> (!air.async.token) {
            %g = air.channel.get async [%t] @chan_b[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
            scf.yield %g : !air.async.token
          }
          %da = air.execute [%gb] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%gb] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// The same, but the second allocation waits on the first nest, so it cannot
// move above it. The channels are then not fused on either side: fusing only
// the puts would interleave the two streams the gets still read in sequence.
// CHECK-LABEL: @alloc_waits_on_first_nest
// CHECK: air.channel.put {{.*}}@chan_a
// CHECK: air.channel.put {{.*}}@chan_b
// CHECK: air.herd
// CHECK: air.channel.get {{.*}}@chan_a
// CHECK: air.channel.get {{.*}}@chan_b
module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @alloc_waits_on_first_nest() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
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
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4_h = arith.constant 4 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %ga = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %ta) -> (!air.async.token) {
            %g = air.channel.get async [%t] @chan_a[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            scf.yield %g : !air.async.token
          }
          %tb, %bufb = air.execute [%ga] -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %gb = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %tb) -> (!air.async.token) {
            %g = air.channel.get async [%t] @chan_b[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
            scf.yield %g : !air.async.token
          }
          %da = air.execute [%gb] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%gb] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}
