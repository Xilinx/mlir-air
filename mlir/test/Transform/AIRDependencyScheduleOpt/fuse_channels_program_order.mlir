//===- fuse_channels_program_order.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" -split-input-file | FileCheck %s

// Two buffers of a core, filled one after the other with work between them
// (ping-pong written out): the core already takes the gets in that order, so
// the merged channel keeps it. The second get stays after the work on the
// first buffer instead of moving up next to the first get.
// CHECK-LABEL: @program_order
// CHECK: air.herd
// CHECK: affine.if
// CHECK-NEXT: air.channel.get {{.*}}@chan_a{{.*}}(%[[A:.*]][] [] [])
// CHECK-NOT: air.channel.get
// CHECK: func.call @work(%[[A]])
// CHECK: affine.if
// CHECK-NEXT: air.channel.get {{.*}}@chan_a{{.*}}(%[[B:.*]][] [] [])
// CHECK: func.call @work(%[[B]])
#col0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @chan_a [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_b [1, 1] {broadcast_shape = [1, 2]}
  func.func private @work(memref<32xi32, 2>)
  func.func @program_order() {
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
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c2) {
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
          %w = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %w) -> (!air.async.token) {
            %ga = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_a[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %wa = air.execute [%ga] {
              func.call @work(%bufa) : (memref<32xi32, 2>) -> ()
            }
            %gb = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_b[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %wb = air.execute [%gb] {
              func.call @work(%bufb) : (memref<32xi32, 2>) -> ()
            }
            %n = air.wait_all async [%wa, %wb]
            scf.yield %n : !air.async.token
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

// -----

// The same with the core taking chan_b's buffer first. The memtile side is
// merged chan_a first, so the core's gets are merged the same way: chan_a's
// buffer first.
// CHECK-LABEL: @reversed_program_order
// CHECK: air.channel.put {{.*}}@chan_a{{.*}}(%[[L2A:[a-z_0-9]+]][]
// CHECK-NEXT: air.channel.put {{.*}}@chan_a{{.*}}(%[[L2B:[a-z_0-9]+]][]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[A:[a-z_0-9]+]][]
// CHECK-NEXT: air.channel.get {{.*}}@chan_a{{.*}}(%[[B:[a-z_0-9]+]][]
// CHECK: func.call @work(%[[A]])
#col0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @chan_a [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_b [1, 1] {broadcast_shape = [1, 2]}
  func.func private @work(memref<32xi32, 2>)
  func.func @reversed_program_order() {
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
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c2) {
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
          %w = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%t = %w) -> (!air.async.token) {
            %gb = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_b[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %wb = air.execute [%gb] {
              func.call @work(%bufb) : (memref<32xi32, 2>) -> ()
            }
            %ga = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_a[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %wa = air.execute [%ga] {
              func.call @work(%bufa) : (memref<32xi32, 2>) -> ()
            }
            %n = air.wait_all async [%wa, %wb]
            scf.yield %n : !air.async.token
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
