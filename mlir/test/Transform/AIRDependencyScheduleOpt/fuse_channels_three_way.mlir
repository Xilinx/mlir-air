//===- fuse_channels_three_way.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" -split-input-file | FileCheck %s

// Several channels into one core, used in order a, b, c, ... are merged into
// one channel. Both the memtile puts and the core gets must keep that order.

// Three channels.
// CHECK-LABEL: @three_way
// CHECK: %{{.*}}, %[[LA:.*]] = air.execute -> (memref<32xi32, 1>)
// CHECK: %{{.*}}, %[[LB:.*]] = air.execute -> (memref<32xi32, 1>)
// CHECK: %{{.*}}, %[[LC:.*]] = air.execute -> (memref<32xi32, 1>)
// CHECK: air.channel.put {{.*}}@chan_a{{.*}}(%[[LA]][]
// CHECK-NEXT: air.channel.put {{.*}}@chan_a{{.*}}(%[[LB]][]
// CHECK-NEXT: air.channel.put {{.*}}@chan_a{{.*}}(%[[LC]][]
// CHECK: air.herd
// CHECK: %{{.*}}, %[[A:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: %{{.*}}, %[[B:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: %{{.*}}, %[[C:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[A]][]
// CHECK: func.call @work(%[[A]])
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[B]][]
// CHECK: func.call @work(%[[B]])
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[C]][]
// CHECK: func.call @work(%[[C]])
#col0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @chan_a [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_b [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_c [1, 1] {broadcast_shape = [1, 2]}
  func.func private @work(memref<32xi32, 2>)
  func.func @three_way() {
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
        %tc2, %l2c = air.execute -> (memref<32xi32, 1>) {
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
        %pc = scf.for %i = %c0 to %c4 step %c1_s iter_args(%t = %tc2) -> (!air.async.token) {
          %p = air.channel.put async [%t] @chan_c[] (%l2c[] [] []) : (memref<32xi32, 1>)
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
          %tc, %bufc = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %w = air.wait_all async [%ta, %tb, %tc]
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
            %gc = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_c[%tx, %ty] (%bufc[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %wc = air.execute [%gc] {
              func.call @work(%bufc) : (memref<32xi32, 2>) -> ()
            }
            %n = air.wait_all async [%wa, %wb, %wc]
            scf.yield %n : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
          %dc = air.execute [%g] {
            memref.dealloc %bufc : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// Four channels.
// CHECK-LABEL: @four_way
// CHECK: %{{.*}}, %[[LA:.*]] = air.execute -> (memref<32xi32, 1>)
// CHECK: %{{.*}}, %[[LB:.*]] = air.execute -> (memref<32xi32, 1>)
// CHECK: %{{.*}}, %[[LC:.*]] = air.execute -> (memref<32xi32, 1>)
// CHECK: %{{.*}}, %[[LD:.*]] = air.execute -> (memref<32xi32, 1>)
// CHECK: air.channel.put {{.*}}@chan_a{{.*}}(%[[LA]][]
// CHECK-NEXT: air.channel.put {{.*}}@chan_a{{.*}}(%[[LB]][]
// CHECK-NEXT: air.channel.put {{.*}}@chan_a{{.*}}(%[[LC]][]
// CHECK-NEXT: air.channel.put {{.*}}@chan_a{{.*}}(%[[LD]][]
// CHECK: air.herd
// CHECK: %{{.*}}, %[[A:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: %{{.*}}, %[[B:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: %{{.*}}, %[[C:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: %{{.*}}, %[[D:.*]] = air.execute -> (memref<32xi32, 2>)
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[A]][]
// CHECK: func.call @work(%[[A]])
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[B]][]
// CHECK: func.call @work(%[[B]])
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[C]][]
// CHECK: func.call @work(%[[C]])
// CHECK: air.channel.get {{.*}}@chan_a{{.*}}(%[[D]][]
// CHECK: func.call @work(%[[D]])
#col0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @chan_a [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_b [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_c [1, 1] {broadcast_shape = [1, 2]}
  air.channel @chan_d [1, 1] {broadcast_shape = [1, 2]}
  func.func private @work(memref<32xi32, 2>)
  func.func @four_way() {
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
        %tc2, %l2c = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %td2, %l2d = air.execute -> (memref<32xi32, 1>) {
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
        %pc = scf.for %i = %c0 to %c4 step %c1_s iter_args(%t = %tc2) -> (!air.async.token) {
          %p = air.channel.put async [%t] @chan_c[] (%l2c[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %pd = scf.for %i = %c0 to %c4 step %c1_s iter_args(%t = %td2) -> (!air.async.token) {
          %p = air.channel.put async [%t] @chan_d[] (%l2d[] [] []) : (memref<32xi32, 1>)
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
          %tc, %bufc = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %td, %bufd = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %w = air.wait_all async [%ta, %tb, %tc, %td]
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
            %gc = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_c[%tx, %ty] (%bufc[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %wc = air.execute [%gc] {
              func.call @work(%bufc) : (memref<32xi32, 2>) -> ()
            }
            %gd = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %x = air.channel.get async [%t] @chan_d[%tx, %ty] (%bufd[] [] []) : (memref<32xi32, 2>)
              affine.yield %x : !air.async.token
            } else {
              %x = air.wait_all async [%t]
              affine.yield %x : !air.async.token
            }
            %wd = air.execute [%gd] {
              func.call @work(%bufd) : (memref<32xi32, 2>) -> ()
            }
            %n = air.wait_all async [%wa, %wb, %wc, %wd]
            scf.yield %n : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
          %dc = air.execute [%g] {
            memref.dealloc %bufc : memref<32xi32, 2>
          }
          %dd = air.execute [%g] {
            memref.dealloc %bufd : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}
