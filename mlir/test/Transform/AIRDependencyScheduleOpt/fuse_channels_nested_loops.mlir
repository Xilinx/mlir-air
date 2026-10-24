//===- fuse_channels_nested_loops.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1,L2" -split-input-file | FileCheck %s
// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" -split-input-file | FileCheck %s --check-prefix=L1ONLY

// Each iteration, a core receives buffer A and then buffer B in two chunks,
// with the chunks in an inner loop. The memtile sends A from one loop and the
// chunks of B from a nested loop in another, then sends each buffer back out.
// One channel carries A and then both chunks. The get of B, the chunk loop
// and the put of B back out move into the loop of A. The chunk loop still
// waits for the get of B, and the put of B back out still waits for the chunk
// loop. The core's gets are already in that order, so they do not change.

// Without L2 multiplexing, the memtile would send A and the chunks of B
// through two DMA channels into a single core channel. So A and B keep
// separate channels.
// L1ONLY-LABEL: func.func @ab_chunks
// L1ONLY: air.herd
// L1ONLY: air.channel.get {{.*}} @a_l1[
// L1ONLY: scf.for
// L1ONLY: air.channel.get {{.*}} @b_l1[
// CHECK-LABEL: func.func @ab_chunks
// CHECK: air.segment
// CHECK: scf.for
// CHECK: %[[GB:.*]] = air.channel.get {{.*}} @a_in[] ({{.*}}) : (memref<64xi32, 1>)
// CHECK: air.channel.put {{.*}} @a_l1[] (%{{.*}}[] [] [])
// CHECK-NEXT: %[[J:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[GB]])
// CHECK: air.channel.put {{.*}} @a_l1[] (%{{.*}}[%{{.*}}] [%c32{{.*}}] [%c1{{.*}}])
// CHECK: air.channel.put async [%[[J]]] {{.*}} : (memref<64xi32, 1>)
// CHECK: air.herd
// CHECK: scf.for
// CHECK: air.channel.get {{.*}} @a_l1[
// CHECK-NEXT: scf.for
// CHECK-NEXT: air.channel.get {{.*}} @a_l1[
// CHECK-NOT: @b_l1[

module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  air.channel @a_out [1, 1]
  air.channel @b_out [1, 1]
  func.func @ab_chunks(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        %g1 = air.channel.get async [%p2] @a_out[] (%la[] [] []) : (memref<64xi32>)
        %g2 = air.channel.get async [%g1] @b_out[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %g2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %o = air.channel.put async [%p] @a_out[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %o : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          %o = air.channel.put async [%pj] @b_out[] (%l2b[] [] []) : (memref<64xi32, 1>)
          scf.yield %o : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
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

// The memtile sends A and the chunks of B from one loop, with the chunk loop
// after the put of A. The transfers are already in the core's order, so no
// op moves.

// CHECK-LABEL: func.func @ab_one_loop
// CHECK: air.segment
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @a_l1[] (%{{.*}}[] [] [])
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @a_l1[] (%{{.*}}[%{{.*}}] [%c32{{.*}}] [%c1{{.*}}])
// CHECK: air.herd
// CHECK-NOT: @b_l1[

module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @ab_one_loop(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %l2 = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %get_b = air.channel.get async [%p] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %get_b) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %put_b = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %put_b : !air.async.token
          }
          scf.yield %pj : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
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

// The loop of A holds only the put of A. The loop of B also holds the get
// that brings B in, so the transfers of the two loops do not pair up. The
// loops are fused whole, level by level. The body of the loop of B, including
// the chunk loop, follows the put of A in a single loop that carries the
// tokens of both loops.

// CHECK-LABEL: func.func @levelwise
// CHECK: air.segment
// CHECK: scf.for {{.*}} -> (!air.async.token, !air.async.token)
// CHECK-NEXT: air.channel.put {{.*}} @a_l1[]
// CHECK-NEXT: air.channel.get {{.*}} @b_in[]
// CHECK-NEXT: scf.for
// CHECK: air.channel.put {{.*}} @a_l1[]
// CHECK-NOT: scf.for
// CHECK: air.herd
// CHECK-NOT: @b_l1[

module {
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @levelwise(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p2 = air.channel.put async [%t] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %p = air.channel.put async [%t] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          scf.yield %pj : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
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

// As in @ab_chunks, but each core get is under an affine.if. The core still
// receives A and then the chunks of B, so one channel carries them.

// CHECK-LABEL: func.func @guarded_gets
// CHECK: air.herd
// CHECK: affine.if
// CHECK-NEXT: air.channel.get {{.*}} @a_l1[
// CHECK: scf.for
// CHECK-NEXT: affine.if
// CHECK-NEXT: air.channel.get {{.*}} @a_l1[
// CHECK-NOT: @b_l1[

#set = affine_set<()[s0, s1] : (s0 == 0, s1 == 0)>
module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @guarded_gets(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          scf.yield %pj : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = affine.if #set()[%tx, %ty] -> !air.async.token {
              %get_a = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
              affine.yield %get_a : !air.async.token
            } else {
              affine.yield %t : !air.async.token
            }
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = affine.if #set()[%tx, %ty] -> !air.async.token {
                %get_b = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
                affine.yield %get_b : !air.async.token
              } else {
                affine.yield %tj : !air.async.token
              }
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
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

// As @levelwise, but the loop of B reads a buffer allocated after the loop of
// A, so the two loops cannot be fused. The loop of B would run on its own next
// to the loop of A, so the channels of A and B stay apart.

// CHECK-LABEL: func.func @unfusable
// CHECK: air.channel.put {{.*}} @a_l1[]
// CHECK: air.channel.put {{.*}} @b_l1[]
// CHECK: air.channel.get {{.*}} @b_l1[

module {
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @unfusable(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p2 = air.channel.put async [%t] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %w = air.wait_all async [%ta2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %p = air.channel.put async [%t] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %tb2) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          scf.yield %pj : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
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

// As @ab_chunks, but the chunk loop of B starts from a chain of waits on the
// get of B. The merge rebuilds the chain where the chunk loop moves to, after
// the put of A, so the chunk loop still starts once B is in.

// CHECK-LABEL: func.func @ab_chunks_waits
// CHECK: air.segment
// CHECK: scf.for
// CHECK: %[[GB:.*]] = air.channel.get {{.*}} @a_in[] (%{{.*}}[] [] []) : (memref<64xi32, 1>)
// CHECK: air.channel.put {{.*}} @a_l1[] (%{{.*}}[] [] []) : (memref<32xi32, 1>)
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[GB]])
// CHECK: air.channel.put {{.*}} @a_l1[]

module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  air.channel @a_out [1, 1]
  air.channel @b_out [1, 1]
  func.func @ab_chunks_waits(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        %g1 = air.channel.get async [%p2] @a_out[] (%la[] [] []) : (memref<64xi32>)
        %g2 = air.channel.get async [%g1] @b_out[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %g2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %o = air.channel.put async [%p] @a_out[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %o : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %w1 = air.wait_all async [%g]
          %w2 = air.wait_all async [%w1, %t]
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %w2) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          %o = air.channel.put async [%pj] @b_out[] (%l2b[] [] []) : (memref<64xi32, 1>)
          scf.yield %o : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
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

