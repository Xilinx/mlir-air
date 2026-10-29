//===- fuse_channels_pinned_memtiles_shared.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels=aggressive-mode=L2 | FileCheck %s

// Two functions use channels A and B, multiplexed for the whole module. Each
// pins A's and B's L2 buffers to different memtiles, the second the other way
// round: each channel uses both memtiles, and A and B stay apart.
// CHECK-LABEL: @first
// CHECK: air.segment
// CHECK: air.channel.get async [%{{.*}}] @a_in[]
// CHECK: air.channel.get async [%{{.*}}] @b_in[]
// CHECK-LABEL: @second
// CHECK: air.segment
// CHECK: air.channel.get async [%{{.*}}] @a_in[]
// CHECK: air.channel.get async [%{{.*}}] @b_in[]
module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @b_l1 [1, 1] {broadcast_shape = [1, 2]}
  func.func @first(%in_a: memref<4096xi32>, %in_b: memref<4096xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<4096xi32>, memref<4096xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<4096xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<4096xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 0 : i32} : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 1 : i32} : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @b_l1[] (%l2b[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c2_s) args(%hn=%sn) : index {
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
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gb = air.channel.get async [%ga] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
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


  func.func @second(%in_a: memref<4096xi32>, %in_b: memref<4096xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<4096xi32>, memref<4096xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<4096xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<4096xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 1 : i32} : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 0 : i32} : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @b_l1[] (%l2b[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c2_s) args(%hn=%sn) : index {
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
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gb = air.channel.get async [%ga] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
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
