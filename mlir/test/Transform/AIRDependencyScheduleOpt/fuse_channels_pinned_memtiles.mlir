//===- fuse_channels_pinned_memtiles.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels | FileCheck %s

// Both functions bring buffers A and B into L2 the same way and list L2 in
// air.channel_multiplexing, as in fuse_channels_function_multiplexing.mlir.
// In the first, A and B are pinned to different memtiles, whose DMA channels
// are their own: A and B keep a channel each. In the second, both are pinned
// to one memtile and share its channel.
// CHECK-LABEL: @pins_differ
// CHECK: air.segment
// CHECK: air.channel.get async [%{{.*}}] @a_in_d[]
// CHECK: air.channel.get async [%{{.*}}] @b_in_d[]
// CHECK: air.herd
// CHECK-LABEL: @pins_agree
// CHECK: air.segment
// CHECK: air.channel.get async [%{{.*}}] @a_in_s[]
// CHECK: air.channel.get async [%{{.*}}] @a_in_s[]
// CHECK-NOT: @b_in_s[]
// CHECK: air.herd
//
// In the third, A is not pinned, B is pinned to one memtile and C to another.
// A and B merge; C then differs from what A's channel already carries and
// keeps its own.
// CHECK-LABEL: @three_pins
// CHECK: air.segment
// CHECK: air.channel.get async [%{{.*}}] @a_in_t[]
// CHECK: air.channel.get async [%{{.*}}] @a_in_t[]
// CHECK: air.channel.get async [%{{.*}}] @c_in_t[]
// CHECK: air.herd
//
// In the fourth, one air.execute makes both L2 buffers, pinned to different
// memtiles: each channel's pin is the one of the buffer it uses, and they
// stay apart.
// CHECK-LABEL: @one_execute
// CHECK: air.segment
// CHECK: air.channel.get async [%{{.*}}] @a_in_e[]
// CHECK: air.channel.get async [%{{.*}}] @b_in_e[]
// CHECK: air.herd
module {
  air.channel @a_in_d [1, 1]
  air.channel @b_in_d [1, 1]
  air.channel @a_l1_d [1, 1] {broadcast_shape = [1, 2]}
  air.channel @b_l1_d [1, 1] {broadcast_shape = [1, 2]}
  air.channel @a_in_s [1, 1]
  air.channel @b_in_s [1, 1]
  air.channel @a_l1_s [1, 1] {broadcast_shape = [1, 2]}
  air.channel @b_l1_s [1, 1] {broadcast_shape = [1, 2]}
  air.channel @a_in_t [1, 1]
  air.channel @b_in_t [1, 1]
  air.channel @c_in_t [1, 1]
  air.channel @a_l1_t [1, 1] {broadcast_shape = [1, 2]}
  air.channel @b_l1_t [1, 1] {broadcast_shape = [1, 2]}
  air.channel @c_l1_t [1, 1] {broadcast_shape = [1, 2]}
  air.channel @a_in_e [1, 1]
  air.channel @b_in_e [1, 1]
  air.channel @a_l1_e [1, 1] {broadcast_shape = [1, 2]}
  air.channel @b_l1_e [1, 1] {broadcast_shape = [1, 2]}
  func.func @pins_differ(%in_a: memref<4096xi32>, %in_b: memref<4096xi32>, %n: index) attributes {air.channel_multiplexing = ["L2"]} {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<4096xi32>, memref<4096xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in_d[] (%la[] [] []) : (memref<4096xi32>)
        %p2 = air.channel.put async [%p] @b_in_d[] (%lb[] [] []) : (memref<4096xi32>)
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
          %g = air.channel.get async [%t] @a_in_d[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1_d[] (%l2a[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in_d[] (%l2b[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @b_l1_d[] (%l2b[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
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
            %ga = air.channel.get async [%t] @a_l1_d[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gb = air.channel.get async [%ga] @b_l1_d[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
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

  func.func @pins_agree(%in_a: memref<4096xi32>, %in_b: memref<4096xi32>, %n: index) attributes {air.channel_multiplexing = ["L2"]} {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<4096xi32>, memref<4096xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in_s[] (%la[] [] []) : (memref<4096xi32>)
        %p2 = air.channel.put async [%p] @b_in_s[] (%lb[] [] []) : (memref<4096xi32>)
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
          %x = memref.alloc() {air.memtile_col = 0 : i32} : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in_s[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1_s[] (%l2a[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in_s[] (%l2b[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @b_l1_s[] (%l2b[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
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
            %ga = air.channel.get async [%t] @a_l1_s[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gb = air.channel.get async [%ga] @b_l1_s[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
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


  func.func @three_pins(%in_a: memref<4096xi32>, %in_b: memref<4096xi32>, %n: index) attributes {air.channel_multiplexing = ["L2"]} {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<4096xi32>, memref<4096xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in_t[] (%la[] [] []) : (memref<4096xi32>)
        %p2 = air.channel.put async [%p] @b_in_t[] (%lb[] [] []) : (memref<4096xi32>)
        %p3 = air.channel.put async [%p2] @c_in_t[] (%lb[] [] []) : (memref<4096xi32>)
        scf.yield %p3 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 0 : i32} : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tc2, %l2c = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 1 : i32} : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2, %tc2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in_t[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1_t[] (%l2a[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in_t[] (%l2b[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @b_l1_t[] (%l2b[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_c = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @c_in_t[] (%l2c[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @c_l1_t[] (%l2c[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
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
          %tc, %bufc = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb, %tc]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1_t[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gb = air.channel.get async [%ga] @b_l1_t[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
            %gc = air.channel.get async [%gb] @c_l1_t[%tx, %ty] (%bufc[] [] []) : (memref<32xi32, 2>)
            scf.yield %gc : !air.async.token
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


  func.func @one_execute(%in_a: memref<4096xi32>, %in_b: memref<4096xi32>, %n: index) attributes {air.channel_multiplexing = ["L2"]} {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<4096xi32>, memref<4096xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in_e[] (%la[] [] []) : (memref<4096xi32>)
        %p2 = air.channel.put async [%p] @b_in_e[] (%lb[] [] []) : (memref<4096xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %ta2, %l2a, %l2b = air.execute -> (memref<32xi32, 1>, memref<32xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 0 : i32} : memref<32xi32, 1>
          %y = memref.alloc() {air.memtile_col = 1 : i32} : memref<32xi32, 1>
          air.execute_terminator %x, %y : memref<32xi32, 1>, memref<32xi32, 1>
        }
        %w = air.wait_all async [%ta2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in_e[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1_e[] (%l2a[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in_e[] (%l2b[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @b_l1_e[] (%l2b[] [] []) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>} : (memref<32xi32, 1>)
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
            %ga = air.channel.get async [%t] @a_l1_e[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gb = air.channel.get async [%ga] @b_l1_e[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
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
