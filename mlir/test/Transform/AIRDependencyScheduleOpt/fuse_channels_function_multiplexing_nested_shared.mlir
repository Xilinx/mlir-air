//===- fuse_channels_function_multiplexing_nested_shared.mlir --*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" | FileCheck %s

// As @ab_chunks in fuse_channels_nested_loops.mlir, in two functions that
// share the channels from the memtile to the core. Only @ab_mux lists L2 in
// air.channel_multiplexing. Merging those channels in @ab_mux would rename
// them in @ab_plain too, whose memtile would then send A and the chunks of B
// through two DMA channels into one core channel. So neither function merges
// them.
// CHECK-LABEL: func.func @ab_mux
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @a_l1[
// CHECK: scf.for
// CHECK: air.channel.get {{.*}} @b_l1[
// CHECK-LABEL: func.func @ab_plain
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @a_l1[
// CHECK: scf.for
// CHECK: air.channel.get {{.*}} @b_l1[

module {
  air.channel @a_in_m [1, 1]
  air.channel @b_in_m [1, 1]
  air.channel @a_out_m [1, 1]
  air.channel @b_out_m [1, 1]
  air.channel @a_in_p [1, 1]
  air.channel @b_in_p [1, 1]
  air.channel @a_out_p [1, 1]
  air.channel @b_out_p [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @ab_mux(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) attributes {air.channel_multiplexing = ["L2"]} {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in_m[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in_m[] (%lb[] [] []) : (memref<64xi32>)
        %g1 = air.channel.get async [%p2] @a_out_m[] (%la[] [] []) : (memref<64xi32>)
        %g2 = air.channel.get async [%g1] @b_out_m[] (%lb[] [] []) : (memref<64xi32>)
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
          %g = air.channel.get async [%t] @a_in_m[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %o = air.channel.put async [%p] @a_out_m[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %o : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in_m[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          %o = air.channel.put async [%pj] @b_out_m[] (%l2b[] [] []) : (memref<64xi32, 1>)
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

  func.func @ab_plain(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in_p[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in_p[] (%lb[] [] []) : (memref<64xi32>)
        %g1 = air.channel.get async [%p2] @a_out_p[] (%la[] [] []) : (memref<64xi32>)
        %g2 = air.channel.get async [%g1] @b_out_p[] (%lb[] [] []) : (memref<64xi32>)
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
          %g = air.channel.get async [%t] @a_in_p[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %o = air.channel.put async [%p] @a_out_p[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %o : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in_p[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          %o = air.channel.put async [%pj] @b_out_p[] (%l2b[] [] []) : (memref<64xi32, 1>)
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
