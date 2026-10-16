//===- runtime_repeat_count.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -split-input-file -verify-diagnostics | FileCheck %s

// The memtile transfers repeat a number of times given by a segment argument.
// Each DMA channel has a single task, so it runs as a circular BD chain and
// needs no repeat count.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[BB:.*]], ^
// CHECK: ^[[BB]]:
// CHECK: aie.dma_bd({{.*}} : memref<8xbf16, 1> offset = 0 len = 8)
// CHECK: aie.next_bd ^[[BB]]

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
func.func @runtime_repeat(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.get @w0[] (%l2[] [] []) : (memref<8xbf16, 1>)
        air.channel.put @r0[] (%l2[] [] []) : (memref<8xbf16, 1>)
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.put @w0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// One slot buffer filled by two async loops with the same run-time bounds that
// do not wait for each other: their transfers run side by side, and one
// circular BD chain takes them in turn.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[BB:.*]], ^
// CHECK: ^[[BB]]:
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.next_bd ^[[BB2:.*]]
// CHECK: ^[[BB2]]:
// CHECK: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK: aie.next_bd ^[[BB]]

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
func.func @side_by_side(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      %f = scf.for %i = %c0 to %k step %c1_0 iter_args(%dt = %t) -> (!air.async.token) {
        %g = air.channel.get async [%dt] @w0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        %g2 = air.channel.get async [%g] @w0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        scf.yield %g2 : !air.async.token
      }
      %l0 = scf.for %i = %c0 to %k step %c1_0 iter_args(%dt = %t) -> (!air.async.token) {
        %p = air.channel.put async [%dt] @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        scf.yield %p : !air.async.token
      }
      %l1 = scf.for %i = %c0 to %k step %c1_0 iter_args(%dt = %t) -> (!air.async.token) {
        %p = air.channel.put async [%dt] @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        scf.yield %p : !air.async.token
      }
      %w = air.wait_all async [%f, %l0, %l1]
      %d_ = air.execute [%w] {memref.dealloc %l2 : memref<16xbf16, 1>}
      %h = air.herd @h0 async tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        %r = scf.for %j = %c0_h to %hn step %c1_h iter_args(%e = %tok0) -> (!air.async.token) {
          %p0 = air.channel.put async [%e] @w0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
          %p1 = air.channel.put async [%p0] @w0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
          %q0 = air.channel.get async [%p1] @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
          %q1 = air.channel.get async [%q0] @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
          scf.yield %q1 : !air.async.token
        }
        %d0 = air.execute [%r] {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// A channel whose DMA has another task as well cannot repeat a run-time number
// of times: the repeat count of a BD task is fixed in the configuration.
// expected-error@unknown {{'aie.memtile_dma' op failed to generate dma bd program}}

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
func.func @runtime_repeat_with_other_task(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      %c8 = arith.constant 8 : index
      air.channel.put @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.get @w0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        // expected-error@+1 {{repeats a run-time number of times on a DMA channel with other tasks}}
        air.channel.put @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<16xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.put @w0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}
