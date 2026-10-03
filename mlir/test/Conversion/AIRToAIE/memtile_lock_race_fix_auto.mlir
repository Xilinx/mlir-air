//===- memtile_lock_race_fix_auto.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2 use-lock-race-condition-fix-auto=true" --split-input-file | FileCheck %s
// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2 use-lock-race-condition-fix-auto=true" --split-input-file | FileCheck %s --check-prefix=NOID

// use-lock-race-condition-fix-auto gives a shared L2 buffer the per-transfer
// locks of use-lock-race-condition-fix when each transfer on its many side has
// a channel of its own, keeps the counted lock when a channel moves several of
// them, and moves a buffer to chain locks when the extra BDs of the first fix
// do not fit the BD pool. An npu2 memtile's even channels share 24 BDs, and so
// do its odd channels.

// NOID-NOT: air.l2_buffer_id

// Four writers on four channels, one reader: the reader's channel gets the
// real BD plus one lock-only BD per other writer.

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<4x8xbf16, 1>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 32)
// CHECK-NOT: air.chain_lock
// CHECK-NOT: air.counted_lock
// CHECK: func.func @fanin_fits

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @w3 [1, 1]
air.channel @r0 [1, 1]
func.func @fanin_fits() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      %k3 = arith.constant 3 : index
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w3[] (%l2[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<4x8xbf16, 1>
      }
      air.herd @h0 tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h2 tile (%tx_h2, %ty_h2) in (%sx_h2=%c1_0, %sy_h2=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h3 tile (%tx_h3, %ty_h3) in (%sx_h3=%c1_0, %sy_h3=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%tx_hr, %ty_hr) in (%sx_hr=%c1_0, %sy_hr=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 5 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
    }
  }
  return
}

// -----

// Eighteen writers, six on each of three channels: each channel moves its six
// in order, so the buffer keeps the counted lock and gets no lock-only BDs.

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK: aie.buffer(%{{.*}}) {air.counted_lock, sym_name = "{{buf[0-9]+}}"} : memref<18x8xbf16, 1>
// CHECK-NOT: len = 0)
// CHECK: func.func @fanin_several_per_channel

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @r0 [1, 1]
func.func @fanin_several_per_channel() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      %k3 = arith.constant 3 : index
      %k4 = arith.constant 4 : index
      %k5 = arith.constant 5 : index
      %k6 = arith.constant 6 : index
      %k7 = arith.constant 7 : index
      %k8 = arith.constant 8 : index
      %k9 = arith.constant 9 : index
      %k10 = arith.constant 10 : index
      %k11 = arith.constant 11 : index
      %k12 = arith.constant 12 : index
      %k13 = arith.constant 13 : index
      %k14 = arith.constant 14 : index
      %k15 = arith.constant 15 : index
      %k16 = arith.constant 16 : index
      %k17 = arith.constant 17 : index
      %t, %l2 = air.execute -> (memref<18x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<18x8xbf16, 1>
        air.execute_terminator %alloc : memref<18x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w0[] (%l2[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w0[] (%l2[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w0[] (%l2[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w0[] (%l2[%k4, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w0[] (%l2[%k5, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k6, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k7, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k8, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k9, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k10, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k11, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k12, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k13, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k14, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k15, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k16, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k17, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<18x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<18x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<18x8xbf16, 1>
      }
      air.herd @h0 tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h2 tile (%tx_h2, %ty_h2) in (%sx_h2=%c1_0, %sy_h2=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%tx_hr, %ty_hr) in (%sx_hr=%c1_0, %sy_hr=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 5 : i64} {
        %tok, %l1 = air.execute -> (memref<144xbf16, 2>) {
          %aa = memref.alloc() : memref<144xbf16, 2>
          air.execute_terminator %aa : memref<144xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<144xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<144xbf16, 2>}
      }
    }
  }
  return
}

// -----

// Three buffers, each written through @w0 and read on six channels: with
// per-transfer locks the even-channel pool would need 27 BDs, so one buffer
// takes chain locks and the pool needs 22.

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK-DAG: %[[CHAINED:.*]] = aie.buffer(%{{.*}}) {air.chain_lock, sym_name = "{{buf[0-9]+}}"} : memref<6x8xbf16, 1>
// CHECK-DAG: %[[B1:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<6x8xbf16, 1>
// CHECK-DAG: %[[B2:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<6x8xbf16, 1>
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%[[CHAINED]] : memref<6x8xbf16, 1> offset = 0 len = 48)
// CHECK-NOT: aie.dma_bd(%[[CHAINED]] : memref<6x8xbf16, 1> offset = 0 len = 0)
// CHECK: aie.dma_bd(%{{.*}} : memref<6x8xbf16, 1> offset = 0 len = 48)
// CHECK-COUNT-5: aie.dma_bd(%{{.*}} : memref<6x8xbf16, 1> offset = 0 len = 0)
// CHECK: aie.dma_bd(%{{.*}} : memref<6x8xbf16, 1> offset = 0 len = 48)
// CHECK-COUNT-5: aie.dma_bd(%{{.*}} : memref<6x8xbf16, 1> offset = 0 len = 0)
// CHECK: func.func @fanouts_share_a_channel

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
air.channel @r2 [1, 1]
air.channel @r3 [1, 1]
air.channel @r4 [1, 1]
air.channel @r5 [1, 1]
func.func @fanouts_share_a_channel() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      %k3 = arith.constant 3 : index
      %k4 = arith.constant 4 : index
      %k5 = arith.constant 5 : index
      %t_bufa, %bufa = air.execute -> (memref<6x8xbf16, 1>) {
        %alloc_bufa = memref.alloc() {air.no_split} : memref<6x8xbf16, 1>
        air.execute_terminator %alloc_bufa : memref<6x8xbf16, 1>
      }
      air.channel.get @w0[] (%bufa[] [] []) : (memref<6x8xbf16, 1>)
      air.channel.put @r0[] (%bufa[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r1[] (%bufa[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r2[] (%bufa[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r3[] (%bufa[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r4[] (%bufa[%k4, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r5[] (%bufa[%k5, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      %t_bufb, %bufb = air.execute -> (memref<6x8xbf16, 1>) {
        %alloc_bufb = memref.alloc() {air.no_split} : memref<6x8xbf16, 1>
        air.execute_terminator %alloc_bufb : memref<6x8xbf16, 1>
      }
      air.channel.get @w0[] (%bufb[] [] []) : (memref<6x8xbf16, 1>)
      air.channel.put @r0[] (%bufb[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r1[] (%bufb[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r2[] (%bufb[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r3[] (%bufb[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r4[] (%bufb[%k4, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r5[] (%bufb[%k5, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      %t_bufc, %bufc = air.execute -> (memref<6x8xbf16, 1>) {
        %alloc_bufc = memref.alloc() {air.no_split} : memref<6x8xbf16, 1>
        air.execute_terminator %alloc_bufc : memref<6x8xbf16, 1>
      }
      air.channel.get @w0[] (%bufc[] [] []) : (memref<6x8xbf16, 1>)
      air.channel.put @r0[] (%bufc[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r1[] (%bufc[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r2[] (%bufc[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r3[] (%bufc[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r4[] (%bufc[%k4, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r5[] (%bufc[%k5, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.herd @hw tile (%tx_hw, %ty_hw) in (%sx_hw=%c1_0, %sy_hw=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<48xbf16, 2>) {
          %aa = memref.alloc() : memref<48xbf16, 2>
          air.execute_terminator %aa : memref<48xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<48xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<48xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<48xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<48xbf16, 2>}
      }
      air.herd @h0 tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h2 tile (%tx_h2, %ty_h2) in (%sx_h2=%c1_0, %sy_h2=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h3 tile (%tx_h3, %ty_h3) in (%sx_h3=%c1_0, %sy_h3=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h4 tile (%tx_h4, %ty_h4) in (%sx_h4=%c1_0, %sy_h4=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r4[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r4[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r4[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h5 tile (%tx_h5, %ty_h5) in (%sx_h5=%c1_0, %sy_h5=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r5[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r5[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r5[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
