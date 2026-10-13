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

// Four writers on four channels, one reader. The reader acquires every
// writer's lock before its one transfer reads the whole buffer, and releases
// them all after it: three acquire-only BDs (releasing by 0), the real BD,
// then three release-only BDs (acquire disabled) on the same lock pairs. A
// writer whose lock came back before the read could overwrite its slice
// while the reader still has to send it.

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<4x8xbf16, 1>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%[[C0:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P0:.*]], Release, %c0_i32)
// CHECK: aie.use_lock(%[[C1:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P1:.*]], Release, %c0_i32)
// CHECK: aie.use_lock(%[[C2:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P2:.*]], Release, %c0_i32)
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 32)
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %c1_i32)
// CHECK: aie.use_lock(%[[C0]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P0]], Release, %c1_i32)
// CHECK: aie.use_lock(%[[C1]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P1]], Release, %c1_i32)
// CHECK: aie.use_lock(%[[C2]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<4x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P2]], Release, %c1_i32)
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

// Three buffers, each written through @w0 and read on six channels. With
// per-transfer locks the writer's channel takes, per buffer, the real BD plus
// five acquire-only and five release-only BDs, more than the even-channel pool
// holds. Two buffers move to chain locks; the third keeps per-transfer locks.

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK-DAG: %[[CHAINED0:.*]] = aie.buffer(%{{.*}}) {air.chain_lock, sym_name = "{{buf[0-9]+}}"} : memref<6x8xbf16, 1>
// CHECK-DAG: %[[CHAINED1:.*]] = aie.buffer(%{{.*}}) {air.chain_lock, sym_name = "{{buf[0-9]+}}"} : memref<6x8xbf16, 1>
// CHECK-DAG: %[[B:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<6x8xbf16, 1>
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%{{.*}} : memref<6x8xbf16, 1> offset = 0 len = 48)
// CHECK: aie.dma_bd(%{{.*}} : memref<6x8xbf16, 1> offset = 0 len = 48)
// CHECK-COUNT-5: aie.dma_bd(%[[B]] : memref<6x8xbf16, 1> offset = 0 len = 0)
// CHECK: aie.dma_bd(%[[B]] : memref<6x8xbf16, 1> offset = 0 len = 48)
// CHECK-COUNT-5: aie.dma_bd(%[[B]] : memref<6x8xbf16, 1> offset = 0 len = 0)
// CHECK-NOT: aie.dma_bd(%[[B]]
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

// -----

// The two reads sit in exclusive branches on the segment index, so each
// unrolled segment keeps one of them. The lock-only BDs are added once the BD
// program is built, after unrolling, so each segment gets a single lock pair
// that its one write and one read share.

// CHECK-LABEL: aie.device({{.*}}) @seg_0_0
// CHECK-NOT: len = 0
// CHECK: aie.dma_start(MM2S, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<8xi32, 1> offset = 0 len = 8)
// CHECK-NEXT: aie.use_lock(%[[P:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<8xi32, 1> offset = 0 len = 8)
// CHECK-NEXT: aie.use_lock(%[[C]], Release, %c1_i32)
// CHECK-LABEL: aie.device({{.*}}) @seg_1_0
// CHECK-NOT: len = 0
// CHECK: func.func @reads_in_exclusive_branches

air.channel @w0 [2]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
func.func @reads_in_exclusive_branches(%arg0: memref<16xi32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%in=%arg0) : memref<16xi32> {
    %c2_l = arith.constant 2 : index
    %c0_l = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c8_l = arith.constant 8 : index
    air.channel.put @w0[%c0_l] (%in[%c0_l] [%c8_l] [%c1_l]) : (memref<16xi32>)
    air.channel.put @w0[%c1_l] (%in[%c8_l] [%c8_l] [%c1_l]) : (memref<16xi32>)
    air.segment @seg unroll(%s) in (%ss=%c2_l) {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %t, %l2 = air.execute -> (memref<8xi32, 1>) {
        %alloc = memref.alloc() : memref<8xi32, 1>
        air.execute_terminator %alloc : memref<8xi32, 1>
      }
      %g = air.channel.get async [%t] @w0[%s] (%l2[] [] []) : (memref<8xi32, 1>)
      %first = arith.cmpi eq, %s, %c0 : index
      %p = scf.if %first -> (!air.async.token) {
        %x = air.channel.put async [%g] @r0[] (%l2[] [] []) : (memref<8xi32, 1>)
        scf.yield %x : !air.async.token
      } else {
        %x = air.channel.put async [%g] @r1[] (%l2[] [] []) : (memref<8xi32, 1>)
        scf.yield %x : !air.async.token
      }
      %d_ = air.execute [%p] {
        memref.dealloc %l2 : memref<8xi32, 1>
      }
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %tk, %l1 = air.execute -> (memref<8xi32, 2>) {
          %aa = memref.alloc() : memref<8xi32, 2>
          air.execute_terminator %aa : memref<8xi32, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xi32, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xi32, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
        %tk, %l1 = air.execute -> (memref<8xi32, 2>) {
          %aa = memref.alloc() : memref<8xi32, 2>
          air.execute_terminator %aa : memref<8xi32, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xi32, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xi32, 2>}
      }
    }
  }
  return
}

// -----

// Two writes, and four reads in exclusive branches on the segment index. The
// side with fewer transfers has two, so its lock-only BDs are dummy ops added
// before unrolling; after it, each segment keeps two of the reads and one
// dummy's lock pair has no reader. The buffer keeps the counted lock.

// CHECK-LABEL: aie.device({{.*}}) @seg_0_0
// CHECK: aie.buffer(%{{.*}}) {air.counted_lock, sym_name = "{{buf[0-9]+}}"} : memref<8xi32, 1>
// CHECK-NOT: len = 0
// CHECK-LABEL: aie.device({{.*}}) @seg_1_0
// CHECK: aie.buffer(%{{.*}}) {air.counted_lock, sym_name = "{{buf[0-9]+}}"} : memref<8xi32, 1>
// CHECK-NOT: len = 0
// CHECK: func.func @several_writes_reads_in_exclusive_branches

air.channel @wa [2]
air.channel @wb [2]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
air.channel @r2 [1, 1]
air.channel @r3 [1, 1]
func.func @several_writes_reads_in_exclusive_branches(%arg0: memref<16xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%in=%arg0) : memref<16xi32> {
    %c2_l = arith.constant 2 : index
    %c0_l = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c4_l = arith.constant 4 : index
    %c8_l = arith.constant 8 : index
    %c12_l = arith.constant 12 : index
    air.channel.put @wa[%c0_l] (%in[%c0_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.channel.put @wb[%c0_l] (%in[%c4_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.channel.put @wa[%c1_l] (%in[%c8_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.channel.put @wb[%c1_l] (%in[%c12_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.segment @seg unroll(%s) in (%ss=%c2_l) {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %t, %l2 = air.execute -> (memref<8xi32, 1>) {
        %alloc = memref.alloc() : memref<8xi32, 1>
        air.execute_terminator %alloc : memref<8xi32, 1>
      }
      %g0 = air.channel.get async [%t] @wa[%s] (%l2[%c0] [%c4] [%c1_0]) : (memref<8xi32, 1>)
      %g1 = air.channel.get async [%g0] @wb[%s] (%l2[%c4] [%c4] [%c1_0]) : (memref<8xi32, 1>)
      %first = arith.cmpi eq, %s, %c0 : index
      %p = scf.if %first -> (!air.async.token) {
        %xr0 = air.channel.put async [%g1] @r0[] (%l2[%c0] [%c4] [%c1_0]) : (memref<8xi32, 1>)
        %xr1 = air.channel.put async [%g1] @r1[] (%l2[%c4] [%c4] [%c1_0]) : (memref<8xi32, 1>)
        %w = air.wait_all async [%xr0, %xr1]
        scf.yield %w : !air.async.token
      } else {
        %xr2 = air.channel.put async [%g1] @r2[] (%l2[%c0] [%c4] [%c1_0]) : (memref<8xi32, 1>)
        %xr3 = air.channel.put async [%g1] @r3[] (%l2[%c4] [%c4] [%c1_0]) : (memref<8xi32, 1>)
        %w = air.wait_all async [%xr2, %xr3]
        scf.yield %w : !air.async.token
      }
      %d_ = air.execute [%p] {
        memref.dealloc %l2 : memref<8xi32, 1>
      }
      air.herd @h0 tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %tk, %l1 = air.execute -> (memref<4xi32, 2>) {
          %aa = memref.alloc() : memref<4xi32, 2>
          air.execute_terminator %aa : memref<4xi32, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<4xi32, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<4xi32, 2>}
      }
      air.herd @h1 tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
        %tk, %l1 = air.execute -> (memref<4xi32, 2>) {
          %aa = memref.alloc() : memref<4xi32, 2>
          air.execute_terminator %aa : memref<4xi32, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<4xi32, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<4xi32, 2>}
      }
      air.herd @h2 tile (%tx_h2, %ty_h2) in (%sx_h2=%c1_0, %sy_h2=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 2 : i64} {
        %tk, %l1 = air.execute -> (memref<4xi32, 2>) {
          %aa = memref.alloc() : memref<4xi32, 2>
          air.execute_terminator %aa : memref<4xi32, 2>
        }
        air.channel.get @r2[] (%l1[] [] []) : (memref<4xi32, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<4xi32, 2>}
      }
      air.herd @h3 tile (%tx_h3, %ty_h3) in (%sx_h3=%c1_0, %sy_h3=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 2 : i64} {
        %tk, %l1 = air.execute -> (memref<4xi32, 2>) {
          %aa = memref.alloc() : memref<4xi32, 2>
          air.execute_terminator %aa : memref<4xi32, 2>
        }
        air.channel.get @r3[] (%l1[] [] []) : (memref<4xi32, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<4xi32, 2>}
      }
    }
  }
  return
}

// -----

// Both writes go to @w[%s] with the same segment index, so they share one
// endpoint and run in order, though the index is not a constant. The buffer
// keeps the counted lock.

// CHECK-LABEL: aie.device({{.*}}) @seg_0_0
// CHECK: aie.buffer(%{{.*}}) {air.counted_lock, sym_name = "{{buf[0-9]+}}"} : memref<8xi32, 1>
// CHECK-NOT: len = 0
// CHECK: func.func @writes_on_one_dynamic_endpoint

air.channel @w [2]
air.channel @r [1, 1]
func.func @writes_on_one_dynamic_endpoint(%arg0: memref<16xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%in=%arg0) : memref<16xi32> {
    %c0_l = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c4_l = arith.constant 4 : index
    %c8_l = arith.constant 8 : index
    %c12_l = arith.constant 12 : index
    %c2_l = arith.constant 2 : index
    air.channel.put @w[%c0_l] (%in[%c0_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.channel.put @w[%c0_l] (%in[%c4_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.channel.put @w[%c1_l] (%in[%c8_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.channel.put @w[%c1_l] (%in[%c12_l] [%c4_l] [%c1_l]) : (memref<16xi32>)
    air.segment @seg unroll(%s) in (%ss=%c2_l) {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %t, %l2 = air.execute -> (memref<8xi32, 1>) {
        %alloc = memref.alloc() : memref<8xi32, 1>
        air.execute_terminator %alloc : memref<8xi32, 1>
      }
      %g0 = air.channel.get async [%t] @w[%s] (%l2[%c0] [%c4] [%c1_0]) : (memref<8xi32, 1>)
      %g1 = air.channel.get async [%g0] @w[%s] (%l2[%c4] [%c4] [%c1_0]) : (memref<8xi32, 1>)
      %p = air.channel.put async [%g1] @r[] (%l2[] [] []) : (memref<8xi32, 1>)
      %d_ = air.execute [%p] {
        memref.dealloc %l2 : memref<8xi32, 1>
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %tk, %l1 = air.execute -> (memref<8xi32, 2>) {
          %aa = memref.alloc() : memref<8xi32, 2>
          air.execute_terminator %aa : memref<8xi32, 2>
        }
        air.channel.get @r[] (%l1[] [] []) : (memref<8xi32, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xi32, 2>}
      }
    }
  }
  return
}
