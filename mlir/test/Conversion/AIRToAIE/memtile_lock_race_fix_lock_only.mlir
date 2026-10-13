//===- memtile_lock_race_fix_lock_only.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1 use-lock-race-condition-fix=true" --split-input-file | FileCheck %s
// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2 use-lock-race-condition-fix=true" --split-input-file | FileCheck %s

// use-lock-race-condition-fix gives each transfer on a shared L2 buffer's many
// side its own lock pair. When the other side moves the buffer in one
// transfer, that transfer must hold every pair: lock-only BDs acquire them
// before it and release them after it.

// Three writers, one reader. The reader's read waits for all three writers,
// and no writer gets its slice back until the read is done.

// CHECK-LABEL: aie.device
// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<3x8xbf16, 1>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P0:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 8)
// CHECK-NEXT: aie.use_lock(%[[C0:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(S2MM, 1
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P1:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 8 len = 8)
// CHECK-NEXT: aie.use_lock(%[[C1:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(S2MM, 2
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P2:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 16 len = 8)
// CHECK-NEXT: aie.use_lock(%[[C2:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(MM2S, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C0]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P0]], Release, %c0_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C1]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P1]], Release, %c0_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C2]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 24)
// CHECK-NEXT: aie.use_lock(%[[P2]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C0]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P0]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C1]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[P1]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK: func.func @fanin

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @r0 [1, 1]
func.func @fanin() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      %t, %l2 = air.execute -> (memref<3x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<3x8xbf16, 1>
        air.execute_terminator %alloc : memref<3x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<3x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<3x8xbf16, 1>
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
      air.herd @hr tile (%tx_hr, %ty_hr) in (%sx_hr=%c1_0, %sy_hr=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<24xbf16, 2>) {
          %aa = memref.alloc() : memref<24xbf16, 2>
          air.execute_terminator %aa : memref<24xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<24xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<24xbf16, 2>}
      }
    }
  }
  return
}

// -----

// One writer, three readers. The write waits for all three readers to finish
// the previous round, and no reader starts until the write is done.

// CHECK-LABEL: aie.device
// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<3x8xbf16, 1>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C0:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 8)
// CHECK-NEXT: aie.use_lock(%[[P0:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(MM2S, 1
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C1:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 8 len = 8)
// CHECK-NEXT: aie.use_lock(%[[P1:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(MM2S, 2
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[C2:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 16 len = 8)
// CHECK-NEXT: aie.use_lock(%[[P2:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P0]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[C0]], Release, %c0_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P1]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[C1]], Release, %c0_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P2]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 24)
// CHECK-NEXT: aie.use_lock(%[[C2]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P0]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[C0]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[P1]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[C1]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK: func.func @fanout

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
air.channel @r2 [1, 1]
func.func @fanout() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      %t, %l2 = air.execute -> (memref<3x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<3x8xbf16, 1>
        air.execute_terminator %alloc : memref<3x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[] [] []) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r2[] (%l2[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<3x8xbf16, 1>
      }
      air.herd @hw tile (%tx_hw, %ty_hw) in (%sx_hw=%c1_0, %sy_hw=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<24xbf16, 2>) {
          %aa = memref.alloc() : memref<24xbf16, 2>
          air.execute_terminator %aa : memref<24xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<24xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<24xbf16, 2>}
      }
      air.herd @h0 tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h2 tile (%tx_h2, %ty_h2) in (%sx_h2=%c1_0, %sy_h2=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// Three writers, and the reader takes the buffer in two transfers on one
// channel. Its lock-only BD keeps the old layout, acquiring and releasing one
// pair.

// CHECK-LABEL: aie.device
// CHECK-NOT: acq_en = false
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<3x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %c1_i32)
// CHECK-NOT: acq_en = false
// CHECK-NOT: Release, %c0_i32
// CHECK: func.func @fanin_two_reads

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @r0 [1, 1]
func.func @fanin_two_reads() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      %t, %l2 = air.execute -> (memref<3x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<3x8xbf16, 1>
        air.execute_terminator %alloc : memref<3x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%c0, %c0] [%c2, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<3x8xbf16, 1>
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
      air.herd @hr tile (%tx_hr, %ty_hr) in (%sx_hr=%c1_0, %sy_hr=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<16xbf16, 2>) {
          %aa = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %aa : memref<16xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<16xbf16, 2>)
        air.channel.get @r0[] (%l1[] [] []) : (memref<16xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<16xbf16, 2>}
      }
    }
  }
  return
}

// -----

// Two writers, one reader. The single side's acquire-only and release-only BDs look alike, but
// the release-only one must survive the BD pattern folding.

// CHECK-LABEL: aie.device
// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<2x8xbf16, 1>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[A0:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 8)
// CHECK-NEXT: aie.use_lock(%[[B0:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(MM2S, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[B0]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[A0]], Release, %c0_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 16)
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[B0]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[A0]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK: func.func @fanin_two

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @r0 [1, 1]
func.func @fanin_two() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k1 = arith.constant 1 : index
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<2x8xbf16, 1>
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
      air.herd @hr tile (%tx_hr, %ty_hr) in (%sx_hr=%c1_0, %sy_hr=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<16xbf16, 2>) {
          %aa = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %aa : memref<16xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<16xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<16xbf16, 2>}
      }
    }
  }
  return
}

// -----

// One writer, two readers. The single side's acquire-only and release-only BDs look alike, but
// the release-only one must survive the BD pattern folding.

// CHECK-LABEL: aie.device
// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "{{buf[0-9]+}}"} : memref<2x8xbf16, 1>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[A0:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 8)
// CHECK-NEXT: aie.use_lock(%[[B0:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[B0]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[A0]], Release, %c0_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 16)
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%[[B0]], AcquireGreaterEqual, %c1_i32) {acq_en = false}
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2x8xbf16, 1> offset = 0 len = 0)
// CHECK-NEXT: aie.use_lock(%[[A0]], Release, %c1_i32)
// CHECK-NEXT: aie.next_bd
// CHECK: func.func @fanout_two

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
func.func @fanout_two() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k1 = arith.constant 1 : index
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<2x8xbf16, 1>
      }
      air.herd @hw tile (%tx_hw, %ty_hw) in (%sx_hw=%c1_0, %sy_hw=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<16xbf16, 2>) {
          %aa = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %aa : memref<16xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<16xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<16xbf16, 2>}
      }
      air.herd @h0 tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
