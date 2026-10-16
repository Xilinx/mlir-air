//===- air_shared_segment_lock_race_fix_auto.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2 use-lock-race-condition-fix-auto=true" | FileCheck %s

// Two funcs repeat one segment body. With the automatic lock race fix, each
// L2 buffer is numbered (air.l2_buffer_id) across the module before lowering;
// the numbers differ between the two copies but do not make the bodies
// different, so the device holds the body once.
// CHECK: aie.device(npu2) @seg
// CHECK-COUNT-2: aie.core(
// CHECK-NOT: aie.core(
// CHECK-NOT: aie.device

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
func.func @memtile_circ_same_flow() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @w0[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<2x8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        %tok1, %l1b = air.execute -> (memref<8xbf16, 2>) {
          %bb = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %bb : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1b[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
        %d0b = air.execute {memref.dealloc %l1b : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0) attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<16xbf16, 2>) {
          %aa = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %aa : memref<16xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<16xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<16xbf16, 2>}
      }
    }
  }
  return
}

air.channel @w1 [1, 1]
air.channel @r1 [1, 1]
func.func @memtile_circ_same_flow_again() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      air.channel.get @w1[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r1[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<2x8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        %tok1, %l1b = air.execute -> (memref<8xbf16, 2>) {
          %bb = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %bb : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1b[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
        %d0b = air.execute {memref.dealloc %l1b : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0) attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<16xbf16, 2>) {
          %aa = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %aa : memref<16xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<16xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<16xbf16, 2>}
      }
    }
  }
  return
}
