//===- runtime_repeat_count_fixed_inner_refeed.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2" | FileCheck %s

// Per trip of a run-time loop, a memtile sends x, re-fed twice
// (air.refeed_count = 2), then z from a fixed loop of three of its own, which
// fills z again each time. Both counts reach the circular chain: x twice and
// z three times in a row.
// CHECK-LABEL: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[BD0:.*]], ^
// CHECK: ^[[BD0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:.*]] : memref<64xbf16, 1>
// CHECK: aie.dma_bd(%[[X]] : memref<64xbf16, 1>
// CHECK: aie.dma_bd(%[[Z:.*]] : memref<64xbf16, 1>
// CHECK: aie.dma_bd(%[[Z]] : memref<64xbf16, 1>
// CHECK: aie.dma_bd(%[[Z]] : memref<64xbf16, 1>
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[BD0]]

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @out [1] {channel_type = "npu_dma_packet"}
func.func @refeed_in_fixed_loop(%a: memref<64xbf16>, %b: memref<64xbf16>, %n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %ln=%n) : memref<64xbf16>, memref<64xbf16>, i32 {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg args(%sn=%ln) : i32 {
      %c0_s = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c3_s = arith.constant 3 : index
      %k = arith.index_cast %sn : i32 to index
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 2 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      scf.for %i = %c0_s to %k step %c1_s {
        air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
        air.channel.put @out[] (%x[] [] []) : (memref<64xbf16, 1>)
        scf.for %j = %c0_s to %c3_s step %c1_s {
          air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
          air.channel.put @out[] (%z[] [] []) : (memref<64xbf16, 1>)
        }
      }
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c7 = arith.constant 7 : index
        %t2, %buf = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        scf.for %i = %c0 to %c7 step %c1_h {
          air.channel.get @out[] (%buf[] [] []) : (memref<64xbf16, 2>)
        }
        %d2 = air.execute { memref.dealloc %buf : memref<64xbf16, 2> }
      }
    }
  }
  return
}
