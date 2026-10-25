//===- runtime_repeat_count_two_in_fixed_loop.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics

// Two transfers in one fixed loop within a run-time loop: each trip runs them
// in turn twice, which copies of each BD in a row cannot express.
// expected-error@unknown {{'aie.memtile_dma' op failed to generate dma bd program}}

air.channel @r0 [1, 1]
func.func @two_in_fixed_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          // expected-error@+1 {{repeats a run-time number of times, but also a fixed number of times}}
          air.channel.put @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
          air.channel.put @r0[] (%l2[%c4] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        }
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
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}
