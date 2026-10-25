//===- packet_producers_mixed_memspace_shim_invalid.mlir -------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=npu2" 2>&1 | FileCheck %s

// Only a packet flow can merge several producers into one receiver, so a
// circuit-switched channel with an L3 and an on-device producer is rejected.
// CHECK: error: 'air.channel' op only a packet channel can mix L3 and on-device producers

air.channel @tomt [1, 1]
air.channel @x [1, 1]
air.channel @r0 [1, 1]
func.func @mixed_stream(%host: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%h=%host) : memref<8xbf16> {
    air.channel.put @x[] (%h[] [] []) {id = 100 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %tm, %l2p = air.execute -> (memref<8xbf16, 1>) {
        %a2 = memref.alloc() {air.memtile_col = 1 : i32} : memref<8xbf16, 1>
        air.execute_terminator %a2 : memref<8xbf16, 1>
      }
      air.channel.get @tomt[] (%l2p[] [] []) : (memref<8xbf16, 1>)
      air.channel.put @x[] (%l2p[] [] []) : (memref<8xbf16, 1>)
      %t, %l2 = air.execute -> (memref<3x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split, air.memtile_col = 2 : i32} : memref<3x8xbf16, 1>
        air.execute_terminator %alloc : memref<3x8xbf16, 1>
      }
      air.channel.get @x[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<3x8xbf16, 1>)
      %d_ = air.execute { memref.dealloc %l2 : memref<3x8xbf16, 1> }
      %dp = air.execute { memref.dealloc %l2p : memref<8xbf16, 1> }
      air.herd @hp tile (%txp, %typ) in (%sxp=%c1_0, %syp=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @tomt[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dpp = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @x[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<24xbf16, 2>) {
          %aa = memref.alloc() : memref<24xbf16, 2>
          air.execute_terminator %aa : memref<24xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<24xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<24xbf16, 2>}
      }
    }
  }
  return
}
