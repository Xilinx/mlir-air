//===- l2_memtile_column_pin_saturation.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Pins are honored even when placement falls back to the round-robin
// "saturation" path. Here two unpinned buckets both derive column 6 (their
// consumer herds are at col 6), which saturates that column and forces the
// round-robin fallback. The pinned bucket (64xi32, col 5) is excluded from the
// saturation trigger and still routes to column 5 via findOrCreateLtoForCol;
// the unpinned 32xi32 buffers distribute across the pool.

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" | FileCheck %s

// CHECK-DAG: %[[M5:.*]] = aie.logical_tile<MemTile>(5, ?)
// CHECK-DAG: aie.buffer(%[[M5]]) {{{.*}}} : memref<64xi32, 1>

module {
  air.channel @ch_a [1, 1]
  air.channel @ch_b [1, 1]
  air.channel @ch_c [1, 1]
  func.func @sat_pin() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 attributes {x_size = 4 : i64} {
        %c1_0 = arith.constant 1 : index
        %ta, %a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb, %b = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tc, %c = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() {air.memtile_col = 5 : i32} : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %pa = air.channel.put async [%ta] @ch_a[] (%a[] [] []) : (memref<32xi32, 1>)
        %pb = air.channel.put async [%tb] @ch_b[] (%b[] [] []) : (memref<32xi32, 1>)
        %pc = air.channel.put async [%tc] @ch_c[] (%c[] [] []) : (memref<64xi32, 1>)
        %ha = air.herd @herd_a async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
          %t, %l = air.execute -> (memref<32xi32, 2>) { %z = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %z : memref<32xi32, 2> }
          %g = air.channel.get async [%t] @ch_a[] (%l[] [] []) : (memref<32xi32, 2>)
          %d = air.execute [%g] { memref.dealloc %l : memref<32xi32, 2> }
        }
        %hb = air.herd @herd_b async tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0) attributes {x_loc = 6 : i64, y_loc = 4 : i64} {
          %t, %l = air.execute -> (memref<32xi32, 2>) { %z = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %z : memref<32xi32, 2> }
          %g = air.channel.get async [%t] @ch_b[] (%l[] [] []) : (memref<32xi32, 2>)
          %d = air.execute [%g] { memref.dealloc %l : memref<32xi32, 2> }
        }
        %hc = air.herd @herd_c async tile (%tx3, %ty3) in (%sx3=%c1_0, %sy3=%c1_0) attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
          %t, %l = air.execute -> (memref<64xi32, 2>) { %z = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %z : memref<64xi32, 2> }
          %g = air.channel.get async [%t] @ch_c[] (%l[] [] []) : (memref<64xi32, 2>)
          %d = air.execute [%g] { memref.dealloc %l : memref<64xi32, 2> }
        }
      }
    }
    return
  }
}
