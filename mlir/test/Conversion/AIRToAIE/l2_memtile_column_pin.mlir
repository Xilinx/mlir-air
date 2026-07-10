//===- l2_memtile_column_pin.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A memref.alloc for an L2 buffer may carry an `air.memtile_col` IntegerAttr to
// pin the memtile column its bucket lands on, overriding the default
// round-robin / column-affinity assignment. Invalid columns warn and fall
// through; a distinct pin on an alloc sharing a channel forces that alloc into
// its own bucket (so the shared channel becomes a real cross-memtile flow).
//
// Here alloc_0 is pinned to column 7 and alloc_1 to column 5, so their lowered
// memtile LogicalTileOps carry those explicit columns (row still deferred to
// the placer). Without the pin they follow the default column-affinity /
// round-robin assignment and do not land on the requested columns.

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" | FileCheck %s

// CHECK-DAG: %[[M7:.*]] = aie.logical_tile<MemTile>(7, ?)
// CHECK-DAG: %[[M5:.*]] = aie.logical_tile<MemTile>(5, ?)
// CHECK-DAG: aie.buffer(%[[M7]]) {{{.*}}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[M5]]) {{{.*}}} : memref<64xi32, 1>

module {
  air.channel @ch_a [1, 1]
  air.channel @ch_b [1, 1]
  func.func @pin_test() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token_0, %alloc_0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 7 : i32} : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %async_token_1, %alloc_1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 5 : i32} : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %t0 = air.channel.put async [%async_token_0] @ch_a[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %t1 = air.channel.put async [%async_token_1] @ch_b[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %h0 = air.herd @herd_col6 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
          %async_token_10, %l1_buf = air.execute -> (memref<32xi32, 2>) {
            %b = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %b : memref<32xi32, 2>
          }
          %g0 = air.channel.get async [%async_token_10] @ch_a[] (%l1_buf[] [] []) : (memref<32xi32, 2>)
          %dealloc = air.execute [%g0] { memref.dealloc %l1_buf : memref<32xi32, 2> }
        }
        %h1 = air.herd @herd_col7 async tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0) attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
          %async_token_20, %l1_buf2 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g1 = air.channel.get async [%async_token_20] @ch_b[] (%l1_buf2[] [] []) : (memref<64xi32, 2>)
          %dealloc = air.execute [%g1] { memref.dealloc %l1_buf2 : memref<64xi32, 2> }
        }
      }
    }
    return
  }
}
