//===- l2_memtile_column_pin_mt_hop.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Two L2 buffers that share ONE air.channel (@ch_mid, the intermediate of an
// L3 -> L2 -> L2 -> L1 hop) normally merge into a single bucket and collapse
// onto one memtile. Distinct `air.memtile_col` pins refuse that merge, so the
// shared channel becomes a real cross-memtile (MT->MT) flow: alloc_0 lands on
// column 7 and alloc_1 on column 5.

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" | FileCheck %s

// CHECK-DAG: %[[M7:.*]] = aie.logical_tile<MemTile>(7, ?)
// CHECK-DAG: %[[M5:.*]] = aie.logical_tile<MemTile>(5, ?)
// CHECK-DAG: aie.buffer(%[[M7]]) {{{.*}}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[M5]]) {{{.*}}} : memref<64xi32, 1>

module {
  air.channel @ch_in [1, 1]
  air.channel @ch_mid [1, 1]
  air.channel @ch_out [1, 1]
  func.func @mt_hop(%ext : memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%e=%ext) : memref<32xi32> {
      %p = air.channel.put async @ch_in[] (%e[] [] []) : (memref<32xi32>)
      air.segment @segment_0 attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token_0, %alloc_0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 7 : i32} : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %g_in = air.channel.get async [%async_token_0] @ch_in[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %p_mid = air.channel.put async [%g_in] @ch_mid[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %async_token_1, %alloc_1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 5 : i32} : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %g_mid = air.channel.get async [%async_token_1] @ch_mid[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %p_out = air.channel.put async [%g_mid] @ch_out[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %h0 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
          %async_token_10, %l1 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g = air.channel.get async [%async_token_10] @ch_out[] (%l1[] [] []) : (memref<64xi32, 2>)
          %d = air.execute [%g] { memref.dealloc %l1 : memref<64xi32, 2> }
        }
      }
    }
    return
  }
}
