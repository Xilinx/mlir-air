//===- l2_memtile_near_end_hop.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// An unpinned L2 buffer that shares a channel (@ch_mid) with a buffer pinned
// to another column (air.memtile_col = 5), while its own cores sit in column
// 7, is the near end of a hop: it stays on its cores' memtile rather than
// joining the pinned one, whichever of the two buffers is visited first.

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" | FileCheck %s

// CHECK-LABEL: aie.device
// CHECK-DAG: %[[M5:.*]] = aie.logical_tile<MemTile>(5, ?)
// CHECK-DAG: %[[M7:.*]] = aie.logical_tile<MemTile>(7, ?)
// CHECK-DAG: aie.buffer(%[[M5]]) {{{.*}}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[M7]]) {{{.*}}} : memref<64xi32, 1>

// The pinned buffer first.
module {
  air.channel @ch_in [1, 1]
  air.channel @ch_mid [1, 1]
  air.channel @ch_out [1, 1]
  func.func @pinned_first(%ext : memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%e=%ext) : memref<32xi32> {
      %p = air.channel.put async @ch_in[] (%e[] [] []) : (memref<32xi32>)
      air.segment @segment_0 attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token_0, %alloc_0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 5 : i32} : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %g_in = air.channel.get async [%async_token_0] @ch_in[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %p_mid = air.channel.put async [%g_in] @ch_mid[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %async_token_1, %alloc_1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %g_mid = air.channel.get async [%async_token_1] @ch_mid[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %p_out = air.channel.put async [%g_mid] @ch_out[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %h0 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
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

// -----

// CHECK-LABEL: aie.device
// CHECK-DAG: %[[M5:.*]] = aie.logical_tile<MemTile>(5, ?)
// CHECK-DAG: %[[M7:.*]] = aie.logical_tile<MemTile>(7, ?)
// CHECK-DAG: aie.buffer(%[[M5]]) {{{.*}}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[M7]]) {{{.*}}} : memref<64xi32, 1>

// The near-end buffer first.
module {
  air.channel @ch_in [1, 1]
  air.channel @ch_mid [1, 1]
  air.channel @ch_out [1, 1]
  func.func @near_end_first(%ext : memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%e=%ext) : memref<32xi32> {
      %p = air.channel.put async @ch_in[] (%e[] [] []) : (memref<32xi32>)
      air.segment @segment_0 attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token_1, %alloc_1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %async_token_0, %alloc_0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 5 : i32} : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %g_in = air.channel.get async [%async_token_0] @ch_in[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %p_mid = air.channel.put async [%g_in] @ch_mid[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %g_mid = air.channel.get async [%async_token_1] @ch_mid[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %p_out = air.channel.put async [%g_mid] @ch_out[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %h0 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
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
