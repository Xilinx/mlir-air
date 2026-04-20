//===- l2_memtile_column_affinity.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Tests column-affinity optimization in L2 memref-to-memtile assignment.
//
// Setup: 3 memtile columns (5, 6, 7), 4 L2 allocs with affinities:
//   alloc_0 -> affinity col 6 (via ch_a, gets in core at col 6)
//   alloc_1 -> affinity col 7 (via ch_b, gets in core at col 7)
//   alloc_2 -> affinity col 5 (via ch_c, gets in core at col 5)
//   alloc_3 -> affinity col 5 (via ch_d, gets in core at col 5)
//
// Without column-affinity swaps, round-robin gives:
//   alloc_0 -> memtile col 5 (WRONG, wants col 6)
//   alloc_1 -> memtile col 6 (WRONG, wants col 7)
//   alloc_2 -> memtile col 7 (WRONG, wants col 5)
//   alloc_3 -> memtile col 5 (correct)
//
// With column-affinity swaps:
//   alloc_0 -> memtile col 6 (correct, swapped with alloc_1)
//   alloc_1 -> memtile col 7 (correct, swapped with alloc_2)
//   alloc_2 -> memtile col 5 (correct, landed here after chain)
//   alloc_3 -> memtile col 5 (correct, unchanged)

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" | FileCheck %s

// Memtile tiles at row 1 (xcve2802 memtile row)
// CHECK-DAG:  %[[MT5:.*]] = aie.tile(5, 1)
// CHECK-DAG:  %[[MT6:.*]] = aie.tile(6, 1)
// CHECK-DAG:  %[[MT7:.*]] = aie.tile(7, 1)

// alloc_0 (ch_a, affinity col 6) -> memtile col 6
// CHECK-DAG:  aie.buffer(%[[MT6]]) {{{.*}}} : memref<32xi32, 1>
// alloc_1 (ch_b, affinity col 7) -> memtile col 7
// CHECK-DAG:  aie.buffer(%[[MT7]]) {{{.*}}} : memref<64xi32, 1>
// alloc_2 (ch_c, affinity col 5) -> memtile col 5
// CHECK-DAG:  aie.buffer(%[[MT5]]) {{{.*}}} : memref<128xi32, 1>
// alloc_3 (ch_d, affinity col 5) -> memtile col 5
// CHECK-DAG:  aie.buffer(%[[MT5]]) {{{.*}}} : memref<16xi32, 1>

module {
  // Per-column channels (each connects one L2 alloc to one column's core)
  air.channel @ch_a [1, 1]
  air.channel @ch_b [1, 1]
  air.channel @ch_c [1, 1]
  air.channel @ch_d [1, 1]

  func.func @column_affinity_test() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index

        // L2 allocs — order matters: alloc_0..3 in order that causes
        // round-robin misplacement across 3 memtiles
        %async_token_0, %alloc_0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %async_token_1, %alloc_1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %async_token_2, %alloc_2 = air.execute -> (memref<128xi32, 1>) {
          %a = memref.alloc() : memref<128xi32, 1>
          air.execute_terminator %a : memref<128xi32, 1>
        }
        %async_token_3, %alloc_3 = air.execute -> (memref<16xi32, 1>) {
          %a = memref.alloc() : memref<16xi32, 1>
          air.execute_terminator %a : memref<16xi32, 1>
        }

        // L2 -> L1 puts (segment side)
        %t0 = air.channel.put async [%async_token_0] @ch_a[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %t1 = air.channel.put async [%async_token_1] @ch_b[] (%alloc_1[] [] []) : (memref<64xi32, 1>)
        %t2 = air.channel.put async [%async_token_2] @ch_c[] (%alloc_2[] [] []) : (memref<128xi32, 1>)
        %t3 = air.channel.put async [%async_token_3] @ch_d[] (%alloc_3[] [] []) : (memref<16xi32, 1>)

        // Herd at col 6: gets from ch_a (alloc_0 affinity = col 6)
        %h0 = air.herd @herd_col6 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
          %async_token_10, %l1_buf = air.execute -> (memref<32xi32, 2>) {
            %b = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %b : memref<32xi32, 2>
          }
          %g0 = air.channel.get async [%async_token_10] @ch_a[] (%l1_buf[] [] []) : (memref<32xi32, 2>)
          %dealloc = air.execute [%g0] {
            memref.dealloc %l1_buf : memref<32xi32, 2>
          }
        }

        // Herd at col 7: gets from ch_b (alloc_1 affinity = col 7)
        %h1 = air.herd @herd_col7 async tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0) attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
          %async_token_20, %l1_buf2 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g1 = air.channel.get async [%async_token_20] @ch_b[] (%l1_buf2[] [] []) : (memref<64xi32, 2>)
          %dealloc = air.execute [%g1] {
            memref.dealloc %l1_buf2 : memref<64xi32, 2>
          }
        }

        // Herd at col 5: gets from ch_c and ch_d (alloc_2, alloc_3 affinity = col 5)
        %h2 = air.herd @herd_col5 async tile (%tx3, %ty3) in (%sx3=%c1_0, %sy3=%c1_0) attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
          %async_token_30, %l1_buf3 = air.execute -> (memref<128xi32, 2>) {
            %b = memref.alloc() : memref<128xi32, 2>
            air.execute_terminator %b : memref<128xi32, 2>
          }
          %async_token_31, %l1_buf4 = air.execute -> (memref<16xi32, 2>) {
            %b = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %b : memref<16xi32, 2>
          }
          %g2 = air.channel.get async [%async_token_30] @ch_c[] (%l1_buf3[] [] []) : (memref<128xi32, 2>)
          %g3 = air.channel.get async [%async_token_31] @ch_d[] (%l1_buf4[] [] []) : (memref<16xi32, 2>)
          %dealloc0 = air.execute [%g2] {
            memref.dealloc %l1_buf3 : memref<128xi32, 2>
          }
          %dealloc1 = air.execute [%g3] {
            memref.dealloc %l1_buf4 : memref<16xi32, 2>
          }
        }
      }
    }
    return
  }
}
