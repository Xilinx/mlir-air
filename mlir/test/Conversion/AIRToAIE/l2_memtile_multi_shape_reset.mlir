//===- l2_memtile_multi_shape_reset.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression: when the saturated round-robin path runs, the memtile counter
// must reset at each new alloc-shape group. Otherwise the leftover counter
// from the prior operand class (e.g. A: 2 buckets) shifts the next class
// (B: 3 buckets) off memtile 0. Mirrors the Triton i8 matmul 4-col C-out
// shift that caused a ~30s pathfinder regression.

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" | FileCheck %s

// First three emitted LTOs are reused by both shape classes.
// CHECK: %[[LTO0:.+]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[LTO1:.+]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[LTO2:.+]] = aie.logical_tile<MemTile>(?, ?)

// Buffers are emitted in reverse-bucket order; with the per-shape reset,
// the first bucket of each shape class lands on LTO0 (not LTO0 + sizeof
// of the prior class).
// CHECK-DAG: aie.buffer(%[[LTO0]]) {{.*}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[LTO1]]) {{.*}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[LTO0]]) {{.*}} : memref<64xi32, 1>
// CHECK-DAG: aie.buffer(%[[LTO1]]) {{.*}} : memref<64xi32, 1>
// CHECK-DAG: aie.buffer(%[[LTO2]]) {{.*}} : memref<64xi32, 1>

module {
  air.channel @ch_a0 [1, 1]
  air.channel @ch_a1 [1, 1]
  air.channel @ch_b0 [1, 1]
  air.channel @ch_b1 [1, 1]
  air.channel @ch_b2 [1, 1]

  func.func @multi_shape_reset() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @seg attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index

        // Shape A (memref<32xi32, 1>): 2 buckets via 2 distinct channels.
        %tok_a0, %a0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %tok_a1, %a1 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        // Shape B (memref<64xi32, 1>): 3 buckets via 3 distinct channels.
        %tok_b0, %b0 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %tok_b1, %b1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %tok_b2, %b2 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }

        %p_a0 = air.channel.put async [%tok_a0] @ch_a0[] (%a0[] [] []) : (memref<32xi32, 1>)
        %p_a1 = air.channel.put async [%tok_a1] @ch_a1[] (%a1[] [] []) : (memref<32xi32, 1>)
        %p_b0 = air.channel.put async [%tok_b0] @ch_b0[] (%b0[] [] []) : (memref<64xi32, 1>)
        %p_b1 = air.channel.put async [%tok_b1] @ch_b1[] (%b1[] [] []) : (memref<64xi32, 1>)
        %p_b2 = air.channel.put async [%tok_b2] @ch_b2[] (%b2[] [] []) : (memref<64xi32, 1>)

        // Three single-tile herds, one A + one B consumer each (except col 7
        // which only consumes a B bucket).
        %h0 = air.herd @herd_col5 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
          %tok_l1a, %l1a = air.execute -> (memref<32xi32, 2>) {
            %b = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %b : memref<32xi32, 2>
          }
          %tok_l1b, %l1b = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g_a0 = air.channel.get async [%tok_l1a] @ch_a0[] (%l1a[] [] []) : (memref<32xi32, 2>)
          %g_b0 = air.channel.get async [%tok_l1b] @ch_b0[] (%l1b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%g_a0] { memref.dealloc %l1a : memref<32xi32, 2> }
          %d1 = air.execute [%g_b0] { memref.dealloc %l1b : memref<64xi32, 2> }
        }
        %h1 = air.herd @herd_col6 async tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0) attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
          %tok_l1a2, %l1a2 = air.execute -> (memref<32xi32, 2>) {
            %b = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %b : memref<32xi32, 2>
          }
          %tok_l1b2, %l1b2 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g_a1 = air.channel.get async [%tok_l1a2] @ch_a1[] (%l1a2[] [] []) : (memref<32xi32, 2>)
          %g_b1 = air.channel.get async [%tok_l1b2] @ch_b1[] (%l1b2[] [] []) : (memref<64xi32, 2>)
          %d2 = air.execute [%g_a1] { memref.dealloc %l1a2 : memref<32xi32, 2> }
          %d3 = air.execute [%g_b1] { memref.dealloc %l1b2 : memref<64xi32, 2> }
        }
        %h2 = air.herd @herd_col7 async tile (%tx3, %ty3) in (%sx3=%c1_0, %sy3=%c1_0) attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
          %tok_l1b3, %l1b3 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g_b2 = air.channel.get async [%tok_l1b3] @ch_b2[] (%l1b3[] [] []) : (memref<64xi32, 2>)
          %d4 = air.execute [%g_b2] { memref.dealloc %l1b3 : memref<64xi32, 2> }
        }
      }
    }
    return
  }
}

// -----

// Sanity: when every bucket has a distinct shape (all singletons), they
// must still spread across the memtile pool via the global counter --
// resetting per shape would collapse them all onto memtiles[0].

// CHECK: %[[LTO0_S:.+]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[LTO1_S:.+]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[LTO2_S:.+]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[LTO3_S:.+]] = aie.logical_tile<MemTile>(?, ?)

// Four distinct-shape buckets must each land on a different LTO.
// CHECK-DAG: aie.buffer(%[[LTO0_S]]) {{.*}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[LTO1_S]]) {{.*}} : memref<64xi32, 1>
// CHECK-DAG: aie.buffer(%[[LTO2_S]]) {{.*}} : memref<128xi32, 1>
// CHECK-DAG: aie.buffer(%[[LTO3_S]]) {{.*}} : memref<16xi32, 1>

module {
  air.channel @ch_s0 [1, 1]
  air.channel @ch_s1 [1, 1]
  air.channel @ch_s2 [1, 1]
  air.channel @ch_s3 [1, 1]

  func.func @singleton_spread() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @seg attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index

        %tok0, %s0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %tok1, %s1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %tok2, %s2 = air.execute -> (memref<128xi32, 1>) {
          %a = memref.alloc() : memref<128xi32, 1>
          air.execute_terminator %a : memref<128xi32, 1>
        }
        %tok3, %s3 = air.execute -> (memref<16xi32, 1>) {
          %a = memref.alloc() : memref<16xi32, 1>
          air.execute_terminator %a : memref<16xi32, 1>
        }

        %p0 = air.channel.put async [%tok0] @ch_s0[] (%s0[] [] []) : (memref<32xi32, 1>)
        %p1 = air.channel.put async [%tok1] @ch_s1[] (%s1[] [] []) : (memref<64xi32, 1>)
        %p2 = air.channel.put async [%tok2] @ch_s2[] (%s2[] [] []) : (memref<128xi32, 1>)
        %p3 = air.channel.put async [%tok3] @ch_s3[] (%s3[] [] []) : (memref<16xi32, 1>)

        // Two herds on col 5 to trigger the saturation fallback (col 5
        // would receive 2 buckets, exceeding the per-col budget of 1).
        %h0 = air.herd @herd_col5_a async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
          %tk, %l1 = air.execute -> (memref<32xi32, 2>) {
            %b = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %b : memref<32xi32, 2>
          }
          %g = air.channel.get async [%tk] @ch_s0[] (%l1[] [] []) : (memref<32xi32, 2>)
          %d = air.execute [%g] { memref.dealloc %l1 : memref<32xi32, 2> }
        }
        %h1 = air.herd @herd_col6 async tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0) attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
          %tk, %l1 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tk] @ch_s1[] (%l1[] [] []) : (memref<64xi32, 2>)
          %d = air.execute [%g] { memref.dealloc %l1 : memref<64xi32, 2> }
        }
        %h2 = air.herd @herd_col5_b async tile (%tx3, %ty3) in (%sx3=%c1_0, %sy3=%c1_0) attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
          %tk0, %l1c = air.execute -> (memref<128xi32, 2>) {
            %b = memref.alloc() : memref<128xi32, 2>
            air.execute_terminator %b : memref<128xi32, 2>
          }
          %tk1, %l1d = air.execute -> (memref<16xi32, 2>) {
            %b = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %b : memref<16xi32, 2>
          }
          %g0 = air.channel.get async [%tk0] @ch_s2[] (%l1c[] [] []) : (memref<128xi32, 2>)
          %g1 = air.channel.get async [%tk1] @ch_s3[] (%l1d[] [] []) : (memref<16xi32, 2>)
          %d0 = air.execute [%g0] { memref.dealloc %l1c : memref<128xi32, 2> }
          %d1 = air.execute [%g1] { memref.dealloc %l1d : memref<16xi32, 2> }
        }
      }
    }
    return
  }
}
