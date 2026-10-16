//===- l2_memtile_unpinned_off_pinned.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" | FileCheck %s

// Two unpinned buffers feed herds in column 5, so column affinity saturates
// and they are placed round-robin; the third buffer is pinned to column 6.
// The round-robin skips the memtile the pin claimed, which serves the herds
// of its own column.
// CHECK: %[[PINNED:.*]] = aie.logical_tile<MemTile>(6, ?)
// CHECK-DAG: aie.buffer(%[[PINNED]]) {{{.*}}} : memref<32xi32, 1>
// CHECK-DAG: aie.buffer(%[[A:logical_mem_[0-9]+]]) {{{.*}}} : memref<64xi32, 1>
// CHECK-DAG: aie.buffer(%[[B:logical_mem_[0-9]+]]) {{{.*}}} : memref<128xi32, 1>

module {
  air.channel @ch_a [1, 1]
  air.channel @ch_b [1, 1]
  air.channel @ch_c [1, 1]
  func.func @unpinned_off_pinned() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index
        %t0, %l2_0 = air.execute -> (memref<32xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 6 : i32} : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %p0 = air.channel.put async [%t0] @ch_a[] (%l2_0[] [] []) : (memref<32xi32, 1>)
        %t1, %l2_1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %p1 = air.channel.put async [%t1] @ch_b[] (%l2_1[] [] []) : (memref<64xi32, 1>)
        %t2, %l2_2 = air.execute -> (memref<128xi32, 1>) {
          %a = memref.alloc() : memref<128xi32, 1>
          air.execute_terminator %a : memref<128xi32, 1>
        }
        %p2 = air.channel.put async [%t2] @ch_c[] (%l2_2[] [] []) : (memref<128xi32, 1>)
        %h0 = air.herd @h0 async tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0) attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
          %u0, %l1_0 = air.execute -> (memref<32xi32, 2>) {
            %b = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %b : memref<32xi32, 2>
          }
          %g0 = air.channel.get async [%u0] @ch_a[] (%l1_0[] [] []) : (memref<32xi32, 2>)
          %d0 = air.execute [%g0] { memref.dealloc %l1_0 : memref<32xi32, 2> }
          %u1, %l1_1 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %g1 = air.channel.get async [%u1] @ch_b[] (%l1_1[] [] []) : (memref<64xi32, 2>)
          %d1 = air.execute [%g1] { memref.dealloc %l1_1 : memref<64xi32, 2> }
        }
        %h1 = air.herd @h1 async tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0) attributes {x_loc = 5 : i64, y_loc = 4 : i64} {
          %u2, %l1_2 = air.execute -> (memref<128xi32, 2>) {
            %b = memref.alloc() : memref<128xi32, 2>
            air.execute_terminator %b : memref<128xi32, 2>
          }
          %g2 = air.channel.get async [%u2] @ch_c[] (%l1_2[] [] []) : (memref<128xi32, 2>)
          %d2 = air.execute [%g2] { memref.dealloc %l1_2 : memref<128xi32, 2> }
        }
      }
    }
    return
  }
}
