//===- cascade_chain_multi_channel.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=6 num-cols=8 row-anchor=2 col-anchor=0' | FileCheck %s

// Isolates the multi-channel dedupe behavior. Three cascade channels
// (Q+K+V style) connect each adjacent pair. The downstream cascade
// graph must collapse them to one edge per pair so multi-producer
// detection doesn't false-positive and route through the
// distinct-producers placement path. Expected placement matches
// cascade_chain_3herd.mlir (single channel per pair).

// CHECK: air.herd @producer_a {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 4 : i64}
// CHECK: air.herd @producer_b {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 3 : i64}
// CHECK: air.herd @consumer   {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}

module {
  air.channel @ab_q [8, 1] {channel_type = "npu_cascade"}
  air.channel @ab_k [8, 1] {channel_type = "npu_cascade"}
  air.channel @ab_v [8, 1] {channel_type = "npu_cascade"}
  air.channel @bc_q [8, 1] {channel_type = "npu_cascade"}
  air.channel @bc_k [8, 1] {channel_type = "npu_cascade"}
  air.channel @bc_v [8, 1] {channel_type = "npu_cascade"}

  func.func @three_herd_multi_channel() {
    %c1 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      air.segment @seg attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index

        air.herd @producer_a tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 3 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.put @ab_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @ab_k[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @ab_v[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @producer_b tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 4 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @ab_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.get @ab_k[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.get @ab_v[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @bc_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @bc_k[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @bc_v[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @consumer tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 5 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @bc_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.get @bc_k[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.get @bc_v[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
