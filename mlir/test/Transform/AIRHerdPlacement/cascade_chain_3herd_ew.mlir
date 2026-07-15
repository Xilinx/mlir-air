//===- cascade_chain_3herd_ew.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=8 num-cols=6 row-anchor=2 col-anchor=0' | FileCheck %s

// West-east analog of cascade_chain_3herd.mlir. A 3-herd cascade chain
// at multi-row herd height must stack west-to-east so per-row cascade
// adjacency holds. Without chain-extent-aware roomEast the head anchors
// with only 1 col east and the tail wraps, breaking the chain.

// CHECK: air.herd @producer_a {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}
// CHECK: air.herd @producer_b {{.*}} attributes {{{.*}}x_loc = 1 : i64, y_loc = 2 : i64}
// CHECK: air.herd @consumer   {{.*}} attributes {{{.*}}x_loc = 2 : i64, y_loc = 2 : i64}

module {
  air.channel @ab [1, 8] {channel_type = "npu_cascade"}
  air.channel @bc [1, 8] {channel_type = "npu_cascade"}

  func.func @three_herd_cascade_chain_ew() {
    %c1 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      air.segment @seg attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index

        air.herd @producer_a tile (%tx, %ty) in (%sx=%c1_0, %sy=%c8) attributes {id = 3 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.put @ab[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @producer_b tile (%tx, %ty) in (%sx=%c1_0, %sy=%c8) attributes {id = 4 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @ab[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @bc[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @consumer tile (%tx, %ty) in (%sx=%c1_0, %sy=%c8) attributes {id = 5 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @bc[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
