//===- cascade_chain_3herd.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=6 num-cols=8 row-anchor=2 col-anchor=0' | FileCheck %s

// Isolates the chain-depth-aware roomSouth fix. A 3-herd cascade chain
// at multi-column herd width must stack north-to-south so per-tile
// cascade adjacency holds. Without the fix the head anchors with only
// 1 row south and the tail wraps north of the chain — which the AIE
// verifier rejects as 'aie.cascade_flow' op tiles must be adjacent.

// CHECK: air.herd @producer_a {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 4 : i64}
// CHECK: air.herd @producer_b {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 3 : i64}
// CHECK: air.herd @consumer   {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}

module {
  air.channel @ab [8, 1] {channel_type = "cascade"}
  air.channel @bc [8, 1] {channel_type = "cascade"}

  func.func @three_herd_cascade_chain() {
    %c1 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      air.segment @seg attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index

        air.herd @producer_a tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 3 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.put @ab[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @producer_b tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 4 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @ab[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @bc[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @consumer tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 5 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @bc[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
