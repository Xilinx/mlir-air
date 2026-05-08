//===- cascade_chain_3herd_with_l1_upstream.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=6 num-cols=8 row-anchor=2 col-anchor=0' | FileCheck %s

// Isolates the topo-order fix. A single-tile `upstream` herd feeds the
// cascade-chain head via a non-cascade L1 broadcast. Both have cascade
// inDegree 0; the topo seeder must pick only cascade-rooted herds so
// `upstream` is placed last into a leftover slot rather than blocking
// the chain's south stack.

// `upstream` lands in a leftover slot (not on top of the chain).
// Cascade chain occupies the south rows in the same column range.
// CHECK-DAG: air.herd @upstream {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 5 : i64}
// CHECK-DAG: air.herd @producer_a {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 4 : i64}
// CHECK-DAG: air.herd @producer_b {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 3 : i64}
// CHECK-DAG: air.herd @consumer {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}

module {
  air.channel @upstream_to_a [1, 1] {broadcast_shape = [8 : index, 1 : index]}
  air.channel @ab_q [8, 1] {channel_type = "npu_cascade"}
  air.channel @bc_q [8, 1] {channel_type = "npu_cascade"}

  func.func @upstream_then_3_chain() {
    %c1 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      air.segment @seg attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index

        // upstream: single-tile herd with a non-cascade L1 broadcast
        // out. Has cascade-inDegree 0, but is not cascade-connected.
        air.herd @upstream tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {id = 3 : i32} {
          %u = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.put @upstream_to_a[%tx, %ty] (%u[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %u : memref<64xbf16, 2 : i32>
        }

        // producer_a: head of cascade chain. Receives upstream
        // broadcast (not a cascade edge) and produces cascade out.
        air.herd @producer_a tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 4 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @upstream_to_a[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @ab_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        // producer_b: middle of cascade chain.
        air.herd @producer_b tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 5 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @ab_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @bc_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        // consumer: tail of cascade chain.
        air.herd @consumer tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 6 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @bc_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
