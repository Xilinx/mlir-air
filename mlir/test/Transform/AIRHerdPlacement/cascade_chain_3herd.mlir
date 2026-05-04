//===- cascade_chain_3herd.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=6 num-cols=8 row-anchor=2 col-anchor=0' | FileCheck %s

// Three-herd cascade chain at multi-column herd width (size [8, 1]).
// The chain producer_a -> producer_b -> consumer needs to stack
// vertically in the same columns so that the per-tile cascade flow
// (one cascade-flow per tile-column index) can land on physically
// adjacent tiles. With multi-column herds, the only legal placement
// for the chain is north-to-south stacking — west-to-east placement
// would put producer's tile-column 0 cascading to consumer's tile-
// column 0 across two physical column hops, which the AIE dialect
// verifier rejects as 'aie.cascade_flow' op tiles must be adjacent.
//
// Three independent fixes in AIRHerdPlacementPass cooperate to make
// this work:
//   1. analyzeCascadeConnections dedupes (producer, consumer) pairs
//      so multiple cascade channels between the same pair (e.g. Q +
//      K + V cascades from rope to attn) don't false-positive the
//      multi-producer-consumer detection.
//   2. The fallback's roomSouth check for cascade producers requires
//      `chainDepth - 1` rows south, not just 1. Without this, a
//      3-herd chain producer anchors with only 1 row south, leaving
//      the third herd no legal south position.
//   3. buildCascadeTopologicalOrder seeds Kahn's queue ONLY with
//      cascade-connected herds. A non-cascade-connected upstream
//      herd (like an L1-broadcast producer) processed first would
//      claim a southern row and block the chain's south-stack —
//      tested in cascade_chain_3herd_with_l1_upstream.mlir.

// Per-tile cascade adjacency requires producer and consumer to
// occupy the same X range with consumer one row south of producer.
// CHECK: air.herd @producer_a {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 4 : i64}
// CHECK: air.herd @producer_b {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 3 : i64}
// CHECK: air.herd @consumer {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}

module {
  // Three cascade channels between each pair: exercises the dedupe
  // path. The placer must NOT count them as multiple distinct
  // producers — they're all the same producer-consumer pair.
  air.channel @ab_q [8, 1] {channel_type = "cascade"}
  air.channel @ab_k [8, 1] {channel_type = "cascade"}
  air.channel @ab_v [8, 1] {channel_type = "cascade"}
  air.channel @bc_q [8, 1] {channel_type = "cascade"}
  air.channel @bc_k [8, 1] {channel_type = "cascade"}
  air.channel @bc_v [8, 1] {channel_type = "cascade"}

  func.func @three_herd_cascade_chain() {
    %c1 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      air.segment @seg attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index

        // producer_a: head of the cascade chain (chain depth 3).
        air.herd @producer_a tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 3 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.put @ab_q[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @ab_k[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @ab_v[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        // producer_b: middle of the chain.
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

        // consumer: tail of the chain.
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
