//===- cascade_chain_3herd_with_l1_upstream.mlir ------------*- MLIR -*-===//
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

// 4-herd composition: a single-tile `upstream` herd connected to
// `producer_a` (head of a 3-herd cascade chain) via a non-cascade
// L1 broadcast channel. Both herds have cascade-inDegree of zero
// in the cascade graph; without fix #3 (cascade-rooted herds first
// in topo order), Kahn's queue processed `upstream` first (LIFO
// order over the std::set of herd names) and placed it at the
// southernmost legal slot, which then blocked the cascade chain
// from stacking south and forced `consumer` to wrap NORTH of the
// chain — violating the cascade direction.
//
// The fix seeds the Kahn queue ONLY with cascade-connected herds.
// Non-cascade herds (like `upstream` here, broadcasting via L1 to
// a chain head) get appended at the end of the topological order
// and placed last into whatever slot remains.

// `upstream` lands in a leftover slot (not on top of the chain).
// Cascade chain occupies the south rows in the same column range.
// CHECK-DAG: air.herd @upstream {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 5 : i64}
// CHECK-DAG: air.herd @producer_a {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 4 : i64}
// CHECK-DAG: air.herd @producer_b {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 3 : i64}
// CHECK-DAG: air.herd @consumer {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}

module {
  air.channel @upstream_to_a [1, 1] {broadcast_shape = [8 : index, 1 : index]}
  air.channel @ab_q [8, 1] {channel_type = "cascade"}
  air.channel @bc_q [8, 1] {channel_type = "cascade"}

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
