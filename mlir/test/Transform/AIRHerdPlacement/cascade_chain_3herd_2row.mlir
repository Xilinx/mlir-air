//===- cascade_chain_3herd_2row.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=8 num-cols=8 row-anchor=0 col-anchor=0' | FileCheck %s

// 3-herd cascade chain of 2-row herds: the south-rows reservation must
// sum consumer heights along the longest path, not count herds. Each
// downstream consumer is 2 rows tall, so the head must reserve 4 rows
// south (2 + 2), not 2 (chain length - 1). With num-rows=8, the head
// anchors at y=4 and the chain occupies rows 0..5.

// CHECK: air.herd @h0 {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 4 : i64}
// CHECK: air.herd @h1 {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}
// CHECK: air.herd @h2 {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 0 : i64}

module {
  air.channel @c01 [4, 2] {channel_type = "cascade"}
  air.channel @c12 [4, 2] {channel_type = "cascade"}

  func.func @three_herd_2row_chain() {
    %c1 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      air.segment @seg attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index

        air.herd @h0 tile (%tx, %ty) in (%sx=%c4, %sy=%c2) attributes {id = 3 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.put @c01[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @h1 tile (%tx, %ty) in (%sx=%c4, %sy=%c2) attributes {id = 4 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @c01[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @c12[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @h2 tile (%tx, %ty) in (%sx=%c4, %sy=%c2) attributes {id = 5 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @c12[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
