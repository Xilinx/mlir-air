//===- cascade_chain_4herd.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=6 num-cols=8 row-anchor=0 col-anchor=0' | FileCheck %s

// Verifies the chainDepth-1 formula scales beyond 3 herds. A 4-herd
// chain at multi-column width must reserve 3 rows south of the head.
// With num-rows=6, the head anchors at y=3 (3 rows south for the
// remaining 3 herds).

// CHECK: air.herd @h0 {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 3 : i64}
// CHECK: air.herd @h1 {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 2 : i64}
// CHECK: air.herd @h2 {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 1 : i64}
// CHECK: air.herd @h3 {{.*}} attributes {{{.*}}x_loc = 0 : i64, y_loc = 0 : i64}

module {
  air.channel @c01 [8, 1] {channel_type = "npu_cascade"}
  air.channel @c12 [8, 1] {channel_type = "npu_cascade"}
  air.channel @c23 [8, 1] {channel_type = "npu_cascade"}

  func.func @four_herd_cascade_chain() {
    %c1 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      air.segment @seg attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index

        air.herd @h0 tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 3 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.put @c01[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @h1 tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 4 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @c01[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @c12[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @h2 tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 5 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @c12[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          air.channel.put @c23[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }

        air.herd @h3 tile (%tx, %ty) in (%sx=%c8, %sy=%c1_0) attributes {id = 6 : i32} {
          %a = memref.alloc() : memref<64xbf16, 2 : i32>
          air.channel.get @c23[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2 : i32>)
          memref.dealloc %a : memref<64xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
