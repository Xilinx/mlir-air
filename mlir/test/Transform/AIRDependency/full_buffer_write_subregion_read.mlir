//===- full_buffer_write_subregion_read.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Regression test: when a channel.get writes an entire L2 buffer (empty offsets)
// and multiple channel.put ops read from different sub-regions of that buffer,
// all puts must depend on the get. Previously, areEqualIndexPartialMemrefs only
// matched the sub-region at offset 0, missing RAW dependencies for other offsets.

// CHECK-LABEL: func.func @full_buffer_write_subregion_read
// CHECK: air.segment
// The channel.get fills the entire buffer (empty offsets)
// CHECK: %[[GET:.*]] = air.channel.get async{{.*}}@src[] (%{{.*}}[] [] [])
// Both channel.put ops must depend on the channel.get
// CHECK: air.channel.put async [%[[GET]]{{.*}}]{{.*}}@dst_0[]
// CHECK: air.channel.put async [%[[GET]]{{.*}}]{{.*}}@dst_1[]

module {
  air.channel @src []
  air.channel @dst_0 []
  air.channel @dst_1 []

  func.func @full_buffer_write_subregion_read() {
    %c1_outer = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1_outer) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %alloc = memref.alloc() : memref<2x16xbf16, 1>

        // Write: channel.get fills entire L2 buffer (no offsets = full buffer)
        %get = air.channel.get async @src[] (%alloc[] [] []) :
            (memref<2x16xbf16, 1>)

        // Read 1: channel.put reads row 0 — offset [0,0]
        %put0 = air.channel.put async @dst_0[] (%alloc[%c0, %c0] [%c1, %c16] [%c16, %c1]) :
            (memref<2x16xbf16, 1>)

        // Read 2: channel.put reads row 1 — offset [1,0]
        %put1 = air.channel.put async @dst_1[] (%alloc[%c1, %c0] [%c1, %c16] [%c16, %c1]) :
            (memref<2x16xbf16, 1>)

        memref.dealloc %alloc : memref<2x16xbf16, 1>
      }
    }
    return
  }
}
