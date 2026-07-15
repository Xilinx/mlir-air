//===- channel_2d_both_dims_gt1_routing.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for AIR 2D-channel index linearization. When a channel is
// declared `size=[X, Y]` with both X>1 AND Y>1, the launch-side
// `air.channel.get/put` and the herd-side tile placement must agree on a
// single linearization of `indices=[i, j]` to a metadataArray slot.
//
// `getIteratorFromMDVector` uses row-major (idx[0] slow, idx[N-1] fast). For
// that to pair correctly with the per-device shim-allocation enumeration in
// `createShimDMAAllocationOpsImpl`, the alloc bucket must be sorted by far-
// end tile (col, row) lex — col slow, row fast — so that physical tile
// (x_loc+i, y_loc+j) gets per-device-idx `i*y_size + j`. Without the sort,
// the bucket arrived in tile-allocation order (col fast, row slow) and
// off-diagonal channel instances routed to the wrong shim allocation. The
// two orderings coincide whenever any dim == 1, so 1D-style herds were
// unaffected.
//
// This test exercises a 2x2 herd with channel @outD [2, 2]: a launch-side
// `air.channel.get @outD[1, 0]` must resolve to the shim allocation for
// tile (col=1, row=2)'s output — the matching producer for herd indices
// (tx=1, ty=0).

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// CHECK-DAG: aie.shim_dma_allocation @air_outD_0(%logical_shim_noc, S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_outD_1(%logical_shim_noc, S2MM, 1)
// CHECK-DAG: aie.shim_dma_allocation @air_outD_2(%logical_shim_noc_0, S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_outD_3(%logical_shim_noc_0, S2MM, 1)

// Per-device alloc index for tile (col, row) = (col-x_loc)*y_size + (row-y_loc).
// For 2x2 with x_loc=0, y_loc=2:
//   tile (0,2) → outD_0     tile (0,3) → outD_1
//   tile (1,2) → outD_2     tile (1,3) → outD_3
// Row-major formula `idx[0]*dims[1] + idx[1]` then picks the matching slot:
//   indices=[0,0]→0  [0,1]→1  [1,0]→2  [1,1]→3
// CHECK: air.channel.get @outD[%c0, %c0]
// CHECK-SAME: metadataArray = [{base = "air_outD_0"
// CHECK-SAME:                   {base = "air_outD_1"
// CHECK-SAME:                   {base = "air_outD_2"
// CHECK-SAME:                   {base = "air_outD_3"

module {
  air.channel @outD [2, 2]

  func.func @test_2d_channel_routing(%out: memref<32xbf16>) {
    %0 = air.launch async () in () args(%output=%out) : memref<32xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index

      air.channel.get @outD[%c0, %c0] (%output[%c0] [%c8] [%c1]) {id = 10 : i32} : (memref<32xbf16>)
      air.channel.get @outD[%c0, %c1] (%output[%c8] [%c8] [%c1]) {id = 11 : i32} : (memref<32xbf16>)
      air.channel.get @outD[%c1, %c0] (%output[%c16] [%c8] [%c1]) {id = 12 : i32} : (memref<32xbf16>)
      air.channel.get @outD[%c1, %c1] (%output[%c24] [%c8] [%c1]) {id = 13 : i32} : (memref<32xbf16>)

      %seg = air.segment @seg async
          attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 2 : i64} {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %herd = air.herd @h async tile (%tx, %ty) in (%htx=%c2_s, %hty=%c2_s)
            attributes {id = 3 : i32} {
          %l1 = memref.alloc() : memref<8xbf16, 2>
          air.channel.put @outD[%tx, %ty] (%l1[] [] []) {id = 20 : i32} : (memref<8xbf16, 2>)
          memref.dealloc %l1 : memref<8xbf16, 2>
        }
      }
    }
    return
  }
}
