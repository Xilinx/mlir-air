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
// Before the fix to `getIteratorFromMDVector` (mlir/lib/Util/Util.cpp), the
// formula was row-major (idx[0] slow) while shim_dma_allocation enumeration
// happened in tile-allocation order (col fast). The two coincide only when
// any dim == 1, so 1D-style herds passed but a true 2D herd misrouted
// off-diagonal channel instances.
//
// This test checks the col-major-fixed behavior: for a 2x2 herd with channel
// @outD [2, 2], the launch-side getter for indices=[1, 0] picks
// metadataArray slot 1 (linIdx = 1 + 0*2 = 1), which is the shim allocation
// for tile (col=1, row=2)'s output — the matching producer.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// CHECK-DAG: aie.shim_dma_allocation @air_outD_0(%logical_shim_noc, S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_outD_1(%logical_shim_noc_0, S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_outD_2(%logical_shim_noc, S2MM, 1)
// CHECK-DAG: aie.shim_dma_allocation @air_outD_3(%logical_shim_noc_0, S2MM, 1)

// Metadata order matches col-major linearization
// (idx[0] fast: outD_0=[0,0], outD_1=[1,0], outD_2=[0,1], outD_3=[1,1]).
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
      air.channel.get @outD[%c1, %c0] (%output[%c8] [%c8] [%c1]) {id = 11 : i32} : (memref<32xbf16>)
      air.channel.get @outD[%c0, %c1] (%output[%c16] [%c8] [%c1]) {id = 12 : i32} : (memref<32xbf16>)
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
