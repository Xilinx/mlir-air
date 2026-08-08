//===- shim_metadata_bundle_order.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// `metadataArray` is consumed POSITIONALLY, by the linearized channel bundle
// index (air::getIndexToMetadataArrayFromChannelIndices). Its entries are
// appended in shim ALLOCATION order, so the shim allocation bucket must itself
// be ordered by the bundle index each allocation serves. Order it by anything
// that merely correlates with placement -- e.g. the far-end tile's (col, row)
// -- and pinning endpoints to columns out of bundle order makes every endpoint
// drive the wrong physical stream, silently: the design still lowers, routes
// and runs, it just permutes its data.
//
// Endpoint [0, 0] is pinned to memtile column 5 and endpoint [0, 1] to column
// 1, so tile-placement order (col 1 first) is the REVERSE of bundle order.
// `air_qin_0` must therefore name the COLUMN-5 stream, which the flows below
// tie back to the memtile the [0, 0] endpoint was pinned to.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2 use-objectfifo=false' | FileCheck %s

// CHECK: %[[SHIM_A:.*]] = aie.logical_tile<ShimNOCTile>
// CHECK: %[[SHIM_B:.*]] = aie.logical_tile<ShimNOCTile>
// CHECK: %[[MEM5:.*]] = aie.logical_tile<MemTile>(5, ?)
// CHECK: %[[MEM1:.*]] = aie.logical_tile<MemTile>(1, ?)

// SHIM_A serves memtile column 5 (bundle index 0), SHIM_B column 1 (index 1).
// CHECK: aie.flow(%[[SHIM_A]], DMA : 0, %[[MEM5]], DMA : 0)
// CHECK: aie.flow(%[[SHIM_B]], DMA : 0, %[[MEM1]], DMA : 0)

// So slot 0 of each bundle must be named _0, on the column-5 stream. Reverse
// these two and bundle index 0 drives the column-1 stream instead.
// CHECK: aie.shim_dma_allocation @air_outc_0(%[[SHIM_A]], S2MM, 0)
// CHECK: aie.shim_dma_allocation @air_outc_1(%[[SHIM_B]], S2MM, 0)
// CHECK: aie.shim_dma_allocation @air_qin_0(%[[SHIM_A]], MM2S, 0)
// CHECK: aie.shim_dma_allocation @air_qin_1(%[[SHIM_B]], MM2S, 0)

// Host-side feed and drain then index that naming directly.
// CHECK: air.channel.put{{.*}}@qin{{.*}}metadataArray = [{base = "air_qin_0", index = 0 : i32}, {base = "air_qin_1", index = 1 : i32}]
// CHECK: air.channel.get{{.*}}@outc{{.*}}metadataArray = [{base = "air_outc_0", index = 0 : i32}, {base = "air_outc_1", index = 1 : i32}]

module {
  air.channel @qin [1, 2]
  air.channel @outc [1, 2]
  air.channel @l1a [1, 1]
  air.channel @l1b [1, 1]
  func.func @md(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%li) in (%ls=%c1) args(%a0=%arg0, %a1=%arg1) : memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      air.channel.put @qin[%c0, %c0] (%a0[] [] []) : (memref<64xi32>)
      air.channel.put @qin[%c0, %c1_l] (%a0[] [] []) : (memref<64xi32>)
      air.channel.get @outc[%c0, %c0] (%a1[] [] []) : (memref<64xi32>)
      air.channel.get @outc[%c0, %c1_l] (%a1[] [] []) : (memref<64xi32>)
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        %c0_0 = arith.constant 0 : index
        %tk0, %b0 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 5 : i32} : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %tk1, %b1 = air.execute -> (memref<64xi32, 1>) {
          %a = memref.alloc() {air.memtile_col = 1 : i32} : memref<64xi32, 1>
          air.execute_terminator %a : memref<64xi32, 1>
        }
        %g0 = air.channel.get async [%tk0] @qin[%c0_0, %c0_0] (%b0[] [] []) : (memref<64xi32, 1>)
        %g1 = air.channel.get async [%tk1] @qin[%c0_0, %c1_0] (%b1[] [] []) : (memref<64xi32, 1>)
        %f0 = air.channel.put async [%g0] @l1a[] (%b0[] [] []) : (memref<64xi32, 1>)
        %f1 = air.channel.put async [%g1] @l1b[] (%b1[] [] []) : (memref<64xi32, 1>)
        %p0 = air.channel.put async [%f0] @outc[%c0_0, %c0_0] (%b0[] [] []) : (memref<64xi32, 1>)
        %p1 = air.channel.put async [%f1] @outc[%c0_0, %c1_0] (%b1[] [] []) : (memref<64xi32, 1>)
        %h0 = air.herd @h5 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 5 : i64, y_loc = 2 : i64} {
          %tkl, %l1 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %gl = air.channel.get async [%tkl] @l1a[] (%l1[] [] []) : (memref<64xi32, 2>)
          %d = air.execute [%gl] { memref.dealloc %l1 : memref<64xi32, 2> }
        }
        %h1 = air.herd @h1 async tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0) attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
          %tkl2, %l12 = air.execute -> (memref<64xi32, 2>) {
            %b = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %b : memref<64xi32, 2>
          }
          %gl2 = air.channel.get async [%tkl2] @l1b[] (%l12[] [] []) : (memref<64xi32, 2>)
          %d2 = air.execute [%gl2] { memref.dealloc %l12 : memref<64xi32, 2> }
        }
      }
    }
    return
  }
}
