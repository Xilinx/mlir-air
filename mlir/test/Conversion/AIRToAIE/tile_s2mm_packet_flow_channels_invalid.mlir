//===- tile_s2mm_packet_flow_channels_invalid.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" -verify-diagnostics

// Three inbound packet flows into one tile, all received in a loop. No two can
// share a BD ring and the tile has two S2MM channels, so this is an error
// rather than a design that delivers into the wrong buffers.

air.channel @fA [1] {channel_type = "npu_dma_packet"}
air.channel @fB [1] {channel_type = "npu_dma_packet"}
air.channel @fC [1] {channel_type = "npu_dma_packet"}
func.func @three_repeat(%a: memref<64x32xbf16>, %b: memref<64x32xbf16>, %c: memref<64x32xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %ab=%b, %ac=%c) : memref<64x32xbf16>, memref<64x32xbf16>, memref<64x32xbf16> {
    air.segment @seg args(%sa=%aa, %sb=%ab, %sc=%ac) : memref<64x32xbf16>, memref<64x32xbf16>, memref<64x32xbf16> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0_0 to %c4 step %c1_0 {
        air.channel.put @fA[] (%sa[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 1 : i32} : (memref<64x32xbf16>)
        air.channel.put @fB[] (%sb[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 2 : i32} : (memref<64x32xbf16>)
        air.channel.put @fC[] (%sc[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 3 : i32} : (memref<64x32xbf16>)
      }
      air.herd @h tile (%tx, %ty) in (%sx2=%c1_0, %sy2=%c1_0) {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c4_1 = arith.constant 4 : index
        %ba = memref.alloc() : memref<1x8xbf16, 2>
        %bb = memref.alloc() : memref<1x8xbf16, 2>
        %bc = memref.alloc() : memref<1x8xbf16, 2>
        scf.for %j = %c0_1 to %c4_1 step %c1_1 {
          air.channel.get @fA[] (%ba[] [] []) {id = 4 : i32} : (memref<1x8xbf16, 2>)
          air.channel.get @fB[] (%bb[] [] []) {id = 5 : i32} : (memref<1x8xbf16, 2>)
          // expected-error @+1 {{no S2MM channel left for this inbound packet flow}}
          air.channel.get @fC[] (%bc[] [] []) {id = 6 : i32} : (memref<1x8xbf16, 2>)
        }
        memref.dealloc %ba : memref<1x8xbf16, 2>
        memref.dealloc %bb : memref<1x8xbf16, 2>
        memref.dealloc %bc : memref<1x8xbf16, 2>
      }
    }
  }
  return
}
