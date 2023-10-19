//===- air_to_ipu_spatial_add_one.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -pass-pipeline='builtin.module(func.func(convert-linalg-to-affine-loops), air-to-aie{row-offset=2 col-offset=0 device=ipu})' | FileCheck %s
// CHECK: %[[VAL0:.*]] = AIE.tile(0, 1)
// CHECK: %[[VAL1:.*]] = AIE.tile(0, 2)
// CHECK: %[[VAL2:.*]] = AIE.tile(0, 0)
// CHECK: %[[VAL3:.*]] = AIE.lock(%[[VAL0]], 3) {init = 1 : i32}
// CHECK: %[[VAL4:.*]] = AIE.lock(%[[VAL0]], 2) {init = 0 : i32}
// CHECK: %[[VAL5:.*]] = AIE.lock(%[[VAL0]], 1) {init = 1 : i32}
// CHECK: %[[VAL6:.*]] = AIE.lock(%[[VAL0]], 0) {init = 0 : i32}
// CHECK: %[[VAL7:.*]] = AIE.lock(%[[VAL1]], 3) {init = 1 : i32}
// CHECK: %[[VAL8:.*]] = AIE.lock(%[[VAL1]], 2) {init = 0 : i32}
// CHECK: %[[VAL9:.*]] = AIE.lock(%[[VAL1]], 1) {init = 1 : i32}
// CHECK: %[[VAL10:.*]] = AIE.lock(%[[VAL1]], 0) {init = 0 : i32}
// CHECK: %[[VAL11:.*]] = AIE.buffer(%[[VAL0]]) {sym_name = {{.*}}} : memref<64xi32, 1>
// CHECK: %[[VAL12:.*]] = AIE.buffer(%[[VAL0]]) {sym_name = {{.*}}} : memref<64xi32, 1>
// CHECK: %[[VAL13:.*]] = AIE.buffer(%[[VAL1]]) {sym_name = {{.*}}} : memref<64xi32, 2>
// CHECK: %[[VAL14:.*]] = AIE.buffer(%[[VAL1]]) {sym_name = {{.*}}} : memref<64xi32, 2>
// CHECK: AIE.mem(%[[VAL1]]) {
// CHECK:   AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL7]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL13]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:   AIE.useLock(%[[VAL8]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb3:  // pred: ^bb0
// CHECK:   AIE.dmaStart(MM2S, 0, ^bb4,
// CHECK: ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:   AIE.useLock(%[[VAL10]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL14]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:   AIE.useLock(%[[VAL9]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: }
// CHECK: AIE.core(%[[VAL1]]) {
// CHECK:   %[[VAL15:.*]] = arith.constant 1 : i32
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   AIE.useLock(%[[VAL8]], AcquireGreaterEqual, 1)
// CHECK:   affine.for %[[VAL16:.*]] = 0 to 64 {
// CHECK:     %[[VAL17:.*]] = affine.load %[[VAL13]][%[[VAL16]]] : memref<64xi32, 2>
// CHECK:     %[[VAL18:.*]] = arith.addi %[[VAL17]], %[[VAL15]] : i32
// CHECK:     affine.store %[[VAL18]], %[[VAL14]][%[[VAL16]]] : memref<64xi32, 2>
// CHECK:   }
// CHECK:   AIE.useLock(%[[VAL9]], AcquireGreaterEqual, 1)
// CHECK:   AIE.useLock(%[[VAL7]], Release, 1)
// CHECK:   AIE.useLock(%[[VAL10]], Release, 1)
// CHECK:   AIE.end
// CHECK: }
// CHECK: AIE.flow(%[[VAL2]], DMA : 0, %[[VAL0]], DMA : 0)
// CHECK: AIE.flow(%[[VAL0]], DMA : 0, %[[VAL1]], DMA : 0)
// CHECK: AIE.flow(%[[VAL1]], DMA : 0, %[[VAL0]], DMA : 1)
// CHECK: AIE.flow(%[[VAL0]], DMA : 1, %[[VAL2]], DMA : 0)
// CHECK: AIE.memTileDMA(%[[VAL0]]) {
// CHECK:   AIE.dmaStart(S2MM, 0, ^bb1, ^bb7)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL5]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL11]] : memref<64xi32, 1>, 0, 64>, 0)
// CHECK:   AIE.useLock(%[[VAL6]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb3:
// CHECK:   AIE.dmaStart(S2MM, 1, ^bb4
// CHECK: ^bb4:
// CHECK:   AIE.useLock(%[[VAL3]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL12]] : memref<64xi32, 1>, 0, 64>, 0)
// CHECK:   AIE.useLock(%[[VAL4]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: ^bb5:
// CHECK:   AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
// CHECK: ^bb6:
// CHECK:   AIE.useLock(%[[VAL6]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL11]] : memref<64xi32, 1>, 0, 64>, 0)
// CHECK:   AIE.useLock(%[[VAL5]], Release, 1)
// CHECK:   AIE.nextBd ^bb6
// CHECK: ^bb7:
// CHECK:   AIE.dmaStart(MM2S, 1, ^bb8, ^bb5)
// CHECK: ^bb8:
// CHECK:   AIE.useLock(%[[VAL4]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL12]] : memref<64xi32, 1>, 0, 64>, 0)
// CHECK:   AIE.useLock(%[[VAL3]], Release, 1)
// CHECK:   AIE.nextBd ^bb8
// CHECK: }

#map2 = affine_map<(d0) -> (d0)>
air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
air.channel @channel_2 [1, 1]
air.channel @channel_3 [1, 1]
func.func @func0(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  air.channel.put @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
  air.segment @segment0 {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %memtile0 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @channel_0[] (%memtile0[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
    air.channel.put @channel_1[] (%memtile0[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    memref.dealloc %memtile0 : memref<64xi32, 1>
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="func4"} {
      %buf0 = memref.alloc() : memref<64xi32, 2>
      %buf1 = memref.alloc() : memref<64xi32, 2>
      air.channel.get @channel_1[%tx, %ty] (%buf0[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
      linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%buf0 : memref<64xi32, 2>) outs(%buf1 : memref<64xi32, 2>) {
      ^bb0(%arg11: i32, %arg12: i32):
        %c1_32 = arith.constant 1 : i32
        %12 = arith.addi %arg11, %c1_32 : i32
        linalg.yield %12 : i32
      }
      air.channel.put @channel_2[%tx, %ty] (%buf1[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
      memref.dealloc %buf0 : memref<64xi32, 2>
      memref.dealloc %buf1 : memref<64xi32, 2>
      air.herd_terminator
    }
    %memtile1 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @channel_2[] (%memtile1[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
    air.channel.put @channel_3[] (%memtile1[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
    memref.dealloc %memtile1 : memref<64xi32, 1>
    air.segment_terminator
  }
  air.channel.get @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
  return
}
