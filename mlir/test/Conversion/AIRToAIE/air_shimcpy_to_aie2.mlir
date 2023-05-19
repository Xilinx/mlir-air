//===- air_shimcpy_to_aie2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -o out.mlir --split-input-file | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:  %[[VAL_0:.*]] = AIE.tile(2, 3)
// CHECK:  %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:  %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2>
// CHECK:  %[[VAL_5:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:    %[[VAL_6:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:  ^bb1:
// CHECK:    AIE.useLock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:    AIE.dmaBd(<%[[VAL_4]] : memref<1024xi32, 2>, 0, 0>, 0)
// CHECK:    AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:    AIE.nextBd ^bb1
// CHECK:  ^bb2:
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %[[VAL_7:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    AIE.useLock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:    AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:    AIE.end
// CHECK:  AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
func.func @func1(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 0 : index
    %c1024 = arith.constant 0 : index
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy_nd (%buf0[] [] [], %ext0[%c0] [%c1024] [%c1]) : (memref<1024xi32, 2>, memref<1024xi32>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    air.herd_terminator
  }
  return
}

// -----

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK: %[[VAL_0:.*]] = AIE.tile(2, 3)
// CHECK: %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK: %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 1 : i32}
// CHECK: %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32}
// CHECK: %[[VAL_4:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK: %[[VAL_5:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK: %[[VAL_6:.*]] = AIE.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2>
// CHECK: %[[VAL_7:.*]] = AIE.buffer(%[[VAL_0]]) {{.*}} : memref<512xi32, 2>
// CHECK: %[[VAL_8:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:   %[[VAL_9:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_6]] : memref<1024xi32, 2>, 0, 0>, 0)
// CHECK:   AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb2:
// CHECK:   AIE.end
// CHECK: ^bb3:
// CHECK:   %[[VAL_10:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   AIE.useLock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_7]] : memref<512xi32, 2>, 0, 0>, 0)
// CHECK:   AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: }
// CHECK: %[[VAL_11:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   AIE.useLock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:   AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:   AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:   AIE.end
// CHECK: AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK: AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
func.func @func1(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 0 : index
    %c1024 = arith.constant 0 : index
    %c512 = arith.constant 0 : index
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    %buf1 = memref.alloc() : memref<512xi32, 2>
    air.dma_memcpy_nd (%buf0[] [] [], %ext0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>)
    air.dma_memcpy_nd (%ext0[%c0] [%c512] [%c1], %buf1[] [] []) {id = 2 : i32} : (memref<1024xi32>, memref<512xi32, 2>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    memref.dealloc %buf1 : memref<512xi32, 2>
    air.herd_terminator
  }
  return
}
