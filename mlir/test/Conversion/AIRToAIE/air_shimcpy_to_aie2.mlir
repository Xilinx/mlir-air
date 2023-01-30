//===- air_shimcpy_to_aie.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=2" | FileCheck %s

module {

// CHECK: module @aie.partition_0
// CHECK:         %[[VAL_12:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_10:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_15:.*]] = AIE.lock(%[[VAL_12]], 1)
// CHECK:         %[[VAL_14:.*]] = AIE.lock(%[[VAL_12]], 0)
// CHECK:         %[[VAL_13:.*]] = AIE.buffer(%[[VAL_12]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_16:.*]] = AIE.buffer(%[[VAL_12]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_12]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_13]] : memref<1024xi32, 2>, 0, 0>, 0)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_15]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_16]] : memref<512xi32, 2>, 0, 0>, 0)
// CHECK:           AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_12]])  {
// CHECK:           AIE.useLock(%[[VAL_15]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_15]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_10]], DMA : 0, %[[VAL_12]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_10]], DMA : 1, %[[VAL_12]], DMA : 1)
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
    air.dma_memcpy_nd (%buf1[] [] [], %ext0[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<512xi32, 2>, memref<1024xi32>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    memref.dealloc %buf1 : memref<512xi32, 2>
    air.herd_terminator
  }
  return
}

}
