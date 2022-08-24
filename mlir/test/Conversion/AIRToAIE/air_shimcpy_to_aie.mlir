// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=2" | FileCheck %s

module {

// CHECK: module @aie.partition_0
// CHECK:         %[[VAL_12:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_10:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_14:.*]] = AIE.lock(%[[VAL_12]], 0)
// CHECK:         %[[VAL_13:.*]] = AIE.buffer(%[[VAL_12]]) {sym_name = {{.*}}} : memref<1024xi32, 2>

// CHECK:    AIE.mem(%[[VAL_12]])  {
// CHECK:           AIE.dmaStart(S2MM0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_13]] : memref<1024xi32, 2>, 0, 0>, 0)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 1)
// CHECK:           br ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_12]])  {
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_10]], DMA : 0, %[[VAL_12]], DMA : 0)
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

}
