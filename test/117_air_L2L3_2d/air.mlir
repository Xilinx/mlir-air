// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

module {

func.func @graph(%arg0 : memref<16x16xi32>, %arg1 : memref<16x16xi32>, %arg2 : memref<4x8xi32, 1>) -> () {
  %herd_cols = constant 1 : index
  %herd_rows = constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1, %sp0 = %arg2) : memref<16x16xi32>, memref<16x16xi32>, memref<4x8xi32, 1> attributes { sym_name="herd_0"} {
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c8 = constant 8 : index
    %c16 = constant 16 : index
    air.dma_memcpy_nd (%sp0[%c0,%c0][%c8,%c4][%c4,%c0], %ext0[%c0,%c0][%c8,%c4][%c16,%c0]) {id = 1 : i32} : (memref<4x8xi32, 1>, memref<16x16xi32>)
    air.dma_memcpy_nd (%ext1[%c0,%c0][%c8,%c4][%c16,%c0], %sp0[%c0,%c0][%c8,%c4][%c4,%c0]) {id = 2 : i32} : (memref<16x16xi32>, memref<4x8xi32, 1>)
    air.herd_terminator
  }
  return
}

}


