// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module {

func.func @graph(%arg0 : memref<32x16xi32>, %arg1 : memref<32x16xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<32x16xi32>, memref<32x16xi32> attributes { sym_name="herd_0"} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %buf0 = memref.alloc() {sym_name = "scratch"}: memref<16x8xi32, 2>
    %buf1 = memref.alloc() {sym_name = "scratch_copy"}: memref<16x8xi32, 2>
    air.dma_memcpy_nd (%buf0[%c0, %c0][%c8, %c16][%c32, %c0], %ext0[%c8, %c0][%c8, %c16][%c32, %c0]) {id = 1 : i32} : (memref<16x8xi32, 2>, memref<32x16xi32>)
    affine.for %j = 0 to 8 {
      affine.for %i = 0 to 16 {
        %0 = affine.load %buf0[%i, %j] : memref<16x8xi32, 2>
        affine.store %0, %buf1[%i, %j] : memref<16x8xi32, 2>
      }
    }
    air.dma_memcpy_nd (%ext1[%c8, %c0][%c8, %c16][%c32, %c0], %buf1[%c0, %c0][%c8, %c16][%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>, memref<16x8xi32, 2>)
    memref.dealloc %buf1 : memref<16x8xi32, 2>
    memref.dealloc %buf0 : memref<16x8xi32, 2>
    air.herd_terminator
  }
  return
}

}


