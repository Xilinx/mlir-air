// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module {

func @graph(%arg0 : memref<16x4x2xi32>) -> () {
  %herd_cols = constant 1 : index
  %herd_rows = constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0) : memref<16x4x2xi32> attributes { sym_name="herd_0"} {
    %c0 = constant 0 : index
    %c64 = constant 64 : index
    %c16 = constant 16 : index
    %c4 = constant 4 : index
    %c2 = constant 2 : index
    %buf0 = memref.alloc() : memref<4x2x2xi32, 2>
    air.dma_memcpy_nd (%buf0[%c0, %c0, %c0][%c2, %c2, %c4][%c64, %c16, %c0], %ext0[%c0, %c0, %c0][%c2, %c2, %c4][%c64, %c16, %c0]) {id = 1 : i32} : (memref<4x2x2xi32, 2>, memref<16x4x2xi32>)
    memref.dealloc %buf0 : memref<4x2x2xi32, 2>
    air.herd_terminator
  }
  return
}

}


