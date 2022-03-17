// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

module {

func @graph(%arg0 : memref<32xi32>, %arg1 : memref<32xi32>, %arg2 : memref<32xi32, 1>, %arg3 : memref<32xi32, 1>) -> () {
  %herd_cols = constant 1 : index
  %herd_rows = constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1, %sp0 = %arg2, %sp1 = %arg3) : memref<32xi32>, memref<32xi32>, memref<32xi32, 1>, memref<32xi32, 1> attributes { sym_name="herd_0"} {
    %c0 = constant 0 : index
    %c32 = constant 32 : index
    air.dma_memcpy_nd (%sp0[%c0][%c32][%c0], %ext0[%c0][%c32][%c0]) {id = 1 : i32} : (memref<32xi32, 1>, memref<32xi32>)
    %buf0 = memref.alloc() : memref<32xi32, 2>
    air.dma_memcpy_nd (%buf0[%c0][%c32][%c0], %sp0[%c0][%c32][%c0]) {id = 2 : i32} : (memref<32xi32, 2>, memref<32xi32, 1>)
    air.dma_memcpy_nd (%sp1[%c0][%c32][%c0], %buf0[%c0][%c32][%c0]) {id = 3 : i32} : (memref<32xi32, 1>, memref<32xi32, 2>)
    memref.dealloc %buf0 : memref<32xi32, 2>
    air.dma_memcpy_nd (%ext1[%c0][%c32][%c0], %sp1[%c0][%c32][%c0]) {id = 4 : i32} : (memref<32xi32>, memref<32xi32, 1>)
    air.herd_terminator
  }
  return
}

}


