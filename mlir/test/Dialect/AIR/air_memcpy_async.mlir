// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s | FileCheck %s
module {

// CHECK-LABEL: module
// CHECK: func.func @foo
func.func @foo(%arg0 : memref<16x16xf32>, %arg1 : memref<16x16xf32>) -> () {
  %cst1 = arith.constant 1 : index

  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) args(%ext0 = %arg0, %ext1 = %arg1) : memref<16x16xf32>, memref<16x16xf32> attributes { "foo" = "bar" } {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %src0 = memref.alloc() : memref<16x16xf32, 2>
    %dst0 = memref.alloc() : memref<16x16xf32, 2>
    %e0 = air.wait_all async
    %e = air.dma_memcpy_2d async [%e0] (%ext0, %src0, [%c0, %c0], [%c0, %c0], %c256, %c256, %c256) : (memref<16x16xf32>, memref<16x16xf32, 2>, [index, index], [index, index], index, index, index) -> ()
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %0 = affine.load %src0[%arg4, %arg5] : memref<16x16xf32, 2>
        %cst = arith.constant 1.000000e+00 : f32
        %1 = arith.addf %0, %cst : f32
        affine.store %1, %dst0[%arg4, %arg5] : memref<16x16xf32, 2>
      }
    }
    "air.dma_memcpy_2d"(%dst0, %ext1, %c0, %c0, %c0, %c0, %c256, %c256, %c256) : (memref<16x16xf32, 2>, memref<16x16xf32>, index, index, index, index, index, index, index) -> ()
    air.herd_terminator
  }
  return
}

// CHECK: func.func @memcpy_nd
func.func @memcpy_nd(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  air.herd tile (%arg1, %arg2) in (%arg3=%c4, %arg4=%c1) args(%arg5=%arg0) : memref<4096xi32>attributes {sym_name = "memcpy_nd"} {
    %c32 = arith.constant 32 : index
    %0 = arith.muli %arg1, %c32 : index
    %1 = memref.alloc() : memref<32xi32, 2>
    %c1_0 = arith.constant 1 : index
    // CHECK: air.dma_memcpy_nd
    air.dma_memcpy_nd (%1[] [] [], %arg5[%0] [%c32] [%c1_0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    air.dma_memcpy_nd (%arg5[%0] [%c32] [%c1_0], %1[] [] []) {id = 2 : i32} : (memref<4096xi32>, memref<32xi32, 2>)
    memref.dealloc %1 : memref<32xi32, 2>
    air.herd_terminator
  }
  return
}

}
