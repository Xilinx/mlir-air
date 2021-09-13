// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s | FileCheck %s
// CHECK: module
module  {
  func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    %0 = airrt.alloc : memref<1024xi32, 1>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        %c16 = constant 16 : index
        %c1_i32 = constant 1 : i32
        %1 = index_cast %arg3 : index to i64
        %2 = index_cast %arg2 : index to i64
        %3 = index_cast %c16 : index to i64
        airrt.dma_memcpy(%c1_i32, %1, %2, %0[%3], %3) : (i32, i64, i64, memref<1024xi32, 1>, [i64], i64) -> ()
        %c2_i32 = constant 2 : i32
        airrt.dma_memcpy(%c2_i32, %1, %2, %0[%3], %3) : (i32, i64, i64, memref<1024xi32, 1>, [i64], i64) -> ()
      } {air.herd_launch = "inner"}
    } {air.herd_launch = "outer"}
    airrt.dealloc %0 : memref<1024xi32, 1>
    return
  }
}

// CHECK: module
module  {
  func @foo(%arg0: memref<256x256xi32>, %arg1: memref<128x128xi32, 1>) {
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        %c16 = constant 16 : index
        %c0 = constant 0 : i64
        %c1 = constant 1 : i64
        %c1_i32 = constant 1 : i32
        %c2_i32 = constant 2 : i32
        %c128_i64 = constant 128 : i64
        %c256_i64 = constant 256 : i64
        %1 = index_cast %arg3 : index to i64
        %2 = index_cast %arg2 : index to i64
        %3 = index_cast %c16 : index to i64
        airrt.dma_memcpy_nd(%c1_i32, %1, %2, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %3, %3], [%c0, %c0, %c256_i64]) : (i32, i64, i64, memref<256x256xi32>, [i64,i64,i64,i64], [i64,i64,i64,i64], [i64,i64,i64]) -> ()
        airrt.dma_memcpy_nd(%c2_i32, %1, %2, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %3, %3], [%c0, %c0, %c128_i64]) {attr = "attr"} : (i32, i64, i64, memref<128x128xi32, 1>, [i64,i64,i64,i64], [i64,i64,i64,i64], [i64,i64,i64]) -> ()
      } {air.herd_launch = "inner"}
    } {air.herd_launch = "outer"}
    return
  }
}