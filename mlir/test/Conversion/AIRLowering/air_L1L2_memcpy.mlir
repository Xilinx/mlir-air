// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-to-std | FileCheck %s
// CHECK: %0 = airrt.alloc : memref<1024xi32, 1>
// CHECK: airrt.dma_memcpy(%c1_i32, {{.*}}, {{.*}}, %0[{{.*}}], {{.*}}) : (i32, i64, i64, memref<1024xi32, 1>, [i64], i64)
// CHECK: airrt.dma_memcpy(%c2_i32, {{.*}}, {{.*}}, %0[{{.*}}], {{.*}}) : (i32, i64, i64, memref<1024xi32, 1>, [i64], i64)
// CHECK: airrt.dealloc %0 : memref<1024xi32, 1>
module  {
  func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1024xi32, 1>
    air.launch_herd tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%0, %arg7=%arg1) : memref<1024xi32, 1>,memref<1024xi32> {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %1 = memref.alloc() : memref<16xi32, 2>
      air.dma_memcpy (%1, %arg6, [%c0], [%c16], %c16) {id = 1 : i32} : (memref<16xi32, 2>, memref<1024xi32, 1>, [index], [index], index) -> ()
      air.dma_memcpy (%arg6, %1, [%c16], [%c0], %c16) {id = 2 : i32} : (memref<1024xi32, 1>, memref<16xi32, 2>, [index], [index], index) -> ()
      air.herd_terminator
    }
    memref.dealloc %0 : memref<1024xi32, 1>
    return
  }
}