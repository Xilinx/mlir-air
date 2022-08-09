// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-dependency | FileCheck %s
// XFAIL: *
// Currently undefined air.launch async loop behaviour (IsolatedFromAbove? Or similar to scf loops?)
// Async dependency tracing for air.launch not yet implemented

module  {
// CHECK-LABEL: module
  func.func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1024xi32, 1>
    // CHECK: %[[EVENT0:.*]], %[[EVENT1:.*]] = air.region async
    // CHECK: air.region_terminator 
    air.launch (%arg2, %arg3) in (%size_x = %c1, %size_y = %c1) {
    //CHECK: %[[EVENT2:.*]] = air.launch async [{{.*}}%[[EVENT0]]{{.*}}]
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %1 = memref.alloc() : memref<16xi32, 2>
      // CHECK: %[[EVENT3:.*]], %[[EVENT4:.*]] = air.region async
      // CHECK: air.region_terminator
      air.dma_memcpy (%1, %0, [%c0], [%c16], %c16) {id = 1 : i32} : (memref<16xi32, 2>, memref<1024xi32, 1>, [index], [index], index) -> ()
      // CHECK: %[[EVENT5:.*]] = air.dma_memcpy async [{{.*}}%[[EVENT3]]{{.*}}]
      air.dma_memcpy (%0, %1, [%c16], [%c0], %c16) {id = 2 : i32} : (memref<1024xi32, 1>, memref<16xi32, 2>, [index], [index], index) -> ()
      // CHECK: %[[EVENT6:.*]] = air.dma_memcpy async [{{.*}}%[[EVENT5]]{{.*}}]
      air.launch_terminator
    }
    memref.dealloc %0 : memref<1024xi32, 1>
    // CHECK: %[[EVENT7:.*]] = air.region async [{{.*}}%[[EVENT2]]{{.*}}]
    // CHECK: air.region_terminator
    return
  }
}