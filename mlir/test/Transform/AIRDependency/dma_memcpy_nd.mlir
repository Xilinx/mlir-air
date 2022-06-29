// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-dependency | FileCheck %s

module {

// CHECK-LABEL: module
func.func @memcpy_nd(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  air.launch_herd tile (%arg1, %arg2) in (%arg3=%c4, %arg4=%c1) args(%arg5=%arg0) : memref<4096xi32>attributes {sym_name = "memcpy_nd"} {
  // CHECK: %[[EVENT0:.*]] = air.launch_herd async
    %c32 = arith.constant 32 : index
    %0 = arith.muli %arg1, %c32 : index
    // CHECK: %[[EVENT1:.*]], %[[EVENT2:.*]] = air.region async
    // CHECK: air.region_terminator
    %1 = memref.alloc() : memref<32xi32, 2>
    // CHECK: %[[EVENT3:.*]], %[[EVENT4:.*]] = air.region async
    // CHECK: air.region_terminator
    %c1_0 = arith.constant 1 : index
    air.dma_memcpy_nd (%1[] [] [], %arg5[%0] [%c32] [%c1_0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    // CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT3]]{{.*}}, {{.*}}%[[EVENT1]]{{.*}}]
    air.dma_memcpy_nd (%arg5[%0] [%c32] [%c1_0], %1[] [] []) {id = 2 : i32} : (memref<4096xi32>, memref<32xi32, 2>)
    // CHECK: %[[EVENT6:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT5]]{{.*}}]
    memref.dealloc %1 : memref<32xi32, 2>
    // CHECK: %[[EVENT7:.*]] = air.region async [{{.*}}%[[EVENT6]]{{.*}}]
    // CHECK: air.region_terminator
    air.herd_terminator
  }
  return
}

}