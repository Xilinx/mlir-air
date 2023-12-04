//===- air_launch_dma_memcpy_nd.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

module {

// CHECK-LABEL: module
func.func @memcpy_nd(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  air.launch (%arg1, %arg2) in (%size_x = %c4, %size_y = %c1) args(%arg3=%arg0) : memref<4096xi32> attributes {sym_name = "memcpy_nd"} {
  // CHECK: %[[EVENT0:.*]] = air.launch @memcpy_nd async
    %c32 = arith.constant 32 : index
    %0 = arith.muli %arg1, %c32 : index
    // CHECK: %[[EVENT1:.*]], %[[EVENT2:.*]] = air.execute
    %1 = memref.alloc() : memref<32xi32, 2>
    // CHECK: %[[EVENT3:.*]], %[[EVENT4:.*]] = air.execute
    %c1_0 = arith.constant 1 : index
    air.dma_memcpy_nd (%1[] [] [], %arg3[%0] [%c32] [%c1_0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    // CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT1]]{{.*}}, {{.*}}%[[EVENT3]]{{.*}}]
    air.dma_memcpy_nd (%arg3[%0] [%c32] [%c1_0], %1[] [] []) {id = 2 : i32} : (memref<4096xi32>, memref<32xi32, 2>)
    // CHECK: %[[EVENT6:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT5]]{{.*}}]
    memref.dealloc %1 : memref<32xi32, 2>
    // CHECK: %[[EVENT7:.*]] = air.execute [{{.*}}%[[EVENT6]]{{.*}}]
    air.launch_terminator
  }
  return
}

}
