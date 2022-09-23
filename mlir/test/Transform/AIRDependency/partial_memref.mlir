// Copyright (C) 2022, Advanced Micro Devices, Inc.

// RUN: air-opt %s -air-dependency | FileCheck %s

// Dependency tracing capable of differentiating different pointers pointing
// to the same memref.

// CHECK-LABEL: func.func @partial_memref
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd async [
// CHECK-NOT: %[[EVENT1:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT0]]
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT0]]
// CHECK: air.herd_terminator

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
module {
  func.func @partial_memref() {
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() : memref<128x128xf32, 2>
    air.herd @herd_0  tile (%arg0, %arg1) in (%arg2=%c8, %arg3=%c2) args(%arg4=%0) : memref<128x128xf32, 2> {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c128 = arith.constant 128 : index
      %1 = affine.apply #map0()[%arg2]
      %2 = affine.apply #map1()[%arg2]
      %3 = memref.alloc() : memref<128x128xf32, 2>
      air.dma_memcpy_nd (%3[%1, %arg1] [%c16, %c32] [%c128, %c1], %arg4[%1, %arg1] [%c16, %c32] [%c128, %c1]) {id = 1 : i32} : (memref<128x128xf32, 2>, memref<128x128xf32, 2>)
      air.dma_memcpy_nd (%3[%2, %arg1] [%c16, %c32] [%c128, %c1], %arg4[%1, %arg1] [%c16, %c32] [%c128, %c1]) {id = 2 : i32} : (memref<128x128xf32, 2>, memref<128x128xf32, 2>)
      air.dma_memcpy_nd (%3[%1, %arg1] [%c16, %c32] [%c128, %c1], %arg4[%1, %arg1] [%c16, %c32] [%c128, %c1]) {id = 3 : i32} : (memref<128x128xf32, 2>, memref<128x128xf32, 2>)
      air.herd_terminator
    }
    return
  }
}