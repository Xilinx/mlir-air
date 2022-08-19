// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-dependency | FileCheck %s
// Async dependency tracing through air hierarchies

module  {
// CHECK-LABEL: module
  func.func @foo(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%size_x1 = %c1, %size_y1 = %c1) args(%arg4 = %arg0) : memref<1024xi32> {
    //CHECK: %[[EVENT0:.*]] = air.launch async
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %1 = memref.alloc() : memref<512xi32, 3>
      // CHECK: %[[EVENT1:.*]], %[[VAL1:.*]] = air.region async
      air.dma_memcpy (%1, %arg4, [%c0_1], [%c0_1], %c0_1) {id = 1 : i32} : (memref<512xi32, 3>, memref<1024xi32>, [index], [index], index) -> ()
      // CHECK: %[[EVENT2:.*]] = air.dma_memcpy async [{{.*}}%[[EVENT1]]{{.*}}]
      air.partition unroll (%arg5, %arg6) in (%size_x2 = %c1_1, %size_y2 = %c1_1) args(%arg7 = %1) : memref<512xi32, 3> {
      // CHECK: %[[EVENT3:.*]] = air.partition async [{{.*}}%[[EVENT2]]{{.*}}]{{.*}}unroll
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %2 = memref.alloc() : memref<256xi32, 2>
        // CHECK: %[[EVENT4:.*]], %[[VAL4:.*]] = air.region async
        air.dma_memcpy (%2, %arg7, [%c0_2], [%c0_2], %c0_2) {id = 2 : i32} : (memref<256xi32, 2>, memref<512xi32, 3>, [index], [index], index) -> ()
        // CHECK: %[[EVENT5:.*]] = air.dma_memcpy async [{{.*}}%[[EVENT4]]{{.*}}]
        air.herd tile (%arg8, %arg9) in (%arg10=%c1_2, %arg11=%c1_2) args(%arg12=%2) : memref<256xi32, 2> {
        // CHECK: %[[EVENT6:.*]] = air.herd async [{{.*}}%[[EVENT5]]{{.*}}]{{.*}}tile
          %c0_3 = arith.constant 0 : index
          %3 = memref.alloc() : memref<128xi32, 1>
          // CHECK: %[[EVENT7:.*]], %[[VAL7:.*]] = air.region async
          air.dma_memcpy (%3, %arg12, [%c0_3], [%c0_3], %c0_3) {id = 3 : i32} : (memref<128xi32, 1>, memref<256xi32, 2>, [index], [index], index) -> ()
          // CHECK: %[[EVENT8:.*]] = air.dma_memcpy async [{{.*}}%[[EVENT7]]{{.*}}]
          memref.dealloc %3 : memref<128xi32, 1>
          // CHECK: %[[EVENT9:.*]] = air.region async [{{.*}}%[[EVENT8]]{{.*}}]
          air.herd_terminator
        }
        memref.dealloc %2 : memref<256xi32, 2>
        // CHECK: %[[EVENT10:.*]] = air.region async [{{.*}}%[[EVENT6]]{{.*}}]
        air.partition_terminator
      }
      memref.dealloc %1 : memref<512xi32, 3>
      // CHECK: %[[EVENT11:.*]] = air.region async [{{.*}}%[[EVENT3]]{{.*}}]
      air.launch_terminator
    }
    return
  }
}