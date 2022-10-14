//===- air_hierarchy.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

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
      // CHECK: %[[EVENT1:.*]], %[[VAL1:.*]] = air.execute
      air.dma_memcpy_nd (%1[][][], %arg4[%c0_1][%c0_1][%c0_1]) {id = 1 : i32} : (memref<512xi32, 3>, memref<1024xi32>)
      // CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT1]]{{.*}}]
      air.partition unroll (%arg5, %arg6) in (%size_x2 = %c1_1, %size_y2 = %c1_1) args(%arg7 = %1) : memref<512xi32, 3> {
      // CHECK: %[[EVENT3:.*]] = air.partition async [{{.*}}%[[EVENT2]]{{.*}}]{{.*}}unroll
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %2 = memref.alloc() : memref<256xi32, 2>
        // CHECK: %[[EVENT4:.*]], %[[VAL4:.*]] = air.execute
        air.dma_memcpy_nd (%2[][][], %arg7[%c0_2][%c0_2][%c0_2]) {id = 2 : i32} : (memref<256xi32, 2>, memref<512xi32, 3>)
        // CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT4]]{{.*}}]
        air.herd tile (%arg8, %arg9) in (%arg10=%c1_2, %arg11=%c1_2) args(%arg12=%2) : memref<256xi32, 2> {
        // CHECK: %[[EVENT6:.*]] = air.herd async [{{.*}}%[[EVENT5]]{{.*}}]{{.*}}tile
          %c0_3 = arith.constant 0 : index
          %3 = memref.alloc() : memref<128xi32, 1>
          // CHECK: %[[EVENT7:.*]], %[[VAL7:.*]] = air.execute
          air.dma_memcpy_nd (%3[][][], %arg12[%c0_3][%c0_3][%c0_3]) {id = 3 : i32} : (memref<128xi32, 1>, memref<256xi32, 2>)
          // CHECK: %[[EVENT8:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT7]]{{.*}}]
          memref.dealloc %3 : memref<128xi32, 1>
          // CHECK: %[[EVENT9:.*]] = air.execute [{{.*}}%[[EVENT8]]{{.*}}]
          air.herd_terminator
        }
        memref.dealloc %2 : memref<256xi32, 2>
        // CHECK: %[[EVENT10:.*]] = air.execute [{{.*}}%[[EVENT6]]{{.*}}]
        air.partition_terminator
      }
      memref.dealloc %1 : memref<512xi32, 3>
      // CHECK: %[[EVENT11:.*]] = air.execute [{{.*}}%[[EVENT3]]{{.*}}]
      air.launch_terminator
    }
    return
  }
}