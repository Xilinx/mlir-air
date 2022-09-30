//===- prune_redundant_edges.mlir ------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// Prune redundant dependency edges
// CHECK: %[[EVENT0:.*]] = air.partition async
// CHECK: %[[EVENT1:.*]] = air.herd async
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd async{{.*}}id = 3
// CHECK-NEXT: %[[EVENT3:.*]] = air.execute async [%[[EVENT2]]]
// CHECK: %[[EVENT4:.*]] = air.execute async [%[[EVENT1]]]
// CHECK: %[[EVENT5:.*]] = air.execute async [%[[EVENT0]]]

module {
  func.func @foo(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<1024xi32> attributes {id = 3 : i32} {
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %asyncToken, %valOut = air.execute async  {
        %3 = memref.alloc() : memref<512xi32, 3>
        air.execute_terminator %3 : memref<512xi32, 3>
      } {id = 1 : i32} : (memref<512xi32, 3>)
      %1 = air.dma_memcpy_nd async [%asyncToken] (%valOut[] [] [], %arg5[%c0_0] [%c0_0] [%c0_0]) {id = 1 : i32} : (memref<512xi32, 3>, memref<1024xi32>)
      %2 = air.partition async [%1]  unroll(%arg6, %arg7) in (%arg8=%c1_1, %arg9=%c1_1) args(%arg10=%valOut) : memref<512xi32, 3> attributes {id = 2 : i32} {
        %c0_3 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        %asyncToken_5, %valOut_6 = air.execute async  {
          %5 = memref.alloc() : memref<256xi32, 2>
          air.execute_terminator %5 : memref<256xi32, 2>
        } {id = 2 : i32} : (memref<256xi32, 2>)
        %3 = air.dma_memcpy_nd async [%asyncToken_5] (%valOut_6[] [] [], %arg10[%c0_3] [%c0_3] [%c0_3]) {id = 2 : i32} : (memref<256xi32, 2>, memref<512xi32, 3>)
        %4 = air.herd async [%3]  tile (%arg11, %arg12) in (%arg13=%c1_4, %arg14=%c1_4) args(%arg15=%valOut_6) : memref<256xi32, 2> attributes {id = 1 : i32} {
          %c0_8 = arith.constant 0 : index
          %asyncToken_9, %valOut_10 = air.execute async  {
            %6 = memref.alloc() : memref<128xi32, 1>
            air.execute_terminator %6 : memref<128xi32, 1>
          } {id = 3 : i32} : (memref<128xi32, 1>)
          %5 = air.dma_memcpy_nd async [%asyncToken_9] (%valOut_10[] [] [], %arg15[%c0_8] [%c0_8] [%c0_8]) {id = 3 : i32} : (memref<128xi32, 1>, memref<256xi32, 2>)
          %asyncToken_11 = air.execute async [%5, %asyncToken_9]  : (!air.async.token, !air.async.token) {
            memref.dealloc %valOut_10 : memref<128xi32, 1>
            air.execute_terminator
          } {id = 4 : i32}
          air.herd_terminator
        }
        %asyncToken_7 = air.execute async [%4, %asyncToken_5, %3]  : (!air.async.token, !air.async.token, !air.async.token) {
          memref.dealloc %valOut_6 : memref<256xi32, 2>
          air.execute_terminator
        } {id = 5 : i32}
        air.partition_terminator
      }
      %asyncToken_2 = air.execute async [%2, %1]  : (!air.async.token, !air.async.token) {
        memref.dealloc %valOut : memref<512xi32, 3>
        air.execute_terminator
      } {id = 6 : i32}
      air.launch_terminator
    }
    return
  }
}