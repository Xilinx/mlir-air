//===- herd_only.mlir ------------------------------*- MLIR -*-===//
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

// Prune redundant dependency edges, when there are only air herds
// CHECK: %[[EVENT0:.*]] = scf.for{{.*}}iter_args(%[[EVENT1:.*]] = 
// CHECK: %[[EVENT2:.*]], %[[VALUE0:.*]] = air.execute [%[[EVENT1]]]
// CHECK: %[[EVENT3:.*]], %[[VALUE1:.*]] = air.execute [%[[EVENT1]]]
// CHECK: %[[EVENT4:.*]], %[[VALUE2:.*]] = air.execute [%[[EVENT1]]]
// CHECK-NOT: %[[EVENT5:.*]] = air.dma_memcpy_nd async [%[[EVENT2]], %[[EVENT1]]]
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd async [%[[EVENT2]]]
// CHECK-NOT: %[[EVENT6:.*]] = air.dma_memcpy_nd async [%[[EVENT3]], %[[EVENT1]]]
// CHECK: %[[EVENT6:.*]] = air.dma_memcpy_nd async [%[[EVENT3]]]
// CHECK-NOT: %[[EVENT7:.*]] = air.dma_memcpy_nd async [%[[EVENT4]], %[[EVENT1]]]
// CHECK: %[[EVENT7:.*]] = air.dma_memcpy_nd async [%[[EVENT4]]]

#map0 = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "MMult_Mult"} {
  func.func @forward(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>, %arg3: memref<128x128xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %async_token, %results = air.execute -> (memref<128x128xf32>) {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
      air.execute_terminator %2 : memref<128x128xf32>
    } {id = 1 : i32}
    %async_token_0, %results_1 = air.execute -> (memref<128x128xf32>) {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
      air.execute_terminator %2 : memref<128x128xf32>
    } {id = 2 : i32}
    %async_token_2, %results_3 = air.execute -> (memref<128x128xf32>) {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
      air.execute_terminator %2 : memref<128x128xf32>
    } {id = 3 : i32}
    %async_token_4 = air.execute [%async_token_0] {
      linalg.fill ins(%cst : f32) outs(%results_1 : memref<128x128xf32>)
    } {id = 4 : i32}
    %async_token_5 = air.execute [%async_token_2, %async_token_4] {
      memref.copy %results_1, %results_3 : memref<128x128xf32> to memref<128x128xf32>
    } {id = 5 : i32}
    %0 = air.herd @herd_0 async [%async_token_5]  tile (%arg4, %arg5) in (%arg6=%c8, %arg7=%c2) args(%arg8=%arg1, %arg9=%arg2, %arg10=%results_3) : memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32> attributes {id = 1 : i32} {
      %c64 = arith.constant 64 : index
      %c1_7 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %async_token_8, %results_9 = air.execute -> (index) {
        %4 = affine.apply #map0()[%arg4]
        air.execute_terminator %4 : index
      } {id = 6 : i32}
      %async_token_10, %results_11 = air.execute -> (index) {
        %4 = affine.apply #map1()[%arg5]
        air.execute_terminator %4 : index
      } {id = 7 : i32}
      %2 = air.wait_all async [%async_token_8, %async_token_10] 
      %3 = scf.for %arg11 = %c0 to %c128 step %c32 iter_args(%arg12 = %2) -> (!air.async.token) {
        %c16_12 = arith.constant 16 : index
        %c32_13 = arith.constant 32 : index
        %c128_14 = arith.constant 128 : index
        %c1_15 = arith.constant 1 : index
        %c64_16 = arith.constant 64 : index
        %async_token_17, %results_18 = air.execute [%arg12] -> (memref<16x32xf32, 2>) {
          %9 = memref.alloc() : memref<16x32xf32, 2>
          air.execute_terminator %9 : memref<16x32xf32, 2>
        } {id = 8 : i32}
        %async_token_19, %results_20 = air.execute [%arg12] -> (memref<32x64xf32, 2>) {
          %9 = memref.alloc() : memref<32x64xf32, 2>
          air.execute_terminator %9 : memref<32x64xf32, 2>
        } {id = 9 : i32}
        %async_token_21, %results_22 = air.execute [%arg12] -> (memref<16x64xf32, 2>) {
          %9 = memref.alloc() : memref<16x64xf32, 2>
          air.execute_terminator %9 : memref<16x64xf32, 2>
        } {id = 10 : i32}
        %4 = air.dma_memcpy_nd async [%async_token_17, %arg12] (%results_18[] [] [], %arg8[%results_9, %arg11] [%c16_12, %c32_13] [%c128_14, %c1_15]) {id = 1 : i32} : (memref<16x32xf32, 2>, memref<128x128xf32>)
        %5 = air.dma_memcpy_nd async [%async_token_19, %arg12] (%results_20[] [] [], %arg9[%arg11, %results_11] [%c32_13, %c64_16] [%c128_14, %c1_15]) {id = 2 : i32} : (memref<32x64xf32, 2>, memref<128x128xf32>)
        %6 = air.dma_memcpy_nd async [%async_token_21, %arg12] (%results_22[] [] [], %arg10[%results_9, %results_11] [%c16_12, %c64_16] [%c128_14, %c1_15]) {id = 3 : i32} : (memref<16x64xf32, 2>, memref<128x128xf32>)
        %async_token_23 = air.execute [%5, %6, %4] {
          linalg.matmul ins(%results_18, %results_20 : memref<16x32xf32, 2>, memref<32x64xf32, 2>) outs(%results_22 : memref<16x64xf32, 2>)
        } {id = 11 : i32}
        %7 = air.dma_memcpy_nd async [%async_token_23] (%arg10[%results_9, %results_11] [%c16_12, %c64_16] [%c128_14, %c1_15], %results_22[] [] []) {id = 4 : i32} : (memref<128x128xf32>, memref<16x64xf32, 2>)
        %async_token_24 = air.execute [%async_token_23] {
          memref.dealloc %results_18 : memref<16x32xf32, 2>
        } {id = 12 : i32}
        %async_token_25 = air.execute [%async_token_23] {
          memref.dealloc %results_20 : memref<32x64xf32, 2>
        } {id = 13 : i32}
        %async_token_26 = air.execute [%7] {
          memref.dealloc %results_22 : memref<16x64xf32, 2>
        } {id = 14 : i32}
        %8 = air.wait_all async [%async_token_24, %async_token_25, %async_token_26] 
        scf.yield %8 : !air.async.token
      }
      air.herd_terminator
    }
    return
  }
}