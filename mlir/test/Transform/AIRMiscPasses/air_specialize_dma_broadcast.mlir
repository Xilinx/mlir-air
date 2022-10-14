//===- air_specialize_dma_broadcast.mlir -----------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-specialize-dma-broadcast | FileCheck %s

// Lowers broadcastable DMAs using affine.if
// CHECK: [[$SET0:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET2:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
// CHECK: [[$SET3:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
// CHECK: %[[EVENT0:.*]] = affine.if [[$SET0]]
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET0]]{{.*}}id = [[#ID0:]]
// CHECK: affine.yield %[[EVENT1]]
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET1]]{{.*}}id = [[#ID0+1]]
// CHECK: affine.yield %[[EVENT2]]
// CHECK: %[[EVENT3:.*]] = affine.if [[$SET2]]
// CHECK: %[[EVENT4:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET2]]{{.*}}id = [[#ID1:]]
// CHECK: affine.yield %[[EVENT4]]
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET3]]{{.*}}id = [[#ID1+1]]
// CHECK: affine.yield %[[EVENT5]]

#map = affine_map<()[s0] -> (s0 * 32)>
#set0 = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
module {
  func.func @matmul(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %asyncToken, %valOut = air.execute -> (memref<512x512xbf16>){
      %1 = memref.alloc() {alignment = 128 : i64} : memref<512x512xbf16>
      air.execute_terminator %1 : memref<512x512xbf16>
    } {id = 1 : i32} 
    %asyncToken_0 = air.execute[%asyncToken] {
      memref.copy %arg2, %valOut : memref<512x512xbf16> to memref<512x512xbf16>
      air.execute_terminator
    } {id = 2 : i32}
    %0 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c64, %c64) init (%asyncToken_0) -> !air.async.token {
      %1 = scf.for %arg5 = %c0 to %c512 step %c64 iter_args(%arg6 = %asyncToken_0) -> (!air.async.token) {
        %asyncToken_1, %valOut_2 = air.execute[%arg6] -> (memref<64x64xbf16, 1>) {
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %8 : memref<64x64xbf16, 1>
        } {id = 3 : i32}
        %asyncToken_3, %valOut_4 = air.execute[%arg6] -> (memref<64x64xbf16, 1>) {
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %8 : memref<64x64xbf16, 1>
        } {id = 4 : i32}
        %asyncToken_5, %valOut_6 = air.execute[%arg6] -> (memref<64x64xbf16, 1>) {
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %8 : memref<64x64xbf16, 1>
        } {id = 5 : i32}
        %2 = air.dma_memcpy_nd async [%asyncToken_1] (%valOut_2[] [] [], %arg0[%arg3, %arg5] [%c64, %c64] [%c512, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
        %3 = air.dma_memcpy_nd async [%asyncToken_3] (%valOut_4[] [] [], %arg1[%arg5, %arg4] [%c64, %c64] [%c512, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
        %4 = air.dma_memcpy_nd async [%asyncToken_5, %arg6] (%valOut_6[] [] [], %valOut[%arg3, %arg4] [%c64, %c64] [%c512, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
        %5 = air.herd async [%3, %2, %4]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%valOut_2, %arg12=%valOut_4, %arg13=%valOut_6) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 1 : i32, sym_name = "herd_0"} {
          %c1_10 = arith.constant 1 : index
          %c64_11 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %c0_12 = arith.constant 0 : index
          %asyncToken_13, %valOut_14 = air.execute -> (index) {
            %12 = affine.apply #map()[%arg7]
            air.execute_terminator %12 : index
          } {id = 6 : i32}
          %asyncToken_15, %valOut_16 = air.execute -> (index) {
            %12 = affine.apply #map()[%arg8]
            air.execute_terminator %12 : index
          } {id = 7 : i32}
          %8 = air.wait_all async [%asyncToken_13, %asyncToken_15] 
          %asyncToken_17, %valOut_18 = air.execute -> (memref<32x32xbf16, 2>) {
            %12 = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %12 : memref<32x32xbf16, 2>
          } {id = 10 : i32}
          %9 = air.dma_memcpy_nd async [%8, %asyncToken_17] (%valOut_18[] [] [], %arg13[%valOut_14, %valOut_16] [%c32, %c32] [%c64_11, %c1_10]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          %10 = scf.for %arg14 = %c0_12 to %c64_11 step %c32 iter_args(%arg15 = %9) -> (!air.async.token) {
            %asyncToken_20, %valOut_21 = air.execute[%arg15] -> (memref<32x32xbf16, 2>) {
              %15 = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %15 : memref<32x32xbf16, 2>
            } {id = 8 : i32}
            %asyncToken_22, %valOut_23 = air.execute[%arg15] -> (memref<32x32xbf16, 2>) {
              %15 = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %15 : memref<32x32xbf16, 2>
            } {id = 9 : i32}
            %12 = air.dma_memcpy_nd async [%asyncToken_20, %arg15] (%valOut_21[] [] [], %arg11[%valOut_14, %arg14] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_pattern = #set0, id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            %13 = air.dma_memcpy_nd async [%asyncToken_22, %arg15] (%valOut_23[] [] [], %arg12[%arg14, %valOut_16] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_pattern = #set1, id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            %asyncToken_24 = air.execute[%13, %arg15, %12] {
              linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%valOut_21, %valOut_23 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%valOut_18 : memref<32x32xbf16, 2>)
              air.execute_terminator
            } {id = 11 : i32}
            %asyncToken_25 = air.execute[%asyncToken_24] {
              memref.dealloc %valOut_21 : memref<32x32xbf16, 2>
              air.execute_terminator
            } {id = 12 : i32}
            %asyncToken_26 = air.execute[%asyncToken_24] {
              memref.dealloc %valOut_23 : memref<32x32xbf16, 2>
              air.execute_terminator
            } {id = 13 : i32}
            %14 = air.wait_all async [%asyncToken_24, %asyncToken_25, %asyncToken_26] 
            scf.yield %14 : !air.async.token
          }
          %11 = air.dma_memcpy_nd async [%10] (%arg13[%valOut_14, %valOut_16] [%c32, %c32] [%c64_11, %c1_10], %valOut_18[] [] []) {broadcast = "both", id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          %asyncToken_19 = air.execute[%11] {
            memref.dealloc %valOut_18 : memref<32x32xbf16, 2>
            air.execute_terminator
          } {id = 14 : i32}
          air.herd_terminator
        }
        %6 = air.dma_memcpy_nd async [%5] (%valOut[%arg3, %arg4] [%c64, %c64] [%c512, %c1], %valOut_6[] [] []) {id = 8 : i32} : (memref<512x512xbf16>, memref<64x64xbf16, 1>)
        %asyncToken_7 = air.execute[%5] {
          memref.dealloc %valOut_2 : memref<64x64xbf16, 1>
          air.execute_terminator
        } {id = 15 : i32}
        %asyncToken_8 = air.execute[%5] {
          memref.dealloc %valOut_4 : memref<64x64xbf16, 1>
          air.execute_terminator
        } {id = 16 : i32}
        %asyncToken_9 = air.execute[%6] {
          memref.dealloc %valOut_6 : memref<64x64xbf16, 1>
          air.execute_terminator
        } {id = 17 : i32}
        %7 = air.wait_all async [%asyncToken_7, %asyncToken_8, %asyncToken_9] 
        scf.yield %7 : !air.async.token
      }
      scf.reduce(%1)  : !air.async.token {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %2 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %2 : !air.async.token
      }
      scf.yield
    }
    return
  }
}