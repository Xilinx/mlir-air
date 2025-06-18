//===- dma_to_channel.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -split-input-file | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
// CHECK: air.channel @channel_0 [2, 2]
// CHECK: air.channel @channel_1 [2, 2]
// CHECK: air.channel @channel_2 [2, 2]
// CHECK: air.channel @channel_3 [2, 2]
// CHECK-LABEL: func.func @mmult
  func.func @mmult(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64x64xi32>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %alloc, %alloc_0 : memref<64x64xi32> to memref<64x64xi32>
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_0

// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_1

// CHECK: %[[EVENT12:.*]] = air.wait_all async
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_2

// CHECK: %[[EVENT18:.*]] = air.wait_all async
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@channel_3
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %0 = affine.apply #map()[%arg2]
      %1 = affine.apply #map()[%arg3]
      scf.for %arg9 = %c0 to %c64 step %c32 {
        %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
        %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
        %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
// CHECK: air.channel.get{{.*}}@channel_0
        air.dma_memcpy_nd (%alloc_1[] [] [], %arg6[%0, %arg9] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.get{{.*}}@channel_1
        air.dma_memcpy_nd (%alloc_2[] [] [], %arg7[%arg9, %1] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.get{{.*}}@channel_2
        air.dma_memcpy_nd (%alloc_3[] [] [], %arg8[%0, %1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.put{{.*}}@channel_3
        linalg.matmul ins(%alloc_1, %alloc_2 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_3 : memref<32x32xi32, 2>)
        air.dma_memcpy_nd (%arg8[%0, %1] [%c32, %c32] [%c64, %c1], %alloc_3[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
        memref.dealloc %alloc_1 : memref<32x32xi32, 2>
        memref.dealloc %alloc_2 : memref<32x32xi32, 2>
        memref.dealloc %alloc_3 : memref<32x32xi32, 2>
      }
    }
    return %alloc_0 : memref<64x64xi32>
  }
}

// -----

#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
module {
// CHECK: air.channel @channel_0 [1, 1]
// CHECK: air.launch
// CHECK: scf.parallel (%[[ARG0:.*]], %[[ARG1:.*]]) = ({{.*}}) to ({{.*}}) step ({{.*}}) init (%{{.*}})
// CHECK: air.channel.get async [%{{.*}}]  @channel_0[%[[ARG0]], %[[ARG1]]]
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0 async  tile (%[[ARG2:.*]], %[[ARG3:.*]]) in ({{.*}}) args({{.*}})
// CHECK: air.channel.put async [%{{.*}}]  @channel_0[%[[ARG2]], %[[ARG3]]]
  func.func @l1tol3(%arg0: memref<16x32xf32>, %arg1: memref<16x32xf32>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg1, %arg7=%arg0) : memref<16x32xf32>, memref<16x32xf32> attributes {id = 3 : i32} {
      %1 = air.segment @segment_0 async  args(%arg8=%arg2, %arg9=%arg3, %arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7) : index, index, index, index, memref<16x32xf32>, memref<16x32xf32> attributes {id = 2 : i32} {
        %c1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (index) {
          %3 = affine.apply #map()[%arg8]
          air.execute_terminator %3 : index
        } {id = 1 : i32}
        %async_token_0, %results_1 = air.execute -> (index) {
          %3 = affine.apply #map1()[%arg9]
          air.execute_terminator %3 : index
        } {id = 2 : i32}
        %subview = memref.subview %arg12[%results, %results_1] [8, 16] [1, 1] : memref<16x32xf32> to memref<8x16xf32, strided<[32, 1], offset: ?>>
        %async_token_2, %results_3 = air.execute -> (index) {
          %3 = affine.apply #map()[%arg8]
          air.execute_terminator %3 : index
        } {id = 3 : i32}
        %async_token_4, %results_5 = air.execute -> (index) {
          %3 = affine.apply #map1()[%arg9]
          air.execute_terminator %3 : index
        } {id = 4 : i32}
        %async_token_6, %results_7 = air.execute -> (memref<8x16xf32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x16xf32, 1 : i32>
          air.execute_terminator %alloc : memref<8x16xf32, 1 : i32>
        } {id = 5 : i32}
        %2 = air.herd @herd_0 async  tile (%arg14, %arg15) in (%arg16=%c1, %arg17=%c1) args(%arg18=%results_7, %arg19=%subview) : memref<8x16xf32, 1 : i32>, memref<8x16xf32, strided<[32, 1], offset: ?>> attributes {id = 1 : i32} {
          %async_token_9, %results_10 = air.execute -> (memref<8x16xf32, 2 : i32>) {
            %alloc = memref.alloc() : memref<8x16xf32, 2 : i32>
            air.execute_terminator %alloc : memref<8x16xf32, 2 : i32>
          } {id = 6 : i32}
          %3 = air.dma_memcpy_nd async [%async_token_9] (%arg19[] [] [], %results_10[] [] []) {id = 1 : i32} : (memref<8x16xf32, strided<[32, 1], offset: ?>>, memref<8x16xf32, 2 : i32>)
          %async_token_11 = air.execute [%3] {
            memref.dealloc %results_10 : memref<8x16xf32, 2 : i32>
          } {id = 7 : i32}
        }
        %async_token_8 = air.execute [%2, %async_token_6] {
          memref.dealloc %results_7 : memref<8x16xf32, 1 : i32>
        } {id = 8 : i32}
      }
    }
    return
  }
}
