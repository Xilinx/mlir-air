//===- dma_to_channel_async_canonicalize.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 32)>
module {
// CHECK: air.channel @channel_0 [2, 2]
// CHECK: air.channel @channel_1 [2, 2]
  func.func @matmul_on_buffers(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %c2 = arith.constant 2 : index
// CHECK: %[[EVENT0:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: %[[EVENT1:.*]] = air.channel.put async{{.*}}@channel_0[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT2:.*]] = scf.parallel (%[[VALUE2:.*]], %[[VALUE3:.*]]) ={{.*}}init
// CHECK: %[[EVENT3:.*]] = air.channel.get async{{.*}}@channel_1[%[[VALUE2]], %[[VALUE3]]]
// CHECK: %[[EVENT4:.*]] = air.herd @herd_0 async{{.*}}tile (%[[VALUE4:.*]], %[[VALUE5:.*]]) in
    %0 = air.herd @herd_0 async  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> attributes {id = 1 : i32, x_loc = 7 : i64, y_loc = 2 : i64} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %async_token, %results = air.execute -> (index) {
        %4 = affine.apply #map()[%arg3]
        air.execute_terminator %4 : index
      }
      %async_token_0, %results_1 = air.execute -> (index) {
        %4 = affine.apply #map()[%arg4]
        air.execute_terminator %4 : index
      }
      %1 = air.wait_all async [%async_token_0, %async_token] 
      %async_token_2, %results_3 = air.execute -> (memref<32x32xf32, 2>) {
        %alloc = memref.alloc() : memref<32x32xf32, 2>
        air.execute_terminator %alloc : memref<32x32xf32, 2>
      }
// CHECK: %[[EVENT5:.*]] = air.channel.get async{{.*}}@channel_0[%[[VALUE4]], %[[VALUE5]]]
// CHECK: %[[EVENT6:.*]] = air.channel.put async{{.*}}@channel_1[%[[VALUE4]], %[[VALUE5]]]
      %2 = air.dma_memcpy_nd async [%async_token_2, %1] (%results_3[] [] [], %arg9[%results, %results_1] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32>)
      %3 = air.dma_memcpy_nd async [%2] (%arg9[%results, %results_1] [%c32, %c32] [%c64, %c1], %results_3[] [] []) {id = 6 : i32} : (memref<64x64xf32>, memref<32x32xf32, 2>)
      %async_token_4 = air.execute [%3] {
        memref.dealloc %results_3 : memref<32x32xf32, 2>
      }
    }
    return
  }
}
