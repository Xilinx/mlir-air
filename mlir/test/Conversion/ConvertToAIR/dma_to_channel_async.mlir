//===- dma_to_channel_async.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
// CHECK: air.channel @channel_0 [1, 1]
// CHECK: air.channel @channel_1 [2, 2]
// CHECK-LABEL: func.func @mmult
  func.func @mmult(%arg0: memref<512x512xbf16>) {
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c8, %arg4=%c8) args(%arg5=%arg0) : memref<512x512xbf16> attributes {id = 3 : i32} {
// CHECK: %[[EVENT0:.*]] = air.launch async
// CHECK: %[[EVENT1:.*]], %[[VALUE0:.*]] = air.execute
// CHECK: %[[EVENT2:.*]] = air.wait_all async [%[[EVENT1]]]
// CHECK: %[[EVENT3:.*]] = scf.for{{.*}}iter_args(%[[EVENT4:.*]] = %[[EVENT2]])
// CHECK: %[[EVENT5:.*]] = air.channel.put async{{.*}}%[[EVENT4]]{{.*}}@channel_0
// CHECK: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT5]]]
// CHECK: scf.yield %[[EVENT6]]
      %1 = air.segment async  args(%arg6=%arg1, %arg7=%arg2, %arg8=%arg3, %arg9=%arg4, %arg10=%arg5) : index, index, index, index, memref<512x512xbf16> attributes {id = 2 : i32} {
// CHECK: %[[EVENT7:.*]] = air.segment async
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
// CHECK: %[[EVENT8:.*]], %[[VALUE1:.*]] = air.execute
        %async_token, %results = air.execute -> (index) {
          %4 = affine.apply #map()[%arg6]
          air.execute_terminator %4 : index
        } {id = 1 : i32}
// CHECK: %[[EVENT9:.*]] = air.wait_all async [%[[EVENT8]]]
        %2 = air.wait_all async [%async_token]  {id = 4 : i32}
// CHECK: %[[EVENT10:.*]] = scf.for{{.*}}iter_args(%[[EVENT11:.*]] = %[[EVENT9]])
        %3 = scf.for %arg11 = %c0 to %c512 step %c64 iter_args(%arg12 = %2) -> (!air.async.token) {
// CHECK: %[[EVENT12:.*]], %[[VALUE2:.*]] = air.execute
          %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 1>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %alloc : memref<64x64xbf16, 1>
          } {id = 2 : i32}
          %4 = air.dma_memcpy_nd async [%async_token_0, %arg12] (%results_1[] [] [], %arg10[%results, %arg11] [%c64, %c64] [%c512, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
// CHECK: %[[EVENT13:.*]] = air.channel.get async{{.*}}%[[EVENT12]], %[[EVENT11]]{{.*}}@channel_0
// CHECK: %[[EVENT14:.*]] = air.wait_all async{{.*}}%[[EVENT13]]

// CHECK: %[[EVENT15:.*]] = scf.parallel{{.*}}init (%[[EVENT14]])
// CHECK: %[[EVENT16:.*]], %[[VALUE3:.*]] = air.execute
// CHECK: %[[EVENT17:.*]] = air.wait_all async{{.*}}%[[EVENT16]]
// CHECK: %[[EVENT18:.*]] = scf.for{{.*}}iter_args(%[[EVENT19:.*]] = %[[EVENT17]])
// CHECK: %[[EVENT20:.*]] = air.channel.put async{{.*}}%[[EVENT19]]{{.*}}@channel_1
// CHECK: %[[EVENT21:.*]] = air.wait_all async [%[[EVENT20]]]
// CHECK: scf.yield %[[EVENT21]]
// CHECK: %[[EVENT22:.*]] = air.wait_all async [%[[EVENT18]]]
// CHECK: scf.reduce(%[[EVENT22]]
// CHECK: %[[EVENT23:.*]] = air.wait_all async [%[[EVENT24:.*]], %[[EVENT25:.*]]]

// CHECK: %[[EVENT26:.*]] = air.herd @herd_0 async
          %5 = air.herd @herd_0 async [%4]  tile (%arg13, %arg14) in (%arg15=%c2, %arg16=%c2) args(%arg17=%results_1) : memref<64x64xbf16, 1> attributes {id = 1 : i32} {
            %c1_3 = arith.constant 1 : index
            %c0_4 = arith.constant 0 : index
            %c64_5 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %async_token_6, %results_7 = air.execute -> (index) {
              %9 = affine.apply #map1()[%arg13]
              air.execute_terminator %9 : index
            } {id = 3 : i32}
            %7 = air.wait_all async [%async_token_6]  {id = 2 : i32}
            %8 = scf.for %arg18 = %c0_4 to %c64_5 step %c32 iter_args(%arg19 = %7) -> (!air.async.token) {
              %async_token_8, %results_9 = air.execute -> (memref<32x32xbf16, 2>) {
                %alloc = memref.alloc() : memref<32x32xbf16, 2>
                air.execute_terminator %alloc : memref<32x32xbf16, 2>
              } {id = 4 : i32}
// CHECK: %[[EVENT27:.*]] = air.channel.get async{{.*}}@channel_1
              %9 = air.dma_memcpy_nd async [%async_token_8, %arg19] (%results_9[] [] [], %arg17[%results_7, %arg18] [%c32, %c32] [%c64_5, %c1_3]) {id = 2 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              %async_token_10 = air.execute [%9] {
                memref.dealloc %results_9 : memref<32x32xbf16, 2>
              } {id = 5 : i32}
              %10 = air.wait_all async [%9]  {id = 1 : i32}
              scf.yield %10 : !air.async.token
            }
          }
          %async_token_2 = air.execute [%5] {
            memref.dealloc %results_1 : memref<64x64xbf16, 1>
          } {id = 6 : i32}
          %6 = air.wait_all async [%5]  {id = 3 : i32}
          scf.yield %6 : !air.async.token
        }
      }
    }
    return
  }
}