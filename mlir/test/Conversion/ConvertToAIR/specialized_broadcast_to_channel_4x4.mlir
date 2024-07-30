//===- specialized_broadcast_to_channel_4x4.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -cse | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 16)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
// CHECK: air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel @channel_1 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel @channel_2 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel @channel_3 [1, 1] {broadcast_shape = [1, 4]}
  func.func @matmul(%arg0: memref<512x512xbf16>) {
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c8, %arg4=%c8) args(%arg5=%arg0) : memref<512x512xbf16> attributes {id = 3 : i32} {
      %1 = air.segment async  args(%arg6=%arg1, %arg7=%arg2, %arg8=%arg3, %arg9=%arg4, %arg10=%arg5) : index, index, index, index, memref<512x512xbf16> attributes {id = 2 : i32} {
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
        %2 = air.wait_all async  {id = 4 : i32}
        %3 = scf.for %arg11 = %c0 to %c512 step %c64 iter_args(%arg12 = %2) -> (!air.async.token) {
          %async_token, %results = air.execute -> (memref<64x64xbf16, 1>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %alloc : memref<64x64xbf16, 1>
          } {id = 1 : i32}
// CHECK: %[[EVENT0:.*]] = air.channel.put async{{.*}}@channel_0[]
// CHECK: %[[EVENT1:.*]] = air.channel.put async{{.*}}@channel_1[]
// CHECK: %[[EVENT2:.*]] = air.channel.put async{{.*}}@channel_2[]
// CHECK: %[[EVENT3:.*]] = air.channel.put async{{.*}}@channel_3[]
// CHECK: %[[EVENT4:.*]] = air.herd @herd_0 async{{.*}}tile (%[[VALUE0:.*]], %[[VALUE1:.*]]) in
          %4 = air.herd @herd_0 async [%async_token]  tile (%arg13, %arg14) in (%arg15=%c4, %arg16=%c4) args(%arg17=%results) : memref<64x64xbf16, 1> attributes {id = 1 : i32} {
            %c1 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c16 = arith.constant 16 : index
            %async_token_3, %results_4 = air.execute -> (index) {
              %8 = affine.apply #map()[%arg13]
              air.execute_terminator %8 : index
            } {id = 2 : i32}
            %6 = air.wait_all async [%async_token_3]  {id = 2 : i32}
            %7 = scf.for %arg18 = %c0_1 to %c64_2 step %c16 iter_args(%arg19 = %6) -> (!air.async.token) {
              %async_token_5, %results_6 = air.execute -> (memref<16x16xbf16, 2>) {
                %alloc = memref.alloc() : memref<16x16xbf16, 2>
                air.execute_terminator %alloc : memref<16x16xbf16, 2>
              } {id = 3 : i32}
              %async_token_7, %results_8 = air.execute -> (memref<16x16xbf16, 2>) {
                %alloc = memref.alloc() : memref<16x16xbf16, 2>
                air.execute_terminator %alloc : memref<16x16xbf16, 2>
              } {id = 4 : i32}
              %async_token_9, %results_10 = air.execute -> (memref<16x16xbf16, 2>) {
                %alloc = memref.alloc() : memref<16x16xbf16, 2>
                air.execute_terminator %alloc : memref<16x16xbf16, 2>
              } {id = 5 : i32}
              %8 = affine.if #set()[%arg13, %arg14] -> !air.async.token {
                %c0_15 = arith.constant 0 : index
// CHECK: %[[EVENT5:.*]] = air.channel.get async{{.*}}@channel_0[%[[VALUE0]], %[[VALUE1]]]
                %10 = air.dma_memcpy_nd async [%async_token_5, %arg19] (%results_6[] [] [], %arg17[%c0_15, %arg18] [%c16, %c16] [%c64_2, %c1]) {broadcast_set = #set, id = 1 : i32} : (memref<16x16xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %10 : !air.async.token
              } else {
                %10 = affine.if #set1()[%arg13, %arg14] -> !air.async.token {
                  %c16_15 = arith.constant 16 : index
// CHECK: %[[EVENT6:.*]] = air.channel.get async{{.*}}@channel_1[%[[VALUE0]], %[[VALUE1]]]
                  %11 = air.dma_memcpy_nd async [%async_token_5, %arg19] (%results_6[] [] [], %arg17[%c16_15, %arg18] [%c16, %c16] [%c64_2, %c1]) {broadcast_set = #set1, id = 2 : i32} : (memref<16x16xbf16, 2>, memref<64x64xbf16, 1>)
                  affine.yield %11 : !air.async.token
                } else {
                  %11 = affine.if #set2()[%arg13, %arg14] -> !air.async.token {
                    %c32 = arith.constant 32 : index
// CHECK: %[[EVENT7:.*]] = air.channel.get async{{.*}}@channel_2[%[[VALUE0]], %[[VALUE1]]]
                    %12 = air.dma_memcpy_nd async [%async_token_5, %arg19] (%results_6[] [] [], %arg17[%c32, %arg18] [%c16, %c16] [%c64_2, %c1]) {broadcast_set = #set2, id = 3 : i32} : (memref<16x16xbf16, 2>, memref<64x64xbf16, 1>)
                    affine.yield %12 : !air.async.token
                  } else {
                    %c48 = arith.constant 48 : index
// CHECK: %[[EVENT8:.*]] = air.channel.get async{{.*}}@channel_3[%[[VALUE0]], %[[VALUE1]]]
                    %12 = air.dma_memcpy_nd async [%async_token_5, %arg19] (%results_6[] [] [], %arg17[%c48, %arg18] [%c16, %c16] [%c64_2, %c1]) {broadcast_set = #set3, id = 4 : i32} : (memref<16x16xbf16, 2>, memref<64x64xbf16, 1>)
                    affine.yield %12 : !air.async.token
                  }
                  affine.yield %11 : !air.async.token
                }
                affine.yield %10 : !air.async.token
              }
              %async_token_11 = air.execute [%async_token_9, %async_token_7, %8] {
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_6, %results_8 : memref<16x16xbf16, 2>, memref<16x16xbf16, 2>) outs(%results_10 : memref<16x16xbf16, 2>)
              } {id = 6 : i32}
              %async_token_12 = air.execute [%async_token_11] {
                memref.dealloc %results_6 : memref<16x16xbf16, 2>
              } {id = 7 : i32}
              %async_token_13 = air.execute [%async_token_11] {
                memref.dealloc %results_8 : memref<16x16xbf16, 2>
              } {id = 8 : i32}
              %async_token_14 = air.execute [%async_token_11] {
                memref.dealloc %results_10 : memref<16x16xbf16, 2>
              } {id = 9 : i32}
              %9 = air.wait_all async [%async_token_11]  {id = 1 : i32}
              scf.yield %9 : !air.async.token
            }
          }
          %async_token_0 = air.execute [%4] {
            memref.dealloc %results : memref<64x64xbf16, 1>
          } {id = 10 : i32}
          %5 = air.wait_all async [%4]  {id = 3 : i32}
          scf.yield %5 : !air.async.token
        }
      }
    }
    return
  }
}

