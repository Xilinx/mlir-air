//===- specialized_broadcast_to_channel.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -cse | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
// CHECK: air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
// CHECK: air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  func.func @mmult(%arg0: memref<512x512xbf16>) {
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c8, %arg4=%c8) args(%arg5=%arg0) : memref<512x512xbf16> attributes {id = 3 : i32} {
      %1 = air.segment async  args(%arg6=%arg1, %arg7=%arg2, %arg8=%arg3, %arg9=%arg4, %arg10=%arg5) : index, index, index, index, memref<512x512xbf16> attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
        %async_token, %results = air.execute -> (index) {
          %4 = affine.apply #map()[%arg6]
          air.execute_terminator %4 : index
        } {id = 1 : i32}
        %2 = air.wait_all async [%async_token]  {id = 4 : i32}
        %3 = scf.for %arg11 = %c0 to %c512 step %c64 iter_args(%arg12 = %2) -> (!air.async.token) {
          %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 1>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %alloc : memref<64x64xbf16, 1>
          } {id = 2 : i32}
// CHECK: %[[EVENT0:.*]] = air.channel.put async{{.*}}@channel_0[]
// CHECK: %[[EVENT1:.*]] = air.channel.put async{{.*}}@channel_1[]
// CHECK: %[[EVENT2:.*]] = air.herd @herd_0 async{{.*}}tile (%[[VALUE0:.*]], %[[VALUE1:.*]]) in
          %4 = air.herd @herd_0 async  tile (%arg13, %arg14) in (%arg15=%c2, %arg16=%c2) args(%arg17=%results_1) : memref<64x64xbf16, 1> attributes {id = 1 : i32} {
            %c1 = arith.constant 1 : index
            %c0_3 = arith.constant 0 : index
            %c64_4 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %async_token_5, %results_6 = air.execute -> (index) {
              %8 = affine.apply #map1()[%arg13]
              air.execute_terminator %8 : index
            } {id = 3 : i32}
            %6 = air.wait_all async [%async_token_5]  {id = 2 : i32}
            %7 = scf.for %arg18 = %c0_3 to %c64_4 step %c32 iter_args(%arg19 = %6) -> (!air.async.token) {
              %async_token_7, %results_8 = air.execute -> (memref<32x32xbf16, 2>) {
                %alloc = memref.alloc() : memref<32x32xbf16, 2>
                air.execute_terminator %alloc : memref<32x32xbf16, 2>
              } {id = 4 : i32}
              %8 = affine.if #set()[%arg13, %arg14] -> !air.async.token {
// CHECK: %[[EVENT3:.*]] = air.channel.get async{{.*}}@channel_0[%[[VALUE0]], %[[VALUE1]]]
                %10 = air.dma_memcpy_nd async [%async_token_7, %arg19] (%results_8[] [] [], %arg17[%c0_3, %arg18] [%c32, %c32] [%c64_4, %c1]) {broadcast_set = #set, id = 1 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %10 : !air.async.token
              } else {
// CHECK: %[[EVENT4:.*]] = air.channel.get async{{.*}}@channel_1[%[[VALUE0]], %[[VALUE1]]]
                %10 = air.dma_memcpy_nd async [%async_token_7, %arg19] (%results_8[] [] [], %arg17[%c32, %arg18] [%c32, %c32] [%c64_4, %c1]) {broadcast_set = #set1, id = 2 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %10 : !air.async.token
              }
              %async_token_9 = air.execute [%8] {
                memref.dealloc %results_8 : memref<32x32xbf16, 2>
              } {id = 5 : i32}
              %9 = air.wait_all async [%8]  {id = 1 : i32}
              scf.yield %9 : !air.async.token
            }
          }
          %async_token_2 = air.execute [%4] {
            memref.dealloc %results_1 : memref<64x64xbf16, 1>
          } {id = 6 : i32}
          %5 = air.wait_all async [%4]  {id = 3 : i32}
          scf.yield %5 : !air.async.token
        }
      }
    }
    return
  }
}