//===- dma_to_channel_nested_par_in_herd.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 256)>
#set = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
module {
// CHECK: air.channel @channel_0 [2, 1] {broadcast_shape = [2, 2]}
// CHECK: air.channel @channel_1 [1, 2] {broadcast_shape = [2, 2]}
// CHECK: air.channel @channel_2 [2, 2]
// CHECK-LABEL: func.func @mmult
  func.func @mmult(%arg0: memref<512x512xi32>, %arg1: memref<512x512xi32>, %arg2: memref<512x512xi32>) {
    %c2 = arith.constant 2 : index
    %async_token, %results = air.execute -> (memref<512x512xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xi32>
      air.execute_terminator %alloc : memref<512x512xi32>
    }
// CHECK: %[[EVENT0:.*]] = scf.parallel{{.*}}init
// CHECK: %[[EVENT1:.*]] = scf.parallel{{.*}}init
// CHECK: %[[EVENT2:.*]] = scf.for{{.*}}iter_args
// CHECK: %[[EVENT3:.*]] = air.channel.put async{{.*}}@channel_0
// CHECK: scf.yield
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: %[[EVENT4:.*]] = scf.parallel{{.*}}init
// CHECK: %[[EVENT5:.*]] = scf.parallel{{.*}}init
// CHECK: %[[EVENT6:.*]] = scf.for{{.*}}iter_args
// CHECK: %[[EVENT7:.*]] = air.channel.put async{{.*}}@channel_1
// CHECK: scf.yield
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: %[[EVENT8:.*]] = scf.parallel{{.*}}init
// CHECK: %[[EVENT9:.*]] = scf.parallel{{.*}}init
// CHECK: %[[EVENT10:.*]] = air.channel.get async{{.*}}@channel_2
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: %[[EVENT11:.*]] = air.herd async
    %0 = air.herd async [%async_token]  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%results) : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %async_token_1, %results_2 = air.execute -> (index) {
        %3 = affine.apply #map()[%arg3]
        air.execute_terminator %3 : index
      }
      %async_token_3, %results_4 = air.execute -> (index) {
        %3 = affine.apply #map()[%arg4]
        air.execute_terminator %3 : index
      }
      %1 = air.wait_all async [%async_token_3, %async_token_1] 
// CHECK: %[[EVENT12:.*]] = scf.parallel{{.*}}init
      %2 = scf.parallel (%arg10, %arg11) = (%c0, %c0) to (%c256, %c256) step (%c32, %c32) init (%1) -> !air.async.token {
        %async_token_5, %results_6 = air.execute [%1] -> (index) {
          %6 = arith.addi %results_2, %arg10 : index
          air.execute_terminator %6 : index
        }
        %async_token_7, %results_8 = air.execute [%1] -> (index) {
          %6 = arith.addi %results_4, %arg11 : index
          air.execute_terminator %6 : index
        }
        %async_token_9, %results_10 = air.execute [%1] -> (memref<32x32xi32, 2>) {
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.execute_terminator %alloc : memref<32x32xi32, 2>
        }
        %async_token_11 = air.execute [%async_token_9] {
          linalg.fill ins(%c0_i32 : i32) outs(%results_10 : memref<32x32xi32, 2>)
        }
        %3 = air.wait_all async [%async_token_11, %async_token_7, %async_token_5] 
// CHECK: %[[EVENT13:.*]] = scf.for{{.*}}iter_args
        %4 = scf.for %arg12 = %c0 to %c512 step %c32 iter_args(%arg13 = %3) -> (!air.async.token) {
          %async_token_12, %results_13 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %async_token_14, %results_15 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
// CHECK: %[[EVENT14:.*]] = air.channel.get async{{.*}}@channel_0
          %6 = air.dma_memcpy_nd async [%async_token_12, %arg13] (%results_13[] [] [], %arg7[%results_6, %arg12] [%c32, %c32] [%c512, %c1]) {broadcast_pattern = #set, id = 1 : i32} : (memref<32x32xi32, 2>, memref<512x512xi32>)
// CHECK: %[[EVENT15:.*]] = air.channel.get async{{.*}}@channel_1
          %7 = air.dma_memcpy_nd async [%async_token_14, %arg13] (%results_15[] [] [], %arg8[%arg12, %results_8] [%c32, %c32] [%c512, %c1]) {broadcast_pattern = #set1, id = 2 : i32} : (memref<32x32xi32, 2>, memref<512x512xi32>)
          %async_token_16 = air.execute [%7, %6] {
            linalg.matmul ins(%results_13, %results_15 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_10 : memref<32x32xi32, 2>)
          }
// CHECK: scf.yield
          scf.yield %async_token_16 : !air.async.token
        }
// CHECK: %[[EVENT16:.*]] = air.channel.put async{{.*}}@channel_2
        %5 = air.dma_memcpy_nd async [%4] (%arg9[%results_6, %results_8] [%c32, %c32] [%c512, %c1], %results_10[] [] []) {id = 3 : i32} : (memref<512x512xi32>, memref<32x32xi32, 2>)
// CHECK: scf.reduce
        scf.reduce(%5 : !air.async.token) {
        ^bb0(%arg12: !air.async.token, %arg13: !air.async.token):
          %6 = air.wait_all async [%arg12, %arg13] 
// CHECK: scf.reduce.return
          scf.reduce.return %6 : !air.async.token
        }
      }
    }
    %async_token_0 = air.execute [%0] {
      memref.copy %results, %arg2 : memref<512x512xi32> to memref<512x512xi32>
    }
    return
  }
}
