//===- dma_to_channel_nested_for_in_partition.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// Hoisting external channel put/get op out of a partition with nested for loops

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
// CHECK-LABEL: func.func @mmult
  func.func @mmult(%arg0: memref<512x512xbf16>) {
    %c8 = arith.constant 8 : index
    %c1_0 = arith.constant 1 : index
// CHECK: %[[EVENT0:.*]] = air.launch async
// CHECK: %[[EVENT1:.*]] = scf.for
// CHECK: %[[EVENT2:.*]] = scf.for
// CHECK: %[[EVENT3:.*]] = air.channel.put{{.*}}@channel_0[]
// CHECK: %[[EVENT4:.*]] = air.partition async
// CHECK: %[[EVENT5:.*]] = scf.for
// CHECK: %[[EVENT6:.*]] = scf.for
// CHECK: %[[EVENT7:.*]] = air.channel.get{{.*}}@channel_0[]
// CHECK: %[[EVENT8:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: %[[EVENT9:.*]] = scf.for
// CHECK: %[[EVENT10:.*]] = air.channel.put{{.*}}@channel_1[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT11:.*]] = air.herd @herd_0 async
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1_0, %arg4=%c8) args(%arg5=%arg0) : memref<512x512xbf16> attributes {id = 3 : i32} {
      %1 = air.partition async  args(%arg6=%arg1, %arg7=%arg2, %arg8=%arg3, %arg9=%arg4, %arg10=%arg5) : index, index, index, index, memref<512x512xbf16> attributes {id = 2 : i32} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
        %2 = air.wait_all async  {id = 4 : i32}
        %newarg1 = scf.for %newarg0 = %c0 to %c512 step %c64 iter_args(%newarg2 = %2) -> (!air.async.token) {
            %newarg3 = air.wait_all async [%newarg2]
            %3 = scf.for %arg11 = %c0 to %c512 step %c64 iter_args(%arg12 = %newarg3) -> (!air.async.token) {
                %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 1>) {
                    %alloc = memref.alloc() : memref<64x64xbf16, 1>
                    air.execute_terminator %alloc : memref<64x64xbf16, 1>
                } {id = 2 : i32}
                %4 = air.dma_memcpy_nd async [%async_token_0, %arg12] (%results_1[] [] [], %arg10[%newarg0, %arg11] [%c64, %c64] [%c512, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
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
                    %9 = air.dma_memcpy_nd async [%async_token_8, %arg19] (%results_9[] [] [], %arg17[%results_7, %arg18] [%c32, %c32] [%c64_5, %c1_3]) {id = 2 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                    %async_token_10 = air.execute [%9] {
                        memref.dealloc %results_9 : memref<32x32xbf16, 2>
                    } {id = 5 : i32}
                    %10 = air.wait_all async [%9]  {id = 1 : i32}
                    scf.yield %10 : !air.async.token
                    }
                    air.herd_terminator
                }
                %async_token_2 = air.execute [%5] {
                    memref.dealloc %results_1 : memref<64x64xbf16, 1>
                } {id = 6 : i32}
                %6 = air.wait_all async [%5]  {id = 3 : i32}
                scf.yield %6 : !air.async.token
            }
            %newarg4 = air.wait_all async [%3]  {id = 3 : i32}
            scf.yield %newarg4 : !air.async.token
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return
  }
}