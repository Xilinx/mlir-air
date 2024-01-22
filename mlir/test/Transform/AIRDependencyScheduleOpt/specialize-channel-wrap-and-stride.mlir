//===- specialize-channel-wrap-and-stride.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride | FileCheck %s

// Specialize air.channel ops in perfectly nested for loops in air.segment with wraps and strides.

#map = affine_map<()[s0] -> (s0 * 32)>
module {

  // CHECK-LABEL: test0
  // CHECK: air.segment
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[CST64:.*]] = arith.constant 64 : index
  // CHECK-DAG: %[[CST512:.*]] = arith.constant 512 : index
  // CHECK: %[[EVENT0:.*]] = air.channel.put async [{{.*}}]  @channel_0[] (%[[VAL0:.*]][%[[CST0]], %[[CST0]]] [%[[CST32]], %[[CST4]]] [%[[CST16]], %[[CST1]]]) : (memref<8x16xi32, 1>)
  // CHECK: scf.for{{.*}}iter_args(%[[EVENT1:.*]] = %[[EVENT0]])
  // CHECK: air.herd
  // CHECK: air.channel.get async{{.*}}@channel_0
  // CHECK: air.herd_terminator
  // CHECK: air.segment_terminator

  func.func @test0(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
        %c0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c4 = arith.constant 4 : index
        %async_token_25, %results_26 = air.execute -> (memref<8x16xi32, 1>) {
          %alloc = memref.alloc() : memref<8x16xi32, 1>
          air.execute_terminator %alloc : memref<8x16xi32, 1>
        }
        %async_token_0 = air.wait_all async
        %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
          %async_token_3 = air.channel.put async [%arg11]  @channel_0[] (%results_26[%c0, %arg10] [%c4, %c4] [%c16, %c1_1]) : (memref<8x16xi32, 1>)
          scf.yield %async_token_3 : !air.async.token
        }
        %4 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %3) -> (!air.async.token) {
          %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
            %async_token_27, %results_28 = air.execute -> (memref<4x4xi32, 2>) {
              %alloc = memref.alloc() : memref<4x4xi32, 2>
              air.execute_terminator %alloc : memref<4x4xi32, 2>
            }
            %5 = air.channel.get async [%async_token_27]  @channel_0[%arg21, %arg22] (%results_28[] [] []) : (memref<4x4xi32, 2>)
            air.herd_terminator
          }
          scf.yield %2 : !air.async.token
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }

  // CHECK-LABEL: test1
  // CHECK: put @channel_1[%c0, %c0] (%arg0[%c0, %c0] [%c4, %c32] [%c32, %c1]) : (memref<128xf32>)
  // CHECK: get @channel_2[%c0, %c0] (%arg1[%c0, %c0] [%c128, %c32] [%c128, %c1]) : (memref<128x128xf32>)
  // CHECK: put @channel_3[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c32, %c32] [%c32, %c128, %c1]) : (memref<128x128xf32>)
  // CHECK: put @channel_4[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c128, %c32] [%c32, %c128, %c1]) : (memref<128x128xf32>)
  // CHECK: get @channel_5[%c0, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c4, %c4, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (memref<128x128xf32>)

  func.func @test1(%arg0: memref<128xf32>, %arg1: memref<128x128xf32>) -> memref<128xf32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<128xf32>
    scf.for %arg2 = %c0 to %c128 step %c32 {
      air.channel.put  @channel_1[%c0, %c0] (%arg0[%arg2] [%c32] [%c1]) : (memref<128xf32>)
    }
    scf.for %arg2 = %c0 to %c128 step %c32 {
      air.channel.get  @channel_2[%c0, %c0] (%arg1[%arg2, %c0] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
    }
    scf.for %arg2 = %c0 to %c128 step %c32 {
      air.channel.put  @channel_3[%c0, %c0] (%arg1[%c0, %arg2] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
    }
    scf.for %arg2 = %c0 to %c128 step %c32 {
      scf.for %arg3 = %c0 to %c128 step %c32 {
        air.channel.put  @channel_4[%c0, %c0] (%arg1[%arg3, %arg2] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
      }
    }
    scf.for %arg2 = %c0 to %c128 step %c32 {
      scf.for %arg3 = %c0 to %c128 step %c32 {
        air.channel.get  @channel_5[%c0, %c0] (%arg1[%arg2, %arg3] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
      }
    }
    return %alloc : memref<128xf32>
  }

  // CHECK-LABEL: test2
  // CHECK: put @channel_6[%c0, %c0] (%arg0[%c0, %c0] [%c4, %c32] [%c32, %c1]) : (memref<128xf32>)
  // CHECK: get @channel_7[%c0, %c0] (%arg1[%c0, %c0] [%c128, %c32] [%c128, %c1]) : (memref<128x128xf32>)
  // CHECK: put @channel_8[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c32, %c32] [%c32, %c128, %c1]) : (memref<128x128xf32>)
  // CHECK: put @channel_9[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c128, %c32] [%c32, %c128, %c1]) : (memref<128x128xf32>)
  // CHECK: get @channel_10[%c0, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c4, %c4, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (memref<128x128xf32>)

  func.func @test2(%arg0: memref<128xf32>, %arg1: memref<128x128xf32>) -> memref<128xf32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<128xf32>
    affine.for %arg2 = 0 to 128 step 32 {
      air.channel.put  @channel_6[%c0, %c0] (%arg0[%arg2] [%c32] [%c1]) : (memref<128xf32>)
    }
    affine.for %arg2 = 0 to 128 step 32 {
      air.channel.get  @channel_7[%c0, %c0] (%arg1[%arg2, %c0] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
    }
    affine.for %arg2 = 0 to 128 step 32 {
      air.channel.put  @channel_8[%c0, %c0] (%arg1[%c0, %arg2] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
    }
    affine.for %arg2 = 0 to 128 step 32 {
      affine.for %arg3 = 0 to 128 step 32 {
        air.channel.put  @channel_9[%c0, %c0] (%arg1[%arg3, %arg2] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
      }
    }
    affine.for %arg2 = 0 to 128 step 32 {
      affine.for %arg3 = 0 to 128 step 32 {
        air.channel.get  @channel_10[%c0, %c0] (%arg1[%arg2, %arg3] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
      }
    }
    return %alloc : memref<128xf32>
  }

  // CHECK-LABEL: test3
  // CHECK: put @channel_11[%c0, %c0] (%arg0[%c0, %c0] [%c4, %c32] [%c32, %c1]) : (memref<128xf32>)
  // CHECK: put @channel_12[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c128, %c32] [%c32, %c128, %c1]) : (memref<128x128xf32>)
  // CHECK: put @channel_13[%c0, %c0] (%arg0[%c0, %c0] [%c4, %c32] [%c32, %c1]) : (memref<128xf32>)
  // CHECK: put async [%0]  @channel_14[%c0, %c0] (%arg0[%c0, %c0] [%c4, %c32] [%c32, %c1]) : (memref<128xf32>)

  func.func @test3(%arg0: memref<128xf32>, %arg1: memref<128x128xf32>) -> memref<128xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<128xf32>
    affine.for %arg2 = 0 to 4 step 1 {
      %0 = affine.apply #map()[%arg2]
      air.channel.put  @channel_11[%c0, %c0] (%arg0[%0] [%c32] [%c1]) : (memref<128xf32>)
    }
    affine.for %arg2 = 0 to 4 step 1 {
      affine.for %arg3 = 0 to 4 step 1 {
        %0 = affine.apply #map()[%arg2]
        %1 = affine.apply #map()[%arg3]
        air.channel.put  @channel_12[%c0, %c0] (%arg1[%1, %0] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
      }
    }
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %0 = affine.apply #map()[%arg2]
      air.channel.put  @channel_13[%c0, %c0] (%arg0[%0] [%c32] [%c1]) : (memref<128xf32>)
    }
    %async_token = air.wait_all async
    %async_token_0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg4 = %async_token) -> (!air.async.token) {
      %async_token_1, %results_0 = air.execute [%arg4] -> (index) {
        %apply = affine.apply #map()[%arg2]
        air.execute_terminator %apply : index
      }
      %async_token_2 = air.channel.put async [%async_token_1]  @channel_14[%c0, %c0] (%arg0[%results_0] [%c32] [%c1]) : (memref<128xf32>)
      scf.yield %async_token_2 : !air.async.token
    }
    return %alloc : memref<128xf32>
  }

  // CHECK-LABEL: test4
  // CHECK: put async  @channel_15[%c0, %c0] (%arg0[%c0] [%c32] [%c1]) : (memref<128xf32>)
  // CHECK: put async  @channel_16[%c0, %c0] (%arg1[%c0, %c0] [%c16, %c4] [%c4, %c1]) : (memref<128x128xf32>)

  func.func @test4(%arg0: memref<128xf32>, %arg1: memref<128x128xf32>) -> memref<128xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %alloc = memref.alloc() : memref<128xf32>
    %0 = air.channel.put async @channel_15[%c0, %c0] (%arg0[%c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c32] [%c128, %c128, %c128, %c1]) : (memref<128xf32>)
    %1 = air.channel.put async @channel_16[%c0, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c1, %c2, %c8, %c4] [%c64, %c32, %c4, %c1]) : (memref<128x128xf32>)
    %2 = air.wait_all async [%0, %1]
    return %alloc : memref<128xf32>
  }
}
