//===- specialize-channel-wrap-and-stride.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride="scope=func" | FileCheck %s

// Specialize air.channel ops in perfectly nested for loops in air.segment with wraps and strides.

#map = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<()[s0] -> (s0 * 64)>
#map4 = affine_map<()[s0] -> (s0 + 3)>
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
  // CHECK: %[[EVENT0:.*]] = air.channel.put async [{{.*}}]  @channel_0[] (%[[VAL0:.*]][%[[CST0]], %[[CST0]]] [%[[CST32]], %[[CST4]]] [%[[CST16]], %[[CST1]]]){{.*}}: (memref<8x16xi32, 1>)
  // CHECK: scf.for{{.*}}iter_args(%[[EVENT1:.*]] = %[[EVENT0]])
  // CHECK: air.herd
  // CHECK: air.channel.get async{{.*}}@channel_0

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
          }
          scf.yield %2 : !air.async.token
        }
      }
    }
    return
  }

  // CHECK-LABEL: test1
  // CHECK: put @channel_1[%c0, %c0] (%arg0[] [] []){{.*}}: (memref<128xf32>)
  // CHECK: get @channel_2[%c0, %c0] (%arg1[%c0, %c0] [%c128, %c32] [%c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: put @channel_3[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c32, %c32] [%c32, %c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: put @channel_4[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c128, %c32] [%c32, %c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: get @channel_5[%c0, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c4, %c4, %c32, %c32] [%c4096, %c32, %c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: put @channel_5[] (%alloc_0[%c0, %c18, %c0] [%c4, %c18, %c8] [%c8, %c32, %c1]){{.*}}: (memref<1x6x6x32xi8, 1>)

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
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c8 = arith.constant 8 : index
    %c192 = arith.constant 192 : index
    %c1152 = arith.constant 1152 : index
    %alloc_0 = memref.alloc() : memref<1x6x6x32xi8, 1>
    scf.for %arg2 = %c0 to %c32 step %c8 {
      air.channel.put @channel_5[] (%alloc_0[%c0, %c3, %c0, %arg2] [%c1, %c3, %c6, %c8] [%c1152, %c192, %c32, %c1]) : (memref<1x6x6x32xi8, 1>)
    }
    return %alloc : memref<128xf32>
  }

  // CHECK-LABEL: test2
  // CHECK: put @channel_6[%c0, %c0] (%arg0[] [] []){{.*}}: (memref<128xf32>)
  // CHECK: get @channel_7[%c0, %c0] (%arg1[%c0, %c0] [%c128, %c32] [%c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: put @channel_8[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c32, %c32] [%c32, %c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: put @channel_9[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c128, %c32] [%c32, %c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: get @channel_10[%c0, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c4, %c4, %c32, %c32] [%c4096, %c32, %c128, %c1]){{.*}}: (memref<128x128xf32>)

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
  // CHECK: put @channel_11[%c0, %c0] (%arg0[] [] []){{.*}}: (memref<128xf32>)
  // CHECK: put @channel_12[%c0, %c0] (%arg1[%c0, %c0, %c0] [%c4, %c128, %c32] [%c32, %c128, %c1]){{.*}}: (memref<128x128xf32>)
  // CHECK: put @channel_13[%c0, %c0] (%arg0[] [] []){{.*}}: (memref<128xf32>)
  // CHECK: put async [%0]  @channel_14[%c0, %c0] (%arg0[] [] []){{.*}}: (memref<128xf32>)

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
  // CHECK: put async  @channel_15[%c0, %c0] (%arg0[%c0] [%c32] [%c1]){{.*}}: (memref<128xf32>)
  // CHECK: put async  @channel_16[%c0, %c0] (%arg1[%c0] [%c64] [%c1]){{.*}}: (memref<128x128xf32>)

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

  // CHECK-LABEL: test5
  // CHECK: put async  @channel_17[] (%arg0[%c0, %c0] [%c8, %c1024] [%c0, %c1]){{.*}}: (memref<32x32xf32>)

  func.func @test5(%arg0: memref<32x32xf32>) -> memref<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %alloc = memref.alloc() : memref<32x32xf32>
    scf.for %arg2 = %c0 to %c8 step %c1 {
      %0 = affine.apply #map()[%arg2]
      %1 = air.channel.put async @channel_17[] (%arg0[] [] []) : (memref<32x32xf32>)
    }
    return %alloc : memref<32x32xf32>
  }

  // CHECK-LABEL: test6
  // CHECK: put async  @channel_18[] (%arg0[] [] []){{.*}}: (memref<1x1x4x2x8x4xi32>)

  func.func @test6(%arg0: memref<1x1x4x2x8x4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %1 = air.channel.put async @channel_18[] (%arg0[%c0, %c0, %c0, %c0] [%c4, %c2, %c8, %c4] [%c64, %c32, %c4, %c1]) : (memref<1x1x4x2x8x4xi32>)
    return
  }

  // CHECK-LABEL: test7
  // CHECK: put async  @channel_19[%c0, %c0] (%arg0[%c0] [%c16384] [%c1]){{.*}}: (memref<256x256xbf16>)
  // CHECK: put async  @channel_19[%c1, %c0] (%arg0[%c16384] [%c16384] [%c1]){{.*}}: (memref<256x256xbf16>)
  // CHECK: put async  @channel_19[%c2, %c0] (%arg0[%c32768] [%c16384] [%c1]){{.*}}: (memref<256x256xbf16>)
  // CHECK: put async  @channel_19[%c3, %c0] (%arg0[%c49152] [%c16384] [%c1]){{.*}}: (memref<256x256xbf16>)

  func.func @test7(%arg0: memref<256x256xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c192 = arith.constant 192 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %1 = air.channel.put async @channel_19[%c0, %c0] (%arg0[%c0, %c0] [%c64, %c256] [%c256, %c1]) : (memref<256x256xbf16>)
    %2 = air.channel.put async @channel_19[%c1, %c0] (%arg0[%c64, %c0] [%c64, %c256] [%c256, %c1]) : (memref<256x256xbf16>)
    %3 = air.channel.put async @channel_19[%c2, %c0] (%arg0[%c128, %c0] [%c64, %c256] [%c256, %c1]) : (memref<256x256xbf16>)
    %4 = air.channel.put async @channel_19[%c3, %c0] (%arg0[%c192, %c0] [%c64, %c256] [%c256, %c1]) : (memref<256x256xbf16>)
    return
  }

  // AIE2 hw limitation: stride <= 1M.
  // CHECK-LABEL: test8
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c0, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c256, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c0, %c0, %c0] [%c4, %c64, %c256] [%c1245184, %c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c0, %c0] [%c512, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)

  func.func @test8(%arg0: memref<2432x9728xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c9728 = arith.constant 9728 : index
    scf.for %arg2 = %c0 to %c512 step %c256 {
      air.channel.put @channel_20[%c0, %c0] (%arg0[%arg2, %c0] [%c64, %c256] [%c9728, %c1]) : (memref<2432x9728xbf16>)
    }
    scf.for %arg2 = %c0 to %c512 step %c128 {
      air.channel.put @channel_20[%c0, %c0] (%arg0[%arg2, %c0] [%c64, %c256] [%c9728, %c1]) : (memref<2432x9728xbf16>)
    }
    scf.for %arg2 = %c0 to %c512 step %c64 {
      air.channel.put @channel_20[%c0, %c0] (%arg0[%arg2, %c0] [%c64, %c256] [%c9728, %c1]) : (memref<2432x9728xbf16>)
    }
    return
  }

  // Offset propagation with wrap-and-stride canonicalization.
  // CHECK-LABEL: test9
  // CHECK: %[[VAL0:.*]] = affine.apply #map()[%arg1]
  // CHECK: put  @channel_21[] (%arg0[%c0, %c0, %[[VAL0]]] [%c8, %c64, %c32] [%c32, %c256, %c1]){{.*}}: (memref<128x256xi32>)
  // CHECK: air.channel.put  @channel_22[] (%arg2[%c256, %c0, %c0] [%c8, %c32, %c4] [%c4, %c32, %c1]){{.*}}: (memref<1x2x32x32xi32, 1 : i32>)
  // CHECK: air.channel.put  @channel_23[] (%arg3[%c128, %c0, %c0] [%c4, %c32, %c8] [%c8, %c32, %c1]){{.*}}: (memref<2x1x32x32xi32, 1 : i32>)
  // CHECK: %[[VAL1:.*]] = affine.apply
  // CHECK: put async  @channel_21[] (%arg0[%c0, %[[VAL1]]] [%c8, %c8] [%c0, %c1]){{.*}}: (memref<128x256xi32>)

  func.func @test9(%arg0: memref<128x256xi32>, %arg1: index, %arg2: memref<1x2x32x32xi32, 1 : i32>, %arg3: memref<2x1x32x32xi32, 1 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %c8192 = arith.constant 8192 : index
    %results = affine.apply #map()[%arg1]
    air.channel.put  @channel_21[] (%arg0[%c0, %c0, %results, %c0, %c0] [%c8, %c2, %c1, %c32, %c32] [%c32, %c8192, %c32, %c256, %c1]) : (memref<128x256xi32>)
    air.channel.put  @channel_22[] (%arg2[%c0, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (memref<1x2x32x32xi32, 1 : i32>)
    air.channel.put  @channel_23[] (%arg3[%c1, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (memref<2x1x32x32xi32, 1 : i32>)
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %0 = affine.apply #map()[%arg1]
      %1 = air.channel.put async @channel_21[] (%arg0[%0] [%c8] [%c1]) : (memref<128x256xi32>)
    }
    return
  }

  // Scf.parallel loop specialization; specialization of affine.apply on induction vars.
  // CHECK-LABEL: test10
  
  // CHECK: air.channel.put async [%{{.*}}]  @channel_24[%c0, %c0] (%{{.*}}[%c0, %c0, %c0, %c0, %c0] [%c4, %c3, %c3, %c8, %c4] [%c288, %c6, %c1, %c36, %c1]){{.*}}: (memref<1x32x6x6xi32, 1>)
  // CHECK: air.channel.put async [%{{.*}}]  @channel_24[%c1, %c0] (%{{.*}}[%c0, %c0, %c0, %c0, %c6] [%c4, %c3, %c3, %c8, %c4] [%c288, %c6, %c1, %c36, %c1]){{.*}}: (memref<1x32x6x6xi32, 1>)
  // CHECK: air.channel.put async [%{{.*}}]  @channel_24[%c2{{.*}}, %c0] (%{{.*}}[%c0, %c0, %c0, %c0, %c12] [%c4, %c3, %c3, %c8, %c4] [%c288, %c6, %c1, %c36, %c1]){{.*}}: (memref<1x32x6x6xi32, 1>)
  // CHECK: air.channel.put async [%{{.*}}]  @channel_24[%c3, %c0] (%{{.*}}[%c0, %c0, %c0, %c0, %c18] [%c4, %c3, %c3, %c8, %c4] [%c288, %c6, %c1, %c36, %c1]){{.*}}: (memref<1x32x6x6xi32, 1>)
  // CHECK: air.channel.get async [%{{.*}}]  @channel_25[%c0, %c0] (%{{.*}}[%c0, %c0] [%c4, %c4] [%c16, %c1]){{.*}}: (memref<1x4x4x4xi32, 1>)
  // CHECK: air.channel.get async [%{{.*}}]  @channel_25[%c1, %c0] (%{{.*}}[%c0, %c4] [%c4, %c4] [%c16, %c1]){{.*}}: (memref<1x4x4x4xi32, 1>)
  // CHECK: air.channel.get async [%{{.*}}]  @channel_25[%c2{{.*}}, %c0] (%{{.*}}[%c0, %c8] [%c4, %c4] [%c16, %c1]){{.*}}: (memref<1x4x4x4xi32, 1>)
  // CHECK: air.channel.get async [%{{.*}}]  @channel_25[%c3, %c0] (%{{.*}}[%c0, %c12] [%c4, %c4] [%c16, %c1]){{.*}}: (memref<1x4x4x4xi32, 1>)

  func.func @test10(%arg0: memref<2x32x6x6xi32>, %arg1: memref<4x32x3x3xi32>, %arg2: memref<2x4x4x4xi32>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg3) in (%arg4=%c2) attributes {id = 1 : i32} {
      %1 = air.segment @conv_static_0 async  attributes {id = 2 : i32} {
        %c8 = arith.constant 8 : index
        %c3 = arith.constant 3 : index
        %c16 = arith.constant 16 : index
        %c36 = arith.constant 36 : index
        %c6 = arith.constant 6 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<1x32x6x6xi32, 1>) {
          %alloc = memref.alloc() : memref<1x32x6x6xi32, 1>
          air.execute_terminator %alloc : memref<1x32x6x6xi32, 1>
        }
        %async_token_0, %results_1 = air.execute -> (memref<1x4x4x4xi32, 1>) {
          %alloc = memref.alloc() : memref<1x4x4x4xi32, 1>
          air.execute_terminator %alloc : memref<1x4x4x4xi32, 1>
        }
        %2 = scf.parallel (%arg5) = (%c0) to (%c4) step (%c1) init (%async_token) -> !air.async.token {
          %5 = scf.for %arg6 = %c0 to %c32 step %c8 iter_args(%arg7 = %async_token) -> (!air.async.token) {
            %6 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %arg7) -> (!air.async.token) {
              %7 = scf.for %arg10 = %c0 to %c3 step %c1 iter_args(%arg11 = %arg9) -> (!air.async.token) {
                %async_token_2, %results_3 = air.execute [%arg11] -> (index) {
                  %9 = affine.apply #map1()[%arg5, %arg8]
                  air.execute_terminator %9 : index
                }
                %8 = air.channel.put async [%async_token_2]  @channel_24[%arg5, %c0] (%results[%arg6, %results_3, %arg10] [%c8, %c1, %c4] [%c36, %c6, %c1]) : (memref<1x32x6x6xi32, 1>)
                scf.yield %8 : !air.async.token
              }
              scf.yield %7 : !air.async.token
            }
            scf.yield %6 : !air.async.token
          }
          scf.reduce(%5 : !air.async.token) {
          ^bb0(%arg6: !air.async.token, %arg7: !air.async.token):
            %6 = air.wait_all async [%arg6, %arg7] 
            scf.reduce.return %6 : !air.async.token
          }
        }
        %3 = scf.parallel (%arg5) = (%c0) to (%c4) step (%c1) init (%async_token_0) -> !air.async.token {
          %5 = air.channel.get async [%async_token_0]  @channel_25[%arg5, %c0] (%results_1[%c0, %arg5, %c0] [%c4, %c1, %c4] [%c16, %c4, %c1]) : (memref<1x4x4x4xi32, 1>)
          scf.reduce(%5 : !air.async.token) {
          ^bb0(%arg6: !air.async.token, %arg7: !air.async.token):
            %6 = air.wait_all async [%arg6, %arg7] 
            scf.reduce.return %6 : !air.async.token
          }
        }
        %4 = air.wait_all async [%2, %3] 
      }
    }
    return
  }

  // Affine.apply with map joining two for loops in a loop nest.
  // CHECK-LABEL: test11
  
  // CHECK: air.channel.put async {{.*}}@channel_26[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c18{{.*}}, %c4{{.*}}] [%c96{{.*}}, %c16{{.*}}, %c1{{.*}}]){{.*}}: (memref<1x6x6x16xbf16, 1>)
  // CHECK: air.channel.put async {{.*}}@channel_26[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c18{{.*}}, %c4{{.*}}] [%c96{{.*}}, %c16{{.*}}, %c1{{.*}}]){{.*}}: (memref<1x6x6x16xbf16, 1>)
  // CHECK: air.channel.put async {{.*}}@channel_26[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c12{{.*}}] [%c3{{.*}}, %c3{{.*}}, %c4{{.*}}, %c4{{.*}}] [%c96{{.*}}, %c16{{.*}}, %c16{{.*}}, %c1{{.*}}]){{.*}}: (memref<1x3x6x16xi32, 1>)

  func.func @test11() {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%arg3, %arg4, %arg5) in (%arg6=%c3, %arg7=%c3, %arg8=%c4) {
      %1 = air.segment @segment_0 async {
        %c576 = arith.constant 576 : index
        %c288 = arith.constant 288 : index
        %c96 = arith.constant 96 : index
        %c3_0 = arith.constant 3 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c12 = arith.constant 12 : index
        %c6 = arith.constant 6 : index
        %c0 = arith.constant 0 : index
        %c4_1 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<1x6x6x16xbf16, 1>) {
          %alloc = memref.alloc() : memref<1x6x6x16xbf16, 1>
          air.execute_terminator %alloc : memref<1x6x6x16xbf16, 1>
        }
        %async_token_23, %results_25 = air.execute -> (memref<1x3x6x16xi32, 1>) {
          %alloc = memref.alloc() : memref<1x3x6x16xi32, 1>
          air.execute_terminator %alloc : memref<1x3x6x16xi32, 1>
        }
        %4 = scf.for %arg9 = %c0 to %c4_1 step %c1 iter_args(%arg13 = %async_token) -> (!air.async.token) {
          %2 = scf.for %arg10 = %c0 to %c3_0 step %c1 iter_args(%arg11 = %arg13) -> (!air.async.token) {
            %async_token_2, %results_3 = air.execute [%arg11] -> (index) {
              %4 = affine.apply #map2(%arg9, %arg10)
              air.execute_terminator %4 : index
            }
            %3 = air.channel.put async [%async_token_2]  @channel_26[%c0, %c0] (%results[%c0, %results_3, %c0, %c0] [%c1, %c1, %c6, %c4_1] [%c576, %c96, %c16, %c1]) : (memref<1x6x6x16xbf16, 1>)
            scf.yield %3 : !air.async.token
          }
          scf.yield %2 : !air.async.token
        }
        %5 = scf.for %arg9 = %c0 to %c4_1 step %c1 iter_args(%arg13 = %async_token) -> (!air.async.token) {
          %2 = scf.for %arg10 = %c0 to %c3_0 step %c1 iter_args(%arg11 = %arg13) -> (!air.async.token) {
            %async_token_2, %results_3 = air.execute [%arg11] -> (index) {
              %6 = affine.apply #map1()[%arg9, %arg10]
              air.execute_terminator %6 : index
            }
            %3 = air.channel.put async [%async_token_2]  @channel_26[%c0, %c0] (%results[%c0, %results_3, %c0, %c0] [%c1, %c1, %c6, %c4_1] [%c576, %c96, %c16, %c1]) : (memref<1x6x6x16xbf16, 1>)
            scf.yield %3 : !air.async.token
          }
          scf.yield %2 : !air.async.token
        }
        scf.for %arg9 = %c0 to %c3_0 step %c1 {
          %60 = scf.for %arg10 = %c0 to %c3_0 step %c1 iter_args(%arg13 = %async_token) -> (!air.async.token) {
            %async_token_54, %results_55 = air.execute [%arg13] -> (index) {
              air.execute_terminator %arg9 : index
            }
            %61 = air.channel.put async [%async_token_54]  @channel_26[%c0, %c0] (%results_25[%c0, %results_55, %arg10, %c12] [%c1, %c1, %c4_1, %c4_1] [%c288, %c96, %c16, %c1]) : (memref<1x3x6x16xi32, 1>)
            scf.yield %61 : !air.async.token
          }
        }
      }
    }
    return
  }

  // Arith.muli and addi folding into loops.
  // CHECK-LABEL: test12
  
  // CHECK: air.channel.put async [%{{.*}}]  @channel_27[%c0, %c0] (%{{.*}}[%c0, %c0, %c8] [%c2, %c2, %c128] [%c8, %c16, %c1]){{.*}}: (memref<32x16xi32>)

  func.func @test12(%arg0: memref<32x16xi32>) {
    %0 = air.launch async () in () args(%arg2=%arg0) : memref<32x16xi32> {
      %1 = air.segment @seg async  args(%arg3=%arg2) : memref<32x16xi32> {
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %2 = air.wait_all async 
        %3 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %2) -> (!air.async.token) {
          %4 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %arg5) -> (!air.async.token) {
            %async_token, %results = air.execute [%arg7] -> (index) {
              %6 = arith.addi %arg4, %c1 : index
              air.execute_terminator %6 : index
            }
            %async_token_0, %results_1 = air.execute [%arg7] -> (index) {
              %6 = arith.muli %arg6, %c16 : index
              air.execute_terminator %6 : index
            }
            %5 = air.channel.put async [%async_token_0, %async_token, %arg7]  @channel_27[%c0, %c0] (%arg3[%results, %results_1] [%c16, %c8] [%c8, %c1]) : (memref<32x16xi32>)
            scf.yield %5 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
      }
    }
    return
  }

  // Offset propagated from scf.for and air.hier induction vars.
  // CHECK-LABEL: test13
  
  // CHECK: air.channel.put async [%{{.*}}]  @channel_14[] (%{{.*}}[%c0, %1, %results, %c0] [%c8, %c2_0, %c32, %c32] [%c32, %c8192, %c256, %c1]){{.*}}: (memref<2x128x256xi32>)
  // CHECK: air.channel.put async [%{{.*}}]  @channel_15[%c0, %c0] (%{{.*}}[%c0, %results, %c32768] [%c8, %c32, %c32] [%c32, %c256, %c1]){{.*}}: (memref<512x512xi32>)

  func.func @test13(%arg0: memref<2x128x256xi32>, %arg1: memref<512x512xi32>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg3, %arg4, %arg5) in (%arg6=%c2, %arg7=%c2, %arg8=%c2) args(%arg10=%arg0, %arg11=%arg1) : memref<2x128x256xi32>, memref<512x512xi32> {
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c16384 = arith.constant 16384 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c8192 = arith.constant 8192 : index
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c2_0 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (index) {
        %7 = affine.apply #map3()[%arg4]
        air.execute_terminator %7 : index
      }
      %2 = scf.for %arg12 = %c0 to %c256 step %c32 iter_args(%arg13 = %async_token) -> (!air.async.token) {
        %7 = air.channel.put async [%arg13, %async_token]  @channel_14[] (%arg10[%arg3, %c0, %c0, %results, %arg12] [%c1, %c2_0, %c1, %c32, %c32] [%c32768, %c8192, %c32, %c256, %c1]) : (memref<2x128x256xi32>)
        scf.yield %7 : !air.async.token
      }
      %3 = scf.for %arg12 = %c0 to %c256 step %c32 iter_args(%arg13 = %async_token) -> (!air.async.token) {
        %7 = air.channel.put async [%arg13, %async_token]  @channel_15[%c0, %c0] (%arg11[%c2_0, %c0, %results, %arg12] [%c1, %c1, %c32, %c32] [%c16384, %c32, %c256, %c1]) {id = 1 : i32} : (memref<512x512xi32>)
        scf.yield %7 : !air.async.token
      }
    }
    return
  }

  // Unroll any data movement loops that cannot directly transform into wraps and strides.
  // CHECK-LABEL: test14
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c0, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg1[%c0, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c256, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg1[%c256, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c0, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg1[%c0, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg0[%c256, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)
  // CHECK: put  @channel_20[%c0, %c0] (%arg1[%c256, %c0] [%c64, %c256] [%c9728, %c1]){{.*}}: (memref<2432x9728xbf16>)

  func.func @test14(%arg0: memref<2432x9728xbf16>, %arg1: memref<2432x9728xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c9728 = arith.constant 9728 : index
    scf.for %arg2 = %c0 to %c512 step %c256 {
      air.channel.put @channel_20[%c0, %c0] (%arg0[%arg2, %c0] [%c64, %c256] [%c9728, %c1]) : (memref<2432x9728xbf16>)
      air.channel.put @channel_20[%c0, %c0] (%arg1[%arg2, %c0] [%c64, %c256] [%c9728, %c1]) : (memref<2432x9728xbf16>)
    }
    affine.for %arg2 = 0 to 512 step 256 {
      air.channel.put @channel_20[%c0, %c0] (%arg0[%arg2, %c0] [%c64, %c256] [%c9728, %c1]) : (memref<2432x9728xbf16>)
      air.channel.put @channel_20[%c0, %c0] (%arg1[%arg2, %c0] [%c64, %c256] [%c9728, %c1]) : (memref<2432x9728xbf16>)
    }
    return
  }

  // Specializing affine.apply on induction vars.
  // CHECK-LABEL: test15
  // CHECK: get async {{.*}} @channel_26[%c3{{.*}}, %c3{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c6144{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c1024{{.*}}] [%c1024{{.*}}, %c8192{{.*}}, %c1{{.*}}])

  func.func @test15() {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1024 = arith.constant 1024 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %async_token, %results = air.execute -> (memref<8x2x32x32xf32, 1>) {
      %alloc = memref.alloc() : memref<8x2x32x32xf32, 1>
      air.execute_terminator %alloc : memref<8x2x32x32xf32, 1>
    }
    %0 = air.wait_all async 
    %1 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %0) -> (!air.async.token) {
      %2 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %arg1) -> (!air.async.token) {
        %async_token_0, %results_1 = air.execute -> (index) {
          %4 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%arg2]
          air.execute_terminator %4 : index
        } {unrolled = true}
        %3 = air.channel.get async [%arg3, %async_token_0]  @channel_26[%c3, %c3] (%results[%results_1, %arg0, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1]) {affine_map = #map, id = 69 : i32, partition_key = 3 : i32, partitioning = true, split_dim = 1 : i32, split_dim_offset = 0 : i32, split_dim_size = 2 : i32, split_dim_stride_factor = 4 : i32, unrolled = true} : (memref<8x2x32x32xf32, 1>)
        scf.yield %3 : !air.async.token
      }
      scf.yield %2 : !air.async.token
    }
    return
  }

  // Specializing arith addi.
  // CHECK-LABEL: test16
  // CHECK: air.channel.get async {{.*}} @channel_14[%c0{{.*}}, %c0{{.*}}] (%results[] [] [])
  // CHECK: air.channel.get async {{.*}} @channel_13[%c0{{.*}}, %c0{{.*}}] (%results_1[] [] [])
  // CHECK: air.channel.get async {{.*}} @channel_14[%c0{{.*}}, %c0{{.*}}] (%results[] [] [])
  // CHECK: air.channel.get async {{.*}} @channel_13[%c0{{.*}}, %c0{{.*}}] (%results_1[] [] [])

  func.func @test16() {
    %0 = air.segment @segment_0 async  {
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %async_token, %results = air.execute -> (memref<64x256xbf16, 1>) {
        %alloc = memref.alloc() : memref<64x256xbf16, 1>
        air.execute_terminator %alloc : memref<64x256xbf16, 1>
      }
      %async_token_0, %results_1 = air.execute -> (memref<256x64xbf16, 1>) {
        %alloc = memref.alloc() : memref<256x64xbf16, 1>
        air.execute_terminator %alloc : memref<256x64xbf16, 1>
      }
      %1 = air.wait_all async 
      %2 = scf.for %arg0 = %c0 to %c1024 step %c512 iter_args(%arg1 = %1) -> (!air.async.token) {
        %3 = arith.addi %arg0, %c256 : index
        %4 = air.channel.get async [%arg1]  @channel_14[%c0, %c0] (%results[%c0, %3] [%c64, %c256] [%c256, %c1]) {id = 13 : i32} : (memref<64x256xbf16, 1>)
        %5 = air.channel.get async [%arg1]  @channel_13[%c0, %c0] (%results_1[%3, %c0] [%c256, %c64] [%c64, %c1]) {id = 17 : i32} : (memref<256x64xbf16, 1>)
        %6 = air.wait_all async [%4, %5] 
        scf.yield %6 : !air.async.token
      }
    }
    return
  }
}
