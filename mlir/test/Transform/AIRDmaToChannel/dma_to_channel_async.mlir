//===- dma_to_channel_async.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel --split-input-file | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
// CHECK: air.channel @channel_0 [1, 1]
// CHECK: air.channel @channel_1 [2, 2]
// CHECK-LABEL: func.func @func0
  func.func @func0(%arg0: memref<512x512xbf16>) {
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
// CHECK: %[[EVENT14:.*]] = air.wait_all async [%[[EVENT11]], %[[EVENT12]], %[[EVENT13]]]

// CHECK: %[[EVENT15:.*]] = scf.parallel{{.*}}init (%[[EVENT14]])
// CHECK: %[[EVENT16:.*]], %[[VALUE3:.*]] = air.execute
// CHECK: %[[EVENT17:.*]] = air.wait_all async [%[[EVENT14]], %[[EVENT16]]]
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

// -----

// Hoisting affine.if from air.herd.

// CHECK: air.launch
// CHECK: scf.for %[[VALUE0:.*]] = %c0{{.*}} to %c8{{.*}} step %c1{{.*}} iter_args(%[[VALUE1:.*]] = %{{.*}})
// CHECK: affine.apply {{.*}}[%[[VALUE0]]]
// CHECK: air.channel.put{{.*}}@channel_4
// CHECK: scf.yield

// CHECK: air.segment
// CHECK: scf.for %[[VALUE2:.*]] = %c0{{.*}} to %c8{{.*}} step %c1{{.*}} iter_args(%[[VALUE3:.*]] = %{{.*}})
// CHECK: affine.apply {{.*}}[%[[VALUE2]]]
// CHECK: %[[GET0:.*]] = air.channel.get async [%{{.*}}, %[[VALUE3]]]  @channel_4
// CHECK: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[GET0]]{{.*}}]  @channel_0
// CHECK: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[GET0]]{{.*}}]  @channel_1
// CHECK: %[[PUT2:.*]] = air.channel.put async [{{.*}}%[[GET0]]{{.*}}]  @channel_2
// CHECK: %[[PUT3:.*]] = air.channel.put async [{{.*}}%[[GET0]]{{.*}}]  @channel_3

// CHECK: air.herd
// CHECK: affine.if
// CHECK: air.channel.get async{{.*}}@channel_0
// CHECK: else
// CHECK-NEXT: affine.if
// CHECK: air.channel.get async{{.*}}@channel_1
// CHECK: else
// CHECK-NEXT: affine.if
// CHECK: air.channel.get async{{.*}}@channel_2
// CHECK: else
// CHECK: air.channel.get async{{.*}}@channel_3

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  func.func @func1(%arg0: memref<512x512xbf16>) {
    %c2_0 = arith.constant 2 : index
    %0 = air.launch async (%launchX, %launchY) in (%launchXSize=%c2_0, %launchYSize=%c2_0) args(%arg6=%arg0) : memref<512x512xbf16> attributes {id = 5 : i32} {
      %1 = air.segment @segment_0 async  args(%launchXArg = %launchX, %launchYArg = %launchY, %arg9=%arg6) : index, index, memref<512x512xbf16> attributes {id = 4 : i32} {
        %c4096 = arith.constant 4096 : index
        %c16384 = arith.constant 16384 : index
        %c64 = arith.constant 64 : index
        %c32768 = arith.constant 32768 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_0 = arith.constant 1 : index
        %c512 = arith.constant 512 : index
        %c256 = arith.constant 256 : index
        %async_token_x, %arg11 = air.execute -> (index) {
          %17 = affine.apply #map1()[%launchXArg]
          air.execute_terminator %17 : index
        } {id = 8 : i32}
        %async_token_y, %arg13 = air.execute -> (index) {
          %17 = affine.apply #map1()[%launchYArg]
          air.execute_terminator %17 : index
        } {id = 8 : i32}
        %async_token_3, %results_4 = air.execute -> (memref<1x1x8x16x4x8xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>
        } {id = 3 : i32}
        %async_token_7, %results_8 = air.execute -> (memref<4x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<4x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<4x1x64x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %2 = air.wait_all async [%async_token_3, %async_token_7]  {id = 6 : i32}
        %3 = scf.for %arg15 = %c0 to %c8 step %c1_0 iter_args(%arg16 = %2) -> (!air.async.token) {
          %async_token_17, %results_18 = air.execute [%arg16] -> (index) {
            %17 = affine.apply #map()[%arg15]
            air.execute_terminator %17 : index
          } {id = 8 : i32}
          %13 = air.dma_memcpy_nd async [%async_token_17, %arg16] (%results_8[] [] [], %arg9[%c0, %c0, %arg11, %results_18] [%c4, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {id = 1 : i32} : (memref<4x1x64x64xbf16, 1 : i32>, memref<512x512xbf16>)
          %15 = air.herd @herd_0 async [%13, %arg16]  tile (%arg17, %arg18) in (%arg19=%c4, %arg20=%c4) args(%arg21=%results_8, %arg22=%results_4) : memref<4x1x64x64xbf16, 1 : i32>, memref<1x1x8x16x4x8xbf16, 2 : i32> {
            %c512_19 = arith.constant 512 : index
            %c16384_20 = arith.constant 16384 : index
            %c0_21 = arith.constant 0 : index
            %c4096_22 = arith.constant 4096 : index
            %c8_23 = arith.constant 8 : index
            %c256_24 = arith.constant 256 : index
            %c64_25 = arith.constant 64 : index
            %c1_26 = arith.constant 1 : index
            %c16 = arith.constant 16 : index
            %c4_27 = arith.constant 4 : index
            %17 = affine.if #set()[%arg17, %arg18] -> !air.async.token {
              %c0_29 = arith.constant 0 : index
              %19 = air.dma_memcpy_nd async (%arg22[] [] [], %arg21[%c0_29, %c0_21, %c0_21, %c0_21, %c0_21, %c0_21] [%c1_26, %c1_26, %c8_23, %c16, %c4_27, %c8_23] [%c4096_22, %c4096_22, %c8_23, %c256_24, %c64_25, %c1_26]) {broadcast_set = #set, id = 3 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<4x1x64x64xbf16, 1 : i32>)
              affine.yield %19 : !air.async.token
            } else {
              %19 = affine.if #set1()[%arg17, %arg18] -> !air.async.token {
                %c1_29 = arith.constant 1 : index
                %20 = air.dma_memcpy_nd async (%arg22[] [] [], %arg21[%c1_29, %c0_21, %c0_21, %c0_21, %c0_21, %c0_21] [%c1_26, %c1_26, %c8_23, %c16, %c4_27, %c8_23] [%c4096_22, %c4096_22, %c8_23, %c256_24, %c64_25, %c1_26]) {broadcast_set = #set1, id = 4 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<4x1x64x64xbf16, 1 : i32>)
                affine.yield %20 : !air.async.token
              } else {
                %20 = affine.if #set2()[%arg17, %arg18] -> !air.async.token {
                  %c2 = arith.constant 2 : index
                  %21 = air.dma_memcpy_nd async (%arg22[] [] [], %arg21[%c2, %c0_21, %c0_21, %c0_21, %c0_21, %c0_21] [%c1_26, %c1_26, %c8_23, %c16, %c4_27, %c8_23] [%c4096_22, %c4096_22, %c8_23, %c256_24, %c64_25, %c1_26]) {broadcast_set = #set2, id = 5 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<4x1x64x64xbf16, 1 : i32>)
                  affine.yield %21 : !air.async.token
                } else {
                  %c3 = arith.constant 3 : index
                  %21 = air.dma_memcpy_nd async (%arg22[] [] [], %arg21[%c3, %c0_21, %c0_21, %c0_21, %c0_21, %c0_21] [%c1_26, %c1_26, %c8_23, %c16, %c4_27, %c8_23] [%c4096_22, %c4096_22, %c8_23, %c256_24, %c64_25, %c1_26]) {broadcast_set = #set3, id = 6 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<4x1x64x64xbf16, 1 : i32>)
                  affine.yield %21 : !air.async.token
                }
                affine.yield %20 : !air.async.token
              }
              affine.yield %19 : !air.async.token
            }
          }
          %16 = air.wait_all async [%arg16, %15]  {id = 1 : i32}
          scf.yield %16 : !air.async.token
        }
        %async_token_12 = air.execute [%3] {
          memref.dealloc %results_8 : memref<4x1x64x64xbf16, 1 : i32>
        } {id = 11 : i32}
        %async_token_14 = air.execute [%3] {
          memref.dealloc %results_4 : memref<1x1x8x16x4x8xbf16, 2 : i32>
        } {id = 13 : i32}
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @func2
// CHECK: air.segment
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for

// CHECK: %[[TOK0:.*]], %[[EXEC0:.*]] = air.execute
// CHECK-NEXT: affine.apply {{.*}}[%{{.*}}, %c0{{.*}}]
// CHECK: air.channel.put async{{.*}}@channel_0[] (%{{.*}}[%[[EXEC0]], 
// CHECK: %[[TOK1:.*]], %[[EXEC1:.*]] = air.execute
// CHECK-NEXT: affine.apply {{.*}}[%{{.*}}, %c1{{.*}}]
// CHECK: air.channel.put async{{.*}}@channel_1[] (%{{.*}}[%[[EXEC1]], 
// CHECK: air.herd

#map = affine_map<()[s0, s1] -> (s0 + s1)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  func.func @func2() {
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c16) attributes {id = 5 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 4 : i32} {
        %c15 = arith.constant 15 : index
        %c4 = arith.constant 4 : index
        %c2_0 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %async_token, %results = air.execute -> (memref<1x1x4x8x4x8xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x4x8x4x8xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x4x8x4x8xbf16, 2 : i32>
        } {id = 7 : i32}
        %async_token_1, %results_2 = air.execute -> (memref<8x16x32x32xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x16x32x32xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<8x16x32x32xbf16, 1 : i32>
        } {id = 9 : i32}
        scf.for %arg7 = %c0 to %c8 step %c4 {
          scf.for %arg8 = %c0 to %c8 step %c4 {
            %2 = scf.for %arg9 = %c1 to %c15 step %c1 iter_args(%arg10 = %async_token) -> (!air.async.token) {
              %3 = air.herd @herd_0 async [%arg10]  tile (%arg11, %arg12) in (%arg13=%c2_0, %arg14=%c2_0) args(%arg15=%arg7, %arg16=%results, %arg17=%results_2, %arg18=%arg9) : index, memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>, index attributes {id = 2 : i32, link_with = "mm.o"} {
                %c0_5 = arith.constant 0 : index
                %c1_6 = arith.constant 1 : index
                %c4_7 = arith.constant 4 : index
                %c8_8 = arith.constant 8 : index
                %c16384 = arith.constant 16384 : index
                %c1024 = arith.constant 1024 : index
                %c128 = arith.constant 128 : index
                %c32 = arith.constant 32 : index
                %async_token_9, %results_10 = air.execute -> (index) {
                  %5 = affine.apply #map()[%arg15, %arg11]
                  air.execute_terminator %5 : index
                } {id = 16 : i32}
                %4 = affine.if #set()[%arg11, %arg12] -> !air.async.token {
                  %5 = air.dma_memcpy_nd async [%async_token_9] (%arg16[] [] [], %arg17[%results_10, %arg18, %c0_5, %c0_5, %c0_5, %c0_5] [%c1_6, %c1_6, %c4_7, %c8_8, %c4_7, %c8_8] [%c16384, %c1024, %c8_8, %c128, %c32, %c1_6]) {broadcast_set = #set, id = 11 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
                  affine.yield %5 : !air.async.token
                } else {
                  %5 = air.dma_memcpy_nd async [%async_token_9] (%arg16[] [] [], %arg17[%results_10, %arg18, %c0_5, %c0_5, %c0_5, %c0_5] [%c1_6, %c1_6, %c4_7, %c8_8, %c4_7, %c8_8] [%c16384, %c1024, %c8_8, %c128, %c32, %c1_6]) {broadcast_set = #set1, id = 12 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
                  affine.yield %5 : !air.async.token
                }
              }
              scf.yield %3 : !air.async.token
            }
          }
        }
        %async_token_3 = air.execute {
          memref.dealloc %results_2 : memref<8x16x32x32xbf16, 1 : i32>
        } {id = 22 : i32}
        %async_token_4 = air.execute [%async_token] {
          memref.dealloc %results : memref<1x1x4x8x4x8xbf16, 2 : i32>
        } {id = 24 : i32}
      }
    }
    return
  }
}


