//===- dma_to_channel_nested_for_in_segment.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -split-input-file | FileCheck %s

// Hoisting external channel put/get op out of a segment with nested for loops

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
// CHECK-LABEL: func.func @func1
  func.func @func1(%arg0: memref<512x512xbf16>) {
    %c8 = arith.constant 8 : index
    %c1_0 = arith.constant 1 : index
// CHECK: %[[EVENT0:.*]] = air.launch async
// CHECK: %[[EVENT1:.*]] = scf.for
// CHECK: %[[EVENT2:.*]] = scf.for
// CHECK: %[[EVENT3:.*]] = air.channel.put{{.*}}@channel_0[]
// CHECK: %[[EVENT4:.*]] = air.segment async
// CHECK: %[[EVENT5:.*]] = scf.for
// CHECK: %[[EVENT6:.*]] = scf.for
// CHECK: %[[EVENT7:.*]] = air.channel.get{{.*}}@channel_0[]
// CHECK: %[[EVENT8:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: %[[EVENT9:.*]] = scf.for
// CHECK: %[[EVENT10:.*]] = air.channel.put{{.*}}@channel_1[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT11:.*]] = air.herd @herd_0 async
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1_0, %arg4=%c8) args(%arg5=%arg0) : memref<512x512xbf16> attributes {id = 3 : i32} {
      %1 = air.segment async  args(%arg6=%arg1, %arg7=%arg2, %arg8=%arg3, %arg9=%arg4, %arg10=%arg5) : index, index, index, index, memref<512x512xbf16> attributes {id = 2 : i32} {
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
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @func2
// CHECK: air.launch

// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: scf.yield

// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: scf.yield

// CHECK: air.segment
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.herd
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.herd
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: scf.yield

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func private @linalg_fill_bf16_view1x1x16x16x4x4xbf16as2(bf16, memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>) attributes {link_with = "mm.o", llvm.emit_c_interface}
  func.func private @matmul_bf16_bf16(memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, 2 : i32>, memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>) attributes {link_with = "mm.o", llvm.emit_c_interface}
  func.func @func2(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg5=%arg0, %arg6=%arg1) : memref<512x512xbf16>, memref<512x512xbf16> attributes {id = 5 : i32} {
      %1 = air.segment @segment_0 async  args(%arg7=%arg5, %arg8=%arg6) : memref<512x512xbf16>, memref<512x512xbf16> attributes {id = 4 : i32} {
        %c64 = arith.constant 64 : index
        %c32768 = arith.constant 32768 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_0 = arith.constant 1 : index
        %c512 = arith.constant 512 : index
        %c256 = arith.constant 256 : index
        %async_token, %results = air.execute -> (memref<1x4x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x4x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x4x64x64xbf16, 1 : i32>
        } {id = 4 : i32}
        %async_token_1, %results_2 = air.execute -> (memref<4x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<4x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<4x1x64x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %2 = air.wait_all async [%async_token, %async_token_1]  {id = 6 : i32}
        %3 = scf.for %arg9 = %c0 to %c512 step %c256 iter_args(%arg10 = %2) -> (!air.async.token) {
          %4 = scf.for %arg11 = %c0 to %c512 step %c256 iter_args(%arg12 = %arg10) -> (!air.async.token) {
            %6 = air.herd @herd_0 async [%arg12]  tile (%arg13, %arg14) in (%arg15=%c4, %arg16=%c4) 
            %7 = scf.for %arg13 = %c0 to %c8 step %c1_0 iter_args(%arg14 = %6) -> (!air.async.token) {
              %async_token_5, %results_6 = air.execute [%arg14] -> (index) {
                %12 = affine.apply #map()[%arg13]
                air.execute_terminator %12 : index
              } {id = 8 : i32}
              %8 = air.dma_memcpy_nd async [%async_token_5, %arg14] (%results_2[] [] [], %arg7[%c0, %c0, %arg9, %results_6] [%c4, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {id = 1 : i32} : (memref<4x1x64x64xbf16, 1 : i32>, memref<512x512xbf16>)
              %9 = air.dma_memcpy_nd async [%async_token_5, %arg14] (%results[] [] [], %arg8[%c0, %c0, %results_6, %arg11] [%c1_0, %c4, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {id = 2 : i32} : (memref<1x4x64x64xbf16, 1 : i32>, memref<512x512xbf16>)
              %10 = air.herd @herd_0 async [%8, %9]  tile (%arg15, %arg16) in (%arg17=%c4, %arg18=%c4) 
              %11 = air.wait_all async [%arg14, %10]  {id = 1 : i32}
              scf.yield %11 : !air.async.token
            }
            scf.yield %7 : !air.async.token
          }
          %5 = air.wait_all async [%arg10, %4]  {id = 5 : i32}
          scf.yield %5 : !air.async.token
        }
        %async_token_3 = air.execute [%3] {
          memref.dealloc %results_2 : memref<4x1x64x64xbf16, 1 : i32>
        } {id = 11 : i32}
        %async_token_4 = air.execute [%3] {
          memref.dealloc %results : memref<1x4x64x64xbf16, 1 : i32>
        } {id = 12 : i32}
      }
    }
    return
  }
}

// -----

// Hoisting memcpy ops from affine.if in herd to segment, nested within scf.for loops.

// CHECK-LABEL: func.func @func3
// CHECK: air.launch
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_4
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: scf.yield

// CHECK: air.segment
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for %[[VALUE0:.*]] = %c0{{.*}} to %c8{{.*}} step %c1{{.*}} iter_args(%[[VALUE1:.*]] = %{{.*}})
// CHECK: affine.apply {{.*}}[%[[VALUE0]]]
// CHECK: %[[GET0:.*]] = air.channel.get async [%{{.*}}, %[[VALUE1]]]  @channel_4
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
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  func.func @func3(%arg0: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg6=%arg0) : memref<512x512xbf16> attributes {id = 5 : i32} {
      %1 = air.segment @segment_0 async  args(%arg9=%arg6) : memref<512x512xbf16> attributes {id = 4 : i32} {
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
        %async_token_3, %results_4 = air.execute -> (memref<1x1x8x16x4x8xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>
        } {id = 3 : i32}
        %async_token_7, %results_8 = air.execute -> (memref<4x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<4x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<4x1x64x64xbf16, 1 : i32>
        } {id = 5 : i32}
        %2 = air.wait_all async [%async_token_3, %async_token_7]  {id = 6 : i32}
        %3 = scf.for %arg11 = %c0 to %c512 step %c256 iter_args(%arg12 = %2) -> (!air.async.token) {
          %4 = air.wait_all async [%arg12]  {id = 4 : i32}
          %5 = scf.for %arg13 = %c0 to %c512 step %c256 iter_args(%arg14 = %4) -> (!air.async.token) {
            %8 = air.wait_all async [%arg14]  {id = 2 : i32}
            %9 = scf.for %arg15 = %c0 to %c8 step %c1_0 iter_args(%arg16 = %8) -> (!air.async.token) {
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
            %12 = air.wait_all async [%arg14]  {id = 3 : i32}
            scf.yield %12 : !air.async.token
          }
          %6 = air.wait_all async [%arg12, %5]  {id = 5 : i32}
          scf.yield %6 : !air.async.token
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
