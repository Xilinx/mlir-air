//===- dma_to_channel.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -split-input-file | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
// CHECK: air.channel @channel_0 [2, 2]
// CHECK: air.channel @channel_1 [2, 2]
// CHECK: air.channel @channel_2 [2, 2]
// CHECK: air.channel @channel_3 [2, 2]
// CHECK-LABEL: func.func @mmult
  func.func @mmult(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64x64xi32>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %alloc, %alloc_0 : memref<64x64xi32> to memref<64x64xi32>
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_0

// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_1

// CHECK: %[[EVENT12:.*]] = air.wait_all async
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_2

// CHECK: %[[EVENT18:.*]] = air.wait_all async
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@channel_3
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %0 = affine.apply #map()[%arg2]
      %1 = affine.apply #map()[%arg3]
      scf.for %arg9 = %c0 to %c64 step %c32 {
        %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
        %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
        %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
// CHECK: air.channel.get{{.*}}@channel_0
        air.dma_memcpy_nd (%alloc_1[] [] [], %arg6[%0, %arg9] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.get{{.*}}@channel_1
        air.dma_memcpy_nd (%alloc_2[] [] [], %arg7[%arg9, %1] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.get{{.*}}@channel_2
        air.dma_memcpy_nd (%alloc_3[] [] [], %arg8[%0, %1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.put{{.*}}@channel_3
        linalg.matmul ins(%alloc_1, %alloc_2 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_3 : memref<32x32xi32, 2>)
        air.dma_memcpy_nd (%arg8[%0, %1] [%c32, %c32] [%c64, %c1], %alloc_3[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
        memref.dealloc %alloc_1 : memref<32x32xi32, 2>
        memref.dealloc %alloc_2 : memref<32x32xi32, 2>
        memref.dealloc %alloc_3 : memref<32x32xi32, 2>
      }
    }
    return %alloc_0 : memref<64x64xi32>
  }
}

// -----

#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
module {
// CHECK: air.channel @channel_0 [1, 1]
// CHECK: air.launch
// CHECK: scf.parallel (%[[ARG0:.*]], %[[ARG1:.*]]) = ({{.*}}) to ({{.*}}) step ({{.*}}) init (%{{.*}})
// CHECK: air.channel.get async [%{{.*}}]  @channel_0[%[[ARG0]], %[[ARG1]]]
// CHECK: scf.reduce
// CHECK: scf.reduce.return
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0 async  tile (%[[ARG2:.*]], %[[ARG3:.*]]) in ({{.*}}) args({{.*}})
// CHECK: air.channel.put async [%{{.*}}]  @channel_0[%[[ARG2]], %[[ARG3]]]
  func.func @l1tol3(%arg0: memref<16x32xf32>, %arg1: memref<16x32xf32>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg1, %arg7=%arg0) : memref<16x32xf32>, memref<16x32xf32> attributes {id = 3 : i32} {
      %1 = air.segment @segment_0 async  args(%arg8=%arg2, %arg9=%arg3, %arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7) : index, index, index, index, memref<16x32xf32>, memref<16x32xf32> attributes {id = 2 : i32} {
        %c1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (index) {
          %3 = affine.apply #map()[%arg8]
          air.execute_terminator %3 : index
        } {id = 1 : i32}
        %async_token_0, %results_1 = air.execute -> (index) {
          %3 = affine.apply #map1()[%arg9]
          air.execute_terminator %3 : index
        } {id = 2 : i32}
        %subview = memref.subview %arg12[%results, %results_1] [8, 16] [1, 1] : memref<16x32xf32> to memref<8x16xf32, strided<[32, 1], offset: ?>>
        %async_token_2, %results_3 = air.execute -> (index) {
          %3 = affine.apply #map()[%arg8]
          air.execute_terminator %3 : index
        } {id = 3 : i32}
        %async_token_4, %results_5 = air.execute -> (index) {
          %3 = affine.apply #map1()[%arg9]
          air.execute_terminator %3 : index
        } {id = 4 : i32}
        %async_token_6, %results_7 = air.execute -> (memref<8x16xf32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x16xf32, 1 : i32>
          air.execute_terminator %alloc : memref<8x16xf32, 1 : i32>
        } {id = 5 : i32}
        %2 = air.herd @herd_0 async  tile (%arg14, %arg15) in (%arg16=%c1, %arg17=%c1) args(%arg18=%results_7, %arg19=%subview) : memref<8x16xf32, 1 : i32>, memref<8x16xf32, strided<[32, 1], offset: ?>> attributes {id = 1 : i32} {
          %async_token_9, %results_10 = air.execute -> (memref<8x16xf32, 2 : i32>) {
            %alloc = memref.alloc() : memref<8x16xf32, 2 : i32>
            air.execute_terminator %alloc : memref<8x16xf32, 2 : i32>
          } {id = 6 : i32}
          %3 = air.dma_memcpy_nd async [%async_token_9] (%arg19[] [] [], %results_10[] [] []) {id = 1 : i32} : (memref<8x16xf32, strided<[32, 1], offset: ?>>, memref<8x16xf32, 2 : i32>)
          %async_token_11 = air.execute [%3] {
            memref.dealloc %results_10 : memref<8x16xf32, 2 : i32>
          } {id = 7 : i32}
        }
        %async_token_8 = air.execute [%2, %async_token_6] {
          memref.dealloc %results_7 : memref<8x16xf32, 1 : i32>
        } {id = 8 : i32}
      }
    }
    return
  }
}

// -----

// Hoisting external channel.put/get op to scf.parallel, with affine.if guarding those ops.

// CHECK: air.channel @channel_0 [2, 4]
// CHECK: air.launch
// CHECK: air.segment @segment_0
// CHECK: scf.parallel (%[[ARG0:.*]], %[[ARG1:.*]]) = (%c0{{.*}}, %c0{{.*}}) to (%c2{{.*}}, %c1{{.*}}) step (%c1{{.*}}, %c1{{.*}})
// CHECK: air.channel.get  @channel_0[%[[ARG0]], %[[ARG1]]]
// CHECK: scf.reduce
// CHECK: air.herd @herd_0  tile (%[[ARG2:.*]], %[[ARG3:.*]]) in
// CHECK: affine.if
// CHECK: else
// CHECK: affine.if
// CHECK: else
// CHECK: air.channel.put  @channel_0[%[[ARG2]], %[[ARG3]]]
#map = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 3 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 >= 0, -s1 + 2 >= 0)>
module {
  func.func @affine_if(%arg0: memref<512xi32>, %arg1: memref<512x256xi32>) -> memref<256xi32> {
    %c4 = arith.constant 4 : index
    %alloc = memref.alloc() : memref<256xi32>
    air.launch (%arg2) in (%arg3=%c4) {
      %c1 = arith.constant 1 : index
      air.segment @segment_0  unroll(%arg4) in (%arg5=%c1) {
        %c4_0 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %alloc_1 = memref.alloc() : memref<64xi32, 1 : i32>
        air.herd @herd_0  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c4_0) args(%arg10=%alloc_1) : memref<64xi32, 1 : i32> {
          %c32 = arith.constant 32 : index
          %c1_2 = arith.constant 1 : index
          %0 = affine.apply #map()[%arg6]
          %alloc_3 = memref.alloc() : memref<32xi32, 2 : i32>
          affine.if #set()[%arg6, %arg7] {
          } else {
            affine.if #set1()[%arg6, %arg7] {
            } else {
              air.dma_memcpy_nd (%arg10[%0] [%c32] [%c1_2], %alloc_3[] [] []) {id = 5 : i32} : (memref<64xi32, 1 : i32>, memref<32xi32, 2 : i32>)
            }
          }
          memref.dealloc %alloc_3 : memref<32xi32, 2 : i32>
        }
        memref.dealloc %alloc_1 : memref<64xi32, 1 : i32>
      }
    }
    return %alloc : memref<256xi32>
  }
}

// -----

// Test case: Multiple broadcast DMAs reading from the same L1 buffer should
// all depend on the DMA that fills that buffer. This is a regression test for
// a bug in areOverlappingPartialMemrefs where only the first channel.put would
// depend on the channel.get that writes to the L1 buffer.

// CHECK-LABEL: func.func @test_overlapping_l1_reads
// CHECK: air.segment
// CHECK: scf.for %[[ARG:.*]] = %c0{{.*}} to %c512{{.*}} step %c64
// The channel.get fills results buffer
// CHECK: %[[GET0:.*]] = air.channel.get async{{.*}}@channel_{{[0-9]+}}[] (%[[RESULTS:.*]][%c0{{.*}}, %[[ARG]]]
// All channel.put operations reading from results must depend on GET0
// CHECK: air.channel.put async [%[[GET0]]{{.*}}]{{.*}}@channel_{{[0-9]+}}[] (%[[RESULTS]][%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %[[ARG]]]
// CHECK: air.channel.put async [%[[GET0]]{{.*}}]{{.*}}@channel_{{[0-9]+}}[] (%[[RESULTS]][%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %[[ARG]]]
// CHECK: air.channel.put async [%[[GET0]]{{.*}}]{{.*}}@channel_{{[0-9]+}}[] (%[[RESULTS]][%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %[[ARG]]]
// CHECK: air.channel.put async [%[[GET0]]{{.*}}]{{.*}}@channel_{{[0-9]+}}[] (%[[RESULTS]][%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %[[ARG]]]

#map2 = affine_map<()[s0] -> (s0 * 64)>
#set10 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set11 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set12 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set13 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>

module {
  func.func @test_overlapping_l1_reads(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%arg9, %arg10, %arg11) in (%arg12=%c4, %arg13=%c4, %arg14=%c1) args(%arg15=%arg0) : memref<*xbf16> attributes {id = 5 : i32} {
      %1 = air.segment @segment async args(%arg20=%arg15) : memref<*xbf16> attributes {id = 4 : i32} {
        %c1_0 = arith.constant 1 : index
        %c512 = arith.constant 512 : index
        %c256 = arith.constant 256 : index
        %c4096 = arith.constant 4096 : index
        %c8 = arith.constant 8 : index
        %c4_1 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c192 = arith.constant 192 : index

        // Allocate L1 buffer
        %async_token, %results = air.execute -> (memref<256x512xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x512xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x512xbf16, 1 : i32>
        } {id = 1 : i32}

        %async_token_16, %results_17 = air.execute -> (memref<8x8x8x8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8x8x8x8xbf16, 2>
          air.execute_terminator %alloc : memref<8x8x8x8xbf16, 2>
        } {id = 6 : i32}

        // Loop that fills L1 buffer and then broadcasts to AIE tiles
        %5 = scf.for %arg23 = %c0 to %c512 step %c64 iter_args(%arg24 = %async_token) -> (!air.async.token) {
          // DMA fills L1 buffer at [0, arg23] with size [256, 64]
          %12 = air.dma_memcpy_nd async [%arg24] (%results[%c0, %arg23] [%c256, %c64] [%c512, %c1_0], %arg20[%c0, %arg23] [%c256, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<256x512xbf16, 1 : i32>, memref<*xbf16>)

          // Herd with broadcast DMAs reading from different parts of the same L1 buffer
          // Each broadcast DMA reads from results buffer at different row offsets (0, 64, 128, 192)
          // but ALL read within the region [0:256, arg23:arg23+64] that was just written
          %14 = air.herd @herd_0 async [%12, %arg24] tile (%arg25, %arg26) in (%arg27=%c4_1, %arg28=%c4_1) args(%arg29=%results, %arg30=%arg23, %arg31=%results_17) : memref<256x512xbf16, 1 : i32>, index, memref<8x8x8x8xbf16, 2> attributes {id = 2 : i32} {
            %c0_11 = arith.constant 0 : index
            %c8_12 = arith.constant 8 : index
            %c4096_c = arith.constant 4096 : index
            %c512_13 = arith.constant 512 : index
            %c1_14 = arith.constant 1 : index
            %c64_c = arith.constant 64 : index
            %c128_c = arith.constant 128 : index
            %c192_c = arith.constant 192 : index

            // Different rows of L1 buffer are broadcast to different herd rows
            // Row 0 of herd gets rows 0-63 of L1
            // Row 1 of herd gets rows 64-127 of L1  
            // Row 2 of herd gets rows 128-191 of L1
            // Row 3 of herd gets rows 192-255 of L1
            // All of these are WITHIN the region written by the DMA above!
            %19 = affine.if #set10()[%arg25, %arg26] -> !air.async.token {
              %24 = air.dma_memcpy_nd async (%arg31[] [] [], %arg29[%c0_11, %c0_11, %c0_11, %arg30] [%c8_12, %c8_12, %c8_12, %c8_12] [%c8_12, %c4096_c, %c512_13, %c1_14]) {broadcast_set = #set10, id = 3 : i32} : (memref<8x8x8x8xbf16, 2>, memref<256x512xbf16, 1 : i32>)
              affine.yield %24 : !air.async.token
            } else {
              %24 = affine.if #set11()[%arg25, %arg26] -> !air.async.token {
                %25 = air.dma_memcpy_nd async (%arg31[] [] [], %arg29[%c0_11, %c0_11, %c64_c, %arg30] [%c8_12, %c8_12, %c8_12, %c8_12] [%c8_12, %c4096_c, %c512_13, %c1_14]) {broadcast_set = #set11, id = 4 : i32} : (memref<8x8x8x8xbf16, 2>, memref<256x512xbf16, 1 : i32>)
                affine.yield %25 : !air.async.token
              } else {
                %25 = affine.if #set12()[%arg25, %arg26] -> !air.async.token {
                  %26 = air.dma_memcpy_nd async (%arg31[] [] [], %arg29[%c0_11, %c0_11, %c128_c, %arg30] [%c8_12, %c8_12, %c8_12, %c8_12] [%c8_12, %c4096_c, %c512_13, %c1_14]) {broadcast_set = #set12, id = 5 : i32} : (memref<8x8x8x8xbf16, 2>, memref<256x512xbf16, 1 : i32>)
                  affine.yield %26 : !air.async.token
                } else {
                  %26 = air.dma_memcpy_nd async (%arg31[] [] [], %arg29[%c0_11, %c0_11, %c192_c, %arg30] [%c8_12, %c8_12, %c8_12, %c8_12] [%c8_12, %c4096_c, %c512_13, %c1_14]) {broadcast_set = #set13, id = 6 : i32} : (memref<8x8x8x8xbf16, 2>, memref<256x512xbf16, 1 : i32>)
                  affine.yield %26 : !air.async.token
                }
                affine.yield %25 : !air.async.token
              }
              affine.yield %24 : !air.async.token
            }
          }
          %15 = air.wait_all async [%14] {id = 12 : i32}
          scf.yield %15 : !air.async.token
        }
      }
    }
    return
  }
}
