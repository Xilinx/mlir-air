//===- shim_packet_flow_npu.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -pass-pipeline='builtin.module(air-to-aie{row-offset=2 col-offset=0 device=npu1_1col use-pkt-flow-at-shim-dma=true})' --split-input-file -verify-diagnostics | FileCheck %s
// RUN: air-opt %s -pass-pipeline='builtin.module(air-to-aie{row-offset=2 col-offset=0 device=npu1 use-pkt-flow-at-shim-dma=true})' --split-input-file | FileCheck %s --check-prefix=WHOLEARRAY

// CHECK: %[[VAL0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL2:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(0) {
// CHECK:   aie.packet_source<%[[VAL2]], DMA : 0>
// CHECK:   aie.packet_dest<%[[VAL0]], DMA : 0>
// CHECK: }
// CHECK: aie.flow(%[[VAL0]], DMA : 0, %[[VAL1]], DMA : 0)
// CHECK: aie.flow(%[[VAL1]], DMA : 0, %[[VAL0]], DMA : 1)
// CHECK: aie.flow(%[[VAL0]], DMA : 1, %[[VAL2]], DMA : 0)
// CHECK: aie.shim_dma_allocation @air_channel_3(S2MM, 0, 0)
// CHECK: memref.global "public" @air_channel_3 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @air_channel_0(MM2S, 0, 0)
// CHECK: memref.global "public" @air_channel_0 : memref<64xi32, 1>
// CHECK: @func0
// CHECK: air.channel.put  @channel_0[] {{.*}} metadataArray = [{base = "air_channel_0", index = 0 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
#map2 = affine_map<(d0) -> (d0)>
air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
air.channel @channel_2 [1, 1]
air.channel @channel_3 [1, 1]
func.func @func0(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  air.channel.put @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
  air.segment @segment0 {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %memtile0 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @channel_0[] (%memtile0[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
    air.channel.put @channel_1[] (%memtile0[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    memref.dealloc %memtile0 : memref<64xi32, 1>
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="func4"} {
      %buf0 = memref.alloc() : memref<64xi32, 2>
      %buf1 = memref.alloc() : memref<64xi32, 2>
      air.channel.get @channel_1[%tx, %ty] (%buf0[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
      linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%buf0 : memref<64xi32, 2>) outs(%buf1 : memref<64xi32, 2>) {
      ^bb0(%arg11: i32, %arg12: i32):
        %c1_32 = arith.constant 1 : i32
        %12 = arith.addi %arg11, %c1_32 : i32
        linalg.yield %12 : i32
      }
      air.channel.put @channel_2[%tx, %ty] (%buf1[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
      memref.dealloc %buf0 : memref<64xi32, 2>
      memref.dealloc %buf1 : memref<64xi32, 2>
    }
    %memtile1 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @channel_2[] (%memtile1[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
    air.channel.put @channel_3[] (%memtile1[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
    memref.dealloc %memtile1 : memref<64xi32, 1>
  }
  air.channel.get @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
  return
}

// -----

// Asynchronous version

// CHECK: %[[VAL0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL2:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(0) {
// CHECK:   aie.packet_source<%[[VAL2]], DMA : 0>
// CHECK:   aie.packet_dest<%[[VAL0]], DMA : 0>
// CHECK: }
// CHECK: aie.flow(%[[VAL0]], DMA : 0, %[[VAL1]], DMA : 0)
// CHECK: aie.flow(%[[VAL1]], DMA : 0, %[[VAL0]], DMA : 1)
// CHECK: aie.flow(%[[VAL0]], DMA : 1, %[[VAL2]], DMA : 0)
// CHECK: aie.shim_dma_allocation @air_channel_3(S2MM, 0, 0)
// CHECK: memref.global "public" @air_channel_3 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @air_channel_0(MM2S, 0, 0)
// CHECK: memref.global "public" @air_channel_0 : memref<64xi32, 1>
// CHECK: @func1
// CHECK: air.channel.put async @channel_0[] {{.*}} metadataArray = [{base = "air_channel_0", index = 0 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
#map = affine_map<(d0) -> (d0)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func1(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %0 = air.channel.put async  @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
    %1 = air.segment @segment0 async  attributes {id = 2 : i32} {
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      } {id = 1 : i32}
      %3 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
      %4 = air.channel.put async [%3]  @channel_1[] (%results[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
      %async_token_2 = air.execute [%4] {
        memref.dealloc %results : memref<64xi32, 1>
      } {id = 2 : i32}
      %5 = air.herd @func4 async  tile (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_1) attributes {id = 1 : i32} {
        %async_token_6, %results_7 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 3 : i32}
        %async_token_8, %results_9 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 4 : i32}
        %8 = air.channel.get async [%async_token_6]  @channel_1[%arg2, %arg3] (%results_7[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        %async_token_10 = air.execute [%async_token_8, %8] {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%results_7 : memref<64xi32, 2>) outs(%results_9 : memref<64xi32, 2>) {
          ^bb0(%in: i32, %out: i32):
            %c1_i32 = arith.constant 1 : i32
            %10 = arith.addi %in, %c1_i32 : i32
            linalg.yield %10 : i32
          }
        } {id = 5 : i32}
        %9 = air.channel.put async [%async_token_10]  @channel_2[%arg2, %arg3] (%results_9[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
        %async_token_11 = air.execute [%async_token_10] {
          memref.dealloc %results_7 : memref<64xi32, 2>
        } {id = 6 : i32}
        %async_token_12 = air.execute [%9] {
          memref.dealloc %results_9 : memref<64xi32, 2>
        } {id = 7 : i32}
      }
      %async_token_3, %results_4 = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      } {id = 8 : i32}
      %6 = air.channel.get async [%async_token_3]  @channel_2[] (%results_4[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
      %7 = air.channel.put async [%6]  @channel_3[] (%results_4[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
      %async_token_5 = air.execute [%7] {
        memref.dealloc %results_4 : memref<64xi32, 1>
      } {id = 9 : i32}
    }
    %2 = air.channel.get async  @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
    return
  }
}

// -----

// 4x4 NPU1 array.


// WHOLEARRAY: %[[shim_noc_tile_0_0:.*]] = aie.tile(0, 0)
// WHOLEARRAY: %[[shim_noc_tile_1_0:.*]] = aie.tile(1, 0)
// WHOLEARRAY: %[[shim_noc_tile_2_0:.*]] = aie.tile(2, 0)
// WHOLEARRAY: %[[shim_noc_tile_3_0:.*]] = aie.tile(3, 0)
// WHOLEARRAY: aie.shim_dma_allocation @air_channel_2_0(MM2S, 0, 0)
// WHOLEARRAY: memref.global "public" @air_channel_2_0 : memref<1x1x64x64xbf16, 1 : i32>
// WHOLEARRAY: aie.shim_dma_allocation @air_channel_2_1(MM2S, 0, 1)
// WHOLEARRAY: memref.global "public" @air_channel_2_1 : memref<1x1x64x64xbf16, 1 : i32>
// WHOLEARRAY: aie.shim_dma_allocation @air_channel_2_2(MM2S, 0, 2)
// WHOLEARRAY: memref.global "public" @air_channel_2_2 : memref<1x1x64x64xbf16, 1 : i32>
// WHOLEARRAY: aie.shim_dma_allocation @air_channel_2_3(MM2S, 0, 3)
// WHOLEARRAY: memref.global "public" @air_channel_2_3 : memref<1x1x64x64xbf16, 1 : i32>
// WHOLEARRAY: @func2
// WHOLEARRAY: air.channel.put {{.*}} @channel_2[%c0{{.*}}, %c0{{.*}}] {{.*}} metadataArray = [{base = "air_channel_2_0", index = 0 : i32}, {base = "air_channel_2_1", index = 1 : i32}, {base = "air_channel_2_2", index = 2 : i32}, {base = "air_channel_2_3", index = 3 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
// WHOLEARRAY: air.channel.put {{.*}} @channel_2[%c1{{.*}}, %c0{{.*}}] {{.*}} metadataArray = [{base = "air_channel_2_0", index = 0 : i32}, {base = "air_channel_2_1", index = 1 : i32}, {base = "air_channel_2_2", index = 2 : i32}, {base = "air_channel_2_3", index = 3 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>
// WHOLEARRAY: air.channel.put {{.*}} @channel_2[%c2{{.*}}, %c0{{.*}}] {{.*}} metadataArray = [{base = "air_channel_2_0", index = 0 : i32}, {base = "air_channel_2_1", index = 1 : i32}, {base = "air_channel_2_2", index = 2 : i32}, {base = "air_channel_2_3", index = 3 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>
// WHOLEARRAY: air.channel.put {{.*}} @channel_2[%c3{{.*}}, %c0{{.*}}] {{.*}} metadataArray = [{base = "air_channel_2_0", index = 0 : i32}, {base = "air_channel_2_1", index = 1 : i32}, {base = "air_channel_2_2", index = 2 : i32}, {base = "air_channel_2_3", index = 3 : i32}], packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>

#map = affine_map<()[s0] -> (s0 * 256)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  air.channel @channel_12 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_13 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_14 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_15 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_2 [4, 1]
  func.func @func2(%arg0: memref<512x512xbf16>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0) : memref<512x512xbf16> attributes {id = 1 : i32} {
      %c3 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg3]
        air.execute_terminator %3 : index
      }
      %1 = scf.for %arg8 = %c0 to %c512 step %c64 iter_args(%arg9 = %async_token) -> (!air.async.token) {
        %3 = air.channel.put async [%arg9]  @channel_2[%c0, %c0] (%arg7[%c0, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 1 : i32} : (memref<512x512xbf16>)
        %4 = air.channel.put async [%arg9]  @channel_2[%c1, %c0] (%arg7[%c1, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 2 : i32} : (memref<512x512xbf16>)
// Note: this error is only expected from "device=npu1_1col"; no error is expected from WHOLEARRAY, i.e. "device=npu1". -verify-diagnostics is thus disabled for WHOLEARRAY.
// expected-error@+1 {{'air.channel.put' op failed to map to shim dma channels: out of channels.}}
        %5 = air.channel.put async [%arg9]  @channel_2[%c2_0, %c0] (%arg7[%c2_0, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
        %6 = air.channel.put async [%arg9]  @channel_2[%c3, %c0] (%arg7[%c3, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 4 : i32} : (memref<512x512xbf16>)
        %7 = air.wait_all async [%3, %4, %5, %6] 
        scf.yield %7 : !air.async.token
      }
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c64_3 = arith.constant 64 : index
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %async_token_6, %results_7 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_8, %results_9 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %3 = air.wait_all async [%async_token_6, %async_token_8] 
        %4 = air.wait_all async [%async_token_6, %async_token_8] 
        %5:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %4, %arg10 = %async_token_8, %arg11 = %3, %arg12 = %3) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c0_4, %c0_4] (%results_7[] [] []) {id = 13 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_12[] (%results_7[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set, id = 14 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c0_4, %c0_4] (%results_9[] [] []) {id = 15 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_12[] (%results_9[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set, id = 16 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_10 = air.execute [%5#1] {
          memref.dealloc %results_7 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_11 = air.execute [%5#1] {
          memref.dealloc %results_9 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_12, %results_13 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_14, %results_15 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %6 = air.wait_all async [%async_token_12, %async_token_14] 
        %7 = air.wait_all async [%async_token_12, %async_token_14] 
        %8:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %7, %arg10 = %async_token_14, %arg11 = %6, %arg12 = %6) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c1_5, %c0_4] (%results_13[] [] []) {id = 17 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_13[] (%results_13[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set1, id = 18 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c1_5, %c0_4] (%results_15[] [] []) {id = 19 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_13[] (%results_15[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set1, id = 20 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_16 = air.execute [%8#1] {
          memref.dealloc %results_13 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%8#1] {
          memref.dealloc %results_15 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_20, %results_21 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %9 = air.wait_all async [%async_token_18, %async_token_20] 
        %10 = air.wait_all async [%async_token_18, %async_token_20] 
        %11:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %10, %arg10 = %async_token_20, %arg11 = %9, %arg12 = %9) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c2_2, %c0_4] (%results_19[] [] []) {id = 21 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_14[] (%results_19[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set2, id = 22 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c2_2, %c0_4] (%results_21[] [] []) {id = 23 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_14[] (%results_21[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set2, id = 24 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_22 = air.execute [%11#1] {
          memref.dealloc %results_19 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_23 = air.execute [%11#1] {
          memref.dealloc %results_21 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_26, %results_27 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %12 = air.wait_all async [%async_token_24, %async_token_26] 
        %13 = air.wait_all async [%async_token_24, %async_token_26] 
        %14:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %13, %arg10 = %async_token_26, %arg11 = %12, %arg12 = %12) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c3_1, %c0_4] (%results_25[] [] []) {id = 25 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_15[] (%results_25[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set3, id = 26 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c3_1, %c0_4] (%results_27[] [] []) {id = 27 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_15[] (%results_27[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set3, id = 28 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %15 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 5 : i32, link_with = "mm.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token_28, %results_29 = air.execute -> (memref<1x1x8x16x4x8xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>
          }
          %16 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
            %17 = air.channel.get async [%async_token_28]  @channel_12[%arg8, %arg9] (%results_29[] [] []) {id = 61 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
            affine.yield %17 : !air.async.token
          } else {
            %17 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
              %18 = air.channel.get async [%async_token_28]  @channel_13[%arg8, %arg9] (%results_29[] [] []) {id = 62 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
              affine.yield %18 : !air.async.token
            } else {
              %18 = affine.if #set2()[%arg8, %arg9] -> !air.async.token {
                %19 = air.channel.get async [%async_token_28]  @channel_14[%arg8, %arg9] (%results_29[] [] []) {id = 63 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                affine.yield %19 : !air.async.token
              } else {
                %19 = air.channel.get async [%async_token_28]  @channel_15[%arg8, %arg9] (%results_29[] [] []) {id = 64 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                affine.yield %19 : !air.async.token
              }
              affine.yield %18 : !air.async.token
            }
            affine.yield %17 : !air.async.token
          }
        }
      }
    }
    return
  }
}
