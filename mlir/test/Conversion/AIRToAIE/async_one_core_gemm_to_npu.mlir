//===- async_one_core_gemm_to_npu.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-fuse-channels="aggressive-mode=L1,L2,L3" -air-to-aie="row-offset=2 col-offset=0 device=npu1_1col" -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) @segment_0 {
// CHECK-DAG:  %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(0, ?)
// CHECK-DAG:  %[[COMPUTE:.*]] = aie.tile(0, 2)
// CHECK-DAG:  %[[CLOCK_3P:.*]] = aie.lock(%[[COMPUTE]], 3) {init = 3 : i32}
// CHECK-DAG:  %[[CLOCK_3C:.*]] = aie.lock(%[[COMPUTE]], 2) {init = 0 : i32}
// CHECK-DAG:  %[[CLOCK_2P:.*]] = aie.lock(%[[COMPUTE]], 1) {init = 1 : i32}
// CHECK-DAG:  %[[CLOCK_2C:.*]] = aie.lock(%[[COMPUTE]], 0) {init = 0 : i32}
// CHECK-DAG:  aie.buffer(%[[COMPUTE]]) {{{.*}}} : memref<32x32xi32, 2>
// CHECK-DAG:  aie.buffer(%[[COMPUTE]]) {{{.*}}} : memref<32x32xi32, 2>
// CHECK-DAG:  aie.buffer(%[[COMPUTE]]) {{{.*}}} : memref<32x32xi32, 2>
// CHECK:  aie.mem(%[[COMPUTE]]) {
// CHECK:  aie.core(%[[COMPUTE]]) {
// CHECK-DAG:  %[[MEMTILE:.*]] = aie.logical_tile<MemTile>(0, ?)
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 7) {init = 1 : i32}
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 6) {init = 0 : i32}
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 5) {init = 1 : i32}
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 4) {init = 0 : i32}
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 3) {init = 1 : i32}
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 2) {init = 0 : i32}
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 1) {init = 1 : i32}
// CHECK-DAG:  aie.lock(%[[MEMTILE]], 0) {init = 0 : i32}
// CHECK-DAG:  aie.buffer(%[[MEMTILE]]) {{{.*}}} : memref<32x32xi32, 1>
// CHECK-DAG:  aie.buffer(%[[MEMTILE]]) {{{.*}}} : memref<32x32xi32, 1>
// CHECK-DAG:  aie.buffer(%[[MEMTILE]]) {{{.*}}} : memref<32x32xi32, 1>
// CHECK-DAG:  aie.buffer(%[[MEMTILE]]) {{{.*}}} : memref<32x32xi32, 1>
// CHECK:  aie.flow(%[[SHIM]], DMA : 0, %[[MEMTILE]], DMA : 0)
// CHECK:  aie.flow(%[[MEMTILE]], DMA : 0, %[[SHIM]], DMA : 0)
// CHECK:  aie.flow(%[[MEMTILE]], DMA : 1, %[[COMPUTE]], DMA : 0)
// CHECK:  aie.flow(%[[COMPUTE]], DMA : 0, %[[MEMTILE]], DMA : 1)
// CHECK:  aie.memtile_dma(%[[MEMTILE]]) {

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_7 [1, 1]
  air.channel @channel_6 [1, 1]
  air.channel @channel_5 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_3 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @forward(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>) {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %async_token, %results = air.execute -> (memref<32x32xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
      air.execute_terminator %alloc : memref<32x32xi32>
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%c0_i32 : i32) outs(%results : memref<32x32xi32>)
    }
    %async_token_1, %results_2 = air.execute -> (memref<32x32xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
      air.execute_terminator %alloc : memref<32x32xi32>
    }
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<32x32xi32> to memref<32x32xi32>
    }
    %0 = air.launch async [%async_token_3] (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%results_2) : memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %async_token_6, %results_7 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %async_token_8, %results_9 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %1 = air.channel.put async [%async_token_8, %async_token_6]  @channel_0[] (%arg9[%results_7, %results_9] [%c32, %c32] [%c32, %c1_5]) {id = 1 : i32} : (memref<32x32xi32>)
      %async_token_10, %results_11 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %2 = air.wait_all async 
      %3 = air.channel.put async [%2, %async_token_10]  @channel_1[] (%arg7[%results_11, %c0] [%c32, %c32] [%c32, %c1_5]) {id = 2 : i32} : (memref<32x32xi32>)
      %async_token_12, %results_13 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %4 = air.wait_all async 
      %5 = air.channel.put async [%4, %async_token_12]  @channel_2[] (%arg8[%c0, %results_13] [%c32, %c32] [%c32, %c1_5]) {id = 3 : i32} : (memref<32x32xi32>)
      %async_token_14, %results_15 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %async_token_16, %results_17 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %6 = air.channel.get async [%async_token_16, %async_token_14]  @channel_7[] (%arg9[%results_15, %results_17] [%c32, %c32] [%c32, %c1_5]) {id = 4 : i32} : (memref<32x32xi32>)
      %7 = air.segment async  attributes {id = 2 : i32} {
        %c0_18 = arith.constant 0 : index
        %c1_19 = arith.constant 1 : index
        %c32_20 = arith.constant 32 : index
        %8 = air.wait_all async 
        %async_token_21, %results_22 = air.execute -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %async_token_23, %results_24 = air.execute -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %9 = air.channel.get async [%async_token_21, %async_token_23, %8]  @channel_0[] (%results_22[] [] []) {id = 5 : i32} : (memref<32x32xi32, 1>)
        %async_token_25, %results_26 = air.execute -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %async_token_27, %results_28 = air.execute -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %10 = air.channel.get async [%async_token_25, %9]  @channel_1[] (%results_26[] [] []) {id = 6 : i32} : (memref<32x32xi32, 1>)
        %11 = air.channel.get async [%async_token_27, %9]  @channel_2[] (%results_28[] [] []) {id = 7 : i32} : (memref<32x32xi32, 1>)
        %12 = air.wait_all async 
        %13 = air.wait_all async 
        %14 = air.channel.put async [%async_token_21, %13, %12, %9]  @channel_3[%c0_18, %c0_18] (%results_22[%c0_18, %c0_18] [%c32_20, %c32_20] [%c32_20, %c1_19]) {id = 8 : i32} : (memref<32x32xi32, 1>)
        %15 = air.wait_all async 
        %16 = air.channel.put async [%15, %10]  @channel_4[%c0_18, %c0_18] (%results_26[%c0_18, %c0_18] [%c32_20, %c32_20] [%c32_20, %c1_19]) {id = 9 : i32} : (memref<32x32xi32, 1>)
        %17 = air.wait_all async 
        %18 = air.channel.put async [%17, %11]  @channel_5[%c0_18, %c0_18] (%results_28[%c0_18, %c0_18] [%c32_20, %c32_20] [%c32_20, %c1_19]) {id = 10 : i32} : (memref<32x32xi32, 1>)
        %19 = air.wait_all async 
        %20 = air.wait_all async 
        %21 = air.channel.get async [%20, %19, %9]  @channel_6[%c0_18, %c0_18] (%results_24[%c0_18, %c0_18] [%c32_20, %c32_20] [%c32_20, %c1_19]) {id = 11 : i32} : (memref<32x32xi32, 1>)
        %22 = air.herd @herd_0 async [%11, %10]  tile (%arg10, %arg11) in (%arg12=%c1_19, %arg13=%c1_19) attributes {id = 3 : i32} {
          %25 = air.wait_all async 
          %async_token_33, %results_34 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %26 = air.channel.get async [%async_token_33, %25]  @channel_3[%arg10, %arg11] (%results_34[] [] []) {id = 12 : i32} : (memref<32x32xi32, 2>)
          %async_token_35, %results_36 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %async_token_37, %results_38 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %27 = air.channel.get async [%async_token_35, %26]  @channel_4[%arg10, %arg11] (%results_36[] [] []) {id = 13 : i32} : (memref<32x32xi32, 2>)
          %28 = air.channel.get async [%async_token_37, %26]  @channel_5[%arg10, %arg11] (%results_38[] [] []) {id = 14 : i32} : (memref<32x32xi32, 2>)
          %async_token_39 = air.execute [%28, %27] {
            linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_36, %results_38 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_34 : memref<32x32xi32, 2>)
          }
          %async_token_40 = air.execute [%async_token_39] {
            memref.dealloc %results_36 : memref<32x32xi32, 2>
          }
          %async_token_41 = air.execute [%async_token_39] {
            memref.dealloc %results_38 : memref<32x32xi32, 2>
          }
          %29 = air.channel.put async [%async_token_39]  @channel_6[%arg10, %arg11] (%results_34[] [] []) {id = 15 : i32} : (memref<32x32xi32, 2>)
          %async_token_42 = air.execute [%29] {
            memref.dealloc %results_34 : memref<32x32xi32, 2>
          }
        }
        %async_token_29 = air.execute [%16] {
          memref.dealloc %results_26 : memref<32x32xi32, 1>
        }
        %async_token_30 = air.execute [%18] {
          memref.dealloc %results_28 : memref<32x32xi32, 1>
        }
        %23 = air.wait_all async [%21, %14, %18, %22, %16] 
        %24 = air.channel.put async [%23]  @channel_7[] (%results_24[] [] []) {id = 16 : i32} : (memref<32x32xi32, 1>)
        %async_token_31 = air.execute {
          memref.dealloc %results_22 : memref<32x32xi32, 1>
        }
        %async_token_32 = air.execute [%24] {
          memref.dealloc %results_24 : memref<32x32xi32, 1>
        }
      }
    }
    %async_token_4 = air.execute [%0] {
      memref.copy %results_2, %arg2 : memref<32x32xi32> to memref<32x32xi32>
    }
    return
  }
}
