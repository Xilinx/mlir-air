//===- async_one_core_gemm_to_ipu.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-fuse-channels -air-to-aie="row-offset=2 col-offset=0 device=ipu" -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL:   AIE.device(ipu) {
// CHECK:  %[[VAL_0:.*]] = AIE.tile(0, 0)
// CHECK:  %[[VAL_1:.*]] = AIE.tile(0, 1)
// CHECK:  %[[VAL_2:.*]] = AIE.tile(0, 2)
// CHECK:  %[[VAL_3:.*]] = AIE.lock(%[[VAL_1]], 7) {init = 1 : i32}
// CHECK:  %[[VAL_4:.*]] = AIE.lock(%[[VAL_1]], 6) {init = 0 : i32}
// CHECK:  %[[VAL_5:.*]] = AIE.lock(%[[VAL_1]], 5) {init = 1 : i32}
// CHECK:  %[[VAL_6:.*]] = AIE.lock(%[[VAL_1]], 4) {init = 0 : i32}
// CHECK:  %[[VAL_7:.*]] = AIE.lock(%[[VAL_1]], 3) {init = 1 : i32}
// CHECK:  %[[VAL_8:.*]] = AIE.lock(%[[VAL_1]], 2) {init = 0 : i32}
// CHECK:  %[[VAL_9:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_10:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_15:.*]] = AIE.lock(%[[VAL_2]], 3) {init = 3 : i32}
// CHECK:  %[[VAL_16:.*]] = AIE.lock(%[[VAL_2]], 2) {init = 0 : i32}
// CHECK:  %[[VAL_17:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_18:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_19:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "buf6"} : memref<32x32xi32, 1>
// CHECK:  %[[VAL_20:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "buf5"} : memref<32x32xi32, 1>
// CHECK:  %[[VAL_21:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "buf4"} : memref<32x32xi32, 1>
// CHECK:  %[[VAL_22:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "buf3"} : memref<32x32xi32, 1>
// CHECK:  %[[VAL_23:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "buf2"} : memref<32x32xi32, 2>
// CHECK:  %[[VAL_24:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "buf1"} : memref<32x32xi32, 2>
// CHECK:  %[[VAL_25:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "buf0"} : memref<32x32xi32, 2>
// CHECK:  %[[VAL_26:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:  %[[VAL_27:.*]] = AIE.core(%[[VAL_2]]) {
// CHECK:  AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:  AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:  AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_2]], DMA : 0)
// CHECK:  AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:  %[[VAL_28:.*]] = AIE.memTileDMA(%[[VAL_1]]) {

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
          air.herd_terminator
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
        air.segment_terminator
      }
      air.launch_terminator
    }
    %async_token_4 = air.execute [%0] {
      memref.copy %results_2, %arg2 : memref<32x32xi32> to memref<32x32xi32>
    }
    return
  }
}
