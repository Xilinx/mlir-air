//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_3 [2, 2]
  air.channel @channel_2 [2, 2]
  air.channel @channel_1 [2, 2]
  air.channel @channel_0 [2, 2]
  func.func @forward(%arg0: memref<64x128xi32>, %arg1: memref<128x64xi32>, %arg2: memref<64x64xi32>) {
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %async_token, %results = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%c0_i32 : i32) outs(%results : memref<64x64xi32>)
    }
    %async_token_1, %results_2 = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<64x64xi32> to memref<64x64xi32>
    }
    %0 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_3) -> !air.async.token {
      %async_token_5, %results_6 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %async_token_7, %results_8 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %7 = air.channel.put async [%async_token_7, %async_token_5, %async_token_3]  @channel_0[%arg3, %arg4] (%results_2[%results_6, %results_8] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<64x64xi32>)
      scf.reduce(%7 : !air.async.token) {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %8 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %8 : !air.async.token
      }
    }
    %1 = air.wait_all async 
    %2 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%1) -> !air.async.token {
      %async_token_5, %results_6 = air.execute -> (index) {
        %9 = affine.apply #map()[%arg3]
        air.execute_terminator %9 : index
      }
      %8 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %async_token_5) -> (!air.async.token) {
        %9 = air.channel.put async [%arg6]  @channel_1[%arg3, %arg4] (%arg0[%results_6, %arg5] [%c32, %c32] [%c128, %c1]) {id = 2 : i32} : (memref<64x128xi32>)
        scf.yield %9 : !air.async.token
      }
      scf.reduce(%8 : !air.async.token) {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %9 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %9 : !air.async.token
      }
    }
    %3 = air.wait_all async 
    %4 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%3) -> !air.async.token {
      %async_token_5, %results_6 = air.execute -> (index) {
        %9 = affine.apply #map()[%arg4]
        air.execute_terminator %9 : index
      }
      %8 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %async_token_5) -> (!air.async.token) {
        %9 = air.channel.put async [%arg6]  @channel_2[%arg3, %arg4] (%arg1[%arg5, %results_6] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<128x64xi32>)
        scf.yield %9 : !air.async.token
      }
      scf.reduce(%8 : !air.async.token) {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %9 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %9 : !air.async.token
      }
    }
    %5 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_3) -> !air.async.token {
      %async_token_5, %results_6 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %async_token_7, %results_8 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %7 = air.channel.get async [%async_token_7, %async_token_5, %async_token_3]  @channel_3[%arg3, %arg4] (%results_2[%results_6, %results_8] [%c32, %c32] [%c64, %c1]) {id = 4 : i32} : (memref<64x64xi32>)
      scf.reduce(%7 : !air.async.token) {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %8 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %8 : !air.async.token
      }
    }
    %6 = air.herd @herd_0 async [%async_token_3]  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) attributes {id = 1 : i32} {
      %c64_5 = arith.constant 64 : index
      %c0_6 = arith.constant 0 : index
      %c128_7 = arith.constant 128 : index
      %7 = air.wait_all async 
      %async_token_8, %results_9 = air.execute -> (memref<32x32xi32, 2>) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.execute_terminator %alloc : memref<32x32xi32, 2>
      }
      %8 = air.channel.get async [%async_token_8, %7]  @channel_0[%arg3, %arg4] (%results_9[] [] []) {id = 5 : i32} : (memref<32x32xi32, 2>)
      %async_token_10, %results_11 = air.execute [%8] -> (memref<32x32xi32, 2>) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.execute_terminator %alloc : memref<32x32xi32, 2>
      }
      %async_token_12, %results_13 = air.execute [%async_token_10] -> (memref<32x32xi32, 2>) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.execute_terminator %alloc : memref<32x32xi32, 2>
      }
      %async_token_14, %results_15 = air.execute [%async_token_12] -> (memref<32x32xi32, 2>) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.execute_terminator %alloc : memref<32x32xi32, 2>
      }
      %async_token_16, %results_17 = air.execute [%async_token_12] -> (memref<32x32xi32, 2>) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.execute_terminator %alloc : memref<32x32xi32, 2>
      }
      %9:4 = scf.for %arg7 = %c0_6 to %c128_7 step %c64_5 iter_args(%arg8 = %async_token_14, %arg9 = %async_token_16, %arg10 = %async_token_16, %arg11 = %async_token_16) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
        %11 = air.channel.get async [%arg11, %async_token_14, %arg8]  @channel_1[%arg3, %arg4] (%results_15[] [] []) {id = 6 : i32} : (memref<32x32xi32, 2>)
        %12 = air.channel.get async [%arg11, %async_token_16, %arg8]  @channel_2[%arg3, %arg4] (%results_17[] [] []) {id = 7 : i32} : (memref<32x32xi32, 2>)
        %async_token_19 = air.execute [%arg10, %12, %11] {
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_15, %results_17 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_9 : memref<32x32xi32, 2>)
        }
        %async_token_20 = air.execute {
          memref.dealloc %results_15 : memref<32x32xi32, 2>
        }
        %async_token_21 = air.execute {
          memref.dealloc %results_17 : memref<32x32xi32, 2>
        }
        %13 = air.channel.get async [%12, %11, %arg9]  @channel_1[%arg3, %arg4] (%results_13[] [] []) {id = 6 : i32} : (memref<32x32xi32, 2>)
        %14 = air.channel.get async [%12, %11, %arg9]  @channel_2[%arg3, %arg4] (%results_11[] [] []) {id = 7 : i32} : (memref<32x32xi32, 2>)
        %async_token_22 = air.execute [%async_token_19, %14, %13] {
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_13, %results_11 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_9 : memref<32x32xi32, 2>)
        }
        %async_token_23 = air.execute {
          memref.dealloc %results_13 : memref<32x32xi32, 2>
        }
        %async_token_24 = air.execute {
          memref.dealloc %results_11 : memref<32x32xi32, 2>
        }
        %15 = air.wait_all async [%13, %14] 
        scf.yield %async_token_19, %async_token_22, %async_token_22, %15 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
      }
      %10 = air.channel.put async [%9#1]  @channel_3[%arg3, %arg4] (%results_9[] [] []) {id = 8 : i32} : (memref<32x32xi32, 2>)
      %async_token_18 = air.execute [%10] {
        memref.dealloc %results_9 : memref<32x32xi32, 2>
      }
      air.herd_terminator
    }
    %async_token_4 = air.execute [%async_token_3] {
      memref.copy %results_2, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    }
    return
  }
}
