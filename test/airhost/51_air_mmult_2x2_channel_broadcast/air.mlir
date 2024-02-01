//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
module {
  air.channel @channel_4 [2, 2]
  air.channel @channel_3 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  func.func @forward(%arg0: memref<64x128xi32>, %arg1: memref<128x64xi32>, %arg2: memref<64x64xi32>) {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
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
    %0 = air.wait_all async 
    %1 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %0) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_0[] (%arg0[%c0, %arg3] [%c32, %c32] [%c128, %c1]) {id = 1 : i32} : (memref<64x128xi32>)
      scf.yield %10 : !air.async.token
    }
    %2 = air.wait_all async 
    %3 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %2) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_1[] (%arg0[%c32, %arg3] [%c32, %c32] [%c128, %c1]) {id = 2 : i32} : (memref<64x128xi32>)
      scf.yield %10 : !air.async.token
    }
    %4 = air.wait_all async 
    %5 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %4) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_2[] (%arg1[%arg3, %c0] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<128x64xi32>)
      scf.yield %10 : !air.async.token
    }
    %6 = air.wait_all async 
    %7 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %6) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_3[] (%arg1[%arg3, %c32] [%c32, %c32] [%c64, %c1]) {id = 4 : i32} : (memref<128x64xi32>)
      scf.yield %10 : !air.async.token
    }
    %8 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_3) -> !air.async.token {
      %async_token_5, %results_6 = air.execute -> (index) {
        %11 = affine.apply #map()[%arg3]
        air.execute_terminator %11 : index
      }
      %async_token_7, %results_8 = air.execute -> (index) {
        %11 = affine.apply #map()[%arg4]
        air.execute_terminator %11 : index
      }
      %10 = air.channel.get async [%async_token_7, %async_token_5, %async_token_3]  @channel_4[%arg3, %arg4] (%results_2[%results_6, %results_8] [%c32, %c32] [%c64, %c1]) {id = 5 : i32} : (memref<64x64xi32>)
      scf.reduce(%10 : !air.async.token) {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %11 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %11 : !air.async.token
      }
    }
    %9 = air.herd @herd_0 async [%async_token_3]  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) attributes {id = 1 : i32} {
      %c0_5 = arith.constant 0 : index
      %c128_6 = arith.constant 128 : index
      %c32_7 = arith.constant 32 : index
      %c0_i32_8 = arith.constant 0 : i32
      %10 = air.wait_all async 
      %async_token_9, %results_10 = air.execute -> (memref<32x32xi32, 2>) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.execute_terminator %alloc : memref<32x32xi32, 2>
      }
      %async_token_11 = air.execute [%async_token_9, %10] {
        linalg.fill ins(%c0_i32_8 : i32) outs(%results_10 : memref<32x32xi32, 2>)
      }
      %11 = scf.for %arg7 = %c0_5 to %c128_6 step %c32_7 iter_args(%arg8 = %async_token_11) -> (!air.async.token) {
        %async_token_13, %results_14 = air.execute -> (memref<32x32xi32, 2>) {
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.execute_terminator %alloc : memref<32x32xi32, 2>
        }
        %async_token_15, %results_16 = air.execute -> (memref<32x32xi32, 2>) {
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.execute_terminator %alloc : memref<32x32xi32, 2>
        }
        %13 = affine.if #set()[%arg3, %arg4] -> !air.async.token {
          %15 = air.channel.get async [%async_token_13, %arg8]  @channel_0[%arg3, %arg4] (%results_14[] [] []) {id = 6 : i32} : (memref<32x32xi32, 2>)
          affine.yield %15 : !air.async.token
        } else {
          %15 = air.channel.get async [%async_token_13, %arg8]  @channel_1[%arg3, %arg4] (%results_14[] [] []) {id = 7 : i32} : (memref<32x32xi32, 2>)
          affine.yield %15 : !air.async.token
        }
        %14 = affine.if #set1()[%arg3, %arg4] -> !air.async.token {
          %15 = air.channel.get async [%async_token_15, %arg8]  @channel_2[%arg3, %arg4] (%results_16[] [] []) {id = 8 : i32} : (memref<32x32xi32, 2>)
          affine.yield %15 : !air.async.token
        } else {
          %15 = air.channel.get async [%async_token_15, %arg8]  @channel_3[%arg3, %arg4] (%results_16[] [] []) {id = 9 : i32} : (memref<32x32xi32, 2>)
          affine.yield %15 : !air.async.token
        }
        %async_token_17 = air.execute [%14, %13] {
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_14, %results_16 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_10 : memref<32x32xi32, 2>)
        }
        %async_token_18 = air.execute [%async_token_17] {
          memref.dealloc %results_14 : memref<32x32xi32, 2>
        }
        %async_token_19 = air.execute [%async_token_17] {
          memref.dealloc %results_16 : memref<32x32xi32, 2>
        }
        scf.yield %async_token_17 : !air.async.token
      }
      %12 = air.channel.put async [%11]  @channel_4[%arg3, %arg4] (%results_10[] [] []) {id = 10 : i32} : (memref<32x32xi32, 2>)
      %async_token_12 = air.execute [%12] {
        memref.dealloc %results_10 : memref<32x32xi32, 2>
      }
      air.herd_terminator
    }
    %async_token_4 = air.execute [%async_token_3] {
      memref.copy %results_2, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    }
    return
  }
}
