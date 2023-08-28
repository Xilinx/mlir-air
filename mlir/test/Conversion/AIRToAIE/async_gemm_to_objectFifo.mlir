//===- async_gemm_to_objectFifo.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-aie="emit-while-loop=false use-objectfifo=true row-offset=3 col-offset=5 device=xcvc1902" %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(5, 3)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(6, 3)
// CHECK:   %[[VAL_2:.*]] = AIE.tile(5, 4)
// CHECK:   %[[VAL_3:.*]] = AIE.tile(6, 4)
// CHECK:   %[[VAL_4:.*]] = AIE.buffer(%[[VAL_3]]){{.*}}memref<64x96xbf16, 2>
// CHECK:   %[[VAL_5:.*]] = AIE.buffer(%[[VAL_3]]){{.*}}memref<96x64xbf16, 2>
// CHECK:   %[[VAL_6:.*]] = AIE.buffer(%[[VAL_3]]){{.*}}memref<64x64xbf16, 2>
// CHECK:   %[[VAL_7:.*]] = AIE.buffer(%[[VAL_2]]){{.*}}memref<64x96xbf16, 2>
// CHECK:   %[[VAL_8:.*]] = AIE.buffer(%[[VAL_2]]){{.*}}memref<96x64xbf16, 2>
// CHECK:   %[[VAL_9:.*]] = AIE.buffer(%[[VAL_2]]){{.*}}memref<64x64xbf16, 2>
// CHECK:   %[[VAL_10:.*]] = AIE.buffer(%[[VAL_1]]){{.*}}memref<64x96xbf16, 2>
// CHECK:   %[[VAL_11:.*]] = AIE.buffer(%[[VAL_1]]){{.*}}memref<96x64xbf16, 2>
// CHECK:   %[[VAL_12:.*]] = AIE.buffer(%[[VAL_1]]){{.*}}memref<64x64xbf16, 2>
// CHECK:   %[[VAL_13:.*]] = AIE.buffer(%[[VAL_0]]){{.*}}memref<64x96xbf16, 2>
// CHECK:   %[[VAL_14:.*]] = AIE.buffer(%[[VAL_0]]){{.*}}memref<96x64xbf16, 2>
// CHECK:   %[[VAL_15:.*]] = AIE.buffer(%[[VAL_0]]){{.*}}memref<64x64xbf16, 2>
// CHECK-COUNT-16:    AIE.objectFifo @
// CHECK:   %[[VAL_16:.*]] = AIE.core(%[[VAL_3]]) {
// CHECK:   %[[VAL_17:.*]] = AIE.core(%[[VAL_2]]) {
// CHECK:   %[[VAL_18:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:   %[[VAL_19:.*]] = AIE.core(%[[VAL_0]]) {

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  air.channel @channel_3 [2, 2]
  air.channel @channel_2 [2, 2]
  air.channel @channel_1 [2, 2]
  air.channel @channel_0 [2, 2]
  func.func @matmul(%arg0: memref<128x384xbf16>, %arg1: memref<384x128xbf16>, %arg2: memref<128x128xbf16>) {
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %async_token, %results = air.execute -> (memref<128x128xbf16>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xbf16>
      air.execute_terminator %alloc : memref<128x128xbf16>
    }
    %async_token_0 = air.execute [%async_token] {
      memref.copy %arg2, %results : memref<128x128xbf16> to memref<128x128xbf16>
    }
    %0 = air.wait_all async 
    %1 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
      %async_token_1, %results_2 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %7 = scf.for %arg5 = %c0 to %c384 step %c96 iter_args(%arg6 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.put async [%arg6]  @channel_0[%arg3, %arg4] (%arg0[%results_2, %arg5] [%c64, %c96] [%c384, %c1]) {id = 1 : i32} : (memref<128x384xbf16>)
        scf.yield %8 : !air.async.token
      }
      scf.reduce(%7)  : !air.async.token {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %8 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %8 : !air.async.token
      }
      scf.yield
    }
    %2 = air.wait_all async 
    %3 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%2) -> !air.async.token {
      %async_token_1, %results_2 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %7 = scf.for %arg5 = %c0 to %c384 step %c96 iter_args(%arg6 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.put async [%arg6]  @channel_1[%arg3, %arg4] (%arg1[%arg5, %results_2] [%c96, %c64] [%c128, %c1]) {id = 2 : i32} : (memref<384x128xbf16>)
        scf.yield %8 : !air.async.token
      }
      scf.reduce(%7)  : !air.async.token {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %8 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %8 : !air.async.token
      }
      scf.yield
    }
    %4 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_0) -> !air.async.token {
      %async_token_1, %results_2 = air.execute -> (index) {
        %9 = affine.apply #map()[%arg3]
        air.execute_terminator %9 : index
      }
      %async_token_3, %results_4 = air.execute -> (index) {
        %9 = affine.apply #map()[%arg4]
        air.execute_terminator %9 : index
      }
      %7 = air.wait_all async [%async_token_3, %async_token_1, %async_token_0] 
      %8 = scf.for %arg5 = %c0 to %c384 step %c96 iter_args(%arg6 = %7) -> (!air.async.token) {
        %9 = air.channel.put async [%arg6]  @channel_2[%arg3, %arg4] (%results[%results_2, %results_4] [%c64, %c64] [%c128, %c1]) {id = 3 : i32} : (memref<128x128xbf16>)
        scf.yield %9 : !air.async.token
      }
      scf.reduce(%8)  : !air.async.token {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %9 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %9 : !air.async.token
      }
      scf.yield
    }
    %5 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_0) -> !air.async.token {
      %async_token_1, %results_2 = air.execute -> (index) {
        %9 = affine.apply #map()[%arg3]
        air.execute_terminator %9 : index
      }
      %async_token_3, %results_4 = air.execute -> (index) {
        %9 = affine.apply #map()[%arg4]
        air.execute_terminator %9 : index
      }
      %7 = air.wait_all async [%async_token_3, %async_token_1, %async_token_0] 
      %8 = scf.for %arg5 = %c0 to %c384 step %c96 iter_args(%arg6 = %7) -> (!air.async.token) {
        %9 = air.channel.get async [%arg6]  @channel_3[%arg3, %arg4] (%results[%results_2, %results_4] [%c64, %c64] [%c128, %c1]) {id = 4 : i32} : (memref<128x128xbf16>)
        scf.yield %9 : !air.async.token
      }
      scf.reduce(%8)  : !air.async.token {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %9 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %9 : !air.async.token
      }
      scf.yield
    }
    %6 = air.herd @herd_0 async [%async_token_0]  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) attributes {id = 1 : i32} {
      %c0_1 = arith.constant 0 : index
      %c384_2 = arith.constant 384 : index
      %c96_3 = arith.constant 96 : index
      %7 = air.wait_all async 
      %8 = scf.for %arg7 = %c0_1 to %c384_2 step %c96_3 iter_args(%arg8 = %7) -> (!air.async.token) {
        %async_token_4, %results_5 = air.execute -> (memref<64x96xbf16, 2>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<64x96xbf16, 2>
          air.execute_terminator %alloc : memref<64x96xbf16, 2>
        }
        %async_token_6, %results_7 = air.execute -> (memref<96x64xbf16, 2>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<96x64xbf16, 2>
          air.execute_terminator %alloc : memref<96x64xbf16, 2>
        }
        %async_token_8, %results_9 = air.execute -> (memref<64x64xbf16, 2>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<64x64xbf16, 2>
          air.execute_terminator %alloc : memref<64x64xbf16, 2>
        }
        %9 = air.channel.get async [%async_token_4, %arg8]  @channel_0[%arg3, %arg4] (%results_5[] [] []) {id = 5 : i32} : (memref<64x96xbf16, 2>)
        %10 = air.channel.get async [%async_token_6, %arg8]  @channel_1[%arg3, %arg4] (%results_7[] [] []) {id = 6 : i32} : (memref<96x64xbf16, 2>)
        %11 = air.channel.get async [%async_token_8, %arg8]  @channel_2[%arg3, %arg4] (%results_9[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 2>)
        %async_token_10 = air.execute [%11, %10, %9] {
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_5, %results_7 : memref<64x96xbf16, 2>, memref<96x64xbf16, 2>) outs(%results_9 : memref<64x64xbf16, 2>)
        }
        %12 = air.channel.put async [%async_token_10]  @channel_3[%arg3, %arg4] (%results_9[] [] []) {id = 8 : i32} : (memref<64x64xbf16, 2>)
        %async_token_11 = air.execute [%async_token_10] {
          memref.dealloc %results_5 : memref<64x96xbf16, 2>
        }
        %async_token_12 = air.execute [%async_token_10] {
          memref.dealloc %results_7 : memref<96x64xbf16, 2>
        }
        %async_token_13 = air.execute [%12] {
          memref.dealloc %results_9 : memref<64x64xbf16, 2>
        }
        scf.yield %12 : !air.async.token
      } {unroll = 2 : i32}
      air.herd_terminator
    }
    return
  }
}
