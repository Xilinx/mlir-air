//===- broadcast_hoist_affine_if.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that air-dma-to-channel correctly handles broadcast DMAs wrapped in
// affine.if when the external channel.put needs to be hoisted outside the
// herd. This is the L3->L1 broadcast case (no segment intermediate layer)
// where the affine.if result token must be properly mapped during hoisting.
// Regression test for GitHub issue #1484.

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse | FileCheck %s

// CHECK: air.channel @channel_{{.*}} [1, 1] {broadcast_shape = [2, 1]}
// CHECK: air.channel @channel_{{.*}} [2, 1]
// CHECK: air.channel @channel_{{.*}} [2, 1]
// CHECK: air.channel.put{{.*}}@channel_{{.*}}[]
// CHECK: air.herd

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
#map = affine_map<()[s0, s1] -> ((s0 + s1 * 64) * 64)>
module {
  func.func @scale(%arg0: memref<8192xbf16>, %arg1: memref<64xbf16>, %arg2: memref<8192xbf16>) {
    %0 = air.launch async () in () args(%arg3=%arg0, %arg4=%arg1, %arg5=%arg2) : memref<8192xbf16>, memref<64xbf16>, memref<8192xbf16> attributes {id = 3 : i32} {
      %1 = air.segment @seg async  args(%arg6=%arg3, %arg7=%arg4, %arg8=%arg5) : memref<8192xbf16>, memref<64xbf16>, memref<8192xbf16> attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %2 = air.herd @herd async  tile (%arg9, %arg10) in (%arg11=%c2, %arg12=%c1) args(%arg13=%arg6, %arg14=%arg7, %arg15=%arg8) : memref<8192xbf16>, memref<64xbf16>, memref<8192xbf16> attributes {id = 1 : i32} {
          %c16 = arith.constant 16 : index
          %c1_0 = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %cst = arith.constant 0.000000e+00 : bf16
          %c0 = arith.constant 0 : index
          %async_token, %results = air.execute -> (memref<64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
          } {id = 1 : i32}
          %async_token_1, %results_2 = air.execute -> (memref<64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
          } {id = 2 : i32}
          %async_token_3, %results_4 = air.execute -> (memref<64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
          } {id = 3 : i32}
          // Broadcast DMA wrapped in affine.if — this is the pattern that
          // caused SSA dominance error before the fix.
          %3 = affine.if #set()[%arg9, %arg10] -> !air.async.token {
            %6 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg14[] [] []) {broadcast_set = #set, id = 1 : i32} : (memref<64xbf16, 2 : i32>, memref<64xbf16>)
            affine.yield %6 : !air.async.token
          } else {
            %6 = air.wait_all async [%async_token]
            affine.yield %6 : !air.async.token
          }
          // %3 (affine.if result) is used by wait_all and execute below.
          %4 = air.wait_all async [%async_token_1, %async_token_3, %3]  {id = 3 : i32}
          %5 = scf.for %arg16 = %c0 to %c64 step %c1_0 iter_args(%arg17 = %4) -> (!air.async.token) {
            %6 = affine.apply #map()[%arg16, %arg9]
            %7 = air.dma_memcpy_nd async [%arg17] (%results_2[] [] [], %arg13[%6] [%c64] [%c1_0]) {id = 2 : i32} : (memref<64xbf16, 2 : i32>, memref<8192xbf16>)
            %8 = air.wait_all async [%7, %arg17, %arg17]  {id = 1 : i32}
            %9 = scf.for %arg18 = %c0 to %c64 step %c16 iter_args(%arg19 = %8) -> (!air.async.token) {
              %subview = memref.subview %results_2[%arg18] [16] [1] : memref<64xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              %subview_8 = memref.subview %results[%arg18] [16] [1] : memref<64xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              %subview_9 = memref.subview %results_4[%arg18] [16] [1] : memref<64xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              %async_token_10, %results_11 = air.execute [%arg19] -> (vector<16xbf16>) {
                %14 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
                air.execute_terminator %14 : vector<16xbf16>
              } {id = 4 : i32}
              %async_token_12, %results_13 = air.execute [%arg19] -> (vector<16xbf16>) {
                %14 = vector.transfer_read %subview_8[%c0], %cst {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
                air.execute_terminator %14 : vector<16xbf16>
              } {id = 5 : i32}
              %12 = arith.mulf %results_11, %results_13 : vector<16xbf16>
              %async_token_14 = air.execute [%arg19] {
                vector.transfer_write %12, %subview_9[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              } {id = 6 : i32}
              %13 = air.wait_all async [%async_token_10, %async_token_12, %async_token_14]  {id = 2 : i32}
              scf.yield %13 : !air.async.token
            }
            %10 = air.dma_memcpy_nd async [%arg17] (%arg15[%6] [%c64] [%c1_0], %results_4[] [] []) {id = 3 : i32} : (memref<8192xbf16>, memref<64xbf16, 2 : i32>)
            %11 = air.wait_all async [%9, %10]  {id = 4 : i32}
            scf.yield %11 : !air.async.token
          }
          %async_token_5 = air.execute [%3] {
            memref.dealloc %results : memref<64xbf16, 2 : i32>
          } {id = 7 : i32}
          %async_token_6 = air.execute [%5] {
            memref.dealloc %results_2 : memref<64xbf16, 2 : i32>
          } {id = 8 : i32}
          %async_token_7 = air.execute [%5] {
            memref.dealloc %results_4 : memref<64xbf16, 2 : i32>
          } {id = 9 : i32}
        }
      }
    }
    return
  }
}
