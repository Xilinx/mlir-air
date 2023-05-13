//===- unroll_scf_for_and_hoist_alloc.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-unroll-loop-for-pipelining-pattern -air-hoist-alloc-in-for-pattern | FileCheck %s

// Unroll an scf for loop, and hoist memory alloc ops, for pipelining.

// CHECK: func @unroll_and_hoist_alloc
// CHECK: %[[EVENT0:.*]], %[[VALUE0:.*]] = air.execute
// CHECK: memref.alloc()
// CHECK: {unrolled_iteration = 1 : i32}
// CHECK: %[[EVENT1:.*]], %[[VALUE1:.*]] = air.execute
// CHECK: memref.alloc()
// CHECK: {unrolled_iteration = 0 : i32}
// CHECK: %[[EVENT2:.*]] = scf.for {{.*}} iter_args(%[[EVENT1:.*]] =
// CHECK: %[[EVENT3:.*]] = air.wait_all async [%[[EVENT1]]]{{.*}}{unrolled_iteration = 0 : i32}
// CHECK: %[[EVENT4:.*]] = air.wait_all async [%[[EVENT3]]]{{.*}}{unrolled_iteration = 1 : i32}
// CHECK: scf.yield %[[EVENT4]]
// CHECK: %[[EVENT5:.*]] = air.execute [%[[EVENT2]]]
// CHECK: memref.dealloc
// CHECK: {unrolled_iteration = 0 : i32}
// CHECK: %[[EVENT6:.*]] = air.execute [%[[EVENT2]]]
// CHECK: memref.dealloc
// CHECK: {unrolled_iteration = 1 : i32}

func.func @unroll_and_hoist_alloc(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
    %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
      %c4 = arith.constant 4 : index
      %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %async_token_0 = air.wait_all async
        %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
          %async_token_3, %results_4 = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() {hoist_alloc = "true"} : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %wait = air.wait_all async [%async_token_3]
          %async_token_5 = air.execute [%wait] {
            memref.dealloc %results_4 : memref<32x32xbf16, 2>
          }
          scf.yield %async_token_5 : !air.async.token
        } {unroll = 2 : i32}
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}