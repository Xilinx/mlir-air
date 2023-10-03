//===- enforce_loop_carried_dealloc.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-enforce-loop-carried-memref-dealloc | FileCheck %s

// Reconnect dealloc ops to loop-carried dependency path, if disconnected.
// CHECK-LABEL: test1
// CHECK: %[[EVENT0:.*]], %[[VALUE0:.*]] = air.execute -> (memref<32x32xbf16, 1>) {
// CHECK-NEXT: memref.alloc() : memref<32x32xbf16, 1>
// CHECK: %[[EVENT1:.*]], %[[VALUE1:.*]] = air.execute
// CHECK-NEXT: memref.alloc() : memref<32x32xbf16, 2>
// CHECK: %[[EVENT2:.*]], %[[VALUE2:.*]] = air.execute
// CHECK-NEXT: memref.alloc() : memref<32x32xbf16, 2>
// CHECK: %[[EVENT3:.*]], %[[VALUE3:.*]] = air.execute
// CHECK-NEXT: memref.alloc() : memref<32x32xbf16, 2>
// CHECK: %[[EVENT4:.*]] = air.execute
// CHECK-NEXT: memref.dealloc
// CHECK: %[[EVENT5:.*]] = air.execute
// CHECK-NEXT: memref.dealloc
// CHECK: %[[EVENT6:.*]] = air.execute
// CHECK-NEXT: memref.dealloc
// CHECK: %[[EVENT7:.*]] = air.wait_all async [%[[EVENT5]], %[[EVENT6]]]
// CHECK: scf.yield %[[EVENT7]] : !air.async.token

func.func @test1() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async  args(%arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7) : index, index, index, index {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %async_token, %results = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %3 = air.herd @herd_0 async [%async_token]  tile (%arg16, %arg17) in (%arg18=%c1_0, %arg19=%c1_0) {
        %c128 = arith.constant 128 : index
        %c0_2 = arith.constant 0 : index
        %c512_3 = arith.constant 512 : index
        %5 = air.wait_all async 
        %6 = scf.for %arg14 = %c0_2 to %c512_3 step %c128 iter_args(%arg15 = %5) -> (!air.async.token) {
          %async_token_4, %results_5 = air.execute [%arg15] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %async_token_6, %results_7 = air.execute [%async_token_4] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %async_token_7, %results_9 = air.execute [%async_token_6] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %async_token_8 = air.execute [%async_token_7] {
            memref.dealloc %results_7 : memref<32x32xbf16, 2>
          }
          %async_token_9 = air.execute [%async_token_7] {
            memref.dealloc %results_5 : memref<32x32xbf16, 2>
          }
          %async_token_10 = air.execute [%async_token_8] {
            memref.dealloc %results_9 : memref<32x32xbf16, 2>
          }
          scf.yield %async_token_10 : !air.async.token
        }
        air.herd_terminator
      }
      %async_token_1 = air.execute [%3] {
        memref.dealloc %results : memref<32x32xbf16, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
