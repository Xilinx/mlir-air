//===- opt_memtile_dma_bds_device_issue.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" | FileCheck %s

// A core reads one L2 row per trip of a ping-pong loop and starts the memtile
// put that sends it. The put sits at the segment, in a loop of its own, and is
// shaped like any other L2 put: four starts of one buffer filled once are four
// puts, one per lock credit.

// CHECK-LABEL: air.segment @seg
// CHECK-COUNT-4: air.channel.put async {{.*}} @Row[] {{.*}} {air.device_issue = 2 : i64
// CHECK-NOT: scf.for
// CHECK: air.herd @fwd
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c2
// CHECK-NOT: air.channel.put {{.*}} @Row[]
// CHECK: air.channel.issue @Row[]
// CHECK: air.channel.issue @Row[]
// CHECK: air.segment_end

module {
  air.channel @WIn []
  air.channel @IdxIn []
  air.channel @Row []
  air.channel @YOut []
  func.func @device_issue_l2(%arg0: memref<8x256xi32>, %arg1: memref<16xi32>, %arg2: memref<4x256xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<8x256xi32>, memref<16xi32>, memref<4x256xi32> attributes {id = 1 : i32} {
      %1 = air.channel.put async  @WIn[] (%arg7[] [] []) {id = 1 : i32} : (memref<8x256xi32>)
      %2 = air.channel.put async  @IdxIn[] (%arg8[] [] []) {id = 2 : i32} : (memref<16xi32>)
      %3 = air.channel.get async  @YOut[] (%arg9[] [] []) {id = 3 : i32} : (memref<4x256xi32>)
      %4 = air.segment @seg async  attributes {id = 2 : i32} {
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<8x256xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x256xi32, 1 : i32>
          air.execute_terminator %alloc : memref<8x256xi32, 1 : i32>
        }
        %5 = air.channel.get async [%async_token]  @WIn[] (%results[] [] []) {id = 4 : i32} : (memref<8x256xi32, 1 : i32>)
        %6 = scf.for %arg10 = %c0 to %c4 step %c1_0 iter_args(%arg11 = %5) -> (!air.async.token) {
          %8 = air.channel.put async [%arg11]  @Row[] (%results[0, 0] [1, 256] [256, 1]) {air.device_issue = 2 : i64, id = 5 : i32} : (memref<8x256xi32, 1 : i32>)
          scf.yield %8 : !air.async.token
        }
        %7 = air.herd @fwd async [%5]  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) attributes {id = 3 : i32} {
          %c2 = arith.constant 2 : index
          %c256_i64 = arith.constant 256 : i64
          %c16 = arith.constant 16 : index
          %c256 = arith.constant 256 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_2 = arith.constant 1 : index
          %c4_3 = arith.constant 4 : index
          %c0_4 = arith.constant 0 : index
          %async_token_5, %results_6 = air.execute -> (memref<16xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<16xi32, 2 : i32>
            air.execute_terminator %alloc : memref<16xi32, 2 : i32>
          }
          %8 = air.channel.get async [%async_token_5]  @IdxIn[] (%results_6[] [] []) {id = 6 : i32} : (memref<16xi32, 2 : i32>)
          %async_token_7, %results_8 = air.execute [%8] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %async_token_9, %results_10 = air.execute [%async_token_7] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %async_token_11, %results_12 = air.execute [%async_token_9] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %async_token_13, %results_14 = air.execute [%async_token_9] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %9 = air.wait_all async [%async_token_11, %async_token_13]
          %10 = air.wait_all async [%async_token_11, %async_token_13]
          %11:4 = scf.for %arg14 = %c0_4 to %c4_3 step %c2 iter_args(%arg15 = %10, %arg16 = %async_token_13, %arg17 = %9, %arg18 = %9) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %async_token_20, %results_21 = air.execute [%arg18, %arg15] -> (i32) {
              %27 = memref.load %results_6[%arg14] : memref<16xi32, 2 : i32>
              air.execute_terminator %27 : i32
            }
            %12 = arith.index_cast %results_21 : i32 to index
            %13 = arith.index_cast %12 : index to i64
            %14 = arith.muli %13, %c256_i64 : i64
            %async_token_22 = air.execute [%arg18, %arg15] {
              air.channel.issue  @Row[] offset(%14 : i64) {issue_id = 2 : i64}
            }
            %15 = air.channel.get async [%arg15, %arg18]  @Row[] (%results_14[] [] []) {id = 7 : i32} : (memref<256xi32, 2 : i32>)
            %16 = scf.for %arg19 = %c0_4 to %c256 step %c16 iter_args(%arg20 = %15) -> (!air.async.token) {
              %async_token_26, %results_27 = air.execute [%arg20] -> (vector<16xi32>) {
                %28 = vector.transfer_read %results_14[%arg19], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %28 : vector<16xi32>
              }
              %async_token_28 = air.execute [%arg20] {
                vector.transfer_write %results_27, %results_12[%arg19] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
              }
              %27 = air.wait_all async [%async_token_26, %async_token_28]
              scf.yield %27 : !air.async.token
            }
            %17 = air.channel.put async [%16, %arg15, %arg18, %arg17]  @YOut[] (%results_12[] [] []) {id = 8 : i32} : (memref<256xi32, 2 : i32>)
            %18 = arith.addi %arg14, %c1_2 : index
            %async_token_23, %results_24 = air.execute [%async_token_20, %arg16] -> (i32) {
              %27 = memref.load %results_6[%18] : memref<16xi32, 2 : i32>
              air.execute_terminator %27 : i32
            }
            %19 = arith.index_cast %results_24 : i32 to index
            %20 = arith.index_cast %19 : index to i64
            %21 = arith.muli %20, %c256_i64 : i64
            %async_token_25 = air.execute [%16, %15, %arg16, %async_token_20] {
              air.channel.issue  @Row[] offset(%21 : i64) {issue_id = 2 : i64}
            }
            %22 = air.channel.get async [%arg16, %15]  @Row[] (%results_8[] [] []) {id = 7 : i32} : (memref<256xi32, 2 : i32>)
            %23 = scf.for %arg19 = %c0_4 to %c256 step %c16 iter_args(%arg20 = %22) -> (!air.async.token) {
              %async_token_26, %results_27 = air.execute [%arg20] -> (vector<16xi32>) {
                %28 = vector.transfer_read %results_8[%arg19], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %28 : vector<16xi32>
              }
              %async_token_28 = air.execute [%arg20] {
                vector.transfer_write %results_27, %results_10[%arg19] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
              }
              %27 = air.wait_all async [%async_token_26, %async_token_28]
              scf.yield %27 : !air.async.token
            }
            %24 = air.channel.put async [%23, %arg15, %arg16]  @YOut[] (%results_10[] [] []) {id = 8 : i32} : (memref<256xi32, 2 : i32>)
            %25 = air.wait_all async [%async_token_23, %24]
            %26 = air.wait_all async [%22, %23]
            scf.yield %async_token_20, %25, %25, %26 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %async_token_15 = air.execute [%11#1] {
            memref.dealloc %results_12 : memref<256xi32, 2 : i32>
          }
          %async_token_16 = air.execute [%11#1] {
            memref.dealloc %results_14 : memref<256xi32, 2 : i32>
          }
          %async_token_17 = air.execute [%11#1] {
            memref.dealloc %results_10 : memref<256xi32, 2 : i32>
          }
          %async_token_18 = air.execute [%11#1] {
            memref.dealloc %results_8 : memref<256xi32, 2 : i32>
          }
          %async_token_19 = air.execute [%11#1] {
            memref.dealloc %results_6 : memref<16xi32, 2 : i32>
          }
        }
        %async_token_1 = air.execute [%6] {
          memref.dealloc %results : memref<8x256xi32, 1 : i32>
        }
      }
    }
    return
  }
}
