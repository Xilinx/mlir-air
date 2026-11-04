//===- opt_memtile_dma_bds_device_issue.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" | FileCheck %s

// A core reads one L2 row per trip of a ping-pong loop and starts the memtile
// put that sends it. The put waits in the herd until this pass,
// which hoists it to the segment and then shapes it like any other L2 put:
// four starts of one buffer filled once are four puts, one per lock credit.
// The core's loop keeps its ping-pong.

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
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<8x256xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x256xi32, 1 : i32>
          air.execute_terminator %alloc : memref<8x256xi32, 1 : i32>
        }
        %5 = air.channel.get async [%async_token]  @WIn[] (%results[] [] []) {id = 4 : i32} : (memref<8x256xi32, 1 : i32>)
        %6 = air.herd @fwd async [%5]  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) args(%arg14=%results) : memref<8x256xi32, 1 : i32> attributes {id = 3 : i32} {
          %c2 = arith.constant 2 : index
          %c256_i64 = arith.constant 256 : i64
          %c16 = arith.constant 16 : index
          %c256 = arith.constant 256 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_2 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c0 = arith.constant 0 : index
          %async_token_3, %results_4 = air.execute -> (memref<16xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<16xi32, 2 : i32>
            air.execute_terminator %alloc : memref<16xi32, 2 : i32>
          }
          %7 = air.channel.get async [%async_token_3]  @IdxIn[] (%results_4[] [] []) {id = 5 : i32} : (memref<16xi32, 2 : i32>)
          %async_token_5, %results_6 = air.execute [%7] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %async_token_7, %results_8 = air.execute [%async_token_5] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %async_token_9, %results_10 = air.execute [%async_token_7] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %async_token_11, %results_12 = air.execute [%async_token_7] -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %8 = air.wait_all async [%async_token_9, %async_token_11] 
          %9 = air.wait_all async [%async_token_9, %async_token_11] 
          %10:4 = scf.for %arg15 = %c0 to %c4 step %c2 iter_args(%arg16 = %9, %arg17 = %async_token_11, %arg18 = %8, %arg19 = %8) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %async_token_18, %results_19 = air.execute [%arg19, %arg16] -> (i32) {
              %28 = memref.load %results_4[%arg15] : memref<16xi32, 2 : i32>
              air.execute_terminator %28 : i32
            }
            %11 = arith.index_cast %results_19 : i32 to index
            %12 = arith.index_cast %11 : index to i64
            %13 = arith.muli %12, %c256_i64 : i64
            %async_token_20 = air.execute [%arg19, %arg16] {
              air.channel.issue  @Row[] offset(%13 : i64) {issue_id = 2 : i64}
            }
            %14 = air.channel.put async [%async_token_20]  @Row[] (%arg14[0, 0] [1, 256] [256, 1]) {air.device_issue = 2 : i64, id = 6 : i32} : (memref<8x256xi32, 1 : i32>)
            %15 = air.channel.get async [%arg16, %arg19]  @Row[] (%results_12[] [] []) {id = 7 : i32} : (memref<256xi32, 2 : i32>)
            %16 = scf.for %arg20 = %c0 to %c256 step %c16 iter_args(%arg21 = %15) -> (!air.async.token) {
              %async_token_24, %results_25 = air.execute [%arg21] -> (vector<16xi32>) {
                %29 = vector.transfer_read %results_12[%arg20], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %29 : vector<16xi32>
              }
              %async_token_26 = air.execute [%arg21] {
                vector.transfer_write %results_25, %results_10[%arg20] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
              }
              %28 = air.wait_all async [%async_token_24, %async_token_26] 
              scf.yield %28 : !air.async.token
            }
            %17 = air.channel.put async [%16, %arg16, %arg19, %arg18]  @YOut[] (%results_10[] [] []) {id = 8 : i32} : (memref<256xi32, 2 : i32>)
            %18 = arith.addi %arg15, %c1_2 : index
            %async_token_21, %results_22 = air.execute [%async_token_18, %arg17] -> (i32) {
              %28 = memref.load %results_4[%18] : memref<16xi32, 2 : i32>
              air.execute_terminator %28 : i32
            }
            %19 = arith.index_cast %results_22 : i32 to index
            %20 = arith.index_cast %19 : index to i64
            %21 = arith.muli %20, %c256_i64 : i64
            %async_token_23 = air.execute [%arg17] {
              air.channel.issue  @Row[] offset(%21 : i64) {issue_id = 2 : i64}
            }
            %22 = air.channel.put async [%async_token_23]  @Row[] (%arg14[0, 0] [1, 256] [256, 1]) {air.device_issue = 2 : i64, id = 6 : i32} : (memref<8x256xi32, 1 : i32>)
            %23 = air.channel.get async [%arg17, %15]  @Row[] (%results_6[] [] []) {id = 7 : i32} : (memref<256xi32, 2 : i32>)
            %24 = scf.for %arg20 = %c0 to %c256 step %c16 iter_args(%arg21 = %23) -> (!air.async.token) {
              %async_token_24, %results_25 = air.execute [%arg21] -> (vector<16xi32>) {
                %29 = vector.transfer_read %results_6[%arg20], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %29 : vector<16xi32>
              }
              %async_token_26 = air.execute [%arg21] {
                vector.transfer_write %results_25, %results_8[%arg20] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
              }
              %28 = air.wait_all async [%async_token_24, %async_token_26] 
              scf.yield %28 : !air.async.token
            }
            %25 = air.channel.put async [%24, %arg16, %arg17]  @YOut[] (%results_8[] [] []) {id = 8 : i32} : (memref<256xi32, 2 : i32>)
            %26 = air.wait_all async [%async_token_21, %22, %25] 
            %27 = air.wait_all async [%23, %24] 
            scf.yield %async_token_18, %26, %26, %27 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %async_token_13 = air.execute [%10#1] {
            memref.dealloc %results_10 : memref<256xi32, 2 : i32>
          }
          %async_token_14 = air.execute [%10#1] {
            memref.dealloc %results_12 : memref<256xi32, 2 : i32>
          }
          %async_token_15 = air.execute [%10#1] {
            memref.dealloc %results_8 : memref<256xi32, 2 : i32>
          }
          %async_token_16 = air.execute [%10#1] {
            memref.dealloc %results_6 : memref<256xi32, 2 : i32>
          }
          %async_token_17 = air.execute [%10#1] {
            memref.dealloc %results_4 : memref<16xi32, 2 : i32>
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

