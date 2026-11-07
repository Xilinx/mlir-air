//===- shrink_memref_sizes_by_access_device_issue.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access | FileCheck %s

// Only half the columns of an L2 buffer are ever accessed, but a core starts
// a put from it with a row offset computed from its 256-element stride, so
// the buffer keeps its shape.

// CHECK: memref.alloc() : memref<8x256xi32, 1 : i32>
// CHECK: air.channel.put async {{.*}}@Row[] (%{{.*}}[0, 0] [1, 128] [256, 1]) {air.device_issue
module {
  air.channel @WIn []
  air.channel @IdxIn []
  air.channel @Row []
  air.channel @YOut []
  func.func @device_issue_l2(%arg0: memref<8x128xi32>, %arg1: memref<16xi32>, %arg2: memref<4x128xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<8x128xi32>, memref<16xi32>, memref<4x128xi32> attributes {id = 1 : i32} {
      %1 = air.channel.put async  @WIn[] (%arg7[] [] []) {id = 1 : i32} : (memref<8x128xi32>)
      %2 = air.channel.put async  @IdxIn[] (%arg8[] [] []) {id = 2 : i32} : (memref<16xi32>)
      %3 = air.channel.get async  @YOut[] (%arg9[] [] []) {id = 3 : i32} : (memref<4x128xi32>)
      %4 = air.segment @seg async  attributes {id = 2 : i32} {
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<8x256xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x256xi32, 1 : i32>
          air.execute_terminator %alloc : memref<8x256xi32, 1 : i32>
        }
        %5 = air.channel.get async [%async_token]  @WIn[] (%results[0, 0] [8, 128] [256, 1]) {id = 4 : i32} : (memref<8x256xi32, 1 : i32>)
        %6 = scf.for %arg10 = %c0 to %c4 step %c1_0 iter_args(%arg11 = %5) -> (!air.async.token) {
          %8 = air.channel.put async [%arg11]  @Row[] (%results[0, 0] [1, 128] [256, 1]) {air.device_issue = 2 : i64, id = 5 : i32} : (memref<8x256xi32, 1 : i32>)
          scf.yield %8 : !air.async.token
        }
        %7 = air.herd @fwd async [%5]  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) attributes {id = 3 : i32} {
          %c256_i64 = arith.constant 256 : i64
          %c16 = arith.constant 16 : index
          %c128 = arith.constant 128 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_2 = arith.constant 1 : index
          %c4_3 = arith.constant 4 : index
          %c0_4 = arith.constant 0 : index
          %async_token_5, %results_6 = air.execute -> (memref<16xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<16xi32, 2 : i32>
            air.execute_terminator %alloc : memref<16xi32, 2 : i32>
          }
          %8 = air.wait_all async
          %9 = air.wait_all async
          %10 = air.channel.get async [%async_token_5]  @IdxIn[] (%results_6[] [] []) {id = 6 : i32} : (memref<16xi32, 2 : i32>)
          %11 = air.wait_all async [%8, %9, %10]
          %12 = scf.for %arg14 = %c0_4 to %c4_3 step %c1_2 iter_args(%arg15 = %11) -> (!air.async.token) {
            %async_token_8, %results_9 = air.execute -> (memref<128xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<128xi32, 2 : i32>
              air.execute_terminator %alloc : memref<128xi32, 2 : i32>
            }
            %async_token_10, %results_11 = air.execute -> (memref<128xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<128xi32, 2 : i32>
              air.execute_terminator %alloc : memref<128xi32, 2 : i32>
            }
            %async_token_12, %results_13 = air.execute [%arg15] -> (i32) {
              %22 = memref.load %results_6[%arg14] : memref<16xi32, 2 : i32>
              air.execute_terminator %22 : i32
            }
            %15 = arith.index_cast %results_13 : i32 to index
            %16 = arith.index_cast %15 : index to i64
            %17 = arith.muli %16, %c256_i64 : i64
            %async_token_14 = air.execute [%arg15] {
              air.channel.issue  @Row[] offset(%17 : i64) {issue_id = 2 : i64}
            }
            %18 = air.channel.get async [%async_token_10, %arg15]  @Row[] (%results_11[] [] []) {id = 7 : i32} : (memref<128xi32, 2 : i32>)
            %19 = scf.for %arg16 = %c0_4 to %c128 step %c16 iter_args(%arg17 = %18) -> (!air.async.token) {
              %async_token_17, %results_18 = air.execute [%async_token_10, %arg17] -> (vector<16xi32>) {
                %23 = vector.transfer_read %results_11[%arg16], %c0_i32 {in_bounds = [true]} : memref<128xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %23 : vector<16xi32>
              }
              %async_token_19 = air.execute [%async_token_8, %arg17] {
                vector.transfer_write %results_18, %results_9[%arg16] {in_bounds = [true]} : vector<16xi32>, memref<128xi32, 2 : i32>
              }
              %22 = air.wait_all async [%async_token_17, %async_token_19]
              scf.yield %22 : !air.async.token
            }
            %20 = air.channel.put async [%async_token_8, %19]  @YOut[] (%results_9[] [] []) {id = 8 : i32} : (memref<128xi32, 2 : i32>)
            %21 = air.wait_all async [%async_token_12, %20]
            %async_token_15 = air.execute [%19, %18] {
              memref.dealloc %results_11 : memref<128xi32, 2 : i32>
            }
            %async_token_16 = air.execute [%20, %19] {
              memref.dealloc %results_9 : memref<128xi32, 2 : i32>
            }
            scf.yield %21 : !air.async.token
          }
          %13 = air.wait_all async [%12]
          %14 = air.wait_all async [%12]
          %async_token_7 = air.execute [%12] {
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
