//===- air_device_issue_on_chip.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu2 use-lock-race-condition-fix-auto=true" | FileCheck %s

// A core starts a memtile MM2S whose offset it computes. The
// descriptor is a task in @__air_issue_templates on a BD of its own, its buffer
// pinned at the top of the memtile; the core writes word 1 (offset field, in
// words, in the memtile's own window) and queues the BD, without reading back.

// CHECK-LABEL: aie.device
// CHECK: %[[READY:.*]] = aie.lock(%[[T:.*]], {{.*}}) {init = 0 : i32, sym_name = "__air_issue_ready_0_2"}
// CHECK: aie.dma_bd(%__air_issue_ctrl_out_0_2 : memref<4xi32, 2 : i32> offset = 0 len = 3)
// CHECK: aie.core
// CHECK: aie.use_lock(%[[READY]], AcquireGreaterEqual
// CHECK: arith.addi %{{.*}}, %c260096_i32
// CHECK: memref.store %c-2130050332_i32, %__air_issue_ctrl_out_0_2
// CHECK: memref.store %c-2130049484_i32, %__air_issue_ctrl_out_0_2
// CHECK: memref.store %c23_i32, %__air_issue_ctrl_out_0_2
// CHECK-NOT: air.channel.issue
// CHECK: aie.buffer(%{{.*}}) {address = 516096 : i32{{.*}}} : memref<8x256xi32, 1 : i32>
// CHECK: aie.runtime_sequence @__air_issue_templates()
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 256) {bd_id = 23 : i32}
// CHECK: } {air.device_issue = 2 : i64, air.device_issue_channel = 0 : i32, air.device_issue_dir = 1 : i32, air.device_issue_lock = "__air_issue_ready_0_2"}
// CHECK: aie.packet_source<%[[T]], DMA : 1>
// CHECK-NEXT: aie.packet_dest<%{{.*}}, TileControl : 0>
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
      %4 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<8x256xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x256xi32, 1 : i32>
          air.execute_terminator %alloc : memref<8x256xi32, 1 : i32>
        }
        %5 = air.channel.get async [%async_token]  @WIn[] (%results[] [] []) {id = 4 : i32} : (memref<8x256xi32, 1 : i32>)
        %6 = air.channel.put async [%5]  @Row[] (%results[0] [256] [1]) {air.device_issue = 2 : i64, id = 5 : i32} : (memref<8x256xi32, 1 : i32>)
        %7 = air.channel.put async [%6]  @Row[] (%results[0] [256] [1]) {air.device_issue = 2 : i64, id = 6 : i32} : (memref<8x256xi32, 1 : i32>)
        %8 = air.channel.put async [%7]  @Row[] (%results[0] [256] [1]) {air.device_issue = 2 : i64, id = 7 : i32} : (memref<8x256xi32, 1 : i32>)
        %9 = air.channel.put async [%8]  @Row[] (%results[0] [256] [1]) {air.device_issue = 2 : i64, id = 8 : i32} : (memref<8x256xi32, 1 : i32>)
        %10 = air.herd @fwd async [%5]  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
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
          %11 = air.channel.get async [%async_token_3]  @IdxIn[] (%results_4[] [] []) {id = 9 : i32} : (memref<16xi32, 2 : i32>)
          %async_token_5, %results_6 = air.execute [%11] -> (memref<256xi32, 2 : i32>) {
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
          %12 = air.wait_all async [%async_token_9, %async_token_11]
          %13 = air.wait_all async [%async_token_9, %async_token_11]
          %14:4 = scf.for %arg14 = %c0 to %c4 step %c2 iter_args(%arg15 = %13, %arg16 = %async_token_11, %arg17 = %12, %arg18 = %12) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %async_token_18, %results_19 = air.execute [%arg18, %arg15] -> (i32) {
              %30 = memref.load %results_4[%arg14] : memref<16xi32, 2 : i32>
              air.execute_terminator %30 : i32
            }
            %15 = arith.index_cast %results_19 : i32 to index
            %16 = arith.index_cast %15 : index to i64
            %17 = arith.muli %16, %c256_i64 : i64
            %async_token_20 = air.execute [%arg18, %arg15] {
              air.channel.issue  @Row[] offset(%17 : i64) {issue_id = 2 : i64}
            }
            %18 = air.channel.get async [%arg15, %arg18]  @Row[] (%results_12[] [] []) {id = 10 : i32} : (memref<256xi32, 2 : i32>)
            %19 = scf.for %arg19 = %c0 to %c256 step %c16 iter_args(%arg20 = %18) -> (!air.async.token) {
              %async_token_24, %results_25 = air.execute [%arg20] -> (vector<16xi32>) {
                %31 = vector.transfer_read %results_12[%arg19], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %31 : vector<16xi32>
              }
              %async_token_26 = air.execute [%arg20] {
                vector.transfer_write %results_25, %results_10[%arg19] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
              }
              %30 = air.wait_all async [%async_token_24, %async_token_26]
              scf.yield %30 : !air.async.token
            }
            %20 = air.channel.put async [%19, %arg15, %arg18, %arg17]  @YOut[] (%results_10[] [] []) {id = 11 : i32} : (memref<256xi32, 2 : i32>)
            %21 = arith.addi %arg14, %c1_2 : index
            %async_token_21, %results_22 = air.execute [%async_token_18, %arg16] -> (i32) {
              %30 = memref.load %results_4[%21] : memref<16xi32, 2 : i32>
              air.execute_terminator %30 : i32
            }
            %22 = arith.index_cast %results_22 : i32 to index
            %23 = arith.index_cast %22 : index to i64
            %24 = arith.muli %23, %c256_i64 : i64
            %async_token_23 = air.execute [%19, %18, %arg16, %async_token_18] {
              air.channel.issue  @Row[] offset(%24 : i64) {issue_id = 2 : i64}
            }
            %25 = air.channel.get async [%arg16, %18]  @Row[] (%results_6[] [] []) {id = 12 : i32} : (memref<256xi32, 2 : i32>)
            %26 = scf.for %arg19 = %c0 to %c256 step %c16 iter_args(%arg20 = %25) -> (!air.async.token) {
              %async_token_24, %results_25 = air.execute [%arg20] -> (vector<16xi32>) {
                %31 = vector.transfer_read %results_6[%arg19], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %31 : vector<16xi32>
              }
              %async_token_26 = air.execute [%arg20] {
                vector.transfer_write %results_25, %results_8[%arg19] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
              }
              %30 = air.wait_all async [%async_token_24, %async_token_26]
              scf.yield %30 : !air.async.token
            }
            %27 = air.channel.put async [%20, %26, %arg15, %arg16]  @YOut[] (%results_8[] [] []) {id = 13 : i32} : (memref<256xi32, 2 : i32>)
            %28 = air.wait_all async [%async_token_21, %27]
            %29 = air.wait_all async [%25, %26]
            scf.yield %async_token_18, %28, %28, %29 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %async_token_13 = air.execute [%14#1] {
            memref.dealloc %results_10 : memref<256xi32, 2 : i32>
          }
          %async_token_14 = air.execute [%14#1] {
            memref.dealloc %results_12 : memref<256xi32, 2 : i32>
          }
          %async_token_15 = air.execute [%14#1] {
            memref.dealloc %results_8 : memref<256xi32, 2 : i32>
          }
          %async_token_16 = air.execute [%14#1] {
            memref.dealloc %results_6 : memref<256xi32, 2 : i32>
          }
          %async_token_17 = air.execute [%14#1] {
            memref.dealloc %results_4 : memref<16xi32, 2 : i32>
          }
        }
        %async_token_1 = air.execute [%9] {
          memref.dealloc %results : memref<8x256xi32, 1 : i32>
        }
        air.wait_all [%10, %async_token_1]  {air.segment_end}
      }
    }
    return
  }
}

// -----

// A core starts its own S2MM, into a slot it computes: its
// buffer pinned at the top of L1, word 0 written as address and length.

// CHECK-LABEL: aie.device
// CHECK-DAG: aie.buffer(%{{.*}}) {address = 57344 : i32{{.*}}} : memref<8x256xi32, 2 : i32>
// CHECK: aie.core
// CHECK: arith.addi %{{.*}}, %c234881280_i32
// CHECK: memref.store %c16896480_i32, %__air_issue_ctrl_out_0_2
// CHECK: memref.store %c-2130584052_i32, %__air_issue_ctrl_out_0_2
// CHECK: memref.store %c15_i32, %__air_issue_ctrl_out_0_2
// CHECK: aie.runtime_sequence @__air_issue_templates()
// CHECK: aiex.dma_configure_task(%[[T:.*]], S2MM, 1)
// CHECK: {bd_id = 15 : i32}
// CHECK: aie.packet_source<%[[T]], DMA : 1>
// CHECK-NEXT: aie.packet_dest<%[[T]], TileControl : 0>
module {
  air.channel @WIn []
  air.channel @IdxIn []
  air.channel @YOut []
  func.func @device_issue_l1(%arg0: memref<4x256xi32>, %arg1: memref<16xi32>, %arg2: memref<8x256xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<4x256xi32>, memref<16xi32>, memref<8x256xi32> attributes {id = 1 : i32} {
      %1 = air.channel.put async  @IdxIn[] (%arg8[] [] []) {id = 1 : i32} : (memref<16xi32>)
      %2 = air.channel.put async  @WIn[] (%arg7[0, 0] [1, 256] [256, 1]) {id = 2 : i32} : (memref<4x256xi32>)
      %3 = air.channel.put async [%2]  @WIn[] (%arg7[1, 0] [1, 256] [256, 1]) {id = 3 : i32} : (memref<4x256xi32>)
      %4 = air.channel.put async [%3]  @WIn[] (%arg7[2, 0] [1, 256] [256, 1]) {id = 4 : i32} : (memref<4x256xi32>)
      %5 = air.channel.put async [%4]  @WIn[] (%arg7[3, 0] [1, 256] [256, 1]) {id = 5 : i32} : (memref<4x256xi32>)
      %6 = air.channel.get async  @YOut[] (%arg9[] [] []) {id = 6 : i32} : (memref<8x256xi32>)
      %7 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %8 = air.herd @fwd async  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c256_i64 = arith.constant 256 : i64
          %c16 = arith.constant 16 : index
          %c256 = arith.constant 256 : index
          %c8 = arith.constant 8 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c0 = arith.constant 0 : index
          %async_token, %results = air.execute -> (memref<16xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<16xi32, 2 : i32>
            air.execute_terminator %alloc : memref<16xi32, 2 : i32>
          }
          %async_token_2, %results_3 = air.execute -> (memref<8x256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<8x256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<8x256xi32, 2 : i32>
          }
          %async_token_4, %results_5 = air.execute -> (memref<8x256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<8x256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<8x256xi32, 2 : i32>
          }
          %9 = air.channel.get async [%async_token]  @IdxIn[] (%results[] [] []) {id = 7 : i32} : (memref<16xi32, 2 : i32>)
          %10 = air.wait_all async [%async_token_2, %9]
          %11 = scf.for %arg14 = %c0 to %c4 step %c1_1 iter_args(%arg15 = %10) -> (!air.async.token) {
            %async_token_9, %results_10 = air.execute [%arg15] -> (i32) {
              %20 = memref.load %results[%arg14] : memref<16xi32, 2 : i32>
              air.execute_terminator %20 : i32
            }
            %15 = arith.index_cast %results_10 : i32 to index
            %16 = arith.index_cast %15 : index to i64
            %17 = arith.muli %16, %c256_i64 : i64
            %async_token_11 = air.execute [%arg15] {
              air.channel.issue  @WIn[] offset(%17 : i64) {issue_id = 7 : i64}
            }
            %18 = air.channel.get async [%async_token_11]  @WIn[] (%results_3[0] [256] [1]) {air.device_issue = 7 : i64, id = 8 : i32} : (memref<8x256xi32, 2 : i32>)
            %19 = air.wait_all async [%async_token_9, %18]
            scf.yield %19 : !air.async.token
          }
          %async_token_6 = air.execute [%11] {
            memref.dealloc %results : memref<16xi32, 2 : i32>
          }
          %12 = air.wait_all async [%async_token_4, %11]
          %13 = scf.for %arg14 = %c0 to %c8 step %c1_1 iter_args(%arg15 = %12) -> (!air.async.token) {
            %15 = scf.for %arg16 = %c0 to %c256 step %c16 iter_args(%arg17 = %arg15) -> (!air.async.token) {
              %async_token_9, %results_10 = air.execute [%arg17] -> (vector<16xi32>) {
                %17 = vector.transfer_read %results_3[%arg14, %arg16], %c0_i32 {in_bounds = [true]} : memref<8x256xi32, 2 : i32>, vector<16xi32>
                air.execute_terminator %17 : vector<16xi32>
              }
              %async_token_11 = air.execute [%arg17] {
                vector.transfer_write %results_10, %results_5[%arg14, %arg16] {in_bounds = [true]} : vector<16xi32>, memref<8x256xi32, 2 : i32>
              }
              %16 = air.wait_all async [%async_token_9, %async_token_11]
              scf.yield %16 : !air.async.token
            }
            scf.yield %15 : !air.async.token
          }
          %async_token_7 = air.execute [%13] {
            memref.dealloc %results_3 : memref<8x256xi32, 2 : i32>
          }
          %14 = air.channel.put async [%13]  @YOut[] (%results_5[] [] []) {id = 9 : i32} : (memref<8x256xi32, 2 : i32>)
          %async_token_8 = air.execute [%14] {
            memref.dealloc %results_5 : memref<8x256xi32, 2 : i32>
          }
        }
        air.wait_all [%8]  {air.segment_end}
      }
    }
    return
  }
}
