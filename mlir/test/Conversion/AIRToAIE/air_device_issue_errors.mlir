//===- air_device_issue_errors.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" -verify-diagnostics -split-input-file

// Two L3 puts claim the descriptor one core issue starts.
module {
  air.channel @Wexp [1]
  air.channel @Wexp2 [1]
  func.func @moe(%arg0: memref<32x256xbf16>, %arg1: memref<256xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %1 = air.channel.put async  @Wexp[%c0] (%arg6[0, 0] [1, 256] [256, 1]) {air.device_issue = 0 : i64, id = 1 : i32} : (memref<32x256xbf16>)
      // expected-error@+1 {{shares its air.device_issue id with another L3 transfer}}
      %11 = air.channel.put async  @Wexp2[%c0] (%arg6[0, 0] [1, 256] [256, 1]) {air.device_issue = 0 : i64, id = 5 : i32} : (memref<32x256xbf16>)
      %2 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %3 = air.herd @router async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c256 = arith.constant 256 : index
          %c0_1 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %async_token, %results = air.execute -> (memref<4xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4xi32, 2 : i32>
            air.execute_terminator %alloc : memref<4xi32, 2 : i32>
          }
          %async_token_2, %results_3 = air.execute [%async_token] -> (i32) {
            %7 = memref.load %results[%c2] : memref<4xi32, 2 : i32>
            air.execute_terminator %7 : i32
          }
          %async_token_4 = air.execute [%async_token_2] {
            memref.dealloc %results : memref<4xi32, 2 : i32>
          }
          %5 = arith.index_cast %results_3 : i32 to index
          %6 = arith.muli %5, %c256 : index
          %async_token_5 = air.execute {
            air.channel.issue @Wexp[%c0_1] offset(%6 : index) {issue_id = 0 : i64}
          }
        }
        %4 = air.herd @mvm async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 4 : i32, x_loc = 1 : i64, y_loc = 2 : i64} {
          %c0_1 = arith.constant 0 : index
          %async_token, %results = air.execute -> (memref<256xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<256xbf16, 2 : i32>
          }
          %5 = air.channel.get async [%async_token]  @Wexp[%c0_1] (%results[] [] []) {id = 2 : i32} : (memref<256xbf16, 2 : i32>)
          %15 = air.channel.get async [%5]  @Wexp2[%c0_1] (%results[] [] []) {id = 6 : i32} : (memref<256xbf16, 2 : i32>)
          %async_token_2 = air.execute [%15] {
            memref.dealloc %results : memref<256xbf16, 2 : i32>
          }
        }
        air.wait_all [%3, %4]  {air.segment_end}
      }
    }
    return
  }
}

// -----

// Two cores of one herd start the one put the host arms.
module {
  air.channel @Wexp [1]
  func.func @moe(%arg0: memref<32x256xbf16>, %arg1: memref<256xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %1 = air.channel.put async  @Wexp[%c0] (%arg6[0, 0] [1, 256] [256, 1]) {air.device_issue = 0 : i64, id = 1 : i32} : (memref<32x256xbf16>)
      %2 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        %3 = air.herd @router async  tile (%arg7, %arg8) in (%arg9=%c2_0, %arg10=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c256 = arith.constant 256 : index
          %c0_1 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %async_token, %results = air.execute -> (memref<4xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4xi32, 2 : i32>
            air.execute_terminator %alloc : memref<4xi32, 2 : i32>
          }
          %async_token_2, %results_3 = air.execute [%async_token] -> (i32) {
            %7 = memref.load %results[%c2] : memref<4xi32, 2 : i32>
            air.execute_terminator %7 : i32
          }
          %async_token_4 = air.execute [%async_token_2] {
            memref.dealloc %results : memref<4xi32, 2 : i32>
          }
          %5 = arith.index_cast %results_3 : i32 to index
          %6 = arith.muli %5, %c256 : index
          %async_token_5 = air.execute {
            // expected-error@+1 {{starts an L3 transfer another issue also starts}}
            air.channel.issue @Wexp[%c0_1] offset(%6 : index) {issue_id = 0 : i64}
          }
        }
        %4 = air.herd @mvm async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 4 : i32, x_loc = 1 : i64, y_loc = 2 : i64} {
          %c0_1 = arith.constant 0 : index
          %async_token, %results = air.execute -> (memref<256xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<256xbf16, 2 : i32>
          }
          %5 = air.channel.get async [%async_token]  @Wexp[%c0_1] (%results[] [] []) {id = 2 : i32} : (memref<256xbf16, 2 : i32>)
          %async_token_2 = air.execute [%5] {
            memref.dealloc %results : memref<256xbf16, 2 : i32>
          }
        }
        air.wait_all [%3, %4]  {air.segment_end}
      }
    }
    return
  }
}

// -----

// The put the core starts is still in its herd.
module {
  air.channel @Wexp [1]
  func.func @moe(%arg0: memref<32x256xbf16>, %arg1: memref<256xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %2 = air.segment @seg async args(%sw=%arg6) : memref<32x256xbf16> attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %3 = air.herd @router async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) args(%w=%sw) : memref<32x256xbf16> attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c256 = arith.constant 256 : index
          %c0_1 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %async_token, %results = air.execute -> (memref<4xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4xi32, 2 : i32>
            air.execute_terminator %alloc : memref<4xi32, 2 : i32>
          }
          %async_token_2, %results_3 = air.execute [%async_token] -> (i32) {
            %7 = memref.load %results[%c2] : memref<4xi32, 2 : i32>
            air.execute_terminator %7 : i32
          }
          %async_token_4 = air.execute [%async_token_2] {
            memref.dealloc %results : memref<4xi32, 2 : i32>
          }
          %5 = arith.index_cast %results_3 : i32 to index
          %6 = arith.muli %5, %c256 : index
          %async_token_5 = air.execute {
            air.channel.issue @Wexp[%c0_1] offset(%6 : index) {issue_id = 0 : i64}
          }
          // expected-error@+1 {{is started by a core, so it must sit outside the herd, at the level of the memory it reads}}
          %p = air.channel.put async [%async_token_5] @Wexp[%c0_1] (%w[0, 0] [1, 256] [256, 1]) {air.device_issue = 0 : i64, id = 1 : i32} : (memref<32x256xbf16>)
        }
        %4 = air.herd @mvm async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 4 : i32, x_loc = 1 : i64, y_loc = 2 : i64} {
          %c0_1 = arith.constant 0 : index
          %async_token, %results = air.execute -> (memref<256xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<256xbf16, 2 : i32>
          }
          %5 = air.channel.get async [%async_token]  @Wexp[%c0_1] (%results[] [] []) {id = 2 : i32} : (memref<256xbf16, 2 : i32>)
          %async_token_2 = air.execute [%5] {
            memref.dealloc %results : memref<256xbf16, 2 : i32>
          }
        }
        air.wait_all [%3, %4]  {air.segment_end}
      }
    }
    return
  }
}
