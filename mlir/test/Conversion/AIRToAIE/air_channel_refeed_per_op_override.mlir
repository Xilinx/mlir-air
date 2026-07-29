//===- air_channel_refeed_per_op_override.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// An EXPLICIT per-emission air.refeed_count on the put/get overrides the
// channel-level declaration for ANY value (including 1). Here the channel
// declares refeed_count=4 but the producer put sets refeed_count=1, so the
// re-broadcast credit lock must be initialized to the value-1 override, NOT the
// channel's 4. A regression that only honored per-op overrides > 1 would fall
// back to the channel declaration and re-introduce the 4-credit lock.

// CHECK-LABEL: aie.device
// CHECK-NOT: {init = 4 : i32}

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_ovr [1, 1] {air.refeed_count = 4 : i32}
func.func @refeed_override() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_ovr[] (%results_7[] [] []) {air.refeed_count = 1 : i32} : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_ovr[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
