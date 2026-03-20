//===- air_hierarchy_to_aie_leaves_channels.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Verify that --air-hierarchy-to-aie creates aie.device/tile/core and
// preserves air.channel declarations and put/get ops (no DMA lowering).

// RUN: air-opt %s -air-hierarchy-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// CHECK: aie.device
// CHECK-DAG: aie.tile(2, 3)
// CHECK-DAG: aie.tile(2, 4)

// Core bodies must contain air.channel.put / air.channel.get.
// CHECK-DAG: air.channel.put{{.*}}@channel_0
// CHECK-DAG: air.channel.get{{.*}}@channel_0

// Channels must survive inside the device.
// CHECK: air.channel @channel_0

// No DMA lowering should have occurred.
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.flow
// CHECK-NOT: aie.dma_bd
// CHECK-NOT: aie.dma_start

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_0 [1, 1]
func.func @one_to_one() {
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
          %4 = air.channel.put async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
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
