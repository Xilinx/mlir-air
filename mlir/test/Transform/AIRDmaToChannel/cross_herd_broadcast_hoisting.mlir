//===- cross_herd_broadcast_hoisting.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that user-written channel ops are not duplicated to segment level
// when DMA ops in the same herd are hoisted by air-dma-to-channel.
//
// Scenario: producer herd [1,1] broadcasts via @bcast, consumer herd [4,1]
// receives the broadcast and writes to L3 via dma_memcpy_nd. The DMA's
// backward slice includes the user-written channel.get (via async token
// dependency). The pass must NOT clone channel.get @bcast to segment level.

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// Verify channel declarations: @bcast should remain unchanged, new channels
// should be created for the DMAs.
// CHECK-DAG: air.channel @bcast [1, 1] {broadcast_shape = [4 : index, 1 : index]}
// CHECK-DAG: air.channel @channel_{{.*}}

// Verify that air.channel.get @bcast does NOT appear at segment level (outside
// the herds). The segment body should have scf.parallel ops with channel ops
// for the DMA-derived channels only, not for @bcast.
// CHECK-LABEL: func.func @cross_herd_broadcast
// CHECK: air.segment
// CHECK-NOT: air.channel.get{{.*}}@bcast

// Verify that air.channel.get @bcast remains inside the consumer herd.
// CHECK: air.herd @consumer
// CHECK: air.channel.get{{.*}}@bcast

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  air.channel @bcast [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  func.func @cross_herd_broadcast(%arg0: memref<64xbf16>, %arg1: memref<256xbf16>) {
    %0 = air.launch async () in () args(%arg2=%arg0, %arg3=%arg1) : memref<64xbf16>, memref<256xbf16> attributes {id = 2 : i32} {
      %1 = air.segment @seg async  args(%arg4=%arg2, %arg5=%arg3) : memref<64xbf16>, memref<256xbf16> attributes {id = 1 : i32} {
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        // Producer herd: load from L3, broadcast to consumer tiles.
        %2 = air.herd @producer async  tile (%arg6, %arg7) in (%arg8=%c1, %arg9=%c1) args(%arg10=%arg4) : memref<64xbf16> attributes {id = 1 : i32} {
          %async_token, %results = air.execute -> (memref<64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
          } {id = 1 : i32}
          %4 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg10[] [] []) {id = 1 : i32} : (memref<64xbf16, 2 : i32>, memref<64xbf16>)
          %5 = air.channel.put async [%4]  @bcast[] (%results[] [] []) {id = 1 : i32} : (memref<64xbf16, 2 : i32>)
          %async_token_0 = air.execute [%5] {
            memref.dealloc %results : memref<64xbf16, 2 : i32>
          } {id = 2 : i32}
        }
        // Consumer herd: receive broadcast, then DMA to L3.
        // The DMA depends on channel.get @bcast via async token %4.
        // This is the critical pattern: the backward slice of the DMA's
        // external half includes the user-written channel.get.
        %3 = air.herd @consumer async  tile (%arg6, %arg7) in (%arg8=%c4, %arg9=%c1) args(%arg10=%arg5) : memref<256xbf16> attributes {id = 2 : i32} {
          %async_token, %results = air.execute -> (memref<64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
          } {id = 3 : i32}
          %4 = air.channel.get async [%async_token]  @bcast[%arg6, %arg7] (%results[] [] []) {id = 2 : i32} : (memref<64xbf16, 2 : i32>)
          %5 = affine.apply #map()[%arg6]
          %c64 = arith.constant 64 : index
          %c1_0 = arith.constant 1 : index
          %6 = air.dma_memcpy_nd async [%4] (%arg10[%5] [%c64] [%c1_0], %results[] [] []) {id = 2 : i32} : (memref<256xbf16>, memref<64xbf16, 2 : i32>)
          %async_token_1 = air.execute [%6] {
            memref.dealloc %results : memref<64xbf16, 2 : i32>
          } {id = 4 : i32}
        }
      }
    }
    return
  }
}
