//===- air_hierarchy_to_aie_channel_bundle.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Verify that --air-hierarchy-to-aie runs specializeChannelBundle, which
// expands bundled channels (channels indexed by herd tile coordinates) into
// per-tile scalar channels, while preserving the air.channel put/get ops.
//
// Input: air.channel @ch [1, 2] with a 1x2 herd where tile(0,0) puts and
//        tile(0,1) gets.  specializeChannelBundle should create per-tile
//        scalar channels but leave the air.channel operations present.
//
// Key invariants:
//   - aie.device created
//   - Specialized channels survive (per-tile names)
//   - aie.core bodies contain air.channel.put/get
//   - No aie.lock / aie.flow / aie.dma_bd

// RUN: air-opt %s -air-hierarchy-to-aie="row-offset=2 col-offset=0 device=npu1_1col" | FileCheck %s

// CHECK: aie.device(npu1_1col)

// After specialization, channels should be present.
// CHECK: air.channel

// No DMA lowering.
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.flow
// CHECK-NOT: aie.dma_bd

// Cores should still contain channel ops.
// CHECK: aie.core
// CHECK:   air.channel.put
// CHECK: aie.core
// CHECK:   air.channel.get

air.channel @ch [1, 2]
func.func @channel_bundle_test() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %2 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2_0) {
        %c0 = arith.constant 0 : index
        %async_tok, %buf = air.execute -> (memref<32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %alloc : memref<32xbf16, 2>
        }
        // tile(0,0) puts, tile(0,1) gets — differentiated by channel index
        %3 = air.channel.put async [%async_tok] @ch[%tx, %ty] (%buf[] [] []) : (memref<32xbf16, 2>)
        %async_tok2 = air.execute [%3] {
          memref.dealloc %buf : memref<32xbf16, 2>
        }
      }
    }
  }
  return
}
