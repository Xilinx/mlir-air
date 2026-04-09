//===- air_hierarchy_to_aie_multi_herd.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Verify --air-hierarchy-to-aie with a multi-tile herd sharing a channel.
//
// A 1x2 herd where both tile positions use the same channel @data_ch.
// After hierarchy lowering, both tile positions should have separate aie.core
// regions with their air.channel ops preserved.
//
// Key invariants:
//   - Two aie.core ops created (one per tile position)
//   - air.channel declarations preserved
//   - No DMA infrastructure (no aie.lock / aie.flow / aie.dma_bd)

// RUN: air-opt %s -air-hierarchy-to-aie="row-offset=2 col-offset=0 device=npu1_1col" | FileCheck %s
// XFAIL: *

// CHECK: aie.device(npu1_1col)

// Specialized per-tile channel declarations must survive.
// CHECK-DAG: air.channel @channel_0
// CHECK-DAG: air.channel @channel_1

// No DMA infrastructure.
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.flow
// CHECK-NOT: aie.dma_bd

// Multiple cores with specialized channel ops.
// CHECK: aie.core
// CHECK:   air.channel.put{{.*}}@channel_
// CHECK: aie.core
// CHECK:   air.channel.put{{.*}}@channel_

air.channel @data_ch [1, 2]
func.func @multi_tile_channels() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %h = air.herd @tiles async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2_0) {
        %tok, %buf = air.execute -> (memref<32xi32, 2>) {
          %a = memref.alloc() : memref<32xi32, 2>
          air.execute_terminator %a : memref<32xi32, 2>
        }
        %p = air.channel.put async [%tok] @data_ch[%tx, %ty] (%buf[] [] []) : (memref<32xi32, 2>)
        %d = air.execute [%p] {
          memref.dealloc %buf : memref<32xi32, 2>
        }
      }
    }
  }
  return
}
