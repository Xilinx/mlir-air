//===- air_hierarchy_to_aie_segment.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Verify that --air-hierarchy-to-aie preserves air.channel declarations and
// put/get ops when air.segment has both L2 memory (memtile) and L1 (herd).
//
// This tests the segment → memtile lowering + channel preservation path.
// Key invariants:
//   - aie.device created
//   - aie.tile created for compute and memtile rows
//   - aie.core created from air.herd
//   - air.channel declarations survive inside aie.device
//   - air.channel.put/get survive inside aie.core bodies
//   - No aie.lock / aie.flow / aie.dma_bd (those come from Pass C)

// RUN: air-opt %s -air-hierarchy-to-aie="row-offset=2 col-offset=0 device=npu1_1col" | FileCheck %s

// CHECK: aie.device(npu1_1col)

// Channels must survive.
// CHECK:   air.channel @inA
// CHECK:   air.channel @outC

// No DMA lowering yet.
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.flow
// CHECK-NOT: aie.dma_bd

// Core should contain channel ops.
// CHECK:   aie.core
// CHECK:     air.channel.get{{.*}}@inA
// CHECK:     air.channel.put{{.*}}@outC

air.channel @inA [1, 1]
air.channel @outC [1, 1]
func.func @segment_with_channels(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%a0=%arg0, %a1=%arg1) : memref<64xi32>, memref<64xi32> {
    %1 = air.segment async args(%sa0=%a0, %sa1=%a1) : memref<64xi32>, memref<64xi32> {
      // L2 alloc simulates memtile buffer
      %async_tok, %l2_buf = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      }
      // Shim → L2 via channel (put side outside herd)
      %ch_tok = air.channel.put async [%async_tok] @inA[] (%l2_buf[] [] []) : (memref<64xi32, 1>)

      %c1_0 = arith.constant 1 : index
      %2 = air.herd @compute async [%ch_tok] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %async_tok2, %l1_buf = air.execute -> (memref<64xi32, 2>) {
          %a = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %a : memref<64xi32, 2>
        }
        // L2 → L1 via channel (get inside core)
        %g = air.channel.get async [%async_tok2] @inA[] (%l1_buf[] [] []) : (memref<64xi32, 2>)
        // L1 → L2 via channel (put inside core)
        %p = air.channel.put async [%g] @outC[] (%l1_buf[] [] []) : (memref<64xi32, 2>)
        %dealloc = air.execute [%p] {
          memref.dealloc %l1_buf : memref<64xi32, 2>
        }
      }
    }
  }
  return
}
