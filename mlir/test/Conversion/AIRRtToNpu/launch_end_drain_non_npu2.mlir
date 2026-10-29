//===- launch_end_drain_non_npu2.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// The launch end drains the launch's transfers on every target, not only on
// the NPU2 family. Here each trip ends a launch, and the loop picks a drain by
// its induction variable in a branch where no wait_all names it. The drain
// awaits it at the end of its branch; without that, every trip would keep its
// buffer descriptors.

// CHECK-LABEL: aie.runtime_sequence @pick_drain
// CHECK: scf.for
// CHECK: scf.index_switch
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg0
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-NEXT: aiex.dma_await_task(%[[A]])
// CHECK: %[[B:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg1
// CHECK: aiex.dma_start_task(%[[B]])
// CHECK-NEXT: aiex.dma_await_task(%[[B]])
// CHECK: %[[C:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg2
// CHECK: aiex.dma_await_task(%[[C]])

module {
  aie.device(npu1) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    %mem = aie.mem(%tile_0_2) { aie.end }
    %core = aie.core(%tile_0_2) { aie.end }
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @pick_drain(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %id = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.index_switch %i
      case 0 {
        %f = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.yield
      }
      default {
        %g = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.yield
      }
      %h = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      airrt.wait_all %h
      airrt.wait_all {"air.launch_end"}
    }
    %p = airrt.segment_load "seg" : i64
    return
  }
}
