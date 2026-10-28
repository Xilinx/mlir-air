//===- unroll_loop_selecting_transfers.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// A loop that picks, by its induction variable, a drain that no wait_all in
// the branch names is unrolled. Rolled, the drain stays in the branch and is
// never awaited, so each trip holds its buffer descriptors. Unrolled, the
// pick folds and the await of the next drain on the channel releases it.

// CHECK-LABEL: aie.runtime_sequence @pick_drain
// CHECK-NOT: scf.for
// CHECK-NOT: scf.index_switch
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg0
// CHECK: %[[B:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg2
// CHECK: aiex.dma_await_task(%[[A]])
// CHECK-NEXT: aiex.dma_await_task(%[[B]])
// CHECK: %[[C:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg1
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg2
// CHECK: aiex.dma_await_task(%[[C]])
// CHECK-NEXT: aiex.dma_await_task(%[[D]])

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

// -----

// Paced feeds (air.preserve_shim_dma_order) are not freed by a wait_all: the
// loop stays rolled.

// CHECK-LABEL: aie.runtime_sequence @pick_paced_feed
// CHECK: scf.for
// CHECK: scf.index_switch

module {
  aie.device(npu1) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %mem = aie.mem(%tile_0_2) { aie.end }
    %core = aie.core(%tile_0_2) { aie.end }
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @pick_paced_feed(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
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
        %f = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.yield
      }
      default {
        %g = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn, air.preserve_shim_dma_order} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.yield
      }
      airrt.wait_all {"air.launch_end"}
    }
    %p = airrt.segment_load "seg" : i64
    return
  }
}

// -----

// A wait_all in the branch names the feed, so the branch frees it: the loop
// stays rolled.

// CHECK-LABEL: aie.runtime_sequence @pick_waited_feed
// CHECK: scf.for
// CHECK: scf.index_switch
// CHECK: aiex.dma_free_task

module {
  aie.device(npu1) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %mem = aie.mem(%tile_0_2) { aie.end }
    %core = aie.core(%tile_0_2) { aie.end }
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @pick_waited_feed(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
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
        %f = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        airrt.wait_all %f
        scf.yield
      }
      default {
        %g = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        airrt.wait_all %g
        scf.yield
      }
      airrt.wait_all {"air.launch_end"}
    }
    %p = airrt.segment_load "seg" : i64
    return
  }
}

// -----

// A switch whose selector does not depend on the induction variable picks the
// same arm on every iteration: the loop stays rolled.

// CHECK-LABEL: aie.runtime_sequence @fixed_feed
// CHECK: scf.for

module {
  aie.device(npu1) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %mem = aie.mem(%tile_0_2) { aie.end }
    %core = aie.core(%tile_0_2) { aie.end }
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @fixed_feed(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %id = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.index_switch %c1
      case 0 {
        %f = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.yield
      }
      default {
        %g = airrt.dma_memcpy_nd(%id, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.yield
      }
      airrt.wait_all {"air.launch_end"}
    }
    %p = airrt.segment_load "seg" : i64
    return
  }
}

