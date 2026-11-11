//===- feed_cap_across_runs.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// A shim feed channel holds at most four started tasks that are not yet
// awaited or freed. A drain await between two runs of feeds splits them, and
// the tasks of the first run still count when the second starts.

// The fifth and sixth feeds start only after the first and second are
// awaited; the awaited tasks are not freed again.
// CHECK-LABEL: aie.runtime_sequence @three_and_three
// CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @feed
// CHECK: %[[F1:.*]] = aiex.dma_configure_task_for @feed
// CHECK: %[[F2:.*]] = aiex.dma_configure_task_for @feed
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @drain
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK: %[[F3:.*]] = aiex.dma_configure_task_for @feed
// CHECK-NOT: aiex.dma_await_task
// CHECK: aiex.dma_start_task(%[[F3]])
// CHECK: %[[F4:.*]] = aiex.dma_configure_task_for @feed
// CHECK: aiex.dma_await_task(%[[F0]])
// CHECK-NEXT: aiex.dma_start_task(%[[F4]])
// CHECK: %[[F5:.*]] = aiex.dma_configure_task_for @feed
// CHECK: aiex.dma_await_task(%[[F1]])
// CHECK-NEXT: aiex.dma_start_task(%[[F5]])
// CHECK-NOT: aiex.dma_free_task(%[[F0]])
// CHECK-NOT: aiex.dma_free_task(%[[F1]])
// CHECK: aiex.dma_free_task(%[[F2]])

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feed(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @drain(%tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @three_and_three(%arg0: memref<512xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %id_feed = arith.constant 4 : i32
    %id_feed1 = arith.constant 6 : i32
    %id_drain = arith.constant 5 : i32
    %o0 = arith.constant 0 : i64
    %f0 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o1 = arith.constant 64 : i64
    %f1 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o1], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o2 = arith.constant 128 : i64
    %f2 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o2], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %g = airrt.dma_memcpy_nd(%id_drain, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %g
    %o3 = arith.constant 192 : i64
    %f3 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o3], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o4 = arith.constant 256 : i64
    %f4 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o4], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o5 = arith.constant 320 : i64
    %f5 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o5], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    airrt.wait_all %f0, %f1, %f2, %f3, %f4, %f5 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}

// -----

// Four feeds in all never reach the cap: only the drain is awaited.
// CHECK-LABEL: aie.runtime_sequence @two_and_two
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @drain
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK-NOT: aiex.dma_await_task

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feed(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @drain(%tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @two_and_two(%arg0: memref<512xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %id_feed = arith.constant 4 : i32
    %id_feed1 = arith.constant 6 : i32
    %id_drain = arith.constant 5 : i32
    %o0 = arith.constant 0 : i64
    %f0 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o1 = arith.constant 64 : i64
    %f1 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o1], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %g = airrt.dma_memcpy_nd(%id_drain, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %g
    %o2 = arith.constant 128 : i64
    %f2 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o2], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o3 = arith.constant 192 : i64
    %f3 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o3], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    airrt.wait_all %f0, %f1, %f2, %f3 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}

// -----

// Six feeds alternating over two channels, three on each: the cap is per
// channel, so neither reaches it.
// CHECK-LABEL: aie.runtime_sequence @two_channels
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @drain
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK-NOT: aiex.dma_await_task

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feed(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @feed1(%tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @drain(%tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @two_channels(%arg0: memref<512xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %id_feed = arith.constant 4 : i32
    %id_feed1 = arith.constant 6 : i32
    %id_drain = arith.constant 5 : i32
    %o0 = arith.constant 0 : i64
    %f0 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o1 = arith.constant 64 : i64
    %f1 = airrt.dma_memcpy_nd(%id_feed1, %c0, %c0, %arg0[%c0, %c0, %c0, %o1], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed1} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o2 = arith.constant 128 : i64
    %f2 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o2], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %g = airrt.dma_memcpy_nd(%id_drain, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %g
    %o3 = arith.constant 192 : i64
    %f3 = airrt.dma_memcpy_nd(%id_feed1, %c0, %c0, %arg0[%c0, %c0, %c0, %o3], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed1} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o4 = arith.constant 256 : i64
    %f4 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o4], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o5 = arith.constant 320 : i64
    %f5 = airrt.dma_memcpy_nd(%id_feed1, %c0, %c0, %arg0[%c0, %c0, %c0, %o5], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed1} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    airrt.wait_all %f0, %f1, %f2, %f3, %f4, %f5 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}

// -----

// The first run's feeds are freed with the drain's wait, before the second
// run starts, so they no longer count against the cap.
// CHECK-LABEL: aie.runtime_sequence @first_run_freed
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @drain
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK-NOT: aiex.dma_await_task

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feed(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @drain(%tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @first_run_freed(%arg0: memref<512xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %id_feed = arith.constant 4 : i32
    %id_feed1 = arith.constant 6 : i32
    %id_drain = arith.constant 5 : i32
    %o0 = arith.constant 0 : i64
    %f0 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o1 = arith.constant 64 : i64
    %f1 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o1], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o2 = arith.constant 128 : i64
    %f2 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o2], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %g = airrt.dma_memcpy_nd(%id_drain, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %g, %f0, %f1, %f2
    %o3 = arith.constant 192 : i64
    %f3 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o3], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o4 = arith.constant 256 : i64
    %f4 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o4], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    %o5 = arith.constant 320 : i64
    %f5 = airrt.dma_memcpy_nd(%id_feed, %c0, %c0, %arg0[%c0, %c0, %c0, %o5], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @feed} : (i32, i64, i64, memref<512xi32>) : !airrt.event
    airrt.wait_all %f3, %f4, %f5 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}
