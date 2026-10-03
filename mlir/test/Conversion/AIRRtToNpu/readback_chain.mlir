//===- readback_chain.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// Five jobs through one host buffer: job k drains row k and its input reads row
// k - 1. Each drain is awaited right before the input that reads it, not before
// the first input after it (which would wait on a drain whose job has not been
// fed yet), and the input channel keeps at most four tasks in flight.

// Drain 0 is awaited before input 1 starts, drain 1 before input 2, and so on.
// CHECK: aie.runtime_sequence @chain
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @air_JobOut
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @air_JobIn
// CHECK: aiex.dma_start_task(%[[F0]])
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @air_JobOut
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK: %[[F1:.*]] = aiex.dma_configure_task_for @air_JobIn
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: aiex.dma_start_task(%[[F1]])
// CHECK: %[[D2:.*]] = aiex.dma_configure_task_for @air_JobOut
// CHECK: aiex.dma_start_task(%[[D2]])
// CHECK: %[[F2:.*]] = aiex.dma_configure_task_for @air_JobIn
// CHECK: aiex.dma_await_task(%[[D1]])
// CHECK: aiex.dma_start_task(%[[F2]])
// CHECK: %[[D3:.*]] = aiex.dma_configure_task_for @air_JobOut
// CHECK: aiex.dma_start_task(%[[D3]])
// CHECK: %[[F3:.*]] = aiex.dma_configure_task_for @air_JobIn
// CHECK: aiex.dma_await_task(%[[D2]])
// CHECK: aiex.dma_start_task(%[[F3]])
// CHECK: %[[D4:.*]] = aiex.dma_configure_task_for @air_JobOut
// CHECK: aiex.dma_start_task(%[[D4]])
// The fifth input waits for the first, which has long completed.
// CHECK: %[[F4:.*]] = aiex.dma_configure_task_for @air_JobIn
// CHECK: aiex.dma_await_task(%[[D3]])
// CHECK: aiex.dma_await_task(%[[F0]])
// CHECK: aiex.dma_start_task(%[[F4]])
// CHECK: aiex.dma_await_task(%[[D4]])

#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<16xi32, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<16xi32, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<80xi32>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<16xi32>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf0 : memref<16xi32, 2 : i32> offset = 0 len = 16) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_1, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf1 : memref<16xi32, 2 : i32> offset = 0 len = 16) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = memref.load %buf1[%arg0] : memref<16xi32, 2 : i32>
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %buf0[%arg0] : memref<16xi32, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = memref.load %buf1[%arg0] : memref<16xi32, 2 : i32>
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %buf0[%arg0] : memref<16xi32, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = memref.load %buf1[%arg0] : memref<16xi32, 2 : i32>
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %buf0[%arg0] : memref<16xi32, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = memref.load %buf1[%arg0] : memref<16xi32, 2 : i32>
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %buf0[%arg0] : memref<16xi32, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %0 = memref.load %buf1[%arg0] : memref<16xi32, 2 : i32>
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %buf0[%arg0] : memref<16xi32, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "herd_0", air.herd_size = array<i64: 1, 1>, stack_size = 2048 : i32}
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.shim_dma_allocation @air_JobOut(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_JobIn(%shim_noc_tile_0_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 12 : i64, location = -1 : i64, row = 0 : i64}, {channel = 0 : i64, col = 0 : i64, id = 14 : i64, location = -1 : i64, row = 0 : i64}, {channel = 0 : i64, col = 0 : i64, id = 16 : i64, location = -1 : i64, row = 0 : i64}, {channel = 0 : i64, col = 0 : i64, id = 18 : i64, location = -1 : i64, row = 0 : i64}, {channel = 0 : i64, col = 0 : i64, id = 20 : i64, location = -1 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 11 : i64, location = -1 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 13 : i64, location = -1 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 15 : i64, location = -1 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 17 : i64, location = -1 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 19 : i64, location = -1 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "herd_0"}
    }
  }
  func.func @chain(%arg0: memref<16xi32>, %arg1: memref<80xi32>) {
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg2 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg2 : index to i64
      %1 = airrt.dma_memcpy_nd(%c12_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 16], [0, 0, 0, 1]) {air.append_barrier, chan_name = @JobOut, metadata = @air_JobOut} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c11_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 16], [0, 0, 0, 1]) {chan_name = @JobIn, metadata = @air_JobIn} : (i32, i64, i64, memref<16xi32>) : !airrt.event
      %3 = airrt.dma_memcpy_nd(%c12_i32, %0, %c0_i64, %arg1[0, 0, 0, 16], [1, 1, 1, 16], [0, 0, 0, 1]) {air.append_barrier, chan_name = @JobOut, metadata = @air_JobOut} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %4 = airrt.dma_memcpy_nd(%c11_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 16], [0, 0, 0, 1]) {air.await_appends, chan_name = @JobIn, metadata = @air_JobIn} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %5 = airrt.dma_memcpy_nd(%c12_i32, %0, %c0_i64, %arg1[0, 0, 0, 32], [1, 1, 1, 16], [0, 0, 0, 1]) {air.append_barrier, chan_name = @JobOut, metadata = @air_JobOut} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %6 = airrt.dma_memcpy_nd(%c11_i32, %0, %c0_i64, %arg1[0, 0, 0, 16], [1, 1, 1, 16], [0, 0, 0, 1]) {air.await_appends, chan_name = @JobIn, metadata = @air_JobIn} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %7 = airrt.dma_memcpy_nd(%c12_i32, %0, %c0_i64, %arg1[0, 0, 0, 48], [1, 1, 1, 16], [0, 0, 0, 1]) {air.append_barrier, chan_name = @JobOut, metadata = @air_JobOut} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %8 = airrt.dma_memcpy_nd(%c11_i32, %0, %c0_i64, %arg1[0, 0, 0, 32], [1, 1, 1, 16], [0, 0, 0, 1]) {air.await_appends, chan_name = @JobIn, metadata = @air_JobIn} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %9 = airrt.dma_memcpy_nd(%c12_i32, %0, %c0_i64, %arg1[0, 0, 0, 64], [1, 1, 1, 16], [0, 0, 0, 1]) {chan_name = @JobOut, metadata = @air_JobOut} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      %10 = airrt.dma_memcpy_nd(%c11_i32, %0, %c0_i64, %arg1[0, 0, 0, 48], [1, 1, 1, 16], [0, 0, 0, 1]) {air.await_appends, chan_name = @JobIn, metadata = @air_JobIn} : (i32, i64, i64, memref<80xi32>) : !airrt.event
      affine.for %arg3 = 0 to 1 {
        %h = airrt.herd_load "herd_0" () {segment_name = "seg"} : () -> i64
      }
      airrt.wait_all %1, %3, %5, %7, %9, %10, %8, %6, %4, %2 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
