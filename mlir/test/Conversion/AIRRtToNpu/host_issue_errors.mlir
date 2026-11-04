//===- host_issue_errors.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-npu -verify-diagnostics

// A length the host sets must be one value per dispatch: every issue of a
// descriptor in a dispatch then writes the same word, so an issue that lands
// while an earlier start of that descriptor is still queued changes nothing.
// A length that varies with a loop of the sequence is refused.

#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aiex.scratchpad_parameter @__air_param_fwd_0 : i32
  aie.device(npu2) @seg {
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 4) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 2) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<256xi32, 2 : i32>
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<256xi32, 2 : i32>
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<8x256xi32>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<8x256xi32>
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf1 : memref<256xi32, 2 : i32> offset = 0 len = 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_1, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf0 : memref<256xi32, 2 : i32> offset = 0 len = 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%__air_herd_lock_0_2, AcquireGreaterEqual, %c1_i32)
      %0 = aiex.read_scratchpad_parameter @__air_param_fwd_0 : i32
      %1 = arith.index_cast %0 : i32 to index
      scf.for %arg0 = %c0 to %1 step %c1 {
        aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
        scf.for %arg1 = %c0 to %c256 step %c16 {
          %2 = vector.transfer_read %buf0[%arg1], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
          vector.transfer_write %2, %buf1[%arg1] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
        } {loop_annotation = #loop_annotation}
        aie.use_lock(%lock_0_2, Release, %c1_i32)
        aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "fwd", air.herd_size = array<i64: 1, 1>, emit_parameter_sync_preamble = false, stack_size = 2048 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_3 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %buf2 = aie.buffer(%mem_tile_0_1) {address = 516096 : i32, air.counted_lock, sym_name = "buf2"} : memref<8x256xi32, 1 : i32>
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf2 : memref<8x256xi32, 1 : i32> offset = 0 len = 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_3, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    aie.runtime_sequence @__air_issue_templates() {
      %c1_i32 = arith.constant 1 : i32
      %0 = aiex.dma_configure_task(%mem_tile_0_1, MM2S, 0) {
        aie.use_lock(%lock_0_1_3, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf2 : memref<8x256xi32, 1 : i32> offset = 0 len = 2048) {bd_id = 23 : i32}
        aie.use_lock(%lock_0_1, Release, %c1_i32)
        aie.end
      } {air.device_issue = 4 : i64, air.device_issue_channel = 0 : i32, air.device_issue_dir = 1 : i32}
    }
    aie.shim_dma_allocation @air_YOut(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_KvIn(%shim_noc_tile_0_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 2 : i64, location = -1 : i64, row = -3 : i64}], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 5 : i64, location = -1 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "fwd"}
    }
  }
  func.func @host_issue_l2(%arg0: memref<8x256xi32>, %arg1: memref<8x256xi32>, %arg2: i32) {
    %c5_i32 = arith.constant 5 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg3 : index to i64
      %1 = arith.index_cast %arg2 : i32 to index
      %2 = arith.index_cast %1 : index to i64
      %3 = airrt.dma_memcpy_nd(%c5_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, %2, 256], [0, 0, 256, 1]) {chan_name = @YOut, metadata = @air_YOut} : (i32, i64, i64, memref<8x256xi32>) : !airrt.event
      %4 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 2048], [0, 0, 0, 1]) {chan_name = @KvIn, metadata = @air_KvIn} : (i32, i64, i64, memref<8x256xi32>) : !airrt.event
      affine.for %arg4 = 0 to 1 {
        // expected-error@+1 {{length is not affine in one argument of the sequence}}
        air.channel.issue  @KvOut[] length(%arg3) {air.length_bound = 8 : i64, issue_id = 4 : i64}
        %h = airrt.herd_load "fwd" (%arg2) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all %3, %4 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
