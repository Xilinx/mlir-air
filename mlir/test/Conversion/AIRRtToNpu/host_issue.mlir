//===- host_issue.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -airrt-to-npu | FileCheck %s

// A descriptor a core patches on a memtile: the sequence writes
// it first, from @__air_issue_templates, then lets the core go, once per
// iteration of the core this dispatch runs.

// CHECK-LABEL: aie.runtime_sequence @device_issue_l2
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK: {bd_id = 23 : i32}
// CHECK: %[[N:.*]] = arith.constant 3 : i32
// CHECK: aiex.set_lock(%__air_issue_ready_0_2, %[[N]]) {air.device_issue}
// CHECK-NOT: __air_issue_templates
#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @seg {
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_issue_ctrl_out_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32, 2 : i32>
    %__air_issue_out_prod_0_2 = aie.lock(%tile_0_2, 6) {init = 1 : i32, sym_name = "__air_issue_out_prod_0_2"}
    %__air_issue_out_cons_0_2 = aie.lock(%tile_0_2, 7) {init = 0 : i32, sym_name = "__air_issue_out_cons_0_2"}
    %__air_issue_ready_0_2 = aie.lock(%tile_0_2, 8) {init = 0 : i32, sym_name = "__air_issue_ready_0_2"}
    %__air_issue_started_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_issue_started_0_2"} : memref<1xi32, 2 : i32> = dense<0>
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 2 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 2 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf4 = aie.buffer(%tile_0_2) {sym_name = "buf4"} : memref<16xi32, 2 : i32>
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<256xi32, 2 : i32>
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<256xi32, 2 : i32>
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<256xi32, 2 : i32>
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<256xi32, 2 : i32>
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<8x256xi32>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<16xi32>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<4x256xi32>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf1 : memref<256xi32, 2 : i32> offset = 0 len = 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf2 : memref<256xi32, 2 : i32> offset = 0 len = 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb6
      %1 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb4:  // pred: ^bb0
      %2 = aie.dma_start(S2MM, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf4 : memref<16xi32, 2 : i32> offset = 0 len = 16) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 1, ^bb7, ^bb3)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf0 : memref<256xi32, 2 : i32> offset = 0 len = 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf3 : memref<256xi32, 2 : i32> offset = 0 len = 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb9:  // 2 preds: ^bb3, ^bb9
      aie.use_lock(%__air_issue_out_cons_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%__air_issue_ctrl_out_0_2 : memref<4xi32, 2 : i32> offset = 0 len = 3)
      aie.use_lock(%__air_issue_out_prod_0_2, Release, %c1_i32)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb3
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1024_i64 = arith.constant 1024 : i64
      %c23_i32 = arith.constant 23 : i32
      %c-2130049484_i32 = arith.constant -2130049484 : i32
      %c-2130050332_i32 = arith.constant -2130050332 : i32
      %c-2147352575_i32 = arith.constant -2147352575 : i32
      %c260096_i32 = arith.constant 260096 : i32
      %c2_i64 = arith.constant 2 : i64
      %c1_i32 = arith.constant 1 : i32
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c4 step %c2 {
        aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
        aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
        %0 = memref.load %buf4[%arg0] : memref<16xi32, 2 : i32>
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.index_cast %1 : index to i64
        %3 = memref.load %__air_issue_started_0_2[%c0] : memref<1xi32, 2 : i32>
        %4 = arith.cmpi ne, %3, %c0_i32 : i32
        scf.if %4 {
        } else {
          aie.use_lock(%__air_issue_ready_0_2, AcquireGreaterEqual, %c1_i32)
          memref.store %c1_i32, %__air_issue_started_0_2[%c0] : memref<1xi32, 2 : i32>
        }
        %5 = arith.muli %2, %c1024_i64 : i64
        %6 = arith.shrui %5, %c2_i64 : i64
        %7 = arith.trunci %6 : i64 to i32
        %8 = arith.addi %7, %c260096_i32 : i32
        aie.use_lock(%__air_issue_out_prod_0_2, AcquireGreaterEqual, %c1_i32)
        memref.store %c-2147352575_i32, %__air_issue_ctrl_out_0_2[%c0] : memref<4xi32, 2 : i32>
        memref.store %c-2130050332_i32, %__air_issue_ctrl_out_0_2[%c1] : memref<4xi32, 2 : i32>
        memref.store %8, %__air_issue_ctrl_out_0_2[%c2] : memref<4xi32, 2 : i32>
        aie.use_lock(%__air_issue_out_cons_0_2, Release, %c1_i32)
        aie.use_lock(%__air_issue_out_prod_0_2, AcquireGreaterEqual, %c1_i32)
        memref.store %c-2147352575_i32, %__air_issue_ctrl_out_0_2[%c0] : memref<4xi32, 2 : i32>
        memref.store %c-2130049484_i32, %__air_issue_ctrl_out_0_2[%c1] : memref<4xi32, 2 : i32>
        memref.store %c23_i32, %__air_issue_ctrl_out_0_2[%c2] : memref<4xi32, 2 : i32>
        aie.use_lock(%__air_issue_out_cons_0_2, Release, %c1_i32)
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
        scf.for %arg1 = %c0 to %c256 step %c16 {
          %19 = vector.transfer_read %buf0[%arg1], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
          vector.transfer_write %19, %buf1[%arg1] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
        } {loop_annotation = #loop_annotation}
        aie.use_lock(%lock_0_2, Release, %c1_i32)
        %9 = arith.addi %arg0, %c1 : index
        %10 = memref.load %buf4[%9] : memref<16xi32, 2 : i32>
        %11 = arith.index_cast %10 : i32 to index
        %12 = arith.index_cast %11 : index to i64
        %13 = memref.load %__air_issue_started_0_2[%c0] : memref<1xi32, 2 : i32>
        %14 = arith.cmpi ne, %13, %c0_i32 : i32
        scf.if %14 {
        } else {
          aie.use_lock(%__air_issue_ready_0_2, AcquireGreaterEqual, %c1_i32)
          memref.store %c1_i32, %__air_issue_started_0_2[%c0] : memref<1xi32, 2 : i32>
        }
        %15 = arith.muli %12, %c1024_i64 : i64
        %16 = arith.shrui %15, %c2_i64 : i64
        %17 = arith.trunci %16 : i64 to i32
        %18 = arith.addi %17, %c260096_i32 : i32
        aie.use_lock(%__air_issue_out_prod_0_2, AcquireGreaterEqual, %c1_i32)
        memref.store %c-2147352575_i32, %__air_issue_ctrl_out_0_2[%c0] : memref<4xi32, 2 : i32>
        memref.store %c-2130050332_i32, %__air_issue_ctrl_out_0_2[%c1] : memref<4xi32, 2 : i32>
        memref.store %18, %__air_issue_ctrl_out_0_2[%c2] : memref<4xi32, 2 : i32>
        aie.use_lock(%__air_issue_out_cons_0_2, Release, %c1_i32)
        aie.use_lock(%__air_issue_out_prod_0_2, AcquireGreaterEqual, %c1_i32)
        memref.store %c-2147352575_i32, %__air_issue_ctrl_out_0_2[%c0] : memref<4xi32, 2 : i32>
        memref.store %c-2130049484_i32, %__air_issue_ctrl_out_0_2[%c1] : memref<4xi32, 2 : i32>
        memref.store %c23_i32, %__air_issue_ctrl_out_0_2[%c2] : memref<4xi32, 2 : i32>
        aie.use_lock(%__air_issue_out_cons_0_2, Release, %c1_i32)
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
        scf.for %arg1 = %c0 to %c256 step %c16 {
          %19 = vector.transfer_read %buf3[%arg1], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
          vector.transfer_write %19, %buf2[%arg1] {in_bounds = [true]} : vector<16xi32>, memref<256xi32, 2 : i32>
        } {loop_annotation = #loop_annotation}
        aie.use_lock(%lock_0_2, Release, %c1_i32)
        aie.use_lock(%lock_0_2_4, Release, %c1_i32)
        aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_1, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "fwd", air.herd_size = array<i64: 1, 1>, stack_size = 2048 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 4 : i32}
    %lock_0_1_5 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %buf5 = aie.buffer(%mem_tile_0_1) {address = 516096 : i32, air.counted_lock, sym_name = "buf5"} : memref<8x256xi32, 1 : i32>
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 1)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %c4_i32 = arith.constant 4 : i32
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%buf5 : memref<8x256xi32, 1 : i32> offset = 0 len = 2048) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_5, Release, %c4_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    aie.runtime_sequence @__air_issue_templates() {
      %c1_i32 = arith.constant 1 : i32
      %0 = aiex.dma_configure_task(%mem_tile_0_1, MM2S, 0) {
        aie.use_lock(%lock_0_1_5, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf5 : memref<8x256xi32, 1 : i32> offset = 0 len = 256) {bd_id = 23 : i32}
        aie.use_lock(%lock_0_1, Release, %c1_i32)
        aie.end
      } {air.device_issue = 2 : i64, air.device_issue_channel = 0 : i32, air.device_issue_dir = 1 : i32, air.device_issue_iterations = 3 : i64, air.device_issue_lock = "__air_issue_ready_0_2"}
    }
    aie.shim_dma_allocation @air_YOut(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_WIn(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_IdxIn(%shim_noc_tile_0_0, MM2S, 1)
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, DMA : 1>
      aie.packet_dest<%mem_tile_0_1, TileControl : 0>
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 4 : i64, location = -1 : i64, row = -3 : i64}], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 11 : i64, location = -1 : i64, row = 0 : i64}, {channel = 0 : i64, col = 0 : i64, id = 13 : i64, location = -1 : i64, row = 0 : i64}, {channel = 3 : i64, col = 0 : i64, id = 9 : i64, location = -1 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "fwd"}
    }
  }
  func.func @device_issue_l2(%arg0: memref<8x256xi32>, %arg1: memref<16xi32>, %arg2: memref<4x256xi32>) {
    %c9_i32 = arith.constant 9 : i32
    %c11_i32 = arith.constant 11 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg3 : index to i64
      %1 = airrt.dma_memcpy_nd(%c11_i32, %0, %c0_i64, %arg2[0, 0, 0, 0], [1, 1, 1, 1024], [0, 0, 0, 1]) {chan_name = @YOut, metadata = @air_YOut} : (i32, i64, i64, memref<4x256xi32>) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c4_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 2048], [0, 0, 0, 1]) {chan_name = @WIn, metadata = @air_WIn} : (i32, i64, i64, memref<8x256xi32>) : !airrt.event
      %3 = airrt.dma_memcpy_nd(%c9_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 16], [0, 0, 0, 1]) {chan_name = @IdxIn, metadata = @air_IdxIn} : (i32, i64, i64, memref<16xi32>) : !airrt.event
      affine.for %arg4 = 0 to 1 {
        %h = airrt.herd_load "fwd" () {segment_name = "seg"} : () -> i64
      }
      airrt.wait_all %1, %2, %3 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// The host sets a memtile descriptor's length: a table with an
// entry per length, word 0 then the queue, is the sequence's last argument; a
// shim MM2S of the column reads the entry for L, a scratchpad offset.

// CHECK: module attributes {air.ctrl_bounds = array<i64: 2, 1, 0, 8>, air.ctrl_table = array<i32: 1, 17433312, 256, 1, -2130049484, 23, 1, 17433312, 512,
// CHECK-LABEL: aie.runtime_sequence @host_issue_l2
// CHECK-SAME: %{{.*}}: i32, %[[TAB:.*]]: memref<48xi32>)
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK: {bd_id = 23 : i32}
// CHECK: aiex.dma_configure_task_for @__air_ctrltab_4
// CHECK-NEXT: aie.dma_bd(%[[TAB]] : memref<48xi32>
// CHECK: aie.packet_source<%{{.*}}, DMA : 1>
// CHECK-NEXT: aie.packet_dest<%{{.*}}, TileControl : 0>
// CHECK: aie.shim_dma_allocation @__air_ctrltab_4(%{{.*}}, MM2S, 1)
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
        air.channel.issue  @KvOut[] length(%1) {air.length_bound = 8 : i64, issue_id = 4 : i64}
        %h = airrt.herd_load "fwd" (%arg2) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all %3, %4 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// The host sets an L3 put's length: its shim task is configured
// on a pinned BD and not started; the entry queues it with a completion token,
// which the host waits on before freeing it.

// CHECK: module attributes {air.ctrl_bounds = array<i64: 2, 1, 0, 8>, air.ctrl_table = array<i32: 1, 16896480, 256, 1, -2130587116, -2147483633,
// CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
// CHECK-LABEL: aie.runtime_sequence @host_issue_l3
// CHECK: %[[T:.*]] = aiex.dma_configure_task_for @air_XIn
// CHECK: {bd_id = 15 : i32}
// CHECK-NOT: aiex.dma_start_task(%[[T]])
// CHECK: aiex.npu.maskwrite32
// CHECK: aiex.dma_configure_task_for @__air_ctrltab_2
// CHECK: aiex.npu.sync
// CHECK-NEXT: aiex.dma_free_task(%[[T]])
#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aiex.scratchpad_parameter @__air_param_fwd_0 : i32
  aie.device(npu2) @seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 4) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 2) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<8x256xi32, 2 : i32>
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
      aie.dma_bd(%buf1 : memref<8x256xi32, 2 : i32> offset = 0 len = 2048) {task_id = 0 : i32}
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
      %c8 = arith.constant 8 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant dense<-1> : vector<16xi32>
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%__air_herd_lock_0_2, AcquireGreaterEqual, %c1_i32)
      %0 = aiex.read_scratchpad_parameter @__air_param_fwd_0 : i32
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c256 step %c16 {
          vector.transfer_write %cst, %buf1[%arg0, %arg1] {in_bounds = [true]} : vector<16xi32>, memref<8x256xi32, 2 : i32>
        } {loop_annotation = #loop_annotation}
      } {loop_annotation = #loop_annotation}
      %1 = arith.index_cast %0 : i32 to index
      scf.for %arg0 = %c0 to %1 step %c1 {
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
        scf.for %arg1 = %c0 to %c256 step %c16 {
          %2 = vector.transfer_read %buf0[%arg1], %c0_i32 {in_bounds = [true]} : memref<256xi32, 2 : i32>, vector<16xi32>
          vector.transfer_write %2, %buf1[%arg0, %arg1] {in_bounds = [true]} : vector<16xi32>, memref<8x256xi32, 2 : i32>
        } {loop_annotation = #loop_annotation}
        aie.use_lock(%lock_0_2, Release, %c1_i32)
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "fwd", air.herd_size = array<i64: 1, 1>, emit_parameter_sync_preamble = false, stack_size = 2048 : i32}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_YOut(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_XIn(%shim_noc_tile_0_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 3 : i64, location = -1 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 2 : i64, location = -1 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "fwd"}
    }
  }
  func.func @host_issue_l3(%arg0: memref<8x256xi32>, %arg1: memref<8x256xi32>, %arg2: i32) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg3 : index to i64
      %1 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 2048], [0, 0, 0, 1]) {chan_name = @YOut, metadata = @air_YOut} : (i32, i64, i64, memref<8x256xi32>) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 2048], [0, 0, 0, 1]) {air.device_issue = 2 : i64, air.device_issue_bd = 15 : i32, air.host_issue, chan_name = @XIn, metadata = @air_XIn} : (i32, i64, i64, memref<8x256xi32>) : !airrt.event
      affine.for %arg4 = 0 to 1 {
        %3 = arith.index_cast %arg2 : i32 to index
        air.channel.issue  @XIn[] length(%3) {air.length_bound = 8 : i64, issue_id = 2 : i64}
        %h = airrt.herd_load "fwd" (%arg2) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all %1, %2 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
