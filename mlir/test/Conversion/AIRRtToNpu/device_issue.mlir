//===- device_issue.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-npu -canonicalize -split-input-file | FileCheck %s

// A core-started put is armed on its pinned BD but not started. Arms written
// back to back are released to the core with one set_lock, after the channel
// is told which controller id its tokens go under; each is then awaited on
// the channel and freed. The core's task-queue word gets the repeat count the
// BD was shaped with: 3 for the first put's outer dimension of 4.

// CHECK: aie.core
// CHECK-DAG: %[[Q0:.*]] = arith.constant -2147287025 : i32
// CHECK-DAG: %[[Q1:.*]] = arith.constant -2147483634 : i32
// CHECK: memref.store %[[Q0]], %{{.*}}[%{{.*}}] {air.device_issue = 0 : i64}
// CHECK: memref.store %[[Q1]], %{{.*}}[%{{.*}}] {air.device_issue = 1 : i64}

// CHECK-LABEL: aie.runtime_sequence @moe
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[MASK:.*]] = arith.constant 7936 : i32
// CHECK-DAG: %[[ID:.*]] = arith.constant 3840 : i32
// CHECK-DAG: %[[CTRL:.*]] = arith.constant 33673744 : i32
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @air_Wexp {
// CHECK: aie.dma_bd({{.*}}) {bd_id = 15 : i32}
// CHECK: } {air.device_issue_lock = "__air_device_issue_lock_0_2", issue_token = true, repeat_count = 3 : i32}
// CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @air_Wexp {
// CHECK: aie.dma_bd({{.*}}) {bd_id = 14 : i32}
// CHECK-NOT: aiex.dma_start_task
// Shim (1, 0) MM2S0 control register 0x201D210: controller id 15 in [12:8].
// CHECK: aiex.npu.maskwrite32(%[[CTRL]], %[[ID]], %[[MASK]])
// CHECK: aiex.set_lock(%{{.*}}, %[[TWO]]) {air.device_issue}
// CHECK: aiex.npu.sync
// CHECK: aiex.dma_free_task(%[[T0]])
// CHECK: aiex.npu.sync
// CHECK: aiex.dma_free_task(%[[T1]])
// CHECK-NOT: aiex.dma_start_task
module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      %q14 = arith.constant -2147483634 : i32
      memref.store %q14, %ctrl[%c2] {air.device_issue = 1 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @moe(%arg0: memref<32x256xbf16>) {
    %c2_i32 = arith.constant 2 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg2 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg2 : index to i64
      %1 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [4, 1, 1, 256], [512, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 256], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 1 : i64, air.device_issue_bd = 14 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      affine.for %arg3 = 0 to 1 {
        %h = airrt.herd_load "router" () {segment_name = "seg"} : () -> i64
      }
      airrt.wait_all %1, %2 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// In a rolled launch loop each iteration arms the pinned BD again, so its task
// is retired inside the iteration. A host put that follows on the same channel
// waits for the core-started task first: it must not overtake it in the queue.

// CHECK-LABEL: aie.runtime_sequence @rearm
// CHECK: scf.for
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @air_Wexp {
// CHECK: aie.dma_bd({{.*}}) {bd_id = 15 : i32}
// CHECK: aiex.set_lock(%{{.*}}, %{{.*}}) {air.device_issue}
// CHECK-NEXT: aiex.npu.sync
// CHECK-NEXT: aiex.dma_free_task(%[[T0]])
// CHECK-NEXT: %[[T1:.*]] = aiex.dma_configure_task_for @air_Wexp {
// CHECK-NEXT: aie.dma_bd
// CHECK-NEXT: aie.end
// CHECK-NEXT: }
// CHECK-NEXT: aiex.dma_start_task(%[[T1]])
// CHECK-NEXT: aiex.dma_free_task(%[[T1]])
// CHECK-NEXT: }
module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @rearm(%arg0: memref<32x256xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %w = %c0 to %c3 step %c1 {
      %c2_i32 = arith.constant 2 : i32
      %c0_i64 = arith.constant 0 : i64
      affine.for %arg2 = 0 to 1 {
        %p = airrt.segment_load "seg" : i64
        %0 = arith.index_cast %arg2 : index to i64
        %1 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
        affine.for %arg3 = 0 to 1 {
          %h = airrt.herd_load "router" () {segment_name = "seg"} : () -> i64
        }
        %2 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 256], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
        airrt.wait_all %1, %2 {air.launch_end}
      } {affine_opt_label = "tiling"}
    }
    return
  }
}

// -----

// Host feeds on another channel go ahead of the release of the core that
// starts @air_Wexp: releasing it at the arm would hold every later feed behind
// it. Five feeds overrun that channel's task queue, so a wait is added before
// the fifth; the release comes before that wait.

// CHECK-LABEL: aie.runtime_sequence @release_before_wait
// CHECK: aiex.dma_configure_task_for @air_Wexp
// CHECK: aiex.dma_configure_task_for @air_In
// CHECK: aiex.dma_start_task
// CHECK-NOT: aiex.dma_await_task
// CHECK: aiex.set_lock(%{{.*}}, %{{.*}}) {air.device_issue}
// CHECK: aiex.dma_await_task
module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_In(%shim_noc_tile_1_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @release_before_wait(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>) {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg2 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg2 : index to i64
      %1 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      %3 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 256], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      %4 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 512], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      %5 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 768], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      %6 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 1024], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      affine.for %arg3 = 0 to 1 {
        %h = airrt.herd_load "router" () {segment_name = "seg"} : () -> i64
      }
      airrt.wait_all %1, %2, %3, %4, %5, %6 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
