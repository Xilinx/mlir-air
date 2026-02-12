//===- merge_unrolled_devices.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-merge-unrolled-devices | FileCheck %s

// Test 1: Basic device merging - verify devices are merged and tiles are offset
// CHECK-LABEL: module
// CHECK: aie.device(npu2) @segment_with_unroll
// CHECK-NOT: aie.device(npu2_4col) @segment_with_unroll_0_0
// CHECK-NOT: aie.device(npu2_4col) @segment_with_unroll_1_0
// CHECK-DAG: aie.tile(0, 0)
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(4, 0)
// CHECK-DAG: aie.tile(4, 2)

// Test 2: Verify that symbol names are made unique with _unroll_N suffix
// (except ShimDMAAllocationOp which already contains unroll coordinates)
// CHECK-DAG: aie.buffer({{.*}}) {sym_name = "buf1_0_unroll_0"}
// CHECK-DAG: aie.buffer({{.*}}) {sym_name = "buf0_0_unroll_0"}
// CHECK-DAG: aie.buffer({{.*}}) {sym_name = "buf3_1_unroll_1"}
// CHECK-DAG: aie.buffer({{.*}}) {sym_name = "buf2_1_unroll_1"}

// Test 3: Verify air.channel symbols are renamed with _unroll_N suffix
// CHECK-DAG: air.channel @channel_2_unroll_0
// CHECK-DAG: air.channel @channel_0_unroll_0
// CHECK-DAG: air.channel @channel_3_unroll_1
// CHECK-DAG: air.channel @channel_1_unroll_1

// Test 4: Verify ShimDMAAllocationOp names are NOT renamed (they already have unique suffixes)
// CHECK-DAG: aie.shim_dma_allocation @air_ChanOut_0
// CHECK-DAG: aie.shim_dma_allocation @air_ChanIn_0
// CHECK-DAG: aie.shim_dma_allocation @air_ChanOut_1
// CHECK-DAG: aie.shim_dma_allocation @air_ChanIn_1

// Test 5: Verify airrt.segment_metadata is merged
// CHECK: airrt.segment_metadata attributes {{{.*}}sym_name = "segment_with_unroll"}
// CHECK-NOT: airrt.segment_metadata attributes {{{.*}}sym_name = "segment_with_unroll_0_0"}
// CHECK-NOT: airrt.segment_metadata attributes {{{.*}}sym_name = "segment_with_unroll_1_0"}

#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2_4col) @segment_with_unroll_0_0 {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 4) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 2) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1_0"} : memref<32xi32, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0_0"} : memref<32xi32, 2 : i32> 
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2_0"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<32xi32, 2 : i32>, 0, 32) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    ^bb3:
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<32xi32, 2 : i32>, 0, 32) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c10_i32 = arith.constant 10 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.use_lock(%__air_herd_lock_0_2, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c32 step %c1 {
        %0 = memref.load %buf1[%arg0] : memref<32xi32, 2 : i32>
        %1 = arith.addi %0, %c10_i32 : i32
        memref.store %1, %buf0[%arg0] : memref<32xi32, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_2, Release, 1)
      cf.br ^bb1
    }
    air.channel @channel_2 [1, 1]
    air.channel @channel_0 [1, 1]
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_ChanOut_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_ChanIn_0(%shim_noc_tile_0_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 0 : i64, segment_unroll_y = 0 : i64}
  aie.device(npu2_4col) @segment_with_unroll_1_0 {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 4) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 2) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3_1"} : memref<32xi32, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2_1"} : memref<32xi32, 2 : i32> 
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2_1"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<32xi32, 2 : i32>, 0, 32) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    ^bb3:
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<32xi32, 2 : i32>, 0, 32) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c10_i32 = arith.constant 10 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.use_lock(%__air_herd_lock_0_2, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c32 step %c1 {
        %0 = memref.load %buf3[%arg0] : memref<32xi32, 2 : i32>
        %1 = arith.addi %0, %c10_i32 : i32
        memref.store %1, %buf2[%arg0] : memref<32xi32, 2 : i32>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_2, Release, 1)
      cf.br ^bb1
    }
    air.channel @channel_3 [1, 1]
    air.channel @channel_1 [1, 1]
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @air_ChanOut_1(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_ChanIn_1(%shim_noc_tile_0_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 1 : i64, segment_unroll_y = 0 : i64}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "segment_with_unroll_0_0"}{
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 6 : i64, location = 0 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "compute_herd"}
    }
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "segment_with_unroll_1_0"}{
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 6 : i64, location = 0 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "compute_herd"}
    }
  }
}
