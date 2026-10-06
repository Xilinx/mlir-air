//===- armed_drain_weave.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// air-to-std arms a launch's device->host drains ahead of all its inputs and
// marks them air.armed_drain. A drain cannot retire before the inputs that
// produce its data, so past the shim task queue depth the control program
// would wait on a drain that depends on an input issued after it: a dropped
// push, or a queue-space poll that never returns. boundShimFeedBursts weaves
// such drains back between the feeds, in proportion, and caps each channel at
// 4 in flight. Drain j - 4 is awaited ahead of drain j's configure, and the
// await that drain had at the launch end is dropped.

// Six drains, six feeds: one drain per feed round, each ahead of its feed.
// CHECK-LABEL: aie.runtime_sequence @even
// CHECK-NOT: air.armed_drain
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @even_out
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @even_in
// CHECK: aiex.dma_start_task(%[[F0]])
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @even_out
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK: %[[F1:.*]] = aiex.dma_configure_task_for @even_in
// CHECK: aiex.dma_start_task(%[[F1]])
// CHECK: %[[D2:.*]] = aiex.dma_configure_task_for @even_out
// CHECK: aiex.dma_start_task(%[[D2]])
// CHECK: %[[F2:.*]] = aiex.dma_configure_task_for @even_in
// CHECK: aiex.dma_start_task(%[[F2]])
// CHECK: %[[D3:.*]] = aiex.dma_configure_task_for @even_out
// CHECK: aiex.dma_start_task(%[[D3]])
// CHECK: %[[F3:.*]] = aiex.dma_configure_task_for @even_in
// CHECK: aiex.dma_start_task(%[[F3]])
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: %[[D4:.*]] = aiex.dma_configure_task_for @even_out
// CHECK: aiex.dma_start_task(%[[D4]])
// CHECK: %[[F4:.*]] = aiex.dma_configure_task_for @even_in
// CHECK: aiex.dma_await_task(%[[F0]])
// CHECK: aiex.dma_start_task(%[[F4]])
// CHECK: aiex.dma_await_task(%[[D1]])
// CHECK: %[[D5:.*]] = aiex.dma_configure_task_for @even_out
// CHECK: aiex.dma_start_task(%[[D5]])
// CHECK: %[[F5:.*]] = aiex.dma_configure_task_for @even_in
// CHECK: aiex.dma_await_task(%[[F1]])
// CHECK: aiex.dma_start_task(%[[F5]])
// The launch end awaits only the drains not already awaited.
// CHECK-NOT: aiex.dma_await_task(%[[D0]])
// CHECK-NOT: aiex.dma_await_task(%[[D1]])
// CHECK: aiex.dma_await_task(%[[D2]])
// CHECK-NOT: aiex.dma_await_task(%[[D0]])
// CHECK-NOT: aiex.dma_await_task(%[[D1]])
// CHECK: aiex.dma_await_task(%[[D5]])
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @even_out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @even_in(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "even_seg"}
  airrt.module_metadata{}
  func.func @even(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "even_seg" : i64
    %6 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %11 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11
    return
  }
}

// -----

// Six drains of one feed. The first four are armed ahead of it; the rest come
// after it, since each waits for a drain that needs that feed.
// CHECK-LABEL: aie.runtime_sequence @few_feeds
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @few_feeds_out
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @few_feeds_out
// CHECK: %[[D2:.*]] = aiex.dma_configure_task_for @few_feeds_out
// CHECK: %[[D3:.*]] = aiex.dma_configure_task_for @few_feeds_out
// CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @few_feeds_in
// CHECK: aiex.dma_start_task(%[[F0]])
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: %[[D4:.*]] = aiex.dma_configure_task_for @few_feeds_out
// CHECK: aiex.dma_start_task(%[[D4]])
// CHECK: aiex.dma_await_task(%[[D1]])
// CHECK: %[[D5:.*]] = aiex.dma_configure_task_for @few_feeds_out
// CHECK: aiex.dma_start_task(%[[D5]])
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @few_feeds_out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @few_feeds_in(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "few_feeds_seg"}
  airrt.module_metadata{}
  func.func @few_feeds(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "few_feeds_seg" : i64
    %6 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5, %6
    return
  }
}

// -----

// Four drains fit their channel: they stay armed ahead of every feed, and only
// the feeds are capped, exactly as without drains.
// CHECK-LABEL: aie.runtime_sequence @fits
// CHECK-COUNT-4: aiex.dma_configure_task_for @fits_out
// CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @fits_in
// CHECK-NOT: aiex.dma_configure_task_for @fits_out
// CHECK: aiex.dma_await_task(%[[F0]])
// CHECK-NOT: aiex.dma_configure_task_for @fits_out
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @fits_out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @fits_in(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "fits_seg"}
  airrt.module_metadata{}
  func.func @fits(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "fits_seg" : i64
    %4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %6 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
    return
  }
}

// -----

// Drains air-to-std did not mark keep their position.
// CHECK-LABEL: aie.runtime_sequence @unmarked
// CHECK-COUNT-6: aiex.dma_configure_task_for @unmarked_out
// CHECK: aiex.dma_configure_task_for @unmarked_in
// CHECK-NOT: aiex.dma_configure_task_for @unmarked_out
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @unmarked_out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @unmarked_in(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "unmarked_seg"}
  airrt.module_metadata{}
  func.func @unmarked(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "unmarked_seg" : i64
    %6 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %11 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @unmarked_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11
    return
  }
}

// -----

// A drain-only launch: six armed drains and no feed. Their producer needs
// nothing later in the sequence (the segment load lowers to no instruction), so
// they keep their order and gain no waits.
// CHECK-LABEL: aie.runtime_sequence @drain_only
// CHECK-COUNT-6: aiex.dma_configure_task_for @drain_only_out
// CHECK-NOT: aiex.dma_await_task
// CHECK-COUNT-6: aiex.dma_await_task
// CHECK-NOT: air.armed_drain
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @drain_only_out(%shim_noc_tile_0_0, S2MM, 0)
  } {sym_name = "drain_only_seg"}
  airrt.module_metadata{}
  func.func @drain_only(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out, air.armed_drain} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "drain_only_seg" : i64
    airrt.wait_all %0, %1, %2, %3, %4, %5
    return
  }
}
