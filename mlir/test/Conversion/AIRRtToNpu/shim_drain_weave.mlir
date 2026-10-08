//===- shim_drain_weave.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// air-to-std arms a launch's device->host drains ahead of the inputs that drive
// their producer. A drain cannot retire before the inputs that produce its
// data, so past the shim task queue depth the control program would wait on a
// drain that depends on an input issued after it: a dropped push, or a
// queue-space poll that never returns. boundShimFeedBursts weaves
// such drains back between the feeds, in proportion, and caps each channel at
// 4 in flight. Drain j - 4 is awaited ahead of drain j's configure, and the
// await that drain had at the launch end is dropped.

// Six drains, six feeds: one drain per feed round, each ahead of its feed.
// CHECK-LABEL: aie.runtime_sequence @even
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
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @even_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
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
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "few_feeds_seg" : i64
    %6 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @few_feeds_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5, %6
    return
  }
}

// -----

// Four drains fit their channel: they stay ahead of every feed, and only
// the feeds are capped, exactly as without drains.
// CHECK-LABEL: aie.runtime_sequence @fits
// CHECK: aiex.dma_configure_task_for @fits_out
// CHECK-NEXT: aie.dma_bd
// CHECK-NEXT: aie.end
// CHECK-NEXT: }
// CHECK-NEXT: aiex.dma_start_task
// CHECK-COUNT-3: aiex.dma_configure_task_for @fits_out
// CHECK-NOT: aiex.dma_configure_task_for @fits_in
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
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @fits_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
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

// A drain-only launch: six drains and no feed. Their producer needs
// nothing later in the sequence, so waiting for drain j - 4 before drain j
// always retires, and the channel is capped at 4 in flight.
// CHECK-LABEL: aie.runtime_sequence @drain_only
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @drain_only_out
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @drain_only_out
// CHECK-COUNT-2: aiex.dma_configure_task_for @drain_only_out
// CHECK-NOT: aiex.dma_configure_task_for
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: aiex.dma_configure_task_for @drain_only_out
// CHECK-NOT: aiex.dma_configure_task_for
// CHECK: aiex.dma_await_task(%[[D1]])
// CHECK-NEXT: aiex.dma_configure_task_for @drain_only_out
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
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drain_only_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "drain_only_seg" : i64
    airrt.wait_all %0, %1, %2, %3, %4, %5
    return
  }
}

// -----

// A launch emitted channel by channel, with a drain between channel A's
// feeds and a feed of A that reads that drain back. The drain stays in the
// burst, armed at the round of the feed it preceded, so channel B's feeds are
// still woven in rather than left behind every task of A; the drain's await
// travels with the feed that reads it, last in its round.
// CHECK-LABEL: aie.runtime_sequence @mid
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @mid_a
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @mid_b
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @mid_a
// CHECK: %[[B1:.*]] = aiex.dma_configure_task_for @mid_b
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @mid_out
// CHECK: aiex.dma_start_task(%[[D]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @mid_a
// CHECK: %[[B2:.*]] = aiex.dma_configure_task_for @mid_b
// CHECK: %[[A3:.*]] = aiex.dma_configure_task_for @mid_a
// CHECK: %[[B3:.*]] = aiex.dma_configure_task_for @mid_b
// CHECK: %[[B4:.*]] = aiex.dma_configure_task_for @mid_b
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK: %[[A4:.*]] = aiex.dma_configure_task_for @mid_a
// CHECK-NOT: aiex.dma_await_task(%[[D]])
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @mid_out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @mid_a(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @mid_b(%shim_noc_tile_1_0, MM2S, 0)
  } {sym_name = "mid_seg"}
  airrt.module_metadata{}
  func.func @mid(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "mid_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %2
    %5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %6 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @mid_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %3, %4, %5, %6, %7, %8, %9, %10
    return
  }
}

// -----

// Three drains ahead of the feeds and three between them, on one channel. The
// cap counts them all, in the order the channel fills them: drain 4 waits for
// drain 0, drain 5 for drain 1, so no more than 4 are in flight.
// CHECK-LABEL: aie.runtime_sequence @lead_mid
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @lm_out
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @lm_out
// CHECK: %[[D2:.*]] = aiex.dma_configure_task_for @lm_out
// CHECK: %[[D3:.*]] = aiex.dma_configure_task_for @lm_out
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: %[[D4:.*]] = aiex.dma_configure_task_for @lm_out
// CHECK: aiex.dma_await_task(%[[D1]])
// CHECK-NEXT: %[[D5:.*]] = aiex.dma_configure_task_for @lm_out
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @lm_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @lm_in(%t0, MM2S, 0)
    aie.shim_dma_allocation @lm_in2(%t1, MM2S, 0)
  } {sym_name = "lm_seg"}
  airrt.module_metadata{}
  func.func @lead_mid(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "lm_seg" : i64
    %3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %6 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %11 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %12 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %13 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %14 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %15 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %16 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %17 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @lm_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17
    return
  }
}
