//===- branch_transfer_wait.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s
// RUN: air-opt -airrt-to-npu="unroll-feed-loops=true" -canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=UNROLL

// A transfer in a branch (a class of waves, say) that no wait in the branch
// covers: its event leaves through the branch's result, which the sequence
// cannot wait on. It is waited for at the branch's end instead -- the drain
// awaited, the feed freed -- rather than dropped.

// CHECK-LABEL: aie.runtime_sequence @branch
// CHECK: scf.if
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @drainOut
// CHECK: aiex.dma_start_task(%[[D]])
// CHECK-NEXT: aiex.dma_await_task(%[[D]])
// CHECK-NEXT: aiex.dma_free_task(%[[D]])
// CHECK: } else {
// CHECK: %[[F:.*]] = aiex.dma_configure_task_for @feedIn
// CHECK: aiex.dma_start_task(%[[F]])
// CHECK-NEXT: aiex.dma_free_task(%[[F]])

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @drainOut(%tile_0_0, S2MM, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @branch(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %sel: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %drain = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %h = airrt.herd_load "herd_0" () {segment_name = "seg"} : () -> i64
    %cond = arith.cmpi eq, %sel, %c0_i32 : i32
    %e = scf.if %cond -> (!airrt.event) {
      %d = airrt.dma_memcpy_nd(%drain, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drainOut} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      scf.yield %d : !airrt.event
    } else {
      %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      scf.yield %f : !airrt.event
    }
    airrt.wait_all %e {"air.launch_end"}
    return
  }
}

// -----

// The same drain, but a feed follows the branch before the launch's end: the
// drain's producer may wait for that feed, and a wait in the branch would
// block the sequence before issuing it. The drain is not awaited there.

// CHECK-LABEL: aie.runtime_sequence @branch_then_feed
// CHECK: scf.if
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @drainOut
// CHECK: aiex.dma_start_task(%[[D]])
// CHECK-NOT: aiex.dma_await_task(%[[D]])
// CHECK: }
// CHECK: aiex.dma_configure_task_for @feedIn

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @drainOut(%tile_0_0, S2MM, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @branch_then_feed(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %sel: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %drain = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %h = airrt.herd_load "herd_0" () {segment_name = "seg"} : () -> i64
    %cond = arith.cmpi eq, %sel, %c0_i32 : i32
    %e = scf.if %cond -> (!airrt.event) {
      %d = airrt.dma_memcpy_nd(%drain, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drainOut} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      scf.yield %d : !airrt.event
    } else {
      %w = airrt.wait_all : !airrt.event
      scf.yield %w : !airrt.event
    }
    %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %e, %f {"air.launch_end"}
    return
  }
}

// -----

// A drain at the sequence's top level, awaited in both arms of a branch:
// under unroll-feed-loops each await frees it on its own path, not only the
// first one found.

// UNROLL-LABEL: aie.runtime_sequence @await_both_arms
// UNROLL: %[[D:.*]] = aiex.dma_configure_task_for @drainOut
// UNROLL: scf.if
// UNROLL: aiex.dma_await_task(%[[D]])
// UNROLL-NEXT: aiex.dma_free_task(%[[D]])
// UNROLL: } else {
// UNROLL: aiex.dma_await_task(%[[D]])
// UNROLL-NEXT: aiex.dma_free_task(%[[D]])

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @drainOut(%tile_0_0, S2MM, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @await_both_arms(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %sel: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %drain = arith.constant 5 : i32
    %p = airrt.segment_load "seg" : i64
    %h = airrt.herd_load "herd_0" () {segment_name = "seg"} : () -> i64
    %d = airrt.dma_memcpy_nd(%drain, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drainOut} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %cond = arith.cmpi eq, %sel, %c0_i32 : i32
    scf.if %cond {
      airrt.wait_all %d
      %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      airrt.wait_all %f
    } else {
      airrt.wait_all %d
    }
    airrt.wait_all {"air.launch_end"}
    return
  }
}
