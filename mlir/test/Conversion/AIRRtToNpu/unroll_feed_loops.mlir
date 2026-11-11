//===- unroll_feed_loops.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=ROLLED
// RUN: air-opt -airrt-to-npu="unroll-feed-loops=true" -canonicalize -cse --split-input-file %s | FileCheck %s
// RUN: air-opt -airrt-to-npu="unroll-feed-loops=true merge-contiguous-feeds=true" -canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=MERGE

// A host loop of waves (one dispatch running a herd several times) frees each
// feed right after its start. Rolled, the next iteration's configure takes the
// descriptor while the feed may still be queued. With unroll-feed-loops the
// loop is unrolled and each feed is freed at the end of its wave. The RTP
// write, hoisted above the chain that computes its value from the wave index,
// still gets each wave's value (0, 0, 1).

// ROLLED-LABEL: aie.runtime_sequence @waves
// ROLLED: scf.for
// ROLLED: aiex.npu.rtp_write
// ROLLED: aiex.dma_start_task(%[[T:.*]])
// ROLLED-NEXT: aiex.dma_free_task(%[[T]])

// CHECK-LABEL: aie.runtime_sequence @waves
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NOT: scf.for
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %[[ZERO]])
// CHECK: aie.dma_bd(%arg0 : memref<192xi32> offset = 0 len = 64
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %[[ZERO]])
// CHECK: aie.dma_bd(%arg0 : memref<192xi32> offset = 64 len = 64
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %[[ONE]])
// CHECK: aie.dma_bd(%arg0 : memref<192xi32> offset = 128 len = 64
// CHECK-NOT: air.unrolled_trip

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @waves(%arg0: memref<192xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    scf.for %w = %c0 to %c3 step %c1 {
      %p = airrt.segment_load "seg" : i64
      %wi = arith.index_cast %w : index to i64
      %off = arith.muli %wi, %c64_i64 : i64
      %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<192xi32>) : !airrt.event
      %lm = arith.cmpi sge, %w, %c2 : index
      %arm = arith.extui %lm : i1 to i32
      %h = airrt.herd_load "herd_0" (%arm) {segment_name = "seg"} : (i32) -> i64
      airrt.wait_all %f {"air.launch_end"}
    }
    return
  }
}

// -----

// The phase loop's bounds select on the wave's class (two phases in the first
// waves, the third phase alone in the last): they are constants only once the
// wave loop is unrolled, and the phase loop is unrolled then too. Its feeds
// belong to the wave: they are freed after the wave's last feed, not after
// their own start. With merge-contiguous-feeds, a wave's feeds that continue
// one another in host memory are one task; the next wave's are not merged
// across the RTP write between them.

// ROLLED-LABEL: aie.runtime_sequence @phases
// ROLLED: scf.for
// ROLLED: scf.for

// CHECK-LABEL: aie.runtime_sequence @phases
// CHECK-NOT: scf.for
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @feedIn
// CHECK-NEXT: aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64
// CHECK: %[[B:.*]] = aiex.dma_configure_task_for @feedIn
// CHECK-NEXT: aie.dma_bd(%arg0 : memref<512xi32> offset = 64 len = 64
// CHECK: aiex.dma_start_task(%[[B]])
// CHECK-NEXT: aiex.dma_free_task(%[[A]])
// CHECK-NEXT: aiex.dma_free_task(%[[B]])
// CHECK: aie.dma_bd(%arg0 : memref<512xi32> offset = 128 len = 64
// CHECK: aie.dma_bd(%arg0 : memref<512xi32> offset = 192 len = 64
// CHECK: aie.dma_bd(%arg0 : memref<512xi32> offset = 384 len = 64

// MERGE-LABEL: aie.runtime_sequence @phases
// MERGE: aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 128
// MERGE: aie.dma_bd(%arg0 : memref<512xi32> offset = 128 len = 128
// MERGE: aie.dma_bd(%arg0 : memref<512xi32> offset = 384 len = 64
// MERGE-NOT: aie.dma_bd

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @phases(%arg0: memref<512xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %feed = arith.constant 4 : i32
    scf.for %w = %c0 to %c3 step %c1 {
      %p = airrt.segment_load "seg" : i64
      %lm = arith.cmpi sge, %w, %c2 : index
      %lo = arith.select %lm, %c2, %c0 : index
      %hi = arith.select %lm, %c3, %c2 : index
      %wi = arith.index_cast %w : index to i64
      %base = arith.muli %wi, %c128_i64 : i64
      scf.for %ph = %lo to %hi step %c1 {
        %phi = arith.index_cast %ph : index to i64
        %po = arith.muli %phi, %c64_i64 : i64
        %off = arith.addi %base, %po : i64
        %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<512xi32>) : !airrt.event
      }
      %arm = arith.extui %lm : i1 to i32
      %h = airrt.herd_load "herd_0" (%arm) {segment_name = "seg"} : (i32) -> i64
      %e = airrt.wait_all {"air.launch_end"} : !airrt.event
    }
    return
  }
}

// -----

// Each wave awaits its drain. Unrolled, the waves' tasks are at the top level,
// where the run-time BD pool (in use: a run-time loop follows) returns a
// descriptor only at a free: each awaited drain is freed after its await, and
// each wave's feed at the end of its wave -- before the run-time loop, not at
// the end of the sequence.

// CHECK-LABEL: aie.runtime_sequence @drains
// MERGE-LABEL: aie.runtime_sequence @drains
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @drainOut
// CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @feedIn
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: aiex.dma_free_task(%[[D0]])
// CHECK-NEXT: aiex.dma_free_task(%[[F0]])
// CHECK: aiex.dma_configure_task_for @drainOut
// CHECK: %[[D2:.*]] = aiex.dma_configure_task_for @drainOut
// CHECK: %[[F2:.*]] = aiex.dma_configure_task_for @feedIn
// CHECK: aiex.dma_await_task(%[[D2]])
// CHECK-NEXT: aiex.dma_free_task(%[[D2]])
// CHECK-NEXT: aiex.dma_free_task(%[[F2]])
// CHECK-NEXT: scf.for

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
  func.func @drains(%arg0: memref<192xi32>, %arg1: memref<192xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %drain = arith.constant 5 : i32
    scf.for %w = %c0 to %c3 step %c1 {
      %p = airrt.segment_load "seg" : i64
      %wi = arith.index_cast %w : index to i64
      %off = arith.muli %wi, %c64_i64 : i64
      %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<192xi32>) : !airrt.event
      %d = airrt.dma_memcpy_nd(%drain, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @drainOut} : (i32, i64, i64, memref<192xi32>) : !airrt.event
      %h = airrt.herd_load "herd_0" () {segment_name = "seg"} : () -> i64
      airrt.wait_all %d {"air.launch_end"}
    }
    scf.for %r = %c0 to %n step %c1 {
      %f2 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<192xi32>) : !airrt.event
      airrt.wait_all %f2
    }
    return
  }
}

// -----

// A shim channel that sources a packet flow: each task is one packet, whose
// end ends a BD on the receiving side, so contiguous feeds stay two tasks.

// MERGE-LABEL: aie.runtime_sequence @packets
// MERGE: aie.dma_bd(%arg0 : memref<128xi32> offset = 0 len = 64
// MERGE: aie.dma_bd(%arg0 : memref<128xi32> offset = 64 len = 64

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @packets(%arg0: memref<128xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %p = airrt.segment_load "seg" : i64
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    %f1 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    %h = airrt.herd_load "herd_0" () {segment_name = "seg"} : () -> i64
    %e = airrt.wait_all %f0, %f1 {"air.launch_end"} : !airrt.event
    return
  }
}
