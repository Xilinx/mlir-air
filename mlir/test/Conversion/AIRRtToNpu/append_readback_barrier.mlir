//===- append_readback_barrier.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// airrt-to-npu auto-derives the append->readback ordering barrier from L3
// aliasing -- no air.await_appends / air.append_barrier frontend markers. A
// device->host S2MM drain (append) writing an L3 buffer, followed by a
// host->device MM2S readback of an overlapping range of the SAME buffer, is a
// read-after-write on shared DDR that the async dependence graph does not
// enforce. The compiler moves the appends' completion awaits before the
// readback's start.

// -----

// (1) Overlapping ranges (identical %arg0 span) -> barrier IS synthesized.
// The appends are configured/started, then their awaits are hoisted before the
// readback's start task.

// CHECK-LABEL: aie.runtime_sequence @overlap
// CHECK: aiex.dma_configure_task_for @appendK
// CHECK: aiex.dma_configure_task_for @appendV
// CHECK: %[[RB:.*]] = aiex.dma_configure_task_for @readback
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_start_task(%[[RB]])
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @readback(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @appendK(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @appendV(%shim_noc_tile_0_0, S2MM, 1)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @overlap(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "seg" : i64
    %ak = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendK} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %av = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendV} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %r = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @readback} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %ak, %av
    return
  }
}

// -----

// (2) Provably-disjoint ranges of the same buffer -> NO barrier. The append
// writes [0, 64); the readback reads [64, 128). The readback's start must NOT
// be preceded by an await of the append (no false read-after-write).

// CHECK-LABEL: aie.runtime_sequence @disjoint
// CHECK: aiex.dma_configure_task_for @appendK
// CHECK: aiex.dma_start_task
// CHECK: %[[RB:.*]] = aiex.dma_configure_task_for @readback
// CHECK-NOT: aiex.dma_await_task
// CHECK: aiex.dma_start_task(%[[RB]])
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @readback(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @appendK(%shim_noc_tile_0_0, S2MM, 0)
  } {sym_name = "seg2"}
  airrt.module_metadata{}
  func.func @disjoint(%arg0: memref<128xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "seg2" : i64
    %ak = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendK} : (i32, i64, i64, memref<128xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %r = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @readback} : (i32, i64, i64, memref<128xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %ak
    return
  }
}
