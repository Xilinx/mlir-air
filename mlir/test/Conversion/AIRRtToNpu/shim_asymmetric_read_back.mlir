//===- shim_asymmetric_read_back.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// Two channels with very different task counts. The awaiting feed follows
// every feed emitted before it; both output drains start before their awaits.
// CHECK-LABEL: aie.runtime_sequence @asym_read_back
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @asym_a
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK-NEXT: %[[D0:.*]] = aiex.dma_configure_task_for @asym_out
// CHECK: aiex.dma_start_task(%[[D0]])
// CHECK-NEXT: %[[A1:.*]] = aiex.dma_configure_task_for @asym_a
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK-NEXT: %[[D1:.*]] = aiex.dma_configure_task_for @asym_out
// CHECK: aiex.dma_start_task(%[[D1]])
// CHECK-NEXT: %[[A2:.*]] = aiex.dma_configure_task_for @asym_a
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK-NEXT: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: aiex.dma_await_task(%[[D1]])
// CHECK-NEXT: %[[F:.*]] = aiex.dma_configure_task_for @asym_fa
// CHECK: aiex.dma_start_task(%[[F]])
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t3 = aie.tile(3, 0)
    %t4 = aie.tile(4, 0)
    aie.shim_dma_allocation @asym_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @asym_a(%t3, MM2S, 0)
    aie.shim_dma_allocation @asym_fa(%t4, MM2S, 0)
  } {sym_name = "asym_seg"}
  airrt.module_metadata{}
  func.func @asym_read_back(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "asym_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %3
    %5 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_fa} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %4, %5
    return
  }
}
