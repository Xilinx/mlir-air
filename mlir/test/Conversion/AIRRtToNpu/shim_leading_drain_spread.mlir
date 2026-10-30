//===- shim_read_back_caps.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// Fifteen drains lead a long run of feeds on another channel, and an early feed
// of a third channel reads all of them back. The leading drains are spread over
// the rounds of the run, but only as far as that read-back wait allows: the
// drains are all armed before it, the wait follows the six inputs before the
// feed, and the run is not rejected.

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @D(%t0, S2MM, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @A(%t1, MM2S, 0)
    %t2 = aie.tile(2, 0)
    aie.shim_dma_allocation @B(%t2, MM2S, 0)
    %t3 = aie.tile(3, 0)
    aie.shim_dma_allocation @C(%t3, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  // CHECK-LABEL: aie.runtime_sequence @leading_drains
  // CHECK-COUNT-6: dma_configure_task_for @A
  // CHECK: dma_configure_task_for @B
  func.func @leading_drains(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c2i = arith.constant 2 : i32
    %c3i = arith.constant 3 : i32
    %c4i = arith.constant 4 : i32
    %p = airrt.segment_load "seg" : i64
    %D0 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D1 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D2 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D3 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D4 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D5 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D6 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D7 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D8 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D9 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D10 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D11 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D12 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D13 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %D14 = airrt.dma_memcpy_nd(%c3i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @D} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A0 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A1 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A2 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A3 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A4 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A5 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %D1, %D2, %D3, %D4, %D5, %D6, %D7, %D8, %D9, %D10, %D11, %D12, %D13, %D14
    %B0 = airrt.dma_memcpy_nd(%c4i, %c0, %c0, %arg1[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @B} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A6 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A7 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A8 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A9 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A10 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A11 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A12 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A13 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A14 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A15 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A16 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A17 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A18 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A19 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A20 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A21 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A22 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A23 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A24 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A25 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A26 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A27 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A28 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A29 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A30 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A31 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A32 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A33 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A34 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A35 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A36 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A37 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A38 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A39 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A40 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A41 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A42 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A43 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A44 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A45 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A46 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A47 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A48 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A49 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A50 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A51 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A52 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A53 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A54 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A55 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A56 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A57 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A58 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A59 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A60 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A61 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A62 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %A63 = airrt.dma_memcpy_nd(%c2i, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c64], [%c0, %c0, %c0, %c1]) {metadata = @A} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %D0, %D1, %D2, %D3, %D4, %D5, %D6, %D7, %D8, %D9, %D10, %D11, %D12, %D13, %D14, %A0, %A1, %A2, %A3, %A4, %A5, %A6, %A7, %A8, %A9, %A10, %A11, %A12, %A13, %A14, %A15, %A16, %A17, %A18, %A19, %A20, %A21, %A22, %A23, %A24, %A25, %A26, %A27, %A28, %A29, %A30, %A31, %A32, %A33, %A34, %A35, %A36, %A37, %A38, %A39, %A40, %A41, %A42, %A43, %A44, %A45, %A46, %A47, %A48, %A49, %A50, %A51, %A52, %A53, %A54, %A55, %A56, %A57, %A58, %A59, %A60, %A61, %A62, %A63, %B0
    return
  }
}
