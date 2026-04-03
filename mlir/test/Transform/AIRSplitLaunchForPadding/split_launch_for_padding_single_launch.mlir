//===- split_launch_for_padding_single_launch.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding='split-mode=single-launch pad-location=source' | FileCheck %s

// Tests single-launch mode: produces one launch with scf.if on block indices.
// M=300, M_TILE=64 -> launchM=5, mRem=300%64=44, interiorM=4
// N=300, N_TILE=64 -> launchN=5, nRem=300%64=44, interiorN=4
// Expected: single launch (5x5), nested scf.if tree with 4 partitions.

// CHECK-LABEL: func.func @single_launch_both_boundary
// The launch keeps the original 5x5 grid size.
// CHECK: %[[C5:.*]] = arith.constant 5 : index
// CHECK: air.launch (%[[M:.*]], %[[N:.*]]) in (%{{.*}}=%[[C5]], %{{.*}}=%[[C5]])
// CHECK:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:   %[[CMP_M:.*]] = arith.cmpi ult, %[[M]], %[[C4]] : index
// Outer scf.if: is M interior?
// CHECK:   scf.if %[[CMP_M]]
// Inner scf.if: is N interior?
// CHECK:     scf.if
// Interior body (no padding):
// CHECK:       air.dma_memcpy_nd
// CHECK-NOT:   pad_after
// CHECK:     } else {
// N-boundary body:
// CHECK:       air.dma_memcpy_nd
// CHECK:   } else {
// M-boundary or corner:
// CHECK:     scf.if
// CHECK:       air.dma_memcpy_nd
// CHECK:     } else {
// Corner body:
// CHECK:       air.dma_memcpy_nd

module {
  func.func @single_launch_both_boundary(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c5 = arith.constant 5 : index
    air.launch (%arg9, %arg10) in (%arg12=%c5, %arg13=%c5) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {air.actual_sizes = array<i64: 300, 300>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c300 = arith.constant 300 : index

      %offset_a = arith.muli %arg9, %c64 : index
      %offset_b = arith.muli %arg10, %c64 : index

      air.dma_memcpy_nd (%arg17[%offset_a, %c0] [%c64, %c64] [%c300, %c1], %arg15[%offset_a, %c0] [%c64, %c64] [%c300, %c1]) {id = 1 : i32} : (memref<*xbf16>, memref<*xbf16>)
      air.dma_memcpy_nd (%arg17[%c0, %offset_b] [%c64, %c64] [%c300, %c1], %arg16[%c0, %offset_b] [%c64, %c64] [%c300, %c1]) {id = 2 : i32} : (memref<*xbf16>, memref<*xbf16>)
    }
    return
  }
}
