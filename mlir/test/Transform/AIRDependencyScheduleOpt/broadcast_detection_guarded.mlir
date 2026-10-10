//===- broadcast_detection_guarded.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-broadcast-detection --split-input-file | FileCheck %s

// The L2 to L1 DMA reads the same region on every core, but it sits under
// `ty == 3`, so only one core runs it: it is not a broadcast.

// CHECK-LABEL: func.func @guarded
// CHECK: scf.if
// CHECK: air.dma_memcpy_nd
// CHECK-NOT: broadcast_pattern
// CHECK-SAME: pad_after
// CHECK: } else {

func.func @guarded(%arg0: memref<16x960xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%a=%arg0) : memref<16x960xbf16> {
    air.segment @seg args(%sa=%a) : memref<16x960xbf16> {
      %c1_s = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %l2 = memref.alloc() : memref<16x960xbf16, 1>
      air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<16x960xbf16, 1>, memref<16x960xbf16>)
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_s, %sy=%c4) args(%h2=%l2) : memref<16x960xbf16, 1> {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %c16 = arith.constant 16 : index
        %c192 = arith.constant 192 : index
        %c256 = arith.constant 256 : index
        %c960 = arith.constant 960 : index
        %c768 = arith.constant 768 : index
        %l1 = memref.alloc() : memref<16x256xbf16, 2>
        %off = arith.muli %ty, %c256 : index
        %last = arith.cmpi eq, %ty, %c3 : index
        scf.if %last {
          air.dma_memcpy_nd (%l1[] [] [], %h2[%c0, %c768] [%c16, %c192] [%c960, %c1_h]) {pad_before = array<i32: 0, 0>, pad_after = array<i32: 0, 64>} : (memref<16x256xbf16, 2>, memref<16x960xbf16, 1>)
        } else {
          air.dma_memcpy_nd (%l1[] [] [], %h2[%c0, %off] [%c16, %c256] [%c960, %c1_h]) : (memref<16x256xbf16, 2>, memref<16x960xbf16, 1>)
        }
        memref.dealloc %l1 : memref<16x256xbf16, 2>
      }
      memref.dealloc %l2 : memref<16x960xbf16, 1>
    }
  }
  return
}

// -----

// The same DMA without the condition runs on all four cores of the column and
// is broadcast to them.

// CHECK-LABEL: func.func @unguarded
// CHECK: air.dma_memcpy_nd {{.*}}broadcast_pattern

func.func @unguarded(%arg0: memref<16x960xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%a=%arg0) : memref<16x960xbf16> {
    air.segment @seg args(%sa=%a) : memref<16x960xbf16> {
      %c1_s = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %l2 = memref.alloc() : memref<16x960xbf16, 1>
      air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<16x960xbf16, 1>, memref<16x960xbf16>)
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_s, %sy=%c4) args(%h2=%l2) : memref<16x960xbf16, 1> {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %c16 = arith.constant 16 : index
        %c192 = arith.constant 192 : index
        %c256 = arith.constant 256 : index
        %c960 = arith.constant 960 : index
        %c768 = arith.constant 768 : index
        %l1 = memref.alloc() : memref<16x256xbf16, 2>
        %off = arith.muli %ty, %c256 : index
        %last = arith.cmpi eq, %ty, %c3 : index
        air.dma_memcpy_nd (%l1[] [] [], %h2[%c0, %c768] [%c16, %c192] [%c960, %c1_h]) {pad_before = array<i32: 0, 0>, pad_after = array<i32: 0, 64>} : (memref<16x256xbf16, 2>, memref<16x960xbf16, 1>)
        memref.dealloc %l1 : memref<16x256xbf16, 2>
      }
      memref.dealloc %l2 : memref<16x960xbf16, 1>
    }
  }
  return
}
