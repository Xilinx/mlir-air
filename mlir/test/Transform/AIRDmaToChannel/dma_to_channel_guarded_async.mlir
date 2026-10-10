//===- dma_to_channel_guarded_async.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -canonicalize -cse | FileCheck %s

// L2 to L1 DMAs under a condition on the herd index. Once async, the scf.if
// yields a token. The memtile-side puts keep the condition, so each core gets
// only its own branch's transfer, and the hoisted loop waits on the scf.if.

// CHECK-LABEL: func.func @guarded
// CHECK: air.segment
// CHECK: %[[L2:.*]] = air.channel.get async {{.*}}@channel_0
// CHECK: scf.parallel (%[[TY:.*]]) = {{.*}} init (%[[L2]])
// CHECK: %[[LAST:.*]] = arith.cmpi eq, %[[TY]], %c3
// CHECK: %[[IF:.*]] = scf.if %[[LAST]] -> (!air.async.token) {
// CHECK: %[[PUT:.*]] = air.channel.put async [%[[L2]]] @channel_1[%c0, %[[TY]]]
// CHECK-SAME: [16, 192] [960, 1]) {pad_after = array<i32: 0, 64>
// CHECK: scf.yield %[[PUT]]
// CHECK: } else {
// CHECK-NOT: air.channel.put
// CHECK: }
// CHECK: scf.reduce(%[[IF]]
// CHECK: scf.parallel (%[[TY2:.*]]) = {{.*}} init (%[[L2]])
// CHECK: %[[IF2:.*]] = scf.if %{{.*}} -> (!air.async.token) {
// CHECK-NOT: air.channel.put
// CHECK: } else {
// CHECK: air.channel.put async [%[[L2]]] @channel_2[%c0, %[[TY2]]]
// CHECK-SAME: [16, 256] [960, 1])
// CHECK: }
// CHECK: scf.reduce(%[[IF2]]
// CHECK: air.herd
// CHECK: scf.if
// CHECK: air.channel.get {{.*}}@channel_1
// CHECK: } else {
// CHECK: air.channel.get {{.*}}@channel_2

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
        %l1 = memref.alloc() : memref<16x256xbf16, 2>
        %off = arith.muli %ty, %c256 : index
        %last = arith.cmpi eq, %ty, %c3 : index
        scf.if %last {
          air.dma_memcpy_nd (%l1[] [] [], %h2[%c0, %off] [%c16, %c192] [%c960, %c1_h]) {pad_before = array<i32: 0, 0>, pad_after = array<i32: 0, 64>} : (memref<16x256xbf16, 2>, memref<16x960xbf16, 1>)
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
