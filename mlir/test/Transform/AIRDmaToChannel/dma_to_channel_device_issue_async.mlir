//===- dma_to_channel_device_issue_async.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -canonicalize -cse | FileCheck %s

// In async IR the put a core starts stays in its herd and waits for its
// issue's token, so every pass up to air-opt-memtile-dma-bds sees the order.
// The issue's offset comes from the load's result, which SSA orders.

// CHECK-LABEL: func.func @split
// CHECK: air.herd @router
// CHECK: %{{.*}}, %[[V:.*]] = air.execute [%{{.*}}] -> (i32) {
// CHECK-NEXT: memref.load
// CHECK: %[[E:.*]] = arith.index_cast %[[V]] : i32 to index
// CHECK: %[[T:.*]] = air.execute {
// CHECK-NEXT: air.channel.issue @cin{{.*}} offset(%{{.*}} : i64)
// CHECK: air.channel.put async [{{.*}}%[[T]]{{.*}}] @cin{{.*}}{air.device_issue = 0 : i64
// CHECK: air.herd @mvm
module {
  air.channel @cin [1, 1]
  func.func @split(%arg0: memref<65536xi32>) {
    %c2 = arith.constant 2 : index
    air.launch (%lx) in (%ls=%c2) args(%a0=%arg0) : memref<65536xi32> {
      air.segment @seg args(%s0=%a0, %layer=%lx) : memref<65536xi32>, index {
        %c1 = arith.constant 1 : index
        air.herd @router tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%w=%s0, %l=%layer) : memref<65536xi32>, index {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c1024 = arith.constant 1024 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          %v = memref.load %topk[%c0] : memref<16xi32, 2 : i32>
          %e = arith.index_cast %v : i32 to index
          %off = affine.apply affine_map<()[s0, s1] -> (s0 * 32768 + s1 * 1024 + 7)>()[%l, %e]
          air.channel.put @cin[%tx, %ty] (%w[%off] [%c1024] [%c1_0]) : (memref<65536xi32>)
        }
        air.herd @mvm tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
          %buf = memref.alloc() : memref<1024xi32, 2 : i32>
          air.channel.get @cin[%tx, %ty] (%buf[] [] []) : (memref<1024xi32, 2 : i32>)
        }
      }
    }
    return
  }
}
