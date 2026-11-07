//===- dma_to_channel_device_issue_async.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -canonicalize -cse -split-input-file | FileCheck %s
// RUN: air-opt %s -air-dependency -air-dma-to-channel -canonicalize -cse -air-label-scf-for-to-ping-pong -split-input-file | FileCheck %s --check-prefix=PINGPONG

// In async IR the put a core starts leaves its herd like any other L3 put and
// does not wait on its issue. The issue's offset comes from the load's
// result, which SSA orders.

// CHECK-LABEL: func.func @split
// CHECK: air.launch
// CHECK: air.channel.put async @cin{{.*}}{air.device_issue = 0 : i64
// CHECK: air.herd @router
// CHECK: %{{.*}}, %[[V:.*]] = air.execute [%{{.*}}] -> (i32) {
// CHECK-NEXT: memref.load
// CHECK: %[[E:.*]] = arith.index_cast %[[V]] : i32 to index
// CHECK: %[[T:.*]] = air.execute {
// CHECK-NEXT: air.channel.issue @cin{{.*}} offset(%{{.*}} : i64)
// CHECK-NOT: air.channel.put
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

// -----

// A put in a core loop leaves for a loop of its own at the segment, so the
// core's loop holds only the issue and still double-buffers.

// CHECK-LABEL: func.func @l2_put_loop
// CHECK: air.segment
// CHECK: scf.for {{.*}} iter_args
// CHECK-NEXT: air.channel.put async {{.*}}@row4[] {{.*}}{air.device_issue = 0 : i64
// CHECK: air.herd @fwd
// CHECK-NOT: air.channel.put
// CHECK: air.channel.issue @row4[]
// PINGPONG-LABEL: func.func @l2_put_loop
// PINGPONG: air.herd @fwd
// PINGPONG: memref.alloc() {hoist_alloc = true} : memref<256xi32, 2 : i32>
// PINGPONG: air.channel.issue @row4[]
// PINGPONG: } {unroll = 2 : i32}
module {
  air.channel @row4 []
  func.func @l2_put_loop() {
    air.launch () in () {
      air.segment @seg {
        %c1 = arith.constant 1 : index
        %l2 = memref.alloc() : memref<8x256xi32, 1 : i32>
        air.herd @fwd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%w=%l2) : memref<8x256xi32, 1 : i32> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          scf.for %i = %c0 to %c4 step %c1_0 {
            %buf = memref.alloc() : memref<256xi32, 2 : i32>
            %v = memref.load %topk[%i] : memref<16xi32, 2 : i32>
            %e = arith.index_cast %v : i32 to index
            air.channel.put @row4[] (%w[%e, %c0] [1, 256] [256, 1]) : (memref<8x256xi32, 1 : i32>)
            air.channel.get @row4[] (%buf[] [] []) : (memref<256xi32, 2 : i32>)
            memref.dealloc %buf : memref<256xi32, 2 : i32>
          }
        }
      }
    }
    return
  }
}

// -----

// A get the core starts writes the core's own memory, so it waits for its
// issue.

// CHECK-LABEL: func.func @l1_get
// CHECK: %[[T:.*]] = air.execute [%{{.*}}] {
// CHECK-NEXT: air.channel.issue @win[] offset(%{{.*}} : i64)
// CHECK: air.channel.get async [%[[T]]] @win[] {{.*}}{air.device_issue = 0 : i64
module {
  air.channel @win []
  func.func @l1_get(%arg0: memref<256xi32>) {
    air.launch () in () args(%a0=%arg0) : memref<256xi32> {
      air.channel.put @win[] (%a0[] [] []) : (memref<256xi32>)
      air.segment @seg {
        %c1 = arith.constant 1 : index
        air.herd @fwd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
          %c0 = arith.constant 0 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          %slots = memref.alloc() : memref<8x256xi32, 2 : i32>
          %v = memref.load %topk[%c0] : memref<16xi32, 2 : i32>
          %e = arith.index_cast %v : i32 to index
          air.channel.get @win[] (%slots[%e, %c0] [1, 256] [256, 1]) : (memref<8x256xi32, 2 : i32>)
        }
      }
    }
    return
  }
}
