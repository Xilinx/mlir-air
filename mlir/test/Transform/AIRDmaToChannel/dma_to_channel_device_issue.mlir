//===- dma_to_channel_device_issue.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse -split-input-file -verify-diagnostics | FileCheck %s

// An L3 put whose offset uses a value the core loads from L1 is split: the
// host part is hoisted onto the put, and an air.channel.issue in the herd
// carries the device part as an i64 element offset.

// CHECK-DAG: #[[HOSTMAP:.*]] = affine_map<()[s0] -> (s0 * 32768 + 7)>
// CHECK-LABEL: func.func @split
// CHECK: air.launch (%[[L:.*]]) in
// CHECK: %[[HOST:.*]] = affine.apply #[[HOSTMAP]]()[%[[L]]]
// CHECK: air.channel.put @cin[{{.*}}] (%{{.*}}[%[[HOST]]] [1024] [1]) {air.device_issue = 0 : i64}
// CHECK: air.herd @router
// CHECK: %[[V:.*]] = memref.load
// CHECK: %[[E:.*]] = arith.index_cast %[[V]] : i32 to index
// CHECK: %[[E64:.*]] = arith.index_cast %[[E]] : index to i64
// CHECK: %[[DEV:.*]] = arith.muli %[[E64]], %c1024_i64 : i64
// CHECK: air.channel.issue @cin[%{{.*}}, %{{.*}}] offset(%[[DEV]] : i64) {issue_id = 0 : i64}
// CHECK: air.herd @mvm
// CHECK: air.channel.get @cin
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

// The device offset of a multi-dimensional access is folded through its
// strides into one element offset.

// CHECK-LABEL: func.func @strided
// CHECK: air.channel.put @cin2[{{.*}}] (%{{.*}}[0, 0] [4, 256] [256, 1]) {air.device_issue = 0 : i64}
// CHECK: air.herd @router
// CHECK: %[[E:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK: %[[DEV:.*]] = arith.muli %[[E]], %c256_i64 : i64
// CHECK: air.channel.issue @cin2[%{{.*}}, %{{.*}}] offset(%[[DEV]] : i64)
module {
  air.channel @cin2 [1, 1]
  func.func @strided(%arg0: memref<64x256xi32>) {
    air.launch () in () args(%a0=%arg0) : memref<64x256xi32> {
      air.segment @seg args(%s0=%a0) : memref<64x256xi32> {
        %c1 = arith.constant 1 : index
        air.herd @router tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%w=%s0) : memref<64x256xi32> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c256 = arith.constant 256 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          %v = memref.load %topk[%c0] : memref<16xi32, 2 : i32>
          %e = arith.index_cast %v : i32 to index
          air.channel.put @cin2[%tx, %ty] (%w[%e, %c0] [%c4, %c256] [%c256, %c1_0]) : (memref<64x256xi32>)
        }
        air.herd @mvm tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
          %buf = memref.alloc() : memref<1024xi32, 2 : i32>
          air.channel.get @cin2[%tx, %ty] (%buf[] [] []) : (memref<1024xi32, 2 : i32>)
        }
      }
    }
    return
  }
}

// -----

module {
  air.channel @cin3 [1, 1]
  func.func @nonlinear(%arg0: memref<65536xi32>) {
    air.launch () in () args(%a0=%arg0) : memref<65536xi32> {
      air.segment @seg args(%s0=%a0) : memref<65536xi32> {
        %c1 = arith.constant 1 : index
        air.herd @router tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%w=%s0) : memref<65536xi32> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c1024 = arith.constant 1024 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          %v = memref.load %topk[%c0] : memref<16xi32, 2 : i32>
          %e = arith.index_cast %v : i32 to index
          %off = affine.apply affine_map<()[s0] -> ((s0 * 1024) mod 4096)>()[%e]
          // expected-error@+1 {{offset depends on a value computed in the herd, but is not a host part plus a linear function of the core's values}}
          air.channel.put @cin3[%tx, %ty] (%w[%off] [%c1024] [%c1_0]) : (memref<65536xi32>)
        }
      }
    }
    return
  }
}

// -----

module {
  air.channel @cin4 [1, 1]
  func.func @device_size(%arg0: memref<65536xi32>) {
    air.launch () in () args(%a0=%arg0) : memref<65536xi32> {
      air.segment @seg args(%s0=%a0) : memref<65536xi32> {
        %c1 = arith.constant 1 : index
        air.herd @router tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%w=%s0) : memref<65536xi32> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          %v = memref.load %topk[%c0] : memref<16xi32, 2 : i32>
          %e = arith.index_cast %v : i32 to index
          // expected-error@+1 {{sizes and strides of a transfer cannot depend on a value computed in the herd; only its offset can}}
          air.channel.put @cin4[%tx, %ty] (%w[%e] [%e] [%c1_0]) : (memref<65536xi32>)
        }
      }
    }
    return
  }
}

// -----

// An L2 put whose offset the core computes, received by that core: the
// memtile descriptor keeps only the constant part.

// CHECK-LABEL: func.func @l2_put
// CHECK: air.segment
// CHECK: air.channel.put @row[] (%{{.*}}[0, 0] [1, 256] [256, 1]) {air.device_issue = {{[0-9]+}} : i64}
// CHECK: air.herd @fwd
// CHECK: %[[E:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK: %[[DEV:.*]] = arith.muli %[[E]], %c256_i64 : i64
// CHECK: air.channel.issue @row[] offset(%[[DEV]] : i64)
// CHECK: air.channel.get @row[]
module {
  air.channel @row []
  func.func @l2_put() {
    air.launch () in () {
      air.segment @seg {
        %c1 = arith.constant 1 : index
        %l2 = memref.alloc() : memref<8x256xi32, 1 : i32>
        air.herd @fwd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%w=%l2) : memref<8x256xi32, 1 : i32> {
          %c0 = arith.constant 0 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          %buf = memref.alloc() : memref<256xi32, 2 : i32>
          %v = memref.load %topk[%c0] : memref<16xi32, 2 : i32>
          %e = arith.index_cast %v : i32 to index
          air.channel.put @row[] (%w[%e, %c0] [1, 256] [256, 1]) : (memref<8x256xi32, 1 : i32>)
          air.channel.get @row[] (%buf[] [] []) : (memref<256xi32, 2 : i32>)
        }
      }
    }
    return
  }
}

// -----

// An L1 get whose destination the core computes: its own descriptor, started
// by the core before the get.

// CHECK-LABEL: func.func @l1_get
// CHECK: air.herd @fwd
// CHECK: air.channel.issue @win[] offset(%{{.*}} : i64)
// CHECK: air.channel.get @win[] (%{{.*}}[0, 0] [1, 256] [256, 1]) {air.device_issue = {{[0-9]+}} : i64}
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

// -----

// A segment's L2 put with a run-time outermost size: the largest stays on the
// op, and the host sets the run-time one.

// CHECK-LABEL: func.func @l2_length
// CHECK: air.segment
// CHECK: %[[N:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: air.channel.issue @kv[] length(%[[N]]) {air.length_bound = 8 : i64, issue_id = {{[0-9]+}} : i64}
// CHECK: air.channel.put @kv[] (%{{.*}}[] [] []) {air.device_issue = {{[0-9]+}} : i64}
module {
  air.channel @kv []
  func.func @l2_length(%arg0: i32) {
    air.launch () in () args(%n0=%arg0) : i32 {
      air.segment @seg args(%n=%n0) : i32 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c256 = arith.constant 256 : index
        %l2 = memref.alloc() : memref<8x256xi32, 1 : i32>
        %len = arith.index_cast %n : i32 to index
        air.channel.put @kv[] (%l2[%c0, %c0] [%len, %c256] [%c256, %c1]) : (memref<8x256xi32, 1 : i32>)
        air.herd @fwd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
          %buf = memref.alloc() : memref<256xi32, 2 : i32>
          air.channel.get @kv[] (%buf[] [] []) : (memref<256xi32, 2 : i32>)
        }
      }
    }
    return
  }
}

// -----

// An L3 put's run-time size goes through control packets only when the launch
// asks for it.

// CHECK-LABEL: func.func @l3_length
// CHECK: air.channel.put @xin[] (%{{.*}}[] [] []) {air.device_issue = {{[0-9]+}} : i64}
// CHECK: air.segment
// CHECK: air.channel.issue @xin[] length(%{{.*}}) {air.length_bound = 8 : i64
module {
  air.channel @xin []
  func.func @l3_length(%arg0: memref<8x256xi32>, %arg1: i32) {
    air.launch () in () args(%a0=%arg0, %n0=%arg1) : memref<8x256xi32>, i32 attributes {air.host_issue_lengths} {
      air.segment @seg args(%x=%a0, %n=%n0) : memref<8x256xi32>, i32 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c256 = arith.constant 256 : index
        %len = arith.index_cast %n : i32 to index
        air.channel.put @xin[] (%x[%c0, %c0] [%len, %c256] [%c256, %c1]) : (memref<8x256xi32>)
        air.herd @fwd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
          %buf = memref.alloc() : memref<256xi32, 2 : i32>
          air.channel.get @xin[] (%buf[] [] []) : (memref<256xi32, 2 : i32>)
        }
      }
    }
    return
  }
}

// -----

module {
  air.channel @row2 []
  func.func @l2_host_offset() {
    air.launch () in () {
      air.segment @seg {
        %c1 = arith.constant 1 : index
        %l2 = memref.alloc() : memref<8x256xi32, 1 : i32>
        air.herd @fwd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%w=%l2) : memref<8x256xi32, 1 : i32> {
          %c0 = arith.constant 0 : index
          %topk = memref.alloc() : memref<16xi32, 2 : i32>
          %buf = memref.alloc() : memref<256xi32, 2 : i32>
          %v = memref.load %topk[%c0] : memref<16xi32, 2 : i32>
          %e = arith.index_cast %v : i32 to index
          %off = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%e, %tx]
          // expected-error@+1 {{offset adds a core-computed value to one that is not a constant; only an L3 transfer can take both}}
          air.channel.put @row2[] (%w[%off, %c0] [1, 256] [256, 1]) : (memref<8x256xi32, 1 : i32>)
          air.channel.get @row2[] (%buf[] [] []) : (memref<256xi32, 2 : i32>)
        }
      }
    }
    return
  }
}
