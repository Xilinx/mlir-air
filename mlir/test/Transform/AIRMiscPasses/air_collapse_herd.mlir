//===- air_collapse_herd.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-collapse-herd="max-col-size=4" -canonicalize --split-input-file | FileCheck %s
// RUN: air-opt %s -air-collapse-herd -canonicalize --split-input-file | FileCheck %s --check-prefix=MAXCOL

// CHECK-LABEL: func.func @test0
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[CST4:.*]] = arith.constant 4 : index
// CHECK: air.herd  tile (%[[VAL0:.*]], %[[VAL1:.*]]) in (%[[VAL2:.*]]=%[[CST1]], %[[VAL3:.*]]=%[[CST4]])
// MAXCOL-LABEL: func.func @test0
// MAXCOL: %[[CST1:.*]] = arith.constant 1 : index
// MAXCOL: %[[CST4:.*]] = arith.constant 4 : index
// MAXCOL: air.herd  tile (%[[VAL0:.*]], %[[VAL1:.*]]) in (%[[VAL2:.*]]=%[[CST1]], %[[VAL3:.*]]=%[[CST4]])

func.func @test0() -> () {
  %c2 = arith.constant 2 : index
  air.herd tile (%x, %y) in (%sx=%c2, %sy=%c2) {
  }
  return
}

// -----

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 2 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 2 >= 0, s1 == 0)>
// CHECK: func.func @test1
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[CST4:.*]] = arith.constant 4 : index
// CHECK: air.herd  tile (%[[ARG0:.*]], %[[ARG1:.*]]) in (%[[ARG2:.*]]=%[[CST1]], %[[ARG3:.*]]=%[[CST4]])
// CHECK:   %[[CST0:.*]] = arith.constant 0 : index
// CHECK:   %[[CST2:.*]] = arith.constant 2 : index
// CHECK:   %[[VAL0:.*]] = arith.remsi %[[ARG1]], %[[CST2]] : index
// CHECK:   %[[VAL1:.*]] = arith.divsi %[[ARG1]], %[[CST2]] : index
// CHECK:   affine.if [[$SET0]]()[%[[VAL1]], %[[VAL0]]] {
// CHECK:   } else {
// CHECK:   }
// CHECK:   affine.if [[$SET1]]()[%[[VAL1]], %[[VAL0]]] {
// CHECK:   } else {

#set0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 2 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 2 >= 0, s1 == 0)>
func.func @test1() -> () {
  %c2 = arith.constant 2 : index
  air.herd tile (%x, %y) in (%sx=%c2, %sy=%c2) {
    %c0 = arith.constant 0 : index
    affine.if #set0()[%x, %y] {
      %alloc = memref.alloc() : memref<8x16xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<8x16xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<8x16xi32, 2>
      memref.dealloc %alloc : memref<8x16xi32, 2>
    } else {
      %alloc = memref.alloc() : memref<16x16xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<16x16xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<16x16xi32, 2>
      memref.dealloc %alloc : memref<16x16xi32, 2>
    }
    affine.if #set1()[%x, %y] {
      %alloc = memref.alloc() : memref<16x8xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<16x8xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<16x8xi32, 2>
      memref.dealloc %alloc : memref<16x8xi32, 2>
    } else {
      %alloc = memref.alloc() : memref<8x8xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<8x8xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<8x8xi32, 2>
      memref.dealloc %alloc : memref<8x8xi32, 2>
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test2
// CHECK: %[[CST3:.*]] = arith.constant 3 : index
// CHECK: air.herd  tile (%[[VAL0:.*]], %[[VAL1:.*]]) in (%[[VAL2:.*]]=%[[CST3]], %[[VAL3:.*]]=%[[CST3]])
// MAXCOL-LABEL: func.func @test2
// MAXCOL: %[[CST1:.*]] = arith.constant 1 : index
// MAXCOL: %[[CST9:.*]] = arith.constant 9 : index
// MAXCOL: air.herd  tile (%[[VAL0:.*]], %[[VAL1:.*]]) in (%[[VAL2:.*]]=%[[CST1]], %[[VAL3:.*]]=%[[CST9]])

func.func @test2() -> () {
  %c3 = arith.constant 3 : index
  air.herd tile (%x, %y) in (%sx=%c3, %sy=%c3) {
  }
  return
}

// -----

// Verify that herds with intra-herd cascade (both put and get on the
// same cascade channel) are NOT collapsed — cascade connections are
// peer-to-peer and rearranging would break cascade routing.

// CHECK-LABEL: func.func @test_intra_herd_cascade_skip
// CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
// CHECK: air.herd @cascade_herd tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])
// MAXCOL-LABEL: func.func @test_intra_herd_cascade_skip
// MAXCOL-DAG: %[[CST2:.*]] = arith.constant 2 : index
// MAXCOL: air.herd @cascade_herd tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])

air.channel @cascade_chan [3] {channel_type = "cascade"}
func.func @test_intra_herd_cascade_skip() -> () {
  %c2 = arith.constant 2 : index
  air.herd @cascade_herd tile (%x, %y) in (%sx=%c2, %sy=%c2) {
    %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
    %alloc2 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
    // Intra-herd cascade: same channel used for both put and get within one herd
    air.channel.get @cascade_chan[%x] (%alloc2[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
    air.channel.put @cascade_chan[%x] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
    memref.dealloc %alloc : memref<1x1x2048xi32, 2 : i32>
    memref.dealloc %alloc2 : memref<1x1x2048xi32, 2 : i32>
  }
  return
}

// -----

// Verify that herds with inter-herd cascade (only put OR get, not both)
// are NOT collapsed — all herds in a segment with cascade skip collapsing.

// CHECK-LABEL: func.func @test_inter_herd_cascade_skip
// CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
// CHECK: air.herd @cascade_producer tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])
// MAXCOL-LABEL: func.func @test_inter_herd_cascade_skip
// MAXCOL-DAG: %[[CST2:.*]] = arith.constant 2 : index
// MAXCOL: air.herd @cascade_producer tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])

air.channel @inter_cascade [1] {channel_type = "cascade"}
func.func @test_inter_herd_cascade_skip() -> () {
  %c2 = arith.constant 2 : index
  air.herd @cascade_producer tile (%x, %y) in (%sx=%c2, %sy=%c2) {
    %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
    air.channel.put @inter_cascade[] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
    memref.dealloc %alloc : memref<1x1x2048xi32, 2 : i32>
  }
  return
}

// -----

// Verify that a non-cascade herd in the same segment as a cascade herd
// is also NOT collapsed — all herds in a cascade segment skip collapsing
// to avoid dimension mismatches that break placement.

// CHECK-LABEL: func.func @test_non_cascade_herd_in_cascade_segment
// CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
// CHECK: air.herd @no_cascade_herd tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])
// CHECK: air.herd @cascade_herd_put tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])
// MAXCOL-LABEL: func.func @test_non_cascade_herd_in_cascade_segment
// MAXCOL-DAG: %[[CST2:.*]] = arith.constant 2 : index
// MAXCOL: air.herd @no_cascade_herd tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])
// MAXCOL: air.herd @cascade_herd_put tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[CST2]], %{{.*}}=%[[CST2]])

air.channel @seg_cascade [1] {channel_type = "cascade"}
func.func @test_non_cascade_herd_in_cascade_segment() -> () {
  %c1 = arith.constant 1 : index
  air.launch (%arg0) in (%arg1=%c1) {
    air.segment @seg {
      %c2 = arith.constant 2 : index
      // This herd has no cascade, but is in the same segment as a cascade herd
      air.herd @no_cascade_herd tile (%x, %y) in (%sx=%c2, %sy=%c2) {
        %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
        memref.dealloc %alloc : memref<1x1x2048xi32, 2 : i32>
      }
      air.herd @cascade_herd_put tile (%x, %y) in (%sx=%c2, %sy=%c2) {
        %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
        air.channel.put @seg_cascade[] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
        memref.dealloc %alloc : memref<1x1x2048xi32, 2 : i32>
      }
    }
  }
  return
}
