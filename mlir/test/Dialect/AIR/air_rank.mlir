//===- air_rank.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_rank_1d
func.func @test_rank_1d() {
  %c2 = arith.constant 2 : index
  // CHECK: air.rank (%{{.*}}) in (%{{.*}}=%c2{{.*}}) {
  // CHECK-NEXT: }
  air.rank (%rx) in (%sx = %c2) {
  }
  return
}

// CHECK-LABEL: func.func @test_rank_2d
func.func @test_rank_2d() {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: air.rank (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c4{{.*}}) {
  air.rank (%rx, %ry) in (%sx = %c2, %sy = %c4) {
  }
  return
}

// CHECK-LABEL: func.func @test_rank_async
func.func @test_rank_async() {
  %c2 = arith.constant 2 : index
  // CHECK: %[[T0:.*]] = air.rank async (%{{.*}}) in (%{{.*}}=%c2{{.*}}) {
  %t0 = air.rank async (%rx) in (%sx = %c2) {
  }
  // CHECK: air.rank async [%[[T0]]] (%{{.*}}) in (%{{.*}}=%c2{{.*}}) {
  %t1 = air.rank async [%t0] (%rx2) in (%sx2 = %c2) {
  }
  return
}

// CHECK-LABEL: func.func @test_rank_args
func.func @test_rank_args(%arg0 : memref<16x16xf32>) {
  %c2 = arith.constant 2 : index
  // CHECK: air.rank (%{{.*}}) in (%{{.*}}=%c2{{.*}}) args(%{{.*}}=%{{.*}}) : memref<16x16xf32>
  air.rank (%rx) in (%sx = %c2) args(%a=%arg0) : memref<16x16xf32> {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls = %c1) args(%la=%a) : memref<16x16xf32> {
    }
  }
  return
}

// CHECK-LABEL: func.func @test_rank_universe
func.func @test_rank_universe() {
  %c4 = arith.constant 4 : index
  // CHECK: %[[U:.*]] = air.universe.alloc(%c4{{.*}})
  %u = air.universe.alloc(%c4)
  // CHECK: air.rank universe(%[[U]]) (%{{.*}}) in (%{{.*}}=%c4{{.*}}) {
  air.rank universe(%u) (%rx) in (%sx = %c4) {
  }
  return
}

// CHECK-LABEL: func.func @test_rank_named
func.func @test_rank_named() {
  %c1 = arith.constant 1 : index
  // CHECK: air.rank @my_rank (%{{.*}}) in (%{{.*}}=%c1{{.*}})
  air.rank @my_rank (%rx) in (%sx = %c1)
  return
}
