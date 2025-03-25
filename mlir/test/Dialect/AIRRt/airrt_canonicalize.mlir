//===- airrt_canonicalize.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: wait_all_0
// CHECK-NEXT: return
func.func @wait_all_0() -> () {
  %0 = airrt.wait_all : !airrt.event
  airrt.wait_all %0
  return
}

// CHECK-LABEL: wait_all_1
// CHECK-SAME: (%[[E0:.*]]: !airrt.event, %[[E1:.*]]: !airrt.event, %[[E2:.*]]: !airrt.event) -> !airrt.event {
// CHECK-NEXT:   %[[E4:.*]] = airrt.wait_all %[[E0]], %[[E1]], %[[E2]] : !airrt.event
// CHECK-NEXT: return %[[E4]]
func.func @wait_all_1(%e0 : !airrt.event, %e1 : !airrt.event, %e2 : !airrt.event) -> (!airrt.event) {
  %1 = airrt.wait_all %e0 : !airrt.event
  %2 = airrt.wait_all %e1 : !airrt.event
  %3 = airrt.wait_all %e2 : !airrt.event
  %4 = airrt.wait_all %1 : !airrt.event
  %5 = airrt.wait_all %4, %2 : !airrt.event
  %6 = airrt.wait_all %5, %3 : !airrt.event
  %7 = airrt.wait_all %6 : !airrt.event
  return %7 : !airrt.event
}

// CHECK-LABEL: alloc_dealloc
// CHECK-NEXT: return
func.func @alloc_dealloc() {
  %0 = airrt.alloc : memref<1x4x4x16xi32, 1>
  airrt.dealloc %0 : memref<1x4x4x16xi32, 1>
  return
}
