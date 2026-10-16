//===- segment_shared_name.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-insert-launch-around-herd -verify-diagnostics | FileCheck %s

// Segments in different funcs may share a name; they lower to one device.
// A segment with no name still gets one that no other segment uses.

// CHECK-LABEL: func.func @f0
// CHECK: air.segment @g_0
// CHECK-LABEL: func.func @f1
// CHECK: air.segment @g_0
// CHECK-LABEL: func.func @g
// CHECK: air.segment @g_1
// A name used later in the module is not generated either.
// CHECK-LABEL: func.func @h
// CHECK: air.segment @h_1
// CHECK-LABEL: func.func @k
// CHECK: air.segment @h_0
func.func @f0() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment @g_0 {
      %c2 = arith.constant 2 : index
      air.herd @h0 tile (%a, %b) in (%sa=%c2, %sb=%c2) {
      }
    }
  }
  return
}

func.func @f1() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment @g_0 {
      %c2 = arith.constant 2 : index
      air.herd @h1 tile (%a, %b) in (%sa=%c2, %sb=%c2) {
      }
    }
  }
  return
}

func.func @g() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment {
      %c2 = arith.constant 2 : index
      air.herd @h2 tile (%a, %b) in (%sa=%c2, %sb=%c2) {
      }
    }
  }
  return
}

func.func @h() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment {
      %c2 = arith.constant 2 : index
      air.herd @h3 tile (%a, %b) in (%sa=%c2, %sb=%c2) {
      }
    }
  }
  return
}

func.func @k() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment @h_0 {
      %c2 = arith.constant 2 : index
      air.herd @h4 tile (%a, %b) in (%sa=%c2, %sb=%c2) {
      }
    }
  }
  return
}
