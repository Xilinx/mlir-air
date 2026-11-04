//===- canonicalize_device_issue.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize | FileCheck %s

// Folding a whole-buffer pattern rebuilds the put; it keeps its pairing with
// the air.channel.issue that starts it.

// CHECK-LABEL: func.func @keep_pairing
// CHECK: air.channel.put @c[] (%{{.*}}[] [] []) {air.device_issue = 3 : i64}
air.channel @c []
func.func @keep_pairing() {
  %buf = memref.alloc() : memref<8x256xi32, 1 : i32>
  air.channel.put @c[] (%buf[0, 0] [8, 256] [256, 1]) {air.device_issue = 3 : i64} : (memref<8x256xi32, 1 : i32>)
  return
}
