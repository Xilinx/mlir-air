//===- canonicalize_broadcast_set.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize | FileCheck %s

// Folding a view into a broadcast put rebuilds it; it keeps the cores it
// feeds (broadcast_set), which AIRToAIE's flow creation reads.

#set = affine_set<(d0, d1)[s0] : (d0 == 0, d1 >= 0, -d1 + 3 >= 0)>
// CHECK: #[[SET:.*]] = affine_set
// CHECK-LABEL: func.func @keep_broadcast_set
// CHECK: air.channel.put @c[] (%{{.*}}[] [] []) {broadcast_set = #[[SET]]}
air.channel @c [1, 1] {broadcast_shape = [1, 4]}
func.func @keep_broadcast_set() {
  %buf = memref.alloc() : memref<8x256xi32, 1 : i32>
  air.channel.put @c[] (%buf[0, 0] [8, 256] [256, 1]) {broadcast_set = #set} : (memref<8x256xi32, 1 : i32>)
  return
}
