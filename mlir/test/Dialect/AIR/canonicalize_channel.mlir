//===- canonicalize_channel.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize | FileCheck %s

// Prune channels with empty uses.
// CHECK-LABEL: module
// CHECK-NEXT: %c2 = arith.constant 2 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: air.channel @channel_0 [1, 1]
// CHECK-NEXT: air.launch (%{{.*}}, %{{.*}}) in (%{{.*}}=%c1, %{{.*}}=%c2) attributes {foo = "bar"} {

module {
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  air.launch (%tx, %ty) in (%size_x = %c1, %size_y = %c2) attributes {foo = "bar"} {
    %alloc = memref.alloc() : memref<16x8xi32>
    air.channel.get  @channel_0[] (%alloc[] [] []) : (memref<16x8xi32>)
  }
}