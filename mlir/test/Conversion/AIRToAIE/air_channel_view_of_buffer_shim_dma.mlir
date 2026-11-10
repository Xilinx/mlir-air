//===- air_channel_view_of_buffer_shim_dma.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802 generate-shim-dma=true" -canonicalize | FileCheck %s

// A two-ended copy from L3 into a subview of a core's buffer: the core's BD
// starts where the subview does; the shim's BD, on the other end, where its
// own access pattern says.

// CHECK: aie.mem(%{{.*}})
// CHECK: aie.dma_bd(%{{.*}} : memref<64xi32, 2> offset = 32 len = 32)
// CHECK: aie.shim_dma(%{{.*}})
// CHECK: aie.dma_bd(%{{.*}} : memref<1024xi32> offset = 0 len = 32)

func.func @two_ended(%arg0 : memref<1024xi32>) -> () {
  %c1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%sx = %c1, %sy = %c1) args(%ext0 = %arg0) : memref<1024xi32> attributes { sym_name="herd1"} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %buf0 = memref.alloc() : memref<64xi32, 2>
    %half = memref.subview %buf0[32] [32] [1] : memref<64xi32, 2> to memref<32xi32, strided<[1], offset: 32>, 2>
    air.dma_memcpy_nd (%half[] [] [], %ext0[%c0] [%c32] [%c1_0]) : (memref<32xi32, strided<[1], offset: 32>, 2>, memref<1024xi32>)
    memref.dealloc %buf0 : memref<64xi32, 2>
  }
  return
}
