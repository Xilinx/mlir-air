//===- unroll_inner_stride_invalid.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -verify-diagnostics %s

// Unrolling a transfer into several BDs moves its start through the innermost
// offset, which is in units of the innermost stride: an outer stride that is
// not a multiple of it cannot be expressed that way.

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
    func.func @outer_stride_not_multiple(%arg0: memref<3x4x4096x16xi32>) {
      %z = arith.constant 0 : i64
      %id = arith.constant 2 : i32
      // expected-error @+1 {{needs each outer stride to be a multiple of the innermost stride to be split into several BDs}}
      %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%z, %z, %z, %z], [3, 2, 2048, 8], [262145, 65536, 16, 2]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<3x4x4096x16xi32>) : !airrt.event
      return
    }
  }
}
