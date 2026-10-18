//===- rolled_runtime_offsets_invalid.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -verify-diagnostics %s

// The runtime offset is added to the innermost offset of each BD, which only
// works if that dimension is contiguous.

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
    func.func @illegal_wrap(%arg0: memref<3x4x4096x16xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %z = arith.constant 0 : i64
      %id = arith.constant 2 : i32
      scf.for %i = %c0 to %c2 step %c1 {
        %ii = arith.index_cast %i : index to i64
        // expected-error @+1 {{with a runtime offset needs an innermost stride of 1 to be split into several BDs}}
        %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%ii, %z, %z, %z], [3, 2, 2048, 8], [262144, 65536, 16, 2]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<3x4x4096x16xi32>) : !airrt.event
      }
      return
    }
  }
}
