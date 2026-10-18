//===- rolled_runtime_offsets.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// Runtime offsets in two dimensions are added into the BD's linear offset.

// CHECK-LABEL: aie.runtime_sequence @two_offsets
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK:           scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
// CHECK-DAG:         %[[II:.*]] = arith.index_cast %[[I]] : index to i64
// CHECK-DAG:         %[[JJ:.*]] = arith.index_cast %[[J]] : index to i64
// CHECK:             %[[ROW:.*]] = arith.muli %[[II]], %c64_i64 : i64
// CHECK:             %[[OFF:.*]] = arith.addi %[[ROW]], %[[JJ]] : i64
// CHECK:             %[[OFF32:.*]] = arith.trunci %[[OFF]] : i64 to i32
// CHECK:             aie.dma_bd(%{{.*}} : memref<64x64xi32> offset = %[[OFF32]] len = 256 sizes = [16, 16] strides = [64, 1])
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId1(%tile_0_0, MM2S, 0)
    func.func @two_offsets(%arg0: memref<64x64xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %z = arith.constant 0 : i64
      %id = arith.constant 1 : i32
      scf.for %i = %c0 to %c2 step %c1 {
        scf.for %j = %c0 to %c2 step %c1 {
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%z, %z, %ii, %jj], [1, 1, 16, 16], [0, 0, 64, 1]) {metadata = @airMemcpyId1} : (i32, i64, i64, memref<64x64xi32>) : !airrt.event
        }
      }
      return
    }
  }
}

// -----

// A transfer with more dimensions than one BD holds is split into several BDs.
// The runtime offset is in a dimension that the split unrolls, and is added to
// the offset of every BD.

// CHECK-LABEL: aie.runtime_sequence @illegal_wrap
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK:           %[[II:.*]] = arith.index_cast %[[I]] : index to i64
// CHECK:           %[[BASE:.*]] = arith.muli %[[II]], %c262144_i64 : i64
// CHECK:           %[[B0:.*]] = arith.trunci %[[BASE]] : i64 to i32
// CHECK:           aie.dma_bd(%{{.*}} offset = %[[B0]] len = 32768 sizes = [2, 4, 512, 16] strides = [65536, 8192, 16, 1])
// CHECK:           %[[O1:.*]] = arith.addi %[[BASE]], %c262144_i64 : i64
// CHECK:           %[[B1:.*]] = arith.trunci %[[O1]] : i64 to i32
// CHECK:           aie.dma_bd(%{{.*}} offset = %[[B1]] len = 32768
// CHECK:           %[[O2:.*]] = arith.addi %[[BASE]], %c524288_i64 : i64
// CHECK:           %[[B2:.*]] = arith.trunci %[[O2]] : i64 to i32
// CHECK:           aie.dma_bd(%{{.*}} offset = %[[B2]] len = 32768
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
        %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%ii, %z, %z, %z], [3, 2, 2048, 16], [262144, 65536, 16, 1]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<3x4x4096x16xi32>) : !airrt.event
      }
      return
    }
  }
}
