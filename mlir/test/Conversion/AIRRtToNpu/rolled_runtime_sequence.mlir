//===- rolled_runtime_sequence.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// airrt-to-npu leaves the runtime sequence's scf.for in place, so what reaches
// aiecc is one configure inside a loop rather than N copies of it. aiecc's
// aie-unroll-runtime-sequence-loops unrolls the constant-trip case itself --
// one unroller in the stack -- so the instruction stream is unchanged; a
// runtime-bound loop stays rolled through to the dynamic BD pool.
//
// The DMA has to be built directly under the runtime_sequence -- the AIEX ops
// require it as an ancestor -- so the control func becomes the sequence before
// the DMA conversion runs, and the loop's !airrt.event iter_args (which
// unrolling used to dissolve) are dropped first.

// CHECK-LABEL: aie.runtime_sequence @func0
// CHECK:         scf.for
// CHECK:           aiex.dma_configure_task_for @airMemcpyId1
// CHECK:             aie.dma_bd(%{{.*}} : memref<64xi32>
// CHECK:           aiex.dma_start_task
// CHECK:         }
#map = affine_map<()[s0] -> (s0)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId1(%tile_0_0, MM2S, 0)
    func.func @func0(%arg0: memref<64xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %z = arith.constant 0 : i64
      %id = arith.constant 1 : i32
      %e0 = airrt.wait_all : !airrt.event
      %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%e = %e0) -> (!airrt.event) {
        %e1 = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[0, 0, 0, 0], [1, 1, 1, 64], [0, 0, 0, 1]) {metadata = @airMemcpyId1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.yield %e1 : !airrt.event
      }
      return
    }
  }
}

// -----

// A runtime trip count survives to the sequence: this is the form the static
// BD-id allocator rejects and the dynamic BD pool exists for.

// CHECK-LABEL: aie.runtime_sequence @func1
// CHECK-SAME:    %{{.*}}: memref<64xi32>, %[[N:[a-zA-Z0-9_]+]]: index
// CHECK:         scf.for %{{.*}} = %c0{{.*}} to %[[N]] step %c1
// CHECK:           aiex.dma_configure_task_for @airMemcpyId2
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
    func.func @func1(%arg0: memref<64xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %z = arith.constant 0 : i64
      %id = arith.constant 2 : i32
      scf.for %i = %c0 to %n step %c1 {
        %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[0, 0, 0, 0], [1, 1, 1, 64], [0, 0, 0, 1]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      }
      return
    }
  }
}
