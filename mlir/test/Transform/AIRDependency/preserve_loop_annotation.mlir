//===- preserve_loop_annotation.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// air-dependency rewrites sync scf.for / scf.parallel that touch async ops
// into async versions by building a fresh op with an extra !air.async.token
// iter_arg / init_val. The fresh op was created with only sym_name copied,
// so a user-attached llvm.loop_annotation (used to disable Peano -O2
// unrolling) was dropped. Verify it survives.

#loop_unroll = #llvm.loop_unroll<disable = true>
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>

// CHECK-DAG: #[[$LOOP_UNROLL:.*]] = #llvm.loop_unroll<disable = true>
// CHECK-DAG: #[[$LOOP_ANNOT:.*]] = #llvm.loop_annotation<unroll = #[[$LOOP_UNROLL]], mustProgress = true>

// CHECK-LABEL: func.func @preserve_loop_annotation_scf_for
// CHECK: scf.for {{.*}} iter_args
// CHECK: } {loop_annotation = #[[$LOOP_ANNOT]]}
func.func @preserve_loop_annotation_scf_for(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %arg1 = %c0 to %c4 step %c1 {
    %0 = arith.muli %arg1, %c32 : index
    %1 = memref.alloc() : memref<32xi32, 2>
    air.dma_memcpy_nd (%1[] [] [], %arg0[%0] [%c32] [%c1]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    memref.dealloc %1 : memref<32xi32, 2>
  } {loop_annotation = #loop_annotation}
  return
}

// CHECK-LABEL: func.func @preserve_loop_annotation_scf_parallel
// CHECK: scf.parallel
// CHECK: } {loop_annotation = #[[$LOOP_ANNOT]]}
func.func @preserve_loop_annotation_scf_parallel(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.parallel (%arg1) = (%c0) to (%c4) step (%c1) {
    %0 = arith.muli %arg1, %c32 : index
    %1 = memref.alloc() : memref<32xi32, 2>
    air.dma_memcpy_nd (%1[] [] [], %arg0[%0] [%c32] [%c1]) {id = 2 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    memref.dealloc %1 : memref<32xi32, 2>
    scf.reduce
  } {loop_annotation = #loop_annotation}
  return
}
