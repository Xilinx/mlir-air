//===- preserve_disable_ping_pong_attr.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// air-dependency rewrites sync scf.for / scf.parallel that touch async ops
// into async versions by building a fresh op. The user-facing ping-pong
// opt-out attr `air.disable_ping_pong` must survive that rewrite so the
// labeling pass running much later in the pipeline still sees it.

// CHECK-LABEL: func.func @preserve_disable_ping_pong_scf_for
// CHECK: scf.for {{.*}} iter_args
// CHECK: } {air.disable_ping_pong}
func.func @preserve_disable_ping_pong_scf_for(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %arg1 = %c0 to %c4 step %c1 {
    %0 = arith.muli %arg1, %c32 : index
    %1 = memref.alloc() : memref<32xi32, 2>
    air.dma_memcpy_nd (%1[] [] [], %arg0[%0] [%c32] [%c1]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    memref.dealloc %1 : memref<32xi32, 2>
  } {air.disable_ping_pong}
  return
}

// CHECK-LABEL: func.func @preserve_disable_ping_pong_scf_parallel
// CHECK: scf.parallel
// CHECK: } {air.disable_ping_pong}
func.func @preserve_disable_ping_pong_scf_parallel(%arg0: memref<4096xi32>) {
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
  } {air.disable_ping_pong}
  return
}
