//===- dump_dot.mlir - Test DOT graph dump ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dependency-parse-graph='output-dir=%T/dot_output' | FileCheck %s --check-prefix=IR
// RUN: cat %T/dot_output/host.dot | FileCheck %s --check-prefix=DOT

// IR: air.herd

// DOT: digraph G {
// DOT:   rankdir=LR;
// DOT-DAG: label="start"
// DOT-DAG: color="yellow"
// DOT-DAG: shape="box"
// DOT: ->
// DOT: }

module {

func.func @test(%arg0: memref<256xi32>) {
  %c1 = arith.constant 1 : index
  air.herd tile (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<256xi32> {
    %alloc = memref.alloc() : memref<256xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %arg5[] [] []) {id = 1 : i32} : (memref<256xi32, 2>, memref<256xi32>)
    air.dma_memcpy_nd (%arg5[] [] [], %alloc[] [] []) {id = 2 : i32} : (memref<256xi32>, memref<256xi32, 2>)
    memref.dealloc %alloc : memref<256xi32, 2>
    air.herd_terminator
  }
  return
}

}
