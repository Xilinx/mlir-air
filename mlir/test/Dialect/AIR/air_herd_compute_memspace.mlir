//===- air_herd_compute_memspace.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @herd_l1_load_store
// Positive test: memref.load/store on L1 memref inside herd is allowed.
func.func @herd_l1_load_store() {
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<32xi32, 2 : i32>
  air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%buf=%alloc) : memref<32xi32, 2 : i32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 42 : i32
    memref.store %cst, %buf[%c0] : memref<32xi32, 2 : i32>
    %v = memref.load %buf[%c0] : memref<32xi32, 2 : i32>
    air.herd_terminator
  }
  memref.dealloc %alloc : memref<32xi32, 2 : i32>
  return
}

// -----

// CHECK-LABEL: func.func @herd_dma_l2_exempt
// Positive test: air.dma_memcpy_nd referencing L2 memrefs is allowed (DMA engine).
func.func @herd_dma_l2_exempt() {
  %c1 = arith.constant 1 : index
  %l2_alloc = memref.alloc() : memref<32xi32, 1 : i32>
  %l1_alloc = memref.alloc() : memref<32xi32, 2 : i32>
  air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%l2=%l2_alloc, %l1=%l1_alloc) : memref<32xi32, 1 : i32>, memref<32xi32, 2 : i32> {
    air.dma_memcpy_nd (%l1[] [] [], %l2[] [] []) : (memref<32xi32, 2 : i32>, memref<32xi32, 1 : i32>)
    air.herd_terminator
  }
  memref.dealloc %l1_alloc : memref<32xi32, 2 : i32>
  memref.dealloc %l2_alloc : memref<32xi32, 1 : i32>
  return
}

// -----

// CHECK-LABEL: func.func @herd_linalg_l2_allowed
// Positive test: linalg ops with non-L1 memrefs are not checked (they are
// higher-level ops that get lowered to loads/stores or DMA later).
func.func @herd_linalg_l2_allowed() {
  %c1 = arith.constant 1 : index
  %l2_in = memref.alloc() : memref<32xf32, 1 : i32>
  %l1_out = memref.alloc() : memref<32xf32, 2 : i32>
  air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%in=%l2_in, %out=%l1_out) : memref<32xf32, 1 : i32>, memref<32xf32, 2 : i32> {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%in : memref<32xf32, 1 : i32>) outs(%out : memref<32xf32, 2 : i32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      linalg.yield %arg0 : f32
    }
    air.herd_terminator
  }
  memref.dealloc %l1_out : memref<32xf32, 2 : i32>
  memref.dealloc %l2_in : memref<32xf32, 1 : i32>
  return
}

// -----

// Negative test: memref.load from L2 memref inside herd is rejected.
func.func @herd_l2_load_rejected() {
  %c1 = arith.constant 1 : index
  %l2_alloc = memref.alloc() : memref<32xi32, 1 : i32>
  air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%buf=%l2_alloc) : memref<32xi32, 1 : i32> {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{'memref.load' op inside 'air.herd' accesses memref with memory_space L2; AIE core tiles can only access L1 or more local memory directly. Use air.dma_memcpy_nd to stage data first.}}
    %v = memref.load %buf[%c0] : memref<32xi32, 1 : i32>
    air.herd_terminator
  }
  memref.dealloc %l2_alloc : memref<32xi32, 1 : i32>
  return
}

// -----

// Negative test: memref.store to L3 (default) memref inside herd is rejected.
func.func @herd_l3_store_rejected() {
  %c1 = arith.constant 1 : index
  %l3_alloc = memref.alloc() : memref<32xi32>
  air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%buf=%l3_alloc) : memref<32xi32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 42 : i32
    // expected-error @+1 {{'memref.store' op inside 'air.herd' accesses memref with memory_space L3; AIE core tiles can only access L1 or more local memory directly. Use air.dma_memcpy_nd to stage data first.}}
    memref.store %cst, %buf[%c0] : memref<32xi32>
    air.herd_terminator
  }
  memref.dealloc %l3_alloc : memref<32xi32>
  return
}

// -----

// Negative test: vector.transfer_read from L3 memref inside herd is rejected.
func.func @herd_l3_transfer_read_rejected() {
  %c1 = arith.constant 1 : index
  %l3_alloc = memref.alloc() : memref<32xf32>
  air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%buf=%l3_alloc) : memref<32xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    // expected-error @+1 {{'vector.transfer_read' op inside 'air.herd' accesses memref with memory_space L3; AIE core tiles can only access L1 or more local memory directly. Use air.dma_memcpy_nd to stage data first.}}
    %v = vector.transfer_read %buf[%c0], %cst : memref<32xf32>, vector<16xf32>
    air.herd_terminator
  }
  memref.dealloc %l3_alloc : memref<32xf32>
  return
}
