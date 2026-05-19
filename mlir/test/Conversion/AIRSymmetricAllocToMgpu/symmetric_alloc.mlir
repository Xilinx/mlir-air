//===- symmetric_alloc.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s --split-input-file -air-symmetric-alloc-to-mgpu | FileCheck %s

// Basic 1D alloc + dealloc.
// CHECK-LABEL: func.func @basic_alloc_dealloc
// CHECK: %[[SZ:.*]] = arith.constant 4096 : i64
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[PTR:.*]] = call @mgpuSymmetricAlloc(%[[SZ]], %[[NULL]]) : (i64, !llvm.ptr) -> !llvm.ptr
// Descriptor build (poison + insertvalue) then unrealized cast.
// CHECK: llvm.mlir.poison
// CHECK: llvm.insertvalue %[[PTR]]
// CHECK: llvm.insertvalue %[[PTR]]
// CHECK: builtin.unrealized_conversion_cast {{.*}} : !llvm.struct<{{.*}}> to memref<1024xf32, #air.symmetric_heap>
// Dealloc -> mgpuSymmetricFree.
// CHECK: call @mgpuSymmetricFree(%[[PTR]],
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.dealloc
func.func @basic_alloc_dealloc() {
  %buf = memref.alloc() : memref<1024xf32, #air.symmetric_heap>
  memref.dealloc %buf : memref<1024xf32, #air.symmetric_heap>
  return
}

// -----

// 2D alloc: 64*64*4 = 16384 bytes; descriptor strides should be [64, 1].
// CHECK-LABEL: func.func @alloc_2d
// CHECK: arith.constant 16384 : i64
// CHECK: call @mgpuSymmetricAlloc
// Strides 64 then 1 in the descriptor (innermost-most-contiguous).
// CHECK: llvm.mlir.constant(64 : i64)
// CHECK: llvm.insertvalue
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: llvm.insertvalue
// CHECK: builtin.unrealized_conversion_cast {{.*}} : !llvm.struct<{{.*}}> to memref<64x64xf32, #air.symmetric_heap>
func.func @alloc_2d() -> memref<64x64xf32, #air.symmetric_heap> {
  %buf = memref.alloc() : memref<64x64xf32, #air.symmetric_heap>
  return %buf : memref<64x64xf32, #air.symmetric_heap>
}

// -----

// f64 element type (8 bytes): 1024 * 8 = 8192 bytes.
// CHECK-LABEL: func.func @f64_element
// CHECK: arith.constant 8192 : i64
func.func @f64_element() {
  %buf = memref.alloc() : memref<1024xf64, #air.symmetric_heap>
  memref.dealloc %buf : memref<1024xf64, #air.symmetric_heap>
  return
}

// -----

// i32 element type (4 bytes): 256 * 4 = 1024 bytes.
// CHECK-LABEL: func.func @i32_element
// CHECK: arith.constant 1024 : i64
func.func @i32_element() {
  %buf = memref.alloc() : memref<256xi32, #air.symmetric_heap>
  memref.dealloc %buf : memref<256xi32, #air.symmetric_heap>
  return
}

// -----

// Multiple symmetric allocs in one function: each lowered independently;
// extern decls are emitted exactly once at module scope.
// Match the actual emission order: Free decl before Alloc decl.
// CHECK-COUNT-1: func.func private @mgpuSymmetricFree
// CHECK-NOT: func.func private @mgpuSymmetricFree
// CHECK-COUNT-1: func.func private @mgpuSymmetricAlloc
// CHECK-NOT: func.func private @mgpuSymmetricAlloc
// CHECK-LABEL: func.func @two_allocs
// CHECK-COUNT-2: call @mgpuSymmetricAlloc
// CHECK-COUNT-2: call @mgpuSymmetricFree
func.func @two_allocs() {
  %a = memref.alloc() : memref<32xf32, #air.symmetric_heap>
  %b = memref.alloc() : memref<64xf32, #air.symmetric_heap>
  memref.dealloc %a : memref<32xf32, #air.symmetric_heap>
  memref.dealloc %b : memref<64xf32, #air.symmetric_heap>
  return
}

// -----

// LAST partition: cases that test the pass leaves things untouched.
// Both `ignores_non_symmetric` and `no_symmetric_alloc` are folded here
// so the trailing CHECK-NOTs only need to match against this one (final)
// partition's text.
// CHECK-LABEL: func.func @no_symmetric_changes
// CHECK: memref.alloc() : memref<1024xf32>
// CHECK: memref.alloc() : memref<32xf32>
// CHECK-NOT: mgpuSymmetricAlloc
// CHECK-NOT: mgpuSymmetricFree
func.func @no_symmetric_changes() {
  %a = memref.alloc() : memref<1024xf32>
  memref.dealloc %a : memref<1024xf32>
  %b = memref.alloc() : memref<32xf32>
  memref.dealloc %b : memref<32xf32>
  return
}
