//===- rank_to_mgpu.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// RUN: air-opt %s --split-input-file -air-rank-to-mgpu                       | FileCheck %s
// RUN: air-opt %s --split-input-file -air-rank-to-mgpu='heap-size=536870912' | FileCheck %s --check-prefix=HEAPOPT

// CHECK-LABEL: func.func @test_rank_1d
// CHECK: call @mgpuSymmetricHeapInit
// CHECK-NOT: air.rank
// CHECK: %[[R:.*]] = call @mgpuGetRank() : () -> i32
// CHECK: arith.extsi %[[R]] : i32 to i64
// CHECK: arith.index_cast
// CHECK: call @mgpuSymmetricHeapDestroy
// CHECK: return

// HEAPOPT-LABEL: func.func @test_rank_1d
// HEAPOPT: arith.constant 536870912 : i64
// HEAPOPT: call @mgpuSymmetricHeapInit
func.func @test_rank_1d(%arg0: memref<16x16xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) args(%a=%arg0) : memref<16x16xf32> {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls = %c1) args(%la=%a) : memref<16x16xf32> {
      air.launch_terminator
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_rank_2d
// 2D rank delinearization: id_x = flat % sx, id_y = flat / sx
// CHECK: %[[FLAT:.*]] = arith.index_cast
// CHECK: %[[IDX:.*]] = arith.remsi %[[FLAT]], %{{.*}}
// CHECK: %[[IDY:.*]] = arith.divsi %[[FLAT]], %{{.*}}
// CHECK-NOT: air.rank
func.func @test_rank_2d(%arg0: memref<16x16xf32>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  air.rank (%rx, %ry) in (%sx = %c2, %sy = %c4) args(%a=%arg0) : memref<16x16xf32> {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls = %c1) args(%la=%a) : memref<16x16xf32> {
      air.launch_terminator
    }
  }
  return
}

// -----

// Default heap size is 256 MB = 268435456.
// CHECK-LABEL: func.func @test_rank_default_heap
// CHECK: arith.constant 268435456 : i64
// CHECK: call @mgpuSymmetricHeapInit
func.func @test_rank_default_heap() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
  }
  return
}

// -----

// Async form: air.rank with async result token. Pass should produce a wait_all
// to replace the token, and the body should still be inlined.
// CHECK-LABEL: func.func @test_rank_async
// CHECK: call @mgpuSymmetricHeapInit
// CHECK: call @mgpuGetRank
// CHECK-NOT: air.rank
// CHECK: air.wait_all
// CHECK: call @mgpuSymmetricHeapDestroy
func.func @test_rank_async() -> !air.async.token {
  %c2 = arith.constant 2 : index
  %t = air.rank async (%rx) in (%sx = %c2) {
  }
  return %t : !air.async.token
}

// -----

// Async dependency: air.rank async [%dep]. Pass must insert a blocking
// wait_all on the dependency before lowering the rank body.
// CHECK-LABEL: func.func @test_rank_async_dep
// CHECK: %[[DEP:.*]] = air.wait_all async
// CHECK: air.wait_all [%[[DEP]]]
// CHECK: call @mgpuGetRank
// CHECK-NOT: air.rank
func.func @test_rank_async_dep() {
  %c2 = arith.constant 2 : index
  %dep = air.wait_all async
  %t = air.rank async [%dep] (%rx) in (%sx = %c2) {
  }
  return
}

// -----

// Multiple air.rank ops in one function: heap init should appear once
// (at function entry) and destroy once (before return), regardless of how
// many rank ops are inlined. Each rank produces its own mgpuGetRank().
// CHECK-LABEL: func.func @test_multiple_ranks
// CHECK-COUNT-1: call @mgpuSymmetricHeapInit
// CHECK-COUNT-2: call @mgpuGetRank
// CHECK-COUNT-1: call @mgpuSymmetricHeapDestroy
// CHECK-NOT: air.rank
func.func @test_multiple_ranks() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
  }
  air.rank (%rx) in (%sx = %c2) {
  }
  return
}

// -----

// Multiple returns: destroy should be inserted before EACH return path.
// CHECK-LABEL: func.func @test_multiple_returns
// CHECK-COUNT-1: call @mgpuSymmetricHeapInit
// CHECK-COUNT-2: call @mgpuSymmetricHeapDestroy
func.func @test_multiple_returns(%cond: i1) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
  }
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  return
^bb2:
  return
}

// -----

// Kernel operand mapping: a value passed as args(%a=%arg0) should be
// substituted into the inlined body so that uses of the block arg are
// replaced with the original SSA value.
// CHECK-LABEL: func.func @test_kernel_args(
// CHECK-SAME: %[[ARG0:.*]]: memref<16x16xf32>
// CHECK-NOT: air.rank
// The store should reference the function arg directly, not a block arg.
// CHECK: memref.store %{{.*}}, %[[ARG0]]
func.func @test_kernel_args(%arg0: memref<16x16xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) args(%a=%arg0) : memref<16x16xf32> {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    memref.store %cst, %a[%c0, %c0] : memref<16x16xf32>
  }
  return
}

// -----

// Idempotent extern decls: only one decl of each mgpu* function in the
// module, even with multiple ranks across multiple functions.
// CHECK-COUNT-1: func.func private @mgpuGetRank
// CHECK-NOT: func.func private @mgpuGetRank
// CHECK-COUNT-1: func.func private @mgpuSymmetricHeapDestroy
// CHECK-NOT: func.func private @mgpuSymmetricHeapDestroy
// CHECK-COUNT-1: func.func private @mgpuSymmetricHeapInit
// CHECK-NOT: func.func private @mgpuSymmetricHeapInit
func.func @test_decls_in_func_a() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {}
  return
}
func.func @test_decls_in_func_b() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {}
  return
}

// -----

// A function with NO air.rank should be left completely untouched.
// (Placed last in the file so CHECK-NOTs aren't matched against later
// partitions that legitimately contain mgpu* decls.)
// CHECK-LABEL: func.func @test_no_rank
// CHECK-NOT: mgpuSymmetricHeapInit
// CHECK-NOT: mgpuSymmetricHeapDestroy
// CHECK-NOT: mgpuGetRank
func.func @test_no_rank(%arg0: memref<16x16xf32>) -> memref<16x16xf32> {
  return %arg0 : memref<16x16xf32>
}
