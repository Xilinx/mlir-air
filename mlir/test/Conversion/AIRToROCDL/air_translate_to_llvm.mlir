//===- air_translate_to_llvm.mlir - air-translate-to-llvm pass -----------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// RUN: air-opt --air-translate-to-llvm --split-input-file %s | FileCheck %s

// 1D static memref: full peer-VA expansion shape.
// CHECK-LABEL: func.func @translate_1d
// CHECK-DAG:   %[[SRC_IDX:.+]] = memref.extract_aligned_pointer_as_index %arg0
// CHECK-DAG:   %[[SRC_I64:.+]] = arith.index_cast %[[SRC_IDX]]
// CHECK-DAG:   %[[SRC_PTR:.+]] = llvm.inttoptr %[[SRC_I64]]
// CHECK-DAG:   %[[FROM_I64:.+]] = arith.index_cast %arg1
// CHECK-DAG:   %[[TO_I64:.+]]   = arith.index_cast %arg2
// CHECK-DAG:   %[[FROM_GEP:.+]] = llvm.getelementptr %arg3[%[[FROM_I64]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-DAG:   %[[FROM_BASE:.+]] = llvm.load %[[FROM_GEP]]
// CHECK-DAG:   %[[TO_GEP:.+]]   = llvm.getelementptr %arg3[%[[TO_I64]]]  : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-DAG:   %[[TO_BASE:.+]]  = llvm.load %[[TO_GEP]]
// CHECK-DAG:   %[[FROM_INT:.+]] = llvm.ptrtoint %[[FROM_BASE]]
// CHECK-DAG:   %[[TO_INT:.+]]   = llvm.ptrtoint %[[TO_BASE]]
// CHECK:       %[[DIFF:.+]]     = arith.subi %[[TO_INT]], %[[FROM_INT]]
// CHECK:       %[[PEER:.+]]     = llvm.getelementptr %[[SRC_PTR]][%[[DIFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:       %[[POISON:.+]]   = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:       %[[D0:.+]] = llvm.insertvalue %[[PEER]], %[[POISON]][0]
// CHECK:       %[[D1:.+]] = llvm.insertvalue %[[PEER]], %[[D0]][1]
// CHECK:       %{{.*}}    = llvm.mlir.constant(0 : i64)
// CHECK:       %[[D2:.+]] = llvm.insertvalue %{{.*}}, %[[D1]][2]
// CHECK:       %{{.*}}    = llvm.mlir.constant(1024 : i64)
// CHECK:       %[[D3:.+]] = llvm.insertvalue %{{.*}}, %[[D2]][3, 0]
// CHECK:       %{{.*}}    = llvm.mlir.constant(1 : i64)
// CHECK:       %[[D4:.+]] = llvm.insertvalue %{{.*}}, %[[D3]][4, 0]
// CHECK:       %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[D4]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1024xf32>
// CHECK:       return %[[CAST]]
// CHECK-NOT:   air.translate
func.func @translate_1d(%src : memref<1024xf32>, %from : index, %to : index, %bases : !llvm.ptr) -> memref<1024xf32> {
  %peer = air.translate %src, %from, %to, %bases : memref<1024xf32>, !llvm.ptr
  return %peer : memref<1024xf32>
}

// -----

// 2D static memref: descriptor includes row-major strides [64, 1].
// CHECK-LABEL: func.func @translate_2d
// CHECK:       llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:   llvm.mlir.constant(64 : i64)
// CHECK-DAG:   llvm.mlir.constant(1 : i64)
// CHECK:       builtin.unrealized_conversion_cast {{.*}} to memref<64x64xf32, 1>
// CHECK-NOT:   air.translate
func.func @translate_2d(%src : memref<64x64xf32, 1>, %from : index, %to : index, %bases : !llvm.ptr) -> memref<64x64xf32, 1> {
  %peer = air.translate %src, %from, %to, %bases : memref<64x64xf32, 1>, !llvm.ptr
  return %peer : memref<64x64xf32, 1>
}

// -----

// Inside a gpu.func (kernel-side use): same expansion shape — purely
// arithmetic, no runtime call.
// CHECK-LABEL: gpu.func @kernel
// CHECK:       memref.extract_aligned_pointer_as_index
// CHECK:       arith.subi
// CHECK:       llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:       builtin.unrealized_conversion_cast {{.*}} to memref<1024xf32, 1>
// CHECK:       memref.store
// CHECK-NOT:   air.translate
gpu.module @kernels {
  gpu.func @kernel(%data : memref<1024xf32, 1>, %from : index, %to : index, %bases : !llvm.ptr) kernel {
    %peer = air.translate %data, %from, %to, %bases : memref<1024xf32, 1>, !llvm.ptr
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42.0 : f32
    memref.store %c42, %peer[%c0] : memref<1024xf32, 1>
    gpu.return
  }
}

// -----

// No air.translate: pass is a no-op.
// CHECK-LABEL: func.func @noop
// CHECK-NEXT:    return
// CHECK-NOT:   memref.extract_aligned_pointer_as_index
// CHECK-NOT:   llvm.mlir.poison
func.func @noop(%a : memref<8xf32>) -> memref<8xf32> {
  return %a : memref<8xf32>
}

