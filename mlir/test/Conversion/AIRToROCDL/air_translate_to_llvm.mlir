//===- air_translate_to_llvm.mlir - air-translate-to-llvm pass -----------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt --air-translate-to-llvm --split-input-file %s | FileCheck %s

// 1D static memref: full peer-VA expansion shape.
// CHECK-LABEL: func.func @translate_1d
// CHECK-DAG:   %[[SRC_IDX:.+]]   = memref.extract_aligned_pointer_as_index %arg0
// CHECK-DAG:   %[[FROM_BASE:.+]] = memref.load %arg3[%arg1] : memref<?xindex>
// CHECK-DAG:   %[[TO_BASE:.+]]   = memref.load %arg3[%arg2] : memref<?xindex>
// CHECK:       %[[DIFF:.+]]      = arith.subi %[[TO_BASE]], %[[FROM_BASE]]
// CHECK:       %[[PEER_IDX:.+]]  = arith.addi %[[SRC_IDX]], %[[DIFF]]
// CHECK:       %[[PEER_I64:.+]]  = arith.index_cast %[[PEER_IDX]] : index to i64
// CHECK:       %[[PEER_PTR:.+]]  = llvm.inttoptr %[[PEER_I64]] : i64 to !llvm.ptr
// CHECK:       %[[POISON:.+]]    = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:       %[[D0:.+]] = llvm.insertvalue %[[PEER_PTR]], %[[POISON]][0]
// CHECK:       %[[D1:.+]] = llvm.insertvalue %[[PEER_PTR]], %[[D0]][1]
// CHECK:       %{{.*}}    = llvm.mlir.constant(0 : i64)
// CHECK:       %[[D2:.+]] = llvm.insertvalue %{{.*}}, %[[D1]][2]
// CHECK:       %{{.*}}    = llvm.mlir.constant(1024 : i64)
// CHECK:       %[[D3:.+]] = llvm.insertvalue %{{.*}}, %[[D2]][3, 0]
// CHECK:       %{{.*}}    = llvm.mlir.constant(1 : i64)
// CHECK:       %[[D4:.+]] = llvm.insertvalue %{{.*}}, %[[D3]][4, 0]
// CHECK:       %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[D4]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1024xf32>
// CHECK:       return %[[CAST]]
// CHECK-NOT:   air.translate
func.func @translate_1d(%src : memref<1024xf32>, %from : index, %to : index, %bases : memref<?xindex>) -> memref<1024xf32> {
  %peer = air.translate %src, %from, %to, %bases : memref<1024xf32>, memref<?xindex>
  return %peer : memref<1024xf32>
}

// -----

// 2D static memref: descriptor includes row-major strides [64, 1].
// CHECK-LABEL: func.func @translate_2d
// CHECK:       memref.load %arg3[%arg1] : memref<?xindex>
// CHECK:       memref.load %arg3[%arg2] : memref<?xindex>
// CHECK:       llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:   llvm.mlir.constant(64 : i64)
// CHECK-DAG:   llvm.mlir.constant(1 : i64)
// CHECK:       builtin.unrealized_conversion_cast {{.*}} to memref<64x64xf32, 1>
// CHECK-NOT:   air.translate
func.func @translate_2d(%src : memref<64x64xf32, 1>, %from : index, %to : index, %bases : memref<?xindex>) -> memref<64x64xf32, 1> {
  %peer = air.translate %src, %from, %to, %bases : memref<64x64xf32, 1>, memref<?xindex>
  return %peer : memref<64x64xf32, 1>
}

// -----

// Inside a gpu.func (kernel-side use): same expansion shape — purely
// memref + arith ops, no runtime call.
// CHECK-LABEL: gpu.func @kernel
// CHECK:       memref.extract_aligned_pointer_as_index
// CHECK:       memref.load %arg3[%arg1] : memref<?xindex>
// CHECK:       memref.load %arg3[%arg2] : memref<?xindex>
// CHECK:       arith.subi
// CHECK:       arith.addi
// CHECK:       builtin.unrealized_conversion_cast {{.*}} to memref<1024xf32, 1>
// CHECK:       memref.store
// CHECK-NOT:   air.translate
gpu.module @kernels {
  gpu.func @kernel(%data : memref<1024xf32, 1>, %from : index, %to : index, %bases : memref<?xindex>) kernel {
    %peer = air.translate %data, %from, %to, %bases : memref<1024xf32, 1>, memref<?xindex>
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

