//===- scf_parallel_to_launch_and_segment.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-par-to-launch='has-air-segment=true' %s -cse | FileCheck %s
// CHECK-LABEL: func.func @f0
// CHECK: %[[C0:.*]] = arith.constant 2 : index
// CHECK: air.launch (%[[V0:.*]], %[[V1:.*]]) in (%[[V2:.*]]=%[[C0]], %[[V3:.*]]=%[[C0]])
// CHECK: air.segment @{{.*}} args({{.*}}=%[[V0]], {{.*}}=%[[V1]], {{.*}}=%[[V2]], {{.*}}=%[[V3]])
func.func @f0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%x,%y) = (%c0,%c0) to (%c2, %c2) step (%c1,%c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// CHECK-LABEL: func.func @f1
// CHECK: %[[C1:.*]] = arith.constant 4 : index
// CHECK: air.launch (%[[V0:.*]]) in (%[[V1:.*]]=%[[C1]])
// CHECK: air.segment @{{.*}}  args({{.*}}=%[[V0]], {{.*}}=%[[V1]])
func.func @f1()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%x) = (%c0) to (%c128) step (%c32) {
    %2 = arith.muli %x, %x : index
  }
  return
}

// CHECK-LABEL: func.func @f2
// CHECK: %[[VAL_0:.*]] = arith.constant 2 : index
// CHECK: air.launch (%[[VAL_1:.*]], %[[VAL_2:.*]]) in (%[[VAL_3:.*]]=%[[VAL_0]], %[[VAL_4:.*]]=%[[VAL_0]]) {
// CHECK:   air.segment @segment_2  args(%[[VAL_5:.*]]=%[[VAL_1]], %[[VAL_6:.*]]=%[[VAL_2]], %[[VAL_7:.*]]=%[[VAL_3]], %[[VAL_8:.*]]=%[[VAL_4]]) : index, index, index, index
func.func @f2()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.forall (%x, %y) in (2, 2) {
    %2 = arith.muli %x, %y : index
  }
  return
}

// CHECK-LABEL: func.func @f3
// CHECK: air.launch
// CHECK:   air.segment @segment_3
// CHECK:     memref.alloc() : memref<1x1x64x128xbf16, 1 : i32>
// CHECK:     memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
// CHECK:     air.dma_memcpy_nd
// CHECK:     memref.dealloc
// CHECK:     memref.dealloc
func.func @f3()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %alloc = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
  %alloc_0 = memref.alloc() : memref<1x1x64x128xbf16, 1 : i32>
  scf.parallel (%x,%y) = (%c0,%c0) to (%c2, %c2) step (%c1,%c1) {
    air.dma_memcpy_nd (%alloc[] [] [], %alloc_0[] [] []) : (memref<1x1x16x8x8x4xbf16, 2 : i32>, memref<1x1x64x128xbf16, 1 : i32>)
  }
  memref.dealloc %alloc_0 : memref<1x1x64x128xbf16, 1 : i32>
  memref.dealloc %alloc : memref<1x1x16x8x8x4xbf16, 2 : i32>
  return
}
