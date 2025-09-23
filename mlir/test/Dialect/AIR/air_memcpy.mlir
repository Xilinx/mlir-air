//===- air_memcpy.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize %s | FileCheck %s
module {

// CHECK-LABEL: module

// CHECK: func.func @test1
func.func @test1(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  air.herd tile (%arg1, %arg2) in (%arg3=%c4, %arg4=%c1) args(%arg5=%arg0) : memref<4096xi32>attributes {sym_name = "memcpy_nd"} {
    %c32 = arith.constant 32 : index
    %0 = arith.muli %arg1, %c32 : index
    %1 = memref.alloc() : memref<32xi32, 2>
    %c1_0 = arith.constant 1 : index
    // CHECK: air.dma_memcpy_nd
    air.dma_memcpy_nd (%1[] [] [], %arg5[%0] [%c32] [%c1_0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    air.dma_memcpy_nd (%arg5[%0] [%c32] [%c1_0], %1[] [] []) {id = 2 : i32} : (memref<4096xi32>, memref<32xi32, 2>)
    memref.dealloc %1 : memref<32xi32, 2>
  }
  return
}

// CHECK-LABEL: test2
// CHECK: %[[alloc:.*]] = memref.alloc() : memref<64x256x1xf32, 2>
// CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<64x16x4x4xf32, 2>
// CHECK: air.dma_memcpy_nd (%[[alloc_0]][] [] [], %[[alloc]][%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c64{{.*}}, %c16{{.*}}, %c4{{.*}}, %c4{{.*}}] [%c4{{.*}}, %c1024{{.*}}, %c256{{.*}}, %c1{{.*}}]) : (memref<64x16x4x4xf32, 2>, memref<64x256x1xf32, 2>)

func.func @test2() {
  %alloc_11 = memref.alloc() : memref<64x256x1xf32, 2>
  %subview_12 = memref.subview %alloc_11[0, 0, 0] [64, 256, 1] [1, 1, 1] : memref<64x256x1xf32, 2> to memref<64x256xf32, strided<[256, 1]>, 2>
  %alloc_15 = memref.alloc() : memref<64x16x4x4xf32, 2>
  %c0_16 = arith.constant 0 : index
  %c4_17 = arith.constant 4 : index
  %c1024_18 = arith.constant 1024 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c64_19 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c4_20 = arith.constant 4 : index
  %c4_21 = arith.constant 4 : index
  air.dma_memcpy_nd (%alloc_15[] [] [], %subview_12[%c0_16, %c0_16, %c0_16, %c0_16] [%c64_19, %c16, %c4_20, %c4_21] [%c4_17, %c1024_18, %c256, %c1]) : (memref<64x16x4x4xf32, 2>, memref<64x256xf32, strided<[256, 1]>, 2>)
  return
}

}
