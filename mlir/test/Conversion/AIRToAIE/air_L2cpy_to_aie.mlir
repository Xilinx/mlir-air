//===- air_L2cpy_to_aie.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=7' | FileCheck %s
// CHECK: [[T:%.*]] = AIE.tile(7, 2)
// CHECK: [[L:%.*]] = AIE.lock([[T]], {{.*}})
// CHECK: {{.*}} = AIE.mem([[T:.*]])  {
// CHECK:   AIE.useLock([[L]], Acquire, 0)
// CHECK:   AIE.dmaBd(<{{.*}} : memref<1024xi32, 2>, 0, 0>, 0)
// CHECK:   AIE.useLock([[L]], Release, 1)
// CHECK: {{.*}} = AIE.core([[T]])  {
// CHECK:   AIE.useLock([[L]], Acquire, 1)
// CHECK:   AIE.useLock([[L]], Release, 0)
// CHECK: AIE.flow({{.*}}, PLIO : 4, [[T]], DMA : 0)
module {

func.func @foo(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  %buf0 = memref.alloc() : memref<1024xi32, 1>
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %buf0, %ext1 = %arg1) : memref<1024xi32, 1>, memref<1024xi32> attributes { } {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 0 : index
    %buf1 = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy_nd (%buf1[][][], %ext0[%c0][%c0][%c1024]) : (memref<1024xi32, 2>, memref<1024xi32, 1>)
    memref.dealloc %buf1 : memref<1024xi32, 2>
    air.herd_terminator
  }
  memref.dealloc %buf0 : memref<1024xi32, 1>
  return
}

}
