//===- dead_global_cleanup.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that dead memref.global / memref.get_global ops created by
// outlineAIECores for L3 herd buffer arguments are cleaned up after
// DMA lowering.  See issue #1404.

// The intermediate stage (to-aie-mlir) creates the globals:
// RUN: air-opt %s -air-to-aie='test-patterns=to-aie-mlir' | FileCheck %s --check-prefix=INTERMEDIATE

// The full pipeline should remove them:
// RUN: air-opt %s -air-to-aie="use-objectfifo=false row-offset=1 col-offset=1 device=xcvc1902 generate-shim-dma=true" | FileCheck %s --check-prefix=CLEAN

// Intermediate stage must have the globals (created by outlineAIECores):
// INTERMEDIATE: memref.global{{.*}}__air_herd_arg
// INTERMEDIATE: memref.get_global{{.*}}__air_herd_arg

// Full pipeline must have cleaned them up:
// CLEAN-NOT: memref.global{{.*}}__air_herd_arg
// CLEAN-NOT: memref.get_global{{.*}}__air_herd_arg
// CLEAN: aie.device
// CLEAN: aie.core

module {
  func.func @three_l3_herd_args(%a0: memref<64x64xi32>, %a1: memref<64x64xi32>, %a2: memref<64x64xi32>) {
    air.segment @segment0 args(%arg0=%a0, %arg1=%a1, %arg2=%a2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      // Three L3 (memory space 0) herd args, all accessed exclusively via DMA.
      air.herd @herd_0 tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c1_0 = arith.constant 1 : index
        %4 = memref.alloc() : memref<32x32xi32, 2>
        %5 = memref.alloc() : memref<32x32xi32, 2>
        %6 = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%4[] [] [], %arg7[%c0, %c0] [%c32, %c32] [%c64, %c1_0]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        air.dma_memcpy_nd (%5[] [] [], %arg8[%c0, %c0] [%c32, %c32] [%c64, %c1_0]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        linalg.matmul ins(%4, %5 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%6 : memref<32x32xi32, 2>)
        air.dma_memcpy_nd (%arg9[%c0, %c0] [%c32, %c32] [%c64, %c1_0], %6[] [] []) {id = 3 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
        memref.dealloc %4 : memref<32x32xi32, 2>
        memref.dealloc %5 : memref<32x32xi32, 2>
        memref.dealloc %6 : memref<32x32xi32, 2>
      }
    }
    return
  }
}
