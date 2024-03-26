//===- dma_to_channel.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
// CHECK: air.channel @channel_3 [2, 2]
// CHECK: air.channel @channel_2 [2, 2]
// CHECK: air.channel @channel_1 [2, 2]
// CHECK: air.channel @channel_0 [2, 2]
// CHECK-LABEL: func.func @mmult
  func.func @mmult(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64x64xi32>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %alloc, %alloc_0 : memref<64x64xi32> to memref<64x64xi32>
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_0

// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_1

// CHECK: %[[EVENT12:.*]] = air.wait_all async
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_2

// CHECK: %[[EVENT18:.*]] = air.wait_all async
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@channel_3
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %0 = affine.apply #map()[%arg2]
      %1 = affine.apply #map()[%arg3]
      scf.for %arg9 = %c0 to %c64 step %c32 {
        %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
        %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
        %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
// CHECK: air.channel.get{{.*}}@channel_0
        air.dma_memcpy_nd (%alloc_1[] [] [], %arg6[%0, %arg9] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.get{{.*}}@channel_1
        air.dma_memcpy_nd (%alloc_2[] [] [], %arg7[%arg9, %1] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.get{{.*}}@channel_2
        air.dma_memcpy_nd (%alloc_3[] [] [], %arg8[%0, %1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
// CHECK: air.channel.put{{.*}}@channel_3
        linalg.matmul ins(%alloc_1, %alloc_2 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_3 : memref<32x32xi32, 2>)
        air.dma_memcpy_nd (%arg8[%0, %1] [%c32, %c32] [%c64, %c1], %alloc_3[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
        memref.dealloc %alloc_1 : memref<32x32xi32, 2>
        memref.dealloc %alloc_2 : memref<32x32xi32, 2>
        memref.dealloc %alloc_3 : memref<32x32xi32, 2>
      }
      air.herd_terminator
    }
    return %alloc_0 : memref<64x64xi32>
  }
}
