//===- lower_air_execute_pattern.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=lower-air-execute' | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 32)>
#set0 = affine_set<(d0, d1)[s0] : (d0 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %ci1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<64x64xi32>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %0, %1 : memref<64x64xi32> to memref<64x64xi32>
    air.herd  tile (%arg3, %arg4) in (%arg5=%ci1, %arg6=%ci1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%1) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32, sym_name = "herd_0"} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      // CHECK: affine.apply #map()[%{{.*}}]
      // CHECK-NEXT: [[T0:%.*]] = air.wait_all async
      %asyncToken, %valOut = air.execute -> (index) {
        %6 = affine.apply #map()[%arg3]
        air.execute_terminator %6 : index
      } {id = 5 : i32}
      // CHECK-NEXT: affine.apply #map()[%{{.*}}]
      // CHECK-NEXT: [[T1:%.*]] = air.wait_all async
      %asyncToken_0, %valOut_1 = air.execute -> (index) {
        %6 = affine.apply #map()[%arg4]
        air.execute_terminator %6 : index
      } {id = 6 : i32}
      // CHECK-NEXT: air.wait_all async [[[T0]], [[T1]]]
      %2 = air.wait_all async [%asyncToken, %asyncToken_0] 
      %6 = memref.alloc() : memref<32x32xi32, 2>
      // CHECK: [[T2:%.*]] = air.dma_memcpy_nd async
      // CHECK: [[T3:%.*]] = air.dma_memcpy_nd async
      %4 = air.dma_memcpy_nd async [%2] (%6[] [] [], %arg9[%valOut, %valOut_1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
      %5 = air.dma_memcpy_nd async [%4] (%arg9[%valOut, %valOut_1] [%c32, %c32] [%c64, %c1], %6[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
      // CHECK: air.wait_all [[[T2]], [[T3]]]
      %asyncToken_4 = air.execute [%4, %5] {
        memref.dealloc %6 : memref<32x32xi32, 2>
      } {id = 15 : i32}
      air.herd_terminator
    }
    memref.copy %1, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    return
  }
}

