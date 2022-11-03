//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "model"} {
  func.func @forward(%arg0: memref<64x64x32xi32>, %arg1: memref<64x64x32xi32>, %arg2: memref<64x64x32xi32>) {
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x32xi32>
    air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%0) : memref<64x64x32xi32>, memref<64x64x32xi32>, memref<64x64x32xi32> {
      %c1 = arith.constant 1 : index
      %c2048 = arith.constant 2048 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %1 = affine.apply #map0()[%arg3]
      %2 = affine.apply #map0()[%arg4]
      scf.for %arg10 = %c0 to %c32 step %c16 {
        scf.for %arg11 = %c0 to %c32 step %c16 {
          scf.for %arg12 = %c0 to %c32 step %c4 {
            %3 = arith.addi %1, %arg10 : index
            %4 = arith.addi %2, %arg11 : index
            %5 = memref.alloc() : memref<16x16x4xi32, 2>
            %6 = memref.alloc() : memref<16x16x4xi32, 2>
            %7 = memref.alloc() : memref<16x16x4xi32, 2>
            air.dma_memcpy_nd (%5[] [] [], %arg7[%3, %4, %arg12] [%c16, %c16, %c4] [%c2048, %c32, %c1]) {id = 1 : i32} : (memref<16x16x4xi32, 2>, memref<64x64x32xi32>)
            air.dma_memcpy_nd (%6[] [] [], %arg8[%3, %4, %arg12] [%c16, %c16, %c4] [%c2048, %c32, %c1]) {id = 2 : i32} : (memref<16x16x4xi32, 2>, memref<64x64x32xi32>)
            air.dma_memcpy_nd (%7[] [] [], %arg9[%3, %4, %arg12] [%c16, %c16, %c4] [%c2048, %c32, %c1]) {id = 3 : i32} : (memref<16x16x4xi32, 2>, memref<64x64x32xi32>)
            linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6 : memref<16x16x4xi32, 2>, memref<16x16x4xi32, 2>) outs(%7 : memref<16x16x4xi32, 2>) {
            ^bb0(%arg13: i32, %arg14: i32, %arg15: i32):
              %8 = arith.addi %arg13, %arg14 : i32
              linalg.yield %8 : i32
            }
            air.dma_memcpy_nd (%arg9[%3, %4, %arg12] [%c16, %c16, %c4] [%c2048, %c32, %c1], %7[] [] []) {id = 4 : i32} : (memref<64x64x32xi32>, memref<16x16x4xi32, 2>)
            memref.dealloc %5 : memref<16x16x4xi32, 2>
            memref.dealloc %6 : memref<16x16x4xi32, 2>
            memref.dealloc %7 : memref<16x16x4xi32, 2>
          }
        }
      }
      air.herd_terminator
    }
    memref.copy %0, %arg2 : memref<64x64x32xi32> to memref<64x64x32xi32>
    return
  }
}

