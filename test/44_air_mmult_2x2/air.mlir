//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%a0: memref<64x64xi32>, %a1: memref<64x64xi32>, %a2: memref<64x64xi32>) {
    air.segment @segment0 args(%arg0=%a0, %arg1=%a1, %arg2=%a2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      %1 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 64 {
          affine.store %c0_i32, %0[%arg3, %arg4] : memref<64x64xi32>
        }
      }
      memref.copy %0, %1 : memref<64x64xi32> to memref<64x64xi32>
      air.herd  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%1) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {sym_name = "herd_0"} {
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %2 = affine.apply #map()[%arg3]
        %3 = affine.apply #map()[%arg4]
        scf.for %arg10 = %c0 to %c64 step %c32 {
          %4 = memref.alloc() : memref<32x32xi32, 2>
          %5 = memref.alloc() : memref<32x32xi32, 2>
          %6 = memref.alloc() : memref<32x32xi32, 2>
          air.dma_memcpy_nd (%4[] [] [], %arg7[%2, %arg10] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          air.dma_memcpy_nd (%5[] [] [], %arg8[%arg10, %3] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          air.dma_memcpy_nd (%6[] [] [], %arg9[%2, %3] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          affine.for %arg11 = 0 to 32 {
            affine.for %arg12 = 0 to 32 {
              affine.for %arg13 = 0 to 32 {
                %7 = affine.load %4[%arg11, %arg13] : memref<32x32xi32, 2>
                %8 = affine.load %5[%arg13, %arg12] : memref<32x32xi32, 2>
                %9 = affine.load %6[%arg11, %arg12] : memref<32x32xi32, 2>
                %10 = arith.muli %7, %8 : i32
                %11 = arith.addi %9, %10 : i32
                affine.store %11, %6[%arg11, %arg12] : memref<32x32xi32, 2>
              }
            }
          }
          air.dma_memcpy_nd (%arg9[%2, %3] [%c32, %c32] [%c64, %c1], %6[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
          memref.dealloc %4 : memref<32x32xi32, 2>
          memref.dealloc %5 : memref<32x32xi32, 2>
          memref.dealloc %6 : memref<32x32xi32, 2>
        }
        air.herd_terminator
      }
      memref.copy %1, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    }
    return
  }
}

