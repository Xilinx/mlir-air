//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {

func.func @graph(%arg0 : memref<512x512xi32>, %arg1 : memref<512x512xi32>, %arg2 : memref<512x512xi32>, %arg3 : memref<64x64xi32, 1>, %arg4 : memref<64x64xi32, 1>, %arg5 : memref<64x64xi32, 1>) -> () {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  scf.for %arg6 = %c0 to %c512 step %c64 {
    scf.for %arg7 = %c0 to %c512 step %c64 {
      air.dma_memcpy_nd (%arg3[][][], %arg0[%arg6, %arg7][%c64, %c64][%c512, %c0]) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<512x512xi32>)
      air.dma_memcpy_nd (%arg4[][][], %arg1[%arg6, %arg7][%c64, %c64][%c512, %c0]) {id = 2 : i32} : (memref<64x64xi32, 1>, memref<512x512xi32>)
      %herd_cols = arith.constant 1 : index
      %herd_rows = arith.constant 1 : index
      air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg3, %ext1 = %arg4, %ext2 = %arg5) : memref<64x64xi32, 1>, memref<64x64xi32, 1>, memref<64x64xi32, 1> attributes { sym_name="herd_0"} {
        %d0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %d64 = arith.constant 64 : index
        scf.for %ha0 = %d0 to %d64 step %c32 {
          scf.for %ha1 = %d0 to %d64 step %c32 {
            %buf0 = memref.alloc() {sym_name = "scratch_a"}: memref<32x32xi32, 2>
            %buf1 = memref.alloc() {sym_name = "scratch_b"}: memref<32x32xi32, 2>
            %buf2 = memref.alloc() {sym_name = "scratch_c"}: memref<32x32xi32, 2>
            air.dma_memcpy_nd (%buf0[][][], %ext0[%ha0, %ha1][%c32, %c32][%d64, %d0]) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
            air.dma_memcpy_nd (%buf1[][][], %ext1[%ha0, %ha1][%c32, %c32][%d64, %d0]) {id = 5 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
            affine.for %ha3 = 0 to 32 {
              affine.for %ha4 = 0 to 32 {
                %a3 = affine.load %buf0[%ha3, %ha4] : memref<32x32xi32, 2>
                %b3 = affine.load %buf1[%ha3, %ha4] : memref<32x32xi32, 2> 
                %c3 = arith.addi %a3, %b3 : i32
                affine.store %c3, %buf2[%ha3, %ha4] : memref<32x32xi32, 2>
              }
            }
            air.dma_memcpy_nd (%ext2[%ha0, %ha1][%c32, %c32][%d64, %d0], %buf2[][][]) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
            memref.dealloc %buf0 : memref<32x32xi32, 2>
            memref.dealloc %buf1 : memref<32x32xi32, 2>
            memref.dealloc %buf2 : memref<32x32xi32, 2>
          }
        }
        air.herd_terminator
      }
      air.dma_memcpy_nd (%arg2[%arg6, %arg7][%c64, %c64][%c512, %c0], %arg5[][][]) {id = 8 : i32} : (memref<512x512xi32>, memref<64x64xi32, 1>)
    }
  }
  return
}

}


