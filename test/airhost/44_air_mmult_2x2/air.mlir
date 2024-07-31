//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<64x64xi32>
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%alloc, %arg8=%arg0, %arg9=%arg1) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      air.segment @segment_0  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
        %c2 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c2, %arg18=%c2) args(%arg21=%arg12, %arg22=%arg13, %arg23=%arg14) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
          %c1_0 = arith.constant 1 : index
          %c0_i32 = arith.constant 0 : i32
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %2 = affine.apply #map1()[%arg15]
          %3 = affine.apply #map1()[%arg16]
          %4 = arith.addi %c64, %2 : index
          %5 = arith.addi %c64, %3 : index
          %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<32x32xi32, 2>)
          scf.for %arg24 = %c0 to %c64 step %c32 {
            %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
            %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
            air.dma_memcpy_nd (%alloc_2[] [] [], %arg22[%4, %arg24] [%c32, %c32] [%c64, %c1_0]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            air.dma_memcpy_nd (%alloc_3[] [] [], %arg23[%arg24, %5] [%c32, %c32] [%c64, %c1_0]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_2, %alloc_3 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_1 : memref<32x32xi32, 2>)
            memref.dealloc %alloc_2 : memref<32x32xi32, 2>
            memref.dealloc %alloc_3 : memref<32x32xi32, 2>
          }
          air.dma_memcpy_nd (%arg21[%4, %5] [%c32, %c32] [%c64, %c1_0], %alloc_1[] [] []) {id = 3 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
          memref.dealloc %alloc_1 : memref<32x32xi32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    memref.copy %alloc, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    return
  }
}

