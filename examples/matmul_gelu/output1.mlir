//===- output1.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c16 = arith.constant 16 : index
    %c384 = arith.constant 384 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    linalg.fill ins(%cst : bf16) outs(%0 : memref<24576x1024xbf16>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    memref.copy %0, %1 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    air.launch (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%1) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> {
      air.partition  args(%arg9=%arg2, %arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%arg6, %arg14=%arg7, %arg15=%arg8) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %3 = affine.apply #map0()[%arg9]
        %4 = affine.apply #map0()[%arg10]
        scf.for %arg16 = %c0 to %c1024 step %c64 {
          %5 = memref.alloc() : memref<64x64xbf16, 1>
          %6 = memref.alloc() : memref<64x64xbf16, 1>
          %7 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%5[] [] [], %arg13[%3, %arg16] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%6[] [] [], %arg14[%arg16, %4] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%7[] [] [], %arg15[%3, %4] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @herd_0  tile (%arg17, %arg18) in (%arg19=%c2, %arg20=%c2) args(%arg21=%5, %arg22=%6, %arg23=%7) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %8 = affine.apply #map1()[%arg17]
            %9 = affine.apply #map1()[%arg18]
            scf.for %arg24 = %c0_1 to %c64_2 step %c32 {
              %10 = memref.alloc() : memref<32x32xbf16, 2>
              %11 = memref.alloc() : memref<32x32xbf16, 2>
              %12 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%10[] [] [], %arg21[%8, %arg24] [%c32, %c32] [%c64_2, %c1_0]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%11[] [] [], %arg22[%arg24, %9] [%c32, %c32] [%c64_2, %c1_0]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%12[] [] [], %arg23[%8, %9] [%c32, %c32] [%c64_2, %c1_0]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%10, %11 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%12 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg23[%8, %9] [%c32, %c32] [%c64_2, %c1_0], %12[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %10 : memref<32x32xbf16, 2>
              memref.dealloc %11 : memref<32x32xbf16, 2>
              memref.dealloc %12 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg15[%3, %4] [%c64, %c64] [%c1024, %c1], %7[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %5 : memref<64x64xbf16, 1>
          memref.dealloc %6 : memref<64x64xbf16, 1>
          memref.dealloc %7 : memref<64x64xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    air.launch (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%1, %arg7=%2) : memref<24576x1024xbf16>, memref<24576x1024xbf16> {
      air.partition  args(%arg8=%arg2, %arg9=%arg3, %arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7) : index, index, index, index, memref<24576x1024xbf16>, memref<24576x1024xbf16> {
        %c1 = arith.constant 1 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %3 = affine.apply #map0()[%arg8]
        %4 = affine.apply #map0()[%arg9]
        %5 = memref.alloc() : memref<64x64xbf16, 1>
        %6 = memref.alloc() : memref<64x64xbf16, 1>
        air.dma_memcpy_nd (%5[] [] [], %arg12[%3, %4] [%c64, %c64] [%c1024, %c1]) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.dma_memcpy_nd (%6[] [] [], %arg13[%3, %4] [%c64, %c64] [%c1024, %c1]) {id = 10 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.herd @herd_1  tile (%arg14, %arg15) in (%arg16=%c2, %arg17=%c2) args(%arg18=%5, %arg19=%6) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
          %c1_0 = arith.constant 1 : index
          %c64_1 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %cst_2 = arith.constant 2.000000e+00 : bf16
          %cst_3 = arith.constant 1.000000e+00 : bf16
          %cst_4 = arith.constant 5.000000e-01 : bf16
          %7 = affine.apply #map1()[%arg14]
          %8 = affine.apply #map1()[%arg15]
          %9 = memref.alloc() : memref<32x32xbf16, 2>
          %10 = memref.alloc() : memref<32x32xbf16, 2>
          air.dma_memcpy_nd (%9[] [] [], %arg18[%7, %8] [%c32, %c32] [%c64_1, %c1_0]) {id = 11 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          air.dma_memcpy_nd (%10[] [] [], %arg19[%7, %8] [%c32, %c32] [%c64_1, %c1_0]) {id = 12 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9 : memref<32x32xbf16, 2>) outs(%10 : memref<32x32xbf16, 2>) {
          ^bb0(%arg20: bf16, %arg21: bf16):
            %11 = math.sqrt %cst_2 : bf16
            %12 = arith.divf %arg20, %11 : bf16
            %13 = math.erf %12 : bf16
            %14 = arith.addf %13, %cst_3 : bf16
            %15 = arith.mulf %14, %cst_4 : bf16
            %16 = arith.mulf %arg20, %15 : bf16
            linalg.yield %16 : bf16
          }
          air.dma_memcpy_nd (%arg19[%7, %8] [%c32, %c32] [%c64_1, %c1_0], %10[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          memref.dealloc %9 : memref<32x32xbf16, 2>
          memref.dealloc %10 : memref<32x32xbf16, 2>
          air.herd_terminator
        }
        air.dma_memcpy_nd (%arg13[%3, %4] [%c64, %c64] [%c1024, %c1], %6[] [] []) {id = 14 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        memref.dealloc %5 : memref<64x64xbf16, 1>
        memref.dealloc %6 : memref<64x64xbf16, 1>
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %2 : memref<24576x1024xbf16>
  }
}
