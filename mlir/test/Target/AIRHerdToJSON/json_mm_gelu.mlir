//===- json_mm_gelu.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-translate %s -air-herds-to-json -num-rows=6 -num-cols=8 | FileCheck %s
// CHECK: "row": 5
// CHECK: "col": 7
// CHECK: partition
// CHECK: [0, "matmul_herd_0", [0, 0], [0, 1], [1, 0], [1, 1]],
// CHECK: [1, "matmul_herd_1", [0, 2], [0, 3], [1, 2], [1, 3]],
// CHECK: [2, "matmul_herd_2", [0, 4], [0, 5], [1, 4], [1, 5]],
// CHECK: [3, "matmul_herd_3", [0, 6], [0, 7], [1, 6], [1, 7]],
// CHECK: [4, "matmul_herd_4", [2, 0], [2, 1], [3, 0], [3, 1]],
// CHECK: [5, "matmul_herd_5", [2, 2], [2, 3], [3, 2], [3, 3]],
// CHECK: [6, "matmul_herd_6", [2, 4], [2, 5], [3, 4], [3, 5]],
// CHECK: [7, "matmul_herd_7", [2, 6], [2, 7], [3, 6], [3, 7]],
// CHECK: [8, "gelu_herd_0", [4, 0], [4, 1], [5, 0], [5, 1]],
// CHECK: [9, "gelu_herd_1", [4, 2], [4, 3], [5, 2], [5, 3]],
// CHECK: [10, "gelu_herd_2", [4, 4], [4, 5], [5, 4], [5, 5]],
// CHECK: [11, "gelu_herd_3", [4, 6], [4, 7], [5, 6], [5, 7]

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
#map2 = affine_map<()[s0] -> (s0 * 32)>
#map3 = affine_map<()[s0] -> (s0 * 512 + 64)>
#map4 = affine_map<()[s0] -> (s0 * 512 + 128)>
#map5 = affine_map<()[s0] -> (s0 * 512 + 192)>
#map6 = affine_map<()[s0] -> (s0 * 512 + 256)>
#map7 = affine_map<()[s0] -> (s0 * 512 + 320)>
#map8 = affine_map<()[s0] -> (s0 * 512 + 384)>
#map9 = affine_map<()[s0] -> (s0 * 512 + 448)>
#map10 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c16 = arith.constant 16 : index
    %c48 = arith.constant 48 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<24576x1024xbf16>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    memref.copy %alloc, %alloc_0 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    air.launch @launch_0 (%arg2, %arg3) in (%arg4=%c48, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0, %arg9=%alloc_1) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 6 : i64, size_y = 2 : i64} {
      air.partition @partition_0  args(%arg10=%arg2, %arg11=%arg3, %arg12=%arg4, %arg13=%arg5, %arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 3 : i64, size_y = 2 : i64} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %0 = affine.apply #map()[%arg11]
        %1 = affine.apply #map1()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%1, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%1, %0] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_0  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 0 : i64, y_loc = 0 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%1, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        %2 = affine.apply #map3()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%2, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 10 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%2, %0] [%c64, %c64] [%c1024, %c1]) {id = 11 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_1  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 2 : i64, y_loc = 0 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 12 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 13 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 14 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 15 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%2, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 16 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        %3 = affine.apply #map4()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%3, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 17 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 18 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%3, %0] [%c64, %c64] [%c1024, %c1]) {id = 19 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_2  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 4 : i64, y_loc = 0 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 20 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 21 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 22 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%3, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 24 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        %4 = affine.apply #map5()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%4, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 25 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 26 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%4, %0] [%c64, %c64] [%c1024, %c1]) {id = 27 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_3  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 6 : i64, y_loc = 0 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 28 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 29 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 30 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%4, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 32 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        %5 = affine.apply #map6()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%5, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 33 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 34 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%5, %0] [%c64, %c64] [%c1024, %c1]) {id = 35 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_4  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 36 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 37 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 38 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%5, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 40 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        %6 = affine.apply #map7()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%6, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 41 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 42 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%6, %0] [%c64, %c64] [%c1024, %c1]) {id = 43 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_5  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 2 : i64, y_loc = 2 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 44 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 45 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 46 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%6, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 48 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        %7 = affine.apply #map8()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%7, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 49 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 50 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%7, %0] [%c64, %c64] [%c1024, %c1]) {id = 51 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_6  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 4 : i64, y_loc = 2 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 52 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 53 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 54 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%7, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 56 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        %8 = affine.apply #map9()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg14[%8, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 57 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%arg18, %0] [%c64, %c64] [%c1024, %c1]) {id = 58 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%8, %0] [%c64, %c64] [%c1024, %c1]) {id = 59 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_7  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3, %arg25=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 6 : i64, y_loc = 2 : i64} {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %9 = affine.apply #map2()[%arg19]
            %10 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg23[%9, %arg26] [%c32, %c32] [%c64_7, %c1_5]) {id = 60 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg24[%arg26, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 61 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5]) {id = 62 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%9, %10] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%8, %0] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 64 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %9 = arith.muli %arg18, %c64 : index
          %10 = arith.addi %1, %9 : index
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg16[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 65 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg17[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 66 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_0  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 0 : i64, y_loc = 4 : i64} {
            %c1_4 = arith.constant 1 : index
            %c64_5 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_6 = arith.constant 2.000000e+00 : bf16
            %cst_7 = arith.constant 1.000000e+00 : bf16
            %cst_8 = arith.constant 5.000000e-01 : bf16
            %11 = affine.apply #map2()[%arg19]
            %12 = affine.apply #map2()[%arg20]
            %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
            %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%alloc_9[] [] [], %arg23[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 67 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%alloc_10[] [] [], %arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 68 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>) {
            ^bb0(%in: bf16, %out: bf16):
              %13 = math.sqrt %cst_6 : bf16
              %14 = arith.divf %in, %13 : bf16
              %15 = math.erf %14 : bf16
              %16 = arith.addf %15, %cst_7 : bf16
              %17 = arith.mulf %16, %cst_8 : bf16
              %18 = arith.mulf %in, %17 : bf16
              linalg.yield %18 : bf16
            }
            air.dma_memcpy_nd (%arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4], %alloc_10[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
            memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%10, %0] [%c64, %c64] [%c1024, %c1], %alloc_3[] [] []) {id = 70 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %9 = arith.muli %arg18, %c64 : index
          %10 = arith.addi %3, %9 : index
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg16[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 71 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg17[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 72 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_1  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 2 : i64, y_loc = 4 : i64} {
            %c1_4 = arith.constant 1 : index
            %c64_5 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_6 = arith.constant 2.000000e+00 : bf16
            %cst_7 = arith.constant 1.000000e+00 : bf16
            %cst_8 = arith.constant 5.000000e-01 : bf16
            %11 = affine.apply #map2()[%arg19]
            %12 = affine.apply #map2()[%arg20]
            %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
            %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%alloc_9[] [] [], %arg23[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 73 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%alloc_10[] [] [], %arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 74 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>) {
            ^bb0(%in: bf16, %out: bf16):
              %13 = math.sqrt %cst_6 : bf16
              %14 = arith.divf %in, %13 : bf16
              %15 = math.erf %14 : bf16
              %16 = arith.addf %15, %cst_7 : bf16
              %17 = arith.mulf %16, %cst_8 : bf16
              %18 = arith.mulf %in, %17 : bf16
              linalg.yield %18 : bf16
            }
            air.dma_memcpy_nd (%arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4], %alloc_10[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
            memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%10, %0] [%c64, %c64] [%c1024, %c1], %alloc_3[] [] []) {id = 76 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %9 = arith.muli %arg18, %c64 : index
          %10 = arith.addi %5, %9 : index
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg16[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 77 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg17[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 78 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_2  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 4 : i64, y_loc = 4 : i64} {
            %c1_4 = arith.constant 1 : index
            %c64_5 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_6 = arith.constant 2.000000e+00 : bf16
            %cst_7 = arith.constant 1.000000e+00 : bf16
            %cst_8 = arith.constant 5.000000e-01 : bf16
            %11 = affine.apply #map2()[%arg19]
            %12 = affine.apply #map2()[%arg20]
            %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
            %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%alloc_9[] [] [], %arg23[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 79 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%alloc_10[] [] [], %arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 80 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>) {
            ^bb0(%in: bf16, %out: bf16):
              %13 = math.sqrt %cst_6 : bf16
              %14 = arith.divf %in, %13 : bf16
              %15 = math.erf %14 : bf16
              %16 = arith.addf %15, %cst_7 : bf16
              %17 = arith.mulf %16, %cst_8 : bf16
              %18 = arith.mulf %in, %17 : bf16
              linalg.yield %18 : bf16
            }
            air.dma_memcpy_nd (%arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4], %alloc_10[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
            memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%10, %0] [%c64, %c64] [%c1024, %c1], %alloc_3[] [] []) {id = 82 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %9 = arith.muli %arg18, %c64 : index
          %10 = arith.addi %7, %9 : index
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg16[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 83 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg17[%10, %0] [%c64, %c64] [%c1024, %c1]) {id = 84 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_3  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%alloc_2, %arg24=%alloc_3) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 6 : i64, y_loc = 4 : i64} {
            %c1_4 = arith.constant 1 : index
            %c64_5 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_6 = arith.constant 2.000000e+00 : bf16
            %cst_7 = arith.constant 1.000000e+00 : bf16
            %cst_8 = arith.constant 5.000000e-01 : bf16
            %11 = affine.apply #map2()[%arg19]
            %12 = affine.apply #map2()[%arg20]
            %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
            %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%alloc_9[] [] [], %arg23[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 85 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%alloc_10[] [] [], %arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4]) {id = 86 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>) {
            ^bb0(%in: bf16, %out: bf16):
              %13 = math.sqrt %cst_6 : bf16
              %14 = arith.divf %in, %13 : bf16
              %15 = math.erf %14 : bf16
              %16 = arith.addf %15, %cst_7 : bf16
              %17 = arith.mulf %16, %cst_8 : bf16
              %18 = arith.mulf %in, %17 : bf16
              linalg.yield %18 : bf16
            }
            air.dma_memcpy_nd (%arg24[%11, %12] [%c32, %c32] [%c64_5, %c1_4], %alloc_10[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
            memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%10, %0] [%c64, %c64] [%c1024, %c1], %alloc_3[] [] []) {id = 88 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %alloc_1 : memref<24576x1024xbf16>
  }
}

