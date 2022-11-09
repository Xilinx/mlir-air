//===- matmul_gelu_random_shapes.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds="num-rows=6 num-cols=11" | air-translate -air-herds-to-json -num-rows=6 -num-cols=11 |& FileCheck %s

// CHECK: "row": 5, 
// CHECK: "col": 10
// CHECK: partition

// CHECK: [0, "matmul_herd_0", [5, 7], [4, 7]],
// CHECK: [1, "matmul_herd_1", [3, 5], [3, 6], [2, 5], [2, 6]],
// CHECK: [2, "matmul_herd_2", [4, 3], [4, 4], [3, 3], [3, 4], [2, 3], [2, 4]],
// CHECK: [3, "matmul_herd_3", [5, 2], [4, 2], [3, 2]],
// CHECK: [4, "matmul_herd_4", [2, 0], [2, 1], [2, 2], [1, 0], [1, 1], [1, 2], [0, 0], [0, 1], [0, 2]],
// CHECK: [5, "matmul_herd_5", [4, 8], [4, 9]],
// CHECK: [6, "matmul_herd_6", [1, 3], [1, 4], [1, 5], [1, 6], [0, 3], [0, 4], [0, 5], [0, 6]],
// CHECK: [7, "matmul_herd_7", [3, 7], [3, 8], [2, 7], [2, 8]],
// CHECK: [8, "gelu_herd_0", [3, 9], [3, 10], [2, 9], [2, 10]],
// CHECK: [9, "gelu_herd_1", [4, 0], [4, 1], [3, 0], [3, 1]],
// CHECK: [10, "gelu_herd_2", [5, 5], [5, 6], [4, 5], [4, 6]],
// CHECK: [11, "gelu_herd_3", [1, 7], [1, 8], [1, 9], [1, 10], [0, 7], [0, 8], [0, 9], [0, 10]]


#map0 = affine_map<()[s0] -> (s0 * 64)>
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
    %0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    linalg.fill ins(%cst : bf16) outs(%0 : memref<24576x1024xbf16>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    memref.copy %0, %1 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    air.launch @launch_0 (%arg2, %arg3) in (%arg4=%c48, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%1, %arg9=%2) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 6 : i64, size_y = 2 : i64} {
      air.partition @partition_0  args(%arg10=%arg2, %arg11=%arg3, %arg12=%arg4, %arg13=%arg5, %arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 3 : i64, size_y = 2 : i64} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %3 = affine.apply #map0()[%arg11]
        %4 = affine.apply #map1()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%4, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%4, %3] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_0  tile (%arg19, %arg20) in (%arg21=%c1, %arg22=%c2) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%4, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        %5 = affine.apply #map3()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%5, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 10 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%5, %3] [%c64, %c64] [%c1024, %c1]) {id = 11 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_1  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 12 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 13 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 14 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 15 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%5, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 16 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        %6 = affine.apply #map4()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%6, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 17 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 18 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%6, %3] [%c64, %c64] [%c1024, %c1]) {id = 19 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_2  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c3) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 20 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 21 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 22 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%6, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 24 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        %7 = affine.apply #map5()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%7, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 25 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 26 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%7, %3] [%c64, %c64] [%c1024, %c1]) {id = 27 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_3  tile (%arg19, %arg20) in (%arg21=%c1, %arg22=%c3) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 28 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 29 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 30 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%7, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 32 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        %8 = affine.apply #map6()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%8, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 33 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 34 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%8, %3] [%c64, %c64] [%c1024, %c1]) {id = 35 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_4  tile (%arg19, %arg20) in (%arg21=%c3, %arg22=%c3) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 36 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 37 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 38 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%8, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 40 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        %9 = affine.apply #map7()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%9, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 41 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 42 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%9, %3] [%c64, %c64] [%c1024, %c1]) {id = 43 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_5  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 44 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 45 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 46 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%9, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 48 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        %10 = affine.apply #map8()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%10, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 49 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 50 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%10, %3] [%c64, %c64] [%c1024, %c1]) {id = 51 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_6  tile (%arg19, %arg20) in (%arg21=%c4, %arg22=%c2) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 52 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 53 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 54 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%10, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 56 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        %11 = affine.apply #map9()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %12 = memref.alloc() : memref<64x64xbf16, 1>
          %13 = memref.alloc() : memref<64x64xbf16, 1>
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%12[] [] [], %arg14[%11, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 57 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 58 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%14[] [] [], %arg16[%11, %3] [%c64, %c64] [%c1024, %c1]) {id = 59 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @matmul_herd_7  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %15 = affine.apply #map2()[%arg19]
            %16 = affine.apply #map2()[%arg20]
            scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              %18 = memref.alloc() : memref<32x32xbf16, 2>
              %19 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 60 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 61 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 62 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %17 : memref<32x32xbf16, 2>
              memref.dealloc %18 : memref<32x32xbf16, 2>
              memref.dealloc %19 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%11, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 64 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %12 : memref<64x64xbf16, 1>
          memref.dealloc %13 : memref<64x64xbf16, 1>
          memref.dealloc %14 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg18, %c64 : index
          %13 = arith.addi %4, %12 : index
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          %15 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%14[] [] [], %arg16[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 65 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%15[] [] [], %arg17[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 66 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_0  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%14, %arg24=%15) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c64_1 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_2 = arith.constant 2.000000e+00 : bf16
            %cst_3 = arith.constant 1.000000e+00 : bf16
            %cst_4 = arith.constant 5.000000e-01 : bf16
            %16 = affine.apply #map2()[%arg19]
            %17 = affine.apply #map2()[%arg20]
            %18 = memref.alloc() : memref<32x32xbf16, 2>
            %19 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%18[] [] [], %arg23[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 67 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%19[] [] [], %arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 68 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%18 : memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>) {
            ^bb0(%arg25: bf16, %arg26: bf16):
              %20 = math.sqrt %cst_2 : bf16
              %21 = arith.divf %arg25, %20 : bf16
              %22 = math.erf %21 : bf16
              %23 = arith.addf %22, %cst_3 : bf16
              %24 = arith.mulf %23, %cst_4 : bf16
              %25 = arith.mulf %arg25, %24 : bf16
              linalg.yield %25 : bf16
            }
            air.dma_memcpy_nd (%arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0], %19[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %18 : memref<32x32xbf16, 2>
            memref.dealloc %19 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%13, %3] [%c64, %c64] [%c1024, %c1], %15[] [] []) {id = 70 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %14 : memref<64x64xbf16, 1>
          memref.dealloc %15 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg18, %c64 : index
          %13 = arith.addi %6, %12 : index
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          %15 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%14[] [] [], %arg16[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 71 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%15[] [] [], %arg17[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 72 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_1  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%14, %arg24=%15) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c64_1 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_2 = arith.constant 2.000000e+00 : bf16
            %cst_3 = arith.constant 1.000000e+00 : bf16
            %cst_4 = arith.constant 5.000000e-01 : bf16
            %16 = affine.apply #map2()[%arg19]
            %17 = affine.apply #map2()[%arg20]
            %18 = memref.alloc() : memref<32x32xbf16, 2>
            %19 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%18[] [] [], %arg23[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 73 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%19[] [] [], %arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 74 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%18 : memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>) {
            ^bb0(%arg25: bf16, %arg26: bf16):
              %20 = math.sqrt %cst_2 : bf16
              %21 = arith.divf %arg25, %20 : bf16
              %22 = math.erf %21 : bf16
              %23 = arith.addf %22, %cst_3 : bf16
              %24 = arith.mulf %23, %cst_4 : bf16
              %25 = arith.mulf %arg25, %24 : bf16
              linalg.yield %25 : bf16
            }
            air.dma_memcpy_nd (%arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0], %19[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %18 : memref<32x32xbf16, 2>
            memref.dealloc %19 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%13, %3] [%c64, %c64] [%c1024, %c1], %15[] [] []) {id = 76 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %14 : memref<64x64xbf16, 1>
          memref.dealloc %15 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg18, %c64 : index
          %13 = arith.addi %8, %12 : index
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          %15 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%14[] [] [], %arg16[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 77 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%15[] [] [], %arg17[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 78 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_2  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%14, %arg24=%15) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c64_1 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_2 = arith.constant 2.000000e+00 : bf16
            %cst_3 = arith.constant 1.000000e+00 : bf16
            %cst_4 = arith.constant 5.000000e-01 : bf16
            %16 = affine.apply #map2()[%arg19]
            %17 = affine.apply #map2()[%arg20]
            %18 = memref.alloc() : memref<32x32xbf16, 2>
            %19 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%18[] [] [], %arg23[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 79 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%19[] [] [], %arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 80 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%18 : memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>) {
            ^bb0(%arg25: bf16, %arg26: bf16):
              %20 = math.sqrt %cst_2 : bf16
              %21 = arith.divf %arg25, %20 : bf16
              %22 = math.erf %21 : bf16
              %23 = arith.addf %22, %cst_3 : bf16
              %24 = arith.mulf %23, %cst_4 : bf16
              %25 = arith.mulf %arg25, %24 : bf16
              linalg.yield %25 : bf16
            }
            air.dma_memcpy_nd (%arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0], %19[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %18 : memref<32x32xbf16, 2>
            memref.dealloc %19 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%13, %3] [%c64, %c64] [%c1024, %c1], %15[] [] []) {id = 82 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %14 : memref<64x64xbf16, 1>
          memref.dealloc %15 : memref<64x64xbf16, 1>
        }
        scf.for %arg18 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg18, %c64 : index
          %13 = arith.addi %10, %12 : index
          %14 = memref.alloc() : memref<64x64xbf16, 1>
          %15 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%14[] [] [], %arg16[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 83 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%15[] [] [], %arg17[%13, %3] [%c64, %c64] [%c1024, %c1]) {id = 84 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @gelu_herd_3  tile (%arg19, %arg20) in (%arg21=%c4, %arg22=%c2) args(%arg23=%14, %arg24=%15) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c64_1 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %cst_2 = arith.constant 2.000000e+00 : bf16
            %cst_3 = arith.constant 1.000000e+00 : bf16
            %cst_4 = arith.constant 5.000000e-01 : bf16
            %16 = affine.apply #map2()[%arg19]
            %17 = affine.apply #map2()[%arg20]
            %18 = memref.alloc() : memref<32x32xbf16, 2>
            %19 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%18[] [] [], %arg23[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 85 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%19[] [] [], %arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0]) {id = 86 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.generic {indexing_maps = [#map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%18 : memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>) {
            ^bb0(%arg25: bf16, %arg26: bf16):
              %20 = math.sqrt %cst_2 : bf16
              %21 = arith.divf %arg25, %20 : bf16
              %22 = math.erf %21 : bf16
              %23 = arith.addf %22, %cst_3 : bf16
              %24 = arith.mulf %23, %cst_4 : bf16
              %25 = arith.mulf %arg25, %24 : bf16
              linalg.yield %25 : bf16
            }
            air.dma_memcpy_nd (%arg24[%16, %17] [%c32, %c32] [%c64_1, %c1_0], %19[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %18 : memref<32x32xbf16, 2>
            memref.dealloc %19 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg17[%13, %3] [%c64, %c64] [%c1024, %c1], %15[] [] []) {id = 88 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %14 : memref<64x64xbf16, 1>
          memref.dealloc %15 : memref<64x64xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %2 : memref<24576x1024xbf16>
  }
}