//===- matmul_gelu_shifted_anchor.mlir ------------------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-place-herds="num-rows=8 num-cols=10 row-anchor=2 col-anchor=1" | FileCheck %s

// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 2
// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 3
// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 4
// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 5
// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 6
// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 7
// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 8
// CHECK: air.herd {{.*}} attributes {x_loc = 1 {{.*}} y_loc = 9

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
      air.segment @segment_0  args(%arg10=%arg2, %arg11=%arg3, %arg12=%arg4, %arg13=%arg5, %arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 3 : i64, size_y = 2 : i64} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c10 = arith.constant 10 : index
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
          air.herd @matmul_herd_0  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
          air.herd @matmul_herd_1  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
          air.herd @matmul_herd_2  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
          air.herd @matmul_herd_3  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
          air.herd @matmul_herd_4  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
          air.herd @matmul_herd_5  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
          air.herd @matmul_herd_6  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
          air.herd @matmul_herd_7  tile (%arg19, %arg20) in (%arg21=%c10, %arg22=%c1) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
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
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %2 : memref<24576x1024xbf16>
  }
}