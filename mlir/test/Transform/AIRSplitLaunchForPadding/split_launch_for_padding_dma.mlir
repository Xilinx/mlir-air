//===- split_launch_for_padding_dma.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding='use-dma-memcpy=true' | FileCheck %s

// M=500, M_TILE=64 → launchM=8, last M-block has 500-7*64=52 rows
// N=500, N_TILE=32 → launchN=16, last N-block has 500-15*32=20 cols
// Expected: 4 partitions (interior 7x15, m_boundary 1x15, n_boundary 7x1, corner 1x1)

// CHECK-LABEL: func.func @matmul_padding_kernel

// Interior: 7x15 launch, no padding on any DMA.
// CHECK: air.segment @matmul_padding_kernel_0_interior
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 1 : i32}
// CHECK-NOT: pad_after
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 2 : i32}
// CHECK-NOT: pad_after

// M-boundary: A DMA has reduced src_sizes[0] from 64→52 and pad_after, B unchanged.
// CHECK: air.segment @matmul_padding_kernel_0_m_boundary
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 1 : i32, pad_after = array<i32: 12, 0>, pad_before = array<i32: 0, 0>}
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 2 : i32}
// CHECK-NOT: pad_after

// N-boundary: A DMA unchanged, B DMA has reduced src_sizes[1] from 32→20 and pad_after.
// CHECK: air.segment @matmul_padding_kernel_0_n_boundary
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 1 : i32}
// CHECK-NOT: pad_after
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 2 : i32, pad_after = array<i32: 0, 12>, pad_before = array<i32: 0, 0>}

// Corner: both A and B have padding.
// CHECK: air.segment @matmul_padding_kernel_0_corner
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 1 : i32, pad_after = array<i32: 12, 0>, pad_before = array<i32: 0, 0>}
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 2 : i32, pad_after = array<i32: 0, 12>, pad_before = array<i32: 0, 0>}

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_padding_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c8, %arg13=%c16, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {air.actual_sizes = array<i64: 500, 500, 1>} {
      air.segment @matmul_padding_kernel_0  args(%arg18=%arg9, %arg19=%arg10, %arg20=%arg15, %arg21=%arg16, %arg22=%arg17) : index, index, memref<*xbf16>, memref<*xbf16>, memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c16_1 = arith.constant 16 : index
        %c32000 = arith.constant 32000 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c500 = arith.constant 500 : index
        %c0_i32 = arith.constant 0 : i32
        %c784_i32 = arith.constant 784 : i32
        %c16_i32 = arith.constant 16 : i32
        %0 = tensor.empty() : tensor<64x32xf32>
        %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %2 = arith.muli %arg18, %c64 : index
        %3 = arith.muli %arg19, %c32 : index
        %4 = scf.for %arg23 = %c0_i32 to %c784_i32 step %c16_i32 iter_args(%arg24 = %1) -> (tensor<64x32xf32>)  : i32 {
          %9 = arith.index_cast %arg23 : i32 to index
          %10 = arith.muli %9, %c500 : index
          %11 = arith.addi %2, %10 : index
          %alloc = memref.alloc() : memref<64x16xbf16, 1 : i32>
          air.dma_memcpy_nd (%alloc[] [] [], %arg20[%c0, %11] [%c64, %c16_1] [%c1_0, %c500]) {id = 1 : i32} : (memref<64x16xbf16, 1 : i32>, memref<*xbf16>)
          %12 = bufferization.to_tensor %alloc restrict writable : memref<64x16xbf16, 1 : i32> to tensor<64x16xbf16>
          %13 = arith.addi %10, %3 : index
          %alloc_2 = memref.alloc() : memref<16x32xbf16, 1 : i32>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg21[%c0, %13] [%c16_1, %c32] [%c500, %c1_0]) {id = 2 : i32} : (memref<16x32xbf16, 1 : i32>, memref<*xbf16>)
          %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x32xbf16, 1 : i32> to tensor<16x32xbf16>
          %15 = linalg.matmul ins(%12, %14 : tensor<64x16xbf16>, tensor<16x32xbf16>) outs(%1 : tensor<64x32xf32>) -> tensor<64x32xf32>
          %16 = bufferization.alloc_tensor() : tensor<64x32xf32>
          %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg24, %15 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%16 : tensor<64x32xf32>) {
          ^bb0(%in: f32, %in_3: f32, %out: f32):
            %18 = arith.addf %in, %in_3 : f32
            linalg.yield %18 : f32
          } -> tensor<64x32xf32>
          scf.yield %17 : tensor<64x32xf32>
        }
        %5 = tensor.empty() : tensor<64x32xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<64x32xf32>) outs(%5 : tensor<64x32xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %9 = arith.truncf %in : f32 to bf16
          linalg.yield %9 : bf16
        } -> tensor<64x32xbf16>
        %7 = arith.muli %arg18, %c32000 : index
        %8 = arith.addi %7, %3 : index
        %reinterpret_cast = memref.reinterpret_cast %arg22 to offset: [%8], sizes: [64, 32], strides: [500, 1] : memref<*xbf16> to memref<64x32xbf16, strided<[500, 1], offset: ?>>
        bufferization.materialize_in_destination %6 in writable %reinterpret_cast : (tensor<64x32xbf16>, memref<64x32xbf16, strided<[500, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
