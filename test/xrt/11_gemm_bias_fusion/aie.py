# aie.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air
import air.compiler.util
from air.dialects import linalg, tensor, arith, func, memref
from air.ir import *
import air.passmanager
from air.dialects import air as airdialect
from air.compiler.util import run_transform
import sys

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Tiling
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<()[s0] -> (s0 * 4)>
    #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
    #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
    #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
    #map4 = affine_map<()[s0] -> (s0 * 32)>
    #map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
    module {
      func.func @matmul_example_dispatch_0_matmul_1024x1024x512_i32(%0: memref<1024x512xi32>, %1: memref<512x1024xi32>, %2: memref<1024x1024xi32>, %3: memref<1024x1024xi32>) {
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %c15 = arith.constant 15 : index
        %alloc = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x16x16x4x4xi32, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x1x64x64xi32, 1 : i32>
        %alloc_2 = memref.alloc() : memref<1x1x64x64xi32, 1 : i32>
        %alloc_3 = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
        %alloc_4 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
        %alloc_5 = memref.alloc() : memref<1x1x32x64xi32, 1 : i32>
        %alloc_6 = memref.alloc() : memref<1x1x64x32xi32, 1 : i32>
        %alloc_7 = memref.alloc() : memref<1x1x16x16x4x4xi32, 2 : i32>
        %alloc_8 = memref.alloc() : memref<1x1x64x64xi32, 1 : i32>
        scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1024, %c1024) step (%c64, %c64) {
          %subview = memref.subview %3[%arg0, %arg1] [64, 64] [1, 1] : memref<1024x1024xi32> to memref<64x64xi32, strided<[1024, 1], offset: ?>>
          %subview_9 = memref.subview %0[%arg0, 0] [64, 32] [1, 1] : memref<1024x512xi32> to memref<64x32xi32, strided<[512, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_6[] [] [], %subview_9[] [] []) : (memref<1x1x64x32xi32, 1 : i32>, memref<64x32xi32, strided<[512, 1], offset: ?>>)
          %subview_10 = memref.subview %1[0, %arg1] [32, 64] [1, 1] : memref<512x1024xi32> to memref<32x64xi32, strided<[1024, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_5[] [] [], %subview_10[] [] []) : (memref<1x1x32x64xi32, 1 : i32>, memref<32x64xi32, strided<[1024, 1], offset: ?>>)
          scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c16, %c16) step (%c8, %c8) {
            %4 = affine.apply #map()[%arg2]
            %subview_16 = memref.subview %alloc_6[0, 0, %4, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32>
            %expand_shape = memref.expand_shape %subview_16 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 4, 8] : memref<1x1x32x32xi32, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x4x8xi32, strided<[2048, 2048, 128, 32, 8, 1], offset: ?>, 1 : i32>
            %transpose_17 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x4x8xi32, strided<[2048, 2048, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x4x8x4x8xi32, strided<[2048, 2048, 8, 128, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_4[] [] [], %transpose_17[] [] []) : (memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x4x8x4x8xi32, strided<[2048, 2048, 8, 128, 32, 1], offset: ?>, 1 : i32>)
            %5 = affine.apply #map()[%arg3]
            %subview_18 = memref.subview %alloc_5[0, 0, 0, %5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x32x64xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32>
            %expand_shape_19 = memref.expand_shape %subview_18 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 8, 8, 4] : memref<1x1x32x32xi32, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32> into memref<1x1x4x8x8x4xi32, strided<[2048, 2048, 512, 64, 4, 1], offset: ?>, 1 : i32>
            %transpose_20 = memref.transpose %expand_shape_19 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x8x8x4xi32, strided<[2048, 2048, 512, 64, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x4x8x4xi32, strided<[2048, 2048, 4, 512, 64, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_3[] [] [], %transpose_20[] [] []) : (memref<1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, strided<[2048, 2048, 4, 512, 64, 1], offset: ?>, 1 : i32>)
            %subview_21 = memref.subview %alloc_7[0, 0, %arg3, %arg2, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x16x16x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.fill ins(%c0_i32 : i32) outs(%subview_21 : memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
            linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_4, %alloc_3 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_21 : memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: i32, %in_22: i32, %out: i32):
              %6 = arith.muli %in, %in_22 : i32
              %7 = arith.addi %out, %6 : i32
              linalg.yield %7 : i32
            }
            scf.reduce 
          }
          scf.for %arg2 = %c1 to %c15 step %c1 {
            %4 = affine.apply #map4()[%arg2]
            %subview_16 = memref.subview %0[%arg0, %4] [64, 32] [1, 1] : memref<1024x512xi32> to memref<64x32xi32, strided<[512, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_6[] [] [], %subview_16[] [] []) : (memref<1x1x64x32xi32, 1 : i32>, memref<64x32xi32, strided<[512, 1], offset: ?>>)
            %5 = affine.apply #map4()[%arg2]
            %subview_17 = memref.subview %1[%5, %arg1] [32, 64] [1, 1] : memref<512x1024xi32> to memref<32x64xi32, strided<[1024, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_5[] [] [], %subview_17[] [] []) : (memref<1x1x32x64xi32, 1 : i32>, memref<32x64xi32, strided<[1024, 1], offset: ?>>)
            scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c16, %c16) step (%c8, %c8) {
              %6 = affine.apply #map()[%arg3]
              %subview_18 = memref.subview %alloc_6[0, 0, %6, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32>
              %expand_shape = memref.expand_shape %subview_18 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 4, 8] : memref<1x1x32x32xi32, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x4x8xi32, strided<[2048, 2048, 128, 32, 8, 1], offset: ?>, 1 : i32>
              %transpose_19 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x4x8xi32, strided<[2048, 2048, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x4x8x4x8xi32, strided<[2048, 2048, 8, 128, 32, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc_4[] [] [], %transpose_19[] [] []) : (memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x4x8x4x8xi32, strided<[2048, 2048, 8, 128, 32, 1], offset: ?>, 1 : i32>)
              %7 = affine.apply #map()[%arg4]
              %subview_20 = memref.subview %alloc_5[0, 0, 0, %7] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x32x64xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32>
              %expand_shape_21 = memref.expand_shape %subview_20 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 8, 8, 4] : memref<1x1x32x32xi32, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32> into memref<1x1x4x8x8x4xi32, strided<[2048, 2048, 512, 64, 4, 1], offset: ?>, 1 : i32>
              %transpose_22 = memref.transpose %expand_shape_21 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x8x8x4xi32, strided<[2048, 2048, 512, 64, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x4x8x4xi32, strided<[2048, 2048, 4, 512, 64, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc_3[] [] [], %transpose_22[] [] []) : (memref<1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, strided<[2048, 2048, 4, 512, 64, 1], offset: ?>, 1 : i32>)
              %subview_23 = memref.subview %alloc_7[0, 0, %arg4, %arg3, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x16x16x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
              linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_4, %alloc_3 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_23 : memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>) {
              ^bb0(%in: i32, %in_24: i32, %out: i32):
                %8 = arith.muli %in, %in_24 : i32
                %9 = arith.addi %out, %8 : i32
                linalg.yield %9 : i32
              }
              scf.reduce 
            }
          }
          %subview_11 = memref.subview %0[%arg0, 480] [64, 32] [1, 1] : memref<1024x512xi32> to memref<64x32xi32, strided<[512, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_6[] [] [], %subview_11[] [] []) : (memref<1x1x64x32xi32, 1 : i32>, memref<64x32xi32, strided<[512, 1], offset: ?>>)
          %subview_12 = memref.subview %1[480, %arg1] [32, 64] [1, 1] : memref<512x1024xi32> to memref<32x64xi32, strided<[1024, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_5[] [] [], %subview_12[] [] []) : (memref<1x1x32x64xi32, 1 : i32>, memref<32x64xi32, strided<[1024, 1], offset: ?>>)
          %subview_13 = memref.subview %2[%arg0, %arg1] [64, 64] [1, 1] : memref<1024x1024xi32> to memref<64x64xi32, strided<[1024, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_2[] [] [], %subview[] [] []) : (memref<1x1x64x64xi32, 1 : i32>, memref<64x64xi32, strided<[1024, 1], offset: ?>>)
          air.dma_memcpy_nd (%alloc_1[] [] [], %subview_13[] [] []) : (memref<1x1x64x64xi32, 1 : i32>, memref<64x64xi32, strided<[1024, 1], offset: ?>>)
          scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c16, %c16) step (%c8, %c8) {
            %4 = affine.apply #map()[%arg2]
            %subview_16 = memref.subview %alloc_6[0, 0, %4, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32>
            %expand_shape = memref.expand_shape %subview_16 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 4, 8] : memref<1x1x32x32xi32, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x4x8xi32, strided<[2048, 2048, 128, 32, 8, 1], offset: ?>, 1 : i32>
            %transpose_17 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x4x8xi32, strided<[2048, 2048, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x4x8x4x8xi32, strided<[2048, 2048, 8, 128, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_4[] [] [], %transpose_17[] [] []) : (memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x4x8x4x8xi32, strided<[2048, 2048, 8, 128, 32, 1], offset: ?>, 1 : i32>)
            %5 = affine.apply #map()[%arg3]
            %subview_18 = memref.subview %alloc_5[0, 0, 0, %5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x32x64xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32>
            %expand_shape_19 = memref.expand_shape %subview_18 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 8, 8, 4] : memref<1x1x32x32xi32, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32> into memref<1x1x4x8x8x4xi32, strided<[2048, 2048, 512, 64, 4, 1], offset: ?>, 1 : i32>
            %transpose_20 = memref.transpose %expand_shape_19 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x8x8x4xi32, strided<[2048, 2048, 512, 64, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x4x8x4xi32, strided<[2048, 2048, 4, 512, 64, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_3[] [] [], %transpose_20[] [] []) : (memref<1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, strided<[2048, 2048, 4, 512, 64, 1], offset: ?>, 1 : i32>)
            %subview_21 = memref.subview %alloc_7[0, 0, %arg3, %arg2, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x16x16x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_4, %alloc_3 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_21 : memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: i32, %in_29: i32, %out: i32):
              %6 = arith.muli %in, %in_29 : i32
              %7 = arith.addi %out, %6 : i32
              linalg.yield %7 : i32
            }
            %subview_22 = memref.subview %alloc_1[0, 0, %4, %5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x64xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32>
            %expand_shape_23 = memref.expand_shape %subview_22 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 8, 4] : memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x8x4xi32, strided<[4096, 4096, 256, 64, 4, 1], offset: ?>, 1 : i32>
            %transpose_24 = memref.transpose %expand_shape_23 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x8x4xi32, strided<[4096, 4096, 256, 64, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 4, 256, 64, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc[] [] [], %transpose_24[] [] []) : (memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 4, 256, 64, 1], offset: ?>, 1 : i32>)
            %subview_25 = memref.subview %alloc_2[0, 0, %4, %5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x64xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32>
            %subview_26 = memref.subview %alloc_0[0, 0, %arg3, %arg2, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x16x16x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
            %expand_shape_27 = memref.expand_shape %subview_25 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 8, 4] : memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x8x4xi32, strided<[4096, 4096, 256, 64, 4, 1], offset: ?>, 1 : i32>
            %transpose_28 = memref.transpose %expand_shape_27 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x8x4xi32, strided<[4096, 4096, 256, 64, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 4, 256, 64, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%subview_26[] [] [], %transpose_28[] [] []) : (memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>, memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 4, 256, 64, 1], offset: ?>, 1 : i32>)
            linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%subview_21, %alloc : memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%subview_26 : memref<1x1x8x8x4x4xi32, strided<[4096, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: i32, %in_29: i32, %out: i32):
              %6 = arith.addi %in, %in_29 : i32
              linalg.yield %6 : i32
            }
            scf.reduce 
          }
          %transpose = memref.transpose %alloc_0 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x16x16x4x4xi32, 2 : i32> to memref<1x1x16x4x16x4xi32, strided<[4096, 4096, 16, 4, 256, 1]>, 2 : i32>
          air.dma_memcpy_nd (%alloc_8[] [] [], %transpose[] [] []) : (memref<1x1x64x64xi32, 1 : i32>, memref<1x1x16x4x16x4xi32, strided<[4096, 4096, 16, 4, 256, 1]>, 2 : i32>)
          %subview_14 = memref.subview %alloc_8[0, 0, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x64xi32, 1 : i32> to memref<64x64xi32, 1 : i32>
          %transpose_15 = memref.transpose %subview_14 (d0, d1) -> (d0, d1) : memref<64x64xi32, 1 : i32> to memref<64x64xi32, strided<[64, 1]>, 1 : i32>
          air.dma_memcpy_nd (%subview[] [] [], %transpose_15[] [] []) : (memref<64x64xi32, strided<[1024, 1], offset: ?>>, memref<64x64xi32, strided<[64, 1]>, 1 : i32>)
          scf.reduce 
        }
        memref.dealloc %alloc_8 : memref<1x1x64x64xi32, 1 : i32>
        memref.dealloc %alloc_7 : memref<1x1x16x16x4x4xi32, 2 : i32>
        memref.dealloc %alloc_6 : memref<1x1x64x32xi32, 1 : i32>
        memref.dealloc %alloc_5 : memref<1x1x32x64xi32, 1 : i32>
        memref.dealloc %alloc_4 : memref<1x1x4x8x4x8xi32, 2 : i32>
        memref.dealloc %alloc_3 : memref<1x1x8x4x8x4xi32, 2 : i32>
        memref.dealloc %alloc_2 : memref<1x1x64x64xi32, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x1x64x64xi32, 1 : i32>
        memref.dealloc %alloc_0 : memref<1x1x16x16x4x4xi32, 2 : i32>
        memref.dealloc %alloc : memref<1x1x8x8x4x4xi32, 2 : i32>
        return
      }
    }
    """

    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "buffer-results-to-out-params",
                "air-par-to-herd{depth=1}",
                "air-par-to-launch{has-air-segment=true}",
                "air-copy-to-dma",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ###############################################
    # Extract event dependency and optimize schedule
    ###############################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-dependency",
                "air-dma-to-channel",
                "canonicalize",
                "cse",
                "air-dependency-canonicalize",
                "canonicalize",
                "cse",
                "air-isolate-async-dma-loop-nests",
                "canonicalize",
                "cse",
                "air-fuse-channels{aggressive-mode=L1}",
                "canonicalize",
                "cse",
                "func.func(air-loop-fusion)",
                "air-label-scf-for-to-ping-pong",
                "air-ping-pong-transform{keep-memref-dealloc=true}",
                "canonicalize",
                "cse",
                "air-specialize-channel-wrap-and-stride",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## Place herd to segment
    ################################################

    air_async_module = Module.parse(str(air_module))
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "func.func(air-collapse-herd{max-col-size=4})",
                "canonicalize",
                "cse",
                "air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}",
                "canonicalize",
                "cse",
                "func.func(air-renumber-dma)",
                "func.func(convert-linalg-to-loops)",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## MLIR-AIR to MLIR-AIE
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "canonicalize",
                "cse",
                "air-to-aie{row-offset=2 col-offset=0 device=npu1_4col emit-while-loop=true}",
                "canonicalize",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## MLIR-AIR runtime lowering
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "func.func(air-opt-shim-dma-bds{device=npu1_4col})",
                "air-to-std",
                "canonicalize",
                "symbol-dce",
                "func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})",
                "func.func(air-unroll-outer-affine-loops{depth=2})",
                "affine-expand-index-ops",
                "airrt-to-npu",
                "canonicalize",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open("aie.mlir", "w") as f:
        f.write(str(air_module))
