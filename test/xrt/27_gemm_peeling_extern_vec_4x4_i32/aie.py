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
    module {
      func.func @matmul_512x512_512xi32__dispatch_0_matmul_512x512x512_i32(%0 : memref<512x512xi32>, %1 : memref<512x512xi32>, %2 : memref<512x512xi32>) {
        %c4 = arith.constant 4 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c15 = arith.constant 15 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %c1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x4x32x32xi32, 1 : i32>
        %alloc_2 = memref.alloc() : memref<4x1x32x32xi32, 1 : i32>
        %alloc_3 = memref.alloc() : memref<4x4x8x8x4x4xi32, 2 : i32>
        %alloc_4 = memref.alloc() : memref<4x4x32x32xi32, 1 : i32>
        scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
          %subview = memref.subview %2[%arg0, %arg1] [128, 128] [1, 1] : memref<512x512xi32> to memref<128x128xi32, strided<[512, 1], offset: ?>>
          %subview_5 = memref.subview %0[%arg0, 0] [128, 32] [1, 1] : memref<512x512xi32> to memref<128x32xi32, strided<[512, 1], offset: ?>>
          %expand_shape = memref.expand_shape %subview_5 [[0, 1], [2, 3]] output_shape [4, 32, 1, 32] : memref<128x32xi32, strided<[512, 1], offset: ?>> into memref<4x32x1x32xi32, strided<[16384, 512, 32, 1], offset: ?>>
          %transpose = memref.transpose %expand_shape (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<4x32x1x32xi32, strided<[16384, 512, 32, 1], offset: ?>> to memref<4x1x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_2[] [] [], %transpose[] [] []) : (memref<4x1x32x32xi32, 1 : i32>, memref<4x1x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>)
          %subview_6 = memref.subview %1[0, %arg1] [32, 128] [1, 1] : memref<512x512xi32> to memref<32x128xi32, strided<[512, 1], offset: ?>>
          %expand_shape_7 = memref.expand_shape %subview_6 [[0, 1], [2, 3]] output_shape [1, 32, 4, 32] : memref<32x128xi32, strided<[512, 1], offset: ?>> into memref<1x32x4x32xi32, strided<[16384, 512, 32, 1], offset: ?>>
          %transpose_8 = memref.transpose %expand_shape_7 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<1x32x4x32xi32, strided<[16384, 512, 32, 1], offset: ?>> to memref<1x4x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_1[] [] [], %transpose_8[] [] []) : (memref<1x4x32x32xi32, 1 : i32>, memref<1x4x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>)
          scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
            %subview_16 = memref.subview %alloc_2[%arg2, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x1x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_17 = memref.expand_shape %subview_16 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 4, 8] : memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32>
            %transpose_18 = memref.transpose %expand_shape_17 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x4x8x4x8xi32, strided<[1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_18[] [] []) : (memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x4x8x4x8xi32, strided<[1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>)
            %subview_19 = memref.subview %alloc_1[0, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x4x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_20 = memref.expand_shape %subview_19 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 8, 8, 4] : memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x4x8x8x4xi32, strided<[4096, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32>
            %transpose_21 = memref.transpose %expand_shape_20 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x8x8x4xi32, strided<[4096, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc[] [] [], %transpose_21[] [] []) : (memref<1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>)
            %subview_22 = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.fill ins(%c0_i32 : i32) outs(%subview_22 : memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_22 : memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: i32, %in_23: i32, %out: i32):
              %3 = arith.muli %in, %in_23 : i32
              %4 = arith.addi %out, %3 : i32
              linalg.yield %4 : i32
            }
            scf.reduce 
          }
          scf.for %arg2 = %c1 to %c15 step %c1 {
            %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg2]
            %subview_16 = memref.subview %0[%arg0, %3] [128, 32] [1, 1] : memref<512x512xi32> to memref<128x32xi32, strided<[512, 1], offset: ?>>
            %expand_shape_17 = memref.expand_shape %subview_16 [[0, 1], [2, 3]] output_shape [4, 32, 1, 32] : memref<128x32xi32, strided<[512, 1], offset: ?>> into memref<4x32x1x32xi32, strided<[16384, 512, 32, 1], offset: ?>>
            %transpose_18 = memref.transpose %expand_shape_17 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<4x32x1x32xi32, strided<[16384, 512, 32, 1], offset: ?>> to memref<4x1x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_2[] [] [], %transpose_18[] [] []) : (memref<4x1x32x32xi32, 1 : i32>, memref<4x1x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>)
            %subview_19 = memref.subview %1[%3, %arg1] [32, 128] [1, 1] : memref<512x512xi32> to memref<32x128xi32, strided<[512, 1], offset: ?>>
            %expand_shape_20 = memref.expand_shape %subview_19 [[0, 1], [2, 3]] output_shape [1, 32, 4, 32] : memref<32x128xi32, strided<[512, 1], offset: ?>> into memref<1x32x4x32xi32, strided<[16384, 512, 32, 1], offset: ?>>
            %transpose_21 = memref.transpose %expand_shape_20 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<1x32x4x32xi32, strided<[16384, 512, 32, 1], offset: ?>> to memref<1x4x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_1[] [] [], %transpose_21[] [] []) : (memref<1x4x32x32xi32, 1 : i32>, memref<1x4x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>)
            scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
              %subview_22 = memref.subview %alloc_2[%arg3, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x1x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32>
              %expand_shape_23 = memref.expand_shape %subview_22 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 4, 8] : memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32>
              %transpose_24 = memref.transpose %expand_shape_23 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x4x8x4x8xi32, strided<[1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_24[] [] []) : (memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x4x8x4x8xi32, strided<[1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>)
              %subview_25 = memref.subview %alloc_1[0, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x4x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32>
              %expand_shape_26 = memref.expand_shape %subview_25 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 8, 8, 4] : memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x4x8x8x4xi32, strided<[4096, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32>
              %transpose_27 = memref.transpose %expand_shape_26 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x8x8x4xi32, strided<[4096, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc[] [] [], %transpose_27[] [] []) : (memref<1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>)
              %subview_28 = memref.subview %alloc_3[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_28 : memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) {
              ^bb0(%in: i32, %in_29: i32, %out: i32):
                %4 = arith.muli %in, %in_29 : i32
                %5 = arith.addi %out, %4 : i32
                linalg.yield %5 : i32
              }
              scf.reduce 
            }
          }
          %subview_9 = memref.subview %0[%arg0, 480] [128, 32] [1, 1] : memref<512x512xi32> to memref<128x32xi32, strided<[512, 1], offset: ?>>
          %expand_shape_10 = memref.expand_shape %subview_9 [[0, 1], [2, 3]] output_shape [4, 32, 1, 32] : memref<128x32xi32, strided<[512, 1], offset: ?>> into memref<4x32x1x32xi32, strided<[16384, 512, 32, 1], offset: ?>>
          %transpose_11 = memref.transpose %expand_shape_10 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<4x32x1x32xi32, strided<[16384, 512, 32, 1], offset: ?>> to memref<4x1x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_2[] [] [], %transpose_11[] [] []) : (memref<4x1x32x32xi32, 1 : i32>, memref<4x1x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>)
          %subview_12 = memref.subview %1[480, %arg1] [32, 128] [1, 1] : memref<512x512xi32> to memref<32x128xi32, strided<[512, 1], offset: ?>>
          %expand_shape_13 = memref.expand_shape %subview_12 [[0, 1], [2, 3]] output_shape [1, 32, 4, 32] : memref<32x128xi32, strided<[512, 1], offset: ?>> into memref<1x32x4x32xi32, strided<[16384, 512, 32, 1], offset: ?>>
          %transpose_14 = memref.transpose %expand_shape_13 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<1x32x4x32xi32, strided<[16384, 512, 32, 1], offset: ?>> to memref<1x4x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_1[] [] [], %transpose_14[] [] []) : (memref<1x4x32x32xi32, 1 : i32>, memref<1x4x32x32xi32, strided<[16384, 32, 512, 1], offset: ?>>)
          scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
            %subview_16 = memref.subview %alloc_2[%arg2, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x1x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_17 = memref.expand_shape %subview_16 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 4, 8] : memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32>
            %transpose_18 = memref.transpose %expand_shape_17 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x4x8x4x8xi32, strided<[1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_18[] [] []) : (memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x4x8x4x8xi32, strided<[1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>)
            %subview_19 = memref.subview %alloc_1[0, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x4x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_20 = memref.expand_shape %subview_19 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 8, 8, 4] : memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x4x8x8x4xi32, strided<[4096, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32>
            %transpose_21 = memref.transpose %expand_shape_20 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x8x8x4xi32, strided<[4096, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc[] [] [], %transpose_21[] [] []) : (memref<1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>)
            %subview_22 = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_22 : memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: i32, %in_25: i32, %out: i32):
              %3 = arith.muli %in, %in_25 : i32
              %4 = arith.addi %out, %3 : i32
              linalg.yield %4 : i32
            }
            %subview_23 = memref.subview %alloc_4[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x4x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32>
            %transpose_24 = memref.transpose %subview_22 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x8x8x4x4xi32, strided<[4096, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32> to memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 16, 4, 128, 1], offset: ?>, 2 : i32>
            air.dma_memcpy_nd (%subview_23[] [] [], %transpose_24[] [] []) : (memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 1 : i32>, memref<1x1x8x4x8x4xi32, strided<[4096, 1024, 16, 4, 128, 1], offset: ?>, 2 : i32>)
            scf.reduce 
          }
          %transpose_15 = memref.transpose %alloc_4 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<4x4x32x32xi32, 1 : i32> to memref<4x32x4x32xi32, strided<[4096, 32, 1024, 1]>, 1 : i32>
          air.dma_memcpy_nd (%subview[] [] [], %transpose_15[] [] []) : (memref<128x128xi32, strided<[512, 1], offset: ?>>, memref<4x32x4x32xi32, strided<[4096, 32, 1024, 1]>, 1 : i32>)
          scf.reduce 
        }
        memref.dealloc %alloc_4 : memref<4x4x32x32xi32, 1 : i32>
        memref.dealloc %alloc_3 : memref<4x4x8x8x4x4xi32, 2 : i32>
        memref.dealloc %alloc_2 : memref<4x1x32x32xi32, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x4x32x32xi32, 1 : i32>
        memref.dealloc %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32>
        memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
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
                "air-dependency-schedule-opt",
                "air-specialize-dma-broadcast",
                "air-dma-to-channel",
                "canonicalize",
                "cse",
                "air-dependency-canonicalize",
                "canonicalize",
                "cse",
                "air-isolate-async-dma-loop-nests",
                "canonicalize",
                "cse",
                "air-fuse-channels",
                "canonicalize",
                "cse",
                ### Scaling to 4 AIE columns
                "func.func(air-split-l2-memref)",
                "air-isolate-async-dma-loop-nests",
                ###
                "canonicalize",
                "cse",
                "func.func(air-loop-fusion)",
                "air-label-scf-for-to-ping-pong",
                "air-ping-pong-transform",
                "canonicalize",
                "cse",
                "func.func(air-opt-memtile-dma-bds{device=npu1_4col})",
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
                "func.func(affine-loop-opt{affine-opt-tile-sizes=2,2})",
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
