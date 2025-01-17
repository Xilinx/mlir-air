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
      func.func @batch_matmul_dispatch_0_batch_matmul_2x128x128x256_i32(%0 : memref<2x128x256xi32>, %1 : memref<2x256x128xi32>, %2 : memref<2x128x128xi32>) {
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c2 = arith.constant 2 : index
        %c7 = arith.constant 7 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %c1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<1x1x1x8x4x8x4xi32, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x1x4x8x4x8xi32, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x1x2x32x32xi32, 1 : i32>
        %alloc_2 = memref.alloc() : memref<1x2x1x32x32xi32, 1 : i32>
        %alloc_3 = memref.alloc() : memref<1x2x2x8x8x4x4xi32, 2 : i32>
        %alloc_4 = memref.alloc() : memref<1x2x2x32x32xi32, 1 : i32>
        scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c2, %c128, %c128) step (%c1, %c64, %c64) {
          %subview = memref.subview %2[%arg0, %arg1, %arg2] [1, 64, 64] [1, 1, 1] : memref<2x128x128xi32> to memref<1x64x64xi32, strided<[16384, 128, 1], offset: ?>>
          %subview_5 = memref.subview %0[%arg0, %arg1, 0] [1, 64, 32] [1, 1, 1] : memref<2x128x256xi32> to memref<1x64x32xi32, strided<[32768, 256, 1], offset: ?>>
          %expand_shape = memref.expand_shape %subview_5 [[0], [1, 2], [3, 4]] output_shape [1, 2, 32, 1, 32] : memref<1x64x32xi32, strided<[32768, 256, 1], offset: ?>> into memref<1x2x32x1x32xi32, strided<[32768, 8192, 256, 32, 1], offset: ?>>
          %transpose = memref.transpose %expand_shape (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x2x32x1x32xi32, strided<[32768, 8192, 256, 32, 1], offset: ?>> to memref<1x2x1x32x32xi32, strided<[32768, 8192, 32, 256, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_2[] [] [], %transpose[] [] []) : (memref<1x2x1x32x32xi32, 1 : i32>, memref<1x2x1x32x32xi32, strided<[32768, 8192, 32, 256, 1], offset: ?>>)
          %subview_6 = memref.subview %1[%arg0, 0, %arg2] [1, 32, 64] [1, 1, 1] : memref<2x256x128xi32> to memref<1x32x64xi32, strided<[32768, 128, 1], offset: ?>>
          %expand_shape_7 = memref.expand_shape %subview_6 [[0], [1, 2], [3, 4]] output_shape [1, 1, 32, 2, 32] : memref<1x32x64xi32, strided<[32768, 128, 1], offset: ?>> into memref<1x1x32x2x32xi32, strided<[32768, 4096, 128, 32, 1], offset: ?>>
          %transpose_8 = memref.transpose %expand_shape_7 (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x1x32x2x32xi32, strided<[32768, 4096, 128, 32, 1], offset: ?>> to memref<1x1x2x32x32xi32, strided<[32768, 4096, 32, 128, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_1[] [] [], %transpose_8[] [] []) : (memref<1x1x2x32x32xi32, 1 : i32>, memref<1x1x2x32x32xi32, strided<[32768, 4096, 32, 128, 1], offset: ?>>)
          scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
            %subview_16 = memref.subview %alloc_2[0, %arg3, 0, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : memref<1x2x1x32x32xi32, 1 : i32> to memref<1x1x1x32x32xi32, strided<[2048, 1024, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_17 = memref.expand_shape %subview_16 [[0], [1], [2], [3, 4], [5, 6]] output_shape [1, 1, 1, 8, 4, 4, 8] : memref<1x1x1x32x32xi32, strided<[2048, 1024, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x1x8x4x4x8xi32, strided<[2048, 1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32>
            %transpose_18 = memref.transpose %expand_shape_17 (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d3, d4, d6) : memref<1x1x1x8x4x4x8xi32, strided<[2048, 1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x1x4x8x4x8xi32, strided<[2048, 1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_18[] [] []) : (memref<1x1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x1x4x8x4x8xi32, strided<[2048, 1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>)
            %subview_19 = memref.subview %alloc_1[0, 0, %arg4, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : memref<1x1x2x32x32xi32, 1 : i32> to memref<1x1x1x32x32xi32, strided<[2048, 2048, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_20 = memref.expand_shape %subview_19 [[0], [1], [2], [3, 4], [5, 6]] output_shape [1, 1, 1, 4, 8, 8, 4] : memref<1x1x1x32x32xi32, strided<[2048, 2048, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x1x4x8x8x4xi32, strided<[2048, 2048, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32>
            %transpose_21 = memref.transpose %expand_shape_20 (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d3, d4, d6) : memref<1x1x1x4x8x8x4xi32, strided<[2048, 2048, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32> to memref<1x1x1x8x4x8x4xi32, strided<[2048, 2048, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc[] [] [], %transpose_21[] [] []) : (memref<1x1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x1x8x4x8x4xi32, strided<[2048, 2048, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>)
            %subview_22 = memref.subview %alloc_3[0, %arg3, %arg4, 0, 0, 0, 0] [1, 1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1, 1] : memref<1x2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.fill ins(%c0_i32 : i32) outs(%subview_22 : memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d3, d6, d4, d7, d9)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d3, d2, d5, d6, d9, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d5, d4, d7, d8)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_22 : memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: i32, %in_23: i32, %out: i32):
              %3 = arith.muli %in, %in_23 : i32
              %4 = arith.addi %out, %3 : i32
              linalg.yield %4 : i32
            }
            scf.reduce 
          }
          scf.for %arg3 = %c1 to %c7 step %c1 {
            %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg3]
            %subview_16 = memref.subview %0[%arg0, %arg1, %3] [1, 64, 32] [1, 1, 1] : memref<2x128x256xi32> to memref<1x64x32xi32, strided<[32768, 256, 1], offset: ?>>
            %expand_shape_17 = memref.expand_shape %subview_16 [[0], [1, 2], [3, 4]] output_shape [1, 2, 32, 1, 32] : memref<1x64x32xi32, strided<[32768, 256, 1], offset: ?>> into memref<1x2x32x1x32xi32, strided<[32768, 8192, 256, 32, 1], offset: ?>>
            %transpose_18 = memref.transpose %expand_shape_17 (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x2x32x1x32xi32, strided<[32768, 8192, 256, 32, 1], offset: ?>> to memref<1x2x1x32x32xi32, strided<[32768, 8192, 32, 256, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_2[] [] [], %transpose_18[] [] []) : (memref<1x2x1x32x32xi32, 1 : i32>, memref<1x2x1x32x32xi32, strided<[32768, 8192, 32, 256, 1], offset: ?>>)
            %subview_19 = memref.subview %1[%arg0, %3, %arg2] [1, 32, 64] [1, 1, 1] : memref<2x256x128xi32> to memref<1x32x64xi32, strided<[32768, 128, 1], offset: ?>>
            %expand_shape_20 = memref.expand_shape %subview_19 [[0], [1, 2], [3, 4]] output_shape [1, 1, 32, 2, 32] : memref<1x32x64xi32, strided<[32768, 128, 1], offset: ?>> into memref<1x1x32x2x32xi32, strided<[32768, 4096, 128, 32, 1], offset: ?>>
            %transpose_21 = memref.transpose %expand_shape_20 (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x1x32x2x32xi32, strided<[32768, 4096, 128, 32, 1], offset: ?>> to memref<1x1x2x32x32xi32, strided<[32768, 4096, 32, 128, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_1[] [] [], %transpose_21[] [] []) : (memref<1x1x2x32x32xi32, 1 : i32>, memref<1x1x2x32x32xi32, strided<[32768, 4096, 32, 128, 1], offset: ?>>)
            scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
              %subview_22 = memref.subview %alloc_2[0, %arg4, 0, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : memref<1x2x1x32x32xi32, 1 : i32> to memref<1x1x1x32x32xi32, strided<[2048, 1024, 1024, 32, 1], offset: ?>, 1 : i32>
              %expand_shape_23 = memref.expand_shape %subview_22 [[0], [1], [2], [3, 4], [5, 6]] output_shape [1, 1, 1, 8, 4, 4, 8] : memref<1x1x1x32x32xi32, strided<[2048, 1024, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x1x8x4x4x8xi32, strided<[2048, 1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32>
              %transpose_24 = memref.transpose %expand_shape_23 (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d3, d4, d6) : memref<1x1x1x8x4x4x8xi32, strided<[2048, 1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x1x4x8x4x8xi32, strided<[2048, 1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_24[] [] []) : (memref<1x1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x1x4x8x4x8xi32, strided<[2048, 1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>)
              %subview_25 = memref.subview %alloc_1[0, 0, %arg5, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : memref<1x1x2x32x32xi32, 1 : i32> to memref<1x1x1x32x32xi32, strided<[2048, 2048, 1024, 32, 1], offset: ?>, 1 : i32>
              %expand_shape_26 = memref.expand_shape %subview_25 [[0], [1], [2], [3, 4], [5, 6]] output_shape [1, 1, 1, 4, 8, 8, 4] : memref<1x1x1x32x32xi32, strided<[2048, 2048, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x1x4x8x8x4xi32, strided<[2048, 2048, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32>
              %transpose_27 = memref.transpose %expand_shape_26 (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d3, d4, d6) : memref<1x1x1x4x8x8x4xi32, strided<[2048, 2048, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32> to memref<1x1x1x8x4x8x4xi32, strided<[2048, 2048, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc[] [] [], %transpose_27[] [] []) : (memref<1x1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x1x8x4x8x4xi32, strided<[2048, 2048, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>)
              %subview_28 = memref.subview %alloc_3[0, %arg4, %arg5, 0, 0, 0, 0] [1, 1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1, 1] : memref<1x2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d3, d6, d4, d7, d9)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d3, d2, d5, d6, d9, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d5, d4, d7, d8)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_28 : memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) {
              ^bb0(%in: i32, %in_29: i32, %out: i32):
                %4 = arith.muli %in, %in_29 : i32
                %5 = arith.addi %out, %4 : i32
                linalg.yield %5 : i32
              }
              scf.reduce 
            }
          }
          %subview_9 = memref.subview %0[%arg0, %arg1, 224] [1, 64, 32] [1, 1, 1] : memref<2x128x256xi32> to memref<1x64x32xi32, strided<[32768, 256, 1], offset: ?>>
          %expand_shape_10 = memref.expand_shape %subview_9 [[0], [1, 2], [3, 4]] output_shape [1, 2, 32, 1, 32] : memref<1x64x32xi32, strided<[32768, 256, 1], offset: ?>> into memref<1x2x32x1x32xi32, strided<[32768, 8192, 256, 32, 1], offset: ?>>
          %transpose_11 = memref.transpose %expand_shape_10 (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x2x32x1x32xi32, strided<[32768, 8192, 256, 32, 1], offset: ?>> to memref<1x2x1x32x32xi32, strided<[32768, 8192, 32, 256, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_2[] [] [], %transpose_11[] [] []) : (memref<1x2x1x32x32xi32, 1 : i32>, memref<1x2x1x32x32xi32, strided<[32768, 8192, 32, 256, 1], offset: ?>>)
          %subview_12 = memref.subview %1[%arg0, 224, %arg2] [1, 32, 64] [1, 1, 1] : memref<2x256x128xi32> to memref<1x32x64xi32, strided<[32768, 128, 1], offset: ?>>
          %expand_shape_13 = memref.expand_shape %subview_12 [[0], [1, 2], [3, 4]] output_shape [1, 1, 32, 2, 32] : memref<1x32x64xi32, strided<[32768, 128, 1], offset: ?>> into memref<1x1x32x2x32xi32, strided<[32768, 4096, 128, 32, 1], offset: ?>>
          %transpose_14 = memref.transpose %expand_shape_13 (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x1x32x2x32xi32, strided<[32768, 4096, 128, 32, 1], offset: ?>> to memref<1x1x2x32x32xi32, strided<[32768, 4096, 32, 128, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_1[] [] [], %transpose_14[] [] []) : (memref<1x1x2x32x32xi32, 1 : i32>, memref<1x1x2x32x32xi32, strided<[32768, 4096, 32, 128, 1], offset: ?>>)
          scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
            %subview_16 = memref.subview %alloc_2[0, %arg3, 0, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : memref<1x2x1x32x32xi32, 1 : i32> to memref<1x1x1x32x32xi32, strided<[2048, 1024, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_17 = memref.expand_shape %subview_16 [[0], [1], [2], [3, 4], [5, 6]] output_shape [1, 1, 1, 8, 4, 4, 8] : memref<1x1x1x32x32xi32, strided<[2048, 1024, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x1x8x4x4x8xi32, strided<[2048, 1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32>
            %transpose_18 = memref.transpose %expand_shape_17 (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d3, d4, d6) : memref<1x1x1x8x4x4x8xi32, strided<[2048, 1024, 1024, 128, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x1x1x4x8x4x8xi32, strided<[2048, 1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_18[] [] []) : (memref<1x1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x1x4x8x4x8xi32, strided<[2048, 1024, 1024, 8, 128, 32, 1], offset: ?>, 1 : i32>)
            %subview_19 = memref.subview %alloc_1[0, 0, %arg4, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : memref<1x1x2x32x32xi32, 1 : i32> to memref<1x1x1x32x32xi32, strided<[2048, 2048, 1024, 32, 1], offset: ?>, 1 : i32>
            %expand_shape_20 = memref.expand_shape %subview_19 [[0], [1], [2], [3, 4], [5, 6]] output_shape [1, 1, 1, 4, 8, 8, 4] : memref<1x1x1x32x32xi32, strided<[2048, 2048, 1024, 32, 1], offset: ?>, 1 : i32> into memref<1x1x1x4x8x8x4xi32, strided<[2048, 2048, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32>
            %transpose_21 = memref.transpose %expand_shape_20 (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d3, d4, d6) : memref<1x1x1x4x8x8x4xi32, strided<[2048, 2048, 1024, 256, 32, 4, 1], offset: ?>, 1 : i32> to memref<1x1x1x8x4x8x4xi32, strided<[2048, 2048, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc[] [] [], %transpose_21[] [] []) : (memref<1x1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x1x8x4x8x4xi32, strided<[2048, 2048, 1024, 4, 256, 32, 1], offset: ?>, 1 : i32>)
            %subview_22 = memref.subview %alloc_3[0, %arg3, %arg4, 0, 0, 0, 0] [1, 1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1, 1] : memref<1x2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d3, d6, d4, d7, d9)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d3, d2, d5, d6, d9, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d5, d4, d7, d8)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_22 : memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: i32, %in_25: i32, %out: i32):
              %3 = arith.muli %in, %in_25 : i32
              %4 = arith.addi %out, %3 : i32
              linalg.yield %4 : i32
            }
            %subview_23 = memref.subview %alloc_4[0, %arg3, %arg4, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : memref<1x2x2x32x32xi32, 1 : i32> to memref<1x1x1x32x32xi32, strided<[4096, 2048, 1024, 32, 1], offset: ?>, 1 : i32>
            %transpose_24 = memref.transpose %subview_22 (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5, d3, d6) : memref<1x1x1x8x8x4x4xi32, strided<[4096, 2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32> to memref<1x1x1x8x4x8x4xi32, strided<[4096, 2048, 1024, 16, 4, 128, 1], offset: ?>, 2 : i32>
            air.dma_memcpy_nd (%subview_23[] [] [], %transpose_24[] [] []) : (memref<1x1x1x32x32xi32, strided<[4096, 2048, 1024, 32, 1], offset: ?>, 1 : i32>, memref<1x1x1x8x4x8x4xi32, strided<[4096, 2048, 1024, 16, 4, 128, 1], offset: ?>, 2 : i32>)
            scf.reduce 
          }
          %transpose_15 = memref.transpose %alloc_4 (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x2x2x32x32xi32, 1 : i32> to memref<1x2x32x2x32xi32, strided<[4096, 2048, 32, 1024, 1]>, 1 : i32>
          air.dma_memcpy_nd (%subview[] [] [], %transpose_15[] [] []) : (memref<1x64x64xi32, strided<[16384, 128, 1], offset: ?>>, memref<1x2x32x2x32xi32, strided<[4096, 2048, 32, 1024, 1]>, 1 : i32>)
          scf.reduce 
        }
        memref.dealloc %alloc_4 : memref<1x2x2x32x32xi32, 1 : i32>
        memref.dealloc %alloc_3 : memref<1x2x2x8x8x4x4xi32, 2 : i32>
        memref.dealloc %alloc_2 : memref<1x2x1x32x32xi32, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x1x2x32x32xi32, 1 : i32>
        memref.dealloc %alloc_0 : memref<1x1x1x4x8x4x8xi32, 2 : i32>
        memref.dealloc %alloc : memref<1x1x1x8x4x8x4xi32, 2 : i32>
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
                "air-to-aie{row-offset=2 col-offset=1 device=npu2 emit-while-loop=true}",
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
                "func.func(air-opt-shim-dma-bds{device=npu2})",
                "air-to-std",
                "canonicalize",
                "symbol-dce",
                "func.func(affine-loop-opt{affine-opt-tile-sizes=2,2,2})",
                "func.func(air-unroll-outer-affine-loops{depth=4})",
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
