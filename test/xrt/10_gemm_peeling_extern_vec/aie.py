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
    #map4 = affine_map<()[s0] -> (s0 * 64)>
    builtin.module {
      func.func @matmul_bf16(%0: memref<512x1024xbf16>, %1: memref<1024x512xbf16>, %2: memref<512x512xbf16>) {
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : bf16
        %c16 = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x1x64x128xbf16, 1 : i32>
        %alloc_2 = memref.alloc() : memref<1x1x128x64xbf16, 1 : i32>
        %alloc_3 = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
        %alloc_4 = memref.alloc() : memref<1x1x128x128xbf16, 1 : i32>
        scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
          %subview = memref.subview %0[%arg0, 0] [128, 1024] [1, 1] : memref<512x1024xbf16> to memref<128x1024xbf16, strided<[1024, 1], offset: ?>>
          %subview_5 = memref.subview %1[0, %arg1] [1024, 128] [1, 1] : memref<1024x512xbf16> to memref<1024x128xbf16, strided<[512, 1], offset: ?>>
          %subview_6 = memref.subview %2[%arg0, %arg1] [128, 128] [1, 1] : memref<512x512xbf16> to memref<128x128xbf16, strided<[512, 1], offset: ?>>
          %subview_7 = memref.subview %subview[0, 0] [128, 64] [1, 1] : memref<128x1024xbf16, strided<[1024, 1], offset: ?>> to memref<128x64xbf16, strided<[1024, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_2[] [] [], %subview_7[] [] []) : (memref<1x1x128x64xbf16, 1 : i32>, memref<128x64xbf16, strided<[1024, 1], offset: ?>>)
          %subview_8 = memref.subview %subview_5[0, 0] [64, 128] [1, 1] : memref<1024x128xbf16, strided<[512, 1], offset: ?>> to memref<64x128xbf16, strided<[512, 1], offset: ?>>
          air.dma_memcpy_nd (%alloc_1[] [] [], %subview_8[] [] []) : (memref<1x1x64x128xbf16, 1 : i32>, memref<64x128xbf16, strided<[512, 1], offset: ?>>)
          scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
            %3 = affine.apply #map()[%arg2]
            %subview_11 = memref.subview %alloc_2[0, 0, %3, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x128x64xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[8192, 8192, 64, 1], offset: ?>, 1 : i32>
            %expand_shape = memref.expand_shape %subview_11 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 16, 4, 8, 8] : memref<1x1x64x64xbf16, strided<[8192, 8192, 64, 1], offset: ?>, 1 : i32> into memref<1x1x16x4x8x8xbf16, strided<[8192, 8192, 256, 64, 8, 1], offset: ?>, 1 : i32>
            %transpose_12 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x16x4x8x8xbf16, strided<[8192, 8192, 256, 64, 8, 1], offset: ?>, 1 : i32> to memref<1x1x8x16x4x8xbf16, strided<[8192, 8192, 8, 256, 64, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_12[] [] []) : (memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x8x16x4x8xbf16, strided<[8192, 8192, 8, 256, 64, 1], offset: ?>, 1 : i32>)
            %4 = affine.apply #map()[%arg3]
            %subview_13 = memref.subview %alloc_1[0, 0, 0, %4] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x128xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[8192, 8192, 128, 1], offset: ?>, 1 : i32>
            %expand_shape_14 = memref.expand_shape %subview_13 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 8, 16, 4] : memref<1x1x64x64xbf16, strided<[8192, 8192, 128, 1], offset: ?>, 1 : i32> into memref<1x1x8x8x16x4xbf16, strided<[8192, 8192, 1024, 128, 4, 1], offset: ?>, 1 : i32>
            %transpose_15 = memref.transpose %expand_shape_14 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x8x16x4xbf16, strided<[8192, 8192, 1024, 128, 4, 1], offset: ?>, 1 : i32> to memref<1x1x16x8x8x4xbf16, strided<[8192, 8192, 4, 1024, 128, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc[] [] [], %transpose_15[] [] []) : (memref<1x1x16x8x8x4xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, strided<[8192, 8192, 4, 1024, 128, 1], offset: ?>, 1 : i32>)
            %subview_16 = memref.subview %alloc_3[0, 0, %arg3, %arg2, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.fill ins(%cst : bf16) outs(%subview_16 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
            linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], library_call = "matmul_bf16_bf16"} ins(%alloc_0, %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, 2 : i32>) outs(%subview_16 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>) {
            ^bb0(%in: bf16, %in_17: bf16, %out: bf16):
              %5 = arith.mulf %in, %in_17 : bf16
              %6 = arith.addf %out, %5 : bf16
              linalg.yield %6 : bf16
            }
            scf.reduce 
          }
          scf.for %arg2 = %c1 to %c16 step %c1 {
            %3 = affine.apply #map4()[%arg2]
            %subview_11 = memref.subview %subview[0, %3] [128, 64] [1, 1] : memref<128x1024xbf16, strided<[1024, 1], offset: ?>> to memref<128x64xbf16, strided<[1024, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_2[] [] [], %subview_11[] [] []) : (memref<1x1x128x64xbf16, 1 : i32>, memref<128x64xbf16, strided<[1024, 1], offset: ?>>)
            %subview_12 = memref.subview %subview_5[%3, 0] [64, 128] [1, 1] : memref<1024x128xbf16, strided<[512, 1], offset: ?>> to memref<64x128xbf16, strided<[512, 1], offset: ?>>
            air.dma_memcpy_nd (%alloc_1[] [] [], %subview_12[] [] []) : (memref<1x1x64x128xbf16, 1 : i32>, memref<64x128xbf16, strided<[512, 1], offset: ?>>)
            scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c32, %c32) step (%c16, %c16) {
              %4 = affine.apply #map()[%arg3]
              %subview_13 = memref.subview %alloc_2[0, 0, %4, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x128x64xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[8192, 8192, 64, 1], offset: ?>, 1 : i32>
              %expand_shape = memref.expand_shape %subview_13 [[0], [1], [2, 3], [4, 5]]  output_shape [1, 1, 16, 4, 8, 8] : memref<1x1x64x64xbf16, strided<[8192, 8192, 64, 1], offset: ?>, 1 : i32> into memref<1x1x16x4x8x8xbf16, strided<[8192, 8192, 256, 64, 8, 1], offset: ?>, 1 : i32>
              %transpose_14 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x16x4x8x8xbf16, strided<[8192, 8192, 256, 64, 8, 1], offset: ?>, 1 : i32> to memref<1x1x8x16x4x8xbf16, strided<[8192, 8192, 8, 256, 64, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_14[] [] []) : (memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x8x16x4x8xbf16, strided<[8192, 8192, 8, 256, 64, 1], offset: ?>, 1 : i32>)
              %5 = affine.apply #map()[%arg4]
              %subview_15 = memref.subview %alloc_1[0, 0, 0, %5] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x128xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[8192, 8192, 128, 1], offset: ?>, 1 : i32>
              %expand_shape_16 = memref.expand_shape %subview_15 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 8, 16, 4] : memref<1x1x64x64xbf16, strided<[8192, 8192, 128, 1], offset: ?>, 1 : i32> into memref<1x1x8x8x16x4xbf16, strided<[8192, 8192, 1024, 128, 4, 1], offset: ?>, 1 : i32>
              %transpose_17 = memref.transpose %expand_shape_16 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x8x16x4xbf16, strided<[8192, 8192, 1024, 128, 4, 1], offset: ?>, 1 : i32> to memref<1x1x16x8x8x4xbf16, strided<[8192, 8192, 4, 1024, 128, 1], offset: ?>, 1 : i32>
              air.dma_memcpy_nd (%alloc[] [] [], %transpose_17[] [] []) : (memref<1x1x16x8x8x4xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, strided<[8192, 8192, 4, 1024, 128, 1], offset: ?>, 1 : i32>)
              %subview_18 = memref.subview %alloc_3[0, 0, %arg4, %arg3, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
              linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], library_call = "matmul_bf16_bf16"} ins(%alloc_0, %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, 2 : i32>) outs(%subview_18 : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>) {
              ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
                %6 = arith.mulf %in, %in_19 : bf16
                %7 = arith.addf %out, %6 : bf16
                linalg.yield %7 : bf16
              }
              scf.reduce 
            }
          }
          %transpose = memref.transpose %alloc_3 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x32x4x32x4xbf16, strided<[16384, 16384, 16, 4, 512, 1]>, 2 : i32>
          air.dma_memcpy_nd (%alloc_4[] [] [], %transpose[] [] []) : (memref<1x1x128x128xbf16, 1 : i32>, memref<1x1x32x4x32x4xbf16, strided<[16384, 16384, 16, 4, 512, 1]>, 2 : i32>)
          %subview_9 = memref.subview %alloc_4[0, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<1x1x128x128xbf16, 1 : i32> to memref<128x128xbf16, 1 : i32>
          %transpose_10 = memref.transpose %subview_9 (d0, d1) -> (d0, d1) : memref<128x128xbf16, 1 : i32> to memref<128x128xbf16, strided<[128, 1]>, 1 : i32>
          air.dma_memcpy_nd (%subview_6[] [] [], %transpose_10[] [] []) : (memref<128x128xbf16, strided<[512, 1], offset: ?>>, memref<128x128xbf16, strided<[128, 1]>, 1 : i32>)
          scf.reduce 
        }
        memref.dealloc %alloc_4 : memref<1x1x128x128xbf16, 1 : i32>
        memref.dealloc %alloc_3 : memref<1x1x32x32x4x4xbf16, 2 : i32>
        memref.dealloc %alloc_2 : memref<1x1x128x64xbf16, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x1x64x128xbf16, 1 : i32>
        memref.dealloc %alloc_0 : memref<1x1x8x16x4x8xbf16, 2 : i32>
        memref.dealloc %alloc : memref<1x1x16x8x8x4xbf16, 2 : i32>
        return
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)
    
    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################

    pipeline = "builtin.module("+",".join([
        "buffer-results-to-out-params",
        "air-linalg-to-func{link-with=mm.o}",
        "air-par-to-herd{depth=1}",
        "air-par-to-launch{has-air-segment=true}",
        "air-copy-to-dma",
        "canonicalize", "cse",
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    
    ###############################################
    # Extract event dependency and optimize schedule
    ###############################################

    pipeline = "builtin.module("+",".join([
        "air-dependency",
        "air-dependency-schedule-opt",
        "air-specialize-dma-broadcast",
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-dependency-canonicalize",
        "canonicalize", "cse",
        "air-isolate-async-dma-loop-nests",
        "canonicalize", "cse",
        "air-fuse-channels",
        "canonicalize", "cse",
        "func.func(air-loop-fusion)",
        "air-label-scf-for-to-ping-pong",
        "air-ping-pong-transform{keep-memref-dealloc=true}",
        "canonicalize", "cse",
        "air-specialize-channel-wrap-and-stride",
        "canonicalize", "cse",
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    
    ################################################
    ## Place herd to segment
    ################################################

    air_async_module = Module.parse(str(air_module))
    pipeline = "builtin.module("+",".join([
        "func.func(air-collapse-herd{max-col-size=4})",
        'canonicalize', 'cse',
        "air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}",
        'canonicalize', 'cse',
        'func.func(air-renumber-dma)'
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    
    ################################################
    ## MLIR-AIR to MLIR-AIE
    ################################################
    
    pipeline = "builtin.module("+",".join([
        'canonicalize', 'cse',
        'air-to-aie{row-offset=2 col-offset=0 device=npu emit-while-loop=true}',
        'canonicalize',
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    
    ################################################
    ## MLIR-AIR runtime lowering
    ################################################

    pipeline = "builtin.module("+",".join([
      'air-to-std',
      'canonicalize',
      'symbol-dce',
      'func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})',
      'func.func(air-unroll-outer-affine-loops{depth=2})',
      'affine-expand-index-ops',
      'airrt-to-npu',
      'canonicalize',
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open('aie.mlir', 'w') as f:
        f.write(str(air_module))
