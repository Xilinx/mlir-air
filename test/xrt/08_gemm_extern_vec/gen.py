# gen.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.backend.xrt import XRTBackend
from air.ir import *
import air.passmanager

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<()[s0] -> (s0 * 128)>
    #map1 = affine_map<()[s0] -> (s0 * 64)>
    #map2 = affine_map<()[s0] -> (s0 * 8)>
    #map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
    #map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
    #map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
    module {
      func.func @matmul(%0 : memref<256x1024xbf16>, %1 : memref<1024x256xbf16>, %2 : memref<256x256xbf16>) {
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c8 = arith.constant 8 : index
          %c128 = arith.constant 128 : index
          %c256 = arith.constant 256 : index
          %c1024 = arith.constant 1024 : index
          %cst = arith.constant 0.000000e+00 : bf16
          %c0 = arith.constant 0 : index
          scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
            %3 = affine.apply #map()[%arg0]
            %4 = affine.apply #map()[%arg1]
            %subview = memref.subview %2[%3, %4] [128, 128] [1, 1] : memref<256x256xbf16> to memref<128x128xbf16, strided<[256, 1], offset: ?>>
            %alloc = memref.alloc() : memref<128x1024xbf16, 1>
            scf.for %arg2 = %c0 to %c1024 step %c256 {
              %5 = affine.apply #map()[%arg0]
              %subview_2 = memref.subview %0[%5, %arg2] [128, 256] [1, 1] : memref<256x1024xbf16> to memref<128x256xbf16, strided<[1024, 1], offset: ?>>
              %subview_3 = memref.subview %alloc[0, %arg2] [128, 256] [1, 1] : memref<128x1024xbf16, 1> to memref<128x256xbf16, strided<[1024, 1], offset: ?>, 1>
              memref.copy %subview_2, %subview_3 : memref<128x256xbf16, strided<[1024, 1], offset: ?>> to memref<128x256xbf16, strided<[1024, 1], offset: ?>, 1>
            }
            %alloc_0 = memref.alloc() : memref<1024x128xbf16, 1>
            scf.for %arg2 = %c0 to %c1024 step %c256 {
              %5 = affine.apply #map()[%arg1]
              %subview_2 = memref.subview %1[%arg2, %5] [256, 128] [1, 1] : memref<1024x256xbf16> to memref<256x128xbf16, strided<[256, 1], offset: ?>>
              %subview_3 = memref.subview %alloc_0[%arg2, 0] [256, 128] [1, 1] : memref<1024x128xbf16, 1> to memref<256x128xbf16, strided<[128, 1], offset: ?>, 1>
              memref.copy %subview_2, %subview_3 : memref<256x128xbf16, strided<[256, 1], offset: ?>> to memref<256x128xbf16, strided<[128, 1], offset: ?>, 1>
            }
            %alloc_1 = memref.alloc() : memref<128x128xbf16, 1>
            scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
              %5 = affine.apply #map1()[%arg2]
              %6 = affine.apply #map1()[%arg3]
              %subview_2 = memref.subview %alloc_1[%5, %6] [64, 64] [1, 1] : memref<128x128xbf16, 1> to memref<64x64xbf16, strided<[128, 1], offset: ?>, 1>
              %alloc_3 = memref.alloc() : memref<16x16x4x4xbf16, 2>
              linalg.fill ins(%cst : bf16) outs(%alloc_3 : memref<16x16x4x4xbf16, 2>)
              scf.for %arg4 = %c0 to %c128 step %c8 {
                %7 = affine.apply #map1()[%arg2]
                %8 = affine.apply #map2()[%arg4]
                %subview_4 = memref.subview %alloc[%7, %8] [64, 64] [1, 1] : memref<128x1024xbf16, 1> to memref<64x64xbf16, strided<[1024, 1], offset: ?>, 1>
                %alloc_5 = memref.alloc() : memref<8x16x4x8xbf16, 2>
                %expand_shape = memref.expand_shape %subview_4 [[0, 1], [2, 3]] output_shape [16, 4, 8, 8] : memref<64x64xbf16, strided<[1024, 1], offset: ?>, 1> into memref<16x4x8x8xbf16, strided<[4096, 1024, 8, 1], offset: ?>, 1>
                %transpose_6 = memref.transpose %expand_shape (d0, d1, d2, d3) -> (d2, d0, d1, d3) : memref<16x4x8x8xbf16, strided<[4096, 1024, 8, 1], offset: ?>, 1> to memref<8x16x4x8xbf16, strided<[8, 4096, 1024, 1], offset: ?>, 1>
                air.dma_memcpy_nd (%alloc_5[] [] [], %transpose_6[] [] []) : (memref<8x16x4x8xbf16, 2>, memref<8x16x4x8xbf16, strided<[8, 4096, 1024, 1], offset: ?>, 1>)
                %9 = affine.apply #map2()[%arg4]
                %10 = affine.apply #map1()[%arg3]
                %subview_7 = memref.subview %alloc_0[%9, %10] [64, 64] [1, 1] : memref<1024x128xbf16, 1> to memref<64x64xbf16, strided<[128, 1], offset: ?>, 1>
                %alloc_8 = memref.alloc() : memref<16x8x8x4xbf16, 2>
                %expand_shape_9 = memref.expand_shape %subview_7 [[0, 1], [2, 3]] output_shape [8, 8, 16, 4]  : memref<64x64xbf16, strided<[128, 1], offset: ?>, 1> into memref<8x8x16x4xbf16, strided<[1024, 128, 4, 1], offset: ?>, 1>
                %transpose_10 = memref.transpose %expand_shape_9 (d0, d1, d2, d3) -> (d2, d0, d1, d3) : memref<8x8x16x4xbf16, strided<[1024, 128, 4, 1], offset: ?>, 1> to memref<16x8x8x4xbf16, strided<[4, 1024, 128, 1], offset: ?>, 1>
                air.dma_memcpy_nd (%alloc_8[] [] [], %transpose_10[] [] []) : (memref<16x8x8x4xbf16, 2>, memref<16x8x8x4xbf16, strided<[4, 1024, 128, 1], offset: ?>, 1>)
                linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], library_call = "matmul_bf16_bf16"} ins(%alloc_5, %alloc_8 : memref<8x16x4x8xbf16, 2>, memref<16x8x8x4xbf16, 2>) outs(%alloc_3 : memref<16x16x4x4xbf16, 2>) {
                ^bb0(%in: bf16, %in_11: bf16, %out: bf16):
                  %11 = arith.mulf %in, %in_11 : bf16
                  %12 = arith.addf %out, %11 : bf16
                  linalg.yield %12 : bf16
                }
                memref.dealloc %alloc_5 : memref<8x16x4x8xbf16, 2>
                memref.dealloc %alloc_8 : memref<16x8x8x4xbf16, 2>
              }
              %transpose = memref.transpose %alloc_3 (d0, d1, d2, d3) -> (d1, d2, d0, d3) : memref<16x16x4x4xbf16, 2> to memref<16x4x16x4xbf16, strided<[16, 4, 256, 1]>, 2>
              air.dma_memcpy_nd (%subview_2[] [] [], %transpose[] [] []) : (memref<64x64xbf16, strided<[128, 1], offset: ?>, 1>, memref<16x4x16x4xbf16, strided<[16, 4, 256, 1]>, 2>)
              memref.dealloc %alloc_3 : memref<16x16x4x4xbf16, 2>
              scf.reduce 
            }
            memref.copy %alloc_1, %subview : memref<128x128xbf16, 1> to memref<128x128xbf16, strided<[256, 1], offset: ?>>
            memref.dealloc %alloc : memref<128x1024xbf16, 1>
            memref.dealloc %alloc_0 : memref<1024x128xbf16, 1>
            memref.dealloc %alloc_1 : memref<128x128xbf16, 1>
            scf.reduce 
          }
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
                "air-linalg-to-func{link-with=mm.o}",
                "air-par-to-herd{depth=-1}",
                "air-par-to-launch{has-air-segment=true}",
                "scf-forall-to-for",
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
    # Run compile and load
    ###############################################

    backend = XRTBackend(
        air_loop_fusion=True,
        use_lock_race_condition_fix=True,
    )
    backend.compile(air_module)
