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
    module {
      func.func @matmul_512x512_512xbf16__dispatch_0_matmul_512x512x512_bf16(%0 : memref<512x512xbf16>, %1 : memref<512x512xbf16>, %2 : memref<512x512xbf16>) {
        %c4 = arith.constant 4 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c7 = arith.constant 7 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : bf16
        %c1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x4x64x64xbf16, 1 : i32>
        %alloc_2 = memref.alloc() : memref<4x1x64x64xbf16, 1 : i32>
        %alloc_3 = memref.alloc() : memref<4x4x16x16x4x4xbf16, 2 : i32>
        %alloc_4 = memref.alloc() : memref<4x4x64x64xbf16, 1 : i32>
        // scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c256, %c256) {
        scf.parallel (%dummyarg) = (%c0) to (%c1) step (%c1) {
          scf.for %arg0 = %c0 to %c512 step %c256 {
            scf.for %arg1 = %c0 to %c512 step %c256 {
              %subview = memref.subview %2[%arg0, %arg1] [256, 256] [1, 1] : memref<512x512xbf16> to memref<256x256xbf16, strided<[512, 1], offset: ?>>
              scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
                %subview_22 = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
                linalg.fill ins(%cst : bf16) outs(%subview_22 : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
                scf.reduce 
              }
              scf.for %arg2 = %c0 to %c8 step %c1 {
                %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg2]
                %subview_16 = memref.subview %0[%arg0, %3] [256, 64] [1, 1] : memref<512x512xbf16> to memref<256x64xbf16, strided<[512, 1], offset: ?>>
                %expand_shape_17 = memref.expand_shape %subview_16 [[0, 1], [2, 3]] output_shape [4, 64, 1, 64] : memref<256x64xbf16, strided<[512, 1], offset: ?>> into memref<4x64x1x64xbf16, strided<[32768, 512, 64, 1], offset: ?>>
                %transpose_18 = memref.transpose %expand_shape_17 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<4x64x1x64xbf16, strided<[32768, 512, 64, 1], offset: ?>> to memref<4x1x64x64xbf16, strided<[32768, 64, 512, 1], offset: ?>>
                air.dma_memcpy_nd (%alloc_2[] [] [], %transpose_18[] [] []) : (memref<4x1x64x64xbf16, 1 : i32>, memref<4x1x64x64xbf16, strided<[32768, 64, 512, 1], offset: ?>>)
                %subview_19 = memref.subview %1[%3, %arg1] [64, 256] [1, 1] : memref<512x512xbf16> to memref<64x256xbf16, strided<[512, 1], offset: ?>>
                %expand_shape_20 = memref.expand_shape %subview_19 [[0, 1], [2, 3]] output_shape [1, 64, 4, 64] : memref<64x256xbf16, strided<[512, 1], offset: ?>> into memref<1x64x4x64xbf16, strided<[32768, 512, 64, 1], offset: ?>>
                %transpose_21 = memref.transpose %expand_shape_20 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<1x64x4x64xbf16, strided<[32768, 512, 64, 1], offset: ?>> to memref<1x4x64x64xbf16, strided<[32768, 64, 512, 1], offset: ?>>
                air.dma_memcpy_nd (%alloc_1[] [] [], %transpose_21[] [] []) : (memref<1x4x64x64xbf16, 1 : i32>, memref<1x4x64x64xbf16, strided<[32768, 64, 512, 1], offset: ?>>)
                scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
                  %subview_22 = memref.subview %alloc_2[%arg3, 0, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x1x64x64xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32>
                  %expand_shape_23 = memref.expand_shape %subview_22 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 16, 4, 8, 8] : memref<1x1x64x64xbf16, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32> into memref<1x1x16x4x8x8xbf16, strided<[4096, 4096, 256, 64, 8, 1], offset: ?>, 1 : i32>
                  %transpose_24 = memref.transpose %expand_shape_23 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x16x4x8x8xbf16, strided<[4096, 4096, 256, 64, 8, 1], offset: ?>, 1 : i32> to memref<1x1x8x16x4x8xbf16, strided<[4096, 4096, 8, 256, 64, 1], offset: ?>, 1 : i32>
                  air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_24[] [] []) : (memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x8x16x4x8xbf16, strided<[4096, 4096, 8, 256, 64, 1], offset: ?>, 1 : i32>)
                  %subview_25 = memref.subview %alloc_1[0, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x4x64x64xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32>
                  %expand_shape_26 = memref.expand_shape %subview_25 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 8, 16, 4] : memref<1x1x64x64xbf16, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32> into memref<1x1x8x8x16x4xbf16, strided<[16384, 4096, 512, 64, 4, 1], offset: ?>, 1 : i32>
                  %transpose_27 = memref.transpose %expand_shape_26 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x8x16x4xbf16, strided<[16384, 4096, 512, 64, 4, 1], offset: ?>, 1 : i32> to memref<1x1x16x8x8x4xbf16, strided<[16384, 4096, 4, 512, 64, 1], offset: ?>, 1 : i32>
                  air.dma_memcpy_nd (%alloc[] [] [], %transpose_27[] [] []) : (memref<1x1x16x8x8x4xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, strided<[16384, 4096, 4, 512, 64, 1], offset: ?>, 1 : i32>)
                  %subview_28 = memref.subview %alloc_3[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
                  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], library_call = "matmul_bf16_bf16"} ins(%alloc_0, %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, 2 : i32>) outs(%subview_28 : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>) {
                  ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
                    %4 = arith.mulf %in, %in_29 : bf16
                    %5 = arith.addf %out, %4 : bf16
                    linalg.yield %5 : bf16
                  }
                  scf.reduce 
                }
              }
              scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
                %subview_22 = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
                %subview_23 = memref.subview %alloc_4[%arg2, %arg3, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x4x64x64xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32>
                %transpose_24 = memref.transpose %subview_22 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32> to memref<1x1x16x4x16x4xbf16, strided<[16384, 4096, 16, 4, 256, 1], offset: ?>, 2 : i32>
                air.dma_memcpy_nd (%subview_23[] [] [], %transpose_24[] [] []) : (memref<1x1x64x64xbf16, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32>, memref<1x1x16x4x16x4xbf16, strided<[16384, 4096, 16, 4, 256, 1], offset: ?>, 2 : i32>)
                scf.reduce 
              }
              %transpose_15 = memref.transpose %alloc_4 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<4x4x64x64xbf16, 1 : i32> to memref<4x64x4x64xbf16, strided<[16384, 64, 4096, 1]>, 1 : i32>
              air.dma_memcpy_nd (%subview[] [] [], %transpose_15[] [] []) : (memref<256x256xbf16, strided<[512, 1], offset: ?>>, memref<4x64x4x64xbf16, strided<[16384, 64, 4096, 1]>, 1 : i32>)
            }
          }
          scf.reduce 
        }
        memref.dealloc %alloc_4 : memref<4x4x64x64xbf16, 1 : i32>
        memref.dealloc %alloc_3 : memref<4x4x16x16x4x4xbf16, 2 : i32>
        memref.dealloc %alloc_2 : memref<4x1x64x64xbf16, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x4x64x64xbf16, 1 : i32>
        memref.dealloc %alloc_0 : memref<1x1x8x16x4x8xbf16, 2 : i32>
        memref.dealloc %alloc : memref<1x1x16x8x8x4xbf16, 2 : i32>
        return
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)

    ###############################################
    # Binding scf.paralell to air hierarchies
    ###############################################

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
        omit_while_true_loop=False,
        omit_pingpong=True,
        lower_linalg_to_func=True,
        air_loop_fusion=False,
    )
    module_function = backend.compile_and_load(air_module)
