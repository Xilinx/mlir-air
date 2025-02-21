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
    #map = affine_map<()[s0, s1] -> (s0 + s1)>
    #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
    #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d8, d7)>
    #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d0, d4, d3, d6, d7)>
    module {
      func.func @matmul_dispatch_0_matmul_512x512x512_bf16xbf16xbf16(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
        %c8192 = arith.constant 8192 : index
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index
        %c131072 = arith.constant 131072 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %c16384 = arith.constant 16384 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst = arith.constant 0.000000e+00 : bf16
        %c15 = arith.constant 15 : index
        %alloc = memref.alloc() : memref<1x1x8x8x4x4xbf16, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x8x4x8x4xbf16, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xbf16, 2 : i32>
        %alloc_2 = memref.alloc() : memref<8x8x32x32xbf16, 1 : i32>
        %alloc_3 = memref.alloc() : memref<8x16x32x32xbf16, 1 : i32>
        %alloc_4 = memref.alloc() : memref<8x16x32x32xbf16, 1 : i32>
        memref.assume_alignment %arg0, 64 : memref<512x512xbf16>
        memref.assume_alignment %arg1, 64 : memref<512x512xbf16>
        memref.assume_alignment %arg2, 64 : memref<512x512xbf16>
        scf.forall (%arg3, %arg4) = (0, 0) to (512, 512) step (256, 256) {
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg0[%c0, %c0, %arg3, %c0] [%c8, %c16, %c32, %c32] [%c16384, %c32, %c512, %c1]) : (memref<8x16x32x32xbf16, 1 : i32>, memref<512x512xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg1[%c0, %c0, %c0, %arg4] [%c8, %c16, %c32, %c32] [%c32, %c16384, %c512, %c1]) : (memref<8x16x32x32xbf16, 1 : i32>, memref<512x512xbf16>)
          scf.forall (%arg5, %arg6) = (0, 0) to (8, 8) step (4, 4) {
            scf.forall (%arg7, %arg8) = (0, 0) to (4, 4) step (1, 1) {
              %0 = affine.apply #map()[%arg5, %arg7]
              air.dma_memcpy_nd (%alloc_1[] [] [], %alloc_4[%0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) : (memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
              %1 = affine.apply #map()[%arg6, %arg8]
              air.dma_memcpy_nd (%alloc_0[] [] [], %alloc_3[%1, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c16384, %c1024, %c4, %c256, %c32, %c1]) : (memref<1x1x8x4x8x4xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
              linalg.fill ins(%cst : bf16) outs(%alloc : memref<1x1x8x8x4x4xbf16, 2 : i32>)
              linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], library_call = "matmul_bf16_bf16"} ins(%alloc_1, %alloc_0 : memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<1x1x8x4x8x4xbf16, 2 : i32>) outs(%alloc : memref<1x1x8x8x4x4xbf16, 2 : i32>) {
              ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
                %4 = arith.mulf %in, %in_5 : bf16
                %5 = arith.addf %out, %4 : bf16
                linalg.yield %5 : bf16
              }
            }
            scf.for %arg7 = %c1 to %c15 step %c1 {
              scf.forall (%arg8, %arg9) = (0, 0) to (4, 4) step (1, 1) {
                %0 = affine.apply #map()[%arg5, %arg8]
                air.dma_memcpy_nd (%alloc_1[] [] [], %alloc_4[%0, %arg7, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) : (memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
                %1 = affine.apply #map()[%arg6, %arg9]
                air.dma_memcpy_nd (%alloc_0[] [] [], %alloc_3[%1, %arg7, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c16384, %c1024, %c4, %c256, %c32, %c1]) : (memref<1x1x8x4x8x4xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
                linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], library_call = "matmul_bf16_bf16"} ins(%alloc_1, %alloc_0 : memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<1x1x8x4x8x4xbf16, 2 : i32>) outs(%alloc : memref<1x1x8x8x4x4xbf16, 2 : i32>) {
                ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
                  %4 = arith.mulf %in, %in_5 : bf16
                  %5 = arith.addf %out, %4 : bf16
                  linalg.yield %5 : bf16
                }
              }
            }
            scf.forall (%arg7, %arg8) = (0, 0) to (4, 4) step (1, 1) {
              %0 = affine.apply #map()[%arg5, %arg7]
              air.dma_memcpy_nd (%alloc_1[] [] [], %alloc_4[%0, %c15, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) : (memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
              %1 = affine.apply #map()[%arg6, %arg8]
              air.dma_memcpy_nd (%alloc_0[] [] [], %alloc_3[%1, %c15, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c16384, %c1024, %c4, %c256, %c32, %c1]) : (memref<1x1x8x4x8x4xbf16, 2 : i32>, memref<8x16x32x32xbf16, 1 : i32>)
              linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], library_call = "matmul_bf16_bf16"} ins(%alloc_1, %alloc_0 : memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<1x1x8x4x8x4xbf16, 2 : i32>) outs(%alloc : memref<1x1x8x8x4x4xbf16, 2 : i32>) {
              ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
                %4 = arith.mulf %in, %in_5 : bf16
                %5 = arith.addf %out, %4 : bf16
                linalg.yield %5 : bf16
              }
              air.dma_memcpy_nd (%alloc_2[%1, %0, %c0, %c0] [%c1, %c1, %c32, %c32] [%c8192, %c1024, %c32, %c1], %alloc[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c16, %c4, %c128, %c1]) : (memref<8x8x32x32xbf16, 1 : i32>, memref<1x1x8x8x4x4xbf16, 2 : i32>)
            }
          }
          air.dma_memcpy_nd (%arg2[%arg3, %arg4] [%c256, %c256] [%c512, %c1], %alloc_2[%c0, %c0, %c0, %c0] [%c8, %c32, %c8, %c32] [%c1024, %c32, %c8192, %c1]) : (memref<512x512xbf16>, memref<8x8x32x32xbf16, 1 : i32>)
        }
        memref.dealloc %alloc_4 : memref<8x16x32x32xbf16, 1 : i32>
        memref.dealloc %alloc_3 : memref<8x16x32x32xbf16, 1 : i32>
        memref.dealloc %alloc_2 : memref<8x8x32x32xbf16, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x1x4x8x4x8xbf16, 2 : i32>
        memref.dealloc %alloc_0 : memref<1x1x8x4x8x4xbf16, 2 : i32>
        memref.dealloc %alloc : memref<1x1x8x8x4x4xbf16, 2 : i32>
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
                "air-par-to-launch{depth=0 has-air-segment=true}",
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
