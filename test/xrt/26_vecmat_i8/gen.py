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
    #map = affine_map<()[s0] -> (s0 * 256)>
    #map1 = affine_map<()[s0] -> (s0 * 128)>
    #map2 = affine_map<()[s0] -> (s0 * 16)>
    #map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
    #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
    #map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
    module {
      func.func @vecmat_i8(%0 : memref<2048xi8>, %1 : memref<2048x1024xi8>, %2 : memref<1024xi32>) {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c128 = arith.constant 128 : index
        %c2048 = arith.constant 2048 : index
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        scf.parallel (%arg0) = (%c0) to (%c4) step (%c1) {
          %3 = affine.apply #map()[%arg0]
          %subview = memref.subview %2[%3] [256] [1] : memref<1024xi32> to memref<256xi32, strided<[1], offset: ?>>
          %alloc = memref.alloc() : memref<2048xi8, 1>
          scf.for %arg1 = %c0 to %c2048 step %c128 {
            %subview_2 = memref.subview %0[%arg1] [128] [1] : memref<2048xi8> to memref<128xi8, strided<[1], offset: ?>>
            %subview_3 = memref.subview %alloc[%arg1] [128] [1] : memref<2048xi8, 1> to memref<128xi8, strided<[1], offset: ?>, 1>
            memref.copy %subview_2, %subview_3 : memref<128xi8, strided<[1], offset: ?>> to memref<128xi8, strided<[1], offset: ?>, 1>
          }
          %alloc_0 = memref.alloc() : memref<2048x256xi8, 1>
          scf.for %arg1 = %c0 to %c2048 step %c128 {
            %4 = affine.apply #map()[%arg0]
            %subview_2 = memref.subview %1[%arg1, %4] [128, 256] [1, 1] : memref<2048x1024xi8> to memref<128x256xi8, strided<[1024, 1], offset: ?>>
            %subview_3 = memref.subview %alloc_0[%arg1, 0] [128, 256] [1, 1] : memref<2048x256xi8, 1> to memref<128x256xi8, strided<[256, 1], offset: ?>, 1>
            memref.copy %subview_2, %subview_3 : memref<128x256xi8, strided<[1024, 1], offset: ?>> to memref<128x256xi8, strided<[256, 1], offset: ?>, 1>
          }
          %alloc_1 = memref.alloc() : memref<256xi32, 1>
          scf.parallel (%arg1) = (%c0) to (%c2) step (%c1) {
            %4 = affine.apply #map1()[%arg1]
            %subview_2 = memref.subview %alloc_1[%4] [128] [1] : memref<256xi32, 1> to memref<128xi32, strided<[1], offset: ?>, 1>
            %alloc_3 = memref.alloc() : memref<16x8xi32, 2>
            linalg.fill ins(%c0_i32 : i32) outs(%alloc_3 : memref<16x8xi32, 2>)
            scf.for %arg2 = %c0 to %c128 step %c8 {
              %5 = affine.apply #map2()[%arg2]
              %subview_4 = memref.subview %alloc[%5] [128] [1] : memref<2048xi8, 1> to memref<128xi8, strided<[1], offset: ?>, 1>
              %alloc_5 = memref.alloc() : memref<8x16xi8, 2>
              %expand_shape = memref.expand_shape %subview_4 [[0, 1]] output_shape [8, 16] : memref<128xi8, strided<[1], offset: ?>, 1> into memref<8x16xi8, strided<[16, 1], offset: ?>, 1>
              air.dma_memcpy_nd (%alloc_5[] [] [], %expand_shape[] [] []) : (memref<8x16xi8, 2>, memref<8x16xi8, strided<[16, 1], offset: ?>, 1>)
              %6 = affine.apply #map2()[%arg2]
              %7 = affine.apply #map1()[%arg1]
              %subview_6 = memref.subview %alloc_0[%6, %7] [128, 128] [1, 1] : memref<2048x256xi8, 1> to memref<128x128xi8, strided<[256, 1], offset: ?>, 1>
              %alloc_7 = memref.alloc() : memref<16x8x16x8xi8, 2>
              %expand_shape_8 = memref.expand_shape %subview_6 [[0, 1], [2, 3]] output_shape [8, 16, 16, 8] : memref<128x128xi8, strided<[256, 1], offset: ?>, 1> into memref<8x16x16x8xi8, strided<[4096, 256, 8, 1], offset: ?>, 1>
              %transpose_9 = memref.transpose %expand_shape_8 (d0, d1, d2, d3) -> (d2, d0, d1, d3) : memref<8x16x16x8xi8, strided<[4096, 256, 8, 1], offset: ?>, 1> to memref<16x8x16x8xi8, strided<[8, 4096, 256, 1], offset: ?>, 1>
              air.dma_memcpy_nd (%alloc_7[] [] [], %transpose_9[] [] []) : (memref<16x8x16x8xi8, 2>, memref<16x8x16x8xi8, strided<[8, 4096, 256, 1], offset: ?>, 1>)
              linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction", "parallel", "reduction"], library_call = "vecmat_i8_i32"} ins(%alloc_5, %alloc_7 : memref<8x16xi8, 2>, memref<16x8x16x8xi8, 2>) outs(%alloc_3 : memref<16x8xi32, 2>) {
              ^bb0(%in: i8, %in_10: i8, %out: i32):
                %8 = arith.extsi %in : i8 to i32
                %9 = arith.extsi %in_10 : i8 to i32
                %10 = arith.muli %8, %9 : i32
                %11 = arith.addi %out, %10 : i32
                linalg.yield %11 : i32
              }
              memref.dealloc %alloc_5 : memref<8x16xi8, 2>
              memref.dealloc %alloc_7 : memref<16x8x16x8xi8, 2>
            }
            %transpose = memref.transpose %alloc_3 (d0, d1) -> (d0, d1) : memref<16x8xi32, 2> to memref<16x8xi32, strided<[8, 1]>, 2>
            air.dma_memcpy_nd (%subview_2[] [] [], %transpose[] [] []) : (memref<128xi32, strided<[1], offset: ?>, 1>, memref<16x8xi32, strided<[8, 1]>, 2>)
            memref.dealloc %alloc_3 : memref<16x8xi32, 2>
            scf.reduce 
          }
          memref.copy %alloc_1, %subview : memref<256xi32, 1> to memref<256xi32, strided<[1], offset: ?>>
          memref.dealloc %alloc : memref<2048xi8, 1>
          memref.dealloc %alloc_0 : memref<2048x256xi8, 1>
          memref.dealloc %alloc_1 : memref<256xi32, 1>
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
                "air-linalg-to-func{link-with=vm.o}",
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
        lower_linalg_to_func=True,
        air_loop_fusion=True,
    )
    module_function = backend.compile_and_load(air_module)
