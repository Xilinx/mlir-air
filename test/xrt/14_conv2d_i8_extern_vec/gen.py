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
      func.func private @conv(memref<1x3x4x6x8xi8, 2 : i32>, memref<3x3x4x1x8x8xi8, 2 : i32>, memref<1x1x4x1x8xi32, 2 : i32>) attributes {link_with = "conv.o", llvm.emit_c_interface}
      func.func @conv_2d_nhwc_hwcf_q_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_i8xi8xi32(%0 : memref<2x14x14x32xi8>, %1 : memref<3x3x32x64xi8>, %2 : memref<2x12x12x64xi32>) {
        %c8 = arith.constant 8 : index
        %c64 = arith.constant 64 : index
        %c12 = arith.constant 12 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<1x1x4x1x8xi32, 2 : i32>
        %alloc_0 = memref.alloc() : memref<3x3x4x1x8x8xi8, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x3x4x6x8xi8, 2 : i32>
        %alloc_2 = memref.alloc() : memref<1x4x4x8xi32, 1 : i32>
        %alloc_3 = memref.alloc() : memref<3x3x32x8xi8, 1 : i32>
        %alloc_4 = memref.alloc() : memref<1x6x6x32xi8, 1 : i32>
        scf.parallel (%arg0, %arg1, %arg2, %arg3) = (%c0, %c0, %c0, %c0) to (%c2, %c12, %c12, %c64) step (%c1, %c4, %c4, %c8) {
          %subview = memref.subview %0[%arg0, %arg1, %arg2, 0] [1, 6, 6, 32] [1, 1, 1, 1] : memref<2x14x14x32xi8> to memref<1x6x6x32xi8, strided<[6272, 448, 32, 1], offset: ?>>
          %subview_5 = memref.subview %1[0, 0, 0, %arg3] [3, 3, 32, 8] [1, 1, 1, 1] : memref<3x3x32x64xi8> to memref<3x3x32x8xi8, strided<[6144, 2048, 64, 1], offset: ?>>
          %subview_6 = memref.subview %2[%arg0, %arg1, %arg2, %arg3] [1, 4, 4, 8] [1, 1, 1, 1] : memref<2x12x12x64xi32> to memref<1x4x4x8xi32, strided<[9216, 768, 64, 1], offset: ?>>
          memref.copy %subview, %alloc_4 : memref<1x6x6x32xi8, strided<[6272, 448, 32, 1], offset: ?>> to memref<1x6x6x32xi8, 1 : i32>
          memref.copy %subview_5, %alloc_3 : memref<3x3x32x8xi8, strided<[6144, 2048, 64, 1], offset: ?>> to memref<3x3x32x8xi8, 1 : i32>
          scf.parallel (%arg4) = (%c0) to (%c4) step (%c1) {
            %subview_7 = memref.subview %alloc_4[0, %arg4, 0, 0] [1, 3, 6, 32] [1, 1, 1, 1] : memref<1x6x6x32xi8, 1 : i32> to memref<1x3x6x32xi8, strided<[1152, 192, 32, 1], offset: ?>, 1 : i32>
            %cast = memref.cast %alloc_3 : memref<3x3x32x8xi8, 1 : i32> to memref<3x3x32x8xi8, strided<[768, 256, 8, 1], offset: ?>, 1 : i32>
            %subview_8 = memref.subview %alloc_2[0, %arg4, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x4x4x8xi32, 1 : i32> to memref<1x1x4x8xi32, strided<[128, 32, 8, 1], offset: ?>, 1 : i32>
            %expand_shape = memref.expand_shape %subview_7 [[0], [1], [2], [3, 4]] output_shape [1, 3, 6, 4, 8] : memref<1x3x6x32xi8, strided<[1152, 192, 32, 1], offset: ?>, 1 : i32> into memref<1x3x6x4x8xi8, strided<[1152, 192, 32, 8, 1], offset: ?>, 1 : i32>
            %transpose = memref.transpose %expand_shape (d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4) : memref<1x3x6x4x8xi8, strided<[1152, 192, 32, 8, 1], offset: ?>, 1 : i32> to memref<1x3x4x6x8xi8, strided<[1152, 192, 8, 32, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_1[] [] [], %transpose[] [] []) : (memref<1x3x4x6x8xi8, 2 : i32>, memref<1x3x4x6x8xi8, strided<[1152, 192, 8, 32, 1], offset: ?>, 1 : i32>)
            %expand_shape_9 = memref.expand_shape %cast [[0], [1], [2, 3], [4, 5]] output_shape [3, 3, 4, 8, 1, 8] : memref<3x3x32x8xi8, strided<[768, 256, 8, 1], offset: ?>, 1 : i32> into memref<3x3x4x8x1x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32>
            %transpose_10 = memref.transpose %expand_shape_9 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5) : memref<3x3x4x8x1x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32> to memref<3x3x4x1x8x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32>
            air.dma_memcpy_nd (%alloc_0[] [] [], %transpose_10[] [] []) : (memref<3x3x4x1x8x8xi8, 2 : i32>, memref<3x3x4x1x8x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32>)
            linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<1x1x4x1x8xi32, 2 : i32>)
            func.call @conv(%alloc_1, %alloc_0, %alloc) : (memref<1x3x4x6x8xi8, 2 : i32>, memref<3x3x4x1x8x8xi8, 2 : i32>, memref<1x1x4x1x8xi32, 2 : i32>) -> ()
            %subview_11 = memref.subview %alloc[0, 0, 0, 0, 0] [1, 1, 4, 1, 8] [1, 1, 1, 1, 1] : memref<1x1x4x1x8xi32, 2 : i32> to memref<1x1x4x8xi32, 2 : i32>
            %transpose_12 = memref.transpose %subview_11 (d0, d1, d2, d3) -> (d0, d1, d2, d3) : memref<1x1x4x8xi32, 2 : i32> to memref<1x1x4x8xi32, strided<[32, 32, 8, 1]>, 2 : i32>
            air.dma_memcpy_nd (%subview_8[] [] [], %transpose_12[] [] []) : (memref<1x1x4x8xi32, strided<[128, 32, 8, 1], offset: ?>, 1 : i32>, memref<1x1x4x8xi32, strided<[32, 32, 8, 1]>, 2 : i32>)
            scf.reduce 
          }
          memref.copy %alloc_2, %subview_6 : memref<1x4x4x8xi32, 1 : i32> to memref<1x4x4x8xi32, strided<[9216, 768, 64, 1], offset: ?>>
          scf.reduce 
        }
        memref.dealloc %alloc_4 : memref<1x6x6x32xi8, 1 : i32>
        memref.dealloc %alloc_3 : memref<3x3x32x8xi8, 1 : i32>
        memref.dealloc %alloc_2 : memref<1x4x4x8xi32, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x3x4x6x8xi8, 2 : i32>
        memref.dealloc %alloc_0 : memref<3x3x4x1x8x8xi8, 2 : i32>
        memref.dealloc %alloc : memref<1x1x4x1x8xi32, 2 : i32>
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
        lower_linalg_to_func="conv.o",
        air_loop_fusion=True,
        trace_offset=73728,
        trace_size=262144,
        runtime_loop_tiling_sizes=[1, 1],
    )
    module_function = backend.compile_and_load(air_module)
