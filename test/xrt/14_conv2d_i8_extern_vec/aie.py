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
from air.dialects.air import module_builder
from air.compiler.util import run_transform
import sys

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Tiling
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
                "air-linalg-to-func{link-with=conv.o}",
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
                "func.func(air-split-l2-memref)",
                "air-isolate-async-dma-loop-nests",
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
                "air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor="
                + str(1)
                + "}",
                "canonicalize",
                "cse",
                "func.func(air-renumber-dma)",
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
                "air-to-aie{row-offset=2 col-offset=0 device=npu1_4col emit-while-loop=true insert-trace-packet-flow=true}",
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
                "air-to-std",
                "canonicalize",
                "symbol-dce",
                "func.func(air-unroll-outer-affine-loops{depth=4})",
                "affine-expand-index-ops",
                "airrt-to-npu{trace-offset=73728 trace-size=262144}",
                "canonicalize",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open("aie.mlir", "w") as f:
        f.write(str(air_module))
