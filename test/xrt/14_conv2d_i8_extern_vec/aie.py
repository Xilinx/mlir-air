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
    #map = affine_map<()[s0] -> (s0 * 4)>
    #map1 = affine_map<()[s0] -> (s0 * 8)>
    module {
      func.func @conv_static_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_i32(%0 : memref<2x14x14x32xi8>, %1 : memref<3x3x32x64xi8>, %2 : memref<2x12x12x64xi32>) {
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c3 = arith.constant 3 : index
        %c1 = arith.constant 1 : index
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        scf.parallel (%arg0, %arg1, %arg2, %arg3) = (%c0, %c0, %c0, %c0) to (%c2, %c3, %c3, %c8) step (%c1, %c1, %c1, %c1) {
          %3 = affine.apply #map()[%arg1]
          %4 = affine.apply #map()[%arg2]
          %5 = affine.apply #map1()[%arg3]
          %subview = memref.subview %0[%arg0, %3, %4, 0] [1, 6, 6, 32] [1, 1, 1, 1] : memref<2x14x14x32xi8> to memref<1x6x6x32xi8, strided<[6272, 448, 32, 1], offset: ?>>
          %subview_0 = memref.subview %1[0, 0, 0, %5] [3, 3, 32, 8] [1, 1, 1, 1] : memref<3x3x32x64xi8> to memref<3x3x32x8xi8, strided<[6144, 2048, 64, 1], offset: ?>>
          %subview_1 = memref.subview %2[%arg0, %3, %4, %5] [1, 4, 4, 8] [1, 1, 1, 1] : memref<2x12x12x64xi32> to memref<1x4x4x8xi32, strided<[9216, 768, 64, 1], offset: ?>>
          %alloc = memref.alloc() : memref<1x6x6x32xi8, 1>
          memref.copy %subview, %alloc : memref<1x6x6x32xi8, strided<[6272, 448, 32, 1], offset: ?>> to memref<1x6x6x32xi8, 1>
          %alloc_2 = memref.alloc() : memref<3x3x32x8xi8, 1>
          memref.copy %subview_0, %alloc_2 : memref<3x3x32x8xi8, strided<[6144, 2048, 64, 1], offset: ?>> to memref<3x3x32x8xi8, 1>
          %alloc_3 = memref.alloc() : memref<1x4x4x8xi32, 1>
          scf.parallel (%arg4) = (%c0) to (%c4) step (%c1) {
            %subview_4 = memref.subview %alloc[0, %arg4, 0, 0] [1, 3, 6, 32] [1, 1, 1, 1] : memref<1x6x6x32xi8, 1> to memref<1x3x6x32xi8, strided<[1152, 192, 32, 1], offset: ?>, 1>
            %subview_5 = memref.subview %alloc_3[0, %arg4, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x4x4x8xi32, 1> to memref<1x1x4x8xi32, strided<[128, 32, 8, 1], offset: ?>, 1>
            %alloc_6 = memref.alloc() : memref<1x1x4x8xi32, 2>
            linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<1x1x4x8xi32, 2>)
            %subview_7 = memref.subview %alloc_6[0, 0, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x1x4x8xi32, 2> to memref<1x4x8xi32, strided<[32, 8, 1]>, 2>
            scf.for %arg5 = %c0 to %c3 step %c1 {
              scf.for %arg6 = %c0 to %c3 step %c1 {
                scf.for %arg7 = %c0 to %c32 step %c8 {
                  %subview_8 = memref.subview %subview_4[0, %arg5, %arg6, %arg7] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x3x6x32xi8, strided<[1152, 192, 32, 1], offset: ?>, 1> to memref<1x1x4x8xi8, strided<[1152, 192, 32, 1], offset: ?>, 1>
                  %subview_9 = memref.subview %alloc_2[%arg5, %arg6, %arg7, 0] [1, 1, 8, 8] [1, 1, 1, 1] : memref<3x3x32x8xi8, 1> to memref<1x1x8x8xi8, strided<[768, 256, 8, 1], offset: ?>, 1>
                  %subview_10 = memref.subview %subview_8[0, 0, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x1x4x8xi8, strided<[1152, 192, 32, 1], offset: ?>, 1> to memref<1x4x8xi8, strided<[1152, 32, 1], offset: ?>, 1>
                  %subview_11 = memref.subview %subview_9[0, 0, 0, 0] [1, 1, 8, 8] [1, 1, 1, 1] : memref<1x1x8x8xi8, strided<[768, 256, 8, 1], offset: ?>, 1> to memref<1x8x8xi8, strided<[768, 8, 1], offset: ?>, 1>
                  %alloc_12 = memref.alloc() : memref<1x4x8xi8, 2>
                  memref.copy %subview_10, %alloc_12 : memref<1x4x8xi8, strided<[1152, 32, 1], offset: ?>, 1> to memref<1x4x8xi8, 2>
                  %alloc_13 = memref.alloc() : memref<1x8x8xi8, 2>
                  memref.copy %subview_11, %alloc_13 : memref<1x8x8xi8, strided<[768, 8, 1], offset: ?>, 1> to memref<1x8x8xi8, 2>
                  linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%alloc_12, %alloc_13 : memref<1x4x8xi8, 2>, memref<1x8x8xi8, 2>) outs(%subview_7 : memref<1x4x8xi32, strided<[32, 8, 1]>, 2>)
                  memref.dealloc %alloc_12 : memref<1x4x8xi8, 2>
                  memref.dealloc %alloc_13 : memref<1x8x8xi8, 2>
                }
              }
            }
            memref.copy %alloc_6, %subview_5 : memref<1x1x4x8xi32, 2> to memref<1x1x4x8xi32, strided<[128, 32, 8, 1], offset: ?>, 1>
            memref.dealloc %alloc_6 : memref<1x1x4x8xi32, 2>
            scf.reduce 
          }
          memref.copy %alloc_3, %subview_1 : memref<1x4x4x8xi32, 1> to memref<1x4x4x8xi32, strided<[9216, 768, 64, 1], offset: ?>>
          memref.dealloc %alloc : memref<1x6x6x32xi8, 1>
          memref.dealloc %alloc_2 : memref<3x3x32x8xi8, 1>
          memref.dealloc %alloc_3 : memref<1x4x4x8xi32, 1>
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
