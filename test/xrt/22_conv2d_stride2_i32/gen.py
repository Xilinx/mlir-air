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
      func.func @conv_2d_nhwc_hwcf_dispatch_0_conv_2d_nhwc_hwcf_2x8x8x64x3x3x32_i32(%0 : memref<2x17x17x32xi32>, %1 : memref<3x3x32x64xi32>, %2 : memref<2x8x8x64xi32>) {
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c3 = arith.constant 3 : index
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c8, %c8, %c64) step (%c4, %c4, %c4) {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%arg0]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%arg1]
          %subview = memref.subview %0[0, %3, %4, 0] [2, 9, 9, 32] [1, 1, 1, 1] : memref<2x17x17x32xi32> to memref<2x9x9x32xi32, strided<[9248, 544, 32, 1], offset: ?>>
          %subview_0 = memref.subview %1[0, 0, 0, %arg2] [3, 3, 32, 4] [1, 1, 1, 1] : memref<3x3x32x64xi32> to memref<3x3x32x4xi32, strided<[6144, 2048, 64, 1], offset: ?>>
          %subview_1 = memref.subview %2[0, %arg0, %arg1, %arg2] [2, 4, 4, 4] [1, 1, 1, 1] : memref<2x8x8x64xi32> to memref<2x4x4x4xi32, strided<[4096, 512, 64, 1], offset: ?>>
          %alloc = memref.alloc() : memref<2x9x9x32xi32, 1 : i32>
          memref.copy %subview, %alloc : memref<2x9x9x32xi32, strided<[9248, 544, 32, 1], offset: ?>> to memref<2x9x9x32xi32, 1 : i32>
          %alloc_2 = memref.alloc() : memref<3x3x32x4xi32, 1 : i32>
          memref.copy %subview_0, %alloc_2 : memref<3x3x32x4xi32, strided<[6144, 2048, 64, 1], offset: ?>> to memref<3x3x32x4xi32, 1 : i32>
          %alloc_3 = memref.alloc() : memref<2x4x4x4xi32, 1 : i32>
          scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c4) step (%c1, %c1) {
            %subview_4 = memref.subview %alloc_3[%arg3, %arg4, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<2x4x4x4xi32, 1 : i32> to memref<1x1x4x4xi32, strided<[64, 16, 4, 1], offset: ?>, 1 : i32>
            %alloc_5 = memref.alloc() : memref<1x1x4x4xi32, 2 : i32>
            linalg.fill ins(%c0_i32 : i32) outs(%alloc_5 : memref<1x1x4x4xi32, 2 : i32>)
            scf.for %arg5 = %c0 to %c3 step %c1 {
              scf.for %arg6 = %c0 to %c3 step %c1 {
                scf.for %arg7 = %c0 to %c32 step %c8 {
                  %5 = affine.apply affine_map<(d0)[s0] -> (d0 * 2 + s0)>(%arg4)[%arg5]
                  %subview_6 = memref.subview %alloc[%arg3, %5, %arg6, %arg7] [1, 1, 7, 8] [1, 1, 1, 1] : memref<2x9x9x32xi32, 1 : i32> to memref<1x1x7x8xi32, strided<[2592, 288, 32, 1], offset: ?>, 1 : i32>
                  %subview_7 = memref.subview %alloc_2[%arg5, %arg6, %arg7, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<3x3x32x4xi32, 1 : i32> to memref<1x1x8x4xi32, strided<[384, 128, 4, 1], offset: ?>, 1 : i32>
                  %alloc_8 = memref.alloc() : memref<1x1x7x8xi32, 2 : i32>
                  memref.copy %subview_6, %alloc_8 : memref<1x1x7x8xi32, strided<[2592, 288, 32, 1], offset: ?>, 1 : i32> to memref<1x1x7x8xi32, 2 : i32>
                  %alloc_9 = memref.alloc() : memref<1x1x8x4xi32, 2 : i32>
                  memref.copy %subview_7, %alloc_9 : memref<1x1x8x4xi32, strided<[384, 128, 4, 1], offset: ?>, 1 : i32> to memref<1x1x8x4xi32, 2 : i32>
                  %subview_10 = memref.subview %alloc_8[0, 0, 0, 0] [1, 1, 7, 8] [1, 1, 1, 1] : memref<1x1x7x8xi32, 2 : i32> to memref<1x7x8xi32, strided<[56, 8, 1]>, 2 : i32>
                  %subview_11 = memref.subview %alloc_9[0, 0, 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<1x1x8x4xi32, 2 : i32> to memref<1x8x4xi32, strided<[32, 4, 1]>, 2 : i32>
                  %subview_12 = memref.subview %alloc_5[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<1x1x4x4xi32, 2 : i32> to memref<1x4x4xi32, strided<[16, 4, 1]>, 2 : i32>
                  linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%subview_10, %subview_11 : memref<1x7x8xi32, strided<[56, 8, 1]>, 2 : i32>, memref<1x8x4xi32, strided<[32, 4, 1]>, 2 : i32>) outs(%subview_12 : memref<1x4x4xi32, strided<[16, 4, 1]>, 2 : i32>)
                  memref.dealloc %alloc_8 : memref<1x1x7x8xi32, 2 : i32>
                  memref.dealloc %alloc_9 : memref<1x1x8x4xi32, 2 : i32>
                }
              }
            }
            memref.copy %alloc_5, %subview_4 : memref<1x1x4x4xi32, 2 : i32> to memref<1x1x4x4xi32, strided<[64, 16, 4, 1], offset: ?>, 1 : i32>
            memref.dealloc %alloc_5 : memref<1x1x4x4xi32, 2 : i32>
            scf.reduce 
          }
          memref.copy %alloc_3, %subview_1 : memref<2x4x4x4xi32, 1 : i32> to memref<2x4x4x4xi32, strided<[4096, 512, 64, 1], offset: ?>>
          memref.dealloc %alloc : memref<2x9x9x32xi32, 1 : i32>
          memref.dealloc %alloc_2 : memref<3x3x32x4xi32, 1 : i32>
          memref.dealloc %alloc_3 : memref<2x4x4x4xi32, 1 : i32>
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
        runtime_loop_tiling_sizes=[1, 1],
        use_lock_race_condition_fix=True,
    )
    backend.compile(air_module)
