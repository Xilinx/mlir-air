# run.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air
import air.compiler.util
from air.dialects import linalg, tensor, arith, func, memref
from air.ir import *
import air.passmanager
from air.dialects import air as airdialect
from air.compiler.util import run_transform
import argparse
import sys

from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

parser = argparse.ArgumentParser(
    prog="run.py",
    description="Builds, runs, and tests the cascade example",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
)
parser.add_argument(
    "--compile-mode",
    type=str,
    choices=["compile-only", "compile-and-run"],
    dest="compile_mode",
    default="compile-and-run",
    help="Configure to whether to run after compile",
)
args = parser.parse_args()

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Tiling
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<(d0) -> (d0 * 64)>
    #map1 = affine_map<(d0) -> (d0 * 32)>
    #map2 = affine_map<(d0, d1) -> ((d0 + d1 * 4) * 32)>
    #set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 3)>
    #set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 >= 0, -s1 + 2 >= 0)>
    module {
      func.func @forward(%arg0: memref<512xi32>, %arg1: memref<512x256xi32>) -> memref<256xi32> {
        %c4 = arith.constant 4 : index
        %alloc = memref.alloc() : memref<256xi32>
        air.launch (%arg2) in (%arg3=%c4) args(%arg4=%arg0, %arg5=%arg1, %arg6=%alloc) : memref<512xi32>, memref<512x256xi32>, memref<256xi32> {
          %c1 = arith.constant 1 : index
          %0 = affine.apply #map(%arg2)
          %subview = memref.subview %arg4[0] [512] [1] : memref<512xi32> to memref<512xi32, strided<[1]>>
          %1 = affine.apply #map(%arg2)
          air.segment  unroll(%arg7) in (%arg8=%c1) args(%arg9=%0, %arg10=%arg5, %arg11=%1, %arg12=%arg6, %arg13=%subview) : index, memref<512x256xi32>, index, memref<256xi32>, memref<512xi32, strided<[1]>> {
            %c4_0 = arith.constant 4 : index
            %c2 = arith.constant 2 : index
            %c1_0 = arith.constant 1 : index
            %2 = affine.apply #map(%arg7)
            %3 = arith.addi %arg9, %2 : index
            %subview_1 = memref.subview %arg10[0, %3] [512, 64] [1, 1] : memref<512x256xi32> to memref<512x64xi32, strided<[256, 1], offset: ?>>
            %4 = affine.apply #map(%arg7)
            %5 = arith.addi %arg11, %4 : index
            %subview_2 = memref.subview %arg12[%5] [64] [1] : memref<256xi32> to memref<64xi32, strided<[1], offset: ?>>
            %alloc_3 = memref.alloc() : memref<64xi32, 1 : i32>
            %alloc_4 = memref.alloc() : memref<512xi32, 1 : i32>
            %alloc_5 = memref.alloc() : memref<512x64xi32, 1 : i32>
            air.dma_memcpy_nd (%alloc_4[] [] [], %arg13[] [] []) {id = 1 : i32} : (memref<512xi32, 1 : i32>, memref<512xi32, strided<[1]>>)
            %c0 = arith.constant 0 : index
            %c512 = arith.constant 512 : index
            %c64 = arith.constant 64 : index
            %c32_33 = arith.constant 32 : index
            %c256 = arith.constant 256 : index
            %c1_6 = arith.constant 1 : index
            air.dma_memcpy_nd (%alloc_5[] [] [], %arg10[%c0, %3] [%c512, %c64] [%c256, %c1_6]) {id = 2 : i32} : (memref<512x64xi32, 1 : i32>, memref<512x256xi32>)

            %cst = arith.constant 0 : i32
            scf.parallel (%y) = (%c0) to (%c2) step (%c1_0){
              %cascade_buffer = memref.alloc() : memref<32xi32, 2 : i32>
              linalg.fill ins(%cst : i32) outs(%cascade_buffer : memref<32xi32, 2 : i32>)
              %sum = scf.parallel (%x) = (%c0) to (%c4_0) step (%c1_0) init (%cascade_buffer) -> memref<32xi32, 2 : i32> {
                %c0_i32 = arith.constant 0 : i32
                %c0_9 = arith.constant 0 : index
                %c16 = arith.constant 16 : index
                %c4_10 = arith.constant 4 : index
                %c1_10 = arith.constant 1 : index
                %6 = affine.apply #map1(%y)
                %7 = affine.apply #map1(%y)
                %subview_11 = memref.subview %alloc_3[%7] [32] [1] : memref<64xi32, 1 : i32> to memref<32xi32, strided<[1], offset: ?>, 1 : i32>
                %alloc_12 = memref.alloc() : memref<32xi32, 2 : i32>
                linalg.fill ins(%c0_i32 : i32) outs(%alloc_12 : memref<32xi32, 2 : i32>)
                scf.for %arg21 = %c0_9 to %c4_10 step %c1_10 {
                  %8 = affine.apply #map2(%arg21, %x)
                  %subview_14 = memref.subview %alloc_4[%8] [32] [1] : memref<512xi32, 1 : i32> to memref<32xi32, strided<[1], offset: ?>, 1 : i32>
                  %subview_15 = memref.subview %alloc_5[%8, %6] [32, 32] [1, 1] : memref<512x64xi32, 1 : i32> to memref<32x32xi32, strided<[64, 1], offset: ?>, 1 : i32>
                  %subview_16 = memref.subview %alloc_12[0] [32] [1] : memref<32xi32, 2 : i32> to memref<32xi32, strided<[1]>, 2 : i32>
                  %alloc_17 = memref.alloc() : memref<32xi32, 2 : i32>
                  %alloc_18 = memref.alloc() : memref<32x32xi32, 2 : i32>
                  %c32_19 = arith.constant 32 : index
                  %c1_20 = arith.constant 1 : index
                  air.dma_memcpy_nd (%alloc_17[] [] [], %alloc_4[%8] [%c32_19] [%c1_20]) {id = 3 : i32} : (memref<32xi32, 2 : i32>, memref<512xi32, 1 : i32>)
                  %c32_21 = arith.constant 32 : index
                  %c32_22 = arith.constant 32 : index
                  %c64_23 = arith.constant 64 : index
                  %c1_24 = arith.constant 1 : index
                  air.dma_memcpy_nd (%alloc_18[] [] [], %alloc_5[%8, %6] [%c32_21, %c32_22] [%c64_23, %c1_24]) {id = 4 : i32} : (memref<32x32xi32, 2 : i32>, memref<512x64xi32, 1 : i32>)
                  linalg.vecmat ins(%alloc_17, %alloc_18 : memref<32xi32, 2 : i32>, memref<32x32xi32, 2 : i32>) outs(%subview_16 : memref<32xi32, strided<[1]>, 2 : i32>)
                  memref.dealloc %alloc_17 : memref<32xi32, 2 : i32>
                  memref.dealloc %alloc_18 : memref<32x32xi32, 2 : i32>
                }                
                
                scf.reduce(%alloc_12 : memref<32xi32, 2 : i32>) {
                ^bb0(%a4: memref<32xi32, 2 : i32>, %a5: memref<32xi32, 2 : i32>):
                  linalg.add ins(%a4, %a5 : memref<32xi32, 2 : i32>, memref<32xi32, 2 : i32>) outs(%a4 : memref<32xi32, 2 : i32>)
                  scf.reduce.return %a4 : memref<32xi32, 2 : i32>
                }
              }
              %77 = affine.apply #map1(%y)
              air.dma_memcpy_nd (%alloc_3[%77] [%c32_33] [%c1_0], %sum[] [] []) : (memref<64xi32, 1 : i32>, memref<32xi32, 2 : i32>)
            }
            memref.dealloc %alloc_4 : memref<512xi32, 1 : i32>
            memref.dealloc %alloc_5 : memref<512x64xi32, 1 : i32>
            %c64_7 = arith.constant 64 : index
            %c1_8 = arith.constant 1 : index
            air.dma_memcpy_nd (%arg12[%5] [%c64_7] [%c1_8], %alloc_3[] [] []) {id = 6 : i32} : (memref<256xi32>, memref<64xi32, 1 : i32>)
            memref.dealloc %alloc_3 : memref<64xi32, 1 : i32>
          }
        }
        return %alloc : memref<256xi32>
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
                "air-par-to-herd{depth=-1}",
                "func.func(air-fuse-nested-herd)",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ###############################################
    # Compile, run and compare results
    ###############################################

    # Default values.
    K = 512
    N = 256
    input_a = np.arange(0, K, dtype=np.int32).reshape(
        K,
    )
    input_b = np.arange(0, K * N, dtype=np.int32).reshape(K, N)
    if args.compile_mode == "compile-and-run":
        output_c = np.dot(input_a.astype(np.int32), input_b.astype(np.int32))
        runner = XRTRunner(verbose=args.verbose, omit_while_true_loop=False)
        exit(
            runner.run_test(
                air_module,
                inputs=[input_a, input_b],
                expected_outputs=[output_c],
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
        )
        module_function = backend.compile(air_module)

        backend.unload()
