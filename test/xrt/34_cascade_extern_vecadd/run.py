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
    #set = affine_set<()[s0] : (s0 - 3 == 0)>
    #set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
    module {
      func.func private @cascade_put(memref<1x1x2048xi32, 2 : i32>) attributes {link_with = "cascade.o", llvm.emit_c_interface}
      func.func private @cascade_get(memref<1x1x2048xi32, 2 : i32>) attributes {link_with = "cascade.o", llvm.emit_c_interface}
      air.channel @channel_1 [1]
      air.channel @channel_2 [1]
      func.func @scf1(%arg0: memref<1x1x2048xi32>, %arg1: memref<1x1x2048xi32>) {
        %c1 = arith.constant 1 : index
        air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<1x1x2048xi32>, memref<1x1x2048xi32> {
          %c4 = arith.constant 4 : index
          %c1_0 = arith.constant 1 : index
          air.channel.put  @channel_1[] (%arg6[] [] []) : (memref<1x1x2048xi32>)
          air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c4) attributes {link_with = "cascade.o"} {
            %c1_i32 = arith.constant 1 : i32
            %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
            linalg.fill ins(%c1_i32 : i32) outs(%alloc : memref<1x1x2048xi32, 2 : i32>)
            affine.if #set()[%arg9] {
              %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
              air.channel.get  @channel_1[] (%alloc_1[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
              linalg.add ins(%alloc_1, %alloc : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%alloc : memref<1x1x2048xi32, 2 : i32>)
              func.call @cascade_put(%alloc) : (memref<1x1x2048xi32, 2 : i32>) -> ()
            } else {
              affine.if #set1()[%arg9] {
                %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
                func.call @cascade_get(%alloc_1) : (memref<1x1x2048xi32, 2 : i32>) -> ()
                linalg.add ins(%alloc_1, %alloc : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%alloc : memref<1x1x2048xi32, 2 : i32>)
                func.call @cascade_put(%alloc) : (memref<1x1x2048xi32, 2 : i32>) -> ()
              } else {
                %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
                func.call @cascade_get(%alloc_1) : (memref<1x1x2048xi32, 2 : i32>) -> ()
                linalg.add ins(%alloc_1, %alloc : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%alloc : memref<1x1x2048xi32, 2 : i32>)
                air.channel.put  @channel_2[] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
              }
            }
          }
          air.channel.get  @channel_2[] (%arg7[] [] []) : (memref<1x1x2048xi32>)
        }
        return
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)

    ###############################################
    # Compile, run and compare results
    ###############################################

    input_a = np.arange(0, 2048, dtype=np.int32)
    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, 2048, num_samples)
            ]
        )

        # Compute reference results for sampled indices
        sampled_values = np.array(
            [
                input_a[i] + 4
                for i in zip(*sampled_indices)
            ],
            dtype=np.int32,
        )

        # Store as a dictionary
        sampled_data = {
            "shape": (2048),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
        )
        exit(
            runner.run_test(
                air_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
