# aie.py -*- Python -*-
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

from air.backend.xrt import compile_air, get_air_runtime
import aie.utils

import numpy as np

np.random.seed(42)

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
parser.add_argument(
    "--output-format",
    type=str,
    dest="output_format",
    default="xclbin",
    choices=["elf", "xclbin"],
    help="Output format: 'xclbin' (default) or 'elf'",
)
args = parser.parse_args()

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Tiling
    ################################################

    air_tiled_ir_string = """
    #set = affine_set<()[s0] : (s0 == 0)>
    #set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
    module {
    air.channel @channel_0 [4]
      air.channel @channel_1 [1]
      air.channel @channel_2 [1]
      func.func @scf1(%arg0: memref<1x1x2048xi32>, %arg1: memref<1x1x2048xi32>) {
        %c1 = arith.constant 1 : index
        air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<1x1x2048xi32>, memref<1x1x2048xi32> {
          %c4 = arith.constant 4 : index
          %c1_0 = arith.constant 1 : index
          air.channel.put  @channel_1[] (%arg6[] [] []) : (memref<1x1x2048xi32>)
          air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c4) {
            %c1_i32 = arith.constant 1 : i32
            %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
            linalg.fill ins(%c1_i32 : i32) outs(%alloc : memref<1x1x2048xi32, 2 : i32>)
            affine.if #set()[%arg9] {
              %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
              air.channel.get  @channel_1[] (%alloc_1[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
              linalg.add ins(%alloc_1, %alloc : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%alloc : memref<1x1x2048xi32, 2 : i32>)
              air.channel.put  @channel_0[%arg9] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
            } else {
              affine.if #set1()[%arg9] {
                %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
                %c1_1 = arith.constant 1 : index
                %iv_sub1 = arith.subi %arg9, %c1_1 : index
                air.channel.get  @channel_0[%iv_sub1] (%alloc_1[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
                linalg.add ins(%alloc_1, %alloc : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%alloc : memref<1x1x2048xi32, 2 : i32>)
                air.channel.put  @channel_0[%arg9] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
              } else {
                %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
                %c1_1 = arith.constant 1 : index
                %iv_sub1 = arith.subi %arg9, %c1_1 : index
                air.channel.get  @channel_0[%iv_sub1] (%alloc_1[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
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

    npu_kernel = compile_air(
        air_module,
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="scf1",
        runtime_loop_tiling_sizes=[4, 4],
    )

    if args.compile_mode == "compile-only":
        exit(0)

    num_samples = 100
    sampled_indices = np.vstack([np.random.randint(0, 2048, num_samples)])

    # Compute reference results for sampled indices
    sampled_values = np.array(
        [input_a[i] + 4 for i in zip(*sampled_indices)],
        dtype=np.int32,
    )

    # Store as a dictionary
    sampled_data = {
        "shape": (2048),
        "indices": sampled_indices,
        "values": sampled_values,
    }

    runtime = get_air_runtime()
    dtype = sampled_data["values"].dtype
    shape = sampled_data["shape"]
    if isinstance(shape, int):
        shape = (shape,)
    io_args = [
        aie.utils.tensor(input_a),
        aie.utils.tensor(np.zeros(shape, dtype)),
    ]
    exit(runtime.run_test(npu_kernel, io_args, refs={}, stochastic_refs=[sampled_data]))
