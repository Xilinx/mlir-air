# run.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import numpy as np
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
import filelock
from ml_dtypes import bfloat16

parser = argparse.ArgumentParser(
    prog="run.py",
    description="Builds, runs, and tests the matmul example",
)
parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform.mlir",
    help="Transform script path",
)
parser.add_argument(
    "--M",
    type=int,
    dest="M",
    default=256,
    help="M (parallel) dimension size",
)
parser.add_argument(
    "--N",
    type=int,
    dest="N",
    default=256,
    help="N (reduction) dimension size",
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


def softmax(x, axis=-1):
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<(d0, d1) -> (d0, d1)>
    #map1 = affine_map<(d0, d1) -> (d0, 0)>
    module {
      func.func @softmax_kernel(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
        %c4_i32 = arith.constant 4 : i32
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0xFF800000 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = arith.muli %arg5, %c4_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %1, %c256 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [4, 256], strides: [256, 1] : memref<*xbf16> to memref<4x256xbf16, strided<[256, 1], offset: ?>>
        %alloc = memref.alloc() : memref<4x256xbf16>
        memref.copy %reinterpret_cast, %alloc : memref<4x256xbf16, strided<[256, 1], offset: ?>> to memref<4x256xbf16>
        %3 = bufferization.to_tensor %alloc restrict writable : memref<4x256xbf16> to tensor<4x256xbf16>
        %4 = tensor.empty() : tensor<4x256xf32>
        %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4x256xbf16>) outs(%4 : tensor<4x256xf32>) {
        ^bb0(%in: bf16, %out: f32):
          %17 = arith.extf %in : bf16 to f32
          linalg.yield %17 : f32
        } -> tensor<4x256xf32>
        %6 = tensor.empty() : tensor<256x4xf32>
        %transposed = linalg.transpose ins(%5 : tensor<4x256xf32>) outs(%6 : tensor<256x4xf32>) permutation = [1, 0] 
        %7 = tensor.empty() : tensor<4xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4xf32>) -> tensor<4xf32>
        %reduced = linalg.reduce ins(%transposed : tensor<256x4xf32>) outs(%8 : tensor<4xf32>) dimensions = [0] 
          (%in: f32, %init: f32) {
            %17 = arith.maxnumf %in, %init : f32
            linalg.yield %17 : f32
          }
        %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
        %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<4x1xf32>) outs(%4 : tensor<4x256xf32>) attrs =  {broadcastDims = array<i64: 1>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x256xf32>
        %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %9 : tensor<4x256xf32>, tensor<4x256xf32>) outs(%5 : tensor<4x256xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %17 = arith.subf %in, %in_5 : f32
          linalg.yield %17 : f32
        } -> tensor<4x256xf32>
        %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<4x256xf32>) outs(%10 : tensor<4x256xf32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = math.exp %in : f32
          linalg.yield %17 : f32
        } -> tensor<4x256xf32>
        %transposed_1 = linalg.transpose ins(%11 : tensor<4x256xf32>) outs(%6 : tensor<256x4xf32>) permutation = [1, 0] 
        %12 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<4xf32>) -> tensor<4xf32>
        %reduced_2 = linalg.reduce ins(%transposed_1 : tensor<256x4xf32>) outs(%12 : tensor<4xf32>) dimensions = [0] 
          (%in: f32, %init: f32) {
            %17 = arith.addf %in, %init : f32
            linalg.yield %17 : f32
          }
        %expanded_3 = tensor.expand_shape %reduced_2 [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
        %13 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_3 : tensor<4x1xf32>) outs(%4 : tensor<4x256xf32>) attrs =  {broadcastDims = array<i64: 1>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x256xf32>
        %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %13 : tensor<4x256xf32>, tensor<4x256xf32>) outs(%11 : tensor<4x256xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %17 = arith.divf %in, %in_5 : f32
          linalg.yield %17 : f32
        } -> tensor<4x256xf32>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [4, 256], strides: [256, 1] : memref<*xbf16> to memref<4x256xbf16, strided<[256, 1], offset: ?>>
        %15 = tensor.empty() : tensor<4x256xbf16>
        %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<4x256xf32>) outs(%15 : tensor<4x256xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %17 = arith.truncf %in : f32 to bf16
          linalg.yield %17 : bf16
        } -> tensor<4x256xbf16>
        bufferization.materialize_in_destination %16 in writable %reinterpret_cast_4 : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[256, 1], offset: ?>>) -> ()
        return
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Tiling
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-resolve-tensor-opoperand-conflicts",
                "air-override-memref-memory-space{scope=func memory-space=1}",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## Tiling
    ################################################

    # Load the MLIR transform IR from an external file
    with open(args.transform_script, "r") as f:
        transform_ir_string = f.read()
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, air_module)

    ###############################################
    # Binding scf.paralell to air hierarchies
    ###############################################
    M, N = args.M, args.N
    input_size = (M, N)
    tile_size = (4, N)  # herd size = 4 (4 AIE cores)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},1}})",
                "air-par-to-launch{depth=-1 has-air-segment=true}",
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

    # input_type = np.float32
    input_type = bfloat16
    A = np.random.rand(M, N).astype(input_type)  # Shape [M, N]
    C = softmax(A).astype(input_type)

    ###### Compile and test
    runner = XRTRunner(
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="softmax_kernel",
    )
    exit(
        runner.run_test(
            air_module,
            inputs=[A],
            expected_outputs=[C],
            rtol=1e-2,
            atol=1e-3,
        )
    )
