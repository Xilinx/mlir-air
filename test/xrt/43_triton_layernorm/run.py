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

parser = argparse.ArgumentParser(
    prog="run.py",
    description="Builds, runs, and tests the LayerNorm example",
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
    default=128,
    help="M (parallel) dimension size",
)
parser.add_argument(
    "--N",
    type=int,
    dest="N",
    default=128,
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


def layer_norm(x_arg, y, weight, bias, mean, rstd, eps=1e-5):
    """
    Numpy implementation of layer normalization.

    Args:
        x_arg: Input array of shape [M, N]
        y: Output array (will be filled)
        weight: Weight array of shape [N]
        bias: Bias array of shape [N]
        mean: Mean array of shape [M] (will be filled)
        rstd: Reciprocal standard deviation array of shape [M] (will be filled)
        eps: Small constant for numerical stability

    Returns:
        y: Normalized output
    """
    M, N = x_arg.shape

    # Compute mean along last axis
    mean[:] = np.mean(x_arg, axis=-1)

    # Compute variance along last axis
    variance = np.mean((x_arg - mean.reshape(-1, 1)) ** 2, axis=-1)

    # Compute reciprocal standard deviation
    rstd[:] = 1.0 / np.sqrt(variance + eps)

    # Normalize: (x - mean) * rstd
    normalized = (x_arg - mean.reshape(-1, 1)) * rstd.reshape(-1, 1)

    # Apply affine transformation: normalized * weight + bias
    y[:] = normalized * weight + bias

    return y


with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<(d0, d1) -> (d0, d1)>
    #map1 = affine_map<(d0, d1) -> (d0, 0)>
    module {
      func.func @_layer_norm_fwd_fused(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
        %cst = arith.constant 0.000000e+00 : f32
        %c128 = arith.constant 128 : index
        %cst_0 = arith.constant 1.000000e+00 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %cst_2 = arith.constant 128.0 : f32
        %c4_i32 = arith.constant 4 : i32
        %0 = tensor.empty() : tensor<4x128xf32>
        %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x128xf32>) -> tensor<4x128xf32>
        %2 = tensor.empty() : tensor<4x1xf32>
        %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<4x1xf32>) -> tensor<4x1xf32>
        %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<4x1xf32>) -> tensor<4x1xf32>
        %5 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<4x1xf32>) -> tensor<4x1xf32>
        %6 = arith.muli %arg5, %c4_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = arith.muli %7, %c128 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [4, 128], strides: [128, 1] : memref<*xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
        %alloc = memref.alloc() : memref<4x128xf32>
        memref.copy %reinterpret_cast, %alloc : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x128xf32>
        %9 = bufferization.to_tensor %alloc restrict writable : memref<4x128xf32> to tensor<4x128xf32>
        %10 = tensor.empty() : tensor<128x4xf32>
        %transposed = linalg.transpose ins(%9 : tensor<4x128xf32>) outs(%10 : tensor<128x4xf32>) permutation = [1, 0] 
        %11 = tensor.empty() : tensor<4xf32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<4xf32>) -> tensor<4xf32>
        %reduced = linalg.reduce ins(%transposed : tensor<128x4xf32>) outs(%12 : tensor<4xf32>) dimensions = [0] 
          (%in: f32, %init: f32) {
            %26 = arith.addf %in, %init : f32
            linalg.yield %26 : f32
          }
        %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
        %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %9 : tensor<4x128xf32>, tensor<4x128xf32>) outs(%9 : tensor<4x128xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.mulf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x128xf32>
        %transposed_3 = linalg.transpose ins(%13 : tensor<4x128xf32>) outs(%10 : tensor<128x4xf32>) permutation = [1, 0] 
        %reduced_4 = linalg.reduce ins(%transposed_3 : tensor<128x4xf32>) outs(%12 : tensor<4xf32>) dimensions = [0] 
          (%in: f32, %init: f32) {
            %26 = arith.addf %in, %init : f32
            linalg.yield %26 : f32
          }
        %expanded_5 = tensor.expand_shape %reduced_4 [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
        %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %5 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%expanded : tensor<4x1xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.divf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x1xf32>
        %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %14 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%14 : tensor<4x1xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.mulf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x1xf32>
        %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_5, %5 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%expanded_5 : tensor<4x1xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.divf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x1xf32>
        %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%16, %15 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%16 : tensor<4x1xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.subf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x1xf32>
        %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %4 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%17 : tensor<4x1xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.addf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x1xf32>
        %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<4x1xf32>) outs(%18 : tensor<4x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %26 = math.sqrt %in : f32
          linalg.yield %26 : f32
        } -> tensor<4x1xf32>
        %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %19 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%3 : tensor<4x1xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.divf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x1xf32>
        %21 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<4x1xf32>) outs(%0 : tensor<4x128xf32>) attrs =  {broadcastDims = array<i64: 1>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x128xf32>
        %22 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %21 : tensor<4x128xf32>, tensor<4x128xf32>) outs(%9 : tensor<4x128xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.subf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x128xf32>
        %23 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<4x1xf32>) outs(%0 : tensor<4x128xf32>) attrs =  {broadcastDims = array<i64: 1>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x128xf32>
        %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%22, %23 : tensor<4x128xf32>, tensor<4x128xf32>) outs(%22 : tensor<4x128xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.mulf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x128xf32>
        %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%24, %1 : tensor<4x128xf32>, tensor<4x128xf32>) outs(%24 : tensor<4x128xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %26 = arith.addf %in, %in_7 : f32
          linalg.yield %26 : f32
        } -> tensor<4x128xf32>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%8], sizes: [4, 128], strides: [128, 1] : memref<*xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
        bufferization.materialize_in_destination %25 in writable %reinterpret_cast_6 : (tensor<4x128xf32>, memref<4x128xf32, strided<[128, 1], offset: ?>>) -> ()
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
    # Binding scf.parallel to air hierarchies
    ###############################################
    M, N = args.M, args.N
    input_size = (M, N)
    tile_size = (4, N)
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

    input_type = np.float32

    # Fix random seed for reproducibility
    np.random.seed(0)

    # Initialize inputs with arange for more distinct distribution (better for bf16 testing)
    x_arg = np.random.rand(M, N).astype(input_type)  # Shape [M, N]

    # Compute expected output - the IR only does normalization without weight/bias
    eps = 9.99999974e-6  # Match the epsilon value in MLIR
    mean = np.mean(x_arg, axis=-1)  # Shape [M]
    variance = np.mean((x_arg - mean.reshape(-1, 1)) ** 2, axis=-1)  # Shape [M]
    rstd = 1.0 / np.sqrt(variance + eps)  # Shape [M]
    y_expected = (x_arg - mean.reshape(-1, 1)) * rstd.reshape(-1, 1)  # Shape [M, N]

    ###### Compile and test
    runner = XRTRunner(
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="_layer_norm_fwd_fused",
    )
    exit(
        runner.run_test(
            air_module,
            inputs=[x_arg],
            expected_outputs=[y_expected],
            rtol=1e-2,
            atol=1e-1,
        )
    )
