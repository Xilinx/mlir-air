# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# BF16 matmul with non-tile-aligned M, N dimensions.
# Uses air-split-launch-for-padding to split the launch grid and apply
# memtile DMA padding (pad_after) on boundary blocks. Shim DMA reads are
# reduced to actual data size; memtile hardware zero-fills the remainder.
#
# Target: NPU2/Strix, BF16 in, F32 accumulation, BF16 out.
# Tile sizes: M_TILE=128, N_TILE=256, K must be multiple of K_L2_TILE (default 64).

import argparse
import math

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
from ml_dtypes import bfloat16
import filelock

import numpy as np

np.random.seed(42)

parser = argparse.ArgumentParser(
    prog="run.py",
    description="BF16 matmul with non-tile-aligned dimensions and DMA padding",
)
parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform_aie2p.mlir",
    help="Transform script path",
)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument(
    "--compile-mode",
    type=str,
    choices=["compile-only", "compile-and-run"],
    dest="compile_mode",
    default="compile-and-run",
    help="Configure to whether to run after compile",
)
parser.add_argument("--M", type=int, default=500, help="Matrix M dimension")
parser.add_argument("--N", type=int, default=500, help="Matrix N dimension")
parser.add_argument("--K", type=int, default=784, help="Matrix K dimension")
parser.add_argument(
    "--k-l2-tile",
    type=int,
    default=16,
    dest="k_l2_tile",
    help="L2 K-dimension tile size (K must be a multiple of this, must be multiple of 8)",
)
args = parser.parse_args()

# Tile dimensions
M_TILE = 128
N_TILE = 256
K_L2_TILE = args.k_l2_tile

# Actual matrix dimensions (may not be tile-aligned)
M_actual = args.M
N_actual = args.N
K_FULL = args.K

# Validate K alignment with L2 tile size
assert (
    K_L2_TILE % 8 == 0
), f"K L2 tile size must be a multiple of 8 (pack size), got {K_L2_TILE}"
assert K_FULL % K_L2_TILE == 0, (
    f"K={K_FULL} must be a multiple of K L2 tile size={K_L2_TILE}. "
    f"Use --k-l2-tile to set a compatible value."
)

# Padded dimensions for launch grid
M_padded = math.ceil(M_actual / M_TILE) * M_TILE
N_padded = math.ceil(N_actual / N_TILE) * N_TILE
LAUNCH_M = M_padded // M_TILE
LAUNCH_N = N_padded // N_TILE

needs_padding = (M_actual % M_TILE != 0) or (N_actual % N_TILE != 0)

if args.verbose:
    print(f"M={M_actual}, N={N_actual}, K={K_FULL}")
    print(f"M_padded={M_padded}, N_padded={N_padded}")
    print(f"Launch grid: {LAUNCH_M}x{LAUNCH_N}x1")
    if needs_padding:
        M_rem = M_actual % M_TILE
        N_rem = N_actual % N_TILE
        if M_rem:
            print(
                f"M padding: {M_TILE - M_rem} rows (last block has {M_rem} actual rows)"
            )
        if N_rem:
            print(
                f"N padding: {N_TILE - N_rem} cols (last block has {N_rem} actual cols)"
            )

# Block-aligned allocation sizes for input buffers.
INNER_BLOCK = 8
M_alloc = math.ceil(M_actual / INNER_BLOCK) * INNER_BLOCK if needs_padding else M_padded
N_alloc = math.ceil(N_actual / INNER_BLOCK) * INNER_BLOCK if needs_padding else N_padded
N_stride_b = N_alloc
# C stride uses tile-padded N (output shim DMAs write full tiles).
N_stride_c = N_padded

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = f"""
    #map = affine_map<(d0, d1) -> (d0, d1)>
    module {{
      func.func @matmul_bf16(%arg0: memref<*xbf16> {{tt.divisibility = 16 : i32}}, %arg1: memref<*xbf16> {{tt.divisibility = 16 : i32}}, %arg2: memref<*xbf16> {{tt.divisibility = 16 : i32}}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {{
        %cst = arith.constant 0.000000e+00 : f32
        %c{K_FULL} = arith.constant {K_FULL} : index
        %c_n_stride_b = arith.constant {N_stride_b} : index
        %c_n_stride_c = arith.constant {N_stride_c} : index
        %c{M_TILE}_i32 = arith.constant {M_TILE} : i32
        %c{N_TILE}_i32 = arith.constant {N_TILE} : i32
        %0 = arith.muli %arg6, %c{M_TILE}_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg7, %c{N_TILE}_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %1, %c{K_FULL} : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [{M_TILE}, {K_FULL}], strides: [{K_FULL}, 1] : memref<*xbf16> to memref<{M_TILE}x{K_FULL}xbf16, strided<[{K_FULL}, 1], offset: ?>>
        %alloc = memref.alloc() : memref<{M_TILE}x{K_FULL}xbf16>
        memref.copy %reinterpret_cast, %alloc : memref<{M_TILE}x{K_FULL}xbf16, strided<[{K_FULL}, 1], offset: ?>> to memref<{M_TILE}x{K_FULL}xbf16>
        %5 = bufferization.to_tensor %alloc restrict writable : memref<{M_TILE}x{K_FULL}xbf16> to tensor<{M_TILE}x{K_FULL}xbf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [{K_FULL}, {N_TILE}], strides: [{N_stride_b}, 1] : memref<*xbf16> to memref<{K_FULL}x{N_TILE}xbf16, strided<[{N_stride_b}, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<{K_FULL}x{N_TILE}xbf16>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<{K_FULL}x{N_TILE}xbf16, strided<[{N_stride_b}, 1], offset: ?>> to memref<{K_FULL}x{N_TILE}xbf16>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<{K_FULL}x{N_TILE}xbf16> to tensor<{K_FULL}x{N_TILE}xbf16>
        %7 = tensor.empty() : tensor<{M_TILE}x{N_TILE}xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<{M_TILE}x{N_TILE}xf32>) -> tensor<{M_TILE}x{N_TILE}xf32>
        %9 = linalg.matmul ins(%5, %6 : tensor<{M_TILE}x{K_FULL}xbf16>, tensor<{K_FULL}x{N_TILE}xbf16>) outs(%8 : tensor<{M_TILE}x{N_TILE}xf32>) -> tensor<{M_TILE}x{N_TILE}xf32>
        %10 = arith.muli %1, %c_n_stride_c : index
        %11 = arith.addi %10, %3 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [{M_TILE}, {N_TILE}], strides: [{N_stride_c}, 1] : memref<*xbf16> to memref<{M_TILE}x{N_TILE}xbf16, strided<[{N_stride_c}, 1], offset: ?>>
        %12 = tensor.empty() : tensor<{M_TILE}x{N_TILE}xbf16>
        %13 = linalg.generic {{indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}} ins(%9 : tensor<{M_TILE}x{N_TILE}xf32>) outs(%12 : tensor<{M_TILE}x{N_TILE}xbf16>) {{
        ^bb0(%in: f32, %out: bf16):
          %14 = arith.truncf %in : f32 to bf16
          linalg.yield %14 : bf16
        }} -> tensor<{M_TILE}x{N_TILE}xbf16>
        bufferization.materialize_in_destination %13 in writable %reinterpret_cast_2 : (tensor<{M_TILE}x{N_TILE}xbf16>, memref<{M_TILE}x{N_TILE}xbf16, strided<[{N_stride_c}, 1], offset: ?>>) -> ()
        return
      }}
    }}
    """
    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Tiling via transform script
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-override-memref-memory-space{scope=func memory-space=1}",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    with open(args.transform_script, "r") as f:
        transform_ir_string = f.read()
    # Parametrize L2 K-tile size in the transform script.
    if K_L2_TILE != 64:
        import re

        transform_ir_string = re.sub(
            r"(tile_using_for %copy1 tile_sizes \[0, )64(\])",
            rf"\g<1>{K_L2_TILE}\2",
            transform_ir_string,
        )
        transform_ir_string = re.sub(
            r"(tile_using_for %copy2 tile_sizes \[)64(\])",
            rf"\g<1>{K_L2_TILE}\2",
            transform_ir_string,
        )
        k_red_tile = K_L2_TILE // 8
        transform_ir_string = re.sub(
            r"(tile_using_for %packed_c tile_sizes \[0, 0, )8(\])",
            rf"\g<1>{k_red_tile}\2",
            transform_ir_string,
        )
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, air_module)

    ################################################
    ## Binding scf.parallel to air hierarchies
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={LAUNCH_M},{LAUNCH_N},1}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
                "canonicalize",
                "cse",
                "air-copy-to-dma",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ###############################################
    # Compile and run
    ###############################################

    pad_actual_m = M_actual if needs_padding and (M_actual % M_TILE != 0) else 0
    pad_actual_n = N_actual if needs_padding and (N_actual % N_TILE != 0) else 0

    input_type = bfloat16
    output_type = bfloat16

    A = np.zeros((M_alloc, K_FULL), dtype=input_type)
    A[:M_actual, :] = np.random.rand(M_actual, K_FULL).astype(input_type)
    B = np.zeros((K_FULL, N_alloc), dtype=input_type)
    B[:, :N_actual] = np.random.rand(K_FULL, N_actual).astype(input_type)

    C_ref_actual = np.matmul(
        A[:M_actual, :].astype(np.float32), B[:, :N_actual].astype(np.float32)
    ).astype(output_type)

    C_ref = np.zeros((M_padded, N_padded), dtype=output_type)
    C_ref[:M_actual, :N_actual] = C_ref_actual

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[1, 1],
            output_format="elf" if needs_padding else "xclbin",
            instance_name="matmul_bf16",
            actual_m=pad_actual_m,
            actual_n=pad_actual_n,
            tile_m=M_TILE,
            tile_n=N_TILE,
        )

        num_samples = 200
        sampled_row = np.random.randint(0, M_actual, num_samples)
        sampled_col = np.random.randint(0, N_actual, num_samples)
        sampled_indices = np.vstack([sampled_row, sampled_col])
        sampled_values = np.array(
            [C_ref_actual[r, c] for r, c in zip(sampled_row, sampled_col)],
            dtype=output_type,
        )
        sampled_data = {
            "shape": (M_padded, N_padded),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        exit(
            runner.run_test(
                air_module,
                inputs=[A, B],
                stochastic_expected_outputs=[sampled_data],
                rtol=max(1e-1, 2e-2 * (K_FULL / K_L2_TILE)),
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
        )
        module_function = backend.compile(air_module)
        backend.unload()
