# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# F32 matmul with bf16/bfp16 emulation, starting from Triton-XDNA asm_src.mlir.
# A is stored in K×M layout (transposed). All host data is f32.
# Uses transform dialect to tile, pack, and vectorize the matmul.
# Uses air-split-launch-for-padding for non-tile-aligned M, N dimensions.
#
# Target: NPU2/Strix, aie2p architecture.

import argparse
import math
import os

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
from ml_dtypes import bfloat16

import numpy as np

np.random.seed(42)

parser = argparse.ArgumentParser(
    prog="run.py",
    description="F32 matmul with bf16 emulation from asm_src.mlir, A in K×M layout",
)
parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform_aie2p.mlir",
    help="Transform script path",
)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-p", "--print-module-only", action="store_true")
parser.add_argument(
    "--compile-mode",
    type=str,
    choices=["compile-only", "compile-and-run"],
    dest="compile_mode",
    default="compile-and-run",
)
parser.add_argument("--M", type=int, default=500, help="Matrix M dimension")
parser.add_argument("--N", type=int, default=500, help="Matrix N dimension")
parser.add_argument("--K", type=int, default=784, help="Matrix K dimension")
parser.add_argument(
    "--k-l2-tile",
    type=int,
    default=16,
    dest="k_l2_tile",
    help="L2 K-dimension tile size (K must be a multiple of this)",
)
parser.add_argument(
    "--herd-m", type=int, default=4, dest="herd_m", help="Herd M dimension"
)
parser.add_argument(
    "--herd-n", type=int, default=4, dest="herd_n", help="Herd N dimension"
)
args = parser.parse_args()

# Tile dimensions (must match asm_src.mlir tile sizes)
TILE_M = 64
TILE_N = 32
K_L2_TILE = args.k_l2_tile
HERD_M = args.herd_m
HERD_N = args.herd_n

# Actual matrix dimensions (may not be tile-aligned)
M_actual = args.M
N_actual = args.N
K_FULL = args.K

assert (
    K_FULL % K_L2_TILE == 0
), f"K={K_FULL} must be a multiple of K_L2_TILE={K_L2_TILE}"

# Padded dimensions for launch grid
LAUNCH_TILE_M = TILE_M * HERD_M
LAUNCH_TILE_N = TILE_N * HERD_N
M_padded = math.ceil(M_actual / LAUNCH_TILE_M) * LAUNCH_TILE_M
N_padded = math.ceil(N_actual / LAUNCH_TILE_N) * LAUNCH_TILE_N
LAUNCH_M = M_padded // LAUNCH_TILE_M
LAUNCH_N = N_padded // LAUNCH_TILE_N

needs_padding = (M_actual % TILE_M != 0) or (N_actual % TILE_N != 0)

if args.verbose:
    print(f"M_actual={M_actual}, N_actual={N_actual}, K={K_FULL}")
    print(f"M_padded={M_padded}, N_padded={N_padded}")
    print(f"Launch grid: {LAUNCH_M}x{LAUNCH_N}x1")

# Allocation sizes: use full padded dimensions so L3→L2 DMA can read
# entire launch tiles from zero-filled host buffers. This handles both
# large matrices (multiple launches) and small matrices (M,N < tile size)
# where the actual data is smaller than a single launch tile.
M_alloc = M_padded
N_alloc = N_padded

################################################
# Load and transform IR
################################################

# Resolve paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
transform_path = (
    args.transform_script
    if os.path.isabs(args.transform_script)
    else os.path.join(script_dir, args.transform_script)
)

with air.ir.Context() as ctx, Location.unknown():

    # Generate full-K matmul IR for the FULL LAUNCH TILE (HERD_M*TILE_M × HERD_N*TILE_N).
    # This matches test 53's pattern where the string IR represents the full herd's work.
    # The transform splits it into multi-core via forall.
    # A is K×M (transposed, strides [1, M_alloc]); B is K×N (strides [N_alloc, 1]).
    LT_M = LAUNCH_TILE_M  # 256
    LT_N = LAUNCH_TILE_N  # 128
    air_tiled_ir_string = f"""
    module {{
      func.func @matmul_padding_kernel(%arg0: memref<*xf32> {{tt.divisibility = 16 : i32}}, %arg1: memref<*xf32> {{tt.divisibility = 16 : i32}}, %arg2: memref<*xf32> {{tt.divisibility = 16 : i32}}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {{
        %cst = arith.constant 0.000000e+00 : f32
        %c{K_FULL} = arith.constant {K_FULL} : index
        %c_m_alloc = arith.constant {M_alloc} : index
        %c_n_alloc = arith.constant {N_alloc} : index
        %c_n_padded = arith.constant {N_padded} : index
        %c{LT_M}_i32 = arith.constant {LT_M} : i32
        %c{LT_N}_i32 = arith.constant {LT_N} : i32
        %0 = arith.muli %arg6, %c{LT_M}_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg7, %c{LT_N}_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [{LT_M}, {K_FULL}], strides: [1, {M_alloc}] : memref<*xf32> to memref<{LT_M}x{K_FULL}xf32, strided<[1, {M_alloc}], offset: ?>>
        %alloc = memref.alloc() : memref<{LT_M}x{K_FULL}xf32>
        memref.copy %reinterpret_cast, %alloc : memref<{LT_M}x{K_FULL}xf32, strided<[1, {M_alloc}], offset: ?>> to memref<{LT_M}x{K_FULL}xf32>
        %5 = bufferization.to_tensor %alloc restrict writable : memref<{LT_M}x{K_FULL}xf32> to tensor<{LT_M}x{K_FULL}xf32>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [{K_FULL}, {LT_N}], strides: [{N_alloc}, 1] : memref<*xf32> to memref<{K_FULL}x{LT_N}xf32, strided<[{N_alloc}, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<{K_FULL}x{LT_N}xf32>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<{K_FULL}x{LT_N}xf32, strided<[{N_alloc}, 1], offset: ?>> to memref<{K_FULL}x{LT_N}xf32>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<{K_FULL}x{LT_N}xf32> to tensor<{K_FULL}x{LT_N}xf32>
        %7 = tensor.empty() : tensor<{LT_M}x{LT_N}xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<{LT_M}x{LT_N}xf32>) -> tensor<{LT_M}x{LT_N}xf32>
        %9 = linalg.matmul ins(%5, %6 : tensor<{LT_M}x{K_FULL}xf32>, tensor<{K_FULL}x{LT_N}xf32>) outs(%8 : tensor<{LT_M}x{LT_N}xf32>) -> tensor<{LT_M}x{LT_N}xf32>
        %10 = arith.muli %1, %c_n_padded : index
        %11 = arith.addi %10, %3 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [{LT_M}, {LT_N}], strides: [{N_padded}, 1] : memref<*xf32> to memref<{LT_M}x{LT_N}xf32, strided<[{N_padded}, 1], offset: ?>>
        bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<{LT_M}x{LT_N}xf32>, memref<{LT_M}x{LT_N}xf32, strided<[{N_padded}, 1], offset: ?>>) -> ()
        return
      }}
    }}
    """
    air_module = Module.parse(air_tiled_ir_string)

    # Pre-transform: override memory spaces to L2
    pipeline = (
        "builtin.module(air-override-memref-memory-space{scope=func memory-space=1})"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    # Apply transform script
    with open(transform_path, "r") as f:
        transform_ir_string = f.read()
    transform_ir = Module.parse(transform_ir_string, context=air_module.context)
    run_transform(transform_ir, air_module)

    if args.print_module_only:
        print(air_module)
        exit(0)

    # Wrap with parallel + convert to AIR hierarchy
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={LAUNCH_M},{LAUNCH_N},1 actual-sizes={M_actual},{N_actual},1}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    if args.verbose:
        # Dump IR before air-copy-to-dma for debugging
        with open("before_copy_to_dma.mlir", "w") as f:
            f.write(str(air_module))

    pipeline = "builtin.module(air-copy-to-dma)"
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    if args.verbose:
        print("Running module:")
        print(air_module)

    ###############################################
    # Compile and run
    ###############################################

    # Host data: f32. A is K×M_alloc (transposed, block-aligned actual size).
    # B is K×N_alloc. Zero-padded beyond M_actual/N_actual.
    input_a = np.zeros((K_FULL, M_alloc), dtype=np.float32)
    input_a[:, :M_actual] = (np.random.rand(K_FULL, M_actual) * 4).astype(np.float32)
    input_b = np.zeros((K_FULL, N_alloc), dtype=np.float32)
    input_b[:, :N_actual] = (np.random.rand(K_FULL, N_actual) * 4).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, M_actual, num_samples),
                np.random.randint(0, N_actual, num_samples),
            ]
        )

        # Add deterministic boundary-tile samples to catch padding errors.
        boundary_m = list(
            set(
                [
                    min(M_actual - 1, m)
                    for m in [M_actual - 1, M_actual - TILE_M + 1, 0]
                    if m >= 0
                ]
            )
        )
        boundary_n = list(
            set(
                [
                    min(N_actual - 1, n)
                    for n in [N_actual - 1, N_actual - TILE_N + 1, 0]
                    if n >= 0
                ]
            )
        )
        boundary_indices = np.array([[m, n] for m in boundary_m for n in boundary_n]).T
        sampled_indices = np.hstack([sampled_indices, boundary_indices])

        # Golden: truncate f32 inputs to bf16 (matching hardware truncf_op),
        # then compute dot product with f32 accumulation.
        input_a_bf16 = input_a.astype(bfloat16)
        input_b_bf16 = input_b.astype(bfloat16)
        sampled_values = np.array(
            [
                np.sum(
                    input_a_bf16[:, i].astype(np.float32)
                    * input_b_bf16[:, j].astype(np.float32),
                    dtype=np.float32,
                )
                for i, j in zip(*sampled_indices)
            ],
            dtype=np.float32,
        )

        sampled_data = {
            "shape": (M_padded, N_padded),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="matmul_padding_kernel",
            bf16_emulation=True,
            debug_ir=True,
        )
        exit(
            runner.run_test(
                air_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=0.1,
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            bf16_emulation=True,
        )
        module_function = backend.compile(air_module)
        backend.unload()
