# run.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
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
    description="Builds, runs, and tests the matmul example",
)
parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform.mlir",
    help="Transform script path (legacy path).",
)
parser.add_argument(
    "--use-cpp-pipeline",
    action="store_true",
    help="Replace the legacy transform script with the C++ matmul codegen "
    "orchestrator (air-matmul-codegen). Targets aie2 / NPU1 (mmul=4x4x8).",
)
args = parser.parse_args()

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    module {
      func.func @bare_matmul(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
        %cst = arith.constant 0.000000e+00 : f32
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        %c256_i32 = arith.constant 256 : i32
        %0 = arith.muli %arg6, %c256_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg7, %c256_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %1, %c512 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [256, 512], strides: [512, 1] : memref<*xbf16> to memref<256x512xbf16, strided<[512, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x512xbf16>
        memref.copy %reinterpret_cast, %alloc : memref<256x512xbf16, strided<[512, 1], offset: ?>> to memref<256x512xbf16>
        %5 = bufferization.to_tensor %alloc restrict writable : memref<256x512xbf16> to tensor<256x512xbf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [512, 256], strides: [1024, 1] : memref<*xbf16> to memref<512x256xbf16, strided<[1024, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<512x256xbf16>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<512x256xbf16, strided<[1024, 1], offset: ?>> to memref<512x256xbf16>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<512x256xbf16> to tensor<512x256xbf16>
        %7 = tensor.empty() : tensor<256x256xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %9 = linalg.matmul ins(%5, %6 : tensor<256x512xbf16>, tensor<512x256xbf16>) outs(%8 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %10 = arith.muli %1, %c1024 : index
        %11 = arith.addi %10, %3 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [256, 256], strides: [1024, 1] : memref<*xf32> to memref<256x256xf32, strided<[1024, 1], offset: ?>>
        bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<256x256xf32>, memref<256x256xf32, strided<[1024, 1], offset: ?>>) -> ()
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
                "air-override-memref-memory-space{scope=func memory-space=1}",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    if args.use_cpp_pipeline:
        # Single-pack-level NPU1 (aie2) flow via the C++ orchestrator.
        # mmul=[4,4,8]. Per-launch matmul is 256x256x512; orchestrator's
        # launch-tile=64,64 creates an outer scf.forall (4x4 herd) wrapping
        # an inner 64x64 matmul. No L3->L2 copy tiling, no fuse-truncf
        # (output is f32). No prologue/epilogue tiling (test 39's transform
        # script doesn't separate them).
        cpp_pipeline = (
            "builtin.module("
            "air-matmul-codegen{"
            # Phase A: launch-tile = 64x64 (the only parallel tile in this
            # flow). Becomes the outer scf.forall, mapped to a 4x4 herd.
            "launch-tile=64,64 "
            # Phase C: bufferize fill output to L2.
            "bufferize-output-l2=true "
            # Phase B: single-pack [4, 4, 8] (aie2 mmul).
            "l2-pack-sizes=4,4,8 "
            "l2-lhs-outer-perm=1,0 "
            "l2-rhs-outer-perm=1,0 l2-rhs-inner-perm=1,0 "
            "l2-acc-outer-perm=1,0 "
            # Phase E: K-tile factor=4 (matches transform's tile_using_for "
            # [0, 0, 4]).
            "outer-k-tile-factor=4 outer-k-iter-index=2 "
            # No core-tile (the launch-tile is the only parallel tile).
            # No inner K-tile, no prologue/epilogue.
            # Phase L: upstream one-shot-bufferize.
            "one-shot-bufferize=true "
            # Phase M: tile-for-vectorize at [1, 1, 1, 0, 0, 0]; no second-
            # level unroll.
            "matmul-vec-tile=1,1,1,0,0,0 "
            "matmul-unroll-factor=1 fill-vec-tile=1,1 "
            # Phase N: no vec-prep (test 39 doesn't run any vec-prep steps).
            "do-vec-prep=false"
            "}, "
            "func.func(scf-forall-to-parallel), "
            "air-par-to-herd, "
            "func.func(air-herd-vectorize), "
            "func.func(canonicalize,cse,fold-memref-alias-ops), "
            # Fold-only orchestrator pass for post-vectorize cleanup.
            "air-matmul-codegen{do-vec-prep=false}"
            ")"
        )
        pm = air.passmanager.PassManager.parse(cpp_pipeline)
        pm.run(air_module.operation)
    else:
        # Load the MLIR transform IR from an external file
        with open(args.transform_script, "r") as f:
            transform_ir_string = f.read()
        transform_ir = Module.parse(transform_ir_string)
        run_transform(transform_ir, air_module)

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################
    M, N, K = 2048, 1024, 512
    input_size = (M, N, K)
    tile_size = (256, 256, K)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
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
    # Run compile and load
    ###############################################

    input_type = bfloat16
    output_type = np.float32
    A = np.random.rand(M, K).astype(input_type)  # Shape [M, K]
    B = np.random.rand(K, N).astype(input_type)  # Shape [K, N]
    C = np.matmul(A, B).astype(output_type)  # Shape [M, N]

    ###### Compile and test
    runner = XRTRunner(
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[4, 4],
    )
    rc = runner.run_test(
        air_module,
        inputs=[A, B],
        expected_outputs=[C],
        rtol=1e-3,
    )
    exit(rc)
