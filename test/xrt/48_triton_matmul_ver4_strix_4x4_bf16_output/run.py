# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
    help="Transform script path",
)
parser.add_argument(
    "--use-cpp-pipeline",
    action="store_true",
    help="Replace the legacy transform script with the C++ matmul codegen "
    "pipeline (M5 — Triton-XDNA single-pack bf16-out flow). See "
    "MATMUL_CODEGEN_PIPELINE_PLAN.md.",
)
parser.add_argument(
    "--profile-iters",
    type=int,
    default=0,
    help="If >0, also benchmark on HW for this many iters (after correctness).",
)
args = parser.parse_args()

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<(d0, d1) -> (d0, d1)>
    module {
      func.func @npu_mm_exact(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
        %cst = arith.constant 0.000000e+00 : f32
        %c256 = arith.constant 256 : index
        %c256_i32 = arith.constant 256 : i32
        %0 = arith.muli %arg6, %c256_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg7, %c256_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %1, %c256 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [256, 256], strides: [256, 1] : memref<*xbf16> to memref<256x256xbf16, strided<[256, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x256xbf16>
        memref.copy %reinterpret_cast, %alloc : memref<256x256xbf16, strided<[256, 1], offset: ?>> to memref<256x256xbf16>
        %5 = bufferization.to_tensor %alloc restrict writable : memref<256x256xbf16> to tensor<256x256xbf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [256, 256], strides: [256, 1] : memref<*xbf16> to memref<256x256xbf16, strided<[256, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<256x256xbf16>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<256x256xbf16, strided<[256, 1], offset: ?>> to memref<256x256xbf16>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<256x256xbf16> to tensor<256x256xbf16>
        %7 = tensor.empty() : tensor<256x256xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %9 = linalg.matmul ins(%5, %6 : tensor<256x256xbf16>, tensor<256x256xbf16>) outs(%8 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %10 = arith.addi %4, %3 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [256, 256], strides: [256, 1] : memref<*xbf16> to memref<256x256xbf16, strided<[256, 1], offset: ?>>
        %11 = tensor.empty() : tensor<256x256xbf16>
        %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<256x256xf32>) outs(%11 : tensor<256x256xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %13 = arith.truncf %in : f32 to bf16
          linalg.yield %13 : bf16
        } -> tensor<256x256xbf16>
        bufferization.materialize_in_destination %12 in writable %reinterpret_cast_2 : (tensor<256x256xbf16>, memref<256x256xbf16, strided<[256, 1], offset: ?>>) -> ()
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
        # Drive Triton-XDNA bf16-out matmul codegen via the C++ orchestrator.
        # Single-pack-level flow: one L2 pack (orchestrator auto-bufferizes
        # its output to L1 since l1-pack-sizes is empty). Per-launch-tile
        # shape is 256x256x256.
        phases = [
            "air-matmul-codegen{"
            # Phase C: bufferize L2 acc + pre-steps for bf16-out flow.
            "bufferize-output-l2=true fuse-output-truncf-first=true "
            "tile-l3-to-l2-copies=true k-l2-tile=64 "
            # Phase B: single-pack L2 pack.
            "l2-pack-sizes=8,8,8 "
            "l2-lhs-outer-perm=1,0 l2-lhs-inner-perm=0,1 "
            "l2-rhs-outer-perm=1,0 l2-rhs-inner-perm=1,0 "
            "l2-acc-outer-perm=1,0 l2-acc-inner-perm=0,1 "
            # Phase E: K-tile factor=8 (single-pack so this is the only K-tile).
            "outer-k-tile-factor=8 outer-k-iter-index=2 "
            # Phase H: per-core tile.
            "core-tile=8,8,0 "
            # Phase K: prologue/epilogue.
            "prologue-tile=8,8 epilogue-tile=64,64 fill-iter-perm=1,0,2,3 "
            # Phase L: upstream one-shot-bufferize.
            "one-shot-bufferize=true "
            # Phase M: tile-for-vectorize.
            "post-bufferize-cleanup-first=true "
            "matmul-vec-tile=2,2,1,0,0,0 "
            "matmul-unroll-vec-tile=1,1,0,0,0,0 "
            "matmul-unroll-factor=2 fill-vec-tile=1,1,0,0 "
            # Phase N: vec-prep deferred to second invocation (after herd).
            "do-vec-prep=false" "}",
            "func.func(scf-forall-to-parallel)",
            "air-par-to-herd",
            "func.func(air-herd-vectorize)",
            "func.func(canonicalize,cse,fold-memref-alias-ops)",
            # Second orchestrator invocation: vec-prep only.
            "air-matmul-codegen{"
            "do-vec-prep=true "
            "vec-prep-cast1-target-element-type=f32 "
            "vec-prep-cast1-input-indices=2 "
            "vec-prep-cast1-output-indices=0 "
            "vec-prep-hoist-cast-pairs=true"
            "}",
            "func.func(canonicalize,cse,fold-memref-alias-ops,"
            "air-fold-unit-extent-dims)",
        ]
        cpp_pipeline = "builtin.module(" + ",".join(phases) + ")"
        pm = air.passmanager.PassManager.parse(cpp_pipeline)
        pm.run(air_module.operation)
    else:
        # Load the MLIR transform IR from an external file
        with open(args.transform_script, "r") as f:
            transform_ir_string = f.read()
        transform_ir = Module.parse(transform_ir_string)
        run_transform(transform_ir, air_module)

    ################################################
    ## Binding scf.parallel to air hierarchies
    ################################################
    M, N, K = 256, 256, 256
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

    import os

    if os.environ.get("AIR_DUMP_FINAL_IR"):
        with open(os.environ["AIR_DUMP_FINAL_IR"], "w") as f:
            f.write(str(air_module))

    ###############################################
    # Run compile and load
    ###############################################

    input_type = bfloat16
    output_type = bfloat16
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
        rtol=1e-1,
    )
    if args.profile_iters > 0 and rc == 0:
        runner.benchmark(
            air_module,
            inputs=[A, B],
            output_shapes_dtypes=[(C.shape, C.dtype)],
            iters=args.profile_iters,
            label=("cpp" if args.use_cpp_pipeline else "legacy"),
        )
    exit(rc)
