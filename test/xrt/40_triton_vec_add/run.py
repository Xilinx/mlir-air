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

# Dtype configuration table:
#   mlir_type: MLIR type string used in the IR
#   np_type: numpy dtype for host data
#   add_op: arith add operation name
#   pad_val: padding value literal in MLIR
#   default_vector_size: default vector lane count (i8 uses 32 due to backend limitation)
#   rtol: relative tolerance for output comparison
DTYPE_CONFIG = {
    "bf16": {
        "mlir_type": "bf16",
        "np_type": bfloat16,
        "add_op": "arith.addf",
        "pad_val": "0.0 : bf16",
        "default_vector_size": 16,
        "rtol": 1e-2,
    },
    "f32": {
        "mlir_type": "f32",
        "np_type": np.float32,
        "add_op": "arith.addf",
        "pad_val": "0.0 : f32",
        "default_vector_size": 16,
        "rtol": 5e-2,
    },
    "i8": {
        "mlir_type": "i8",
        "np_type": np.int8,
        "add_op": "arith.addi",
        "pad_val": "0 : i8",
        "default_vector_size": 32,
        "rtol": 0,
    },
    "i16": {
        "mlir_type": "i16",
        "np_type": np.int16,
        "add_op": "arith.addi",
        "pad_val": "0 : i16",
        "default_vector_size": 32,
        "rtol": 0,
    },
}

parser = argparse.ArgumentParser(
    prog="run.py",
    description="Builds, runs, and tests the vecadd example",
)
parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform.mlir",
    help="Transform script path",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=list(DTYPE_CONFIG.keys()),
    default="bf16",
    help="Element data type (default: bf16)",
)
parser.add_argument(
    "--vector-size",
    type=int,
    dest="vector_size",
    default=None,
    help="Vector size for SIMD operations (default: auto based on dtype)",
)
parser.add_argument(
    "--num-tiles",
    type=int,
    dest="num_tiles",
    default=4,
    help="Number of AIE compute tiles (herd size). NPU1 has 4 columns, NPU2 has 8 (default: 4).",
)
parser.add_argument(
    "--bf16-emulation",
    dest="bf16_emulation",
    default=False,
    action="store_true",
    help="Use f32 input data type and emulate f32 vector arithmetic using bf16 operations.",
)
args = parser.parse_args()

# --bf16-emulation is shorthand for --dtype f32 with bf16_emulation enabled
if args.bf16_emulation:
    args.dtype = "f32"

cfg = DTYPE_CONFIG[args.dtype]
dtype_str = cfg["mlir_type"]
input_type = cfg["np_type"]
output_type = cfg["np_type"]
add_op = cfg["add_op"]
pad_val = cfg["pad_val"]
vector_size = (
    args.vector_size if args.vector_size is not None else cfg["default_vector_size"]
)
rtol = cfg["rtol"]
num_tiles = args.num_tiles
herd_tile_size = 256 // num_tiles

# bf16_emulation only applies to f32 dtype
bf16_emulation = args.bf16_emulation and args.dtype == "f32"

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = f"""
    #map = affine_map<(d0, d1) -> (d0, d1)>
    module {{
      func.func @vecadd(%arg0: memref<*x{dtype_str}> {{tt.divisibility = 16 : i32}}, %arg1: memref<*x{dtype_str}> {{tt.divisibility = 16 : i32}}, %arg2: memref<*x{dtype_str}> {{tt.divisibility = 16 : i32}}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {{
        %c256_i32 = arith.constant 256 : i32
        %0 = arith.muli %arg6, %c256_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256, 1], strides: [1, 1] : memref<*x{dtype_str}> to memref<256x1x{dtype_str}, strided<[1, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x1x{dtype_str}>
        memref.copy %reinterpret_cast, %alloc : memref<256x1x{dtype_str}, strided<[1, 1], offset: ?>> to memref<256x1x{dtype_str}>
        %2 = bufferization.to_tensor %alloc restrict writable : memref<256x1x{dtype_str}> to tensor<256x1x{dtype_str}>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [256, 1], strides: [1, 1] : memref<*x{dtype_str}> to memref<256x1x{dtype_str}, strided<[1, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<256x1x{dtype_str}>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<256x1x{dtype_str}, strided<[1, 1], offset: ?>> to memref<256x1x{dtype_str}>
        %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<256x1x{dtype_str}> to tensor<256x1x{dtype_str}>
        %4 = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}} ins(%2, %3 : tensor<256x1x{dtype_str}>, tensor<256x1x{dtype_str}>) outs(%2 : tensor<256x1x{dtype_str}>) {{
        ^bb0(%in: {dtype_str}, %in_3: {dtype_str}, %out: {dtype_str}):
          %5 = {add_op} %in, %in_3 : {dtype_str}
          linalg.yield %5 : {dtype_str}
        }} -> tensor<256x1x{dtype_str}>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [256, 1], strides: [1, 1] : memref<*x{dtype_str}> to memref<256x1x{dtype_str}, strided<[1, 1], offset: ?>>
        bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<256x1x{dtype_str}>, memref<256x1x{dtype_str}, strided<[1, 1], offset: ?>>) -> ()
        return
      }}
    }}
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
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    # Load the MLIR transform IR from an external file
    with open(args.transform_script, "r") as f:
        transform_ir_string = f.read()
    transform_ir_string = transform_ir_string.replace("@PAD_VAL@", pad_val)
    transform_ir_string = transform_ir_string.replace("@VECTOR_SIZE@", str(vector_size))
    transform_ir_string = transform_ir_string.replace(
        "@HERD_TILE_SIZE@", str(herd_tile_size)
    )
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, air_module)

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################
    M, N, K = 1024, 1, 1
    input_size = (M, N, K)
    tile_size = (256, 1, K)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
                "func.func(air-fuse-alloc-dealloc)",
                "canonicalize",
                "cse",
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

    if np.issubdtype(input_type, np.integer):
        iinfo = np.iinfo(input_type)
        half_max = iinfo.max // 2
        A = np.random.randint(0, half_max, size=(M,), dtype=input_type)
        B = np.random.randint(0, half_max, size=(M,), dtype=input_type)
    else:
        A = np.random.rand(M).astype(input_type)
        B = np.random.rand(M).astype(input_type)
    C = np.add(A, B).astype(output_type)

    ###### Compile and test
    runner = XRTRunner(
        omit_while_true_loop=False,
        use_lock_race_condition_fix=True,
        runtime_loop_tiling_sizes=[4, 4],
        bf16_emulation=bf16_emulation,
    )
    exit(
        runner.run_test(
            air_module,
            inputs=[A, B],
            expected_outputs=[C],
            rtol=rtol,
        )
    )
