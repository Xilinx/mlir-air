# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Multi-Launch Attention Test Driver
==================================
Tests: O = softmax(Q @ K^T) @ V

This test validates the multi-launch attention example by:
1. Generating random Q, K^T, V matrices
2. Computing expected output using numpy
3. Running on AIE hardware
4. Comparing results
"""

from air.backend.xrt import compile_air, get_air_runtime
from air.ir import *
from ml_dtypes import bfloat16
import numpy as np
import os
import aie.utils


def softmax(x, axis=-1):
    """Numerically stable softmax using numpy only."""
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Matrix dimensions
M, N, K = 256, 256, 256

with Context() as ctx, Location.unknown():

    # Read MLIR IR from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file_path = os.path.join(script_dir, "attention.mlir")
    with open(mlir_file_path, "r") as f:
        air_tiled_ir_string = f.read()

    air_module = Module.parse(air_tiled_ir_string)

    ###############################################
    # Generate test inputs and expected outputs
    ###############################################

    input_type = bfloat16
    output_type = bfloat16

    # Random inputs
    np.random.seed(42)
    Q = np.random.uniform(low=-512.0, high=1024.0, size=(M, K)).astype(input_type)
    K_T = np.random.uniform(low=-512.0, high=1024.0, size=(K, N)).astype(input_type)
    V = np.random.uniform(low=-512.0, high=1024.0, size=(K, N)).astype(input_type)

    # Compute expected output: O = softmax(Q @ K^T) @ V
    # Use bf16 inputs cast to float32 for matmul precision, then cast back
    # to bf16 at each stage to match hardware behavior
    # Step 1: S = Q @ K^T
    S = np.matmul(Q, K_T)
    # Step 2: P = softmax(S) (row-wise)
    P = softmax(S, axis=-1)
    # Step 3: O = P @ V
    O_expected = np.matmul(P, V).astype(output_type)

    # Allocate intermediate buffers (now function arguments)
    S_buffer = np.zeros((M, N), dtype=output_type)
    P_buffer = np.zeros((M, N), dtype=output_type)

    ###############################################
    # Run test
    ###############################################

    npu_kernel = compile_air(
        air_module,
        output_format="elf",
        instance_name="attention",
        omit_while_true_loop=False,
        verbose=False,
        runtime_loop_tiling_sizes=[4, 4],
    )
    runtime = get_air_runtime()
    io_args = [
        aie.utils.tensor(Q),
        aie.utils.tensor(K_T),
        aie.utils.tensor(V),
        aie.utils.tensor(S_buffer),
        aie.utils.tensor(P_buffer),
        aie.utils.tensor(np.zeros(O_expected.shape, O_expected.dtype)),
    ]
    exit(runtime.run_test(npu_kernel, io_args, refs={5: O_expected}, atol=2e3))
