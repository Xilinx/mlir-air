# run.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.dialects import linalg, arith, func, tensor, memref, bufferization
from air.dialects.air import module_builder
from air.dialects.linalg.opdsl.lang import *
from air.compiler.util import run_transform
import argparse

from air.backend.xrt import XRTBackend
from air.ir import *
import air.passmanager

from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

from ml_dtypes import bfloat16
import numpy as np

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
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform.mlir",
    help="Transform script path",
)
parser.add_argument(
    "--M",
    type=int,
    default=512,
    help="Matrix dimension M (rows of A, rows of C)",
)
parser.add_argument(
    "--K",
    type=int,
    default=1024,
    help="Matrix dimension K (cols of A, rows of B)",
)
parser.add_argument(
    "--N",
    type=int,
    default=512,
    help="Matrix dimension N (cols of B, cols of C)",
)
args = parser.parse_args()


@linalg_structured_op
def matmul(
    A=TensorDef(TV.T, S.M, S.K),
    B=TensorDef(TV.T, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True),
):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


@module_builder
def matmul_on_tensors(m, k, n):
    dtype_in = BF16Type.get()
    dtype_out = F32Type.get()

    @func.FuncOp.from_py_func(
        MemRefType.get((m, k), dtype_in), MemRefType.get((k, n), dtype_in)
    )
    def forward(lhs, rhs):
        lhs_tensor = bufferization.to_tensor(
            buffer=lhs,
            result=RankedTensorType.get((m, k), dtype_in),
            restrict=True,
            writable=True,
        )
        rhs_tensor = bufferization.to_tensor(
            buffer=rhs,
            result=RankedTensorType.get((k, n), dtype_in),
            restrict=True,
            writable=True,
        )
        out = tensor.EmptyOp((m, n), dtype_out).result
        zero = arith.ConstantOp(dtype_out, 0.0)
        zero_fill = linalg.fill(zero, outs=[out])
        matmul_tensor = matmul(lhs_tensor, rhs_tensor, outs=[zero_fill])
        result_memref = bufferization.to_buffer(
            tensor=matmul_tensor, buffer=MemRefType.get((m, n), dtype_out)
        )
        return result_memref


air_module = matmul_on_tensors(args.M, args.K, args.N)
context = air_module.context

with open("air_input.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Tiling
################################################

# Load the MLIR transform IR from an external file
with open(args.transform_script, "r") as f:
    transform_ir_string = f.read()
transform_ir = Module.parse(transform_ir_string, context=context)
run_transform(transform_ir, air_module)

with open("air_tiled.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Binding scf.paralell to air hierarchies
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-par-to-herd{depth=-1}",
            "air-par-to-launch{depth=-1 has-air-segment=true}",
            "func.func(air-herd-vectorize)",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

###############################################
# Compile, run and compare results
###############################################


# Use parsed arguments for matrix dimensions.
M = args.M
K = args.K
N = args.N
input_a = np.arange(0, M * K, dtype=bfloat16).reshape(M, K)
input_b = np.arange(0, K * N, dtype=bfloat16).reshape(K, N)
if args.compile_mode == "compile-and-run":
    # Stochastically sample num_sample results, and pass to XRTRunner backend for verification.
    num_samples = 100
    sampled_indices = np.vstack(
        [
            np.random.randint(0, args.M, num_samples),  # i indices
            np.random.randint(0, args.N, num_samples),  # j indices
        ]
    )

    # Compute reference results for sampled indices
    sampled_values = np.array(
        [
            np.sum(
                (input_a[i, :].astype(np.float32) * input_b[:, j].astype(np.float32)),
                dtype=np.float32,
            )
            for i, j in zip(*sampled_indices)
        ],
        dtype=np.float32,
    )

    # Store as a dictionary
    sampled_data = {
        "shape": (args.M, args.N),
        "indices": sampled_indices,
        "values": sampled_values,
    }
    runner = XRTRunner(verbose=args.verbose, omit_while_true_loop=False)
    exit(
        runner.run_test(
            air_module,
            inputs=[input_a, input_b],
            stochastic_expected_outputs=[sampled_data],
            rtol=1e-1,
        )
    )

elif args.compile_mode == "compile-only":
    ###### Compile only
    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
    )
    module_function = backend.compile(air_module)

    backend.unload()
