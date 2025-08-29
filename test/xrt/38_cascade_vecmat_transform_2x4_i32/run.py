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
    "--K",
    type=int,
    default=512,
    help="Matrix dimension K (cols of A, rows of B)",
)
parser.add_argument(
    "--N",
    type=int,
    default=256,
    help="Matrix dimension N (cols of B, cols of C)",
)
args = parser.parse_args()


@linalg_structured_op
def vecmat(
    y=TensorDef(TV.T1, S.M),
    A=TensorDef(TV.T2, S.M, S.N),
    x=TensorDef(U, S.N, output=True),
):
    """Performs a vector-matrix multiplication.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.n, D.m)
    implements(ContractionOpInterface)
    x[D.n] += TypeFn.cast_signed(U, y[D.m]) * TypeFn.cast_signed(U, A[D.m, D.n])


@module_builder
def vecmat_on_tensors(k, n):
    dtype = IntegerType.get_signless(width=32)

    @func.FuncOp.from_py_func(
        MemRefType.get((k,), dtype), MemRefType.get((k, n), dtype)
    )
    def forward(lhs, rhs):
        lhs_tensor = bufferization.to_tensor(
            buffer=lhs,
            result=RankedTensorType.get((k,), dtype),
            restrict=True,
            writable=True,
        )
        rhs_tensor = bufferization.to_tensor(
            buffer=rhs,
            result=RankedTensorType.get((k, n), dtype),
            restrict=True,
            writable=True,
        )
        out = tensor.EmptyOp((n,), dtype).result
        zero = arith.ConstantOp(dtype, 0)
        zero_fill = linalg.fill(zero, outs=[out])
        vecmat_tensor = vecmat(lhs_tensor, rhs_tensor, outs=[zero_fill])
        result_memref = bufferization.to_buffer(
            tensor=vecmat_tensor, buffer=MemRefType.get((n,), dtype)
        )
        return result_memref


air_module = vecmat_on_tensors(args.K, args.N)
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
            "buffer-results-to-out-params{hoist-static-allocs=true}",
            "air-copy-to-dma",
            "air-par-to-herd{depth=-1}",
            "air-par-to-herd{depth=-1}",
            "air-par-to-launch{depth=-1 has-air-segment=true}",
            "func.func(air-fuse-nested-herd)",
            "canonicalize",
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
K = args.K
N = args.N

# Set random seed for reproducibility
np.random.seed(42)

# Use small positive integers to avoid overflow
# Max value chosen so that K * max_val^2 < 2^31
max_val = 1024  # Conservative estimate
input_a = np.random.randint(1, max_val + 1, size=(1, K), dtype=np.int32)
input_b = np.random.randint(1, max_val + 1, size=(K, N), dtype=np.int32)
if args.compile_mode == "compile-and-run":
    # Stochastically sample num_sample results, and pass to XRTRunner backend for verification.
    num_samples = 100
    sampled_indices = np.vstack(
        [
            np.random.randint(0, 1, num_samples),  # i indices
            np.random.randint(0, args.N, num_samples),  # j indices
        ]
    )

    # Compute reference results for sampled indices
    sampled_values = np.array(
        [
            np.sum(
                (input_a[i, :].astype(np.int32) * input_b[:, j].astype(np.int32)),
                dtype=np.int32,
            )
            for i, j in zip(*sampled_indices)
        ],
        dtype=np.int32,
    )
    print(sampled_values)

    # Store as a dictionary
    sampled_data = {
        "shape": (1, args.N),
        "indices": sampled_indices,
        "values": sampled_values,
    }
    runner = XRTRunner(verbose=args.verbose, omit_while_true_loop=False)
    exit(
        runner.run_test(
            air_module,
            inputs=[input_a, input_b],
            stochastic_expected_outputs=[sampled_data],
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
