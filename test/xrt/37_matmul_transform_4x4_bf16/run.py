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

np.random.seed(42)

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
    "--use-cpp-pipeline",
    action="store_true",
    help="Replace the legacy transform script with the C++ matmul codegen "
    "pipeline (M4 two-pack-level flow). See MATMUL_CODEGEN_PIPELINE_PLAN.md.",
)
parser.add_argument(
    "--profile-iters",
    type=int,
    default=0,
    help="If >0, also benchmark on HW for this many iters (after correctness).",
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
parser.add_argument(
    "--output-format",
    type=str,
    dest="output_format",
    default="xclbin",
    choices=["elf", "xclbin"],
    help="Output format: 'xclbin' (default) or 'elf'",
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

if args.use_cpp_pipeline:
    # Two-pack-level matmul codegen via the single C++ orchestrator pass.
    # Hand-tuned options match the legacy transform_aie2p.mlir values for
    # M=512/N=512/K=1024.
    pipeline = (
        "builtin.module(air-matmul-codegen{"
        # Phase A: outer launch tile.
        "launch-tile=256,256 "
        # Phase B: L2 pack.
        "l2-pack-sizes=64,64,64 "
        "l2-lhs-outer-perm=0,1 l2-lhs-inner-perm=0,1 "
        "l2-rhs-outer-perm=1,0 l2-rhs-inner-perm=1,0 "
        "l2-acc-outer-perm=0,1 l2-acc-inner-perm=0,1 "
        # Phase C: bufferize L2 accumulator init.
        "bufferize-output-l2=true "
        # Phase D: L1 pack on the L2-packed generic; bufferize pack_c to L1.
        "l1-pack-sizes=0,0,0,8,8,8 "
        "l1-lhs-outer-perm=0,1,3,2 "
        "l1-rhs-outer-perm=0,1,3,2 l1-rhs-inner-perm=1,0 "
        "l1-acc-outer-perm=0,1,3,2 "
        # Phase E: outer K-tile (factor=1 over K_L2/64 = 16 chunks).
        # Chain-fuses both L1 and L2 packs into the K-loop; orchestrator
        # auto-bufferizes the L2 packs into L2 (Phase F).
        "outer-k-tile-factor=1 outer-k-iter-index=2 "
        # Phase H: per-core tile (4x4 forall).
        "core-tile=1,1,0,0,0,0,0,0,0 "
        # Phase I: inner K-tile (factor=8 over k_L2/8 = 8 chunks).
        # Orchestrator auto-bufferizes L1 input packs (Phase J).
        "inner-k-tile-factor=8 inner-k-iter-index=5 "
        # Phase K: prologue/epilogue. hoist-static-alloc-first hoists the L1
        # acc alloc out of the K-reduction loop (K-peel flow).
        "prologue-tile=1,1 epilogue-tile=1,1 hoist-static-alloc-first=true "
        # Phase L: upstream one-shot-bufferize.
        "one-shot-bufferize=true "
        # Phase M: tile-for-vectorize (9-iter matmul tiled by 1; fill 4-iter).
        # post-bufferize-cleanup-first removes uninitialized copies and
        # sibling-fuses pingpong loops.
        "post-bufferize-cleanup-first=true "
        "matmul-vec-tile=1,1,1,1,1,1,0,0,0 "
        "matmul-unroll-vec-tile=0,0,0,0,0,0,0,0,0 "
        "matmul-unroll-factor=1 fill-vec-tile=1,1,1,1 "
        # Phase N: vec-prep is gated off — this test does not need any of
        # the vec-prep sub-steps (no vector-cast emulation, no cast-pair
        # hoist; the simple flatten/hoist passes are not used here).
        "do-vec-prep=false"
        "})"
    )
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)
else:
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
            "air-copy-to-dma",
            "air-par-to-herd{depth=-1}",
            "air-par-to-launch{depth=-1 has-air-segment=true}",
            "func.func(air-fuse-alloc-dealloc)",
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
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[4, 4],
        output_format=args.output_format,
        instance_name="forward",
    )
    rc = runner.run_test(
        air_module,
        inputs=[input_a, input_b],
        stochastic_expected_outputs=[sampled_data],
        rtol=1e-1,
    )
    if args.profile_iters > 0 and rc == 0:
        runner.benchmark(
            air_module,
            inputs=[input_a, input_b],
            stochastic_expected_outputs=[sampled_data],
            iters=args.profile_iters,
            label=("cpp" if args.use_cpp_pipeline else "legacy"),
        )
    exit(rc)

elif args.compile_mode == "compile-only":
    ###### Compile only
    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="forward",
        runtime_loop_tiling_sizes=[4, 4],
    )
    module_function = backend.compile(air_module)

    backend.unload()
