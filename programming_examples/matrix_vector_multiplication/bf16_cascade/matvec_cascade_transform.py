# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Transform-dialect-based cascade bf16 matrix-vector multiplication.
#
# C[M] = A[M,K] @ B[K]   (bf16 in/out)
#
# Uses linalg.matvec on tensors, tiled and cascade-split via transform.mlir.
# After vectorization, vector.contract is decomposed via
# transform.apply_patterns.vector.lower_contraction{dot} into
# arith.mulf + vector.reduction<add> which maps to aie::mac + aie::reduce_add.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

from air.dialects import linalg, arith, func, tensor, memref, bufferization
from air.dialects.air import module_builder
from air.dialects.linalg.opdsl.lang import *
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend


@module_builder
def matvec_on_tensors(m, k):
    dtype = BF16Type.get()

    @func.FuncOp.from_py_func(
        MemRefType.get((m, k), dtype), MemRefType.get((k,), dtype)
    )
    def forward(A, B):
        A_tensor = bufferization.to_tensor(
            buffer=A,
            result=RankedTensorType.get((m, k), dtype),
            restrict=True,
            writable=True,
        )
        B_tensor = bufferization.to_tensor(
            buffer=B,
            result=RankedTensorType.get((k,), dtype),
            restrict=True,
            writable=True,
        )
        out = tensor.EmptyOp((m,), dtype).result
        zero = arith.ConstantOp(dtype, 0.0)
        zero_fill = linalg.fill(zero, outs=[out])
        result = linalg.matvec(A_tensor, B_tensor, outs=[zero_fill])
        result_memref = bufferization.to_buffer(
            tensor=result, buffer=MemRefType.get((m,), dtype)
        )
        return result_memref


# Transform script to lower vector.contract into arith.mulf + vector.reduction
LOWER_CONTRACT_SCRIPT = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "dot"
    } : !transform.any_op
    transform.yield
  }
}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="matvec_cascade_transform.py",
        description="Transform-dialect-based cascade bf16 matvec",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument(
        "--transform-script",
        type=str,
        dest="transform_script",
        default="transform.mlir",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-only"],
        dest="compile_mode",
        default="compile-and-run",
    )
    args = parser.parse_args()

    air_module = matvec_on_tensors(args.m, args.k)
    context = air_module.context

    if args.print_module_only:
        print(air_module)
        exit(0)

    # === Phase 1: Apply transform script for tiling ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transform_path = (
        args.transform_script
        if os.path.isabs(args.transform_script)
        else os.path.join(script_dir, args.transform_script)
    )
    with open(transform_path, "r") as f:
        transform_ir_string = f.read()
    transform_ir = Module.parse(transform_ir_string, context=context)
    run_transform(transform_ir, air_module)

    with open("air_tiled.mlir", "w") as f:
        f.write(str(air_module))

    # === Phase 2: Convert to AIR hierarchy + vectorize ===
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "buffer-results-to-out-params{hoist-static-allocs=true modify-public-functions=true}",
                "air-copy-to-dma",
                "air-par-to-herd{depth=-1}",
                "air-par-to-herd{depth=-1}",
                "air-par-to-launch{depth=-1 has-air-segment=true}",
                "func.func(air-fuse-nested-herd)",
                # Note: air-herd-vectorize is NOT used. The linalg.matvec stays
                # as scalar code and Peano handles vectorization at LLVM level.
                # Vectorizing here creates per-element subview writes that break
                # the cascade dependency chain.
                "canonicalize",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)

    with open("air_herd.mlir", "w") as f:
        f.write(str(air_module))

    # === Phase 4: Compile and run ===
    M = args.m
    K = args.k

    if args.compile_mode == "compile-and-run":
        np.random.seed(42)
        input_a = (np.random.randn(M, K) * 4).astype(bfloat16)
        input_b = (np.random.randn(K) * 4).astype(bfloat16)
        output_c = np.dot(
            input_a.astype(np.float32), input_b.astype(np.float32)
        ).astype(bfloat16)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[4, 4],
            output_format=args.output_format,
        )
        exit(
            runner.run_test(
                air_module,
                inputs=[input_a, input_b],
                expected_outputs=[output_c],
                rtol=0.04,
                atol=1e-3,
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[4, 4],
            output_format=args.output_format,
        )
        module_function = backend.compile(air_module)
        backend.unload()
