# ./python/test/compiler/run_transform.py -*- Python -*-

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s
from air.ir import *
from air.dialects import air as airdialect
from air.dialects import arith, func, linalg
from air.compiler.util import run_transform


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: gemm_module
# CHECK: scf.parallel
# CHECK: scf.for
@run
def gemm_module():
    with Context() as ctx, Location.unknown():
        M = 256
        N = 256
        K = 256
        dtype = F32Type.get()
        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                MemRefType.get((M, K), dtype),
                MemRefType.get((K, N), dtype),
                MemRefType.get((M, N), dtype),
            )
            def matmul(lhs, rhs, out):
                zero = arith.ConstantOp(dtype, FloatAttr.get(dtype, 0))
                linalg.fill(zero, outs=[out])
                linalg.matmul(lhs, rhs, outs=[out])
                return out

        transform_ir_string = """
        transform.with_pdl_patterns {
        ^bb0(%arg0: !pdl.operation):
            transform.sequence %arg0 : !pdl.operation failures(propagate) {
            ^bb1(%arg1: !pdl.operation):
                %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation
                %matmul_1, %loops:3 = transform.air.linalg_tile %matmul [64, 64, 64]
            }
        }
        """
        transform_ir = Module.parse(transform_ir_string)
        run_transform(transform_ir, module)
        print(module)
