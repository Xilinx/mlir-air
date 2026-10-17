# ./python/test/compiler/run_transform.py -*- Python -*-

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s
from air.ir import *
from air.dialects import air as airdialect
from air.dialects import arith, func, linalg
from air.dialects.air import module_builder
from air.compiler.util import run_transform
from air.dialects.linalg.opdsl.lang import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


@linalg_structured_op
def matmul_mono(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True),
):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


# CHECK-LABEL: TEST: gemm_module
# CHECK: scf.parallel
# CHECK: scf.for
@run
def gemm_module():
    @module_builder
    def build_module():
        M = 256
        N = 256
        K = 256
        dtype = F32Type.get()

        @func.FuncOp.from_py_func(
            MemRefType.get((M, K), dtype),
            MemRefType.get((K, N), dtype),
            MemRefType.get((M, N), dtype),
        )
        def matmul(lhs, rhs, out):
            zero = arith.ConstantOp(dtype, FloatAttr.get(dtype, 0))
            linalg.fill(zero, outs=[out])
            matmul_mono(lhs, rhs, outs=[out])
            return out

    module = build_module()
    transform_ir_string = """
        module attributes {transform.with_named_sequence} {
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
            %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
            %matmul_1, %forall = transform.air.linalg_tile %matmul [64, 64, 0]
            %parallal = transform.loop.forall_to_parallel %forall  : (!transform.any_op) -> !transform.any_op
            %matmul_2 = transform.structured.match ops{["linalg.generic"]} in %parallal  : (!transform.any_op) -> !transform.any_op
            %matmul_3, %forall_1 = transform.air.linalg_tile %matmul_2 [0, 0, 64]
            %scffor = transform.loop.forall_to_for %forall_1  : (!transform.any_op) -> !transform.any_op
          transform.yield
        }
        }
    """
    transform_ir = Module.parse(transform_ir_string, context=module.context)
    run_transform(transform_ir, module)
    print(module)


# A failing transform raises, and the message carries the diagnostic.
# CHECK-LABEL: TEST: failing_transform
# CHECK: RuntimeError: transform failed:
# CHECK-SAME: expected to contain 2 payloads but it contains 1 payloads
@run
def failing_transform():
    module = Module.parse(
        """
        func.func @f(%a: memref<4xf32>) {
          return
        }
    """,
        context=Context(),
    )
    transform_ir = Module.parse(
        """
        module attributes {transform.with_named_sequence} {
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
            %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            %a, %b = transform.split_handle %f : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
            transform.yield
          }
        }
        """,
        context=module.context,
    )
    try:
        run_transform(transform_ir, module)
        print("no error")
    except RuntimeError as e:
        print(f"RuntimeError: {' '.join(str(e).split())}")
