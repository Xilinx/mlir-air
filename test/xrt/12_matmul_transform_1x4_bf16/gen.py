# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air
import air.compiler.util
from air.dialects import linalg, arith, func
from air.dialects.linalg.opdsl.lang import *
from air.ir import *
import air.passmanager
from air._mlir_libs._air import run_transform
from air.dialects.air import module_builder

import argparse


@linalg_structured_op
def matmul_poly(
    A=TensorDef(TV.T1, S.M, S.K),
    B=TensorDef(TV.T2, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


@module_builder
def matmul_on_tensors(m, n, k):
    dtype = BF16Type.get()

    @func.FuncOp.from_py_func(
        RankedTensorType.get((m, k), dtype),
        RankedTensorType.get((k, n), dtype),
        RankedTensorType.get((m, n), F32Type.get()),
    )
    def forward(lhs, rhs, out):
        zero = arith.ConstantOp(F32Type.get(), 0.0)
        zero_fill = linalg.fill(zero, outs=[out])
        o = matmul_poly(lhs, rhs, outs=[zero_fill])
        return o


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", required=True, dest="transform_filename", help="transform script filename"
)
opts = parser.parse_args()

air_module = matmul_on_tensors(128, 128, 256)
context = air_module.context

################################################
## Tiling
################################################

with open(opts.transform_filename, "r") as f:
    transform_ir_string = f.read()

transform_ir = Module.parse(transform_ir_string, context=context)
run_transform(transform_ir, air_module)

with open("air_transform.mlir", "w") as f:
    f.write(str(air_module))

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "one-shot-bufferize{bufferize-function-boundaries=1 unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map}",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)

pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

transform_ir_string = """
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

        %fill_0 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %generic = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul_0 = transform.structured.specialize %generic : (!pdl.operation) -> !pdl.operation
        %ps = transform.merge_handles %fill_0, %matmul_0 : !pdl.operation
        transform.air.linalg_promote %ps {"operands_to_promote"=[1,4], "group_size"=2, "memory_space"="L1"}

        %matmul_1, %loops:3 = transform.air.linalg_tile %matmul_0 [16, 16, 16]

        transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[0,1], "memory_space"="L1"}

        %f = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %f : !pdl.operation
    }
}
"""
transform_ir = Module.parse(transform_ir_string, context=context)
run_transform(transform_ir, air_module)

with open("air_tiled.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Binding parallel loops to air hierarchies
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-copy-to-dma",
            "air-linalg-to-func{link-with=kernel.o}",
            "air-par-to-herd{depth=1}",
            "air-par-to-launch{has-air-segment=1}",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)

pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

with open("air_sync.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Extract event dependency and optimize schedule
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-dependency",
            "air-dependency-schedule-opt",
            "air-specialize-dma-broadcast",
            "air-dma-to-channel",
            "canonicalize",
            "cse",
            "air-dependency-canonicalize",
            "canonicalize",
            "cse",
            "func.func(air-loop-fusion)",
            "air-label-scf-for-to-ping-pong",
            "air-ping-pong-transform{keep-memref-dealloc=true}",
            "canonicalize",
            "cse",
            "air-specialize-channel-wrap-and-stride",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aircc_input.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Place herd to segment
################################################

air_async_module = Module.parse(str(air_module), context=context)
pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-collapse-herd)",
            "canonicalize",
            "cse",
            "air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor=0}",
            "canonicalize",
            "cse",
            "func.func(air-renumber-dma)",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("air_placed.mlir", "w") as f:
    f.write(str(air_module))

################################################
## MLIR-AIR to MLIR-AIE
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-to-aie{row-offset=2 col-offset=0 device=npu1_4col emit-while-loop=true}",
            "canonicalize",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aircc_decomp_aiecc.mlir", "w") as f:
    f.write(str(air_module))

################################################
## MLIR-AIR runtime lowering
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-to-std",
            "symbol-dce",
            "airrt-to-npu",
            "canonicalize",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aie.mlir", "w") as f:
    f.write(str(air_module))

import aie.compiler.aiecc.main as aiecc

aiecc_options = [
    "--no-aiesim",
    "--xchesscc",
    "--xbridge",
    "--aie-generate-cdo",
    "--aie-generate-npu",
    "--no-compile-host",
    "--npu-insts-name=insts.txt",
    "--xclbin-name=aie.xclbin",
    "aie.mlir",
]
aiecc.run(air_module, aiecc_options)
