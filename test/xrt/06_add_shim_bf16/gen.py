# gen.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.passmanager import *
from air.dialects import func, linalg
from air.dialects.air import module_builder
from air.dialects.linalg.opdsl.lang import *
from air.compiler.util import run_transform
import aie.compiler.aiecc.main as aiecc


@module_builder
def generate_add_module(shape):
    dtype = BF16Type.get()

    @func.FuncOp.from_py_func(
        MemRefType.get(shape, dtype),
        MemRefType.get(shape, dtype),
        MemRefType.get(shape, F32Type.get()),
    )
    def mul(lhs, rhs, out):
        linalg.elemwise_binary(
            lhs, rhs, outs=[out], fun=BinaryFn.add, cast=TypeFn.cast_unsigned
        )
        return


transform_ir_string = """
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_copy : benefit(1) {
    %args = pdl.operands
    %results = pdl.types
    %op = pdl.operation "memref.copy"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    pdl.rewrite %op with "transform.dialect"
    }
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
    %l0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %l1, %herd_tile_loops = transform.air.linalg_tile %l0 [0,128]
    %l3, %inner_tile_loops:2 = transform.air.linalg_tile %l1 [32,32]
    transform.air.linalg_promote %l3 {"operands_to_promote"=[0,1,2], "memory_space"="L1"}
    %herd = transform.air.par_to_herd %herd_tile_loops
    %copies = transform.pdl_match @match_copy in %arg0 : (!pdl.operation) -> !pdl.operation
    %h = transform.air.copy_to_dma %copies
    }
}
"""

module = generate_add_module([128, 128])
context = module.context

pm = PassManager.parse(
    "builtin.module(func.func(linalg-generalize-named-ops))", context=context
)
pm.run(module.operation)
transform_ir = Module.parse(transform_ir_string, context=context)
run_transform(transform_ir, module)

pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=context)
pm.run(module.operation)

print("\nTiled AIR Module:\n\n", module)
# with open("add.air.mlir", "w") as f:
#     f.write(str(module))

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-lower-herd-parallel)",
            # "air-dependency",
            # "air-dependency-schedule-opt",
            "air-dma-to-channel",
            "canonicalize",
            "cse",
            "func.func(air-opt-memtile-dma-bds{device=npu1_4col})",
            "func.func(convert-linalg-to-loops)",
            "func.func(air-renumber-dma)",
        ]
    )
    + ")"
)
pm = PassManager.parse(pipeline, context=context)
pm.run(module.operation)

# print("\nAIE Module:\n\n", module)
# with open("add.chan.mlir", "w") as f:
#     f.write(str(module))

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-to-aie{emit-while-loop=true device=npu1_4col row-offset=2 col-offset=0 use-objectfifo=false}",
            "func.func(air-opt-shim-dma-bds{device=npu1_4col})",
            "air-to-std",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)
pm = PassManager.parse(pipeline, context=context)
pm.run(module.operation)

# print("\nAIE Module:\n\n", module)
# with open("add.aieairrt.mlir", "w") as f:
#     f.write(str(module))

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "airrt-to-npu",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)
pm = PassManager.parse(pipeline, context=context)
pm.run(module.operation)

# print("\nAIE Module:\n\n", module)
# with open("add.aienpu.mlir", "w") as f:
#     f.write(str(module))

aiecc_options = [
    "--no-aiesim",
    "--no-xchesscc",
    "--no-xbridge",
    "--aie-generate-cdo",
    "--aie-generate-npu",
    "--no-compile-host",
    "--npu-insts-name=insts.txt",
    "--xclbin-name=add.xclbin",
    "aie.mlir",
]
aiecc.run(module, aiecc_options)
