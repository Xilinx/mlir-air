# gen.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.backend.xrt import XRTBackend
from air.ir import *
from air.passmanager import *
from air.dialects.air import module_builder
from air.compiler.util import run_transform
from air.dialects import func, linalg
from air.dialects.linalg.opdsl.lang import *


@module_builder
def generate_add_module(shape):
    dtype = BF16Type.get()

    @func.FuncOp.from_py_func(
        MemRefType.get(shape, dtype),
        MemRefType.get(shape, dtype),
        MemRefType.get(shape, dtype),
    )
    def add(lhs, rhs, out):
        linalg.elemwise_binary(
            lhs, rhs, outs=[out], fun=BinaryFn.add, cast=TypeFn.cast_unsigned
        )
        return

    # print ("\nlinalg Module:\n\n", module)


module = generate_add_module([128, 128])
context = module.context

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
    %name = transform.param.constant "add_bf16" -> !transform.any_param
    transform.annotate %l3 "library_call" = %name : !pdl.operation, !transform.any_param
    transform.air.linalg_promote %l3 {"operands_to_promote"=[0,1,2], "memory_space"="L1"}
    %herd = transform.air.par_to_herd %herd_tile_loops
    %library = transform.param.constant "kernel.o" -> !transform.any_param
    transform.annotate %herd "link_with" = %library : !pdl.operation, !transform.any_param
    %copies = transform.pdl_match @match_copy in %arg0 : (!pdl.operation) -> !pdl.operation
    %h = transform.air.copy_to_dma %copies
    }
}
"""

pm = PassManager.parse(
    "builtin.module(func.func(linalg-generalize-named-ops))", context=context
)
pm.run(module.operation)
transform_ir = Module.parse(transform_ir_string, context=context)
run_transform(transform_ir, module)

pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=context)
pm.run(module.operation)

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-lower-herd-parallel)",
        ]
    )
    + ")"
)
pm = PassManager.parse(pipeline, context=context)
pm.run(module.operation)

###############################################
# Run compile and load
###############################################

backend = XRTBackend(
    lower_linalg_to_func="kernel.o",
)
module_function = backend.compile_and_load(module)
