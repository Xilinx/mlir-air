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
import air.dialects.linalg.opdsl.lang as linalg_lang


# elemwise_binary (with operand type cast) is deprecated from upstream. Definition is moved here as a custom linalg structured op.
@linalg_lang.linalg_structured_op
def elemwise_binary(
    lhs=linalg_lang.TensorDef(linalg_lang.TV.T1),
    rhs=linalg_lang.TensorDef(linalg_lang.TV.T2),
    O=linalg_lang.TensorDef(linalg_lang.U, output=True),
    fun=linalg_lang.BinaryFnAttrDef(default=linalg_lang.BinaryFn.add),
    cast=linalg_lang.TypeFnAttrDef(default=linalg_lang.TypeFn.cast_signed),
):
    """Applies the binary function fun elementwise.
    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    O[None] = fun(cast(linalg_lang.U, lhs[None]), cast(linalg_lang.U, rhs[None]))


@module_builder
def generate_add_module(shape):
    dtype = BF16Type.get()

    @func.FuncOp.from_py_func(
        MemRefType.get(shape, dtype),
        MemRefType.get(shape, dtype),
        MemRefType.get(shape, F32Type.get()),
    )
    def mul(lhs, rhs, out):
        elemwise_binary(
            lhs,
            rhs,
            outs=[out],
            fun=linalg_lang.BinaryFn.add,
            cast=linalg_lang.TypeFn.cast_unsigned,
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
    %l1, %herd_tile_loop = transform.air.linalg_tile %l0 [0,128]
    %l3, %inner_tile_loop = transform.air.linalg_tile %l1 [32,32]
    transform.air.linalg_promote %l3 {"operands_to_promote"=[0,1,2], "memory_space"="L1"}
    %inner_tile_par = transform.loop.forall_to_parallel %inner_tile_loop  : (!pdl.operation) -> !pdl.operation
    %herd_tile_par = transform.loop.forall_to_parallel %herd_tile_loop  : (!pdl.operation) -> !pdl.operation
    %herd = transform.air.par_to_herd %herd_tile_par
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

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-lower-herd-parallel)",
            "func.func(convert-linalg-to-loops)",
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
    air_loop_fusion=True,
    use_lock_race_condition_fix=True,
)
module_function = backend.compile(module)
