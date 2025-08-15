# ./python/test/spensor/spensor_dce.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: spensor

# RUN: %PYTHON %s | FileCheck %s

from xdsl.dialects.builtin import (
    ModuleOp,
    IndexType,
    TensorType,
    f32,
    FunctionType,
)
from xdsl.context import Context
from xdsl.dialects import func, arith
import spensor.dialects.spensor_dialect as spensor
from spensor.passes.spensor_dce import SpensorDeadCodeElimination



def getModule(ctx: Context) -> ModuleOp:
    L1_memory = spensor.MemoryType("L1", [6])
    L2_memory = spensor.MemoryType("L2", [])
    l3_memory = spensor.MemoryType("L3", [])
    declare_l1 = spensor.DeclareMemoryOp(
        "L1", L1_memory.memory_shape, ["L2", "L3"], ["L2", "L3"], 2
    )
    declare_l2 = spensor.DeclareMemoryOp(
        "L2", L2_memory.memory_shape, ["L3"], ["L3"], 1
    )
    declare_l3 = spensor.DeclareMemoryOp("L3", l3_memory.memory_shape, [], [])

    spensor_input_type = spensor.SpensorType(TensorType(f32, [16, 32]), l3_memory)
    function_type = FunctionType.from_lists(
        [
            spensor_input_type,
        ],
        [spensor_input_type],
    )
    # function(arg0: <<16x32xf32>, L3>) -> <<16x32xf32>, L3>
    func_op = func.FuncOp("dce", function_type)
    block = func_op.body.block
    arg0 = block.args[0]

    return_op = func.ReturnOp(arg0)

    block.add_ops(
        [
            return_op,
        ]
    )
    module_op = ModuleOp([declare_l1, declare_l2, declare_l3, func_op])
    return module_op

ctx = Context()
module_op = getModule(ctx)
SpensorDeadCodeElimination().apply(ctx, module_op)
print(module_op)

# CHECK: builtin.module {
# CHECK-NEXT: func.func @dce(%0 : !spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>>) -> !spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>> {
# CHECK-NEXT:     func.return %0 : !spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
