# ./python/test/spensor/append_constant_pass.py -*- Python -*-
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
from spensor.utils.spensor_util import getConstantOpByIndex
from spensor.passes.append_constant import AppendConstant



def getModule(ctx: Context) -> ModuleOp:
    l3_memory = spensor.MemoryType("L3", [])
    spensor_input_type = spensor.SpensorType(TensorType(f32, [16, 32]), l3_memory)
    function_type = FunctionType.from_lists(
        [
            spensor_input_type,
        ],
        [spensor_input_type],
    )
    # function(arg0: <<16x32xf32>, L3>) -> <<16x32xf32>, L3>
    func_op = func.FuncOp("constant", function_type)
    block = func_op.body.block
    arg0 = block.args[0]

    const_1 = getConstantOpByIndex(1)
    const_0 = getConstantOpByIndex(0)
    const_4 = getConstantOpByIndex(4)

    return_op = func.ReturnOp(arg0)

    block.add_ops(
        [
            return_op,
        ]
    )
    module_op = ModuleOp([func_op])
    return module_op

ctx = Context()
module_op = getModule(ctx)
AppendConstant().apply(ctx, module_op)
print(module_op)

# CHECK: builtin.module {
# CHECK-NEXT: func.func @constant(%0 : !spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>>) -> !spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>> {
# CHECK-NEXT:     %autogen_1_index = arith.constant 1 : index
# CHECK-NEXT:     %autogen_0_index = arith.constant 0 : index
# CHECK-NEXT:     %autogen_4_index = arith.constant 4 : index
# CHECK-NEXT:     func.return %0 : !spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
