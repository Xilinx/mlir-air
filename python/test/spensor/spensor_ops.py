# ./python/test/spensor/spensor_ops.py -*- Python -*-
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

def getConsantIndex(val: int):
    return arith.ConstantOp.from_int_and_width(val, IndexType())


def getReduceCascade(ctx: Context) -> ModuleOp:
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
    spensor_output_type = spensor.SpensorType(TensorType(f32, [16, 1]), l3_memory)
    function_type = FunctionType.from_lists(
        [
            spensor_input_type,
        ],
        [spensor_output_type],
    )
    # function(arg0: <<16x32xf32>, L3>) -> <<16x1xf32>, L1>
    func_op = func.FuncOp("reduce", function_type)
    block = func_op.body.block
    arg0 = block.args[0]

    const_1 = getConsantIndex(1)
    const_0 = getConsantIndex(0)
    const_4 = getConsantIndex(4)

    split_arg0 = spensor.SplitOp(arg0, num_partitions=const_4, dim=const_1)
    arg0_l2 = spensor.MoveOp(split_arg0, L2_memory)

    arg0_l1 = spensor.MoveOp(arg0_l2, L1_memory)
    reduce_res = spensor.ReduceSumOp(arg0_l1)

    combine_res = spensor.NDCombineOp(reduce_res, nd_dim=const_0, dim=const_1, memory=L2_memory)
    combine_res_l1 = spensor.MoveOp(combine_res, L1_memory)
    combine_reduce_res = spensor.ReduceSumOp(combine_res_l1)

    combine_reduce_res_l2 = spensor.MoveOp(combine_reduce_res.result, L2_memory)

    reduce_res_l3 = spensor.MoveOp(combine_reduce_res_l2.result, l3_memory)
    result = spensor.CombineToSpensorOp(reduce_res_l3.result, (0,))
    return_op = func.ReturnOp(result.result)

    block.add_ops(
        [
            const_0,
            const_1,
            const_4,
            split_arg0,
            arg0_l2,
            arg0_l1,
            reduce_res,
            combine_res,
            combine_res_l1,
            combine_reduce_res,
            combine_reduce_res_l2,
            reduce_res_l3,
            result,
            return_op,
        ]
    )
    module_op = ModuleOp([declare_l1, declare_l2, declare_l3, func_op])
    return module_op

ctx = Context()
module_op = getReduceCascade(ctx)
print(module_op)

# CHECK: builtin.module {
# CHECK-NEXT: "spensor.declare_memory"() <{memory_name = "L1", memory_shape = [6 : i64], load = ["L2", "L3"], store = ["L2", "L3"]}> {memory_space = 2 : i64} : () -> ()
# CHECK-NEXT: "spensor.declare_memory"() <{memory_name = "L2", memory_shape = [], load = ["L3"], store = ["L3"]}> {memory_space = 1 : i64} : () -> ()
# CHECK-NEXT: "spensor.declare_memory"() <{memory_name = "L3", memory_shape = [], load = [], store = []}> {memory_space = 0 : i64} : () -> ()
# CHECK-NEXT:   func.func @reduce(%0 : !spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>>) -> !spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L3", []>> {
# CHECK-NEXT:     %1 = arith.constant 0 : index
# CHECK-NEXT:     %2 = arith.constant 1 : index
# CHECK-NEXT:     %3 = arith.constant 4 : index
# CHECK-NEXT:     %4 = "spensor.split"(%0, %3, %2) : (!spensor.spensor<tensor<16x32xf32>, !spensor.memory<"L3", []>>, index, index) -> !spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x8xf32>, !spensor.memory<"L3", []>>>>
# CHECK-NEXT:     %5 = "spensor.move"(%4) <{memory_name = "L2"}> : (!spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x8xf32>, !spensor.memory<"L3", []>>>>) -> !spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x8xf32>, !spensor.memory<"L2", []>>>>
# CHECK-NEXT:     %6 = "spensor.move"(%5) <{memory_name = "L1"}> : (!spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x8xf32>, !spensor.memory<"L2", []>>>>) -> !spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x8xf32>, !spensor.memory<"L1", [6 : i64]>>>>
# CHECK-NEXT:     %7 = "spensor.reduce_sum"(%6) : (!spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x8xf32>, !spensor.memory<"L1", [6 : i64]>>>>) -> !spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L1", [6 : i64]>>>>
# CHECK-NEXT:     %8 = "ndspensor.combine"(%7, %1, %2) <{memory_name = "L2"}> : (!spensor.ndspensor<tensor<4x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L1", [6 : i64]>>>>, index, index) -> !spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x4xf32>, !spensor.memory<"L2", []>>>>
# CHECK-NEXT:     %9 = "spensor.move"(%8) <{memory_name = "L1"}> : (!spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x4xf32>, !spensor.memory<"L2", []>>>>) -> !spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x4xf32>, !spensor.memory<"L1", [6 : i64]>>>>
# CHECK-NEXT:     %10 = "spensor.reduce_sum"(%9) : (!spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x4xf32>, !spensor.memory<"L1", [6 : i64]>>>>) -> !spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L1", [6 : i64]>>>>
# CHECK-NEXT:     %11 = "spensor.move"(%10) <{memory_name = "L2"}> : (!spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L1", [6 : i64]>>>>) -> !spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L2", []>>>>
# CHECK-NEXT:     %12 = "spensor.move"(%11) <{memory_name = "L3"}> : (!spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L2", []>>>>) -> !spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L3", []>>>>
# CHECK-NEXT:     %13 = "ndspensor.combine_to_spensor"(%12) <{combine_index = [0 : i64]}> : (!spensor.ndspensor<tensor<1x!spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L3", []>>>>) -> !spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L3", []>>
# CHECK-NEXT:     func.return %13 : !spensor.spensor<tensor<16x1xf32>, !spensor.memory<"L3", []>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
