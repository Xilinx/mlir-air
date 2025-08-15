# ./python/test/spensor/spensor_to_tensor.py -*- Python -*-
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
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.context import Context
from xdsl.dialects import func, arith
import spensor.dialects.spensor_dialect as spensor
from spensor.utils.spensor_util import getConstantOpByIndex
from spensor.passes.spensor_to_tensor import SpensorToTensor
from spensor.passes.memory_analysis import MemoryAnalysis
from xdsl.parser import Parser
from spensor.dialects.spensor_dialect import SpensorDialect, NDSpensorDialect
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from xdsl.dialects.affine import Affine
from xdsl.dialects.scf import Scf

mlir_input = """
builtin.module {
  "spensor.declare_memory"() <{memory_name = "L1", memory_shape = [6 : i64, 4 : i64], load = ["L2", "L3"], store = ["L2", "L3"]}> {memory_space = 2 : i64} : () -> ()
  "spensor.declare_memory"() <{memory_name = "L2", memory_shape = [], load = ["L3"], store = ["L3"]}> {memory_space = 1 : i64} : () -> ()
  "spensor.declare_memory"() <{memory_name = "L3", memory_shape = [], load = [], store = []}> {memory_space = 0 : i64} : () -> ()
  func.func @"64_add"(%0 : !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>, %1 : !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>, %2 : !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>) {
    %autogen_0_index = arith.constant 0 : index
    %autogen_1_index = arith.constant 1 : index
    %autogen_8_index = arith.constant 8 : index
    "scf.parallel"(%autogen_0_index, %autogen_0_index, %autogen_8_index, %autogen_8_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^0(%3 : index, %4 : index):
      %5 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%3]
      %6 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%4]
      %7 = "spensor.subview"(%0, %5, %6, %autogen_8_index, %autogen_8_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>}> : (!spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>, index, index, index, index, index, index) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L3", []>>
      %8 = "spensor.move"(%7) <{memory_name = "L2"}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L3", []>>) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L2", []>>
      %9 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%3]
      %10 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%4]
      %11 = "spensor.subview"(%1, %9, %10, %autogen_8_index, %autogen_8_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>}> : (!spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>, index, index, index, index, index, index) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L3", []>>
      %12 = "spensor.move"(%11) <{memory_name = "L2"}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L3", []>>) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L2", []>>
      "scf.parallel"(%autogen_0_index, %autogen_0_index, %autogen_1_index, %autogen_1_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
      ^1(%13 : index, %14 : index):
        %15 = affine.apply affine_map<()[s0] -> (s0)> ()[%13]
        %16 = affine.apply affine_map<()[s0] -> (s0)> ()[%14]
        %17 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%15]
        %18 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%16]
        %19 = "spensor.subview"(%12, %17, %18, %autogen_8_index, %autogen_8_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L2", []>>, index, index, index, index, index, index) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>
        %20 = "spensor.move"(%19) <{memory_name = "L1"}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>
        %21 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%15]
        %22 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%16]
        %23 = "spensor.subview"(%8, %21, %22, %autogen_8_index, %autogen_8_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L2", []>>, index, index, index, index, index, index) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>
        %24 = "spensor.move"(%23) <{memory_name = "L1"}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>
        %25 = "spensor.add"(%24, %20) {self_assign = false} : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>, !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>
        %26 = "spensor.subview"(%25, %autogen_0_index, %autogen_0_index, %autogen_8_index, %autogen_8_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 1, 2, 2, 2>}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L1", [6 : i64, 4 : i64]>>, index, index, index, index, index, index) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L2", []>>
        %27 = "spensor.move"(%26) <{memory_name = "L2"}> : (!spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L2", []>>) -> !spensor.spensor<tensor<8x8xf32>, !spensor.memory<"L2", []>>
        scf.reduce
      }) {memory_tag = "L1"} : (index, index, index, index, index, index) -> ()
      scf.reduce
    }) {memory_tag = "L2"} : (index, index, index, index, index, index) -> ()
    func.return
  }
}
"""


def parse_file(ctx: Context, mlir_input: str) -> Operation:
    parser = Parser(ctx, mlir_input)
    module = parser.parse_module()
    return module


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
ctx.register_dialect(Builtin.name, lambda: Builtin)
ctx.register_dialect(Func.name, lambda: Func)
ctx.register_dialect(Arith.name, lambda: Arith)
ctx.register_dialect(Affine.name, lambda: Affine)
ctx.register_dialect(Scf.name, lambda: Scf)
ctx.register_dialect(SpensorDialect.name, lambda: SpensorDialect)
ctx.register_dialect(NDSpensorDialect.name, lambda: NDSpensorDialect)
module_op = parse_file(ctx, mlir_input)
MemoryAnalysis().apply(ctx, module_op)
SpensorToTensor().apply(ctx, module_op)
print(module_op)

# CHECK: builtin.module {
# CHECK-NEXT:   "spensor.declare_memory"() <{memory_name = "L1", memory_shape = [6 : i64, 4 : i64], load = ["L2", "L3"], store = ["L2", "L3"]}> {memory_space = 2 : i64} : () -> ()
# CHECK-NEXT:   "spensor.declare_memory"() <{memory_name = "L2", memory_shape = [], load = ["L3"], store = ["L3"]}> {memory_space = 1 : i64} : () -> ()
# CHECK-NEXT:   "spensor.declare_memory"() <{memory_name = "L3", memory_shape = [], load = [], store = []}> {memory_space = 0 : i64} : () -> ()
# CHECK-NEXT:   func.func @"64_add"(%0 : memref<64x64xf32>, %1 : memref<64x64xf32>, %2 : memref<64x64xf32>) {
# CHECK-NEXT:     %3 = bufferization.to_tensor %2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
# CHECK-NEXT:     %4 = bufferization.to_tensor %1 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
# CHECK-NEXT:     %5 = bufferization.to_tensor %0 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
# CHECK-NEXT:     %autogen_0_index = arith.constant 0 : index
# CHECK-NEXT:     %autogen_1_index = arith.constant 1 : index
# CHECK-NEXT:     %autogen_8_index = arith.constant 8 : index
# CHECK-NEXT:     "scf.parallel"(%autogen_0_index, %autogen_0_index, %autogen_8_index, %autogen_8_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
# CHECK-NEXT:     ^0(%6 : index, %7 : index):
# CHECK-NEXT:       %8 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<8x8xf32>
# CHECK-NEXT:       %9 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<8x8xf32>
# CHECK-NEXT:       %10 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<8x8xf32>
# CHECK-NEXT:       %11 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%6]
# CHECK-NEXT:       %12 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%7]
# CHECK-NEXT:       %13 = "tensor.extract_slice"(%5, %11, %12, %autogen_1_index, %autogen_1_index) <{static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>, operandSegmentSizes = array<i32: 1, 2, 0, 2>}> : (tensor<64x64xf32>, index, index, index, index) -> tensor<8x8xf32>
# CHECK-NEXT:       %14 = linalg.copy ins(%13 : tensor<8x8xf32>) outs(%10 : tensor<8x8xf32>) -> tensor<8x8xf32>
# CHECK-NEXT:       %15 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%6]
# CHECK-NEXT:       %16 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%7]
# CHECK-NEXT:       %17 = "tensor.extract_slice"(%4, %15, %16, %autogen_1_index, %autogen_1_index) <{static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>, operandSegmentSizes = array<i32: 1, 2, 0, 2>}> : (tensor<64x64xf32>, index, index, index, index) -> tensor<8x8xf32>
# CHECK-NEXT:       %18 = linalg.copy ins(%17 : tensor<8x8xf32>) outs(%9 : tensor<8x8xf32>) -> tensor<8x8xf32>
# CHECK-NEXT:       "scf.parallel"(%autogen_0_index, %autogen_0_index, %autogen_1_index, %autogen_1_index, %autogen_1_index, %autogen_1_index) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
# CHECK-NEXT:       ^1(%19 : index, %20 : index):
# CHECK-NEXT:         %21 = bufferization.alloc_tensor() {memory_space = 2 : i64} : tensor<8x8xf32>
# CHECK-NEXT:         %22 = bufferization.alloc_tensor() {memory_space = 2 : i64} : tensor<8x8xf32>
# CHECK-NEXT:         %23 = bufferization.alloc_tensor() {memory_space = 2 : i64} : tensor<8x8xf32>
# CHECK-NEXT:         %24 = affine.apply affine_map<()[s0] -> (s0)> ()[%19]
# CHECK-NEXT:         %25 = affine.apply affine_map<()[s0] -> (s0)> ()[%20]
# CHECK-NEXT:         %26 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%24]
# CHECK-NEXT:         %27 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%25]
# CHECK-NEXT:         %28 = "tensor.extract_slice"(%9, %26, %27, %autogen_1_index, %autogen_1_index) <{static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>, operandSegmentSizes = array<i32: 1, 2, 0, 2>}> : (tensor<8x8xf32>, index, index, index, index) -> tensor<8x8xf32>
# CHECK-NEXT:         %29 = linalg.copy ins(%28 : tensor<8x8xf32>) outs(%23 : tensor<8x8xf32>) -> tensor<8x8xf32>
# CHECK-NEXT:         %30 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%24]
# CHECK-NEXT:         %31 = affine.apply affine_map<()[s0] -> ((s0 * 8))> ()[%25]
# CHECK-NEXT:         %32 = "tensor.extract_slice"(%10, %30, %31, %autogen_1_index, %autogen_1_index) <{static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>, operandSegmentSizes = array<i32: 1, 2, 0, 2>}> : (tensor<8x8xf32>, index, index, index, index) -> tensor<8x8xf32>
# CHECK-NEXT:         %33 = linalg.copy ins(%32 : tensor<8x8xf32>) outs(%22 : tensor<8x8xf32>) -> tensor<8x8xf32>
# CHECK-NEXT:         %34 = linalg.add ins(%22, %23 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%21 : tensor<8x8xf32>) -> tensor<8x8xf32>
# CHECK-NEXT:         %35 = "tensor.extract_slice"(%21, %autogen_0_index, %autogen_0_index, %autogen_1_index, %autogen_1_index) <{static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 8, 8>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>, operandSegmentSizes = array<i32: 1, 2, 0, 2>}> : (tensor<8x8xf32>, index, index, index, index) -> tensor<8x8xf32>
# CHECK-NEXT:         %36 = linalg.copy ins(%35 : tensor<8x8xf32>) outs(%8 : tensor<8x8xf32>) -> tensor<8x8xf32>
# CHECK-NEXT:         scf.reduce
# CHECK-NEXT:       }) {memory_tag = "L1"} : (index, index, index, index, index, index) -> ()
# CHECK-NEXT:       scf.reduce
# CHECK-NEXT:     }) {memory_tag = "L2"} : (index, index, index, index, index, index) -> ()
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }