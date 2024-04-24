# gen.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air
from air.ir import *
from air.passmanager import *
from air.dialects import air as airdialect
from air.dialects import arith, func, linalg, memref
from air.dialects.linalg.opdsl.lang import *
from air.compiler.util import run_transform

def generate_add_module(shape, dtype):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(
            MemRefType.get(shape, dtype), MemRefType.get(shape, dtype), MemRefType.get(shape, dtype))
        def add(lhs, rhs, out):
            linalg.elemwise_binary(
                lhs,
                rhs,
                outs=[out],
                fun=BinaryFn.add,
                cast=TypeFn.cast_unsigned)
            return

    #print ("\nlinalg Module:\n\n", module)

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

    pm = PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))')
    pm.run(module.operation)
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, module)

    pm = PassManager.parse('builtin.module(func.func(canonicalize,cse))')
    pm.run(module.operation)
    return module

with Context() as ctx, Location.unknown():
    mlir_module = generate_add_module([128,128], BF16Type.get())

    # print("\nTiled AIR Module:\n\n", mlir_module)
    # with open("add.air.mlir", "w") as f:
    #     f.write(str(mlir_module))

    pipeline = "builtin.module(" + ",".join([
        "func.func(air-lower-herd-parallel)",
        #"air-dependency",
        #"air-dependency-schedule-opt",
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-specialize-channel-wrap-and-stride",
        "air-linalg-to-func",
        'func.func(air-renumber-dma)'
    ]) + ")"
    pm = PassManager.parse(pipeline)
    pm.run(mlir_module.operation)

    # print("\nAIE Module:\n\n", mlir_module)
    # with open("add.chan.mlir", "w") as f:
    #     f.write(str(mlir_module))

    pipeline = "builtin.module(" + ",".join([
        "air-to-aie{device=npu row-offset=2 col-offset=0}",
        "air-to-std",
        "symbol-dce",
        "canonicalize", "cse",
    ]) + ")"
    pm = PassManager.parse(pipeline)
    pm.run(mlir_module.operation)

    # print("\nAIE Module:\n\n", mlir_module)
    # with open("add.aieairrt.mlir", "w") as f:
    #     f.write(str(mlir_module))

    pipeline = "builtin.module(" + ",".join([
        "airrt-to-npu",
        "canonicalize", "cse",
    ]) + ")"
    pm = PassManager.parse(pipeline)
    pm.run(mlir_module.operation)

    # print("\nAIE Module:\n\n", mlir_module)
    # with open("add.aienpu.mlir", "w") as f:
    #     f.write(str(mlir_module))

    import aie.compiler.aiecc.main as aiecc

    aiecc_options = ['--no-aiesim',
                     '--xbridge',
                     '--xchesscc',
                     '--aie-generate-cdo',
                     '--aie-generate-npu',
                     '--no-compile-host',
                     '--npu-insts-name=insts.txt',
                     '--xclbin-name=add.xclbin',
                     'aie.mlir']
    aiecc.run(mlir_module, aiecc_options)
