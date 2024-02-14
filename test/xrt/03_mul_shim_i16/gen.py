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
from air._mlir_libs._airMlir import _run_air_transform as run_air_transform

def generate_add_module(shape, dtype):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(
            MemRefType.get(shape, dtype), MemRefType.get(shape, dtype), MemRefType.get(shape, dtype))
        def mul(lhs, rhs, out):
            linalg.elemwise_binary(
                lhs,
                rhs,
                outs=[out],
                fun=BinaryFn.mul,
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
        %l1, %outer_tile_loops:1 = transform.air.linalg_tile %l0 [1024]
        %l2, %inner_tile_loops:1 = transform.air.linalg_tile %l1 [32]
        transform.air.linalg_promote %l2 {"operands_to_promote"=[0,1,2], "memory_space"="L1"}
        %herds = transform.air.par_to_herd %outer_tile_loops#0
        %copies = transform.pdl_match @match_copy in %arg0 : (!pdl.operation) -> !pdl.operation
        %h = transform.air.copy_to_dma %copies
      }
    }
    """

    pm = PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))')
    pm.run(module.operation)
    transform_ir = Module.parse(transform_ir_string)
    run_air_transform(transform_ir, module)

    pm = PassManager.parse('builtin.module(func.func(canonicalize,cse))')
    pm.run(module.operation)
    return module

with Context() as ctx, Location.unknown():
    airdialect.register_dialect(ctx)
    mlir_module = generate_add_module([32*32], IntegerType.get_signless(16))

    # print("\nTiled AIR Module:\n\n", mlir_module)
    # with open("mul.air.mlir", "w") as f:
    #     f.write(str(mlir_module))

    pipeline = "builtin.module(" + ",".join([
        "func.func(air-lower-herd-parallel)",
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-specialize-channel-wrap-and-stride",
        "func.func(convert-linalg-to-loops)",
        'func.func(air-renumber-dma)',
        "air-to-aie{emit-while-loop=true device=ipu row-offset=2 col-offset=0}",
        "air-to-std",
        "airrt-to-ipu",
        "canonicalize", "cse",
    ]) + ")"
    pm = PassManager.parse(pipeline)
    pm.run(mlir_module.operation)

    # print("\nAIE Module:\n\n", mlir_module)
    # with open("mul.air_ipu.mlir", "w") as f:
    #     f.write(str(mlir_module))

    import aie.compiler.aiecc.main as aiecc

    aiecc_options = ['--no-aiesim',
                     '--aie-generate-cdo',
                     '--aie-generate-ipu',
                     '--no-compile-host',
                     '--ipu-insts-name=insts.txt',
                     '--xclbin-name=mul.xclbin',
                     'aie.mlir']
    aiecc.run(mlir_module, aiecc_options)
