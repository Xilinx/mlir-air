# run.py -*- Python -*-
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
import air.compiler.aircc.main as aircc

import numpy as np
import pyxrt as xrt
import sys
import filelock

in_size = 32*32
out_size = 32*32

in_size_bytes = in_size * 4
out_size_bytes = out_size * 4

opts_xclbin = 'aie.xclbin'
opts_kernel = 'MLIR_AIE'
opts_insts = opts_xclbin.removesuffix('.xclbin') + ".insts.txt"

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

    mlir_module = generate_add_module([32*32], IntegerType.get_signless(32))

    aircc_options = ['--device', 'ipu', 'air.mlir', '-o', opts_xclbin]
    aircc.run(mlir_module, aircc_options)

with open(opts_insts, 'r') as f:
    instr_text = f.read().split('\n')
    instr_text = [l for l in instr_text if l != '']
    instr_v = np.array([int(i,16) for i in instr_text], dtype=np.uint32)

with filelock.FileLock("/tmp/ipu.lock"):
    device = xrt.device(0)
    xclbin = xrt.xclbin(opts_xclbin)
    kernels = xclbin.get_kernels()
    try:
        xkernel = [k for k in kernels if opts_kernel in k.get_name()][0]
    except:
        print(f"Kernel '{opts_kernel}' not found in '{opts_xclbin}'")
        exit(-1)

    device.register_xclbin(xclbin)
    context = xrt.hw_context(device, xclbin.get_uuid())
    kernel = xrt.kernel(context, xkernel.get_name())

    bo_instr = xrt.bo(device, len(instr_v)*4, xrt.bo.cacheable, kernel.group_id(0))
    bo_a = xrt.bo(device, in_size_bytes, xrt.bo.host_only, kernel.group_id(2))
    bo_b = xrt.bo(device, in_size_bytes, xrt.bo.host_only, kernel.group_id(3))
    bo_c = xrt.bo(device, out_size_bytes, xrt.bo.host_only, kernel.group_id(4))

    bo_instr.write(instr_v, 0)
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    input_a = np.arange(in_size, dtype=np.uint32)
    bo_a.write(input_a, 0)
    bo_a.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    input_b = np.arange(in_size, dtype=np.uint32)
    bo_b.write(input_b, 0)
    bo_b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    h = kernel(bo_instr, len(instr_v), bo_a, bo_b, bo_c)
    h.wait()

    bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    output_buffer = bo_c.read(out_size_bytes, 0).view(np.uint32)

print("input:", input_a)
print("input:", input_b)
print("output:", output_buffer)

ref = input_a * input_b
if np.equal(ref, output_buffer).all():
    print("PASS!")
    exit(0)
else:
    print("failed.")
    exit(-1)
