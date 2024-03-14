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
from air.compiler.util import run_transform
import air.compiler.aircc.main as aircc

import numpy as np
import pyxrt as xrt
import sys
import filelock
from bfloat16 import bfloat16

sizes = [
    [1024],
]

dtypes = [
    (np.int32, np.int32),
    (np.int16, np.int32),
    (np.int16, np.int16),
    (np.float32, np.float32),
    # (bfloat16, np.float32),
    # (bfloat16, bfloat16),
]

opts_xclbin = 'aie.xclbin'
opts_kernel = 'MLIR_AIE'
opts_insts = opts_xclbin.removesuffix('.xclbin') + ".insts.txt"

def to_type(dtype):
    if dtype == np.int32:
        return IntegerType.get_signless(32)
    if dtype == np.int16:
        return IntegerType.get_signless(16)
    if dtype == np.float32:
        return F32Type.get()
    if dtype == bfloat16:
        return BF16Type.get()
    return None

def generate_add_module(shape, idtype, odtype):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(
            MemRefType.get(shape, idtype), MemRefType.get(shape, idtype), MemRefType.get(shape, odtype))
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
    run_transform(transform_ir, module)

    pm = PassManager.parse('builtin.module(func.func(canonicalize,cse))')
    pm.run(module.operation)
    return module

def run_test(size, idtype, odtype):
    with Context() as ctx, Location.unknown():
        mlir_input_type = to_type(idtype)
        mlir_output_type = to_type(odtype)

        mlir_module = generate_add_module(size, mlir_input_type, mlir_output_type)

        aircc_options = ['--device', 'ipu', 'air.mlir', '-xchesscc', '-xbridge', '-o', opts_xclbin]
        aircc.run(mlir_module, aircc_options)

    with open(opts_insts, 'r') as f:
        instr_text = f.read().split('\n')
        instr_text = [l for l in instr_text if l != '']
        instr_v = np.array([int(i,16) for i in instr_text], dtype=np.uint32)

    input_a = (np.random.rand(*size) * 127).astype(idtype).reshape(size)
    input_b = (np.random.rand(*size) * 127).astype(idtype).reshape(size)
    ref = (input_a * input_b).astype(odtype)

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

        in_size_bytes = input_a.size * input_a.itemsize
        out_size_bytes = ref.size * ref.itemsize

        bo_instr = xrt.bo(device, len(instr_v)*4, xrt.bo.cacheable, kernel.group_id(0))
        bo_a = xrt.bo(device, in_size_bytes, xrt.bo.host_only, kernel.group_id(2))
        bo_b = xrt.bo(device, in_size_bytes, xrt.bo.host_only, kernel.group_id(3))
        bo_c = xrt.bo(device, out_size_bytes, xrt.bo.host_only, kernel.group_id(4))

        bo_instr.write(instr_v, 0)
        bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        bo_a.write(input_a, 0)
        bo_a.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        bo_b.write(input_b, 0)
        bo_b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        h = kernel(bo_instr, len(instr_v), bo_a, bo_b, bo_c)
        h.wait()

        bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_buffer = bo_c.read(out_size_bytes, 0).view(odtype)

    print("input:", input_a)
    print("input:", input_b)
    print("output:", output_buffer)

    if np.allclose(ref, output_buffer, 0.01):
        print("PASS!")
        return 1
    else:
        print("failed.")
        return 0

passed = 0
for (idtype, odtype) in dtypes:
    for size in sizes:
        try:
            print("Testing size:", size, "dtype:", idtype, odtype)
            passed = passed + run_test(size, idtype, odtype)
        except Exception as e:
            print(e)

num_tests = len(sizes)*len(dtypes)
if passed != num_tests:
    print (f"failed. {passed}/{num_tests}")
    exit(-1)
else:
    print (f"PASSED! {passed}/{num_tests}")
    exit(0)
