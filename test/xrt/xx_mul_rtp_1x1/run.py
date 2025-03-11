# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

import air.backend.xrt as xrt_backend
import air.compiler.aircc.main as aircc
from air.dialects.air import *
from air.dialects.func import FuncOp, ReturnOp
from air.dialects.linalg import elemwise_binary
from air.dialects.linalg.opdsl.lang import BinaryFn, TypeFn
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.ir import *

import numpy as np
import filelock
from ml_dtypes import bfloat16

verbose = True

sizes = [
    [1024],
]

dtypes = [
    (np.int32, np.int32),
    # (np.int16, np.int32),
    # (np.int16, np.int16),
    # (np.float32, np.float32),
    # (bfloat16, np.float32),
    # (bfloat16, bfloat16),
]


def to_type(dtype):
    if dtype == np.int32:
        return T.i32()
    if dtype == np.int16:
        return T.i16()
    if dtype == np.float32:
        return F32Type.get()
    if dtype == bfloat16:
        return BF16Type.get()
    return None


@module_builder
def build_module(shape, idtype, odtype, tile_size):
    memrefTyIn = MemRefType.get(shape, to_type(idtype))
    memrefTyOut = MemRefType.get(shape, to_type(odtype))
    ChannelOp("ChanA")
    ChannelOp("ChanB")
    ChannelOp("ChanC")

    @FuncOp.from_py_func(memrefTyIn, memrefTyIn, memrefTyOut)
    def mul(arg0, arg1, arg2):
        @launch(operands=[arg0, arg1, arg2])
        def launch_body(a, b, c):
            ChannelPut("ChanA", a)
            ChannelPut("ChanB", b)
            ChannelGet("ChanC", c)

            @segment(name="segment_0")
            def segment_body():
                c = arith.ConstantOp.create_index(tile_size)
                @herd(name="herd_0", sizes=[1, 1], operands=[c])
                def herd_body(x, y, sx, sy, count):
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    itile_type = MemRefType.get(
                        shape=[tile_size],
                        element_type=to_type(idtype),
                        memory_space=mem_space,
                    )
                    otile_type = MemRefType.get(
                        shape=[tile_size],
                        element_type=to_type(odtype),
                        memory_space=mem_space,
                    )
                    for _ in for_(count):
                        tile_a = AllocOp(itile_type, [], [])
                        tile_b = AllocOp(itile_type, [], [])
                        tile_c = AllocOp(otile_type, [], [])
                        ChannelGet("ChanA", tile_a)
                        ChannelGet("ChanB", tile_b)
                        elemwise_binary(
                            tile_a,
                            tile_b,
                            outs=[tile_c],
                            fun=BinaryFn.mul,
                            cast=TypeFn.cast_unsigned,
                        )
                        ChannelPut("ChanC", tile_c)
                        DeallocOp(tile_a)
                        DeallocOp(tile_b)
                        DeallocOp(tile_c)
                        yield_([])


def run_test(size, idtype, odtype):

    mlir_module = build_module(size, idtype, odtype, 32)
    print(mlir_module)

    input_a = (np.random.rand(*size) * 127).astype(idtype).reshape(size)
    input_b = (np.random.rand(*size) * 127).astype(idtype).reshape(size)
    ref = (input_a * input_b).astype(odtype)
    input_c = np.ones_like(ref)

    backend = xrt_backend.XRTBackend(verbose=verbose)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        mul = backend.compile_and_load(mlir_module)
        print("running")
        (_, _, output_c) = mul(input_a, input_b, input_c)

    backend.unload()

    print("inputA:", input_a)
    print("inputB:", input_b)
    print("output:", output_c)

    if np.allclose(ref, output_c, 0.01):
        print("PASS!")
        return 1
    else:
        print("failed.")
        return 0


passed = 0
for idtype, odtype in dtypes:
    for size in sizes:
        try:
            print("Testing size:", size, "dtype:", idtype, odtype)
            passed = passed + run_test(size, idtype, odtype)
        except Exception as e:
            print(e)

num_tests = len(sizes) * len(dtypes)
if passed != num_tests:
    print(f"failed. {passed}/{num_tests}")
    exit(-1)
else:
    print(f"PASSED! {passed}/{num_tests}")
    exit(0)
