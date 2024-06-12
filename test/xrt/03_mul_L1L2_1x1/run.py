# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

import air.backend.xrt as xrt_backend
from air.dialects.air import *
from air.dialects.func import FuncOp
from air.dialects.linalg import elemwise_binary
from air.dialects.linalg.opdsl.lang import BinaryFn, TypeFn
from air.dialects.scf import for_, yield_
from air.ir import *

import numpy as np
import filelock
from bfloat16 import bfloat16

verbose = False

sizes = [
    [4096],
]

dtypes = [
    (np.int32, np.int32),
    (np.int16, np.int32),
    (np.int16, np.int16),
    (np.float32, np.float32),
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
def build_module(idtype, odtype, l3_shape, l2_shape, l1_shape):
    memrefTyIn = MemRefType.get(l3_shape, to_type(idtype))
    memrefTyOut = MemRefType.get(l3_shape, to_type(odtype))

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_itile_type = MemRefType.get(
        shape=l1_shape,
        element_type=to_type(idtype),
        memory_space=l1_mem_space,
    )
    l1_otile_type = MemRefType.get(
        shape=l1_shape,
        element_type=to_type(odtype),
        memory_space=l1_mem_space,
    )
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_itile_type = MemRefType.get(
        shape=l2_shape,
        element_type=to_type(idtype),
        memory_space=l2_mem_space,
    )
    l2_otile_type = MemRefType.get(
        shape=l2_shape,
        element_type=to_type(odtype),
        memory_space=l2_mem_space,
    )

    ChannelOp("ChanL2A")
    ChannelOp("ChanL2B")
    ChannelOp("ChanL2C")
    ChannelOp("ChanL1A")
    ChannelOp("ChanL1B")
    ChannelOp("ChanL1C")

    @FuncOp.from_py_func(memrefTyIn, memrefTyIn, memrefTyOut)
    def mul(arg0, arg1, arg2):
        @launch(sizes=[l3_shape[0] // l2_shape[0]], operands=[arg0, arg1, arg2])
        def launch_body(i, s, a, b, c):
            m = arith.ConstantOp.create_index(l2_shape[0])
            o = arith.MulIOp(m, i)
            ChannelPut(
                "ChanL2A", a, src_offsets=[o], src_strides=[m], src_sizes=[1]
            )
            ChannelPut(
                "ChanL2B", b, src_offsets=[o], src_strides=[m], src_sizes=[1]
            )
            ChannelGet(
                "ChanL2C", c, dst_offsets=[o], dst_strides=[m], dst_sizes=[1]
            )

            @segment(name="segment_0")
            def segment_body():
                for _ in for_(l3_shape[0] // l2_shape[0]):
                    l2_tile_a = Alloc(l2_itile_type)
                    l2_tile_b = Alloc(l2_itile_type)
                    l2_tile_c = Alloc(l2_otile_type)

                    # get from L2, put to L1
                    ChannelGet("ChanL2A", l2_tile_a)
                    ChannelPut("ChanL1A", l2_tile_a)
                    ChannelGet("ChanL2B", l2_tile_b)
                    ChannelPut("ChanL1B", l2_tile_b)

                    @herd(name="herd_0", sizes=[1, 1])
                    def herd_body(x, y, sx, sy):
                        for _ in for_(l2_shape[0] // l1_shape[0]):
                            l1_tile_a = Alloc(l1_itile_type)
                            l1_tile_b = Alloc(l1_itile_type)
                            l1_tile_c = Alloc(l1_otile_type)
                            ChannelGet("ChanL1A", l1_tile_a)
                            ChannelGet("ChanL1B", l1_tile_b)
                            elemwise_binary(
                                l1_tile_a,
                                l1_tile_b,
                                outs=[l1_tile_c],
                                fun=BinaryFn.mul,
                                cast=TypeFn.cast_unsigned,
                            )
                            ChannelPut("ChanL1C", l1_tile_c)
                            Dealloc(l1_tile_a)
                            Dealloc(l1_tile_b)
                            Dealloc(l1_tile_c)
                            yield_([])

                    # get from L1, put to L2
                    ChannelGet("ChanL1C", l2_tile_c)
                    ChannelPut("ChanL2C", l2_tile_c)

                    Dealloc(l2_tile_a)
                    Dealloc(l2_tile_b)
                    Dealloc(l2_tile_c)
                    yield_([])


def run_test(size, idtype, odtype):

    mlir_module = build_module(idtype, odtype, size, [1024], [64])
    print(mlir_module)
    input_a = (np.random.rand(*size) * 127).astype(idtype).reshape(size)
    input_b = (np.random.rand(*size) * 127).astype(idtype).reshape(size)
    ref = (input_a * input_b).astype(odtype)
    input_c = np.ones_like(ref)

    backend = xrt_backend.XRTBackend(verbose=verbose)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        mul = backend.compile_and_load(mlir_module)
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
