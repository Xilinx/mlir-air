# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

from air.ir import *
from air.dialects.air import *
from air.dialects import func
from air.dialects.memref import AllocOp, DeallocOp


def constructAndPrintInFunc(f):
    print("\nTEST:", f.__name__)

    @module_builder
    def build_module():
        @func.FuncOp.from_py_func()
        def build_function():
            f()

    module = build_module()
    print(module)


# CHECK-LABEL: TEST: dmaMemcpyNdWithPadding
# CHECK: air.dma_memcpy_nd
# CHECK-SAME: pad_after = array<i32: 2, 1>
# CHECK-SAME: pad_before = array<i32: 0, 2>
@constructAndPrintInFunc
def dmaMemcpyNdWithPadding():
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l1_type = MemRefType.get([16, 16], T.i32(), memory_space=l1_space)
    l2_type = MemRefType.get([16, 16], T.i32(), memory_space=l2_space)
    dst = AllocOp(l1_type, [], [])
    src = AllocOp(l2_type, [], [])
    dma_memcpy_nd(
        dst,
        src,
        pad_before=[0, 2],
        pad_after=[2, 1],
    )
    DeallocOp(dst)
    DeallocOp(src)


# CHECK-LABEL: TEST: dmaMemcpyNdWithoutPadding
# CHECK: air.dma_memcpy_nd
# CHECK-NOT: pad_before
# CHECK-NOT: pad_after
@constructAndPrintInFunc
def dmaMemcpyNdWithoutPadding():
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l1_type = MemRefType.get([16, 16], T.i32(), memory_space=l1_space)
    l2_type = MemRefType.get([16, 16], T.i32(), memory_space=l2_space)
    dst = AllocOp(l1_type, [], [])
    src = AllocOp(l2_type, [], [])
    dma_memcpy_nd(dst, src)
    DeallocOp(dst)
    DeallocOp(src)


# CHECK-LABEL: TEST: channelPutWithPadding
# CHECK: air.channel
# CHECK-SAME: sym_name = "PutPadChan"
# CHECK: air.channel.put
# CHECK-SAME: pad_after = array<i32: 1>
# CHECK-SAME: pad_before = array<i32: 2>
@constructAndPrintInFunc
def channelPutWithPadding():
    Channel("PutPadChan")
    l2_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2_type = MemRefType.get([32], T.i32(), memory_space=l2_space)
    src = AllocOp(l2_type, [], [])
    ChannelPut("PutPadChan", src, pad_before=[2], pad_after=[1])
    DeallocOp(src)


# CHECK-LABEL: TEST: channelGetWithPadding
# CHECK: air.channel
# CHECK-SAME: sym_name = "GetPadChan"
# CHECK: air.channel.get
# CHECK-SAME: pad_after = array<i32: 3>
# CHECK-SAME: pad_before = array<i32: 1>
@constructAndPrintInFunc
def channelGetWithPadding():
    Channel("GetPadChan")
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_type = MemRefType.get([32], T.i32(), memory_space=l1_space)
    dst = AllocOp(l1_type, [], [])
    ChannelGet("GetPadChan", dst, pad_before=[1], pad_after=[3])
    DeallocOp(dst)
