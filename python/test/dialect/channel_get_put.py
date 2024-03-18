# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

from air.ir import *
from air.dialects.air import *
from air.dialects.func import FuncOp, ReturnOp
from air.dialects.linalg import elemwise_binary
from air.dialects.linalg.opdsl.lang import BinaryFn, TypeFn
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_

import numpy as np

def to_type(dtype):
    if dtype == np.int32:
        return T.i32()
    return None

def build_module(shape, idtype, odtype):
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            memrefTyIn = MemRefType.get(shape, to_type(idtype))
            memrefTyOut = MemRefType.get(shape, to_type(odtype))
            # CHECK: air.channel @Chan0
            # CHECK: air.channel @Chan1
            # CHECK: air.channel @Chan2
            ChannelOp("Chan0")
            ChannelOp("Chan1")
            ChannelOp("Chan2")
            @FuncOp.from_py_func(memrefTyIn, memrefTyIn, memrefTyOut)
            def mul(arg0, arg1, arg2):
                @launch(operands=[arg0, arg1, arg2])
                def launch_body(a, b, c):
                    # CHECK: air.channel.put  @ChanA[] (%{{.*}}[] [] []) : (memref<1024xi32>)
                    # CHECK: air.channel.put  @ChanB[] (%{{.*}}[] [] []) : (memref<1024xi32>)
                    # CHECK: air.channel.get  @ChanC[] (%{{.*}}[] [] []) : (memref<1024xi32>)
                    ChannelPut("ChanA", [], a)
                    ChannelPut("ChanB", [], b)
                    ChannelGet("ChanC", [], c)
                    @segment(name="segment_0")
                    def segment_body():
                        @herd(name="herd_0", sizes=[1, 1])
                        def herd_body(x, y, sx, sy):
                            mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                            tile_type = MemRefType.get(shape=[32],
                                                       element_type=to_type(idtype),
                                                       memory_space=mem_space)
                            tile_a = AllocOp(tile_type, [], [])
                            tile_b = AllocOp(tile_type, [], [])
                            tile_c = AllocOp(tile_type, [], [])
                            # CHECK: air.channel.get  @ChanA[] (%{{.*}}[] [] []) : (memref<32xi32, 2 : i32>)
                            # CHECK: air.channel.get  @ChanB[] (%{{.*}}[] [] []) : (memref<32xi32, 2 : i32>)
                            # CHECK: air.channel.put  @ChanC[] (%{{.*}}[] [] []) : (memref<32xi32, 2 : i32>)
                            for _ in for_(shape[0] // 32):
                                ChannelGet("ChanA", [], tile_a)
                                ChannelGet("ChanB", [], tile_b)
                                elemwise_binary(tile_a, tile_b, outs=[tile_c],
                                                fun=BinaryFn.mul,
                                                cast=TypeFn.cast_unsigned)
                                ChannelPut("ChanC", [], tile_c)
                                yield_([])
                            DeallocOp(tile_a)
                            DeallocOp(tile_b)
                            DeallocOp(tile_c)
                            HerdTerminatorOp()
                        SegmentTerminatorOp()
                    LaunchTerminatorOp()
        return module

module = build_module([1024], np.int32, np.int32)
print(module)