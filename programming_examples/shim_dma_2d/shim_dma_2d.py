# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects.affine import load, store
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_

range_ = for_


def build_module():  # try to add input arguments without the Insertion Pont from the module
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            memrefTyIn = MemRefType.get([32, 16], T.i32())

            @FuncOp.from_py_func(memrefTyIn, memrefTyIn)
            def copy(arg0, arg1):
                @launch(operands=[arg0, arg1])
                def launch_body(a, b):
                    @segment(name="seg", operands=[a, b])
                    def segment_body(arg2, arg3):
                        @herd(name="copyherd", sizes=[1, 1], operands=[arg2, arg3])
                        def herd_body(tx, ty, sx, sy, a, b):
                            mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                            tile_type = MemRefType.get(
                                shape=[16, 8],
                                element_type=T.i32(),
                                memory_space=mem_space,
                            )
                            buf0 = AllocOp(tile_type, [], [])
                            buf1 = AllocOp(tile_type, [], [])
                            dma_memcpy_nd(
                                buf0,
                                a,
                                src_offsets=[0, 0],
                                src_sizes=[8, 16],
                                src_strides=[32, 1],
                            )
                            for j in range_(8):
                                for i in range_(16):
                                    val = load(buf0, [i, j])
                                    store(val, buf1, [i, j])
                                    yield_([])
                                yield_([])
                            dma_memcpy_nd(
                                b,
                                buf1,
                                dst_offsets=[0, 0],
                                dst_sizes=[8, 16],
                                dst_strides=[32, 1],
                            )
                            DeallocOp(buf0)
                            DeallocOp(buf1)
                            HerdTerminatorOp()

                        SegmentTerminatorOp()

                    LaunchTerminatorOp()

        return module


if __name__ == "__main__":
    module = build_module()
    print(module)
