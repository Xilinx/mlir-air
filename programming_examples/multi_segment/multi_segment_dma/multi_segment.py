# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path  # if you haven't already done so

# Python paths are a bit complex. Taking solution from : https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_

from common import *


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(VECTOR_SIZE, T.i32())

    # We want to store our data in L1 memory
    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the tile
    image_type_l1 = MemRefType.get(
        shape=VECTOR_SIZE,
        element_type=T.i32(),
        memory_space=mem_space_l1,
    )

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut, memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1, arg2, arg3):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1, arg2, arg3])
        def launch_body(a, b, c, d):

            @segment(name="seg1", operands=[a, c])
            def segment_body(arg0, arg2):

                @herd(name="addherd1", sizes=[1, 1], operands=[arg0, arg2])
                def herd_body(tx, ty, sx, sy, a, c):

                    image_in_a = AllocOp(image_type_l1, [], [])
                    image_out_a = AllocOp(image_type_l1, [], [])

                    dma_memcpy_nd(image_in_a, a)

                    # Access every value in the tile
                    c0 = arith.ConstantOp.create_index(0)
                    for j in range_(VECTOR_LEN):
                        val_a = load(image_in_a, [c0, j])
                        val_outa = arith.addi(val_a, arith.constant(T.i32(), 10))
                        store(val_outa, image_out_a, [c0, j])
                        yield_([])

                    dma_memcpy_nd(c, image_out_a)
                    DeallocOp(image_in_a)
                    DeallocOp(image_out_a)

            @segment(name="seg2", operands=[b, d])
            def segment_body(arg1, arg3):

                @herd(name="addherd2", sizes=[1, 1], operands=[arg1, arg3])
                def herd_body(tx, ty, sx, sy, b, d):

                    image_in_b = AllocOp(image_type_l1, [], [])
                    image_out_b = AllocOp(image_type_l1, [], [])

                    dma_memcpy_nd(image_in_b, b)

                    # Access every value in the tile
                    c0 = arith.ConstantOp.create_index(0)
                    for j in range_(VECTOR_LEN):
                        val_b = load(image_in_b, [c0, j])
                        val_outb = arith.addi(arith.constant(T.i32(), 10), val_b)
                        store(val_outb, image_out_b, [c0, j])
                        yield_([])

                    dma_memcpy_nd(d, image_out_b)

                    DeallocOp(image_in_b)
                    DeallocOp(image_out_b)


if __name__ == "__main__":
    module = build_module()
    print(module)
