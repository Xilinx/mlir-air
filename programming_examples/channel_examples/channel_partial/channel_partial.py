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
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    ChannelOp("ChanIn")
    ChannelOp("ChanOut")
    ChannelOp("MiddleChannel")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            ChannelPut("ChanIn", a)
            ChannelGet("ChanOut", b)

            @segment(name="seg")
            def segment_body():

                @herd(name="partial_herd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    # We want to store our data in L1 memory
                    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # Just get one little piece of data
                    tiny_tile_type = MemRefType.get(
                        shape=[1, 1],
                        element_type=T.i32(),
                        memory_space=mem_space_l1,
                    )

                    for i in range_(IMAGE_HEIGHT * IMAGE_WIDTH):
                        tiny_tile_in = AllocOp(tiny_tile_type, [], [])
                        tiny_tile_out = AllocOp(tiny_tile_type, [], [])

                        ChannelGet("ChanIn", tiny_tile_in)

                        # Load the input value from tile_in
                        index0 = arith.ConstantOp.create_index(0)
                        val_in = load(tiny_tile_in, [index0, index0])
                        val_out = arith.addi(val_in, arith.index_cast(T.i32(), i))
                        store(val_out, tiny_tile_out, [index0, index0])

                        ChannelPut("MiddleChannel", tiny_tile_out)

                        DeallocOp(tiny_tile_in)
                        DeallocOp(tiny_tile_out)
                        yield_([])

                    for i in range_(IMAGE_HEIGHT * IMAGE_WIDTH):
                        tiny_tile_in = AllocOp(tiny_tile_type, [], [])
                        tiny_tile_out = AllocOp(tiny_tile_type, [], [])

                        ChannelGet("MiddleChannel", tiny_tile_in)

                        # Load the input value from tile_in
                        index0 = arith.ConstantOp.create_index(0)
                        val_in = load(tiny_tile_in, [index0, index0])
                        val_out = arith.addi(val_in, arith.index_cast(T.i32(), i))
                        store(val_out, tiny_tile_out, [index0, index0])

                        ChannelPut("ChanOut", tiny_tile_out)

                        DeallocOp(tiny_tile_in)
                        DeallocOp(tiny_tile_out)
                        yield_([])


if __name__ == "__main__":
    module = build_module()
    print(module)
