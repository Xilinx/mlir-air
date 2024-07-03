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
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp


from common import *


@module_builder
def build_module(m, k):
    memrefTyIn = MemRefType.get(shape=[m, k], element_type=T.i32())
    memrefTyOut = MemRefType.get(shape=[k, m], element_type=T.i32())

    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def transpose(arg0, arg1):

        @launch(operands=[arg0, arg1])
        def launch_body(a, b):
            # Put data into the channel
            ChannelPut("ChanIn", a)

            # Write data back out to the channel
            ChannelGet("ChanOut", b)

            @segment(name="seg")
            def segment_body():

                @herd(name="herd", sizes=[1, 1])
                def herd_body(_tx, _ty, _sx, _sy):
                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # This is the type definition of the tensor
                    tensor_type = MemRefType.get(
                        shape=[k * m],  # Read as one large array
                        element_type=T.i32(),
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    tensor_in = AllocOp(tensor_type, [], [])

                    ChannelGet("ChanIn", tensor_in)
                    ChannelPut("ChanOut", tensor_in, sizes=[1, k, m], strides=[1, 1, k])

                    DeallocOp(tensor_in)


if __name__ == "__main__":
    module = build_module()
    print(module)
