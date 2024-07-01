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
from air.dialects.linalg import elemwise_binary
from air.dialects.linalg.opdsl.lang import BinaryFn, TypeFn
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_


from common import *


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # We want to store our data in L1 memory
    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the tile
    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=T.i32(),
        memory_space=mem_space_l1,
    )

    # Create two channels which will send/receive the
    # input/output data respectively
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # Create a channel we will use to pass data between works in two herds
    ChannelOp("WorkerToWorker")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # Fetch all input data into the channel
            ChannelPut("ChanIn", a)

            # Push all output data out of the channel
            ChannelGet("ChanOut", b)

            @segment(name="seg")
            def segment_body():

                @herd(name="producer_herd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    # We must allocate a buffer of tilße size for the input/output
                    image_in = AllocOp(image_type_l1, [], [])
                    image_out = AllocOp(image_type_l1, [], [])

                    ChannelGet("ChanIn", image_in)

                    elemwise_binary(
                        image_in,
                        image_in,
                        outs=[image_out],
                        fun=BinaryFn.mul,
                        cast=TypeFn.cast_unsigned,
                    )

                    ChannelPut("WorkerToWorker", image_out)

                    DeallocOp(image_in)
                    DeallocOp(image_out)

                @herd(name="consumer_herd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    # We must allocate a buffer of image size for the input/output
                    image_in = AllocOp(image_type_l1, [], [])
                    image_out = AllocOp(image_type_l1, [], [])

                    ChannelGet("WorkerToWorker", image_in)

                    # Access every value in the image
                    for j in range_(IMAGE_HEIGHT):
                        for i in range_(IMAGE_WIDTH):
                            # Load the input value
                            val_in = load(image_in, [i, j])

                            # Calculate the output value
                            val_out = arith.addi(val_in, arith.ConstantOp(T.i32(), 1))

                            # Store the output value
                            store(val_out, image_out, [i, j])
                            yield_([])
                        yield_([])

                    ChannelPut("ChanOut", image_out)

                    DeallocOp(image_in)
                    DeallocOp(image_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
