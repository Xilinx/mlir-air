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

CHANNEL_SIZE = [1, 2]
NUM_CHANNELS = CHANNEL_SIZE[0] * CHANNEL_SIZE[1]
assert IMAGE_SIZE[0] // CHANNEL_SIZE[0] == 0
assert IMAGE_SIZE[1] // CHANNEL_SIZE[1] == 0

PARTIAL_IMAGE_SIZE = [
    IMAGE_SIZE[0] // CHANNEL_SIZE[0],
    IMAGE_SIZE[1] // CHANNEL_SIZE[1],
]


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # We want to store our data in L1 memory
    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the tile
    partial_image_type_l1 = MemRefType.get(
        shape=PARTIAL_IMAGE_SIZE,  # Process a portion of the image at a time
        element_type=T.i32(),
        memory_space=mem_space_l1,
    )

    # Create two channels which will send/receive the
    # input/output data respectively
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # Create channel(s) we will use to pass data between workers in two herds
    ChannelOp("Herd2Herd", size=CHANNEL_SIZE)

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

                    for channel_index0 in range_(CHANNEL_SIZE[0]):
                        for channel_index1 in range_(CHANNEL_SIZE[1]):
                            # We must allocate a buffer of tile size for the input/output
                            partial_image_in = AllocOp(partial_image_type_l1, [], [])
                            partial_image_out = AllocOp(partial_image_type_l1, [], [])

                            ChannelGet("ChanIn", partial_image_in)

                            elemwise_binary(
                                partial_image_in,
                                partial_image_in,
                                outs=[partial_image_out],
                                fun=BinaryFn.mul,
                                cast=TypeFn.cast_unsigned,
                            )

                            ChannelPut(
                                "Herd2Herd",
                                partial_image_out,
                                indices=[channel_index0, channel_index1],
                            )

                            DeallocOp(partial_image_in)
                            DeallocOp(partial_image_out)

                            yield_([])
                        yield_([])

                @herd(name="consumer_herd", sizes=[1, 1])
                def herd_body(_tx, _ty, _sx, _sy):

                    for channel_index0 in range_(CHANNEL_SIZE[0]):
                        for channel_index1 in range_(CHANNEL_SIZE[1]):
                            # We must allocate a buffer of image size for the input/output
                            partial_image_in = AllocOp(partial_image_type_l1, [], [])
                            partial_image_out = AllocOp(partial_image_type_l1, [], [])

                            ChannelGet(
                                "Herd2Herd",
                                partial_image_in,
                                indices=[channel_index0, channel_index1],
                            )

                            # Access every value in the image
                            for j in range_(PARTIAL_IMAGE_SIZE[0]):
                                for i in range_(PARTIAL_IMAGE_SIZE[1]):
                                    # Load the input value
                                    # val_in = load(partial_image_in, [i, j])

                                    # Calculate the output value
                                    """
                                    val_out = arith.addi(
                                        val_in, arith.ConstantOp(T.i32(), 1)
                                    )
                                    """

                                    # TODO: for debugging, temporarily just set it to a current channel number
                                    val_out = arith.ConstantOp(
                                        T.i32(),
                                        channel_index0 * CHANNEL_SIZE[0]
                                        + channel_index1,
                                    )

                                    # Store the output value
                                    store(val_out, partial_image_out, [i, j])
                                    yield_([])
                                yield_([])

                            ChannelPut("ChanOut", partial_image_out)

                            DeallocOp(partial_image_in)
                            DeallocOp(partial_image_out)

                            yield_([])
                        yield_([])


if __name__ == "__main__":
    module = build_module()
    print(module)
