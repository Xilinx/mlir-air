# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # Create an input/output channel pair per worker
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")
    # ChannelOp("ToSelf")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            ChannelPut("ChanIn", a)
            ChannelGet("ChanOut", b)

            # The arguments are still the input and the output
            @segment(name="seg")
            def segment_body():

                @herd(name="xaddherd", sizes=[1, 1])
                def herd_body(_th, _tw, _sx, _sy):

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    image_type = MemRefType.get(
                        shape=IMAGE_SIZE,
                        element_type=T.i32(),
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    image_in = AllocOp(image_type, [], [])
                    image_out = AllocOp(image_type, [], [])
                    # image_in2 = AllocOp(image_type, [], [])
                    # image_out2 = AllocOp(image_type, [], [])

                    # Copy a tile from the input image (a) into the L1 memory region (image_in)
                    ChannelGet("ChanIn", image_in)

                    # Access every value in the time
                    for j in range_(IMAGE_HEIGHT):
                        for i in range_(IMAGE_WIDTH):
                            # Load the input value from image_in
                            val = load(image_in, [i, j])

                            # Store the output value in image_out
                            store(val, image_out, [i, j])
                            yield_([])
                        yield_([])

                    # Channel to self
                    # ChannelPut("ToSelf", image_out)
                    # ChannelGet("ToSelf", image_in2)

                    """
                    # Access every value in the tile
                    for j in range_(IMAGE_HEIGHT):
                        for i in range_(IMAGE_WIDTH):
                            # Load the input value from image_in
                            val = load(image_in2, [i, j])

                            # Store the output value in image_out
                            store(val, image_out2, [i, j])
                            yield_([])
                        yield_([])
                    """

                    ChannelGet("ChanOut", image_out)

                    # Deallocate our L1 buffers
                    DeallocOp(image_in)
                    DeallocOp(image_out)
                    # DeallocOp(image_in2)
                    # DeallocOp(image_out2)


if __name__ == "__main__":
    module = build_module()
    print(module)
