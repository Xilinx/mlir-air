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

    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # ChannelPut("ChanIn", a)
            # ChannelGet("ChanOut", b)

            # The arguments are still the input and the output
            @segment(name="seg")
            def segment_body():

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="copyherd", sizes=[1, 1])
                def herd_body(_tx, _ty, _sx, _sy):

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # This is the type definition of the image
                    image_type = MemRefType.get(
                        shape=IMAGE_SIZE,
                        element_type=T.i32(),
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of image size for the input/output
                    image_in = AllocOp(image_type, [], [])
                    image_out = AllocOp(image_type, [], [])

                    """
                    # Place the input image (a) into the L1 memory region
                    ChannelGet("ChanIn", image_in)

                    # Access every value in the image
                    for j in range_(IMAGE_HEIGHT):
                        for i in range_(IMAGE_WIDTH):
                            # Load the input value
                            val = load(image_in, [i, j])

                            # Store the output value
                            store(val, image_out, [i, j])
                            yield_([])
                        yield_([])

                    # Copy the output image into the output
                    ChannelPut("ChanOut", image_out)
                    """

                    # Deallocate our L1 buffers
                    DeallocOp(image_in)
                    DeallocOp(image_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
