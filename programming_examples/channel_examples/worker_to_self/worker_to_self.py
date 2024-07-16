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

    # Type and method of input/output
    memrefTyInOut = T.MemRefType.get(IMAGE_SIZE, T.i32())
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")
    ChannelOp("ToSelf")

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=T.i32(),
        memory_space=mem_space_l1,
    )

    mem_space_l2 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    image_type_l2 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=T.i32(),
        memory_space=mem_space_l2,
    )

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

                tensor_in_l2 = AllocOp(image_type_l2, [], [])
                tensor_out_l2 = AllocOp(image_type_l2, [], [])

                ChannelGet("ChanIn", tensor_in_l2)
                ChannelPut("ChanOut", tensor_out_l2)

                DeallocOp(tensor_in_l2)
                DeallocOp(tensor_out_l2)

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(
                    name="copyherd",
                    sizes=[1, 1],
                    operands=[tensor_in_l2, tensor_out_l2],
                )
                def herd_body(tx, ty, sx, sy, tensor_in_l2, tensor_out_l2):

                    # We must allocate a buffer of image size for the input/output
                    tensor_in_l1 = AllocOp(image_type_l1, [], [])
                    tensor_out_l1 = AllocOp(image_type_l1, [], [])

                    ChannelPut("ToSelf", tensor_in_l2)
                    ChannelGet("ToSelf", tensor_in_l1)

                    # Access every value in the tile
                    for j in range_(IMAGE_HEIGHT):
                        for i in range_(IMAGE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tensor_in_l1, [i, j])

                            # Store the output value in tile_out
                            store(val, tensor_out_l1, [i, j])
                            yield_([])
                        yield_([])

                    ChannelPut("ToSelf", tensor_out_l1)
                    ChannelGet("ToSelf", tensor_out_l2)

                    # Deallocate our L1 buffers
                    DeallocOp(tensor_in_l1)
                    DeallocOp(tensor_out_l1)


if __name__ == "__main__":
    module = build_module()
    print(module)
