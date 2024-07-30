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
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = T.i32()
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, xrt_dtype)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=xrt_dtype,
        memory_space=mem_space_l1,
    )

    ChannelOp("ChanIn")
    ChannelOp("ChanOut0")
    ChannelOp("ChanOut1")
    ChannelOp("ChanOut2")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut, memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1, arg2, arg3):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1, arg2, arg3])
        def launch_body(a, b, c, d):

            ChannelPut("ChanIn", a)
            ChannelGet("ChanOut0", b)
            ChannelGet("ChanOut1", c)
            ChannelGet("ChanOut2", d)

            @segment(name="seg")
            def segment_body():

                for herd_num in range(3):

                    @herd(name="broadcastherd" + str(herd_num), sizes=[1, 1])
                    def herd_body(_tx, _ty, _sx, _sy):

                        # We must allocate a buffer of image size for the input/output
                        image_in = AllocOp(image_type_l1, [], [])
                        image_out = AllocOp(image_type_l1, [], [])

                        ChannelGet("ChanIn", image_in)

                        # Access every value in the image
                        for i in range_(IMAGE_HEIGHT):
                            for j in range_(IMAGE_WIDTH):
                                # Load the input value
                                val_in = load(image_in, [i, j])

                                # Calculate the output value
                                val_out = arith.addi(
                                    val_in, arith.ConstantOp(T.i32(), herd_num + 1)
                                )

                                # Store the output value
                                store(val_out, image_out, [i, j])
                                yield_([])
                            yield_([])

                        ChannelPut("ChanOut" + str(herd_num), image_out)

                        DeallocOp(image_in)
                        DeallocOp(image_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
