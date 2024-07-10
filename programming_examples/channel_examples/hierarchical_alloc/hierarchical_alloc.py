# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_

IMAGE_WIDTH = 1
IMAGE_HEIGHT = 4
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    mem_space_l2 = IntegerAttr.get(T.i32(), MemorySpace.L2)
    mem_space_l3 = IntegerAttr.get(T.i32(), MemorySpace.L3)

    image_type_l1 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=T.i32(),
        memory_space=mem_space_l1,
    )
    image_type_l2 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=T.i32(),
        memory_space=mem_space_l2,
    )
    image_type_l3 = MemRefType.get(
        shape=IMAGE_SIZE,
        element_type=T.i32(),
        memory_space=mem_space_l3,
    )

    ChannelOp("ChanInL2")
    ChannelOp("ChanOutL2")
    ChannelOp("ChanInL1")
    ChannelOp("ChanOutL1")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):
            image_in_l3 = AllocOp(image_type_l3, [], [])

            # Do some data manipulation instead of a straight copy
            # 0, 1, 2, 3 -> 1, 1, 2, 2
            c0 = arith.constant(T.index(), 0)
            c1 = arith.constant(T.index(), 1)
            c2 = arith.constant(T.index(), 2)
            c3 = arith.constant(T.index(), 3)
            val1 = load(a, [c0, c1])
            store(val1, image_in_l3, [c0, c0])
            store(val1, image_in_l3, [c0, c1])
            val2 = load(a, [c0, c2])
            store(val2, image_in_l3, [c0, c2])
            store(val2, image_in_l3, [c0, c3])

            ChannelPut("ChanInL2", image_in_l3)
            DeallocOp(image_in_l3)

            ChannelGet("ChanOutL2", b)

            @segment(name="seg")
            def segment_body():
                image_in_l2a = AllocOp(image_type_l2, [], [])
                image_in_l2b = AllocOp(image_type_l2, [], [])

                ChannelGet("ChanInL2", image_in_l2a)

                # Do some data manipulation instead of a straight copy
                # 1, 1, 2, 2 -> 2, 1, 2, 2
                c0 = arith.constant(T.index(), 0)
                c1 = arith.constant(T.index(), 1)
                c2 = arith.constant(T.index(), 2)
                c3 = arith.constant(T.index(), 3)
                val1 = load(image_in_l2a, [c0, c0])
                store(val1, image_in_l2b, [c0, c1])
                val2 = load(image_in_l2a, [c0, c0])
                store(val1, image_in_l2b, [c0, c0])
                store(val2, image_in_l2b, [c0, c2])
                store(val1, image_in_l2b, [c0, c3])

                ChannelPut("ChanInL1", image_in_l2b)
                DeallocOp(image_in_l2a)
                DeallocOp(image_in_l2b)

                image_out_l2 = AllocOp(image_type_l2, [], [])
                ChannelGet("ChanOutL1", image_out_l2)
                ChannelPut("ChanOutL2", image_out_l2)
                DeallocOp(image_out_l2)

                @herd(name="addherd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    # We must allocate a buffer of image size for the input/output
                    image_in = AllocOp(image_type_l1, [], [])
                    image_out = AllocOp(image_type_l1, [], [])

                    ChannelGet("ChanInL1", image_in)

                    # Access every value in the image
                    for j in range_(IMAGE_HEIGHT):
                        for i in range_(IMAGE_WIDTH):
                            # Load the input value
                            val_in = load(image_in, [i, j])

                            # Calculate the output value
                            # 2, 1, 2, 2 -> 3, 2, 3, 3
                            val_out = arith.addi(val_in, arith.ConstantOp(T.i32(), 1))

                            # Store the output value
                            store(val_out, image_out, [i, j])
                            yield_([])
                        yield_([])

                    ChannelPut("ChanOutL1", image_out)

                    DeallocOp(image_in)
                    DeallocOp(image_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
