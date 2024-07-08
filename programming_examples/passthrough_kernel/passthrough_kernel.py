# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.dialects.arith import constant

range_ = for_

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]

def external_func(name, inputs, outputs=None, visibility="private"):
    if outputs is None:
        outputs = []
    my_func = FuncOp(
        name=name, type=FunctionType.get(inputs, outputs), visibility=visibility
    )
    return my_func


# Wrapper for func CallOp.
class call(CallOp):
    """Specialize CallOp class constructor to take python integers"""

    def __init__(self, calleeOrResults, inputs=[], input_types=[]):
        attrInputs = []
        
        for (i, itype) in zip(inputs, input_types):
            if isinstance(i, int):
                attrInputs.append(constant(itype, i))
            else:
                attrInputs.append(i)
        if isinstance(calleeOrResults, FuncOp):
            super().__init__(
                calleeOrResults=calleeOrResults, argumentsOrCallee=attrInputs,
            )
        else:
            super().__init__(
                calleeOrResults=input_types,
                argumentsOrCallee=FlatSymbolRefAttr.get(calleeOrResults),
                arguments=attrInputs,
            )


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

            # The arguments are still the input and the output
            @segment(name="seg", operands=[a, b])
            def segment_body(arg0, arg1):

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="copyherd", sizes=[1, 1], operands=[arg0, arg1])
                def herd_body(tx, ty, sx, sy, a, b):

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    lineWidthInBytes = 4096 // 4

                    # This is the type definition of the image
                    image_type = MemRefType.get(
                        shape=[lineWidthInBytes],
                        element_type=T.ui8(),
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of image size for the input/output
                    image_in = AllocOp(image_type, [], [])
                    image_out = AllocOp(image_type, [], [])

                    # AIE Core Function declarations
                    memRef_ty = T.memref(lineWidthInBytes, T.ui8())
                    passThroughLineFunc = external_func(
                        "passThroughLine", inputs=[memRef_ty, memRef_ty, T.i32()]
                    )
                    call(passThroughLineFunc, [image_in, image_out, lineWidthInBytes])

                    """
                    # Access every value in the image
                    for j in range_(IMAGE_HEIGHT):
                        for i in range_(IMAGE_WIDTH):
                            # Load the input value
                            val = load(image_in, [i, j])

                            # Store the output value
                            store(val, image_out, [i, j])
                            yield_([])
                        yield_([])
                    """

                    # Deallocate our L1 buffers
                    DeallocOp(image_in)
                    DeallocOp(image_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
