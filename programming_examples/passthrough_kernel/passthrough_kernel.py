# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.dialects.arith import constant

range_ = for_

NUM_VECTORS = 4


def external_func(name, inputs, outputs=None, visibility="private"):
    if outputs is None:
        outputs = []
    my_func = FuncOp(
        name=name, type=FunctionType.get(inputs, outputs), visibility=visibility
    )
    my_func.operation.attributes["link_with"] = StringAttr.get("passThrough.cc.o")
    my_func.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    return my_func


# Wrapper for func CallOp.
class call(CallOp):
    """Specialize CallOp class constructor to take python integers"""

    def __init__(self, calleeOrResults, inputs=[], input_types=[]):
        attrInputs = []

        for i, itype in zip(inputs, input_types):
            if isinstance(i, int):
                attrInputs.append(constant(itype, i))
            else:
                attrInputs.append(i)
        if isinstance(calleeOrResults, FuncOp):
            super().__init__(
                calleeOrResults=calleeOrResults,
                argumentsOrCallee=attrInputs,
            )
        else:
            super().__init__(
                calleeOrResults=input_types,
                argumentsOrCallee=FlatSymbolRefAttr.get(calleeOrResults),
                arguments=attrInputs,
            )


@module_builder
def build_module(vector_size):
    assert vector_size % NUM_VECTORS == 0

    # chop input in 4 sub-tensors
    lineWidthInBytes = vector_size // NUM_VECTORS

    # Type and method of input/output
    memrefTyInOut = T.memref(vector_size, T.ui8())
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # We want to store our data in L1 memory
    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the image
    tensor_type = MemRefType.get(
        shape=[lineWidthInBytes],
        element_type=T.ui8(),
        memory_space=mem_space,
    )

    passThroughLine = external_func(
        "passThroughLine", inputs=[tensor_type, tensor_type, T.i32()]
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

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="copyherd", sizes=[1, 1], link_with="passThrough.cc.o")
                def herd_body(tx, ty, sx, sy):

                    for i in range_(NUM_VECTORS):
                        # We must allocate a buffer of image size for the input/output
                        tensor_in = AllocOp(tensor_type, [], [])
                        tensor_out = AllocOp(tensor_type, [], [])

                        ChannelGet("ChanIn", tensor_in)

                        call(
                            passThroughLine,
                            inputs=[tensor_in, tensor_out, lineWidthInBytes],
                            input_types=[tensor_type, tensor_type, T.i32()],
                        )

                        ChannelPut("ChanOut", tensor_out)

                        # Deallocate our L1 buffers
                        DeallocOp(tensor_in)
                        DeallocOp(tensor_out)
                        yield_([])


if __name__ == "__main__":
    module = build_module()
    print(module)
