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
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_

from common import *


@module_builder
def build_module(vector_size):
    assert vector_size % NUM_VECTORS == 0

    # chop input in 4 sub-tensors
    lineWidthInBytes = vector_size // NUM_VECTORS

    # Type and method of input/output
    memrefTyInOut = T.memref(vector_size, T.ui8())

    # We want to store our data in L1 memory
    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # This is the type definition of the image
    tensor_type = MemRefType.get(
        shape=[lineWidthInBytes],
        element_type=T.ui8(),
        memory_space=mem_space,
    )

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

                    for _i in range_(NUM_VECTORS):
                        # We must allocate a buffer of image size for the input/output
                        tensor_in = AllocOp(tensor_type, [], [])
                        tensor_out = AllocOp(tensor_type, [], [])

                        # Place the input image (a) into the L1 memory region
                        dma_memcpy_nd(tensor_in, a)

                        for j in range_(lineWidthInBytes):
                            # Load the input value
                            val = load(tensor_in, [j])

                            # Store the output value
                            store(val, tensor_out, [j])
                            yield_([])

                        dma_memcpy_nd(b, tensor_out)

                        # Deallocate our L1 buffers
                        DeallocOp(tensor_in)
                        DeallocOp(tensor_out)
                        yield_([])


if __name__ == "__main__":
    module = build_module()
    print(module)
