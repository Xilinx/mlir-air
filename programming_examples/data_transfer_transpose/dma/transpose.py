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
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_

from common import *


@module_builder
def build_module(m, k):
    memrefTyIn = MemRefType.get(shape=[m, k], element_type=T.i32())
    memrefTyOut = MemRefType.get(shape=[k, m], element_type=T.i32())

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def transpose(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # The arguments are still the input and the output
            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="herd", sizes=[1, 1], operands=[arg2, arg3])
                def herd_body(_tx, _ty, _sx, _sy, a, b):

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # This is the type definition of the tensor
                    tensor_type = MemRefType.get(
                        shape=[m * k],  # Read as one large array
                        element_type=T.i32(),
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    tensor_in = AllocOp(tensor_type, [], [])
                    tensor_out = AllocOp(tensor_type, [], [])

                    # The strides below are configured to read across all rows in the same column
                    # Stride of K in dim/wrap 2 skips an entire row to read a full column
                    dma_memcpy_nd(
                        tensor_in,
                        a,
                    )

                    dma_memcpy_nd(
                        b,
                        tensor_in,
                        src_sizes=[1, k, m],
                        src_strides=[1, 1, k],
                    )

                    # Deallocate our L1 buffer
                    DeallocOp(tensor_in)
                    DeallocOp(tensor_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
