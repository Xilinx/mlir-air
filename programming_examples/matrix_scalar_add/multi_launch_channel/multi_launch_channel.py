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
from air.dialects.affine import apply as affine_apply

range_ = for_

from common import *


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # Manually unroll launch creation so the launch parameters are available as constants
        # (e.g., python vars during this code generation phase)
        for w in range(IMAGE_WIDTH // TILE_WIDTH):
            for h in range(IMAGE_HEIGHT // TILE_HEIGHT):
                ChannelOp(
                    "ChanIn",
                    size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH],
                )
                ChannelOp(
                    "ChanOut",
                    size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH],
                )

                @launch(operands=[arg0, arg1])
                def launch_body(a, b):
                    # Put data into the channel tile by tile
                    ChannelPut(
                        "ChanIn",
                        a,
                        indices=[w, h],
                        offsets=[w * IMAGE_HEIGHT, h * IMAGE_HEIGHT],
                        sizes=[TILE_HEIGHT, TILE_WIDTH],
                        strides=[IMAGE_WIDTH, 1],
                    )

                    # Write data back out to the channel tile by tile
                    ChannelGet(
                        "ChanOut",
                        b,
                        indices=[w, h],
                        offsets=[w * IMAGE_HEIGHT, h * IMAGE_HEIGHT],
                        sizes=[TILE_HEIGHT, TILE_WIDTH],
                        strides=[IMAGE_WIDTH, 1],
                    )

                    @segment(name="seg")
                    def segment_body():

                        @herd(name="xaddherd")
                        def herd_body(_tx, _ty, _sx, _sy):
                            # We want to store our data in L1 memory
                            mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                            # This is the type definition of the tile
                            tile_type = MemRefType.get(
                                shape=TILE_SIZE,
                                element_type=T.i32(),
                                memory_space=mem_space,
                            )

                            # We must allocate a buffer of tile size for the input/output
                            tile_in = AllocOp(tile_type, [], [])
                            tile_out = AllocOp(tile_type, [], [])

                            # Copy a tile from the input image (a) into the L1 memory region (tile_in)
                            ChannelGet("ChanIn", tile_in, indices=[w, h])

                            # Access every value in the tile
                            for j in range_(TILE_HEIGHT):
                                for i in range_(TILE_WIDTH):
                                    # Load the input value from tile_in
                                    val_in = load(tile_in, [i, j])

                                    val_out = arith.addi(
                                        val_in,
                                        arith.ConstantOp(
                                            T.i32(),
                                            h * (IMAGE_HEIGHT // TILE_HEIGHT) + w,
                                        ),
                                    )

                                    # Store the output value in tile_out
                                    store(
                                        val_out,
                                        tile_out,
                                        [i, j],
                                    )
                                    yield_([])
                                yield_([])

                            # Copy the output tile into the output
                            ChannelPut("ChanOut", tile_out, indices=[w, h])

                            # Deallocate our L1 buffers
                            DeallocOp(tile_in)
                            DeallocOp(tile_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
