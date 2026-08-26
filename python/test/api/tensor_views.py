# ./python/test/api/tensor_views.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""An L3 tensor reshapes and transposes, the same way a buffer does.

``reshape`` and ``transpose`` were buffer-only, which made the two ends of a
transfer unequal for no reason anyone had decided: the L1 side of a 4-D scatter
could be described as the shape the hardware walks and the L3 side could not.
conv2d_14x14 is the case -- its input is declared ``[4, 802816]`` and read as
``[4, 14, 64, 56]`` with strides ``[802816, 3584, 56, 1]``.

Both now come from one implementation. ``_StridedView`` holds reshape and
transpose over an (offsets, sizes, strides) triple and is mixed into both slice
types; ``_Reshapable`` holds the two-line delegation from a whole Tensor or
Buffer to a region of itself. The only thing either slice supplies is what kind
of region to build.
"""

from air import api as air
from air.api.types import i8


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: a_tensor_region_reshapes_into_the_walk_the_hardware_wants
# [4, 802816] sliced to [4, 50176] and re-described as [4, 14, 64, 56]. The
# strides are the giveaway that nothing moved: the outer axis keeps the tensor's
# own 802816, and the three inner ones are the row-major strides of the slice.
# This is conv2d_14x14's input scatter, which was previously inexpressible.
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0, 0, %{{.*}}] [4, 14, 64, 56] [802816, 3584, 56, 1])
@run
def a_tensor_region_reshapes_into_the_walk_the_hardware_wants():
    A = air.tensor([4, 802816], i8)
    OUT = air.tensor([200704], i8)

    with air.launch(name="scatter") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2 = air.alloc([4, 14, 3584], i8, scope=seg.private())

                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            for yp in air.sequential(16):
                                y0 = yp * 50176
                                air.ops.load(
                                    l2, A[:, y0 : y0 + 50176].reshape(4, 14, 64, 56)
                                )
                            b = air.alloc([256], i8, scope=h.private())
                            air.ops.store(b, OUT[0:256])

    print(launch.build(target="npu1"))


# CHECK-LABEL: TEST: a_tensor_transposes_like_a_buffer
# transpose permutes offsets, sizes and strides together, so a [64, 32] tensor
# read with its axes swapped is sizes [32, 64] and strides [1, 32] -- the same
# descriptor data_transfer_transpose builds from the L1 side.
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0] [32, 64] [1, 32])
@run
def a_tensor_transposes_like_a_buffer():
    A = air.tensor([64, 32], i8)
    OUT = air.tensor([2048], i8)

    with air.launch(name="t") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    b = air.alloc([32, 64], i8, scope=h.private())
                    air.ops.load(b, A.transpose(1, 0))
                    air.ops.store(b.reshape(2048), OUT[0:2048])

    print(launch.build(target="npu1"))
