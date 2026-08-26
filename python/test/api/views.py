# ./python/test/api/views.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""A reshaped buffer is a transfer endpoint in both directions.

``reshape`` and ``transpose`` produce a view -- the same buffer walked with
different sizes and strides -- and a view has always been a valid *source* for
``ops.store``: ``store(t.transpose(1, 0), B[:, :])`` is the whole of
data_transfer_transpose. Filling one was refused, which read as a rule and was
an omission: the direction simply had no caller until rope_sincos, whose L1
tiles are shaped [3, head_size] because that is what rope.cc is compiled
against, while its L3 tensor is one flat run per head.

What a view does *not* do is excuse a shape mismatch. A buffer whose shape
genuinely disagrees with the region is still refused; the view is how you say
the two describe the same elements, not a way to stop being asked.
"""

from air import api as air
from air.api.types import bf16

HEADS, ROWS, COLS = 4, 3, 48
FLAT = ROWS * COLS


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def build(load_view=True, store_view=True):
    IN = air.tensor([HEADS * FLAT], bf16)
    OUT = air.tensor([HEADS * FLAT], bf16)

    with air.launch(name="v") as launch:

        @launch.body
        def _():
            with air.herd([range(HEADS)], name="herd_0", shape=(1,)) as h:

                @h.body
                def _(tx):
                    o = tx * FLAT
                    b = air.alloc([ROWS, COLS], bf16, scope=h.private())
                    air.ops.load(b.reshape(FLAT) if load_view else b, IN[o : o + FLAT])
                    air.ops.store(
                        b.reshape(FLAT) if store_view else b, OUT[o : o + FLAT]
                    )

    print(launch.build(target="npu1"))


# CHECK-LABEL: TEST: a_reshaped_destination_is_filled_in_place
# The buffer keeps the shape the kernel wants -- memref<3x48xbf16> -- while the
# transfer carries the shape the tensor has: one contiguous run of 144. Both
# directions are the same view, so both DMAs walk it the same way, and no
# memref.reshape/collapse_shape op is emitted at all: a view is sizes and
# strides on the transfer, not an operation on the buffer.
# CHECK: %[[B:.*]] = memref.alloc() : memref<3x48xbf16, 2 : i32>
# CHECK: air.dma_memcpy_nd (%[[B]][0] [144] [1], %{{.*}}[%{{.*}}] [144] [1])
# CHECK-SAME: (memref<3x48xbf16, 2 : i32>, memref<576xbf16>)
# CHECK: air.dma_memcpy_nd (%{{.*}}[%{{.*}}] [144] [1], %[[B]][0] [144] [1])
# CHECK-SAME: (memref<576xbf16>, memref<3x48xbf16, 2 : i32>)
# CHECK-NOT: memref.collapse_shape
# CHECK-NOT: memref.reshape
@run
def a_reshaped_destination_is_filled_in_place():
    build(load_view=True, store_view=True)


# CHECK-LABEL: TEST: without_the_view_the_mismatch_is_still_refused
# The view is how you say the buffer and the region describe the same elements.
# Handing over the buffer itself, whose shape genuinely disagrees, is the same
# mistake it always was -- and the message names both shapes rather than the
# rank.
# CHECK: ValueError: transfer shape mismatch in air.api.ops.load
# CHECK-SAME: destination buffer is (3, 48)
# CHECK-SAME: source tensor slice is (144,)
@run
def without_the_view_the_mismatch_is_still_refused():
    try:
        build(load_view=False)
    except ValueError as e:
        print(f"ValueError: {e}")
    else:
        print("ERROR: no exception raised")
