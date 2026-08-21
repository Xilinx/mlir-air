# ./python/test/api/unsigned.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""Unsigned element types: movable, and not computable.

MLIR spells signedness in the *operation* -- ``arith.divsi`` and ``arith.divui``
both take signless operands -- so the whole ``arith`` dialect, and every named
``linalg`` contraction whose region is built from it, rejects a ``ui8``. What
that leaves is everything that only moves bytes: a memref of any element type,
``air.dma_memcpy_nd``, ``air.channel``, and a ``func.call`` into a hand-written
kernel.

That is the whole surface a ``uint8`` kernel needs, and declaring ``i8`` instead
would be an untruth in the one place it is load-bearing: the emitted
``func.func private`` declaration is the signature aircc links the object file
against.

The refusals live in ``errors.py``; this file pins what does work.
"""

from air import api as air
from air.api.types import i32, ui8, ui16, ui32

N = 64
CHUNK = 16


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: extern_kernel_over_ui8
# The declaration aircc links against carries ui8, matching the uint8_t* the
# object file was compiled from -- this is the reason the type exists.
# CHECK: func.func private @passThroughLine(memref<16xui8, 2 : i32>, memref<16xui8, 2 : i32>, i32)
# CHECK-SAME: link_with = "passThrough.cc.o"
# CHECK: air.channel @In
# CHECK: air.channel @Out
# CHECK: func.func @k(%{{.*}}: memref<64xui8>, %{{.*}}: memref<64xui8>)
# The L3 endpoints keep the tensor's element type through the channel...
# CHECK: air.channel.put{{ *}}@In[] (%{{.*}}[] [] []) : (memref<64xui8>)
# ...and the L1 tiles are ui8 too, allocated in memory space 2.
# CHECK: %[[IN:.*]] = memref.alloc() : memref<16xui8, 2 : i32>
# CHECK: %[[OUT:.*]] = memref.alloc() : memref<16xui8, 2 : i32>
# CHECK: air.channel.get{{ *}}@In[] (%[[IN]][] [] []) : (memref<16xui8, 2 : i32>)
# CHECK: func.call @passThroughLine(%[[IN]], %[[OUT]], %{{.*}})
# CHECK: air.channel.put{{ *}}@Out[] (%[[OUT]][] [] []) : (memref<16xui8, 2 : i32>)
# CHECK: air.channel.get{{ *}}@Out[] (%{{.*}}[] [] []) : (memref<64xui8>)
@run
def extern_kernel_over_ui8():
    A = air.tensor([N], ui8)
    B = air.tensor([N], ui8)
    chan_in = air.channel("In")
    chan_out = air.channel("Out")
    line = air.extern("passThroughLine", object="passThrough.cc.o", scalars=[i32])

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    chan_in.put(A)

                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            tile_in = air.alloc([CHUNK], ui8, scope=h.private())
                            tile_out = air.alloc([CHUNK], ui8, scope=h.private())
                            for _i in air.sequential(0, N // CHUNK):
                                chan_in.get(tile_in)
                                line(tile_in, tile_out, CHUNK)
                                chan_out.put(tile_out)

                    chan_out.get(B)

    print(launch.mlir())


# CHECK-LABEL: TEST: elementwise_copy_is_scalar
# A copy carries no operator, so it emits memref.load/store and no arith op at
# all -- which is exactly the loop the hand-written uint8 examples spell out.
# The vector path is unavailable even here: vector.transfer_read takes a padding
# value, and that padding value is an arith.constant.
# The CHECK-NEXTs are the assertion: the copy loop's body is those two ops and
# nothing else, so neither a transfer_read nor its padding constant is there.
# CHECK: air.herd @copy
# CHECK: air.dma_memcpy_nd
# CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
# CHECK-NEXT: %[[V:.*]] = memref.load %{{.*}} : memref<16xui8, 2 : i32>
# CHECK-NEXT: memref.store %[[V]], %{{.*}} : memref<16xui8, 2 : i32>
# CHECK-NEXT: }
@run
def elementwise_copy_is_scalar():
    A = air.tensor([N], ui8)
    B = air.tensor([N], ui8)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd([range(0, N, CHUNK)], name="copy", shape=(1,)) as h:

                @h.body
                def _(tx):
                    src = air.alloc([CHUNK], ui8, scope=h.private())
                    dst = air.alloc([CHUNK], ui8, scope=h.private())
                    air.ops.load(src, A[tx : tx + CHUNK])
                    dst[:] = src[:]
                    air.ops.store(dst, B[tx : tx + CHUNK])

    print(launch.mlir())


# CHECK-LABEL: TEST: wider_unsigned_widths
# ui16 and ui32 travel the same path; the DMA's element count comes from the
# shape and its stride from the memref, so the width only has to reach the type.
# CHECK: func.func @k(%{{.*}}: memref<64xui16>, %{{.*}}: memref<64xui32>
# CHECK: memref.alloc() : memref<16xui16, 2 : i32>
# CHECK: memref.alloc() : memref<16xui32, 2 : i32>
@run
def wider_unsigned_widths():
    A = air.tensor([N], ui16)
    B = air.tensor([N], ui32)
    C = air.tensor([N], ui32)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd([range(0, N, CHUNK)], name="widths", shape=(1,)) as h:

                @h.body
                def _(tx):
                    narrow = air.alloc([CHUNK], ui16, scope=h.private())
                    wide = air.alloc([CHUNK], ui32, scope=h.private())
                    air.ops.load(narrow, A[tx : tx + CHUNK])
                    air.ops.load(wide, B[tx : tx + CHUNK])
                    air.ops.store(wide, C[tx : tx + CHUNK])

    print(launch.mlir())
