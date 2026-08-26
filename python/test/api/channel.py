# ./python/test/api/channel.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api moves data through named channels.

A channel differs from ``ops.load``/``ops.store`` in two ways that this file is
written to pin down.

**It crosses scopes without being an operand.** ``air.herd`` and ``air.segment``
are ``IsolatedFromAbove``, so the DSL threads staged L2 buffers in explicitly. A
channel needs none of that: it is a module-level ``Symbol``, so the herd's
``get`` finds what the segment's ``put`` sent by name alone.

**Put and get sizes need not match.** Below, one put of the whole ``[4096]``
tensor feeds four gets of ``[1024]`` -- a channel is a stream, and each get
takes the next chunk. This is the shape ``passthrough_channel`` uses, and
applying ``ops.load``'s shape-equality rule to it would reject it.

An endpoint in L3 has to sit inside an ``air.segment``: reaching L3 needs a shim
DMA allocation, and outside a segment there is none to link to. That is measured
rather than assumed -- see ``_channel.py`` -- and ``errors.py`` pins the message.
"""

from air import api as air
from air.api.types import i8, i32

N = 4096
SUB = 4


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: plain_channels
#
# Both channels are declared at module scope, ahead of the func, and in the
# order they were written -- they are materialised on first use, which happens
# deep inside the body, so the ordering is deliberate rather than incidental.
# CHECK: air.channel @ChanIn []
# CHECK: air.channel @ChanOut []
# CHECK: func.func @copy(%{{.*}}: memref<4096xi8>, %{{.*}}: memref<4096xi8>)
#
# The put is at segment scope and sends the whole tensor: no access pattern, so
# the op prints [] [] [].
# CHECK: air.segment @seg
# CHECK: air.channel.put @ChanIn[] (%{{.*}}[] [] []) : (memref<4096xi8>)
#
# The herd carries no channel operand -- only the two tensors the DSL passes
# everywhere -- yet its get resolves @ChanIn by name.
# CHECK: air.herd @copyherd
# CHECK: scf.for
# CHECK: air.channel.get @ChanIn[] (%{{.*}}[] [] []) : (memref<1024xi8, 2 : i32>)
# CHECK: air.channel.put @ChanOut[] (%{{.*}}[] [] []) : (memref<1024xi8, 2 : i32>)
#
# ...and the matching get is back at segment scope, into L3.
# CHECK: air.channel.get @ChanOut[] (%{{.*}}[] [] []) : (memref<4096xi8>)
@run
def plain_channels():
    A = air.tensor([N], i8)
    B = air.tensor([N], i8)
    cin = air.channel("ChanIn")
    cout = air.channel("ChanOut")

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    cin.put(A)
                    with air.herd([range(1)], name="copyherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            tin = air.alloc([N // SUB], i8, scope=h.private())
                            tout = air.alloc([N // SUB], i8, scope=h.private())
                            for _i in air.sequential(0, SUB):
                                cin.get(tin)
                                tout[:] = tin[:]
                                cout.put(tout)

                    cout.get(B)

    print(launch.mlir())


# CHECK-LABEL: TEST: channel_slice
# A region endpoint reuses the same slice machinery ops.load/store use, so the
# offsets, sizes and strides land on the channel op unchanged.
# CHECK: air.channel.put @Tiles[] (%{{.*}}[0, 8] [16, 8] [64, 1]) : (memref<32x64xi32>)
@run
def channel_slice():
    A = air.tensor([32, 64], i32)
    B = air.tensor([16, 8], i32)
    tiles = air.channel("Tiles")
    back = air.channel("Back")

    with air.launch(name="sliced") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    tiles.put(A[0:16, 8:16])
                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([16, 8], i32, scope=h.private())
                            tiles.get(buf)
                            back.put(buf)

                    back.get(B)

    print(launch.mlir())


# CHECK-LABEL: TEST: channel_array
# size= makes the channel an array, and indices= selects one of its members.
# The herd coordinate is a legitimate index, so the subscript is dynamic.
# CHECK: air.channel @Grid [2, 2]
# CHECK: air.channel.put @Grid[%c0{{.*}}, %c1{{.*}}] (%{{.*}}[0, 8] [16, 8] [64, 1])
# CHECK: air.channel.get @Grid[%{{.*}}, %{{.*}}] (%{{.*}}[] [] [])
@run
def channel_array():
    A = air.tensor([32, 64], i32)
    B = air.tensor([16, 8], i32)
    grid = air.channel("Grid", size=[2, 2])
    back = air.channel("Back")

    with air.launch(name="arrayed") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    grid.put(A[0:16, 8:16], indices=[0, 1])
                    with air.herd([range(2), range(2)], name="h", shape=(2, 2)) as h:

                        @h.body
                        def _(tx, ty):
                            buf = air.alloc([16, 8], i32, scope=h.private())
                            grid.get(buf, indices=[tx, ty])
                            back.put(buf)

                    back.get(B)

    print(launch.mlir())


# CHECK-LABEL: TEST: channel_broadcast
# broadcast_shape is a declaration-level attribute: one put fans out to the
# three gets that follow, each naming its own destination with indices=.
# CHECK: air.channel @Bcast [1, 1] {broadcast_shape = [1 : index, 3 : index]}
# CHECK: air.channel.put @Bcast[] (%{{.*}}[] [] []) : (memref<64xi32>)
# CHECK: air.channel.get @Bcast[%{{.*}}, %{{.*}}] (%{{.*}}[] [] []) : (memref<64xi32, 2 : i32>)
#
# Each core adds its own coordinate, so the fan-out is observable rather than
# merely asserted. A herd coordinate broadcast into an elementwise expression
# materialises as an affine.apply and an index_cast to the buffer's element
# type -- which is what the hand-written broadcast example spells by hand.
# CHECK: affine.apply
# CHECK: arith.index_cast %{{.*}} : index to i32
# CHECK: arith.addi
@run
def channel_broadcast():
    A = air.tensor([64], i32)
    B = air.tensor([64], i32)
    bcast = air.channel("Bcast", size=[1, 1], broadcast_shape=[1, 3])
    back = air.channel("Back")

    with air.launch(name="fanout") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    bcast.put(A)
                    with air.herd([range(1), range(3)], name="h", shape=(1, 3)) as h:

                        @h.body
                        def _(tx, ty):
                            buf = air.alloc([64], i32, scope=h.private())
                            out = air.alloc([64], i32, scope=h.private(), vector=0)
                            bcast.get(buf, indices=[tx, ty])
                            out[:] = buf[:] + ty
                            back.put(out)

                    back.get(B)

    print(launch.mlir())


# CHECK-LABEL: TEST: whole_tensor_transfer
# Channels take a bare tensor, and they share ops._endpoint with load/store, so
# load/store take one too: `ops.load(buf, A)` is the whole of A. Pinned here
# because it is a widening of those two, not only of the channel ops -- the
# shape check still applies (errors.py) and store still marks its tensor an
# output, which is what fixes the kernel's calling convention.
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[] [] []) : (memref<4x64xi32, 2 : i32>, memref<4x64xi32>)
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[] [] []) : (memref<4x64xi32>, memref<4x64xi32, 2 : i32>)
@run
def whole_tensor_transfer():
    A = air.tensor([4, 64], i32)
    B = air.tensor([4, 64], i32)

    with air.launch(name="whole") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="h", shape=(1,)) as h:

                @h.body
                def _(tx):
                    buf = air.alloc([4, 64], i32, scope=h.private())
                    air.ops.load(buf, A)
                    air.ops.store(buf, B)

    print(launch.mlir())


# CHECK-LABEL: TEST: channel_pack
# A put walks a flat L2 region block-first, so the DMA performs the pack and the
# consumer does a plain whole-buffer get. The walk is written where it happens,
# with reshape and transpose on the region -- put and get are separate ops with
# the blocked side at the other end of the stream, so there is nothing else the
# channel could derive it from.
#
# Splitting a [K, N] region into (K/k, k, N/n, n) and permuting to
# [N/n, K/k, k, n] gives strides [n, k*N, N, 1] -- here [6, 6, 16, 8] over
# [8, 768, 48, 1] for a [96, 48] region, which is the same walk the
# hand-written kernel spells collapsed as [6, 96, 8] over [8, 48, 1].
# CHECK: air.channel.put @Bpack[] (%{{.*}}[0, 0, 0, 0] [6, 6, 16, 8] [8, 768, 48, 1])
# CHECK: air.channel.get @Bpack[] (%{{.*}}[] [] [])
@run
def channel_pack():
    TILE_K, TILE_N = 96, 48
    MM_K, MM_N = 16, 8

    A = air.tensor([TILE_K, TILE_N], i8)
    Out = air.tensor([TILE_K, TILE_N], i8)
    pack = air.channel("Bpack")

    with air.launch(name="packed") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2_b = air.alloc([TILE_K, TILE_N], i8, scope=seg.private())
                    air.ops.load(l2_b, A)
                    pack.put(
                        l2_b[0:TILE_K, 0:TILE_N]
                        .reshape(TILE_K // MM_K, MM_K, TILE_N // MM_N, MM_N)
                        .transpose(2, 0, 1, 3)
                    )

                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            l1_b = air.alloc([6, 6, 16, 8], i8, scope=h.private())
                            pack.get(l1_b)

                    air.ops.store(l2_b, Out)

    print(launch.mlir())


# CHECK-LABEL: TEST: l3_endpoint_after_a_segment_reenters_the_launch
# A gridless launch opens its region lazily, so the segment is what opens it and
# a get written afterwards is back out at trace scope. It has to step into the
# region anyway, because reaching L3 needs the shim DMA allocation the launch
# brings; without that it lands at func scope and aircc reports "failed to link
# to any shim dma allocation".
#
# The endpoint is resolved inside that region rather than before choosing it.
# Resolving early would materialise the slice's offset arithmetic out here and
# then again inside, orphaning the first copy at func scope. With a constant
# offset there is no arithmetic to orphan, which is why the pin is on the op's
# position: the get is the last thing in the launch body, after air.segment.
# CHECK: air.launch
# CHECK: air.segment @seg
# CHECK: air.channel.get @drain[] (%{{.*}}[32] [32] [1]) : (memref<64xi32>)
@run
def l3_endpoint_after_a_segment_reenters_the_launch():
    A = air.tensor([64], i32)
    B = air.tensor([64], i32)
    feed = air.channel("feed")
    drain = air.channel("drain")

    with air.launch(name="reentry") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([32], i32, scope=h.private())
                            air.ops.load(buf, A[0:32])
                            feed.put(buf)
                            drain.put(buf)

                    l2 = air.alloc([32], i32, scope=seg.private())
                    feed.get(l2)

            drain.get(B[32:64])

    print(launch.mlir())


# CHECK-LABEL: TEST: a_computed_offset_cannot_follow_the_segment
# Same position, but the offset is built from a loop variable that lives outside
# the launch. Moving the op into the region cannot take that value with it --
# air.launch is IsolatedFromAbove -- so this is refused by name rather than left
# to fail as "'affine.apply' op using value defined outside the region", which
# reads as a DSL bug instead of as the shape of the program.
# CHECK: RuntimeError: air.channel.get on an L3 tensor slice whose offset is
# CHECK-SAME: computed from a coordinate or loop variable
@run
def a_computed_offset_cannot_follow_the_segment():
    B = air.tensor([64], i32)
    drain = air.channel("drain2")

    with air.launch(name="computed") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([32], i32, scope=h.private())
                            buf[:] = 0
                            drain.put(buf)

            for i in air.sequential(2):
                drain.get(B[i * 32 : i * 32 + 32])

    try:
        launch.mlir()
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    else:
        print("ERROR: no exception raised")


# CHECK-LABEL: TEST: a_parallel_loop_indexes_a_channel_bundle
# Staging that fans a memtile buffer out to a row of cores has to be parallel
# twice over, and air.sequential is wrong on both counts. The trip index names
# one slot of the bundle, which air-place-herds refuses to take from a temporal
# loop ("channel bundle indices must not be temporal scf.for induction
# variables"); and the trips share one set of buffer descriptors, where a Python
# `for` would unroll into that many independent DMAs.
#
# Emitted as scf.forall, which the pipeline's scf-forall-to-parallel turns into
# the scf.parallel the hand-written examples spell directly. One put, inside the
# loop -- not four.
# CHECK: scf.forall (%[[COL:.*]]) in (4) {
# CHECK: air.channel.put @fan[%{{.*}}, %{{.*}}] (%{{.*}}[0, %{{.*}}] [8, 8] [32, 1])
# CHECK: air.herd @h
# CHECK: air.channel.get @fan[%{{.*}}, %{{.*}}]
@run
def a_parallel_loop_indexes_a_channel_bundle():
    A = air.tensor([8, 32], i32)
    Out = air.tensor([8, 32], i32)
    fan = air.channel("fan", size=[4, 1])
    back = air.channel("back", size=[4, 1])

    with air.launch(name="fanout") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    staged = air.alloc([8, 32], i32, scope=seg.private())
                    air.ops.load(staged, A)

                    for col in air.parallel(4):
                        lo = col * 8
                        fan.put(staged[:, lo : lo + 8], indices=[col, 0])

                    with air.herd([range(4)], name="h", shape=(4,)) as h:

                        @h.body
                        def _(tx):
                            tile = air.alloc([8, 8], i32, scope=h.private())
                            fan.get(tile, indices=[tx, 0])
                            back.put(tile, indices=[tx, 0])

                    for col in air.parallel(4):
                        lo = col * 8
                        back.get(staged[:, lo : lo + 8], indices=[col, 0])
                    air.ops.store(staged, Out)

    print(launch.mlir())
