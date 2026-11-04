# ./python/test/api/device_issue.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""An L3 slice can take its offset from an element of an L1 buffer.

``W[topk[i], :]`` reads ``topk[i]`` on the core and uses it as the row offset:
the core picks which rows of ``W`` to fetch. The compiler hoists the rest of the
transfer to the host and lets the core start it.
"""

from air import api as air
from air.api.types import bf16, i32, ui8

NE, E = 32, 256


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: expert_rows
#
# The element is read on the core and cast to an index, which becomes the row
# offset of the L3 put inside the router herd.
# CHECK: air.herd @router
# CHECK: %[[V:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<4xi32, 2 : i32>
# CHECK: %[[E:.*]] = arith.index_cast %[[V]] : i32 to index
# CHECK: %[[OFF:.*]] = affine.apply #{{.*}}()[%[[E]]]
# CHECK: air.channel.put @Wexp[%c{{.*}}] (%{{.*}}[%[[OFF]], 0] [1, 256] [256, 1]) : (memref<32x256xbf16>)
# CHECK: air.herd @mvm
# CHECK: air.channel.get @Wexp
@run
def expert_rows():
    W = air.tensor([NE, E], bf16)
    Y = air.tensor([E], bf16)
    wexp = air.channel("Wexp", size=[1])

    with air.launch(name="moe") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="router", shape=(1,)) as r:

                        @r.body
                        def _(tx):
                            topk = air.alloc([4], i32, scope=r.private())
                            wexp.put(W[topk[2], :], indices=[0])

                    with air.herd([range(1)], name="mvm", shape=(1,)) as m:

                        @m.body
                        def _(tx):
                            w = air.alloc([E], bf16, scope=m.private())
                            wexp.get(w, indices=[0])

    print(launch.mlir())


# CHECK-LABEL: TEST: needs_a_core
# CHECK: only its core can read it
@run
def needs_a_core():
    W = air.tensor([NE, E], bf16)
    wexp = air.channel("Wexp2", size=[1])
    with air.launch(name="moe2") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="router", shape=(1,)) as r:

                        @r.body
                        def _(tx):
                            global topk_outside
                            topk_outside = air.alloc([4], i32, scope=r.private())

                    wexp.put(W[topk_outside[0], :], indices=[0])

    try:
        launch.mlir()
    except TypeError as e:
        print(e)


# CHECK-LABEL: TEST: view_index
# CHECK: cannot read {{.*}} as an index: it is a reshaped or transposed view
@run
def view_index():
    W = air.tensor([NE, E], bf16)
    wexp = air.channel("Wexp3", size=[1])
    with air.launch(name="moe3") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="router", shape=(1,)) as r:

                        @r.body
                        def _(tx):
                            topk = air.alloc([2, 2], i32, scope=r.private())
                            wexp.put(W[topk.transpose(1, 0)[1, 0], :], indices=[0])

    try:
        launch.mlir()
    except TypeError as e:
        print(e)


# CHECK-LABEL: TEST: unsigned_index
# CHECK: an L1 element used as an index
@run
def unsigned_index():
    W = air.tensor([NE, E], bf16)
    wexp = air.channel("Wexp4", size=[1])
    with air.launch(name="moe4") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="router", shape=(1,)) as r:

                        @r.body
                        def _(tx):
                            topk = air.alloc([4], ui8, scope=r.private())
                            wexp.put(W[topk[0], :], indices=[0])

    try:
        launch.mlir()
    except NotImplementedError as e:
        print(e)


# CHECK-LABEL: TEST: l2_extent
#
# A whole L2 buffer takes a run-time outermost extent: its row-major pattern,
# cut to that many rows.
# CHECK: air.segment @seg
# CHECK: %[[N:.*]] = arith.index_cast %{{.*}} : i32 to index
# CHECK: air.channel.put @Kv[] (%{{.*}}[0, 0] [%{{.*}}, 256] [256, 1]) : (memref<8x256xi32, 1 : i32>)
@run
def l2_extent():
    L = air.tensor([], i32, name="L")
    kv = air.channel("Kv")
    with air.launch(name="l2x", repeat=1) as launch:

        @launch.body
        def _(w):
            n = air.rtp(L.value)
            with air.segment(name="seg", params=[n]) as seg:

                @seg.body
                def _():
                    buf = air.alloc([8, E], i32, scope=seg.private())
                    kv.put(buf, extent=n.as_index())
                    with air.herd([range(1)], name="fwd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            blk = air.alloc([E], i32, scope=h.private())
                            kv.get(blk)

    print(launch.mlir())
