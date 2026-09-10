# ./python/test/api/switch_region.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""``ops.switch``'s region form, and its index-typed value form.

``switch.py`` covers the expression: ``ops.switch(k, [a, b])`` picks a number.
This covers the two forms added for the shared ``fused_decode`` builder.

**Region form** -- ``ops.switch(k)`` with no values -- runs statements, and
lowers to a single-case ``scf.index_switch``. It exists because the op a
position needs is not always the op its control flow suggests: an N-way
statement is nested ``ops.branch``, which is where this was first routed, but
that emits an ``scf.if``, leaving a region that must be an ``scf.index_switch``
with no spelling.

**Index form** -- the value form typed as ``index`` rather than as a buffer
element, so a switch can bound an ``air.sequential`` or offset a region.

The shapes checked here are the ones the shared ``fused_decode`` builder emits,
which is why they are pinned rather than described: ten ``llms/*_q4*`` models
compile from that one builder.
"""

from air import api as air
from air.api import ops
from air.api.types import bf16

W, NW, UNI_DEC = 256, 4, 2


def gated_launch():
    """A repeated launch, gated by switch regions holding L3 channel traffic.

    The whole shape the shared fused_decode builder uses: an scf.for around
    air.launch (`repeat=`), a unit attribute on the launch, switch regions
    gating the L3 puts, and a switch bounding a loop inside the herd.
    """
    A = air.tensor([NW * W], bf16)
    D = air.tensor([NW * W], bf16)
    ch = air.channel("toc")
    bk = air.channel("back")

    with air.launch(
        repeat=NW, name="gated", attrs=["air.preserve_shim_dma_order"]
    ) as lch:

        @lch.body
        def _(wave):
            # Both arms populated: the body is the default, otherwise() fills
            # the case 0 that __enter__ builds either way. Still one case
            # region -- there is no way to ask for a second.
            with ops.switch(wave < UNI_DEC) as arm:
                ch.put(A[wave * W : wave * W + W])
            with arm.otherwise():
                ch.put(A[0:W])
            with air.segment(name="s") as seg:

                @seg.body
                def _():
                    # The gate key, evaluated once here and carried into the
                    # herd as i32 -- an index-typed one folds to constant 0.
                    arm = air.rtp(wave < UNI_DEC)
                    # A second parameter: a per-dispatch trip count, which is
                    # what an attention block loop over ceil(L/16) needs. Two
                    # live parameters make the threading ambiguous, so the herd
                    # has to name them -- and their order here is their operand
                    # order.
                    trips = air.rtp(wave + 1)
                    l1 = air.alloc([W], bf16, scope=seg.per_core())
                    with air.herd(
                        [range(1), range(1)],
                        name="h",
                        at=(0, 2),
                        params=[trips, arm],
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            with ops.switch(arm):
                                pass
                            ch.get(l1)
                            # A runtime trip count: the index form of a switch,
                            # emitted above the loop it bounds.
                            for _k in air.sequential(ops.switch(tx, [1, 4])):
                                l1[:] = l1[:] * 2.0
                            # And one straight from an air.rtp parameter. The
                            # bound coercion covers both, so neither needs an
                            # explicit .as_index() at the call site.
                            for _k in air.sequential(trips):
                                l1[:] = l1[:] + 1.0
                            bk.put(l1)

            with ops.switch(wave < UNI_DEC):
                bk.get(D[wave * W : wave * W + W])

    return lch.build(target="npu2")


# repeat= is an scf.for AROUND air.launch, with the dispatch index threaded in
# as the last launch operand -- one device driven N times. A grid would be N
# devices: spatial, one segment per point, and the 5-bit dma_bd Packet ID field
# caps that at about four launches.
# CHECK-LABEL: func.func @gated
# CHECK: scf.for %[[WAVE:.*]] = %{{.*}} to %{{.*}} step
# CHECK: air.launch {{.*}}%[[ARG:.*]]=%[[WAVE]]) :{{.*}}, index attributes {air.preserve_shim_dma_order}
#
# A Condition reaches the switch as select(pred, 1, 0) then an index_cast, not
# as a raw i1. That is not a detour: cloneL2AndL3MemcpysToDeviceOp pins
# INDEX-typed segment arguments to constant 0, so a key built as an index inside
# a segment would fold to one arm and erase every other branch.
# CHECK: %[[P:.*]] = arith.cmpi slt
# CHECK: %[[S:.*]] = arith.select %[[P]], %{{.*}}, %{{.*}} : i32
# CHECK: %[[K:.*]] = arith.index_cast %[[S]] : i32 to index
# CHECK: scf.index_switch %[[K]]
# The body is the default; case 0 is otherwise(), or empty if unclaimed. One
# case region, never two -- see SwitchRegion's docstring.
# CHECK-NEXT: case 0 {
# CHECK-NEXT: air.channel.put @toc[] (%{{.*}}[0]
# CHECK-NEXT: scf.yield
# CHECK-NEXT: }
# CHECK-NEXT: default {
# CHECK: air.channel.put @toc
# The dispatch index crosses into the segment: air.herd and air.segment are both
# IsolatedFromAbove, so it is an operand, not a reference.
# CHECK: air.segment @s {{.*}}args(%{{.*}}=%{{.*}}) : index
# air.rtp crosses into the herd as i32, and is read back with an index_cast.
# Two runtime parameters, in the order params= names them.
# CHECK: air.herd @h {{.*}}, i32, i32
# CHECK: %[[K:.*]] = arith.index_cast %{{.*}} : i32 to index
# The identity affine.apply between the cast and the switch is the DSL's
# convention for every index it builds, and canonicalises away.
# CHECK: %[[KB:.*]] = affine.apply {{.*}}[%[[K]]]
# CHECK: scf.index_switch %[[KB]]
# The index form yields an index and bounds the loop; the switch sits above the
# scf.for rather than inside it. The identity affine.apply between the two is
# the DSL's existing convention for every index it builds, and canonicalises
# away -- it is not particular to a switch.
# CHECK: %[[N:.*]] = scf.index_switch %{{.*}} -> index
# CHECK: %[[NB:.*]] = affine.apply {{.*}}[%[[N]]]
# CHECK: scf.for %{{.*}} = %{{.*}} to %[[NB]]
# An air.rtp parameter bounds a loop the same way, with no .as_index() at the
# call site: air.sequential coerces whatever coerce_index accepts. Without that
# the DSL refused the bound outright, and blamed a tile coordinate for it.
# CHECK: %[[T:.*]] = arith.index_cast %{{.*}} : i32 to index
# CHECK: %[[TB:.*]] = affine.apply {{.*}}[%[[T]]]
# CHECK: scf.for %{{.*}} = %{{.*}} to %[[TB]]
print(gated_launch())


def refusals():
    """The three things the new surface must refuse, and say why."""
    D = air.tensor([W], bf16)

    with air.launch(grid=[range(NW)], name="refuse") as lch:

        @lch.body
        def _(wave):
            with air.segment(name="s") as seg:

                @seg.body
                def _():
                    l1 = air.alloc([W], bf16, scope=seg.per_core())
                    with air.herd([range(1), range(1)], name="h", at=(0, 2)) as h:

                        @h.body
                        def _(tx, ty):
                            # otherwise() fills case 0, and there is only one
                            # of those -- a second case region is what breaks
                            # air-to-aie's L2 receiver allocation.
                            g = ops.switch(tx == 0)
                            with g:
                                l1[:] = 1.0
                            with g.otherwise():
                                l1[:] = 2.0
                            try:
                                g.otherwise()
                            except RuntimeError as e:
                                print("OTHERWISE:", e)
                            # An index arm has to be a whole number.
                            try:
                                ops.switch(tx, [1.5, 2.5]).as_index()
                            except TypeError as e:
                                print("FRACTIONAL:", e)
                            # The region form still refuses a buffer key.
                            try:
                                ops.switch(l1)
                            except TypeError as e:
                                print("BUFFER:", type(e).__name__)
                            ops.store(l1, D[0:W])

    lch.build(target="npu2")


# CHECK: OTHERWISE: this ops.switch region already has an otherwise()
# CHECK: FRACTIONAL: air.api.ops.switch: 1.5 is not an integer
# CHECK-SAME: used as an index
# CHECK: BUFFER: TypeError
refusals()
