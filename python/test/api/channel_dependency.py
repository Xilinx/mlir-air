# ./python/test/api/channel_dependency.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""``dependency=`` on a channel ``put``/``get`` orders it after the named ops.

The DSL emits synchronous ops and leaves the asynchronous graph to
``air-dependency``, which derives it from program order and from the regions
each op touches. Where that is not enough -- a read of host memory that the
device wrote earlier in the same launch, at an offset known only at run time --
the builder names the dependency, and the two ops it relates become
asynchronous. Everything else stays synchronous.
"""

from air import api as air
from air.api.types import i32


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: readback_after_drain
# The herd's own traffic stays synchronous.
# CHECK: air.channel.put  @Tiles[]
# CHECK: air.channel.get  @Tiles[]
# CHECK: air.channel.put  @Back[]
# The drain becomes asynchronous, and the readback depends on its token.
# CHECK: %[[D:.*]] = air.channel.get async  @Back[] (%{{.*}}[0] [64] [1])
# CHECK: air.channel.put async [%[[D]]]  @Again[] (%{{.*}}[0] [64] [1])
@run
def readback_after_drain():
    A = air.tensor([64], i32)
    KV = air.tensor([256], i32)
    tiles = air.channel("Tiles")
    back = air.channel("Back")
    again = air.channel("Again")

    with air.launch(name="rb") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    tiles.put(A)
                    with air.herd([range(1)], name="h", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([64], i32, scope=h.private())
                            tiles.get(buf)
                            back.put(buf)

                    drained = back.get(KV[0:64])
                    again.put(KV[0:64], dependency=drained)

    print(launch.mlir())


# CHECK-LABEL: TEST: two_dependencies
# A list names several ops; each becomes asynchronous once.
# CHECK: %[[D0:.*]] = air.channel.get async  @Back0[]
# CHECK: %[[D1:.*]] = air.channel.get async  @Back1[]
# CHECK: air.channel.put async [%[[D0]], %[[D1]]]  @Again2[]
@run
def two_dependencies():
    A = air.tensor([64], i32)
    KV = air.tensor([256], i32)
    tiles = air.channel("Tiles2")
    back0 = air.channel("Back0")
    back1 = air.channel("Back1")
    again = air.channel("Again2")

    with air.launch(name="rb2") as launch:

        @launch.body
        def _():
            with air.segment(name="seg2") as seg:

                @seg.body
                def _():
                    tiles.put(A)
                    with air.herd([range(1)], name="h2", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([64], i32, scope=h.private())
                            tiles.get(buf)
                            back0.put(buf)
                            back1.put(buf)

                    d0 = back0.get(KV[0:64])
                    d1 = back1.get(KV[64:128])
                    again.put(KV[0:128], dependency=[d0, d1])

    print(launch.mlir())


# CHECK-LABEL: TEST: one_token_two_readers
# A token named by two ops is made asynchronous once; both depend on it.
# CHECK: %[[D:.*]] = air.channel.get async  @Back3[]
# CHECK-NOT: air.channel.get{{.*}}@Back3
# CHECK: air.channel.put async [%[[D]]]  @Again3[] (%{{.*}}[0] [32] [1])
# CHECK: air.channel.put async [%[[D]]]  @Again3[] (%{{.*}}[32] [32] [1])
@run
def one_token_two_readers():
    A = air.tensor([64], i32)
    KV = air.tensor([256], i32)
    tiles = air.channel("Tiles3")
    back = air.channel("Back3")
    again = air.channel("Again3")

    with air.launch(name="rb3") as launch:

        @launch.body
        def _():
            with air.segment(name="seg3") as seg:

                @seg.body
                def _():
                    tiles.put(A)
                    with air.herd([range(1)], name="h3", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            buf = air.alloc([64], i32, scope=h.private())
                            tiles.get(buf)
                            back.put(buf)

                    drained = back.get(KV[0:64])
                    again.put(KV[0:32], dependency=drained)
                    again.put(KV[32:64], dependency=drained)

    print(launch.mlir())
