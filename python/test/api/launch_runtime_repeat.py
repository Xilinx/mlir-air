# ./python/test/api/launch_runtime_repeat.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""A launch repeated a run-time number of times, each wave on a run-time row.

``repeat=`` takes a rank-0 integer tensor: the scf.for around air.launch is
bounded by the scalar argument. A rank-0 integer tensor is also an index where
one is used, so
``ops.switch(wave, [lambda: e0, ...])`` picks the wave's own scalar, and an
offset built from one scalar twice is one term: ``W[o : o + n]`` keeps its
static size.
"""

from air import api as air
from air.api import ops
from air.api.types import i32

E, D, K = 64, 1024, 4

W = air.tensor([E, D], i32)
Y = air.tensor([K, D], i32)
n = air.tensor([], i32, name="n")
es = [air.tensor([], i32, name=f"e{k}") for k in range(K)]
win = air.channel("Win")
yout = air.channel("Yout")

# CHECK-LABEL: func.func @rt(
# CHECK-SAME:    %[[W:arg0]]: memref<64x1024xi32>, %{{.*}}: memref<4x1024xi32>, %[[N:arg2]]: i32
# CHECK:         %[[NI:.*]] = arith.index_cast %[[N]] : i32 to index
# CHECK:         %[[B:.*]] = affine.apply #map()[%[[NI]]]
# CHECK:         scf.for %{{.*}} = %c0 to %[[B]] step %c1 {
# CHECK:           air.launch
# CHECK:             scf.index_switch %{{.*}} -> index
# CHECK:             case 0 {
# CHECK:               arith.index_cast %{{.*}} : i32 to index
# CHECK:             default {
# CHECK:             air.channel.put  @Win[] (%{{.*}}[%{{.*}}, 0] [1, 1024] [1024, 1])
with air.launch(repeat=n, name="rt") as lch:

    @lch.body
    def _(wave):
        e = ops.switch(wave, [lambda t=t: t for t in es])
        win.put(W[e, :])
        yout.get(Y[wave, :])
        with air.segment(name="seg") as seg:

            @seg.body
            def _():
                with air.herd([range(1)], name="h", shape=(1,), at=(0, 2)) as h:

                    @h.body
                    def _(tx):
                        a = air.alloc([D], i32, scope=h.private())
                        b = air.alloc([D], i32, scope=h.private())
                        win.get(a)
                        b[:] = a[:] + 1
                        yout.put(b)


print(lch.build(target="npu2"))

# A scalar read twice is one term: the slice's size stays static.
# CHECK-LABEL: func.func @flat(
# CHECK:         air.channel.put  @F[] (%{{.*}}[%{{.*}}] [256] [1])
F = air.tensor([E * 256], i32)
s = air.tensor([], i32, name="s")
f = air.channel("F")
with air.launch(repeat=2, name="flat") as lch2:

    @lch2.body
    def _(wave):
        f.put(F[(wave + s) * 256 : (wave + s) * 256 + 256])
        with air.segment(name="seg") as seg:

            @seg.body
            def _():
                with air.herd([range(1)], name="h", shape=(1,), at=(0, 2)) as h:

                    @h.body
                    def _(tx):
                        a = air.alloc([256], i32, scope=h.private())
                        f.get(a)


print(lch2.build(target="npu2"))

# A repeat count must be an integer scalar.
# CHECK: rank-0 integer tensor
try:
    air.launch(repeat=air.tensor([2], i32))
except TypeError as err:
    print(err)
